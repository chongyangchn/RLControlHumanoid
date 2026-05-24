import mujoco
import numpy as np
import torch
from pathlib import Path

# 导入奖励模块
from humanoid.envs.reward import compute_total_reward

class HumanoidG1Env:
    """
    G1 人形机器人环境（MuJoCo only，无 Gym 依赖）
    所有输入输出均为 torch.Tensor，适配 PPO 算法
    """
    def __init__(self, xml_path, cfg, reward_scaling=None):
        """

        """
        # -------- 1. 加载模型 --------
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)


        # -------- 2. 动作重复与时间步长 --------
        self.action_repeat = cfg['env']['action_repeat']                    # 动作重复次数 25Hz (dt=0.002*20=0.04s)
        self.action_scale = cfg['env']['action_scale']                      # 动作缩放系数
        self.dt = self.model.opt.timestep * self.action_repeat              # 实际控制周期
        self.reset_count = 0


        # ---------- 3 动作与控制相关信息 + 观测量信息----------
        self.num_joints = self.model.nq - 7                                 # 关节数量：29
        self.action_dim = self.model.nu                                     # 动作维度29 (双腿：6 × 2 = 12 ；腰部：3；双臂：7 × 2 = 14；) (合计：12 + 3 + 14 = 29)
        self.obs_dim = 11 + 2 + 2 * self.num_joints + self.action_dim           # 加历史动作：98维  基础量：基座高度(1) + 姿态(4) + 躯干IMU(6) = 11

        # 关节限位（
        self.joint_low = self.model.actuator_ctrlrange[:, 0] # 获取XML中定义的关节限位
        self.joint_high = self.model.actuator_ctrlrange[:, 1] # 获取XML中定义的关节限位
        self.joint_qpos_idx = slice(7, self.model.nq)  # 关节位置索引（跳过基座 7 维）
        self.joint_qvel_idx = slice(6, self.model.nv)  # 关节速度索引（跳过基座 6 维）

        # ---------- 4 默认站立姿态（优先从 keyframe "stand" 读取）----------
        stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if stand_key_id >= 0:
            # keyframe 中的 ctrl 顺序与 actuator 一致
            self.default_stand = self.model.key_ctrl[stand_key_id].copy()
        else:
            # 若没有 keyframe，取当前 ctrl（需确保模型初始为直立）
            self.default_stand = self.data.ctrl.copy()
            print("[WARNING] No 'stand' keyframe found. Using current ctrl as default.")
        self.last_action = np.zeros(self.action_dim)
        self.invalid_steps = 0 # 无效步数

        # -------- 5. 传感器与几何体 ID --------
        # 躯干 site ID (需要你的xml中有这个site名)
        self.torso_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso")
        # 如果没有这个site，尝试用 trunk 或 base
        if self.torso_site_id < 0:
            self.torso_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "trunk")


        # -------- 6. 设备与 Tensor 缓存  --------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"当前设备所使用的设备是：{self.device}")

        # 缓存物理数据的 tensor 版本，用于加速观测构造
        self.last_action_tensor = torch.zeros(self.action_dim, device=self.device)
        self.qpos_tensor = torch.zeros(self.model.nq, dtype=torch.float32, device=self.device)
        self.qvel_tensor = torch.zeros(self.model.nv, dtype=torch.float32, device=self.device)
        self.qacc_tensor = torch.zeros(self.model.nv, dtype=torch.float32, device=self.device)
        self.ctrl_tensor = torch.zeros(self.model.nu, dtype=torch.float32, device=self.device)
        self.foot_contacts_tensor = torch.zeros(2, dtype=torch.float32, device=self.device)  # 2维：[左脚, 右脚]

        # 关节限位与默认姿态的 tensor 版本
        self.joint_low_tensor = torch.tensor(self.joint_low, dtype=torch.float32, device=self.device)
        self.joint_high_tensor = torch.tensor(self.joint_high, dtype=torch.float32, device=self.device)
        self.default_stand_tensor = torch.tensor(self.default_stand, dtype=torch.float32, device=self.device)


        self.reward_scaling = reward_scaling  # 奖励函数系数

        # 奖励系数（可配置）
        if reward_scaling is None:
            self.reward_scaling = {
                # === 核心正向奖励 ===
                'vel': 1.0,          # 速度跟踪（最重要的任务目标）
                'survival': 0.05,    # 生存奖励（只要没摔，就给一点点甜头）
                'height': 0.2,       # 高度奖励（维持稳定高度，防止躺地）
                'upright': 0.2,      # 直立奖励（防止侧翻/仰翻，但不能太强，否则机器人不敢动）

                # === 致命惩罚（解决抽搐和空翻的关键！） ===
                'smooth': -0.02,     # 动作平滑惩罚（防止高频抖动。如果还在抽搐，请改为 -0.05 或 -0.1！）
                'torque': -1e-5,     # 扭矩惩罚（防止用蛮力维持平衡，导致突然摔倒）
                'joint_acc': -5e-7,  # ★★★ 关节加速度惩罚（防止抽搐的核武器！必须要有！）
                'foot_contact': 0.1, # 足部接触奖励（鼓励双脚交替落地，防止滑行）
            }
        print(f"[Env] Loaded G1 | 控制量维度={self.action_dim} | 观测量维度={self.obs_dim} | 时间步长{self.dt:.3f}s")



    # ================================================================
    #  核心接口：reset, step, _get_obs, _get_reward, _is_done
    # ================================================================

    def reset(self):
        """重置仿真状态"""
        # 1. 物理重置
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # 2. 设置到默认站立姿态
        # 将关节设为默认站立姿态（若需从站立状态开始）
        # 注意：mj_resetData 会重置所有 qpos/qvel，我们需要手动设置
        # 将 keyframe 的 qpos 应用到模型
        stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if stand_key_id >= 0:
            # 设置基座位置和姿态（keyframe 中 qpos 前 7 个值）
            self.data.qpos[:7] = self.model.key_qpos[stand_key_id][:7]
            # 设置关节角度 (qpos[7:])
            self.data.qpos[7:] = self.model.key_qpos[stand_key_id][7:]
            # 设置关节速度 (qvel) 为0
            self.data.qvel[:] = 0.0
            # 设置控制量为默认姿态
            self.data.ctrl[:] = self.default_stand
        else:
            # 没有 keyframe，直接将默认姿态作为控制量
            self.data.ctrl[:] = self.default_stand

        mujoco.mj_forward(self.model, self.data)

        # 3. 重置内部状态
        self.reset_count += 1
        self.invalid_steps = 0
        self.last_action = np.zeros(self.action_dim)
        self.last_action_tensor.zero_()
        self.foot_contacts_tensor = self._get_foot_contacts_tensor()  # 初始化接触信息

        return self._get_obs_tensor()


    def step(self, action: torch.Tensor):
        """
        执行一步仿真。
        Args:
            action: 29维动作 Tensor (范围应在 [-1, 1] 之间)
        Returns:
            obs: 下一步观测 Tensor
            reward: 奖励 Tensor (标量)
            done: 是否结束 (bool)
            info: 信息字典
        """
        # 1. 动作处理（已经是 tensor 了，或者转成 tensor）
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action_np = action.detach().cpu().numpy()

        # 2. 动作缩放与限幅 (以默认姿态为中心)
        action_np = np.clip(action_np, -1, 1)
        target_angle = self.default_stand + self.action_scale * action_np
        target_angle = np.clip(target_angle, self.joint_low, self.joint_high)
        self.data.ctrl[:] = target_angle

        # 3. 物理步进（多个子步）
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)

        # 4. 更新缓存 Tensor (用于奖励计算和观测)
        self.qpos_tensor.copy_(torch.as_tensor(self.data.qpos, device=self.device))
        self.qvel_tensor.copy_(torch.as_tensor(self.data.qvel, device=self.device))
        self.qacc_tensor.copy_(torch.as_tensor(self.data.qacc, device=self.device))
        self.ctrl_tensor.copy_(torch.as_tensor(self.data.ctrl, device=self.device))

        # 5. 更新动作历史
        self.last_action = action_np.copy()
        self.last_action_tensor.copy_(action)

        # 6. 计算奖励
        reward = compute_total_reward(self, action)

        # 7. 判断是否 done
        done = self._is_done()
        if done:
            reward -= 10.0  # 摔倒惩罚

        obs = self._get_obs_tensor()
        return  obs, reward, done, {}

    # ─────────────────────────── 观测 ────────────────────────────
    def _get_obs_tensor(self):
        """构建并返回 torch.Tensor 格式的观测向量"""
        # 1. 基座与传感器 (从缓存的 tensor 读取)

        base_height = self.qpos_tensor[2:3]  # (1,)
        base_quat = self.qpos_tensor[3:7]  # (4,)
        torso_angvel = self.qvel_tensor[3:6]  # (3,)
        # 加速度需要从传感器读取（暂省略，用0占位）
        torso_acc = torch.zeros(3, device=self.device)
        joint_pos = self.qpos_tensor[7:]  # (29,)
        joint_vel = self.qvel_tensor[6:]  # (29,)
        foot_contacts = self.foot_contacts_tensor  # (4,)
        last_action = self.last_action_tensor  # (29,)

        obs = torch.cat([
            base_height,
            base_quat,
            torso_angvel,
            torso_acc,
            joint_pos,
            joint_vel,
            foot_contacts,
            last_action
        ])

        # # 关节位置（粗略归一化到 [-1,1]）
        # base_pos = (self.data.qpos[2:3].flatten()  - 0.78) / 0.3 # 只取高度 Z   # 确保是 (1,)  # 大约归一化到 [-1.5,1.5]
        # base_quat = self.data.qpos[3:7].flatten()    # 确保是 (4,)   # 已在 [-1,1]
        # base_vel_raw = self.data.qvel[0:6].flatten()   # 6 维
        # base_vel = np.clip(base_vel_raw / [5.0,5.0,5.0,10.0,10.0,10.0], -1.0, 1.0)
        #
        # # 关节位置与速度 (仅限受控的29个关节)
        # joint_pos_raw = self.data.qpos[7:].flatten()   # 确保是 (29,)
        # joint_range = self.joint_high - self.joint_low
        # joint_pos = 2.0 * (joint_pos_raw - self.joint_low) / (joint_range + 1e-8) - 1.0
        #
        # joint_vel_raw = self.data.qvel[6:].flatten()  # qvel前6位是自由座的速度   # 确保是 (29,)
        # joint_vel = np.clip(joint_vel_raw / 20.0, -1.0, 1.0)
        # last_action = self.last_action.flatten()   # 确保是 (29,)   # 已在 [-1,1] 内
        #
        #
        # # obs_data = np.concatenate([base_pos, base_quat, joint_pos, joint_vel]).astype(np.float32)  # 维度为：base_pos： 1   base_quat：4    joint_pos：29  joint_vel：29  相加是63
        # obs_data = np.concatenate([
        #     base_pos, base_quat, base_vel,          # 1 + 4 + 6 = 11
        #     joint_pos, joint_vel, last_action       # 29 + 29 + 29 = 87
        # ]).astype(np.float32) # 维度为：base_pos： 1   base_quat：4    joint_pos：29  joint_vel：29   last_action： 29  相加是92
        # # 总维度: 11 + 87 = 98

        return obs.to(torch.float32)


    # ─────────────────────────── Reward ──────────────────────────
    # def _get_reward(self, action):
        """初步的奖励函数：比如根据躯干高度和前进速度"""
        # forward_vel = self.data.qvel[0]  # 是世界系 x 轴速度，
        # 20260523更新：
        # 获取当前状态量
        # height = self.data.qpos[2]
        # base_vel = self.data.qvel[0:3]  # 基座速度 (x, y, z)
        # base_angvel = self.data.qvel[3:6]  # 基座角速度 (roll, pitch, yaw)
        # upright = self._get_upright_reward()  # 躯干直立度 (0~1)
        # torques = self.data.ctrl.copy()  # 实际输出力矩 (用于惩罚)
        # joint_acc = self.data.qacc[6:]  # 关节加速度 (跳过基座6维)
        # action_rate = np.mean(np.square(action - self.last_action))
        #
        # reward = 0.0
        # # (1) 生存奖励
        # reward += self.reward_scaling['survival']
        #
        # # (2) 速度跟踪奖励 (仅鼓励前进 x 方向)
        # target_vel_x = 0.6  # 目标速度 m/s
        # vel_reward = np.exp(-5.0 * (base_vel[0] - target_vel_x) ** 2)
        # reward += self.reward_scaling['vel'] * vel_reward
        #
        # # (3) 高度奖励 (鼓励保持稳定高度)
        # target_height = 0.78  # 根据 G1 实际站立高度微调
        # height_reward = np.exp(-10.0 * (height - target_height) ** 2)
        # reward += self.reward_scaling['height'] * height_reward
        #
        # # (4) 直立奖励
        # reward += self.reward_scaling['upright'] * upright
        #
        # # (5) 动作平滑惩罚
        # reward += self.reward_scaling['smooth'] * action_rate
        #
        # # (6) 扭矩惩罚 (防止暴力维持平衡)
        # reward += self.reward_scaling['torque'] * np.mean(np.square(torques))
        #
        # # (7) 关节加速度惩罚 (抑制抽搐)
        # reward += self.reward_scaling['joint_acc'] * np.mean(np.square(joint_acc))
        #
        # # (8) 足部接触奖励 (鼓励双脚交替落地)
        # foot_contacts = self._get_foot_contacts()
        # contact_reward = self.reward_scaling.get('foot_contact', 0.1) * np.mean(foot_contacts)
        # reward += contact_reward
        #
        # # (9) 惩罚垂直速度 (防止空翻)
        # reward -= 2.0 * abs(base_vel[2])
        #
        # # (10) 惩罚侧向角速度 (防止侧倒)
        # reward -= 0.5 * (abs(base_angvel[0]) + abs(base_angvel[1]))

        # quat = self.data.qpos[3:7].copy()
        # rot_mat = np.zeros(9)
        # mujoco.mju_quat2Mat(rot_mat, quat)
        # rot_mat = rot_mat.reshape(3, 3)
        # global_vel = self.data.qvel[0:3]
        # local_vel = rot_mat.T @ global_vel # 局部坐标系速度
        # forward_vel = local_vel[0]  # 机器人前进方向（通常是 x 轴）
        #
        # height = self.data.qpos[2]
        # upright = self._get_upright_reward()  # [0,1] 躯干直立程度
        #
        # reward_vel = self.reward_scaling['vel'] * np.exp(-2.0 * (forward_vel - 0.6)**2)                             # 1. 速度奖励 (目标 0.6 m/s)
        # reward_height = self.reward_scaling['height'] * np.exp(-5.0 * (height - 0.78)**2)                           # 2. 高度惩罚 (G1 站立高度约为 0.7-0.8m)
        # upright_reward = self.reward_scaling['upright'] * upright                                                   # 3. 躯干竖直奖励
        # reward_smooth = self.reward_scaling['smooth']  * np.mean(np.square(action - self.last_action))              # 4. 平滑惩罚  sum换成mean
        # reward_survival = self.reward_scaling['survival']                                                           # 5. 生存奖励
        #
        # reward = reward_vel + reward_height + reward_smooth + reward_survival + upright_reward
        # # 如果高度低于0.6m，但没到摔倒阈值0.4m，使用指数级惩罚
        # if 0.4 < height < 0.6:
        #     penalty = 1.0 / (height - 0.39 + 1e-8)  # 指数惩罚
        #     reward -= penalty

        # 20260523 更新
        # terms = [
        #     RewTerm(track_lin_vel_xy_yaw_frame_exp, weight=1.0),
        #     RewTerm(track_ang_vel_z, weight=0.5),
        #     RewTerm(is_alive, weight=0.1),
        #     RewTerm(flat_orientation_l2, weight=-1.0),
        #     RewTerm(base_height_l2, weight=-10.0, params={"target_height": 0.78}),
        #     RewTerm(joint_acc, weight=-5e-7),
        #     RewTerm(action_rate, weight=-0.02, params={"action": action}),
        #     RewTerm(feet_slide, weight=-0.5),
        #     RewTerm(gait, weight=0.2),
        #     RewTerm(joint_vel, weight=-0.001),
        #     RewTerm(dof_pos_limits, weight=-1.0),
        #     RewTerm(energy, weight=-0.0001),
        #     RewTerm(joint_deviation_arms, weight=-0.1),
        #     RewTerm(joint_deviation_waists, weight=-0.1),
        #     RewTerm(joint_deviation_legs, weight=-0.1),
        #     RewTerm(feet_clearance, weight=-0.1),
        #     RewTerm(undesired_contacts, weight=-1.0),
        # ]
        # total_reward = torch.tensor(0.0, device=action.device)
        # for t in terms:
        #     total += t(env)
        # return reward


    # ─────────────────────────── 终止条件 ────────────────────────
    def _is_done(self):
        """判断是否摔倒或任务结束"""
        height = self.data.qpos[2]
        upright = self._get_upright_reward()

        if height < 0.4:
            return True

        if height > 1.8:
            return True

        if upright < 0.7:
            return True

        return False


    # ─────────────────────────── 辅助方法 ────────────────────────
    def _get_upright_reward(self):
        if self.torso_site_id < 0:
            return 1.0
        xmat = self.data.site_xmat[self.torso_site_id].reshape(3, 3)
        torso_up = xmat[:, 2]  # 局部 Z 轴在世界坐标系中的方向
        upright = np.dot(torso_up, np.array([0, 0, 1]))
        return max(0.0, upright)

    def _get_foot_contacts(self) -> np.ndarray:
        """
        检测左右脚是否接触地面 (Numpy 版本，用于物理)
        返回: (4,) 数组 [左脚前, 左脚后, 右脚前, 右脚后]
        """
        contact = np.zeros(2)

        # 获取 site ID (每次调用时获取，或者缓存起来)
        left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

        # 如果找不到 site，返回 0，避免报错
        if left_foot_id < 0 or right_foot_id < 0:
            print("找不到 left_foot_id 和 right_foot_id")
            return contact

        # 获取 site 的世界坐标
        left_pos = self.data.site_xpos[left_foot_id]
        right_pos = self.data.site_xpos[right_foot_id]

        # 如果 Z 坐标小于一个很小的值（比如 0.005m），我们认为它接触地面
        contact[0] = 1.0 if left_pos[2] < 0.005 else 0.0
        contact[1] = 1.0 if right_pos[2] < 0.005 else 0.0
        return contact

    def _get_foot_contacts_tensor(self) -> torch.Tensor:
        """返回足部接触信息的 Tensor 版本"""
        # 为性能，我们可以在 step 中缓存，这里直接调用并转 Tensor
        contact_np = self._get_foot_contacts()
        return torch.as_tensor(contact_np, device=self.device)





