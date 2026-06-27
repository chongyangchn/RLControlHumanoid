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
    def __init__(self, xml_path, cfg):

        # -------- 1. 加载模型 --------
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)


        # -------- 2. 动作重复与时间步长 --------
        self.action_repeat = cfg['env']['action_repeat']                    # 动作重复次数 25Hz (dt=0.002*20=0.04s)
        self.action_scale = cfg['env']['action_scale']                      # 动作缩放系数
        self.dt = self.model.opt.timestep * self.action_repeat              # 实际控制周期
        self.reset_count = 0
        self.step_count = 0
        self.max_episode_steps = cfg['env'].get('max_episode_steps', 1000)
        self.training_stage = cfg['env'].get('training_stage', 'stand')
        self.command_lin_vel_x = cfg['env'].get('command_lin_vel_x', 0.0)
        self.command_yaw_rate = cfg['env'].get('command_yaw_rate', 0.0)


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
        self.last_action_tensor = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
        self.qpos_tensor = torch.zeros(self.model.nq, dtype=torch.float32, device=self.device)
        self.qvel_tensor = torch.zeros(self.model.nv, dtype=torch.float32, device=self.device)
        self.qacc_tensor = torch.zeros(self.model.nv, dtype=torch.float32, device=self.device)
        self.ctrl_tensor = torch.zeros(self.model.nu, dtype=torch.float32, device=self.device)
        self.foot_contacts_tensor = torch.zeros(2, dtype=torch.float32, device=self.device)  # 2维：[左脚, 右脚]

        # 关节限位与默认姿态的 tensor 版本
        self.joint_low_tensor = torch.tensor(self.joint_low, dtype=torch.float32, device=self.device)
        self.joint_high_tensor = torch.tensor(self.joint_high, dtype=torch.float32, device=self.device)
        self.default_stand_tensor = torch.tensor(self.default_stand, dtype=torch.float32, device=self.device)

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

        # 3. 关键修复：立即更新缓存
        self.qpos_tensor.copy_(torch.as_tensor(self.data.qpos, device=self.device))
        self.qvel_tensor.copy_(torch.as_tensor(self.data.qvel, device=self.device))
        self.qacc_tensor.copy_(torch.as_tensor(self.data.qacc, device=self.device))
        self.ctrl_tensor.copy_(torch.as_tensor(self.data.ctrl, device=self.device))
        self.foot_contacts_tensor = self._get_foot_contacts_tensor()

        # 3. 重置内部状态
        self.reset_count += 1
        self.step_count = 0
        self.invalid_steps = 0
        self.last_action = np.zeros(self.action_dim)
        self.last_action_tensor.zero_()
        self.foot_contacts_tensor = self._get_foot_contacts_tensor()  # 初始化接触信息

        return self._get_obs_tensor()


    def step(self, action: torch.Tensor):
        # 1. 动作处理（已经是 tensor 了，或者转成 tensor）
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = torch.clamp(action.to(self.device), -1.0, 1.0)
        prev_action_tensor = self.last_action_tensor.clone()
        action_np = action.detach().cpu().numpy()

        # 2. 动作缩放与限幅 (以默认姿态为中心)
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
        self.foot_contacts_tensor = self._get_foot_contacts_tensor()
        self.prev_action_tensor = prev_action_tensor

        # 6. 计算奖励
        reward = compute_total_reward(self, action)

        self.last_action = action_np.copy()
        self.last_action_tensor.copy_(action)
        self.step_count += 1

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
        return obs.to(torch.float32)


    # ─────────────────────────── Reward ──────────────────────────
    # def _get_reward(self, action):

        # return reward


    # ─────────────────────────── 终止条件 ────────────────────────
    def _is_done(self):
        height = self.data.qpos[2]
        upright = self._get_upright_reward()

        if height < 0.6 or height > 1.0:
            return True

        if upright < 0.85:
            return True

        if self.step_count >= self.max_episode_steps:
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





