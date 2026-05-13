import mujoco
import time
import numpy as np
from pathlib import Path

class HumanoidG1Env:
    """
    G1 人形机器人环境（MuJoCo only，无 Gym 依赖）
    """
    def __init__(self, xml_path, cfg, reward_scaling=None):
        """
        xml_path: G1模型路径
        action_repeat: 每个控制命令重复多少次物理步长 (控制频率 = 模拟频率 / action_repeat)
        reward_scaling: 奖励缩放系数 (可选字典，如 {'vel': 2.0, 'upright': 3.0})
        """
        # ── 加载模型 ──────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.action_repeat = cfg['env']['action_repeat']   # 👇 动作重复次数 25Hz (dt=0.002*20=0.04s)
        self.dt = self.model.opt.timestep * self.action_repeat
        self.reward_scaling = reward_scaling  # 奖励函数系数
        self.reset_count = 0

        # ---------- 确定关节控制数量 ----------
        self.nu = self.model.nu  # 动作维度（执行器数量）                      # 29
        self.action_dim = self.nu  # 控制量维度
        self.num_joints = self.model.nq - 7   # G 关节数量：29 （12 个腿部Waist + 3 个腰部Legs + 14 个手臂Arms）

        # 观测量维度后续试一下两种不同的观测量：
        # self.obs_dim = 5 + self.num_joints * 2   # z轴高度 + 四元数（5） + 关节位置、速度 + Torso 角速度 + 线加速度 + Pelvis 角速度 + 线加速度
        self.obs_dim = 1 + 4 + 6 + self.num_joints * 2 + self.action_dim  # z轴高度 + 四元数（5） + 关节位置、速度 + Torso 角速度 + 线加速度 + Pelvis 角速度 + 线加速度 + 上一时刻动作

        # 默认站立姿态 (读取初始的控制量)
        self.default_stand = self.data.ctrl.copy()
        self.last_action = np.zeros(self.action_dim)

        self.joint_low = self.model.actuator_ctrlrange[:, 0] # 获取XML中定义的关节限位
        self.joint_high = self.model.actuator_ctrlrange[:, 1] # 获取XML中定义的关节限位
        self.joint_qpos_idx = slice(7, self.model.nq)  # 关节位置索引（跳过基座 7 维）
        self.joint_qvel_idx = slice(6, self.model.nv)  # 关节速度索引（跳过基座 6 维）

        # 躯干 site ID (需要你的xml中有这个site名)
        self.torso_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso"
        )
        # 如果没有这个site，尝试用 trunk 或 base
        if self.torso_site_id < 0:
            self.torso_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "trunk"
            )

        # 奖励系数（可配置）
        if reward_scaling is None:
            self.reward_scaling = {
                'vel': 0.5,  # 速度奖励系数
                'height': 0.5,  # 高度奖励系数
                'upright': 0.8,  # 竖直奖励系数
                'smooth': -0.03,  # 动作平滑惩罚系数
                'survival': 0.05,  # 每步生存奖励
            }
        print(f"[Env] Loaded G1 | 控制量维度={self.action_dim} | 观测量维度={self.obs_dim} | 时间步长{self.dt:.3f}s")

    # ─────────────────────────── 公共接口 ────────────────────────

    def reset(self):
        """重置仿真状态"""
        mujoco.mj_resetData(self.model, self.data)

        # 使用 XML 中的 Keyframe "stand" 初始化姿态
        stand_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if stand_key_id != -1:
            self.data.qpos[:] = self.model.key_qpos[stand_key_id]
            self.data.ctrl[:] = self.model.key_ctrl[stand_key_id]

        self.reset_count += 1
        self.last_action = np.zeros(self.action_dim)
        return self._get_obs().astype(np.float32)


    def step(self, action):
        """执行一步动作"""
        # 关键点 1：确保传入的 action 是 1 维的 # 如果 action 是 [[...]] 这种形状，就取第一行
        if action.ndim > 1:
            action = action.flatten()

        # 1. 将网络输出的 [-1, 1] 映射到 XML 的真实弧度范围
        ctrl_action = self.joint_low + 0.5 * (action + 1) * (self.joint_high - self.joint_low)
        self.data.ctrl[:] = ctrl_action

        # 2. 物理仿真推进
        for _ in range(self.action_repeat):   # 执行多次物理步长
            mujoco.mj_step(self.model, self.data)

        # 关键点 2：保存副本时确保是平坦的 1 维数组
        self.last_action = action.copy().flatten()

        obs = self._get_obs().astype(np.float32)
        reward = self._get_reward(action)
        done = self._is_done()
        info = {"height": self.data.qpos[2], "forward_vel": self.data.qvel[0]}

        return  obs, reward, done, info

    # ─────────────────────────── 观测 ────────────────────────────
    def _get_obs(self):
        """提取观测向量：包括躯干姿态、关节角度、速度等"""
        # 提取高度和四元数 (Base Position/Orientation)
        # qpos[:3] 是 XYZ, qpos[3:7] 是四元数
        # 关节位置（粗略归一化到 [-1,1]）
        base_pos = (self.data.qpos[2:3].flatten()  - 0.78) / 0.3 # 只取高度 Z   # 确保是 (1,)  # 大约归一化到 [-1.5,1.5]
        base_quat = self.data.qpos[3:7].flatten()    # 确保是 (4,)   # 已在 [-1,1]
        base_vel_raw = self.data.qvel[0:6].flatten()   # 6 维
        base_vel = np.clip(base_vel_raw / [5.0,5.0,5.0,10.0,10.0,10.0], -1.0, 1.0)

        # 关节位置与速度 (仅限受控的29个关节)
        joint_pos_raw = self.data.qpos[7:].flatten()   # 确保是 (29,)
        joint_range = self.joint_high - self.joint_low
        joint_pos = 2.0 * (joint_pos_raw - self.joint_low) / (joint_range + 1e-8) - 1.0

        joint_vel_raw = self.data.qvel[6:].flatten()  # qvel前6位是自由座的速度   # 确保是 (29,)
        joint_vel = np.clip(joint_vel_raw / 20.0, -1.0, 1.0)
        last_action = self.last_action.flatten()   # 确保是 (29,)   # 已在 [-1,1] 内


        # obs_data = np.concatenate([base_pos, base_quat, joint_pos, joint_vel]).astype(np.float32)  # 维度为：base_pos： 1   base_quat：4    joint_pos：29  joint_vel：29  相加是63
        obs_data = np.concatenate([
            base_pos, base_quat, base_vel,          # 1 + 4 + 6 = 11
            joint_pos, joint_vel, last_action       # 29 + 29 + 29 = 87
        ]).astype(np.float32) # 维度为：base_pos： 1   base_quat：4    joint_pos：29  joint_vel：29   last_action： 29  相加是92
        # 总维度: 11 + 87 = 98
        return obs_data


    # ─────────────────────────── Reward ──────────────────────────
    def _get_reward(self, action):
        """初步的奖励函数：比如根据躯干高度和前进速度"""
        # forward_vel = self.data.qvel[0]  # 是世界系 x 轴速度，

        quat = self.data.qpos[3:7].copy()
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        global_vel = self.data.qvel[0:3]
        local_vel = rot_mat.T @ global_vel # 局部坐标系速度
        forward_vel = local_vel[0]  # 机器人前进方向（通常是 x 轴）

        height = self.data.qpos[2]
        upright = self._get_upright_reward()  # [0,1] 躯干直立程度

        reward_vel = self.reward_scaling['vel'] * np.exp(-2.0 * (forward_vel - 0.6)**2)                             # 1. 速度奖励 (目标 0.6 m/s)
        reward_height = self.reward_scaling['height'] * np.exp(-5.0 * (height - 0.78)**2)                           # 2. 高度惩罚 (G1 站立高度约为 0.7-0.8m)
        upright_reward = self.reward_scaling['upright'] * upright                                                   # 3. 躯干竖直奖励
        reward_smooth = self.reward_scaling['smooth']  * np.sum(np.square(action - self.last_action))               # 4. 平滑惩罚
        reward_survival = self.reward_scaling['survival']                                                           # 5. 生存奖励
        return reward_vel + reward_height + reward_smooth + reward_survival + upright_reward


    # ─────────────────────────── 终止条件 ────────────────────────
    def _is_done(self):
        """判断是否摔倒或任务结束"""
        height = self.data.qpos[2]
        upright = self._get_upright_reward()
        return (height < 0.4)  or (upright < 0.3)


    # ─────────────────────────── 辅助方法 ────────────────────────
    def _get_upright_reward(self):
        if self.torso_site_id < 0:
            return 1.0
        xmat = self.data.site_xmat[self.torso_site_id].reshape(3, 3)
        torso_up = xmat[:, 2]
        upright = np.dot(torso_up, np.array([0, 0, 1]))
        return max(0.0, upright)


