#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/23 17:45
# @Author  : cychn
# @File    : reward.py
# @Software: PyCharm

"""
奖励函数
"""
import torch
import numpy as np


class RewTerm:
    def __init__(self, func, weight=1.0, params=None):
        self.func = func
        self.weight = weight
        self.params = params if params is not None else {}

    def __call__(self, env):
        return self.weight * self.func(env, **self.params)


# --------------------- 核心奖励项 ---------------------

def track_lin_vel_xy_yaw_frame_exp(env, std: float = 0.25) -> torch.Tensor:
    """
    在机器人局部坐标系下追踪线速度（X, Y）。
    """
    # 使用缓存 Tensor (而不是 env.data)
    base_quat = env.qpos_tensor[3:7]  # (4,)
    lin_vel = env.qvel_tensor[0:3]  # (3,)

    # 四元数解绑
    qw, qx, qy, qz = base_quat.unbind(-1)

    # 将全局速度旋转到局部坐标系
    vx = (1 - 2 * qy ** 2 - 2 * qz ** 2) * lin_vel[0] + (2 * qx * qy - 2 * qz * qw) * lin_vel[1] + (
                2 * qx * qz + 2 * qy * qw) * lin_vel[2]
    vy = (2 * qx * qy + 2 * qz * qw) * lin_vel[0] + (1 - 2 * qx ** 2 - 2 * qz ** 2) * lin_vel[1] + (
                2 * qy * qz - 2 * qx * qw) * lin_vel[2]

    local_vel = torch.stack([vx, vy], dim=-1)  # (2,)

    cmd_vel = torch.tensor([0.6, 0.0], device=local_vel.device)  # 目标速度
    error = torch.norm(cmd_vel - local_vel, dim=-1)
    return torch.exp(-error / (std ** 2))


def track_ang_vel_z(env, std: float = 0.5) -> torch.Tensor:
    yaw_rate = env.qvel_tensor[5]  # 偏航角速度
    cmd_yaw_rate = torch.tensor(0.0, device=yaw_rate.device)
    error = torch.abs(cmd_yaw_rate - yaw_rate)
    return torch.exp(-error / (std ** 2))


def is_alive(env) -> torch.Tensor:
    return torch.tensor(1.0, device=env.device)


def flat_orientation_l2(env) -> torch.Tensor:
    quat = env.qpos_tensor[3:7]  # 从缓存读取
    qw, qx, qy, qz = quat.unbind(-1)
    zx = 2 * (qx * qz + qy * qw)
    zy = 2 * (qy * qz - qx * qw)
    zz = 1 - 2 * (qx ** 2 + qy ** 2)
    error = torch.sqrt(zx ** 2 + zy ** 2 + (zz - 1) ** 2)
    return torch.exp(-10.0 * error)  # 返回 [0, 1]


def base_height_l2(env, target_height: float = 0.78) -> torch.Tensor:
    height = env.qpos_tensor[2]  # 从缓存读取
    return torch.exp(-10.0 * (height - target_height) ** 2)  # 返回 [0, 1]


def joint_acc(env) -> torch.Tensor:
    acc = env.qacc_tensor[6:]  # 跳过基座
    return -torch.exp(-0.1 * torch.mean(acc ** 2))


def action_rate(env, action: torch.Tensor) -> torch.Tensor:
    rate = torch.mean((action - env.last_action_tensor) ** 2)
    return -rate


def feet_slide(env) -> torch.Tensor:
    # 使用缓存 Tensor
    contacts = env.foot_contacts_tensor  # (2,) [左, 右]
    foot_vel = env.qvel_tensor[0:3]  # 用基座速度近似
    slide = torch.norm(foot_vel[:2])
    penalty = torch.where((contacts[0] == 1) | (contacts[1] == 1), -0.5 * slide, 0.0)
    return penalty


def gait(env) -> torch.Tensor:
    contacts = env.foot_contacts_tensor  # (2,)
    left = contacts[0]
    right = contacts[1]
    # 一只脚接触，另一只离地 -> 迈步奖励
    reward = torch.where((left == 1) & (right == 0) | (left == 0) & (right == 1), 0.5, 0.0)
    return reward


def joint_vel(env) -> torch.Tensor:
    vel = env.qvel_tensor[6:]  # 跳过基座
    return -torch.exp(-0.1 * torch.mean(vel ** 2)) * 0.001


def dof_pos_limits(env) -> torch.Tensor:
    qpos = env.qpos_tensor[7:]  # 跳过基座
    low = env.joint_low_tensor
    high = env.joint_high_tensor
    violation = torch.sum((qpos - low < 0.05) | (high - qpos < 0.05))
    return -violation.float()


def energy(env) -> torch.Tensor:
    torque = env.ctrl_tensor
    vel = env.qvel_tensor[6:]
    power = torch.sum(torch.abs(torque * vel))
    return -1e-4 * power


def joint_deviation_arms(env) -> torch.Tensor:
    # 假设索引（需根据实际 XML 调整）
    indices = torch.tensor([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], device=env.device)
    qpos = env.qpos_tensor[7:]
    default = env.default_stand_tensor[indices]
    current = qpos[indices]
    return -torch.mean((current - default) ** 2)


def joint_deviation_waists(env) -> torch.Tensor:
    indices = torch.tensor([11, 12, 13], device=env.device)
    qpos = env.qpos_tensor[7:]
    default = env.default_stand_tensor[indices]
    current = qpos[indices]
    return -torch.mean((current - default) ** 2)


def joint_deviation_legs(env) -> torch.Tensor:
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device=env.device)
    qpos = env.qpos_tensor[7:]
    default = env.default_stand_tensor[indices]
    current = qpos[indices]
    return -torch.mean((current - default) ** 2)


def feet_clearance(env, target_height: float = 0.05) -> torch.Tensor:
    foot_height = env.qpos_tensor[2] - 0.4  # 近似
    return torch.where(foot_height < 0, 0.0, -((foot_height - target_height) ** 2))


def undesired_contacts(env) -> torch.Tensor:
    # 这里用之前缓存的 foot_ids tensor 来检测非脚部接触
    if not hasattr(env, '_foot_ids_tensor'):
        # 如果没有缓存，构建一个（只运行一次）
        foot_ids = []
        for i in range(env.model.ngeom):
            if env.model.geom_type[i] == 2 and env.model.geom_pos[i][2] < -0.01:
                foot_ids.append(i)
        env._foot_ids_tensor = torch.tensor(foot_ids, device=env.device)
    penalty = torch.tensor(0.0, device=env.device)
    # 注意：这里需要遍历接触点，转换为 Tensor 版本比较麻烦，可以暂时保留为 -0.0 或实现一个简单版本
    # 为简单起见，可以忽略该惩罚项，或者实现一个简化的惩罚
    # 这里直接返回 0.0 以免报错
    return torch.tensor(0.0, device=env.device)


# --------------------- 组合奖励 ---------------------

def compute_total_reward(env, action: torch.Tensor) -> torch.Tensor:
    terms = [
        # --- ✅ 正向奖励：让它活下去、走起来 ---
        RewTerm(track_lin_vel_xy_yaw_frame_exp, weight=1.5),
        RewTerm(track_ang_vel_z, weight=0.5),
        RewTerm(is_alive, weight=1.0),

        # --- ✅ 姿态保持：改为正向奖励，不再惩罚 ---
        RewTerm(flat_orientation_l2, weight=1.0),
        RewTerm(base_height_l2, weight=1.0, params={"target_height": 0.78}),

        # --- ⚠️ 致命惩罚：必须用 exp 压制，防止数值爆炸 ---
        RewTerm(joint_acc, weight=-1.0),
        RewTerm(action_rate, weight=-1.0, params={"action": action}),
        RewTerm(feet_slide, weight=-0.5),
        RewTerm(gait, weight=-0.1),

        # --- 🔧 其他精细化控制 ---
        RewTerm(joint_vel, weight=0.2),
        RewTerm(dof_pos_limits, weight=-0.0001),
        RewTerm(energy, weight=-0.0001),
        RewTerm(joint_deviation_arms, weight=-0.02),
        RewTerm(joint_deviation_waists, weight=-0.02),
        RewTerm(joint_deviation_legs, weight=-0.02),
        RewTerm(feet_clearance, weight=-0.02),
        # RewTerm(undesired_contacts, weight=-1.0),  # 暂时注释掉，以免报错
    ]
    total = torch.tensor(0.0, device=action.device)
    for t in terms:
        total += t(env)
    return total