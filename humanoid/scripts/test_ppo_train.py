#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/15 00:00
# @Author  : cychn
# @File    : test_ppo_train.py
# @Software: PyCharm

"""

"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 解决 libiomp5md.dll 冲突

import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import yaml

from humanoid.envs.g1_env import HumanoidG1Env
from humanoid.algorithm.ppo.agent_ppo import Actor


class ObsNormalizer:
    def __init__(self, state=None, epsilon=1e-8):
        if state is None:
            self.mean = None
            self.var = None
            self.epsilon = epsilon
        else:
            self.mean = np.asarray(state["mean"], dtype=np.float32)
            self.var = np.asarray(state["var"], dtype=np.float32)
            self.epsilon = state.get("epsilon", epsilon)

    def normalize(self, obs):
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if self.mean is None or self.var is None:
            return obs
        return (obs - self.mean) / (np.sqrt(self.var) + self.epsilon)

# ---------- 配置 ----------
with open(r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\humanoid\configs\ppo_walking.yaml", "r",
          encoding="utf-8") as f:
    config = yaml.safe_load(f)

DETERMINISTIC = True
RENDER = True
EPISODE_STEPS = 2000

MODEL_PATH = r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\outputs\model\ppo_train\g1_actor_7200.pth"



# ---------- 加载环境 ----------
xml_path = r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\resources\robots\unitree_g1\scene.xml"
env = HumanoidG1Env(xml_path, config)
obs_dim = env.obs_dim
action_dim = env.action_dim

# ---------- 加载模型 - ---------
device = torch.device("cpu") # 测试通常用 CPU 就够了
actor = Actor(obs_dim, action_dim, hidden_dim=256).to(device)
# checkpoint = torch.load(MODEL_PATH, map_location=device)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
if 'actor_state_dict' in checkpoint:
    actor.load_state_dict(checkpoint['actor_state_dict'])
    obs_normalizer = ObsNormalizer(checkpoint.get("obs_normalizer"))
    if "obs_normalizer" not in checkpoint:
        print("[WARNING] 这个模型没有保存 obs_normalizer。建议用新训练脚本重新训练后再测试。")
else:
    actor.load_state_dict(checkpoint)
    obs_normalizer = ObsNormalizer()
    print("[WARNING] 这个模型是旧格式，只包含 actor 权重。建议用新训练脚本重新训练后再测试。")
actor.eval()  # 切换到评估模式（主要是关闭 dropout 等，这里不影响）


# ---------- 测试循环 ----------
obs = env.reset()
total_reward = 0.0

if RENDER:
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    # 可选设置视角
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    viewer.cam.distance = 3.0

for step in range(EPISODE_STEPS):
    obs_for_policy = obs_normalizer.normalize(obs)
    obs_tensor = torch.as_tensor(obs_for_policy, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"当前机器人的基座高度是：{obs[0]:.4f} 米")
    print(f"当前机器人的四元数 qx 分量是：{obs[2]:.4f}")

    with torch.no_grad():
        mu, _ = actor(obs_tensor)
        action = mu.squeeze(0).cpu().numpy()
        print(f"action = {action}")

    # 环境步进
    next_obs, reward, done, info = env.step(action)
    total_reward += float(reward.detach().cpu())

    if RENDER:
        viewer.sync()

    time.sleep(env.dt * 0.5)

    if done:
        print(f"Episode ended at step {step}, total reward = {total_reward:.2f}")
        time.sleep(5)
        break

    obs = next_obs

print(f"Test finished. Total reward: {total_reward:.2f}")

if RENDER:
    time.sleep(2)
    viewer.close()
