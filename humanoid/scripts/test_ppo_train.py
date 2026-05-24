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

# ---------- 配置 ----------
with open(r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\humanoid\configs\ppo_walking.yaml", "r",
          encoding="utf-8") as f:
    config = yaml.safe_load(f)

DETERMINISTIC = True
RENDER = True
EPISODE_STEPS = 2000

MODEL_PATH = r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\outputs\model\ppo_train\g1_actor_3900.pth"



# ---------- 加载环境 ----------
xml_path = r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\resources\robots\unitree_g1\scene.xml"
env = HumanoidG1Env(xml_path, config)
obs_dim = env.obs_dim
action_dim = env.action_dim

# ---------- 加载模型 - ---------
device = torch.device("cpu") # 测试通常用 CPU 就够了
actor = Actor(obs_dim, action_dim, hidden_dim=256).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'actor_state_dict' in checkpoint:
    actor.load_state_dict(checkpoint['actor_state_dict'])
else:
    actor.load_state_dict(checkpoint)
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
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        mu, _ = actor(obs_tensor)
        action = mu.squeeze(0).cpu().numpy()
        print(f"action = {action}")

    # 环境步进
    next_obs, reward, done, info = env.step(action)
    total_reward += reward

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