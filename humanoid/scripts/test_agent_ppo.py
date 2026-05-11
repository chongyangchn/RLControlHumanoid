#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/10 01:41
# @Author  : cychn
# @File    : test_agent_ppo.py
# @Software: PyCharm

"""
在写正式训练代码前，先运行这个脚本，确保 环境 -> 智能体 -> 动作输出 链路是通的，且没有维度冲突。
"""
from mpmath.libmp import agm_fixed

from humanoid.envs.g1_env import HumanoidG1Env
from humanoid.algorithm.ppo.agent_ppo import PPOAgent
import numpy as np
import yaml

def test():
    xml_path = r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\resources\robots\unitree_g1\scene.xml"
    with open(r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\humanoid\configs\ppo_walking.yaml", "r",
              encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # 注意：这里的 xml 路径请改为你电脑上的实际路径
    env = HumanoidG1Env(xml_path, config)
    obs = env.reset()

    # 2. 初始化智能体
    obs_dim = len(obs)
    action_dim = env.action_dim
    cfg = config["ppo"]
    agent = PPOAgent(obs_dim, action_dim, cfg)
    print(f"检测到观测维度: {obs_dim}, 动作维度: {action_dim}")


    for i in range(100):
        # 3. 获取动作
        action, log_prob, val = agent.get_action(obs)
        action = action.flatten()  # 将 (1, act_dim) 转为 (act_dim,)
        # 4. 执行动作
        next_obs, reward, done, _ = env.step(action)

        if i % 10 == 0:
            print(f"Step {i} | Action[0]: {action[0]:.2f} | Reward: {reward:.2f}")

        obs = next_obs
        if done:
            print("机器人摔倒，重置环境...")
            obs = env.reset()
    print("测试成功：链路完全打通！")


if __name__ == "__main__":
    test()
