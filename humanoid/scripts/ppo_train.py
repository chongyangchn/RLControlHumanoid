#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/10 14:58
# @Author  : cychn
# @File    : ppo_train.py
# @Software: PyCharm

"""

"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 解决 libiomp5md.dll 冲突

import torch
import yaml
from humanoid.algorithm.ppo.agent_ppo import PPOAgent, RolloutBuffer
from humanoid.envs.g1_env import HumanoidG1Env
from humanoid.utils.logger import RLLogger

def test():
    with open(r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\humanoid\configs\ppo_walking.yaml", "r",
              encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger_dir = config["path"]["logger_dir"]
    model_dir = config["path"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True) # 确保文件夹存在

    # 1. 初始化 Logger
    logger = RLLogger(log_dir=logger_dir)

    xml_path = r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\resources\robots\unitree_g1\scene.xml"
    env = HumanoidG1Env(xml_path, config)
    obs_dim = env.obs_dim
    action_dim = env.action_dim

    agent = PPOAgent(obs_dim, action_dim, config["ppo"])
    buffer = RolloutBuffer(config["ppo"])


    max_episodes = config["train"]["max_episodes"]
    steps_per_epoch = config["train"]["steps_per_epoch"]
    print(f"max_episodes:{max_episodes}")
    print(f"max_episodes:{steps_per_epoch}")

    episode_rewards = []

    for episode in range(max_episodes):
        obs = env.reset()
        episode_reward = 0

        for t in range(steps_per_epoch):
            # 1. 采集数据
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.store(obs, action, reward, log_prob, value, done)
            obs = next_obs
            episode_reward += reward

            if done:
                obs = env.reset()  # 注意：这里重置后继续收集，无需特殊处理, 但 done 会在 GAE 中影响计算

        # 2. 计算 GAE 并更新网络
        with torch.no_grad():
            _, _, last_value = agent.get_action(obs) # obs最后一步之后的 next_obs
        buffer.finish_path(last_value)

        # 3. PPO更新
        loss_dict = agent.update(buffer)
        buffer.clear()

        # 3. 计算并记录日志
        logger.log_scalar("Train/Reward", episode_reward, episode)
        logger.log_scalar("Loss/Actor", loss_dict["actor_loss"], episode)
        logger.log_scalar("Loss/Critic", loss_dict["critic_loss"], episode)

        if episode % 50 == 0:
            save_path = os.path.join(model_dir, f"g1_actor_{episode}.pth")
            torch.save(agent.actor.state_dict(), save_path)
            logger.info(f"保存中间模型至: {save_path}")

    final_save_path = os.path.join(model_dir, "g1_actor_final.pth")
    # 建议同时保存 Actor 和 Critic，方便以后“断点续训”
    torch.save({
        'episode': config["train"]["max_episodes"],
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, final_save_path)
    logger.info(f"训练完成！最终模型已保存至: {final_save_path}")

if __name__ == "__main__":
    test()
