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

import wandb
import torch
import numpy as np
import yaml
from humanoid.algorithm.ppo.agent_ppo import PPOAgent, RolloutBuffer
from humanoid.envs.g1_env import HumanoidG1Env
from humanoid.utils.logger import RLLogger
import time

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.epsilon = epsilon

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            x = x.reshape(1)
        if self.mean.shape != () and x.shape == self.mean.shape:
            x = x.reshape(1, *self.mean.shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)) / (self.count + batch_count)
        self.count += batch_count

    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            # 关键修复：将 numpy 数据转为�?x 相同设备�?tensor
            device = x.device
            mean_tensor = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
            std_tensor = torch.as_tensor(np.sqrt(self.var) + self.epsilon, dtype=torch.float32, device=device)
            return (x - mean_tensor) / std_tensor
        else:
            return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)

    def state_dict(self):
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = int(state.get("count", 0))
        self.epsilon = state.get("epsilon", self.epsilon)

class RewardNormalizer:
    def __init__(self, shape=(), epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.epsilon = epsilon

    def update(self, x):
        # --- 核心修复：兼容标量、tensor、numpy ---
        if isinstance(x, torch.Tensor):
            # 标量 Tensor (0�? 转为 1�?
            if x.ndim == 0:
                x = x.view(-1)
            x = x.detach().cpu().numpy()
        elif isinstance(x, (int, float)):
            x = np.array([x])
        elif isinstance(x, list):
            x = np.array(x)

        # 确保 x �?numpy 数组
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float64)

        # 如果 x 是标�?numpy 数组（形状为 ()），�?reshape �?(1,)
        if x.ndim == 0:
            x = x.reshape(1)

        # 获取批次大小
        batch_size = x.shape[0]

        # Welford 算法更新均值和方差
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)

        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_size / (self.count + batch_size)
        m_a = self.var * self.count
        m_b = batch_var * batch_size
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_size / (self.count + batch_size)) / (
                    self.count + batch_size)
        self.count += batch_size

    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=torch.float32, device=x.device)
            std = torch.as_tensor(np.sqrt(self.var) + self.epsilon, dtype=torch.float32, device=x.device)
            return (x - mean) / std
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)


def test():
    start = time.time()

    # wandb.init(
    #     project="g1_ppo_walking",
    #     config={
    #         "learning_rate": 1e-4,
    #         "batch_size": 256,
    #         "num_envs": 1,  # 你的单环�?
    #         "gamma": 0.99,
    #         # 其他超参�?..
    #     }
    # )

    with open("configs/ppo_walking.yaml", "r",
              encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger_dir = config["path"]["logger_dir"]
    model_dir = config["path"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True) # 确保文件夹存�?

    # 1. 初始�?Logger
    logger = RLLogger(log_dir=logger_dir)

    xml_path = "resources/robots/unitree_g1/scene.xml"
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

    # 初始化奖励归一化器
    reward_norm = RewardNormalizer()
    obs_normalizer = RunningMeanStd(shape=(env.obs_dim,))

    resume_from = config["train"].get("resume_from")
    start_episode = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=agent.device, weights_only=False)
        if "actor_state_dict" in checkpoint:
            agent.actor.load_state_dict(checkpoint["actor_state_dict"])
            if "critic_state_dict" in checkpoint:
                agent.critic.load_state_dict(checkpoint["critic_state_dict"])
            if config["train"].get("resume_optimizer", False) and "optimizer_state_dict" in checkpoint:
                agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if not config["train"].get("reset_obs_normalizer", False) and "obs_normalizer" in checkpoint:
                obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
            start_episode = int(checkpoint.get("episode", 0)) + 1
            logger.info(f"�?checkpoint 继续训练: {resume_from}")
        else:
            agent.actor.load_state_dict(checkpoint)
            logger.info(f"加载旧格�?actor 权重: {resume_from}")

    for episode in range(start_episode, max_episodes):
        obs = env.reset()
        # ----- 新增：对初始观测进行归一�?-----
        obs = obs_normalizer.normalize(obs)
        rollout_reward = 0.0
        completed_episode_rewards = []
        completed_episode_lengths = []
        current_episode_reward = 0.0
        current_episode_length = 0

        for step in range(steps_per_epoch):
            # 1. 采集数据
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            # ----- 新增：更新观测归一化统计量，再归一化下一时刻观测 -----
            obs_normalizer.update(next_obs)  # 用新观测更新统计�?
            next_obs = obs_normalizer.normalize(next_obs)  # 再归一化，存入buffer

            # ----------- 修改核心：奖励归一�?-----------
            reward_norm.update(reward)
            # 2. 获得归一化后的奖�?
            normalized_reward = reward_norm.normalize(reward)
            buffer.store(obs, action, normalized_reward, log_prob, value, done)
            # ------------------------------------------------
            obs = next_obs
            reward_value = float(reward.detach().cpu())
            rollout_reward += reward_value
            current_episode_reward += reward_value
            current_episode_length += 1

            if done:
                completed_episode_rewards.append(current_episode_reward)
                completed_episode_lengths.append(current_episode_length)
                current_episode_reward = 0.0
                current_episode_length = 0
                obs = env.reset()  # 注意：这里重置后继续收集，无需特殊处理, �?done 会在 GAE 中影响计�?                obs = obs_normalizer.normalize(obs)

        # 2. 计算 GAE 并更新网�?        with torch.no_grad():
            _, _, last_value = agent.get_action(obs) # obs最后一步之后的 next_obs
        buffer.finish_path(last_value)
        action_array = np.asarray(buffer.actions, dtype=np.float32)
        action_abs_mean = float(np.mean(np.abs(action_array)))
        action_abs_max = float(np.max(np.abs(action_array)))

        # 3. PPO更新
        loss_dict = agent.update(buffer)
        buffer.clear()

        # 3. 计算并记录日�?
        mean_episode_reward = float(np.mean(completed_episode_rewards)) if completed_episode_rewards else current_episode_reward
        mean_episode_length = float(np.mean(completed_episode_lengths)) if completed_episode_lengths else current_episode_length
        done_rate = len(completed_episode_lengths) / max(1, steps_per_epoch)
        logger.log_scalar("Train/RolloutReward", rollout_reward, episode)
        logger.log_scalar("Train/EpisodeReward", mean_episode_reward, episode)
        logger.log_scalar("Train/EpisodeLength", mean_episode_length, episode)
        logger.log_scalar("Train/DoneCount", len(completed_episode_lengths), episode)
        logger.log_scalar("Train/DoneRate", done_rate, episode)
        logger.log_scalar("Train/CommandLinVelX", env.command_lin_vel_x, episode)
        logger.log_scalar("Policy/ActionAbsMean", action_abs_mean, episode)
        logger.log_scalar("Policy/ActionAbsMax", action_abs_max, episode)
        logger.log_scalar("Loss/Actor", loss_dict["actor_loss"], episode)
        logger.log_scalar("Loss/Critic", loss_dict["critic_loss"], episode)

        checkpoint_interval = config["train"].get("checkpoint_interval", 300)
        if episode % checkpoint_interval == 0:
            save_path = os.path.join(model_dir, f"g1_actor_{episode}.pth")
            torch.save({
                'episode': episode,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'obs_normalizer': obs_normalizer.state_dict(),
                'config': config,
            }, save_path)
            logger.info(f"保存中间模型�? {save_path}")

        # if episode % 100 == 0:
        #     wandb.log({
        #         "episode_reward": episode_reward,
        #         "actor_loss": loss_dict['actor_loss'],
        #         "critic_loss": loss_dict['critic_loss']
        #         # "steps": total_steps
        #     })

    final_save_path = os.path.join(model_dir, "g1_actor_final_2026014.pth")
    # 建议同时保存 Actor �?Critic，方便以后“断点续训�?
    torch.save({
        'episode': config["train"]["max_episodes"],
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'obs_normalizer': obs_normalizer.state_dict(),
        'config': config,
    }, final_save_path)
    logger.info(f"训练完成！最终模型已保存�? {final_save_path}")

    print(f"Total time for {max_episodes} episodes: {time.time() - start:.2f} sec")

if __name__ == "__main__":
    test()

