#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/10 01:22
# @Author  : cychn
# @File    : agent_ppo.py
# @Software: PyCharm

"""
针对 Unitree G1 这种高维连续控制任务，智能体需要具备：
高斯策略 (Gaussian Policy)：输出均值和标准差，用于探索。
正交初始化 (Orthogonal Initialization)：这能显著提高仿人机器人训练的稳定性。
GAE (广义优势估计)：平衡方差与偏差。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# 1. 策略网络 (Actor)
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 将均值限制在 [-1, 1]
        )
        # 可学习的标准差 log_std，初始设为 -0.5 (对应 std 约为 0.6)
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def forward(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        return mu, std

# 2. 价值网络 (Critic)
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        return self.net(obs)


# 3. PPO 智能体主体
class PPOAgent:
    def __init__(self, obs_dim, action_dim, cfg):
        self.gamma = cfg["gamma"]
        self.lam = cfg["lam"]
        self.clip_epsilon = cfg["clip_epsilon"]
        self.lr = cfg["lr"]
        self.hidden_dim = cfg["hidden_dim"]
        self.batch_size = cfg["batch_size"]
        self.update_epochs = cfg["update_epochs"]


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(obs_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, self.hidden_dim).to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr}
        ])

        # 初始化：正交初始化有利于稳定
        self._init_weights(self.actor)
        self._init_weights(self.critic)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def get_action(self, obs, deterministic=False):
        # obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        # 修改后（兼容 CPU 和 CUDA）：
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        # 确保已经在正确的设备上，并增加 batch 维度
        obs = obs.to(self.device).unsqueeze(0)

        with torch.no_grad():
            mu, std = self.actor(obs)
            std = std + 1e-6 # 防止除零，也提供最小探索
            if deterministic:
                action = mu
                return action.squeeze(0).cpu().numpy(), None, None

            dist = Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(obs)

        return action.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu().numpy(), value.squeeze(0).cpu().item()

    def update(self, buffer):
        # 1. 提取 Buffer 中的所有数据并转为 Tensor
        # 这里将数据修改为tensor后就不能使用np数组了
        # states = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        # actions = torch.FloatTensor(np.array(buffer.actions)).to(self.device)
        # log_probs_old = torch.FloatTensor(np.array(buffer.log_probs)).to(self.device)
        # returns = torch.FloatTensor(np.array(buffer.returns)).to(self.device)
        # advantages = torch.FloatTensor(np.array(buffer.advantages)).to(self.device)

        # 安全地提取数据，兼容 Tensor 或 NumPy 元素
        def to_tensor(lst, device):
            # 如果列表里的第一个元素是 numpy 或 tensor，都转为 tensor 并堆叠
            tensors = [torch.as_tensor(item, dtype=torch.float32) for item in lst]
            return torch.stack(tensors).to(device)

        # 使用 to_tensor 辅助函数替代 torch.stack
        states = to_tensor(buffer.states, self.device)
        actions = to_tensor(buffer.actions, self.device)
        log_probs_old = to_tensor(buffer.log_probs, self.device)
        returns = to_tensor(buffer.returns, self.device)
        advantages = to_tensor(buffer.advantages, self.device)

        # 优势归一化：这能显著稳定仿人机器人的训练
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        for _ in range(self.update_epochs):
            # 生成随机索引用于 Mini-batch 更新
            indices = np.arange(len(buffer.states))
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                idx = indices[start: start + self.batch_size]

                # 计算当前的动作分布
                mu, std = self.actor(states[idx])
                dist = Normal(mu, std)
                log_probs_now = dist.log_prob(actions[idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # 计算 Ratio (重要性采样比率)
                ratio = torch.exp(log_probs_now - log_probs_old[idx])

                # PPO 核心：裁剪损失函数
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic 损失：均方误差
                values_now = self.critic(states[idx]).squeeze()
                critic_loss = nn.MSELoss()(values_now, returns[idx])

                # 总损失 (加入熵正则化鼓励探索)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

        # 返回平均 Loss 字典
        num_updates = self.update_epochs * (len(buffer.states) // self.batch_size)
        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates
        }

# 4 实现数据缓冲区
class RolloutBuffer:
    def __init__(self, cfg):
        self.gamma = cfg["gamma"]
        self.lam = cfg["lam"]
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.values = []

        self.returns = []
        self.advantages = []

    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def finish_path(self, last_value=0.0):
        n = len(self.rewards)
        returns = torch.zeros(n, dtype=torch.float32)
        advantages = torch.zeros(n, dtype=torch.float32)
        next_value = last_value   # 初始化为传入的最终状态价值
        gae = 0.0
        for t in reversed(range(n)):            # 从后往前倒序计算 GAE
            if self.dones[t]:                   # 如果这一步之后环境终止，则后续价值为0，且GAE不继续传递
                next_value = 0.0
                gae = 0.0                           # 关键：done后GAE重置，因为新的轨迹开始

            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            gae = delta + self.gamma * self.lam * (1.0 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]  # 回报 = 优势 + 当前时刻的价值
            next_value = self.values[t]

        self.advantages = advantages
        self.returns = returns

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.values = []
        self.returns = []
        self.advantages = []

