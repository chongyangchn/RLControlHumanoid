import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # 策略网络 (Actor): 负责输出动作的均值 (Mean)
        # 具身智能中通常使用 ELU 或 Tanh，比 ReLU 在控制任务中更平滑
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()       # 将动作限制在 [-1, 1] 之间，便于对应底盘关节限位
        )

        # 价值网络 (Critic): 输入状态 -> 输出该状态的期望得分 (V值)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

        # 动作的标准差 (Action variance): 决定了探索的幅度
        # 初始值设为 -0.5 (exp(-0.5)约为0.6)，随着训练缩小，动作会趋于稳定
        # 动作标准差: 表示探索的动力。初期给大一点，后期自动收敛
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.2)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """采样动作，用于训练过程中的探索"""
        action_mean = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        action = dist.sample()
        log_prob  = dist.log_prob(action).sum(dim=-1)
        state_val = self.critic(state)
        return action, log_prob, state_val

    def evaluate(self, state, action):
        """评估已有动作，用于 PPO 损失计算"""
        action_mean = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        log_probs  = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)

        return log_probs , state_values, dist_entropy
