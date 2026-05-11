import torch
import torch.nn as nn
from .actor_critic import ActorCritic

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.state_values = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.state_values[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPOAgent:
    def __init__(self, state_dim, action_dim, config, device):
        self.device = device
        self.gamma = config['gamma']                # 折扣因子
        self.lam = config['lam']                    # GAE 参数
        self.eps_clip = config['eps_clip']          # PPO 裁剪范围
        self.k_epochs = config['k_epochs']          # 每次更新迭代次数

        self.policy = ActorCritic(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['lr'])

        # policy_old 用于计算重要性采样中的概率比值
        self.policy_old = ActorCritic(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def selection_action(self, state):
        """给 train.py 调用：根据当前状态选动作"""
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, log_prob, state_val = self.policy_old.act(state)
        # 转换为 CPU 并去掉 batch 维度 (假设 batch=1)
        action = action.squeeze(0).cpu()
        log_prob = log_prob.squeeze(0).cpu()
        state_val = state_val.squeeze(0).cpu()  # 标量
        # 移除 batch 维度 (假设网络输出 batch_size=1)
        return action, log_prob, state_val

    def update(self, buffer):
        """核心更新逻辑"""
        # 1. 转换 Buffer 里的数据
        old_states = torch.stack(buffer.states).to(self.device)
        old_actions = torch.stack(buffer.actions).to(self.device)
        old_log_probs = torch.stack(buffer.log_probs).to(self.device)
        old_state_values = torch.stack(buffer.state_values).squeeze(-1).to(self.device)

        # 2. 计算 GAE (优势估计)
        # 这部分代码模拟了控制系统中的“误差前馈”
        advantages = []
        gae = 0

        # 逆序计算，因为当前优势依赖于未来奖励
        for i in reversed(range(len(buffer.rewards))):
            mask = 1.0 - buffer.is_terminals[i]
            # TD Error
            delta = buffer.rewards[i] + self.gamma * old_state_values[i+1] * mask - old_state_values[i]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + old_state_values[:-1]
        # 归一化 Advantage 是稳定人形机器人步态的关键
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. 策略优化 (K 次迭代)
        for _ in range(self.k_epochs):
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 计算重要性采样率 (Ratio)
            ratios = torch.exp(log_probs - old_log_probs)

            # PPO Clip 损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # 总损失 = 策略损失 + 价值损失 - 熵损失
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.MseLoss(state_values.squeeze(), returns)
            entropy_loss = -0.01 * dist_entropy.mean()
            loss = policy_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # 同步权重
        self.policy_old.load_state_dict(self.policy.state_dict())
        buffer.clear()
        return loss.item()