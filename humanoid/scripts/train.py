#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/4 15:58
# @Author  : cychn
# @File    : train.py
# @Software: PyCharm

"""

"""

import os
# 解决 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from humanoid.envs.g1_env import HumanoidG1Env
from humanoid.algorithm.ppo.agent_ppo import PPOAgent, RolloutBuffer
from utils.logger import RLLogger
import yaml
import time
import mujoco.viewer
from tqdm import tqdm

def train():

    # 1. 加载配置 (这里假设你已经有了 configs/ppo_walking.yaml)
    with open("configs/ppo_walking.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 初始化环境、智能体和日志
    render_interval = 100  # 每 50 个 episode 渲染一次，让你看看它走得怎么样

    xml_path = config['xml_path']
    env = HumanoidG1Env(xml_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 自动获取维度
    state_dim = env.observation_space_dim
    action_dim = env.action_space_dim
    agent = PPOAgent(state_dim, action_dim, config, device)
    buffer = RolloutBuffer()
    logger = RLLogger(log_dir=config['log_dir'])

    # 3. 训练循环变量
    max_episodes = config['max_episodes']
    update_timestep = config['update_timestep'] # 每积累多少步更新一次，建议 2048
    timestep = 0

    # 设置训练过程中的渲染
    viewer = None  # 初始化 viewer 为 None
    render_interval = 50  # 每隔 50 个 episode 显示一次


    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        # 判断本回合是否需要可视化
        # render_this_episode = (episode % render_interval == 0)
        # if render_this_episode and viewer is None:
        #     viewer = mujoco.viewer.launch_passive(env.model, env.data)
        #     print("--- 已开启可视化窗口 ---")
        #     print(f"正在可视化第 {episode} 回合的表现...")

        for t in range(config['max_steps_per_episodes']):
            timestep += 1
            episode_steps += 1

            with torch.no_grad():
            # 选择动作并与环境交互
                action_tensor, log_prob, state_value = agent.selection_action(state)
                action_np = action_tensor.numpy().flatten()

            next_state, reward, done, _ = env.step(action_np)

            # 存入缓存(注意这里存的是 tensor)
            buffer.states.append(torch.FloatTensor(state))
            buffer.actions.append(action_tensor)
            buffer.log_probs.append(log_prob)
            buffer.state_values.append(state_value)
            buffer.rewards.append(reward)
            buffer.is_terminals.append(done)

            state = next_state
            episode_reward = episode_reward + reward

            # 定时更新策略
            if timestep % update_timestep == 0:
                # 传入最后一个状态的 V 值用于处理非正常结束（Time-limit truncation)
                with torch.no_grad():
                    last_state_tensor = torch.FloatTensor(state).to(device)
                    last_val = agent.policy.critic(last_state_tensor).detach().cpu().squeeze()
                buffer.state_values.append(last_val)

                loss = agent.update(buffer)
                logger.log_scalar("Train/Loss", loss, timestep)
                logger.info(f"Step {timestep} | 策略已更新 | 当前 Loss: {loss:.4f}")

            # # 关键：只有在本回合需要渲染时，才调用 sync()
            # if render_this_episode and viewer is not  None:
            #     if viewer.is_running():
            #         viewer.sync()# 同步渲染画面
            #         # 控制渲染速度，否则训练太快，肉眼看不清动作
            #         time.sleep(env.model.opt.timestep)
            #     else:
            #         viewer = None # 如果用户手动关了窗口，下次需要时再开

            if done:
                break

        # 记录每回合奖励
        logger.log_scalar("Train/EpisodeRewarde", episode_reward, episode)

        if episode % 10 == 0:
            logger.info(f"Episode {episode} \t 奖励: {episode_reward:.2f}")

        if episode % 100 == 0:
            save_path = os.path.join(config['log_dir'], f"ppo_g1_{episode}.pt")
            torch.save(agent.policy.state_dict(), save_path)
            logger.info(f"模型已保存至: {save_path}")

    logger.close()
    # viewer.close()


def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


if __name__ == "__main__":
    train()
    # check_device()