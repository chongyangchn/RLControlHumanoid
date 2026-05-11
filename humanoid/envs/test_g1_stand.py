#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/5 21:31
# @Author  : cychn
# @File    : test_g1_stand.py
# @Software: PyCharm

"""
该文件用于测试G1机器人保持站立以确保env接口的准确性
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from humanoid.envs.g1_env import HumanoidG1Env



def test_stand_and_move(xml_path):
    env = HumanoidG1Env(xml_path)
    print(f"G1人形机器人的动作控制量维度为：{env.action_space_dim}\n")
    print("开始环境验证...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs = env.reset()  # 重置环境
        start_time = time.time()  # 记录初始时间
        stage = "STANDING"  # 阶段标记  # STANDING -> ARM_MOVE
        while viewer.is_running():
            t = time.time() - start_time
            step_time = time.time()

            if stage == "STANDING":
                action = np.zeros(env.action_space_dim)
                if time.time() - start_time > 5.0:
                    stage = "ARM_MOVE"
                    print("已站立 5 秒，开始手臂测试...")
            elif stage == "ARM_MOVE":
                # 手臂移动阶段：让特定的 actuator 动起来
                action = np.zeros(env.action_space_dim)
                # 假设 G1 的手臂 actuator 索引在最后几位（需查阅 XML）
                # 我们给一个随时间变化的正弦信号
                t = time.time()
                action[-1] = np.sin(t*2)*0.5 # 最后一个关节来回摆动

            # 执行环境步进
            # 打印当前高度和奖励，观察数值变化
            obs, reward, done, _ = env.step(action)
            if int(t * 10) % 20 == 0: # 每 2 秒打印一次
                print(f"当前高度: {env.data.qpos[2]:.3f} | 奖励值: {reward:.3f}")
            viewer.sync()
            time.sleep(0.01)  # 减慢演示速度
            if t > 8:
                break
        print("环境验证完成，渲染正常且逻辑闭环。")

def hand_ctrl_G1(xml_path):
    env = HumanoidG1Env(xml_path)
    print("动作维度 =", env.model.nu)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs = env.reset()  # 重置环境
        start_time = time.time()  # 记录初始时间
        print("初始高度 =", env.data.qpos[2])

        # 构造一套粗略站立基准姿态 (29维全0基础上，只弯膝盖)
        stand_ctrl = np.zeros(env.model.nu)
        # 左膝、右膝弯曲一点，保持直立不塌
        stand_ctrl[3] = 0
        stand_ctrl[9] = 0

        while viewer.is_running():
            t = time.time()
            dt = t - start_time
            # 直接给动作！让机器人膝盖弯曲
            # 索引 3 = 左膝，索引 9 = 右膝

            # 运行100步
            for i in range(1000):
                env.data.ctrl[:] = stand_ctrl
                mujoco.mj_step(env.model, env.data)
                print("高度 =", env.data.qpos[2])
                if i % 100 == 0:
                    print(f"step {i}, 高度: {env.data.qpos[2]:.4f}")

            viewer.sync()
            time.sleep(0.01)  # 减慢演示速度
            if dt > 10:
                # 打印已经稳定后的关节位置 qpos（去掉根坐标前7维）
                print("稳定后关节姿态 qpos[7:]：")
                print(np.round(env.data.qpos[7:], 4))
                print("长度:", len(env.data.qpos[7:]))
                break

if __name__ == "__main__":
    xml_path = r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\resources\robots\unitree_g1\scene.xml"
    # test_stand_and_move(xml_path)
    hand_ctrl_G1(xml_path)