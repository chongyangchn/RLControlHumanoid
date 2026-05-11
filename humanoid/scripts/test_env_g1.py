#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/8 23:00
# @Author  : cychn
# @File    : test_env_g1.py
# @Software: PyCharm

"""
测试宇树机器人脚本文件
"""
import os
# 解决 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mujoco
import mujoco.viewer
import numpy as np
import time
import yaml
from humanoid.envs.g1_env import HumanoidG1Env

def test(xml_path, config):
    env = HumanoidG1Env(xml_path, config)
    print("开始环境验证...")
    print(f"G1人形机器人的动作控制量维度为：{env.nu}")   # 29
    print(f"G1人形机器人的关节数量为：{env.num_joints}") # 29
    print(f"G1人形机器人的关节位置状态为：{env.data.qpos}，该数组长度为：{len(env.data.qpos)}")   # 29个受控关节的当前角度 + 7个自由基座位姿（前3位是 XYZ 坐标，后4位是四元数姿态）。  # 36(7 + 29)
    print(f"G1人形机器人的关节速度状态为：{env.data.qvel}，该数组长度为：{len(env.data.qpos)}")   # 35(6 + 29)
    print(f"G1人形机器人的关节速度状态为：{env.joint_low}") # 关节限位
    print(f"G1人形机器人的关节速度状态为：{env.joint_high}") # 关节限位

    env.reset()
    env._get_obs()


if __name__ == "__main__":
    xml_path = r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\resources\robots\unitree_g1\scene.xml"
    with open(r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\humanoid\configs\ppo_walking.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    test(xml_path, config)