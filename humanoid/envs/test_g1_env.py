#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/22 23:53
# @Author  : cychn
# @File    : test_g1_env.py
# @Software: PyCharm

"""
G1环境的测试以及状态向量和动作向量的获取
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


# ---------- 加载环境 ----------
xml_path = r"Y:\RobotTransition\Project\cyRobotic\RLControlHumanoid\resources\robots\unitree_g1\scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
print(model.ngeom)

# 获取左右脚踝连杆的 Body ID
left_ankle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
right_ankle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

print(f"left_ankle_body_id is {left_ankle_body_id}")  # 7
print(f"right_ankle_body_id is {right_ankle_body_id}")  # 13

# # print("开始验证脚部几何体...")
# for i in range(model.ngeom):
#
# # G1Robot = HumanoidG1Env(xml_path, config)

print("=== 29个控制/状态关节的顺序 (从左到右) ===")


