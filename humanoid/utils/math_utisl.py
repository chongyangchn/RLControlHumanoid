#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/4 19:16
# @Author  : cychn
# @File    : math_utisl.py
# @Software: PyCharm

"""
由于机器人观测涉及坐标系转换，这个工具类必不可少
"""

import numpy as np
import mujoco

def get_projected_gravity(model, data):
    """将世界坐标系的重力向量投影到机器人躯干局部坐标系"""
    gravity = np.array([0, 0, -1])
    # data.qpos[3:7] 是躯干的世界四元数
    quat = data.qpos[3:7]
    inv_quat = np.zeros(4)
    mujoco.mju_negQuat(inv_quat, quat) # 求逆四元数

    projected_gravity = np.zeros(3)
    mujoco.mju_rotVecQuat(projected_gravity, gravity, inv_quat)
    return projected_gravity