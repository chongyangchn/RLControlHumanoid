#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/30 15:54
# @Author  : cychn
# @File    : test_tqdm.py
# @Software: PyCharm

"""

"""

from tqdm import tqdm
import time

# total = 2_000_000
# for i in tqdm(range(total), desc="Training", unit="step"):
#     time.sleep(0.00001)  # 模拟计算

import tqdm
dir(tqdm)                     # 列出所有属性、方法
help(tqdm.tqdm)               # 查看 tqdm 类的详细文档

from stable_baselines3 import PPO
help(PPO)                     # 查看类的文档
help(PPO.learn)               # 查看 learn 方法的参数