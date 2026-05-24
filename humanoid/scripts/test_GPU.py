#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/5/13 21:54
# @Author  : cychn
# @File    : test_GPU.py
# @Software: PyCharm

"""

"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
# 在训练循环开始前记录时间
start = time.time()

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

print("test")