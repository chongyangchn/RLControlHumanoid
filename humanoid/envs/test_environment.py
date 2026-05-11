# -*- coding: utf-8 -*-
import time
import mujoco.viewer
import numpy as np

from g1_env import HumanoidG1Env

def verify_environment(xml_path):
    print("开始环境验证...")
    env = HumanoidG1Env(xml_path)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # 重置环境
        obs = env.reset()
        print(f"初始观测向量维度: {obs.shape}")

        for _ in range(1000):
            # 随机生成动作 (范围假设在 -1 到 1 之间)
            random_action = np.random.uniform(-1.0, 1.0, size=env.nu)

            # 执行动作
            obs, reward, done, _ = env.step(random_action)

            # 刷新渲染
            viewer.sync()
            # if done:
            #     print(" 机器人摔倒了， 重置环境...")
            #     env.reset()

            time.sleep(0.01) # 减慢演示速度


    print("环境验证完成，渲染正常且逻辑闭环。")



print("hello")
# xml_path = "unitree_g1/g1.xml"
xml_path = r"Y:/RobotTransition/Project/cyRobotic/RLControlHumanoid/envs/unitree_g1/scene.xml"
verify_environment(xml_path)