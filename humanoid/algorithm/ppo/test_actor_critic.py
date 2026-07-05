import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from actor_critic import ActorCritic


ac = ActorCritic(num_actor_obs=98, num_critic_obs=98, num_actions=29)

obs = torch.rand(4, 98)
action = ac.act(obs)
value = ac.evaluate(obs)
print("Action:", action.shape)
print("Critic:", value.shape)


# import inspect
# print("ActorCritic 类的来源文件:", inspect.getfile(ActorCritic))
# print("ActorCritic 的 __init__ 签名:", inspect.signature(ActorCritic.__init__))