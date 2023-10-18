from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import time
import math
import random
from typing import Tuple

# import gym
import gymnasium as gym
env = gym.make('CartPole-v1')

def policy(obs): return 1

for _ in range(5):
    obs = env.reset()
    for _ in range(80):
        actions = policy(*obs)
        obs, reward, done, info = env.step(actions)
        env.render()
        time.sleep(0.05)

env.close()
