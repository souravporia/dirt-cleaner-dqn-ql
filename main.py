import numpy as np
import gymnasium as gym
import random
from gymnasium.spaces import Discrete, Box
import pygame
from agent import QLAgent, DQNAgent
from env import DirtEnv

env = DirtEnv(n=5, p=0.8, dirt_count=1)  # 80% empty space probability
obs = env.reset(seed=11)

agent = QLAgent(env, 0.1, 1, 0.01, 0.1, 0.9)
done = False

# Run
timestamps = 1000
while timestamps <= 1000:
    env.render()
    cur_obs = int(np.copy(env.obs))
    action = agent.get_action(cur_obs)
    obs, reward, done, _ = env.step(action)
    agent.update(cur_obs, action, reward, done, obs)
    if(done) :
        env.reset(seed=11)
    timestamps -= 1
env.close()
