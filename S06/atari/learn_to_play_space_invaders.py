# learn_to_play_space_invaders.py

# Arz
# 2024 JUN 03 (MON)

"""
train agent to play <space invaders>
"""

# reference:


import gymnasium
from learning_algorithms import DQN
from flax import nnx


# load environment
env = gymnasium.make("ALE/SpaceInvaders-v5", obs_type="rgb", render_mode="rgb_array")

# load agent
memory_capacity = int(1E4)
alpha = 2.5E-4  # learning rate
gamma = 0.99
agent = DQN(env, alpha, gamma, memory_capacity, rngs=nnx.Rngs(0))

# training setting
max_num_episodes = 3

# training
agent.train(max_num_episodes)
