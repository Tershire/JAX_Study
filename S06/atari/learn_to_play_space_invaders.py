# learn_to_play_space_invaders.py

# Arz
# 2024 JUN 03 (MON)

"""
train agent to play <space invaders>
"""

# reference:


import gymnasium
import numpy as np
import matplotlib.pyplot as plt
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
max_num_episodes = 200

# training
agent.train(max_num_episodes)

# training result
def compute_average_cumulative_rewards(agent, M=10):
    average_cumulative_rewards = []
    for i in range(M, len(agent.cumulative_rewards) + 1):
        average_cumulative_reward = np.mean(agent.cumulative_rewards[i - M:i])
        average_cumulative_rewards.append(average_cumulative_reward)

    return average_cumulative_rewards

plt.plot(np.arange(1, len(agent.cumulative_rewards) + 1), agent.cumulative_rewards)
M = 10
plt.plot(np.arange(M, len(agent.cumulative_rewards) + 1),
         compute_average_cumulative_rewards(agent, M))
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.grid()
plt.show()
