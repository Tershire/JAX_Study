# learn_to_control_cartpole.py

# Arz
# 2024 JUN 15 (MON)

"""
train agent to control cartpole
"""

# reference:


import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from learning_algorithms import DQN
from flax import nnx


# load environment
render_mode = "rgb_array"
do_render = False
if do_render:
    render_mode = "human"

env = gymnasium.make("CartPole-v1", render_mode=render_mode)

# load agent
alpha = 1.0E-4  # learning rate
memory_capacity = int(1E4)  # replay memory capacity
agent = DQN(env, alpha, memory_capacity, rngs=nnx.Rngs(0))

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
