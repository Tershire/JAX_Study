# learn_to_control_cartpole_pytorch_prodo.py

# Arz
# 2024 AUG 31 (FRI)

"""
train agent to control cartpole.
"""

# reference:
# - https://github.com/seungeunrho/minimalRL/blob/master/dqn.py


import torch
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from learning_algorithms_pytorch import DQN_Prodo


mode = "train"  # train | test
model_path = Path("./model/dqn.pt")
save_model = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"  # found GPU is slower for this implementation.
print(device)

# load environment
render_mode = "rgb_array"
do_render = False
if do_render:
    render_mode = "human"

env = gymnasium.make("CartPole-v1", render_mode=render_mode)

# load agent
learning_rate = 1.0E-3  # learning rate
memory_capacity = int(1E4)  # replay memory capacity
agent = DQN_Prodo(env, learning_rate, memory_capacity)

# training or test
match mode:
    case "train":
        max_num_episodes = 300
        agent.train(max_num_episodes)

        if save_model:
            agent.save_model(model_path=model_path)

    case "test":
        state_trajectory, action_trajectory = agent.test(model_path=model_path)

# result
if mode == "train":
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
