# learn_to_control_cartpole_pytorch.py

# Arz
# 2024 AUG 02 (FRI)

"""
train agent to control cartpole.
"""
import torch

# reference:
# - https://github.com/seungeunrho/minimalRL/blob/master/dqn.py


import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from learning_algorithms_pytorch import DQN
from pathlib import Path


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
learning_rate = 1.0E-4  # learning rate
memory_capacity = int(2E4)  # replay memory capacity
agent = DQN(env, learning_rate, memory_capacity)

# training or test
match mode:
    case "train":
        max_num_episodes = 100
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
