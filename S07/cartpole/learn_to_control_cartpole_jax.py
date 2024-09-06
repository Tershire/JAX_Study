# learn_to_control_cartpole_jax.py

# Arz
# 2024 SEP 06 (FRI)

"""
train agent to control cartpole.
"""

# reference:


import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from learning_algorithms_jax import DQN
from flax import nnx
from pathlib import Path


mode = "test"  # train | test
model_path = Path("./model/dqn.bin")
save_model = False
save_video = False

# load environment
render_mode = "rgb_array"
do_render = False
if do_render:
    render_mode = "human"

env = gymnasium.make("CartPole-v1", render_mode=render_mode)

# load agent
learning_rate = 1E-3  # learning rate
memory_capacity = int(1E4)  # replay memory capacity
agent = DQN(env, learning_rate, memory_capacity, rngs=nnx.Rngs(0))

# training or test
match mode:
    case "train":
        max_num_episodes = 200
        agent.train(max_num_episodes)

        if save_model:
            agent.save_model(model_path=model_path)

    case "test":
        if save_video:
            env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
        else:
            env = gymnasium.make("CartPole-v1", render_mode="human")
        agent = DQN(env, learning_rate, memory_capacity, rngs=nnx.Rngs(0))

        cumulative_reward, action_trajectory = agent.test(model_path=model_path, save_video=save_video)
        print("cumulative_reward:", cumulative_reward)

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
