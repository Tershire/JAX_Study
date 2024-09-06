# learn_to_control_cartpole_pytorch.py

# Arz
# 2024 AUG 02 (FRI)

"""
train agent to escape maze.
"""
import torch

# reference:


import maze_environment
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
wall_positions = np.array([(1, 3), (1, 4), (2, 1), (3, 1), (3, 3), (4, 3)])
env = maze_environment.Maze((5, 5), wall_positions)

print(env.render("cell_name"))

# load agent
learning_rate = 1E-4  # learning rate
memory_capacity = int(1E4)  # replay memory capacity
agent = DQN(env, learning_rate, memory_capacity)

# training or test
match mode:
    case "train":
        max_num_episodes = 1200
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


def build_q_table(env, agent):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    for state_id in range(q_table.shape[0]):
        state = agent.one_hot_encode(state_id)
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_value = agent.q_estimator(state)
        q_table[state_id] = q_value.cpu()

    return q_table

q_table = build_q_table(env, agent)
print(q_table)


arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
def display_greedy_action_per_state(q_table, env):
    grid = np.full(env.grid_size, '')
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            state = env.observation_space.compute_observation(np.array([i, j]))
            action = np.argmax(q_table[state])
            grid[i, j] = arrows[action]

            # case when all the actions have the equal value
            if np.all(q_table[state] == q_table[state][0]):
                grid[i, j] = '*'

    print(grid)

display_greedy_action_per_state(q_table, env)

if mode == "test":
    def display_escape_route(state_trajectory, action_trajectory):
        grid = np.full(env.grid_size, ' ')
        for i, (agent_position, action) in enumerate(zip(state_trajectory, action_trajectory)):
            grid[agent_position] = arrows[action]

        print(grid)

    display_escape_route(state_trajectory, action_trajectory)
