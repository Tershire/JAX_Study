# benchmark_escape_maze.py

# Arz
# 2024 MAY 19 (SUN)

"""escape maze using RL."""

# reference:


import numpy as np
import maze_environment
import learning_algorithms
import learning_algorithms_jax
import jax
import matplotlib.pyplot as plt


# environment setting
wall_positions = np.array([(1, 3), (1, 4), (2, 1), (3, 1), (3, 3), (4, 3)])
env = maze_environment.Maze((5, 5), wall_positions)

# initialize environment
env.reset()
print(env.render("cell_name"))

# initialize agent
learning_algorithm = "Q-Learning"
alpha = 0.1  # learning rate
gamma = 0.99  # discount
epsilon = 0.1  # epsilon-greedy param
match learning_algorithm:
    case "Q-Learning":
        agent = learning_algorithms.Q_Learning(env, alpha, gamma, epsilon)
        agent_jax = learning_algorithms_jax.Q_Learning(env, alpha, gamma, epsilon)
    case "SARSA":
        agent = learning_algorithms.SARSA(env, alpha, gamma, epsilon)

# training
max_num_episodes = 1000

agent.learn(max_num_episodes)
print("training complete.")

key = jax.random.key(0)
agent_jax.learn(max_num_episodes, key)
print("training complete.")

# training result
print(agent.q_table)
print(agent_jax.q_table)
