# escape_maze.py

# Arz
# 2024 MAY 13 (MON)

"""escape maze using RL."""

# reference:
# https://zoo.cs.yale.edu/classes/cs470/materials/hws/hw7/FrozenLake.html


import numpy as np
import maze_environment
import learning_algorithms
from enum import Enum


# environment setting
wall_positions = np.array([(1, 3), (1, 4), (2, 1), (3, 1), (3, 3), (4, 3)])
env = maze_environment.Maze((5, 5), wall_positions)

# initialize environment
env.reset()
print(env.render("cell_name"))

# initialize agent
agent = learning_algorithms.Q_Learning(env, 0.1, 0.99, 0.1)

# training
agent.learn(1000)
print("training complete.")

# training result
print(agent.q_table)

arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
def render_greedy_action_per_state(env, agent):
    grid = np.full(env.grid_size, '')
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            state = env.observation_space.compute_observation(np.array([i, j]))
            action = np.argmax(agent.q_table[state])
            grid[i, j] = arrows[action]

            # case when all the actions have the equal value
            if np.all(agent.q_table[state] == agent.q_table[state][0]):
                grid[i, j] = '*'

    print(grid)

render_greedy_action_per_state(env, agent)