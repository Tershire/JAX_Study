# benchmark_escape_maze.py

# Arz
# 2024 MAY 19 (SUN)

"""escape maze using RL."""

# reference:
# https://blog.joonas.io/239  # time benchmark


import numpy as np
import maze_environment
import learning_algorithms
import learning_algorithms_jax
import jax
import matplotlib.pyplot as plt
import time


# check JAX status
print(f"Jax backend: {jax.default_backend()}")

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

agent_jax.learn(max_num_episodes)
print("training complete.")


# training result
# 1. Q-table
# print(agent.q_table)
# print(agent_jax.q_table)

# 2. reward plot
def compute_average_cumulative_rewards(cumulative_rewards, M=10):
    average_cumulative_rewards = np.array([])
    for i in range(M, len(agent.cumulative_rewards) + 1):
        average_cumulative_reward = np.mean(agent.cumulative_rewards[i - M:i])
        average_cumulative_rewards = np.append(average_cumulative_rewards, average_cumulative_reward)

    return average_cumulative_rewards


def plot_reward_plot(agent, M=10):
    plt.plot(np.arange(1, len(agent.cumulative_rewards) + 1), agent.cumulative_rewards)
    plt.plot(np.arange(M, len(agent.cumulative_rewards) + 1),
             compute_average_cumulative_rewards(agent.cumulative_rewards, M=M))
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid()
    plt.show()


plot_reward_plot(agent, M=10)
plot_reward_plot(agent_jax, M=10)

# benchmark time
def measure_runtime(function):
    t_ini = time.time()
    function()
    t_fin = time.time()
    delta_t = t_fin - t_ini
    return delta_t * 1000


def benchmark_runtime(function, k=1, params: list=[{}]):
    ts = []
    for _ in range(k):
        for param in params:
            t = measure_runtime(lambda: function(**param))
            ts.append(t)
    average = sum(ts) / len(ts)
    del ts
    print(f">>> average {average:.5f} [ms]")


benchmark_runtime(agent.learn, k=1, params=[{"max_num_episodes": max_num_episodes}])

agent_jax.learn(1)  # warm-up
t2 = benchmark_runtime(agent_jax.learn, k=1, params=[{"max_num_episodes": max_num_episodes}])
