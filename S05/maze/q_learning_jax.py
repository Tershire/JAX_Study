# q_learning_jax.py

# Arz
# 2024 MAY 18 (SAT)

"""
⚠️ work in progress
Q-Learning agent for maze.
compare cases without and with JAX.
"""

# reference:


import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import maze_environment


# environment setting
wall_positions = np.array([(1, 3), (1, 4), (2, 1), (3, 1), (3, 3), (4, 3)])
env = maze_environment.Maze((5, 5), wall_positions)

# initialize environment
env.reset()
print(env.render("cell_name"))

# training setting
max_num_episodes = 1000
alpha = 0.1  # learning rate
gamma = 0.99  # discount
epsilon = 0.1  # epsilon-greedy param


# 1) without JAX
q_table = np.zeros((env.observation_space.n, env.action_space.n))


def select_action(q_table, state, epsilon):
    """epsilon-greedy"""
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])

    return action


def run_step(state, q_table, alpha, gamma, epsilon):
    action = select_action(q_table, state, epsilon)

    state_tp1, reward_tp1, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    q_value = q_table[state, action]
    td_target = reward_tp1 + gamma*np.max(q_table[state_tp1])
    q_table[state, action] = q_value + alpha*(td_target - q_value)

    state = state_tp1

    return state, q_table, done


def run_episode(env, q_table, alpha, gamma, epsilon):
    state, _ = env.reset()
    done = False

    while not done:
        state, q_table, done = run_step(state, q_table, alpha, gamma, epsilon)


def learn(max_num_episodes, env, q_table, alpha, gamma, epsilon):
    for episode in range(max_num_episodes):
        run_episode(env, q_table, alpha, gamma, epsilon)


learn(max_num_episodes, env, q_table, alpha, gamma, epsilon)

# training result
print(q_table)

arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
def display_greedy_action_per_state(env, q_table):
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

display_greedy_action_per_state(env, q_table)


# 1) with JAX
q_table = np.zeros((env.observation_space.n, env.action_space.n))
"""
should q_table be np or jnp (?)
jnp is immutable. then how can (?)
"""

key = jax.random.key(0)


def random_sample_a_action(q_values_for_a_state, subkey):
    jax.random.choice()



def select_action(q_table, state, epsilon, subkey):
    """epsilon-greedy"""
    if jax.random.uniform(subkey) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])

    return action
"""
how and where should I use jax.random for a function (?)
should I pass key as a parameter (?)
if epsilon rarely changes, setting it as static using @partial could be a good solution.
however, even in this case, the condition depends on the value of subkey.
then, one could try to set subkey as static using @partial,
but 'PRNGKeyArray' is unhashable and non-hashable static arguments are not supported.
"""

@partial(jit, static_argnames=["epsilon", "subkey"])
def run_step(state, q_table, alpha, gamma, epsilon, subkey):
    _, subkey = jax.random.split(subkey)
    action = select_action(q_table, state, epsilon, subkey)

    state_tp1, reward_tp1, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    q_value = q_table[state, action]
    td_target = reward_tp1 + gamma*np.max(q_table[state_tp1])
    q_table[state, action] = q_value + alpha*(td_target - q_value)

    state = state_tp1

    return state, q_table, done, subkey


def run_episode(env, q_table, alpha, gamma, epsilon, subkey):
    state, _ = env.reset()
    done = False

    while not done:
        state, q_table, done, subkey = run_step(state, q_table, alpha, gamma, epsilon, subkey)


def learn(max_num_episodes, env, q_table, alpha, gamma, epsilon, key):
    for episode in range(max_num_episodes):
        run_episode(env, q_table, alpha, gamma, epsilon, key)


learn(max_num_episodes, env, q_table, alpha, gamma, epsilon, key)

# training result
print(q_table)

display_greedy_action_per_state(env, q_table)
