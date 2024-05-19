# learning_algorithms_jax.py

# Arz
# 2024 MAY 19 (SUN)

"""
⚠️ work in progress
agent learning algorithms for maze.
JAX implemented.
    - Q-Learning
    - SARSA
"""

# reference:
# https://github.com/google-deepmind/rlax/blob/master/examples/simple_dqn.py
# https://github.com/hamishs/JAX-RL/blob/main/src/jax_rl/algorithms/dqn.py



import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial


class Q_Learning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.q_table = jnp.zeros((env.observation_space.n, env.action_space.n), dtype=jnp.float32)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount
        self.epsilon = epsilon  # epsilon-greedy param

    def learn(self, max_num_episodes, subkey):
        for episode in range(max_num_episodes):
            self.run_episode(self.env, self.q_table, self.alpha, self.gamma, self.epsilon, subkey)

    def run_episode(self, env, q_table, alpha, gamma, epsilon, subkey):
        state, _ = env.reset()
        done = False

        while not done:
            state, q_table, done, subkey = self.run_step(state, env, q_table, done, alpha, gamma, epsilon, subkey)

    @partial(jit, static_argnames=["self", "env", "alpha", "gamma", "epsilon"])
    def run_step(self, state, env, q_table, done, alpha, gamma, epsilon, subkey):
        _, *subkeys = jax.random.split(subkey, 3)
        action = self.select_action(state, epsilon, subkeys[0], subkeys[1])

        state_tp1, reward_tp1, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        q_value = q_table[state, action]
        td_target = reward_tp1 + gamma * np.max(q_table[state_tp1])
        q_table.at[state, action].set(q_value + alpha * (td_target - q_value))

        state = state_tp1

        return state, q_table, done, subkey

    def select_action(self, state, epsilon, subkey1, subkey2):
        """epsilon-greedy"""
        action = lax.cond(jax.random.uniform(subkey1) < epsilon,
                          self.select_random_action,
                          self.select_greedy_action,
                          state, subkey2)

        return action

    def select_random_action(self, state, subkey):
        action = jax.random.choice(subkey, jnp.arange(self.env.action_space.n))

        return action

    def select_greedy_action(self, state, subkey):
        action = jnp.argmax(self.q_table[state])

        return action
