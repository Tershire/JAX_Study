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
from functools import partial


class Q_Learning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.q_table = jnp.zeros((env.observation_space.n, env.action_space.n), dtype=jnp.float32)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount
        self.epsilon = epsilon  # epsilon-greedy param

        # result
        self.cumulative_rewards = np.array([])

    def learn(self, max_num_episodes):
        # result
        self.cumulative_rewards = np.array([])

        for episode in range(max_num_episodes):
            state, _ = self.env.reset()
            done = False

            # result
            cumulative_reward = 0

            while not done:
                # select action
                action = self.select_action(state)

                # take the action and receive state and reward at t + 1
                state_tp1, reward_tp1, terminated, truncated, info = self.env.step(action)  # tp1: t + 1
                done = terminated or truncated

                # update Q-table
                q_table = self.update_q_table(self.q_table, state, action, state_tp1, reward_tp1)
                self.q_table = q_table

                # update state
                state = state_tp1

                # collect result
                cumulative_reward += reward_tp1

            # collect result
            self.cumulative_rewards = np.append(self.cumulative_rewards, cumulative_reward)

    @partial(jax.jit, static_argnames=["self"])
    def update_q_table(self, q_table, state, action, state_tp1, reward_tp1):
        q_value = q_table[state, action]
        td_target = reward_tp1 + self.gamma * jnp.max(q_table[state_tp1])
        q_table = q_table.at[state, action].set(q_value + self.alpha * (td_target - q_value))

        return q_table

    def select_action(self, state):
        """epsilon-greedy"""
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = jnp.argmax(self.q_table[state])

        return action
