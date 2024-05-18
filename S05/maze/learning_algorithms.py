# learning_algorithms.py

# Arz
# 2024 MAY 14 (TUE)

"""Q-learning agent for maze."""

# reference:
#

import numpy as np


class Q_Learning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount
        self.epsilon = epsilon  # epsilon-greedy param

    def learn(self, max_num_episodes):
        for episode in range(max_num_episodes):
            state, _ = self.env.reset()
            done = False

            # result
            # agent_positions = []

            while not done:
                # select action
                action = self.select_action(state)

                # take the action and receive state and reward at t + 1
                state_tp1, reward_tp1, terminated, truncated, info = self.env.step(action)  # tp1: t + 1
                done = terminated or truncated

                # update Q-table
                q_value = self.q_table[state, action]
                td_target = reward_tp1 + self.gamma*np.max(self.q_table[state_tp1])
                self.q_table[state, action] = q_value + self.alpha*(td_target - q_value)

                # update state
                state = state_tp1

                # collect result
                # agent_positions.append(self.env.agent_position)

            # show result
            # print("episode:", episode)
            # print("\tagent positions:", agent_positions)

    def select_action(self, state):
        """epsilon-greedy"""
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])

        return action