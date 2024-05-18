# learning_algorithms.py

# Arz
# 2024 MAY 14 (TUE)

"""
agent learning algorithms for maze.
    - Q-Learning
    - SARSA
"""

# reference:
# https://hh-bigdata-career.tistory.com/14


import numpy as np


class Q_Learning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
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
            # agent_positions = np.array([])

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
                cumulative_reward += reward_tp1
                # np.append(agent_positions, self.env.agent_position)

            # collect result
            self.cumulative_rewards = np.append(self.cumulative_rewards, cumulative_reward)

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


class SARSA:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
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
            # agent_positions = np.array([])

            while not done:
                # select action
                action = self.select_action(state)

                # take the action and receive state and reward at t + 1
                state_tp1, reward_tp1, terminated, truncated, info = self.env.step(action)  # tp1: t + 1
                done = terminated or truncated

                # select action at t + 1
                action_tp1 = self.select_action(state_tp1)

                # update Q-table
                q_value = self.q_table[state, action]
                td_target = reward_tp1 + self.gamma*self.q_table[state_tp1, action_tp1]
                self.q_table[state, action] = q_value + self.alpha*(td_target - q_value)

                # update state
                state = state_tp1

                # collect result
                cumulative_reward += reward_tp1
                # np.append(agent_positions, self.env.agent_position)

            # collect result
            self.cumulative_rewards = np.append(self.cumulative_rewards, cumulative_reward)

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