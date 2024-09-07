# learning_algorithms_jax.py

# Arz
# 2024 AUG 31 (SUN)

"""
agent learning algorithms.
numpy implementation.
    - DQN
"""

# reference:


import numpy as np


class Q_Learning:
    def __init__(self, env, alpha=0.1, gamma=0.99, num_bins=(10, 10, 10, 10)):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount
        self.num_bins = num_bins

        # state discretization
        self.cart_position_bins, self.cart_velocity_bins, self.pole_angle_bins, self.pole_velocity_bins = \
            self.define_discrete_states(env, num_bins)
        self.q_table = np.zeros(num_bins + (env.action_space.n,))

        # result
        self.cumulative_rewards = np.array([])

    def train(self, max_num_episodes):
        # result
        self.cumulative_rewards = np.array([])

        for episode in range(max_num_episodes):
            state_t, _ = self.env.reset()
            state_t = self.discretize_state(state_t)
            done = False

            # result
            cumulative_reward = 0

            while not done:
                # select action
                # print(self.q_table.shape, self.q_table.dtype)  # test
                # self.q_table[state_t]  # test
                action_t = self.select_action(state_t, episode, max_num_episodes)

                # take the action and receive state and reward at t + 1
                state_tp1, reward_tp1, terminated, truncated, info = self.env.step(action_t)  # tp1: t + 1
                state_tp1 = self.discretize_state(state_tp1)
                done = terminated or truncated

                # update Q-table
                q_value = self.q_table[state_t + (action_t,)]
                td_target = reward_tp1 + self.gamma * np.max(self.q_table[state_tp1])
                self.q_table[state_t + (action_t,)] = q_value + self.alpha * (td_target - q_value)

                # update state
                state_t = state_tp1

                # collect result
                cumulative_reward += reward_tp1

            # collect result
            self.cumulative_rewards = np.append(self.cumulative_rewards, cumulative_reward)

    def select_action(self, state, episode, max_num_episodes):
        """epsilon-greedy"""
        epsilon = 0.1 + (1 - 0.1) * np.exp(-episode / (max_num_episodes * 1))
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action

    def define_discrete_states(self, env, num_bins=(10, 10, 10, 10)):
        """ChatGPT & Yellow Card Prodo"""
        cart_position_bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num_bins[0])
        cart_velocity_bins = np.linspace(-3.0, 3.0, num_bins[1])
        pole_angle_bins = np.linspace(env.observation_space.low[2], env.observation_space.high[2], num_bins[2])
        pole_velocity_bins = np.linspace(-2.0, 2.0, num_bins[3])
        return cart_position_bins, cart_velocity_bins, pole_angle_bins, pole_velocity_bins

    def discretize_state(self, state):
        """ChatGPT & Yellow Card Prodo"""
        state = [
            np.digitize(state[0], self.cart_position_bins) - 1,
            np.digitize(state[1], self.cart_velocity_bins) - 1,
            np.digitize(state[2], self.pole_angle_bins) - 1,
            np.digitize(state[3], self.pole_velocity_bins) - 1
        ]
        return tuple(state)

    def test(self, model_path):
        self.q_table = np.load(model_path)

        # result
        cumulative_reward = 0
        actions = []

        # reset world
        state_t, _ = self.env.reset()
        state_t = self.discretize_state(state_t)

        while True:
            # {select & do} action
            action_t = np.argmax(self.q_table[state_t])
            state_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
            state_tp1 = self.discretize_state(state_tp1)
            done = terminated or truncated

            # step forward
            state_t = state_tp1

            # result
            cumulative_reward += reward_tp1
            actions.append(action_t)

            # termination
            if done:
                break

        return cumulative_reward, actions

    def save_model(self, model_path):
        print(f"model saved at {model_path}")
        np.save(model_path, self.q_table)
