# learning_algorithms_pytorch.py

# Arz
# 2024 AUG 31 (SUN)

"""
agent learning algorithms.
PyTorch implemented.
    - DQN
"""

# reference:
# - https://www.akshaymakes.com/blogs/deep_q_learning
# = https://github.com/jordanlei/deep-reinforcement-learning-cartpole/blob/master/dqn_cartpole.py


import random
import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

Experience = namedtuple("experience", ("state_t", "action_t", "reward_tp1", "state_tp1"))

Experience_With_Dones = namedtuple("experience_with_dones", ("state_t", "action_t", "reward_tp1", "state_tp1", "done"))


class DQN:
    def __init__(self, env, learning_rate, memory_capacity):
        self.t = 0  # time step
        self.env = env
        self.learning_rate = learning_rate  # learning rate
        self.gamma = 0.99  # discount factor
        self.minibatch_size = 64
        self.num_actions = env.action_space.n
        observation, observation_info = env.reset()
        self.num_observations = len(observation)
        self.memory_capacity = memory_capacity
        self.replay_memory = self.Replay_Memory(self.memory_capacity)

        self.q_estimator = self.Q_Estimator(self.num_observations, self.num_actions).to(device)
        # self.loss_function = nn.MSELoss()
        self.loss_function = nn.SmoothL1Loss()
        # self.optimizer = torch.optim.RMSprop(self.q_estimator.parameters(), lr=learning_rate, momentum=0.95)
        self.optimizer = torch.optim.Adam(self.q_estimator.parameters(), lr=learning_rate)
        self.optimizer_update_interval = 2
        self.target_q_estimator = self.Q_Estimator(self.num_observations, self.num_actions).to(device)
        self.target_q_estimator_update_interval = int(1E3)

        # result
        self.cumulative_rewards = []

    def train(self, max_num_episodes):
        for episode in range(max_num_episodes):
            print("episode:", episode)

            # reset world
            state_t, _ = self.env.reset()
            state_t = torch.FloatTensor(state_t).to(device)

            # result
            cumulative_reward = 0

            while True:
                # {select & do} action
                action_t = self.select_action(state_t, episode, max_num_episodes)
                state_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
                state_tp1 = torch.FloatTensor(state_tp1).to(device)

                # remember experience
                experience = Experience(state_t, action_t, reward_tp1, state_tp1)
                self.replay_memory.remember(experience)

                # retrospect and update Q
                if len(self.replay_memory) >= self.minibatch_size * 50 and self.t % self.optimizer_update_interval == 0:
                    experiences = self.replay_memory.retrieve_random_experiences(self.minibatch_size)

                    experience_minibatch = Experience(*zip(*experiences))
                    state_j_minibatch = torch.stack(experience_minibatch.state_t).to(device)
                    action_j_minibatch = torch.LongTensor(np.array(experience_minibatch.action_t)).unsqueeze(1).to(device)
                    reward_jp1_minibatch = torch.FloatTensor(np.array(experience_minibatch.reward_tp1)).to(device)
                    state_jp1_minibatch = torch.stack(experience_minibatch.state_tp1).to(device)

                    # update Q
                    self.update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch,
                                  self.q_estimator, self.target_q_estimator, self.optimizer, self.gamma,
                                  self.loss_function)

                # update target Q-estimator
                if self.t % self.target_q_estimator_update_interval == 0:
                    self.target_q_estimator.load_state_dict(self.q_estimator.state_dict())

                    # soft update
                    # tau = 0.005
                    # target_net_state_dict = self.target_q_estimator.state_dict()
                    # policy_net_state_dict = self.q_estimator.state_dict()
                    # for key in policy_net_state_dict:
                    #     target_net_state_dict[key] = tau * policy_net_state_dict[key] + (1 - tau) * \
                    #                                  target_net_state_dict[key]
                    #     self.target_q_estimator.load_state_dict(target_net_state_dict)

                # step forward
                state_t = state_tp1
                self.t += 1

                # result
                cumulative_reward += reward_tp1

                # termination
                done = terminated or truncated
                if done:
                    break

            # result
            self.cumulative_rewards.append(cumulative_reward)

    def select_action(self, state, episode, max_num_episodes):
        """
        epsilon-greedy with decaying epsilon.
        x: state history
        """
        # epsilon = 0.1 + (1 - 0.1) * np.exp(-episode / (max_num_episodes * 100))
        epsilon = max(0.01, 0.1 - 0.01 * (episode / 50))
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_estimator(state)
                action = torch.argmax(q_values).item()

        return action

    @staticmethod
    def update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch,
                 q_estimator, target_q_estimator, optimizer, gamma, loss_function):

        with torch.no_grad():
            target_q_values = reward_jp1_minibatch + \
                              gamma * torch.max(target_q_estimator(state_jp1_minibatch), dim=1).values  # (minibatch_size,)

        q_values = q_estimator(state_j_minibatch).gather(1, action_j_minibatch).squeeze(1)  # (minibatch_size,)
        loss = loss_function(target_q_values, q_values)

        # optimize Q-estimator
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(q_estimator.parameters(), 100)  # in-place gradient clipping
        optimizer.step()

    def test(self, model_path):
        self.q_estimator.load_state_dict(torch.load(model_path))

        # result
        agent_positions = []

        episode = 0
        while self.env.grid[self.env.agent_position[0], self.env.agent_position[1]] != maze_environment.Cell.GOAL:
            print("episode:", episode)

            # result
            agent_positions = []
            actions = []

            # reset world
            state_t, _ = self.env.reset()
            state_t = torch.FloatTensor(state_t).to(device)

            while True:
                # {select & do} action
                with torch.no_grad():
                    action_t = torch.argmax(self.q_estimator(state_t)).item()
                state_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
                state_tp1 = torch.FloatTensor(state_tp1).to(device)

                # step forward
                state_t = state_tp1

                # result
                agent_positions.append((self.env.agent_position[0], self.env.agent_position[1]))
                actions.append(action_t)

                # termination
                done = terminated or truncated
                if done:
                    break

            episode += 1

        return agent_positions, actions

    def save_model(self, model_path):
        torch.save(self.q_estimator.state_dict(), model_path)
        print(f"model saved to {model_path}.")

    class Q_Estimator(nn.Module):
        """
        action value estimator.
        """

        def __init__(self, num_observations: int, num_actions: int):
            super().__init__()

            self.main = nn.Sequential(
                nn.Linear(in_features=num_observations, out_features=128),  # (minibatch_size, num_observations) -> (minibatch_size, 128)
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=64),  # (minibatch_size, 128) -> (minibatch_size, 64)
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=num_actions)   # (minibatch_size, 64) -> (minibatch_size, num_actions)
            )

        def forward(self, x):
            x = self.main(x)
            return x

    class Replay_Memory():
        """
        replay memory.
        """

        def __init__(self, memory_capacity):
            self.storage = deque([], maxlen=memory_capacity)

        def remember(self, experience: Experience):
            self.storage.append(experience)

        def retrieve_random_experiences(self, batch_size):
            return random.sample(self.storage, batch_size)

        def __len__(self):
            return len(self.storage)


class DQN_Prodo:
    """DQN written by Yellow Card Prodo."""
    def __init__(self, env, learning_rate=1e-3, memory_capacity=int(1e4)):
        self.t = 0
        self.env = env
        self.learning_rate = learning_rate  # learning rate
        self.gamma = 0.99
        self.minibatch_size = 64
        self.num_actions = env.action_space.n
        observation, observation_info = env.reset()
        self.num_observations = len(observation)
        self.memory_capacity = memory_capacity
        self.replay_memory = self.Replay_Memory(self.memory_capacity)

        self.q_estimator = self.Q_Estimator(self.num_observations, self.num_actions).to(device)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_estimator.parameters(), lr=learning_rate)
        self.optimizer_update_interval = 1
        self.target_q_estimator = self.Q_Estimator(self.num_observations, self.num_actions).to(device)
        self.target_q_estimator_update_interval = int(1E1)

        self.target_q_estimator.load_state_dict(self.q_estimator.state_dict())

        # action selection
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995

        # result
        self.cumulative_rewards = []

        video_path = f"./video/final_episode_training.mp4"
        frame = self.env.render()
        frame_height, frame_width, _ = frame.shape
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv.VideoWriter(video_path, fourcc, 30, (frame_width, frame_height))

    def train(self, max_num_episodes, save_final_episode_video=False):
        for episode in range(max_num_episodes):
            print("episode:", episode)

            # reset world
            state_t, _ = self.env.reset()
            state_t = torch.FloatTensor(state_t).to(device)

            # result
            cumulative_reward = 0
            frames = []  # for video

            while True:
                # video
                if save_final_episode_video and episode == max_num_episodes - 1:
                    frame = self.env.render()
                    frames.append(frame)

                # {select & do} action
                action_t = self.select_action(state_t)
                state_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
                state_tp1 = torch.FloatTensor(state_tp1).to(device)
                done = terminated or truncated

                # remember experience
                experience = Experience_With_Dones(state_t, action_t, reward_tp1, state_tp1, done)
                self.replay_memory.remember(experience)

                # retrospect and update Q
                if len(self.replay_memory) >= self.minibatch_size and self.t % self.optimizer_update_interval == 0:
                    experiences = self.replay_memory.retrieve_random_experiences(self.minibatch_size)

                    experience_minibatch = Experience_With_Dones(*zip(*experiences))
                    state_j_minibatch = torch.stack(experience_minibatch.state_t).to(device)
                    action_j_minibatch = torch.LongTensor(np.array(experience_minibatch.action_t)).unsqueeze(1).to(device)
                    reward_jp1_minibatch = torch.FloatTensor(np.array(experience_minibatch.reward_tp1)).to(device)
                    state_jp1_minibatch = torch.stack(experience_minibatch.state_tp1).to(device)
                    done_j_minibatch = torch.FloatTensor(experience_minibatch.done).to(device)
                    # done_j_minibatch = torch.IntTensor(experience_minibatch.done).to(device)

                    # update Q
                    self.update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch, done_j_minibatch,
                                  self.q_estimator, self.target_q_estimator, self.optimizer, self.gamma,
                                  self.loss_function)

                # step forward
                state_t = state_tp1
                self.t += 1

                # result
                cumulative_reward += reward_tp1

                # termination
                if done:
                    break

            # update target Q-estimator
            if episode % self.target_q_estimator_update_interval == 0:
                self.target_q_estimator.load_state_dict(self.q_estimator.state_dict())

            # result
            self.cumulative_rewards.append(cumulative_reward)

            if save_final_episode_video and episode == max_num_episodes - 1:
                for frame in frames:
                    self.video_writer.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))

        self.video_writer.release()

    def select_action(self, state):
        """
        epsilon-greedy with decaying epsilon.
        x: state history
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_estimator(state)
                action = torch.argmax(q_values).item()
        return action

    @staticmethod
    def update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch, done_j_minibatch,
                 q_estimator, target_q_estimator, optimizer, gamma, loss_function):

        with torch.no_grad():
            target_q_values = reward_jp1_minibatch + \
                              gamma * torch.max(target_q_estimator(state_jp1_minibatch), dim=1).values * (1 - done_j_minibatch)  # (minibatch_size,)

            # TODO: investigate the reason.
            # below is the original code. without the (1 - done_j_minibatch), the performance is very bad.
            # target_q_values = reward_jp1_minibatch + \
            #                   gamma * torch.max(target_q_estimator(state_jp1_minibatch), dim=1).values  # (minibatch_size,)

        q_values = q_estimator(state_j_minibatch).gather(1, action_j_minibatch).squeeze(1)  # (minibatch_size,)

        loss = loss_function(target_q_values, q_values)

        # optimize Q-estimator
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    class Q_Estimator(nn.Module):
        """
        action value estimator.
        """

        def __init__(self, num_observations: int, num_actions: int):
            super().__init__()

            self.main = nn.Sequential(
                nn.Linear(in_features=num_observations, out_features=64),  # (minibatch_size, num_observations) -> (minibatch_size, 64)
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=64),  # (minibatch_size, 64) -> (minibatch_size, 64)
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=num_actions)   # (minibatch_size, 64) -> (minibatch_size, num_actions)
            )

        def forward(self, x):
            x = self.main(x)
            return x

    class Replay_Memory():
        """
        replay memory.
        """

        def __init__(self, memory_capacity):
            self.storage = deque([], maxlen=memory_capacity)

        def remember(self, experience: Experience_With_Dones):
            self.storage.append(experience)

        def retrieve_random_experiences(self, batch_size):
            return random.sample(self.storage, batch_size)

        def __len__(self):
            return len(self.storage)


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
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            done = False

            # result
            cumulative_reward = 0

            while not done:
                # select action
                action = self.select_action(state, episode, max_num_episodes)

                # take the action and receive state and reward at t + 1
                state_tp1, reward_tp1, terminated, truncated, info = self.env.step(action)  # tp1: t + 1
                state_tp1 = self.discretize_state(state_tp1)
                done = terminated or truncated

                # update Q-table
                q_value = self.q_table[state + (action,)]
                td_target = reward_tp1 + self.gamma * np.max(self.q_table[state_tp1])
                self.q_table[state + (action,)] = q_value + self.alpha * (td_target - q_value)

                # update state
                state = state_tp1

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
