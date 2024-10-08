# learning_algorithms_pytorch.py

# Arz
# 2024 AUG 02 (FRI)

"""
agent learning algorithms.
PyTorch implemented.
    - DQN
"""

# reference:


import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, deque

import maze_environment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

# Experience = namedtuple("experience", ("state_t", "action_t", "reward_tp1", "state_tp1"))

Experience = namedtuple("experience_with_dones", ("state_t", "action_t", "reward_tp1", "state_tp1", "done"))


class DQN:
    def __init__(self, env, learning_rate, memory_capacity):
        self.t = 0  # time step
        self.env = env
        self.learning_rate = learning_rate  # learning rate
        self.gamma = 0.99  # discount factor
        self.minibatch_size = 32
        self.num_actions = env.action_space.n
        self.num_observations = env.observation_space.n
        self.memory_capacity = memory_capacity
        self.replay_memory = self.Replay_Memory(self.memory_capacity)

        self.q_estimator = self.Q_Estimator(self.num_observations, self.num_actions).to(device)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.q_estimator.parameters(), lr=learning_rate, momentum=0.95)
        self.optimizer_update_interval = 4
        self.target_q_estimator = self.Q_Estimator(self.num_observations, self.num_actions).to(device)
        self.target_q_estimator_update_interval = 20

        self.target_q_estimator.load_state_dict(self.q_estimator.state_dict())

        # result
        self.cumulative_rewards = []

    def train(self, max_num_episodes):
        for episode in range(max_num_episodes):
            print("episode:", episode)

            # reset world
            state_t, _ = self.env.reset()
            state_t = self.one_hot_encode(state_t)
            state_t = torch.FloatTensor(state_t).to(device)

            # result
            cumulative_reward = 0

            while True:
                # {select & do} action
                action_t = self.select_action(state_t, episode, max_num_episodes)
                state_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
                state_tp1 = self.one_hot_encode(state_tp1)
                state_tp1 = torch.FloatTensor(state_tp1).to(device)
                done = terminated or truncated

                # remember experience
                experience = Experience(state_t, action_t, reward_tp1, state_tp1, done)
                self.replay_memory.remember(experience)

                # retrospect and update Q
                if len(self.replay_memory) >= self.minibatch_size and self.t % self.optimizer_update_interval == 0:
                    experiences = self.replay_memory.retrieve_random_experiences(self.minibatch_size)

                    experience_minibatch = Experience(*zip(*experiences))
                    state_j_minibatch = torch.stack(experience_minibatch.state_t).to(device)
                    action_j_minibatch = torch.LongTensor(np.array(experience_minibatch.action_t)).unsqueeze(1).to(device)
                    reward_jp1_minibatch = torch.FloatTensor(np.array(experience_minibatch.reward_tp1)).to(device)
                    state_jp1_minibatch = torch.stack(experience_minibatch.state_tp1).to(device)
                    done_j_minibatch = torch.IntTensor(experience_minibatch.done).to(device)

                    # update Q
                    self.update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch, done_j_minibatch,
                                  self.q_estimator, self.target_q_estimator, self.optimizer, self.gamma,
                                  self.loss_function)

                # update target Q-estimator
                if self.t % self.target_q_estimator_update_interval == 0:
                    self.target_q_estimator.load_state_dict(self.q_estimator.state_dict())

                # step forward
                state_t = state_tp1
                self.t += 1

                # result
                cumulative_reward += reward_tp1

                # termination
                if done:
                    break

            # result
            self.cumulative_rewards.append(cumulative_reward)

    def select_action(self, state, episode, max_num_episodes):
        """
        epsilon-greedy with decaying epsilon.
        x: state history
        """
        epsilon = 0.1 + (1 - 0.1) * np.exp(-episode / (max_num_episodes * 0.1))
        if np.random.rand() < epsilon:
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
                              gamma * torch.max(target_q_estimator(state_jp1_minibatch), dim=1).values  # (minibatch_size,)
                              # gamma * torch.max(target_q_estimator(state_jp1_minibatch), dim=1).values * (1 - done_j_minibatch)  # (minibatch_size,)

        q_values = q_estimator(state_j_minibatch).gather(1, action_j_minibatch).squeeze(1)  # (minibatch_size,)
        loss = loss_function(target_q_values, q_values)

        # optimize Q-estimator
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def one_hot_encode(self, state):
        encoded_state = np.zeros(self.num_observations)
        encoded_state[state] = 1
        return encoded_state

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
            state_t = self.one_hot_encode(state_t)
            state_t = torch.FloatTensor(state_t).to(device)

            while True:
                # {select & do} action
                with torch.no_grad():
                    action_t = torch.argmax(self.q_estimator(state_t)).item()
                state_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
                state_tp1 = self.one_hot_encode(state_tp1)
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
