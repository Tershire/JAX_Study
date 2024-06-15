# learning_algorithms.py

# Arz
# 2024 JUN 15 (MON)

"""
agent learning algorithms.
JAX implemented.
    - DQN
"""

# reference:


import cv2 as cv
import random
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
from flax import nnx
from collections import namedtuple, deque
from pathlib import Path


Experience = namedtuple("experience", ("state_t", "action_t", "reward_tp1", "state_tp1"))


class DQN:
    def __init__(self, env, alpha, memory_capacity, rngs):
        self.t = 0  # time step
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = 0.99  # discount factor
        self.minibatch_size = 128
        self.num_actions = env.action_space.n
        observation, observation_info = env.reset()
        self.num_observations = len(observation)
        self.memory_capacity = memory_capacity
        self.replay_memory = self.Replay_Memory(self.memory_capacity)
        self.q_estimator = self.Q_Estimator(self.num_observations, self.num_actions, rngs)
        self.optimizer = nnx.Optimizer(self.q_estimator,
                                       optax.rmsprop(learning_rate=alpha, momentum=0.95))
        self.optimizer_update_interval = 4
        self.target_q_estimator = self.q_estimator
        self.target_q_estimator_update_interval = 1E2

        # result
        self.cumulative_rewards = []

    def train(self, max_num_episodes, save_model=False):
        for episode in range(max_num_episodes):
            print("episode:", episode)

            # reset world
            state_t, _ = self.env.reset()

            # result
            cumulative_reward = 0

            while True:
                # {select & do} action
                action_t = self.select_action(state_t, episode, max_num_episodes)
                state_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
                done = terminated or truncated
                if done:
                    break

                # remember experience
                experience = Experience(state_t, action_t, reward_tp1, state_tp1)
                self.replay_memory.remember(experience)

                # retrospect and update Q
                if len(self.replay_memory) >= self.minibatch_size and self.t % self.optimizer_update_interval == 0:
                    experiences = self.replay_memory.retrieve_random_experiences(self.minibatch_size)
                    state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch = \
                        map(jnp.array, Experience(*zip(*experiences)))

                    # update Q
                    self.update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch,
                                  self.q_estimator, self.target_q_estimator, self.optimizer, self.gamma)

                # update target Q-estimator
                if self.t % self.target_q_estimator_update_interval == 0:
                    params = nnx.state(self.q_estimator, nnx.Param)
                    nnx.update(self.target_q_estimator, params)

                # step forward
                state_t = state_tp1
                self.t += 1

                # result
                cumulative_reward += reward_tp1

            # result
            self.cumulative_rewards.append(cumulative_reward)

        # save model
        if save_model:
            model_path = Path("/home/tershire/Documents/academics/extracurricular_learning/jax_study_models/cartpole.arz_model")
            _, params, = nnx.split(self.q_estimator, nnx.Param)
            # nnx.display(params)
            # with open(model_path, "wb") as f:
            #     f.write(flax.serialization.to_bytes(params))
            # print(f"model saved to {model_path}.")

    def select_action(self, state, episode, max_num_episodes):
        """
        epsilon-greedy with decaying epsilon.
        x: state history
        """
        epsilon = 0.1 + (1 - 0.1) * np.exp(-episode / max_num_episodes)
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = np.expand_dims(state, 0)
            q_values = self.q_estimator(state)
            action = np.argmax(q_values)

        return np.array(action)  # sometimes gets jax array time if without np. weird.

    @staticmethod
    @nnx.jit
    def update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch,
                 q_estimator, target_q_estimator, optimizer, gamma):
        """

        """
        target_q_values = reward_jp1_minibatch + \
                          gamma * jnp.max(target_q_estimator(state_jp1_minibatch))  # (minibatch_size,)

        # optimize Q-estimator
        def loss_function(model):
            q_values = model(state_j_minibatch)  # (minibatch_size, num_actions)
            q_values = q_values[jnp.arange(q_values.shape[0]), action_j_minibatch.squeeze()]  # (minibatch_size,) (?)
            return ((target_q_values - q_values) ** 2).mean()

        grads = nnx.grad(loss_function)(q_estimator)
        optimizer.update(grads)

    class Q_Estimator(nnx.Module):
        """
        action value estimator.
        """

        def __init__(self, num_observations: int, num_actions: int, rngs: nnx.Rngs):
            self.linear1 = nnx.Linear(
                in_features=num_observations, out_features=128,
                rngs=rngs)
            self.linear2 = nnx.Linear(
                in_features=128, out_features=64,
                rngs=rngs)
            self.linear3 = nnx.Linear(
                in_features=64, out_features=num_actions,
                rngs=rngs)

        def __call__(self, x):
            x = nnx.relu(self.linear1(x))  # (minibatch_size, num_observations) -> (minibatch_size, 128)
            x = nnx.relu(self.linear2(x))  # (minibatch_size, 128) -> (minibatch_size, 64)
            x = self.linear3(x)  # (minibatch_size, 64) -> (minibatch_size, num_actions)
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


class DQN_With_Conv:
    def __init__(self, env, alpha, memory_capacity, rngs):
        self.t = 0  # time step
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = 0.99  # discount factor
        self.minibatch_size = 128
        self.obsv_history_length = 1
        self.num_actions = env.action_space.n
        self.memory_capacity = memory_capacity
        self.replay_memory = self.Replay_Memory(self.memory_capacity)
        self.obsv_history = deque([], maxlen=self.obsv_history_length)
        self.q_estimator = self.Q_Estimator(self.obsv_history_length, self.num_actions, rngs)
        self.optimizer = nnx.Optimizer(self.q_estimator,
                                       optax.rmsprop(learning_rate=alpha, momentum=0.95))
        self.optimizer_update_interval = 1
        self.target_q_estimator = self.q_estimator
        self.target_q_estimator_update_interval = 4

        # result
        self.cumulative_rewards = []

    def train(self, max_num_episodes, save_model=False):
        for episode in range(max_num_episodes):
            print("episode:", episode)

            # reset world
            obsv_t, _ = self.env.reset()
            self.obsv_history.append(obsv_t)

            # initialize state: observation history
            while len(self.obsv_history) < self.obsv_history_length:
                action_t = self.env.action_space.sample()
                obsv_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
                done = terminated or truncated
                if done:
                    break

                self.obsv_history.append(obsv_tp1)

            state_t = np.array(self.obsv_history)
            state_t = np.moveaxis(state_t, [0], [1])

            # result
            cumulative_reward = 0

            while True:
                # {select & do} action
                action_t = self.select_action(state_t, episode, max_num_episodes)
                obsv_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
                done = terminated or truncated
                if done:
                    break

                # update image history
                self.obsv_history.append(obsv_tp1)

                # remember experience
                state_tp1 = np.array(self.obsv_history)
                state_tp1 = np.moveaxis(state_tp1, [0], [1])
                experience = Experience(state_t, action_t, reward_tp1, state_tp1)
                self.replay_memory.remember(experience)

                # retrospect and update Q
                if len(self.replay_memory) >= self.minibatch_size and self.t % self.optimizer_update_interval == 0:
                    experiences = self.replay_memory.retrieve_random_experiences(self.minibatch_size)
                    state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch = \
                        map(jnp.array, Experience(*zip(*experiences)))

                    # update Q
                    self.update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch,
                                  self.q_estimator, self.target_q_estimator, self.optimizer, self.gamma)

                # update target Q-estimator
                if self.t % self.target_q_estimator_update_interval == 0:
                    params = nnx.state(self.q_estimator, nnx.Param)
                    nnx.update(self.target_q_estimator, params)

                # step forward
                state_t = state_tp1
                self.t += 1

                # result
                cumulative_reward += reward_tp1

            # result
            self.cumulative_rewards.append(cumulative_reward)

        # save model
        if save_model:
            model_path = Path("/home/tershire/Documents/academics/extracurricular_learning/jax_study_models/cartpole.arz_model")
            _, params, = nnx.split(self.q_estimator, nnx.Param)
            # nnx.display(params)
            # with open(model_path, "wb") as f:
            #     f.write(flax.serialization.to_bytes(params))
            # print(f"model saved to {model_path}.")

    def select_action(self, state, episode, max_num_episodes):
        """
        epsilon-greedy with decaying epsilon.
        x: state history
        """
        epsilon = 0.1 + (1 - 0.1) * np.exp(-episode / max_num_episodes)
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = np.expand_dims(state, 0)
            q_values = self.q_estimator(state)
            action = np.argmax(q_values)

        return np.array(action)  # sometimes gets jax array time if without np. weird.

    @staticmethod
    @nnx.jit
    def update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch,
                 q_estimator, target_q_estimator, optimizer, gamma):
        """

        """
        target_q_values = reward_jp1_minibatch + \
                          gamma * jnp.max(target_q_estimator(state_jp1_minibatch))  # (minibatch_size,)

        # optimize Q-estimator
        def loss_function(model):
            q_values = model(state_j_minibatch)  # (minibatch_size, num_actions)
            q_values = q_values[jnp.arange(q_values.shape[0]), action_j_minibatch.squeeze()]  # (minibatch_size,) (?)
            return ((target_q_values - q_values) ** 2).mean()

        grads = nnx.grad(loss_function)(q_estimator)
        optimizer.update(grads)

    class Q_Estimator(nnx.Module):
        """
        action value estimator.
        """

        def __init__(self, obsv_history_length: int, num_actions: int, rngs: nnx.Rngs):
            self.conv1 = nnx.Conv(
                in_features=obsv_history_length, out_features=16,
                kernel_size=4, strides=2, padding=2,
                rngs=rngs)
            self.conv2 = nnx.Conv(
                in_features=16, out_features=32,
                kernel_size=2, strides=1, padding=1,
                rngs=rngs)
            self.linear1 = nnx.Linear(
                in_features=128, out_features=128,
                rngs=rngs)
            self.linear2 = nnx.Linear(
                in_features=128, out_features=64,
                rngs=rngs)
            self.linear3 = nnx.Linear(
                in_features=64, out_features=num_actions,
                rngs=rngs)

        def __call__(self, x):
            x = nnx.relu(self.conv1(x))  # (minibatch_size, 4, obsv_history_length) -> (minibatch_size, 3, 16)
            x = nnx.relu(self.conv2(x))  # (minibatch_size, 1, 4, 32) -> (minibatch_size, 1, 4, 128)
            x = x.reshape(x.shape[0], -1)  # flatten: (minibatch_size, 1, 4, 32) -> (minibatch_size, 128)
            x = nnx.relu(self.linear1(x))  # (minibatch_size, 128) -> (minibatch_size, 128)
            x = nnx.relu(self.linear2(x))  # (minibatch_size, 128) -> (minibatch_size, 64)
            x = self.linear3(x)  # (minibatch_size, 64) -> (minibatch_size, num_actions)
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
