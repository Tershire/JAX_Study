# learning_algorithms_jax.py

# Arz
# 2024 SEP 06 (FRI)

"""
agent learning algorithms.
JAX implemented.
    - DQN
"""

# reference:


import random
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
import cv2 as cv
from flax import nnx
from collections import namedtuple, deque


# Experience = namedtuple("experience", ("state_t", "action_t", "reward_tp1", "state_tp1"))

Experience = namedtuple("experience_with_dones", ("state_t", "action_t", "reward_tp1", "state_tp1", "done"))


class DQN:
    def __init__(self, env, learning_rate, memory_capacity, rngs):
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

        self.q_estimator = self.Q_Estimator(self.num_observations, self.num_actions, rngs)
        self.optimizer = nnx.Optimizer(self.q_estimator, optax.adam(learning_rate=learning_rate))
        # self.optimizer = nnx.Optimizer(self.q_estimator,
        #                                optax.rmsprop(learning_rate=learning_rate, momentum=0.95))
        self.optimizer_update_interval = 1
        self.target_q_estimator = self.Q_Estimator(self.num_observations, self.num_actions, rngs)
        self.target_q_estimator_update_interval = int(1E2)  # 1E3 -> very bad, 1E2 -> very good

        params = nnx.state(self.q_estimator, nnx.Param)
        nnx.update(self.target_q_estimator, params)

        # result
        self.cumulative_rewards = []

    def train(self, max_num_episodes):
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

                # remember experience
                experience = Experience(state_t, action_t, reward_tp1, state_tp1, done)
                self.replay_memory.remember(experience)

                # retrospect and update Q
                if len(self.replay_memory) >= self.minibatch_size and self.t % self.optimizer_update_interval == 0:
                    experiences = self.replay_memory.retrieve_random_experiences(self.minibatch_size)
                    state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch, done_j_minibatch = \
                        map(jnp.array, Experience(*zip(*experiences)))

                    # update Q
                    self.update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch, done_j_minibatch,
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
        epsilon = 0.05 + (1 - 0.05) * np.exp(-episode / (max_num_episodes * 0.1))  # arz 1 -> bad, 0.1 -> good
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            q_values = self.q_estimator(state)
            action = np.argmax(q_values)

        return np.array(action)

    @staticmethod
    @nnx.jit
    def update_q(state_j_minibatch, action_j_minibatch, reward_jp1_minibatch, state_jp1_minibatch, done_j_minibatch,
                 q_estimator, target_q_estimator, optimizer, gamma):

        target_q_values = reward_jp1_minibatch + \
                          gamma * jnp.max(target_q_estimator(state_jp1_minibatch), axis=1) * (1 - done_j_minibatch)  # (minibatch_size,)
                          # gamma * jnp.max(target_q_estimator(state_jp1_minibatch), axis=1)  # (minibatch_size,)

        def loss_function(model):
            q_values = model(state_j_minibatch)  # (minibatch_size, num_actions)
            q_values = q_values[jnp.arange(q_values.shape[0]), action_j_minibatch.squeeze()]  # (minibatch_size,)
            return ((target_q_values - q_values) ** 2).mean()

        # optimize Q-estimator
        loss, grads = nnx.value_and_grad(loss_function)(q_estimator)
        optimizer.update(grads)

    def test(self, model_path, save_video=False):
        state = self.load_model(model_path)

        nnx.update(self.q_estimator, state)
        nnx.update(self.target_q_estimator, state)

        # result
        cumulative_reward = 0
        actions = []

        if save_video:
            video_path = f"./video/dqn_test.mp4"
            frame = self.env.render()
            frame_height, frame_width, _ = frame.shape
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv.VideoWriter(video_path, fourcc, 30, (frame_width, frame_height))

            frames = []

        # reset world
        state_t, _ = self.env.reset()

        while True:
            # video
            if save_video:
                frame = self.env.render()
                frames.append(frame)

            # {select & do} action
            action_t = np.argmax(self.q_estimator(state_t))
            action_t = np.array(action_t)
            state_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
            done = terminated or truncated

            # step forward
            state_t = state_tp1

            # result
            cumulative_reward += reward_tp1
            actions.append(int(action_t))

            # termination
            if done:
                break

        # result
        if save_video:
            for frame in frames:
                self.video_writer.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
            self.video_writer.release()

        return cumulative_reward, actions

    def save_model(self, model_path):
        _, state, = nnx.split(self.q_estimator)

        def save_pytree(pytree, file_path):
            with open(file_path, 'wb') as f:
                f.write(flax.serialization.to_bytes(pytree))

        flat_state, _ = jax.tree_util.tree_flatten(state)
        save_pytree(flat_state, model_path)
        print(f"model saved at {model_path}.")

    def load_model(self, model_path):
        def load_pytree(filepath, tree_structure):
            with open(filepath, 'rb') as f:
                flat_state = f.read()
            return flax.serialization.from_bytes(tree_structure, flat_state)

        _, state = nnx.split(self.q_estimator)
        _, tree_structure = jax.tree_util.tree_flatten(state)
        flat_state = load_pytree(model_path, tree_structure)

        # restore model
        state = jax.tree_util.tree_unflatten(tree_structure, flat_state.values())
        return state

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
