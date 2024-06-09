# learning_algorithms.py

# Arz
# 2024 JUN 03 (MON)

"""
agent learning algorithms.
JAX implemented.
    - DQN
"""

# reference:
# https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html
# https://flax.readthedocs.io/en/latest/nnx/mnist_tutorial.html
# https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/training/optimizer.html
# https://github.com/davidreiman/pytorch-atari-dqn/blob/master/dqn.ipynb
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_jax.py
# https://flax.readthedocs.io/en/latest/nnx/transforms.html
# https://github.com/Tershire/JAX_Study/blob/master/S04/S04P01_flax_basics.ipynb


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


# debugging
def show_state(state):
    reel = np.hstack(state)
    reel = cv.resize(reel, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
    cv.imshow("state", reel)
    cv.waitKey(0)


Experience = namedtuple("experience", ("state_t", "action_t", "reward_tp1", "state_tp1"))


class DQN:
    def __init__(self, env, alpha, gamma, memory_capacity, rngs):
        self.t = 0  # time step
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # forgetting factor
        self.minibatch_size = 32
        self.image_history_length = 4
        self.num_actions = env.action_space.n
        self.memory_capacity = memory_capacity
        self.replay_memory = self.Replay_Memory(self.memory_capacity)
        self.image_history = deque([], maxlen=self.image_history_length)
        self.q_estimator = self.Q_Estimator(self.image_history_length, self.num_actions, rngs)
        self.optimizer = nnx.Optimizer(self.q_estimator,
                                       optax.rmsprop(learning_rate=alpha, momentum=0.95))
        self.optimizer_update_interval = 4
        self.target_q_estimator = self.q_estimator
        self.target_q_estimator_update_interval = 1E3

        # result
        self.cumulative_rewards = []

    def train(self, max_num_episodes, save_model=False):
        for episode in range(max_num_episodes):
            print("episode:", episode)

            # reset world
            frame_t, _ = self.env.reset()
            image_t = self.preprocess(frame_t)
            self.image_history.append(image_t)

            # initialize state: image history
            while len(self.image_history) < self.image_history_length:
                action_t = 0  # idle
                frame_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
                done = terminated or truncated
                if done:
                    break

                image_tp1 = self.preprocess(frame_t, frame_tp1)
                self.image_history.append(image_tp1)
                frame_t = frame_tp1

            state_t = np.array(self.image_history)
            state_t = np.moveaxis(state_t, [0], [2])

            # result
            cumulative_reward = 0

            while True:
                # {select & do} action
                action_t = self.select_action(state_t, episode, max_num_episodes)
                frame_tp1, reward_tp1, terminated, truncated, _ = self.env.step(action_t)
                done = terminated or truncated
                if done:
                    break

                # preprocess and update image history
                image_tp1 = self.preprocess(frame_t, frame_tp1)
                self.image_history.append(image_tp1)
                frame_t = frame_tp1

                # remember experience
                state_tp1 = np.array(self.image_history)
                state_tp1 = np.moveaxis(state_tp1, [0], [2])
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
            model_path = Path("/home/tershire/Documents/academics/extracurricular_learning/jax_study_models/atari.arz_model")
            _, params, = nnx.split(self.q_estimator, nnx.Param)
            nnx.display(params)
            # with open(model_path, "wb") as f:
            #     f.write(flax.serialization.to_bytes(params))
            # print(f"model saved to {model_path}.")

    def preprocess(self, frame1, frame2=None):
        """
        preprocess.
        """
        # reduce flickering
        if frame1 is not None and frame2 is not None:
            image = np.fmax(frame1, frame2)
        else:
            image = frame1

        # convert color space: BGR -> YCC, then extract Y channel.
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)[:, :, 0]

        # down-sampling
        image = cv.resize(image, (84, 110), interpolation=cv.INTER_LINEAR)

        # crop
        image = image[13 + 7:97 + 7, :]

        return image

    def select_action(self, state, episode, max_num_episodes):
        """
        epsilon-greedy with decaying epsilon.
        x: state history
        """
        epsilon = 0.1 + (1 - 0.1) * np.exp(-episode / max_num_episodes)
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = jnp.expand_dims(state, 0)
            q_values = self.q_estimator(state)
            action = np.argmax(q_values)

        return action

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

        def __init__(self, image_history_length: int, num_actions: int, rngs: nnx.Rngs):
            self.conv1 = nnx.Conv(
                in_features=image_history_length, out_features=16,
                kernel_size=(8, 8), strides=4, padding=2,
                rngs=rngs)
            self.conv2 = nnx.Conv(
                in_features=16, out_features=32,
                kernel_size=(4, 4), strides=2, padding=1,
                rngs=rngs)
            self.linear1 = nnx.Linear(
                in_features=3200, out_features=256,
                rngs=rngs)
            self.linear2 = nnx.Linear(
                in_features=256, out_features=num_actions,
                rngs=rngs)

        def __call__(self, x):
            x = nnx.relu(self.conv1(x))  # (minibatch_size, 80, 80, 4) -> (minibatch_size, 21, 21, 16)
            x = nnx.relu(self.conv2(x))  # (minibatch_size, 21, 21, 16) -> (minibatch_size, 10, 10, 32)
            x = x.reshape(x.shape[0], -1)  # flatten: (minibatch_size, 10, 10, 32) -> (minibatch_size, 3200)
            x = nnx.relu(self.linear1(x))  # (minibatch_size, 3200) -> (minibatch_size, 256)
            x = self.linear2(x)  # (minibatch_size, 256) -> (minibatch_size, num_actions)
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
