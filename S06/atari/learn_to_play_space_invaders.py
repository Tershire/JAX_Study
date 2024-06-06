# learn_to_play_space_invaders.py

# Arz
# 2024 JUN 03 (MON)

"""
train agent to play <space invaders>
"""

# reference:
# https://flax.readthedocs.io/en/latest/nnx/mnist_tutorial.html
# https://github.com/davidreiman/pytorch-atari-dqn/blob/master/dqn.ipynb


import gymnasium
import numpy as np
import jax
import jax.numpy as jnp
import cv2 as cv
import itertools
from flax import nnx


# load environment
env = gymnasium.make("ALE/SpaceInvaders-v5", obs_type="rgb", render_mode="rgb_array")


# preprocessing
def preprocess(s_t, s_tp1):
    """
    preprocess
    :param s_t:
    :param s_tp1:
    :return:
    """
    # reduce flickering
    phi_tp1 = np.fmax(s_t, s_tp1)

    # convert color space: BGR -> YCC
    phi_tp1 = cv.cvtColor(phi_tp1, cv.COLOR_BGR2YCrCb)[:, :, 0]

    # down-sampling
    phi_tp1 = cv.resize(phi_tp1, (84, 110), interpolation=cv.INTER_LINEAR)

    # crop
    phi_tp1 = phi_tp1[13 + 7:97 + 7, :]

    return phi_tp1


# action value estimator
class Q_Estimator(nnx.Module):
    """
    action value estimator.
    """
    def __init__(self, num_frames: int, num_actions: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=num_frames,
            out_features=16,
            kernel_size=8,
            strides=4,
            padding=2,
            rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=16,
            out_features=32,
            kernel_size=4,
            strides=2,
            padding=1,
            rngs=rngs)
        self.linear1 = nnx.Linear(
            in_features=3200,
            out_features=256,
            rngs=rngs)
        self.linear2 = nnx.Linear(
            in_features=256,
            out_features=num_actions,
            rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def select_action(episode, max_num_episodes):
    """
    epsilon-greedy with decaying epsilon.
    :param episode:
    :param max_num_episodes:
    :return:
    """
    epsilon = np.exp(-episode / max_num_episodes)

    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        pass


# training setting
max_num_episodes = 10
num_frames = 4
num_actions = env.action_space.n
Q = Q_Estimator(num_frames, num_actions)


# training
for episode in range(max_num_episodes):
    env.reset()
    done = False
    while not done:
        # {select & do} action
        a_t = select_action()
        s_tp1, r_tp1, terminated, truncated, _ = env.step(a_t)

        # preprocess
        phi_tp1 = preprocess(s_t, s_tp1)
        pass

