# learn_to_play_space_invaders.py

# Arz
# 2024 JUN 03 (MON)

"""
train agent to play <space invaders>
"""

# reference:
#


import gymnasium
import numpy as np
import cv2 as cv


# load environment
env = gymnasium.make("ALE/SpaceInvaders-v5", obs_type="rgb", render_mode="rgb_array")


# preprocessing
def preprocess(s_t, s_tp1):
    # reduce flickering
    phi_tp1 = np.fmax(s_t, s_tp1)

    # convert color space: BGR -> YCC
    phi_tp1 = cv.cvtColor(phi_tp1, cv.COLOR_BGR2YCrCb)[:, :, 0]

    # down-sampling
    phi_tp1 = cv.resize(phi_tp1, (84, 110), interpolation=cv.INTER_LINEAR)

    # crop
    phi_tp1 = phi_tp1[13 + 7:97 + 7, :]

    return phi_tp1


# training
max_num_episodes = 10
for episode in range(max_num_episodes):
    done = False
    while not done:
        pass
    