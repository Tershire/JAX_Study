# test_preprocessing.py

# Arz
# 2024 JUN 06 (WED)

"""
DQN preprocessing
"""

# reference:
# https://github.com/sudharsan13296/Deep-Reinforcement-Learning-With-Python/blob/master/ \
# 09.%20%20Deep%20Q%20Network%20and%20its%20Variants/9.03.%20Playing%20Atari%20Games%20using%20DQN.ipynb
# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# https://limepencil.tistory.com/38


import gymnasium
import numpy as np
import cv2 as cv


# load environment
env = gymnasium.make("ALE/SpaceInvaders-v5", obs_type="rgb", render_mode="rgb_array")

# test
do_test = True
if do_test:
    env.reset()
    frame = env.render()
    frame_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.imshow("frame_image", frame_image)
    cv.waitKey(0)

# preprocessing
def preprocess(s_t, s_tp1):
    # reduce flickering
    phi_tp1 = np.fmax(s_t, s_tp1)
    print(phi_tp1.shape)

    # convert color space: BGR -> YCC
    phi_tp1 = cv.cvtColor(phi_tp1, cv.COLOR_BGR2YCrCb)[:, :, 0]
    print(phi_tp1.shape)

    # down-sampling
    phi_tp1 = cv.resize(phi_tp1, (84, 110), interpolation=cv.INTER_LINEAR)
    print(phi_tp1.shape)

    # crop
    phi_tp1 = phi_tp1[13+7:97+7, :]
    print(phi_tp1.shape)

    cv.imshow("phi_tp1", phi_tp1)
    cv.waitKey(0)

do_test = True
if do_test:
    env.reset()
    s_0 = env.render()
    s_1, r_1, terminated, truncated, info = env.step(0)
    preprocess(s_0, s_1)
