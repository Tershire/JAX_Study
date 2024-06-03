# learn_to_play_space_invaders.py

# Arz
# 2024 JUN 03 (MON)

"""
train agent to play <space invaders>
"""


import gymnasium
import matplotlib.pyplot as plt
import cv2 as cv


# load environment
env = gymnasium.make("ALE/SpaceInvaders-v5", obs_type="rgb", render_mode="rgb_array")

env.reset()
frame = env.render()
frame_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
cv.imshow("frame_image", frame_image)
cv.waitKey(0)
