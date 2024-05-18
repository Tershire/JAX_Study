# maze_environment.py

# Arz
# 2024 MAY 13 (MON)

"""maze environment."""

# reference:
# http://romain.raveaux.free.fr/document/ReinforcementLearningbyQLearning.html


import numpy as np
import random
from enum import Enum


class Cell(Enum):
    BLANK = 0
    START = 1
    GOAL = 2
    WALL = 3


class Action(Enum):
    MOVE_LEFT = 0
    MOVE_DOWN = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3


class Maze:
    def __init__(self, grid_size: (int, int), wall_positions: list[(int, int)]):
        self.grid = np.full(grid_size, Cell.BLANK)
        self.grid_size = grid_size
        self.wall_positions = wall_positions
        self.agent_position = np.array([0, 0])
        self.episode_length = 0
        self.observation_space = self.Observation_Space(self.grid_size)
        self.action_space = self.Action_Space()

    class Observation_Space:
        def __init__(self, grid_size):
            self.grid_size = grid_size
            self.n = grid_size[0]*grid_size[1]

        def compute_observation(self, agent_position):
            return agent_position[0]*self.grid_size[0] + agent_position[1]

    class Action_Space:
        def __init__(self):
            self.n = len(Action)
            self.Action = [action for action in Action]

        def sample(self):
            return random.sample(self.Action, 1)[0].value

    def step(self, action):
        match action:
            case Action.MOVE_LEFT.value:
                self.agent_position[1] -= 1
            case Action.MOVE_DOWN.value:
                self.agent_position[0] += 1
            case Action.MOVE_RIGHT.value:
                self.agent_position[1] += 1
            case Action.MOVE_UP.value:
                self.agent_position[0] -= 1

        # check for various cases
        reward = -1
        terminated = False
        truncated = False
        if self.episode_length > 150:
            truncated = True

        # check if agent is off-grid
        agent_is_off_grid = self.check_and_control_if_agent_is_off_grid()
        if agent_is_off_grid:
            reward = -2
            terminated = True

        else:  # cases when agent is on-grid
            match self.grid[self.agent_position[0], self.agent_position[1]]:
                case Cell.BLANK:
                    reward = -1
                case Cell.START:
                    reward = -1
                case Cell.WALL:
                    reward = -2  # how much is reasonable (?)
                case Cell.GOAL:
                    reward = +10  # how much is reasonable (?)
                    terminated = True

        observation = self.observation_space.compute_observation(self.agent_position)

        # count episode length
        self.episode_length += 1

        info = dict()

        return observation, reward, terminated, truncated, info

    def reset(self):
        self.place_start()
        self.place_goal()
        self.place_walls()
        self.agent_position = np.array([0, 0])
        self.episode_length = 0

        observation = self.observation_space.compute_observation(self.agent_position)
        info = dict()

        return observation, info

    def place_start(self):
        self.grid[0, 0] = Cell.START

    def place_goal(self):
        self.grid[self.grid_size[0] - 1, self.grid_size[1] - 1] = Cell.GOAL

    def place_walls(self):
        for wall_position in self.wall_positions:
            self.grid[wall_position[0], wall_position[1]] = Cell.WALL

    def check_and_control_if_agent_is_off_grid(self):
        agent_is_off_grid = False
        if self.agent_position[0] < 0:
            self.agent_position[0] += 1
            agent_is_off_grid = True
        if self.agent_position[0] >= self.grid_size[0]:
            self.agent_position[0] -= 1
            agent_is_off_grid = True
        if self.agent_position[1] < 0:
            self.agent_position[1] += 1
            agent_is_off_grid = True
        if self.agent_position[1] >= self.grid_size[1]:
            self.agent_position[1] -= 1
            agent_is_off_grid = True

        return agent_is_off_grid


    # render
    def render(self, render_mode):
        match render_mode:
            case "cell_name":
                rendered_grid = np.full(self.grid_size, '')
                for i in np.arange(self.grid_size[0]):
                    for j in np.arange(self.grid_size[1]):
                        rendered_grid[i, j] = self.grid[i, j].name
                return rendered_grid
