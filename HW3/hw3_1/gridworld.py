import os
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib import colors


COLORS = [
    "white",
    "black",
    "green",
    "red",
    "darkorange",
    "springgreen",
    "yellow",
    "brown",
    "aquamarine",
    "skyblue"
]

class GridWorld:

    OBJECT_TO_INDEX = {
        " ": 0,
        "W": 1,
        "G": 2,
        "T": 3,
        "L": 4,
        "E": 5,
        "K": 6,
        "D": 7,
        "B": 8,
        "P": 9,
        "A": 10,
    }
    OBJECT_INDEX_TO_CHAR = {
        0: " ",
        1: "#",
        2: "G",
        3: "T",
        4: "L",
        5: "E",
        6: "K",
        7: "D",
        8: "B",
        9: "P",
        10: "A",
    }

    def __init__(
        self,
        maze_file: str,
        goal_reward: float = 1,
        trap_reward: float = -1,
        step_reward: float = -1,
        exit_reward: float = 0.1,
        bait_reward: float = 1,
        bait_step_penalty: float = -0.25,
        max_step: int = 1000,
    ):
        self._goal_reward = goal_reward
        self._trap_reward = trap_reward
        self._step_reward = step_reward
        self._exit_reward = exit_reward
        self._bait_reward = bait_reward
        self._bait_step_penalty = bait_step_penalty
        self.step_reward = self._step_reward
        self._step_count = 0
        self._maze = np.array([])
        self._state_list = []
        self._current_state = 0
        self.max_step = max_step
        self.maze_name = os.path.split(maze_file)[1].replace(".txt", "").capitalize()
        self._read_maze(maze_file)
        self._state_list_len = len(self._state_list)
        self._state_list_index = {}
        for idx, state in enumerate(self._state_list):
            self._state_list_index[state] = idx
        self.render_init(self.maze_name)

        min_y = None

        lava_states = []
        for state in range(self.get_grid_space()):
            if self._is_lava_state(self._state_list[state]):
                lava_states.append(self._state_list[state])

        if len(lava_states) > 0:
            min_y = min(lava_states, key=lambda x: x[1])[1]

        self._door_state = None
        self._key_state = None
        for state in range(self.get_grid_space()):
            if self._is_door_state(self._state_list[state]):
                self._door_state = state
            if self._is_key_state(self._state_list[state]):
                self._key_state = state

        if self._door_state is not None:

            if min_y is not None:
                min_y = min(min_y, self._state_list[self._door_state][1])
            else:
                min_y = self._state_list[self._door_state][1]

        self._bait_state = None
        for state in range(self.get_grid_space()):
            if self._is_bait_state(self._state_list[state]):
                self._bait_state = state

        self._portal_state = []
        for state in range(self.get_grid_space()):
            if self._is_portal_state(self._state_list[state]):
                self._portal_state.append(state)

        self.portal_next_state = {}
        if len(self._portal_state) == 2:
            self.portal_next_state[self._state_list[self._portal_state[0]]
                                   ] = self._state_list[self._portal_state[1]]
            self.portal_next_state[self._state_list[self._portal_state[1]]
                                   ] = self._state_list[self._portal_state[0]]

        self._init_states = []
        for state in range(self.get_grid_space()):
            if state == self._bait_state:
                continue
            if state == self._key_state:
                continue
            if min_y is not None and self._state_list[state][1] < min_y:
                self._init_states.append(state)
            elif min_y is None:
                self._init_states.append(state)

        self.reset()

    def _read_maze(self, maze_file: str) -> None:
        self._maze = np.loadtxt(maze_file, dtype=np.uint8)
        for i in range(self._maze.shape[0]):
            for j in range(self._maze.shape[1]):
                if self._maze[i, j] != 1:
                    self._state_list.append((i, j))

    def get_current_state(self) -> int:
        return self._current_state

    def set_current_state(self, state) -> None:
        self._current_state = state

    def get_step_count(self) -> int:
        return self._step_count

    def get_action_space(self) -> int:
        return 4

    def get_grid_space(self) -> int:
        return len(self._state_list)

    def get_state_space(self) -> int:
        return len(self._state_list) * 2


    def _is_valid_state(self, state_coord: tuple) -> bool:
        if self._is_door_state(state_coord):
            return False
        if state_coord[0] < 0 or state_coord[0] >= self._maze.shape[0]:
            return False
        if state_coord[1] < 0 or state_coord[1] >= self._maze.shape[1]:
            return False
        if self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["W"]:
            return False
        return True

    def _is_goal_state(self, state_coord: tuple) -> bool:
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["G"]

    def _is_trap_state(self, state_coord: tuple) -> bool:
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["T"]

    def _is_lava_state(self, state_coord: tuple) -> bool:
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["L"]

    def _is_door_state(self, state_coord: tuple) -> bool:
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["D"]

    def _is_key_state(self, state_coord: tuple) -> bool:
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["K"]

    def _is_exit_state(self, state_coord: tuple) -> bool:
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["E"]

    def _is_bait_state(self, state_coord: tuple) -> bool:
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["B"]

    def _is_portal_state(self, state_coord: tuple) -> bool:
        return self._maze[state_coord[0], state_coord[1]] == self.OBJECT_TO_INDEX["P"]


    @property
    def _is_closed(self):
        if self._door_state is None:
            return True
        return self._maze[self._state_list[self._door_state][0], self._state_list[self._door_state][1]] == self.OBJECT_TO_INDEX["D"]

    @property
    def _is_opened(self):
        if self._door_state is None:
            return False
        return self._maze[self._state_list[self._door_state][0], self._state_list[self._door_state][1]] == self.OBJECT_TO_INDEX[" "]

    @property
    def _is_baited(self):
        if self._bait_state is None:
            return False
        return self._maze[self._state_list[self._bait_state][0], self._state_list[self._bait_state][1]] == self.OBJECT_TO_INDEX[" "]


    def close_door(self):
        if self._door_state is None or self._is_closed:
            return
        self._maze[self._state_list[self._door_state][0],
                   self._state_list[self._door_state][1]] = self.OBJECT_TO_INDEX["D"]
        self.render_maze()

    def open_door(self):
        if self._door_state is None or self._is_opened:
            return
        self._maze[self._state_list[self._door_state][0],
                   self._state_list[self._door_state][1]] = self.OBJECT_TO_INDEX[" "]
        self.render_maze()

    def bite(self):
        if self._bait_state is None or self._is_baited:
            return
        self.step_reward = self._step_reward + self._bait_step_penalty
        self._maze[self._state_list[self._bait_state][0],
                   self._state_list[self._bait_state][1]] = self.OBJECT_TO_INDEX[" "]
        self.render_maze()

    def place_bait(self):
        if self._bait_state is None:
            return
        self.step_reward = self._step_reward
        self._maze[self._state_list[self._bait_state][0],
                   self._state_list[self._bait_state][1]] = self.OBJECT_TO_INDEX["B"]
        self.render_maze()

    def _get_next_state(self, state_coord: tuple, action: int) -> tuple:
        next_state_coord = np.array(state_coord)
        if action == 0:
            next_state_coord[0] -= 1
        elif action == 1:
            next_state_coord[0] += 1
        elif action == 2:
            next_state_coord[1] -= 1
        elif action == 3:
            next_state_coord[1] += 1
        if not self._is_valid_state(next_state_coord) and self._is_portal_state(state_coord):
            next_state_coord = self.portal_next_state[state_coord]
        if not self._is_valid_state(next_state_coord):
            next_state_coord = state_coord
        return tuple(next_state_coord)

    def step(self, action: int) -> tuple:
        self._step_count += 1
        Truncate = (self._step_count >= self.max_step)
        state_coord = self._state_list[self._current_state]
        if self._is_goal_state(state_coord):
            return self._current_state, self._goal_reward, True, Truncate
        if self._is_trap_state(state_coord):
            return self._current_state, self._trap_reward, True, Truncate
        if self._is_exit_state(state_coord):
            return self._current_state, self._exit_reward, True, Truncate

        next_state_coord = self._get_next_state(state_coord, action)
        next_state = self._state_list_index[next_state_coord]

        reward = self.step_reward

        if self._is_portal_state(next_state_coord) and self._current_state == next_state:
            next_state = next_state ^ self._portal_state[0] ^ self._portal_state[1]
        elif self._is_bait_state(next_state_coord):
            self.bite()
            reward = self._bait_reward
        elif self._is_key_state(state_coord):
            self.open_door()

        self._current_state = next_state

        return next_state + self._is_opened * (self._state_list_len), reward, (self._is_lava_state(next_state_coord)), Truncate

    def reset(self) -> int:
        self._step_count = 0
        self._current_state = np.random.choice(self._init_states)
        if self._is_opened:
            self.close_door()
        if self._is_baited:
            self.place_bait()
        return self._current_state


    def __str__(self):
        maze_str = f"Size: {self._maze.shape}\n"
        current_state_position = self._state_list[self._current_state]
        for i in range(self._maze.shape[0]):
            for j in range(self._maze.shape[1]):
                if (i, j) == current_state_position:
                    maze_str += "A"
                else:
                    maze_str += self.OBJECT_INDEX_TO_CHAR[self._maze[i, j]]
            maze_str += "\n"
        return maze_str

    def render_maze(self):
        num_colors = len(self.OBJECT_INDEX_TO_CHAR) - 1
        grid_colors = COLORS[:num_colors]
        cmap = colors.ListedColormap(grid_colors)
        self.ax.imshow(self._maze, cmap=cmap, vmin=0, vmax=num_colors)

    def render_init(self, title="GridWorld"):
        plt.close("all")

        self.fig, self.ax = plt.subplots(
            figsize=(self._maze.shape[1], self._maze.shape[0]))
        self.render_maze()
        self.ax.grid(which="major", axis="both",
                     linestyle="-", color="gray", linewidth=2)
        self.ax.set_xticks(np.arange(-0.5, self._maze.shape[1], 1))
        self.ax.set_yticks(np.arange(-0.5, self._maze.shape[0], 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.tick_params(length=0)
        self.state_to_text = {}
        self.previous_state = None
        text_count = 0

        for i in range(self._maze.shape[0]):
            for j in range(self._maze.shape[1]):
                if self._maze[i, j] == 1:
                    continue

                state = self._state_list_index[(i, j)]
                label = f"{state}"

                self.state_to_text[state] = text_count
                text_count += 1

                self.ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color="k",
                )

        if title is not None:
            plt.title(title)

        plt.tight_layout()

    def visualize(self, filename=None):
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def set_text_color(self, state, color):
        text_id = self.state_to_text[state]
        text = "Agent" if color == "b" else str(state)
        self.ax.texts[text_id].set(c=color, text=text)

    def rgb_render(
        self,
    ) -> np.ndarray | None:
        if self.previous_state is not None:
            self.set_text_color(self.previous_state, "k")
        self.set_text_color(self._current_state, "b")
        self.previous_state = self._current_state

        if self._step_count == 0:
            plt.pause(1)
        else:
            plt.pause(0.25)

    def get_rgb(self) -> np.ndarray:
        if self.previous_state is not None:
            self.set_text_color(self.previous_state, "k")
        self.set_text_color(self._current_state, "b")
        self.previous_state = self._current_state
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        data = np.asarray(buf)
        return data


class GridWorldEnv(gym.Env):
    def __init__(self, maze_file, goal_reward, trap_reward, step_reward, exit_reward, bait_reward, bait_step_penalty, max_step, render_mode="human") -> None:
        super(GridWorldEnv, self).__init__()
        self.render_mode = render_mode
        self.grid_world = GridWorld(maze_file, goal_reward, trap_reward,
                                    step_reward, exit_reward, bait_reward, bait_step_penalty, max_step)

        self.metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 60}
        self.action_space = spaces.Discrete(self.grid_world.get_action_space())
        self.observation_space = spaces.Discrete(self.grid_world.get_state_space())

    def reset(self, seed=None, **kwds: Any):
        next_state = self.grid_world.reset()
        return next_state, {}

    def step(self, action):
        next_state, reward, done, trucated = self.grid_world.step(action)
        return next_state, reward, done, trucated, {}

    def render(self, mode="human"):
        if self.render_mode == "ansi":
            print(self.grid_world)
        if self.render_mode == "human":
            self.grid_world.rgb_render()

    def seed(self, seed):
        pass
