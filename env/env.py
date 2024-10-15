# env/env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict
from ems import EMSManager

class BinPacking3DEnv(gym.Env):
    """
    Environment for the Online 3D Bin Packing problem.
    """
    def __init__(
        self,
        bin_size: Tuple[int, int, int],
        items: List[Tuple[int, int, int]],
        buffer_size: int = 2,
        num_rotations: int = 2
    ):
        super(BinPacking3DEnv, self).__init__()
        """
        :param bin_size: The size of the bin (W, L, H).
        :param items: A list of items, each item is a vector of 3 integers (w, l, h).
        :param buffer_size: The size of the buffer to store k items.
        :param num_rotations: The number of rotations for each item.
        
        The default number of rotations is 2 because robots can only rotate items by 90 degrees.
        """
        self.W, self.L, self.H = bin_size
        self.items: List[Tuple[int, int, int]] = items
        self.buffer_size: int = buffer_size
        self.num_rotations = num_rotations

        if self.buffer_size > len(self.items):
            raise ValueError("Buffer size must be less than the number of items.")

        if self.num_rotations < 1 or self.num_rotations > 6:
            raise ValueError("The number of rotations must be between 1 and 6.")

        self.current_item_index: int = 0
        self.height_map: np.ndarray = np.zeros((self.W, self.L), dtype=np.int32)
        self.placed_items: List[Dict] = [] # To store placed items' positions and rotations

        # Intialize EMSManager
        self.ems_manager = EMSManager(bin_size=bin_size)
        
        """
        Define action space:
        - Action is a tuple of 4 integers (x, y, rotation, item_index).
        - x: Position x in the bin (0 to W - 1).
        - y: Position y in the bin (0 to L - 1).
        - rotation: Rotation of the item (0 to num_rotations - 1).
        - item_index: Index of the item in the buffer (0 to buffer_size - 1).
        - The total number of actions is W x L x (num_rotations) x (buffer_size).
        """
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.W), # x
            spaces.Discrete(self.L), # y
            spaces.Discrete(self.num_rotations), # rotation
            spaces.Discrete(self.buffer_size) # item_index
        ))
        
        """
        Define observation space:
        - A buffer with k items, each item is a vector of 3 integers (width, length, height).
        - A list of empty maximal spaces (EMSs) in the bin.
        - Each EMS is a vector of 6 integers: left-back-bottom corner and right-front-top corner. 
        """
        self.max_ems = self.W * self.L * self.H
        self.observation_space = spaces.Dict({
            # 'height_map': spaces.Box(
            #     low=0,
            #     high=self.H,
            #     shape=(self.W, self.L),
            #     dtype=np.int32
            # ),
            'buffer': spaces.Box(
                low=0,
                high=max(self.W, self.L, self.H),
                shape=(self.buffer_size, 3), 
                dtype=np.int32
            ),
            'ems': spaces.Box(
                low=0,
                high=max(self.W, self.L, self.H),
                shape=(self.max_ems, 6),
                dtype=np.int32
            ),
        })
        
    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.current_item_index = 0
        self.height_map = np.zeros((self.W, self.L), dtype=int)
        self.placed_items = []
        
        # Reset EMSManager
        self.ems_manager = EMSManager(bin_size=(self.W, self.L, self.H))

        # Initialize the buffer with k items
        self.buffer: List[Tuple[int, int, int]] = []
        for i in range(self.buffer_size):
            if i < len(self.items):
                self.buffer.append(self.items[i])
                self.current_item_index += 1
            else:
                self.buffer.append((0, 0, 0))
    
    def step(self, action: Tuple[int, int, int, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Perform a step in the environment.

        :param action: The tuple of (x, y, rotation, item_index).
        :return: A tuple of (observation, reward, done, truncated, info).

        - observation: The current observation.
        - reward: The agent receives a reward after placing an item.
        - done: The episode is done when all items are placed.
        - truncated: Set to be false.
        - info: Additional information.

        - Action is always valid because of the action mask we use in the training loop.
        """
        done = False
        truncated = False
        info = {}
        reward = 0.0

        # Unpack the action
        x, y, rotation, item_index = action 

        selected_item = self.buffer[item_index]
        rotated_item = self._get_rotated_item(selected_item, rotation)
        rotated_w, rotated_l, rotated_h = rotated_item

        # Determine z-coordinate based on the height map (place the item on top of the existing items)
        z = max([self.height_map[xi][yi] for xi in range(x, x + rotated_w) for yi in range(y, y + rotated_l)])

        # Update the height map after placing the item
        for xi in range(x, x + rotated_w):
            for yi in range(y, y + rotated_l):
                self.height_map[xi][yi] = z + rotated_h

        # Update the EMS list after placing the item
        self.ems_manager.update_ems_after_placement((x, y, z, rotated_w, rotated_l, rotated_h))

        # Record the placed item
        self.placed_items.append({
            'position': (x, y, z),
            'size': rotated_item,
            'rotation': rotation,
        })

        # Reward is the percentage of the item's volume in the bin
        reward += (100 * rotated_w * rotated_l * rotated_h) / (self.W * self.L * self.H)

        # Update the buffer
        self.buffer.pop(item_index)
        if self.current_item_index < len(self.items):
            self.buffer.append(self.items[self.current_item_index])
            self.current_item_index += 1
        else:
            self.buffer.append((0, 0, 0))

        # Check if all items are placed
        if all(item == (0, 0, 0) for item in self.buffer):
            done = True
            reward += 100.0
            info['sucess'] = True

        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Create an observation from the current state.
        """
        return {
            # 'height_map': self.height_map,
            'buffer': np.array(self.buffer, dtype=np.int32),
            'ems': np.array(self.ems_manager.ems_list, dtype=np.int32)
        }
    
    def _get_rotated_item(self, item: Tuple[int, int, int], rotation: int) -> Tuple[int, int, int]:
        """
        Rotate the item based on the given rotation.
        """
        w, l, h = item
        if rotation == 0:
            return w, l, h
        elif rotation == 1:
            return l, w, h
        elif rotation == 2:
            return w, h, l
        elif rotation == 3:
            return h, w, l
        elif rotation == 4:
            return l, h, w
        elif rotation == 5:
            return h, l, w
        else:
            raise ValueError(f"Invalid rotation: {rotation}")

    def generate_action_mask(self) -> np.ndarray:
        """
        Get the action mask for the current state.

        :return: A 4D array of shape (W, L, num_rotations, buffer_size) with 0s and 1s.

        - 0: The action is invalid.
        - 1: The action is valid.
        """
        action_mask = np.zeros((self.W, self.L, self.num_rotations, self.buffer_size), dtype=np.int8)

        for buffer_idx, item in enumerate(self.buffer):
            if item == (0, 0, 0):
                # All actions are invalid (this is a padding item)
                continue

            for rot in range(self.num_rotations):
                rotated_box = self._get_rotated_item(item, rot)
                rotated_w, rotated_l, rotated_h = rotated_box

                # Iterate over all possible (x, y) positions
                for x_pos in range(self.W - rotated_w + 1):
                    for y_pos in range(self.L - rotated_l + 1):
                        # Determine z based on height_map (placement on top)
                        z = max([self.height_map[xi][yi] for xi in range(x_pos, x_pos + rotated_w) for yi in range(y_pos, y_pos + rotated_l)])

                        # Check height constraint
                        if z + rotated_h > self.H:
                            continue

                        # Check support ratio
                        supported_cells = 0
                        total_cells = rotated_w * rotated_l
                        for xi in range(x_pos, x_pos + rotated_w):
                            for yi in range(y_pos, y_pos + rotated_l):
                                if self.height_map[xi][yi] >= z:
                                    supported_cells += 1
                        support_ratio = supported_cells / total_cells
                        if support_ratio < 0.8:
                            continue  # Insufficient support

                        action_mask[x_pos, y_pos, rot, buffer_idx] = 1

        return action_mask

    def render(self):
        """
        Render the environment.
        """
        print("\nCurrent height map:")
        for y in reversed(range(self.height_map.shape[1])):
            row = ""
            for x in range(self.height_map.shape[0]):
                row += f"{self.height_map[x][y]:2} "
            print(row)
        self.ems_manager.print_ems_list()
        print("\nBuffer:")
        for item in self.buffer:
            print(item)

    def close(self):
        """
        Close the environment.
        """
        pass