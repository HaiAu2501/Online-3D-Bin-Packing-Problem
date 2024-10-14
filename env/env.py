# env/env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict
from env.ems import EMSManager

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
        '''
        :param bin_size: The size of the bin (W, L, H).
        :param items: A list of items, each item is a vector of 3 integers (w, l, h).
        :param buffer_size: The size of the buffer to store k items.
        :param num_rotations: The number of rotations for each item.
        
        The default number of rotations is 2 because robots can only rotate items by 90 degrees.
        '''
        self.W, self.L, self.H = bin_size
        self.items: List[Tuple[int, int, int]] = items
        self.buffer_size: int = buffer_size
        self.num_rotations = num_rotations

        if self.buffer_size > len(self.items):
            raise ValueError("Buffer size must be less than the number of items.")

        if self.num_rotations < 1 or self.num_rotations > 6:
            raise ValueError("The number of rotations must be between 1 and 6.")

        self.current_item_index: int = 0
        self.height_map: np.ndarray = np.zeros((self.W, self.L), dtype=np.float32)
        self.placed_items: List[Dict] = [] # To store placed items' positions and rotations

        # Intialize EMSManager
        self.ems_manager = EMSManager(bin_size=bin_size)
        
        '''
        Define action space:
        - Action is a tuple of 4 integers (x, y, rotation, item_index).
        - x: Position x in the bin (0 to W - 1).
        - y: Position y in the bin (0 to L - 1).
        - rotation: Rotation of the item (0 to num_rotations - 1).
        - item_index: Index of the item in the buffer (0 to buffer_size - 1).
        - The total number of actions is W x L x (num_rotations) x (buffer_size).
        '''
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.W), # x
            spaces.Discrete(self.L), # y
            spaces.Discrete(self.num_rotations), # rotation
            spaces.Discrete(self.buffer_size) # item_index
        ))
        
        '''
        Define observation space:
        - A buffer with k items, each item is a vector of 3 integers (width, length, height).
        - A list of empty maximal spaces (EMSs) in the bin.
        - Each EMS is a vector of 6 integers: left-back-bottom corner and right-front-top corner. 
        '''
        self.max_ems = self.W * self.L * self.H
        self.observation_space = spaces.Dict({
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
            'action_mask': spaces.Box(
                low=0, 
                high=1, 
                shape=(self.W, self.L, self.num_rotations, self.buffer_size),
                dtype=np.int8
            )
        })

        self.ems_list: List[Tuple[int, int, int, int, int, int]] = [(0, 0, 0, self.W, self.L, self.H)]
        
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
    
    def step(self, action: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Perform a step in the environment.

        :param action: The tuple of (x, y, rotation, item_index).
        :return: A tuple of (observation, reward, done, truncated, info).
        """
        done = False
        truncated = False
        info = {}
        reward = 0.0

        # Unpack the action
        x, y, rotation, item_index = action

        selected_item = self.buffer[item_index]
        if selected_item == (0, 0, 0):
            # Skip the action if the item is not selected
            done = True
            return self._get_observation(), reward, done, truncated, info

        rotated_item = self._get_rotated_item(selected_item, rotation)
        rotated_w, rotated_l, rotated_h = rotated_item

        # Determine the z-coordinate of the item based on the height map
        z = max([self.height_map[xi][y_] for xi in range(x, x + rotated_w) for y_ in range(y, y + rotated_l)])
        

    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Create an observation from the current state.
        """
    
    def _place_item(self, x: int, y: int, rotation: int, item: Tuple[int, int, int]) -> float:
        """
        Update the height_map after placing the item.
        """
    
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