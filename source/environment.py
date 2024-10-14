# environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict

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

        self.current_item_index: int = 0
        self.height_map: np.ndarray = np.zeros((self.W, self.L), dtype=np.float32)
        self.placed_items: List[Dict] = [] # To store placed items' positions and rotations
        
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
        self.max_ems = self.W * self.L
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
        self.height_map = np.zeros((self.W, self.L), dtype=np.float32)
        self.placed_items = []
        self.ems_list = [(0, 0, 0, self.W, self.L, self.H)]

        # Initialize the buffer with k items
        self.buffer: List[Tuple[int, int, int]] = []
        for i in range(self.buffer_size):
            if i < len(self.items):
                self.buffer.append(self.items[i])
                self.current_item_index += 1
            else:
                self.buffer.append((0, 0, 0))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Perform a step in the environment.
        """
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Create an observation from the current state.
        """


    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """
        Decode an action into position x, y and rotation.
        """
    
    def _is_valid_position(self, x: int, y: int, rotation: int) -> bool:
        """
        Check if the item can be placed at the given position.
        - The item must be inside the bin.
        - The height must not exceed the bin height.
        - The item must be supported by at least 80% of its base area. 
        - The item must not overlap with other items.
        """
    
    def _place_item(self, x: int, y: int, rotation: int, item: Tuple[int, int, int]) -> float:
        """
        Update the height_map after placing the item.
        """
    
    def _get_rotated_item(self, item: Tuple[int, int, int], rotation: int) -> Tuple[int, int, int]:
        """
        Rotate the item based on the given rotation.
        """