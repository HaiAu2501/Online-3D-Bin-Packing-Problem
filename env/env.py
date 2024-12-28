import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns

from .ems import EMSManager

class BinPacking3DEnv(gym.Env):
    """
    Environment for the Online 3D Bin Packing problem.
    """
    def __init__(
        self,
        bin_size: Tuple[int, int, int],
        items: List[Tuple[int, int, int]],
        buffer_size: int = 2,
        num_rotations: int = 2,
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
          Each EMS is a vector of 6 integers: left-back-bottom corner and right-front-top corner. 
        - A mask for the EMS list to indicate the valid EMSs.
        - A mask for the action space to indicate the valid actions.
        """
        self.observation_space = spaces.Dict({
            'buffer': spaces.Box(
                low=0,
                high=max(self.W, self.L, self.H),
                shape=(self.buffer_size, 3),
                dtype=np.int32
            ),
            'ems_list': spaces.Sequence(
                spaces.Box(
                    low=0,
                    high=max(self.W, self.L, self.H),
                    shape=(6,),
                    dtype=np.int32
                )
            ),
            'action_mask': spaces.Box(
                low=0,
                high=1,
                shape=(self.W, self.L, self.num_rotations, self.buffer_size),
                dtype=np.int8
            ),
        })
        
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to the initial state or start a new episode.
        """
        self.current_item_index = 0
        self.height_map = np.zeros((self.W, self.L), dtype=np.int32)
        self.placed_items = []
        
        # Reset EMSManager
        self.ems_manager.reset()

        # Reset the buffer with buffer_size items
        self.buffer = []
        for i in range(self.buffer_size):
            if i < len(self.items):
                self.buffer.append(self.items[i])
                self.current_item_index += 1
            else:
                self.buffer.append((0, 0, 0))

        return self._get_observation(), {}
    
    def step(self, action: Optional[Tuple[int, int, int, int]]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Perform a step in the environment.

        :param action: The tuple of (x, y, rotation, item_index) or None.
        :return: A tuple of (observation, reward, done, truncated, info).

        - observation: The current observation.
        - reward: The agent receives a reward after placing an item.
        - done: The episode is done when all items are placed.
        - truncated: Set to be false.
        - info: Additional information.

        - Action is always valid because of the action mask we use in the training loop.
        - Action can be None (when the action mask is all zeros).
        """
        done = False
        truncated = False
        info = {}
        reward = 0.0

        # Unpack the action
        x, y, rotation, item_index = action 

        if self.action_mask[x, y, rotation, item_index] == 0:
            raise ValueError(f"Invalid action: {action}")

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
        reward += (rotated_w * rotated_l * rotated_h) / (self.W * self.L * self.H)

        # Update the buffer
        self.buffer.pop(item_index)
        if self.current_item_index < len(self.items):
            self.buffer.append(self.items[self.current_item_index])
            self.current_item_index += 1
        else:
            self.buffer.append((0, 0, 0))

        # Check if all items are placed
        if all(item == (0, 0, 0) for item in self.buffer) or not self.generate_action_mask().any():
            done = True

        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Create an observation from the current state.
        """
        ems_array = np.array(self.ems_manager.ems_list, dtype=np.int32)
        self.action_mask = self.generate_action_mask()

        return {
            'buffer': np.array(self.buffer, dtype=np.int32), # Size: (buffer_size, 3)
            'ems_list': ems_array, # Size: (num_ems, 6)
            'action_mask': self.action_mask, # Size: (W, L, num_rotations, buffer_size)
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
        self.valid_actions = []
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
                        self.valid_actions.append((x_pos, y_pos, rot, buffer_idx))

        return action_mask

    def render(self, **kwargs) -> None:
        """
        Render the environment.
        - verbose: Print all information.
        - buffer: Print the buffer.
        - ems_list: Print the EMS list.
        - height_map: Print the height map.
        - placed_items: Print the placed items.
        - action_mask: Print the action mask.
        """
        if kwargs['verbose']:
            kwargs['buffer'] = True
            kwargs['ems_list'] = True
            kwargs['height_map'] = True
            kwargs['placed_items'] = True
            kwargs['action_mask'] = True

        if kwargs['buffer']:
            print("- Buffer:")
            for item in self.buffer:
                print(item)
            print()
        
        if kwargs['ems_list']:
            print("- EMS List:")
            self.ems_manager.print_ems_list()
            print()

        if kwargs['height_map']:
            print("- Height Map:")
            for i in range(self.W):
                for j in range(self.L):
                    print(f"{self.height_map[i][j]:2d}", end=" ")
                print()
            print()

        if kwargs['placed_items']:
            print("- Placed Items:")
            for item in self.placed_items:
                print(item)
            print()

        if kwargs['action_mask']:
            print("- Action Mask:")
            for k in range(self.buffer_size):
                print(f"Item {k} in the buffer with size {self.buffer[k]}")
                for rot in range(self.num_rotations):
                    w, l, h = self._get_rotated_item(self.buffer[k], rot)
                    print(f"Rotation {rot} with size {w} x {l} x {h}")
                    for i in range(self.W):
                        for j in range(self.L):
                            print(self.action_mask[i, j, rot, k], end=" ")
                        print()
                    print()

        print("--------------------")

    def visualize(self) -> None:
        """
        Visualize the 3D bin packing problem.
        """
        if not self.placed_items:
            print("No items have been placed yet.")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Vẽ khung thùng hàng
        W, L, H = self.W, self.L, self.H
        bin_corners = [
            [0, 0, 0],
            [W, 0, 0],
            [W, L, 0],
            [0, L, 0],
            [0, 0, H],
            [W, 0, H],
            [W, L, H],
            [0, L, H],
        ]

        bin_faces = [
            [bin_corners[0], bin_corners[1], bin_corners[2], bin_corners[3]],
            [bin_corners[4], bin_corners[5], bin_corners[6], bin_corners[7]],
            [bin_corners[0], bin_corners[1], bin_corners[5], bin_corners[4]],
            [bin_corners[2], bin_corners[3], bin_corners[7], bin_corners[6]],
            [bin_corners[1], bin_corners[2], bin_corners[6], bin_corners[5]],
            [bin_corners[4], bin_corners[7], bin_corners[3], bin_corners[0]],
        ]

        bin_collection = Poly3DCollection(bin_faces, linewidths=1, edgecolors='black', alpha=0.1)
        bin_collection.set_facecolor((0.5, 0.5, 0.5, 0.1))  # Màu xám nhạt cho thùng
        ax.add_collection3d(bin_collection)

        # Tạo danh sách màu sắc khác nhau
        num_items = len(self.placed_items)
        palette = sns.color_palette("Set2", num_items)

        for idx, item in enumerate(self.placed_items):
            x, y, z = item['position']
            w, l, h = item['size']

            # Định nghĩa các góc của hộp
            item_corners = [
                [x, y, z],
                [x + w, y, z],
                [x + w, y + l, z],
                [x, y + l, z],
                [x, y, z + h],
                [x + w, y, z + h],
                [x + w, y + l, z + h],
                [x, y + l, z + h],
            ]

            # Định nghĩa các mặt của hộp
            item_faces = [
                [item_corners[0], item_corners[1], item_corners[2], item_corners[3]],
                [item_corners[4], item_corners[5], item_corners[6], item_corners[7]],
                [item_corners[0], item_corners[1], item_corners[5], item_corners[4]],
                [item_corners[2], item_corners[3], item_corners[7], item_corners[6]],
                [item_corners[1], item_corners[2], item_corners[6], item_corners[5]],
                [item_corners[4], item_corners[7], item_corners[3], item_corners[0]],
            ]

            # Tạo Poly3DCollection cho mỗi mục
            item_color = palette[idx % len(palette)]
            item_collection = Poly3DCollection(item_faces, linewidths=1, edgecolors='black', alpha=0.5)
            item_collection.set_facecolor(item_color)
            ax.add_collection3d(item_collection)

        # Thiết lập tỷ lệ trục bằng nhau
        max_range = np.array([W, L, H]).max()
        ax.set_xlim(0, max_range)
        ax.set_ylim(0, max_range)
        ax.set_zlim(0, max_range)

        # Thiết lập nhãn trục
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Thiết lập góc nhìn (tuỳ chọn)
        ax.view_init(elev=20., azim=30)

        plt.title('3D Bin Packing Visualization')
        plt.show()

    def clone(self) -> 'BinPacking3DEnv':
        """
        Clone the environment.
        """
        cloned_env = BinPacking3DEnv(
            bin_size=(self.W, self.L, self.H),
            items=self.items.copy(),
            buffer_size=self.buffer_size,
            num_rotations=self.num_rotations
        )
        cloned_env.current_item_index = self.current_item_index
        cloned_env.height_map = self.height_map.copy()
        cloned_env.placed_items = self.placed_items.copy()
        cloned_env.buffer = self.buffer.copy()
        cloned_env.ems_manager = self.ems_manager.clone()

        return cloned_env

    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        pass    