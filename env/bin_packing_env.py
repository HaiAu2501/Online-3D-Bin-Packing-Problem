import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, List, Optional, Any, Union


class BinPackingEnv(gym.Env):
    """3D Bin Packing Environment using Gymnasium API.
    
    This environment implements the Online 3D Bin Packing problem as described in
    the documentation. Items arrive sequentially and must be placed in a bin.
    
    State representation follows the algorithm flow:
        - Bin kích thước: [W, L, H] (số nguyên).
        - Occupancy Map: O ∈ {0,1}^(W * L * H).
        - Height Map: H_t ∈ Z^(W * L), với H_t(x,y)= max{ z | O(x,y,z)=1 }.
        - Buffer: B_t ∈ Z^(B * 3) với mỗi vật phẩm s_i = [w_i, l_i, h_i].
        - State: s_t = { H_t, B_t, O }.
    
    Action:
        - a_t = (b, r, x, y): vật phẩm trong buffer, xoay, vị trí x, y
    
    Reward:
        - r_t = (w * l * h) / (W * L * H) - tỷ lệ thể tích sử dụng
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self,
        bin_size: List[int] = [10, 10, 10],  # [W, L, H]
        buffer_size: int = 5,
        item_list: Optional[List[List[int]]] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize the 3D Bin Packing Environment.
        
        Args:
            bin_size: Dimensions of the bin [width, length, height]
            buffer_size: Number of items in the buffer B
            item_list: List of items, each with dimensions [w, l, h]
            seed: Random seed
            render_mode: Rendering mode (human or ansi)
        """
        super().__init__()
        
        self.bin_size = bin_size
        self.W, self.L, self.H = bin_size
        self.buffer_size = buffer_size
        self.item_list = item_list if item_list is not None else []
        self.render_mode = render_mode
        
        # Determine max item size for observation space
        max_item_size = max(max(bin_size), 1)
        if self.item_list:
            for item in self.item_list:
                max_item_size = max(max_item_size, max(item))

        # Thêm max_items dựa trên độ dài của item_list
        self.max_items = len(self.item_list) if self.item_list else 0
        
        # Define action spaces
        self.action_space = spaces.MultiDiscrete([
            buffer_size,  # b: Item index in buffer (0 to B-1)
            2,            # r: Rotation (0 or 1)
            self.W,       # x: X position (0 to W-1)
            self.L        # y: Y position (0 to L-1)
        ])
        
        # Define observation spaces
        self.observation_space = spaces.Dict({
            # Height map: 2D grid storing the max height at each (x,y) position
            'height_map': spaces.Box(
                low=0, 
                high=self.H, 
                shape=(self.W, self.L), 
                dtype=np.int32
            ),
            # Buffer: B items, each with dimensions [w, l, h]
            'buffer': spaces.Box(
                low=0,
                high=max_item_size,
                shape=(buffer_size, 3),
                dtype=np.int32
            ),
            # Occupancy map: 3D grid with binary values (0: empty, 1: occupied)
            'occupancy_map': spaces.Box(
                low=0,
                high=1,
                shape=(self.W, self.L, self.H),
                dtype=np.int8
            )
        })
        
        # Initialize random number generator
        self.np_random = np.random.RandomState(seed)
        
        # State variables
        self.height_map = None
        self.buffer = None
        self.occupancy_map = None
        self.remaining_items = None
        
        # Episode tracking
        self.current_step = None
        self.items_placed = None
        self.volume_utilized = None
    
    def set_item_list(self, item_list: List[List[int]]):
        """Set or update the item list for the environment.
        
        Args:
            item_list: List of items, each with dimensions [w, l, h]
        """
        self.item_list = item_list.copy()
        self.max_items = len(self.item_list)
    
    def _refill_buffer(self):
        """Refill the buffer with items from the remaining items queue.
        
        When an item is removed from the buffer, a new item from the queue
        is added to maintain the buffer size.
        """
        # Determine how many items needed to refill buffer
        num_to_add = self.buffer_size - len(self.buffer)
        
        if num_to_add > 0:
            # Add items from remaining_items if available
            if len(self.remaining_items) > 0:
                # Take items from remaining_items
                items_to_add = self.remaining_items[:num_to_add]
                self.remaining_items = self.remaining_items[num_to_add:]
                
                # Add to buffer
                self.buffer = np.vstack([self.buffer, items_to_add])
            else:
                # No more real items, add [0,0,0] placeholders
                zero_items = np.zeros((num_to_add, 3), dtype=np.int32)
                self.buffer = np.vstack([self.buffer, zero_items])
    
    def _get_support_ratio(self, item: np.ndarray, rotation: int, x: int, y: int) -> float:
        """Calculate the support ratio for an item at a given position.
        
        Args:
            item: Item dimensions [w, l, h]
            rotation: Rotation (0 or 1)
            x: X position
            y: Y position
            
        Returns:
            Support ratio (percentage of item base supported)
        """
        w, l, h = item
        
        # Apply rotation if needed
        if rotation == 1:
            w, l = l, w
        
        # Get the region beneath the item
        base_heights = self.height_map[x:x+w, y:y+l]
        
        # Get max height (where the item will be placed)
        max_height = np.max(base_heights)
        
        # Count support points (positions where height equals max_height)
        support_area = np.sum(base_heights == max_height)
        total_area = w * l
        
        return support_area / total_area
    
    def _is_valid_placement(self, item_idx: int, rotation: int, x: int, y: int) -> bool:
        """Check if item placement is valid.
        
        Args:
            item_idx: Index of the item in the buffer
            rotation: Rotation (0 or 1)
            x: X position
            y: Y position
            
        Returns:
            True if placement is valid, False otherwise
        """
        # Check if item index is valid
        if item_idx < 0 or item_idx >= len(self.buffer):
            return False
        
        # Get item dimensions
        item = self.buffer[item_idx]
        w, l, h = item
        
        # Check if item is a placeholder [0,0,0]
        if w == 0 and l == 0 and h == 0:
            return False
        
        # Apply rotation
        if rotation == 1:
            w, l = l, w
        
        # Check if item placement is within bin boundaries
        if x < 0 or y < 0 or x + w > self.W or y + l > self.L:
            return False
        
        # Get region beneath the item
        try:
            base_heights = self.height_map[x:x+w, y:y+l]
        except IndexError:
            return False
        
        if base_heights.size == 0:
            return False
        
        # Find the height at which the item should be placed
        base_height = np.max(base_heights)
        
        # Check if item fits within height constraint
        if base_height + h > self.H:
            return False
        
        # Check support constraint (at least 80% of the base must be supported)
        # Ground level (height 0) is always fully supported
        if base_height > 0:
            support_ratio = self._get_support_ratio(item, rotation, x, y)
            if support_ratio < 0.8:  # Based on support ratio requirement
                return False
        
        return True
    
    def _place_item(self, item_idx: int, rotation: int, x: int, y: int) -> float:
        """Place an item in the bin.
        
        Args:
            item_idx: Index of the item in the buffer
            rotation: Rotation (0 or 1)
            x: X position
            y: Y position
            
        Returns:
            Reward for placing the item
        """
        # Get item dimensions
        item = self.buffer[item_idx].copy()
        w, l, h = item
        
        # Apply rotation
        if rotation == 1:
            w, l = l, w
        
        # Find the height at which to place the item
        base_heights = self.height_map[x:x+w, y:y+l]
        base_height = np.max(base_heights)
        
        # Update height map - set new heights for the area covered by the item
        self.height_map[x:x+w, y:y+l] = base_height + h
        
        # Update occupancy map - mark all voxels as occupied
        for k in range(base_height, base_height + h):
            self.occupancy_map[x:x+w, y:y+l, k] = 1
        
        # Calculate reward as volume utilization (as per the algorithm description)
        item_volume = w * l * h
        reward = item_volume / (self.W * self.L * self.H)
        
        # Remove placed item from buffer
        mask = np.ones(len(self.buffer), dtype=bool)
        mask[item_idx] = False
        self.buffer = self.buffer[mask]
        
        # Refill the buffer
        self._refill_buffer()
        
        # Update statistics
        self.items_placed += 1
        self.volume_utilized += item_volume
        
        return reward
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (item_list can be provided here)
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Process options if provided
        if options and 'item_list' in options:
            self.item_list = options['item_list']
            self.max_items = len(self.item_list)
            
        # Check if item_list is available
        if not self.item_list:
            raise ValueError("No item list provided. Set item_list during initialization or through reset options.")
        
        # Initialize height map (all positions start at height 0)
        self.height_map = np.zeros((self.W, self.L), dtype=np.int32)
        
        # Initialize occupancy map (all positions empty)
        self.occupancy_map = np.zeros((self.W, self.L, self.H), dtype=np.int8)
        
        # Convert item list to numpy array if needed
        item_array = np.array(self.item_list, dtype=np.int32)
        
        # Split items into buffer and remaining items
        self.buffer = item_array[:min(self.buffer_size, len(item_array))]
        self.remaining_items = item_array[min(self.buffer_size, len(item_array)):]
        
        # Ensure buffer is correct size (add zero items if needed)
        if len(self.buffer) < self.buffer_size:
            zero_items = np.zeros((self.buffer_size - len(self.buffer), 3), dtype=np.int32)
            self.buffer = np.vstack([self.buffer, zero_items])
        
        # Reset episode tracking
        self.current_step = 0
        self.items_placed = 0
        self.volume_utilized = 0
        
        # Prepare observation
        observation = {
            'height_map': self.height_map,
            'buffer': self.buffer,
            'occupancy_map': self.occupancy_map
        }
        
        info = {
            'bin_size': self.bin_size,
            'items_placed': self.items_placed,
            'volume_utilized': self.volume_utilized,
            'total_items': len(self.item_list)
        }
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: (b, r, x, y) - Item index, rotation, x position, y position
            
        Returns:
            observation: New state
            reward: Reward for the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        self.current_step += 1
        
        # Parse action
        item_idx, rotation, x, y = action
        
        # Check if action is valid
        valid_action = self._is_valid_placement(item_idx, rotation, x, y)
        
        # Place item if valid, otherwise give zero reward
        if valid_action:
            reward = self._place_item(item_idx, rotation, x, y)
        else:
            reward = 0
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Episode ends when all non-zero items have been placed
        # (all items in buffer are [0,0,0] placeholders)
        if np.all(self.buffer == 0):
            terminated = True
        
        # Truncate if max steps reached (avoid infinite loops)
        if self.current_step >= self.max_items * 2:
            truncated = True
        
        # Prepare observation
        observation = {
            'height_map': self.height_map,
            'buffer': self.buffer,
            'occupancy_map': self.occupancy_map
        }
        
        info = {
            'valid_action': valid_action,
            'items_placed': self.items_placed,
            'volume_utilized': self.volume_utilized,
            'volume_utilization_ratio': self.volume_utilized / (self.W * self.L * self.H)
        }
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "ansi":
            return self._render_frame()
        else:  # human mode
            self._render_frame()
            return None
    
    def _render_frame(self):
        """Render a simple text representation of the environment."""
        output = []
        
        # Add header information
        output.append(f"3D Bin Packing - Items Placed: {self.items_placed}")
        output.append(f"Bin Size: {self.W}×{self.L}×{self.H}")
        output.append(f"Volume Utilization: {self.volume_utilized/(self.W*self.L*self.H):.2%}")
        output.append("")
        
        # Display height map
        output.append("Height Map:")
        for y in range(self.L):
            row = []
            for x in range(self.W):
                height = self.height_map[x, y]
                if height == 0:
                    row.append(".")
                else:
                    row.append(str(min(height, 9)))  # Limit display to single digit
            output.append(" ".join(row))
        output.append("")
        
        # Display buffer items
        output.append("Buffer Items:")
        for i, item in enumerate(self.buffer):
            w, l, h = item
            if w == 0 and l == 0 and h == 0:
                output.append(f"{i}: Empty")
            else:
                output.append(f"{i}: {w} * {l} * {h}")
        
        # Join all lines into a single string
        rendered = "\n".join(output)
        
        # Print or return based on render mode
        if self.render_mode == "human":
            print(rendered)
        
        return rendered
    
    def close(self):
        """Close the environment."""
        pass