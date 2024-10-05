# environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict

class BinPacking3DEnv(gym.Env):
    """
    Môi trường Gymnasium cho bài toán Online 3D Bin Packing.
    """
    def __init__(self, bin_size: Tuple[int, int, int], items: List[Tuple[int, int, int]]):
        super(BinPacking3DEnv, self).__init__()
        self.W, self.L, self.H = bin_size
        self.items = items
        self.current_item_index = 0
        self.height_map = np.zeros((self.W, self.L), dtype=np.float32)
        self.current_height = np.zeros((self.W, self.L), dtype=np.float32)
        
        # Định nghĩa không gian hành động:
        # Hành động là vị trí x, y và hướng xoay (6 hướng)
        self.num_rotations = 6
        self.action_space = spaces.Discrete(self.W * self.L * self.num_rotations)
        
        # Định nghĩa không gian quan sát:
        # 4 kênh: height_map và 3 kênh cho kích thước của vật phẩm hiện tại
        self.observation_space = spaces.Box(
            low=0, high=max(self.W, self.L, self.H),
            shape=(4, self.W, self.L), dtype=np.float32
        )
        
    def reset(self):
        """
        Reset môi trường về trạng thái ban đầu.
        """
        self.current_item_index = 0
        self.height_map = np.zeros((self.W, self.L), dtype=np.float32)
        self.current_height = np.zeros((self.W, self.L), dtype=np.float32)
        observation = self._get_observation()
        return observation, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Thực hiện một bước trong môi trường.
        """
        x, y, rotation = self._decode_action(action)
        if not self._is_valid_position(x, y, rotation):
            # Nếu hành động không hợp lệ, kết thúc episode với phần thưởng âm
            reward = -1.0
            done = True
            return self._get_observation(), reward, done, False, {}
        
        # Đặt vật phẩm vào vị trí (x, y) với hướng xoay
        item = self.items[self.current_item_index]
        placed_volume = self._place_item(x, y, rotation, item)
        reward = (item[0] * item[1] * item[2]) / (self.W * self.L * self.H)
        
        self.current_item_index += 1
        done = self.current_item_index >= len(self.items)
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Tạo quan sát từ trạng thái hiện tại.
        """
        if self.current_item_index < len(self.items):
            current_item = self.items[self.current_item_index]
            # Mở rộng và xếp chồng cùng height map theo trục kênh
            item_tensor = np.zeros((3, self.W, self.L), dtype=np.float32)
            item_tensor[0, :, :] = current_item[0]
            item_tensor[1, :, :] = current_item[1]
            item_tensor[2, :, :] = current_item[2]
        else:
            item_tensor = np.zeros((3, self.W, self.L), dtype=np.float32)
        
        observation = np.concatenate([self.height_map[np.newaxis, :, :], item_tensor], axis=0)
        return observation
    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """
        Giải mã hành động thành vị trí x, y và hướng xoay.
        """
        rotation = action % self.num_rotations
        pos = action // self.num_rotations
        x = pos // self.L
        y = pos % self.L
        return x, y, rotation
    
    def _is_valid_position(self, x: int, y: int, rotation: int) -> bool:
        """
        Kiểm tra xem vị trí đặt vật phẩm có hợp lệ hay không.
        """
        if self.current_item_index >= len(self.items):
            return False  # Không còn vật phẩm để đặt
        
        item = self.items[self.current_item_index]
        rotated_item = self._get_rotated_item(item, rotation)
        w, l, h = rotated_item
        
        # Kiểm tra giới hạn thùng
        if x < 0 or y < 0 or (x + w) > self.W or (y + l) > self.L:
            return False  # Vượt ra ngoài thùng
        
        # Kiểm tra chiều cao không vượt quá thùng
        current_max_height = np.max(self.height_map[x:x + w, y:y + l])
        if (current_max_height + h) > self.H:
            return False  # Vượt quá chiều cao thùng
        
        # Kiểm tra hỗ trợ 60% diện tích đáy
        base_area = w * l
        
        if current_max_height == 0:
            # Nếu đặt trên mặt đáy thùng, toàn bộ diện tích đáy được hỗ trợ
            return True
        else:
            # Đặt trên vật phẩm khác, đảm bảo ít nhất 60% diện tích đáy được hỗ trợ bởi vật phẩm khác
            support_area = np.sum(self.height_map[x:x + w, y:y + l] > 0)
            required_support = 0.6 * base_area
            return support_area >= required_support
    
    def _place_item(self, x: int, y: int, rotation: int, item: Tuple[int, int, int]) -> float:
        """
        Cập nhật height_map sau khi đặt vật phẩm.
        """
        rotated_item = self._get_rotated_item(item, rotation)
        w, l, h = rotated_item
        
        # Cập nhật height_map: tăng chiều cao tại vùng đặt vật phẩm
        current_region = self.height_map[x:x + w, y:y + l]
        new_height = current_region + h
        self.height_map[x:x + w, y:y + l] = new_height
        
        # Trả về thể tích đã đặt
        return (w * l * h) / (self.W * self.L * self.H)
    
    def _get_rotated_item(self, item: Tuple[int, int, int], rotation: int) -> Tuple[int, int, int]:
        """
        Lấy kích thước vật phẩm sau khi xoay dựa trên hướng xoay.
        Giả sử 6 hướng xoay tương ứng với các cách xoay khác nhau của vật phẩm.
        """
        w, l, h = item
        if rotation == 0:
            return (w, l, h)
        elif rotation == 1:
            return (w, h, l)
        elif rotation == 2:
            return (l, w, h)
        elif rotation == 3:
            return (l, h, w)
        elif rotation == 4:
            return (h, w, l)
        elif rotation == 5:
            return (h, l, w)
        else:
            return (w, l, h)  # Default không xoay
