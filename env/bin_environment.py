# env/bin_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.free_space_manager import FreeSpaceManager
from env.buffer_manager import BufferManager

class BinEnvironment(gym.Env):
    """
    Môi trường đóng gói 3D dạng online kiểu bottom-up sử dụng Gymnasium.

    Các thành phần:
      - Bin: kích thước [W, L, H] (số nguyên).
      - Mặt hàng: danh sách các item dạng [w, l, h] (số nguyên).
      - Buffer: chứa buffer_size item được cập nhật theo danh sách mặt hàng.
      - FreeSpaceManager: quản lý height_map của bin với grid cố định (grid_shape).
    
    Observation bao gồm:
      - height_map: mảng 2D kích thước (grid_w, grid_l) với các giá trị số nguyên.
      - buffer: mảng (buffer_size, 3) chứa các item hiện có.
      - action_mask: tensor (buffer_size, 2, grid_w, grid_l) cho biết vị trí hợp lệ
        để đặt từng item với mỗi rotation.
    
    Action là tuple (b, r, x, y):
      - b: chỉ số item trong buffer.
      - r: rotation (0: giữ nguyên, 1: xoay 90° => hoán đổi w và l).
      - x, y: tọa độ đặt item trên mặt đáy bin (số nguyên, trong [0,W] và [0,L]).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        """
        Khởi tạo môi trường với config chứa:
          - "W", "L", "H": kích thước bin (số nguyên).
          - "buffer_size": kích thước buffer.
          - "grid_shape": tuple (grid_w, grid_l), ví dụ (128,128).
          - "items_list": danh sách các mặt hàng, mỗi mặt hàng dạng [w, l, h] (số nguyên).
        """
        super(BinEnvironment, self).__init__()
        self.W = config["W"]
        self.L = config["L"]
        self.H = config["H"]
        self.buffer_size = config["buffer_size"]
        self.grid_shape = config["grid_shape"]  # (grid_w, grid_l)
        self.items_list = config["items_list"]

        # Khởi tạo FreeSpaceManager và BufferManager
        self.free_space_manager = FreeSpaceManager(self.W, self.L, self.H, self.grid_shape)
        self.buffer_manager = BufferManager(self.buffer_size, self.items_list)

        grid_w, grid_l = self.grid_shape

        # Định nghĩa observation space
        self.observation_space = spaces.Dict({
            "height_map": spaces.Box(low=0, high=self.H, shape=(grid_w, grid_l), dtype=np.int32),
            "buffer": spaces.Box(low=0, high=max(self.W, self.L, self.H), shape=(self.buffer_size, 3), dtype=np.int32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.buffer_size, 2, grid_w, grid_l), dtype=np.int32)
        })

        # Action space: (b, r, x, y) với x và y là số nguyên
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.buffer_size),  # chỉ số item trong buffer
            spaces.Discrete(2),                   # rotation: 0 hoặc 1
            spaces.Box(low=0, high=self.W, shape=(1,), dtype=np.int32),  # x
            spaces.Box(low=0, high=self.L, shape=(1,), dtype=np.int32)   # y
        ))

        self.current_step = 0
        self.done = False
        # Số bước tối đa bằng số mặt hàng (vì mỗi bước sử dụng 1 mặt hàng)
        self.max_steps = len(self.items_list)

        # Tham số reward
        self.valid_reward = 1  # thưởng đơn vị (có thể điều chỉnh)
        self.invalid_penalty = -1

    def reset(self, seed=None, options=None):
        """
        Reset môi trường về trạng thái ban đầu.
        """
        super().reset(seed=seed)
        self.free_space_manager.reset()
        self.buffer_manager.reset()
        self.current_step = 0
        self.done = False
        return self._get_obs(), {}

    def compute_full_action_mask(self):
        """
        Tính toán action mask với kích thước (buffer_size, 2, grid_w, grid_l).

        Với mỗi item trong buffer và mỗi rotation (0,1), quét toàn bộ grid để xác định
        xem đặt item (với footprint tương ứng và chiều cao của item) tại mỗi vị trí có hợp lệ hay không.
        """
        buffer_arr = self.buffer_manager.get_buffer()  # shape: (buffer_size, 3)
        grid_w, grid_l = self.grid_shape
        full_mask = np.zeros((self.buffer_size, 2, grid_w, grid_l), dtype=np.int32)

        for b in range(self.buffer_size):
            item = buffer_arr[b]  # [w, l, h]
            if np.array_equal(item, [0, 0, 0]):
                continue  # dummy item, mask toàn 0
            w, l, h = item
            for r in [0, 1]:
                if r == 1:
                    footprint_w, footprint_l = l, w
                else:
                    footprint_w, footprint_l = w, l
                mask = self.free_space_manager.compute_action_mask(footprint_w, footprint_l, h)
                full_mask[b, r] = mask
        return full_mask

    def _get_obs(self):
        """
        Tạo observation gồm:
          - height_map: trạng thái hiện tại của bin.
          - buffer: mảng các item trong buffer.
          - action_mask: tensor (buffer_size, 2, grid_w, grid_l).
        """
        height_map = self.free_space_manager.height_map.copy()
        buffer_arr = self.buffer_manager.get_buffer()
        action_mask = self.compute_full_action_mask()
        obs = {
            "height_map": height_map,
            "buffer": buffer_arr,
            "action_mask": action_mask
        }
        return obs

    def step(self, action):
        """
        Thực hiện hành động (b, r, x, y) và cập nhật môi trường.

        Quy trình:
          1. Lấy item từ buffer theo chỉ số b.
          2. Nếu r == 1, hoán đổi kích thước w và l.
          3. Kiểm tra tính hợp lệ của việc đặt item tại (x, y) qua is_valid_placement.
          4. Nếu hợp lệ, cập nhật height_map và buffer, tính reward theo thể tích (w * l * h).
             Nếu không hợp lệ, áp dụng penalty.
          5. Cập nhật bước và nếu đã sử dụng hết mặt hàng hoặc bin đầy, set done.
        """
        if self.done:
            return self._get_obs(), 0, True, False, {}

        b, r, x_arr, y_arr = action
        x = int(x_arr[0])
        y = int(y_arr[0])

        buffer_arr = self.buffer_manager.get_buffer()
        if b < 0 or b >= self.buffer_size:
            reward = self.invalid_penalty
            info = {"error": "Chỉ số item không hợp lệ."}
            return self._get_obs(), reward, False, False, info

        item = buffer_arr[b]  # [w, l, h]
        w, l, h = item
        if r == 1:
            w, l = l, w

        valid, base = self.free_space_manager.is_valid_placement(x, y, w, l, h)
        if valid:
            self.free_space_manager.update_height_map(x, y, w, l, h)
            self.buffer_manager.pop_item(b)
            reward = self.valid_reward * (w * l * h)
        else:
            reward = self.invalid_penalty

        self.current_step += 1
        if self.current_step >= self.max_steps or np.all(self.free_space_manager.height_map >= self.H):
            self.done = True

        obs = self._get_obs()
        return obs, reward, self.done, False, {}

    def render(self, mode="human"):
        """
        Render: hiển thị height_map dưới dạng text.
        """
        print("Height Map:")
        print(self.free_space_manager.height_map)

    def close(self):
        pass
