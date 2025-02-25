# env/free_space_manager.py
import numpy as np
import math

class FreeSpaceManager:
    """
    Quản lý free-space của bin thông qua height_map.

    - height_map là mảng 2D với kích thước cố định (grid_w x grid_l),
      mỗi ô biểu diễn chiều cao hiện tại tại vị trí đó (số nguyên).
    - Khi đặt một item, đáy của item sẽ "tiếp xúc" với ô có giá trị
      cao nhất (base) trong vùng chiếm. Vị trí được coi là ổn định nếu
      ít nhất 80% các ô trong vùng có giá trị bằng base.
    """
    def __init__(self, W, L, H, grid_shape):
        """
        :param W: chiều rộng bin (số nguyên).
        :param L: chiều dài bin (số nguyên).
        :param H: chiều cao tối đa của bin (số nguyên).
        :param grid_shape: tuple (grid_w, grid_l), ví dụ (128,128).
        """
        self.W = W
        self.L = L
        self.H = H
        self.grid_w, self.grid_l = grid_shape
        # Tính độ phân giải của lưới theo trục x và y (có thể không phải số nguyên)
        self.resolution_x = W / self.grid_w
        self.resolution_y = L / self.grid_l
        # Sử dụng dtype int32 vì các giá trị là số nguyên
        self.height_map = np.zeros((self.grid_w, self.grid_l), dtype=np.int32)

    def reset(self):
        """Khởi tạo lại height_map về 0."""
        self.height_map = np.zeros((self.grid_w, self.grid_l), dtype=np.int32)

    def _get_grid_indices(self, x, y):
        """
        Chuyển tọa độ liên tục (x,y) thành chỉ số trên grid.
        Vì các tọa độ là số nguyên, ta làm tròn bằng cách chia và ép kiểu int.
        """
        i = int(x / self.resolution_x)
        j = int(y / self.resolution_y)
        return i, j

    def _get_grid_dimensions(self, item_w, item_l):
        """
        Tính số ô trên grid mà item chiếm, dựa trên kích thước item (item_w, item_l).
        """
        gw = int(math.ceil(item_w / self.resolution_x))
        gl = int(math.ceil(item_l / self.resolution_y))
        return gw, gl

    def is_valid_placement(self, x, y, item_w, item_l, item_h, stability_threshold=0.8):
        """
        Kiểm tra xem có thể đặt item tại (x,y) (tọa độ bottom-left) với kích thước (item_w, item_l, item_h)
        hay không.

        Các điều kiện:
          - Vùng chiếm không vượt quá biên của bin.
          - Base của vùng (giá trị max) được xác định trên vùng đó.
          - Ít nhất 80% các ô trong vùng có giá trị bằng base.
          - Chiều cao mới (base + item_h) không vượt quá H.
        
        :return: (is_valid: bool, base: int) – base là giá trị max của vùng nếu hợp lệ.
        """
        i, j = self._get_grid_indices(x, y)
        gw, gl = self._get_grid_dimensions(item_w, item_l)
        
        # Kiểm tra ranh giới
        if i < 0 or j < 0 or i + gw > self.grid_w or j + gl > self.grid_l:
            return False, None
        
        region = self.height_map[i:i+gw, j:j+gl]
        total_cells = region.size
        if total_cells == 0:
            return False, None

        # Base là giá trị max của vùng
        base = int(np.max(region))
        # Đếm số ô có giá trị bằng base (vì giá trị là số nguyên, không cần tolerance)
        count = np.sum(region == base)
        ratio = count / total_cells
        if ratio < stability_threshold:
            return False, None

        # Kiểm tra chiều cao sau khi đặt item
        if base + item_h > self.H:
            return False, None

        return True, base

    def update_height_map(self, x, y, item_w, item_l, item_h):
        """
        Sau khi đặt item hợp lệ, cập nhật height_map:
          - Các ô trong vùng được cập nhật thành (base + item_h).
        """
        i, j = self._get_grid_indices(x, y)
        gw, gl = self._get_grid_dimensions(item_w, item_l)
        valid, base = self.is_valid_placement(x, y, item_w, item_l, item_h)
        if not valid:
            raise ValueError("Vị trí không hợp lệ để cập nhật height_map.")
        new_value = base + item_h
        self.height_map[i:i+gw, j:j+gl] = new_value

    def compute_action_mask(self, item_w, item_l, item_h, stability_threshold=0.8):
        """
        Tính action mask 2D trên toàn bộ lưới cho một footprint nhất định (item_w, item_l, item_h).
        Mỗi ô (i,j) trả về 1 nếu đặt item tại (i * resolution_x, j * resolution_y) là hợp lệ, ngược lại 0.
        
        :return: mask, mảng 2D có kích thước (grid_w, grid_l)
        """
        mask = np.zeros_like(self.height_map, dtype=np.int32)
        for i in range(self.grid_w):
            for j in range(self.grid_l):
                x = i * self.resolution_x
                y = j * self.resolution_y
                valid, _ = self.is_valid_placement(x, y, item_w, item_l, item_h,
                                                    stability_threshold=stability_threshold)
                mask[i, j] = 1 if valid else 0
        return mask
