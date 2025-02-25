# env/buffer_manager.py
import numpy as np

class BufferManager:
    """
    Quản lý buffer chứa danh sách mặt hàng ban đầu.

    - Buffer có kích thước cố định (buffer_size).
    - Danh sách items (mặt hàng) được truyền vào lúc khởi tạo.
    - Mỗi lần sử dụng, lấy item theo thứ tự. Nếu hết mặt hàng, sử dụng dummy item [0,0,0].
    """
    def __init__(self, buffer_size, items_list):
        """
        :param buffer_size: Số lượng item tối đa trong buffer.
        :param items_list: Danh sách các item, mỗi item là [w, l, h] (số nguyên).
        """
        self.buffer_size = buffer_size
        self.items_list = items_list  # danh sách toàn bộ mặt hàng
        self.pointer = 0  # vị trí hiện tại trong danh sách
        self.buffer = []
        self._initialize_buffer()

    def _initialize_buffer(self):
        """Khởi tạo buffer với buffer_size item từ danh sách mặt hàng."""
        self.buffer = []
        for _ in range(self.buffer_size):
            if self.pointer < len(self.items_list):
                self.buffer.append(self.items_list[self.pointer])
                self.pointer += 1
            else:
                self.buffer.append([0, 0, 0])  # dummy item

    def reset(self):
        """Reset buffer và con trỏ về trạng thái ban đầu."""
        self.pointer = 0
        self._initialize_buffer()

    def get_buffer(self):
        """
        Trả về buffer hiện tại dưới dạng numpy array với shape (buffer_size, 3)
        và các giá trị là số nguyên.
        """
        return np.array(self.buffer, dtype=np.int32)

    def pop_item(self, index):
        """
        Loại bỏ item tại vị trí index trong buffer và bổ sung item mới từ danh sách.
        :param index: chỉ số của item được chọn.
        :return: item đã bị loại bỏ.
        """
        if index < 0 or index >= self.buffer_size:
            raise IndexError("Chỉ số item trong buffer không hợp lệ.")
        removed_item = self.buffer.pop(index)
        if self.pointer < len(self.items_list):
            new_item = self.items_list[self.pointer]
            self.pointer += 1
        else:
            new_item = [0, 0, 0]  # dummy item nếu hết mặt hàng
        self.buffer.append(new_item)
        return removed_item
