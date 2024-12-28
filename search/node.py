import numpy as np
from typing import Tuple, Optional, Dict
from env.env import BinPacking3DEnv

class Node:
    def __init__(
        self,
        env: BinPacking3DEnv,
        parent: Optional['Node'] = None,
        action: Optional[Tuple[int, int, int, int]] = None,
        p: float = 0.0,
    ):
        self.env: BinPacking3DEnv = env.clone()
        self.parent: Optional[Node] = parent
        self.action: Optional[Tuple[int, int, int, int]] = action # Hành động của node cha dẫn đến node này

        self.children: Dict[Tuple[int, int, int, int], Node] = {} # Lưu trữ các node con
        self.n: int = 0 # Số lượt node này được thăm
        self.w: float = 0.0 # Tổng giá trị Q (phần thưởng) khi đi qua node này
        self.p: float = p # Xác suất chọn node này (được tính từ policy network của node cha)

    def is_leaf(self) -> bool:
        """
        Kiểm tra xem node này có phải là node lá hay không.
        """
        return len(self.children) == 0

    def get_ucb_score(self, c_puct: float = 1.0) -> float:
        """
        Tính toán điểm số UCB của node.

        :param c_puct: Hằng số khai thác.
        """
        if self.n == 0:
            return float('inf')  # Ưu tiên các node chưa được thăm dò
        return self.w / self.n + c_puct * self.p * (np.sqrt(self.parent.n) / (1 + self.n))

    def select_best_child(self, c_puct: float = 1.0) -> Tuple[Tuple[int, int, int, int], 'Node']:
        """
        Chọn node con có điểm số UCB cao nhất.

        :param c_puct: Hằng số khai thác.
        """
        best_action = None
        best_child = None
        best_score = -float('inf')

        for action, child in self.children.items():
            score = child.get_ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, policy: np.ndarray):
        """
        Mở rộng node bằng cách tạo các node con dựa trên các hành động hợp lệ.

        :param policy: Xác suất cho các hành động, đầu ra của policy network.
        """
        obs, _ = self.env._get_observation()
        action_mask = obs['action_mask']
        valid_actions = np.argwhere(action_mask == 1)

        for action in valid_actions:
            x, y, rot, item_index = action
            action_tuple = (x, y, rot, item_index)
            
            next_env = self.env.clone()
            _, _, _, _, _ = next_env.step(action_tuple)
            self.children[action_tuple] = Node(env=next_env, parent=self, action=action_tuple, p=policy[tuple(action)])

    def backpropagate(self, value: float):
        """
        Cập nhật giá trị của node và các node cha.

        :param value: Giá trị trả về từ quá trình mô phỏng.
        """
        self.n += 1
        self.w += value
        if self.parent:
            self.parent.backpropagate(value) # Truyền ngược giá trị lên node cha