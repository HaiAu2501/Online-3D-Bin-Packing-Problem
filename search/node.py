# search/node.py

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import copy
import numpy as np

if TYPE_CHECKING:
    from env.env import BinPacking3DEnv

class Node:
    def __init__(
        self,
        state: BinPacking3DEnv,
        parent: Optional[Node] = None,
        action: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Khởi tạo một nút trong cây MCTS.

        :param state: Trạng thái hiện tại của môi trường (sao chép của BinPacking3DEnv).
        :param parent: Nút cha của nút hiện tại.
        :param action: Hành động dẫn đến trạng thái này từ nút cha.
        """
        self.state = state
        self.parent = parent
        self.children: Dict[Tuple[int, int, int, int], 'Node'] = {}
        self.action = action
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_actions: List[Tuple[int, int, int, int]] = self.get_valid_actions()

    def get_valid_actions(self) -> List[Tuple[int, int, int, int]]:
        """
        Lấy danh sách các hành động hợp lệ từ trạng thái hiện tại.

        :return: Danh sách các hành động hợp lệ.
        """
        action_mask = self.state.generate_action_mask()
        W, L, num_rotations, buffer_size = self.state.W, self.state.L, self.state.num_rotations, self.state.buffer_size
        valid_actions = []
        for x in range(W):
            for y in range(L):
                for rot in range(num_rotations):
                    for buf_idx in range(buffer_size):
                        if action_mask[x, y, rot, buf_idx]:
                            valid_actions.append((x, y, rot, buf_idx))
        return valid_actions

    def is_fully_expanded(self) -> bool:
        """
        Kiểm tra xem nút đã được mở rộng đầy đủ chưa (tất cả hành động khả dụng đã được mở rộng).

        :return: True nếu đầy đủ, ngược lại False.
        """
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = math.sqrt(2)) -> 'Node':
        """
        Chọn nút con tốt nhất dựa trên công thức UCB1.

        :param c_param: Hệ số cân bằng giữa khai thác và khám phá.
        :return: Nút con tốt nhất.
        """
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children.values()
        ]
        return list(self.children.values())[choices_weights.index(max(choices_weights))]

    def expand(self) -> Node:
        """
        Mở rộng nút bằng cách chọn một hành động chưa được thử và tạo nút con tương ứng.

        :return: Nút con mới được mở rộng.
        """
        action = self.untried_actions.pop()
        # Tạo bản sao của môi trường để thực hiện hành động
        next_state = copy.deepcopy(self.state)
        _, _, done, _, _ = next_state.step(action)
        child_node = Node(state=next_state, parent=self, action=action)
        self.children[action] = child_node
        return child_node

    def update(self, reward: float):
        """
        Cập nhật giá trị và số lượt thăm của nút.

        :param reward: Giá trị thu được từ việc mô phỏng.
        """
        self.visits += 1
        self.value += reward

    def fully_expanded_children(self) -> List['Node']:
        """
        Trả về danh sách các nút con đã được mở rộng đầy đủ.

        :return: Danh sách các nút con.
        """
        return [child for child in self.children.values() if child.is_fully_expanded()]
