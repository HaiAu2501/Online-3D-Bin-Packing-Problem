# search/mcts.py

from __future__ import annotations

import math
import copy
from typing import Optional, Tuple, TYPE_CHECKING
import torch
import numpy as np

if TYPE_CHECKING:
    from env.env import BinPacking3DEnv

from node import Node
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork

class MCTS:
    def __init__(
        self,
        env: BinPacking3DEnv,
        policy_network: PolicyNetwork,
        value_network: ValueNetwork,
        num_simulations: int = 1000,
        c_param: float = math.sqrt(2)
    ):
        """
        Khởi tạo thuật toán MCTS.

        :param env: Môi trường Gym tương ứng.
        :param policy_network: Mạng Policy.
        :param value_network: Mạng Value.
        :param num_simulations: Số lần mô phỏng trong mỗi lượt tìm kiếm.
        :param c_param: Hệ số cân bằng giữa khai thác và khám phá trong UCB.
        """
        self.env = env
        self.policy_network = policy_network
        self.value_network = value_network
        self.num_simulations = num_simulations
        self.c_param = c_param
        self.root = Node(state=self._get_env_state())

    def _get_env_state(self):
        """
        Lấy trạng thái hiện tại của môi trường.

        :return: Một bản sao của môi trường hiện tại.
        """
        # Giả sử môi trường hỗ trợ copy.deepcopy
        return copy.deepcopy(self.env)

    def search(self):
        """
        Thực hiện tìm kiếm MCTS để chọn hành động tốt nhất từ trạng thái gốc.
        """
        for _ in range(self.num_simulations):
            node = self.root
            state = self._get_env_state()

            # SELECTION
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                action = node.action
                state.step(action)

            # EXPANSION
            if not node.is_fully_expanded():
                node = node.expand()
                action = node.action
                state.step(action)

            # SIMULATION
            reward, done = self._simulate(state)

            # BACKPROPAGATION
            self._backpropagate(node, reward)

        # CHOOSE THE BEST ACTION
        best_action = self._get_best_action()
        return best_action

    def _simulate(self, state: BinPacking3DEnv) -> Tuple[float, bool]:
        """
        Thực hiện mô phỏng từ trạng thái hiện tại đến khi kết thúc.

        :param state: Trạng thái môi trường tại thời điểm bắt đầu mô phỏng.
        :return: Tổng phần thưởng và trạng thái kết thúc.
        """
        done = False
        total_reward = 0.0

        while not done:
            observation = state._get_observation()
            action_mask = state.generate_action_mask()
            policy_logits = self.policy_network(torch.tensor(observation, dtype=torch.float32))
            policy_logits = policy_logits.view(-1)
            action_mask = torch.tensor(action_mask, dtype=torch.float32).view(-1)
            masked_logits = policy_logits * action_mask
            if torch.sum(action_mask) == 0:
                break  # Không còn hành động hợp lệ
            action_probs = torch.softmax(masked_logits, dim=0).detach().numpy()
            action = np.random.choice(len(action_probs), p=action_probs / np.sum(action_probs))
            # Chuyển đổi chỉ số thành hành động
            W, L, num_rotations, buffer_size = self.env.W, self.env.L, self.env.num_rotations, self.env.buffer_size
            x = action // (L * num_rotations * buffer_size)
            y = (action % (L * num_rotations * buffer_size)) // (num_rotations * buffer_size)
            rotation = (action % (num_rotations * buffer_size)) // buffer_size
            item_index = action % buffer_size
            selected_action = (x, y, rotation, item_index)

            observation, reward, done, truncated, info = state.step(selected_action)
            total_reward += reward

            if truncated:
                done = True

        # Đánh giá bằng Value Network
        final_observation = state._get_observation()
        value = self.value_network(torch.tensor(final_observation, dtype=torch.float32))
        value = torch.tanh(value).item()
        total_reward += value

        return total_reward, done

    def _backpropagate(self, node: Node, reward: float):
        """
        Cập nhật giá trị và số lượt thăm cho các nút trên đường đi từ nút hiện tại đến gốc.

        :param node: Nút bắt đầu từ nút được mở rộng.
        :param reward: Giá trị thu được từ mô phỏng.
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _get_best_action(self) -> Tuple[int, int, int, int]:
        """
        Chọn hành động tốt nhất từ nút gốc dựa trên số lượt thăm.

        :return: Hành động tốt nhất.
        """
        best_child = max(self.root.children.values(), key=lambda n: n.visits)
        return best_child.action
