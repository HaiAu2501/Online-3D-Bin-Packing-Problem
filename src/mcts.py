# mcts.py

import math
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict
import torch
import torch.nn.functional as F
from networks import PolicyNetwork, ValueNetwork
from environment import BinPacking3DEnv
import copy

class MCTSNode:
    """
    Node trong cây MCTS.
    """
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

class MCTS:
    """
    Thuật toán MCTS.
    """
    def __init__(self, policy_network: PolicyNetwork, value_network: ValueNetwork, env: BinPacking3DEnv, 
                 c_puct: float = 1.4, num_simulations: int = 100):
        self.policy_network = policy_network
        self.value_network = value_network
        self.env = env
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.root: Optional[MCTSNode] = None

    def search(self, initial_state: Any) -> Dict:
        """
        Thực hiện tìm kiếm MCTS từ trạng thái ban đầu.
        """
        self.root = MCTSNode(state=initial_state)
        for simulation in range(1, self.num_simulations + 1):
            node = self.root
            env_clone = self._clone_env(self.env)
            reward = 0.0  # Khởi tạo reward mặc định
            done = False   # Khởi tạo done mặc định

            # Selection
            while node.children and not done:
                action, node = self._select_child(node)
                state, step_reward, done, truncated, _ = env_clone.step(action)
                reward += step_reward
                if done:
                    break

            # Expansion
            if not done and not env_clone.current_item_index >= len(env_clone.items):
                valid_actions = self._get_valid_actions(env_clone)
                if node.visit_count > 0 and valid_actions:
                    # Node chưa được mở rộng
                    policy, value = self._evaluate(env_clone)
                    if not policy:
                        # Không có hành động hợp lệ từ policy mạng, không mở rộng node
                        continue
                    for action in valid_actions:
                        node.children[action] = MCTSNode(state=None, parent=node)
                        node.children[action].prior = policy.get(action, 1e-6)  # Đảm bảo không bằng 0
                    # Sử dụng giá trị từ mạng value làm reward
                    reward += value
                    node.value_sum += value
                    node.visit_count += 1
                elif valid_actions:
                    # Node đã được mở rộng, nhưng chưa có children
                    for action in valid_actions:
                        node.children[action] = MCTSNode(state=None, parent=node)
                        node.children[action].prior = 1.0 / len(valid_actions)  # Uniform prior
            else:
                # Leaf node (episode kết thúc)
                if done:
                    # Reward đã được cộng trong quá trình selection
                    pass
                else:
                    # Nếu không còn vật phẩm để đặt, có thể thêm một reward tùy chỉnh hoặc giữ reward hiện tại
                    pass

            # Backpropagation
            self._backpropagate(node, reward)

        # Sau khi tìm kiếm, trích xuất chính sách từ root node
        policy = self._extract_policy(self.root)
        return policy

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """
        Chọn child với giá trị UCT cao nhất.
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in node.children.items():
            uct_score = (child.value +
                         self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count))
            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_child = child
        return best_action, best_child

    def _evaluate(self, env_clone: BinPacking3DEnv) -> Tuple[Dict[int, float], float]:
        """
        Đánh giá trạng thái bằng mạng policy và value.
        """
        observation = env_clone._get_observation()
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        policy_logits = self.policy_network(observation_tensor)
        policy_probs = F.softmax(policy_logits, dim=-1).detach().numpy()[0]
        
        # Lấy danh sách các hành động hợp lệ
        valid_actions = self._get_valid_actions(env_clone)
        policy = {action: policy_probs[action] for action in valid_actions}
        total_prob = sum(policy.values())
        if total_prob > 0:
            for action in policy:
                policy[action] /= total_prob
        else:
            # Nếu không có hành động hợp lệ, đặt uniform prior
            policy = {action: 1.0 / len(valid_actions) for action in valid_actions} if valid_actions else {}
        
        # Lấy giá trị từ mạng value
        value = self.value_network(observation_tensor).item()
        return policy, value

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Cập nhật giá trị và visit count cho các node trong đường dẫn từ node đến root.
        """
        while node is not None:
            node.visit_count += 1
            node.value_sum += reward
            node = node.parent

    def _extract_policy(self, node: MCTSNode) -> Dict:
        """
        Trích xuất chính sách từ root node sau khi tìm kiếm.
        """
        policy = {}
        for action, child in node.children.items():
            policy[action] = child.visit_count
        # Chuẩn hóa để thành phân phối xác suất
        total = sum(policy.values())
        for action in policy:
            policy[action] /= total
        return policy

    def _get_valid_actions(self, env_clone: BinPacking3DEnv) -> List[int]:
        """
        Lấy danh sách hành động hợp lệ từ trạng thái hiện tại.
        """
        valid_actions = []
        for action in range(env_clone.action_space.n):
            x, y, rotation = env_clone._decode_action(action)
            if env_clone._is_valid_position(x, y, rotation):
                valid_actions.append(action)
        return valid_actions

    def _clone_state(self, state: Any) -> Any:
        """
        Clone trạng thái để không ảnh hưởng đến trạng thái gốc.
        """
        return np.copy(state)

    def _clone_env(self, env: BinPacking3DEnv) -> BinPacking3DEnv:
        """
        Clone môi trường để sử dụng trong simulation.
        """
        cloned_env = BinPacking3DEnv(bin_size=(env.W, env.L, env.H), items=copy.deepcopy(env.items))
        cloned_env.current_item_index = env.current_item_index
        cloned_env.height_map = np.copy(env.height_map)
        cloned_env.current_height = np.copy(env.current_height)
        return cloned_env
