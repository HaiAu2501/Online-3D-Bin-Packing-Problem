# search/node.py

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from env.env import BinPacking3DEnv

class Node:
    def __init__(
        self,
        state: BinPacking3DEnv,
        parent: Optional[Node] = None,
        action: Optional[Tuple[int, int, int, int]] = None,
        prior_prob: float = 0.0
    ):
        """
        Initialize a node in the MCTS tree.

        :param state: The current state of the environment.
        :param parent: Parent node that leads to this state.
        :param action: Action taken to reach this state.
        :param prior_prob: Prior probability of selecting this node by using the policy network.
        """
        self.state: BinPacking3DEnv = state.clone()  # Use the clone method instead of deepcopy
        self.parent: Node = parent
        self.action = action
        self.prior_prob: float = prior_prob

        self.children: Dict[Tuple[int, int, int, int], Node] = {}  # Mapping from actions to child nodes
        self.visits: int = 0  # Number of times node has been visited
        self.value: float = 0.0  # Accumulated value of the node
        self.total_action_value: float = 0.0  # Accumulated value of the node's actions to estimate Q-value
        self.is_terminal: bool = False  # Whether the node is a terminal state

        self.untried_actions: List[Tuple[int, int, int, int]] = self.get_valid_actions()

    def get_valid_actions(self) -> List[Tuple[int, int, int, int]]:
        """
        Retrieve a list of valid actions from the current state.

        :return: A list of valid actions represented as tuples.
        """
        if self.is_terminal:
            return []

        return self.state.valid_actions

    def is_fully_expanded(self) -> bool:
        """
        Check if all possible actions have been tried from this node.

        :return: True if fully expanded, False otherwise.
        """
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = math.sqrt(2)) -> Optional[Node]:
        """
        Select the best child node based on the UCB1 formula.

        :param c_param: Exploration parameter.
        :return: The child node with the highest UCB1 value.
        """
        best_score = - float('inf')
        best_child = None

        for child in self.children.values():
            if child.visits == 0:
                score = float('inf')  # Ưu tiên các node chưa thăm
            else:
                Q = child.total_action_value / child.visits  # Giá trị trung bình
                U = c_param * child.prior_prob * math.sqrt(self.visits) / (1 + child.visits)  # Thành phần khám phá
                score = Q + U

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, action: Tuple[int, int, int, int], prior_prob: float) -> Optional[Node]:
        """
        Expand the node by adding a new child for an untried action.

        :param action: The action to create a new child node.
        :param prior_prob: The prior probability of selecting this action.

        :return: The newly created child node or None if no actions to try.
        """
        new_state = self.state.clone()
        new_state.step(action)  # Placeholder: Hàm này áp dụng hành động vào trạng thái
        child_node = Node(state=new_state, parent=self, action=action, prior_prob=prior_prob)
        self.children[action] = child_node
        self.untried_actions.remove(action)
        return child_node

    def update(self, value: float):
        """
        Update the node's statistics based on the received reward.

        :param value: The value obtained from the simulation.
        """
        self.visits += 1
        self.total_action_value += value # Accumulate the reward

