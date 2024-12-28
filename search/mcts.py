import numpy as np
from typing import Tuple, Dict
from .node import Node
from env.env import BinPacking3DEnv
from models.model import BinPackingModel
import torch
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from .prb import PrioritizedReplayBuffer

class MCTS:
    def __init__(
        self,
        model: BinPackingModel,
        env: BinPacking3DEnv,
        replay_buffer: PrioritizedReplayBuffer = PrioritizedReplayBuffer(10000),
        c_puct: float = 1.0,
        n_simulations: int = 100,
        num_parallel_simulations: int = 5,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.model = model
        self.env = env
        self.replay_buffer = replay_buffer
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.num_parallel_simulations = num_parallel_simulations
        self.device = device

    def run(self, root_node: Node) -> Tuple[Tuple[int, int, int, int], Node]:
        """
        Chạy thuật toán MCTS từ node gốc.

        :param root_node: Node gốc.
        :return: Hành động tốt nhất và node tương ứng.
        """
        for _ in range(self.n_simulations):
            node = root_node
            # 1. Selection
            while not node.is_leaf():
                _, node = node.select_best_child(self.c_puct)

            # 2. Expansion
            obs, _ = node.env._get_observation()
            buffer_tensor = torch.tensor(obs['buffer'], dtype=torch.float32).unsqueeze(0).to(self.device)
            ems_list_tensor = torch.tensor(obs['ems_list'], dtype=torch.float32).unsqueeze(0).to(self.device)
            action_mask_tensor = torch.tensor(obs['action_mask'], dtype=torch.float32).unsqueeze(0).to(self.device)

            if not node.env.action_mask.any(): # Kiểm tra action_mask có toàn 0 không
                # Nếu không có hành động hợp lệ, backpropagate giá trị từ value_net
                value = self.model.get_value(ems_list_tensor, buffer_tensor).squeeze(0).cpu().detach().item()
                node.backpropagate(value)
                continue

            _, policy = self.model(ems_list_tensor, buffer_tensor, action_mask_tensor)
            policy = policy.squeeze(0).cpu().detach().numpy()
            node.expand(policy)

            # 3. Simulation (Leaf Parallelization)
            selected_children = random.sample(list(node.children.values()), min(self.num_parallel_simulations, len(node.children)))

            with ThreadPoolExecutor(max_workers=self.num_parallel_simulations) as executor:
                futures = [executor.submit(self.simulate, child.env) for child in selected_children]
                values = [future.result() for future in as_completed(futures)]

            if values:
                z = np.mean(values)
                # Tính giá trị bằng cách tổ hợp giá trị với mạng Value
                value_net = self.model.get_value(ems_list_tensor, buffer_tensor).squeeze(0).cpu().detach().item()
                alpha = min(1.0, self.replay_buffer.frame / 100000)
                value = alpha * value_net + (1 - alpha) * z

                # 4. Backpropagation
                node.backpropagate(value)

        # Chọn hành động tốt nhất từ node gốc
        best_action, best_child = root_node.select_best_child(c_puct=0) # Chọn node con có w/n cao nhất

        # Lưu trữ dữ liệu vào Replay Buffer
        obs, _ = root_node.env._get_observation()
        next_obs, _ = best_child.env._get_observation()

        # Tính value mục tiêu (target value)
        target_value = 0
        if not best_child.env.generate_action_mask().any() or all(item == (0, 0, 0) for item in best_child.env.buffer):
            target_value = best_child.env.step(best_action)[1] # Lấy reward
        else:
            next_buffer_tensor = torch.tensor(next_obs['buffer'], dtype=torch.float32).unsqueeze(0).to(self.device)
            next_ems_list_tensor = torch.tensor(next_obs['ems_list'], dtype=torch.float32).unsqueeze(0).to(self.device)
            next_value = self.model.get_value(next_ems_list_tensor, next_buffer_tensor).squeeze(0).cpu().detach().item()
            target_value = best_child.env.step(best_action)[1] + 0.95 * next_value # Lấy reward, gamma = 0.95

        _, policy = self.model(ems_list_tensor, buffer_tensor, action_mask_tensor)
        policy = policy.squeeze(0).cpu().detach().numpy()

        value = self.model.get_value(ems_list_tensor, buffer_tensor).squeeze(0).cpu().detach().item()

        self.replay_buffer.add(
            state=obs,
            action_mask=obs['action_mask'],
            action=best_action,
            reward=best_child.env.step(best_action)[1], # Lấy reward từ best_child
            next_state=next_obs,
            done=best_child.env.step(best_action)[2], # Lấy done từ best_child
            value=value,
            policy=policy
        )

        return best_action, best_child

    def simulate(self, env: BinPacking3DEnv) -> float:
        """
        Mô phỏng một episode từ trạng thái hiện tại.

        :param env: Môi trường hiện tại.
        :return: Tổng phần thưởng của episode.
        """
        done = False
        total_reward = 0
        while not done:
            obs, _ = env._get_observation()
            action_mask = obs['action_mask']

            # Nếu không có hành động hợp lệ, kết thúc mô phỏng
            if not action_mask.any():
                break

            valid_actions = np.argwhere(action_mask == 1)
            action = tuple(valid_actions[np.random.choice(len(valid_actions))]) # Chọn ngẫu nhiên một hành động hợp lệ

            _, reward, done, _, _ = env.step(action)
            total_reward += reward
        return total_reward