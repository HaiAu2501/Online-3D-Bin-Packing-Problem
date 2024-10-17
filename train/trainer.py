# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

from typing import Tuple, List, Optional
from env.env import BinPacking3DEnv
from search.mcts import MCTS
from search.replay_buffer import PrioritizedReplayBuffer
from search.node import Node
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork
import os

class Trainer:
    def __init__(
        self,
        env: BinPacking3DEnv,
        policy_network: PolicyNetwork,
        value_network: ValueNetwork,
        replay_buffer: PrioritizedReplayBuffer,
        num_simulations: int = 1000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr_policy: float = 1e-4,
        lr_value: float = 1e-3,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        save_path: str = "./models/"
    ):
        """
        Initialize the Trainer.

        :param env: The bin packing environment.
        :param policy_network: The Policy Network.
        :param value_network: The Value Network.
        :param replay_buffer: The Prioritized Replay Buffer.
        :param num_simulations: Number of MCTS simulations per episode.
        :param batch_size: Batch size for training.
        :param gamma: Discount factor.
        :param lr_policy: Learning rate for Policy Network.
        :param lr_value: Learning rate for Value Network.
        :param beta_start: Initial value of beta for importance sampling.
        :param beta_frames: Number of frames over which beta will be annealed from beta_start to 1.
        :param save_path: Directory to save the trained models.
        """
        self.env = env
        self.policy_network = policy_network
        self.value_network = value_network
        self.replay_buffer = replay_buffer
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.save_path = save_path

        # Define optimizers
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=lr_value)

        # Create directory to save models if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

    def train(self, num_episodes: int = 10000, update_every: int = 10):
        """
        Train the Policy and Value networks using MCTS and PRB.

        :param num_episodes: Number of training episodes.
        :param update_every: Number of episodes between each network update.
        """
        for episode in range(1, num_episodes + 1):
            self.env.reset()
            mcts = MCTS(
                env=self.env,
                policy_network=self.policy_network,
                value_network=self.value_network,
                num_simulations=self.num_simulations
            )

            # Perform MCTS search to generate data
            mcts.search()

            # Collect experiences from the MCTS tree
            experiences = self._collect_experiences(mcts.root)

            # Add experiences to the replay buffer
            for exp in experiences:
                state, action, reward, next_state, done = exp
                priority = reward  # You can adjust priority based on reward or another metric
                self.replay_buffer.add(exp, priority)

            # Update networks every 'update_every' episodes
            if episode % update_every == 0:
                self._update_networks()
                print(f"Episode {episode}/{num_episodes} - Networks updated.")

            # Save models periodically
            if episode % 1000 == 0:
                self._save_models(episode)
                print(f"Episode {episode}/{num_episodes} - Models saved.")

    def _collect_experiences(self, node: Node, parent_state: Optional[BinPacking3DEnv] = None) -> List[Tuple]:
        """
        Traverse the MCTS tree and collect experiences.

        :param node: The current node in the MCTS tree.
        :param parent_state: The state before taking the action.
        :return: A list of experiences.
        """
        experiences = []
        if node.parent is not None and parent_state is not None:
            # Current state is the state after taking node.action from parent_state
            state = parent_state.clone()
            _, reward, done, _, _ = state.step(node.action)
            next_state = state.clone()
            experiences.append((parent_state._get_observation(), node.action, reward, next_state._get_observation(), done))

        for child in node.children.values():
            experiences.extend(self._collect_experiences(child, node.state))

        return experiences

    def _update_networks(self):
        """
        Sample a batch from the replay buffer and perform a training step for both networks.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        # Anneal beta from beta_start to 1 over time
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        # Sample a batch of experiences
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size, beta)

        # Convert batch of experiences to tensors
        batch = list(zip(*experiences))
        states = batch[0]
        actions = batch[1]
        rewards = batch[2]
        next_states = batch[3]
        dones = batch[4]

        # Process states and next_states for Policy and Value networks
        # Assuming the networks take concatenated buffer and ems as input
        state_inputs = [self._prepare_input(state) for state in states]
        next_state_inputs = [self._prepare_input(state) for state in next_states]

        state_inputs = torch.stack(state_inputs)
        next_state_inputs = torch.stack(next_state_inputs)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        # -------------------- Train Policy Network --------------------
        # Predict action probabilities
        policy_logits = self.policy_network(state_inputs)  # Shape: (batch_size, num_actions)
        # Flatten actions for loss computation
        log_probs = torch.log_softmax(policy_logits, dim=1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Compute policy loss (negative log likelihood)
        policy_loss = -selected_log_probs * weights

        # Optimize Policy Network
        self.optimizer_policy.zero_grad()
        policy_loss.mean().backward()
        self.optimizer_policy.step()

        # -------------------- Train Value Network --------------------
        # Predict state values
        state_values = self.value_network(state_inputs).squeeze(1)  # Shape: (batch_size,)
        # Compute target values
        with torch.no_grad():
            next_values = self.value_network(next_state_inputs).squeeze(1)
            targets = rewards + self.gamma * next_values * (1 - dones)
        # Compute value loss (mean squared error)
        value_loss = (state_values - targets) ** 2 * weights
        value_loss = value_loss.mean()

        # Optimize Value Network
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # -------------------- Update Priorities in Replay Buffer --------------------
        # Here we can use the TD error as priority
        td_errors = (state_values - targets).abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors + 1e-5)  # Add a small epsilon to avoid zero priority

    def _prepare_input(self, observation: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Prepare the input tensor for the networks from the observation.

        :param observation: The observation dictionary containing 'buffer' and 'ems'.
        :return: A tensor ready to be input into the networks.
        """
        buffer = torch.tensor(observation['buffer'], dtype=torch.float32)
        ems = torch.tensor(observation['ems'], dtype=torch.float32)
        # Flatten and concatenate buffer and ems
        input_tensor = torch.cat((buffer.flatten(), ems.flatten()))
        return input_tensor

    def _save_models(self, episode: int):
        """
        Save the Policy and Value networks.

        :param episode: The current episode number (used in the filename).
        """
        policy_path = os.path.join(self.save_path, f"policy_net_episode_{episode}.pth")
        value_path = os.path.join(self.save_path, f"value_net_episode_{episode}.pth")
        torch.save(self.policy_network.state_dict(), policy_path)
        torch.save(self.value_network.state_dict(), value_path)

    def _load_models(self, episode: int):
        """
        Load the Policy and Value networks from saved files.

        :param episode: The episode number of the saved models.
        """
        policy_path = os.path.join(self.save_path, f"policy_net_episode_{episode}.pth")
        value_path = os.path.join(self.save_path, f"value_net_episode_{episode}.pth")
        self.policy_network.load_state_dict(torch.load(policy_path))
        self.value_network.load_state_dict(torch.load(value_path))
