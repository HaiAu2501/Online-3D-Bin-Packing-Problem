import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional


class CoarsePolicy(nn.Module):
    """
    CoarsePolicy network for predicting coarse logits.
    
    This network takes the encoded state z_t and predicts coarse logits:
    L_coarse = P_coarse(z_t) ∈ ℝ^(B×2×W_c×L_c)
    
    where:
    - B is the buffer size
    - 2 is the number of rotations
    - W_c is the coarse width grid size
    - L_c is the coarse length grid size
    """
    
    def __init__(
        self,
        W: int,  # Full bin width
        L: int,  # Full bin length
        buffer_size: int,
        embedding_dim: int,
        coarse_grid_size: Tuple[int, int] = (5, 5),  # (W_c, L_c)
        hidden_dim: int = 256,
    ):
        """Initialize the CoarsePolicy network.
        
        Args:
            W: Width of the bin
            L: Length of the bin
            buffer_size: Number of items in buffer
            embedding_dim: Dimension of the encoded state z_t
            coarse_grid_size: (W_c, L_c) - Size of coarse grid
            hidden_dim: Hidden dimension for the network
        """
        super().__init__()
        
        self.W = W
        self.L = L
        self.buffer_size = buffer_size
        self.W_c, self.L_c = coarse_grid_size
        
        # Calculate grid cell sizes
        self.delta_x = W / self.W_c
        self.delta_y = L / self.L_c
        
        # Store dimensions for attention mechanism
        self.embedding_dim = embedding_dim
        
        # Fully connected layers to process encoded state
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers
        # B×2×W_c×L_c = buffer_size × rotations × coarse_width × coarse_length
        total_outputs = buffer_size * 2 * self.W_c * self.L_c
        self.fc_out = nn.Linear(hidden_dim, total_outputs)
        
        # Record feature maps for use in skip connections
        self.feature_maps = {}
        
    def forward(
        self, 
        z_t: torch.Tensor, 
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for coarse policy.
        
        Args:
            z_t: [batch_size, embedding_dim] tensor of encoded states
            valid_mask: Optional [batch_size, buffer_size, 2, W_c, L_c] boolean tensor
                        where True indicates valid actions, False invalid ones
        
        Returns:
            coarse_logits: [batch_size, buffer_size, 2, W_c, L_c] tensor of logits
            coarse_probs: [batch_size, buffer_size, 2, W_c, L_c] tensor of probabilities
        """
        batch_size = z_t.size(0)
        
        # Process encoded state
        x = F.relu(self.fc1(z_t))
        self.feature_maps['fc1'] = x  # Store for skip connections
        
        x = F.relu(self.fc2(x))
        self.feature_maps['fc2'] = x  # Store for skip connections
        
        # Get logits
        logits = self.fc_out(x)
        
        # Reshape to [batch_size, buffer_size, 2, W_c, L_c]
        coarse_logits = logits.view(batch_size, self.buffer_size, 2, self.W_c, self.L_c)
        
        # Apply mask if provided (invalid actions)
        if valid_mask is not None:
            # Set logits for invalid actions to large negative value
            masked_logits = torch.where(
                valid_mask,
                coarse_logits,
                torch.tensor(-1e9, device=coarse_logits.device, dtype=coarse_logits.dtype)
            )
        else:
            masked_logits = coarse_logits
        
        # Calculate probabilities
        coarse_probs = F.softmax(masked_logits.view(batch_size, -1), dim=1)
        coarse_probs = coarse_probs.view(batch_size, self.buffer_size, 2, self.W_c, self.L_c)
        
        return coarse_logits, coarse_probs
    
    def get_candidate_region(
        self, 
        coarse_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get candidate region based on coarse probabilities.
        
        Args:
            coarse_probs: [batch_size, buffer_size, 2, W_c, L_c] tensor of probabilities
            
        Returns:
            batch_indices: [batch_size] tensor of batch indices
            selected_items: [batch_size] tensor of selected item indices
            selected_rotations: [batch_size] tensor of selected rotation indices
            coarse_x: [batch_size] tensor of coarse x indices
            coarse_y: [batch_size] tensor of coarse y indices
        """
        batch_size = coarse_probs.size(0)
        
        # Flatten the probability distribution
        flat_probs = coarse_probs.view(batch_size, -1)
        
        # Get the indices of the maximum probability
        flat_indices = torch.argmax(flat_probs, dim=1)
        
        # Convert flat indices to multi-dimensional indices
        # Calculate the number of elements in each dimension
        buffer_size = coarse_probs.size(1)
        rotations = coarse_probs.size(2)
        W_c = coarse_probs.size(3)
        L_c = coarse_probs.size(4)
        
        # Calculate indices for each dimension
        rot_x_y_size = rotations * W_c * L_c
        x_y_size = W_c * L_c
        
        selected_items = (flat_indices // rot_x_y_size).long()
        remainder = flat_indices % rot_x_y_size
        
        selected_rotations = (remainder // x_y_size).long()
        remainder = remainder % x_y_size
        
        coarse_x = (remainder // L_c).long()
        coarse_y = (remainder % L_c).long()
        
        # Create batch indices
        batch_indices = torch.arange(batch_size, device=coarse_probs.device)
        
        return batch_indices, selected_items, selected_rotations, coarse_x, coarse_y
    
    def get_region_boundaries(
        self, 
        coarse_x: torch.Tensor, 
        coarse_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert coarse grid indices to region boundaries.
        
        Args:
            coarse_x: [batch_size] tensor of coarse x indices
            coarse_y: [batch_size] tensor of coarse y indices
            
        Returns:
            x_min: [batch_size] tensor of minimum x values
            y_min: [batch_size] tensor of minimum y values
            x_max: [batch_size] tensor of maximum x values
            y_max: [batch_size] tensor of maximum y values
        """
        # Convert coarse indices to region boundaries
        # Make sure to round to integers to avoid float values
        x_min = (coarse_x.float() * self.delta_x).floor().long()
        y_min = (coarse_y.float() * self.delta_y).floor().long()
        x_max = ((coarse_x.float() + 1) * self.delta_x).ceil().long()
        y_max = ((coarse_y.float() + 1) * self.delta_y).ceil().long()
        
        # Clamp to bin dimensions
        x_max = torch.clamp(x_max, max=self.W)
        y_max = torch.clamp(y_max, max=self.L)
        
        return x_min, y_min, x_max, y_max
    
    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        """Get intermediate feature maps for skip connections.
        
        Returns:
            Dictionary of feature maps
        """
        return self.feature_maps