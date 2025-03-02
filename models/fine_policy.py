import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Union

class FinePolicy(nn.Module):
    """
    FinePolicy network for predicting fine-grained placements within selected regions.
    
    This network takes:
    1. Encoded state z_t
    2. Feature maps from CoarsePolicy
    3. Selected candidate region
    
    And predicts fine logits:
    L_fine = P_fine(z_fine, R_c*) ∈ ℝ^(1×1×W×L)
    """
    
    def __init__(
        self,
        W: int,  # Full bin width
        L: int,  # Full bin length
        H: int,  # Full bin height
        embedding_dim: int,
        hidden_dim: int = 256,
        use_skip_connections: bool = True
    ):
        """Initialize the FinePolicy network.
        
        Args:
            W: Width of the bin
            L: Length of the bin
            H: Height of the bin
            embedding_dim: Dimension of the encoded state z_t
            hidden_dim: Hidden dimension for the network
            use_skip_connections: Whether to use skip connections from encoder/coarse policy
        """
        super().__init__()
        
        self.W = W
        self.L = L
        self.H = H
        self.use_skip_connections = use_skip_connections
        
        # Feature combination network
        if use_skip_connections:
            # We expect feature maps from both encoder and coarse policy
            self.feature_combination = nn.Sequential(
                nn.Linear(embedding_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            # Only use z_t
            self.feature_combination = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Region-aware processing
        # Input: combined features + region indices (x_min, y_min, x_max, y_max) + item + rotation
        region_input_dim = hidden_dim + 4 + 3 + 1  # 4 for region bounds, 3 for item dims, 1 for rotation
        self.region_processor = nn.Sequential(
            nn.Linear(region_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fine-grained prediction network
        # This will output a feature map that can be mapped to W×L
        self.fine_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, W * L)
        )
        
        # Constants for gradient-based refinement
        self.alpha_t = nn.Parameter(torch.tensor(0.7))  # Volume utilization weight
        self.beta_t = nn.Parameter(torch.tensor(0.3))   # Support constraint weight
        self.gamma = 0.2  # Penalty for uneven support
        
    def forward(
        self,
        z_t: torch.Tensor,
        region_data: Dict[str, torch.Tensor],
        coarse_features: Optional[Dict[str, torch.Tensor]] = None,
        encoder_features: Optional[Dict[str, torch.Tensor]] = None,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for fine policy.
        
        Args:
            z_t: [batch_size, embedding_dim] tensor of encoded states
            region_data: Dictionary containing:
                - 'item_indices': [batch_size] tensor of item indices
                - 'rotation': [batch_size] tensor of rotation indices
                - 'x_min', 'y_min', 'x_max', 'y_max': [batch_size] tensors of region boundaries
                - 'items': [batch_size, 3] tensor of item dimensions (w, l, h)
            coarse_features: Optional dictionary of feature maps from coarse policy
            encoder_features: Optional dictionary of feature maps from encoder
            valid_mask: Optional [batch_size, W, L] boolean tensor for valid placement positions
            
        Returns:
            fine_logits: [batch_size, W, L] tensor of logits
            fine_probs: [batch_size, W, L] tensor of probabilities
        """
        batch_size = z_t.size(0)
        
        # 1. Combine features (z_t and skip connections if available)
        if self.use_skip_connections and coarse_features is not None:
            # Extract a relevant feature map from coarse_features
            coarse_feature = coarse_features['fc2']  # Assuming this is the most relevant
            combined_features = torch.cat([z_t, coarse_feature], dim=1)
            z_fine = self.feature_combination(combined_features)
        else:
            # Just use z_t
            z_fine = self.feature_combination(z_t)
        
        # 2. Extract region information
        x_min = region_data['x_min']
        y_min = region_data['y_min']
        x_max = region_data['x_max']
        y_max = region_data['y_max']
        items = region_data['items']
        rotation = region_data['rotation'].float().unsqueeze(1)  # Add dimension to match others
        
        # Combine z_fine with region information
        region_info = torch.cat([
            z_fine,
            x_min.float().unsqueeze(1),
            y_min.float().unsqueeze(1),
            x_max.float().unsqueeze(1),
            y_max.float().unsqueeze(1),
            items.float(),  # [batch_size, 3]
            rotation
        ], dim=1)
        
        # Process region-aware features
        region_features = self.region_processor(region_info)
        
        # 3. Predict fine-grained logits
        fine_logits_flat = self.fine_predictor(region_features)
        fine_logits = fine_logits_flat.view(batch_size, self.W, self.L)
        
        # 4. Apply mask if provided
        if valid_mask is not None:
            # Set logits for invalid positions to large negative value
            masked_logits = torch.where(
                valid_mask,
                fine_logits,
                torch.tensor(-1e9, device=fine_logits.device, dtype=fine_logits.dtype)
            )
        else:
            masked_logits = fine_logits
        
        # 5. Calculate probabilities
        fine_probs = F.softmax(masked_logits.view(batch_size, -1), dim=1)
        fine_probs = fine_probs.view(batch_size, self.W, self.L)
        
        return fine_logits, fine_probs
    
    def gradient_refinement(
        self,
        height_map: torch.Tensor,
        initial_logits: torch.Tensor,
        initial_probs: torch.Tensor,
        region_data: Dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
        num_iterations: int = 5,
        learning_rate: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform gradient-based refinement on fine logits.
        
        Args:
            height_map: [batch_size, W, L] tensor of height map
            initial_logits: [batch_size, W, L] tensor of initial logits
            initial_probs: [batch_size, W, L] tensor of initial probabilities
            region_data: Dictionary containing region and item information
            valid_mask: [batch_size, W, L] boolean tensor for valid placement positions
            num_iterations: Number of gradient updates
            learning_rate: Learning rate for gradient updates
            
        Returns:
            refined_logits: [batch_size, W, L] tensor of refined logits
            refined_probs: [batch_size, W, L] tensor of refined probabilities
        """
        batch_size = initial_logits.size(0)
        
        # Create tensor to optimize (clone to keep gradients)
        refined_logits = initial_logits.clone().detach().requires_grad_(True)
        
        # Extract item information
        items = region_data['items']  # [batch_size, 3]
        rotation = region_data['rotation']  # [batch_size]
        
        # Calculate item dimensions based on rotation
        w_items = torch.where(
            rotation == 0,
            items[:, 0],  # w if no rotation
            items[:, 1]   # l if rotated
        )
        l_items = torch.where(
            rotation == 0,
            items[:, 1],  # l if no rotation
            items[:, 0]   # w if rotated
        )
        h_items = items[:, 2]  # height is always the same
        
        # Create optimizer
        optimizer = torch.optim.Adam([refined_logits], lr=learning_rate)
        
        # Gradient refinement iterations
        for _ in range(num_iterations):
            optimizer.zero_grad()
            
            # Calculate probabilities
            masked_logits = torch.where(
                valid_mask,
                refined_logits,
                torch.tensor(-1e9, device=refined_logits.device, dtype=refined_logits.dtype)
            )
            probs = F.softmax(masked_logits.view(batch_size, -1), dim=1).view(batch_size, self.W, self.L)
            
            # Initialize loss
            loss = torch.zeros(1, device=refined_logits.device)
            
            # For each item in the batch
            for b in range(batch_size):
                # Get item dimensions
                w_item = w_items[b]
                l_item = l_items[b]
                h_item = h_items[b]
                
                # For each possible position in the grid
                for x in range(self.W - w_item.int() + 1):
                    for y in range(self.L - l_item.int() + 1):
                        # Calculate the region covered by the item
                        x_end = x + w_item.int()
                        y_end = y + l_item.int()
                        
                        # Skip if this position is invalid
                        if not valid_mask[b, x, y]:
                            continue
                        
                        # Calculate H_support and H_min for the region
                        region_heights = height_map[b, x:x_end, y:y_end]
                        H_support = torch.max(region_heights)
                        H_min = torch.min(region_heights)
                        
                        # Calculate volume utilization
                        volume = w_item * l_item * h_item
                        penalty = self.gamma * w_item * l_item * (H_support - H_min)
                        utilization = volume - penalty
                        
                        # Calculate support ratio
                        support_count = torch.sum(region_heights == H_support)
                        support_ratio = support_count / (w_item * l_item)
                        support_penalty = torch.max(
                            torch.tensor(0.0, device=refined_logits.device),
                            torch.tensor(0.8, device=refined_logits.device) - support_ratio
                        ) ** 2
                        
                        # Calculate objective function
                        obj_value = self.alpha_t * utilization - self.beta_t * support_penalty
                        
                        # Add to loss (negative since we want to maximize)
                        loss = loss - probs[b, x, y] * obj_value
            
            # Backpropagate and update
            loss.backward()
            optimizer.step()
        
        # Calculate final probabilities
        masked_logits = torch.where(
            valid_mask,
            refined_logits,
            torch.tensor(-1e9, device=refined_logits.device, dtype=refined_logits.dtype)
        )
        refined_probs = F.softmax(masked_logits.view(batch_size, -1), dim=1).view(batch_size, self.W, self.L)
        
        return refined_logits.detach(), refined_probs