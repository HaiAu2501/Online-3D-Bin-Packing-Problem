import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Union


def compute_support_ratio(
    height_map: torch.Tensor,
    item_dims: torch.Tensor,
    rotation: torch.Tensor,
    position: torch.Tensor
) -> torch.Tensor:
    """Compute the support ratio for items at given positions.
    
    Args:
        height_map: [batch_size, W, L] tensor of height map
        item_dims: [batch_size, 3] tensor of item dimensions (w, l, h)
        rotation: [batch_size] tensor of rotation indices (0 or 1)
        position: [batch_size, 2] tensor of (x, y) positions
        
    Returns:
        [batch_size] tensor of support ratios
    """
    batch_size = height_map.size(0)
    device = height_map.device
    
    # Extract item dimensions
    w = item_dims[:, 0]
    l = item_dims[:, 1]
    
    # Apply rotation
    w_rotated = torch.where(rotation == 0, w, l)
    l_rotated = torch.where(rotation == 0, l, w)
    
    # Extract positions
    x = position[:, 0]
    y = position[:, 1]
    
    # Initialize support ratios
    support_ratios = torch.zeros(batch_size, device=device)
    
    # For each item in the batch
    for b in range(batch_size):
        # Get region under the item - convert all indices to integers
        x_start = int(x[b].item())
        y_start = int(y[b].item())
        w_r = int(w_rotated[b].item())
        l_r = int(l_rotated[b].item())
        
        # Check boundaries
        if x_start < 0 or y_start < 0 or x_start + w_r > height_map.size(1) or y_start + l_r > height_map.size(2):
            support_ratios[b] = 0
            continue
        
        region = height_map[b, x_start:x_start+w_r, y_start:y_start+l_r]
        
        # Find maximum height in the region (support height)
        max_height = torch.max(region)
        
        # For ground level, support is 100%
        if max_height == 0:
            support_ratios[b] = 1.0
            continue
        
        # Calculate number of points at max height (supporting surface)
        support_area = torch.sum(region == max_height).float()
        total_area = w_r * l_r
        
        # Calculate support ratio
        support_ratios[b] = support_area / total_area
    
    return support_ratios


def compute_volume_utilization(
    height_map: torch.Tensor,
    item_dims: torch.Tensor,
    rotation: torch.Tensor,
    position: torch.Tensor,
    bin_size: Tuple[int, int, int],
    gamma: float = 0.2
) -> torch.Tensor:
    """Compute the volume utilization for items at given positions.
    
    Args:
        height_map: [batch_size, W, L] tensor of height map
        item_dims: [batch_size, 3] tensor of item dimensions (w, l, h)
        rotation: [batch_size] tensor of rotation indices (0 or 1)
        position: [batch_size, 2] tensor of (x, y) positions
        bin_size: (W, L, H) dimensions of the bin
        gamma: Penalty factor for uneven support (default 0.2)
        
    Returns:
        [batch_size] tensor of volume utilizations
    """
    batch_size = height_map.size(0)
    device = height_map.device
    
    # Extract item dimensions
    w = item_dims[:, 0]
    l = item_dims[:, 1]
    h = item_dims[:, 2]
    
    # Apply rotation
    w_rotated = torch.where(rotation == 0, w, l)
    l_rotated = torch.where(rotation == 0, l, w)
    
    # Extract positions
    x = position[:, 0]
    y = position[:, 1]
    
    # Initialize utilizations
    utilizations = torch.zeros(batch_size, device=device)
    
    # For each item in the batch
    for b in range(batch_size):
        # Get region under the item - convert all indices to integers
        x_start = int(x[b].item())
        y_start = int(y[b].item())
        w_r = int(w_rotated[b].item())
        l_r = int(l_rotated[b].item())
        h_item = int(h[b].item())
        
        # Check boundaries
        if x_start < 0 or y_start < 0 or x_start + w_r > height_map.size(1) or y_start + l_r > height_map.size(2):
            utilizations[b] = 0
            continue
        
        region = height_map[b, x_start:x_start+w_r, y_start:y_start+l_r]
        
        # Find H_support and H_min
        H_support = torch.max(region)
        H_min = torch.min(region)
        
        # Calculate volume and penalty
        volume = w_r * l_r * h_item
        penalty = gamma * w_r * l_r * (H_support - H_min)
        
        # Calculate utilization
        utilizations[b] = volume - penalty
    
    return utilizations


def compute_objective_function(
    height_map: torch.Tensor,
    item_dims: torch.Tensor,
    rotation: torch.Tensor,
    position: torch.Tensor,
    bin_size: Tuple[int, int, int],
    alpha_t: float = 0.7,
    beta_t: float = 0.3,
    gamma: float = 0.2,
    min_support_ratio: float = 0.8
) -> torch.Tensor:
    """Compute the objective function for items at given positions.
    
    The objective function combines volume utilization and support constraint:
    L_obj = α_t * U(b*, r*, x, y) - β_t * max(0, 0.8 - S(b*, r*, x, y))^2
    
    Args:
        height_map: [batch_size, W, L] tensor of height map
        item_dims: [batch_size, 3] tensor of item dimensions (w, l, h)
        rotation: [batch_size] tensor of rotation indices (0 or 1)
        position: [batch_size, 2] tensor of (x, y) positions
        bin_size: (W, L, H) dimensions of the bin
        alpha_t: Weight for volume utilization (default 0.7)
        beta_t: Weight for support constraint (default 0.3)
        gamma: Penalty factor for uneven support (default 0.2)
        min_support_ratio: Minimum support ratio required (default 0.8)
        
    Returns:
        [batch_size] tensor of objective function values
    """
    # Compute support ratio
    support_ratios = compute_support_ratio(height_map, item_dims, rotation, position)
    
    # Compute volume utilization
    utilizations = compute_volume_utilization(height_map, item_dims, rotation, position, bin_size, gamma)
    
    # Calculate support penalty
    support_penalty = torch.clamp(min_support_ratio - support_ratios, min=0) ** 2
    
    # Calculate objective function
    objective = alpha_t * utilizations - beta_t * support_penalty
    
    return objective


def apply_gradient_update(
    logits: torch.Tensor,
    probs: torch.Tensor,
    height_map: torch.Tensor,
    item_dims: torch.Tensor,
    rotation: torch.Tensor,
    bin_size: Tuple[int, int, int],
    region_bounds: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    valid_mask: Optional[torch.Tensor] = None,
    alpha_t: float = 0.7,
    beta_t: float = 0.3,
    gamma: float = 0.2,
    learning_rate: float = 0.01,
    num_iterations: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply gradient updates to refine logits.
    
    For the demo, we'll use a simplified version that doesn't require gradient computation.
    Instead, we'll adjust the logits based on a heuristic approach.
    
    Args:
        logits: [batch_size, W, L] tensor of initial logits
        probs: [batch_size, W, L] tensor of initial probabilities
        height_map: [batch_size, W, L] tensor of height map
        item_dims: [batch_size, 3] tensor of item dimensions
        rotation: [batch_size] tensor of rotation indices
        bin_size: (W, L, H) dimensions of the bin
        region_bounds: Optional tuple (x_min, y_min, x_max, y_max) to focus refinement
        valid_mask: [batch_size, W, L] boolean tensor for valid positions
        alpha_t, beta_t, gamma: Weights for objective function
        learning_rate: Learning rate for gradient updates
        num_iterations: Number of gradient updates
        
    Returns:
        refined_logits: [batch_size, W, L] tensor of refined logits
        refined_probs: [batch_size, W, L] tensor of refined probabilities
    """
    batch_size, W, L = logits.shape
    device = logits.device
    
    # Create a copy of logits
    refined_logits = logits.clone().detach()
    
    # If region bounds are provided, limit the search to those regions
    if region_bounds is not None:
        x_min, y_min, x_max, y_max = region_bounds
    else:
        # Otherwise, use the entire bin
        x_min = torch.zeros(batch_size, dtype=torch.long, device=device)
        y_min = torch.zeros(batch_size, dtype=torch.long, device=device)
        x_max = torch.full((batch_size,), W, dtype=torch.long, device=device)
        y_max = torch.full((batch_size,), L, dtype=torch.long, device=device)
    
    # For each batch, find the best position based on objective function
    for b in range(batch_size):
        # Get item dimensions
        w, l, h = item_dims[b].int().tolist()
        r = rotation[b].item()
        
        # Apply rotation
        if r == 0:
            w_r, l_r = w, l
        else:
            w_r, l_r = l, w
        
        # Get region bounds as integers
        x_min_val = int(x_min[b].item())
        y_min_val = int(y_min[b].item())
        x_max_val = int(min(x_max[b].item(), W - w_r + 1))
        y_max_val = int(min(y_max[b].item(), L - l_r + 1))
        
        # Find the best position based on objective function
        best_x, best_y = -1, -1
        best_score = float('-inf')
        
        for x in range(x_min_val, x_max_val):
            for y in range(y_min_val, y_max_val):
                # Skip invalid positions
                if valid_mask is not None and not valid_mask[b, x, y]:
                    continue
                
                # Calculate objective function for this position
                position = torch.tensor([[x, y]], device=device)
                obj_value = compute_objective_function(
                    height_map[b:b+1],
                    item_dims[b:b+1],
                    rotation[b:b+1],
                    position,
                    bin_size,
                    alpha_t,
                    beta_t,
                    gamma
                )
                
                # Update best position if better
                score = obj_value.item()
                if score > best_score:
                    best_score = score
                    best_x, best_y = x, y
        
        # Boost the logits for the best position
        if best_x >= 0 and best_y >= 0:
            # Set the logits for the best position to a high value
            # and set other positions to low values
            mask = torch.ones_like(refined_logits[b], dtype=torch.bool)
            mask[best_x, best_y] = False
            
            # Apply the mask
            refined_logits[b] = torch.where(
                mask, 
                torch.full_like(refined_logits[b], -1e9),
                torch.full_like(refined_logits[b], 10.0)
            )
    
    # Calculate final probabilities
    if valid_mask is not None:
        masked_logits = torch.where(
            valid_mask,
            refined_logits,
            torch.tensor(-1e9, device=device, dtype=refined_logits.dtype)
        )
    else:
        masked_logits = refined_logits
    
    refined_probs = torch.softmax(masked_logits.view(batch_size, -1), dim=1).view(batch_size, W, L)
    
    return refined_logits, refined_probs