import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from .gradients import compute_objective_function


def heuristic_refinement(
    height_map: torch.Tensor,
    fine_logits: torch.Tensor,
    region_data: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    bin_size: Tuple[int, int, int],
    method: str = 'greedy',
    alpha_t: float = 0.7,
    beta_t: float = 0.3,
    gamma: float = 0.2,
    temp: float = 1.0,
    max_iterations: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Refine logits using heuristic methods.
    
    Args:
        height_map: [batch_size, W, L] tensor of height map
        fine_logits: [batch_size, W, L] tensor of initial fine logits
        region_data: Dictionary containing:
            - 'x_min', 'y_min', 'x_max', 'y_max': [batch_size] tensors of region boundaries
            - 'items': [batch_size, 3] tensor of item dimensions
            - 'rotation': [batch_size] tensor of rotation indices
        valid_mask: [batch_size, W, L] boolean tensor for valid positions
        bin_size: (W, L, H) dimensions of the bin
        method: Heuristic method ('greedy' or 'simulated_annealing')
        alpha_t, beta_t, gamma: Weights for objective function
        temp: Initial temperature for simulated annealing
        max_iterations: Maximum iterations for simulated annealing
        
    Returns:
        refined_logits: [batch_size, W, L] tensor of refined logits
        refined_probs: [batch_size, W, L] tensor of refined probabilities
    """
    batch_size, W, L = fine_logits.shape
    device = fine_logits.device
    
    # Extract region boundaries
    x_min = region_data['x_min']
    y_min = region_data['y_min']
    x_max = region_data['x_max']
    y_max = region_data['y_max']
    items = region_data['items']
    rotation = region_data['rotation']
    
    # Create output tensors
    refined_logits = fine_logits.clone().detach()
    
    # Apply the selected heuristic method
    if method == 'greedy':
        refined_logits = greedy_search(
            height_map,
            refined_logits,
            items,
            rotation,
            (x_min, y_min, x_max, y_max),
            valid_mask,
            bin_size,
            alpha_t,
            beta_t,
            gamma
        )
    elif method == 'simulated_annealing':
        refined_logits = simulated_annealing(
            height_map,
            refined_logits,
            items,
            rotation,
            (x_min, y_min, x_max, y_max),
            valid_mask,
            bin_size,
            alpha_t,
            beta_t,
            gamma,
            temp,
            max_iterations
        )
    else:
        raise ValueError(f"Unknown heuristic method: {method}")
    
    # Calculate final probabilities
    masked_logits = torch.where(
        valid_mask,
        refined_logits,
        torch.tensor(-1e9, device=device, dtype=refined_logits.dtype)
    )
    refined_probs = torch.softmax(masked_logits.view(batch_size, -1), dim=1).view(batch_size, W, L)
    
    return refined_logits, refined_probs


def greedy_search(
    height_map: torch.Tensor,
    init_logits: torch.Tensor,
    items: torch.Tensor,
    rotation: torch.Tensor,
    region_bounds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    valid_mask: torch.Tensor,
    bin_size: Tuple[int, int, int],
    alpha_t: float = 0.7,
    beta_t: float = 0.3,
    gamma: float = 0.2
) -> torch.Tensor:
    """Apply greedy search to find optimal positions.
    
    Args:
        height_map: [batch_size, W, L] tensor of height map
        init_logits: [batch_size, W, L] tensor of initial logits
        items: [batch_size, 3] tensor of item dimensions
        rotation: [batch_size] tensor of rotation indices
        region_bounds: (x_min, y_min, x_max, y_max) tensors of region boundaries
        valid_mask: [batch_size, W, L] boolean tensor for valid positions
        bin_size: (W, L, H) dimensions of the bin
        alpha_t, beta_t, gamma: Weights for objective function
        
    Returns:
        [batch_size, W, L] tensor of refined logits
    """
    batch_size, W, L = init_logits.shape
    device = init_logits.device
    x_min, y_min, x_max, y_max = region_bounds
    
    # Create output tensor
    refined_logits = init_logits.clone()
    
    # Initialize with large negative values
    base_value = torch.tensor(-1e9, device=device, dtype=refined_logits.dtype)
    
    # For each batch
    for b in range(batch_size):
        # Get item dimensions with rotation applied
        w, l, h = items[b].int().tolist()  # Convert to integers
        r = rotation[b].item()
        
        if r == 0:
            w_r, l_r = w, l
        else:
            w_r, l_r = l, w
        
        # Initialize best position and score
        best_x, best_y = -1, -1
        best_score = float('-inf')
        
        # Get region bounds as integers
        x_min_val = int(x_min[b].item())
        y_min_val = int(y_min[b].item())
        x_max_val = int(min(x_max[b].item(), W - w_r + 1))
        y_max_val = int(min(y_max[b].item(), L - l_r + 1))
        
        # Search within the region
        for x in range(x_min_val, x_max_val):
            for y in range(y_min_val, y_max_val):
                # Skip invalid positions
                if not valid_mask[b, x, y]:
                    continue
                
                # Calculate objective function
                position = torch.tensor([[x, y]], device=device)
                obj_value = compute_objective_function(
                    height_map[b:b+1],
                    items[b:b+1],
                    rotation[b:b+1],
                    position,
                    bin_size,
                    alpha_t,
                    beta_t,
                    gamma
                )
                
                # Update best if better
                if obj_value.item() > best_score:
                    best_score = obj_value.item()
                    best_x, best_y = x, y
        
        # Set all positions to large negative value
        refined_logits[b, :, :] = base_value
        
        # Set best position to a high value
        if best_x >= 0 and best_y >= 0:
            refined_logits[b, best_x, best_y] = 10.0  # High value to make it dominant
    
    return refined_logits


def simulated_annealing(
    height_map: torch.Tensor,
    init_logits: torch.Tensor,
    items: torch.Tensor,
    rotation: torch.Tensor,
    region_bounds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    valid_mask: torch.Tensor,
    bin_size: Tuple[int, int, int],
    alpha_t: float = 0.7,
    beta_t: float = 0.3,
    gamma: float = 0.2,
    init_temp: float = 1.0,
    max_iterations: int = 100
) -> torch.Tensor:
    """Apply simulated annealing to find optimal positions.
    
    Args:
        height_map: [batch_size, W, L] tensor of height map
        init_logits: [batch_size, W, L] tensor of initial logits
        items: [batch_size, 3] tensor of item dimensions
        rotation: [batch_size] tensor of rotation indices
        region_bounds: (x_min, y_min, x_max, y_max) tensors of region boundaries
        valid_mask: [batch_size, W, L] boolean tensor for valid positions
        bin_size: (W, L, H) dimensions of the bin
        alpha_t, beta_t, gamma: Weights for objective function
        init_temp: Initial temperature
        max_iterations: Maximum iterations
        
    Returns:
        [batch_size, W, L] tensor of refined logits
    """
    batch_size, W, L = init_logits.shape
    device = init_logits.device
    x_min, y_min, x_max, y_max = region_bounds
    
    # Create output tensor
    refined_logits = init_logits.clone()
    
    # Initialize with large negative values
    base_value = torch.tensor(-1e9, device=device, dtype=refined_logits.dtype)
    
    # For each batch
    for b in range(batch_size):
        # Get item dimensions with rotation applied
        w, l, h = items[b].int().tolist()  # Convert to integers
        r = rotation[b].item()
        
        if r == 0:
            w_r, l_r = w, l
        else:
            w_r, l_r = l, w
        
        # Get region bounds as integers
        x_min_val = int(x_min[b].item())
        y_min_val = int(y_min[b].item())
        x_max_val = int(min(x_max[b].item(), W - w_r + 1))
        y_max_val = int(min(y_max[b].item(), L - l_r + 1))
        
        # Find valid positions
        valid_positions = []
        for x in range(x_min_val, x_max_val):
            for y in range(y_min_val, y_max_val):
                if valid_mask[b, x, y]:
                    valid_positions.append((x, y))
        
        # If no valid positions, skip
        if not valid_positions:
            continue
        
        # Initialize with a random valid position
        current_pos = valid_positions[np.random.randint(len(valid_positions))]
        position = torch.tensor([[current_pos[0], current_pos[1]]], device=device)
        current_score = compute_objective_function(
            height_map[b:b+1],
            items[b:b+1],
            rotation[b:b+1],
            position,
            bin_size,
            alpha_t,
            beta_t,
            gamma
        ).item()
        
        # Track best position and score
        best_pos = current_pos
        best_score = current_score
        
        # Simulated annealing
        temp = init_temp
        for i in range(max_iterations):
            # Choose a random neighbor from valid positions
            neighbor_idx = np.random.randint(len(valid_positions))
            neighbor_pos = valid_positions[neighbor_idx]
            
            # Skip if it's the same position
            if neighbor_pos == current_pos:
                continue
                
            # Calculate neighbor's score
            position = torch.tensor([[neighbor_pos[0], neighbor_pos[1]]], device=device)
            neighbor_score = compute_objective_function(
                height_map[b:b+1],
                items[b:b+1],
                rotation[b:b+1],
                position,
                bin_size,
                alpha_t,
                beta_t,
                gamma
            ).item()
            
            # Decide whether to accept the neighbor
            if neighbor_score > current_score:
                # Always accept better solutions
                current_pos = neighbor_pos
                current_score = neighbor_score
                
                # Update best if improved
                if current_score > best_score:
                    best_pos = current_pos
                    best_score = current_score
            else:
                # Accept worse solutions with a probability based on temperature
                delta = neighbor_score - current_score
                probability = np.exp(delta / temp)
                if np.random.random() < probability:
                    current_pos = neighbor_pos
                    current_score = neighbor_score
            
            # Cool down
            temp = init_temp * (1 - i / max_iterations)
        
        # Set all positions to large negative value
        refined_logits[b, :, :] = base_value
        
        # Set best position to a high value
        refined_logits[b, best_pos[0], best_pos[1]] = 10.0  # High value to make it dominant
    
    return refined_logits