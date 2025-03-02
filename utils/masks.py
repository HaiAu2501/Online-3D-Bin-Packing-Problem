import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Union


def create_coarse_mask(
    height_map: torch.Tensor,
    buffer: torch.Tensor,
    bin_size: Tuple[int, int, int],
    coarse_grid_size: Tuple[int, int]
) -> torch.Tensor:
    """Create a mask for valid coarse actions.
    
    Args:
        height_map: [batch_size, W, L] tensor of height map
        buffer: [batch_size, buffer_size, 3] tensor of items in buffer
        bin_size: (W, L, H) dimensions of the bin
        coarse_grid_size: (W_c, L_c) size of coarse grid
        
    Returns:
        [batch_size, buffer_size, 2, W_c, L_c] boolean tensor where True indicates a valid action
    """
    batch_size, W, L = height_map.shape
    _, buffer_size, _ = buffer.shape
    W_bin, L_bin, H_bin = bin_size
    W_c, L_c = coarse_grid_size
    
    # Calculate grid cell sizes for coarse grid
    delta_x = W_bin / W_c
    delta_y = L_bin / L_c
    
    # Initialize mask with all True
    mask = torch.ones((batch_size, buffer_size, 2, W_c, L_c), dtype=torch.bool, device=height_map.device)
    
    # For each batch and each item in buffer
    for b in range(batch_size):
        for item_idx in range(buffer_size):
            # Get item dimensions
            item = buffer[b, item_idx]
            w, l, h = item.int().tolist()  # Convert tensor to integers
            
            # Skip placeholder items [0,0,0]
            if w == 0 and l == 0 and h == 0:
                mask[b, item_idx, :, :, :] = False
                continue
            
            # For each rotation
            for r in range(2):
                # Apply rotation if needed
                if r == 0:
                    w_r, l_r = w, l
                else:
                    w_r, l_r = l, w
                
                # For each coarse grid cell
                for i in range(W_c):
                    for j in range(L_c):
                        # Calculate the region boundaries for this coarse cell
                        x_min = int(i * delta_x)
                        y_min = int(j * delta_y)
                        x_max = int(min((i + 1) * delta_x, W_bin))
                        y_max = int(min((j + 1) * delta_y, L_bin))
                        
                        # Check if item can fit within the bin when placed in this region
                        can_fit_x = x_min + w_r <= W_bin
                        can_fit_y = y_min + l_r <= L_bin
                        
                        # If item cannot fit, mark as invalid
                        if not (can_fit_x and can_fit_y):
                            mask[b, item_idx, r, i, j] = False
                            continue
                        
                        # Check height constraints
                        # For simplicity, we check if placing the item at the min position within 
                        # the region would exceed the height limit
                        region_heights = height_map[b, x_min:min(x_min+w_r, W_bin), y_min:min(y_min+l_r, L_bin)]
                        if region_heights.numel() > 0:
                            max_height = torch.max(region_heights)
                            if max_height + h > H_bin:
                                mask[b, item_idx, r, i, j] = False
                                continue
    
    return mask


def create_fine_mask(
    height_map: torch.Tensor,  # [batch_size, W, L]
    item: torch.Tensor,        # [batch_size, 3]
    rotation: torch.Tensor,    # [batch_size]
    bin_size: Tuple[int, int, int],
    region_bounds: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    min_support_ratio: float = 0.8
) -> torch.Tensor:
    """Create a mask for valid fine-grained actions.
    
    Args:
        height_map: [batch_size, W, L] tensor of height map
        item: [batch_size, 3] tensor of item dimensions (w, l, h)
        rotation: [batch_size] tensor of rotation indices (0 or 1)
        bin_size: (W, L, H) dimensions of the bin
        region_bounds: Optional tuple (x_min, y_min, x_max, y_max) of tensors for candidate regions
        min_support_ratio: Minimum support ratio required (default 0.8)
        
    Returns:
        [batch_size, W, L] boolean tensor where True indicates a valid action
    """
    batch_size, W, L = height_map.shape
    W_bin, L_bin, H_bin = bin_size
    
    assert item.shape[0] == batch_size and item.shape[1] == 3, f"Item tensor must have shape [batch_size, 3], got {item.shape}"
    assert rotation.shape[0] == batch_size, f"Rotation tensor must have shape [batch_size], got {rotation.shape}"
    
    # Initialize mask with all False
    mask = torch.zeros((batch_size, W, L), dtype=torch.bool, device=height_map.device)
    
    # If region bounds are provided, limit the search to those regions
    if region_bounds is not None:
        x_min, y_min, x_max, y_max = region_bounds
    else:
        # Otherwise, check the entire bin
        x_min = torch.zeros(batch_size, dtype=torch.long, device=height_map.device)
        y_min = torch.zeros(batch_size, dtype=torch.long, device=height_map.device)
        x_max = torch.full((batch_size,), W, dtype=torch.long, device=height_map.device)
        y_max = torch.full((batch_size,), L, dtype=torch.long, device=height_map.device)
    
    # For each batch
    for b in range(batch_size):
        try:
            # Lấy an toàn kích thước vật phẩm
            if item.size(0) <= b:
                # Nếu batch index vượt quá số lượng item, sử dụng item đầu tiên
                current_item = item[0]
            else:
                current_item = item[b]
                
            # Convert tensor to list of integers
            if current_item.dim() > 0 and current_item.size(0) >= 3:
                w, l, h = current_item.int().tolist()[:3]  # Lấy 3 giá trị đầu tiên
            else:
                # Fallback nếu vẫn không có đủ kích thước
                print(f"Warning: Cannot extract dimensions from item. Using default [1,1,1]")
                w, l, h = 1, 1, 1
                
            r = rotation[b].item() if b < rotation.size(0) else 0  # Default rotation = 0
            
            # Apply rotation if needed
            if r == 0:
                w_r, l_r = w, l
            else:
                w_r, l_r = l, w
            
            # Get region bounds as integers
            x_min_val = int(x_min[b].item() if b < x_min.size(0) else 0)
            y_min_val = int(y_min[b].item() if b < y_min.size(0) else 0)
            x_max_val = int(min(x_max[b].item() if b < x_max.size(0) else W_bin, W_bin - w_r + 1))
            y_max_val = int(min(y_max[b].item() if b < y_max.size(0) else L_bin, L_bin - l_r + 1))
            
            # For each position in the region
            for x in range(x_min_val, x_max_val):
                for y in range(y_min_val, y_max_val):
                    # Check height constraints
                    region_heights = height_map[b, x:x+w_r, y:y+l_r]
                    
                    # Skip if region is not valid
                    if region_heights.numel() == 0:
                        continue
                        
                    max_height = torch.max(region_heights)
                    
                    # Check if item fits within height constraints
                    if max_height + h > H_bin:
                        continue
                    
                    # Calculate support ratio
                    # Count positions where height equals max_height (supporting surface)
                    if max_height > 0:  # Only check support for non-ground placements
                        support_area = torch.sum(region_heights == max_height).item()
                        total_area = w_r * l_r
                        support_ratio = support_area / total_area
                        
                        # Check if support ratio is sufficient
                        if support_ratio < min_support_ratio:
                            continue
                    
                    # If all constraints are satisfied, mark as valid
                    mask[b, x, y] = True
                    
        except Exception as e:
            print(f"Error processing batch {b}: {e}")
            print(f"Item shape: {item.shape}, Rotation shape: {rotation.shape}")
            print(f"Region bounds shapes: x_min={x_min.shape}, y_min={y_min.shape}, x_max={x_max.shape}, y_max={y_max.shape}")
            # Continue with the next batch instead of crashing
            continue
    
    return mask