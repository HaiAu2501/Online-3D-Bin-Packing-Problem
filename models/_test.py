import torch
from transformer import BinPackingTransformer
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    batch_size = 2
    num_ems = 10
    num_items = 3
    d_model = 128
    max_ems = 1000
    
    # Create dummy input
    ems_input = torch.randint(0, 100, (batch_size, num_ems, 6)).float()
    buffer_input = torch.randint(0, 50, (batch_size, num_items, 3)).float()
    
    # Create dummy mask
    ems_mask = None 
    buffer_mask = None  
    
    # Initialize the model
    model = BinPackingTransformer(d_model=d_model, nhead=8, num_layers=3, dim_feedforward=512, max_ems=max_ems)
    
    # Forward pass
    ems_features, item_features = model(ems_input, buffer_input, ems_mask, buffer_mask)
    
    logger.debug(f"EMS Features shape: {ems_features.shape}")  # [batch_size, d_model]
    logger.debug(f"Item Features shape: {item_features.shape}")  # [batch_size, d_model]