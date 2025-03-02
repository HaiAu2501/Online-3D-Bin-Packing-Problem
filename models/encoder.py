import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List


class HeightMapCNN(nn.Module):
    """CNN module for encoding the height map."""
    
    def __init__(self, W: int, L: int, H: int, cnn_out_dim: int = 128):
        """Initialize the CNN encoder for height map.
        
        Args:
            W: Width of the bin
            L: Length of the bin
            H: Height of the bin
            cnn_out_dim: Output dimension of the CNN
        """
        super().__init__()
        
        # Normalize height values by max height
        self.H = H
        
        # CNN layers for height map
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate output size after convolutions (assuming W and L are reasonably large)
        conv_out_h = W
        conv_out_w = L
        
        # Global average pooling + FC layer to get fixed output size
        self.fc = nn.Linear(128 * conv_out_h * conv_out_w // 16, cnn_out_dim)
        
    def forward(self, height_map: torch.Tensor) -> torch.Tensor:
        """Forward pass for height map.
        
        Args:
            height_map: [batch_size, W, L] tensor
            
        Returns:
            [batch_size, cnn_out_dim] tensor
        """
        batch_size = height_map.size(0)
        
        # Normalize height values
        x = height_map.float() / self.H
        
        # Add channel dimension [batch_size, 1, W, L]
        x = x.unsqueeze(1)
        
        # Apply convolutions with max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        
        # Flatten and apply FC layer
        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x))
        
        return x


class OccupancyMapCNN(nn.Module):
    """CNN module for encoding the occupancy map."""
    
    def __init__(self, W: int, L: int, H: int, cnn_out_dim: int = 128):
        """Initialize the CNN encoder for occupancy map.
        
        Args:
            W: Width of the bin
            L: Length of the bin
            H: Height of the bin
            cnn_out_dim: Output dimension of the CNN
        """
        super().__init__()
        
        # 3D CNN layers for occupancy map
        self.conv3d_1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3d_3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        
        # Calculate output size after convolutions and pooling
        conv_out_w = W // 4  # After two 2x2x2 3D max pooling
        conv_out_l = L // 4
        conv_out_h = H // 4
        
        # Global average pooling + FC layer to get fixed output size
        self.fc = nn.Linear(64 * conv_out_w * conv_out_l * conv_out_h, cnn_out_dim)
        
    def forward(self, occupancy_map: torch.Tensor) -> torch.Tensor:
        """Forward pass for occupancy map.
        
        Args:
            occupancy_map: [batch_size, W, L, H] tensor
            
        Returns:
            [batch_size, cnn_out_dim] tensor
        """
        batch_size = occupancy_map.size(0)
        
        # Rearrange dimensions to [batch_size, channels, W, L, H]
        x = occupancy_map.float().unsqueeze(1)
        
        # Apply 3D convolutions with max pooling
        x = F.relu(self.conv3d_1(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv3d_2(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv3d_3(x))
        
        # Flatten and apply FC layer
        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x))
        
        return x


class BufferTransformer(nn.Module):
    """Transformer module for encoding the buffer."""
    
    def __init__(self, buffer_size: int, buffer_out_dim: int = 128, hidden_dim: int = 64):
        """Initialize the transformer encoder for buffer.
        
        Args:
            buffer_size: Maximum number of items in the buffer
            buffer_out_dim: Output dimension of the buffer encoder
            hidden_dim: Hidden dimension of the transformer
        """
        super().__init__()
        
        # Embedding layer for item dimensions
        self.item_embedding = nn.Linear(3, hidden_dim)
        
        # Position encoding
        self.pos_embedding = nn.Parameter(torch.zeros(buffer_size, hidden_dim))
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final linear layer for output
        self.fc_out = nn.Linear(hidden_dim * buffer_size, buffer_out_dim)
        
    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        """Forward pass for buffer.
        
        Args:
            buffer: [batch_size, buffer_size, 3] tensor containing item dimensions
            
        Returns:
            [batch_size, buffer_out_dim] tensor
        """
        batch_size, buffer_size, _ = buffer.shape
        
        # Normalize item dimensions (assuming max value is around 10)
        buffer = buffer.float() / 10.0
        
        # Item embeddings
        item_embedded = self.item_embedding(buffer)
        
        # Add positional encoding
        pos_embedded = item_embedded + self.pos_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(pos_embedded)
        
        # Flatten and apply final FC layer
        transformer_out = transformer_out.reshape(batch_size, -1)
        out = self.fc_out(transformer_out)
        
        return out


class Encoder(nn.Module):
    """Encoder that combines height map, buffer, and occupancy map features."""
    
    def __init__(
        self, 
        W: int, 
        L: int, 
        H: int, 
        buffer_size: int,
        embedding_dim: int = 256,
        use_occupancy: bool = True
    ):
        """Initialize the encoder.
        
        Args:
            W: Width of the bin
            L: Length of the bin
            H: Height of the bin
            buffer_size: Maximum number of items in the buffer
            embedding_dim: Dimension of the final embedding
            use_occupancy: Whether to use occupancy map in encoding
        """
        super().__init__()
        
        # Sub-networks for each input type
        self.height_map_cnn = HeightMapCNN(W, L, H, cnn_out_dim=embedding_dim//2)
        self.buffer_transformer = BufferTransformer(buffer_size, buffer_out_dim=embedding_dim//2)
        
        # Optional occupancy map encoder
        self.use_occupancy = use_occupancy
        if use_occupancy:
            self.occupancy_map_cnn = OccupancyMapCNN(W, L, H, cnn_out_dim=embedding_dim//3)
            # Adjust output FC layer for combined features
            self.fc_out = nn.Linear(embedding_dim//2 + embedding_dim//2 + embedding_dim//3, embedding_dim)
        else:
            self.fc_out = nn.Linear(embedding_dim//2 + embedding_dim//2, embedding_dim)
            
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the encoder.
        
        Args:
            observation: Dictionary containing 'height_map', 'buffer', and optionally 'occupancy_map'
            
        Returns:
            [batch_size, embedding_dim] tensor representing the encoded state
        """
        height_map = observation['height_map']
        buffer = observation['buffer']
        
        # Encode height map
        height_features = self.height_map_cnn(height_map)
        
        # Encode buffer
        buffer_features = self.buffer_transformer(buffer)
        
        # Combine features
        if self.use_occupancy and 'occupancy_map' in observation:
            occupancy_map = observation['occupancy_map']
            occupancy_features = self.occupancy_map_cnn(occupancy_map)
            combined_features = torch.cat([height_features, buffer_features, occupancy_features], dim=1)
        else:
            combined_features = torch.cat([height_features, buffer_features], dim=1)
        
        # Final embedding
        z_t = self.fc_out(combined_features)
        z_t = self.layer_norm(z_t)
        
        return z_t
    
    def get_feature_maps(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get intermediate feature maps for skip connections.
        
        This method is useful for fine policy which needs access to feature maps
        from the encoder for skip connections.
        
        Args:
            observation: Dictionary containing 'height_map', 'buffer', and optionally 'occupancy_map'
            
        Returns:
            Dictionary of feature maps
        """
        height_map = observation['height_map']
        
        # Track activations to return
        feature_maps = {}
        
        # Pass height map through CNN layers and store intermediate activations
        x = height_map.float() / self.height_map_cnn.H
        x = x.unsqueeze(1)  # Add channel dimension
        
        x = F.relu(self.height_map_cnn.conv1(x))
        feature_maps['height_conv1'] = x
        
        x = F.max_pool2d(x, 2)
        x = F.relu(self.height_map_cnn.conv2(x))
        feature_maps['height_conv2'] = x
        
        x = F.max_pool2d(x, 2)
        x = F.relu(self.height_map_cnn.conv3(x))
        feature_maps['height_conv3'] = x
        
        return feature_maps