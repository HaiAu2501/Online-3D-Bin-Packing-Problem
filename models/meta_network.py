import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Union


class MetaNetwork(nn.Module):
    """
    MetaNetwork that combines and ensembles gradient-based and heuristic refinement results.
    
    This network takes two sets of logits from different refinement approaches:
    1. Gradient-based refinement logits
    2. Heuristic-based refinement logits
    
    And outputs the final combined logits through adaptive weighting.
    """
    
    def __init__(
        self,
        W: int,  # Full bin width
        L: int,  # Full bin length
        hidden_dim: int = 64,
        embedding_dim: int = 128
    ):
        """Initialize the MetaNetwork.
        
        Args:
            W: Width of the bin
            L: Length of the bin
            hidden_dim: Hidden dimension of the network
            embedding_dim: Dimension of state embedding (if used)
        """
        super().__init__()
        
        self.W = W
        self.L = L
        
        # Feature extraction networks for analyzing logit patterns
        # We could use CNN here, but for simplicity, using MLPs with flattened inputs
        self.grad_feature_extractor = nn.Sequential(
            nn.Linear(W * L, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.heur_feature_extractor = nn.Sequential(
            nn.Linear(W * L, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Combination network
        # This takes features from both sets of logits and outputs weights for combination
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 weights: grad_weight and heur_weight
        )
        
        # Optional context network to incorporate state information
        self.use_context = embedding_dim > 0
        if self.use_context:
            self.context_network = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.context_combiner = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )
        
    def forward(
        self,
        grad_logits: torch.Tensor,
        heur_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        z_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ensembling refinement results.
        
        Args:
            grad_logits: [batch_size, W, L] tensor of gradient-based refinement logits
            heur_logits: [batch_size, W, L] tensor of heuristic-based refinement logits
            valid_mask: Optional [batch_size, W, L] boolean tensor for valid placement positions
            z_t: Optional [batch_size, embedding_dim] tensor of state embedding for context-aware weighting
            
        Returns:
            ensemble_logits: [batch_size, W, L] tensor of ensembled logits
            ensemble_probs: [batch_size, W, L] tensor of ensembled probabilities
        """
        batch_size = grad_logits.size(0)
        
        # Flatten logits for feature extraction
        flat_grad_logits = grad_logits.view(batch_size, -1)
        flat_heur_logits = heur_logits.view(batch_size, -1)
        
        # Extract features from each set of logits
        grad_features = self.grad_feature_extractor(flat_grad_logits)
        heur_features = self.heur_feature_extractor(flat_heur_logits)
        
        # Combine features
        combined_features = torch.cat([grad_features, heur_features], dim=1)
        
        # Determine weights for combining logits
        if self.use_context and z_t is not None:
            # Context-aware weighting
            context_features = self.context_network(z_t)
            context_combined = torch.cat([combined_features, context_features], dim=1)
            weights = self.context_combiner(context_combined)
        else:
            # Context-free weighting
            weights = self.combiner(combined_features)
        
        # Apply softmax to get normalized weights
        norm_weights = F.softmax(weights, dim=1)
        
        # Combine logits using the learned weights
        grad_weight = norm_weights[:, 0].view(batch_size, 1, 1)
        heur_weight = norm_weights[:, 1].view(batch_size, 1, 1)
        
        ensemble_logits = grad_weight * grad_logits + heur_weight * heur_logits
        
        # Apply mask if provided
        if valid_mask is not None:
            # Set logits for invalid positions to large negative value
            masked_logits = torch.where(
                valid_mask,
                ensemble_logits,
                torch.tensor(-1e9, device=ensemble_logits.device, dtype=ensemble_logits.dtype)
            )
        else:
            masked_logits = ensemble_logits
        
        # Calculate probabilities
        ensemble_probs = F.softmax(masked_logits.view(batch_size, -1), dim=1)
        ensemble_probs = ensemble_probs.view(batch_size, self.W, self.L)
        
        return ensemble_logits, ensemble_probs
    
    def get_weights(
        self,
        grad_logits: torch.Tensor,
        heur_logits: torch.Tensor,
        z_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get the weights used for ensembling.
        
        Args:
            grad_logits: [batch_size, W, L] tensor of gradient-based refinement logits
            heur_logits: [batch_size, W, L] tensor of heuristic-based refinement logits
            z_t: Optional [batch_size, embedding_dim] tensor of state embedding
            
        Returns:
            weights: [batch_size, 2] tensor of weights for [gradient, heuristic]
        """
        batch_size = grad_logits.size(0)
        
        # Flatten logits for feature extraction
        flat_grad_logits = grad_logits.view(batch_size, -1)
        flat_heur_logits = heur_logits.view(batch_size, -1)
        
        # Extract features from each set of logits
        grad_features = self.grad_feature_extractor(flat_grad_logits)
        heur_features = self.heur_feature_extractor(flat_heur_logits)
        
        # Combine features
        combined_features = torch.cat([grad_features, heur_features], dim=1)
        
        # Determine weights for combining logits
        if self.use_context and z_t is not None:
            # Context-aware weighting
            context_features = self.context_network(z_t)
            context_combined = torch.cat([combined_features, context_features], dim=1)
            weights = self.context_combiner(context_combined)
        else:
            # Context-free weighting
            weights = self.combiner(combined_features)
        
        # Apply softmax to get normalized weights
        norm_weights = F.softmax(weights, dim=1)
        
        return norm_weights