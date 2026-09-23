# Copyright (c) 2026 Yuanzhou Huang. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.mlp import MLP


class LMC(nn.Module):
    """
    Long-term Memory Clustering (LMC) module for trajectory modeling.
    
    Clusters temporal features into K prototypes using learnable soft assignment.
    Supports bidirectional mapping: pooling (L->K) and unpooling (K->L).
    """
    
    def __init__(
        self, 
        trajectory_length: int, 
        embed_dim: int, 
        ffn_dim_ratio: int,
        k: int, 
        use_norm: bool = False,
        use_checkpoint: bool = False,
        min_valid_nodes: int | None = 0,
        temperature: float = 0.5
    ):
        """
        Initialize LMC module.
        
        Args:
            trajectory_length: Maximum length of trajectories
            embed_dim: Dimension of input embeddings
            ffn_dim_ratio: FFN expansion ratio
            k: Number of cluster prototypes
            use_norm: Whether to use LayerNorm
            use_checkpoint: Whether to use gradient checkpointing
            min_valid_nodes: Minimum number of valid nodes required for clustering.
                           If None, defaults to k (require at least K valid nodes).
                           If set to a value, trajectories with fewer valid nodes will be marked as invalid.
        """
        super().__init__()
        self.traj_len = trajectory_length
        self.embed_dim = embed_dim
        self.k = k
        self.use_norm = use_norm
        self.use_checkpoint = use_checkpoint
        self.min_valid_nodes = k if min_valid_nodes is None else min_valid_nodes
        self.temperature = temperature

        # Cluster assignment head
        self.assign_head = MLP(embed_dim, embed_dim, k, 2, activation=nn.GELU())
        
        # Cluster prototype refinement
        self.mlp_pool = MLP(embed_dim, embed_dim * ffn_dim_ratio, embed_dim, 3, activation=nn.GELU())
        
        if use_norm:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = None

    def compute_valid_mask(self, traj_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute validity mask based on number of valid nodes.
        
        Args:
            traj_mask: Boolean mask of shape [N, L], True for invalid positions
            
        Returns:
            valid_mask: Boolean mask of shape [N], True for valid trajectories
                       (trajectories with >= min_valid_nodes valid nodes)
        """
        # Count valid nodes per trajectory
        num_valid_nodes = (~traj_mask).sum(dim=1)  # [N]
        
        # Check if valid nodes >= min_valid_nodes
        valid_mask = num_valid_nodes >= self.min_valid_nodes  # [N]
        
        return valid_mask

    def predict_cam(
        self, 
        traj_embeds: torch.Tensor, 
        traj_mask: torch.Tensor, 
        traj_simi: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict Cluster Assignment Matrix (CAM).
        
        Args:
            traj_embeds: Trajectory embeddings of shape [N, L, C]
            traj_mask: Boolean mask of shape [N, L], True for invalid positions
            traj_simi: Similarity matrix of shape [N, L, L] or None
            
        Returns:
            pool_cam: Pooling CAM of shape [N, K, L] (softmax over L)
            unpool_cam: Unpooling CAM of shape [N, L, K] (softmax over K)
        """
        # Aggregate temporal context
        if traj_simi is not None:
            context_features = torch.einsum('nij,njc->nic', traj_simi, traj_embeds)
        else:
            context_features = traj_embeds
        
        # Predict assignment scores
        assignment_scores = self.assign_head(context_features)  # [N, L, K]
        assignment_scores = assignment_scores / self.temperature
        
        # Mask invalid positions before softmax
        masked_scores = assignment_scores.masked_fill(
            traj_mask.unsqueeze(-1),  # [N, L, 1]
            -1e9
        )
        
        # Pool CAM: softmax over temporal dimension L
        pool_cam = F.softmax(masked_scores, dim=1).transpose(-1, -2)  # [N, K, L]
        
        # Unpool CAM: softmax over cluster dimension K
        unpool_cam = F.softmax(masked_scores, dim=-1)  # [N, L, K]
        
        return pool_cam, unpool_cam
    
    def aggregate_to_clusters(
        self, 
        traj_embeds: torch.Tensor, 
        pool_cam: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate trajectory features into cluster prototypes.
        
        Args:
            traj_embeds: Trajectory embeddings of shape [N, L, C]
            pool_cam: Pooling CAM of shape [N, K, L]
            
        Returns:
            cluster_prototypes: Refined cluster features of shape [N, K, C]
        """
        # Weighted aggregation: [N, K, L] @ [N, L, C] -> [N, K, C]
        cluster_prototypes = torch.einsum('nkl,nlc->nkc', pool_cam, traj_embeds)
        
        # Refine with MLP
        refined_prototypes = self.mlp_pool(cluster_prototypes)
        
        if self.use_norm:
            refined_prototypes = self.norm(refined_prototypes)
        
        return refined_prototypes
    
    def _forward_impl(
        self, 
        traj_embeds: torch.Tensor, 
        traj_mask: torch.Tensor, 
        traj_simi: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core forward implementation."""
        
        # Compute validity mask
        valid_mask = self.compute_valid_mask(traj_mask)  # [N]
        
        # Predict cluster assignment matrix
        pool_cam, unpool_cam = self.predict_cam(traj_embeds, traj_mask, traj_simi)
        
        # Aggregate to clusters
        lt_memories = self.aggregate_to_clusters(traj_embeds, pool_cam)

        return lt_memories, unpool_cam, pool_cam, valid_mask
    
    def forward(
        self, 
        traj_embeds: torch.Tensor, 
        traj_mask: torch.Tensor, 
        traj_simi: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cluster trajectory features into long-term memory prototypes.
        
        Args:
            traj_embeds: Trajectory embeddings of shape [N, L, C]
            traj_mask: Boolean mask of shape [N, L], True for invalid positions
            traj_simi: Similarity matrix of shape [N, L, L] or None
            
        Returns:
            lt_memories: Long-term memory prototypes of shape [N, K, C]
            unpool_cam: Unpooling CAM of shape [N, L, K]
            pool_cam: Pooling CAM of shape [N, K, L]
            valid_mask: Boolean mask of shape [N], True for valid trajectories
        """
        if self.use_checkpoint and self.training:
            return checkpoint(
                self._forward_impl,
                traj_embeds,
                traj_mask,
                traj_simi,
                use_reentrant=False
            )
        else:
            return self._forward_impl(traj_embeds, traj_mask, traj_simi)
