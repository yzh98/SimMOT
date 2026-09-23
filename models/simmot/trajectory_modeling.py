# Copyright (c) 2026 Yuanzhou Huang. All Rights Reserved.

import math
import einops
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.simmot.dsa import DSA
from models.simmot.lmc import LMC
from models.simmot.sfs import SFS
from models.mlp import MLP


class TrajectoryModeling(nn.Module):
    def __init__(
        self, 
        trajectory_length: int, 
        feat_dim: int, 
        ffn_dim_ratio: int,
        keep_recent_ratio: float = 0.25,
        num_memories: int = 5, 
        share_dsa: bool = False,
        use_adapter: bool = False,
        use_checkpoint: bool = False
    ):
        """
        Trajectory modeling module with salient feature selection and long-term memory.
        
        Args:
            trajectory_length: Maximum trajectory length (L)
            feat_dim: Feature dimension (C)
            ffn_dim_ratio: FFN expansion ratio
            keep_recent_ratio: Ratio of recent frames to keep in SFS (0 < ratio <= 1)
            num_memories: Number of long-term memory clusters
            share_dsa: Whether to share DSA between SFS and LMC
            use_adapter: Whether to use adapter after FAA
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        self.trajectory_length = trajectory_length
        self.feat_dim = feat_dim
        self.share_dsa = share_dsa
        self.use_adapter = use_adapter
        self.use_checkpoint = use_checkpoint
        
        assert 0 < keep_recent_ratio <= 1, "keep_recent_ratio must be in (0, 1]"
        self.keep_recent_frames = math.ceil(keep_recent_ratio * trajectory_length)
        
        # Core modules
        self.dsa = DSA(trajectory_length, order=['features', 'wh'], weights=[1.0, 0.01])
        self.dsa_lmc = None if share_dsa else DSA(trajectory_length, order=['features', 'wh'], weights=[1.0, 0.01])
        self.sfs = SFS(feat_dim, ffn_dim_ratio, self.keep_recent_frames)
        self.lmc = LMC(trajectory_length, feat_dim, ffn_dim_ratio, num_memories)
        
        # Feature alignment & aggregation
        self.faa_mlp = MLP(feat_dim, feat_dim * ffn_dim_ratio, feat_dim, 3, activation=nn.GELU())
        self.faa_norm = nn.LayerNorm(feat_dim)
        
        if use_adapter:
            self.adapter = MLP(feat_dim, feat_dim * ffn_dim_ratio, feat_dim, 2, activation=nn.GELU())
            self.norm = nn.LayerNorm(feat_dim)
        else:
            self.adapter = self.norm = None
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for name, param in self.named_parameters():
            if param.dim() > 1 and "related_encoding" not in name:
                nn.init.xavier_uniform_(param)
    
    def compute_similarity_matrices(
        self,
        trajectory_features: torch.Tensor,
        trajectory_boxes: torch.Tensor,
        trajectory_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute intra-trajectory similarity matrices for SFS and LMC."""
        sim_mat = self.dsa(trajectory_features, trajectory_boxes, traj_mask=trajectory_masks)
        
        if self.share_dsa:
            sim_mat_sfs = sim_mat_lmc = sim_mat
        else:
            sim_mat_sfs = sim_mat
            sim_mat_lmc = self.dsa_lmc(trajectory_features, trajectory_boxes, traj_mask=trajectory_masks)
        
        return sim_mat_sfs, sim_mat_lmc
    
    def feature_alignment_aggregation(
        self, 
        salient_features: torch.Tensor,
        lt_memories: torch.Tensor,
        select_indices: torch.Tensor | None,
        lt_uncam: torch.Tensor,
        lt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Align and aggregate short-term salient features with long-term memories."""
        K_clus = lt_memories.shape[1]
        
        # Extract CAM entries for selected frames
        if select_indices is not None:
            selected_indices_expanded = select_indices.unsqueeze(-1).expand(-1, -1, K_clus)
            selected_cam = torch.gather(lt_uncam, dim=1, index=selected_indices_expanded)
        else:
            selected_cam = lt_uncam
        
        # Redistribute and refine long-term memories
        redistributed_memories = torch.bmm(selected_cam, lt_memories)
        redistributed_memories = self.faa_mlp(redistributed_memories)
        # redistributed_memories = redistributed_memories * lt_mask.view(-1, 1, 1).float()
        
        # Aggregate via residual connection
        aggregated_features = self.faa_norm(salient_features + redistributed_memories)
        
        if self.use_adapter:
            aggregated_features = aggregated_features + self.adapter(aggregated_features)
            aggregated_features = self.norm(aggregated_features)
        
        return aggregated_features

    def _forward_core(
        self, 
        trajectory_features: torch.Tensor,
        trajectory_boxes: torch.Tensor,
        trajectory_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core forward pass through DSA, LMC, and SFS modules."""
        sim_mat_sfs, sim_mat_lmc = self.compute_similarity_matrices(
            trajectory_features, trajectory_boxes, trajectory_masks
        )
        
        lt_memories, lt_uncam, _, lt_mask = self.lmc(trajectory_features, trajectory_masks, sim_mat_lmc)
        salient_features, updated_masks, select_indices = self.sfs(trajectory_features, trajectory_masks, sim_mat_sfs)
        
        # Feature alignment & aggregation
        aggregated_features = self.feature_alignment_aggregation(
            salient_features, lt_memories, select_indices, lt_uncam, lt_mask
        )
        
        return aggregated_features, updated_masks, select_indices
    
    def _prepare_temporal_data(self, seq_info: dict) -> tuple[torch.Tensor, ...]:
        """Prepare temporal data by rearranging dimensions."""
        trajectory_features = einops.rearrange(seq_info["trajectory_features"], "b g t l n c -> (b g t n) l c")
        trajectory_boxes = einops.rearrange(seq_info["trajectory_boxes"], "b g t l n c -> (b g t n) l c")
        trajectory_id_labels = einops.rearrange(seq_info["trajectory_id_labels"], "b g t l n -> (b g t n) l")
        trajectory_times = einops.rearrange(seq_info["trajectory_times"], "b g t l n -> (b g t n) l")
        trajectory_masks = einops.rearrange(seq_info["trajectory_masks"], "b g t l n -> (b g t n) l")
        
        return trajectory_features, trajectory_boxes, trajectory_id_labels, trajectory_times, trajectory_masks
    
    def _collect_results(
        self, 
        trajectory_features: torch.Tensor, 
        trajectory_masks: torch.Tensor, 
        trajectory_id_labels: torch.Tensor,
        trajectory_times: torch.Tensor, 
        select_indices: torch.Tensor,
        B: int, 
        G: int, 
        N: int
    ) -> dict:
        """Collect and reshape results back to original format."""
        if select_indices is not None:
            trajectory_id_labels =  torch.gather(trajectory_id_labels, dim=1, index=select_indices)
            trajectory_times = torch.gather(trajectory_times, dim=1, index=select_indices)
        
        return {
            "trajectory_features": einops.rearrange(
                trajectory_features, "(b g t n) l c -> b g t l n c", b=B, g=G, n=N
            ),
            "trajectory_masks": einops.rearrange(
                trajectory_masks, "(b g t n) l -> b g t l n", b=B, g=G, n=N
            ),
            "trajectory_id_labels": einops.rearrange(
                trajectory_id_labels, "(b g t n) l -> b g t l n", b=B, g=G, n=N
            ),
            "trajectory_times": einops.rearrange(
                trajectory_times, "(b g t n) l -> b g t l n", b=B, g=G, n=N
            ),
        }
    
    def forward(self, seq_info: dict) -> dict:
        """
        Forward pass with trajectory modeling.
        
        Args:
            seq_info: Dictionary with trajectory data [B, G, T, L, N, ...]
        
        Returns:
            Updated seq_info with processed features [B, G, T, K_selc, N, C]
        """
        B, G, T, L, N, C = seq_info["trajectory_features"].shape
        if B * G * T * N * L == 0:
            return seq_info

        # Prepare temporal data
        trajectory_features, trajectory_boxes, trajectory_id_labels, trajectory_times, trajectory_masks = \
            self._prepare_temporal_data(seq_info)
        
        # Core forward pass with optional gradient checkpointing
        if self.use_checkpoint and self.training:
            aggregated_features, updated_masks, select_indices = checkpoint(
                self._forward_core, trajectory_features, trajectory_boxes, trajectory_masks, use_reentrant=False
            )
        else:
            aggregated_features, updated_masks, select_indices = self._forward_core(
                trajectory_features, trajectory_boxes, trajectory_masks
            )
        
        # Apply mask to features
        trajectory_features = aggregated_features * (~updated_masks[..., None]).float()
        
        # Collect and reshape results
        results = self._collect_results(trajectory_features, updated_masks, trajectory_id_labels, trajectory_times, select_indices, B, G, N)
        seq_info.update(results)
        
        return seq_info
