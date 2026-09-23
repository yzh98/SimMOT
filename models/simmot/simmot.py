# ------------------------------------------------------------------------
# SimMOT
# Copyright (c) 2026 Yuanzhou Huang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import einops


class SimMOT(nn.Module):
    def __init__(
        self,
        detr: nn.Module,
        detr_framework: str,
        only_detr: bool,
        trajectory_modeling: nn.Module,
        id_decoder: nn.Module,
        temporal_stride: int = 1,
        temporal_sampling_strategy: str = 'random',
        exclude_first_frame: bool = True,
    ):
        """
        SimMOT: Simple Multi-Object Tracking with temporal sampling optimization.
        
        Args:
            detr: DETR-based detection model
            detr_framework: Framework name (e.g., 'DINO', 'DeformableDETR')
            only_detr: Whether to use only DETR without tracking
            trajectory_modeling: Trajectory modeling module
            id_decoder: ID decoder module
            temporal_stride: Temporal sampling stride (1=no sampling, 2=half, 4=quarter)
            temporal_sampling_strategy: Sampling strategy ('random' or 'uniform')
            exclude_first_frame: Whether to exclude frame 0 (no historical trajectory)
        """
        super().__init__()
        self.detr = detr
        self.detr_framework = detr_framework
        self.only_detr = only_detr
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder
        self.temporal_stride = temporal_stride
        self.temporal_sampling_strategy = temporal_sampling_strategy
        self.exclude_first_frame = exclude_first_frame

        self.num_id_vocabulary = id_decoder.num_id_vocabulary if id_decoder is not None else 1000

    def _sample_temporal_indices(self, T: int, device: torch.device) -> torch.Tensor:
        """Sample temporal indices. Last frame always kept, first frame optionally excluded."""
        if self.temporal_stride == 1 or not self.training:
            return torch.arange(T, device=device)
        
        start_idx = 1 if self.exclude_first_frame else 0
        T_sampled = max(1, T // self.temporal_stride)
        
        if T_sampled == 1:
            return torch.tensor([T - 1], device=device)
        
        if self.temporal_sampling_strategy == 'uniform':
            return torch.linspace(start_idx, T - 1, T_sampled, device=device).long()
        
        elif self.temporal_sampling_strategy == 'random':
            available_range = T - 1 - start_idx
            num_middle = T_sampled - 1
            
            if available_range <= 0:
                return torch.tensor([T - 1], device=device)
            
            if num_middle >= available_range:
                middle_indices = torch.arange(start_idx, T - 1, device=device)
            else:
                available = torch.arange(start_idx, T - 1, device=device)
                perm = torch.randperm(available_range, device=device)[:num_middle]
                middle_indices = available[perm].sort()[0]
            
            return torch.cat([middle_indices, torch.tensor([T - 1], device=device)])
        
        else:
            raise ValueError(f"Unknown temporal_sampling_strategy: {self.temporal_sampling_strategy}")
    
    def _apply_temporal_sampling(self, seq_info: dict, sampled_indices: torch.Tensor) -> dict:
        """Apply temporal sampling to all tensors with shape [B, G, T, ...]."""
        B, G, T = seq_info["trajectory_features"].shape[:3]
        sampled_seq_info = {}
        
        for key, value in seq_info.items():
            if isinstance(value, torch.Tensor) and list(value.shape[:3]) == [B, G, T]:
                sampled_seq_info[key] = value[:, :, sampled_indices]
            else:
                sampled_seq_info[key] = value
        
        return sampled_seq_info
    
    def _prepare_trajectory(self, seq_info: dict) -> dict:
        """Prepare trajectory data with temporal expansion and causal masking."""
        B, G, L, N = seq_info["trajectory_features"].shape[:4]
        T = L if self.training else 1
        
        # Expand temporal dimension: [B, G, L, N, ...] -> [B, G, T, L, N, ...]
        seq_info["trajectory_features"] = einops.repeat(
            seq_info["trajectory_features"], "b g l n c -> b g t l n c", t=T
        )
        seq_info["trajectory_boxes"] = einops.repeat(
            seq_info["trajectory_boxes"], "b g l n c -> b g t l n c", t=T
        )
        seq_info["trajectory_masks"] = einops.repeat(
            seq_info["trajectory_masks"], "b g l n -> b g t l n", t=T
        ).clone()  # in-place operation
        seq_info["trajectory_id_labels"] = einops.repeat(
            seq_info["trajectory_id_labels"], "b g l n -> b g t l n", t=T
        )
        seq_info["trajectory_times"] = einops.repeat(
            seq_info["trajectory_times"], "b g l n -> b g t l n", t=T
        )
        
        # Apply causal temporal mask: at time t, only frames [0, t-1] are visible
        if self.training:
            temporal_mask = torch.triu(
                torch.ones(T, L, dtype=torch.bool, device=seq_info["trajectory_masks"].device),
                diagonal=0
            )
            temporal_mask = einops.repeat(temporal_mask, "t l -> b g t l n", b=B, g=G, n=N)
            seq_info["trajectory_masks"][temporal_mask] = True
        
        return seq_info

    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for SimMOT forward."
        
        match kwargs["part"]:
            case "detr":
                frames = kwargs["frames"]
                use_checkpoint = kwargs.get("use_checkpoint", False)
                if use_checkpoint:
                    return checkpoint(self.detr, frames, use_reentrant=False)
                return self.detr(samples=frames)
            
            case "trajectory_modeling":
                seq_info = self._prepare_trajectory(kwargs["seq_info"])
                
                # Apply temporal sampling during training
                if self.training and self.temporal_stride > 1:
                    T = seq_info["trajectory_features"].shape[2]
                    device = seq_info["trajectory_features"].device
                    sampled_indices = self._sample_temporal_indices(T, device)
                    seq_info = self._apply_temporal_sampling(seq_info, sampled_indices)
                
                return self.trajectory_modeling(seq_info)
            
            case "id_decoder":
                seq_info = kwargs["seq_info"]
                use_decoder_checkpoint = kwargs.get("use_decoder_checkpoint", False)
                return self.id_decoder(seq_info, use_decoder_checkpoint=use_decoder_checkpoint)
            
            case _:
                raise NotImplementedError(f"SimMOT forwarding doesn't support part={kwargs['part']}.")
