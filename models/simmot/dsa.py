# Copyright (c) 2026 Yuanzhou Huang. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_temporal_related(L, related_temporal_pe_idx_map, device):
    if related_temporal_pe_idx_map.device != device:
        related_temporal_pe_idx_map = related_temporal_pe_idx_map.to(device)
    
    if L == related_temporal_pe_idx_map.shape[0]:
        return related_temporal_pe_idx_map
    else:
        return related_temporal_pe_idx_map[:L, :L]


class DSA(nn.Module):
    """
    Dual Similarity Aggregation module for SFS and LMC.
    
    Computes similarity matrices from multiple trajectory features
    (appearance, geometry, scores) and aggregates them with learnable weights.
    Incorporates temporal relative position encoding.
    """
    
    def __init__(self, trajectory_length, order: list[str], weights: list[int], 
                 learning: bool=True, use_related_enc:bool=True, chunk_size: int=None):
        super().__init__()
        
        valid_orders = {'features', 'boxes', 'wh', 'scores', 'trends'}
        assert all(n in valid_orders for n in order), f"Invalid order: {set(order) - valid_orders}"
        assert len(order) == len(weights)
        
        self.order = order
        self.traj_len = trajectory_length
        self.chunk_size = chunk_size or trajectory_length
        self.use_related_enc = use_related_enc
        
        # Initialize learnable weights
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float))
        if not learning:
            self.weights.requires_grad = False
        
        # Precompute relative position index mapping
        if self.use_related_enc:
            max_relative_position = trajectory_length - 1
            self.related_encoding = nn.Parameter(
                torch.randn(2 * max_relative_position + 1) * 0.01
            )
            self.register_buffer('pe_offset', torch.tensor(max_relative_position))
            
            t_idxs = torch.arange(trajectory_length)
            t_in_dim0, t_in_dim1 = torch.meshgrid([t_idxs, t_idxs], indexing='ij')
            pe_idx_map = (t_in_dim0 - t_in_dim1).to(torch.long)
            self.register_buffer('related_temporal_pe_idx_map', pe_idx_map)
    
    def _compute_similarity_chunked(self, embeds: torch.Tensor):
        """Compute scaled dot-product similarity with optional chunking."""
        N, L, C = embeds.shape
        scale = C ** -0.5
        
        if L <= self.chunk_size:
            return torch.matmul(embeds, embeds.transpose(-2, -1)) * scale
        
        # Chunked computation for memory efficiency
        simi_list = []
        for i in range(0, L, self.chunk_size):
            end_i = min(i + self.chunk_size, L)
            chunk = embeds[:, i:end_i, :]
            chunk_simi = torch.matmul(chunk, embeds.transpose(-2, -1)) * scale
            simi_list.append(chunk_simi)
        
        return torch.cat(simi_list, dim=1)
    
    def forward(self, traj_features, traj_boxes, traj_scores=None, 
                traj_mask=None, softmax=True, mask_fill=-1e9):
        """
        Args:
            traj_features: [N, L, C] appearance features
            traj_boxes: [N, L, 4] bounding boxes (cxcywh)
            traj_scores: [N, L, 1] detection scores (optional)
            traj_mask: [N, L] boolean mask (True for invalid)
            softmax: whether to apply softmax normalization
            mask_fill: fill value for masked positions
            
        Returns:
            Aggregated similarity matrix [N, L, L]
        """
        _N, _L, _ = traj_features.shape
        device = traj_features.device
        
        if _L > self.traj_len:
            raise ValueError(f"Input length {_L} exceeds max length {self.traj_len}")
        
        # Compute similarity matrices for each feature type
        sim_mat_list = []
        for n in self.order:
            if n == 'features':
                embed_cur = traj_features
            elif n == 'boxes':
                embed_cur = traj_boxes
            elif n == 'wh':
                embed_cur = traj_boxes[..., 2:]
            elif n == 'scores':
                if traj_scores is None:
                    raise ValueError("traj_scores required when 'scores' in order")
                embed_cur = traj_scores
            elif n == 'trends':
                raise NotImplementedError("Trend similarity not implemented")
            
            simi = self._compute_similarity_chunked(embed_cur)
            sim_mat_list.append(simi)
        
        sim_mat = torch.stack(sim_mat_list, dim=-1)  # [N, L, L, num_features]
        weights_norm = self.weights.norm(p=2).clamp(min=1e-8)
        sim_mat = (sim_mat * self.weights).sum(dim=-1) / weights_norm
        
        # Hack: Add temporal relative position encoding
        if self.use_related_enc:
            related_idx = _get_temporal_related(_L, self.related_temporal_pe_idx_map, device)
            related_idx_shifted = related_idx + self.pe_offset
            related_enc = self.related_encoding[related_idx_shifted]
            sim_mat = sim_mat + related_enc.unsqueeze(0)
        
        # Apply mask to invalid positions
        if traj_mask is not None:
            invalid_mask = traj_mask.unsqueeze(1)
            sim_mat = sim_mat.masked_fill(invalid_mask, mask_fill)
        # Apply softmax normalization
        if softmax:
            sim_mat = F.softmax(sim_mat, dim=-1)
            if traj_mask is not None:
                sim_mat = sim_mat * (~invalid_mask).float()
        
        return sim_mat
