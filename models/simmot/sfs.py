# Copyright (c) 2026 Yuanzhou Huang. All Rights Reserved.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.mlp import MLP


class SFS(nn.Module):
    """
    Salient Feature Selection (SFS) module for trajectory modeling.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        ffn_dim_ratio: int,
        keep_latest: int, 
        selection_mode: str = 'mask',
        use_checkpoint: bool = False, 
        use_adapter: bool = True,
        non_linear: nn.Module = nn.Tanh(),
        std_scale: float = 1.0,
        min_std: float = 0.005,
        max_std: float = 0.05,
    ):
        """
        Args:
            embed_dim: Dimension of input embeddings
            keep_latest: Number of most recent frames to always retain
            selection_mode: Selection mode, either 'mask' or 'prune'
                - 'mask': Mark unselected frames as invalid (default)
                - 'prune': Physically remove unselected frames
            use_checkpoint: Whether to use gradient checkpointing to save memory
            non_linear: Activation function for score modulation (default: Tanh)
            use_adapter: Whether to use adapter MLP for feature refinement
        """
        super().__init__()
        self.keep_latest = keep_latest
        self.embed_dim = embed_dim
        self.selection_mode = selection_mode
        self.use_checkpoint = use_checkpoint
        self.use_adapter = use_adapter
        self.std_scale = std_scale
        self.min_std = min_std
        self.max_std = max_std
        
        assert selection_mode in ['mask', 'prune'], \
            f"selection_mode must be 'mask' or 'prune', got {selection_mode}"

        # Score prediction head
        self.score_head = MLP(embed_dim, embed_dim, 1, 2, activation=nn.GELU())
        
        # Non-linear activation for score modulation
        self.non_linear = non_linear

        # Optional adapter for feature refinement
        self.adapter = MLP(embed_dim, embed_dim * ffn_dim_ratio, embed_dim, 3, activation=nn.GELU()) if use_adapter else None
        self.norm = nn.LayerNorm(embed_dim)

    def keep_latest_mask(self, scores: torch.Tensor, traj_mask: torch.Tensor) -> torch.Tensor:
        """
        Create mask to retain K most recent valid frames.
        
        Args:
            scores: Saliency scores of shape [N, L]
            traj_mask: Boolean mask of shape [N, L], True for invalid positions
            
        Returns:
            Boolean mask of shape [N, L] with True for selected recent frames
        """
        N, L = scores.shape
        device = scores.device
        latest_k = min(self.keep_latest, L)

        # Create reverse time indices [L-1, L-2, ..., 1, 0]
        reverse_time_idx = torch.arange(L - 1, -1, -1, device=device, dtype=torch.long).float()
        reverse_time_idx = reverse_time_idx.unsqueeze(0).expand(N, -1)
        
        # Assign large values to invalid positions to exclude them
        reverse_time_idx = reverse_time_idx.masked_fill(traj_mask, L)
        
        # Select K smallest indices (most recent valid frames)
        recent_frame_indices = torch.topk(
            reverse_time_idx, k=latest_k, dim=-1, largest=False
        ).indices
        
        # Create selection mask
        selection_mask = torch.zeros_like(traj_mask, dtype=torch.bool)
        selection_mask.scatter_(1, recent_frame_indices, True)
        
        return selection_mask

    def predict_saliency_scores(
        self, 
        traj_embeds: torch.Tensor, 
        traj_simi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict saliency scores for each temporal frame.
        
        Args:
            traj_embeds: Trajectory embeddings of shape [N, L, C]
            traj_simi: Temporal similarity matrix of shape [N, L, L]
            
        Returns:
            raw_scores: Unnormalized saliency scores of shape [N, L]
            modulated_scores: Non-linearly transformed scores of shape [N, L, 1]
        """
        # Aggregate temporal context via similarity-weighted pooling
        context_features = torch.einsum('nij,njc->nic', traj_simi, traj_embeds)
        
        # Predict saliency scores
        raw_scores = self.score_head(context_features).squeeze(-1)
        
        # Apply non-linear modulation (e.g., Tanh maps to [-1, 1])
        modulated_scores = self.non_linear(raw_scores)
        
        return raw_scores, modulated_scores.unsqueeze(-1)
    
    def compute_adaptive_threshold(
        self, 
        scores: torch.Tensor, 
        traj_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive threshold as mean + std of valid saliency scores.
        Handles edge case where all frames in a trajectory are invalid.
        
        Args:
            scores: Saliency scores of shape [N, L]
            traj_mask: Boolean mask of shape [N, L], True for invalid positions
            
        Returns:
            Adaptive threshold values of shape [N]
        """
        valid_scores = scores.masked_fill(traj_mask, float('nan'))
        means = torch.nanmean(valid_scores, dim=-1)
        diff = valid_scores - means.unsqueeze(1)
        diff_squared = diff ** 2
        sum_diff_squared = torch.nansum(diff_squared, dim=-1)
        valid_counts = (~traj_mask).sum(dim=1)
        stds = torch.where(
            valid_counts > 0,
            torch.sqrt(sum_diff_squared / valid_counts),
            torch.tensor(float('nan'), device=scores.device)
        )
        stds = torch.clamp(stds, min=self.min_std, max=self.max_std)
        
        thresh = means + self.std_scale * stds

        max_valid_scores = torch.where(
            traj_mask,
            torch.tensor(-float('inf'), device=valid_scores.device),
            scores
        ).max(dim=1).values  # [N]
        max_valid_scores = torch.where(
            torch.isinf(max_valid_scores),
            torch.tensor(float('nan'), device=valid_scores.device),
            max_valid_scores
        )
        thresh = torch.minimum(thresh, max_valid_scores)

        return thresh
    
    def prune_features(
        self,
        refined_embeds: torch.Tensor,
        traj_mask: torch.Tensor,
        selection_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prune unselected features to reduce sequence length.
        
        This implements Mode 1: physically remove unselected frames.
        
        Args:
            refined_embeds: Refined embeddings of shape [N, L, C]
            traj_mask: Original mask of shape [N, L]
            selection_mask: Boolean mask of shape [N, L], True for selected frames
            
        Returns:
            pruned_embeds: Pruned embeddings of shape [N, K, C] where K <= L
            pruned_mask: Pruned mask of shape [N, K]
            selection_indices: Indices of selected frames of shape [N, K]
        """
        N, L, C = refined_embeds.shape
        device = refined_embeds.device
        
        selection_mask = (~traj_mask) & selection_mask
        num_selected_per_traj = selection_mask.sum(dim=1)  # [N]
        K = num_selected_per_traj.max().item()

        if K >= L:
            return refined_embeds, traj_mask, None

        indices = torch.arange(L, device=device).unsqueeze(0).expand(N, -1)  # [N, L]
        masked_indices = torch.where(
            selection_mask,
            indices,
            torch.full_like(indices, L)  # Use L as sentinel value
        )  # [N, L]
        sorted_indices, _ = torch.sort(masked_indices, dim=1)
        selection_indices = sorted_indices[:, :K]  # [N, K]
        valid_mask = (selection_indices < L)  # [N, K]

        clamped_indices = selection_indices.clamp(max=L-1)  # [N, K]
        gather_indices = clamped_indices.unsqueeze(-1).expand(-1, -1, C)  # [N, K, C]
        pruned_embeds = torch.gather(refined_embeds, dim=1, index=gather_indices)  # [N, K, C]
        pruned_mask = torch.gather(traj_mask, dim=1, index=clamped_indices)  # [N, K]
        pruned_mask = pruned_mask | (~valid_mask)  # [N, K]
        
        return pruned_embeds, pruned_mask, clamped_indices

    def _forward_impl(
        self, 
        traj_embeds: torch.Tensor, 
        traj_mask: torch.Tensor, 
        traj_simi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            traj_embeds: Trajectory embeddings of shape [N, L, C]
            traj_mask: Boolean mask of shape [N, L]
            traj_simi: Similarity matrix of shape [N, L, L]
            
        Returns:
            refined_embeds: Refined embeddings of shape [N, L, C]
            updated_mask: Updated mask after selection of shape [N, L]
            selection_indices: None (not used in mask mode)
        """
        # Step 1: Predict saliency scores
        raw_scores, modulated_scores = self.predict_saliency_scores(traj_embeds, traj_simi)

        # Step 2: Determine selection mask via adaptive thresholding
        adaptive_thresh = self.compute_adaptive_threshold(raw_scores, traj_mask)
        threshold_mask = (raw_scores >= adaptive_thresh[:, None])
        
        # Step 3: Ensure recent frames are always retained
        recent_frame_mask = self.keep_latest_mask(raw_scores, traj_mask)
        
        # Step 4: Combine masks (union of threshold-based and recency-based selection)
        selection_mask = threshold_mask | recent_frame_mask

        # Step 5: Refine features via score-modulated adaptation
        if self.use_adapter:
            # Adapter pathway: modulated_scores act as attention weights
            adapted_features = self.adapter(traj_embeds * modulated_scores)
            refined_embeds = traj_embeds + adapted_features  # Residual connection
            refined_embeds = self.norm(refined_embeds)
        else:
            refined_embeds = traj_embeds + traj_embeds * modulated_scores
            refined_embeds = self.norm(refined_embeds)
        
        if self.selection_mode == "mask":  # Update trajectory mask (mark unselected frames as invalid)
            updated_mask = traj_mask | (~selection_mask)
            return refined_embeds, updated_mask, None
        elif self.selection_mode == "prune":  #  Prune unselected features
            return self.prune_features(refined_embeds, traj_mask, selection_mask)
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

    def forward(
        self, 
        traj_embeds: torch.Tensor, 
        traj_mask: torch.Tensor, 
        traj_simi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select and refine salient temporal features.
        
        Args:
            traj_embeds: Trajectory embeddings of shape [N, L, C]
            traj_mask: Boolean mask of shape [N, L], True for invalid positions
            traj_simi: Temporal similarity matrix of shape [N, L, L] from DSA module
            
        Returns:
            refined_embeds: Score-modulated and refined embeddings
                - Mask mode: [N, L, C]
                - Prune mode: [N, K, C] where K <= L
            updated_mask: Updated mask after saliency-based filtering
                - Mask mode: [N, L]
                - Prune mode: [N, K]
            selection_info: Selection information
                - Mask mode: None
                - Prune mode: selection_indices of shape [N, K]
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
