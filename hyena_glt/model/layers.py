"""Dynamic layers that combine BLT's token merging with Hyena operators."""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..config import HyenaGLTConfig
from .operators import HyenaOperator, GenomicPositionalEncoding


class AdaptiveTokenMerger(nn.Module):
    """
    Adaptive token merger inspired by BLT's dynamic patching.
    Merges tokens based on genomic sequence content and patterns.
    """
    
    def __init__(
        self,
        config: HyenaGLTConfig,
        d_model: int,
        min_patch_size: int = 4,
        max_patch_size: int = 16,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        
        # Content-based merger network
        self.content_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Pattern detector for genomic motifs
        self.pattern_detector = nn.Conv1d(
            d_model, 
            d_model // 2, 
            kernel_size=3, 
            padding=1,
            groups=d_model // 4
        )
        
        # Boundary predictor
        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Merge operation
        self.merge_proj = nn.Linear(d_model * max_patch_size, d_model)
        
        # Position encoding adjustment
        self.pos_adjustment = nn.Linear(max_patch_size, 1)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: (batch, seq_len, d_model) input tokens
            attention_mask: (batch, seq_len) attention mask
            
        Returns:
            merged_tokens: (batch, new_seq_len, d_model) merged tokens
            segment_boundaries: (batch, new_seq_len) boundary indicators  
            merge_info: Dictionary with merging statistics
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute content scores for each token
        content_scores = self.content_scorer(x).squeeze(-1)  # (batch, seq_len)
        
        # Detect genomic patterns
        x_conv = rearrange(x, 'b l d -> b d l')
        pattern_features = self.pattern_detector(x_conv)  # (batch, d_model//2, seq_len)
        pattern_features = rearrange(pattern_features, 'b d l -> b l d')
        
        # Combine content and pattern features
        combined_features = torch.cat([x, pattern_features], dim=-1)
        boundary_scores = self.boundary_predictor(combined_features).squeeze(-1)
        
        # Determine merge boundaries
        merge_boundaries = self._determine_merge_boundaries(
            content_scores, 
            boundary_scores, 
            attention_mask
        )
        
        # Perform adaptive merging
        merged_tokens, segment_boundaries, merge_stats = self._perform_merging(
            x, merge_boundaries, attention_mask
        )
        
        merge_info = {
            "original_length": seq_len,
            "merged_length": merged_tokens.size(1),
            "compression_ratio": seq_len / merged_tokens.size(1),
            "merge_stats": merge_stats,
        }
        
        return merged_tokens, segment_boundaries, merge_info
    
    def _determine_merge_boundaries(
        self,
        content_scores: torch.Tensor,
        boundary_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Determine where to place merge boundaries."""
        batch_size, seq_len = content_scores.shape
        
        # Combine content and boundary information
        merge_signal = content_scores * (1 - boundary_scores)
        
        # Apply attention mask
        if attention_mask is not None:
            merge_signal = merge_signal * attention_mask
        
        # Use adaptive thresholding
        threshold = torch.quantile(merge_signal, 0.7, dim=1, keepdim=True)
        merge_boundaries = (merge_signal < threshold).float()
        
        # Ensure minimum and maximum patch sizes
        merge_boundaries = self._enforce_patch_size_constraints(merge_boundaries)
        
        return merge_boundaries
    
    def _enforce_patch_size_constraints(self, boundaries: torch.Tensor) -> torch.Tensor:
        """Enforce minimum and maximum patch size constraints."""
        batch_size, seq_len = boundaries.shape
        
        # Find boundary positions
        for b in range(batch_size):
            boundary_pos = torch.nonzero(boundaries[b], as_tuple=True)[0]
            
            if len(boundary_pos) == 0:
                continue
                
            # Add boundary at start and end
            boundary_pos = torch.cat([
                torch.tensor([0], device=boundaries.device),
                boundary_pos,
                torch.tensor([seq_len - 1], device=boundaries.device)
            ])
            
            # Check patch sizes and adjust
            new_boundaries = torch.zeros_like(boundaries[b])
            current_start = 0
            
            for i in range(1, len(boundary_pos)):
                patch_size = boundary_pos[i] - current_start
                
                if patch_size < self.min_patch_size:
                    # Merge with next patch
                    continue
                elif patch_size > self.max_patch_size:
                    # Split into smaller patches
                    num_splits = (patch_size + self.max_patch_size - 1) // self.max_patch_size
                    split_size = patch_size // num_splits
                    
                    for j in range(1, num_splits):
                        split_pos = current_start + j * split_size
                        if split_pos < seq_len:
                            new_boundaries[split_pos] = 1
                
                new_boundaries[boundary_pos[i]] = 1
                current_start = boundary_pos[i]
            
            boundaries[b] = new_boundaries
        
        return boundaries
    
    def _perform_merging(
        self,
        x: torch.Tensor,
        boundaries: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Perform the actual token merging."""
        batch_size, seq_len, d_model = x.shape
        
        merged_sequences = []
        segment_boundaries_list = []
        merge_stats = {"patches_per_batch": []}
        
        for b in range(batch_size):
            # Get boundary positions for this batch
            boundary_pos = torch.nonzero(boundaries[b], as_tuple=True)[0]
            
            if len(boundary_pos) == 0:
                # No merging needed
                merged_sequences.append(x[b])
                segment_boundaries_list.append(torch.zeros(seq_len, device=x.device))
                merge_stats["patches_per_batch"].append(seq_len)
                continue
            
            # Add start and end boundaries
            boundary_pos = torch.cat([
                torch.tensor([0], device=x.device),
                boundary_pos,
                torch.tensor([seq_len], device=x.device)
            ])
            boundary_pos = torch.unique(boundary_pos, sorted=True)
            
            # Merge tokens within each patch
            merged_patches = []
            segment_boundaries_seq = []
            
            for i in range(len(boundary_pos) - 1):
                start_pos = boundary_pos[i]
                end_pos = boundary_pos[i + 1]
                patch = x[b, start_pos:end_pos]  # (patch_len, d_model)
                
                # Apply attention mask if available
                if attention_mask is not None:
                    patch_mask = attention_mask[b, start_pos:end_pos]
                    patch = patch * patch_mask.unsqueeze(-1)
                
                # Merge patch tokens
                merged_patch = self._merge_patch(patch)
                merged_patches.append(merged_patch)
                
                # Mark as segment boundary except for first patch
                segment_boundaries_seq.append(1 if i > 0 else 0)
            
            # Stack patches
            merged_seq = torch.stack(merged_patches, dim=0)  # (num_patches, d_model)
            merged_sequences.append(merged_seq)
            segment_boundaries_list.append(torch.tensor(segment_boundaries_seq, device=x.device))
            merge_stats["patches_per_batch"].append(len(merged_patches))
        
        # Pad sequences to same length
        max_len = max(seq.size(0) for seq in merged_sequences)
        
        padded_sequences = []
        padded_boundaries = []
        
        for seq, boundaries_seq in zip(merged_sequences, segment_boundaries_list):
            pad_len = max_len - seq.size(0)
            if pad_len > 0:
                padded_seq = F.pad(seq, (0, 0, 0, pad_len))
                padded_bound = F.pad(boundaries_seq, (0, pad_len))
            else:
                padded_seq = seq
                padded_bound = boundaries_seq
                
            padded_sequences.append(padded_seq)
            padded_boundaries.append(padded_bound)
        
        merged_tokens = torch.stack(padded_sequences, dim=0)
        segment_boundaries = torch.stack(padded_boundaries, dim=0).float()
        
        return merged_tokens, segment_boundaries, merge_stats
    
    def _merge_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """Merge tokens within a patch."""
        patch_len, d_model = patch.shape
        
        if patch_len == 1:
            return patch.squeeze(0)
        
        # Weighted average based on content importance
        weights = torch.softmax(torch.sum(patch ** 2, dim=-1), dim=0)
        merged = torch.sum(patch * weights.unsqueeze(-1), dim=0)
        
        return merged


class DynamicHyenaLayer(nn.Module):
    """
    Hyena layer that handles dynamic token sequences from adaptive merging.
    """
    
    def __init__(
        self,
        config: HyenaGLTConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Token merger for this layer
        if config.dynamic_patching:
            self.token_merger = AdaptiveTokenMerger(
                config=config,
                d_model=config.hidden_size,
                min_patch_size=config.min_patch_size,
                max_patch_size=config.max_patch_size,
            )
        else:
            self.token_merger = None
        
        # Hyena operator
        self.hyena_op = HyenaOperator(
            config=config,
            d_model=config.hidden_size,
            l_max=config.max_position_embeddings,
            order=config.hyena_order,
            filter_order=config.hyena_filter_size,
            dropout=config.hyena_dropout,
            layer_idx=layer_idx,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_boundaries: Optional[torch.Tensor] = None,
        return_merge_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
            segment_boundaries: (batch, seq_len) from previous merging
            return_merge_info: Whether to return merging information
            
        Returns:
            hidden_states: (batch, new_seq_len, hidden_size)
            merge_info: Optional merging information
        """
        
        merge_info = None
        
        # Apply token merging if configured
        if self.token_merger is not None:
            merged_states, new_boundaries, merge_info = self.token_merger(
                hidden_states, attention_mask
            )
            
            # Update attention mask for merged sequence
            if attention_mask is not None:
                # Create new attention mask based on merged length
                new_seq_len = merged_states.size(1)
                attention_mask = torch.ones(
                    (hidden_states.size(0), new_seq_len),
                    device=hidden_states.device,
                    dtype=attention_mask.dtype
                )
            
            segment_boundaries = new_boundaries
            hidden_states = merged_states
        
        # Apply Hyena operator with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.hyena_op(
            hidden_states,
            segment_boundaries=segment_boundaries,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + residual
        
        # Apply feed-forward network with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = hidden_states + residual
        
        if return_merge_info:
            return hidden_states, merge_info
        else:
            return hidden_states, None


class HyenaGLTBlock(nn.Module):
    """
    Complete Hyena-GLT block with optional cross-attention to original sequence.
    """
    
    def __init__(
        self,
        config: HyenaGLTConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Main Hyena layer
        self.hyena_layer = DynamicHyenaLayer(config, layer_idx)
        
        # Optional cross-attention to original sequence
        if config.cross_attention_layers > 0 and layer_idx is not None:
            if layer_idx < config.cross_attention_layers:
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    dropout=config.attention_dropout,
                    batch_first=True,
                )
                self.cross_norm = nn.LayerNorm(config.hidden_size)
            else:
                self.cross_attention = None
                self.cross_norm = None
        else:
            self.cross_attention = None
            self.cross_norm = None
        
        # Positional encoding
        self.pos_encoding = GenomicPositionalEncoding(
            d_model=config.hidden_size,
            max_len=config.max_position_embeddings,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        original_sequence: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        segment_boundaries: Optional[torch.Tensor] = None,
        return_merge_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            hidden_states: Current hidden states
            original_sequence: Original unmerged sequence for cross-attention
            attention_mask: Attention mask
            segment_boundaries: Segment boundaries from previous layers
            return_merge_info: Whether to return merge information
        """
        
        # Add positional encoding
        seq_len = hidden_states.size(1)
        pos_encoding = self.pos_encoding(seq_len)
        hidden_states = hidden_states + pos_encoding.unsqueeze(0)
        
        # Apply main Hyena layer
        hidden_states, merge_info = self.hyena_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            segment_boundaries=segment_boundaries,
            return_merge_info=return_merge_info,
        )
        
        # Apply cross-attention if available and original sequence provided
        if self.cross_attention is not None and original_sequence is not None:
            residual = hidden_states
            hidden_states = self.cross_norm(hidden_states)
            
            # Cross-attention to original sequence
            attn_output, _ = self.cross_attention(
                query=hidden_states,
                key=original_sequence,
                value=original_sequence,
            )
            
            hidden_states = attn_output + residual
        
        return hidden_states, merge_info
