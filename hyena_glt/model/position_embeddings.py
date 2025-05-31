"""
Position embedding systems for BLT-style token merging in Hyena-GLT.

Based on the "Hyena-BLT-Genome Technical Guide.pdf", this implements proper
position embedding handling when tokens are dynamically merged into patches.
"""

import math
from typing import Any

import torch
import torch.nn as nn


class SegmentAwarePositionalEncoding(nn.Module):
    """
    Position encoding that properly handles dynamic token merging.

    This implementation follows the BLT technical guide's recommendation for
    handling position embeddings when tokens are merged into variable-length patches.
    It maintains position information through:
    1. Segment-aware position computation
    2. Cross-attention bridges between local and global representations
    3. Patch boundary tracking
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 32768,
        use_learned_encoding: bool = True,
        segment_encoding_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_learned_encoding = use_learned_encoding
        self.segment_encoding_dim = segment_encoding_dim

        # Standard sinusoidal position encoding
        self.register_buffer(
            "pe_base", self._create_sinusoidal_encoding(max_len, d_model)
        )

        # Learned components for handling merged tokens
        if use_learned_encoding:
            self.segment_encoder = nn.Linear(
                3, segment_encoding_dim
            )  # [pos_in_patch, patch_length, global_pos]
            self.position_projection = nn.Linear(
                d_model + segment_encoding_dim, d_model
            )

        # Genomic-specific encodings
        self._add_genomic_patterns()

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create standard sinusoidal position encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _add_genomic_patterns(self) -> None:
        """Add genomic-specific positional patterns."""
        # Codon patterns (period 3)
        codon_freqs = torch.arange(0, self.d_model // 4, 2).float() * (2 * math.pi / 3)
        self.register_buffer("codon_freqs", codon_freqs)

        # Common genomic motif patterns
        motif_periods = [8, 10, 21, 147]  # nucleosome, etc.
        motif_freqs = []
        for period in motif_periods:
            freqs = torch.arange(0, min(self.d_model // 8, 8), 2).float() * (
                2 * math.pi / period
            )
            motif_freqs.append(freqs)
        self.register_buffer("motif_freqs", torch.cat(motif_freqs))

    def forward(
        self,
        seq_len: int,
        patch_boundaries: torch.Tensor | None = None,
        original_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate position embeddings that are aware of token merging.

        Args:
            seq_len: Current sequence length (after merging)
            patch_boundaries: (batch, seq_len) indicating patch boundaries
            original_positions: (batch, seq_len) original positions before merging

        Returns:
            Position embeddings of shape (batch, seq_len, d_model)
        """
        # Base sinusoidal encoding
        base_pe = self.pe_base[:seq_len]  # (seq_len, d_model)

        if patch_boundaries is None or not self.use_learned_encoding:
            # Simple case: no merging or learned encoding
            return base_pe.unsqueeze(0)  # type: ignore[no-any-return]

        batch_size = patch_boundaries.size(0)
        device = patch_boundaries.device

        # Compute segment-aware position features
        segment_features = self._compute_segment_features(
            batch_size, seq_len, patch_boundaries, original_positions, device
        )

        # Encode segment features
        segment_encoded = self.segment_encoder(
            segment_features
        )  # (batch, seq_len, segment_encoding_dim)

        # Combine base encoding with segment information
        base_pe_batch = base_pe.unsqueeze(0).expand(batch_size, -1, -1)
        combined = torch.cat([base_pe_batch, segment_encoded], dim=-1)

        # Project to final dimension
        final_pe = self.position_projection(combined)

        # Add genomic patterns
        final_pe = self._add_genomic_position_patterns(final_pe, original_positions)

        return final_pe

    def _compute_segment_features(
        self,
        batch_size: int,
        seq_len: int,
        patch_boundaries: torch.Tensor,
        original_positions: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute features for each position based on its patch context.

        Returns:
            segment_features: (batch, seq_len, 3) containing:
                - position within patch (normalized)
                - patch length (normalized)
                - global position (normalized)
        """
        segment_features = torch.zeros(batch_size, seq_len, 3, device=device)

        for b in range(batch_size):
            # Find patch boundaries for this batch
            boundaries = patch_boundaries[b]
            boundary_indices = torch.nonzero(boundaries, as_tuple=True)[0]

            # Add start and end boundaries
            boundary_indices = torch.cat(
                [
                    torch.tensor([0], device=device),
                    boundary_indices,
                    torch.tensor([seq_len], device=device),
                ]
            )
            boundary_indices = torch.unique(boundary_indices, sorted=True)

            # Compute features for each patch
            for i in range(len(boundary_indices) - 1):
                start_pos = boundary_indices[i]
                end_pos = boundary_indices[i + 1]
                patch_length = end_pos - start_pos

                # Position within patch (0 to 1)
                pos_in_patch = torch.arange(
                    patch_length, device=device, dtype=torch.float
                )
                pos_in_patch = pos_in_patch / max(patch_length - 1, 1)

                # Patch length (normalized by max possible)
                patch_len_norm = patch_length / seq_len

                # Global position
                if original_positions is not None:
                    global_pos = original_positions[b, start_pos:end_pos]
                    global_pos_norm = global_pos / self.max_len
                else:
                    global_pos_norm = (
                        torch.arange(
                            start_pos, end_pos, device=device, dtype=torch.float
                        )
                        / seq_len
                    )

                # Store features
                segment_features[b, start_pos:end_pos, 0] = pos_in_patch
                segment_features[b, start_pos:end_pos, 1] = patch_len_norm
                segment_features[b, start_pos:end_pos, 2] = global_pos_norm

        return segment_features

    def _add_genomic_position_patterns(
        self, pe: torch.Tensor, original_positions: torch.Tensor | None
    ) -> torch.Tensor:
        """Add genomic-specific positional patterns."""
        if original_positions is None or not self.use_learned_encoding:
            return pe

        batch_size, seq_len, d_model = pe.shape
        device = pe.device

        # Ensure original_positions has the right shape and type
        if original_positions.size(1) != seq_len:
            # Create positions for current sequence length
            positions = torch.arange(
                seq_len, device=device, dtype=torch.float
            ).unsqueeze(0)
            positions = positions.expand(batch_size, -1)
        else:
            positions = original_positions.float()

        # Add codon patterns
        if hasattr(self, "codon_freqs") and len(self.codon_freqs) > 0:
            codon_pos = positions.unsqueeze(-1) * self.codon_freqs.unsqueeze(
                0
            ).unsqueeze(0)
            codon_encoding = torch.sin(codon_pos)  # (batch, seq_len, codon_dims)

            # Add to appropriate dimensions
            codon_dims = min(codon_encoding.size(-1), d_model // 4)
            if codon_dims > 0:
                pe[:, :, :codon_dims] += codon_encoding[:, :, :codon_dims] * 0.1

        # Add motif patterns
        if hasattr(self, "motif_freqs") and len(self.motif_freqs) > 0:
            motif_pos = positions.unsqueeze(-1) * self.motif_freqs.unsqueeze(
                0
            ).unsqueeze(0)
            motif_encoding = torch.sin(motif_pos)  # (batch, seq_len, motif_dims)

            # Add to appropriate dimensions
            motif_dims = min(motif_encoding.size(-1), d_model // 8)
            if motif_dims > 0:
                start_dim = d_model - motif_dims
                pe[:, :, start_dim:] += motif_encoding[:, :, :motif_dims] * 0.05

        return pe


class CrossAttentionPositionBridge(nn.Module):
    """
    Cross-attention bridge between local (byte-level) and global (patch-level) representations.

    This implements the "U-shape information flow" described in the BLT technical guide,
    allowing position information to flow between byte and patch representations.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_patch_size: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_patch_size = max_patch_size

        # Cross-attention for byte -> patch
        self.byte_to_patch_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention for patch -> byte
        self.patch_to_byte_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Position encoding for relative positions within patches
        self.relative_pos_encoder = nn.Embedding(max_patch_size * 2 + 1, d_model)

        # Layer norms
        self.norm_byte = nn.LayerNorm(d_model)
        self.norm_patch = nn.LayerNorm(d_model)

    def encode_byte_to_patch(
        self,
        byte_repr: torch.Tensor,
        patch_boundaries: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode byte-level representations into patch-level representations.

        Args:
            byte_repr: (batch, byte_seq_len, d_model) byte-level representations
            patch_boundaries: (batch, byte_seq_len) patch boundary indicators
            patch_ids: (batch, byte_seq_len) patch IDs for each byte position

        Returns:
            patch_repr: (batch, num_patches, d_model) patch-level representations
        """
        batch_size, byte_seq_len, d_model = byte_repr.shape
        device = byte_repr.device

        # Generate patch representations by grouping bytes
        patch_reprs = []
        max_patches = 0

        for b in range(batch_size):
            boundaries = patch_boundaries[b]
            boundary_indices = torch.nonzero(boundaries, as_tuple=True)[0]

            # Add start and end
            boundary_indices = torch.cat(
                [
                    torch.tensor([0], device=device),
                    boundary_indices,
                    torch.tensor([byte_seq_len], device=device),
                ]
            )
            boundary_indices = torch.unique(boundary_indices, sorted=True)

            # Create patch representations
            batch_patches = []
            for i in range(len(boundary_indices) - 1):
                start_pos = boundary_indices[i]
                end_pos = boundary_indices[i + 1]

                # Get byte representations for this patch
                patch_bytes = byte_repr[b, start_pos:end_pos]  # (patch_len, d_model)

                # Use cross-attention to create patch representation
                # Query: mean of patch bytes, Key/Value: all patch bytes
                patch_query = patch_bytes.mean(dim=0, keepdim=True)  # (1, d_model)

                patch_repr, _ = self.byte_to_patch_attn(
                    query=patch_query.unsqueeze(0),  # (1, 1, d_model)
                    key=patch_bytes.unsqueeze(0),  # (1, patch_len, d_model)
                    value=patch_bytes.unsqueeze(0),  # (1, patch_len, d_model)
                )

                batch_patches.append(patch_repr.squeeze(0))  # (1, d_model)

            if len(batch_patches) > 0:
                patch_reprs.append(
                    torch.cat(batch_patches, dim=0)
                )  # (num_patches, d_model)
                max_patches = max(max_patches, len(batch_patches))
            else:
                # Handle edge case: no patches
                patch_reprs.append(byte_repr[b : b + 1, 0])  # (1, d_model)
                max_patches = max(max_patches, 1)

        # Pad to same length
        padded_patches = []
        for patches in patch_reprs:
            num_patches = patches.size(0)
            if num_patches < max_patches:
                pad_size = max_patches - num_patches
                padding = torch.zeros(
                    pad_size, d_model, device=device, dtype=patches.dtype
                )
                patches = torch.cat([patches, padding], dim=0)
            padded_patches.append(patches)

        return torch.stack(padded_patches, dim=0)  # (batch, max_patches, d_model)

    def decode_patch_to_byte(
        self,
        patch_repr: torch.Tensor,
        target_byte_len: int,
        patch_boundaries: torch.Tensor,
        original_byte_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode patch-level representations back to byte-level representations.

        Args:
            patch_repr: (batch, num_patches, d_model) patch-level representations
            target_byte_len: Target length for byte sequence
            patch_boundaries: (batch, target_byte_len) patch boundaries
            original_byte_repr: (batch, target_byte_len, d_model) original byte representations

        Returns:
            byte_repr: (batch, target_byte_len, d_model) decoded byte representations
        """
        batch_size, num_patches, d_model = patch_repr.shape
        device = patch_repr.device

        # Create target byte representations
        byte_reprs = torch.zeros(batch_size, target_byte_len, d_model, device=device)

        for b in range(batch_size):
            boundaries = patch_boundaries[b]
            boundary_indices = torch.nonzero(boundaries, as_tuple=True)[0]

            # Add start and end
            boundary_indices = torch.cat(
                [
                    torch.tensor([0], device=device),
                    boundary_indices,
                    torch.tensor([target_byte_len], device=device),
                ]
            )
            boundary_indices = torch.unique(boundary_indices, sorted=True)

            # Decode each patch
            patch_idx = 0
            for i in range(len(boundary_indices) - 1):
                if patch_idx >= num_patches:
                    break

                start_pos = boundary_indices[i]
                end_pos = boundary_indices[i + 1]
                patch_len = end_pos - start_pos

                # Get patch representation
                patch_vec = patch_repr[b, patch_idx]  # (d_model,)

                # Create queries for each byte position in patch
                byte_queries = torch.zeros(patch_len, d_model, device=device)
                if original_byte_repr is not None:
                    byte_queries = original_byte_repr[b, start_pos:end_pos]
                else:
                    # Use positional encoding as queries
                    for j in range(patch_len):
                        pos_id = min(j, self.max_patch_size - 1)
                        byte_queries[j] = self.relative_pos_encoder.weight[pos_id]

                # Cross-attention from patch to bytes
                byte_output, _ = self.patch_to_byte_attn(
                    query=byte_queries.unsqueeze(0),  # (1, patch_len, d_model)
                    key=patch_vec.unsqueeze(0).unsqueeze(0),  # (1, 1, d_model)
                    value=patch_vec.unsqueeze(0).unsqueeze(0),  # (1, 1, d_model)
                )

                byte_reprs[b, start_pos:end_pos] = byte_output.squeeze(0)
                patch_idx += 1

        return byte_reprs


class BLTPositionManager(nn.Module):
    """
    Complete position embedding manager for BLT-style token merging.

    This integrates all components needed for proper position handling:
    1. Segment-aware position encoding
    2. Cross-attention bridges
    3. Position tracking across merging operations
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 32768,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_patch_size: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Core components
        self.position_encoder = SegmentAwarePositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            use_learned_encoding=True,
        )

        self.cross_attention_bridge = CrossAttentionPositionBridge(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_patch_size=max_patch_size,
        )

        # Position tracking for merging operations
        self.register_buffer("position_map", torch.arange(max_len))

    def encode_positions(
        self,
        hidden_states: torch.Tensor,
        patch_boundaries: torch.Tensor | None = None,
        original_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Add position encodings to hidden states.

        Args:
            hidden_states: (batch, seq_len, d_model)
            patch_boundaries: (batch, seq_len) patch boundary indicators
            original_positions: (batch, seq_len) original positions before merging

        Returns:
            Position-encoded hidden states
        """
        seq_len = hidden_states.size(1)

        # Generate position embeddings
        pos_embeddings = self.position_encoder(
            seq_len=seq_len,
            patch_boundaries=patch_boundaries,
            original_positions=original_positions,
        )

        # Add to hidden states
        return hidden_states + pos_embeddings  # type: ignore[no-any-return]

    def create_patch_representations(
        self,
        byte_hidden_states: torch.Tensor,
        patch_boundaries: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Create patch representations from byte-level hidden states.

        Args:
            byte_hidden_states: (batch, byte_seq_len, d_model)
            patch_boundaries: (batch, byte_seq_len) patch boundaries

        Returns:
            patch_representations: (batch, num_patches, d_model)
            position_info: Dictionary with position tracking information
        """
        # Encode byte-to-patch
        patch_repr = self.cross_attention_bridge.encode_byte_to_patch(
            byte_repr=byte_hidden_states,
            patch_boundaries=patch_boundaries,
        )

        # Track position information
        position_info = {
            "original_length": byte_hidden_states.size(1),
            "patch_length": patch_repr.size(1),
            "patch_boundaries": patch_boundaries,
        }

        return patch_repr, position_info

    def reconstruct_byte_representations(
        self,
        patch_hidden_states: torch.Tensor,
        position_info: dict[str, Any],
        original_byte_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Reconstruct byte-level representations from patch-level representations.

        Args:
            patch_hidden_states: (batch, num_patches, d_model)
            position_info: Position tracking information from create_patch_representations
            original_byte_repr: Original byte representations for cross-attention

        Returns:
            byte_representations: (batch, byte_seq_len, d_model)
        """
        target_byte_len = position_info["original_length"]
        patch_boundaries = position_info["patch_boundaries"]

        # Decode patch-to-byte
        byte_repr = self.cross_attention_bridge.decode_patch_to_byte(
            patch_repr=patch_hidden_states,
            target_byte_len=target_byte_len,
            patch_boundaries=patch_boundaries,
            original_byte_repr=original_byte_repr,
        )

        return byte_repr
