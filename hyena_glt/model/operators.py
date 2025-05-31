"""Hyena operators adapted for genomic sequences with dynamic token merging."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..config import HyenaGLTConfig


class DynamicConvolution(nn.Module):
    """Dynamic convolution that handles variable sequence lengths from token merging."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.groups = groups
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Use depthwise convolution for efficiency
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
            padding=kernel_size - 1,  # causal padding
        )

        # Segment boundary detection for handling merged tokens
        self.boundary_detector = nn.Linear(d_model, 1)
        self.boundary_threshold = 0.5

    def forward(
        self, x: torch.Tensor, segment_boundaries: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            segment_boundaries: (batch, seq_len) - binary mask indicating segment boundaries
        """
        batch_size, seq_len, d_model = x.shape

        # Detect segment boundaries if not provided
        if segment_boundaries is None:
            boundary_scores = F.sigmoid(
                self.boundary_detector(x)
            )  # (batch, seq_len, 1)
            segment_boundaries = (
                boundary_scores.squeeze(-1) > self.boundary_threshold
            ).float()

        # Apply convolution
        x_conv = rearrange(x, "b l d -> b d l")
        x_conv = self.conv(x_conv)
        x_conv = x_conv[..., :seq_len]  # remove causal padding
        x_conv = rearrange(x_conv, "b d l -> b l d")

        # Apply segment-aware gating
        if segment_boundaries is not None:
            # Create segment mask that prevents information flow across boundaries
            segment_mask = self._create_segment_mask(
                segment_boundaries, self.kernel_size
            )
            x_conv = x_conv * segment_mask.unsqueeze(-1)

        if self.dropout is not None:
            x_conv = self.dropout(x_conv)

        # Explicit type annotation to fix MyPy "returning Any" error
        result_conv: torch.Tensor = x_conv
        return result_conv

    def _create_segment_mask(
        self, boundaries: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        """Create mask that prevents convolution across segment boundaries."""
        _, seq_len = boundaries.shape
        mask: torch.Tensor = torch.ones_like(boundaries)

        # For each boundary, mask the kernel_size positions around it
        boundary_positions = torch.nonzero(boundaries, as_tuple=True)

        for b, pos in zip(*boundary_positions, strict=False):
            start = max(0, pos - kernel_size // 2)
            end = min(seq_len, pos + kernel_size // 2 + 1)
            mask[b, start:end] *= 0.5  # Reduce but don't completely eliminate

        # Explicit type annotation to fix MyPy "returning Any" error
        result_mask: torch.Tensor = mask
        return result_mask


class HyenaOperator(nn.Module):
    """
    Hyena operator adapted for genomic sequences with support for dynamic token merging.
    Based on Savanna's implementation but modified for variable-length sequences.
    """

    def __init__(
        self,
        config: HyenaGLTConfig,
        d_model: int,
        l_max: int = 32768,
        order: int = 2,
        filter_order: int = 64,
        dropout: float = 0.1,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        self.filter_order = filter_order
        self.layer_idx = layer_idx

        # Input projections
        self.in_proj = nn.Linear(d_model, d_model * (order + 1), bias=config.use_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=config.use_bias)

        # Hyena filter (implicit parametrization)
        self.filter_fn = HyenaFilter(
            d_model=d_model,
            order=order,
            filter_order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=dropout,
        )

        # Dynamic convolution for short-range dependencies
        self.short_conv = DynamicConvolution(
            d_model=d_model,
            kernel_size=config.hyena_short_filter_size,
            groups=d_model,
            dropout=dropout,
        )

        # Gating mechanism
        self.glu: nn.GLU | None
        if config.use_glu:
            self.glu = nn.GLU(dim=-1)
        else:
            self.glu = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm for residual connection
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        segment_boundaries: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            segment_boundaries: (batch, seq_len) - binary mask for merged token boundaries
            attention_mask: (batch, seq_len) - attention mask
        """
        batch_size, seq_len, d_model = x.shape

        # Input projection
        projected = self.in_proj(x)  # (batch, seq_len, d_model * (order + 1))

        # Split into multiple streams
        splits = torch.split(projected, d_model, dim=-1)
        u = splits[0]  # Base signal

        # Apply short-range convolution to base signal
        u_conv = self.short_conv(u, segment_boundaries)

        # Generate long-range filters
        filter_coeffs = self.filter_fn(seq_len)

        # Apply Hyena recurrence with dynamic segments
        y = self._apply_hyena_recurrence(
            list(splits[1:]), filter_coeffs, segment_boundaries, attention_mask
        )

        # Combine short and long range
        output = u_conv * y

        # Apply GLU if configured
        if self.glu is not None:
            glu_input = torch.cat([output, u], dim=-1)
            output = self.glu(glu_input)

        # Output projection
        output = self.out_proj(output)

        # Apply dropout
        output = self.dropout(output)

        # Residual connection with layer norm
        output = self.norm(output + x)

        # Explicit type annotation to fix MyPy "returning Any" error
        result_output: torch.Tensor = output
        return result_output

    def _apply_hyena_recurrence(
        self,
        x_streams: list[torch.Tensor],
        filter_coeffs: torch.Tensor,
        segment_boundaries: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply Hyena recurrence with segment awareness."""

        # For genomic sequences, we use a simplified recurrence
        # that respects segment boundaries from token merging

        x1, x2 = x_streams[0], x_streams[1] if len(x_streams) > 1 else x_streams[0]

        # Apply convolution with learned filters
        x_filtered: torch.Tensor = self._segment_aware_convolution(
            x1, filter_coeffs, segment_boundaries
        )

        # Element-wise gating
        output = x2 * x_filtered

        # Apply attention mask if provided
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)

        # Explicit type annotation to fix MyPy "returning Any" error
        result_output: torch.Tensor = output
        return result_output

    def _segment_aware_convolution(
        self,
        x: torch.Tensor,
        filter_coeffs: torch.Tensor,
        segment_boundaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply convolution while respecting segment boundaries."""

        _, seq_len, d_model = x.shape

        # Reshape for convolution
        x_conv = rearrange(x, "b l d -> b d l")

        # Apply 1D convolution with adaptive filter length
        # Limit filter to reasonable size relative to sequence length
        max_filter_len = min(seq_len, 64)  # Cap at 64 for stability
        filter_len = min(filter_coeffs.size(-1), max_filter_len)
        filter_truncated = filter_coeffs[..., :filter_len]

        # DEBUG: Print shapes
        # print(f"DEBUG _segment_aware_convolution: seq_len={seq_len}, d_model={d_model}")
        # print(f"DEBUG filter_coeffs.shape: {filter_coeffs.shape}")
        # print(f"DEBUG filter_len: {filter_len}, max_filter_len: {max_filter_len}")
        # print(f"DEBUG filter_truncated.shape: {filter_truncated.shape}")
        # print(f"DEBUG x_conv.shape: {x_conv.shape}")

        # CRITICAL FIX: Handle multi-channel filter correctly
        # filter_truncated shape: (d_model, filter_len)
        # We need to create a grouped convolution where each input channel has its own filter

        if filter_truncated.dim() == 2 and filter_truncated.size(0) == d_model:
            # Multi-channel filter: (d_model, filter_len) -> (d_model, 1, filter_len)
            conv_filter = filter_truncated.unsqueeze(1)  # (d_model, 1, filter_len)
        else:
            # Single filter for all channels: expand to (d_model, 1, filter_len)
            conv_filter = filter_truncated.view(1, 1, -1).expand(d_model, 1, -1)

        # print(f"DEBUG conv_filter.shape: {conv_filter.shape}")

        # Pad for causal convolution
        padding = filter_len - 1
        x_padded = F.pad(x_conv, (padding, 0))

        # Perform grouped convolution (each input channel convolved with its own filter)
        output = F.conv1d(
            x_padded,
            conv_filter,
            groups=d_model,  # Each input channel uses its own filter
            padding=0,
        )

        # Truncate to original length
        output = output[..., :seq_len]

        # Apply segment boundaries if provided
        if segment_boundaries is not None:
            segment_mask = self._create_causal_segment_mask(
                segment_boundaries, filter_len
            )
            output = output * segment_mask.unsqueeze(1)

        # Reshape back
        output_reshaped: torch.Tensor = rearrange(output, "b d l -> b l d")

        return output_reshaped

    def _create_causal_segment_mask(
        self, boundaries: torch.Tensor, filter_len: int
    ) -> torch.Tensor:
        """Create causal mask that respects segment boundaries."""
        batch_size, seq_len = boundaries.shape

        # Create base causal mask
        mask = torch.ones(batch_size, seq_len, device=boundaries.device)

        # Identify boundary positions
        boundary_positions = torch.nonzero(boundaries, as_tuple=True)

        # For each boundary, create a causal break
        for b, pos in zip(*boundary_positions, strict=False):
            # Mask positions that would look across the boundary
            start_mask = max(0, pos - filter_len + 1)
            mask[b, start_mask:pos] *= torch.linspace(
                0.1, 1.0, pos - start_mask, device=boundaries.device
            )

        return mask


class HyenaFilter(nn.Module):
    """
    Implicit filter parameterization for Hyena operator.
    Generates position-dependent filters for long-range convolutions.
    """

    def __init__(
        self,
        d_model: int,
        order: int = 2,
        filter_order: int = 64,
        seq_len: int = 32768,
        channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.order = order
        self.filter_order = filter_order
        self.seq_len = seq_len
        self.channels = channels

        # Learnable frequency components
        self.freqs = nn.Parameter(torch.randn(channels, filter_order // 2))
        self.phases = nn.Parameter(torch.randn(channels, filter_order // 2))
        self.amplitudes = nn.Parameter(torch.randn(channels, filter_order // 2))

        # Decay factors for genomic sequences
        self.decay = nn.Parameter(torch.randn(channels, filter_order // 2))

        # Positional encoding for genomic context
        self.pos_encoder = GenomicPositionalEncoding(filter_order, seq_len)

        # Output projection
        self.output_proj = nn.Linear(filter_order, d_model)

        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize filter parameters."""
        # Initialize frequencies for biological-relevant scales
        with torch.no_grad():
            # Genomic sequences have patterns at multiple scales
            # Initialize to capture periodicity at codon (3), amino acid (20), and domain scales
            self.freqs.uniform_(-math.pi, math.pi)
            self.phases.uniform_(-math.pi, math.pi)
            self.amplitudes.normal_(0, 0.1)
            self.decay.uniform_(-1, 0)  # Negative for decay

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate filter coefficients for given sequence length.

        Args:
            seq_len: Length of sequence to generate filter for

        Returns:
            filter_coeffs: (d_model, seq_len) filter coefficients
        """
        # Generate time indices
        t = torch.arange(seq_len, device=self.freqs.device, dtype=self.freqs.dtype)
        t = t.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

        # Generate sinusoidal components
        freqs = self.freqs.unsqueeze(1)  # (channels, 1, filter_order//2)
        phases = self.phases.unsqueeze(1)
        amplitudes = self.amplitudes.unsqueeze(1)
        decay = self.decay.unsqueeze(1)

        # Compute sinusoidal components with decay
        sin_components = (
            amplitudes * torch.sin(freqs * t + phases) * torch.exp(decay * t)
        )
        cos_components = (
            amplitudes * torch.cos(freqs * t + phases) * torch.exp(decay * t)
        )

        # Combine components
        filter_components = torch.cat(
            [sin_components, cos_components], dim=-1
        )  # (channels, seq_len, filter_order)

        # Add positional encoding for genomic context
        pos_encoding = self.pos_encoder(seq_len)  # (seq_len, filter_order)
        filter_components = filter_components + pos_encoding.unsqueeze(0)

        # Project to d_model dimensions
        filter_coeffs = self.output_proj(
            filter_components
        )  # (channels, seq_len, d_model)

        # Apply dropout
        filter_coeffs = self.dropout(filter_coeffs)

        # Average across channels and transpose
        filter_output: torch.Tensor = filter_coeffs.mean(dim=0).transpose(
            0, 1
        )  # (d_model, seq_len)

        return filter_output


class GenomicPositionalEncoding(nn.Module):
    """
    Positional encoding specialized for genomic sequences.
    Incorporates biological priors like codon structure and reading frames.
    """

    def __init__(self, d_model: int, max_len: int = 32768):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add genomic-specific encodings
        self._add_genomic_encodings(pe, max_len, d_model)

        self.register_buffer("pe", pe)

    def _add_genomic_encodings(
        self, pe: torch.Tensor, max_len: int, d_model: int
    ) -> None:
        """Add genomic-specific positional encodings."""
        # Codon-based encoding (period 3)
        codon_positions = torch.arange(max_len) % 3
        codon_encoding = torch.sin(2 * math.pi * codon_positions / 3)

        # Reading frame encoding
        frame_encoding = torch.sin(2 * math.pi * torch.arange(max_len) / 6)

        # Add to a subset of dimensions
        if d_model >= 4:
            pe[:, -4] += codon_encoding * 0.1
            pe[:, -3] += frame_encoding * 0.1

        # Add periodic patterns for common genomic motifs
        for period in [8, 10, 21]:  # Common motif lengths
            if d_model >= 6:
                # Use modulo to ensure we stay within bounds
                dim_idx = (d_model - 1) - (period % (d_model - 2))
                if dim_idx >= 0 and dim_idx < d_model:
                    motif_encoding = torch.sin(
                        2 * math.pi * torch.arange(max_len) / period
                    )
                    pe[:, dim_idx] += motif_encoding * 0.05

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: Length of sequence

        Returns:
            Positional encoding of shape (seq_len, d_model)
        """
        pe_slice: torch.Tensor = self.pe[:seq_len]
        return pe_slice
