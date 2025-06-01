"""
Attention Analysis Module for Hyena-GLT Models

This module provides specialized tools for analyzing attention patterns in Hyena-GLT models,
with focus on genomic sequence understanding and pattern discovery.
"""

import logging
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyenaAttentionAnalyzer:
    """Specialized attention analyzer for Hyena operators."""

    def __init__(self, model: nn.Module, device: str = "auto"):
        self.model = model
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger = logging.getLogger(__name__)

        # Move model to device
        self.model = self.model.to(self.device)

    def extract_hyena_patterns(
        self, input_ids: torch.Tensor, layer_indices: list[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Extract attention-like patterns from Hyena convolution layers."""
        self.model.eval()
        patterns = {}

        # Hook function to capture intermediate activations
        activations = {}

        def hook_fn(name: str) -> Callable:
            def hook(module: torch.nn.Module, input: Any, output: Any) -> None:
                if hasattr(output, "shape") and len(output.shape) >= 3:
                    # Store activation patterns
                    activations[name] = output.detach()

            return hook

        # Register hooks on Hyena layers
        hooks = []
        for name, module in self.model.named_modules():
            if "hyena" in name.lower() and hasattr(module, "conv"):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            _ = self.model(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process activations to create attention-like patterns
        for name, activation in activations.items():
            if len(activation.shape) == 3:  # [batch, seq_len, hidden_size]
                # Compute self-similarity matrix as attention proxy
                activation_norm = F.normalize(activation, dim=-1)
                similarity = torch.matmul(
                    activation_norm, activation_norm.transpose(-2, -1)
                )
                patterns[name] = similarity

        return patterns

    def analyze_positional_patterns(
        self, attention_patterns: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, float]]:
        """Analyze positional attention patterns specific to genomic sequences."""
        analysis = {}

        for layer_name, pattern in attention_patterns.items():
            # pattern shape: [batch, seq_len, seq_len]
            batch_size, seq_len, _ = pattern.shape

            # Local attention (neighboring positions)
            local_attention = 0.0
            local_count = 0
            max_offset = min(5, seq_len // 2)  # Adaptive offset based on sequence length
            
            for offset in range(1, max_offset + 1):
                if offset < seq_len:
                    diagonal_pos = torch.diagonal(
                        pattern, offset=offset, dim1=-2, dim2=-1
                    )
                    diagonal_neg = torch.diagonal(
                        pattern, offset=-offset, dim1=-2, dim2=-1
                    )
                    if len(diagonal_pos) > 0:
                        local_attention += diagonal_pos.mean().item()
                        local_count += 1
                    if len(diagonal_neg) > 0:
                        local_attention += diagonal_neg.mean().item()
                        local_count += 1
            
            if local_count > 0:
                local_attention /= local_count
            else:
                local_attention = 0.0

            # Long-range attention (distant positions)
            # Adaptive mask size based on sequence length
            mask_radius = min(10, seq_len // 4)  # Use smaller mask for short sequences
            long_range_mask = torch.ones_like(pattern[0])
            
            for i in range(seq_len):
                # Mask out local region (±mask_radius positions)
                start_mask = max(0, i - mask_radius)
                end_mask = min(seq_len, i + mask_radius + 1)
                long_range_mask[i, start_mask:end_mask] = 0

            # Check if mask has any elements left
            mask_sum = long_range_mask.sum()
            if mask_sum > 0:
                long_range_attention_tensor = (
                    pattern * long_range_mask.unsqueeze(0)
                ).sum() / mask_sum
                long_range_attention = float(long_range_attention_tensor.item())
            else:
                # If no long-range positions exist, set to 0
                long_range_attention = 0.0

            # Periodicity detection (for genomic repeats)
            periodicity_scores = []
            for period in [3, 6, 9, 12]:  # Common genomic periods
                if period < seq_len // 2:
                    period_pattern = 0.0
                    count = 0
                    for i in range(seq_len - period):
                        if i + period < seq_len:
                            period_value = pattern[:, i, i + period].mean().item()
                            if not np.isnan(period_value) and not np.isinf(period_value):
                                period_pattern += period_value
                                count += 1
                    if count > 0:
                        periodicity_scores.append(period_pattern / count)

            avg_periodicity = np.mean(periodicity_scores) if periodicity_scores else 0.0

            analysis[layer_name] = {
                "local_attention": float(local_attention),
                "long_range_attention": float(long_range_attention),
                "periodicity": float(avg_periodicity),
                "sparsity": float((pattern < 0.1).float().mean().item()),
                "max_attention": float(pattern.max().item()),
            }

        return analysis

    def visualize_genomic_attention(
        self,
        attention_pattern: torch.Tensor,
        sequence: str | None = None,
        save_path: str | None = None,
        title: str = "Genomic Attention Pattern",
    ) -> None:
        """Visualize attention patterns with genomic sequence annotations."""
        # Take first batch item
        pattern = attention_pattern[0].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Main attention heatmap
        im1 = axes[0].imshow(pattern, cmap="Blues", aspect="auto")
        axes[0].set_title(f"{title} - Full Pattern")
        axes[0].set_xlabel("Key Position")
        axes[0].set_ylabel("Query Position")
        plt.colorbar(im1, ax=axes[0])

        # Diagonal pattern (local attention)
        diagonal_region = 20  # Show ±20 positions around diagonal
        seq_len = pattern.shape[0]
        local_pattern = np.zeros((diagonal_region * 2 + 1, seq_len))

        for i in range(seq_len):
            for offset in range(-diagonal_region, diagonal_region + 1):
                j = i + offset
                if 0 <= j < seq_len:
                    local_pattern[offset + diagonal_region, i] = pattern[i, j]

        im2 = axes[1].imshow(local_pattern, cmap="Blues", aspect="auto")
        axes[1].set_title(f"{title} - Local Pattern")
        axes[1].set_xlabel("Position")
        axes[1].set_ylabel("Offset from Diagonal")
        axes[1].set_yticks(range(0, diagonal_region * 2 + 1, 5))
        axes[1].set_yticklabels(range(-diagonal_region, diagonal_region + 1, 5))
        plt.colorbar(im2, ax=axes[1])

        # Add sequence annotations if provided
        if sequence:
            # Mark important genomic features
            self._annotate_genomic_features(axes[0], sequence)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def _annotate_genomic_features(self, ax: Any, sequence: str) -> None:
        """Add genomic feature annotations to attention plot."""
        # Look for common genomic patterns
        features = []

        # Start codons
        for i in range(len(sequence) - 2):
            if sequence[i : i + 3] == "ATG":
                features.append(("Start Codon", i, i + 3, "red"))

        # Stop codons
        stop_codons = ["TAA", "TAG", "TGA"]
        for stop in stop_codons:
            for i in range(len(sequence) - 2):
                if sequence[i : i + 3] == stop:
                    features.append(("Stop Codon", i, i + 3, "orange"))

        # TATA boxes (promoter elements)
        for i in range(len(sequence) - 3):
            if sequence[i : i + 4] == "TATA":
                features.append(("TATA Box", i, i + 4, "green"))

        # Add feature markers
        for _feature_name, start, _end, color in features[:10]:  # Limit annotations
            ax.axhline(y=start, color=color, alpha=0.7, linewidth=1)
            ax.axvline(x=start, color=color, alpha=0.7, linewidth=1)

    def detect_attention_motifs(
        self,
        attention_patterns: dict[str, torch.Tensor],
        sequences: list[str],
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Detect recurring attention motifs in genomic sequences."""
        motifs = []

        for layer_name, patterns in attention_patterns.items():
            # Find high-attention regions
            for _batch_idx, (pattern, sequence) in enumerate(
                zip(patterns, sequences, strict=False)
            ):

                # Find positions with high attention
                attention_scores = pattern.sum(dim=-1)  # Sum over keys
                high_positions = (attention_scores > threshold).nonzero().squeeze()

                if len(high_positions.shape) == 0:
                    high_positions = [high_positions.item()]
                else:
                    high_positions = high_positions.tolist()

                # Extract subsequences around high-attention positions
                for pos in high_positions:
                    if isinstance(pos, torch.Tensor):
                        pos = pos.item()

                    # Extract context around position
                    start = max(0, pos - 5)
                    end = min(len(sequence), pos + 6)

                    if end - start >= 6:  # Minimum motif length
                        motif_seq = sequence[start:end]
                        attention_score = attention_scores[pos].item()

                        motifs.append(
                            {
                                "sequence": motif_seq,
                                "position": pos,
                                "layer": layer_name,
                                "attention_score": attention_score,
                                "context_start": start,
                                "context_end": end,
                            }
                        )

        # Sort by attention score
        motifs.sort(key=lambda x: x["attention_score"], reverse=True)

        return motifs
