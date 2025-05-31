"""
Interpretability Tools for Hyena-GLT Models

This module provides tools for understanding and interpreting Hyena-GLT model behavior,
including attention analysis, gradient-based methods, and genomic-specific interpretability.

Author: Hyena-GLT Development Team
Version: 1.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import pandas as pd

# Import genomic-specific utilities
from hyena_glt.data.tokenizer import DNATokenizer, RNATokenizer, ProteinTokenizer
# Note: Using matplotlib directly for visualizations
# from examples.utils.visualization_utils import plot_attention_heatmap


@dataclass
class InterpretabilityConfig:
    """Configuration for interpretability analysis."""
    
    # Attention analysis
    analyze_attention: bool = True
    attention_layers: Optional[List[int]] = None  # Which layers to analyze
    attention_heads: Optional[List[int]] = None   # Which heads to analyze
    
    # Gradient-based methods
    analyze_gradients: bool = True
    gradient_methods: List[str] = None  # ['vanilla', 'integrated', 'guided']
    
    # Genomic-specific analysis
    motif_analysis: bool = True
    sequence_importance: bool = True
    
    # Output settings
    save_plots: bool = True
    output_dir: str = "./interpretability_outputs"
    
    def __post_init__(self):
        if self.gradient_methods is None:
            self.gradient_methods = ['vanilla', 'integrated', 'guided']


class AttentionAnalyzer:
    """Analyze attention patterns in Hyena-GLT models."""
    
    def __init__(self, model: nn.Module, tokenizer: Optional[Any] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    def extract_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            layers: Which layers to extract (None for all)
            heads: Which heads to extract (None for all)
        
        Returns:
            Dictionary of attention weights
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass with attention output
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                self.logger.warning("Model does not output attention weights")
                return {}
            
            attention_weights = {}
            
            for layer_idx, layer_attention in enumerate(outputs.attentions):
                if layers is None or layer_idx in layers:
                    if heads is None:
                        attention_weights[f'layer_{layer_idx}'] = layer_attention
                    else:
                        selected_heads = layer_attention[:, heads, :, :]
                        attention_weights[f'layer_{layer_idx}'] = selected_heads
            
            return attention_weights
    
    def compute_attention_statistics(
        self,
        attention_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistical measures of attention patterns."""
        stats = {}
        
        for layer_name, attention in attention_weights.items():
            # attention shape: [batch, heads, seq_len, seq_len]
            
            # Entropy (higher = more distributed attention)
            attention_probs = F.softmax(attention, dim=-1)
            entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-9), dim=-1)
            mean_entropy = entropy.mean().item()
            
            # Sparsity (fraction of weights below threshold)
            sparsity = (attention < 0.1).float().mean().item()
            
            # Diagonal dominance (how much attention focuses on nearby positions)
            seq_len = attention.shape[-1]
            diagonal_weights = torch.diagonal(attention, dim1=-2, dim2=-1).mean().item()
            
            # Maximum attention value
            max_attention = attention.max().item()
            
            stats[layer_name] = {
                'entropy': mean_entropy,
                'sparsity': sparsity,
                'diagonal_dominance': diagonal_weights,
                'max_attention': max_attention
            }
        
        return stats
    
    def visualize_attention_patterns(
        self,
        attention_weights: Dict[str, torch.Tensor],
        sequences: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize attention patterns as heatmaps."""
        
        for layer_name, attention in attention_weights.items():
            # Take first batch item and average over heads
            attention_matrix = attention[0].mean(dim=0).cpu().numpy()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                attention_matrix,
                cmap='Blues',
                cbar=True,
                square=True,
                xticklabels=False,
                yticklabels=False
            )
            plt.title(f'Attention Pattern - {layer_name}')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            
            if save_path:
                plt.savefig(f"{save_path}/attention_{layer_name}.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def find_important_positions(
        self,
        attention_weights: Dict[str, torch.Tensor],
        threshold: float = 0.8
    ) -> Dict[str, List[int]]:
        """Find positions that receive high attention."""
        important_positions = {}
        
        for layer_name, attention in attention_weights.items():
            # Average over batch and heads
            avg_attention = attention.mean(dim=(0, 1))  # [seq_len, seq_len]
            
            # Sum attention received by each position
            attention_received = avg_attention.sum(dim=0)
            
            # Find positions above threshold
            important_pos = (attention_received > threshold).nonzero().squeeze().tolist()
            if isinstance(important_pos, int):
                important_pos = [important_pos]
            
            important_positions[layer_name] = important_pos
        
        return important_positions


class GradientAnalyzer:
    """Analyze gradients for feature importance."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def vanilla_gradients(
        self,
        input_ids: torch.Tensor,
        target_class: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute vanilla gradients."""
        input_ids.requires_grad_(True)
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Get gradient with respect to target class
        target_logit = logits[0, target_class]
        target_logit.backward()
        
        gradients = input_ids.grad.clone()
        input_ids.grad.zero_()
        
        return gradients
    
    def integrated_gradients(
        self,
        input_ids: torch.Tensor,
        target_class: int,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute integrated gradients."""
        if baseline is None:
            baseline = torch.zeros_like(input_ids)
        
        # Create interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(input_ids.device)
        gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_ids - baseline)
            interpolated.requires_grad_(True)
            
            outputs = self.model(interpolated, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            target_logit = logits[0, target_class]
            grad = torch.autograd.grad(target_logit, interpolated)[0]
            gradients.append(grad)
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_grads = (input_ids - baseline) * avg_gradients
        
        return integrated_grads
    
    def guided_backpropagation(
        self,
        input_ids: torch.Tensor,
        target_class: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute guided backpropagation gradients."""
        # This is a simplified version - full implementation would modify ReLU behavior
        return self.vanilla_gradients(input_ids, target_class, attention_mask)


class GenomicMotifAnalyzer:
    """Analyze genomic motifs and patterns."""
    
    def __init__(self, tokenizer: Optional[Any] = None):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    def extract_important_subsequences(
        self,
        sequence: str,
        importance_scores: torch.Tensor,
        window_size: int = 6,
        threshold: float = 0.5
    ) -> List[Tuple[str, float, int]]:
        """Extract important subsequences based on importance scores."""
        important_subseqs = []
        
        for i in range(len(sequence) - window_size + 1):
            window_score = importance_scores[i:i+window_size].mean().item()
            
            if window_score > threshold:
                subseq = sequence[i:i+window_size]
                important_subseqs.append((subseq, window_score, i))
        
        # Sort by importance score
        important_subseqs.sort(key=lambda x: x[1], reverse=True)
        
        return important_subseqs
    
    def find_consensus_motifs(
        self,
        sequences: List[str],
        importance_scores: List[torch.Tensor],
        motif_length: int = 6,
        min_frequency: int = 3
    ) -> List[Dict[str, Any]]:
        """Find consensus motifs across multiple sequences."""
        motif_counts = {}
        
        for seq, scores in zip(sequences, importance_scores):
            important_subseqs = self.extract_important_subsequences(
                seq, scores, motif_length
            )
            
            for subseq, score, pos in important_subseqs:
                if subseq not in motif_counts:
                    motif_counts[subseq] = {'count': 0, 'scores': [], 'positions': []}
                
                motif_counts[subseq]['count'] += 1
                motif_counts[subseq]['scores'].append(score)
                motif_counts[subseq]['positions'].append(pos)
        
        # Filter by frequency and create consensus motifs
        consensus_motifs = []
        for motif, data in motif_counts.items():
            if data['count'] >= min_frequency:
                consensus_motifs.append({
                    'sequence': motif,
                    'frequency': data['count'],
                    'avg_importance': np.mean(data['scores']),
                    'positions': data['positions']
                })
        
        # Sort by frequency and importance
        consensus_motifs.sort(key=lambda x: (x['frequency'], x['avg_importance']), reverse=True)
        
        return consensus_motifs


class ModelInterpreter:
    """Main class for model interpretability analysis."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        config: Optional[InterpretabilityConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InterpretabilityConfig()
        
        # Initialize analyzers
        self.attention_analyzer = AttentionAnalyzer(model, tokenizer)
        self.gradient_analyzer = GradientAnalyzer(model)
        self.motif_analyzer = GenomicMotifAnalyzer(tokenizer)
        
        # Setup output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_sequence(
        self,
        sequence: str,
        target_class: Optional[int] = None,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive analysis of a single sequence."""
        self.logger.info(f"Analyzing sequence of length {len(sequence)}")
        
        # Tokenize sequence
        if self.tokenizer:
            tokens = self.tokenizer.encode(sequence, padding=True, truncation=True)
            input_ids = torch.tensor([tokens['input_ids']])
            attention_mask = torch.tensor([tokens['attention_mask']])
        else:
            # Fallback: assume sequence is already tokenized
            input_ids = torch.tensor([[ord(c) for c in sequence[:512]]])
            attention_mask = torch.ones_like(input_ids)
        
        results = {'sequence': sequence, 'length': len(sequence)}
        
        # Attention analysis
        if self.config.analyze_attention:
            self.logger.info("Analyzing attention patterns...")
            attention_weights = self.attention_analyzer.extract_attention_weights(
                input_ids, attention_mask,
                layers=self.config.attention_layers,
                heads=self.config.attention_heads
            )
            
            if attention_weights:
                attention_stats = self.attention_analyzer.compute_attention_statistics(attention_weights)
                results['attention'] = {
                    'weights': attention_weights,
                    'statistics': attention_stats
                }
                
                # Find important positions
                important_positions = self.attention_analyzer.find_important_positions(attention_weights)
                results['attention']['important_positions'] = important_positions
                
                # Visualize if requested
                if self.config.save_plots:
                    self.attention_analyzer.visualize_attention_patterns(
                        attention_weights,
                        [sequence],
                        save_path=self.config.output_dir
                    )
        
        # Gradient analysis
        if self.config.analyze_gradients and target_class is not None:
            self.logger.info("Analyzing gradients...")
            gradient_results = {}
            
            if 'vanilla' in self.config.gradient_methods:
                vanilla_grads = self.gradient_analyzer.vanilla_gradients(
                    input_ids, target_class, attention_mask
                )
                gradient_results['vanilla'] = vanilla_grads
            
            if 'integrated' in self.config.gradient_methods:
                integrated_grads = self.gradient_analyzer.integrated_gradients(
                    input_ids, target_class, attention_mask=attention_mask
                )
                gradient_results['integrated'] = integrated_grads
            
            if 'guided' in self.config.gradient_methods:
                guided_grads = self.gradient_analyzer.guided_backpropagation(
                    input_ids, target_class, attention_mask
                )
                gradient_results['guided'] = guided_grads
            
            results['gradients'] = gradient_results
        
        # Motif analysis
        if self.config.motif_analysis and 'gradients' in results:
            self.logger.info("Analyzing motifs...")
            # Use integrated gradients if available, otherwise vanilla
            importance_scores = results['gradients'].get('integrated', 
                                                       results['gradients'].get('vanilla'))
            
            if importance_scores is not None:
                important_subseqs = self.motif_analyzer.extract_important_subsequences(
                    sequence, importance_scores[0]
                )
                results['motifs'] = important_subseqs
        
        return results
    
    def analyze_batch(
        self,
        sequences: List[str],
        target_classes: Optional[List[int]] = None,
        save_summary: bool = True
    ) -> Dict[str, Any]:
        """Analyze multiple sequences and find common patterns."""
        self.logger.info(f"Analyzing batch of {len(sequences)} sequences")
        
        individual_results = []
        all_importance_scores = []
        
        for i, sequence in enumerate(sequences):
            target_class = target_classes[i] if target_classes else None
            result = self.analyze_sequence(sequence, target_class, detailed=False)
            individual_results.append(result)
            
            # Collect importance scores for consensus analysis
            if 'gradients' in result:
                importance_scores = result['gradients'].get('integrated',
                                                          result['gradients'].get('vanilla'))
                if importance_scores is not None:
                    all_importance_scores.append(importance_scores[0])
        
        # Find consensus motifs
        consensus_motifs = []
        if all_importance_scores and self.config.motif_analysis:
            consensus_motifs = self.motif_analyzer.find_consensus_motifs(
                sequences, all_importance_scores
            )
        
        batch_results = {
            'individual_results': individual_results,
            'consensus_motifs': consensus_motifs,
            'summary_statistics': self._compute_batch_statistics(individual_results)
        }
        
        # Save summary
        if save_summary:
            self._save_batch_summary(batch_results)
        
        return batch_results
    
    def _compute_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics for batch analysis."""
        stats = {
            'num_sequences': len(results),
            'avg_length': np.mean([r['length'] for r in results]),
            'attention_stats': {},
            'gradient_stats': {}
        }
        
        # Aggregate attention statistics
        if any('attention' in r for r in results):
            attention_entropies = []
            attention_sparsities = []
            
            for result in results:
                if 'attention' in result:
                    for layer_stats in result['attention']['statistics'].values():
                        attention_entropies.append(layer_stats['entropy'])
                        attention_sparsities.append(layer_stats['sparsity'])
            
            if attention_entropies:
                stats['attention_stats'] = {
                    'avg_entropy': np.mean(attention_entropies),
                    'avg_sparsity': np.mean(attention_sparsities)
                }
        
        return stats
    
    def _save_batch_summary(self, batch_results: Dict[str, Any]) -> None:
        """Save batch analysis summary to file."""
        summary_path = Path(self.config.output_dir) / "batch_analysis_summary.json"
        
        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(batch_results)
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Batch summary saved to {summary_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert tensors and other non-serializable objects for JSON."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() if k != 'weights'}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def example_interpretability_analysis():
    """Example usage of interpretability tools."""
    from hyena_glt.model import HyenaGLTForSequenceClassification
    from hyena_glt.config import HyenaGLTConfig
    from hyena_glt.data import DNATokenizer
    
    # Setup
    config = HyenaGLTConfig(num_labels=2, task_type="sequence_classification")
    model = HyenaGLTForSequenceClassification(config)
    tokenizer = DNATokenizer(k=3)
    
    # Configure interpretability analysis
    interp_config = InterpretabilityConfig(
        analyze_attention=True,
        analyze_gradients=True,
        motif_analysis=True,
        save_plots=True,
        output_dir="./interpretability_example"
    )
    
    # Initialize interpreter
    interpreter = ModelInterpreter(model, tokenizer, interp_config)
    
    # Analyze single sequence
    test_sequence = "ATGCGATCGATCGATCGATCGATCGATCGTAG"
    results = interpreter.analyze_sequence(test_sequence, target_class=1)
    
    print("Single sequence analysis completed!")
    print(f"Found {len(results.get('motifs', []))} important motifs")
    
    # Analyze batch
    test_sequences = [
        "ATGCGATCGATCGATCGATCGATCGATCGTAG",
        "AAATTTGGGCCCGATCGATCGATCGATCGATC",
        "TATAAAAGGCCGGCCATATCCGGTACCGATCG"
    ]
    
    batch_results = interpreter.analyze_batch(test_sequences)
    
    print("Batch analysis completed!")
    print(f"Found {len(batch_results['consensus_motifs'])} consensus motifs")
    
    return results, batch_results


# Create main framework class alias for backward compatibility
HyenaInterpretabilityFramework = ModelInterpreter

# Export main classes
__all__ = [
    'InterpretabilityConfig',
    'AttentionAnalyzer',
    'GradientAnalyzer', 
    'GenomicMotifAnalyzer',
    'ModelInterpreter',
    'HyenaInterpretabilityFramework',
    'example_interpretability_analysis'
]

if __name__ == "__main__":
    # Run example
    example_interpretability_analysis()
