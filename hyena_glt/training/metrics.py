"""Comprehensive metrics for genomic sequence modeling."""

from typing import Dict, List, Optional, Union, Any
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, matthews_corrcoef
)
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from collections import defaultdict
import warnings


class GenomicMetrics:
    """Comprehensive metrics for genomic sequence tasks."""
    
    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.sequences = []
        self.metadata = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sequences: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        """Update metrics with new predictions and targets."""
        # Convert to CPU and numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()
        
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if sequences is not None:
            self.sequences.extend(sequences)
        if metadata is not None:
            self.metadata.append(metadata)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.predictions:
            return {}
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        if self.task_type == "classification":
            return self._compute_classification_metrics(all_preds, all_targets)
        elif self.task_type == "token_classification":
            return self._compute_token_classification_metrics(all_preds, all_targets)
        elif self.task_type == "generation":
            return self._compute_generation_metrics(all_preds, all_targets)
        else:
            return self._compute_basic_metrics(all_preds, all_targets)
    
    def _compute_classification_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        # Convert logits to predictions
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=-1)
            probabilities = F.softmax(predictions, dim=-1)
        else:
            pred_classes = predictions
            probabilities = predictions
        
        # Convert to numpy
        pred_classes = pred_classes.numpy()
        targets = targets.numpy()
        
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(targets, pred_classes)
        
        # Precision, recall, F1
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, pred_classes, average='weighted', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            
            # Macro averages
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                targets, pred_classes, average='macro', zero_division=0
            )
            metrics['precision_macro'] = precision_macro
            metrics['recall_macro'] = recall_macro
            metrics['f1_macro'] = f1_macro
        except Exception as e:
            warnings.warn(f"Error computing precision/recall/F1: {e}")
        
        # Matthews Correlation Coefficient
        try:
            metrics['mcc'] = matthews_corrcoef(targets, pred_classes)
        except Exception:
            metrics['mcc'] = 0.0
        
        # AUC-ROC for binary/multiclass
        try:
            if predictions.dim() > 1 and predictions.shape[1] > 1:
                if predictions.shape[1] == 2:
                    # Binary classification
                    metrics['auc_roc'] = roc_auc_score(targets, probabilities[:, 1].numpy())
                else:
                    # Multiclass
                    metrics['auc_roc'] = roc_auc_score(
                        targets, probabilities.numpy(), multi_class='ovr'
                    )
        except Exception as e:
            warnings.warn(f"Error computing AUC-ROC: {e}")
        
        # Per-class metrics
        try:
            unique_labels = np.unique(np.concatenate([targets, pred_classes]))
            for label in unique_labels:
                label_mask = (targets == label)
                if label_mask.sum() > 0:
                    label_acc = accuracy_score(targets[label_mask], pred_classes[label_mask])
                    metrics[f'accuracy_class_{label}'] = label_acc
        except Exception:
            pass
        
        return metrics
    
    def _compute_token_classification_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute token-level classification metrics."""
        # Flatten predictions and targets
        if predictions.dim() > 2:
            pred_classes = torch.argmax(predictions, dim=-1)
        else:
            pred_classes = predictions
        
        # Flatten to 1D
        pred_flat = pred_classes.view(-1).numpy()
        target_flat = targets.view(-1).numpy()
        
        # Remove padding tokens (assuming -100 or 0 is padding)
        valid_mask = (target_flat != -100) & (target_flat != 0)
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        metrics = {}
        
        # Token-level accuracy
        metrics['token_accuracy'] = accuracy_score(target_valid, pred_valid)
        
        # Sequence-level accuracy
        seq_length = predictions.shape[1] if predictions.dim() > 1 else len(predictions)
        pred_seqs = pred_classes.view(-1, seq_length)
        target_seqs = targets.view(-1, seq_length)
        
        seq_matches = (pred_seqs == target_seqs).all(dim=1).float().mean().item()
        metrics['sequence_accuracy'] = seq_matches
        
        # Per-class token metrics
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                target_valid, pred_valid, average='weighted', zero_division=0
            )
            metrics['token_precision'] = precision
            metrics['token_recall'] = recall
            metrics['token_f1'] = f1
        except Exception:
            pass
        
        # Entity-level metrics (if applicable)
        if hasattr(self, '_compute_entity_metrics'):
            entity_metrics = self._compute_entity_metrics(pred_classes, targets)
            metrics.update(entity_metrics)
        
        return metrics
    
    def _compute_generation_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute generation metrics."""
        metrics = {}
        
        # Perplexity
        if predictions.dim() > 2:
            # predictions are logits
            log_probs = F.log_softmax(predictions, dim=-1)
            target_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
            
            # Mask padding tokens
            mask = (targets != 0) & (targets != -100)
            valid_log_probs = target_log_probs[mask]
            
            if len(valid_log_probs) > 0:
                avg_log_prob = valid_log_probs.mean()
                perplexity = torch.exp(-avg_log_prob).item()
                metrics['perplexity'] = perplexity
                metrics['cross_entropy'] = -avg_log_prob.item()
        
        # BLEU score (if we have sequence strings)
        if self.sequences and len(self.sequences) >= 2:
            try:
                bleu_score = self._compute_bleu_score()
                metrics['bleu'] = bleu_score
            except Exception:
                pass
        
        # Sequence similarity metrics
        try:
            similarity_metrics = self._compute_sequence_similarity(predictions, targets)
            metrics.update(similarity_metrics)
        except Exception:
            pass
        
        return metrics
    
    def _compute_basic_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute basic metrics for any task."""
        metrics = {}
        
        # MSE for continuous targets
        if targets.dtype in [torch.float32, torch.float64]:
            mse = F.mse_loss(predictions.float(), targets.float()).item()
            metrics['mse'] = mse
            metrics['rmse'] = np.sqrt(mse)
        
        # Cosine similarity
        try:
            pred_flat = predictions.view(predictions.shape[0], -1)
            target_flat = targets.view(targets.shape[0], -1)
            
            cosine_sim = F.cosine_similarity(pred_flat.float(), target_flat.float(), dim=1)
            metrics['cosine_similarity'] = cosine_sim.mean().item()
        except Exception:
            pass
        
        return metrics
    
    def _compute_bleu_score(self) -> float:
        """Compute BLEU score for sequence generation."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Assume sequences come in pairs (predicted, target)
            bleu_scores = []
            smoothing = SmoothingFunction().method1
            
            for i in range(0, len(self.sequences), 2):
                if i + 1 < len(self.sequences):
                    reference = [self.sequences[i+1].split()]
                    candidate = self.sequences[i].split()
                    
                    score = sentence_bleu(
                        reference, candidate, 
                        smoothing_function=smoothing
                    )
                    bleu_scores.append(score)
            
            return np.mean(bleu_scores) if bleu_scores else 0.0
        except ImportError:
            warnings.warn("NLTK not available for BLEU score computation")
            return 0.0
    
    def _compute_sequence_similarity(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute sequence-level similarity metrics."""
        metrics = {}
        
        # Hamming distance for sequences
        if predictions.shape == targets.shape:
            matches = (predictions == targets).float()
            hamming_similarity = matches.mean().item()
            metrics['hamming_similarity'] = hamming_similarity
        
        # Edit distance (approximation using dynamic programming concepts)
        try:
            edit_distances = []
            for pred_seq, target_seq in zip(predictions, targets):
                edit_dist = self._compute_edit_distance(
                    pred_seq.tolist(), target_seq.tolist()
                )
                edit_distances.append(edit_dist)
            
            if edit_distances:
                avg_edit_distance = np.mean(edit_distances)
                max_len = max(len(predictions[0]), len(targets[0]))
                normalized_edit_distance = avg_edit_distance / max_len if max_len > 0 else 0
                
                metrics['edit_distance'] = avg_edit_distance
                metrics['normalized_edit_distance'] = normalized_edit_distance
                metrics['edit_similarity'] = 1.0 - normalized_edit_distance
        except Exception:
            pass
        
        return metrics
    
    def _compute_edit_distance(self, seq1: List[int], seq2: List[int]) -> float:
        """Compute edit distance between two sequences."""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[m][n]
    
    def get_detailed_report(self) -> str:
        """Get detailed metrics report."""
        metrics = self.compute()
        
        report_lines = [f"=== Genomic Metrics Report ({self.task_type}) ==="]
        
        # Group metrics by category
        accuracy_metrics = {k: v for k, v in metrics.items() if 'accuracy' in k}
        precision_recall_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['precision', 'recall', 'f1'])}
        similarity_metrics = {k: v for k, v in metrics.items() if 'similarity' in k}
        other_metrics = {k: v for k, v in metrics.items() if k not in accuracy_metrics and k not in precision_recall_metrics and k not in similarity_metrics}
        
        if accuracy_metrics:
            report_lines.append("\nAccuracy Metrics:")
            for metric, value in accuracy_metrics.items():
                report_lines.append(f"  {metric}: {value:.4f}")
        
        if precision_recall_metrics:
            report_lines.append("\nPrecision/Recall/F1 Metrics:")
            for metric, value in precision_recall_metrics.items():
                report_lines.append(f"  {metric}: {value:.4f}")
        
        if similarity_metrics:
            report_lines.append("\nSimilarity Metrics:")
            for metric, value in similarity_metrics.items():
                report_lines.append(f"  {metric}: {value:.4f}")
        
        if other_metrics:
            report_lines.append("\nOther Metrics:")
            for metric, value in other_metrics.items():
                report_lines.append(f"  {metric}: {value:.4f}")
        
        return "\n".join(report_lines)


class MultiTaskMetrics:
    """Metrics aggregator for multi-task learning."""
    
    def __init__(self, task_names: List[str]):
        self.task_names = task_names
        self.task_metrics = {name: GenomicMetrics() for name in task_names}
    
    def update(
        self,
        task_name: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ):
        """Update metrics for a specific task."""
        if task_name in self.task_metrics:
            self.task_metrics[task_name].update(predictions, targets, **kwargs)
    
    def compute(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics for all tasks."""
        all_metrics = {}
        for task_name, task_metric in self.task_metrics.items():
            all_metrics[task_name] = task_metric.compute()
        return all_metrics
    
    def compute_aggregated(self) -> Dict[str, float]:
        """Compute aggregated metrics across tasks."""
        all_metrics = self.compute()
        aggregated = {}
        
        # Collect all metric names
        all_metric_names = set()
        for task_metrics in all_metrics.values():
            all_metric_names.update(task_metrics.keys())
        
        # Aggregate each metric across tasks
        for metric_name in all_metric_names:
            values = []
            for task_name in self.task_names:
                if metric_name in all_metrics.get(task_name, {}):
                    values.append(all_metrics[task_name][metric_name])
            
            if values:
                aggregated[f'avg_{metric_name}'] = np.mean(values)
                aggregated[f'std_{metric_name}'] = np.std(values)
                aggregated[f'min_{metric_name}'] = np.min(values)
                aggregated[f'max_{metric_name}'] = np.max(values)
        
        return aggregated
    
    def reset(self):
        """Reset all task metrics."""
        for task_metric in self.task_metrics.values():
            task_metric.reset()
    
    def get_summary_report(self) -> str:
        """Get summary report for all tasks."""
        all_metrics = self.compute()
        aggregated = self.compute_aggregated()
        
        report_lines = ["=== Multi-Task Metrics Summary ==="]
        
        # Individual task metrics
        for task_name, task_metrics in all_metrics.items():
            report_lines.append(f"\n{task_name.upper()}:")
            for metric, value in task_metrics.items():
                report_lines.append(f"  {metric}: {value:.4f}")
        
        # Aggregated metrics
        report_lines.append("\nAGGREGATED METRICS:")
        for metric, value in aggregated.items():
            report_lines.append(f"  {metric}: {value:.4f}")
        
        return "\n".join(report_lines)
