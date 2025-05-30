"""
Evaluation metrics for genomic sequence modeling tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from scipy.stats import pearsonr, spearmanr
import warnings
from dataclasses import dataclass

try:
    from Bio.Seq import Seq
    from Bio.SeqUtils import GC, molecular_weight
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    task_name: str
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseMetric:
    """Base class for evaluation metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        pass
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        """Update metric with new predictions and targets."""
        raise NotImplementedError
    
    def compute(self) -> Dict[str, float]:
        """Compute final metric values."""
        raise NotImplementedError


class ClassificationMetrics(BaseMetric):
    """Comprehensive classification metrics."""
    
    def __init__(self, num_classes: int, average: str = 'weighted'):
        super().__init__("classification")
        self.num_classes = num_classes
        self.average = average
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               probabilities: Optional[torch.Tensor] = None):
        """
        Update with batch predictions.
        
        Args:
            predictions: Predicted class indices [batch_size] or [batch_size, seq_len]
            targets: True class indices [batch_size] or [batch_size, seq_len]
            probabilities: Class probabilities [batch_size, num_classes] or [batch_size, seq_len, num_classes]
        """
        # Handle sequence-level predictions
        if predictions.dim() > 1:
            predictions = predictions.flatten()
            targets = targets.flatten()
            if probabilities is not None:
                probabilities = probabilities.view(-1, probabilities.size(-1))
        
        # Filter out padding tokens (assuming -100 or num_classes as ignore index)
        valid_mask = (targets != -100) & (targets < self.num_classes)
        if valid_mask.any():
            self.predictions.extend(predictions[valid_mask].cpu().numpy())
            self.targets.extend(targets[valid_mask].cpu().numpy())
            if probabilities is not None:
                self.probabilities.extend(probabilities[valid_mask].cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average=self.average, zero_division=0),
            'recall': recall_score(targets, predictions, average=self.average, zero_division=0),
            'f1': f1_score(targets, predictions, average=self.average, zero_division=0),
            'mcc': matthews_corrcoef(targets, predictions)
        }
        
        # Add per-class metrics for multiclass
        if self.num_classes > 2:
            per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
            for i, f1_val in enumerate(per_class_f1):
                metrics[f'f1_class_{i}'] = f1_val
        
        # Add probabilistic metrics if available
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            try:
                if self.num_classes == 2:
                    metrics['auc_roc'] = roc_auc_score(targets, probabilities[:, 1])
                    metrics['auc_pr'] = average_precision_score(targets, probabilities[:, 1])
                else:
                    metrics['auc_roc'] = roc_auc_score(targets, probabilities, 
                                                     multi_class='ovr', average=self.average)
            except ValueError as e:
                warnings.warn(f"Could not compute AUC metrics: {e}")
        
        return metrics


class RegressionMetrics(BaseMetric):
    """Regression evaluation metrics."""
    
    def __init__(self):
        super().__init__("regression")
        self.predictions = []
        self.targets = []
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        self.predictions.extend(predictions.flatten().cpu().numpy())
        self.targets.extend(targets.flatten().cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(predictions, targets)
        spearman_corr, spearman_p = spearmanr(predictions, targets)
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p
        }


class PerplexityMetric(BaseMetric):
    """Perplexity metric for language modeling."""
    
    def __init__(self):
        super().__init__("perplexity")
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor, 
               ignore_index: int = -100):
        """
        Update perplexity with batch.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            ignore_index: Index to ignore in loss computation
        """
        # Flatten tensors
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='sum')
        
        # Count valid tokens
        valid_tokens = (targets != ignore_index).sum().item()
        
        self.total_loss += loss.item()
        self.total_tokens += valid_tokens
    
    def compute(self) -> Dict[str, float]:
        if self.total_tokens == 0:
            return {'perplexity': float('inf')}
        
        avg_loss = self.total_loss / self.total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'perplexity': perplexity,
            'cross_entropy': avg_loss
        }


class GenomicSequenceMetrics(BaseMetric):
    """Genomic sequence-specific evaluation metrics."""
    
    def __init__(self, sequence_type: str = 'dna'):
        super().__init__("genomic_sequence")
        self.sequence_type = sequence_type.lower()
        self.generated_sequences = []
        self.reference_sequences = []
    
    def reset(self):
        self.generated_sequences = []
        self.reference_sequences = []
    
    def update(self, generated: List[str], reference: List[str]):
        """
        Update with generated and reference sequences.
        
        Args:
            generated: List of generated sequences
            reference: List of reference sequences
        """
        self.generated_sequences.extend(generated)
        self.reference_sequences.extend(reference)
    
    def compute(self) -> Dict[str, float]:
        if not self.generated_sequences or not HAS_BIOPYTHON:
            return {}
        
        metrics = {}
        
        # Sequence length statistics
        gen_lengths = [len(seq) for seq in self.generated_sequences]
        ref_lengths = [len(seq) for seq in self.reference_sequences]
        
        metrics.update({
            'avg_length_generated': np.mean(gen_lengths),
            'avg_length_reference': np.mean(ref_lengths),
            'length_correlation': pearsonr(gen_lengths, ref_lengths)[0] if len(gen_lengths) == len(ref_lengths) else 0,
        })
        
        # DNA/RNA specific metrics
        if self.sequence_type in ['dna', 'rna']:
            gen_gc = [GC(seq) for seq in self.generated_sequences if seq]
            ref_gc = [GC(seq) for seq in self.reference_sequences if seq]
            
            if gen_gc and ref_gc:
                metrics.update({
                    'avg_gc_generated': np.mean(gen_gc),
                    'avg_gc_reference': np.mean(ref_gc),
                    'gc_correlation': pearsonr(gen_gc, ref_gc)[0] if len(gen_gc) == len(ref_gc) else 0,
                })
        
        # Protein specific metrics
        elif self.sequence_type == 'protein':
            gen_mw = []
            ref_mw = []
            
            for seq in self.generated_sequences:
                try:
                    gen_mw.append(molecular_weight(seq, seq_type='protein'))
                except:
                    pass
            
            for seq in self.reference_sequences:
                try:
                    ref_mw.append(molecular_weight(seq, seq_type='protein'))
                except:
                    pass
            
            if gen_mw and ref_mw:
                metrics.update({
                    'avg_mw_generated': np.mean(gen_mw),
                    'avg_mw_reference': np.mean(ref_mw),
                    'mw_correlation': pearsonr(gen_mw, ref_mw)[0] if len(gen_mw) == len(ref_mw) else 0,
                })
        
        # Edit distance metrics
        if len(self.generated_sequences) == len(self.reference_sequences):
            edit_distances = []
            for gen, ref in zip(self.generated_sequences, self.reference_sequences):
                edit_distances.append(self._edit_distance(gen, ref))
            
            metrics['avg_edit_distance'] = np.mean(edit_distances)
            metrics['normalized_edit_distance'] = np.mean([
                dist / max(len(gen), len(ref)) 
                for dist, gen, ref in zip(edit_distances, self.generated_sequences, self.reference_sequences)
                if max(len(gen), len(ref)) > 0
            ])
        
        return metrics
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class ComputationalMetrics(BaseMetric):
    """Computational efficiency metrics."""
    
    def __init__(self):
        super().__init__("computational")
        self.inference_times = []
        self.memory_usage = []
        self.flops_count = []
    
    def reset(self):
        self.inference_times = []
        self.memory_usage = []
        self.flops_count = []
    
    def update(self, inference_time: float, memory_mb: Optional[float] = None, 
               flops: Optional[int] = None):
        """
        Update computational metrics.
        
        Args:
            inference_time: Inference time in seconds
            memory_mb: Memory usage in MB
            flops: Number of floating point operations
        """
        self.inference_times.append(inference_time)
        if memory_mb is not None:
            self.memory_usage.append(memory_mb)
        if flops is not None:
            self.flops_count.append(flops)
    
    def compute(self) -> Dict[str, float]:
        metrics = {}
        
        if self.inference_times:
            metrics.update({
                'avg_inference_time': np.mean(self.inference_times),
                'std_inference_time': np.std(self.inference_times),
                'median_inference_time': np.median(self.inference_times),
                'throughput_samples_per_sec': 1.0 / np.mean(self.inference_times)
            })
        
        if self.memory_usage:
            metrics.update({
                'avg_memory_mb': np.mean(self.memory_usage),
                'peak_memory_mb': np.max(self.memory_usage)
            })
        
        if self.flops_count:
            metrics.update({
                'avg_flops': np.mean(self.flops_count),
                'total_flops': np.sum(self.flops_count)
            })
        
        return metrics


class MultiTaskEvaluator:
    """Evaluator for multi-task learning scenarios."""
    
    def __init__(self, task_configs: Dict[str, Dict]):
        """
        Initialize multi-task evaluator.
        
        Args:
            task_configs: Dictionary mapping task names to their configurations
        """
        self.task_configs = task_configs
        self.task_metrics = {}
        
        for task_name, config in task_configs.items():
            task_type = config.get('type', 'classification')
            
            if task_type == 'classification':
                self.task_metrics[task_name] = ClassificationMetrics(
                    num_classes=config.get('num_classes', 2),
                    average=config.get('average', 'weighted')
                )
            elif task_type == 'regression':
                self.task_metrics[task_name] = RegressionMetrics()
            elif task_type == 'generation':
                self.task_metrics[task_name] = PerplexityMetric()
            elif task_type == 'genomic_sequence':
                self.task_metrics[task_name] = GenomicSequenceMetrics(
                    sequence_type=config.get('sequence_type', 'dna')
                )
    
    def reset(self):
        """Reset all task metrics."""
        for metric in self.task_metrics.values():
            metric.reset()
    
    def update(self, task_name: str, **kwargs):
        """Update metrics for a specific task."""
        if task_name in self.task_metrics:
            self.task_metrics[task_name].update(**kwargs)
    
    def compute(self) -> Dict[str, EvaluationResult]:
        """Compute metrics for all tasks."""
        results = {}
        
        for task_name, metric in self.task_metrics.items():
            task_metrics = metric.compute()
            results[task_name] = EvaluationResult(
                task_name=task_name,
                metrics=task_metrics
            )
        
        return results
    
    def compute_summary(self) -> Dict[str, float]:
        """Compute summary metrics across all tasks."""
        all_results = self.compute()
        summary = {}
        
        # Aggregate metrics across tasks
        metric_names = set()
        for result in all_results.values():
            metric_names.update(result.metrics.keys())
        
        for metric_name in metric_names:
            values = []
            for result in all_results.values():
                if metric_name in result.metrics:
                    values.append(result.metrics[metric_name])
            
            if values:
                summary[f'avg_{metric_name}'] = np.mean(values)
                summary[f'std_{metric_name}'] = np.std(values)
        
        return summary


class BenchmarkEvaluator:
    """Comprehensive benchmark evaluator for genomic tasks."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.evaluators = {}
        self.computational_metrics = ComputationalMetrics()
        
        # Initialize task-specific evaluators
        for task_name, task_config in config.get('tasks', {}).items():
            self.evaluators[task_name] = MultiTaskEvaluator({task_name: task_config})
    
    def evaluate_model(self, model, data_loader, device: str = 'cuda') -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            
        Returns:
            Dictionary containing all evaluation results
        """
        model.eval()
        results = {
            'task_results': {},
            'computational_metrics': {},
            'summary_metrics': {}
        }
        
        # Reset all metrics
        for evaluator in self.evaluators.values():
            evaluator.reset()
        self.computational_metrics.reset()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Time inference
                torch.cuda.synchronize()
                start_time.record()
                
                outputs = model(**batch)
                
                end_time.record()
                torch.cuda.synchronize()
                
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                
                # Memory usage
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                    self.computational_metrics.update(inference_time, memory_mb)
                else:
                    self.computational_metrics.update(inference_time)
                
                # Update task-specific metrics
                for task_name, evaluator in self.evaluators.items():
                    if task_name in outputs:
                        self._update_task_metrics(
                            task_name, evaluator, outputs[task_name], batch
                        )
        
        # Compute all results
        for task_name, evaluator in self.evaluators.items():
            results['task_results'][task_name] = evaluator.compute()
        
        results['computational_metrics'] = self.computational_metrics.compute()
        
        # Compute summary metrics
        all_task_metrics = {}
        for task_results in results['task_results'].values():
            for task_name, eval_result in task_results.items():
                all_task_metrics.update({
                    f'{task_name}_{k}': v for k, v in eval_result.metrics.items()
                })
        
        results['summary_metrics'] = all_task_metrics
        
        return results
    
    def _update_task_metrics(self, task_name: str, evaluator: MultiTaskEvaluator, 
                           outputs: Dict, batch: Dict):
        """Update metrics for a specific task."""
        task_config = self.config['tasks'][task_name]
        task_type = task_config.get('type', 'classification')
        
        if task_type == 'classification':
            predictions = outputs.get('predictions', outputs.get('logits'))
            targets = batch.get('labels', batch.get('targets'))
            probabilities = outputs.get('probabilities')
            
            if predictions is not None and targets is not None:
                if predictions.dim() > 1 and predictions.size(-1) > 1:
                    # Convert logits to predictions
                    predictions = torch.argmax(predictions, dim=-1)
                
                evaluator.update(
                    task_name, 
                    predictions=predictions, 
                    targets=targets,
                    probabilities=probabilities
                )
        
        elif task_type == 'regression':
            predictions = outputs.get('predictions')
            targets = batch.get('targets')
            
            if predictions is not None and targets is not None:
                evaluator.update(task_name, predictions=predictions, targets=targets)
        
        elif task_type == 'generation':
            logits = outputs.get('logits')
            targets = batch.get('labels', batch.get('targets'))
            
            if logits is not None and targets is not None:
                evaluator.update(task_name, logits=logits, targets=targets)
