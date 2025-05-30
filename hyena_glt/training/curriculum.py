"""Curriculum learning for Hyena-GLT training."""

from typing import Dict, List, Optional, Union, Callable, Any
import torch
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class CurriculumExample:
    """Single example with curriculum information."""
    data: Any
    difficulty: float
    sequence_length: int
    complexity_score: Optional[float] = None
    metadata: Optional[Dict] = None


class DifficultyMeasure(ABC):
    """Abstract base class for difficulty measures."""
    
    @abstractmethod
    def compute_difficulty(self, example: Any) -> float:
        """Compute difficulty score for an example."""
        pass


class SequenceLengthDifficulty(DifficultyMeasure):
    """Difficulty based on sequence length."""
    
    def __init__(self, normalize: bool = True, max_length: Optional[int] = None):
        self.normalize = normalize
        self.max_length = max_length
    
    def compute_difficulty(self, example: Any) -> float:
        """Compute difficulty based on sequence length."""
        if hasattr(example, 'input_ids'):
            length = len(example.input_ids)
        elif hasattr(example, 'sequence'):
            length = len(example.sequence)
        elif isinstance(example, (list, tuple)):
            length = len(example)
        else:
            return 0.5  # Default difficulty
        
        if self.normalize and self.max_length:
            return min(length / self.max_length, 1.0)
        return float(length)


class GenomicComplexityDifficulty(DifficultyMeasure):
    """Difficulty based on genomic sequence complexity."""
    
    def __init__(self, measures: List[str] = None):
        self.measures = measures or ['gc_content', 'entropy', 'repetitiveness']
    
    def compute_difficulty(self, example: Any) -> float:
        """Compute difficulty based on genomic complexity."""
        sequence = self._extract_sequence(example)
        if not sequence:
            return 0.5
        
        scores = []
        
        if 'gc_content' in self.measures:
            scores.append(self._gc_content_score(sequence))
        
        if 'entropy' in self.measures:
            scores.append(self._entropy_score(sequence))
        
        if 'repetitiveness' in self.measures:
            scores.append(self._repetitiveness_score(sequence))
        
        if 'rare_kmers' in self.measures:
            scores.append(self._rare_kmers_score(sequence))
        
        return np.mean(scores) if scores else 0.5
    
    def _extract_sequence(self, example: Any) -> str:
        """Extract sequence string from example."""
        if hasattr(example, 'sequence'):
            return example.sequence
        elif hasattr(example, 'input_ids') and hasattr(example, 'tokenizer'):
            return example.tokenizer.decode(example.input_ids)
        elif isinstance(example, str):
            return example
        return ""
    
    def _gc_content_score(self, sequence: str) -> float:
        """Score based on GC content deviation from 50%."""
        gc_count = sequence.count('G') + sequence.count('C')
        gc_content = gc_count / len(sequence) if sequence else 0.5
        # Distance from balanced GC content (0.5)
        return abs(gc_content - 0.5) * 2
    
    def _entropy_score(self, sequence: str) -> float:
        """Score based on sequence entropy."""
        if not sequence:
            return 0.0
        
        # Count nucleotides
        counts = {}
        for nucleotide in sequence:
            counts[nucleotide] = counts.get(nucleotide, 0) + 1
        
        # Compute entropy
        entropy = 0.0
        length = len(sequence)
        for count in counts.values():
            if count > 0:
                prob = count / length
                entropy -= prob * np.log2(prob)
        
        # Normalize by maximum entropy (log2(4) for 4 nucleotides)
        max_entropy = np.log2(4)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _repetitiveness_score(self, sequence: str) -> float:
        """Score based on repetitive patterns."""
        if len(sequence) < 3:
            return 0.0
        
        # Count 3-mer repetitions
        kmers = {}
        for i in range(len(sequence) - 2):
            kmer = sequence[i:i+3]
            kmers[kmer] = kmers.get(kmer, 0) + 1
        
        # Compute repetitiveness as fraction of non-unique 3-mers
        total_kmers = len(sequence) - 2
        unique_kmers = len(kmers)
        repetitive_kmers = sum(count - 1 for count in kmers.values() if count > 1)
        
        return repetitive_kmers / total_kmers if total_kmers > 0 else 0.0
    
    def _rare_kmers_score(self, sequence: str, k: int = 4) -> float:
        """Score based on presence of rare k-mers."""
        if len(sequence) < k:
            return 0.0
        
        # Common k-mers in genomic sequences (this is a simplified heuristic)
        common_kmers = {'AAAA', 'TTTT', 'GGGG', 'CCCC', 'ATCG', 'CGAT'}
        
        total_kmers = len(sequence) - k + 1
        rare_count = 0
        
        for i in range(total_kmers):
            kmer = sequence[i:i+k]
            if kmer not in common_kmers:
                rare_count += 1
        
        return rare_count / total_kmers if total_kmers > 0 else 0.0


class TaskSpecificDifficulty(DifficultyMeasure):
    """Difficulty based on task-specific properties."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
    
    def compute_difficulty(self, example: Any) -> float:
        """Compute task-specific difficulty."""
        if self.task_type == "classification":
            return self._classification_difficulty(example)
        elif self.task_type == "generation":
            return self._generation_difficulty(example)
        elif self.task_type == "token_classification":
            return self._token_classification_difficulty(example)
        else:
            return 0.5
    
    def _classification_difficulty(self, example: Any) -> float:
        """Difficulty for classification tasks."""
        # Base difficulty on label frequency (rare labels are harder)
        if hasattr(example, 'label_frequency'):
            return 1.0 - example.label_frequency
        elif hasattr(example, 'label') and hasattr(example, 'label_counts'):
            total_samples = sum(example.label_counts.values())
            label_count = example.label_counts.get(example.label, 1)
            return 1.0 - (label_count / total_samples)
        return 0.5
    
    def _generation_difficulty(self, example: Any) -> float:
        """Difficulty for generation tasks."""
        # Base on target sequence properties
        if hasattr(example, 'target'):
            target_length = len(example.target)
            # Longer targets are generally harder
            return min(target_length / 1000, 1.0)  # Normalize by typical max length
        return 0.5
    
    def _token_classification_difficulty(self, example: Any) -> float:
        """Difficulty for token classification tasks."""
        # Base on label density and complexity
        if hasattr(example, 'labels'):
            labels = example.labels
            # Count non-background labels (assuming 0 is background)
            active_labels = sum(1 for label in labels if label != 0)
            label_density = active_labels / len(labels) if labels else 0
            return label_density
        return 0.5


class CurriculumLearning:
    """Curriculum learning scheduler for training."""
    
    def __init__(
        self,
        difficulty_measures: List[DifficultyMeasure],
        curriculum_strategy: str = "linear",
        start_difficulty: float = 0.1,
        end_difficulty: float = 1.0,
        curriculum_steps: int = 10000,
        difficulty_weights: Optional[List[float]] = None
    ):
        self.difficulty_measures = difficulty_measures
        self.curriculum_strategy = curriculum_strategy
        self.start_difficulty = start_difficulty
        self.end_difficulty = end_difficulty
        self.curriculum_steps = curriculum_steps
        self.difficulty_weights = difficulty_weights or [1.0] * len(difficulty_measures)
        
        # Curriculum state
        self.current_step = 0
        self.current_difficulty_threshold = start_difficulty
        
    def compute_example_difficulty(self, example: Any) -> float:
        """Compute overall difficulty for an example."""
        difficulties = []
        for measure in self.difficulty_measures:
            difficulty = measure.compute_difficulty(example)
            difficulties.append(difficulty)
        
        # Weighted average of difficulties
        weighted_sum = sum(d * w for d, w in zip(difficulties, self.difficulty_weights))
        total_weight = sum(self.difficulty_weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def should_include_example(self, example: Any) -> bool:
        """Determine if example should be included at current curriculum stage."""
        example_difficulty = self.compute_example_difficulty(example)
        return example_difficulty <= self.current_difficulty_threshold
    
    def update_curriculum(self, step: int) -> float:
        """Update curriculum difficulty threshold."""
        self.current_step = step
        
        if step >= self.curriculum_steps:
            self.current_difficulty_threshold = self.end_difficulty
        else:
            progress = step / self.curriculum_steps
            
            if self.curriculum_strategy == "linear":
                self.current_difficulty_threshold = (
                    self.start_difficulty + 
                    progress * (self.end_difficulty - self.start_difficulty)
                )
            elif self.curriculum_strategy == "exponential":
                # Exponential growth in difficulty
                alpha = np.log(self.end_difficulty / self.start_difficulty)
                self.current_difficulty_threshold = self.start_difficulty * np.exp(alpha * progress)
            elif self.curriculum_strategy == "cosine":
                # Cosine annealing
                cosine_factor = 0.5 * (1 - np.cos(np.pi * progress))
                self.current_difficulty_threshold = (
                    self.start_difficulty + 
                    cosine_factor * (self.end_difficulty - self.start_difficulty)
                )
            elif self.curriculum_strategy == "step":
                # Step-wise increase
                num_steps = 5  # Number of curriculum steps
                step_size = (self.end_difficulty - self.start_difficulty) / num_steps
                step_idx = min(int(progress * num_steps), num_steps - 1)
                self.current_difficulty_threshold = self.start_difficulty + step_idx * step_size
        
        return self.current_difficulty_threshold
    
    def filter_dataset(self, dataset: List[Any]) -> List[Any]:
        """Filter dataset based on current curriculum stage."""
        filtered = []
        for example in dataset:
            if self.should_include_example(example):
                filtered.append(example)
        return filtered
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get current curriculum statistics."""
        return {
            'current_step': self.current_step,
            'difficulty_threshold': self.current_difficulty_threshold,
            'curriculum_progress': min(self.current_step / self.curriculum_steps, 1.0),
            'strategy': self.curriculum_strategy
        }


class AdaptiveCurriculum(CurriculumLearning):
    """Adaptive curriculum that adjusts based on model performance."""
    
    def __init__(
        self,
        difficulty_measures: List[DifficultyMeasure],
        target_accuracy: float = 0.8,
        adjustment_rate: float = 0.01,
        **kwargs
    ):
        super().__init__(difficulty_measures, **kwargs)
        self.target_accuracy = target_accuracy
        self.adjustment_rate = adjustment_rate
        self.performance_history = []
        
    def update_with_performance(self, accuracy: float, step: int):
        """Update curriculum based on model performance."""
        self.performance_history.append(accuracy)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Adjust difficulty based on performance
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-10:])
            
            if recent_performance > self.target_accuracy:
                # Model is doing well, increase difficulty faster
                self.current_difficulty_threshold += self.adjustment_rate
            elif recent_performance < self.target_accuracy * 0.8:
                # Model is struggling, slow down or decrease difficulty
                self.current_difficulty_threshold -= self.adjustment_rate * 0.5
        
        # Ensure threshold stays within bounds
        self.current_difficulty_threshold = np.clip(
            self.current_difficulty_threshold,
            self.start_difficulty,
            self.end_difficulty
        )
        
        # Also update based on standard curriculum
        base_threshold = super().update_curriculum(step)
        
        # Take minimum to ensure we don't go too fast
        self.current_difficulty_threshold = min(
            self.current_difficulty_threshold,
            base_threshold
        )
        
        return self.current_difficulty_threshold


class CurriculumDataLoader:
    """Data loader with curriculum learning support."""
    
    def __init__(
        self,
        dataset: List[Any],
        curriculum: CurriculumLearning,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.curriculum = curriculum
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Precompute difficulties for all examples
        self.example_difficulties = [
            curriculum.compute_example_difficulty(example)
            for example in dataset
        ]
        
    def get_dataloader(self, step: int) -> torch.utils.data.DataLoader:
        """Get dataloader for current curriculum stage."""
        # Update curriculum
        self.curriculum.update_curriculum(step)
        
        # Filter examples based on difficulty
        filtered_indices = []
        for i, difficulty in enumerate(self.example_difficulties):
            if difficulty <= self.curriculum.current_difficulty_threshold:
                filtered_indices.append(i)
        
        # Create subset dataset
        filtered_dataset = torch.utils.data.Subset(self.dataset, filtered_indices)
        
        # Create dataloader
        return torch.utils.data.DataLoader(
            filtered_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get information about current curriculum state."""
        total_examples = len(self.dataset)
        available_examples = sum(
            1 for difficulty in self.example_difficulties
            if difficulty <= self.curriculum.current_difficulty_threshold
        )
        
        return {
            'total_examples': total_examples,
            'available_examples': available_examples,
            'utilization': available_examples / total_examples if total_examples > 0 else 0,
            'difficulty_threshold': self.curriculum.current_difficulty_threshold,
            'curriculum_stats': self.curriculum.get_curriculum_stats()
        }
