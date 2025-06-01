"""
Genomic data loading utilities for pretraining Hyena-GLT models.

This module provides data loading functionality adapted from savanna's approach
for handling large-scale genomic datasets during pretraining.
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Iterator
import mmap

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np

from ..data.tokenizer import DNATokenizer, RNATokenizer, ProteinTokenizer

logger = logging.getLogger(__name__)


class GenomicPretrainingDataset(IterableDataset):
    """
    Iterable dataset for genomic pretraining data.
    
    Supports large-scale genomic data files with efficient memory usage
    through memory-mapped files and streaming data loading.
    """
    
    def __init__(
        self,
        data_paths: List[str],
        tokenizer: Union[DNATokenizer, RNATokenizer, ProteinTokenizer],
        max_length: int = 1024,
        sequence_type: str = "dna",
        enforce_length: bool = True,
        weights: Optional[List[float]] = None,
        seed: int = 42,
        worker_init_fn_seed: Optional[int] = None
    ):
        """
        Initialize genomic pretraining dataset.
        
        Args:
            data_paths: List of paths to genomic data files
            tokenizer: Tokenizer for the specific sequence type
            max_length: Maximum sequence length
            sequence_type: Type of sequences ("dna", "rna", "protein")
            enforce_length: Whether to enforce exact sequence length
            weights: Optional weights for data mixing
            seed: Random seed for reproducibility
            worker_init_fn_seed: Seed for worker initialization
        """
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequence_type = sequence_type.lower()
        self.enforce_length = enforce_length
        self.weights = weights or [1.0] * len(data_paths)
        self.seed = seed
        self.worker_init_fn_seed = worker_init_fn_seed
        
        # Validate inputs
        if len(self.weights) != len(self.data_paths):
            raise ValueError("Number of weights must match number of data paths")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Store file handles and metadata
        self.file_handles = []
        self.file_sizes = []
        self.file_num_sequences = []
        
        # Initialize file metadata
        self._initialize_files()
        
        logger.info(f"Initialized dataset with {len(self.data_paths)} files")
        logger.info(f"Total sequences: {sum(self.file_num_sequences)}")
    
    def _initialize_files(self):
        """Initialize file handles and compute metadata."""
        for file_path in self.data_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            # Get file size
            file_size = os.path.getsize(file_path)
            self.file_sizes.append(file_size)
            
            # Estimate number of sequences (rough estimate based on file format)
            if file_path.endswith('.txt') or file_path.endswith('.seq'):
                # Assume one sequence per line
                with open(file_path, 'r') as f:
                    num_lines = sum(1 for _ in f)
                self.file_num_sequences.append(num_lines)
            elif file_path.endswith('.fasta') or file_path.endswith('.fa'):
                # Count FASTA entries
                with open(file_path, 'r') as f:
                    num_seqs = sum(1 for line in f if line.startswith('>'))
                self.file_num_sequences.append(num_seqs)
            else:
                # Default estimate based on file size
                estimated_seqs = file_size // (self.max_length * 2)  # Rough estimate
                self.file_num_sequences.append(max(1, estimated_seqs))
    
    def _get_file_iterator(self, file_path: str) -> Iterator[str]:
        """Get iterator over sequences in a file."""
        if file_path.endswith('.fasta') or file_path.endswith('.fa'):
            return self._fasta_iterator(file_path)
        else:
            return self._text_iterator(file_path)
    
    def _fasta_iterator(self, file_path: str) -> Iterator[str]:
        """Iterator for FASTA files."""
        with open(file_path, 'r') as f:
            sequence = ""
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if sequence:
                        yield sequence
                        sequence = ""
                else:
                    sequence += line.upper()
            if sequence:
                yield sequence
    
    def _text_iterator(self, file_path: str) -> Iterator[str]:
        """Iterator for text files (one sequence per line)."""
        with open(file_path, 'r') as f:
            for line in f:
                sequence = line.strip().upper()
                if sequence:
                    yield sequence
    
    def _process_sequence(self, sequence: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process a raw sequence into model inputs.
        
        Args:
            sequence: Raw genomic sequence string
            
        Returns:
            Dictionary with tokenized inputs or None if sequence should be skipped
        """
        # Clean sequence (remove non-standard characters)
        if self.sequence_type == "dna":
            valid_chars = set("ATCGN")
        elif self.sequence_type == "rna":
            valid_chars = set("AUCGN")
        elif self.sequence_type == "protein":
            valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
        else:
            raise ValueError(f"Unknown sequence type: {self.sequence_type}")
        
        # Filter invalid characters
        cleaned_sequence = ''.join(c for c in sequence if c in valid_chars)
        
        if len(cleaned_sequence) < 10:  # Skip very short sequences
            return None
        
        # Handle sequence length
        if self.enforce_length:
            if len(cleaned_sequence) < self.max_length:
                # Pad sequence to max length
                padding_needed = self.max_length - len(cleaned_sequence)
                if self.sequence_type == "protein":
                    pad_char = "X"  # Unknown amino acid
                else:
                    pad_char = "N"  # Unknown nucleotide
                cleaned_sequence += pad_char * padding_needed
            elif len(cleaned_sequence) > self.max_length:
                # Randomly crop sequence
                start_idx = random.randint(0, len(cleaned_sequence) - self.max_length)
                cleaned_sequence = cleaned_sequence[start_idx:start_idx + self.max_length]
        else:
            # Truncate if too long
            if len(cleaned_sequence) > self.max_length:
                cleaned_sequence = cleaned_sequence[:self.max_length]
        
        # Tokenize sequence
        try:
            tokens = self.tokenizer.encode(cleaned_sequence)
            
            # Convert to tensor
            input_ids = torch.tensor(tokens, dtype=torch.long)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones_like(input_ids)
            
            # If we padded the sequence, update attention mask
            if self.enforce_length and len(tokens) < self.max_length:
                pad_length = self.max_length - len(tokens)
                attention_mask[-pad_length:] = 0
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "sequence_length": torch.tensor(len(tokens), dtype=torch.long),
                "original_sequence": cleaned_sequence
            }
            
        except Exception as e:
            logger.warning(f"Failed to tokenize sequence: {e}")
            return None
    
    def __iter__(self):
        """Iterate over the dataset."""
        # Initialize random state for this worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Multi-worker setting
            worker_seed = self.worker_init_fn_seed or self.seed
            random.seed(worker_seed + worker_info.id)
            np.random.seed(worker_seed + worker_info.id)
        else:
            # Single worker
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        while True:  # Infinite iterator for training
            # Select file based on weights
            file_idx = np.random.choice(len(self.data_paths), p=self.weights)
            file_path = self.data_paths[file_idx]
            
            # Get sequences from selected file
            try:
                for sequence in self._get_file_iterator(file_path):
                    processed = self._process_sequence(sequence)
                    if processed is not None:
                        yield processed
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue


class GenomicDatasetFromConfig:
    """Create genomic datasets from configuration similar to savanna's approach."""
    
    @staticmethod
    def build_datasets_from_paths(
        train_data_paths: List[str],
        valid_data_paths: Optional[List[str]] = None,
        test_data_paths: Optional[List[str]] = None,
        train_weights: Optional[List[float]] = None,
        valid_weights: Optional[List[float]] = None,
        test_weights: Optional[List[float]] = None,
        tokenizer: Union[DNATokenizer, RNATokenizer, ProteinTokenizer] = None,
        max_length: int = 1024,
        sequence_type: str = "dna",
        enforce_length: bool = True,
        seed: int = 42
    ) -> Tuple[GenomicPretrainingDataset, Optional[GenomicPretrainingDataset], Optional[GenomicPretrainingDataset]]:
        """
        Build train, validation, and test datasets from file paths.
        
        Args:
            train_data_paths: Paths to training data files
            valid_data_paths: Paths to validation data files
            test_data_paths: Paths to test data files
            train_weights: Weights for training data mixing
            valid_weights: Weights for validation data mixing
            test_weights: Weights for test data mixing
            tokenizer: Tokenizer for sequences
            max_length: Maximum sequence length
            sequence_type: Type of sequences
            enforce_length: Whether to enforce exact length
            seed: Random seed
            
        Returns:
            Tuple of (train_dataset, valid_dataset, test_dataset)
        """
        if tokenizer is None:
            if sequence_type.lower() == "dna":
                tokenizer = DNATokenizer()
            elif sequence_type.lower() == "rna":
                tokenizer = RNATokenizer()
            elif sequence_type.lower() == "protein":
                tokenizer = ProteinTokenizer()
            else:
                raise ValueError(f"Unknown sequence type: {sequence_type}")
        
        # Create training dataset
        train_dataset = GenomicPretrainingDataset(
            data_paths=train_data_paths,
            tokenizer=tokenizer,
            max_length=max_length,
            sequence_type=sequence_type,
            enforce_length=enforce_length,
            weights=train_weights,
            seed=seed
        )
        
        # Create validation dataset
        valid_dataset = None
        if valid_data_paths:
            valid_dataset = GenomicPretrainingDataset(
                data_paths=valid_data_paths,
                tokenizer=tokenizer,
                max_length=max_length,
                sequence_type=sequence_type,
                enforce_length=enforce_length,
                weights=valid_weights,
                seed=seed + 1  # Different seed for validation
            )
        
        # Create test dataset
        test_dataset = None
        if test_data_paths:
            test_dataset = GenomicPretrainingDataset(
                data_paths=test_data_paths,
                tokenizer=tokenizer,
                max_length=max_length,
                sequence_type=sequence_type,
                enforce_length=enforce_length,
                weights=test_weights,
                seed=seed + 2  # Different seed for test
            )
        
        return train_dataset, valid_dataset, test_dataset


class GenomicDataLoader:
    """Data loader wrapper with genomic-specific functionality."""
    
    def __init__(
        self,
        dataset: GenomicPretrainingDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        worker_init_fn: Optional[callable] = None
    ):
        """
        Initialize genomic data loader.
        
        Args:
            dataset: Genomic dataset
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            worker_init_fn: Worker initialization function
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        # Default worker initialization function
        if worker_init_fn is None:
            worker_init_fn = self._worker_init_fn
        
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            collate_fn=self._collate_fn
        )
    
    def _worker_init_fn(self, worker_id: int):
        """Initialize worker with proper random seed."""
        # Set different seed for each worker
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching genomic sequences.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched dictionary
        """
        # Find maximum sequence length in batch
        max_len = max(sample["input_ids"].size(0) for sample in batch)
        
        # Initialize batch tensors
        batch_size = len(batch)
        
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        sequence_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill batch tensors
        for i, sample in enumerate(batch):
            seq_len = sample["input_ids"].size(0)
            input_ids[i, :seq_len] = sample["input_ids"]
            attention_mask[i, :seq_len] = sample["attention_mask"]
            sequence_lengths[i] = sample["sequence_length"]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sequence_lengths": sequence_lengths
        }
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Get number of batches (estimated for iterable dataset)."""
        # For iterable datasets, this is an estimate
        total_sequences = sum(self.dataset.file_num_sequences)
        return total_sequences // self.batch_size


def create_genomic_dataloaders(
    train_data_paths: List[str],
    valid_data_paths: Optional[List[str]] = None,
    test_data_paths: Optional[List[str]] = None,
    tokenizer: Union[DNATokenizer, RNATokenizer, ProteinTokenizer] = None,
    max_length: int = 1024,
    sequence_type: str = "dna",
    batch_size: int = 32,
    num_workers: int = 4,
    train_weights: Optional[List[float]] = None,
    valid_weights: Optional[List[float]] = None,
    test_weights: Optional[List[float]] = None,
    seed: int = 42,
    enforce_length: bool = True
) -> Tuple[GenomicDataLoader, Optional[GenomicDataLoader], Optional[GenomicDataLoader]]:
    """
    Create genomic data loaders for pretraining.
    
    This is the main convenience function for creating data loaders
    similar to savanna's build_train_valid_test_data_iterators.
    
    Args:
        train_data_paths: Paths to training data files
        valid_data_paths: Paths to validation data files
        test_data_paths: Paths to test data files
        tokenizer: Tokenizer for sequences
        max_length: Maximum sequence length
        sequence_type: Type of sequences ("dna", "rna", "protein")
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_weights: Weights for training data mixing
        valid_weights: Weights for validation data mixing
        test_weights: Weights for test data mixing
        seed: Random seed
        enforce_length: Whether to enforce exact sequence length
        
    Returns:
        Tuple of (train_dataloader, valid_dataloader, test_dataloader)
    """
    logger.info("Creating genomic datasets and data loaders...")
    
    # Create datasets
    train_dataset, valid_dataset, test_dataset = GenomicDatasetFromConfig.build_datasets_from_paths(
        train_data_paths=train_data_paths,
        valid_data_paths=valid_data_paths,
        test_data_paths=test_data_paths,
        train_weights=train_weights,
        valid_weights=valid_weights,
        test_weights=test_weights,
        tokenizer=tokenizer,
        max_length=max_length,
        sequence_type=sequence_type,
        enforce_length=enforce_length,
        seed=seed
    )
    
    # Create data loaders
    train_dataloader = GenomicDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_dataloader = None
    if valid_dataset:
        valid_dataloader = GenomicDataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    test_dataloader = None
    if test_dataset:
        test_dataloader = GenomicDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    logger.info(f"Created data loaders with batch size {batch_size}")
    logger.info(f"Training batches: {len(train_dataloader)}")
    if valid_dataloader:
        logger.info(f"Validation batches: {len(valid_dataloader)}")
    if test_dataloader:
        logger.info(f"Test batches: {len(test_dataloader)}")
    
    return train_dataloader, valid_dataloader, test_dataloader


# Example usage and testing
if __name__ == "__main__":
    # Example: Create a simple genomic dataset for testing
    
    # Create sample data
    sample_sequences = [
        "ATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTA",
        "TTTTAAAACCCCGGGG",
        "NNNNNNNNNNNNNNNN",
        "ATGCGTACGTACGTAC"
    ]
    
    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for seq in sample_sequences:
            f.write(seq + '\n')
        temp_file = f.name
    
    try:
        # Create tokenizer
        tokenizer = DNATokenizer()
        
        # Create dataset
        dataset = GenomicPretrainingDataset(
            data_paths=[temp_file],
            tokenizer=tokenizer,
            max_length=32,
            sequence_type="dna",
            enforce_length=True,
            seed=42
        )
        
        # Create data loader
        dataloader = GenomicDataLoader(
            dataset=dataset,
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            drop_last=False
        )
        
        # Test iteration
        print("Testing data loader:")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}:")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            print(f"  Sequence lengths: {batch['sequence_lengths']}")
            
            if i >= 2:  # Test a few batches
                break
        
        print("Data loader test completed successfully!")
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file)
