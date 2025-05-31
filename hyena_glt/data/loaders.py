"""Advanced data loaders for genomic sequences with multi-modal support and optimization."""

import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
from Bio.SeqRecord import SeqRecord
from torch.utils.data import DataLoader, Dataset, Sampler

from .collators import AdaptiveBatchCollator, MultiModalCollator, SequenceCollator
from .dataset import (
    GenomicDataset,
    TokenClassificationDataset,
)
from .preprocessing import GenomicPreprocessor
from .tokenizer import DNATokenizer, GenomicTokenizer, ProteinTokenizer, RNATokenizer


class LengthGroupedSampler(Sampler):
    """
    Sampler that groups sequences by length to minimize padding overhead.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        length_fn: Callable | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize length-grouped sampler.

        Args:
            dataset: Dataset to sample from
            batch_size: Size of each batch
            length_fn: Function to extract length from dataset item
            shuffle: Whether to shuffle within length groups
            drop_last: Whether to drop incomplete batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.length_fn = length_fn or self._default_length_fn
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Pre-compute lengths and group indices
        self.lengths = []
        for i in range(len(dataset)):  # type: ignore[arg-type]
            try:
                length = self.length_fn(dataset[i])
                self.lengths.append(length)
            except Exception:
                self.lengths.append(0)  # Fallback for problematic items

        self._create_length_groups()

    def _default_length_fn(self, item: dict[str, Any]) -> int:
        """Default function to extract sequence length."""
        if "input_ids" in item:
            if isinstance(item["input_ids"], list | np.ndarray):
                return len(item["input_ids"])  # type: ignore[no-any-return]
            elif hasattr(item["input_ids"], "shape"):
                return item["input_ids"].shape[0]  # type: ignore[no-any-return]
        return 0

    def _create_length_groups(self) -> None:
        """Group indices by sequence length."""
        length_to_indices = defaultdict(list)
        for idx, length in enumerate(self.lengths):
            length_to_indices[length].append(idx)

        self.length_groups = list(length_to_indices.values())

        # Shuffle groups if requested
        if self.shuffle:
            for group in self.length_groups:
                random.shuffle(group)
            random.shuffle(self.length_groups)

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batches of indices."""
        all_batches = []

        for group in self.length_groups:
            # Create batches within each length group
            for i in range(0, len(group), self.batch_size):
                batch = group[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle batches if requested
        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        total_batches = 0
        for group in self.length_groups:
            if self.drop_last:
                total_batches += len(group) // self.batch_size
            else:
                total_batches += (len(group) + self.batch_size - 1) // self.batch_size
        return total_batches


class MultiModalSampler(Sampler):
    """
    Sampler for multi-modal datasets that ensures balanced sampling across modalities.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        modality_key: str = "modality",
        balance_modalities: bool = True,
        shuffle: bool = True,
    ):
        """
        Initialize multi-modal sampler.

        Args:
            dataset: Dataset to sample from
            batch_size: Size of each batch
            modality_key: Key indicating the modality in dataset items
            balance_modalities: Whether to balance modalities in each batch
            shuffle: Whether to shuffle samples
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.modality_key = modality_key
        self.balance_modalities = balance_modalities
        self.shuffle = shuffle

        self._group_by_modality()

    def _group_by_modality(self) -> None:
        """Group dataset indices by modality."""
        self.modality_groups = defaultdict(list)

        for idx in range(len(self.dataset)):  # type: ignore[arg-type]
            try:
                item = self.dataset[idx]
                modality = item.get(self.modality_key, "default")
                self.modality_groups[modality].append(idx)
            except Exception:
                self.modality_groups["default"].append(idx)

        self.modalities = list(self.modality_groups.keys())

        # Shuffle within modalities if requested
        if self.shuffle:
            for indices in self.modality_groups.values():
                random.shuffle(indices)

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over balanced batches."""
        if self.balance_modalities and len(self.modalities) > 1:
            # Create balanced batches
            modality_iterators = {
                modality: iter(indices)
                for modality, indices in self.modality_groups.items()
            }

            samples_per_modality = self.batch_size // len(self.modalities)
            remainder = self.batch_size % len(self.modalities)

            while any(modality_iterators.values()):
                batch = []

                for i, modality in enumerate(self.modalities):
                    iterator = modality_iterators[modality]
                    num_samples = samples_per_modality + (1 if i < remainder else 0)

                    for _ in range(num_samples):
                        try:
                            batch.append(next(iterator))
                        except StopIteration:
                            modality_iterators[modality] = iter([])  # Empty iterator
                            break

                if len(batch) > 0:
                    if self.shuffle:
                        random.shuffle(batch)
                    yield batch
        else:
            # Simple batching without balancing
            all_indices = []
            for indices in self.modality_groups.values():
                all_indices.extend(indices)

            if self.shuffle:
                random.shuffle(all_indices)

            for i in range(0, len(all_indices), self.batch_size):
                yield all_indices[i : i + self.batch_size]

    def __len__(self) -> int:
        """Return number of batches."""
        total_samples = sum(len(indices) for indices in self.modality_groups.values())
        return (total_samples + self.batch_size - 1) // self.batch_size


class GenomicDataLoader:
    """
    Enhanced data loader for genomic sequences with built-in optimization and preprocessing.
    """

    def __init__(
        self,
        dataset: Dataset | str | Path,
        tokenizer: GenomicTokenizer,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        max_length: int | None = None,
        padding: bool = True,
        preprocessing: GenomicPreprocessor | None = None,
        length_grouped: bool = False,
        adaptive_batching: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        """
        Initialize genomic data loader.

        Args:
            dataset: Dataset instance or path to data file
            tokenizer: Genomic tokenizer instance
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            collate_fn: Custom collate function
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            preprocessing: Optional preprocessing pipeline
            length_grouped: Whether to group by sequence length
            adaptive_batching: Whether to use adaptive batching
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop incomplete batches
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.max_length = max_length
        self.padding = padding
        self.preprocessing = preprocessing
        self.length_grouped = length_grouped
        self.adaptive_batching = adaptive_batching
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # Declare sampler attribute with proper type
        self.sampler: LengthGroupedSampler | None

        # Create or load dataset
        if isinstance(dataset, str | Path):
            self.dataset = self._load_dataset_from_file(dataset)
        else:
            self.dataset = dataset

        # Set up collate function
        if collate_fn is not None:
            self.collate_fn = collate_fn
        elif adaptive_batching:
            self.collate_fn = AdaptiveBatchCollator(
                tokenizer=tokenizer,
                target_tokens_per_batch=batch_size * (max_length or 512),
            )
        else:
            self.collate_fn = SequenceCollator(
                tokenizer=tokenizer,
                max_length=max_length,
                padding=padding,
                return_attention_mask=True,
            )

        # Set up sampler
        if length_grouped:
            self.sampler = LengthGroupedSampler(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            # Override shuffle since sampler handles it
            self.shuffle = False
        else:
            self.sampler = None

    def _load_dataset_from_file(self, file_path: str | Path) -> Dataset:
        """Load dataset from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Determine file format and load accordingly
        if file_path.suffix.lower() == ".jsonl":
            return self._load_jsonl_dataset(file_path)
        elif file_path.suffix.lower() in [".fa", ".fasta"]:
            return self._load_fasta_dataset(file_path)
        elif file_path.suffix.lower() in [".fq", ".fastq"]:
            return self._load_fastq_dataset(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_jsonl_dataset(self, file_path: Path) -> Dataset:
        """Load dataset from JSONL file."""
        data = []
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        # Determine dataset type based on data structure
        if data and "labels" in data[0] and isinstance(data[0]["labels"], list):
            # Token classification
            return TokenClassificationDataset(
                data=data,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
        else:
            # Sequence classification or other tasks
            return GenomicDataset(
                data=data, tokenizer=self.tokenizer, max_length=self.max_length or 512
            )

    def _load_fasta_dataset(self, file_path: Path) -> Dataset:
        """Load dataset from FASTA file."""
        try:
            from Bio import SeqIO

            sequences = []
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append(str(record.seq))

            # Create data in the expected format
            data = [{"sequence": seq} for seq in sequences]

            return GenomicDataset(
                data=data,
                tokenizer=self.tokenizer,
                max_length=self.max_length or 512,
            )
        except ImportError as e:
            raise ImportError("BioPython required for FASTA file loading") from e

    def _load_fastq_dataset(self, file_path: Path) -> Dataset:
        """Load dataset from FASTQ file."""
        try:
            from Bio import SeqIO

            sequences = []
            qualities = []
            for record in SeqIO.parse(file_path, "fastq"):
                sequences.append(str(record.seq))
                qualities.append(record.letter_annotations.get("phred_quality", []))

            # Apply preprocessing if available
            if self.preprocessing:
                sequences, _, _ = self.preprocessing.preprocess_sequences(
                    cast(list[str | SeqRecord], sequences), qualities=qualities
                )

            # Create data in the expected format
            data = [{"sequence": seq} for seq in sequences]

            return GenomicDataset(
                data=data,
                tokenizer=self.tokenizer,
                max_length=self.max_length or 512,
            )
        except ImportError as e:
            raise ImportError("BioPython required for FASTQ file loading") from e

    def get_dataloader(self) -> DataLoader:
        """Get PyTorch DataLoader instance."""
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size if not self.adaptive_batching else 1,
            shuffle=self.shuffle,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def __iter__(self) -> Iterator[Any]:
        """Make this class iterable."""
        return iter(self.get_dataloader())

    def __len__(self) -> int:
        """Return number of batches."""
        if self.sampler:
            return len(self.sampler)
        else:
            dataset_size = len(self.dataset)  # type: ignore[arg-type]
            if self.drop_last:
                return dataset_size // self.batch_size
            else:
                return (dataset_size + self.batch_size - 1) // self.batch_size


class MultiModalDataLoader:
    """
    Data loader for multi-modal genomic data (DNA, RNA, protein combinations).
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
        tokenizers: dict[str, GenomicTokenizer],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        balance_modalities: bool = True,
        max_lengths: dict[str, int] | None = None,
        align_lengths: bool = False,
    ):
        """
        Initialize multi-modal data loader.

        Args:
            datasets: Dict mapping modality names to datasets
            tokenizers: Dict mapping modality names to tokenizers
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            balance_modalities: Whether to balance modalities in batches
            max_lengths: Maximum lengths per modality
            align_lengths: Whether to align sequence lengths across modalities
        """
        self.datasets = datasets
        self.tokenizers = tokenizers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.balance_modalities = balance_modalities
        self.max_lengths = max_lengths or {}
        self.align_lengths = align_lengths

        # Declare sampler attribute with proper type
        self.sampler: MultiModalSampler | None

        # Create combined dataset
        self.combined_dataset = self._create_combined_dataset()

        # Set up collate function
        self.collate_fn = MultiModalCollator(
            tokenizers=tokenizers, max_lengths=max_lengths, align_lengths=align_lengths
        )

        # Set up sampler
        if balance_modalities:
            self.sampler = MultiModalSampler(
                dataset=self.combined_dataset, batch_size=batch_size, shuffle=shuffle
            )
            self.shuffle = False  # Sampler handles shuffling
        else:
            self.sampler = None

    def _create_combined_dataset(self) -> Dataset:
        """Create a combined dataset from multiple modalities."""

        class CombinedMultiModalDataset(Dataset):
            def __init__(self, datasets: dict[str, Dataset]):
                self.datasets = datasets
                self.modality_offsets = {}
                self.total_size = 0

                # Calculate offsets for each modality
                for modality, dataset in datasets.items():
                    self.modality_offsets[modality] = self.total_size
                    self.total_size += len(dataset)  # type: ignore[arg-type]

            def __len__(self) -> int:
                return self.total_size

            def __getitem__(self, idx: int) -> dict[str, Any]:
                # Find which modality this index belongs to
                for modality, offset in self.modality_offsets.items():
                    dataset = self.datasets[modality]
                    if idx < offset + len(dataset):  # type: ignore[arg-type]
                        item = dataset[idx - offset]
                        item["modality"] = modality
                        return item  # type: ignore[no-any-return]

                raise IndexError(f"Index {idx} out of range")

        return CombinedMultiModalDataset(self.datasets)

    def get_dataloader(self) -> DataLoader:
        """Get PyTorch DataLoader instance."""
        return DataLoader(
            dataset=self.combined_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def __iter__(self) -> Iterator[Any]:
        """Make this class iterable."""
        return iter(self.get_dataloader())

    def __len__(self) -> int:
        """Return number of batches."""
        if self.sampler:
            return len(self.sampler)
        else:
            return (len(self.combined_dataset) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


class StreamingDataLoader:
    """
    Data loader for streaming very large datasets that don't fit in memory.
    """

    def __init__(
        self,
        data_source: str | Path | Iterator,
        tokenizer: GenomicTokenizer,
        batch_size: int = 32,
        max_length: int | None = None,
        buffer_size: int = 10000,
        preprocessing: GenomicPreprocessor | None = None,
        num_workers: int = 0,
    ):
        """
        Initialize streaming data loader.

        Args:
            data_source: Source of streaming data (file path or iterator)
            tokenizer: Genomic tokenizer instance
            batch_size: Batch size for training
            max_length: Maximum sequence length
            buffer_size: Size of internal buffer
            preprocessing: Optional preprocessing pipeline
            num_workers: Number of worker processes
        """
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.preprocessing = preprocessing
        self.num_workers = num_workers

        self.collate_fn = SequenceCollator(
            tokenizer=tokenizer, max_length=max_length, padding=True
        )

    def _create_data_iterator(self) -> Iterator[dict[str, Any]]:
        """Create iterator over data source."""
        if isinstance(self.data_source, str | Path):
            file_path = Path(self.data_source)

            if file_path.suffix.lower() == ".jsonl":
                return self._jsonl_iterator(file_path)
            elif file_path.suffix.lower() in [".fa", ".fasta"]:
                return self._fasta_iterator(file_path)
            else:
                raise ValueError(f"Unsupported streaming format: {file_path.suffix}")
        else:
            return self.data_source

    def _jsonl_iterator(self, file_path: Path) -> Iterator[dict[str, Any]]:
        """Create iterator over JSONL file."""
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def _fasta_iterator(self, file_path: Path) -> Iterator[dict[str, Any]]:
        """Create iterator over FASTA file."""
        try:
            from Bio import SeqIO

            for record in SeqIO.parse(file_path, "fasta"):
                yield {
                    "sequence": str(record.seq),
                    "sequence_id": record.id,
                    "description": record.description,
                }
        except ImportError as e:
            raise ImportError("BioPython required for FASTA streaming") from e

    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches from streaming data."""
        data_iterator = self._create_data_iterator()
        buffer = []

        for item in data_iterator:
            # Apply preprocessing if available
            if self.preprocessing:
                try:
                    sequences, headers, _ = self.preprocessing.preprocess_sequences(
                        sequences=[item.get("sequence", "")],
                        headers=[item.get("sequence_id", "")],
                    )
                    if sequences:
                        item["sequence"] = sequences[0]
                        item["sequence_id"] = headers[0]
                    else:
                        continue  # Skip filtered sequences
                except Exception:
                    continue  # Skip problematic sequences

            # Tokenize sequence
            if "sequence" in item:
                tokens = self.tokenizer.encode(item["sequence"])
                item["input_ids"] = tokens

            buffer.append(item)

            # Yield batch when buffer is full
            if len(buffer) >= self.batch_size:
                batch = buffer[: self.batch_size]
                buffer = buffer[self.batch_size :]
                yield self.collate_fn(batch)

        # Yield remaining items in buffer
        if buffer:
            yield self.collate_fn(buffer)


def create_genomic_dataloaders(
    train_data: Dataset | str | Path,
    val_data: Dataset | str | Path | None = None,
    test_data: Dataset | str | Path | None = None,
    tokenizer: GenomicTokenizer | None = None,
    sequence_type: str = "dna",
    batch_size: int = 32,
    max_length: int | None = None,
    num_workers: int = 0,
    preprocessing: GenomicPreprocessor | None = None,
    **kwargs: Any,
) -> dict[str, GenomicDataLoader]:
    """
    Convenience function to create train/validation/test data loaders.

    Args:
        train_data: Training dataset or path
        val_data: Validation dataset or path (optional)
        test_data: Test dataset or path (optional)
        tokenizer: Genomic tokenizer (auto-created if None)
        sequence_type: Type of sequences ('dna', 'rna', 'protein')
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        preprocessing: Optional preprocessing pipeline
        **kwargs: Additional arguments for GenomicDataLoader

    Returns:
        Dictionary with train/val/test data loaders
    """
    if tokenizer is None:
        if sequence_type == "dna":
            tokenizer = DNATokenizer()
        elif sequence_type == "rna":
            tokenizer = RNATokenizer()
        elif sequence_type == "protein":
            tokenizer = ProteinTokenizer()
        else:
            raise ValueError(f"Unsupported sequence type: {sequence_type}")

    loaders = {}

    # Create dataset objects if raw data is provided
    if not isinstance(train_data, Dataset):
        train_dataset = GenomicDataset(
            data=train_data,
            tokenizer=tokenizer,
            max_length=max_length or tokenizer.model_max_length,
        )
    else:
        train_dataset = cast(GenomicDataset, train_data)

    # Training loader
    loaders["train"] = GenomicDataLoader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_length=max_length,
        preprocessing=preprocessing,
        **kwargs,
    )

    # Validation loader
    if val_data is not None:
        if not isinstance(val_data, Dataset):
            val_dataset = GenomicDataset(
                data=val_data,
                tokenizer=tokenizer,
                max_length=max_length or tokenizer.model_max_length,
            )
        else:
            val_dataset = cast(GenomicDataset, val_data)

        loaders["val"] = GenomicDataLoader(
            dataset=val_dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            max_length=max_length,
            preprocessing=preprocessing,
            **kwargs,
        )

    # Test loader
    if test_data is not None:
        if not isinstance(test_data, Dataset):
            test_dataset = GenomicDataset(
                data=test_data,
                tokenizer=tokenizer,
                max_length=max_length or tokenizer.model_max_length,
            )
        else:
            test_dataset = cast(GenomicDataset, test_data)

        loaders["test"] = GenomicDataLoader(
            dataset=test_dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            max_length=max_length,
            preprocessing=preprocessing,
            **kwargs,
        )

    return loaders
