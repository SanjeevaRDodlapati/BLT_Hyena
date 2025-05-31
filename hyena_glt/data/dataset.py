"""Dataset classes for genomic sequence data."""

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import GenomicTokenizer
from .utils import mask_sequence, reverse_complement


class GenomicDataset(Dataset[dict[str, Any]]):
    """Base dataset class for genomic sequences."""

    def __init__(
        self,
        data: list[dict[str, Any]] | str | Path,
        tokenizer: GenomicTokenizer,
        max_length: int = 512,
        include_reverse_complement: bool = False,
        augment_data: bool = False,
        mask_probability: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_reverse_complement = include_reverse_complement
        self.augment_data = augment_data
        self.mask_probability = mask_probability

        # Load data
        if isinstance(data, str | Path):
            self.data = self._load_data(data)
        else:
            self.data = data

        # Validate data format
        self._validate_data()

        # Apply augmentation if requested
        if self.include_reverse_complement:
            self._add_reverse_complements()

    def _load_data(self, data_path: str | Path) -> list[dict[Any, Any]]:
        """Load data from file."""
        data_path = Path(data_path)

        if data_path.suffix == ".json":
            with open(data_path) as f:
                data: list[dict[Any, Any]] = json.load(f)
                return data
        elif data_path.suffix == ".jsonl":
            data = []
            with open(data_path) as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        elif data_path.suffix in [".csv", ".tsv"]:
            df = pd.read_csv(data_path, sep="\t" if data_path.suffix == ".tsv" else ",")
            records: list[dict[Any, Any]] = df.to_dict("records")
            return records
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def _validate_data(self) -> None:
        """Validate that data has required fields."""
        if not self.data:
            raise ValueError("Dataset is empty")

        required_fields = {"sequence"}
        sample = self.data[0]

        if not isinstance(sample, dict):
            raise ValueError("Data samples must be dictionaries")

        missing_fields = required_fields - set(sample.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

    def _add_reverse_complements(self) -> None:
        """Add reverse complement sequences to dataset."""
        if self.tokenizer.sequence_type not in ["dna"]:
            return

        original_data = self.data.copy()
        for sample in original_data:
            rc_sample = sample.copy()
            rc_sample["sequence"] = reverse_complement(sample["sequence"])
            rc_sample["is_reverse_complement"] = True
            self.data.append(rc_sample)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.data[idx].copy()
        sequence = sample["sequence"]

        # Apply masking if specified
        if self.mask_probability > 0:
            sequence, masked_positions = mask_sequence(
                sequence, self.mask_probability, self.tokenizer.mask_token
            )
            sample["masked_positions"] = masked_positions

        # Tokenize sequence
        encoding = self.tokenizer.encode_plus(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare output
        output = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sequence": sequence,
            "original_sequence": sample["sequence"],
        }

        # Add additional fields from sample
        for key, value in sample.items():
            if key not in ["sequence"]:
                output[key] = value

        return output


class SequenceClassificationDataset(GenomicDataset):
    """Dataset for sequence classification tasks."""

    def __init__(
        self,
        data: list[dict[Any, Any]] | str | Path,
        tokenizer: GenomicTokenizer,
        label_column: str = "label",
        num_classes: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.label_column = label_column
        super().__init__(data, tokenizer, **kwargs)

        # Process labels
        self.labels = [sample[label_column] for sample in self.data]
        self.label2id, self.id2label = self._create_label_mappings()
        self.num_classes = num_classes or len(self.label2id)

    def _validate_data(self) -> None:
        """Validate data with labels."""
        super()._validate_data()

        if self.label_column not in self.data[0]:
            raise ValueError(f"Label column '{self.label_column}' not found in data")

    def _create_label_mappings(self) -> tuple[dict[str, int], dict[int, str]]:
        """Create label to ID mappings."""
        unique_labels = sorted(set(self.labels))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        return label2id, id2label

    def __getitem__(self, idx: int) -> dict[str, Any]:
        output = super().__getitem__(idx)

        # Add label information
        label = self.data[idx][self.label_column]
        output["labels"] = torch.tensor(self.label2id[label], dtype=torch.long)
        output["label_text"] = label

        return output


class TokenClassificationDataset(GenomicDataset):
    """Dataset for token-level classification tasks (e.g., gene annotation)."""

    def __init__(
        self,
        data: list[dict[Any, Any]] | str | Path,
        tokenizer: GenomicTokenizer,
        labels_column: str = "labels",
        **kwargs: Any,
    ) -> None:
        self.labels_column = labels_column
        super().__init__(data, tokenizer, **kwargs)

        # Process label vocabulary
        all_labels = []
        for sample in self.data:
            if self.labels_column in sample:
                all_labels.extend(sample[self.labels_column])

        self.label2id, self.id2label = self._create_label_mappings(all_labels)
        self.num_classes = len(self.label2id)

    def _validate_data(self) -> None:
        """Validate data with token labels."""
        super()._validate_data()

        # Check if at least some samples have labels
        has_labels = any(self.labels_column in sample for sample in self.data)
        if not has_labels:
            print(f"Warning: No samples found with '{self.labels_column}' column")

    def _create_label_mappings(
        self, all_labels: list[str]
    ) -> tuple[dict[str, int], dict[int, str]]:
        """Create label mappings for token classification."""
        # Add special labels
        special_labels = ["O", "PAD"]  # Outside, Padding
        unique_labels = special_labels + sorted(set(all_labels) - set(special_labels))

        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        return label2id, id2label

    def __getitem__(self, idx: int) -> dict[str, Any]:
        output = super().__getitem__(idx)

        # Handle token labels if available
        if self.labels_column in self.data[idx]:
            token_labels = self.data[idx][self.labels_column]

            # Align labels with tokenized sequence
            sequence_length = output["input_ids"].size(0)
            aligned_labels = self._align_labels_with_tokens(
                token_labels, sequence_length, output["sequence"]
            )

            output["labels"] = torch.tensor(aligned_labels, dtype=torch.long)
        else:
            # No labels available (inference mode)
            output["labels"] = torch.full(
                (output["input_ids"].size(0),), self.label2id["PAD"], dtype=torch.long
            )

        return output

    def _align_labels_with_tokens(
        self, token_labels: list[str], sequence_length: int, tokenized_sequence: str
    ) -> list[int]:
        """Align token labels with tokenized sequence."""
        # Simple alignment - can be improved based on specific requirements
        aligned_labels = []

        # Handle special tokens
        aligned_labels.append(self.label2id["O"])  # CLS token

        # Align remaining labels
        label_idx = 0
        for _i in range(1, sequence_length - 1):  # Skip CLS and SEP
            if label_idx < len(token_labels):
                label = token_labels[label_idx]
                aligned_labels.append(self.label2id.get(label, self.label2id["O"]))
                label_idx += 1
            else:
                aligned_labels.append(self.label2id["PAD"])

        # SEP token
        aligned_labels.append(self.label2id["O"])

        # Pad to sequence length
        while len(aligned_labels) < sequence_length:
            aligned_labels.append(self.label2id["PAD"])

        return aligned_labels[:sequence_length]


class SequenceGenerationDataset(GenomicDataset):
    """Dataset for sequence generation tasks."""

    def __init__(
        self,
        data: list[dict[Any, Any]] | str | Path,
        tokenizer: GenomicTokenizer,
        target_column: str = "target",
        **kwargs: Any,
    ) -> None:
        self.target_column = target_column
        super().__init__(data, tokenizer, **kwargs)

    def _validate_data(self) -> None:
        """Validate data with targets."""
        super()._validate_data()

        # Check if samples have target sequences
        has_targets = any(self.target_column in sample for sample in self.data)
        if not has_targets:
            print(f"Warning: No samples found with '{self.target_column}' column")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        output = super().__getitem__(idx)

        # Add target sequence if available
        if self.target_column in self.data[idx]:
            target_sequence = self.data[idx][self.target_column]

            # Tokenize target
            target_encoding = self.tokenizer.encode_plus(
                target_sequence,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            output["labels"] = target_encoding["input_ids"].squeeze(0)
            output["target_sequence"] = target_sequence

        return output


class MultiTaskDataset(Dataset[dict[str, Any]]):
    """Dataset that combines multiple genomic tasks."""

    def __init__(
        self,
        datasets: dict[str, GenomicDataset],
        task_sampling_strategy: str = "round_robin",  # 'round_robin', 'weighted', 'uniform'
        task_weights: dict[str, float] | None = None,
    ):
        self.datasets = datasets
        self.task_sampling_strategy = task_sampling_strategy
        self.task_weights = task_weights or dict.fromkeys(datasets.keys(), 1.0)

        # Calculate total length and task boundaries
        self.task_lengths = {task: len(dataset) for task, dataset in datasets.items()}
        self.total_length = sum(self.task_lengths.values())

        # Create task sampling indices
        self._create_sampling_indices()

    def _create_sampling_indices(self) -> None:
        """Create indices for task sampling."""
        if self.task_sampling_strategy == "round_robin":
            self.indices = []
            max_length = max(self.task_lengths.values())

            for i in range(max_length):
                for task in self.datasets.keys():
                    if i < self.task_lengths[task]:
                        self.indices.append((task, i))

        elif self.task_sampling_strategy == "weighted":
            self.indices = []
            total_weight = sum(self.task_weights.values())

            for task, _dataset in self.datasets.items():
                weight = self.task_weights[task] / total_weight
                num_samples = int(weight * self.total_length)

                for i in range(num_samples):
                    idx = i % self.task_lengths[task]
                    self.indices.append((task, idx))

        else:  # uniform
            self.indices = []
            for task, _dataset in self.datasets.items():
                for i in range(self.task_lengths[task]):
                    self.indices.append((task, i))

            random.shuffle(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task, task_idx = self.indices[idx]
        sample = self.datasets[task][task_idx]

        # Add task information
        sample["task"] = task
        sample["task_id"] = list(self.datasets.keys()).index(task)

        return sample
