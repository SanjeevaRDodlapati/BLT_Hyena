"""Advanced data collators for genomic sequence batching with efficient padding and processing."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .tokenizer import GenomicTokenizer


@dataclass
class GenomicCollatorOutput:
    """Output structure for genomic collators."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None
    sequence_lengths: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None


class SequenceCollator:
    """
    Enhanced collator for genomic sequences with dynamic padding and batching.
    Optimized for variable-length genomic sequences with efficient memory usage.
    """

    def __init__(
        self,
        tokenizer: GenomicTokenizer,
        max_length: int | None = None,
        padding: bool | str = True,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        return_sequence_lengths: bool = True,
        truncation: bool = True,
        include_metadata: bool = False,
    ):
        """
        Initialize the sequence collator.

        Args:
            tokenizer: Genomic tokenizer instance
            max_length: Maximum sequence length (None for dynamic)
            padding: Whether to pad sequences ('longest', True, False)
            pad_to_multiple_of: Pad to multiple of this value
            return_attention_mask: Whether to return attention masks
            return_token_type_ids: Whether to return token type IDs
            return_sequence_lengths: Whether to return sequence lengths
            truncation: Whether to truncate long sequences
            include_metadata: Whether to include metadata in output
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.return_sequence_lengths = return_sequence_lengths
        self.truncation = truncation
        self.include_metadata = include_metadata

        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)

    def __call__(self, features: list[dict[str, Any]]) -> GenomicCollatorOutput:
        """
        Collate a batch of features into tensors.

        Args:
            features: List of feature dictionaries from dataset

        Returns:
            GenomicCollatorOutput with batched tensors
        """
        # Extract and prepare sequences
        input_ids = [f["input_ids"] for f in features]

        # Handle different input formats
        if isinstance(input_ids[0], list):
            input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        elif isinstance(input_ids[0], np.ndarray):
            input_ids = [torch.from_numpy(ids).long() for ids in input_ids]

        # Get original sequence lengths
        sequence_lengths = torch.tensor(
            [len(ids) for ids in input_ids], dtype=torch.long
        )

        # Determine target length for padding
        if self.max_length is not None:
            target_length = self.max_length
        elif self.padding == "longest" or self.padding is True:
            target_length = max(len(ids) for ids in input_ids)
        else:
            target_length = None

        # Apply padding to multiple constraint
        if target_length and self.pad_to_multiple_of:
            target_length = (
                (target_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad/truncate sequences
        if target_length:
            padded_input_ids = []
            for ids in input_ids:
                if len(ids) > target_length and self.truncation:
                    # Truncate from the end
                    ids = ids[:target_length]
                elif len(ids) < target_length:
                    # Pad with pad_token_id
                    padding_length = target_length - len(ids)
                    ids = torch.cat(
                        [
                            ids,
                            torch.full(
                                (padding_length,), self.pad_token_id, dtype=torch.long
                            ),
                        ]
                    )
                padded_input_ids.append(ids)
            input_ids_tensor = torch.stack(padded_input_ids)
        else:
            # No padding - return as list (for models that handle variable lengths)
            input_ids_tensor = (
                torch.stack(input_ids)
                if isinstance(input_ids[0], torch.Tensor)
                else torch.tensor(input_ids)
            )

        # Create attention mask
        attention_mask = None
        if self.return_attention_mask and target_length:
            attention_mask = torch.zeros_like(input_ids_tensor, dtype=torch.long)
            for i, length in enumerate(sequence_lengths):
                actual_length = min(length, target_length)
                attention_mask[i, :actual_length] = 1

        # Handle labels
        labels = None
        if "labels" in features[0]:
            labels_list = [f["labels"] for f in features]

            # Check if labels are for token classification (sequences) or sequence classification (single values)
            if isinstance(labels_list[0], list | np.ndarray | torch.Tensor):
                # Token classification - pad label sequences
                if target_length:
                    padded_labels = []
                    label_pad_id = -100  # Standard ignore index for CrossEntropyLoss

                    for label_seq in labels_list:
                        if isinstance(label_seq, list | np.ndarray):
                            label_seq = torch.tensor(label_seq, dtype=torch.long)

                        if len(label_seq) > target_length and self.truncation:
                            label_seq = label_seq[:target_length]
                        elif len(label_seq) < target_length:
                            padding_length = target_length - len(label_seq)
                            label_seq = torch.cat(
                                [
                                    label_seq,
                                    torch.full(
                                        (padding_length,),
                                        label_pad_id,
                                        dtype=torch.long,
                                    ),
                                ]
                            )
                        padded_labels.append(label_seq)
                    labels = torch.stack(padded_labels)
                else:
                    labels = (
                        torch.stack(labels_list)
                        if isinstance(labels_list[0], torch.Tensor)
                        else torch.tensor(labels_list)
                    )
            else:
                # Sequence classification - simple tensor
                labels = torch.tensor(labels_list, dtype=torch.long)

        # Handle token type IDs (if requested)
        token_type_ids = None
        if self.return_token_type_ids and target_length:
            token_type_ids = torch.zeros_like(input_ids_tensor, dtype=torch.long)

        # Collect metadata
        metadata = None
        if self.include_metadata:
            metadata = {}
            for key in features[0].keys():
                if key not in [
                    "input_ids",
                    "labels",
                    "attention_mask",
                    "token_type_ids",
                ]:
                    metadata[key] = [f.get(key) for f in features]

        return GenomicCollatorOutput(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
            sequence_lengths=sequence_lengths if self.return_sequence_lengths else None,
            metadata=metadata,
        )


class MultiModalCollator:
    """
    Collator for multi-modal genomic data (DNA + protein, DNA + RNA, etc.).
    Handles multiple sequence types in a single batch.
    """

    def __init__(
        self,
        tokenizers: dict[str, GenomicTokenizer],
        modality_key: str = "modality",
        max_lengths: dict[str, int] | None = None,
        padding: bool = True,
        return_attention_mask: bool = True,
        align_lengths: bool = False,
    ):
        """
        Initialize multi-modal collator.

        Args:
            tokenizers: Dict mapping modality names to tokenizers
            modality_key: Key in features indicating the modality
            max_lengths: Maximum lengths per modality
            padding: Whether to pad sequences
            return_attention_mask: Whether to return attention masks
            align_lengths: Whether to align all modalities to same length
        """
        self.tokenizers = tokenizers
        self.modality_key = modality_key
        self.max_lengths = max_lengths or {}
        self.padding = padding
        self.return_attention_mask = return_attention_mask
        self.align_lengths = align_lengths

        # Create individual collators for each modality
        self.collators = {}
        for modality, tokenizer in tokenizers.items():
            self.collators[modality] = SequenceCollator(
                tokenizer=tokenizer,
                max_length=self.max_lengths.get(modality),
                padding=padding,
                return_attention_mask=return_attention_mask,
            )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate multi-modal features.

        Args:
            features: List of feature dictionaries

        Returns:
            Dictionary with collated features per modality
        """
        # Group features by modality
        modality_features: dict[str, list[dict[str, Any]]] = {}
        for feature in features:
            modality = feature.get(self.modality_key, "default")
            if modality not in modality_features:
                modality_features[modality] = []
            modality_features[modality].append(feature)

        # Collate each modality separately
        collated_output = {}
        max_length_across_modalities = 0

        for modality, modal_features in modality_features.items():
            if modality in self.collators:
                modal_output = self.collators[modality](modal_features)
                collated_output[modality] = modal_output

                if self.align_lengths and hasattr(modal_output, "input_ids"):
                    if isinstance(modal_output.input_ids, torch.Tensor):
                        max_length_across_modalities = max(
                            max_length_across_modalities,
                            modal_output.input_ids.shape[1],
                        )

        # Align lengths across modalities if requested
        if self.align_lengths and max_length_across_modalities > 0:
            for modality, modal_output in collated_output.items():
                if hasattr(modal_output, "input_ids") and isinstance(
                    modal_output.input_ids, torch.Tensor
                ):
                    current_length = modal_output.input_ids.shape[1]
                    if current_length < max_length_across_modalities:
                        # Pad to align with longest modality
                        pad_length = max_length_across_modalities - current_length
                        pad_token_id = self.tokenizers[modality].pad_token_id

                        # Pad input_ids
                        padding = torch.full(
                            (modal_output.input_ids.shape[0], pad_length),
                            pad_token_id,
                            dtype=modal_output.input_ids.dtype,
                        )
                        modal_output.input_ids = torch.cat(
                            [modal_output.input_ids, padding], dim=1
                        )

                        # Pad attention_mask if present
                        if modal_output.attention_mask is not None:
                            mask_padding = torch.zeros(
                                (modal_output.attention_mask.shape[0], pad_length),
                                dtype=modal_output.attention_mask.dtype,
                            )
                            modal_output.attention_mask = torch.cat(
                                [modal_output.attention_mask, mask_padding], dim=1
                            )

        return collated_output


class AdaptiveBatchCollator:
    """
    Adaptive collator that optimizes batch composition based on sequence lengths
    to minimize padding overhead and maximize GPU utilization.
    """

    def __init__(
        self,
        tokenizer: GenomicTokenizer,
        target_tokens_per_batch: int = 32768,
        max_sequences_per_batch: int = 64,
        length_tolerance: float = 0.2,
        sorting_strategy: str = "length",
    ):
        """
        Initialize adaptive batch collator.

        Args:
            tokenizer: Genomic tokenizer instance
            target_tokens_per_batch: Target number of tokens per batch
            max_sequences_per_batch: Maximum sequences per batch
            length_tolerance: Tolerance for length-based grouping (0.0-1.0)
            sorting_strategy: Strategy for sorting ('length', 'random', 'none')
        """
        self.tokenizer = tokenizer
        self.target_tokens_per_batch = target_tokens_per_batch
        self.max_sequences_per_batch = max_sequences_per_batch
        self.length_tolerance = length_tolerance
        self.sorting_strategy = sorting_strategy

        self.base_collator = SequenceCollator(
            tokenizer=tokenizer, padding=True, return_attention_mask=True
        )

    def create_adaptive_batches(
        self, features: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """
        Create adaptive batches from features based on length optimization.

        Args:
            features: List of feature dictionaries

        Returns:
            List of batches (each batch is a list of features)
        """
        # Extract sequence lengths
        lengths = []
        for feature in features:
            if isinstance(feature["input_ids"], list | np.ndarray):
                lengths.append(len(feature["input_ids"]))
            elif isinstance(feature["input_ids"], torch.Tensor):
                lengths.append(feature["input_ids"].shape[0])
            else:
                lengths.append(0)

        # Sort by length if requested
        if self.sorting_strategy == "length":
            sorted_indices = sorted(range(len(features)), key=lambda i: lengths[i])
            features = [features[i] for i in sorted_indices]
            lengths = [lengths[i] for i in sorted_indices]
        elif self.sorting_strategy == "random":
            indices = list(range(len(features)))
            np.random.shuffle(indices)
            features = [features[i] for i in indices]
            lengths = [lengths[i] for i in indices]

        # Create adaptive batches
        batches = []
        current_batch: list[dict[str, Any]] = []
        current_max_length = 0

        for _i, (feature, length) in enumerate(zip(features, lengths, strict=False)):
            # Calculate tokens if this feature were added
            proposed_max_length = max(current_max_length, length)
            proposed_batch_size = len(current_batch) + 1
            proposed_tokens = proposed_max_length * proposed_batch_size

            # Check if we should start a new batch
            should_start_new_batch = (
                len(current_batch) >= self.max_sequences_per_batch
                or proposed_tokens > self.target_tokens_per_batch
                or (
                    current_batch
                    and abs(length - current_max_length)
                    / max(length, current_max_length)
                    > self.length_tolerance
                )
            )

            if should_start_new_batch and current_batch:
                batches.append(current_batch)
                current_batch = [feature]
                current_max_length = length
            else:
                current_batch.append(feature)
                current_max_length = proposed_max_length

        # Add the last batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def __call__(self, features: list[dict[str, Any]]) -> list[GenomicCollatorOutput]:
        """
        Create adaptive batches and collate each one.

        Args:
            features: List of feature dictionaries

        Returns:
            List of collated batch outputs
        """
        adaptive_batches = self.create_adaptive_batches(features)
        return [self.base_collator(batch) for batch in adaptive_batches]


class StreamingCollator:
    """
    Streaming collator for very large datasets that can't fit in memory.
    Handles online batching and preprocessing.
    """

    def __init__(
        self,
        tokenizer: GenomicTokenizer,
        batch_size: int = 32,
        max_length: int | None = None,
        buffer_size: int = 1000,
        preprocessing_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        """
        Initialize streaming collator.

        Args:
            tokenizer: Genomic tokenizer instance
            batch_size: Target batch size
            max_length: Maximum sequence length
            buffer_size: Size of internal buffer for sequence sorting
            preprocessing_fn: Optional preprocessing function
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.preprocessing_fn = preprocessing_fn

        self.base_collator = SequenceCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            padding=True,
            return_attention_mask=True,
        )

        self.buffer: list[dict[str, Any]] = []

    def add_to_buffer(self, feature: dict[str, Any]) -> GenomicCollatorOutput | None:
        """
        Add a feature to the buffer and return a batch if ready.

        Args:
            feature: Feature dictionary to add

        Returns:
            Collated batch if buffer is full, None otherwise
        """
        # Apply preprocessing if available
        if self.preprocessing_fn:
            feature = self.preprocessing_fn(feature)

        self.buffer.append(feature)

        # Check if we have enough for a batch
        if len(self.buffer) >= self.batch_size:
            batch_features = self.buffer[: self.batch_size]
            self.buffer = self.buffer[self.batch_size :]
            return self.base_collator(batch_features)

        return None

    def flush_buffer(self) -> GenomicCollatorOutput | None:
        """
        Flush remaining features in buffer as a final batch.

        Returns:
            Collated batch of remaining features, or None if buffer is empty
        """
        if self.buffer:
            batch = self.base_collator(self.buffer)
            self.buffer = []
            return batch
        return None
