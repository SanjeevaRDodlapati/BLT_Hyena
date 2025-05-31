"""
Unit tests for Hyena-GLT data processing components.
"""

from typing import List

import numpy as np
import pytest
import torch

from hyena_glt.data.dataset import GenomicDataset, SequenceClassificationDataset
from hyena_glt.data.tokenizer import DNATokenizer, ProteinTokenizer, RNATokenizer
from hyena_glt.data.utils import (
    calculate_gc_content,
    one_hot_encode,
    reverse_complement,
    sliding_window,
    translate_dna_to_protein,
)
from tests.utils import DataGenerator, TestConfig


class TestDNATokenizer:
    """Test DNA tokenizer functionality."""

    def test_basic_tokenization(self):
        """Test basic DNA sequence tokenization."""
        tokenizer = DNATokenizer()

        # Test simple sequence
        sequence = "ATCG"
        tokens = tokenizer.encode(sequence)

        assert isinstance(tokens, list)
        assert len(tokens) == len(sequence)
        assert all(isinstance(token, int) for token in tokens)

        # Test decoding
        decoded = tokenizer.decode(tokens)
        assert decoded == sequence

    def test_vocab_consistency(self):
        """Test vocabulary consistency."""
        tokenizer = DNATokenizer()

        # Check basic nucleotides
        nucleotides = ["A", "T", "C", "G"]
        for nucleotide in nucleotides:
            token = tokenizer.encode(nucleotide)[0]
            decoded = tokenizer.decode([token])
            assert decoded == nucleotide

    def test_special_tokens(self):
        """Test special token handling."""
        tokenizer = DNATokenizer()

        # Test padding token
        if hasattr(tokenizer, "pad_token"):
            pad_token_id = tokenizer.pad_token_id
            assert isinstance(pad_token_id, int)

        # Test unknown token
        if hasattr(tokenizer, "unk_token"):
            unk_token_id = tokenizer.unk_token_id
            assert isinstance(unk_token_id, int)

    def test_sequence_with_padding(self):
        """Test sequence tokenization with padding."""
        tokenizer = DNATokenizer()

        sequences = ["ATCG", "ATCGATCG", "AT"]
        max_length = 10

        # Test batch encoding with padding
        if hasattr(tokenizer, "encode_batch"):
            tokens = tokenizer.encode_batch(sequences, max_length=max_length)

            assert len(tokens) == len(sequences)
            assert all(len(seq_tokens) == max_length for seq_tokens in tokens)

    def test_invalid_characters(self):
        """Test handling of invalid DNA characters."""
        tokenizer = DNATokenizer()

        # Test sequence with invalid character
        invalid_sequence = "ATCGX"

        # Should either handle gracefully or raise appropriate error
        try:
            tokens = tokenizer.encode(invalid_sequence)
            assert isinstance(tokens, list)
        except (ValueError, KeyError):
            # Expected behavior for invalid characters
            pass

    @pytest.mark.parametrize("sequence", ["ATCG", "AAAAA", "CCCCC", "ATCGATCGATCG", ""])
    def test_encode_decode_consistency(self, sequence):
        """Test encode-decode consistency for various sequences."""
        tokenizer = DNATokenizer()

        if sequence:  # Non-empty sequence
            tokens = tokenizer.encode(sequence)
            decoded = tokenizer.decode(tokens)
            assert decoded == sequence
        else:  # Empty sequence
            tokens = tokenizer.encode(sequence)
            assert tokens == []


class TestProteinTokenizer:
    """Test protein tokenizer functionality."""

    def test_basic_tokenization(self):
        """Test basic protein sequence tokenization."""
        tokenizer = ProteinTokenizer()

        # Test simple sequence
        sequence = "MKTL"
        tokens = tokenizer.encode(sequence)

        assert isinstance(tokens, list)
        assert len(tokens) == len(sequence)
        assert all(isinstance(token, int) for token in tokens)

        # Test decoding
        decoded = tokenizer.decode(tokens)
        assert decoded == sequence

    def test_amino_acid_vocab(self):
        """Test amino acid vocabulary."""
        tokenizer = ProteinTokenizer()

        # Standard amino acids
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        for aa in amino_acids:
            tokens = tokenizer.encode(aa)
            decoded = tokenizer.decode(tokens)
            assert decoded == aa

    @pytest.mark.parametrize(
        "sequence", ["MKTL", "AAAAA", "MKTLQIALPLAALLLLLLAAQFVAQEYQKGK", ""]
    )
    def test_protein_encode_decode(self, sequence):
        """Test protein encode-decode consistency."""
        tokenizer = ProteinTokenizer()

        if sequence:
            tokens = tokenizer.encode(sequence)
            decoded = tokenizer.decode(tokens)
            assert decoded == sequence


class TestRNATokenizer:
    """Test RNA tokenizer functionality."""

    def test_basic_rna_tokenization(self):
        """Test basic RNA sequence tokenization."""
        tokenizer = RNATokenizer()

        # Test RNA sequence (with U instead of T)
        sequence = "AUCG"
        tokens = tokenizer.encode(sequence)

        assert isinstance(tokens, list)
        assert len(tokens) == len(sequence)

        decoded = tokenizer.decode(tokens)
        assert decoded == sequence

    def test_rna_nucleotides(self):
        """Test RNA nucleotide vocabulary."""
        tokenizer = RNATokenizer()

        # RNA nucleotides
        nucleotides = ["A", "U", "C", "G"]

        for nucleotide in nucleotides:
            tokens = tokenizer.encode(nucleotide)
            decoded = tokenizer.decode(tokens)
            assert decoded == nucleotide


class TestGenomicUtils:
    """Test genomic utility functions."""

    def test_reverse_complement(self):
        """Test DNA reverse complement calculation."""
        # Test basic sequences
        assert reverse_complement("ATCG") == "CGAT"
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("CCCC") == "GGGG"
        assert reverse_complement("") == ""

        # Test longer sequence
        sequence = "ATCGATCGATCG"
        rc = reverse_complement(sequence)
        assert len(rc) == len(sequence)
        assert reverse_complement(rc) == sequence  # Double complement

    def test_gc_content(self):
        """Test GC content calculation."""
        # Test known sequences
        assert calculate_gc_content("ATCG") == 0.5  # 2 GC out of 4
        assert calculate_gc_content("AAAA") == 0.0  # No GC
        assert calculate_gc_content("CCCC") == 1.0  # All GC
        assert calculate_gc_content("ATATATAT") == 0.0  # No GC

        # Test empty sequence
        assert calculate_gc_content("") == 0.0

    def test_one_hot_encoding(self):
        """Test one-hot encoding of sequences."""
        # Test DNA encoding
        sequence = "ATCG"
        encoded = one_hot_encode(sequence, alphabet="ATCG")

        assert encoded.shape == (len(sequence), 4)
        assert torch.all(encoded.sum(dim=1) == 1)  # Each position sums to 1

        # Test specific positions
        assert torch.argmax(encoded[0]) == 0  # A
        assert torch.argmax(encoded[1]) == 1  # T
        assert torch.argmax(encoded[2]) == 2  # C
        assert torch.argmax(encoded[3]) == 3  # G

    def test_sliding_window(self):
        """Test sliding window generation."""
        sequence = "ATCGATCG"
        window_size = 4
        stride = 2

        windows = sliding_window(sequence, window_size, stride)

        assert len(windows) > 0
        assert all(len(window) == window_size for window in windows)

        # Check first few windows
        assert windows[0] == "ATCG"
        assert windows[1] == "CGAT"

    def test_translate_dna_to_protein(self):
        """Test DNA to protein translation."""
        # Test known codon
        dna = "ATG"  # Start codon -> Methionine
        protein = translate_dna_to_protein(dna)
        assert protein == "M"

        # Test multiple codons
        dna = "ATGAAATAG"  # ATG (M) + AAA (K) + TAG (stop)
        protein = translate_dna_to_protein(dna)
        assert "M" in protein
        assert "K" in protein

        # Test incomplete codon handling
        dna = "ATGAA"  # Incomplete last codon
        protein = translate_dna_to_protein(dna)
        assert len(protein) == 1  # Only complete codons translated


class TestGenomicDataset:
    """Test genomic dataset functionality."""

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        sequences = ["ATCG", "GCTA", "TTTT"]
        labels = [0, 1, 0]

        dataset = SequenceClassificationDataset(sequences, labels)

        assert len(dataset) == 3
        assert dataset[0]["input_ids"] is not None
        assert dataset[0]["labels"] == 0

    def test_dataset_with_tokenizer(self):
        """Test dataset with custom tokenizer."""
        sequences = ["ATCG", "GCTA", "TTTT"]
        labels = [0, 1, 0]
        tokenizer = DNATokenizer()

        dataset = SequenceClassificationDataset(sequences, labels, tokenizer=tokenizer)

        item = dataset[0]
        assert isinstance(item["input_ids"], torch.Tensor)
        assert item["input_ids"].dtype == torch.long

    def test_dataset_padding(self):
        """Test dataset padding functionality."""
        sequences = ["ATCG", "ATCGATCGATCG", "AT"]
        labels = [0, 1, 0]

        dataset = SequenceClassificationDataset(sequences, labels, max_length=10)

        # All sequences should be padded to max_length
        for i in range(len(dataset)):
            item = dataset[i]
            assert len(item["input_ids"]) == 10

    def test_dataset_iteration(self):
        """Test dataset iteration."""
        sequences = ["ATCG", "GCTA", "TTTT"]
        labels = [0, 1, 0]

        dataset = SequenceClassificationDataset(sequences, labels)

        items = list(dataset)
        assert len(items) == 3

        for item in items:
            assert "input_ids" in item
            assert "labels" in item

    @pytest.mark.parametrize("max_length", [None, 8, 16, 32])
    def test_dataset_different_lengths(self, max_length):
        """Test dataset with different max lengths."""
        sequences = ["ATCG", "ATCGATCGATCG", "AT"]
        labels = [0, 1, 0]

        dataset = SequenceClassificationDataset(
            sequences, labels, max_length=max_length
        )

        item = dataset[0]
        if max_length is not None:
            assert len(item["input_ids"]) == max_length
        else:
            assert len(item["input_ids"]) >= len(sequences[0])


class TestDataLoading:
    """Test data loading and batch processing."""

    def test_batch_creation(self):
        """Test creating batches from dataset."""
        from torch.utils.data import DataLoader

        sequences = ["ATCG"] * 10
        labels = [0, 1] * 5

        dataset = SequenceClassificationDataset(sequences, labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(dataloader))

        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 4  # Batch size
        assert batch["labels"].shape[0] == 4

    def test_collate_function(self):
        """Test custom collate function for variable length sequences."""
        sequences = ["ATCG", "ATCGATCG", "AT"]
        labels = [0, 1, 0]

        dataset = SequenceClassificationDataset(sequences, labels)

        # Create batch manually
        batch_items = [dataset[i] for i in range(len(dataset))]

        # Should handle variable lengths
        assert len(batch_items) == 3

        # Check individual items
        for item in batch_items:
            assert isinstance(item["input_ids"], torch.Tensor)
            assert isinstance(item["labels"], (int, torch.Tensor))


class TestDataValidation:
    """Test data validation and error handling."""

    def test_sequence_validation(self):
        """Test sequence validation in datasets."""
        # Test with invalid sequences
        invalid_sequences = ["ATCGX", "123", ""]
        labels = [0, 1, 0]

        # Should handle or reject invalid sequences appropriately
        try:
            dataset = SequenceClassificationDataset(invalid_sequences, labels)
            # If creation succeeds, check handling
            item = dataset[0]
            assert "input_ids" in item
        except (ValueError, AssertionError):
            # Expected for invalid data
            pass

    def test_label_validation(self):
        """Test label validation."""
        sequences = ["ATCG", "GCTA", "TTTT"]

        # Test mismatched lengths
        with pytest.raises((ValueError, AssertionError)):
            SequenceClassificationDataset(sequences, [0, 1])  # Too few labels

        # Test invalid label types
        with pytest.raises((ValueError, TypeError)):
            SequenceClassificationDataset(sequences, ["a", "b", "c"])

    def test_empty_dataset(self):
        """Test empty dataset handling."""
        dataset = SequenceClassificationDataset([], [])
        assert len(dataset) == 0

        # Test iteration over empty dataset
        items = list(dataset)
        assert len(items) == 0
