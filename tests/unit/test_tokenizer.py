"""
Unit tests for BLT tokenizer components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from hyena_glt.data.tokenizer import (
    BLTTokenizer, DNATokenizer, ProteinTokenizer, RNATokenizer,
    SpecialTokens, TokenizerConfig
)
from hyena_glt.data.blt_layers import (
    ByteLatentTransformer, TokenMerger, AdaptiveEncoder
)
from tests.utils import TestConfig, DataGenerator


class TestBLTTokenizer:
    """Test BLT tokenizer base class."""
    
    def test_blt_tokenizer_creation(self):
        """Test creating BLT tokenizer."""
        config = TokenizerConfig(
            vocab_size=1000,
            byte_vocab_size=256,
            merge_ratio=0.5,
            max_length=512
        )
        
        tokenizer = BLTTokenizer(config)
        
        assert tokenizer.vocab_size == 1000
        assert tokenizer.byte_vocab_size == 256
        assert tokenizer.merge_ratio == 0.5
        assert tokenizer.max_length == 512
    
    def test_encode_decode_consistency(self):
        """Test that encode/decode are consistent."""
        config = TokenizerConfig(vocab_size=64, max_length=128)
        tokenizer = DNATokenizer(config)
        
        sequence = "ATCGATCGATCG"
        encoded = tokenizer.encode(sequence)
        decoded = tokenizer.decode(encoded)
        
        # Should recover original sequence (possibly with padding differences)
        assert isinstance(encoded, torch.Tensor)
        assert isinstance(decoded, str)
        assert "ATCGATCGATCG" in decoded.replace("[PAD]", "").replace("[UNK]", "")
    
    def test_batch_encoding(self):
        """Test batch encoding functionality."""
        config = TokenizerConfig(vocab_size=64, max_length=128)
        tokenizer = DNATokenizer(config)
        
        sequences = ["ATCG", "GCTA", "AATTCC"]
        encoded = tokenizer.encode_batch(sequences)
        
        assert encoded.shape[0] == len(sequences)
        assert encoded.shape[1] <= config.max_length
        assert encoded.dtype == torch.long
    
    def test_special_tokens_handling(self):
        """Test special tokens are handled correctly."""
        config = TokenizerConfig(vocab_size=64, max_length=128)
        tokenizer = DNATokenizer(config)
        
        # Test that special tokens are in vocabulary
        assert tokenizer.pad_token_id is not None
        assert tokenizer.unk_token_id is not None
        assert tokenizer.cls_token_id is not None
        assert tokenizer.sep_token_id is not None
        
        # Test special token IDs are valid
        assert 0 <= tokenizer.pad_token_id < tokenizer.vocab_size
        assert 0 <= tokenizer.unk_token_id < tokenizer.vocab_size


class TestDNATokenizer:
    """Test DNA-specific tokenizer."""
    
    def test_dna_vocabulary(self):
        """Test DNA tokenizer has correct vocabulary."""
        config = TokenizerConfig(vocab_size=64)
        tokenizer = DNATokenizer(config)
        
        # Should contain ATCG and variants
        vocab = tokenizer.get_vocab()
        assert 'A' in vocab
        assert 'T' in vocab
        assert 'C' in vocab
        assert 'G' in vocab
        assert 'N' in vocab  # Unknown nucleotide
    
    def test_dna_sequence_encoding(self):
        """Test encoding DNA sequences."""
        config = TokenizerConfig(vocab_size=64, max_length=128)
        tokenizer = DNATokenizer(config)
        
        sequence = "ATCGATCGATCGAATTCCGGAA"
        encoded = tokenizer.encode(sequence)
        
        assert encoded.shape[0] <= config.max_length
        assert torch.all(encoded >= 0)
        assert torch.all(encoded < tokenizer.vocab_size)
    
    def test_invalid_dna_characters(self):
        """Test handling of invalid DNA characters."""
        config = TokenizerConfig(vocab_size=64)
        tokenizer = DNATokenizer(config)
        
        # Should handle invalid characters gracefully
        sequence = "ATCGXYZ123"
        encoded = tokenizer.encode(sequence)
        
        # Should not raise error and return valid tensor
        assert isinstance(encoded, torch.Tensor)
    
    def test_case_insensitivity(self):
        """Test DNA tokenizer is case insensitive."""
        config = TokenizerConfig(vocab_size=64)
        tokenizer = DNATokenizer(config)
        
        upper_seq = "ATCG"
        lower_seq = "atcg"
        
        encoded_upper = tokenizer.encode(upper_seq)
        encoded_lower = tokenizer.encode(lower_seq)
        
        assert torch.equal(encoded_upper, encoded_lower)


class TestProteinTokenizer:
    """Test protein-specific tokenizer."""
    
    def test_protein_vocabulary(self):
        """Test protein tokenizer has correct vocabulary."""
        config = TokenizerConfig(vocab_size=64)
        tokenizer = ProteinTokenizer(config)
        
        vocab = tokenizer.get_vocab()
        
        # Should contain standard amino acids
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
        for aa in standard_aa:
            assert aa in vocab
    
    def test_protein_sequence_encoding(self):
        """Test encoding protein sequences."""
        config = TokenizerConfig(vocab_size=64, max_length=256)
        tokenizer = ProteinTokenizer(config)
        
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        encoded = tokenizer.encode(sequence)
        
        assert encoded.shape[0] <= config.max_length
        assert torch.all(encoded >= 0)
        assert torch.all(encoded < tokenizer.vocab_size)
    
    def test_stop_codon_handling(self):
        """Test handling of stop codons in protein sequences."""
        config = TokenizerConfig(vocab_size=64)
        tokenizer = ProteinTokenizer(config)
        
        # '*' represents stop codon
        sequence = "MKTVR*"
        encoded = tokenizer.encode(sequence)
        
        assert isinstance(encoded, torch.Tensor)


class TestRNATokenizer:
    """Test RNA-specific tokenizer."""
    
    def test_rna_vocabulary(self):
        """Test RNA tokenizer has correct vocabulary."""
        config = TokenizerConfig(vocab_size=64)
        tokenizer = RNATokenizer(config)
        
        vocab = tokenizer.get_vocab()
        assert 'A' in vocab
        assert 'U' in vocab  # RNA has U instead of T
        assert 'C' in vocab
        assert 'G' in vocab
        assert 'N' in vocab
    
    def test_rna_sequence_encoding(self):
        """Test encoding RNA sequences."""
        config = TokenizerConfig(vocab_size=64, max_length=256)
        tokenizer = RNATokenizer(config)
        
        sequence = "AUCGAUCGAUCGAAUCCGGAA"
        encoded = tokenizer.encode(sequence)
        
        assert encoded.shape[0] <= config.max_length
        assert torch.all(encoded >= 0)
        assert torch.all(encoded < tokenizer.vocab_size)


class TestBLTLayers:
    """Test BLT layer components."""
    
    def test_byte_latent_transformer(self):
        """Test ByteLatentTransformer layer."""
        config = {
            'hidden_size': 256,
            'num_heads': 4,
            'intermediate_size': 512,
            'hidden_dropout_prob': 0.1
        }
        
        layer = ByteLatentTransformer(**config)
        
        batch_size, seq_len = 2, 64
        hidden_states = torch.randn(batch_size, seq_len, config['hidden_size'])
        
        output = layer(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.allclose(output, hidden_states)  # Should transform input
    
    def test_token_merger(self):
        """Test TokenMerger component."""
        hidden_size = 256
        merge_ratio = 0.5
        
        merger = TokenMerger(hidden_size, merge_ratio)
        
        batch_size, seq_len = 2, 64
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        merged, attention_weights = merger(hidden_states)
        
        expected_len = int(seq_len * merge_ratio)
        assert merged.shape == (batch_size, expected_len, hidden_size)
        assert attention_weights.shape[1] == expected_len
    
    def test_adaptive_encoder(self):
        """Test AdaptiveEncoder component."""
        config = {
            'hidden_size': 256,
            'num_layers': 2,
            'merge_layers': [0],
            'merge_ratio': 0.5
        }
        
        encoder = AdaptiveEncoder(**config)
        
        batch_size, seq_len = 2, 64
        hidden_states = torch.randn(batch_size, seq_len, config['hidden_size'])
        
        output = encoder(hidden_states)
        
        # Should reduce sequence length due to merging
        assert output.shape[0] == batch_size
        assert output.shape[1] <= seq_len
        assert output.shape[2] == config['hidden_size']


class TestTokenizerConfig:
    """Test tokenizer configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TokenizerConfig()
        
        assert config.vocab_size > 0
        assert config.byte_vocab_size == 256
        assert 0 < config.merge_ratio <= 1
        assert config.max_length > 0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TokenizerConfig(
            vocab_size=2048,
            byte_vocab_size=512,
            merge_ratio=0.3,
            max_length=1024
        )
        
        assert config.vocab_size == 2048
        assert config.byte_vocab_size == 512
        assert config.merge_ratio == 0.3
        assert config.max_length == 1024
    
    def test_invalid_config(self):
        """Test validation of invalid configuration."""
        with pytest.raises(ValueError):
            TokenizerConfig(vocab_size=0)
        
        with pytest.raises(ValueError):
            TokenizerConfig(merge_ratio=0)
        
        with pytest.raises(ValueError):
            TokenizerConfig(merge_ratio=1.5)


@pytest.mark.benchmark
class TestTokenizerPerformance:
    """Test tokenizer performance."""
    
    def test_encoding_speed(self):
        """Test tokenizer encoding speed."""
        config = TokenizerConfig(vocab_size=1024, max_length=512)
        tokenizer = DNATokenizer(config)
        
        # Generate test sequences
        sequences = [DataGenerator.generate_dna_sequence(400).tolist() for _ in range(100)]
        sequences = [''.join(['ATCG'[i] for i in seq]) for seq in sequences]
        
        import time
        start = time.time()
        
        for seq in sequences:
            tokenizer.encode(seq)
        
        end = time.time()
        
        # Should process 100 sequences in reasonable time
        assert (end - start) < 10.0  # Less than 10 seconds
    
    def test_batch_encoding_speed(self):
        """Test batch encoding speed."""
        config = TokenizerConfig(vocab_size=1024, max_length=512)
        tokenizer = DNATokenizer(config)
        
        # Generate test sequences
        sequences = [DataGenerator.generate_dna_sequence(400).tolist() for _ in range(100)]
        sequences = [''.join(['ATCG'[i] for i in seq]) for seq in sequences]
        
        import time
        start = time.time()
        
        tokenizer.encode_batch(sequences)
        
        end = time.time()
        
        # Batch encoding should be faster than individual encoding
        assert (end - start) < 5.0  # Less than 5 seconds
