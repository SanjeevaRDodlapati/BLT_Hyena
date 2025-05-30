"""
Unit tests for utility modules.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from hyena_glt.utils.visualization import (
    plot_attention_weights, plot_training_metrics, plot_sequence_embeddings,
    visualize_genomic_features, create_genomic_browser
)
from hyena_glt.utils.model_utils import (
    count_parameters, get_model_size, save_checkpoint, load_checkpoint,
    freeze_layers, unfreeze_layers, get_layer_names
)
from hyena_glt.utils.analysis import (
    analyze_attention_patterns, compute_sequence_similarity,
    find_conserved_regions, analyze_variant_effects
)
from hyena_glt.utils.genomic_utils import (
    reverse_complement, translate_dna, gc_content,
    find_orfs, compute_codon_usage
)
from hyena_glt.model.hyena_glt import HyenaGLT
from hyena_glt.config import HyenaGLTConfig
from tests.utils import TestConfig, DataGenerator


class TestVisualization:
    """Test visualization utilities."""
    
    def test_plot_attention_weights(self):
        """Test attention weights plotting."""
        # Create mock attention weights
        batch_size, num_heads, seq_len = 1, 4, 32
        attention_weights = torch.rand(batch_size, num_heads, seq_len, seq_len)
        
        # Should not raise error
        try:
            fig = plot_attention_weights(attention_weights[0])
            assert fig is not None
        except ImportError:
            # Skip if matplotlib not available
            pytest.skip("Matplotlib not available")
    
    def test_plot_training_metrics(self):
        """Test training metrics plotting."""
        metrics = {
            'train_loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3],
            'train_accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
            'val_accuracy': [0.55, 0.65, 0.75, 0.8, 0.85]
        }
        
        try:
            fig = plot_training_metrics(metrics)
            assert fig is not None
        except ImportError:
            pytest.skip("Matplotlib not available")
    
    def test_plot_sequence_embeddings(self):
        """Test sequence embeddings visualization."""
        # Create mock embeddings
        embeddings = torch.randn(100, 256)  # 100 sequences, 256 dimensions
        labels = torch.randint(0, 3, (100,))  # 3 classes
        
        try:
            fig = plot_sequence_embeddings(embeddings, labels)
            assert fig is not None
        except ImportError:
            pytest.skip("Required visualization packages not available")
    
    def test_visualize_genomic_features(self):
        """Test genomic features visualization."""
        # Create mock genomic data
        sequence = "ATCGATCGATCGAATTCCGGAA"
        features = {
            'gc_content': [0.4, 0.6, 0.5, 0.3, 0.7],
            'orfs': [(2, 10), (15, 20)],
            'conserved_regions': [(5, 15)]
        }
        
        try:
            fig = visualize_genomic_features(sequence, features)
            assert fig is not None
        except ImportError:
            pytest.skip("Required visualization packages not available")


class TestModelUtils:
    """Test model utility functions."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)
        
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    def test_get_model_size(self):
        """Test model size calculation."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)
        
        size_mb = get_model_size(model)
        assert size_mb > 0
        assert isinstance(size_mb, float)
    
    def test_save_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        checkpoint_data = {
            'epoch': 5,
            'step': 1000,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': 0.5,
            'val_loss': 0.6
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Test saving
            save_checkpoint(checkpoint_data, checkpoint_path)
            assert os.path.exists(checkpoint_path)
            
            # Test loading
            loaded_data = load_checkpoint(checkpoint_path)
            
            assert loaded_data['epoch'] == 5
            assert loaded_data['step'] == 1000
            assert loaded_data['train_loss'] == 0.5
            
            # Test model state loading
            model.load_state_dict(loaded_data['model_state_dict'])
            
        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
    
    def test_freeze_unfreeze_layers(self):
        """Test layer freezing and unfreezing."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)
        
        # Initially all parameters should be trainable
        trainable_before = count_parameters(model, trainable_only=True)
        total_params = count_parameters(model, trainable_only=False)
        assert trainable_before == total_params
        
        # Freeze embedding layer
        freeze_layers(model, ['embeddings'])
        trainable_after_freeze = count_parameters(model, trainable_only=True)
        assert trainable_after_freeze < trainable_before
        
        # Unfreeze all layers
        unfreeze_layers(model, ['embeddings'])
        trainable_after_unfreeze = count_parameters(model, trainable_only=True)
        assert trainable_after_unfreeze == trainable_before
    
    def test_get_layer_names(self):
        """Test getting layer names."""
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)
        
        layer_names = get_layer_names(model)
        
        assert isinstance(layer_names, list)
        assert len(layer_names) > 0
        assert 'embeddings' in layer_names


class TestAnalysis:
    """Test analysis utilities."""
    
    def test_analyze_attention_patterns(self):
        """Test attention pattern analysis."""
        # Create mock attention weights
        batch_size, num_heads, seq_len = 2, 4, 32
        attention_weights = torch.rand(batch_size, num_heads, seq_len, seq_len)
        
        analysis = analyze_attention_patterns(attention_weights)
        
        assert 'entropy' in analysis
        assert 'sparsity' in analysis
        assert 'locality' in analysis
        assert isinstance(analysis['entropy'], torch.Tensor)
    
    def test_compute_sequence_similarity(self):
        """Test sequence similarity computation."""
        seq1 = torch.randint(0, 4, (50,))
        seq2 = torch.randint(0, 4, (50,))
        
        similarity = compute_sequence_similarity(seq1, seq2)
        
        assert 0 <= similarity <= 1
        assert isinstance(similarity, float)
        
        # Test identical sequences
        identical_sim = compute_sequence_similarity(seq1, seq1)
        assert identical_sim == 1.0
    
    def test_find_conserved_regions(self):
        """Test conserved region finding."""
        # Create sequences with some conservation
        sequences = [
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
            torch.tensor([0, 1, 2, 3, 1, 2, 3, 0]),
            torch.tensor([0, 1, 2, 3, 2, 1, 0, 3])
        ]
        
        conserved_regions = find_conserved_regions(sequences, min_length=3, threshold=0.8)
        
        assert isinstance(conserved_regions, list)
        # First 4 positions should be conserved
        assert (0, 4) in conserved_regions
    
    def test_analyze_variant_effects(self):
        """Test variant effect analysis."""
        reference = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        variant = torch.tensor([0, 1, 3, 3, 0, 1, 2, 3])  # SNP at position 2
        
        config = HyenaGLTConfig(**TestConfig.SMALL_CONFIG)
        model = HyenaGLT(config)
        model.eval()
        
        with torch.no_grad():
            effects = analyze_variant_effects(model, reference, variant)
        
        assert 'embedding_change' in effects
        assert 'attention_change' in effects
        assert isinstance(effects['embedding_change'], torch.Tensor)


class TestGenomicUtils:
    """Test genomic utility functions."""
    
    def test_reverse_complement(self):
        """Test reverse complement calculation."""
        # Test DNA
        dna_seq = "ATCG"
        rev_comp = reverse_complement(dna_seq, sequence_type='dna')
        assert rev_comp == "CGAT"
        
        # Test RNA
        rna_seq = "AUCG"
        rev_comp = reverse_complement(rna_seq, sequence_type='rna')
        assert rev_comp == "CGAU"
    
    def test_translate_dna(self):
        """Test DNA translation."""
        # Standard genetic code test
        dna_seq = "ATGAAATAG"  # Start codon + AAA (Lys) + Stop codon
        protein = translate_dna(dna_seq)
        assert protein.startswith("MK")
        assert protein.endswith("*")
    
    def test_gc_content(self):
        """Test GC content calculation."""
        # 50% GC content
        sequence = "ATCG"
        gc = gc_content(sequence)
        assert gc == 0.5
        
        # 100% GC content
        sequence = "GCGC"
        gc = gc_content(sequence)
        assert gc == 1.0
        
        # 0% GC content
        sequence = "ATAT"
        gc = gc_content(sequence)
        assert gc == 0.0
    
    def test_find_orfs(self):
        """Test ORF finding."""
        # Simple sequence with one ORF
        sequence = "ATGAAAAAATAG"  # Start + codons + stop
        orfs = find_orfs(sequence)
        
        assert len(orfs) >= 1
        assert orfs[0]['start'] == 0
        assert orfs[0]['stop'] == 12
        assert orfs[0]['length'] == 12
    
    def test_compute_codon_usage(self):
        """Test codon usage computation."""
        sequence = "ATGAAAAAATAG"
        codon_usage = compute_codon_usage(sequence)
        
        assert isinstance(codon_usage, dict)
        assert 'ATG' in codon_usage  # Start codon
        assert 'AAA' in codon_usage  # Lysine
        assert 'TAG' in codon_usage  # Stop codon
        
        # Check frequencies sum to 1
        assert abs(sum(codon_usage.values()) - 1.0) < 1e-6


@pytest.mark.memory_intensive
class TestMemoryUsage:
    """Test memory usage of utilities."""
    
    def test_large_sequence_processing(self):
        """Test processing large sequences."""
        # Create large sequence
        large_sequence = "ATCG" * 10000  # 40k nucleotides
        
        # Should handle without memory error
        gc = gc_content(large_sequence)
        assert 0 <= gc <= 1
        
        orfs = find_orfs(large_sequence)
        assert isinstance(orfs, list)
    
    def test_large_attention_analysis(self):
        """Test analyzing large attention matrices."""
        # Create large attention matrix
        seq_len = 1000
        attention_weights = torch.rand(1, 8, seq_len, seq_len)
        
        # Should handle without memory error
        analysis = analyze_attention_patterns(attention_weights)
        assert 'entropy' in analysis


@pytest.mark.slow
class TestPerformance:
    """Test performance of utility functions."""
    
    def test_similarity_computation_speed(self):
        """Test sequence similarity computation speed."""
        seq1 = torch.randint(0, 4, (10000,))
        seq2 = torch.randint(0, 4, (10000,))
        
        import time
        start = time.time()
        
        for _ in range(10):
            compute_sequence_similarity(seq1, seq2)
        
        end = time.time()
        
        # Should complete 10 comparisons in reasonable time
        assert (end - start) < 5.0
    
    def test_orf_finding_speed(self):
        """Test ORF finding speed."""
        sequence = "ATCG" * 2500  # 10k nucleotides
        
        import time
        start = time.time()
        
        orfs = find_orfs(sequence)
        
        end = time.time()
        
        # Should find ORFs in reasonable time
        assert (end - start) < 1.0
        assert isinstance(orfs, list)
