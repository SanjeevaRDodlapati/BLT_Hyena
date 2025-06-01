"""
Test script to validate the pretraining system setup.

This script performs basic validation of the pretraining components
to ensure everything is working correctly.
"""

import os
import sys
import tempfile
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from hyena_glt.training.pretraining_config import (
        HyenaGLTPretrainingConfig,
        create_pretraining_configs
    )
    from hyena_glt.training.data_utils import GenomicPretrainingDataset
    from hyena_glt.training.pretraining import HyenaGLTPretrainer
    print("✓ Successfully imported pretraining modules")
except ImportError as e:
    print(f"✗ Failed to import pretraining modules: {e}")
    sys.exit(1)


def test_config_creation():
    """Test configuration creation and serialization."""
    print("\n=== Testing Configuration System ===")
    
    # Test pre-built configurations
    configs = create_pretraining_configs()
    assert "mlm_small" in configs
    assert "ar_base" in configs
    print("✓ Pre-built configurations created successfully")
    
    # Test configuration serialization
    config = configs["mlm_small"]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.save_to_file(f.name)
        
        # Load it back
        loaded_config = HyenaGLTPretrainingConfig.load_from_file(f.name)
        assert loaded_config.model.hidden_size == config.model.hidden_size
        print("✓ Configuration serialization/deserialization works")
        
        # Clean up
        os.unlink(f.name)


def test_data_utils():
    """Test data loading utilities."""
    print("\n=== Testing Data Loading ===")
    
    # Create a temporary FASTA file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">sequence1\n")
        f.write("ATCGATCGATCGATCGATCGATCGATCGATCG\n")
        f.write(">sequence2\n")
        f.write("GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\n")
        fasta_path = f.name
    
    # Create tokenizer
    from hyena_glt.data.tokenizer import DNATokenizer
    tokenizer = DNATokenizer()
    
    # Test dataset creation
    dataset = GenomicPretrainingDataset(
        data_paths=[fasta_path],
        tokenizer=tokenizer,
        max_length=32,
        sequence_type="dna"
    )
    
    # Test data loading (use iterator for IterableDataset)
    sample = next(iter(dataset))
    assert 'input_ids' in sample
    assert len(sample['input_ids']) <= 32
    print("✓ Genomic dataset creation and sampling works")
    
    # Clean up
    os.unlink(fasta_path)


def test_model_initialization():
    """Test model and pretrainer initialization."""
    print("\n=== Testing Model Initialization ===")
    
    # Get a small configuration
    configs = create_pretraining_configs()
    config = configs["mlm_small"]
    
    # Override for testing
    config.model.hidden_size = 64  # Very small for testing
    config.model.num_layers = 2
    config.model.num_attention_heads = 2
    config.optimization.max_steps = 1  # Just one step
    
    # Create temporary data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(">sequence1\n")
        f.write("ATCGATCGATCGATCGATCGATCGATCGATCG\n")
        config.data.data_paths = [f.name]
    
    # Test pretrainer initialization
    pretrainer = HyenaGLTPretrainer(config)
    assert pretrainer.model is not None
    assert pretrainer.tokenizer is not None
    print("✓ Model and pretrainer initialization works")
    
    # Clean up
    os.unlink(f.name)


def test_masking_utils():
    """Test masking utilities."""
    print("\n=== Testing Masking Utilities ===")
    
    from hyena_glt.training.pretraining import GenomicMaskingUtils
    
    # Create test input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 4, (batch_size, seq_len))
    
    masking_utils = GenomicMaskingUtils()
    
    # Test MLM masking
    masked_input, labels = masking_utils.get_mlm_masks(
        input_ids, mask_prob=0.5, mask_token_id=4
    )
    assert masked_input.shape == input_ids.shape
    assert labels.shape == input_ids.shape
    print("✓ MLM masking works")
    
    # Test span masking
    masked_input, labels = masking_utils.get_span_masks(
        input_ids, span_prob=0.2, mask_token_id=4
    )
    assert masked_input.shape == input_ids.shape
    print("✓ Span masking works")


def test_loss_functions():
    """Test loss function utilities."""
    print("\n=== Testing Loss Functions ===")
    
    from hyena_glt.training.pretraining import GenomicLossFunctions
    
    loss_functions = GenomicLossFunctions()
    
    # Test cross entropy loss
    logits = torch.randn(2, 8, 4)  # batch_size=2, seq_len=8, vocab_size=4
    labels = torch.randint(0, 4, (2, 8))
    
    loss = loss_functions.cross_entropy_loss(logits, labels)
    assert torch.is_tensor(loss)
    assert loss.numel() == 1
    print("✓ Cross entropy loss works")
    
    # Test focal loss
    loss = loss_functions.focal_loss(logits, labels, alpha=1.0, gamma=2.0)
    assert torch.is_tensor(loss)
    print("✓ Focal loss works")


def run_all_tests():
    """Run all validation tests."""
    print("Starting Hyena-GLT Pretraining System Validation...")
    print("=" * 60)
    
    tests = [
        ("Configuration System", test_config_creation),
        ("Data Loading", test_data_utils),
        ("Model Initialization", test_model_initialization),
        ("Masking Utilities", test_masking_utils),
        ("Loss Functions", test_loss_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✓ {test_name} - PASSED")
        except Exception as e:
            print(f"✗ {test_name} - ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pretraining system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
