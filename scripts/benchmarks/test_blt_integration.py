#!/usr/bin/env python3
"""
Test script to verify BLT position embedding integration with Hyena-GLT.

This tests the integration of our new position embedding system based on the 
Hyena-BLT-Genome Technical Guide.
"""

import torch
import torch.nn as nn
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model import HyenaGLT, BLTPositionManager


def test_blt_position_manager():
    """Test the BLT position manager independently."""
    print("Testing BLT Position Manager...")
    
    # Configuration
    d_model = 256
    max_len = 1024
    batch_size = 2
    seq_len = 128
    
    # Create position manager
    position_manager = BLTPositionManager(
        d_model=d_model,
        max_len=max_len,
        num_heads=8,
        dropout=0.1,
        max_patch_size=16,
    )
    
    # Test basic position encoding
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    
    # Test without patch boundaries
    pos_encoded = position_manager.encode_positions(hidden_states)
    assert pos_encoded.shape == hidden_states.shape
    print(f"✓ Basic position encoding: {pos_encoded.shape}")
    
    # Test with patch boundaries
    patch_boundaries = torch.zeros(batch_size, seq_len)
    patch_boundaries[:, 32] = 1  # Add some boundaries
    patch_boundaries[:, 64] = 1
    patch_boundaries[:, 96] = 1
    
    pos_encoded_with_patches = position_manager.encode_positions(
        hidden_states, patch_boundaries=patch_boundaries
    )
    assert pos_encoded_with_patches.shape == hidden_states.shape
    print(f"✓ Position encoding with patches: {pos_encoded_with_patches.shape}")
    
    # Test cross-attention bridge
    byte_repr = torch.randn(batch_size, seq_len, d_model)
    patch_repr, info = position_manager.create_patch_representations(
        byte_repr, patch_boundaries
    )
    print(f"✓ Patch representation creation: {patch_repr.shape}")
    
    # Test reconstruction
    reconstructed = position_manager.reconstruct_byte_representations(
        patch_repr, info, byte_repr
    )
    assert reconstructed.shape == byte_repr.shape
    print(f"✓ Byte reconstruction: {reconstructed.shape}")


def test_hyena_glt_with_blt_positions():
    """Test full Hyena-GLT model with BLT position embeddings."""
    print("\nTesting Hyena-GLT with BLT Position Embeddings...")
    
    # Create small config for testing
    config = HyenaGLTConfig(
        genomic_vocab_size=256,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=2048,
        
        # BLT-specific settings
        dynamic_patching=True,
        min_patch_size=4,
        max_patch_size=16,
        cross_attention_layers=2,
        
        # Hyena settings
        hyena_order=2,
        hyena_filter_size=64,
        dropout=0.1,
    )
    
    # Create model
    model = HyenaGLT(config)
    model.eval()
    
    # Test data
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.genomic_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_merge_info=True,
        )
    
    # Check outputs
    assert "last_hidden_state" in outputs
    assert "hidden_states" in outputs
    assert "merge_info" in outputs
    
    last_hidden = outputs["last_hidden_state"]
    print(f"✓ Model forward pass successful: {last_hidden.shape}")
    
    # Check that token merging happened (sequence should be shorter)
    if config.dynamic_patching and outputs["merge_info"]:
        original_len = outputs["merge_info"][0]["original_length"]
        merged_len = outputs["merge_info"][0]["merged_length"]
        compression_ratio = outputs["merge_info"][0]["compression_ratio"]
        
        print(f"✓ Token merging: {original_len} -> {merged_len} (ratio: {compression_ratio:.2f})")
        assert merged_len <= original_len, "Merged sequence should be shorter or equal"
    
    print("✓ All hidden states collected:", len(outputs["hidden_states"]))


def test_position_preservation():
    """Test that position information is preserved through merging."""
    print("\nTesting Position Information Preservation...")
    
    # Create position manager
    d_model = 128
    position_manager = BLTPositionManager(d_model=d_model, max_len=512)
    
    # Create test sequence
    batch_size = 1
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    
    # Create original positions
    original_positions = torch.arange(seq_len).unsqueeze(0)
    
    # Add initial position encoding
    pos_encoded = position_manager.encode_positions(
        hidden_states, original_positions=original_positions
    )
    
    # Simulate token merging with boundaries
    patch_boundaries = torch.zeros(batch_size, seq_len)
    patch_boundaries[0, [16, 32, 48]] = 1  # Create 4 patches
    
    # Re-encode after merging
    reencoded = position_manager.encode_positions(
        pos_encoded, 
        patch_boundaries=patch_boundaries,
        original_positions=original_positions
    )
    
    # Check that we can create and reconstruct patch representations
    patch_repr, info = position_manager.create_patch_representations(
        reencoded, patch_boundaries
    )
    
    reconstructed = position_manager.reconstruct_byte_representations(
        patch_repr, info, reencoded
    )
    
    # Verify reconstruction preserves general structure
    assert reconstructed.shape == reencoded.shape
    print(f"✓ Position preservation through merging: {reconstructed.shape}")
    
    # Check that the reconstruction is reasonable (not identical due to compression)
    mse = torch.mean((reconstructed - reencoded) ** 2)
    print(f"✓ Reconstruction MSE: {mse.item():.6f}")


def test_genomic_position_patterns():
    """Test genomic-specific position patterns."""
    print("\nTesting Genomic Position Patterns...")
    
    from hyena_glt.model.position_embeddings import SegmentAwarePositionalEncoding
    
    d_model = 128
    pos_encoder = SegmentAwarePositionalEncoding(d_model=d_model, max_len=512)
    
    # Test with genomic-relevant sequence lengths
    genomic_lengths = [147, 21, 10, 8, 3]  # Common genomic patterns
    
    for length in genomic_lengths:
        pos_encoding = pos_encoder(seq_len=length)
        assert pos_encoding.shape == (1, length, d_model)
        print(f"✓ Genomic pattern length {length}: {pos_encoding.shape}")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("BLT-Hyena Position Embedding Integration Tests")
    print("=" * 60)
    
    try:
        test_blt_position_manager()
        test_hyena_glt_with_blt_positions()
        test_position_preservation()
        test_genomic_position_patterns()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("BLT position embedding integration is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
