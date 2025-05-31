#!/usr/bin/env python3
"""
Test script to verify Hyena operators functionality.
"""
import torch
import torch.nn as nn

from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model.operators import (
    DynamicConvolution,
    GenomicPositionalEncoding,
    HyenaFilter,
    HyenaOperator,
)


def test_dynamic_convolution():
    """Test DynamicConvolution operator."""
    print("Testing DynamicConvolution...")

    d_model = 64
    kernel_size = 7
    batch_size = 2
    seq_len = 32

    conv = DynamicConvolution(d_model, kernel_size)
    x = torch.randn(batch_size, seq_len, d_model)

    # Test without segment boundaries
    output = conv(x)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    # Test with segment boundaries
    segment_boundaries = torch.zeros(batch_size, seq_len)
    segment_boundaries[:, seq_len // 2] = 1.0  # Add boundary in middle
    output_with_boundaries = conv(x, segment_boundaries)
    assert output_with_boundaries.shape == x.shape

    print("✅ DynamicConvolution test passed")


def test_hyena_filter():
    """Test HyenaFilter operator."""
    print("Testing HyenaFilter...")

    d_model = 64
    filter_order = 32
    seq_len = 128

    filter_fn = HyenaFilter(
        d_model=d_model,
        order=2,
        filter_order=filter_order,
        seq_len=seq_len,
        channels=1,
        dropout=0.1,
    )

    # Test filter generation
    filter_coeffs = filter_fn(seq_len)
    expected_shape = (d_model, seq_len)
    assert (
        filter_coeffs.shape == expected_shape
    ), f"Expected {expected_shape}, got {filter_coeffs.shape}"

    print("✅ HyenaFilter test passed")


def test_genomic_positional_encoding():
    """Test GenomicPositionalEncoding."""
    print("Testing GenomicPositionalEncoding...")

    d_model = 64
    max_len = 512
    seq_len = 128

    pos_encoder = GenomicPositionalEncoding(d_model, max_len)
    encoding = pos_encoder(seq_len)

    expected_shape = (seq_len, d_model)
    assert (
        encoding.shape == expected_shape
    ), f"Expected {expected_shape}, got {encoding.shape}"

    print("✅ GenomicPositionalEncoding test passed")


def test_hyena_operator():
    """Test full HyenaOperator."""
    print("Testing HyenaOperator...")

    config = HyenaGLTConfig(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
        hyena_order=2,
        hyena_filter_size=32,
        hyena_short_filter_size=7,
        use_bias=True,
        use_glu=True,
        hyena_dropout=0.1,
    )

    d_model = config.hidden_size
    l_max = 128
    batch_size = 2
    seq_len = 64

    hyena_op = HyenaOperator(
        config=config,
        d_model=d_model,
        l_max=l_max,
        order=2,
        filter_order=32,
        dropout=0.1,
    )

    x = torch.randn(batch_size, seq_len, d_model)

    # Test forward pass
    output = hyena_op(x)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    # Test with attention mask
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, seq_len // 2 :] = 0  # Mask second half
    output_masked = hyena_op(x, attention_mask=attention_mask)
    assert output_masked.shape == x.shape

    print("✅ HyenaOperator test passed")


def test_gradient_flow():
    """Test gradient flow through operators."""
    print("Testing gradient flow...")

    config = HyenaGLTConfig(
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        max_position_embeddings=256,
        hyena_order=2,
        hyena_filter_size=16,
        hyena_short_filter_size=7,
    )

    hyena_op = HyenaOperator(
        config=config,
        d_model=32,
        l_max=64,
        order=2,
        filter_order=16,
    )

    x = torch.randn(1, 32, 32, requires_grad=True)
    target = torch.randn(1, 32, 32)

    output = hyena_op(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "No gradients computed for input"
    assert any(
        p.grad is not None for p in hyena_op.parameters()
    ), "No gradients for parameters"

    print("✅ Gradient flow test passed")


def main():
    """Run all tests."""
    print("🧪 Testing Hyena-GLT Operators Functionality\n")

    try:
        test_dynamic_convolution()
        test_hyena_filter()
        test_genomic_positional_encoding()
        test_hyena_operator()
        test_gradient_flow()

        print(
            "\n🎉 All operator tests passed! The Hyena-GLT operators are working correctly."
        )

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
