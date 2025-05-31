#!/usr/bin/env python3
"""
Simple Hyena-GLT Framework Demo

A streamlined demonstration of the complete Hyena-GLT framework capabilities:
- BLT dynamic token merging with entropy-based compression
- Hyena long-range convolutions for genomic pattern capture
- Comprehensive interpretability tools for model analysis

Usage:
    python demo_simple_framework.py

Author: Hyena-GLT Development Team
Date: May 30, 2025
"""

import time

import numpy as np
import torch

# Hyena-GLT imports
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data import DNATokenizer, GenomicDataset
from hyena_glt.interpretability import HyenaInterpretabilityFramework
from hyena_glt.model import HyenaGLT


def create_sample_genomic_data(num_samples: int = 20, seq_length: int = 64) -> list:
    """Create sample genomic sequences for demonstration."""
    nucleotides = ["A", "T", "C", "G"]
    sequences = []

    for i in range(num_samples):
        # Create a random genomic sequence
        sequence = "".join(np.random.choice(nucleotides, seq_length))

        # Add some realistic patterns
        if i % 3 == 0:
            # Add some common motifs
            sequence = sequence[:10] + "TATAAA" + sequence[16:]  # TATA box
        elif i % 3 == 1:
            sequence = sequence[:15] + "AAGCTT" + sequence[21:]  # HindIII site

        sequences.append(
            {"sequence": sequence, "label": i % 3}  # Simple classification labels
        )

    return sequences


def demo_data_processing(seq_length: int = 64, batch_size: int = 4):
    """Demonstrate data processing capabilities."""
    print("\n🧬 DATA PROCESSING DEMO")
    print("=" * 50)

    # Create sample data
    print("📋 Creating sample genomic data...")
    sample_sequences = create_sample_genomic_data(20, seq_length)
    print(f"   ✅ Created {len(sample_sequences)} sample sequences")

    # Initialize tokenizer
    print("🔤 Initializing DNA tokenizer...")
    tokenizer = DNATokenizer(vocab_size=77, sequence_length=seq_length)
    print(f"   ✅ Tokenizer ready (vocab_size={tokenizer.vocab_size})")

    # Test tokenization
    test_sequence = sample_sequences[0]["sequence"]
    tokens = tokenizer.encode(
        test_sequence, max_length=seq_length, padding=True, truncation=True
    )

    print("📊 Tokenization test:")
    print(f"   Original: {test_sequence[:50]}...")
    if isinstance(tokens, dict):
        print(f"   Tokens: {tokens['input_ids'][:10]}...")
        print(f"   Token shape: {len(tokens['input_ids'])}")
    else:
        print(f"   Tokens: {tokens[:10]}...")
        print(f"   Token shape: {len(tokens)}")

    # Create dataset
    print("📦 Creating dataset...")
    dataset = GenomicDataset(
        data=sample_sequences, tokenizer=tokenizer, max_length=seq_length
    )
    print(f"   ✅ Dataset created with {len(dataset)} samples")

    # Test dataset item
    sample_item = dataset[0]
    print(
        f"   ✅ Dataset item shape: input_ids={sample_item['input_ids'].shape}, label={sample_item['label']}"
    )

    return {
        "dataset": dataset,
        "tokenizer": tokenizer,
        "sample_sequences": sample_sequences,
    }


def demo_model_architecture(config: HyenaGLTConfig, seq_length: int = 64):
    """Demonstrate model architecture and forward pass."""
    print("\n🏗️ MODEL ARCHITECTURE DEMO")
    print("=" * 50)

    # Create model
    print("🔧 Creating Hyena-GLT model...")
    model = HyenaGLT(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model created: {param_count:,} parameters")

    # Test forward pass
    print("⚡ Testing forward pass...")
    batch_size = 2
    input_ids = torch.randint(0, config.genomic_vocab_size, (batch_size, seq_length))

    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids, output_merge_info=True)
    forward_time = time.time() - start_time

    print(f"   ✅ Forward pass successful ({forward_time:.3f}s)")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {outputs['last_hidden_state'].shape}")

    # Check for token merging
    if outputs["last_hidden_state"].shape[1] != input_ids.shape[1]:
        compression_ratio = input_ids.shape[1] / outputs["last_hidden_state"].shape[1]
        print(
            f"   ✅ BLT Token Merging: {input_ids.shape[1]} -> {outputs['last_hidden_state'].shape[1]} tokens ({compression_ratio:.1f}x compression)"
        )

    return {"model": model, "outputs": outputs, "param_count": param_count}


def demo_interpretability(model, config):
    """Demonstrate interpretability framework."""
    print("\n🔍 INTERPRETABILITY DEMO")
    print("=" * 50)

    # Initialize interpretability framework
    print("🧠 Initializing interpretability framework...")
    interpreter = HyenaInterpretabilityFramework(model, config)
    print("   ✅ Interpretability framework ready")

    # Test sequence analysis
    print("🔬 Testing sequence analysis...")
    test_sequence = torch.randint(0, config.genomic_vocab_size, (1, 32))
    analysis = interpreter.analyze_sequence(test_sequence)
    print(f"   ✅ Sequence analysis completed: {len(analysis)} components")

    # Test batch analysis
    print("📊 Testing batch analysis...")
    batch_sequences = torch.randint(0, config.genomic_vocab_size, (3, 32))
    batch_analysis = interpreter.analyze_batch(batch_sequences)
    print(f"   ✅ Batch analysis completed: {len(batch_analysis)} items")

    return {
        "interpreter": interpreter,
        "analysis": analysis,
        "batch_analysis": batch_analysis,
    }


def run_complete_demo(seq_length: int = 64, batch_size: int = 4):
    """Run the complete framework demonstration."""
    print("🎉 HYENA-GLT SIMPLE FRAMEWORK DEMO")
    print("=" * 60)
    print("Demonstrating BLT + Hyena + Interpretability")
    print("=" * 60)

    # Configuration
    config = HyenaGLTConfig(
        hidden_size=144,  # Divisible by 12 heads
        num_layers=3,
        max_position_embeddings=seq_length,
        num_attention_heads=12,
        dynamic_patching=True,
    )

    print("🔧 Configuration:")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Number of layers: {config.num_layers}")
    print(f"   Max sequence length: {config.max_position_embeddings}")
    print(f"   Dynamic patching (BLT): {config.dynamic_patching}")

    # Run demos
    results = {}

    # Data processing demo
    results["data"] = demo_data_processing(seq_length, batch_size)

    # Model architecture demo
    results["model"] = demo_model_architecture(config, seq_length)

    # Interpretability demo
    results["interpretability"] = demo_interpretability(
        results["model"]["model"], config
    )

    # Summary
    print("\n🎯 DEMO SUMMARY")
    print("=" * 50)
    print(f"✅ Data processing: {len(results['data']['dataset'])} sequences tokenized")
    print(f"✅ Model architecture: {results['model']['param_count']:,} parameters")
    print("✅ Forward pass: Successful with dynamic token compression")
    print("✅ Interpretability: Analysis framework operational")
    print("\n🚀 HYENA-GLT FRAMEWORK FULLY OPERATIONAL! 🚀")

    return results


def main():
    """Main demo function."""
    try:
        results = run_complete_demo(seq_length=64, batch_size=4)
        print("\n✅ Demo completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
