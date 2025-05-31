#!/usr/bin/env python3
"""
Hyena-GLT Framework Demo

This script demonstrates the complete Hyena-GLT framework capabilities:
- BLT dynamic token merging with entropy-based compression
- Hyena long-range convolutions for genomic pattern capture
- Enhanced training pipeline with multi-modal support
- Comprehensive interpretability tools for model analysis

Usage:
    python demo_complete_framework.py [--sequence-length 128] [--batch-size 4]

Author: Hyena-GLT Development Team
Date: May 30, 2025
"""

import argparse
import logging
import time

import numpy as np
import torch

from examples.enhanced_training_pipeline import (
    EnhancedTrainingConfig,
    EnhancedTrainingPipeline,
)

# Hyena-GLT imports
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.data import DNATokenizer, GenomicDataset, create_genomic_dataloaders
from hyena_glt.interpretability import HyenaInterpretabilityFramework
from hyena_glt.model import HyenaGLT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_genomic_data(num_samples: int = 100, seq_length: int = 128) -> list:
    """Create sample genomic sequences for demonstration."""
    nucleotides = ["A", "T", "C", "G"]
    sequences = []

    for i in range(num_samples):
        # Create realistic genomic sequences with some patterns
        sequence = ""

        # Add some repetitive regions (common in genomes)
        if i % 5 == 0:
            sequence += "ATCGATCG" * (seq_length // 32)

        # Add random nucleotides for the rest
        remaining_length = seq_length - len(sequence)
        sequence += "".join(np.random.choice(nucleotides, remaining_length))

        # Ensure exact length
        sequence = sequence[:seq_length]

        sequences.append(
            {
                "sequence": sequence,
                "labels": i % 3,  # 3-class classification
                "metadata": {"sample_id": i, "seq_type": "genomic"},
            }
        )

    return sequences


def demo_model_architecture(config: HyenaGLTConfig):
    """Demonstrate the model architecture and forward pass."""
    print("\n🏗️  MODEL ARCHITECTURE DEMO")
    print("=" * 50)

    # Create model
    model = HyenaGLT(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("✅ Model created successfully")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # Test forward pass
    batch_size = 2
    seq_len = config.max_position_embeddings
    input_ids = torch.randint(0, config.genomic_vocab_size, (batch_size, seq_len))

    print("\n📊 Forward Pass Test")
    print(f"   Input shape: {input_ids.shape}")

    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids, output_merge_info=True)
    forward_time = time.time() - start_time

    output_shape = outputs["last_hidden_state"].shape
    compression_ratio = seq_len / output_shape[1]

    print(f"   Output shape: {output_shape}")
    print(f"   Original length: {seq_len}")
    print(f"   Compressed length: {output_shape[1]}")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    print(f"   Forward pass time: {forward_time:.3f}s")

    if "merge_info" in outputs:
        print(f"   Token merging layers: {len(outputs['merge_info'])}")

    return model, outputs


def demo_interpretability(model: HyenaGLT, sample_data: torch.Tensor):
    """Demonstrate interpretability capabilities."""
    print("\n🔍 INTERPRETABILITY DEMO")
    print("=" * 50)

    # Create interpretability framework
    interpreter = HyenaInterpretabilityFramework(model)

    # Sequence analysis
    print("📝 Sequence Analysis")
    start_time = time.time()
    sequence_analysis = interpreter.analyze_sequence(sample_data[0])
    analysis_time = time.time() - start_time

    print(f"   ✅ Sequence analysis completed in {analysis_time:.3f}s")
    print(f"   Analysis components: {len(sequence_analysis)}")

    for key, value in sequence_analysis.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape} tensor")
        else:
            print(f"     {key}: {value}")

    # Batch analysis
    print("\n📊 Batch Analysis")
    start_time = time.time()
    batch_analysis = interpreter.analyze_batch(sample_data)
    batch_time = time.time() - start_time

    print(f"   ✅ Batch analysis completed in {batch_time:.3f}s")
    print(f"   Batch components: {len(batch_analysis)}")

    return sequence_analysis, batch_analysis


def demo_training_pipeline():
    """Demonstrate enhanced training pipeline."""
    print("\n🚂 ENHANCED TRAINING PIPELINE DEMO")
    print("=" * 50)

    # Create training configuration
    training_config = EnhancedTrainingConfig(
        output_dir="./demo_training_output",
        experiment_name="hyena_glt_demo",
        use_multimodal=False,  # Simplify for demo
        enable_interpretability=True,
        real_time_plotting=False,  # Disable for demo
        profile_performance=False,
    )

    # Initialize training pipeline
    print("🔧 Initializing training pipeline...")
    pipeline = EnhancedTrainingPipeline(training_config)

    print("   ✅ Training pipeline initialized successfully")
    print(f"   Output directory: {training_config.output_dir}")
    print(f"   Experiment name: {training_config.experiment_name}")
    print(f"   Interpretability enabled: {training_config.enable_interpretability}")

    return pipeline


def demo_data_processing(seq_length: int = 128, batch_size: int = 4):
    """Demonstrate data processing capabilities."""
    print("\n🧬 DATA PROCESSING DEMO")
    print("=" * 50)

    # Create sample data
    print("📋 Creating sample genomic data...")
    sample_sequences = create_sample_genomic_data(num_samples=20, seq_length=seq_length)
    print(f"   ✅ Created {len(sample_sequences)} sample sequences")

    # Initialize tokenizer
    print("🔤 Initializing DNA tokenizer...")
    tokenizer = DNATokenizer(vocab_size=1000, kmer_size=3)
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

    # Create data loaders
    print("📦 Creating data loaders...")

    dataset = GenomicDataset(
        data=sample_sequences, tokenizer=tokenizer, max_length=seq_length
    )

    loaders = create_genomic_dataloaders(
        train_data=dataset, tokenizer=tokenizer, batch_size=batch_size, val_split=0.2
    )

    print("   ✅ Data loaders created")
    print(f"   Train batches: {len(loaders['train'])}")
    print(f"   Validation batches: {len(loaders['val'])}")

    # Test batch processing
    sample_batch = next(iter(loaders["train"]))
    print("📊 Sample batch:")
    print(f"   Input IDs shape: {sample_batch.input_ids.shape}")
    print(f"   Labels shape: {sample_batch.labels.shape}")
    print(f"   Attention mask shape: {sample_batch.attention_mask.shape}")

    return sample_batch.input_ids


def run_complete_demo(seq_length: int = 128, batch_size: int = 4):
    """Run the complete framework demonstration."""
    print("🎉 HYENA-GLT COMPLETE FRAMEWORK DEMO")
    print("=" * 60)
    print("Demonstrating BLT + Hyena + Enhanced Training + Interpretability")
    print("=" * 60)

    # Configuration
    config = HyenaGLTConfig(
        hidden_size=256,
        num_layers=3,
        num_attention_heads=8,
        max_position_embeddings=seq_length,
        dynamic_patching=True,  # Enable BLT
        local_encoder_layers=0,  # Simplify for demo
        local_decoder_layers=0,
    )

    print("🔧 Configuration:")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Number of layers: {config.num_layers}")
    print(f"   Max sequence length: {config.max_position_embeddings}")
    print(f"   Dynamic patching (BLT): {config.dynamic_patching}")

    # 1. Data Processing Demo
    sample_data = demo_data_processing(seq_length, batch_size)

    # 2. Model Architecture Demo
    model, outputs = demo_model_architecture(config)

    # 3. Interpretability Demo
    sequence_analysis, batch_analysis = demo_interpretability(model, sample_data)

    # 4. Training Pipeline Demo
    training_pipeline = demo_training_pipeline()

    # Final Summary
    print("\n🎯 DEMO SUMMARY")
    print("=" * 50)
    print("✅ Data Processing: Tokenization and batch creation")
    print("✅ Model Architecture: BLT + Hyena forward pass")
    print("✅ Token Compression: Achieved compression ratio")
    print("✅ Interpretability: Sequence and batch analysis")
    print("✅ Training Pipeline: Enhanced training infrastructure")
    print("\n🚀 Framework ready for genomic modeling!")

    return {
        "config": config,
        "model": model,
        "outputs": outputs,
        "interpretability": {
            "sequence_analysis": sequence_analysis,
            "batch_analysis": batch_analysis,
        },
        "training_pipeline": training_pipeline,
    }


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Hyena-GLT Framework Demo")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Length of genomic sequences (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run demo
    run_complete_demo(seq_length=args.sequence_length, batch_size=args.batch_size)

    print("\n📊 Demo completed successfully!")
    print("All framework components verified and operational.")


if __name__ == "__main__":
    main()
