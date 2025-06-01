#!/usr/bin/env python3
"""
Complete example showing how to configure and run Hyena-GLT preprocessing and training
with different input data types and configurations.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def create_sample_data():
    """Create sample genomic data files for demonstration."""
    # Create sample FASTA file
    sample_fasta = """
>seq1_human_chr1
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
>seq2_mouse_chr2
CGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA
TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
>seq3_plant_chloroplast
AATTCCGGAATTCCGGAATTCCGGAATTCCGGAATTCCGGAATTCCGGAATTCCGGAATT
CCGGTTAACCGGTTAACCGGTTAACCGGTTAACCGGTTAACCGGTTAACCGGTTAACC
""".strip()

    os.makedirs("example_data", exist_ok=True)
    with open("example_data/sample_sequences.fasta", "w") as f:
        f.write(sample_fasta)
    
    print("✓ Created sample FASTA file: example_data/sample_sequences.fasta")


def create_preprocessing_configs():
    """Create different preprocessing configuration examples."""
    
    # Basic DNA sequence classification config
    basic_config = {
        "preprocessing": {
            "task": "sequence_classification",
            "input_path": "example_data/sample_sequences.fasta",
            "max_length": 512,
            "min_length": 100,
            "sequence_type": "dna"
        },
        "tokenization": {
            "kmer_size": 6,
            "vocab_size": 4096
        },
        "filtering": {
            "filter_n_bases": True,
            "max_ambiguous_fraction": 0.1
        },
        "splitting": {
            "train_fraction": 0.7,
            "val_fraction": 0.2,
            "test_fraction": 0.1,
            "random_seed": 42
        },
        "output": {
            "output_dir": "processed_basic/",
            "format": "hdf5",
            "compress": True
        },
        "system": {
            "num_workers": 2,
            "show_progress": True
        }
    }
    
    # Advanced preprocessing config with quality control
    advanced_config = {
        "preprocessing": {
            "task": "sequence_classification",
            "input_path": "example_data/sample_sequences.fasta",
            "max_length": 1024,
            "min_length": 50,
            "sequence_type": "dna",
            "quality_threshold": 20.0,
            "max_n_ratio": 0.05,
            "remove_duplicates": True,
            "normalize_case": True,
            "filter_non_standard": True
        },
        "tokenization": {
            "kmer_size": 8,
            "vocab_size": 8192
        },
        "filtering": {
            "filter_n_bases": True,
            "filter_repeats": True,
            "min_quality": 15.0,
            "max_ambiguous_fraction": 0.05
        },
        "splitting": {
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "random_seed": 123
        },
        "output": {
            "output_dir": "processed_advanced/",
            "format": "hdf5",
            "compress": True,
            "chunk_size": 5000
        },
        "system": {
            "num_workers": 4,
            "memory_limit": "4GB",
            "show_progress": True
        }
    }
    
    os.makedirs("configs", exist_ok=True)
    
    with open("configs/basic_preprocessing.json", "w") as f:
        json.dump(basic_config, f, indent=2)
    
    with open("configs/advanced_preprocessing.json", "w") as f:
        json.dump(advanced_config, f, indent=2)
    
    print("✓ Created preprocessing configs:")
    print("  - configs/basic_preprocessing.json")
    print("  - configs/advanced_preprocessing.json")


def create_training_configs():
    """Create training configuration examples."""
    
    # Small model training config
    small_model_config = {
        "model": {
            "model_type": "hyena_glt",
            "vocab_size": 4096,
            "hidden_size": 256,
            "num_layers": 6,
            "num_attention_heads": 8,
            "max_position_embeddings": 16384,
            "sequence_type": "dna",
            "kmer_size": 6
        },
        "training": {
            "data_path": "processed_basic/",
            "output_dir": "training_small/",
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "max_length": 512,
            "warmup_steps": 100,
            "save_steps": 200,
            "eval_steps": 50,
            "logging_steps": 10
        },
        "data": {
            "train_file": "processed_basic/train.hdf5",
            "val_file": "processed_basic/val.hdf5",
            "test_file": "processed_basic/test.hdf5"
        },
        "system": {
            "device": "auto",
            "mixed_precision": False,
            "gradient_checkpointing": False,
            "num_workers": 2
        },
        "logging": {
            "log_level": "INFO",
            "wandb": False
        }
    }
    
    # Large model training config
    large_model_config = {
        "model": {
            "model_type": "hyena_glt",
            "vocab_size": 8192,
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 32768,
            "sequence_type": "dna",
            "kmer_size": 8
        },
        "training": {
            "data_path": "processed_advanced/",
            "output_dir": "training_large/",
            "epochs": 10,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "max_length": 1024,
            "warmup_steps": 500,
            "save_steps": 500,
            "eval_steps": 100,
            "logging_steps": 25
        },
        "data": {
            "train_file": "processed_advanced/train.hdf5",
            "val_file": "processed_advanced/val.hdf5",
            "test_file": "processed_advanced/test.hdf5"
        },
        "system": {
            "device": "auto",
            "mixed_precision": True,
            "gradient_checkpointing": True,
            "num_workers": 4
        },
        "logging": {
            "log_level": "INFO",
            "wandb": False
        }
    }
    
    with open("configs/small_model_training.json", "w") as f:
        json.dump(small_model_config, f, indent=2)
    
    with open("configs/large_model_training.json", "w") as f:
        json.dump(large_model_config, f, indent=2)
    
    print("✓ Created training configs:")
    print("  - configs/small_model_training.json")
    print("  - configs/large_model_training.json")


def run_preprocessing_examples():
    """Run preprocessing examples with different configurations."""
    print("\n" + "="*60)
    print("RUNNING PREPROCESSING EXAMPLES")
    print("="*60)
    
    # Example 1: Basic preprocessing with command line
    print("\n1. Basic preprocessing with command line arguments:")
    cmd = [
        "hyena-glt-preprocess",
        "--input", "example_data/sample_sequences.fasta",
        "--output", "processed_cli/",
        "--task", "sequence_classification",
        "--max-length", "512",
        "--kmer-size", "6",
        "--log-level", "INFO"
    ]
    print("Command:", " ".join(cmd))
    print("Note: This would run the preprocessing with CLI arguments")
    
    # Example 2: Preprocessing with config file
    print("\n2. Preprocessing with configuration file:")
    cmd = [
        "hyena-glt-preprocess",
        "--config", "configs/basic_preprocessing.json"
    ]
    print("Command:", " ".join(cmd))
    print("Note: This would use the JSON configuration file")
    
    # Example 3: Advanced preprocessing
    print("\n3. Advanced preprocessing with quality control:")
    cmd = [
        "hyena-glt-preprocess",
        "--config", "configs/advanced_preprocessing.json",
        "--log-level", "DEBUG"
    ]
    print("Command:", " ".join(cmd))
    print("Note: This would run advanced preprocessing with detailed logging")


def run_training_examples():
    """Show training examples with different configurations."""
    print("\n" + "="*60)
    print("RUNNING TRAINING EXAMPLES")
    print("="*60)
    
    # Example 1: Quick training with CLI
    print("\n1. Quick training with command line:")
    cmd = [
        "hyena-glt-train",
        "--model", "tiny",
        "--data", "processed_basic/",
        "--output", "training_cli/",
        "--epochs", "3",
        "--batch-size", "8"
    ]
    print("Command:", " ".join(cmd))
    print("Note: Quick training with minimal parameters")
    
    # Example 2: Small model training
    print("\n2. Small model training with config:")
    cmd = [
        "hyena-glt-train",
        "--config", "configs/small_model_training.json"
    ]
    print("Command:", " ".join(cmd))
    print("Note: Training with small model configuration")
    
    # Example 3: Large model training
    print("\n3. Large model training with optimizations:")
    cmd = [
        "hyena-glt-train",
        "--config", "configs/large_model_training.json",
        "--mixed-precision",
        "--gradient-checkpointing"
    ]
    print("Command:", " ".join(cmd))
    print("Note: Training with performance optimizations")


def show_evaluation_examples():
    """Show evaluation examples."""
    print("\n" + "="*60)
    print("EVALUATION EXAMPLES")
    print("="*60)
    
    print("\n1. Basic model evaluation:")
    cmd = [
        "hyena-glt-eval",
        "--model", "training_small/final_model",
        "--data", "processed_basic/test.hdf5",
        "--output", "evaluation_results/"
    ]
    print("Command:", " ".join(cmd))
    
    print("\n2. Detailed evaluation with metrics:")
    cmd = [
        "hyena-glt-eval",
        "--model", "training_large/final_model",
        "--data", "processed_advanced/test.hdf5",
        "--output", "detailed_evaluation/",
        "--metrics", "accuracy,f1,precision,recall",
        "--batch-size", "16"
    ]
    print("Command:", " ".join(cmd))


def create_pipeline_script():
    """Create a complete pipeline execution script."""
    script_content = '''#!/bin/tcsh
# Complete Hyena-GLT preprocessing and training pipeline

echo "Starting Hyena-GLT Pipeline..."

# Step 1: Preprocess data
echo "\\n=== STEP 1: Data Preprocessing ==="
hyena-glt-preprocess \\
  --input example_data/sample_sequences.fasta \\
  --output processed_data/ \\
  --task sequence_classification \\
  --max-length 1024 \\
  --kmer-size 6 \\
  --train-split 0.8 \\
  --val-split 0.1 \\
  --test-split 0.1

if ($status != 0) then
    echo "Error: Preprocessing failed"
    exit 1
endif

# Step 2: Train model
echo "\\n=== STEP 2: Model Training ==="
hyena-glt-train \\
  --model small \\
  --data processed_data/ \\
  --output model_output/ \\
  --epochs 5 \\
  --batch-size 16 \\
  --learning-rate 1e-4

if ($status != 0) then
    echo "Error: Training failed"
    exit 1
endif

# Step 3: Evaluate model
echo "\\n=== STEP 3: Model Evaluation ==="
hyena-glt-eval \\
  --model model_output/final_model \\
  --data processed_data/test.hdf5 \\
  --output evaluation_results/

if ($status != 0) then
    echo "Error: Evaluation failed"
    exit 1
endif

echo "\\n=== Pipeline completed successfully! ==="
echo "Results available in:"
echo "  - Processed data: processed_data/"
echo "  - Trained model: model_output/"
echo "  - Evaluation: evaluation_results/"
'''
    
    with open("run_pipeline.csh", "w") as f:
        f.write(script_content)
    
    os.chmod("run_pipeline.csh", 0o755)
    print("✓ Created pipeline script: run_pipeline.csh")


def main():
    """Main function to demonstrate configuration and usage."""
    print("Hyena-GLT Configuration and Pipeline Demo")
    print("="*50)
    
    # Create example files and configurations
    create_sample_data()
    create_preprocessing_configs()
    create_training_configs()
    create_pipeline_script()
    
    # Show examples
    run_preprocessing_examples()
    run_training_examples()
    show_evaluation_examples()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Files created:
  - example_data/sample_sequences.fasta  # Sample genomic data
  - configs/basic_preprocessing.json     # Basic preprocessing config
  - configs/advanced_preprocessing.json  # Advanced preprocessing config
  - configs/small_model_training.json    # Small model training config
  - configs/large_model_training.json    # Large model training config
  - run_pipeline.csh                     # Complete pipeline script

To run the complete pipeline:
  ./run_pipeline.csh

To run individual steps:
  1. Preprocessing: hyena-glt-preprocess --config configs/basic_preprocessing.json
  2. Training:     hyena-glt-train --config configs/small_model_training.json
  3. Evaluation:   hyena-glt-eval --model model_output/ --data processed_data/test.hdf5

For more options, use --help with any command:
  hyena-glt-preprocess --help
  hyena-glt-train --help
  hyena-glt-eval --help
""")


if __name__ == "__main__":
    main()
