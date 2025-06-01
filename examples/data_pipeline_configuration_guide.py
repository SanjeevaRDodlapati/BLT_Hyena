#!/usr/bin/env python3
"""
Data Pipeline Configuration Guide for Hyena-GLT Framework
=========================================================

This guide demonstrates how to configure and run data preprocessing and training
jobs when input data paths are provided. It covers:

1. CLI-based preprocessing configuration
2. YAML configuration files for complex pipelines
3. Training pipeline setup with data paths
4. Complete end-to-end workflow examples

Use with: crun -p ~/envs/blthyenapy312/ python data_pipeline_configuration_guide.py
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any


def create_preprocessing_config_examples():
    """Create example preprocessing configuration files for different scenarios."""
    
    print("🔧 Creating Preprocessing Configuration Examples...")
    
    # Create configs directory
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # 1. Basic DNA sequence classification preprocessing
    dna_classification_config = {
        "preprocessing": {
            "task": "sequence_classification",
            "input_path": "data/genome_sequences.fasta",  # Your input data path
            "max_length": 1024,
            "min_length": 50,
            "sequence_type": "dna",
            "remove_duplicates": True,
            "normalize_case": True,
            "filter_non_standard": True
        },
        "tokenization": {
            "vocab_size": 4096,
            "kmer_size": 6
        },
        "filtering": {
            "filter_n_bases": True,
            "max_ambiguous_fraction": 0.1
        },
        "splitting": {
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "random_seed": 42
        },
        "output": {
            "output_dir": "processed_data/dna_classification/",
            "format": "hdf5",
            "compress": True,
            "chunk_size": 10000
        },
        "system": {
            "num_workers": 4,
            "memory_limit": "8GB",
            "show_progress": True
        }
    }
    
    # 2. Variant effect prediction preprocessing
    variant_config = {
        "preprocessing": {
            "task": "variant_effect",
            "input_path": "data/variants.vcf",  # Your VCF file path
            "reference_path": "data/reference_genome.fa",  # Reference genome path
            "max_length": 2048,
            "min_length": 100,
            "sequence_type": "dna"
        },
        "tokenization": {
            "vocab_size": 8192,
            "kmer_size": 8
        },
        "output": {
            "output_dir": "processed_data/variant_effects/",
            "format": "hdf5"
        },
        "system": {
            "num_workers": 8,
            "memory_limit": "16GB"
        }
    }
    
    # 3. Protein sequence preprocessing
    protein_config = {
        "preprocessing": {
            "task": "sequence_classification", 
            "input_path": "data/proteins.fasta",  # Your protein data path
            "max_length": 512,
            "min_length": 20,
            "sequence_type": "protein",
            "remove_duplicates": True
        },
        "tokenization": {
            "vocab_size": 2048,
            "kmer_size": 3  # Smaller k-mer for proteins
        },
        "output": {
            "output_dir": "processed_data/proteins/",
            "format": "hdf5"
        }
    }
    
    # 4. Large-scale genomic data preprocessing
    large_scale_config = {
        "preprocessing": {
            "task": "sequence_classification",
            "input_path": "data/large_genome_dataset/",  # Directory with multiple files
            "max_length": 4096,
            "min_length": 100,
            "sequence_type": "dna",
            "stride": 2048,  # 50% overlap for long sequences
            "overlap": 2048
        },
        "tokenization": {
            "vocab_size": 16384,
            "kmer_size": 8
        },
        "filtering": {
            "filter_n_bases": True,
            "filter_repeats": True,
            "min_quality": 10.0
        },
        "output": {
            "output_dir": "processed_data/large_scale/",
            "format": "hdf5",
            "compress": True,
            "chunk_size": 50000
        },
        "system": {
            "num_workers": 16,
            "memory_limit": "32GB",
            "show_progress": True
        }
    }
    
    # Save all configs
    configs = {
        "dna_classification_preprocessing.yml": dna_classification_config,
        "variant_effect_preprocessing.yml": variant_config,
        "protein_preprocessing.yml": protein_config,
        "large_scale_preprocessing.yml": large_scale_config
    }
    
    for filename, config in configs.items():
        config_path = config_dir / filename
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2, default_flow_style=False)
        print(f"  ✓ Created: {config_path}")
    
    return configs


def create_training_config_examples():
    """Create example training configuration files."""
    
    print("\n🎯 Creating Training Configuration Examples...")
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # 1. Basic training configuration
    basic_training_config = {
        "model": {
            "model_type": "hyena_glt",
            "model_size": "small",
            "vocab_size": 4096,
            "hidden_size": 512,
            "num_layers": 8,
            "max_position_embeddings": 16384,
            "sequence_type": "dna",
            "kmer_size": 6
        },
        "training": {
            "data_path": "processed_data/dna_classification/",  # Path to preprocessed data
            "output_dir": "training_output/dna_model/",
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "max_length": 1024,
            "warmup_steps": 1000,
            "save_steps": 500,
            "eval_steps": 100
        },
        "data": {
            "train_file": "processed_data/dna_classification/train.hdf5",
            "val_file": "processed_data/dna_classification/val.hdf5",
            "test_file": "processed_data/dna_classification/test.hdf5",
            "tokenizer_path": "processed_data/dna_classification/tokenizer.json"
        },
        "optimization": {
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "mixed_precision": True
        },
        "system": {
            "device": "auto",
            "num_workers": 4,
            "gradient_checkpointing": False
        }
    }
    
    # 2. Large model training configuration
    large_model_config = {
        "model": {
            "model_type": "hyena_glt",
            "model_size": "large",
            "vocab_size": 16384,
            "hidden_size": 1024,
            "num_layers": 24,
            "max_position_embeddings": 32768,
            "sequence_type": "dna",
            "kmer_size": 8
        },
        "training": {
            "data_path": "processed_data/large_scale/",
            "output_dir": "training_output/large_model/",
            "epochs": 5,
            "batch_size": 8,  # Smaller batch for large model
            "learning_rate": 5e-5,
            "max_length": 4096,
            "warmup_steps": 2000,
            "save_steps": 1000,
            "eval_steps": 500,
            "gradient_accumulation_steps": 4
        },
        "data": {
            "train_file": "processed_data/large_scale/train.hdf5",
            "val_file": "processed_data/large_scale/val.hdf5",
            "tokenizer_path": "processed_data/large_scale/tokenizer.json"
        },
        "optimization": {
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "mixed_precision": True,
            "use_fsdp": True  # For distributed training
        },
        "system": {
            "device": "auto",
            "num_workers": 8,
            "gradient_checkpointing": True
        }
    }
    
    # 3. Fine-tuning configuration
    finetuning_config = {
        "model": {
            "pretrained_model_path": "models/pretrained_hyena_glt.pt",
            "model_type": "hyena_glt",
            "freeze_embeddings": False,
            "freeze_encoder_layers": 0  # Number of layers to freeze
        },
        "training": {
            "data_path": "processed_data/task_specific/",
            "output_dir": "training_output/finetuned_model/",
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 2e-5,  # Lower LR for fine-tuning
            "max_length": 1024,
            "warmup_steps": 500,
            "save_steps": 200
        },
        "data": {
            "train_file": "processed_data/task_specific/train.hdf5",
            "val_file": "processed_data/task_specific/val.hdf5",
            "tokenizer_path": "processed_data/task_specific/tokenizer.json"
        },
        "optimization": {
            "optimizer": "adamw",
            "weight_decay": 0.001,  # Lower weight decay for fine-tuning
            "layer_wise_decay": 0.8,
            "mixed_precision": True
        }
    }
    
    # Save training configs
    training_configs = {
        "basic_training.yml": basic_training_config,
        "large_model_training.yml": large_model_config,
        "finetuning.yml": finetuning_config
    }
    
    for filename, config in training_configs.items():
        config_path = config_dir / filename
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2, default_flow_style=False)
        print(f"  ✓ Created: {config_path}")
    
    return training_configs


def generate_cli_commands():
    """Generate CLI command examples for different scenarios."""
    
    print("\n🚀 CLI Command Examples for Your Environment:")
    print("=" * 60)
    
    print("\n1. PREPROCESSING COMMANDS:")
    print("-" * 30)
    
    # Basic preprocessing
    print("\n# Basic DNA sequence preprocessing")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \\")
    print("  --input data/genome_sequences.fasta \\")
    print("  --output processed_data/dna/ \\")
    print("  --task sequence_classification \\")
    print("  --max-length 1024 \\")
    print("  --format hdf5")
    
    # Config-based preprocessing
    print("\n# Using configuration file")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \\")
    print("  --config configs/dna_classification_preprocessing.yml")
    
    # Variant effect preprocessing
    print("\n# Variant effect preprocessing")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \\")
    print("  --input data/variants.vcf \\")
    print("  --reference data/reference_genome.fa \\")
    print("  --output processed_data/variants/ \\")
    print("  --task variant_effect \\")
    print("  --max-length 2048")
    
    # Large-scale preprocessing
    print("\n# Large-scale data preprocessing with chunking")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \\")
    print("  --input data/large_genome_dataset/ \\")
    print("  --output processed_data/large_scale/ \\")
    print("  --max-length 4096 \\")
    print("  --overlap 2048 \\")
    print("  --num-workers 16 \\")
    print("  --memory-limit 32GB \\")
    print("  --chunk-size 50000")
    
    print("\n2. TRAINING COMMANDS:")
    print("-" * 25)
    
    # Basic training
    print("\n# Basic model training")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \\")
    print("  --data processed_data/dna/ \\")
    print("  --output training_output/dna_model/ \\")
    print("  --model small \\")
    print("  --epochs 10 \\")
    print("  --batch-size 16")
    
    # Config-based training
    print("\n# Using training configuration file")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \\")
    print("  --config configs/basic_training.yml")
    
    # Large model training
    print("\n# Large model training with distributed setup")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \\")
    print("  --config configs/large_model_training.yml \\")
    print("  --distributed \\")
    print("  --num-gpus 4")
    
    # Resume training
    print("\n# Resume training from checkpoint")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \\")
    print("  --config configs/basic_training.yml \\")
    print("  --resume training_output/dna_model/checkpoints/latest.pt")
    
    print("\n3. EVALUATION COMMANDS:")
    print("-" * 25)
    
    # Model evaluation
    print("\n# Evaluate trained model")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.eval \\")
    print("  --model training_output/dna_model/final_model.pt \\")
    print("  --data processed_data/dna/test.hdf5 \\")
    print("  --output evaluation_results/ \\")
    print("  --batch-size 32")
    
    print("\n4. COMPLETE PIPELINE EXAMPLE:")
    print("-" * 35)
    
    print("\n# Step 1: Preprocess data")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \\")
    print("  --input /path/to/your/genome_data.fasta \\")
    print("  --output processed_data/my_dataset/ \\")
    print("  --task sequence_classification \\")
    print("  --max-length 1024 \\")
    print("  --train-split 0.8 \\")
    print("  --val-split 0.1 \\")
    print("  --test-split 0.1")
    
    print("\n# Step 2: Train model")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \\")
    print("  --data processed_data/my_dataset/ \\")
    print("  --output training_output/my_model/ \\")
    print("  --model small \\")
    print("  --epochs 10 \\")
    print("  --batch-size 16 \\")
    print("  --learning-rate 1e-4")
    
    print("\n# Step 3: Evaluate model")
    print("crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.eval \\")
    print("  --model training_output/my_model/final_model.pt \\")
    print("  --data processed_data/my_dataset/test.hdf5 \\")
    print("  --output evaluation_results/my_model/")


def create_batch_job_scripts():
    """Create batch job scripts for different scenarios."""
    
    print("\n📝 Creating Batch Job Scripts...")
    
    scripts_dir = Path("job_scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # 1. Preprocessing job script
    preprocess_script = """#!/bin/tcsh
# Preprocessing job script for Hyena-GLT
# Usage: ./job_scripts/preprocess_job.csh

echo "Starting data preprocessing job..."
echo "Timestamp: `date`"

# Set environment variables
setenv PYTHONPATH "/home/sdodl001/BLT_Hyena:$PYTHONPATH"
setenv CUDA_VISIBLE_DEVICES "0"

# Create output directories
mkdir -p processed_data/
mkdir -p logs/

# Run preprocessing with your data path
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \\
  --input data/your_genome_sequences.fasta \\
  --output processed_data/genome_classification/ \\
  --task sequence_classification \\
  --max-length 1024 \\
  --min-length 50 \\
  --train-split 0.8 \\
  --val-split 0.1 \\
  --test-split 0.1 \\
  --format hdf5 \\
  --compress \\
  --num-workers 8 \\
  --log-level INFO \\
  --progress > logs/preprocessing.log 2>&1

if ($status == 0) then
    echo "Preprocessing completed successfully!"
    echo "Output saved to: processed_data/genome_classification/"
else
    echo "Preprocessing failed with exit code: $status"
    exit 1
endif

echo "Job completed at: `date`"
"""
    
    # 2. Training job script
    training_script = """#!/bin/tcsh
# Training job script for Hyena-GLT
# Usage: ./job_scripts/training_job.csh

echo "Starting model training job..."
echo "Timestamp: `date`"

# Set environment variables
setenv PYTHONPATH "/home/sdodl001/BLT_Hyena:$PYTHONPATH"
setenv CUDA_VISIBLE_DEVICES "0,1,2,3"  # Adjust based on available GPUs

# Create output directories
mkdir -p training_output/
mkdir -p logs/

# Run training
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \\
  --config configs/basic_training.yml \\
  > logs/training.log 2>&1

if ($status == 0) then
    echo "Training completed successfully!"
    echo "Model saved to: training_output/"
else
    echo "Training failed with exit code: $status"
    exit 1
endif

echo "Job completed at: `date`"
"""
    
    # 3. Complete pipeline script
    pipeline_script = """#!/bin/tcsh
# Complete pipeline job script for Hyena-GLT
# Usage: ./job_scripts/complete_pipeline.csh /path/to/your/data.fasta

echo "Starting complete Hyena-GLT pipeline..."
echo "Timestamp: `date`"

# Check if data path argument is provided
if ($#argv != 1) then
    echo "Usage: $0 <input_data_path>"
    echo "Example: $0 /path/to/your/genome_sequences.fasta"
    exit 1
endif

set INPUT_DATA = $1
echo "Input data: $INPUT_DATA"

# Set environment variables
setenv PYTHONPATH "/home/sdodl001/BLT_Hyena:$PYTHONPATH"
setenv CUDA_VISIBLE_DEVICES "0"

# Create directories
mkdir -p processed_data/
mkdir -p training_output/
mkdir -p evaluation_results/
mkdir -p logs/

echo "Step 1: Data Preprocessing..."
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \\
  --input $INPUT_DATA \\
  --output processed_data/pipeline_data/ \\
  --task sequence_classification \\
  --max-length 1024 \\
  --format hdf5 \\
  --progress > logs/preprocessing.log 2>&1

if ($status != 0) then
    echo "Preprocessing failed!"
    exit 1
endif

echo "Step 2: Model Training..."
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \\
  --data processed_data/pipeline_data/ \\
  --output training_output/pipeline_model/ \\
  --model small \\
  --epochs 5 \\
  --batch-size 16 > logs/training.log 2>&1

if ($status != 0) then
    echo "Training failed!"
    exit 1
endif

echo "Step 3: Model Evaluation..."
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.eval \\
  --model training_output/pipeline_model/final_model.pt \\
  --data processed_data/pipeline_data/test.hdf5 \\
  --output evaluation_results/pipeline_model/ > logs/evaluation.log 2>&1

if ($status != 0) then
    echo "Evaluation failed!"
    exit 1
endif

echo "Pipeline completed successfully!"
echo "Results available in:"
echo "  - Processed data: processed_data/pipeline_data/"
echo "  - Trained model: training_output/pipeline_model/"
echo "  - Evaluation results: evaluation_results/pipeline_model/"
echo "Timestamp: `date`"
"""
    
    # Save scripts
    scripts = {
        "preprocess_job.csh": preprocess_script,
        "training_job.csh": training_script,
        "complete_pipeline.csh": pipeline_script
    }
    
    for filename, script in scripts.items():
        script_path = scripts_dir / filename
        with open(script_path, 'w') as f:
            f.write(script)
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"  ✓ Created: {script_path}")


def create_data_path_examples():
    """Create examples of how to specify different types of data paths."""
    
    print("\n📁 Data Path Configuration Examples:")
    print("=" * 50)
    
    path_examples = {
        "Single FASTA file": {
            "path": "/path/to/genome_sequences.fasta",
            "description": "Single file with genomic sequences",
            "preprocessing_args": "--input /path/to/genome_sequences.fasta --task sequence_classification"
        },
        "Directory with multiple files": {
            "path": "/path/to/genome_dataset/",
            "description": "Directory containing multiple FASTA files",
            "preprocessing_args": "--input /path/to/genome_dataset/ --task sequence_classification"
        },
        "VCF file for variant analysis": {
            "path": "/path/to/variants.vcf",
            "description": "VCF file with genetic variants",
            "preprocessing_args": "--input /path/to/variants.vcf --reference /path/to/reference.fa --task variant_effect"
        },
        "Protein sequences": {
            "path": "/path/to/proteins.fasta", 
            "description": "FASTA file with protein sequences",
            "preprocessing_args": "--input /path/to/proteins.fasta --task sequence_classification --sequence-type protein"
        },
        "BED file for regulatory elements": {
            "path": "/path/to/regulatory_regions.bed",
            "description": "BED file with genomic coordinates",
            "preprocessing_args": "--input /path/to/regulatory_regions.bed --reference /path/to/reference.fa --task regulatory_elements"
        },
        "JSON/JSONL structured data": {
            "path": "/path/to/structured_data.jsonl",
            "description": "Structured data with sequences and labels",
            "preprocessing_args": "--input /path/to/structured_data.jsonl --task sequence_classification"
        }
    }
    
    for data_type, info in path_examples.items():
        print(f"\n{data_type}:")
        print(f"  Path: {info['path']}")
        print(f"  Description: {info['description']}")
        print(f"  Command: crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess {info['preprocessing_args']}")


def main():
    """Main function to demonstrate data pipeline configuration."""
    
    print("🧬 Hyena-GLT Data Pipeline Configuration Guide")
    print("=" * 60)
    print("Environment: crun -p ~/envs/blthyenapy312/ python")
    print("=" * 60)
    
    # Create configuration examples
    create_preprocessing_config_examples()
    create_training_config_examples()
    
    # Generate CLI commands
    generate_cli_commands()
    
    # Create batch job scripts
    create_batch_job_scripts()
    
    # Show data path examples
    create_data_path_examples()
    
    print("\n" + "=" * 60)
    print("📋 QUICK START SUMMARY")
    print("=" * 60)
    print("\n1. For immediate use with your data:")
    print("   Replace '/path/to/your/data.fasta' with your actual data path")
    print("\n2. Run preprocessing:")
    print("   crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \\")
    print("     --input /your/data/path \\")
    print("     --output processed_data/ \\")
    print("     --task sequence_classification")
    print("\n3. Run training:")
    print("   crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \\")
    print("     --data processed_data/ \\")
    print("     --output training_output/ \\")
    print("     --model small")
    print("\n4. Use batch scripts:")
    print("   chmod +x job_scripts/complete_pipeline.csh")
    print("   ./job_scripts/complete_pipeline.csh /your/data/path.fasta")
    
    print("\n✅ Configuration files and scripts created successfully!")
    print("📁 Check the following directories:")
    print("   - configs/ - Configuration YAML files")
    print("   - job_scripts/ - Batch job scripts for tcsh")


if __name__ == "__main__":
    main()
