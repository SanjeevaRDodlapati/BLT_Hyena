# Hyena-GLT Data Preprocessing and Training Configuration Guide

## Overview

The Hyena-GLT framework provides CLI tools for data preprocessing and training with flexible configuration options. This guide shows how to configure and run data preprocessing and training steps when input data paths are provided.

## CLI Tools Available

1. **`hyena-glt-preprocess`** - Data preprocessing and preparation
2. **`hyena-glt-train`** - Model training 
3. **`hyena-glt-eval`** - Model evaluation

## 1. Data Preprocessing Configuration

### Basic Usage with Command Line Arguments

```bash
# Basic sequence preprocessing
hyena-glt-preprocess \
  --input data/genome_sequences.fasta \
  --output processed_data/ \
  --task sequence_classification \
  --max-length 1024 \
  --min-length 50

# Variant effect preprocessing
hyena-glt-preprocess \
  --input data/variants.vcf \
  --reference data/reference_genome.fa \
  --output variant_data/ \
  --task variant_effect

# Auto-detect task type
hyena-glt-preprocess \
  --input data/mixed_genomic_data/ \
  --output processed/ \
  --task auto
```

### Configuration File Usage

Create a preprocessing configuration file (see `examples/preprocessing_config.json`):

```bash
# Use configuration file
hyena-glt-preprocess --config examples/preprocessing_config.json

# Override specific parameters
hyena-glt-preprocess \
  --config examples/preprocessing_config.json \
  --max-length 2048 \
  --batch-size 32
```

### Supported Input Formats

- **FASTA/FA** - Sequence files (`.fa`, `.fasta`, `.fas`)
- **FASTQ/FQ** - Sequence files with quality scores
- **VCF** - Variant call format files
- **BED/GTF/GFF** - Genomic annotation files
- **JSON/JSONL** - Structured genomic data

### Auto-Detection Features

The preprocessing tool automatically detects:
- File format based on extension and content
- Sequence type (DNA, RNA, protein)
- Task type based on input format
- Optimal processing parameters

## 2. Training Configuration

### Basic Training Command

```bash
# Quick training with minimal parameters
hyena-glt-train \
  --model tiny \
  --data processed_data/ \
  --output training_output/ \
  --epochs 10 \
  --batch-size 16

# Training with config file
hyena-glt-train --config examples/training_config.json

# Resume from checkpoint
hyena-glt-train \
  --config examples/training_config.json \
  --resume training_output/checkpoints/latest.pt
```

### Configuration File Structure

The training configuration includes:

- **Model parameters**: Architecture, size, vocabulary
- **Training parameters**: Learning rate, batch size, epochs
- **Data paths**: Training, validation, test files
- **System settings**: Device, mixed precision, workers
- **Logging**: Weights & Biases, log levels

## 3. Complete Pipeline Example

### Step 1: Preprocess Data

```bash
# Create preprocessing config
cat > preprocess_config.json << 'EOF'
{
  "preprocessing": {
    "input_path": "data/raw_sequences.fasta",
    "max_length": 1024,
    "min_length": 50,
    "sequence_type": "dna"
  },
  "output": {
    "output_dir": "processed_data/",
    "format": "hdf5"
  },
  "splitting": {
    "train_fraction": 0.8,
    "val_fraction": 0.1,
    "test_fraction": 0.1
  }
}
EOF

# Run preprocessing
hyena-glt-preprocess --config preprocess_config.json
```

### Step 2: Train Model

```bash
# Create training config
cat > train_config.json << 'EOF'
{
  "model": {
    "hidden_size": 768,
    "num_layers": 12,
    "sequence_type": "dna"
  },
  "training": {
    "data_path": "processed_data/",
    "output_dir": "model_output/",
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 1e-4
  }
}
EOF

# Run training
hyena-glt-train --config train_config.json
```

### Step 3: Evaluate Model

```bash
# Evaluate trained model
hyena-glt-eval \
  --model model_output/final_model \
  --data processed_data/test.hdf5 \
  --output evaluation_results/
```

## 4. Advanced Configuration Options

### Preprocessing Options

```bash
# Advanced preprocessing with filtering
hyena-glt-preprocess \
  --input data/sequences.fasta \
  --output processed/ \
  --max-length 2048 \
  --filter-n \
  --filter-repeats \
  --min-quality 20.0 \
  --max-ambiguous 0.05 \
  --kmer-size 6 \
  --vocab-size 8192
```

### Training with Mixed Precision

```bash
# Training with performance optimizations
hyena-glt-train \
  --config config.json \
  --mixed-precision \
  --gradient-checkpointing \
  --compile \
  --num-workers 8
```

### Distributed Training

```bash
# Multi-GPU training
torchrun --nproc_per_node=4 \
  $(which hyena-glt-train) \
  --config config.json \
  --distributed
```

## 5. Configuration Templates

### Small Model Configuration

```json
{
  "model": {
    "hidden_size": 256,
    "num_layers": 6,
    "num_attention_heads": 8
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 2e-4,
    "epochs": 15
  }
}
```

### Large Model Configuration

```json
{
  "model": {
    "hidden_size": 1024,
    "num_layers": 24,
    "num_attention_heads": 16
  },
  "training": {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "epochs": 50,
    "gradient_checkpointing": true
  }
}
```

## 6. Best Practices

1. **Start with preprocessing**: Always preprocess data first
2. **Use configuration files**: Better for reproducibility
3. **Monitor resources**: Check GPU memory and disk space
4. **Validate data**: Verify preprocessing output before training
5. **Save checkpoints**: Enable regular model saving
6. **Use auto-detection**: Let the tool detect formats and tasks
7. **Progressive training**: Start with small models, scale up

## 7. Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or sequence length
2. **Format errors**: Check input file format and structure
3. **Path errors**: Use absolute paths for data files
4. **Device errors**: Verify CUDA availability for GPU training

### Debug Commands

```bash
# Check preprocessing output
hyena-glt-preprocess --input data.fa --output test/ --log-level DEBUG

# Dry run training
hyena-glt-train --config config.json --dry-run

# Validate data
hyena-glt-eval --data processed_data/ --validate-only
```

## 8. Output Structure

After preprocessing and training, you'll have:

```
processed_data/
├── train.hdf5           # Training data
├── val.hdf5             # Validation data
├── test.hdf5            # Test data
├── tokenizer.json       # Tokenizer configuration
└── preprocessing_stats.json  # Processing statistics

model_output/
├── config.json          # Model configuration
├── pytorch_model.bin    # Trained model weights
├── tokenizer.json       # Tokenizer files
├── training.log         # Training logs
└── checkpoints/         # Training checkpoints
```

This comprehensive configuration system allows you to efficiently preprocess genomic data and train Hyena-GLT models with full control over all parameters.
