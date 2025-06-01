# Data Preprocessing Guide for Hyena-GLT Framework

## Overview

This guide provides comprehensive instructions for preparing input data for the Hyena-GLT framework. It covers configuration files, CLI commands, and best practices for different types of biological sequence data.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Files](#configuration-files)
3. [CLI Commands](#cli-commands)
4. [Data Types and Formats](#data-types-and-formats)
5. [Batch Processing](#batch-processing)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Basic DNA Sequence Preprocessing

```tcsh
# Using YAML configuration (recommended)
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --config configs/dna_classification_preprocessing.yml

# Or using direct CLI arguments
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --input /path/to/your/genome_sequences.fasta \
  --output processed_data/dna/ \
  --task sequence_classification \
  --max-length 1024 \
  --format hdf5
```

### 2. Edit Configuration for Your Data

Edit `configs/dna_classification_preprocessing.yml`:

```yaml
preprocessing:
  input_path: /path/to/your/actual/data.fasta  # ← Change this path
  task: sequence_classification
  max_length: 1024
  # ... rest of configuration
```

## Configuration Files

### Available YAML Configurations

| Configuration File | Purpose | Sequence Type |
|-------------------|---------|---------------|
| `dna_classification_preprocessing.yml` | Basic DNA sequence classification | DNA |
| `variant_effect_preprocessing.yml` | Variant effect prediction | DNA + VCF |
| `protein_preprocessing.yml` | Protein sequence analysis | Protein |
| `large_scale_preprocessing.yml` | Large genomic datasets | DNA |

### DNA Classification Configuration

```yaml
# configs/dna_classification_preprocessing.yml
preprocessing:
  task: sequence_classification
  input_path: data/genome_sequences.fasta
  max_length: 1024
  min_length: 50
  sequence_type: dna
  remove_duplicates: true
  normalize_case: true
  filter_non_standard: true

tokenization:
  vocab_size: 4096
  kmer_size: 6

filtering:
  filter_n_bases: true
  max_ambiguous_fraction: 0.1

splitting:
  train_fraction: 0.8
  val_fraction: 0.1
  test_fraction: 0.1
  random_seed: 42

output:
  output_dir: processed_data/dna_classification/
  format: hdf5
  compress: true
  chunk_size: 10000

system:
  num_workers: 4
  memory_limit: 8GB
  show_progress: true
```

### Variant Effect Configuration

```yaml
# configs/variant_effect_preprocessing.yml
preprocessing:
  task: variant_effect
  input_path: data/variants.vcf
  reference_path: data/reference_genome.fa
  max_length: 2048
  min_length: 100
  sequence_type: dna

tokenization:
  vocab_size: 8192
  kmer_size: 8

output:
  output_dir: processed_data/variant_effects/
  format: hdf5

system:
  num_workers: 8
  memory_limit: 16GB
```

### Protein Sequence Configuration

```yaml
# configs/protein_preprocessing.yml
preprocessing:
  task: sequence_classification
  input_path: data/proteins.fasta
  max_length: 512
  min_length: 20
  sequence_type: protein
  remove_duplicates: true

tokenization:
  vocab_size: 2048
  kmer_size: 3  # Smaller k-mer for proteins

output:
  output_dir: processed_data/proteins/
  format: hdf5
```

## CLI Commands

### Basic Preprocessing Commands

```tcsh
# DNA sequences
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --input data/genome_sequences.fasta \
  --output processed_data/dna/ \
  --task sequence_classification \
  --max-length 1024 \
  --format hdf5

# Protein sequences
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --input data/proteins.fasta \
  --output processed_data/proteins/ \
  --task sequence_classification \
  --sequence-type protein \
  --max-length 512

# Variant analysis
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --input data/variants.vcf \
  --reference data/reference_genome.fa \
  --output processed_data/variants/ \
  --task variant_effect \
  --max-length 2048
```

### Configuration-Based Commands

```tcsh
# Using YAML configuration files (recommended)
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --config configs/dna_classification_preprocessing.yml

crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --config configs/protein_preprocessing.yml

crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --config configs/variant_effect_preprocessing.yml
```

### Large-Scale Data Processing

```tcsh
# For large datasets with chunking and parallel processing
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --input data/large_genome_dataset/ \
  --output processed_data/large_scale/ \
  --max-length 4096 \
  --overlap 2048 \
  --num-workers 16 \
  --memory-limit 32GB \
  --chunk-size 50000
```

## Data Types and Formats

### Supported Input Formats

| Format | Description | Example Usage |
|--------|-------------|---------------|
| **FASTA** | Single or multiple sequence files | `--input data/sequences.fasta` |
| **Directory** | Directory with multiple FASTA files | `--input data/genome_dataset/` |
| **VCF** | Variant Call Format (with reference) | `--input data/variants.vcf --reference data/ref.fa` |
| **BED** | Genomic coordinates (with reference) | `--input data/regions.bed --reference data/ref.fa` |
| **JSONL** | Structured data with sequences and labels | `--input data/structured_data.jsonl` |

### Sequence Types

- **DNA**: `--sequence-type dna` (default)
- **Protein**: `--sequence-type protein`
- **RNA**: `--sequence-type rna`

### Output Formats

- **HDF5**: `--format hdf5` (recommended, compressed)
- **JSON**: `--format json`
- **Parquet**: `--format parquet`

## Batch Processing

### Using Batch Scripts

```tcsh
# Make scripts executable
chmod +x job_scripts/*.csh

# Run complete pipeline
./job_scripts/complete_pipeline.csh /path/to/your/data.fasta

# Run preprocessing only
./job_scripts/preprocess_job.csh
```

### Custom Batch Script Template

```tcsh
#!/bin/tcsh
# Custom preprocessing job

echo "Starting preprocessing: `date`"

# Set environment
setenv PYTHONPATH "/home/sdodl001/BLT_Hyena:$PYTHONPATH"
setenv CUDA_VISIBLE_DEVICES "0"

# Create directories
mkdir -p processed_data/
mkdir -p logs/

# Run preprocessing
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --config configs/your_custom_config.yml \
  > logs/preprocessing.log 2>&1

if ($status == 0) then
    echo "Preprocessing completed successfully!"
else
    echo "Preprocessing failed with exit code: $status"
    exit 1
endif

echo "Job completed: `date`"
```

## Configuration Parameters Reference

### Preprocessing Section

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `task` | string | Task type: sequence_classification, variant_effect | Required |
| `input_path` | string | Path to input data file/directory | Required |
| `reference_path` | string | Path to reference genome (for VCF/BED) | Optional |
| `max_length` | int | Maximum sequence length | 1024 |
| `min_length` | int | Minimum sequence length | 50 |
| `sequence_type` | string | dna, protein, rna | dna |
| `remove_duplicates` | bool | Remove duplicate sequences | true |
| `normalize_case` | bool | Convert to uppercase | true |
| `filter_non_standard` | bool | Remove non-standard bases | true |
| `stride` | int | Sliding window stride for long sequences | null |
| `overlap` | int | Overlap between chunks | null |

### Tokenization Section

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `vocab_size` | int | Vocabulary size for tokenization | 4096 |
| `kmer_size` | int | K-mer size for tokenization | 6 |

### Filtering Section

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `filter_n_bases` | bool | Filter sequences with N bases | true |
| `max_ambiguous_fraction` | float | Max fraction of ambiguous bases | 0.1 |
| `filter_repeats` | bool | Filter repetitive sequences | false |
| `min_quality` | float | Minimum quality score | null |

### Splitting Section

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `train_fraction` | float | Training set fraction | 0.8 |
| `val_fraction` | float | Validation set fraction | 0.1 |
| `test_fraction` | float | Test set fraction | 0.1 |
| `random_seed` | int | Random seed for splitting | 42 |

### Output Section

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | string | Output directory path | Required |
| `format` | string | Output format: hdf5, json, parquet | hdf5 |
| `compress` | bool | Compress output files | true |
| `chunk_size` | int | Number of sequences per chunk | 10000 |

### System Section

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_workers` | int | Number of parallel workers | 4 |
| `memory_limit` | string | Memory limit (e.g., "8GB") | null |
| `show_progress` | bool | Show progress bars | true |

## Common Workflows

### 1. DNA Sequence Classification

```tcsh
# Step 1: Prepare configuration
# Edit configs/dna_classification_preprocessing.yml with your data path

# Step 2: Run preprocessing
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --config configs/dna_classification_preprocessing.yml

# Step 3: Train model
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \
  --config configs/basic_training.yml
```

### 2. Variant Effect Prediction

```tcsh
# Step 1: Prepare VCF and reference files
# Edit configs/variant_effect_preprocessing.yml

# Step 2: Run preprocessing
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --config configs/variant_effect_preprocessing.yml

# Step 3: Train variant effect model
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \
  --config configs/variant_training.yml
```

### 3. Protein Function Prediction

```tcsh
# Step 1: Prepare protein FASTA
# Edit configs/protein_preprocessing.yml

# Step 2: Run preprocessing
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --config configs/protein_preprocessing.yml

# Step 3: Train protein model
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.train \
  --config configs/protein_training.yml
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues

```yaml
# Reduce memory usage in config
system:
  num_workers: 2  # Reduce workers
  memory_limit: 4GB  # Set lower limit

output:
  chunk_size: 5000  # Smaller chunks
```

#### 2. Large Files

```yaml
# For very large files
preprocessing:
  stride: 2048  # Use sliding window
  overlap: 1024  # With overlap

system:
  num_workers: 16  # More parallel processing
  memory_limit: 32GB
```

#### 3. CUDA Out of Memory

```tcsh
# Set environment variable to limit GPU usage
setenv CUDA_VISIBLE_DEVICES "0"  # Use only GPU 0
```

#### 4. File Format Issues

```tcsh
# Validate FASTA file
head -20 /path/to/your/data.fasta

# Check file encoding
file /path/to/your/data.fasta
```

### Log Analysis

```tcsh
# Check preprocessing logs
tail -f logs/preprocessing.log

# Check for errors
grep -i error logs/preprocessing.log
grep -i "failed" logs/preprocessing.log
```

### Performance Optimization

1. **Use HDF5 format** for better I/O performance
2. **Set appropriate chunk_size** based on available memory
3. **Use compression** to reduce storage
4. **Adjust num_workers** based on CPU cores
5. **Use SSD storage** for faster I/O

## Advanced Usage

### Custom Tokenization

```yaml
tokenization:
  vocab_size: 16384  # Larger vocabulary
  kmer_size: 8       # Larger k-mers
  use_bpe: true      # Use Byte-Pair Encoding
```

### Multi-GPU Processing

```tcsh
# Set multiple GPUs
setenv CUDA_VISIBLE_DEVICES "0,1,2,3"

# Use distributed processing
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --config configs/large_scale_preprocessing.yml \
  --distributed
```

### Custom Data Loaders

```python
# For custom data formats, extend the preprocessing pipeline
from hyena_glt.data.preprocessing import BasePreprocessor

class CustomPreprocessor(BasePreprocessor):
    def load_data(self, path):
        # Custom data loading logic
        pass
```

## Reference Files

- **Configuration Generator**: `examples/data_pipeline_configuration_guide.py`
- **Batch Scripts**: `job_scripts/`
- **Example Configs**: `configs/`
- **CLI Tools**: `hyena_glt/cli/`

## Getting Help

1. Check the logs in `logs/preprocessing.log`
2. Run with `--verbose` flag for more details
3. Use `--help` to see all available options:

```tcsh
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess --help
```

## Next Steps

After preprocessing, you can proceed with:

1. **Model Training**: See `docs/TRAINING_AND_INTERPRETABILITY_GUIDE.md`
2. **Fine-tuning**: See `docs/FINE_TUNING.md`
3. **Evaluation**: Use `hyena_glt.cli.eval`

---

*This guide is part of the Hyena-GLT framework documentation. For updates and additional resources, check the main repository.*
