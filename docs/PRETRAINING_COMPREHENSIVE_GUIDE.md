# Comprehensive Pretraining Guide for Hyena-GLT Framework

## Overview

This comprehensive guide covers the complete pretraining pipeline for the Hyena-GLT genomic language model framework. The pretraining stage is crucial for developing foundation models that can understand biological sequence patterns and be fine-tuned for downstream tasks.

## Table of Contents

1. [Pretraining Fundamentals](#pretraining-fundamentals)
2. [Architecture Overview](#architecture-overview)
3. [Data Pipeline](#data-pipeline)
4. [Configuration System](#configuration-system)
5. [Training Process](#training-process)
6. [Monitoring and Evaluation](#monitoring-and-evaluation)
7. [Hardware Requirements](#hardware-requirements)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Topics](#advanced-topics)

## Pretraining Fundamentals

### What is Pretraining?

Pretraining is the initial phase of training a foundation model on large amounts of unlabeled biological sequence data. The model learns to:

- **Predict masked tokens** in DNA/protein sequences
- **Understand sequence patterns** and biological motifs
- **Capture evolutionary relationships** between sequences
- **Learn hierarchical representations** from nucleotides to functional domains

### Objectives

1. **Self-Supervised Learning**: Learn from sequence structure without labels
2. **Representation Learning**: Create meaningful embeddings for biological sequences
3. **Transfer Learning**: Build a foundation for downstream tasks
4. **Scale**: Handle genomic data at population and species levels

### Key Concepts

- **Masked Language Modeling (MLM)**: Predict masked tokens in sequences
- **Autoregressive Modeling**: Predict next tokens in sequences
- **Contrastive Learning**: Learn similarities/differences between sequences
- **Multi-Scale Learning**: Capture patterns from k-mers to genes

## Architecture Overview

### Hyena-GLT Architecture

```
Input Sequences → Tokenization → Embedding → Hyena Layers → Output Head
     ↓              ↓            ↓             ↓            ↓
   ATCG...      [1,2,3,4...]   [emb_dim]   [hidden_dim]  [vocab_size]
```

#### Key Components

1. **Tokenizer**: Converts sequences to numerical tokens
   - K-mer based tokenization (default k=6)
   - Vocabulary size: 4096-16384
   - Special tokens: [MASK], [CLS], [SEP], [PAD]

2. **Embedding Layer**: Maps tokens to dense vectors
   - Learnable embeddings for each token
   - Positional encodings for sequence order
   - Dimension: 256-1024

3. **Hyena Layers**: Core transformer-like architecture
   - Hyena operator for efficient long-range modeling
   - Multi-head attention mechanisms
   - Feed-forward networks
   - Layer normalization and residual connections

4. **Output Head**: Task-specific prediction layers
   - Masked language modeling head
   - Classification heads for downstream tasks

### Model Variants

| Model Size | Parameters | Layers | Hidden Dim | Heads | Context Length |
|------------|------------|---------|------------|-------|----------------|
| **Small**  | 125M       | 12      | 768        | 12    | 1024           |
| **Base**   | 350M       | 24      | 1024       | 16    | 2048           |
| **Large**  | 1.3B       | 48      | 1536       | 24    | 4096           |
| **XL**     | 2.7B       | 64      | 2048       | 32    | 8192           |

## Data Pipeline

### Data Sources

1. **Reference Genomes**: Human, mouse, plant genomes
2. **Population Data**: 1000 Genomes, UK Biobank
3. **Protein Sequences**: UniProt, PDB structures
4. **Functional Annotations**: Gene ontology, regulatory elements

### Preprocessing Steps

```yaml
# Example preprocessing configuration
preprocessing:
  input_path: /data/genomes/
  sequence_type: dna
  max_length: 2048
  min_length: 100
  remove_duplicates: true
  filter_n_bases: true
  max_ambiguous_fraction: 0.05

tokenization:
  vocab_size: 4096
  kmer_size: 6
  stride: 1024
  overlap: 512

output:
  format: hdf5
  chunk_size: 10000
  compress: true
```

### Data Flow

```
Raw FASTA → Quality Filter → Tokenization → Chunking → HDF5 Storage
     ↓            ↓             ↓           ↓           ↓
  Genome.fa   Clean.fa      tokens.txt   chunks/    train.h5
```

## Configuration System

### Pretraining Configuration Structure

```yaml
# configs/pretraining/base_config.yaml
model:
  name: "hyena-glt-base"
  vocab_size: 4096
  hidden_size: 1024
  num_layers: 24
  num_attention_heads: 16
  intermediate_size: 4096
  max_position_embeddings: 2048
  
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  warmup_steps: 10000
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  
data:
  train_path: "data/processed/train.h5"
  val_path: "data/processed/val.h5"
  sequence_length: 2048
  mask_probability: 0.15
  
hardware:
  use_mixed_precision: true
  gradient_checkpointing: true
  num_workers: 8
  pin_memory: true
  
logging:
  use_wandb: true
  log_interval: 100
  eval_interval: 1000
  save_interval: 5000
```

### Configuration Variants

1. **Small Model** (`small_model.yaml`): For development and testing
2. **Base Model** (`base_config.yaml`): Standard pretraining setup
3. **Large Model** (`large_model.yaml`): Production-scale training

## Training Process

### Phase 1: Data Preparation

```tcsh
# 1. Preprocess raw sequences
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --config configs/large_scale_preprocessing.yml

# 2. Validate preprocessed data
python scripts/validate_preprocessed_data.py \
  --data_path processed_data/train.h5
```

### Phase 2: Model Initialization

```tcsh
# Initialize model with configuration
crun -p ~/envs/blthyenapy312/ python scripts/run_pretraining.py \
  --config configs/pretraining/base_config.yaml \
  --mode init_only
```

### Phase 3: Pretraining Execution

```tcsh
# Single GPU training
crun -p ~/envs/blthyenapy312/ python scripts/run_pretraining.py \
  --config configs/pretraining/base_config.yaml

# Multi-GPU training (distributed)
crun -p ~/envs/blthyenapy312/ torchrun --nproc_per_node=4 \
  scripts/run_pretraining.py \
  --config configs/pretraining/base_config.yaml \
  --distributed
```

### Training Stages

#### Stage 1: Initial Learning (Epochs 1-20)
- **Objective**: Basic token prediction
- **Learning Rate**: 1e-4 with warmup
- **Batch Size**: Start small, gradually increase
- **Validation**: Every 1000 steps

#### Stage 2: Pattern Recognition (Epochs 21-60)
- **Objective**: Sequence motif learning
- **Learning Rate**: 5e-5 (reduced)
- **Batch Size**: Full capacity
- **Validation**: Every 500 steps

#### Stage 3: Fine-tuning (Epochs 61-100)
- **Objective**: Optimization and convergence
- **Learning Rate**: 1e-5 (further reduced)
- **Regularization**: Increased dropout
- **Early Stopping**: Monitor validation loss

## Monitoring and Evaluation

### Key Metrics

1. **Training Metrics**
   - Loss: Cross-entropy for masked tokens
   - Perplexity: Model uncertainty measure
   - Accuracy: Masked token prediction accuracy
   - Learning Rate: Current optimization rate

2. **Validation Metrics**
   - Validation Loss: Generalization performance
   - Validation Accuracy: Out-of-sample prediction
   - BLEU Score: Sequence generation quality

3. **Hardware Metrics**
   - GPU Utilization: Computing efficiency
   - Memory Usage: Resource consumption
   - Throughput: Sequences per second

### Logging and Visualization

```python
# Weights & Biases integration
import wandb

# Initialize logging
wandb.init(
    project="hyena-glt-pretraining",
    config=config,
    name=f"pretraining-{model_size}-{timestamp}"
)

# Log metrics during training
wandb.log({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    "lr": current_lr,
    "gpu_memory": gpu_memory_usage
})
```

### Evaluation Protocol

```tcsh
# Evaluate model checkpoints
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.eval \
  --model_path checkpoints/epoch_50/ \
  --eval_data data/processed/test.h5 \
  --metrics mlm_accuracy,perplexity,bleu

# Generate evaluation report
python scripts/generate_evaluation_report.py \
  --checkpoint_dir checkpoints/ \
  --output_dir evaluation_results/
```

## Hardware Requirements

### Minimum Requirements

- **GPU**: NVIDIA RTX 3080 (10GB VRAM)
- **RAM**: 32GB system memory
- **Storage**: 500GB SSD
- **CPU**: 8+ cores

### Recommended Setup

- **GPU**: NVIDIA A100 (40GB VRAM) or multiple RTX 4090
- **RAM**: 128GB system memory
- **Storage**: 2TB NVMe SSD
- **CPU**: 32+ cores (AMD EPYC or Intel Xeon)

### Multi-GPU Configuration

```yaml
# Multi-GPU training configuration
hardware:
  num_gpus: 4
  distributed_backend: nccl
  gradient_accumulation_steps: 8  # Effective batch size = 4 * 32 * 8 = 1024
  mixed_precision: true
  gradient_checkpointing: true
```

## Best Practices

### Data Management

1. **Data Quality**: Filter low-quality sequences
2. **Data Balance**: Ensure diverse sequence representation
3. **Data Augmentation**: Use reverse complements, masking strategies
4. **Storage**: Use efficient formats (HDF5, Parquet)

### Training Optimization

1. **Learning Rate Scheduling**: Warmup + cosine decay
2. **Batch Size Scaling**: Gradual increase during training
3. **Gradient Accumulation**: Handle memory constraints
4. **Mixed Precision**: Use fp16 for efficiency

### Model Architecture

1. **Layer Initialization**: Proper weight initialization
2. **Regularization**: Dropout, weight decay
3. **Normalization**: Layer norm for stability
4. **Residual Connections**: Prevent vanishing gradients

### Checkpointing Strategy

```tcsh
# Save checkpoints regularly
checkpoints/
├── epoch_10/
│   ├── model.pt
│   ├── optimizer.pt
│   └── config.yaml
├── epoch_20/
├── best_model/
└── latest/
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```yaml
# Solutions:
hardware:
  gradient_checkpointing: true
  gradient_accumulation_steps: 8  # Increase
training:
  batch_size: 16  # Reduce
```

#### 2. Slow Training

```yaml
# Optimizations:
data:
  num_workers: 16  # Increase data loading workers
  pin_memory: true
hardware:
  mixed_precision: true
  compile_model: true  # PyTorch 2.0+
```

#### 3. Loss Not Decreasing

```yaml
# Debugging:
training:
  learning_rate: 1e-5  # Reduce
  warmup_steps: 5000   # Increase warmup
  gradient_clip_norm: 0.5  # Add gradient clipping
```

#### 4. Validation Overfitting

```yaml
# Regularization:
model:
  dropout: 0.2  # Increase dropout
training:
  weight_decay: 0.01  # Add weight decay
  early_stopping_patience: 10
```

### Debugging Tools

```tcsh
# Monitor GPU usage
nvidia-smi -l 1

# Profile training
crun -p ~/envs/blthyenapy312/ python -m torch.profiler \
  scripts/run_pretraining.py --config configs/debug.yaml

# Check data loading
python scripts/debug_dataloader.py --config configs/pretraining/base_config.yaml
```

## Advanced Topics

### Curriculum Learning

Train on increasingly complex sequences:

```python
# Phase 1: Short sequences (512 tokens)
# Phase 2: Medium sequences (1024 tokens)  
# Phase 3: Long sequences (2048 tokens)

def curriculum_scheduler(epoch):
    if epoch < 20:
        return 512
    elif epoch < 60:
        return 1024
    else:
        return 2048
```

### Multi-Task Pretraining

```yaml
tasks:
  masked_lm:
    weight: 1.0
    mask_probability: 0.15
  
  next_sentence_prediction:
    weight: 0.5
    negative_sampling: 0.5
  
  sequence_order_prediction:
    weight: 0.3
    shuffle_probability: 0.15
```

### Domain Adaptation

```python
# Adapt pretrained model to specific domains
domains = [
    "human_genome",
    "plant_genome", 
    "microbial_genome",
    "protein_sequences"
]

# Fine-tune on domain-specific data
for domain in domains:
    model = load_pretrained_model("checkpoints/base_model/")
    domain_data = load_domain_data(domain)
    fine_tune(model, domain_data, epochs=10)
```

### Efficiency Optimizations

#### Model Parallelism

```python
# Split large models across GPUs
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# Pipeline parallelism for very large models
from torch.distributed.pipeline.sync import Pipe
model = Pipe(model, balance=[6, 6, 6, 6])  # 24 layers across 4 GPUs
```

#### Gradient Compression

```yaml
compression:
  method: "fp16"  # or "int8", "pruning"
  compression_ratio: 0.01
  sparsity_threshold: 1e-6
```

## Usage Examples

### Quick Start

```tcsh
# 1. Prepare data
crun -p ~/envs/blthyenapy312/ python -m hyena_glt.cli.preprocess \
  --input data/raw/genome.fasta \
  --output data/processed/ \
  --config configs/dna_classification_preprocessing.yml

# 2. Start pretraining
crun -p ~/envs/blthyenapy312/ python scripts/run_pretraining.py \
  --config configs/pretraining/small_model.yaml \
  --output_dir checkpoints/small_model/

# 3. Monitor progress
tensorboard --logdir checkpoints/small_model/logs/
```

### Production Pipeline

```tcsh
# Large-scale pretraining pipeline
./job_scripts/complete_pipeline.csh \
  --data_path /data/genomes/ \
  --model_size large \
  --num_gpus 8 \
  --batch_size 64 \
  --epochs 100
```

## File Structure

```
BLT_Hyena/
├── configs/pretraining/          # Pretraining configurations
│   ├── small_model.yaml
│   ├── base_config.yaml
│   └── large_model.yaml
├── scripts/
│   ├── run_pretraining.py        # Main pretraining script
│   └── validate_model.py         # Model validation
├── hyena_glt/training/           # Training framework
│   ├── pretraining.py           # Core pretraining logic
│   ├── data_utils.py            # Data loading utilities
│   └── evaluation.py            # Evaluation metrics
├── job_scripts/                  # Batch execution scripts
├── checkpoints/                  # Model checkpoints
├── logs/                        # Training logs
└── docs/                        # Documentation
```

## Next Steps

After successful pretraining:

1. **Evaluation**: Comprehensive model assessment
2. **Fine-tuning**: Task-specific adaptation
3. **Deployment**: Model serving and inference
4. **Monitoring**: Production performance tracking

## References

- **Hyena Paper**: "Hyena Hierarchy: Towards Larger Convolutional Language Models"
- **GLT Framework**: "Genomic Language Transformers for Biological Sequence Analysis"
- **Pretraining Best Practices**: "Scaling Laws for Neural Language Models"
- **Biological Applications**: "Large Language Models for Genomics"

---

*This guide provides a comprehensive overview of the pretraining process in the Hyena-GLT framework. For specific implementation details, refer to the code documentation and example configurations.*
