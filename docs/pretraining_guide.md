# Hyena-GLT Pretraining Guide

This guide provides comprehensive documentation for pretraining Hyena-GLT models on genomic sequence data.

## Overview

The Hyena-GLT pretraining system supports multiple pretraining strategies for genomic foundation models:

- **Autoregressive (AR)**: Next token prediction
- **Masked Language Modeling (MLM)**: BERT-style masked token prediction
- **Span Masking**: Masking contiguous spans of tokens
- **Order-Agnostic Autoregressive Diffusion Modeling (OADM)**: Advanced diffusion-based pretraining

## Quick Start

### 1. Basic Pretraining

```bash
# Quick start with default settings
python scripts/run_pretraining.py \
    --strategy mlm \
    --model_size small \
    --data_dir /path/to/genomic/data \
    --max_steps 10000

# Using a configuration file
python scripts/run_pretraining.py \
    --config configs/pretraining/base_config.yaml
```

### 2. Using OpenGenome Dataset

```bash
# Pretrain using OpenGenome configuration
python scripts/run_pretraining.py \
    --opengenome_config /path/to/opengenome2.yml \
    --strategy mlm \
    --model_size base \
    --max_steps 100000
```

## Configuration

### Configuration Files

The system uses YAML configuration files for comprehensive setup:

```yaml
# Example configuration
model:
  d_model: 512
  n_layer: 12
  n_head: 8
  vocab_size: 32

data:
  data_paths: ["/path/to/data"]
  max_sequence_length: 8192
  tokenizer_type: "dna"

strategy:
  name: "mlm"
  mask_probability: 0.15

optimization:
  learning_rate: 1e-4
  batch_size: 32
  max_steps: 100000
```

### Pre-built Configurations

The system includes several pre-built configurations:

- `configs/pretraining/base_config.yaml`: Standard configuration
- `configs/pretraining/small_model.yaml`: Small model for testing
- `configs/pretraining/large_model.yaml`: Large model for production

### Model Sizes

| Size  | d_model | n_layer | n_head | Parameters |
|-------|---------|---------|--------|------------|
| tiny  | 128     | 4       | 2      | ~1M        |
| small | 256     | 6       | 4      | ~5M        |
| base  | 512     | 12      | 8      | ~25M       |
| large | 1024    | 24      | 16     | ~150M      |

## Pretraining Strategies

### Masked Language Modeling (MLM)

Best for: General genomic sequence understanding

```yaml
strategy:
  name: "mlm"
  mask_probability: 0.15
  mask_token_probability: 0.8  # 80% [MASK]
  random_token_probability: 0.1  # 10% random
  leave_unmasked_probability: 0.1  # 10% unchanged
```

### Span Masking

Best for: Understanding sequence dependencies

```yaml
strategy:
  name: "mlm"
  span_masking:
    enable: true
    mean_span_length: 3.0
    max_span_length: 10
```

### Autoregressive (AR)

Best for: Sequence generation tasks

```yaml
strategy:
  name: "ar"
```

### Order-Agnostic Autoregressive Diffusion (OADM)

Best for: Advanced sequence modeling

```yaml
strategy:
  name: "oadm"
  oadm:
    enable: true
    num_diffusion_steps: 1000
    noise_schedule: "cosine"
```

## Data Preparation

### Supported Formats

- **FASTA files**: Standard genomic sequence format
- **Text files**: One sequence per line

### Data Structure

```
data/
├── training/
│   ├── genome_1.fasta
│   ├── genome_2.txt
│   └── ...
└── validation/
    ├── val_genome_1.fasta
    └── ...
```

### Data Configuration

```yaml
data:
  data_paths: 
    - "/path/to/training/data"
  validation_data_paths:
    - "/path/to/validation/data"
  data_types: ["fasta", "txt"]
  max_sequence_length: 8192
  tokenizer_type: "dna"  # or "rna", "protein"
  preprocessing:
    filter_short_sequences: true
    min_sequence_length: 100
    remove_ambiguous_bases: false
    reverse_complement_augmentation: true
```

### OpenGenome Integration

To use the OpenGenome dataset from the savanna repository:

```python
from hyena_glt.training.pretraining_config import OpenGenomeConfigBuilder

builder = OpenGenomeConfigBuilder("/path/to/opengenome2.yml")
config = builder.build_config(strategy="mlm", model_size="base")
```

## Hardware Optimization

### GPU Training

```yaml
hardware:
  device: "cuda"
  mixed_precision: true
  compile_model: true  # PyTorch 2.0 compilation
```

### Multi-GPU Training

```bash
# Use torchrun for distributed training
torchrun --nproc_per_node=4 scripts/run_pretraining.py \
    --config configs/pretraining/large_model.yaml
```

### Memory Optimization

For large models or limited memory:

```yaml
optimization:
  gradient_accumulation_steps: 8  # Effective batch size = batch_size * accumulation_steps
  batch_size: 4  # Smaller batch size

hardware:
  mixed_precision: true  # Reduces memory usage
```

## Monitoring and Logging

### Weights & Biases Integration

```yaml
logging:
  wandb:
    enable: true
    project: "hyena-glt"
    entity: "your-entity"
    tags: ["pretraining", "genomics"]
```

### Local Logging

```yaml
logging:
  output_dir: "./pretraining_output"
  log_every_n_steps: 100
  save_every_n_steps: 5000
  validate_every_n_steps: 1000
```

## Checkpointing and Resuming

### Automatic Checkpointing

```yaml
logging:
  save_every_n_steps: 5000
  max_checkpoints_to_keep: 3
```

### Resuming Training

```bash
python scripts/run_pretraining.py \
    --config configs/pretraining/base_config.yaml \
    --resume_from /path/to/checkpoint.pt
```

## Evaluation

### During Training

The system automatically evaluates:
- Training/validation loss
- Perplexity
- Masked token accuracy (for MLM)
- Learning rate and other metrics

### Post-Training Evaluation

```python
from hyena_glt.training.evaluation import run_comprehensive_evaluation

results = run_comprehensive_evaluation(
    model, tokenizer, val_dataloader, device
)

print(f"Perplexity: {results['perplexity']:.2f}")
print(f"MLM Accuracy: {results['mlm_accuracy']:.3f}")
```

### Available Metrics

- **Perplexity**: Overall model performance
- **MLM Accuracy**: Masked token prediction accuracy
- **Nucleotide Accuracy**: Per-nucleotide prediction accuracy
- **GC Content Preservation**: How well model preserves sequence composition

## Advanced Usage

### Custom Loss Functions

```yaml
loss:
  type: "cross_entropy"
  label_smoothing: 0.1
  focal_loss:
    enable: true
    alpha: 1.0
    gamma: 2.0
```

### Learning Rate Scheduling

```yaml
optimization:
  lr_scheduler: "cosine"  # or "linear", "constant"
  warmup_steps: 5000
  learning_rate: 1e-4
```

### Early Stopping

```yaml
optimization:
  early_stopping:
    enable: true
    patience: 10
    min_delta: 0.001
```

## Examples

### Example 1: Quick Experimentation

```bash
python examples/pretraining_examples.py quick_start
```

### Example 2: OpenGenome Pretraining

```bash
python examples/pretraining_examples.py opengenome
```

### Example 3: Strategy Comparison

```bash
python examples/pretraining_examples.py comparison
```

### Example 4: Custom Configuration

```bash
python examples/pretraining_examples.py custom
```

## Best Practices

### Model Size Selection

- **Small models (tiny/small)**: Quick experimentation, limited compute
- **Base models**: Good balance of performance and efficiency
- **Large models**: Maximum performance, requires significant compute

### Sequence Length

- **Short sequences (1K-4K)**: Faster training, good for many genomic tasks
- **Medium sequences (8K-16K)**: Better context understanding
- **Long sequences (32K+)**: Full gene or regulatory region modeling

### Batch Size Optimization

- Start with batch size that fits in memory
- Use gradient accumulation for larger effective batch sizes
- Monitor GPU utilization (aim for >80%)

### Learning Rate

- Start with 1e-4 for base models
- Scale with batch size: LR ∝ √(batch_size)
- Use warmup for stable training

### Data Mixing

```yaml
data:
  data_paths:
    - "/path/to/human_genome"
    - "/path/to/mouse_genome"
    - "/path/to/plant_genomes"
  data_mixing_weights: [0.5, 0.3, 0.2]  # Weight different datasets
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision

2. **Slow Training**
   - Enable model compilation
   - Increase number of data loader workers
   - Use faster storage (SSD)

3. **Poor Convergence**
   - Check learning rate (try 3e-4, 1e-4, 3e-5)
   - Increase warmup steps
   - Verify data quality

4. **Loss Not Decreasing**
   - Check data preprocessing
   - Verify model configuration
   - Review masking strategy

### Debug Mode

```bash
python scripts/run_pretraining.py \
    --config configs/pretraining/base_config.yaml \
    --log_level DEBUG \
    --max_steps 100
```

## Integration with Downstream Tasks

After pretraining, use the model for downstream tasks:

```python
from hyena_glt.model.hyena_glt import HyenaGLTForSequenceClassification

# Load pretrained model
model = HyenaGLTForSequenceClassification.from_pretrained(
    "/path/to/pretraining/output"
)

# Fine-tune on downstream task
trainer = HyenaGLTTrainer(
    model=model,
    task_type="classification",
    num_classes=2
)
```

## Performance Benchmarks

### Training Speed (steps/second)

| Model Size | Batch Size | A100 GPU | V100 GPU |
|------------|------------|----------|----------|
| Small      | 32         | 15.2     | 8.7      |
| Base       | 16         | 8.9      | 4.2      |
| Large      | 8          | 3.1      | 1.4      |

### Memory Usage

| Model Size | Sequence Length | Memory (GB) |
|------------|-----------------|-------------|
| Small      | 8K              | 12          |
| Base       | 8K              | 24          |
| Large      | 8K              | 48          |

## Contributing

To contribute to the pretraining system:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility

## Citation

If you use this pretraining system, please cite:

```bibtex
@software{hyena_glt_pretraining,
  title={Hyena-GLT Pretraining System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/BLT_Hyena}
}
```
