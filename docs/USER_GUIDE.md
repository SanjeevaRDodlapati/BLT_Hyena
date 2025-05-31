# Hyena-GLT User Guide

Welcome to the comprehensive Hyena-GLT user guide! This document provides practical guidance for using the Hyena-GLT framework for genomic sequence modeling, from installation to advanced applications.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Data Preparation](#data-preparation)
4. [Model Configuration](#model-configuration)
5. [Training Your Models](#training-your-models)
6. [Evaluation and Analysis](#evaluation-and-analysis)
7. [Advanced Techniques](#advanced-techniques)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Getting Started

### Installation

#### Quick Installation

```bash
# Install from PyPI (when available)
pip install hyena-glt

# Or install from source
git clone https://github.com/your-username/hyena-glt.git
cd hyena-glt
pip install -e .
```

#### Development Installation

```bash
# Clone repository
git clone https://github.com/your-username/hyena-glt.git
cd hyena-glt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,docs,notebooks]"

# Verify installation
python -c "import hyena_glt; print(hyena_glt.__version__)"
```

#### System Requirements

**Minimum Requirements:**
- Python 3.8+
- PyTorch 2.0+
- 8GB RAM
- 2GB GPU memory (optional but recommended)

**Recommended Setup:**
- Python 3.10+
- PyTorch 2.1+
- 32GB RAM
- 24GB GPU memory
- CUDA 11.8+ for GPU acceleration

### Your First Model

Let's start with a simple DNA sequence classification example:

```python
import torch
from hyena_glt.models import HyenaGLT, HyenaGLTConfig
from hyena_glt.tokenizers import DNATokenizer
from hyena_glt.data import GenomicDataset

# 1. Configure the model
config = HyenaGLTConfig(
    vocab_size=4,          # DNA bases: A, T, G, C
    d_model=256,           # Model dimension
    n_layers=4,            # Number of Hyena layers
    sequence_length=1024,  # Input sequence length
    num_classes=2          # Binary classification
)

# 2. Create model and tokenizer
model = HyenaGLT(config)
tokenizer = DNATokenizer()

# 3. Prepare data
sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
labels = [0, 1]  # Example labels

# Tokenize sequences
tokens = [tokenizer.encode(seq) for seq in sequences]
input_ids = torch.tensor(tokens)
labels_tensor = torch.tensor(labels)

# 4. Forward pass
with torch.no_grad():
    outputs = model(input_ids)
    predictions = torch.softmax(outputs, dim=-1)

print(f"Predictions: {predictions}")
```

This example demonstrates the core workflow: configure → create → prepare → predict.

## Understanding the Architecture

### The Hyena-GLT Framework

Hyena-GLT combines two powerful innovations:

1. **BLT (Byte Latent Transformer)**: Efficient tokenization that compresses sequences
2. **Striped Hyena**: Linear-complexity alternative to attention mechanisms

#### Key Benefits

- **Efficiency**: Linear O(n) complexity vs quadratic O(n²) attention
- **Long Sequences**: Handle genomic sequences up to 1M+ tokens
- **Compression**: BLT reduces sequence length while preserving information
- **Flexibility**: Adaptable to DNA, RNA, and protein sequences

#### Architecture Overview

```
Input Sequence (Raw DNA/RNA/Protein)
         ↓
    Tokenization (BLT)
         ↓
    Embedding Layer
         ↓
    Hyena Blocks (×N layers)
         ↓
    Task-Specific Head
         ↓
    Output (Classifications/Predictions)
```

### When to Use Hyena-GLT

**Ideal Use Cases:**
- Long genomic sequences (>1000 tokens)
- Real-time genomic analysis
- Large-scale genomic datasets
- Resource-constrained environments
- Multi-modal genomic tasks

**Consider Alternatives When:**
- Very short sequences (<100 tokens)
- Extremely high accuracy requirements with unlimited compute
- Tasks requiring complex reasoning over sequence structure

## Data Preparation

### Supported Sequence Types

#### DNA Sequences
```python
from hyena_glt.tokenizers import DNATokenizer

tokenizer = DNATokenizer()
dna_sequence = "ATCGATCGATCG"
tokens = tokenizer.encode(dna_sequence)
```

#### RNA Sequences
```python
from hyena_glt.tokenizers import RNATokenizer

tokenizer = RNATokenizer()
rna_sequence = "AUCGAUCGAUCG"
tokens = tokenizer.encode(rna_sequence)
```

#### Protein Sequences
```python
from hyena_glt.tokenizers import ProteinTokenizer

tokenizer = ProteinTokenizer()
protein_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
tokens = tokenizer.encode(protein_sequence)
```

### Data Format Guidelines

#### Input Data Structure

Your data should be organized as:

```python
# For classification tasks
data = {
    'sequences': ['ATCGATCG', 'GCTAGCTA', ...],
    'labels': [0, 1, ...],
    'metadata': {
        'sequence_ids': ['seq1', 'seq2', ...],
        'organism': ['human', 'mouse', ...],
        # Additional metadata
    }
}
```

#### Creating Datasets

```python
from hyena_glt.data import GenomicDataset

# Create dataset
dataset = GenomicDataset(
    sequences=data['sequences'],
    labels=data['labels'],
    tokenizer=tokenizer,
    max_length=1024,
    metadata=data['metadata']
)

# Create data loader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### Data Quality Guidelines

#### Sequence Quality
- **Remove ambiguous bases**: N, X, or other non-standard characters
- **Quality filtering**: Remove low-quality sequences
- **Length filtering**: Filter sequences outside target length range
- **Duplicate removal**: Remove identical or near-identical sequences

#### Label Quality
- **Balanced datasets**: Ensure reasonable class balance
- **Label validation**: Verify label accuracy and consistency
- **Missing data**: Handle missing labels appropriately

#### Example Quality Control

```python
def quality_control(sequences, labels, min_length=50, max_length=2000):
    """Apply quality control to genomic sequences."""
    filtered_sequences = []
    filtered_labels = []
    
    for seq, label in zip(sequences, labels):
        # Length filter
        if min_length <= len(seq) <= max_length:
            # Remove ambiguous characters
            if 'N' not in seq and 'X' not in seq:
                # Ensure valid DNA bases only
                if set(seq).issubset({'A', 'T', 'G', 'C'}):
                    filtered_sequences.append(seq)
                    filtered_labels.append(label)
    
    return filtered_sequences, filtered_labels
```

## Model Configuration

### Basic Configuration

The `HyenaGLTConfig` class controls all model parameters:

```python
from hyena_glt.models import HyenaGLTConfig

# Basic configuration
config = HyenaGLTConfig(
    # Tokenization
    vocab_size=4,           # 4 for DNA, 20 for proteins
    sequence_length=1024,   # Maximum input length
    
    # Architecture
    d_model=256,            # Hidden dimension
    n_layers=6,             # Number of Hyena layers
    
    # Task-specific
    num_classes=2,          # For classification
    task_type="classification",
    
    # Training
    dropout=0.1,            # Regularization
    layer_norm_eps=1e-5
)
```

### Task-Specific Configurations

#### DNA Sequence Classification

```python
# Promoter prediction
promoter_config = HyenaGLTConfig(
    vocab_size=4,
    d_model=512,
    n_layers=8,
    sequence_length=2000,   # Promoter regions
    num_classes=2,          # Promoter/non-promoter
    dropout=0.15
)
```

#### Protein Function Prediction

```python
# Multi-label protein function
protein_config = HyenaGLTConfig(
    vocab_size=20,          # 20 amino acids
    d_model=768,
    n_layers=12,
    sequence_length=512,    # Typical protein length
    num_classes=1000,       # GO term classes
    task_type="multi_label_classification",
    dropout=0.2
)
```

#### Variant Effect Prediction

```python
# Pathogenicity prediction
variant_config = HyenaGLTConfig(
    vocab_size=4,
    d_model=384,
    n_layers=10,
    sequence_length=1000,   # Sequence context around variant
    num_classes=3,          # Benign/VUS/Pathogenic
    dropout=0.1
)
```

### Performance vs. Accuracy Trade-offs

#### Fast Model (Lower Resource Requirements)
```python
fast_config = HyenaGLTConfig(
    d_model=128,
    n_layers=4,
    sequence_length=512
)
# ~1M parameters, good for quick experiments
```

#### Balanced Model (Good Performance/Cost Ratio)
```python
balanced_config = HyenaGLTConfig(
    d_model=256,
    n_layers=6,
    sequence_length=1024
)
# ~5M parameters, recommended for most tasks
```

#### High-Performance Model (Best Accuracy)
```python
large_config = HyenaGLTConfig(
    d_model=512,
    n_layers=12,
    sequence_length=2048
)
# ~50M parameters, for demanding applications
```

## Training Your Models

### Basic Training Setup

```python
from hyena_glt.training import HyenaGLTTrainer, TrainingConfig

# Training configuration
training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=16,
    num_epochs=10,
    warmup_steps=1000,
    weight_decay=0.01,
    eval_steps=500,
    save_steps=1000
)

# Create trainer
trainer = HyenaGLTTrainer(
    model=model,
    config=training_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()
```

### Advanced Training Techniques

#### Learning Rate Scheduling

```python
# Cosine annealing with warmup
training_config = TrainingConfig(
    learning_rate=5e-4,
    scheduler_type="cosine_with_warmup",
    warmup_ratio=0.1,
    min_lr=1e-6
)
```

#### Gradient Accumulation

```python
# For effective large batch sizes
training_config = TrainingConfig(
    batch_size=4,               # Physical batch size
    gradient_accumulation_steps=8,  # Effective batch size = 32
    max_grad_norm=1.0          # Gradient clipping
)
```

#### Mixed Precision Training

```python
# Enhanced mixed precision with task-specific optimization
from hyena_glt.training.task_specific import get_optimal_precision_config

# Get optimal precision for your task
precision_config = get_optimal_precision_config('genome_annotation')

training_config = TrainingConfig(
    precision_config=precision_config,  # Task-optimized mixed precision
    dataloader_num_workers=4           # Parallel data loading
)

# Available tasks: 'genome_annotation', 'variant_effect', 'protein_function',
#                  'genome_generation', 'domain_adaptation'
```

### Monitoring Training

#### Using Weights & Biases

```python
training_config = TrainingConfig(
    logging_strategy="wandb",
    project_name="genomic-modeling",
    experiment_name="dna-classifier-v1"
)
```

#### Using TensorBoard

```python
training_config = TrainingConfig(
    logging_strategy="tensorboard",
    logging_dir="./logs"
)
```

#### Custom Monitoring

```python
class CustomCallback:
    def on_epoch_end(self, trainer, epoch, logs):
        # Custom monitoring logic
        if logs['val_accuracy'] > 0.95:
            print(f"High accuracy achieved at epoch {epoch}!")

trainer.add_callback(CustomCallback())
```

## Evaluation and Analysis

### Basic Evaluation

```python
from hyena_glt.evaluation import ModelEvaluator

evaluator = ModelEvaluator(model, tokenizer)

# Evaluate on test set
results = evaluator.evaluate(test_dataset)
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")
```

### Detailed Analysis

#### Confusion Matrix

```python
import matplotlib.pyplot as plt
from hyena_glt.evaluation import plot_confusion_matrix

# Generate predictions
predictions, labels, probabilities = evaluator.predict(test_dataset)

# Plot confusion matrix
plot_confusion_matrix(labels, predictions, 
                     class_names=['Class 0', 'Class 1'])
plt.show()
```

#### ROC Curves

```python
from hyena_glt.evaluation import plot_roc_curve

plot_roc_curve(labels, probabilities, num_classes=2)
plt.show()
```

#### Per-Class Metrics

```python
from sklearn.metrics import classification_report

report = classification_report(labels, predictions, 
                             target_names=['Negative', 'Positive'])
print(report)
```

### Model Interpretability

#### Attention Visualization

```python
from hyena_glt.analysis import AttentionAnalyzer

analyzer = AttentionAnalyzer(model)
attention_maps = analyzer.get_attention_patterns(sequences)

# Visualize attention
analyzer.plot_attention_heatmap(attention_maps[0], sequence=sequences[0])
```

#### Feature Importance

```python
from hyena_glt.analysis import FeatureImportanceAnalyzer

importance_analyzer = FeatureImportanceAnalyzer(model)
importance_scores = importance_analyzer.analyze_sequence(sequence)

# Plot importance
importance_analyzer.plot_importance(importance_scores, sequence)
```

## Advanced Techniques

### Transfer Learning

#### Using Pre-trained Models

```python
# Load pre-trained model
from hyena_glt.models import HyenaGLT

pretrained_model = HyenaGLT.from_pretrained("hyena-glt-dna-large")

# Fine-tune for your task
config = HyenaGLTConfig(num_classes=5)  # Your specific task
model = HyenaGLT(config)
model.load_pretrained_weights(pretrained_model)
```

#### Domain Adaptation

```python
# Adapt from DNA to RNA
dna_model = HyenaGLT.from_pretrained("hyena-glt-dna-base")
rna_config = HyenaGLTConfig(vocab_size=4, num_classes=3)
rna_model = dna_model.adapt_to_config(rna_config)
```

### Multi-Task Learning

```python
from hyena_glt.training import MultiTaskTrainer

# Define multiple tasks
tasks = {
    'promoter_prediction': {
        'dataset': promoter_dataset,
        'num_classes': 2,
        'loss_weight': 1.0
    },
    'splice_site_detection': {
        'dataset': splice_dataset,
        'num_classes': 3,
        'loss_weight': 0.5
    }
}

# Multi-task training
mt_trainer = MultiTaskTrainer(model, tasks)
mt_trainer.train()
```

### Curriculum Learning

```python
from hyena_glt.training import CurriculumTrainer

# Define curriculum stages
curriculum = [
    {'sequence_length': 256, 'epochs': 5},   # Start with short sequences
    {'sequence_length': 512, 'epochs': 5},   # Gradually increase
    {'sequence_length': 1024, 'epochs': 10}  # Full length
]

curriculum_trainer = CurriculumTrainer(model, curriculum)
curriculum_trainer.train()
```

## Production Deployment

### Model Optimization

#### Quantization

```python
from hyena_glt.optimization import quantize_model

# Dynamic quantization (easiest)
quantized_model = quantize_model(model, quantization_type="dynamic")

# Static quantization (better performance)
quantized_model = quantize_model(model, quantization_type="static", 
                                calibration_dataset=cal_dataset)
```

#### Pruning

```python
from hyena_glt.optimization import prune_model

# Magnitude-based pruning
pruned_model = prune_model(model, sparsity=0.5, method="magnitude")

# Structured pruning
pruned_model = prune_model(model, sparsity=0.3, method="structured")
```

#### Knowledge Distillation

```python
from hyena_glt.optimization import distill_model

# Distill large model to smaller one
teacher_model = large_model
student_config = HyenaGLTConfig(d_model=128, n_layers=4)
student_model = distill_model(teacher_model, student_config, 
                             distillation_dataset=train_dataset)
```

### Export and Deployment

#### ONNX Export

```python
# Export to ONNX for cross-platform deployment
import torch.onnx

dummy_input = torch.randint(0, 4, (1, 1024))
torch.onnx.export(model, dummy_input, "hyena_glt_model.onnx",
                  export_params=True, opset_version=11)
```

#### TorchScript

```python
# Export to TorchScript for mobile deployment
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("hyena_glt_mobile.pt")
```

#### Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-runtime

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY hyena_glt/ ./hyena_glt/
COPY model_checkpoint.pt .

EXPOSE 8000
CMD ["python", "serve.py"]
```

### Serving Infrastructure

#### REST API with FastAPI

```python
from fastapi import FastAPI
from hyena_glt.models import HyenaGLT
import torch

app = FastAPI()
model = HyenaGLT.load_from_checkpoint("model_checkpoint.pt")

@app.post("/predict")
async def predict(sequence: str):
    tokens = tokenizer.encode(sequence)
    input_ids = torch.tensor([tokens])
    
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = torch.softmax(outputs, dim=-1)
    
    return {"predictions": predictions.tolist()}
```

#### Batch Processing

```python
from hyena_glt.serving import BatchProcessor

processor = BatchProcessor(model, batch_size=32, max_workers=4)
results = processor.process_sequences(large_sequence_list)
```

## Troubleshooting

### Common Issues and Solutions

#### Memory Issues

**Problem**: CUDA out of memory
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size
2. Use gradient accumulation
3. Enable mixed precision training
4. Reduce sequence length
5. Use gradient checkpointing

```python
# Gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Mixed precision
training_config.fp16 = True

# Smaller batch size with accumulation
training_config.batch_size = 8
training_config.gradient_accumulation_steps = 4
```

#### Slow Training

**Problem**: Training is too slow

**Solutions**:
1. Use multiple GPUs
2. Optimize data loading
3. Use compiled models
4. Profile bottlenecks

```python
# Multi-GPU training
import torch.nn as nn
model = nn.DataParallel(model)

# Faster data loading
dataloader = DataLoader(dataset, num_workers=8, pin_memory=True)

# Model compilation (PyTorch 2.0+)
model = torch.compile(model)
```

#### Poor Performance

**Problem**: Model not learning or poor accuracy

**Solutions**:
1. Check data quality
2. Adjust learning rate
3. Increase model capacity
4. Use better initialization
5. Add regularization

```python
# Learning rate finder
from hyena_glt.training import find_learning_rate
optimal_lr = find_learning_rate(model, train_dataloader)

# Better initialization
model.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) 
           if hasattr(m, 'weight') else None)
```

#### Convergence Issues

**Problem**: Training loss not decreasing

**Solutions**:
1. Reduce learning rate
2. Add gradient clipping
3. Check for label noise
4. Use learning rate scheduling

```python
# Gradient clipping
training_config.max_grad_norm = 1.0

# Learning rate scheduling
training_config.scheduler_type = "reduce_on_plateau"
training_config.patience = 5
```

### Debugging Tools

#### Memory Profiling

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    with_stack=True
) as prof:
    trainer.train()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

#### Gradient Monitoring

```python
def monitor_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

# Use during training
grad_norm = monitor_gradients(model)
print(f"Gradient norm: {grad_norm}")
```

## Best Practices

### Data Preparation Best Practices

1. **Quality First**: Always prioritize data quality over quantity
2. **Balanced Datasets**: Ensure reasonable class balance
3. **Validation Strategy**: Use proper train/validation/test splits
4. **Data Augmentation**: Use sequence-aware augmentation techniques
5. **Reproducibility**: Set random seeds and document preprocessing steps

### Model Configuration Best Practices

1. **Start Small**: Begin with smaller models and scale up
2. **Progressive Training**: Use curriculum learning for complex tasks
3. **Regular Validation**: Monitor validation metrics frequently
4. **Hyperparameter Search**: Use systematic hyperparameter optimization
5. **Documentation**: Document all configuration choices

### Training Best Practices

1. **Monitoring**: Use comprehensive logging and monitoring
2. **Checkpointing**: Save model checkpoints regularly
3. **Early Stopping**: Implement early stopping to prevent overfitting
4. **Resource Management**: Monitor GPU memory and compute usage
5. **Reproducibility**: Ensure experiments are reproducible

### Production Best Practices

1. **Model Versioning**: Use proper model versioning and tracking
2. **A/B Testing**: Test new models against baselines
3. **Monitoring**: Monitor model performance in production
4. **Fallback Strategies**: Implement fallback mechanisms
5. **Security**: Ensure model security and data privacy

### Code Organization Best Practices

```
project/
├── configs/           # Configuration files
├── data/             # Data processing scripts
├── models/           # Model definitions
├── training/         # Training scripts
├── evaluation/       # Evaluation and analysis
├── deployment/       # Deployment configurations
├── notebooks/        # Jupyter notebooks
├── tests/           # Unit and integration tests
└── docs/            # Documentation
```

## Getting Help

### Documentation Resources

- **API Reference**: Complete API documentation in `docs/API.md`
- **Architecture Guide**: Detailed architecture explanation in `docs/ARCHITECTURE.md`
- **Examples**: Working examples in `examples/` directory
- **Tutorials**: Interactive tutorials in `examples/notebooks/`

### Community and Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Join community discussions
- **Documentation**: Contribute to documentation
- **Examples**: Share your examples and use cases

### Contributing

We welcome contributions! Please see:
- **Contributing Guide**: `CONTRIBUTING.md`
- **Code of Conduct**: `CODE_OF_CONDUCT.md`
- **Development Setup**: Instructions in this guide

---

**Happy genomic modeling with Hyena-GLT!** 🧬🚀

This user guide provides comprehensive coverage of the Hyena-GLT framework. For specific technical details, refer to the API documentation and architecture guide. For hands-on learning, check out the tutorial notebooks in the `examples/notebooks/` directory.
