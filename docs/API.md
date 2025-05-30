# Hyena-GLT API Documentation

## Overview

Hyena-GLT (Genome Language Transformer) is a comprehensive framework that combines BLT's byte latent tokenization with Savanna's Striped Hyena blocks for genomic sequence modeling. This document provides detailed API documentation for all modules and classes.

## Table of Contents

1. [Configuration](#configuration)
2. [Data Processing](#data-processing)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Optimization](#optimization)
7. [Examples](#examples)

## Configuration

### HyenaGLTConfig

The main configuration class for Hyena-GLT models.

```python
from hyena_glt.config import HyenaGLTConfig

config = HyenaGLTConfig(
    vocab_size=8192,
    d_model=512,
    n_layers=8,
    sequence_length=4096,
    task_type="dna_sequence_modeling"
)
```

#### Parameters

- `vocab_size` (int): Size of the vocabulary. Default: 8192
- `d_model` (int): Hidden dimension size. Default: 512
- `n_layers` (int): Number of transformer layers. Default: 8
- `sequence_length` (int): Maximum sequence length. Default: 4096
- `task_type` (str): Type of genomic task. Options: "dna_sequence_modeling", "protein_function", "genome_annotation", "variant_effect"
- `use_dynamic_merging` (bool): Enable dynamic token merging. Default: True
- `hyena_order` (int): Order of Hyena operator. Default: 2
- `conv_kernel_size` (int): Kernel size for convolutions. Default: 3
- `learning_rate` (float): Learning rate. Default: 1e-4
- `batch_size` (int): Batch size. Default: 32
- `num_epochs` (int): Number of training epochs. Default: 10

## Data Processing

### Tokenizers

#### DNATokenizer

Tokenizes DNA sequences using byte-level encoding.

```python
from hyena_glt.data import DNATokenizer

tokenizer = DNATokenizer(vocab_size=8192)
tokens = tokenizer.encode("ATCGATCGATCG")
sequence = tokenizer.decode(tokens)
```

#### RNATokenizer

Tokenizes RNA sequences with specialized handling for RNA structure.

```python
from hyena_glt.data import RNATokenizer

tokenizer = RNATokenizer(vocab_size=8192)
tokens = tokenizer.encode("AUCGAUCGAUCG")
```

#### ProteinTokenizer

Tokenizes protein sequences using amino acid encodings.

```python
from hyena_glt.data import ProteinTokenizer

tokenizer = ProteinTokenizer(vocab_size=8192)
tokens = tokenizer.encode("MKTLLLTLLCLVAAYLA")
```

### Datasets

#### GenomicDataset

Base dataset class for genomic sequences.

```python
from hyena_glt.data import GenomicDataset

dataset = GenomicDataset(
    sequences=["ATCGATCG", "GCTAGCTA"],
    labels=[0, 1],
    tokenizer=tokenizer,
    max_length=1024
)
```

#### ProteinFunctionDataset

Dataset for protein function prediction tasks.

```python
from hyena_glt.data import ProteinFunctionDataset

dataset = ProteinFunctionDataset(
    data_path="protein_data.csv",
    tokenizer=protein_tokenizer,
    max_length=512
)
```

#### GenomeAnnotationDataset

Dataset for genome annotation tasks.

```python
from hyena_glt.data import GenomeAnnotationDataset

dataset = GenomeAnnotationDataset(
    data_path="annotation_data.csv",
    tokenizer=dna_tokenizer,
    max_length=2048
)
```

#### VariantEffectDataset

Dataset for variant effect prediction.

```python
from hyena_glt.data import VariantEffectDataset

dataset = VariantEffectDataset(
    data_path="variant_data.csv",
    tokenizer=dna_tokenizer,
    max_length=1024
)
```

## Model Architecture

### Core Operators

#### HyenaOperator

The core Hyena operator implementing efficient long-range convolutions.

```python
from hyena_glt.model import HyenaOperator

hyena_op = HyenaOperator(
    d_model=512,
    l_max=4096,
    order=2,
    filter_order=64
)
```

#### StrippedHyenaOperator

Striped Hyena operator with alternating attention and convolution.

```python
from hyena_glt.model import StrippedHyenaOperator

stripped_op = StrippedHyenaOperator(
    d_model=512,
    l_max=4096,
    order=2,
    num_heads=8
)
```

### Layers

#### DynamicMergingLayer

Layer that implements dynamic token merging from BLT.

```python
from hyena_glt.model import DynamicMergingLayer

merge_layer = DynamicMergingLayer(
    d_model=512,
    merge_ratio=0.5,
    threshold=0.1
)
```

#### HyenaDynamicLayer

Combined layer with Hyena operator and dynamic merging.

```python
from hyena_glt.model import HyenaDynamicLayer

hyena_layer = HyenaDynamicLayer(
    d_model=512,
    l_max=4096,
    merge_ratio=0.5
)
```

### Task-Specific Heads

#### SequenceClassificationHead

Head for sequence-level classification tasks.

```python
from hyena_glt.model import SequenceClassificationHead

clf_head = SequenceClassificationHead(
    d_model=512,
    num_classes=10,
    dropout=0.1
)
```

#### TokenClassificationHead

Head for token-level classification tasks.

```python
from hyena_glt.model import TokenClassificationHead

token_head = TokenClassificationHead(
    d_model=512,
    num_classes=5,
    dropout=0.1
)
```

#### RegressionHead

Head for regression tasks.

```python
from hyena_glt.model import RegressionHead

reg_head = RegressionHead(
    d_model=512,
    output_dim=1,
    dropout=0.1
)
```

### Complete Models

#### HyenaGLTModel

The main Hyena-GLT model combining all components.

```python
from hyena_glt.model import HyenaGLTModel

model = HyenaGLTModel(config)
outputs = model(input_ids, attention_mask=attention_mask)
```

#### HyenaGLTForSequenceClassification

Model specifically designed for sequence classification.

```python
from hyena_glt.model import HyenaGLTForSequenceClassification

model = HyenaGLTForSequenceClassification(config)
logits = model(input_ids, labels=labels)
```

#### HyenaGLTForTokenClassification

Model for token-level classification tasks.

```python
from hyena_glt.model import HyenaGLTForTokenClassification

model = HyenaGLTForTokenClassification(config)
logits = model(input_ids, labels=labels)
```

#### HyenaGLTForRegression

Model for regression tasks.

```python
from hyena_glt.model import HyenaGLTForRegression

model = HyenaGLTForRegression(config)
predictions = model(input_ids, labels=labels)
```

## Training

### HyenaGLTTrainer

Main trainer class for Hyena-GLT models.

```python
from hyena_glt.training import HyenaGLTTrainer

trainer = HyenaGLTTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

#### Methods

- `train()`: Start training process
- `evaluate()`: Evaluate model on validation set
- `save_checkpoint(path)`: Save model checkpoint
- `load_checkpoint(path)`: Load model checkpoint
- `get_metrics()`: Get training metrics

### Multi-Task Learning

#### MultiTaskLearner

Framework for multi-task learning with multiple genomic tasks.

```python
from hyena_glt.training import MultiTaskLearner

learner = MultiTaskLearner(
    tasks=["protein_function", "genome_annotation"],
    models=[protein_model, annotation_model],
    weights=[0.5, 0.5]
)
```

### Curriculum Learning

#### CurriculumLearner

Implements curriculum learning for progressive training.

```python
from hyena_glt.training import CurriculumLearner

curriculum = CurriculumLearner(
    stages=[
        {"max_length": 512, "epochs": 5},
        {"max_length": 1024, "epochs": 5},
        {"max_length": 2048, "epochs": 10}
    ]
)
```

### Fine-Tuning

#### FineTuner

Utilities for fine-tuning pre-trained models.

```python
from hyena_glt.training import FineTuner

finetuner = FineTuner(
    pretrained_model="hyena-glt-base",
    task_type="protein_function",
    num_classes=10
)

finetuner.finetune(train_dataset, val_dataset)
```

## Evaluation

### Metrics

#### GenomicMetrics

Comprehensive metrics for genomic tasks.

```python
from hyena_glt.evaluation import GenomicMetrics

metrics = GenomicMetrics()
results = metrics.compute_all(predictions, targets, task_type="protein_function")
```

#### Available Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score, AUC-ROC, AUC-PR
- **Regression**: MSE, MAE, R², Pearson correlation, Spearman correlation
- **Sequence**: BLEU, edit distance, sequence identity
- **Genomic-specific**: Conservation score, functional impact score

### Benchmarking

#### ModelBenchmark

Benchmark models across multiple tasks and datasets.

```python
from hyena_glt.evaluation import ModelBenchmark

benchmark = ModelBenchmark(
    models=[model1, model2],
    datasets=[dataset1, dataset2],
    metrics=["accuracy", "f1", "auc"]
)

results = benchmark.run()
```

### Analysis

#### PerformanceAnalyzer

Analyze model performance and generate reports.

```python
from hyena_glt.evaluation import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
report = analyzer.generate_report(results, save_path="performance_report.html")
```

#### AttentionAnalyzer

Analyze attention patterns in the model.

```python
from hyena_glt.evaluation import AttentionAnalyzer

attention_analyzer = AttentionAnalyzer(model)
attention_maps = attention_analyzer.get_attention_maps(sequences)
```

## Optimization

### Quantization

#### ModelQuantizer

Quantize models for efficient deployment.

```python
from hyena_glt.optimization import ModelQuantizer, QuantizationConfig

config = QuantizationConfig(
    quantization_type="dynamic",
    dtype=torch.qint8
)

quantizer = ModelQuantizer(config)
quantized_model = quantizer.quantize(model)
```

### Pruning

#### ModelPruner

Prune models to reduce size and improve efficiency.

```python
from hyena_glt.optimization import ModelPruner, PruningConfig

config = PruningConfig(
    pruning_type="magnitude",
    sparsity=0.5
)

pruner = ModelPruner(config)
pruned_model = pruner.prune(model, train_dataloader)
```

### Knowledge Distillation

#### KnowledgeDistiller

Distill knowledge from large teacher models to smaller student models.

```python
from hyena_glt.optimization import KnowledgeDistiller, DistillationConfig

config = DistillationConfig(
    temperature=4.0,
    alpha=0.7,
    distillation_type="response"
)

distiller = KnowledgeDistiller(config)
student_model = distiller.distill(teacher_model, student_model, train_dataloader)
```

### Deployment

#### ModelOptimizer

Optimize models for deployment.

```python
from hyena_glt.optimization import ModelOptimizer, DeploymentConfig

config = DeploymentConfig(
    export_format="onnx",
    optimization_level="all"
)

optimizer = ModelOptimizer(config)
optimized_model = optimizer.optimize(model)
```

## Examples

### Basic Usage

```python
import torch
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model import HyenaGLTForSequenceClassification
from hyena_glt.data import DNATokenizer
from hyena_glt.training import HyenaGLTTrainer

# Configuration
config = HyenaGLTConfig(
    vocab_size=8192,
    d_model=512,
    n_layers=8,
    num_classes=10,
    task_type="dna_sequence_modeling"
)

# Model and tokenizer
model = HyenaGLTForSequenceClassification(config)
tokenizer = DNATokenizer(vocab_size=config.vocab_size)

# Prepare data
sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
tokens = [tokenizer.encode(seq) for seq in sequences]
input_ids = torch.tensor(tokens)

# Forward pass
outputs = model(input_ids)
logits = outputs.logits
```

### Training Example

```python
from hyena_glt.data import GenomicDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = GenomicDataset(
    sequences=train_sequences,
    labels=train_labels,
    tokenizer=tokenizer,
    max_length=1024
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
trainer = HyenaGLTTrainer(
    model=model,
    config=config,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
```

### Fine-Tuning Example

```python
from hyena_glt.training import FineTuner

# Fine-tune pre-trained model
finetuner = FineTuner(
    pretrained_model="hyena-glt-base",
    task_type="protein_function",
    num_classes=5
)

# Fine-tune on custom dataset
finetuned_model = finetuner.finetune(
    train_dataset=protein_dataset,
    val_dataset=val_dataset,
    num_epochs=10
)
```

### Evaluation Example

```python
from hyena_glt.evaluation import GenomicMetrics, ModelBenchmark

# Evaluate single model
metrics = GenomicMetrics()
results = metrics.compute_all(predictions, targets, task_type="protein_function")

# Benchmark multiple models
benchmark = ModelBenchmark(
    models=[model1, model2, model3],
    datasets=[test_dataset],
    metrics=["accuracy", "f1", "auc"]
)

benchmark_results = benchmark.run()
```

### Optimization Example

```python
from hyena_glt.optimization import ModelQuantizer, ModelPruner

# Quantize model
quantizer = ModelQuantizer(QuantizationConfig(quantization_type="dynamic"))
quantized_model = quantizer.quantize(model)

# Prune model
pruner = ModelPruner(PruningConfig(pruning_type="magnitude", sparsity=0.5))
pruned_model = pruner.prune(model, train_dataloader)
```

## Error Handling

### Common Errors

1. **ConfigurationError**: Raised when configuration parameters are invalid
2. **TokenizationError**: Raised during tokenization failures
3. **ModelError**: Raised for model-specific issues
4. **TrainingError**: Raised during training failures
5. **OptimizationError**: Raised during optimization failures

### Best Practices

1. Always validate configuration before training
2. Use appropriate tokenizers for your data type
3. Monitor training metrics regularly
4. Save checkpoints frequently
5. Test optimized models before deployment
6. Use appropriate evaluation metrics for your task

## Version Compatibility

- Python: 3.8+
- PyTorch: 1.12+
- Transformers: 4.20+
- NumPy: 1.21+
- SciPy: 1.7+

## Support

For issues and questions:
- Check the documentation first
- Look at example scripts in `/examples/`
- Review test cases in `/tests/`
- File issues on GitHub

## License

This project is licensed under the MIT License. See LICENSE file for details.
