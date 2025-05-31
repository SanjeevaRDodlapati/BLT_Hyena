# Hyena-GLT Quick Start Guide

This guide will help you get started with Hyena-GLT for genomic sequence modeling in just a few minutes.

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.12 or higher
- CUDA (optional, for GPU acceleration)

### Install Hyena-GLT

```bash
# Clone the repository
git clone https://github.com/your-org/hyena-glt.git
cd hyena-glt

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install hyena-glt
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Examples

### 1. DNA Sequence Classification

```python
import torch
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model import HyenaGLTForSequenceClassification
from hyena_glt.data import DNATokenizer

# Configuration
config = HyenaGLTConfig(
    vocab_size=4096,
    d_model=256,
    n_layers=6,
    num_classes=2,  # Binary classification
    task_type="dna_sequence_modeling"
)

# Initialize model and tokenizer
model = HyenaGLTForSequenceClassification(config)
tokenizer = DNATokenizer(vocab_size=config.vocab_size)

# Example DNA sequences
sequences = [
    "ATCGATCGATCGATCGATCG",
    "GCTAGCTAGCTAGCTAGCTA"
]

# Tokenize sequences
tokens = tokenizer.batch_encode(sequences, max_length=512, padding=True)
input_ids = torch.tensor(tokens['input_ids'])
attention_mask = torch.tensor(tokens['attention_mask'])

# Forward pass
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.softmax(outputs.logits, dim=-1)
    
print(f"Predictions: {predictions}")
```

### 2. Protein Function Prediction

```python
from hyena_glt.data import ProteinTokenizer, ProteinFunctionDataset
from hyena_glt.training import HyenaGLTTrainer

# Configuration for protein task
config = HyenaGLTConfig(
    vocab_size=8192,
    d_model=512,
    n_layers=8,
    num_classes=10,  # 10 functional categories
    task_type="protein_function"
)

# Initialize components
model = HyenaGLTForSequenceClassification(config)
tokenizer = ProteinTokenizer(vocab_size=config.vocab_size)

# Example protein sequence
protein_sequence = "MKTLLLTLLCLVAAYLA"
tokens = tokenizer.encode(protein_sequence, max_length=512)

# Predict function
with torch.no_grad():
    input_ids = torch.tensor([tokens])
    outputs = model(input_ids)
    predicted_function = torch.argmax(outputs.logits, dim=-1)
    
print(f"Predicted function class: {predicted_function.item()}")
```

### 3. Training a Model

```python
from hyena_glt.data import GenomicDataset
from hyena_glt.training import HyenaGLTTrainer
from torch.utils.data import DataLoader

# Prepare your data
train_sequences = [
    "ATCGATCGATCG",
    "GCTAGCTAGCTA",
    # ... more sequences
]
train_labels = [0, 1]  # Classification labels

# Create dataset
train_dataset = GenomicDataset(
    sequences=train_sequences,
    labels=train_labels,
    tokenizer=tokenizer,
    max_length=1024
)

# Initialize trainer
trainer = HyenaGLTTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()
```

### 4. Fine-Tuning a Pre-trained Model

```python
from hyena_glt.training import FineTuner

# Fine-tune for specific task
finetuner = FineTuner(
    pretrained_model="hyena-glt-base",  # Pre-trained model name
    task_type="genome_annotation",
    num_classes=5
)

# Fine-tune on your dataset
finetuned_model = finetuner.finetune(
    train_dataset=your_train_dataset,
    val_dataset=your_val_dataset,
    num_epochs=10,
    learning_rate=1e-5
)
```

### 5. Model Evaluation

```python
from hyena_glt.evaluation import GenomicMetrics

# Evaluate model performance
metrics = GenomicMetrics()

# Get predictions
with torch.no_grad():
    outputs = model(test_input_ids)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Compute metrics
results = metrics.compute_all(
    predictions=predictions,
    targets=test_labels,
    task_type="protein_function"
)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-Score: {results['f1']:.4f}")
print(f"AUC-ROC: {results['auc_roc']:.4f}")
```

### 6. Model Optimization

```python
from hyena_glt.optimization import ModelQuantizer, QuantizationConfig

# Quantize model for deployment
quant_config = QuantizationConfig(
    quantization_type="dynamic",
    dtype=torch.qint8
)

quantizer = ModelQuantizer(quant_config)
quantized_model = quantizer.quantize(model)

# Compare model sizes
original_size = sum(p.numel() for p in model.parameters())
quantized_size = sum(p.numel() for p in quantized_model.parameters())

print(f"Original model parameters: {original_size:,}")
print(f"Quantized model parameters: {quantized_size:,}")
print(f"Compression ratio: {original_size / quantized_size:.2f}x")
```

## Configuration Guide

### Basic Configuration

```python
from hyena_glt.config import HyenaGLTConfig

# Minimal configuration
config = HyenaGLTConfig(
    vocab_size=4096,
    d_model=256,
    n_layers=6
)

# Advanced configuration
config = HyenaGLTConfig(
    # Model architecture
    vocab_size=8192,
    d_model=512,
    n_layers=12,
    n_heads=8,
    
    # Sequence processing
    sequence_length=4096,
    use_dynamic_merging=True,
    
    # Hyena-specific
    hyena_order=2,
    conv_kernel_size=3,
    filter_order=64,
    
    # Training
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=20,
    warmup_steps=1000,
    
    # Task-specific
    task_type="dna_sequence_modeling",
    num_classes=10
)
```

### Task-Specific Configurations

#### DNA Sequence Modeling
```python
dna_config = HyenaGLTConfig(
    vocab_size=4096,
    d_model=512,
    sequence_length=4096,
    task_type="dna_sequence_modeling"
)
```

#### Protein Function Prediction
```python
protein_config = HyenaGLTConfig(
    vocab_size=8192,
    d_model=768,
    sequence_length=1024,
    task_type="protein_function",
    num_classes=10
)
```

#### Genome Annotation
```python
annotation_config = HyenaGLTConfig(
    vocab_size=4096,
    d_model=512,
    sequence_length=8192,
    task_type="genome_annotation",
    num_classes=20
)
```

## Data Preparation

### DNA Sequences

```python
from hyena_glt.data import DNATokenizer, GenomicDataset

# Load your DNA sequences
sequences = ["ATCGATCG", "GCTAGCTA"]
labels = [0, 1]

# Create tokenizer and dataset
tokenizer = DNATokenizer(vocab_size=4096)
dataset = GenomicDataset(
    sequences=sequences,
    labels=labels,
    tokenizer=tokenizer,
    max_length=1024
)
```

### Protein Sequences

```python
from hyena_glt.data import ProteinTokenizer, ProteinFunctionDataset

# For protein sequences
protein_tokenizer = ProteinTokenizer(vocab_size=8192)
protein_dataset = ProteinFunctionDataset(
    data_path="protein_data.csv",
    tokenizer=protein_tokenizer,
    max_length=512
)
```

### Data Format

Your data should be in one of these formats:

#### CSV Format
```csv
sequence,label
ATCGATCGATCG,0
GCTAGCTAGCTA,1
```

#### JSON Format
```json
[
    {"sequence": "ATCGATCGATCG", "label": 0},
    {"sequence": "GCTAGCTAGCTA", "label": 1}
]
```

#### FASTA Format (for sequences only)
```
>sequence1
ATCGATCGATCG
>sequence2
GCTAGCTAGCTA
```

## Training Tips

### 1. Start Small
Begin with smaller models and datasets to verify everything works:
```python
config = HyenaGLTConfig(
    d_model=256,
    n_layers=4,
    batch_size=16
)
```

### 2. Monitor Training
Use the built-in metrics:
```python
trainer = HyenaGLTTrainer(...)
metrics = trainer.train()
print(f"Final validation accuracy: {metrics['val_accuracy']:.4f}")
```

### 3. Use Curriculum Learning
For long sequences, start with shorter lengths:
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

### 4. Leverage Pre-trained Models
Start with pre-trained weights when available:
```python
model = HyenaGLTForSequenceClassification.from_pretrained(
    "hyena-glt-base",
    num_classes=your_num_classes
)
```

## Common Issues and Solutions

### 1. Out of Memory
- Reduce batch size: `config.batch_size = 8`
- Use gradient checkpointing: `config.gradient_checkpointing = True`
- Reduce sequence length: `config.sequence_length = 1024`

### 2. Slow Training
- Use task-optimized mixed precision: `from hyena_glt.training.task_specific import get_optimal_precision_config`
- Increase batch size if you have memory
- Use multiple GPUs: `config.num_gpus = 2`

### 3. Poor Performance
- Increase model size: `config.d_model = 768`
- Add more layers: `config.n_layers = 12`
- Tune learning rate: `config.learning_rate = 5e-4`
- Use learning rate scheduling

### 4. Convergence Issues
- Lower learning rate: `config.learning_rate = 1e-5`
- Add warmup: `config.warmup_steps = 1000`
- Use gradient clipping: `config.max_grad_norm = 1.0`

## Next Steps

1. **Read the full documentation**: Check out `docs/API.md` for complete API reference
2. **Explore examples**: Look at scripts in the `examples/` directory
3. **Try optimization**: Use quantization and pruning for deployment
4. **Join the community**: Contribute to the project or ask questions

## Getting Help

- **Documentation**: Full API docs in `docs/`
- **Examples**: Working examples in `examples/`
- **Issues**: Report bugs on GitHub
- **Discussions**: Join community discussions

Happy modeling with Hyena-GLT! 🧬🚀
