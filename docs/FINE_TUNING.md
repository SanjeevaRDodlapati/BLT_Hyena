# Fine-tuning Pipeline for Hyena-GLT Models

This document provides a comprehensive guide to fine-tuning pre-trained Hyena-GLT models for various genomic tasks.

## Overview

The Hyena-GLT fine-tuning pipeline provides:

- **Pre-trained Model Management**: Download, cache, and load pre-trained models
- **Task-Specific Optimizations**: Specialized configurations for genomic tasks
- **Advanced Training Strategies**: Layer freezing, discriminative learning, curriculum learning
- **Comprehensive Evaluation**: Genomic-specific metrics and benchmarking
- **Command-Line Interface**: Easy-to-use scripts for common tasks

## Available Pre-trained Models

| Model | Size | Description | Best For |
|-------|------|-------------|----------|
| `hyena-glt-small` | 150MB | Small model for quick experiments | Development, simple tasks |
| `hyena-glt-base` | 500MB | General-purpose model | Most genomic tasks |
| `hyena-glt-large` | 1.2GB | Large model for complex tasks | Long sequences, complex patterns |
| `hyena-glt-protein` | 450MB | Specialized for protein sequences | Protein function prediction |
| `hyena-glt-human-genome` | 900MB | Specialized for human genomic data | Human variant analysis |

## Quick Start

### 1. Basic Sequence Classification

```python
from hyena_glt.training.finetuning import finetune_for_sequence_classification
from hyena_glt.data.dataset import SequenceClassificationDataset
from hyena_glt.data.tokenizer import DNATokenizer

# Create dataset
tokenizer = DNATokenizer(k=6)
train_dataset = SequenceClassificationDataset(
    data_path="data/train.jsonl",
    tokenizer=tokenizer,
    max_length=2048,
    label_names=["promoter", "enhancer"]
)

# Fine-tune
trainer = finetune_for_sequence_classification(
    pretrained_model_path="hyena-glt-base",
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    output_dir="outputs/promoter_classifier",
    num_labels=2
)
```

### 2. Command-Line Fine-tuning

```bash
python scripts/finetune_hyena_glt.py \
  --task_type sequence_classification \
  --pretrained_model hyena-glt-base \
  --train_data data/train.jsonl \
  --eval_data data/eval.jsonl \
  --output_dir outputs/my_model \
  --num_labels 2 \
  --learning_rate 2e-5 \
  --batch_size 16 \
  --num_epochs 5
```

### 3. Task-Specific Fine-tuning

```python
# Genome annotation
from hyena_glt.training.task_specific import GenomeAnnotationFineTuner

finetuner = GenomeAnnotationFineTuner("hyena-glt-base", "outputs/annotator")
trainer = finetuner.fine_tune("data/annotation_train.jsonl", "data/annotation_eval.jsonl")

# Variant effect prediction
from hyena_glt.training.task_specific import VariantEffectFineTuner

finetuner = VariantEffectFineTuner("hyena-glt-human-genome", "outputs/variant_predictor")
trainer = finetuner.fine_tune("data/variants_train.jsonl", "data/variants_eval.jsonl")

# Protein function prediction
from hyena_glt.training.task_specific import ProteinFunctionFineTuner

finetuner = ProteinFunctionFineTuner("hyena-glt-protein", "outputs/protein_classifier", "go")
trainer = finetuner.fine_tune("data/proteins_train.jsonl", "data/proteins_eval.jsonl")
```

## Data Formats

### Sequence Classification (JSONL)
```json
{"sequence": "ATGCGTACGTAGCTAG", "label": 0}
{"sequence": "GCTAGCTACGTACGTA", "label": 1}
```

### Token Classification (JSONL)
```json
{"sequence": "ATGCGTACGTAGCTAG", "labels": [0, 0, 1, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0]}
```

### Sequence Generation (JSONL)
```json
{"sequence": "ATGCGTACGTAGCTAG"}
{"sequence": "GCTAGCTACGTACGTA"}
```

## Advanced Configuration

### Layer Freezing and Learning Rate Strategies

```python
config = FinetuningConfig(
    pretrained_model_path="hyena-glt-large",
    output_dir="outputs/advanced",
    
    # Freezing strategies
    freeze_backbone=False,
    freeze_layers=["embeddings"],  # Freeze embedding layers
    unfreeze_layers=["heads"],     # Ensure heads are trainable
    
    # Discriminative learning
    use_layer_wise_decay=True,
    layer_wise_lr_decay=0.95,      # Lower layers get lower LR
    
    # Training parameters
    learning_rate=1e-5,
    batch_size=4,
    gradient_accumulation_steps=8,
    warmup_ratio=0.2,
    
    # Regularization
    label_smoothing=0.1,
    mixup_alpha=0.2,
    weight_decay=0.01
)
```

### Curriculum Learning

```python
from hyena_glt.training.curriculum import CurriculumLearning, SequenceLengthDifficulty

# Define difficulty measure
difficulty_measure = SequenceLengthDifficulty()

# Create curriculum
curriculum = CurriculumLearning(
    difficulty_measure=difficulty_measure,
    initial_difficulty=0.3,
    max_difficulty=1.0,
    increase_rate=0.1
)

# Use in trainer
trainer = HyenaGLTTrainer(
    model=model,
    config=training_config,
    train_dataset=train_dataset,
    curriculum_learning=curriculum
)
```

## Task-Specific Examples

### 1. Genome Annotation
```bash
python examples/finetune_genome_annotation.py \
  --pretrained_model hyena-glt-base \
  --train_data data/genome_annotation_train.jsonl \
  --eval_data data/genome_annotation_eval.jsonl \
  --output_dir outputs/genome_annotator \
  --learning_rate 3e-5 \
  --batch_size 4 \
  --max_length 8192
```

### 2. Variant Effect Prediction
```bash
python examples/finetune_variant_effect.py \
  --pretrained_model hyena-glt-human-genome \
  --train_data data/clinvar_train.jsonl \
  --eval_data data/clinvar_eval.jsonl \
  --output_dir outputs/variant_classifier \
  --learning_rate 2e-5 \
  --label_smoothing 0.1
```

### 3. Protein Function Prediction
```bash
python examples/finetune_protein_function.py \
  --pretrained_model hyena-glt-protein \
  --train_data data/uniprot_train.jsonl \
  --eval_data data/uniprot_eval.jsonl \
  --output_dir outputs/protein_classifier \
  --function_type go \
  --learning_rate 1e-5 \
  --num_epochs 10
```

## Model Management

### Loading Pre-trained Models

```python
from hyena_glt.training.pretrained import load_pretrained_model, list_pretrained_models

# List available models
models = list_pretrained_models(sequence_type="DNA", task="classification")

# Load a model
model = load_pretrained_model(
    model_name="hyena-glt-base",
    task="sequence_classification",
    num_labels=5,
    device="cuda"
)
```

### Converting Models

```python
from hyena_glt.training.pretrained import ModelConverter

# Convert to ONNX
ModelConverter.convert_to_onnx(
    model=fine_tuned_model,
    output_path="model.onnx",
    input_shape=(1, 512)
)

# Convert to TorchScript
ModelConverter.convert_to_torchscript(
    model=fine_tuned_model,
    output_path="model.pt"
)

# Quantize model
quantized_model = ModelConverter.quantize_model(
    model=fine_tuned_model,
    quantization_type="dynamic"
)
```

## Best Practices

### 1. Data Preparation
- Use balanced datasets when possible
- Validate sequence formats (DNA: ATGC, Protein: 20 amino acids)
- Consider data augmentation (reverse complement for DNA)
- Split data properly (train/validation/test)

### 2. Hyperparameter Selection
- Start with task-specific defaults
- Use lower learning rates for larger models (1e-5 to 1e-6)
- Increase batch size with gradient accumulation for memory constraints
- Apply early stopping to prevent overfitting

### 3. Training Strategy
- Begin with frozen backbone, then gradually unfreeze
- Use layer-wise learning rate decay
- Monitor both training and validation metrics
- Save checkpoints regularly

### 4. Model Selection
- **hyena-glt-small**: Quick experiments, proof of concept
- **hyena-glt-base**: Most genomic tasks, good starting point
- **hyena-glt-large**: Complex tasks, large datasets
- **Specialized models**: Use domain-specific models when available

### 5. Evaluation
- Use genomic-specific metrics
- Evaluate on held-out test sets
- Consider biological interpretation of results
- Compare with baseline methods

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use gradient checkpointing
   - Reduce sequence length

2. **Slow Convergence**
   - Increase learning rate
   - Reduce layer freezing
   - Check data quality and balance
   - Use curriculum learning

3. **Overfitting**
   - Apply early stopping
   - Increase regularization (dropout, weight decay)
   - Use label smoothing
   - Reduce model size or training epochs

4. **Poor Performance**
   - Check data preprocessing
   - Verify label alignment
   - Try different pre-trained models
   - Adjust learning rate schedule

### Performance Optimization

```python
# Enable mixed precision training
config.fp16 = True

# Use gradient checkpointing
config.gradient_checkpointing = True

# Optimize data loading
config.dataloader_num_workers = 4
config.dataloader_pin_memory = True

# Use appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Integration with Hugging Face

The fine-tuned models are compatible with Hugging Face Transformers:

```python
from transformers import AutoModel, AutoTokenizer

# Save in Hugging Face format
model.save_pretrained("my_fine_tuned_model")

# Load later
model = AutoModel.from_pretrained("my_fine_tuned_model")
```

## Next Steps

1. **Experiment with Different Tasks**: Try the provided task-specific fine-tuners
2. **Custom Datasets**: Prepare your own genomic datasets
3. **Advanced Techniques**: Explore curriculum learning and multi-task training
4. **Evaluation**: Use the comprehensive evaluation framework
5. **Deployment**: Convert models for production use

For more detailed examples, see the `examples/` directory and the fine-tuning tutorial notebook.
