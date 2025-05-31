"""
Fine-tuning Tutorial for Hyena-GLT Models

This notebook demonstrates how to fine-tune pre-trained Hyena-GLT models
for various genomic tasks with step-by-step examples.
"""

"""
Fine-tuning Tutorial for Hyena-GLT Models

This notebook demonstrates how to fine-tune pre-trained Hyena-GLT models
for various genomic tasks with step-by-step examples.
"""

# %%
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Import Hyena-GLT components
from hyena_glt.training.finetuning import FinetuningConfig, TaskSpecificFineTuner
from hyena_glt.training.metrics import GenomicMetrics
from hyena_glt.training.pretrained import list_pretrained_models

# Add project root to path
project_root = Path.cwd().parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("✓ Hyena-GLT imports successful")

# %%
# 1. EXPLORING AVAILABLE PRE-TRAINED MODELS
print("=" * 60)
print("1. AVAILABLE PRE-TRAINED MODELS")
print("=" * 60)

# List all available models
available_models = list_pretrained_models()

print(f"Found {len(available_models)} pre-trained models:")
for model in available_models:
    print(f"\n• {model.name}")
    print(f"  Description: {model.description}")
    print(f"  Size: {model.size_mb:.1f} MB")
    print(f"  Tasks: {', '.join(model.tasks)}")
    print(f"  Species: {', '.join(model.species)}")
    print(f"  Sequence types: {', '.join(model.sequence_types)}")

# %%
# Filter models by criteria
print("\nDNA sequence models:")
dna_models = list_pretrained_models(sequence_type="DNA")
for model in dna_models:
    print(f"• {model.name}: {model.description}")

print("\nProtein sequence models:")
protein_models = list_pretrained_models(sequence_type="protein")
for model in protein_models:
    print(f"• {model.name}: {model.description}")

# %%
# 2. CREATING SAMPLE DATA FOR DEMONSTRATION
print("\n" + "=" * 60)
print("2. CREATING SAMPLE DATA")
print("=" * 60)

# Create sample DNA sequences for classification
def create_sample_classification_data(num_samples: int = 100) -> dict[str, list]:
    """Create sample DNA sequence classification data."""
    np.random.seed(42)

    sequences = []
    labels = []

    # Generate promoter sequences (label 0)
    for _ in range(num_samples // 2):
        # Promoter-like sequence with TATA box
        seq = "TATAAA" + "".join(np.random.choice(["A", "T", "G", "C"], size=194))
        sequences.append(seq)
        labels.append(0)

    # Generate enhancer sequences (label 1)
    for _ in range(num_samples // 2):
        # Enhancer-like sequence with some motifs
        seq = "GGGCGG" + "".join(np.random.choice(["A", "T", "G", "C"], size=194))
        sequences.append(seq)
        labels.append(1)

    return {
        "sequences": sequences,
        "labels": labels,
        "label_names": ["promoter", "enhancer"]
    }

# Create sample token classification data
def create_sample_token_data(num_samples: int = 50) -> dict[str, list]:
    """Create sample DNA sequence token classification data."""
    np.random.seed(42)

    sequences = []
    token_labels = []

    for _ in range(num_samples):
        # Create sequence with gene structure
        seq_len = 500
        seq = "".join(np.random.choice(["A", "T", "G", "C"], size=seq_len))

        # Create labels: 0=intergenic, 1=exon, 2=intron
        labels = [0] * seq_len

        # Add some exon regions
        for _ in range(2):
            start = np.random.randint(0, seq_len - 50)
            end = start + np.random.randint(30, 50)
            for i in range(start, min(end, seq_len)):
                labels[i] = 1

        # Add some intron regions
        for _ in range(1):
            start = np.random.randint(0, seq_len - 100)
            end = start + np.random.randint(50, 100)
            for i in range(start, min(end, seq_len)):
                if labels[i] == 0:  # Don't override exons
                    labels[i] = 2

        sequences.append(seq)
        token_labels.append(labels)

    return {
        "sequences": sequences,
        "token_labels": token_labels,
        "label_names": ["intergenic", "exon", "intron"]
    }

# Create sample data
print("Creating sample sequence classification data...")
classification_data = create_sample_classification_data(100)
print(f"Created {len(classification_data['sequences'])} sequences")
print(f"Labels: {classification_data['label_names']}")
print(f"Sample sequence length: {len(classification_data['sequences'][0])}")

print("\nCreating sample token classification data...")
token_data = create_sample_token_data(50)
print(f"Created {len(token_data['sequences'])} sequences")
print(f"Labels: {token_data['label_names']}")
print(f"Sample sequence length: {len(token_data['sequences'][0])}")

# %%
# Save sample data to files
def save_classification_data(data: dict, filepath: str):
    """Save classification data to JSONL format."""
    with open(filepath, 'w') as f:
        for seq, label in zip(data['sequences'], data['labels'], strict=False):
            json.dump({"sequence": seq, "label": label}, f)
            f.write('\n')

def save_token_data(data: dict, filepath: str):
    """Save token classification data to JSONL format."""
    with open(filepath, 'w') as f:
        for seq, labels in zip(data['sequences'], data['token_labels'], strict=False):
            json.dump({"sequence": seq, "labels": labels}, f)
            f.write('\n')

# Create data directory
data_dir = Path("sample_data")
data_dir.mkdir(exist_ok=True)

# Save data
classification_train_path = data_dir / "classification_train.jsonl"
classification_eval_path = data_dir / "classification_eval.jsonl"
token_train_path = data_dir / "token_train.jsonl"
token_eval_path = data_dir / "token_eval.jsonl"

# Split data for train/eval
train_split = 0.8
train_size_clf = int(len(classification_data['sequences']) * train_split)
train_size_tok = int(len(token_data['sequences']) * train_split)

# Save classification data
train_clf_data = {
    'sequences': classification_data['sequences'][:train_size_clf],
    'labels': classification_data['labels'][:train_size_clf]
}
eval_clf_data = {
    'sequences': classification_data['sequences'][train_size_clf:],
    'labels': classification_data['labels'][train_size_clf:]
}

save_classification_data(train_clf_data, classification_train_path)
save_classification_data(eval_clf_data, classification_eval_path)

# Save token classification data
train_tok_data = {
    'sequences': token_data['sequences'][:train_size_tok],
    'token_labels': token_data['token_labels'][:train_size_tok]
}
eval_tok_data = {
    'sequences': token_data['sequences'][train_size_tok:],
    'token_labels': token_data['token_labels'][train_size_tok:]
}

save_token_data(train_tok_data, token_train_path)
save_token_data(eval_tok_data, token_eval_path)

print(f"✓ Sample data saved to {data_dir}")

# %%
# 3. SEQUENCE CLASSIFICATION FINE-TUNING
print("\n" + "=" * 60)
print("3. SEQUENCE CLASSIFICATION FINE-TUNING")
print("=" * 60)

# Create a simple config for demonstration (using base model)
config = FinetuningConfig(
    pretrained_model_path="hyena-glt-base",  # This would be downloaded
    output_dir="outputs/sequence_classification",
    task_type="sequence_classification",
    num_labels=2,
    label_names=["promoter", "enhancer"],
    learning_rate=2e-5,
    batch_size=4,  # Small batch for demo
    num_epochs=2,  # Few epochs for demo
    max_length=512,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    logging_steps=10,
    early_stopping_patience=2
)

print("Fine-tuning configuration:")
print(f"• Task: {config.task_type}")
print(f"• Model: {config.pretrained_model_path}")
print(f"• Output: {config.output_dir}")
print(f"• Labels: {config.label_names}")
print(f"• Epochs: {config.num_epochs}")
print(f"• Batch size: {config.batch_size}")
print(f"• Learning rate: {config.learning_rate}")

# NOTE: In a real scenario, you would run the fine-tuning like this:
# finetuner = FineTuner(config)
# trainer = finetuner.fine_tune(train_dataset, eval_dataset, compute_metrics)

print("\n📝 To run fine-tuning:")
print("1. Ensure you have a pre-trained model available")
print("2. Create FineTuner with the config")
print("3. Call fine_tune() with your datasets")

# %%
# 4. TOKEN CLASSIFICATION FINE-TUNING
print("\n" + "=" * 60)
print("4. TOKEN CLASSIFICATION FINE-TUNING")
print("=" * 60)

# Create config for token classification
token_config = FinetuningConfig(
    pretrained_model_path="hyena-glt-base",
    output_dir="outputs/token_classification",
    task_type="token_classification",
    num_labels=3,
    label_names=["intergenic", "exon", "intron"],
    learning_rate=3e-5,
    batch_size=2,  # Smaller batch for longer sequences
    num_epochs=3,
    max_length=1024,
    gradient_accumulation_steps=4,
    warmup_ratio=0.15,
    use_layer_wise_decay=True,
    layer_wise_lr_decay=0.9,
    eval_strategy="epoch",
    logging_steps=5
)

print("Token classification configuration:")
print(f"• Task: {token_config.task_type}")
print(f"• Labels: {token_config.label_names}")
print(f"• Sequence length: {token_config.max_length}")
print(f"• Layer-wise decay: {token_config.use_layer_wise_decay}")
print(f"• Decay factor: {token_config.layer_wise_lr_decay}")

# %%
# 5. USING TASK-SPECIFIC FINE-TUNERS
print("\n" + "=" * 60)
print("5. TASK-SPECIFIC FINE-TUNERS")
print("=" * 60)

# These are convenience classes for common genomic tasks
print("Available task-specific fine-tuners:")

print("\n• TaskSpecificFineTuner.create_sequence_classification_config()")
print("  - Optimized for sequence-level classification")
print("  - Good for: promoter detection, variant effect prediction")

print("\n• TaskSpecificFineTuner.create_token_classification_config()")
print("  - Optimized for token-level classification")
print("  - Good for: gene annotation, regulatory element detection")

print("\n• TaskSpecificFineTuner.create_generation_config()")
print("  - Optimized for sequence generation")
print("  - Good for: sequence completion, synthetic biology")

print("\n• TaskSpecificFineTuner.create_domain_adaptation_config()")
print("  - Optimized for adapting across species/domains")
print("  - Good for: transfer learning between species")

# Example of using task-specific config
seq_clf_config = TaskSpecificFineTuner.create_sequence_classification_config(
    pretrained_model_path="hyena-glt-base",
    output_dir="outputs/task_specific",
    num_labels=2,
    learning_rate=2e-5,
    batch_size=8
)

print("\nTask-specific config created:")
print(f"• Learning rate: {seq_clf_config.learning_rate}")
print(f"• Batch size: {seq_clf_config.batch_size}")
print(f"• Task type: {seq_clf_config.task_type}")

# %%
# 6. ADVANCED FINE-TUNING STRATEGIES
print("\n" + "=" * 60)
print("6. ADVANCED FINE-TUNING STRATEGIES")
print("=" * 60)

print("🔧 LAYER FREEZING STRATEGIES:")
print("• freeze_backbone=True: Freeze all backbone weights")
print("• freeze_layers=['encoder']: Freeze specific layer patterns")
print("• unfreeze_layers=['heads']: Unfreeze specific patterns")

print("\n🎯 DISCRIMINATIVE LEARNING:")
print("• use_layer_wise_decay=True: Different LR for different layers")
print("• layer_wise_lr_decay=0.9: Decay factor for lower layers")
print("• Typically: heads > hyena_blocks > encoder > embeddings")

print("\n📚 CURRICULUM LEARNING:")
print("• Start with easier examples, gradually increase difficulty")
print("• Based on sequence length or complexity")
print("• Integrated with CurriculumLearning class")

print("\n⚖️ REGULARIZATION:")
print("• label_smoothing: Smooth target labels")
print("• mixup_alpha: Mix input sequences")
print("• dropout_rate: Override model dropout")

# Example advanced config
advanced_config = FinetuningConfig(
    pretrained_model_path="hyena-glt-large",
    output_dir="outputs/advanced",
    task_type="sequence_classification",
    num_labels=5,

    # Layer-wise learning
    use_layer_wise_decay=True,
    layer_wise_lr_decay=0.95,
    freeze_layers=["embeddings"],  # Freeze embeddings

    # Regularization
    label_smoothing=0.1,
    mixup_alpha=0.2,

    # Training strategy
    learning_rate=1e-5,
    batch_size=4,
    gradient_accumulation_steps=8,
    warmup_ratio=0.2,

    # Early stopping
    early_stopping_patience=3,
    early_stopping_threshold=0.001
)

print("\nAdvanced configuration example:")
print(f"• Layer-wise decay: {advanced_config.layer_wise_lr_decay}")
print(f"• Label smoothing: {advanced_config.label_smoothing}")
print(f"• Mixup alpha: {advanced_config.mixup_alpha}")
print(f"• Frozen layers: {advanced_config.freeze_layers}")

# %%
# 7. EVALUATION AND METRICS
print("\n" + "=" * 60)
print("7. EVALUATION AND METRICS")
print("=" * 60)

print("📊 GENOMIC-SPECIFIC METRICS:")
print("• Classification: accuracy, precision, recall, F1, AUC")
print("• Token classification: token-level and sequence-level metrics")
print("• Generation: perplexity, BLEU (for sequence similarity)")
print("• Genomic-specific: GC content analysis, motif recovery")

# Example metrics computation
def example_compute_metrics(eval_pred):
    """Example metrics function for fine-tuning."""
    metrics = GenomicMetrics()
    predictions, labels = eval_pred

    # For classification
    predictions = predictions.argmax(axis=-1)

    # Compute standard metrics
    results = metrics.compute_classification_metrics(predictions, labels)

    # Add custom genomic metrics
    results['custom_metric'] = 0.95  # Placeholder

    return results

print("\nExample metrics function created ✓")

# %%
# 8. COMMAND-LINE USAGE EXAMPLES
print("\n" + "=" * 60)
print("8. COMMAND-LINE USAGE")
print("=" * 60)

print("🚀 BASIC FINE-TUNING COMMANDS:")

print("\n1. Sequence Classification:")
print("python scripts/finetune_hyena_glt.py \\")
print("  --task_type sequence_classification \\")
print("  --pretrained_model hyena-glt-base \\")
print("  --train_data data/promoter_train.jsonl \\")
print("  --eval_data data/promoter_eval.jsonl \\")
print("  --output_dir outputs/promoter_classifier \\")
print("  --num_labels 2 \\")
print("  --learning_rate 2e-5 \\")
print("  --batch_size 16 \\")
print("  --num_epochs 5")

print("\n2. Token Classification:")
print("python scripts/finetune_hyena_glt.py \\")
print("  --task_type token_classification \\")
print("  --pretrained_model hyena-glt-base \\")
print("  --train_data data/annotation_train.jsonl \\")
print("  --output_dir outputs/gene_annotator \\")
print("  --num_labels 10 \\")
print("  --use_layer_wise_decay \\")
print("  --batch_size 4 \\")
print("  --max_length 4096")

print("\n3. Protein Function Prediction:")
print("python examples/finetune_protein_function.py \\")
print("  --pretrained_model hyena-glt-protein \\")
print("  --train_data data/protein_functions.jsonl \\")
print("  --output_dir outputs/protein_classifier \\")
print("  --function_type go \\")
print("  --learning_rate 1e-5 \\")
print("  --num_epochs 10")

# %%
# 9. BEST PRACTICES AND TIPS
print("\n" + "=" * 60)
print("9. BEST PRACTICES")
print("=" * 60)

print("💡 FINE-TUNING TIPS:")

print("\n📋 DATA PREPARATION:")
print("• Use JSONL format for easy streaming")
print("• Ensure balanced datasets when possible")
print("• Validate sequence formats (DNA: ATGC, Protein: 20 AAs)")
print("• Consider data augmentation (reverse complement for DNA)")

print("\n⚙️ HYPERPARAMETER TUNING:")
print("• Start with task-specific defaults")
print("• Lower learning rates for larger models")
print("• Use gradient accumulation for large sequences")
print("• Monitor validation metrics to avoid overfitting")

print("\n🎛️ MODEL SELECTION:")
print("• hyena-glt-small: Quick experiments, simple tasks")
print("• hyena-glt-base: General-purpose, good balance")
print("• hyena-glt-large: Complex tasks, large datasets")
print("• Specialized models: Domain-specific (protein, human)")

print("\n🔄 TRAINING STRATEGY:")
print("• Use early stopping to prevent overfitting")
print("• Start with frozen backbone, then unfreeze gradually")
print("• Apply layer-wise learning rate decay")
print("• Monitor both training and validation losses")

print("\n💾 CHECKPOINT MANAGEMENT:")
print("• Save checkpoints regularly")
print("• Keep best model based on validation metrics")
print("• Save fine-tuning configuration for reproducibility")

# %%
print("\n" + "=" * 80)
print("🎉 FINE-TUNING TUTORIAL COMPLETED!")
print("=" * 80)

print("You've learned:")
print("✓ How to explore available pre-trained models")
print("✓ How to create and prepare genomic datasets")
print("✓ How to configure fine-tuning for different tasks")
print("✓ How to use task-specific optimizations")
print("✓ How to apply advanced training strategies")
print("✓ How to evaluate and monitor training")
print("✓ Command-line usage for automated fine-tuning")
print("✓ Best practices for successful fine-tuning")

print("\n🚀 Next Steps:")
print("1. Prepare your own genomic datasets")
print("2. Choose appropriate pre-trained models")
print("3. Start with task-specific configurations")
print("4. Experiment with advanced strategies")
print("5. Evaluate on held-out test sets")
print("6. Deploy your fine-tuned models")

print("\n📚 For more information:")
print("• Check the examples/ directory for complete scripts")
print("• See the documentation for detailed API reference")
print("• Join the community for tips and best practices")

# %%
# Clean up sample data (optional)
# import shutil
# shutil.rmtree("sample_data", ignore_errors=True)
# print("✓ Sample data cleaned up")
