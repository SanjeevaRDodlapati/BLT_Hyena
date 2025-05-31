#!/usr/bin/env python3
"""
Evaluation Example for Hyena-GLT Framework

This example demonstrates comprehensive evaluation of Hyena-GLT models
including performance metrics, analysis, and visualization.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Hyena-GLT imports
from hyena_glt import HyenaGLT
from hyena_glt.utils import (
    analyze_tokenization,
    compute_sequence_statistics,
)


def load_model_checkpoint(checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = checkpoint['config']
    model = HyenaGLT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = checkpoint['tokenizer']

    return model, config, tokenizer, checkpoint

def create_test_data(num_samples=200, seq_length=500, num_classes=3):
    """Create test data for evaluation."""
    np.random.seed(123)  # Different seed from training
    sequences = []
    labels = []

    bases = ['A', 'T', 'G', 'C']

    for i in range(num_samples):
        sequence = []
        label = i % num_classes

        for _j in range(seq_length):
            if label == 0:  # AT-rich
                base = np.random.choice(['A', 'T'], p=[0.6, 0.4])
            elif label == 1:  # GC-rich
                base = np.random.choice(['G', 'C'], p=[0.5, 0.5])
            else:  # Balanced
                base = np.random.choice(bases)
            sequence.append(base)

        sequences.append(''.join(sequence))
        labels.append(label)

    return sequences, labels

def evaluate_perplexity(model, tokenizer, sequences, device):
    """Evaluate model perplexity on sequences."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for sequence in tqdm(sequences, desc="Computing perplexity"):
            tokens = tokenizer.encode(sequence)
            if len(tokens) < 2:
                continue

            input_ids = torch.tensor([tokens[:-1]]).to(device)
            targets = torch.tensor([tokens[1:]]).to(device)

            outputs = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                targets.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += targets.numel()

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def evaluate_classification_task(model, test_loader, device):
    """Evaluate classification performance."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating classification"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            sequence_repr = outputs.hidden_states[-1].mean(dim=1)
            logits = model.classifier(sequence_repr)

            all_logits.append(logits.cpu())
            all_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels, torch.cat(all_logits, dim=0)

def analyze_model_performance(predictions, labels, class_names=None):
    """Analyze classification performance."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(max(labels) + 1)]

    # Classification report
    report = classification_report(
        labels, predictions,
        target_names=class_names,
        output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    return report, cm

def plot_evaluation_results(report, cm, class_names, output_dir):
    """Create evaluation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    plt.close()

    # Plot per-class metrics
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    metrics = ['precision', 'recall', 'f1-score']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        bars = axes[i].bar(range(len(classes)), values, color='skyblue')
        axes[i].set_title(f'{metric.capitalize()}')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_xticks(range(len(classes)))
        axes[i].set_xticklabels(class_names, rotation=45)
        axes[i].set_ylim(0, 1)

        # Add value labels on bars
        for _j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.png', dpi=300)
    plt.close()

def main():
    print("🧬 Hyena-GLT Evaluation Example")
    print("=" * 50)

    # 1. Load model
    checkpoint_path = "./fine_tuned_model/model.pt"
    if not Path(checkpoint_path).exists():
        print("❌ No fine-tuned model found. Please run fine_tuning.py first.")
        return

    print("1. Loading fine-tuned model...")
    model, config, tokenizer, checkpoint = load_model_checkpoint(checkpoint_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"   ✓ Model loaded on {device}")
    print(f"   ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Create test data
    print("\n2. Creating test dataset...")
    test_sequences, test_labels = create_test_data(num_samples=200, num_classes=3)

    class_names = ['AT-rich', 'GC-rich', 'Balanced']
    print(f"   ✓ Test samples: {len(test_sequences)}")
    print(f"   ✓ Classes: {class_names}")

    # 3. Prepare test dataset
    print("\n3. Preparing test dataset...")
    from torch.utils.data import DataLoader

    from examples.fine_tuning import GenomicClassificationDataset

    test_dataset = GenomicClassificationDataset(
        test_sequences, test_labels, tokenizer, max_length=512
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"   ✓ Test loader: {len(test_loader)} batches")

    # 4. Classification evaluation
    print("\n4. Evaluating classification performance...")
    predictions, labels, logits = evaluate_classification_task(model, test_loader, device)

    # Compute metrics
    report, cm = analyze_model_performance(predictions, labels, class_names)

    accuracy = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']

    print(f"   ✓ Accuracy: {accuracy:.4f}")
    print(f"   ✓ Macro F1: {macro_f1:.4f}")
    print(f"   ✓ Weighted F1: {weighted_f1:.4f}")

    # 5. Perplexity evaluation
    print("\n5. Computing perplexity...")
    perplexity = evaluate_perplexity(model, tokenizer, test_sequences[:50], device)
    print(f"   ✓ Perplexity: {perplexity:.4f}")

    # 6. Sequence analysis
    print("\n6. Analyzing sequences...")

    # Tokenization analysis
    token_stats = analyze_tokenization(tokenizer, test_sequences[:20])
    print(f"   ✓ Avg tokens per sequence: {token_stats['avg_tokens']:.1f}")
    print(f"   ✓ Compression ratio: {token_stats['compression_ratio']:.2f}")

    # Sequence statistics
    seq_stats = compute_sequence_statistics(test_sequences[:20])
    print(f"   ✓ Avg sequence length: {seq_stats['avg_length']:.1f}")
    print(f"   ✓ Avg GC content: {seq_stats['avg_gc_content']:.3f}")

    # 7. Model analysis
    print("\n7. Analyzing model internals...")

    # Sample prediction analysis
    sample_idx = 0
    sample_sequence = test_sequences[sample_idx]
    sample_tokens = tokenizer.encode(sample_sequence)

    with torch.no_grad():
        input_ids = torch.tensor([sample_tokens]).to(device)
        outputs = model(input_ids)

        # Get embeddings and attention if available
        embeddings = outputs.hidden_states[-1]
        print(f"   ✓ Sample embedding shape: {embeddings.shape}")
        print(f"   ✓ Sample embedding norm: {torch.norm(embeddings).item():.4f}")

    # 8. Generate evaluation report
    print("\n8. Generating evaluation report...")

    output_dir = Path("./evaluation_results")
    output_dir.mkdir(exist_ok=True)

    # Plot results
    plot_evaluation_results(report, cm, class_names, output_dir)

    # Save detailed report
    eval_report = {
        'model_config': {
            'num_layers': config.num_layers,
            'hidden_size': config.hidden_size,
            'vocab_size': config.vocab_size,
            'total_parameters': sum(p.numel() for p in model.parameters())
        },
        'evaluation_metrics': {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'perplexity': perplexity
        },
        'per_class_metrics': {
            class_names[i]: {
                'precision': report[f'{i}']['precision'],
                'recall': report[f'{i}']['recall'],
                'f1_score': report[f'{i}']['f1-score'],
                'support': report[f'{i}']['support']
            } for i in range(len(class_names))
        },
        'sequence_analysis': {
            'tokenization': token_stats,
            'sequence_stats': seq_stats
        }
    }

    with open(output_dir / 'evaluation_report.json', 'w') as f:
        json.dump(eval_report, f, indent=2)

    print(f"   ✓ Results saved to: {output_dir}")
    print("   ✓ Plots: confusion_matrix.png, per_class_metrics.png")
    print("   ✓ Report: evaluation_report.json")

    # 9. Training history (if available)
    if 'train_losses' in checkpoint and 'val_accuracies' in checkpoint:
        print("\n9. Plotting training history...")

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(checkpoint['train_losses'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(checkpoint['val_accuracies'])
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / 'training_history.png', dpi=300)
        plt.close()

        print("   ✓ Training history saved")

    print("\n" + "=" * 50)
    print("✅ Evaluation completed successfully!")
    print(f"Final test accuracy: {accuracy:.4f}")
    print(f"Model perplexity: {perplexity:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
