"""
Visualization utilities for genomic data analysis and model evaluation.
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | None = None,
    figsize: tuple[int, int] = (15, 5),
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot comprehensive training history with loss, accuracy, and learning rate.

    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot
        figsize: Figure size
        show_plot: Whether to display the plot

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot training and validation loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.8)
    if 'eval_loss' in history or 'val_loss' in history:
        val_loss = history.get('eval_loss', history.get('val_loss'))
        axes[0].plot(val_loss, label='Val Loss', color='red', alpha=0.8)

    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot training and validation accuracy
    if 'train_accuracy' in history:
        axes[1].plot(history['train_accuracy'], label='Train Acc', color='blue', alpha=0.8)
    if 'eval_accuracy' in history or 'val_accuracy' in history:
        val_acc = history.get('eval_accuracy', history.get('val_accuracy'))
        axes[1].plot(val_acc, label='Val Acc', color='red', alpha=0.8)

    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot learning rate
    if 'learning_rate' in history:
        axes[2].plot(history['learning_rate'], color='green', alpha=0.8)
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Learning Rate\nNot Available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title('Learning Rate Schedule')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: str | None = None,
    figsize: tuple[int, int] = (8, 6),
    show_plot: bool = True,
    normalize: bool = False
) -> plt.Figure:
    """
    Plot confusion matrix with customizable styling.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the plot
        figsize: Figure size
        show_plot: Whether to display the plot
        normalize: Whether to normalize the confusion matrix

    Returns:
        Figure object
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    sequence: str,
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    max_length: int = 100,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention weights matrix
        sequence: Input sequence
        save_path: Path to save the plot
        figsize: Figure size
        max_length: Maximum sequence length to display
        show_plot: Whether to display the plot

    Returns:
        Figure object
    """
    # Truncate if sequence is too long
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
        attention_weights = attention_weights[:max_length, :max_length]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')

    # Set labels
    ax.set_xticks(range(len(sequence)))
    ax.set_yticks(range(len(sequence)))
    ax.set_xticklabels(list(sequence))
    ax.set_yticklabels(list(sequence))

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # Set title and labels
    ax.set_title(f'Attention Heatmap (Sequence Length: {len(sequence)})')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')

    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=90)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig


def plot_sequence_analysis(
    sequences: list[str],
    labels: list[int] | None = None,
    class_names: list[str] | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (15, 10),
    show_plot: bool = True
) -> plt.Figure:
    """
    Create comprehensive sequence analysis plots.

    Args:
        sequences: List of sequences
        labels: Optional labels for sequences
        class_names: Optional class names
        save_path: Path to save the plot
        figsize: Figure size
        show_plot: Whether to display the plot

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # 1. Sequence length distribution
    lengths = [len(seq) for seq in sequences]
    axes[0].hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Sequence Length Distribution')
    axes[0].grid(True, alpha=0.3)

    # 2. GC content distribution (for DNA/RNA sequences)
    gc_contents = []
    for seq in sequences:
        if seq:
            gc = (seq.upper().count('G') + seq.upper().count('C')) / len(seq)
            gc_contents.append(gc)

    axes[1].hist(gc_contents, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('GC Content')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('GC Content Distribution')
    axes[1].grid(True, alpha=0.3)

    # 3. Nucleotide composition
    all_seq = ''.join(sequences).upper()
    from collections import Counter
    char_counts = Counter(all_seq)
    common_chars = char_counts.most_common(10)

    chars, counts = zip(*common_chars, strict=False) if common_chars else ([], [])
    axes[2].bar(chars, counts, color='lightcoral', alpha=0.7)
    axes[2].set_xlabel('Character')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Character Frequency')
    axes[2].tick_params(axis='x', rotation=45)

    # 4. Class distribution (if labels provided)
    if labels is not None:
        label_counts = Counter(labels)
        classes = list(label_counts.keys())
        counts = list(label_counts.values())

        if class_names:
            class_labels = [class_names[i] if i < len(class_names) else f'Class {i}' for i in classes]
        else:
            class_labels = [f'Class {i}' for i in classes]

        axes[3].bar(class_labels, counts, color='lightsalmon', alpha=0.7)
        axes[3].set_xlabel('Class')
        axes[3].set_ylabel('Count')
        axes[3].set_title('Class Distribution')
        axes[3].tick_params(axis='x', rotation=45)
    else:
        axes[3].text(0.5, 0.5, 'No Labels\nProvided',
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[3].transAxes, fontsize=12)
        axes[3].set_title('Class Distribution')

    # 5. Sequence complexity (entropy)
    entropies = []
    for seq in sequences:
        char_counts = Counter(seq.upper())
        total = len(seq)
        entropy = -sum((count/total) * np.log2(count/total) for count in char_counts.values())
        entropies.append(entropy)

    axes[4].hist(entropies, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    axes[4].set_xlabel('Sequence Entropy')
    axes[4].set_ylabel('Frequency')
    axes[4].set_title('Sequence Complexity Distribution')
    axes[4].grid(True, alpha=0.3)

    # 6. Summary statistics
    stats_text = f"""
    Total Sequences: {len(sequences)}
    Avg Length: {np.mean(lengths):.1f}
    Avg GC Content: {np.mean(gc_contents):.3f}
    Avg Entropy: {np.mean(entropies):.3f}
    Unique Chars: {len(char_counts)}
    """

    axes[5].text(0.1, 0.5, stats_text, transform=axes[5].transAxes,
                fontsize=10, verticalalignment='center',
                bbox={'boxstyle': "round,pad=0.3", 'facecolor': "lightgray", 'alpha': 0.7})
    axes[5].set_title('Summary Statistics')
    axes[5].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig


def create_genomic_dashboard(
    sequences: list[str],
    predictions: list[int],
    probabilities: np.ndarray,
    true_labels: list[int] | None = None,
    class_names: list[str] | None = None,
    save_path: str | None = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create a comprehensive genomic analysis dashboard.

    Args:
        sequences: Input sequences
        predictions: Model predictions
        probabilities: Prediction probabilities
        true_labels: True labels (optional)
        class_names: Class names (optional)
        save_path: Path to save the dashboard
        show_plot: Whether to display the plot

    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(20, 12))

    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Prediction confidence distribution
    ax1 = fig.add_subplot(gs[0, 0])
    max_probs = np.max(probabilities, axis=1)
    ax1.hist(max_probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Confidence Distribution')
    ax1.axvline(np.mean(max_probs), color='red', linestyle='--',
                label=f'Mean: {np.mean(max_probs):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Class prediction distribution
    ax2 = fig.add_subplot(gs[0, 1])
    pred_counts = Counter(predictions)
    classes = list(pred_counts.keys())
    counts = list(pred_counts.values())

    if class_names:
        labels = [class_names[i] if i < len(class_names) else f'Class {i}' for i in classes]
    else:
        labels = [f'Class {i}' for i in classes]

    ax2.bar(labels, counts, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Predicted Class')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Confusion matrix (if true labels available)
    if true_labels is not None:
        from sklearn.metrics import confusion_matrix
        ax3 = fig.add_subplot(gs[0, 2:])
        cm = confusion_matrix(true_labels, predictions)

        im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax3.figure.colorbar(im, ax=ax3)

        ax3.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels,
               yticklabels=labels,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')

        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    else:
        ax3 = fig.add_subplot(gs[0, 2:])
        ax3.text(0.5, 0.5, 'Confusion Matrix\n(True labels not available)',
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Confusion Matrix')

    # 4. Sequence analysis plots
    # GC content vs prediction confidence
    ax4 = fig.add_subplot(gs[1, 0])
    gc_contents = [(seq.upper().count('G') + seq.upper().count('C')) / len(seq) for seq in sequences]
    ax4.scatter(gc_contents, max_probs, c=predictions, alpha=0.6, cmap='tab10')
    ax4.set_xlabel('GC Content')
    ax4.set_ylabel('Prediction Confidence')
    ax4.set_title('GC Content vs Confidence')
    ax4.grid(True, alpha=0.3)

    # 5. Sequence length vs prediction confidence
    ax5 = fig.add_subplot(gs[1, 1])
    lengths = [len(seq) for seq in sequences]
    ax5.scatter(lengths, max_probs, c=predictions, alpha=0.6, cmap='tab10')
    ax5.set_xlabel('Sequence Length')
    ax5.set_ylabel('Prediction Confidence')
    ax5.set_title('Length vs Confidence')
    ax5.grid(True, alpha=0.3)

    # 6. Per-class confidence
    ax6 = fig.add_subplot(gs[1, 2:])
    if class_names and len(class_names) <= 10:  # Only if reasonable number of classes
        class_confidences = {}
        for i, pred in enumerate(predictions):
            if pred not in class_confidences:
                class_confidences[pred] = []
            class_confidences[pred].append(max_probs[i])

        box_data = []
        box_labels = []
        for class_idx in sorted(class_confidences.keys()):
            box_data.append(class_confidences[class_idx])
            box_labels.append(class_names[class_idx] if class_idx < len(class_names) else f'Class {class_idx}')

        ax6.boxplot(box_data, labels=box_labels)
        ax6.set_xlabel('Predicted Class')
        ax6.set_ylabel('Prediction Confidence')
        ax6.set_title('Confidence by Class')
        ax6.tick_params(axis='x', rotation=45)
    else:
        ax6.text(0.5, 0.5, 'Per-class Confidence\n(Too many classes to display)',
                horizontalalignment='center', verticalalignment='center',
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Confidence by Class')

    # 7. Sequence composition analysis
    ax7 = fig.add_subplot(gs[2, :2])
    all_seq = ''.join(sequences).upper()
    char_counts = Counter(all_seq)
    common_chars = char_counts.most_common(10)

    if common_chars:
        chars, counts = zip(*common_chars, strict=False)
        ax7.bar(chars, counts, color='lightcoral', alpha=0.7)
        ax7.set_xlabel('Character')
        ax7.set_ylabel('Count')
        ax7.set_title('Character Frequency Across All Sequences')

    # 8. Model performance summary
    ax8 = fig.add_subplot(gs[2, 2:])

    # Calculate summary statistics
    avg_confidence = np.mean(max_probs)
    std_confidence = np.std(max_probs)

    if true_labels is not None:
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(true_labels, predictions)
        summary_text = f"""
        PERFORMANCE SUMMARY

        Total Samples: {len(sequences)}
        Average Confidence: {avg_confidence:.3f} ± {std_confidence:.3f}

        Accuracy: {accuracy:.3f}

        Class Distribution:
        """
        for _i, (class_idx, count) in enumerate(pred_counts.items()):
            class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f'Class {class_idx}'
            summary_text += f"\n  {class_name}: {count}"
    else:
        summary_text = f"""
        PREDICTION SUMMARY

        Total Samples: {len(sequences)}
        Average Confidence: {avg_confidence:.3f} ± {std_confidence:.3f}

        Class Distribution:
        """
        for _i, (class_idx, count) in enumerate(pred_counts.items()):
            class_name = class_names[class_idx] if class_names and class_idx < len(class_names) else f'Class {class_idx}'
            summary_text += f"\n  {class_name}: {count}"

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox={'boxstyle': "round,pad=0.5", 'facecolor': "lightgray", 'alpha': 0.8})
    ax8.set_title('Summary Statistics')
    ax8.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    return fig


# Import Counter at module level
from collections import Counter
