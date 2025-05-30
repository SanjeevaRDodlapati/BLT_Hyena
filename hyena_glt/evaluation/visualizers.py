"""
Visualization utilities for evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EvaluationVisualizer:
    """Comprehensive visualization utilities for evaluation results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_task_performance_comparison(self, results_dict: Dict[str, Dict[str, float]], 
                                       metrics: List[str],
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive comparison plot of task performance.
        
        Args:
            results_dict: Dictionary mapping task names to their metrics
            metrics: List of metrics to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        tasks = list(results_dict.keys())
        n_tasks = len(tasks)
        n_metrics = len(metrics)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Task Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Bar plot comparison
        ax1 = axes[0, 0]
        x = np.arange(n_tasks)
        width = 0.8 / n_metrics
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_metrics))
        
        for i, metric in enumerate(metrics):
            values = [results_dict[task].get(metric, 0) for task in tasks]
            ax1.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Tasks')
        ax1.set_ylabel('Score')
        ax1.set_title('Metric Comparison Across Tasks')
        ax1.set_xticks(x + width * (n_metrics - 1) / 2)
        ax1.set_xticklabels(tasks, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap
        ax2 = axes[0, 1]
        heatmap_data = []
        for task in tasks:
            row = [results_dict[task].get(metric, 0) for metric in metrics]
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        im = ax2.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
        
        ax2.set_xticks(np.arange(n_metrics))
        ax2.set_yticks(np.arange(n_tasks))
        ax2.set_xticklabels(metrics)
        ax2.set_yticklabels(tasks)
        ax2.set_title('Performance Heatmap')
        
        # Add text annotations
        for i in range(n_tasks):
            for j in range(n_metrics):
                text = ax2.text(j, i, f'{heatmap_data[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax2)
        
        # 3. Radar plot for first task (if multiple metrics)
        ax3 = axes[1, 0]
        if n_metrics >= 3:
            angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for i, task in enumerate(tasks[:3]):  # Show max 3 tasks
                values = [results_dict[task].get(metric, 0) for metric in metrics]
                values += values[:1]  # Complete the circle
                
                ax3.plot(angles, values, 'o-', linewidth=2, label=task, alpha=0.8)
                ax3.fill(angles, values, alpha=0.1)
            
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(metrics)
            ax3.set_ylim(0, 1)
            ax3.set_title('Radar Plot (Top 3 Tasks)')
            ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'Radar plot requires\n≥3 metrics', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Radar Plot (Insufficient Metrics)')
        
        # 4. Box plot of metric distributions
        ax4 = axes[1, 1]
        box_data = []
        box_labels = []
        
        for metric in metrics:
            values = [results_dict[task].get(metric, 0) for task in tasks]
            box_data.append(values)
            box_labels.append(metric)
        
        box_plot = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_title('Metric Distribution Across Tasks')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_scalability_analysis(self, scalability_data: Dict[str, Any],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize model scalability analysis.
        
        Args:
            scalability_data: Dictionary containing scalability metrics
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Scalability Analysis', fontsize=16, fontweight='bold')
        
        # 1. Batch size scaling
        if 'batch_size_scaling' in scalability_data:
            ax1 = axes[0, 0]
            batch_data = scalability_data['batch_size_scaling']
            
            if 'metrics' in batch_data:
                batch_sizes = []
                inference_times = []
                memory_usage = []
                
                for metric in batch_data['metrics']:
                    if 'error' not in metric:
                        batch_sizes.append(metric['batch_size'])
                        inference_times.append(metric.get('speed_metrics', {}).get('mean_inference_time', 0))
                        memory_usage.append(metric.get('memory_metrics', {}).get('peak_memory_mb', 0))
                
                ax1_twin = ax1.twinx()
                
                line1 = ax1.plot(batch_sizes, inference_times, 'bo-', label='Inference Time', linewidth=2)
                line2 = ax1_twin.plot(batch_sizes, memory_usage, 'ro-', label='Memory Usage', linewidth=2)
                
                ax1.set_xlabel('Batch Size')
                ax1.set_ylabel('Inference Time (s)', color='blue')
                ax1_twin.set_ylabel('Memory Usage (MB)', color='red')
                ax1.set_title('Batch Size Scaling')
                ax1.grid(True, alpha=0.3)
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')
        
        # 2. Sequence length scaling
        if 'sequence_length_scaling' in scalability_data:
            ax2 = axes[0, 1]
            seq_data = scalability_data['sequence_length_scaling']
            
            if 'metrics' in seq_data:
                seq_lengths = []
                inference_times = []
                memory_usage = []
                
                for metric in seq_data['metrics']:
                    if 'error' not in metric:
                        seq_lengths.append(metric['sequence_length'])
                        inference_times.append(metric.get('speed_metrics', {}).get('mean_inference_time', 0))
                        memory_usage.append(metric.get('memory_metrics', {}).get('peak_memory_mb', 0))
                
                ax2_twin = ax2.twinx()
                
                line1 = ax2.plot(seq_lengths, inference_times, 'go-', label='Inference Time', linewidth=2)
                line2 = ax2_twin.plot(seq_lengths, memory_usage, 'mo-', label='Memory Usage', linewidth=2)
                
                ax2.set_xlabel('Sequence Length')
                ax2.set_ylabel('Inference Time (s)', color='green')
                ax2_twin.set_ylabel('Memory Usage (MB)', color='magenta')
                ax2.set_title('Sequence Length Scaling')
                ax2.grid(True, alpha=0.3)
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax2.legend(lines, labels, loc='upper left')
        
        # 3. Throughput analysis
        ax3 = axes[1, 0]
        if 'batch_size_scaling' in scalability_data:
            batch_data = scalability_data['batch_size_scaling']
            
            if 'metrics' in batch_data:
                batch_sizes = []
                throughputs = []
                
                for metric in batch_data['metrics']:
                    if 'error' not in metric:
                        batch_sizes.append(metric['batch_size'])
                        throughput = metric.get('speed_metrics', {}).get('throughput_samples_per_sec', 0)
                        throughputs.append(throughput)
                
                ax3.plot(batch_sizes, throughputs, 'co-', linewidth=2, markersize=8)
                ax3.set_xlabel('Batch Size')
                ax3.set_ylabel('Throughput (samples/sec)')
                ax3.set_title('Throughput vs Batch Size')
                ax3.grid(True, alpha=0.3)
        
        # 4. Memory efficiency
        ax4 = axes[1, 1]
        if 'parameter_analysis' in scalability_data:
            param_data = scalability_data['parameter_analysis']
            
            categories = ['Total', 'Trainable', 'Non-trainable']
            values = [
                param_data.get('total_parameters', 0),
                param_data.get('trainable_parameters', 0),
                param_data.get('non_trainable_parameters', 0)
            ]
            
            # Convert to millions for readability
            values = [v / 1e6 for v in values]
            
            bars = ax4.bar(categories, values, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
            ax4.set_ylabel('Parameters (Millions)')
            ax4.set_title('Parameter Distribution')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}M', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_training_curves(self, training_history: Dict[str, List[float]],
                           validation_history: Optional[Dict[str, List[float]]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training and validation curves.
        
        Args:
            training_history: Dictionary of training metrics over epochs
            validation_history: Dictionary of validation metrics over epochs
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        metrics = list(training_history.keys())
        n_metrics = len(metrics)
        
        # Determine subplot layout
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metrics):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            epochs = range(1, len(training_history[metric]) + 1)
            
            # Plot training curve
            ax.plot(epochs, training_history[metric], 'b-', label=f'Training {metric}', 
                   linewidth=2, marker='o', markersize=4)
            
            # Plot validation curve if available
            if validation_history and metric in validation_history:
                ax.plot(epochs, validation_history[metric], 'r-', label=f'Validation {metric}', 
                       linewidth=2, marker='s', markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add best value annotation
            if validation_history and metric in validation_history:
                best_val = max(validation_history[metric]) if 'acc' in metric or 'f1' in metric else min(validation_history[metric])
                best_epoch = validation_history[metric].index(best_val) + 1
                ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
                ax.text(best_epoch, best_val, f'Best: {best_val:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        # Hide empty subplots
        for i in range(n_metrics, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            elif cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrices(self, confusion_matrices: Dict[str, np.ndarray],
                              class_names: Optional[Dict[str, List[str]]] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrices for multiple tasks.
        
        Args:
            confusion_matrices: Dictionary mapping task names to confusion matrices
            class_names: Optional dictionary mapping task names to class names
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        n_tasks = len(confusion_matrices)
        cols = min(3, n_tasks)
        rows = (n_tasks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        if n_tasks == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (task_name, cm) in enumerate(confusion_matrices.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Set labels
            if class_names and task_name in class_names:
                tick_marks = np.arange(len(class_names[task_name]))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(class_names[task_name], rotation=45)
                ax.set_yticklabels(class_names[task_name])
            else:
                n_classes = cm.shape[0]
                tick_marks = np.arange(n_classes)
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels([f'Class {i}' for i in range(n_classes)], rotation=45)
                ax.set_yticklabels([f'Class {i}' for i in range(n_classes)])
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for i_cm in range(cm.shape[0]):
                for j_cm in range(cm.shape[1]):
                    ax.text(j_cm, i_cm, f'{cm[i_cm, j_cm]}\n({cm_normalized[i_cm, j_cm]:.2f})',
                           ha="center", va="center",
                           color="white" if cm_normalized[i_cm, j_cm] > thresh else "black",
                           fontsize=9)
            
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title(f'{task_name}')
        
        # Hide empty subplots
        for i in range(n_tasks, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            elif cols > 1:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


def create_evaluation_dashboard(results: Dict[str, Any], 
                              output_dir: str,
                              model_name: str = "HyenaGLT") -> List[str]:
    """
    Create a comprehensive visualization dashboard.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save visualizations
        model_name: Name of the model
        
    Returns:
        List of paths to generated visualization files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = EvaluationVisualizer()
    generated_files = []
    
    # 1. Task performance comparison
    if 'task_results' in results:
        task_metrics = {}
        for task_name, task_data in results['task_results'].items():
            if isinstance(task_data, dict) and 'summary_metrics' in task_data:
                task_metrics[task_name] = task_data['summary_metrics']
        
        if task_metrics:
            # Determine available metrics
            all_metrics = set()
            for metrics in task_metrics.values():
                all_metrics.update(metrics.keys())
            
            # Select key metrics for visualization
            key_metrics = []
            for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc_roc']:
                if metric in all_metrics:
                    key_metrics.append(metric)
            
            if key_metrics:
                performance_path = output_path / f'{model_name}_task_performance.png'
                fig = visualizer.plot_task_performance_comparison(
                    task_metrics, key_metrics, str(performance_path)
                )
                plt.close(fig)
                generated_files.append(str(performance_path))
    
    # 2. Scalability analysis
    scalability_data = {}
    for key in ['batch_size_scaling', 'sequence_length_scaling', 'parameter_analysis']:
        if key in results:
            scalability_data[key] = results[key]
    
    if scalability_data:
        scalability_path = output_path / f'{model_name}_scalability.png'
        fig = visualizer.plot_scalability_analysis(scalability_data, str(scalability_path))
        plt.close(fig)
        generated_files.append(str(scalability_path))
    
    # 3. Training curves (if available)
    if 'training_history' in results:
        training_path = output_path / f'{model_name}_training_curves.png'
        fig = visualizer.plot_training_curves(
            results['training_history'],
            results.get('validation_history'),
            str(training_path)
        )
        plt.close(fig)
        generated_files.append(str(training_path))
    
    return generated_files
