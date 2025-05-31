"""
Analysis utilities for model evaluation and results interpretation.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats

from .metrics import EvaluationResult

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from captum.attr import IntegratedGradients, LayerConductance

    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


class ModelAnalyzer:
    """Comprehensive model analysis utilities."""

    def __init__(self, model: torch.nn.Module, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def analyze_attention_patterns(
        self, input_ids: torch.Tensor, layer_indices: list[int] | None = None
    ) -> dict[str, np.ndarray]:
        """
        Analyze attention patterns in the model.

        Args:
            input_ids: Input sequence [batch_size, seq_len]
            layer_indices: Specific layers to analyze (default: all)

        Returns:
            Dictionary containing attention weights for each layer
        """
        if not hasattr(self.model, "get_attention_weights"):
            warnings.warn(
                "Model doesn't support attention weight extraction", stacklevel=2
            )
            return {}

        self.model.eval()
        attention_patterns = {}

        with torch.no_grad():
            # Get attention weights
            outputs = self.model(input_ids, output_attentions=True)
            attentions = outputs.attentions if hasattr(outputs, "attentions") else []

            if layer_indices is None:
                layer_indices = list(range(len(attentions)))

            for layer_idx in layer_indices:
                if layer_idx < len(attentions):
                    # Average across heads and batch
                    attn_weights = attentions[layer_idx].cpu().numpy()
                    attn_weights = np.mean(
                        attn_weights, axis=(0, 1)
                    )  # Average batch and heads
                    attention_patterns[f"layer_{layer_idx}"] = attn_weights

        return attention_patterns

    def analyze_hidden_representations(
        self, input_ids: torch.Tensor, layer_indices: list[int] | None = None
    ) -> dict[str, np.ndarray]:
        """
        Extract and analyze hidden representations from different layers.

        Args:
            input_ids: Input sequence [batch_size, seq_len]
            layer_indices: Specific layers to analyze

        Returns:
            Dictionary containing hidden states for each layer
        """
        self.model.eval()
        hidden_states = {}

        # Register hooks to capture intermediate representations
        activations = {}

        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()

            return hook

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if "layer" in name or "block" in name:
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)

        with torch.no_grad():
            _ = self.model(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process activations
        for name, activation in activations.items():
            if layer_indices is None or any(
                f"layer_{i}" in name or f"block_{i}" in name for i in layer_indices
            ):
                # Average across sequence length
                hidden_states[name] = activation.mean(dim=1).cpu().numpy()

        return hidden_states

    def compute_feature_importance(
        self, input_ids: torch.Tensor, target_class: int | None = None
    ) -> np.ndarray:
        """
        Compute feature importance using integrated gradients.

        Args:
            input_ids: Input sequence [batch_size, seq_len]
            target_class: Target class for attribution (for classification)

        Returns:
            Feature importance scores
        """
        if not HAS_CAPTUM:
            warnings.warn(
                "Captum not available for feature importance analysis", stacklevel=2
            )
            return np.array([])

        self.model.eval()

        # Create baseline (typically zeros or random)
        baseline = torch.zeros_like(input_ids)

        # Initialize integrated gradients
        ig = IntegratedGradients(self.model)

        # Compute attributions
        attributions = ig.attribute(
            input_ids, baseline, target=target_class, return_convergence_delta=False
        )

        return attributions.cpu().numpy()

    def analyze_layer_importance(self, input_ids: torch.Tensor) -> dict[str, float]:
        """
        Analyze the importance of different layers using layer conductance.

        Args:
            input_ids: Input sequence [batch_size, seq_len]

        Returns:
            Dictionary mapping layer names to importance scores
        """
        if not HAS_CAPTUM:
            warnings.warn(
                "Captum not available for layer importance analysis", stacklevel=2
            )
            return {}

        self.model.eval()
        layer_importance = {}

        # Analyze each layer
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear | torch.nn.Conv1d):
                try:
                    lc = LayerConductance(self.model, module)
                    conductance = lc.attribute(input_ids)
                    importance = conductance.abs().sum().item()
                    layer_importance[name] = importance
                except Exception as e:
                    warnings.warn(f"Could not analyze layer {name}: {e}", stacklevel=2)

        return layer_importance


class ResultsAnalyzer:
    """Analyze and visualize evaluation results."""

    def __init__(self, results: dict[str, EvaluationResult]):
        self.results = results

    def create_performance_summary(self) -> pd.DataFrame:
        """Create a summary DataFrame of all results."""
        summary_data = []

        for task_name, result in self.results.items():
            row = {"task": task_name}
            row.update(result.metrics)
            summary_data.append(row)

        return pd.DataFrame(summary_data)

    def plot_metrics_comparison(
        self, metrics: list[str], save_path: str | None = None
    ) -> plt.Figure:
        """
        Create comparison plots for specified metrics across tasks.

        Args:
            metrics: List of metric names to compare
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        summary_df = self.create_performance_summary()

        # Filter to only include specified metrics
        plot_data = summary_df[
            ["task"] + [m for m in metrics if m in summary_df.columns]
        ]

        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            if metric in plot_data.columns:
                ax = axes[i]
                bars = ax.bar(plot_data["task"], plot_data[metric])
                ax.set_title(f"{metric.capitalize()} by Task")
                ax.set_ylabel(metric.capitalize())
                ax.tick_params(axis="x", rotation=45)

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height,
                            f"{height:.3f}",
                            ha="center",
                            va="bottom",
                        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_correlation_heatmap(self, save_path: str | None = None) -> plt.Figure:
        """
        Create correlation heatmap of metrics across tasks.

        Args:
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        summary_df = self.create_performance_summary()

        # Select only numeric columns
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        correlation_data = summary_df[numeric_cols]

        # Compute correlation matrix
        corr_matrix = correlation_data.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", center=0, square=True, ax=ax
        )
        ax.set_title("Metric Correlation Heatmap")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def analyze_task_difficulty(self) -> dict[str, float]:
        """
        Analyze relative difficulty of tasks based on performance metrics.

        Returns:
            Dictionary mapping task names to difficulty scores
        """
        summary_df = self.create_performance_summary()

        # Define metrics that indicate good performance (higher is better)
        positive_metrics = [
            "accuracy",
            "f1",
            "precision",
            "recall",
            "auc_roc",
            "auc_pr",
        ]

        # Define metrics that indicate poor performance (lower is better)
        negative_metrics = ["loss", "perplexity", "mse", "mae"]

        difficulty_scores = {}

        for _, row in summary_df.iterrows():
            task = row["task"]
            scores = []

            # Process positive metrics (invert so higher difficulty = higher score)
            for metric in positive_metrics:
                if metric in row and not pd.isna(row[metric]):
                    # Normalize to 0-1 and invert
                    max_val = (
                        summary_df[metric].max() if summary_df[metric].max() > 0 else 1
                    )
                    normalized = row[metric] / max_val
                    scores.append(1 - normalized)  # Invert

            # Process negative metrics (higher values = higher difficulty)
            for metric in negative_metrics:
                if metric in row and not pd.isna(row[metric]):
                    # Normalize to 0-1
                    max_val = (
                        summary_df[metric].max() if summary_df[metric].max() > 0 else 1
                    )
                    normalized = row[metric] / max_val
                    scores.append(normalized)

            # Average difficulty score
            difficulty_scores[task] = np.mean(scores) if scores else 0.5

        return difficulty_scores


class StatisticalAnalyzer:
    """Statistical analysis of evaluation results."""

    def __init__(self, results_list: list[dict[str, EvaluationResult]]):
        """
        Initialize with multiple evaluation runs.

        Args:
            results_list: List of result dictionaries from multiple runs
        """
        self.results_list = results_list

    def compute_significance_tests(
        self, metric: str, comparison_pairs: list[tuple[str, str]] | None = None
    ) -> dict[str, dict]:
        """
        Perform statistical significance tests between tasks/models.

        Args:
            metric: Metric to analyze
            comparison_pairs: Pairs of tasks to compare (default: all pairs)

        Returns:
            Dictionary containing test results
        """
        # Extract metric values for each task across runs
        task_values = {}
        for results in self.results_list:
            for task_name, result in results.items():
                if metric in result.metrics:
                    if task_name not in task_values:
                        task_values[task_name] = []
                    task_values[task_name].append(result.metrics[metric])

        # Generate comparison pairs if not provided
        if comparison_pairs is None:
            tasks = list(task_values.keys())
            comparison_pairs = [
                (tasks[i], tasks[j])
                for i in range(len(tasks))
                for j in range(i + 1, len(tasks))
            ]

        test_results = {}

        for task1, task2 in comparison_pairs:
            if task1 in task_values and task2 in task_values:
                values1 = np.array(task_values[task1])
                values2 = np.array(task_values[task2])

                # Perform t-test
                t_stat, t_p_value = stats.ttest_ind(values1, values2)

                # Perform Mann-Whitney U test (non-parametric)
                u_stat, u_p_value = stats.mannwhitneyu(
                    values1, values2, alternative="two-sided"
                )

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (
                        (len(values1) - 1) * np.var(values1, ddof=1)
                        + (len(values2) - 1) * np.var(values2, ddof=1)
                    )
                    / (len(values1) + len(values2) - 2)
                )
                cohens_d = (
                    (np.mean(values1) - np.mean(values2)) / pooled_std
                    if pooled_std > 0
                    else 0
                )

                test_results[f"{task1}_vs_{task2}"] = {
                    "t_statistic": t_stat,
                    "t_p_value": t_p_value,
                    "u_statistic": u_stat,
                    "u_p_value": u_p_value,
                    "cohens_d": cohens_d,
                    "mean_diff": np.mean(values1) - np.mean(values2),
                }

        return test_results

    def compute_confidence_intervals(
        self, metric: str, confidence: float = 0.95
    ) -> dict[str, tuple[float, float]]:
        """
        Compute confidence intervals for each task.

        Args:
            metric: Metric to analyze
            confidence: Confidence level (default: 0.95)

        Returns:
            Dictionary mapping task names to (lower, upper) confidence bounds
        """
        confidence_intervals = {}
        alpha = 1 - confidence

        # Extract metric values for each task
        for task_name in self.results_list[0].keys():
            values = []
            for results in self.results_list:
                if task_name in results and metric in results[task_name].metrics:
                    values.append(results[task_name].metrics[metric])

            if len(values) > 1:
                values = np.array(values)
                mean = np.mean(values)
                sem = stats.sem(values)

                # Use t-distribution for small samples
                if len(values) < 30:
                    t_val = stats.t.ppf(1 - alpha / 2, len(values) - 1)
                    margin = t_val * sem
                else:
                    z_val = stats.norm.ppf(1 - alpha / 2)
                    margin = z_val * sem

                confidence_intervals[task_name] = (mean - margin, mean + margin)

        return confidence_intervals


class VisualizationUtils:
    """Utilities for creating advanced visualizations."""

    @staticmethod
    def create_interactive_performance_dashboard(
        results: dict[str, EvaluationResult], save_path: str | None = None
    ):
        """
        Create an interactive dashboard for performance analysis.

        Args:
            results: Evaluation results
            save_path: Path to save HTML file
        """
        if not HAS_PLOTLY:
            warnings.warn(
                "Plotly not available for interactive visualizations", stacklevel=2
            )
            return

        # Prepare data
        summary_data = []
        for task_name, result in results.items():
            row = {"Task": task_name}
            row.update(result.metrics)
            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Accuracy Metrics",
                "F1 Scores",
                "Precision vs Recall",
                "Metric Distribution",
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "box"}],
            ],
        )

        # Accuracy metrics
        if "accuracy" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df["Task"], y=df["accuracy"], name="Accuracy", marker_color="blue"
                ),
                row=1,
                col=1,
            )

        # F1 scores
        if "f1" in df.columns:
            fig.add_trace(
                go.Bar(x=df["Task"], y=df["f1"], name="F1 Score", marker_color="green"),
                row=1,
                col=2,
            )

        # Precision vs Recall scatter
        if "precision" in df.columns and "recall" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["precision"],
                    y=df["recall"],
                    mode="markers+text",
                    text=df["Task"],
                    textposition="top center",
                    marker={"size": 10, "color": "red"},
                    name="Precision vs Recall",
                ),
                row=2,
                col=1,
            )

        # Box plot of all metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != "Task":
                fig.add_trace(go.Box(y=df[col], name=col), row=2, col=2)

        # Update layout
        fig.update_layout(
            title="Model Performance Dashboard", height=800, showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive dashboard saved to: {save_path}")

        return fig

    @staticmethod
    def plot_learning_curves(
        training_history: dict[str, list[float]],
        validation_history: dict[str, list[float]],
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot learning curves for training and validation metrics.

        Args:
            training_history: Training metrics over epochs
            validation_history: Validation metrics over epochs
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        metrics = set(training_history.keys()) & set(validation_history.keys())
        n_metrics = len(metrics)

        if n_metrics == 0:
            return plt.figure()

        # Create subplots
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i, metric in enumerate(metrics):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            epochs = range(1, len(training_history[metric]) + 1)
            ax.plot(
                epochs, training_history[metric], label=f"Training {metric}", marker="o"
            )
            ax.plot(
                epochs,
                validation_history[metric],
                label=f"Validation {metric}",
                marker="s",
            )

            ax.set_title(f"{metric.capitalize()} Learning Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)

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
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


def create_comprehensive_report(
    results: dict[str, EvaluationResult], output_dir: str, model_name: str = "HyenaGLT"
) -> str:
    """
    Create a comprehensive evaluation report.

    Args:
        results: Evaluation results
        output_dir: Directory to save report files
        model_name: Name of the model being evaluated

    Returns:
        Path to the main report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create analyzer
    analyzer = ResultsAnalyzer(results)

    # Generate visualizations
    summary_df = analyzer.create_performance_summary()

    # Performance comparison plot
    key_metrics = ["accuracy", "f1", "precision", "recall"]
    available_metrics = [m for m in key_metrics if m in summary_df.columns]
    if available_metrics:
        comparison_fig = analyzer.plot_metrics_comparison(
            available_metrics, save_path=str(output_path / "metrics_comparison.png")
        )
        plt.close(comparison_fig)

    # Correlation heatmap
    corr_fig = analyzer.create_correlation_heatmap(
        save_path=str(output_path / "correlation_heatmap.png")
    )
    plt.close(corr_fig)

    # Task difficulty analysis
    difficulty_scores = analyzer.analyze_task_difficulty()

    # Create main report
    report_path = output_path / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(f"# {model_name} Evaluation Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write(
            f"This report presents comprehensive evaluation results for the {model_name} model "
        )
        f.write(f"across {len(results)} different tasks.\n\n")

        f.write("## Performance Overview\n\n")
        f.write("### Summary Statistics\n\n")
        f.write(summary_df.describe().to_markdown())
        f.write("\n\n")

        f.write("### Task-wise Performance\n\n")
        for task_name, result in results.items():
            f.write(f"#### {task_name}\n\n")
            for metric, value in result.metrics.items():
                f.write(f"- **{metric}**: {value:.4f}\n")
            f.write("\n")

        f.write("## Task Difficulty Analysis\n\n")
        f.write("Relative difficulty scores (higher = more difficult):\n\n")
        sorted_difficulty = sorted(
            difficulty_scores.items(), key=lambda x: x[1], reverse=True
        )
        for task, score in sorted_difficulty:
            f.write(f"- **{task}**: {score:.3f}\n")
        f.write("\n")

        f.write("## Visualizations\n\n")
        f.write("![Metrics Comparison](metrics_comparison.png)\n\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")

        f.write("## Recommendations\n\n")

        # Find best and worst performing tasks
        if "f1" in summary_df.columns:
            best_task = summary_df.loc[summary_df["f1"].idxmax(), "task"]
            worst_task = summary_df.loc[summary_df["f1"].idxmin(), "task"]
            f.write(f"- **Best performing task**: {best_task}\n")
            f.write(f"- **Most challenging task**: {worst_task}\n")

        f.write("- Consider additional training data for low-performing tasks\n")
        f.write("- Investigate task-specific architectural modifications\n")
        f.write(
            "- Implement transfer learning from high-performing to low-performing tasks\n\n"
        )

    # Save detailed results as JSON
    results_dict = {}
    for task_name, result in results.items():
        results_dict[task_name] = {
            "metrics": result.metrics,
            "task_name": result.task_name,
        }

    with open(output_path / "detailed_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Comprehensive evaluation report saved to: {report_path}")
    return str(report_path)
