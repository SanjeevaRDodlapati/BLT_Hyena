"""
Comprehensive evaluation example for Hyena-GLT model.

This example demonstrates:
1. Model evaluation on multiple genomic tasks
2. Benchmarking performance and computational efficiency
3. Statistical analysis and visualization
4. Report generation
"""

import logging
from pathlib import Path

import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run comprehensive evaluation example."""

    # Import necessary modules
    from hyena_glt.config import HyenaGLTConfig
    from hyena_glt.data import GenomicDataset, GenomicTokenizer
    from hyena_glt.evaluation.analysis import (
        ResultsAnalyzer,
        create_comprehensive_report,
    )
    from hyena_glt.evaluation.benchmark import (
        BenchmarkConfig,
        BenchmarkRunner,
        ModelProfiler,
    )
    from hyena_glt.evaluation.metrics import (
        BenchmarkEvaluator,
    )
    from hyena_glt.model import (
        HyenaGLTForSequenceClassification,
        HyenaGLTForTokenClassification,
    )

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ========================
    # 1. Model Configuration
    # ========================

    logger.info("Setting up model configuration...")

    # Create model configuration
    config = HyenaGLTConfig(
        vocab_size=8,  # DNA tokens: A, C, G, T, N, PAD, BOS, EOS
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=2048,
        hyena_order=2,
        hyena_filter_size=64,
        local_window_size=128,
        merge_ratio=0.25,
        use_genomic_encoding=True,
        dropout=0.1
    )

    # ========================
    # 2. Create Sample Data
    # ========================

    logger.info("Creating sample genomic datasets...")

    # Initialize tokenizer
    tokenizer = GenomicTokenizer(sequence_type='dna', k=1)

    # Create sample datasets for different tasks
    def create_sample_data(task_type: str, num_samples: int = 100, seq_length: int = 512):
        """Create sample genomic data for evaluation."""
        sequences = []
        labels = []

        for _i in range(num_samples):
            # Generate random DNA sequence
            nucleotides = ['A', 'C', 'G', 'T']
            sequence = ''.join(np.random.choice(nucleotides, seq_length))
            sequences.append(sequence)

            if task_type == 'sequence_classification':
                # Binary classification: GC content > 0.5
                gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
                labels.append(1 if gc_content > 0.5 else 0)
            elif task_type == 'token_classification':
                # Token-level: label each position (e.g., promoter regions)
                token_labels = np.random.randint(0, 3, seq_length)  # 3 classes
                labels.append(token_labels.tolist())

        return sequences, labels

    # Create datasets for different tasks
    tasks_data = {
        'promoter_detection': create_sample_data('sequence_classification', 200, 512),
        'splice_site_prediction': create_sample_data('sequence_classification', 200, 256),
        'gene_annotation': create_sample_data('token_classification', 100, 1024)
    }

    # ========================
    # 3. Model Evaluation
    # ========================

    logger.info("Starting model evaluation...")

    # Task configurations
    task_configs = {
        'promoter_detection': {
            'type': 'classification',
            'num_classes': 2,
            'model_class': HyenaGLTForSequenceClassification
        },
        'splice_site_prediction': {
            'type': 'classification',
            'num_classes': 2,
            'model_class': HyenaGLTForSequenceClassification
        },
        'gene_annotation': {
            'type': 'classification',
            'num_classes': 3,
            'model_class': HyenaGLTForTokenClassification
        }
    }

    # Evaluate each task
    evaluation_results = {}

    for task_name, task_config in task_configs.items():
        logger.info(f"Evaluating task: {task_name}")

        # Create model for this task
        model_config = config.copy()
        model_config.num_labels = task_config['num_classes']

        model = task_config['model_class'](model_config)
        model.to(device)
        model.eval()

        # Get task data
        sequences, labels = tasks_data[task_name]

        # Create data loader
        dataset = GenomicDataset(
            sequences=sequences[:50],  # Use subset for demo
            labels=labels[:50],
            tokenizer=tokenizer,
            max_length=512,
            task_type=task_config['type']
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False
        )

        # Create evaluator
        evaluator = BenchmarkEvaluator({
            'tasks': {task_name: task_config}
        })

        # Run evaluation
        task_results = evaluator.evaluate_model(model, data_loader, device)
        evaluation_results[task_name] = task_results

        logger.info(f"Completed evaluation for {task_name}")

    # ========================
    # 4. Performance Benchmarking
    # ========================

    logger.info("Running performance benchmarks...")

    # Create benchmark configuration
    benchmark_config = BenchmarkConfig(
        name="hyena_glt_genomic_benchmark",
        description="Comprehensive benchmark for Hyena-GLT on genomic tasks",
        datasets=list(task_configs.keys()),
        metrics=["accuracy", "f1", "inference_time", "memory_usage"],
        batch_sizes=[1, 4, 8, 16],
        sequence_lengths=[256, 512, 1024],
        num_runs=3,
        device=str(device)
    )

    # Run scalability analysis
    def model_factory(config_dict):
        """Factory function to create models for benchmarking."""
        return HyenaGLTForSequenceClassification(config)

    benchmark_runner = BenchmarkRunner(benchmark_config)

    # Create sample data loaders for benchmarking
    benchmark_data_loaders = {}
    for task_name in task_configs.keys():
        sequences, labels = tasks_data[task_name]
        dataset = GenomicDataset(
            sequences=sequences[:20],  # Small subset for benchmarking
            labels=labels[:20],
            tokenizer=tokenizer,
            max_length=512,
            task_type='sequence_classification'
        )
        benchmark_data_loaders[task_name] = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=False
        )

    # Run benchmark
    try:
        benchmark_runner.run_full_benchmark(
            model_factory=model_factory,
            base_model_config=config.__dict__,
            data_loaders=benchmark_data_loaders
        )
        logger.info("Benchmark completed successfully")
    except Exception as e:
        logger.warning(f"Benchmark failed: {e}")

    # ========================
    # 5. Model Analysis
    # ========================

    logger.info("Performing model analysis...")

    # Create a model for analysis
    analysis_model = HyenaGLTForSequenceClassification(config)
    analysis_model.to(device)
    analysis_model.eval()

    # Model profiler
    profiler = ModelProfiler(analysis_model)

    # Count parameters
    param_info = profiler.count_parameters()
    logger.info(f"Model parameters: {param_info}")

    # Measure memory usage
    input_shapes = {'input_ids': (8, 512)}  # batch_size=8, seq_len=512
    try:
        memory_info = profiler.measure_memory_usage(input_shapes)
        logger.info(f"Memory usage: {memory_info}")
    except Exception as e:
        logger.warning(f"Memory analysis failed: {e}")
        memory_info = {}

    # Measure inference speed
    try:
        speed_info = profiler.profile_inference_speed(input_shapes, num_runs=10)
        logger.info(f"Inference speed: {speed_info}")
    except Exception as e:
        logger.warning(f"Speed analysis failed: {e}")
        speed_info = {}

    # ========================
    # 6. Results Analysis
    # ========================

    logger.info("Analyzing evaluation results...")

    # Convert evaluation results to the expected format
    formatted_results = {}
    for task_name, task_result in evaluation_results.items():
        if 'summary_metrics' in task_result:
            from hyena_glt.evaluation.metrics import EvaluationResult
            formatted_results[task_name] = EvaluationResult(
                task_name=task_name,
                metrics=task_result['summary_metrics']
            )
        else:
            logger.warning(f"No summary metrics found for {task_name}")

    if formatted_results:
        # Create results analyzer
        results_analyzer = ResultsAnalyzer(formatted_results)

        # Generate performance summary
        try:
            summary_df = results_analyzer.create_performance_summary()
            logger.info("Performance Summary:")
            logger.info(f"\n{summary_df}")
        except Exception as e:
            logger.warning(f"Could not create performance summary: {e}")

        # Analyze task difficulty
        try:
            difficulty_scores = results_analyzer.analyze_task_difficulty()
            logger.info("Task Difficulty Analysis:")
            for task, score in difficulty_scores.items():
                logger.info(f"  {task}: {score:.3f}")
        except Exception as e:
            logger.warning(f"Could not analyze task difficulty: {e}")

    # ========================
    # 7. Generate Comprehensive Report
    # ========================

    logger.info("Generating comprehensive evaluation report...")

    # Create output directory
    output_dir = Path("./evaluation_results")
    output_dir.mkdir(exist_ok=True)

    if formatted_results:
        try:
            report_path = create_comprehensive_report(
                results=formatted_results,
                output_dir=str(output_dir),
                model_name="Hyena-GLT"
            )
            logger.info(f"Comprehensive report generated: {report_path}")
        except Exception as e:
            logger.warning(f"Could not generate comprehensive report: {e}")

    # ========================
    # 8. Summary and Recommendations
    # ========================

    logger.info("=== EVALUATION SUMMARY ===")
    logger.info(f"Evaluated {len(task_configs)} genomic tasks")
    logger.info(f"Model parameters: {param_info.get('total_parameters', 'Unknown'):,}")

    if memory_info:
        logger.info(f"Peak memory usage: {memory_info.get('peak_memory_mb', 'Unknown'):.1f} MB")

    if speed_info:
        logger.info(f"Average inference time: {speed_info.get('mean_inference_time', 'Unknown'):.3f} seconds")
        logger.info(f"Throughput: {speed_info.get('throughput_samples_per_sec', 'Unknown'):.1f} samples/sec")

    # Print task performance
    for task_name, task_result in evaluation_results.items():
        if 'summary_metrics' in task_result:
            metrics = task_result['summary_metrics']
            if metrics:
                logger.info(f"\n{task_name} Performance:")
                for metric, value in metrics.items():
                    if isinstance(value, int | float):
                        logger.info(f"  {metric}: {value:.4f}")

    logger.info("\n=== RECOMMENDATIONS ===")
    logger.info("1. Consider task-specific fine-tuning for improved performance")
    logger.info("2. Experiment with different hyperparameters (hidden_size, num_layers)")
    logger.info("3. Implement curriculum learning for complex tasks")
    logger.info("4. Add more diverse training data for underperforming tasks")
    logger.info("5. Consider ensemble methods for critical applications")

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
