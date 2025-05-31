"""
Evaluation framework for Hyena-GLT models.
"""

from .analysis import (
    ModelAnalyzer,
    ResultsAnalyzer,
    StatisticalAnalyzer,
    VisualizationUtils,
    create_comprehensive_report,
)
from .benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    ComparisonBenchmark,
    ModelProfiler,
    ScalabilityBenchmark,
    SystemProfiler,
    create_genomic_benchmark_suite,
)
from .metrics import (
    BaseMetric,
    BenchmarkEvaluator,
    ClassificationMetrics,
    ComputationalMetrics,
    EvaluationResult,
    GenomicSequenceMetrics,
    MultiTaskEvaluator,
    PerplexityMetric,
    RegressionMetrics,
)

__all__ = [
    # Metrics
    "BaseMetric",
    "ClassificationMetrics",
    "RegressionMetrics",
    "PerplexityMetric",
    "GenomicSequenceMetrics",
    "ComputationalMetrics",
    "MultiTaskEvaluator",
    "BenchmarkEvaluator",
    "EvaluationResult",
    # Benchmarking
    "BenchmarkConfig",
    "BenchmarkResult",
    "ModelProfiler",
    "ScalabilityBenchmark",
    "ComparisonBenchmark",
    "SystemProfiler",
    "BenchmarkRunner",
    "create_genomic_benchmark_suite",
    # Analysis
    "ModelAnalyzer",
    "ResultsAnalyzer",
    "StatisticalAnalyzer",
    "VisualizationUtils",
    "create_comprehensive_report",
]
