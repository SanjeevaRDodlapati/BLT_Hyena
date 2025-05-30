"""
Evaluation framework for Hyena-GLT models.
"""

from .metrics import (
    BaseMetric,
    ClassificationMetrics,
    RegressionMetrics,
    PerplexityMetric,
    GenomicSequenceMetrics,
    ComputationalMetrics,
    MultiTaskEvaluator,
    BenchmarkEvaluator,
    EvaluationResult
)

from .benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    ModelProfiler,
    ScalabilityBenchmark,
    ComparisonBenchmark,
    SystemProfiler,
    BenchmarkRunner,
    create_genomic_benchmark_suite
)

from .analysis import (
    ModelAnalyzer,
    ResultsAnalyzer,
    StatisticalAnalyzer,
    VisualizationUtils,
    create_comprehensive_report
)

__all__ = [
    # Metrics
    'BaseMetric',
    'ClassificationMetrics', 
    'RegressionMetrics',
    'PerplexityMetric',
    'GenomicSequenceMetrics',
    'ComputationalMetrics',
    'MultiTaskEvaluator',
    'BenchmarkEvaluator',
    'EvaluationResult',
    
    # Benchmarking
    'BenchmarkConfig',
    'BenchmarkResult', 
    'ModelProfiler',
    'ScalabilityBenchmark',
    'ComparisonBenchmark',
    'SystemProfiler',
    'BenchmarkRunner',
    'create_genomic_benchmark_suite',
    
    # Analysis
    'ModelAnalyzer',
    'ResultsAnalyzer',
    'StatisticalAnalyzer', 
    'VisualizationUtils',
    'create_comprehensive_report'
]
