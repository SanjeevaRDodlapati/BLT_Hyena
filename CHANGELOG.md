# Changelog

All notable changes to the Hyena-GLT framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-05-30

### Added
- **Performance Monitoring Utilities**: Comprehensive performance monitoring system
  - `ProfilerContext`: Context manager for detailed operation profiling
  - `memory_usage()`: Real-time memory usage statistics
  - `gpu_memory_usage()`: GPU memory monitoring (when PyTorch/CUDA available)
  - `benchmark_model()`: Model benchmarking with statistical analysis
  - `measure_throughput()`: Throughput measurement for production workloads
  - `monitor_resources()`: Continuous resource monitoring during execution

- **Enhanced Utilities Module**: 
  - Added performance monitoring imports to `hyena_glt.utils`
  - Version tracking with `__version__` and `__author__` metadata
  - Comprehensive `__all__` exports for better API discovery

- **Example Scripts**:
  - `performance_monitoring_demo.py`: Complete demonstration of performance utilities

- **Multi-Repository Support**:
  - Interactive setup script for pushing to multiple GitHub accounts
  - Comprehensive push guide with authentication options
  - Username-based remote naming convention

### Changed
- Updated package version from 0.1.0 to 1.0.1
- Enhanced documentation with performance monitoring capabilities

### Technical Details
- Performance utilities support both CPU and GPU monitoring
- Thread-safe resource monitoring with configurable sampling intervals
- Statistical analysis for benchmark results (mean, min, max, std deviation)
- Memory efficient implementation with minimal overhead

## [0.1.0] - 2025-05-30

### Added
- Initial release of Hyena-GLT framework
- Core HyenaGLT model implementation combining BLT and Hyena architectures
- Genomic tokenization and data processing pipeline
- Distributed training infrastructure
- Comprehensive test suite with 90%+ coverage
- Documentation and example notebooks
- Analysis and visualization utilities
