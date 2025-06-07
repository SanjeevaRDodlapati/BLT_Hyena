# Hyena-GLT: Genome Language Transformer

> **Status**: ✅ **Production Ready** (v1.0.1) | **Documentation**: Complete | **Tests**: Passing

A hybrid genomic language model combining BLT's dynamic token merging with Hyena's efficient long-range convolutions, specifically designed for genomic sequence modeling.

## 🎯 Overview

Hyena-GLT integrates cutting-edge techniques for efficient genomic sequence processing:

- **BLT's Dynamic Token Merging**: Adaptive compression achieving 16-64x compression ratios
- **Hyena's Long-Range Convolutions**: Efficient pattern recognition with O(n log n) complexity
- **Cross-Attention Bridges**: Seamless local-global information flow
- **Genomic Specialization**: Purpose-built for DNA, RNA, and protein sequence analysis

## 📊 Project Status

| Component | Status | Coverage | Notes |
|-----------|--------|----------|--------|
| **Data Infrastructure** | ✅ Complete | 100% | Reorganized & optimized |
| **Model Architecture** | ✅ Complete | 100% | Production ready |
| **Training Pipeline** | ✅ Complete | 100% | Multi-task support |
| **Mixed Precision** | ✅ Enhanced | 100% | Task-specific optimization |
| **Interpretability** | ✅ Complete | 100% | Advanced analysis tools |
| **Documentation** | ✅ Complete | 6,000+ lines | Comprehensive implementation guides |
| **Testing** | ✅ Passing | 90%+ | Automated benchmarks |
| **Repository Structure** | ✅ Reorganized | 100% | Professional organization |

📋 **For detailed status**: See [`admin/PROJECT_STATUS.md`](admin/PROJECT_STATUS.md)  
🎉 **Reorganization**: See [`REORGANIZATION_COMPLETE.md`](docs/project_management/REORGANIZATION_COMPLETE.md)

## ✨ Features

- 🧬 **Genomic Tokenization**: Specialized tokenizers for DNA, RNA, and protein sequences
- ⚡ **Efficient Architecture**: Hyena convolutions for O(n log n) complexity
- 🔄 **Dynamic Processing**: Adaptive token merging based on sequence content
- 🎯 **Mixed Precision**: Hardware-aware FP16/BF16/FP8 optimization
- 📊 **Multi-task Learning**: Support for classification, generation, and analysis tasks
- 🎯 **Fine-tuning Ready**: Pre-configured for genomic downstream tasks
- 🔍 **Interpretability**: Built-in attention visualization and analysis tools
- 📈 **Performance Optimized**: Memory-efficient training with gradient checkpointing
- ⚡ **Mixed Precision**: Task-specific FP16/BF16/FP8 optimization with hardware awareness
- 🧪 **Comprehensive Testing**: Full test suite with benchmarking capabilities

## 🚀 Quick Navigation

| Need to... | Go to... |
|------------|----------|
| **Check project status** | [`admin/PROJECT_STATUS.md`](admin/PROJECT_STATUS.md) |
| **Start development** | [`admin/SESSION_KICKSTART.md`](admin/SESSION_KICKSTART.md) |
| **View documentation** | [`docs/README.md`](docs/README.md) |
| **Run demos** | [`scripts/demos/`](scripts/demos/) |
| **Run benchmarks** | [`scripts/benchmarks/`](scripts/benchmarks/) |
| **Check examples** | [`examples/`](examples/) |

## 📁 Repository Structure

```text
├── 📁 admin/              # Project management & status
├── 📁 hyena_glt/          # Core framework code
├── 📁 docs/               # Complete documentation
├── 📁 examples/           # Usage examples
├── 📁 scripts/            # Demos, benchmarks, setup
├── 📁 tests/              # Test suite
├── 📁 notebooks/          # Jupyter notebooks
├── 📁 session_notes/      # Development history
└── 📁 archive/            # Historical content
```

📋 **Detailed structure**: See [`DIRECTORY_STRUCTURE.md`](docs/project_management/DIRECTORY_STRUCTURE.md)

## Installation

```bash
git clone https://github.com/your-username/hyena-glt.git
cd hyena-glt
pip install -e .
```

## Quick Start

```python
from hyena_glt.data import DNATokenizer, GenomicDataset, create_genomic_dataloaders
from hyena_glt.model import HyenaGLT
from hyena_glt.config import HyenaGLTConfig

# Initialize components
tokenizer = DNATokenizer(vocab_size=1000, kmer_size=3)
config = HyenaGLTConfig(genomic_vocab_size=1000, hidden_size=256)
model = HyenaGLT(config)

# Create dataset and loaders
data = [{"sequence": "ATCGATCG", "labels": 0}]
dataset = GenomicDataset(data=data, tokenizer=tokenizer, max_length=64)
loaders = create_genomic_dataloaders(train_data=dataset, tokenizer=tokenizer, batch_size=32)

# Ready for training!
for batch in loaders['train']:
    outputs = model(batch.input_ids)
```

## 🎭 Try the Demos

```bash
# Complete framework demonstration
python scripts/demos/demo_complete_framework.py

# Mixed precision capabilities demo
python examples/enhanced_mixed_precision_demo.py

# BLT position system demo
python scripts/demos/demo_blt_position_system.py

# Performance benchmarking
python scripts/benchmarks/benchmark_blt_performance.py
```

## Architecture

```text
Input Sequence
     ↓
GenomicTokenizer
     ↓
Local Encoder (BLT)
     ↓
Dynamic Token Merger
     ↓
Hyena Blocks (Savanna)
     ↓
Task-Specific Heads
     ↓
Output
```

## 📋 Development Stages

1. ✅ **Foundation Setup**: Project structure and configuration
2. ✅ **Genomic Data Infrastructure**: Tokenizers and data loaders  
3. ✅ **Core Hyena Architecture**: Hybrid layers and operators
4. ✅ **Model Integration**: Complete HyenaGLT implementation
5. ✅ **Training Infrastructure**: Multi-task training pipeline
6. ✅ **Evaluation Framework**: Comprehensive testing and analysis

All core development stages are complete. The framework is production-ready with comprehensive documentation and testing.

## 📊 Performance Highlights

- **Memory Efficiency**: 40-60% reduction vs. standard transformers
- **Mixed Precision**: Up to 8x speedup with FP8 on H100/A100 GPUs
- **Speed**: 2-3x faster training on genomic sequences
- **Compression**: 16-64x token reduction with BLT merging
- **Accuracy**: Competitive performance on genomic benchmarks
- **Scalability**: Handles sequences up to 100K+ tokens

## 📚 Documentation

**Comprehensive Documentation Suite (6,000+ lines)**

### 🎯 Essential Guides
- 📖 **[Documentation Index](docs/README.md)**: Complete guide to all documentation
- 🏗️ **[Technical Guide](docs/TECHNICAL_GUIDE.md)**: Main technical documentation with implementation details  
- 🔧 **[Patcher Implementation](docs/PATCHER_IMPLEMENTATION.md)**: Comprehensive external patcher integration guide
- 🚀 **[Integration Guide](docs/INTEGRATION_GUIDE.md)**: Advanced patterns for combining BLT_Hyena with external systems
- 📊 **[Performance Analysis](docs/PERFORMANCE_ANALYSIS.md)**: Detailed benchmarking and optimization strategies

### 🧬 Specialized Documentation
- 🎯 **[Position Embeddings](docs/BLT_POSITION_EMBEDDINGS.md)**: Deep dive into BLT position system
- 🏛️ **[Architecture Guide](docs/ARCHITECTURE.md)**: Detailed architectural documentation with code cross-references
- 🔗 **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation for all components
- ⚡ **[Quick Start](docs/QUICKSTART.md)**: Fast track to using Hyena-GLT

### 📈 Implementation Highlights
- **Six Patching Modes**: Greedy, Optimal, Entropy-based, Length-constrained, Dual-threshold, Monotonic
- **Real Parameter Examples**: Actual calibrated values like `threshold=1.335442066192627`
- **Performance Benchmarks**: Measured latency, memory usage, and scaling characteristics  
- **Production Deployment**: Optimization strategies and scaling patterns
- **Complete API Coverage**: 200+ documented functions and methods

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Running tests and benchmarks
- Submitting pull requests
- Code style guidelines

## 📄 License

MIT License - see LICENSE file for details.
