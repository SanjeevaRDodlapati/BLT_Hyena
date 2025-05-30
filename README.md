# Hyena-GLT: Genome Language Transformer

A hybrid model combining BLT's byte latent tokenization with Savanna's Striped Hyena blocks for efficient genomic sequence modeling.

## Overview

Hyena-GLT integrates:
- **BLT's Dynamic Token Merging**: Adaptive tokenization for variable-length genomic sequences
- **Savanna's Hyena Operators**: Efficient long-range convolutions for genomic pattern recognition
- **Genomic Specialization**: Custom positional encodings and task-specific heads

## Features

- 🧬 **Genomic Tokenization**: Specialized tokenizers for DNA, RNA, and protein sequences
- ⚡ **Efficient Architecture**: Hyena convolutions for O(n log n) complexity
- 🔄 **Dynamic Processing**: Adaptive token merging based on sequence content
- 📊 **Multi-task Learning**: Support for classification, generation, and analysis tasks
- 🎯 **Fine-tuning Ready**: Pre-configured for genomic downstream tasks

## Installation

```bash
git clone https://github.com/your-username/hyena-glt.git
cd hyena-glt
pip install -e .
```

## Quick Start

```python
from hyena_glt import HyenaGLT, GenomicTokenizer

# Initialize model and tokenizer
model = HyenaGLT.from_pretrained("hyena-glt-base")
tokenizer = GenomicTokenizer()

# Process genomic sequence
sequence = "ATCGATCGATCG..."
tokens = tokenizer.encode(sequence)
outputs = model(tokens)
```

## Architecture

```
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

## Development Stages

1. ✅ **Foundation Setup**: Project structure and configuration
2. 🔄 **Genomic Data Infrastructure**: Tokenizers and data loaders
3. ⏳ **Core Hyena Architecture**: Hybrid layers and operators
4. ⏳ **Model Integration**: Complete HyenaGLT implementation
5. ⏳ **Training Infrastructure**: Multi-task training pipeline
6. ⏳ **Evaluation Framework**: Comprehensive testing and analysis

## License

MIT License - see LICENSE file for details.
