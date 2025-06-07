# Working Examples

This directory contains complete, runnable examples that demonstrate BLT_Hyena functionality.

## Basic Examples

### [Quick Start Example](quick_start.py)
Simple 5-minute demo showing basic model usage
```bash
python quick_start.py
```

### [Configuration Example](configuration.py)
Different ways to configure BLT_Hyena models
```bash
python configuration.py
```

## Data Processing Examples

### [FASTA Processing](data_processing/fasta_processing.py)
Complete pipeline for processing genomic FASTA files

### [Dataset Creation](data_processing/dataset_creation.py)
Creating custom datasets for training

### [Data Augmentation](data_processing/data_augmentation.py)
Genomic-specific data augmentation techniques

## Training Examples

### [Basic Training](training/basic_training.py)
Simple training loop with BLT_Hyena

### [Advanced Training](training/advanced_training.py)
Advanced training with custom optimizers and schedulers

### [Distributed Training](training/distributed_training.py)
Multi-GPU training setup

### [Transfer Learning](training/transfer_learning.py)
Fine-tuning pre-trained models

## Evaluation Examples

### [Model Evaluation](evaluation/model_evaluation.py)
Comprehensive model evaluation pipeline

### [Benchmarking](evaluation/benchmarking.py)
Performance benchmarking against baselines

### [Statistical Analysis](evaluation/statistical_analysis.py)
Statistical significance testing of results

## Production Examples

### [Model Serving](production/model_serving.py)
FastAPI service for model deployment

### [Batch Processing](production/batch_processing.py)
Large-scale batch inference

### [Docker Deployment](production/docker/)
Complete Docker setup for production

### [Kubernetes Deployment](production/kubernetes/)
Kubernetes manifests for scalable deployment

## Research Examples

### [Multi-Modal Learning](research/multimodal_learning.py)
Combining sequence and structural data

### [Meta-Learning](research/meta_learning.py)
Few-shot learning for genomic tasks

### [Neural Architecture Search](research/nas_example.py)
Automated architecture optimization

### [Interpretability](research/interpretability.py)
Model interpretation and visualization

## Notebook Examples

### [Jupyter Notebooks](notebooks/)
Interactive examples and tutorials
- [Getting Started](notebooks/01_getting_started.ipynb)
- [Data Exploration](notebooks/02_data_exploration.ipynb)
- [Model Training](notebooks/03_model_training.ipynb)
- [Results Analysis](notebooks/04_results_analysis.ipynb)

## Usage Instructions

1. **Set up environment:**
```bash
cd examples/
pip install -r requirements.txt
```

2. **Download sample data:**
```bash
python download_sample_data.py
```

3. **Run examples:**
```bash
# Basic example
python quick_start.py

# Training example
python training/basic_training.py

# Production example
python production/model_serving.py
```

## Example Data

Sample datasets are provided for testing:
- `data/sample_sequences.fasta` - Small genomic sequences
- `data/sample_labels.csv` - Corresponding labels
- `data/sample_structure.pkl` - Structural features

## Contributing Examples

To contribute a new example:

1. Create a new Python file in the appropriate directory
2. Include comprehensive comments and docstrings
3. Add requirements to the directory's `requirements.txt`
4. Test the example on clean environment
5. Submit pull request with example and documentation

### Example Template

```python
#!/usr/bin/env python3
"""
BLT_Hyena Example: [Title]

Description: Brief description of what this example demonstrates

Requirements:
- torch>=1.12.0
- transformers>=4.20.0
- hyena_glt

Usage:
    python example_name.py [optional arguments]

Author: Your Name
Date: YYYY-MM-DD
"""

import torch
from hyena_glt import HyenaGLT, HyenaGLTConfig

def main():
    """Main example function."""
    print("Starting BLT_Hyena example...")
    
    # Your example code here
    
    print("Example completed successfully!")

if __name__ == "__main__":
    main()
```
