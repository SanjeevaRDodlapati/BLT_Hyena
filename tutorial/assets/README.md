# Assets and Resources

This directory contains assets, diagrams, and resources for the BLT_Hyena tutorial system.

## Directory Structure

```
assets/
├── images/           # Diagrams and illustrations
├── data/            # Sample datasets
├── configs/         # Configuration files
├── scripts/         # Utility scripts
└── templates/       # Code templates
```

## Images and Diagrams

### Architecture Diagrams
- `hyena_architecture.png` - BLT_Hyena model architecture overview
- `attention_vs_hyena.png` - Comparison between attention and Hyena mechanisms
- `data_pipeline.png` - Data processing pipeline diagram
- `training_workflow.png` - Training process flowchart

### Performance Plots
- `performance_comparison.png` - Benchmark results
- `scaling_analysis.png` - Model scaling characteristics
- `memory_usage.png` - Memory usage analysis

### Tutorial Illustrations
- `quick_start_flow.png` - Quick start tutorial workflow
- `production_deployment.png` - Production deployment architecture

## Sample Data

### Genomic Sequences
- `sample_sequences.fasta` - Small sample of genomic sequences (10KB)
- `sample_large.fasta` - Larger sample dataset (100KB)
- `sample_labels.csv` - Corresponding labels for sequences
- `sample_metadata.json` - Metadata for sequences

### Synthetic Data
- `synthetic_dna.fasta` - Computer-generated DNA sequences for testing
- `synthetic_rna.fasta` - Computer-generated RNA sequences
- `benchmark_data.tar.gz` - Standardized benchmark datasets

## Configuration Files

### Model Configurations
- `small_model.yaml` - Configuration for small/fast model
- `base_model.yaml` - Standard configuration
- `large_model.yaml` - Large model configuration
- `research_config.yaml` - Research-oriented configuration

### Training Configurations
- `quick_training.yaml` - Fast training for demos
- `production_training.yaml` - Production-ready training setup
- `distributed_training.yaml` - Multi-GPU training configuration

### Deployment Configurations
- `docker_config.yaml` - Docker deployment settings
- `kubernetes_config.yaml` - Kubernetes deployment manifests
- `monitoring_config.yaml` - Monitoring and logging setup

## Utility Scripts

### Data Processing
- `prepare_data.py` - Data preprocessing utilities
- `validate_data.py` - Data validation and quality checks
- `download_datasets.py` - Download public genomic datasets

### Model Utilities
- `convert_checkpoint.py` - Convert between model formats
- `benchmark_model.py` - Performance benchmarking
- `analyze_model.py` - Model analysis and statistics

### Deployment Scripts
- `setup_environment.sh` - Environment setup script
- `deploy_model.sh` - Model deployment automation
- `health_check.py` - Service health monitoring

## Code Templates

### Basic Templates
- `model_template.py` - Basic model implementation template
- `training_template.py` - Training script template
- `evaluation_template.py` - Evaluation script template

### Advanced Templates
- `custom_layer_template.py` - Custom layer implementation
- `research_experiment_template.py` - Research experiment setup
- `production_service_template.py` - Production API service

### Configuration Templates
- `config_template.yaml` - Base configuration template
- `hyperparameter_template.yaml` - Hyperparameter search template

## Usage

### Accessing Assets in Code

```python
import os
from pathlib import Path

# Get assets directory
assets_dir = Path(__file__).parent.parent / "assets"

# Load sample data
sample_data = assets_dir / "data" / "sample_sequences.fasta"

# Load configuration
config_file = assets_dir / "configs" / "base_model.yaml"

# Load template
template_file = assets_dir / "templates" / "model_template.py"
```

### Using Sample Data

```python
from hyena_glt.data import FASTADataset

# Load sample sequences
dataset = FASTADataset(
    fasta_file="assets/data/sample_sequences.fasta",
    labels_file="assets/data/sample_labels.csv"
)
```

### Loading Configurations

```python
import yaml

with open("assets/configs/base_model.yaml", 'r') as f:
    config = yaml.safe_load(f)

model_config = HyenaGLTConfig(**config['model'])
```

## Contributing Assets

To contribute new assets:

1. **Images/Diagrams:**
   - Use high resolution (300 DPI minimum for print)
   - Save in PNG format for diagrams
   - Include source files when possible (e.g., .svg, .drawio)

2. **Data Files:**
   - Keep files small (<10MB for repository)
   - Use standard formats (FASTA, CSV, JSON)
   - Include metadata and documentation

3. **Scripts:**
   - Follow Python style guidelines
   - Include comprehensive docstrings
   - Add error handling and validation

4. **Templates:**
   - Include clear comments and documentation
   - Use placeholder values that are easy to identify
   - Test templates before contributing

## File Naming Conventions

- Use lowercase with underscores: `model_architecture.png`
- Include version numbers when relevant: `benchmark_v2.csv`
- Use descriptive names: `hyena_vs_transformer_comparison.png`
- Group related files with prefixes: `config_base.yaml`, `config_large.yaml`

## License

All assets are provided under the same license as BLT_Hyena. Please ensure any contributed assets are appropriately licensed and include attribution where required.
