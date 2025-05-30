# 🧬 Hyena-GLT Training & Interpretability Guide

**Purpose**: Comprehensive guide for enhanced training pipelines and model interpretability  
**Created**: 2025-01-20  
**Dependencies**: PyTorch, Transformers, Matplotlib, Seaborn, NumPy  

---

## 🚀 Quick Start

### Basic Training
```python
from examples.streamlined_training_examples import run_basic_dna_classification

# Simple DNA classification
results = run_basic_dna_classification(
    sequences=["ATCGATCG", "GCTAGCTA"],
    labels=[0, 1],
    num_epochs=10
)
```

### Advanced Multi-Modal Training
```python
from examples.enhanced_training_pipeline import EnhancedTrainingPipeline

# Multi-modal genomic learning
pipeline = EnhancedTrainingPipeline()
results = pipeline.run_multi_modal_training(
    dna_data=dna_sequences,
    rna_data=rna_sequences,
    protein_data=protein_sequences
)
```

### Model Interpretability
```python
from hyena_glt.interpretability import ModelInterpreter

# Comprehensive model analysis
interpreter = ModelInterpreter(model, tokenizer)
analysis = interpreter.analyze_comprehensive(sequences)
interpreter.generate_report(analysis, save_path="analysis_report.html")
```

---

## 📋 Training Pipeline Features

### 🎯 Enhanced Training Pipeline
**Location**: `examples/enhanced_training_pipeline.py`

#### Key Capabilities
- **Multi-Modal Learning**: DNA, RNA, and protein sequence integration
- **Real-Time Monitoring**: Live metrics visualization and logging
- **Curriculum Learning**: Progressive training strategies
- **Performance Profiling**: Resource usage and optimization insights
- **Attention Analysis**: Built-in interpretability during training

#### Configuration Example
```python
config = {
    'multi_modal': {
        'dna': {'weight': 0.4, 'max_length': 1024},
        'rna': {'weight': 0.3, 'max_length': 512},
        'protein': {'weight': 0.3, 'max_length': 256}
    },
    'curriculum_learning': {
        'strategy': 'length_based',
        'start_length': 128,
        'max_length': 1024,
        'progression_steps': [128, 256, 512, 1024]
    },
    'monitoring': {
        'log_interval': 100,
        'save_checkpoints': True,
        'visualize_attention': True
    }
}
```

### 🎓 Streamlined Training Examples
**Location**: `examples/streamlined_training_examples.py`

#### Progressive Examples
1. **Basic DNA Classification**: Simple binary classification workflow
2. **Advanced Configuration**: Curriculum learning and monitoring
3. **Protein Function Prediction**: Multi-class prediction with custom metrics
4. **Multi-Task Learning**: Simultaneous multiple task training

#### Usage Patterns
```python
# Example 1: Basic workflow
results = run_basic_dna_classification(sequences, labels)

# Example 2: Advanced features
results = run_advanced_training_example(
    data, 
    use_curriculum=True,
    monitor_attention=True
)

# Example 3: Protein tasks
results = run_protein_function_prediction(
    protein_sequences,
    function_labels,
    num_classes=10
)

# Example 4: Multi-task
results = run_multi_task_example(
    sequences,
    classification_labels,
    regression_targets
)
```

---

## 🔍 Interpretability Framework

### 🧠 Core Components
**Location**: `hyena_glt/interpretability/`

#### ModelInterpreter (Main Interface)
- **Purpose**: Unified interface for all interpretability analyses
- **Features**: Batch processing, report generation, visualization
- **Usage**: One-stop shop for model interpretation

```python
from hyena_glt.interpretability import ModelInterpreter

interpreter = ModelInterpreter(model, tokenizer)

# Single sequence analysis
results = interpreter.analyze_sequence("ATCGATCG")

# Batch analysis
batch_results = interpreter.analyze_batch(sequences)

# Comprehensive analysis with all tools
full_analysis = interpreter.analyze_comprehensive(
    sequences,
    include_attention=True,
    include_gradients=True,
    include_motifs=True
)
```

#### AttentionAnalyzer
- **Purpose**: Attention pattern visualization and analysis
- **Features**: Heatmaps, pattern extraction, genomic annotation
- **Hyena-Specific**: Convolution pattern analysis

```python
from hyena_glt.interpretability import AttentionAnalyzer

analyzer = AttentionAnalyzer(model)
attention_results = analyzer.analyze(sequences)

# Visualize attention patterns
analyzer.plot_attention_heatmap(attention_results[0])
analyzer.plot_attention_summary(attention_results)
```

#### GradientAnalyzer
- **Purpose**: Gradient-based feature importance analysis
- **Methods**: Vanilla, Integrated, Guided backpropagation
- **Features**: Per-position importance, motif highlighting

```python
from hyena_glt.interpretability import GradientAnalyzer

analyzer = GradientAnalyzer(model)

# Different gradient methods
vanilla_grads = analyzer.vanilla_gradients(sequences, labels)
integrated_grads = analyzer.integrated_gradients(sequences, labels)
guided_grads = analyzer.guided_backprop(sequences, labels)

# Visualization
analyzer.plot_importance_scores(integrated_grads[0])
```

#### GenomicMotifAnalyzer
- **Purpose**: Motif discovery and consensus analysis
- **Features**: Pattern identification, conservation analysis
- **Output**: Motif logos, consensus sequences, significance scores

```python
from hyena_glt.interpretability import GenomicMotifAnalyzer

analyzer = GenomicMotifAnalyzer(model, tokenizer)
motif_results = analyzer.discover_motifs(sequences)

# Generate motif logos
analyzer.create_motif_logo(motif_results['top_motifs'][0])
analyzer.plot_motif_conservation(motif_results)
```

### 🔬 Hyena-Specific Analysis
**Location**: `hyena_glt/interpretability/attention_analysis.py`

#### HyenaAttentionAnalyzer
- **Purpose**: Specialized analysis for Hyena convolution patterns
- **Features**: Convolution pattern extraction, positional analysis
- **Benefits**: Hyena-optimized interpretability

```python
from hyena_glt.interpretability.attention_analysis import HyenaAttentionAnalyzer

analyzer = HyenaAttentionAnalyzer(model)

# Hyena-specific analysis
patterns = analyzer.extract_convolution_patterns(sequences)
positional_analysis = analyzer.analyze_positional_patterns(sequences)

# Genomic feature annotation
annotated_results = analyzer.annotate_genomic_features(
    sequences, 
    patterns,
    feature_annotations=gene_annotations
)
```

---

## 🔧 Configuration and Customization

### Training Configuration
```python
# Enhanced Training Pipeline Config
enhanced_config = {
    'model': {
        'name': 'hyena-glt-base',
        'config_overrides': {
            'max_position_embeddings': 2048,
            'num_attention_heads': 16
        }
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 2e-4,
        'num_epochs': 100,
        'warmup_steps': 1000
    },
    'multi_modal': {
        'enabled': True,
        'data_sources': ['dna', 'rna', 'protein'],
        'balancing_strategy': 'weighted'
    },
    'curriculum_learning': {
        'enabled': True,
        'strategy': 'difficulty_based',
        'progression_metric': 'loss'
    },
    'monitoring': {
        'wandb_project': 'hyena-glt-training',
        'log_frequency': 100,
        'save_frequency': 1000
    }
}
```

### Interpretability Configuration
```python
# Interpretability Analysis Config
interpretability_config = {
    'attention': {
        'layer_range': (0, -1),  # All layers
        'head_range': (0, -1),   # All heads
        'aggregation': 'mean'    # mean, max, or per_head
    },
    'gradients': {
        'methods': ['vanilla', 'integrated', 'guided'],
        'baseline_strategy': 'zero',  # zero, random, or blur
        'steps': 50  # For integrated gradients
    },
    'motifs': {
        'min_length': 6,
        'max_length': 15,
        'significance_threshold': 0.01,
        'max_motifs': 20
    },
    'visualization': {
        'save_plots': True,
        'plot_format': 'png',
        'dpi': 300,
        'color_scheme': 'genomic'
    }
}
```

---

## 📊 Analysis Outputs

### Training Metrics
- **Loss Curves**: Training and validation loss over time
- **Performance Metrics**: Accuracy, F1, precision, recall
- **Resource Usage**: GPU memory, training speed, efficiency
- **Attention Patterns**: Evolution during training

### Interpretability Results
- **Attention Heatmaps**: Per-layer, per-head attention visualization
- **Importance Scores**: Position-wise feature importance
- **Motif Discoveries**: Identified genomic patterns and logos
- **Pattern Analysis**: Hyena convolution pattern characterization

### Report Generation
```python
# Generate comprehensive HTML report
interpreter.generate_report(
    analysis_results,
    save_path="model_analysis_report.html",
    include_plots=True,
    include_statistics=True
)

# Generate summary statistics
summary = interpreter.generate_summary(analysis_results)
print(f"Top motifs: {summary['motifs']['count']}")
print(f"Attention complexity: {summary['attention']['avg_entropy']}")
print(f"Feature importance: {summary['gradients']['top_positions']}")
```

---

## 🚨 Best Practices

### Training Guidelines
1. **Start Simple**: Begin with basic examples before advanced features
2. **Monitor Resources**: Use profiling to optimize memory and speed
3. **Curriculum Learning**: Gradually increase sequence complexity
4. **Multi-Modal Balance**: Adjust data source weights based on task
5. **Regular Checkpointing**: Save models frequently during long training

### Interpretability Guidelines
1. **Multiple Methods**: Use different analysis methods for validation
2. **Batch Analysis**: Process multiple sequences for robust patterns
3. **Statistical Significance**: Validate motif discoveries with proper statistics
4. **Visualization**: Always visualize results for intuitive understanding
5. **Domain Knowledge**: Incorporate biological knowledge in interpretation

### Performance Optimization
1. **Gradient Checkpointing**: Enable for memory efficiency with long sequences
2. **Mixed Precision**: Use automatic mixed precision for speed
3. **Batch Size Tuning**: Optimize batch size for your hardware
4. **Attention Caching**: Cache attention patterns for repeated analysis
5. **Lazy Loading**: Load large datasets incrementally

---

## 🔗 Integration with Existing Framework

### Compatibility
- **Full Compatibility**: Works with existing `HyenaGLTTrainer`
- **Data Pipeline**: Uses established tokenizers and datasets
- **Configuration**: Extends `HyenaGLTConfig` seamlessly
- **Utilities**: Leverages existing visualization and monitoring tools

### Migration Path
1. **Existing Training**: Continue using current trainer while exploring enhanced features
2. **Gradual Adoption**: Add interpretability tools to existing workflows
3. **Advanced Features**: Migrate to enhanced pipeline for new projects
4. **Custom Integration**: Use modular components in custom training loops

---

## 📚 Examples and Tutorials

### Complete Workflow Example
```python
from examples.enhanced_training_pipeline import EnhancedTrainingPipeline
from hyena_glt.interpretability import ModelInterpreter

# 1. Enhanced Training
pipeline = EnhancedTrainingPipeline()
trained_model = pipeline.run_multi_modal_training(
    dna_data=dna_sequences,
    rna_data=rna_sequences,
    protein_data=protein_sequences,
    config=enhanced_config
)

# 2. Model Interpretation
interpreter = ModelInterpreter(trained_model, pipeline.tokenizer)
analysis = interpreter.analyze_comprehensive(test_sequences)

# 3. Report Generation
interpreter.generate_report(analysis, "analysis_report.html")

# 4. Pattern Discovery
motifs = analysis['motifs']['discovered_patterns']
attention_patterns = analysis['attention']['layer_patterns']
important_positions = analysis['gradients']['top_positions']

print(f"Discovered {len(motifs)} significant motifs")
print(f"Attention complexity: {attention_patterns['avg_entropy']}")
print(f"Most important positions: {important_positions[:10]}")
```

### Custom Analysis Pipeline
```python
# Custom interpretability workflow
from hyena_glt.interpretability import (
    AttentionAnalyzer, 
    GradientAnalyzer, 
    GenomicMotifAnalyzer
)

# Initialize analyzers
attention_analyzer = AttentionAnalyzer(model)
gradient_analyzer = GradientAnalyzer(model)
motif_analyzer = GenomicMotifAnalyzer(model, tokenizer)

# Custom analysis sequence
sequences = load_genomic_sequences("data/test_sequences.fasta")
labels = load_labels("data/test_labels.csv")

# 1. Attention analysis
attention_results = attention_analyzer.analyze(sequences)
attention_analyzer.plot_attention_heatmap(attention_results[0])

# 2. Gradient analysis
gradient_results = gradient_analyzer.integrated_gradients(sequences, labels)
gradient_analyzer.plot_importance_scores(gradient_results[0])

# 3. Motif discovery
motif_results = motif_analyzer.discover_motifs(sequences)
motif_analyzer.create_motif_logo(motif_results['top_motifs'][0])

# 4. Combined visualization
create_combined_analysis_plot(
    attention_results, 
    gradient_results, 
    motif_results
)
```

This guide provides comprehensive documentation for the enhanced training capabilities and interpretability framework, making it easy for users to leverage these advanced features in their genomic modeling workflows.
