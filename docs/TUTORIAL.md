# Hyena-GLT Tutorial: Advanced Genomic Modeling

This comprehensive tutorial covers advanced features of Hyena-GLT for genomic sequence modeling, including multi-task learning, optimization techniques, and deployment strategies.

## Table of Contents

1. [Advanced Model Architecture](#advanced-model-architecture)
2. [Multi-Task Learning](#multi-task-learning)
3. [Curriculum Learning](#curriculum-learning)
4. [Model Optimization](#model-optimization)
5. [Deployment Strategies](#deployment-strategies)
6. [Performance Analysis](#performance-analysis)
7. [Real-World Applications](#real-world-applications)

## Advanced Model Architecture

### Custom Hyena Configurations

Learn how to customize the Hyena architecture for specific genomic tasks.

```python
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model import HyenaGLTModel

# Advanced configuration for long-range genomic modeling
config = HyenaGLTConfig(
    # Core architecture
    vocab_size=8192,
    d_model=768,
    n_layers=16,
    n_heads=12,
    
    # Extended context for genomic sequences
    sequence_length=16384,  # 16K context
    
    # Hyena-specific optimizations
    hyena_order=3,          # Higher order for complex patterns
    conv_kernel_size=7,     # Larger kernels for genomic motifs
    filter_order=128,       # More complex filters
    
    # Dynamic merging for efficiency
    use_dynamic_merging=True,
    merge_ratio=0.6,        # Aggressive merging for long sequences
    merge_threshold=0.15,   # Fine-tuned threshold
    
    # Memory optimizations
    gradient_checkpointing=True,
    use_flash_attention=True,
)

model = HyenaGLTModel(config)
```

### Specialized Genomic Layers

```python
from hyena_glt.model.layers import GenomicConvolutionLayer, MotifDetectionLayer

class CustomGenomicModel(HyenaGLTModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Add genomic-specific layers
        self.motif_detector = MotifDetectionLayer(
            d_model=config.d_model,
            num_motifs=256,
            motif_length=10
        )
        
        self.genomic_conv = GenomicConvolutionLayer(
            d_model=config.d_model,
            kernel_sizes=[3, 5, 7, 11],  # Multi-scale convolutions
            dilation_rates=[1, 2, 4, 8]  # Dilated convolutions
        )
```

## Multi-Task Learning

### Setting Up Multi-Task Training

Train a single model on multiple genomic tasks simultaneously.

```python
from hyena_glt.training import MultiTaskLearner
from hyena_glt.data import (
    ProteinFunctionDataset, 
    GenomeAnnotationDataset, 
    VariantEffectDataset
)

# Define multiple tasks
tasks = {
    "protein_function": {
        "dataset": ProteinFunctionDataset("protein_data.csv"),
        "num_classes": 10,
        "weight": 0.4
    },
    "genome_annotation": {
        "dataset": GenomeAnnotationDataset("annotation_data.csv"),
        "num_classes": 20,
        "weight": 0.4
    },
    "variant_effect": {
        "dataset": VariantEffectDataset("variant_data.csv"),
        "num_classes": 3,
        "weight": 0.2
    }
}

# Initialize multi-task learner
learner = MultiTaskLearner(
    base_model=model,
    tasks=tasks,
    shared_layers=12,      # Share first 12 layers
    task_specific_layers=4  # 4 task-specific layers each
)

# Train with adaptive task weighting
learner.train(
    num_epochs=50,
    adaptive_weighting=True,
    temperature=2.0
)
```

### Advanced Multi-Task Strategies

```python
# Gradient balancing for multi-task learning
from hyena_glt.training.multitask import GradientBalancer

balancer = GradientBalancer(
    method="gradnorm",      # Options: "equal", "gradnorm", "pcgrad"
    alpha=0.12,             # GradNorm hyperparameter
    normalize_gradients=True
)

learner = MultiTaskLearner(
    base_model=model,
    tasks=tasks,
    gradient_balancer=balancer
)
```

## Curriculum Learning

### Sequence Length Curriculum

Gradually increase sequence length during training for better convergence.

```python
from hyena_glt.training import CurriculumLearner

# Define curriculum stages
curriculum_stages = [
    {
        "stage_name": "short_sequences",
        "max_length": 512,
        "epochs": 10,
        "learning_rate": 1e-3,
        "batch_size": 64
    },
    {
        "stage_name": "medium_sequences", 
        "max_length": 2048,
        "epochs": 15,
        "learning_rate": 5e-4,
        "batch_size": 32
    },
    {
        "stage_name": "long_sequences",
        "max_length": 8192,
        "epochs": 25,
        "learning_rate": 1e-4,
        "batch_size": 16
    }
]

curriculum = CurriculumLearner(
    model=model,
    stages=curriculum_stages,
    transition_strategy="gradual",  # Options: "sudden", "gradual"
    overlap_ratio=0.2               # 20% overlap between stages
)

# Train with curriculum
curriculum.train(dataset=train_dataset)
```

### Complexity-Based Curriculum

```python
from hyena_glt.training.curriculum import ComplexityCurriculum

# Sort data by complexity (sequence length, GC content, etc.)
complexity_curriculum = ComplexityCurriculum(
    dataset=train_dataset,
    complexity_metrics=[
        "sequence_length",
        "gc_content", 
        "repeat_content",
        "entropy"
    ],
    stages=5,
    progression="exponential"  # How fast to increase complexity
)

complexity_curriculum.train(model, num_epochs=30)
```

## Model Optimization

### Quantization for Deployment

```python
from hyena_glt.optimization import (
    ModelQuantizer, 
    QuantizationConfig,
    QATTrainer
)

# Post-training quantization
quant_config = QuantizationConfig(
    quantization_type="static",     # More aggressive than dynamic
    dtype=torch.qint8,
    calibration_dataset_size=1000,
    percentile=99.9
)

quantizer = ModelQuantizer(quant_config)

# Calibrate on representative data
quantizer.calibrate(model, calibration_dataloader)

# Quantize the model
quantized_model = quantizer.quantize(model)

# Evaluate quantization impact
accuracy_drop = quantizer.evaluate_quantization(
    original_model=model,
    quantized_model=quantized_model,
    test_dataloader=test_dataloader
)

print(f"Accuracy drop from quantization: {accuracy_drop:.2%}")
```

### Quantization-Aware Training (QAT)

```python
# Train with quantization in mind
qat_trainer = QATTrainer(
    model=model,
    config=quant_config,
    num_epochs=10  # Fine-tune with fake quantization
)

qat_model = qat_trainer.train(train_dataloader, val_dataloader)
final_quantized = qat_trainer.convert_to_quantized(qat_model)
```

### Model Pruning

```python
from hyena_glt.optimization import ModelPruner, PruningConfig

# Structured pruning (remove entire neurons/channels)
pruning_config = PruningConfig(
    pruning_type="structured",
    sparsity=0.5,               # Remove 50% of parameters
    structured_type="channel",   # Prune entire channels
    importance_metric="magnitude" # How to measure importance
)

pruner = ModelPruner(pruning_config)

# Gradual pruning during training
pruned_model = pruner.prune_gradually(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    num_epochs=20,
    pruning_schedule="polynomial"  # Gradual sparsity increase
)
```

### Knowledge Distillation

```python
from hyena_glt.optimization import KnowledgeDistiller, DistillationConfig

# Create a smaller student model
student_config = config.copy()
student_config.d_model = 256
student_config.n_layers = 8

student_model = HyenaGLTModel(student_config)

# Distillation configuration
distill_config = DistillationConfig(
    temperature=4.0,
    alpha=0.7,                    # Weight for distillation loss
    beta=0.3,                     # Weight for student loss
    distillation_type="comprehensive"  # Feature + attention + response
)

distiller = KnowledgeDistiller(distill_config)

# Distill knowledge from teacher to student
distilled_student = distiller.distill(
    teacher_model=model,          # Large model
    student_model=student_model,  # Small model
    train_dataloader=train_dataloader,
    num_epochs=15
)

# Compare performance and efficiency
teacher_params = sum(p.numel() for p in model.parameters())
student_params = sum(p.numel() for p in distilled_student.parameters())
compression_ratio = teacher_params / student_params

print(f"Compression ratio: {compression_ratio:.2f}x")
```

## Deployment Strategies

### ONNX Export for Cross-Platform Deployment

```python
from hyena_glt.optimization import ONNXExporter, DeploymentConfig

deploy_config = DeploymentConfig(
    export_format="onnx",
    optimization_level="all",
    target_device="cpu",          # or "gpu", "mobile"
    input_shapes={"input_ids": [1, 2048]}  # Static shapes for optimization
)

exporter = ONNXExporter(deploy_config)

# Export to ONNX
onnx_model = exporter.export(
    model=optimized_model,
    output_path="hyena_glt_model.onnx",
    opset_version=11
)

# Validate ONNX model
exporter.validate_export(
    original_model=model,
    onnx_path="hyena_glt_model.onnx",
    test_inputs=sample_inputs
)
```

### TensorRT Optimization (GPU Deployment)

```python
from hyena_glt.optimization import TensorRTOptimizer

# Optimize for NVIDIA GPUs
tensorrt_optimizer = TensorRTOptimizer(
    precision="fp16",            # Mixed precision
    max_workspace_size=1024**3,  # 1GB workspace
    max_batch_size=32
)

# Build TensorRT engine
trt_engine = tensorrt_optimizer.build_engine(
    onnx_path="hyena_glt_model.onnx",
    engine_path="hyena_glt_model.trt"
)

# Benchmark TensorRT performance
trt_benchmark = tensorrt_optimizer.benchmark(
    engine_path="hyena_glt_model.trt",
    test_data=benchmark_data
)

print(f"TensorRT speedup: {trt_benchmark['speedup']:.2f}x")
```

### Mobile Deployment with TorchScript

```python
import torch

# Prepare model for mobile deployment
model.eval()

# Trace the model
example_input = torch.randint(0, config.vocab_size, (1, 512))
traced_model = torch.jit.trace(model, example_input)

# Optimize for mobile
mobile_model = torch.jit.optimize_for_inference(traced_model)

# Save for mobile deployment
mobile_model.save("hyena_glt_mobile.pt")

# Test mobile model
mobile_output = mobile_model(example_input)
print(f"Mobile model output shape: {mobile_output.shape}")
```

## Performance Analysis

### Comprehensive Benchmarking

```python
from hyena_glt.evaluation import PerformanceBenchmark, ModelProfiler

# Profile model performance
profiler = ModelProfiler()

# Memory profiling
memory_stats = profiler.profile_memory(
    model=model,
    input_sizes=[(1, 512), (1, 1024), (1, 2048), (1, 4096)],
    device="cuda"
)

# Speed profiling
speed_stats = profiler.profile_speed(
    model=model,
    input_sizes=[(1, 512), (8, 512), (16, 512), (32, 512)],
    num_runs=100,
    warmup_runs=10
)

# Generate performance report
profiler.generate_report(
    memory_stats=memory_stats,
    speed_stats=speed_stats,
    output_path="performance_report.html"
)
```

### Attention Pattern Analysis

```python
from hyena_glt.evaluation import AttentionAnalyzer

analyzer = AttentionAnalyzer(model)

# Analyze attention patterns on genomic sequences
sequences = [
    "ATGCGATCGATCGATCGATCGTAG",  # Gene sequence
    "AAAAAAAAAAAAAAAAAAAAAA",    # Low complexity
    "ATCGATCGATCGATCGATCGAT"     # Repetitive
]

attention_analysis = analyzer.analyze_patterns(
    sequences=sequences,
    layer_range=(8, 16),  # Analyze layers 8-16
    head_range=None       # All attention heads
)

# Visualize attention patterns
analyzer.plot_attention_maps(
    attention_analysis,
    save_path="attention_patterns.png"
)

# Find important genomic motifs
motifs = analyzer.extract_motifs(
    attention_analysis,
    threshold=0.8,        # High attention threshold
    min_length=6,         # Minimum motif length
    max_length=15         # Maximum motif length
)

print(f"Discovered {len(motifs)} important motifs")
```

### Model Interpretability

```python
from hyena_glt.evaluation import ModelInterpreter

interpreter = ModelInterpreter(model)

# Feature importance analysis
importance_scores = interpreter.compute_feature_importance(
    sequences=test_sequences,
    method="integrated_gradients",  # or "lime", "shap"
    target_class=1
)

# Generate saliency maps
saliency_maps = interpreter.generate_saliency_maps(
    sequences=test_sequences,
    method="guided_backprop"
)

# Visualize important regions
interpreter.visualize_importance(
    sequences=test_sequences,
    importance_scores=importance_scores,
    save_path="feature_importance.html"
)
```

## Real-World Applications

### Gene Expression Prediction

```python
# Configure for gene expression prediction
expression_config = HyenaGLTConfig(
    vocab_size=4096,
    d_model=768,
    n_layers=12,
    sequence_length=8192,    # Long promoter regions
    task_type="regression",   # Continuous expression values
    output_dim=1
)

expression_model = HyenaGLTForRegression(expression_config)

# Train on promoter sequences and expression data
trainer = HyenaGLTTrainer(
    model=expression_model,
    config=expression_config,
    train_dataset=expression_dataset,
    loss_function="mse_loss"
)

trainer.train()

# Predict expression for new sequences
predictions = trainer.predict(new_promoter_sequences)
```

### Variant Effect Prediction

```python
from hyena_glt.applications import VariantEffectPredictor

# Specialized model for variant effects
vep_model = VariantEffectPredictor(
    base_model=model,
    variant_types=["snp", "indel", "cnv"],
    effect_categories=["benign", "likely_benign", "uncertain", 
                      "likely_pathogenic", "pathogenic"]
)

# Analyze variant effects
variant_data = {
    "chromosome": "chr17",
    "position": 43094692,
    "ref_allele": "G",
    "alt_allele": "A",
    "context_sequence": "ATCGATCG...GATCGATC"  # 4KB context
}

effect_prediction = vep_model.predict_effect(variant_data)
print(f"Predicted effect: {effect_prediction['category']}")
print(f"Confidence: {effect_prediction['confidence']:.3f}")
```

### Drug Target Discovery

```python
from hyena_glt.applications import DrugTargetDiscovery

# Multi-modal model for drug-target interactions
drug_target_model = DrugTargetDiscovery(
    protein_model=protein_model,
    compound_model=compound_model,  # Separate model for chemical compounds
    interaction_predictor=interaction_model
)

# Predict drug-target interactions
protein_sequence = "MKTLLLTLLCLVAAYLA..."
compound_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

interaction_score = drug_target_model.predict_interaction(
    protein_sequence=protein_sequence,
    compound_smiles=compound_smiles
)

print(f"Interaction probability: {interaction_score:.3f}")
```

## Advanced Tips and Best Practices

### 1. Memory-Efficient Training

```python
# Enable memory optimizations
config.gradient_checkpointing = True
config.use_flash_attention = True
config.cpu_offload = True

# Use mixed precision training
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group(backend="nccl")

# Wrap model for distributed training
model = DistributedDataParallel(model)

# Use distributed sampler
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

### 3. Hyperparameter Optimization

```python
from hyena_glt.training import HyperparameterOptimizer
import optuna

def objective(trial):
    config.learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config.d_model = trial.suggest_categorical("d_model", [256, 512, 768])
    config.n_layers = trial.suggest_int("n_layers", 4, 16)
    
    trainer = HyenaGLTTrainer(model, config, train_dataset, val_dataset)
    metrics = trainer.train()
    
    return metrics["val_accuracy"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"Best hyperparameters: {study.best_params}")
```

## Conclusion

This tutorial covered advanced features of Hyena-GLT for genomic modeling. Key takeaways:

1. **Architecture Customization**: Adapt the model for specific genomic tasks
2. **Multi-Task Learning**: Train on multiple tasks simultaneously for better generalization
3. **Optimization**: Use quantization, pruning, and distillation for deployment
4. **Analysis**: Understand model behavior through attention and interpretability analysis
5. **Applications**: Apply to real-world genomic problems

For more information, check out:
- API documentation: `docs/API.md`
- Example scripts: `examples/`
- Optimization guide: `docs/OPTIMIZATION.md`

Happy modeling! 🧬🚀
