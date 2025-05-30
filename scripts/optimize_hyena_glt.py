#!/usr/bin/env python3
"""
Comprehensive Model Optimization Script

This script demonstrates how to apply various optimization techniques
to the Hyena-GLT model including quantization, pruning, distillation,
and deployment optimization.
"""

import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
import json

# Import Hyena-GLT components
from hyena_glt.config import HyenaGLTConfig
from hyena_glt.model import HyenaGLT
from hyena_glt.data import create_sample_data

# Import optimization modules
from hyena_glt.optimization import (
    QuantizationConfig, ModelQuantizer, QuantizationBenchmark,
    PruningConfig, ModelPruner,
    DistillationConfig, KnowledgeDistiller, StudentModelFactory,
    DeploymentConfig, ModelOptimizer, ModelProfiler,
    MemoryConfig, MemoryOptimizer, MemoryBenchmark
)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_sample_model_and_data(device='cuda'):
    """Create sample model and data for testing."""
    # Create small model for testing
    config = HyenaGLTConfig(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=4,
        max_length=512
    )
    
    model = HyenaGLT(config).to(device)
    
    # Create sample data
    batch_size = 8
    seq_length = 256
    sample_input = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    sample_target = torch.randint(0, 10, (batch_size,)).to(device)
    
    return model, sample_input, sample_target, config

def run_quantization_optimization(model, sample_input, save_path):
    """Run quantization optimization."""
    print("\n" + "="*50)
    print("QUANTIZATION OPTIMIZATION")
    print("="*50)
    
    # Test different quantization methods
    quantization_methods = ['dynamic', 'static']
    quantized_models = {}
    
    for method in quantization_methods:
        print(f"\nTesting {method} quantization...")
        
        config = QuantizationConfig(
            method=method,
            bits=8,
            backend='fbgemm'
        )
        
        quantizer = ModelQuantizer(config)
        
        # Create dummy calibration data for static quantization
        if method == 'static':
            dataset = torch.utils.data.TensorDataset(
                sample_input.repeat(10, 1, 1),
                torch.randint(0, 10, (80,))
            )
            calibration_loader = torch.utils.data.DataLoader(
                dataset, batch_size=8, shuffle=False
            )
        else:
            calibration_loader = None
        
        try:
            quantized_model = quantizer.quantize_model(
                model,
                calibration_loader=calibration_loader,
                save_path=save_path / f"quantized_{method}"
            )
            quantized_models[method] = quantized_model
            print(f"  ✓ {method} quantization completed")
        except Exception as e:
            print(f"  ✗ {method} quantization failed: {e}")
    
    # Benchmark quantized models
    if quantized_models:
        print("\nBenchmarking quantized models...")
        benchmark = QuantizationBenchmark()
        
        # Create test data
        test_dataset = torch.utils.data.TensorDataset(
            sample_input, torch.randint(0, 10, (sample_input.size(0),))
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
        
        for method, quantized_model in quantized_models.items():
            try:
                results = benchmark.benchmark(
                    model, quantized_model, test_loader,
                    metrics=['latency', 'memory']
                )
                print(f"  {method} quantization benchmark completed")
            except Exception as e:
                print(f"  {method} quantization benchmark failed: {e}")
        
        benchmark.print_results()

def run_pruning_optimization(model, sample_input, save_path):
    """Run pruning optimization."""
    print("\n" + "="*50)
    print("PRUNING OPTIMIZATION")
    print("="*50)
    
    # Test different pruning methods
    pruning_methods = ['magnitude', 'random']
    pruned_models = {}
    
    for method in pruning_methods:
        print(f"\nTesting {method} pruning...")
        
        config = PruningConfig(
            method=method,
            sparsity=0.3,  # 30% sparsity
            global_pruning=True
        )
        
        pruner = ModelPruner(config)
        
        try:
            pruned_model = pruner.prune_model(
                model,
                save_path=save_path / f"pruned_{method}"
            )
            pruned_models[method] = pruned_model
            
            # Get sparsity information
            sparsity_info = pruner.get_sparsity_info(pruned_model)
            print(f"  ✓ {method} pruning completed")
            print(f"    Overall sparsity: {sparsity_info['overall_sparsity']:.2%}")
            print(f"    Total parameters: {sparsity_info['total_params']}")
            print(f"    Pruned parameters: {sparsity_info['pruned_params']}")
        except Exception as e:
            print(f"  ✗ {method} pruning failed: {e}")

def run_distillation_optimization(model, sample_input, sample_target, config, save_path):
    """Run knowledge distillation optimization."""
    print("\n" + "="*50)
    print("KNOWLEDGE DISTILLATION OPTIMIZATION")
    print("="*50)
    
    # Create student model (smaller version)
    print("\nCreating student model...")
    student_model = StudentModelFactory.create_compressed_model(
        config, compression_ratio=0.5
    )
    print(f"Teacher parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # Setup distillation
    distillation_config = DistillationConfig(
        temperature=4.0,
        alpha=0.7,
        beta=0.3,
        epochs=3,  # Reduced for demo
        distillation_type="response"
    )
    
    distiller = KnowledgeDistiller(distillation_config)
    
    # Create training data
    train_dataset = torch.utils.data.TensorDataset(
        sample_input.repeat(5, 1, 1),
        sample_target.repeat(5)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True
    )
    
    try:
        print("\nStarting knowledge distillation...")
        distilled_model = distiller.distill(
            teacher_model=model,
            student_model=student_model,
            train_loader=train_loader,
            save_path=save_path / "distilled"
        )
        print("  ✓ Knowledge distillation completed")
        
        # Benchmark distillation
        print("\nBenchmarking distilled model...")
        from hyena_glt.optimization.distillation import DistillationBenchmark
        
        benchmark = DistillationBenchmark()
        test_dataset = torch.utils.data.TensorDataset(
            sample_input, sample_target
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
        
        results = benchmark.benchmark(
            teacher_model=model,
            student_model=student_model,
            distilled_model=distilled_model,
            test_loader=test_loader
        )
        benchmark.print_results()
        
    except Exception as e:
        print(f"  ✗ Knowledge distillation failed: {e}")

def run_deployment_optimization(model, sample_input, save_path):
    """Run deployment optimization."""
    print("\n" + "="*50)
    print("DEPLOYMENT OPTIMIZATION")
    print("="*50)
    
    # Test deployment optimization
    deployment_config = DeploymentConfig(
        export_format="onnx",
        onnx_optimize=True,
        optimize_for_inference=True,
        device='cpu'  # Use CPU for broader compatibility
    )
    
    optimizer = ModelOptimizer(deployment_config)
    
    try:
        print("\nOptimizing model for deployment...")
        
        # Move to CPU for export
        cpu_model = model.cpu()
        cpu_input = sample_input.cpu()
        
        optimized_models = optimizer.optimize_for_deployment(
            cpu_model,
            cpu_input,
            save_path=save_path / "deployment"
        )
        
        print("  ✓ Deployment optimization completed")
        print(f"    Exported formats: {list(optimized_models.keys())}")
        
        # Profile different formats
        print("\nProfiling optimized models...")
        profiler = ModelProfiler(deployment_config)
        
        # Only profile available formats
        available_formats = {}
        for format_name, model_path in optimized_models.items():
            if Path(model_path).exists():
                available_formats[format_name] = model_path
        
        if available_formats:
            results = profiler.profile_model(
                available_formats,
                cpu_input[:1],  # Single sample for profiling
                num_runs=10
            )
            profiler.print_profiling_results()
        
    except Exception as e:
        print(f"  ✗ Deployment optimization failed: {e}")

def run_memory_optimization(model, sample_input, sample_target):
    """Run memory optimization."""
    print("\n" + "="*50)
    print("MEMORY OPTIMIZATION")
    print("="*50)
    
    # Test different memory optimization configurations
    configs = {
        'baseline': MemoryConfig(
            gradient_checkpointing=False,
            mixed_precision=False,
            aggressive_gc=False
        ),
        'checkpointing': MemoryConfig(
            gradient_checkpointing=True,
            checkpoint_ratio=0.5,
            mixed_precision=False
        ),
        'mixed_precision': MemoryConfig(
            gradient_checkpointing=False,
            mixed_precision=True,
            aggressive_gc=False
        ),
        'full_optimization': MemoryConfig(
            gradient_checkpointing=True,
            checkpoint_ratio=0.5,
            mixed_precision=True,
            aggressive_gc=True,
            adaptive_batch_size=False  # Disable for demo
        )
    }
    
    print("\nBenchmarking memory optimization strategies...")
    benchmark = MemoryBenchmark()
    
    try:
        results = benchmark.benchmark_optimizations(
            base_model=model,
            optimization_configs=configs,
            sample_input=sample_input,
            num_iterations=5  # Reduced for demo
        )
        
        benchmark.print_benchmark_results()
        
    except Exception as e:
        print(f"  ✗ Memory optimization benchmark failed: {e}")

def main():
    """Main optimization demonstration."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Hyena-GLT Model Optimization"
    )
    parser.add_argument(
        '--output', '-o', type=str, default='./optimization_results',
        help='Output directory for optimized models'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for optimization'
    )
    parser.add_argument(
        '--skip', nargs='+', default=[],
        choices=['quantization', 'pruning', 'distillation', 'deployment', 'memory'],
        help='Optimization techniques to skip'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Create output directory
    save_path = Path(args.output)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("HYENA-GLT MODEL OPTIMIZATION DEMONSTRATION")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Output directory: {save_path}")
    
    # Create sample model and data
    print("\nCreating sample model and data...")
    model, sample_input, sample_target, config = create_sample_model_and_data(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Sample input shape: {sample_input.shape}")
    
    # Run optimization techniques
    if 'quantization' not in args.skip:
        run_quantization_optimization(model, sample_input, save_path)
    
    if 'pruning' not in args.skip:
        run_pruning_optimization(model, sample_input, save_path)
    
    if 'distillation' not in args.skip:
        run_distillation_optimization(model, sample_input, sample_target, config, save_path)
    
    if 'deployment' not in args.skip:
        run_deployment_optimization(model, sample_input, save_path)
    
    if 'memory' not in args.skip:
        run_memory_optimization(model, sample_input, sample_target)
    
    print("\n" + "="*60)
    print("OPTIMIZATION DEMONSTRATION COMPLETED")
    print("="*60)
    print(f"Results saved to: {save_path}")
    
    # Save configuration summary
    summary = {
        'model_config': config.__dict__,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'device': args.device,
        'optimization_techniques': [
            technique for technique in 
            ['quantization', 'pruning', 'distillation', 'deployment', 'memory']
            if technique not in args.skip
        ]
    }
    
    with open(save_path / 'optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
