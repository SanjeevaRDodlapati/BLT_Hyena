#!/usr/bin/env python3
"""
Mixed Precision Fine-Tuning Examples for BLT_Hyena

This script demonstrates the enhanced mixed precision capabilities for different
genomic tasks using the BLT_Hyena framework. It showcases task-specific 
optimizations, hardware-aware precision selection, and performance monitoring.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

# Import BLT_Hyena components
try:
    from hyena_glt.training.task_specific import (
        GenomeAnnotationFineTuner,
        VariantEffectFineTuner,
        ProteinFunctionFineTuner,
        GenomeGenerationFineTuner,
        DomainAdaptationFineTuner,
        get_optimal_precision_config,
        apply_task_specific_optimizations,
    )
    from hyena_glt.training.mixed_precision import (
        MixedPrecisionConfig,
        PrecisionMode,
        create_mixed_precision_manager,
    )
except ImportError as e:
    print(f"Warning: Could not import BLT_Hyena components: {e}")
    print("This demo will run with mock implementations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
    # Test data
    batch_size = 4
    seq_len = 1024
    input_ids = torch.randint(0, config.genomic_vocab_size, (batch_size, seq_len))
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    # Test different precision modes
    precision_modes = [
        (PrecisionMode.FP32, "FP32 Baseline"),
        (PrecisionMode.FP16, "FP16 Mixed Precision"),
        (PrecisionMode.BF16, "BF16 Mixed Precision"),
    ]
    
    # Add FP8 if available
    try:
        from transformer_engine import pytorch as te
        precision_modes.append((PrecisionMode.FP8, "FP8 Mixed Precision"))
    except ImportError:
        pass
    
    results = {}
    
    for precision_mode, description in precision_modes:
        print(f"\nTesting {description}...")
        
        try:
            # Create precision manager
            mp_config = MixedPrecisionConfig(
                mode=precision_mode,
                monitor_overflow=True,
                log_precision_stats=True,
            )
            precision_manager = MixedPrecisionManager(mp_config)
            
            # Optimize model for precision
            test_model = precision_manager.optimize_model_for_precision(model)
            
            # Warmup
            for _ in range(3):
                with precision_manager.get_autocast_context():
                    with precision_manager.get_fp8_context():
                        _ = test_model(input_ids)
            
            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                
                with precision_manager.get_autocast_context():
                    with precision_manager.get_fp8_context():
                        outputs = test_model(input_ids)
                
                end_event.record()
                torch.cuda.synchronize()
                
                latency = start_event.elapsed_time(end_event)
                memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
                
                results[precision_mode.value] = {
                    'latency_ms': latency,
                    'memory_mb': memory_used,
                    'output_shape': outputs['last_hidden_state'].shape,
                }
                
                print(f"  ✓ Latency: {latency:.2f}ms")
                print(f"  ✓ Memory: {memory_used:.1f}MB")
                print(f"  ✓ Output shape: {outputs['last_hidden_state'].shape}")
                
                # Reset memory stats
                torch.cuda.reset_peak_memory_stats()
            else:
                print("  ⚠ CUDA not available, skipping timing")
                
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results[precision_mode.value] = {'error': str(e)}
    
    # Print summary
    print("\n=== Performance Summary ===")
    print(f"{'Mode':<20} {'Latency (ms)':<15} {'Memory (MB)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_latency = results.get('fp32', {}).get('latency_ms')
    baseline_memory = results.get('fp32', {}).get('memory_mb')
    
    for mode, data in results.items():
        if 'error' in data:
            print(f"{mode:<20} {'ERROR':<15} {'ERROR':<15} {'N/A':<10}")
            continue
            
        latency = data.get('latency_ms', 0)
        memory = data.get('memory_mb', 0)
        
        speedup = "N/A"
        if baseline_latency and latency:
            speedup = f"{baseline_latency / latency:.2f}x"
        
        print(f"{mode:<20} {latency:<15.2f} {memory:<15.1f} {speedup:<10}")


def demonstrate_training_step():
    """Demonstrate mixed precision training step."""
    print("\n=== Mixed Precision Training Step Demo ===\n")
    
    # Setup
    config = HyenaGLTConfig(
        genomic_vocab_size=256,
        hidden_size=256,
        num_layers=2,
        num_attention_heads=4,
    )
    
    model = HyenaGLT(config)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create precision manager
    mp_config = MixedPrecisionConfig(
        mode=PrecisionMode.FP16,
        monitor_overflow=True,
        log_precision_stats=True,
        gradient_clipping=1.0,
    )
    precision_manager = MixedPrecisionManager(mp_config)
    
    # Optimize model
    model = precision_manager.optimize_model_for_precision(model)
    
    # Training data
    batch_size = 2
    seq_len = 512
    input_ids = torch.randint(0, config.genomic_vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.genomic_vocab_size, (batch_size, seq_len))
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        labels = labels.cuda()
    
    print("Performing mixed precision training step...")
    
    # Training step with context manager
    with MixedPrecisionTrainingStep(precision_manager, optimizer) as step:
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with precision_manager.get_autocast_context():
            with precision_manager.get_fp8_context():
                outputs = model(input_ids)
                logits = outputs['last_hidden_state']
                
                # Compute loss
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=0
                )
        
        # Backward pass with scaling
        step.backward(loss)
        
        print(f"  ✓ Loss: {loss.item():.6f}")
        print(f"  ✓ Loss scale: {precision_manager.scaler.get_scale() if precision_manager.scaler else 'N/A'}")
    
    # Get precision stats
    stats = precision_manager.get_precision_stats()
    print(f"  ✓ Precision stats: {stats}")
    
    print("Training step completed successfully!")


def demonstrate_adaptive_precision():
    """Demonstrate adaptive precision switching."""
    print("\n=== Adaptive Precision Demo ===\n")
    
    # Create adaptive precision manager
    mp_config = MixedPrecisionConfig(
        mode=PrecisionMode.ADAPTIVE,
        monitor_overflow=True,
        dynamic_loss_scaling=True,
    )
    precision_manager = MixedPrecisionManager(mp_config)
    
    print(f"Initial precision mode: {precision_manager.config.mode.value}")
    
    # Simulate training with overflow detection
    for step in range(10):
        # Simulate training step
        fake_loss = torch.tensor(float('inf') if step == 5 else 1.0)
        
        if precision_manager.scaler:
            scaled_loss = precision_manager.scaler.scale(fake_loss)
            
            # Check for overflow
            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                print(f"  Step {step}: Overflow detected!")
            else:
                print(f"  Step {step}: Normal training, scale: {precision_manager.scaler.get_scale():.0f}")
    
    final_stats = precision_manager.get_precision_stats()
    print(f"\nFinal precision statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")


def create_optimized_training_config(task_type: str) -> TrainingConfig:
    """Create optimized training configuration for different tasks."""
    base_config = TrainingConfig(
        num_epochs=5,
        batch_size=4,
        learning_rate=2e-5,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        precision_monitoring=True,
    )
    
    if task_type == "genome_annotation":
        # Large sequences, need memory efficiency
        base_config.mixed_precision_mode = "adaptive"
        base_config.gradient_checkpointing = True
        base_config.fp16 = True
        base_config.gradient_clipping = 1.0
        
    elif task_type == "variant_effect":
        # Numerical stability important
        base_config.bf16 = True
        base_config.gradient_clipping = 0.5
        
    elif task_type == "protein_function":
        # Try FP8 on capable hardware
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            base_config.fp8 = True
            base_config.kernel_precision = "tf32"
        else:
            base_config.bf16 = True
        base_config.gradient_checkpointing = True
        
    elif task_type == "generation":
        # Memory-critical generation tasks
        base_config.fp16 = True
        base_config.gradient_checkpointing = True
        base_config.gradient_clipping = 2.0
        
    return base_config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Mixed Precision Training Examples")
    parser.add_argument(
        "--demo",
        choices=["benchmark", "training_step", "adaptive", "all"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    print("BLT_Hyena Mixed Precision Training Examples")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available. Some features may not work.\n")
    
    try:
        if args.demo in ["benchmark", "all"]:
            benchmark_precision_modes()
        
        if args.demo in ["training_step", "all"]:
            demonstrate_training_step()
        
        if args.demo in ["adaptive", "all"]:
            demonstrate_adaptive_precision()
            
        # Show optimized configs for different tasks
        print("\n=== Optimized Training Configurations ===\n")
        task_types = ["genome_annotation", "variant_effect", "protein_function", "generation"]
        
        for task_type in task_types:
            config = create_optimized_training_config(task_type)
            print(f"{task_type.replace('_', ' ').title()}:")
            
            precision_settings = []
            if config.fp16:
                precision_settings.append("FP16")
            if config.bf16:
                precision_settings.append("BF16")
            if config.fp8:
                precision_settings.append("FP8")
            if config.mixed_precision_mode != "fp16":
                precision_settings.append(f"Mode: {config.mixed_precision_mode}")
            
            print(f"  Precision: {', '.join(precision_settings) if precision_settings else 'FP32'}")
            print(f"  Gradient Checkpointing: {config.gradient_checkpointing}")
            print(f"  Gradient Clipping: {config.gradient_clipping}")
            print()
        
        print("✅ All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
