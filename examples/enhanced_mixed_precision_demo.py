#!/usr/bin/env python3
"""
Enhanced Mixed Precision Demo for BLT_Hyena

This script demonstrates the comprehensive mixed precision capabilities 
for genomic tasks, including task-specific optimizations and hardware-aware 
precision selection.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockPrecisionMode:
    """Mock precision modes for demonstration."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    ADAPTIVE = "adaptive"


class MockMixedPrecisionConfig:
    """Mock mixed precision configuration."""
    def __init__(self, mode, gradient_clipping=1.0, dynamic_loss_scale=True, 
                 monitor_overflow=True, gradient_checkpointing=False,
                 fp8_format="E4M3", cpu_offload=False, growth_interval=2000):
        self.mode = mode
        self.gradient_clipping = gradient_clipping
        self.dynamic_loss_scale = dynamic_loss_scale
        self.monitor_overflow = monitor_overflow
        self.gradient_checkpointing = gradient_checkpointing
        self.fp8_format = fp8_format
        self.cpu_offload = cpu_offload
        self.growth_interval = growth_interval


class MockPrecisionManager:
    """Mock precision manager for demonstration."""
    def __init__(self, config):
        self.config = config
        self.stats = {
            'overflow_count': 0,
            'scale_updates': 0,
            'current_scale': 65536.0,
            'mode': config.mode,
        }
    
    def get_precision_stats(self):
        return self.stats
    
    def optimize_model_for_precision(self, model):
        # Mock optimization
        logger.info(f"Optimizing model for {self.config.mode} precision")
        return model


class MixedPrecisionDemo:
    """Comprehensive mixed precision demonstration."""
    
    def __init__(self, output_dir: str = "./demo_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get current hardware information."""
        hardware_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            compute_cap = torch.cuda.get_device_capability(0)
            hardware_info.update({
                'device_name': device_props.name,
                'compute_capability': f"{compute_cap[0]}.{compute_cap[1]}",
                'memory_gb': device_props.total_memory / (1024**3),
                'multiprocessor_count': device_props.multi_processor_count,
            })
            
        return hardware_info
    
    def get_optimal_precision_config(self, task_type: str, model_size: str = "medium", 
                                   hardware_info: Optional[Dict] = None) -> MockMixedPrecisionConfig:
        """Mock implementation of optimal precision configuration selection."""
        if hardware_info is None:
            hardware_info = self.get_hardware_info()
        
        base_config = {
            'dynamic_loss_scale': True,
            'monitor_overflow': True,
        }
        
        # Task-specific precision selection logic
        if task_type == "genome_annotation":
            # Long sequences need careful precision management
            compute_cap = hardware_info.get('compute_capability', '0.0')
            if float(compute_cap) >= 8.0:
                config = MockMixedPrecisionConfig(
                    mode=MockPrecisionMode.FP8,
                    gradient_clipping=1.0,
                    gradient_checkpointing=True,
                    fp8_format="E4M3",
                    dynamic_loss_scale=base_config['dynamic_loss_scale'],
                    monitor_overflow=base_config['monitor_overflow'],
                )
            else:
                config = MockMixedPrecisionConfig(
                    mode=MockPrecisionMode.ADAPTIVE,
                    gradient_clipping=1.0,
                    gradient_checkpointing=True,
                    dynamic_loss_scale=base_config['dynamic_loss_scale'],
                    monitor_overflow=base_config['monitor_overflow'],
                )
        
        elif task_type == "variant_effect":
            # Stability is crucial for variant effect prediction
            config = MockMixedPrecisionConfig(
                mode=MockPrecisionMode.BF16,
                gradient_clipping=0.5,
                growth_interval=1000,
                dynamic_loss_scale=base_config['dynamic_loss_scale'],
                monitor_overflow=base_config['monitor_overflow'],
            )
        
        elif task_type == "protein_function":
            # Protein sequences can be very long
            memory_gb = hardware_info.get('memory_gb', 0)
            compute_cap = hardware_info.get('compute_capability', '0.0')
            
            if memory_gb > 32 and float(compute_cap) >= 8.0:
                config = MockMixedPrecisionConfig(
                    mode=MockPrecisionMode.FP8,
                    gradient_clipping=1.0,
                    gradient_checkpointing=True,
                    cpu_offload=(model_size == "large"),
                    fp8_format="E4M3",
                    dynamic_loss_scale=base_config['dynamic_loss_scale'],
                    monitor_overflow=base_config['monitor_overflow'],
                )
            else:
                config = MockMixedPrecisionConfig(
                    mode=MockPrecisionMode.BF16,
                    gradient_clipping=1.0,
                    gradient_checkpointing=True,
                    cpu_offload=(model_size == "large"),
                    dynamic_loss_scale=base_config['dynamic_loss_scale'],
                    monitor_overflow=base_config['monitor_overflow'],
                )
        
        elif task_type == "generation":
            # Memory optimization is key for generation
            config = MockMixedPrecisionConfig(
                mode=MockPrecisionMode.FP16,
                gradient_clipping=2.0,
                gradient_checkpointing=True,
                cpu_offload=True,
                growth_interval=500,
                dynamic_loss_scale=base_config['dynamic_loss_scale'],
                monitor_overflow=base_config['monitor_overflow'],
            )
        
        else:
            # Default configuration
            config = MockMixedPrecisionConfig(
                mode=MockPrecisionMode.FP16,
                gradient_clipping=1.0,
                dynamic_loss_scale=base_config['dynamic_loss_scale'],
                monitor_overflow=base_config['monitor_overflow'],
            )
        
        return config
    
    def demonstrate_task_specific_optimization(self):
        """Demonstrate task-specific mixed precision optimization."""
        logger.info("=== Task-Specific Mixed Precision Optimization ===")
        
        tasks = [
            'genome_annotation',
            'variant_effect', 
            'protein_function',
            'generation'
        ]
        
        model_sizes = ['small', 'medium', 'large']
        hardware_info = self.get_hardware_info()
        
        results = {}
        
        for task in tasks:
            logger.info(f"\n--- {task.upper()} TASK ---")
            task_results = {}
            
            for size in model_sizes:
                config = self.get_optimal_precision_config(task, size, hardware_info)
                precision_manager = MockPrecisionManager(config)
                
                logger.info(f"{size} model:")
                logger.info(f"  Precision mode: {config.mode}")
                logger.info(f"  Gradient clipping: {config.gradient_clipping}")
                logger.info(f"  Gradient checkpointing: {config.gradient_checkpointing}")
                logger.info(f"  CPU offload: {config.cpu_offload}")
                
                # Simulate some training stats
                stats = precision_manager.get_precision_stats()
                logger.info(f"  Training stats: {stats}")
                
                task_results[size] = {
                    'precision_mode': config.mode,
                    'gradient_clipping': config.gradient_clipping,
                    'gradient_checkpointing': config.gradient_checkpointing,
                    'cpu_offload': config.cpu_offload,
                    'stats': stats,
                }
            
            results[task] = task_results
        
        return results
    
    def demonstrate_hardware_aware_selection(self):
        """Demonstrate hardware-aware precision mode selection."""
        logger.info("\n=== Hardware-Aware Precision Selection ===")
        
        hardware_info = self.get_hardware_info()
        
        logger.info("Hardware Information:")
        for key, value in hardware_info.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\nTask-specific precision recommendations:")
        
        tasks = ['genome_annotation', 'variant_effect', 'protein_function', 'generation']
        model_sizes = ['small', 'medium', 'large']
        
        recommendations = {}
        
        for task in tasks:
            logger.info(f"\n{task.upper()}:")
            task_recommendations = {}
            
            for size in model_sizes:
                config = self.get_optimal_precision_config(task, size, hardware_info)
                logger.info(f"  {size}: {config.mode} "
                           f"(clipping: {config.gradient_clipping}, "
                           f"checkpointing: {config.gradient_checkpointing})")
                
                task_recommendations[size] = {
                    'mode': config.mode,
                    'gradient_clipping': config.gradient_clipping,
                    'gradient_checkpointing': config.gradient_checkpointing,
                }
            
            recommendations[task] = task_recommendations
        
        return recommendations
    
    def benchmark_precision_modes(self):
        """Benchmark different precision modes."""
        logger.info("\n=== Precision Mode Performance Benchmark ===")
        
        # Create a simple model for testing
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Test different precision modes
        modes = [
            MockPrecisionMode.FP32,
            MockPrecisionMode.FP16,
            MockPrecisionMode.BF16,
        ]
        
        # Add FP8 if hardware supports it
        hardware_info = self.get_hardware_info()
        if hardware_info.get('compute_capability', '0.0') >= '8.0':
            modes.append(MockPrecisionMode.FP8)
        
        results = {}
        
        for mode in modes:
            logger.info(f"\nTesting {mode} precision...")
            
            # Create precision manager
            config = MockMixedPrecisionConfig(mode=mode)
            precision_manager = MockPrecisionManager(config)
            
            # Simulate training step timing
            start_time = time.time()
            
            # Create dummy data
            batch_size = 32
            seq_length = 512
            dummy_input = torch.randn(batch_size, seq_length)
            
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            
            # Simulate forward pass
            with torch.no_grad():
                memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                output = model(dummy_input)
                memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            end_time = time.time()
            
            execution_time = end_time - start_time
            memory_usage = (memory_after - memory_before) / (1024 * 1024)  # MB
            
            results[mode] = {
                'execution_time': execution_time,
                'memory_usage_mb': memory_usage,
                'precision_stats': precision_manager.get_precision_stats(),
            }
            
            logger.info(f"  Execution time: {execution_time:.4f}s")
            logger.info(f"  Memory usage: {memory_usage:.2f}MB")
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def generate_report(self, task_results, hardware_results, benchmark_results):
        """Generate comprehensive report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hardware_info': self.get_hardware_info(),
            'task_specific_results': task_results,
            'hardware_aware_results': hardware_results,
            'benchmark_results': benchmark_results,
        }
        
        # Save to file
        report_file = self.output_dir / 'mixed_precision_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nComprehensive report saved to: {report_file}")
        
        # Print summary
        logger.info("\n=== SUMMARY ===")
        logger.info("Task-specific optimizations demonstrated:")
        for task in task_results.keys():
            logger.info(f"  ✓ {task}")
        
        logger.info("Hardware-aware selection tested for:")
        for task in hardware_results.keys():
            logger.info(f"  ✓ {task}")
        
        logger.info("Precision modes benchmarked:")
        for mode in benchmark_results.keys():
            logger.info(f"  ✓ {mode}")
        
        return report


def main():
    """Main function to run mixed precision demonstrations."""
    parser = argparse.ArgumentParser(
        description="Demonstrate enhanced mixed precision capabilities for BLT_Hyena"
    )
    parser.add_argument(
        '--demo',
        choices=['task-specific', 'hardware-aware', 'benchmark', 'all'],
        default='all',
        help='Which demonstration to run'
    )
    parser.add_argument(
        '--output-dir',
        default='./demo_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("BLT_Hyena Enhanced Mixed Precision Demonstration")
    logger.info("=" * 60)
    
    demo = MixedPrecisionDemo(args.output_dir)
    
    task_results = None
    hardware_results = None
    benchmark_results = None
    
    if args.demo in ['task-specific', 'all']:
        task_results = demo.demonstrate_task_specific_optimization()
    
    if args.demo in ['hardware-aware', 'all']:
        hardware_results = demo.demonstrate_hardware_aware_selection()
    
    if args.demo in ['benchmark', 'all']:
        benchmark_results = demo.benchmark_precision_modes()
    
    # Generate comprehensive report
    if all(x is not None for x in [task_results, hardware_results, benchmark_results]):
        demo.generate_report(task_results, hardware_results, benchmark_results)
    
    logger.info("\n" + "=" * 60)
    logger.info("Mixed precision demonstration completed!")
    logger.info(f"Results saved to: {demo.output_dir}")


if __name__ == "__main__":
    main()
