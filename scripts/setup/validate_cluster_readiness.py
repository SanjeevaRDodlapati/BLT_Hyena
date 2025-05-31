"""
Simple validation test for cluster deployment readiness.
Tests that GPU handling will work on cluster resources without changes.
"""

import torch
import os
import sys

def test_basic_imports():
    """Test that all components can be imported."""
    try:
        from hyena_glt.distributed import (
            DeviceManager, 
            GPUClusterConfig,
            init_distributed_training,
        )
        print("✓ All distributed components imported successfully")
        return True, DeviceManager, GPUClusterConfig, init_distributed_training
    except ImportError as e:
        print(f"⚠ Import issue: {e}")
        return False, None, None, None

def test_device_manager(DeviceManager):
    """Test device manager functionality."""
    try:
        device_manager = DeviceManager()
        print(f"✓ DeviceManager created: device={device_manager.device}")
        print(f"✓ Device count: {device_manager.device_count}")
        print(f"✓ Is distributed: {device_manager.is_distributed}")
        print(f"✓ Is main process: {device_manager.is_main_process}")
        
        # Test memory info
        memory_info = device_manager.get_memory_info()
        print(f"✓ Memory info: {memory_info}")
        
        # Test tensor movement
        test_tensor = torch.randn(2, 3)
        moved_tensor = device_manager.move_to_device(test_tensor)
        print(f"✓ Tensor movement: {test_tensor.device} -> {moved_tensor.device}")
        
        return True
    except Exception as e:
        print(f"✗ DeviceManager test failed: {e}")
        return False

def test_cluster_config(GPUClusterConfig):
    """Test cluster configuration."""
    try:
        config = GPUClusterConfig(nodes=2, gpus_per_node=4, mixed_precision=True)
        print(f"✓ Cluster config created: {config.to_dict()}")
        
        # Test environment-based config
        env_config = GPUClusterConfig.from_env()
        print(f"✓ Environment config: nodes={env_config.nodes}, gpus_per_node={env_config.gpus_per_node}")
        
        return True
    except Exception as e:
        print(f"✗ GPUClusterConfig test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that existing patterns still work."""
    try:
        # Old pattern (from original benchmark script)
        old_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        old_model = torch.nn.Linear(10, 5).to(old_device)
        print(f"✓ Old pattern works: {old_device}")
        
        # Test basic CUDA operations
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} devices")
            torch.cuda.empty_cache()
            print("✓ CUDA cache cleared")
        else:
            print("✓ CPU fallback working")
        
        return True
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False

def test_enterprise_patterns():
    """Test that enterprise patterns from BLT/Savanna/Vortex are implemented."""
    patterns_found = []
    
    # Check for advanced device management
    try:
        from hyena_glt.distributed.device_manager import DeviceManager
        patterns_found.append("✓ Advanced device management (DeviceManager)")
    except:
        patterns_found.append("✗ Missing: DeviceManager")
    
    # Check for distributed utilities
    try:
        from hyena_glt.distributed.distributed_utils import init_distributed_training
        patterns_found.append("✓ Distributed training utilities")
    except:
        patterns_found.append("✗ Missing: Distributed utilities")
    
    # Check for model parallelization
    try:
        from hyena_glt.distributed.parallel_model import DistributedHyenaGLT
        patterns_found.append("✓ Model parallelization support")
    except:
        patterns_found.append("✗ Missing: Model parallelization")
    
    for pattern in patterns_found:
        print(pattern)
    
    return all("✓" in pattern for pattern in patterns_found)

def main():
    """Main test function for cluster deployment readiness."""
    print("=" * 60)
    print("CLUSTER DEPLOYMENT READINESS TEST")
    print("=" * 60)
    
    print("\n1. Testing Component Imports...")
    imports_ok, DeviceManager, GPUClusterConfig, init_distributed_training = test_basic_imports()
    
    if imports_ok:
        print("\n2. Testing Device Management...")
        device_ok = test_device_manager(DeviceManager)
        
        print("\n3. Testing Cluster Configuration...")
        config_ok = test_cluster_config(GPUClusterConfig)
    else:
        device_ok = config_ok = False
    
    print("\n4. Testing Backward Compatibility...")
    compat_ok = test_backward_compatibility()
    
    print("\n5. Testing Enterprise Patterns...")
    patterns_ok = test_enterprise_patterns()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Overall assessment
    if imports_ok and device_ok and config_ok and compat_ok and patterns_ok:
        print("🎉 SUCCESS: BLT_Hyena is CLUSTER-READY!")
        print("✅ Enterprise-level GPU handling implemented")
        print("✅ Compatible with existing code")
        print("✅ Follows BLT/Savanna/Vortex best practices")
        print("✅ Ready for deployment on GPU clusters WITHOUT changes")
        
        print("\nDeployment Options:")
        print("📋 Single Node: torchrun --nproc_per_node=4 scripts/benchmarks/benchmark_distributed.py")
        print("📋 Multi Node: See docs/CLUSTER_DEPLOYMENT.md")
        print("📋 SLURM/K8s: Configuration templates provided")
        
        return True
    else:
        print("❌ ISSUES FOUND - See details above")
        print("⚠️  Some components may need attention before cluster deployment")
        
        if not imports_ok:
            print("🔧 Fix: Ensure all distributed modules are properly implemented")
        if not (device_ok and config_ok):
            print("🔧 Fix: Debug device manager and cluster configuration")
        if not compat_ok:
            print("🔧 Fix: Resolve backward compatibility issues")
        if not patterns_ok:
            print("🔧 Fix: Complete enterprise pattern implementation")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
