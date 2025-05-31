# Cluster Deployment Guide for HyenaGLT

This guide provides instructions for deploying HyenaGLT on GPU clusters with enterprise-level resource management.

## Overview

The HyenaGLT repository now includes sophisticated GPU cluster handling based on patterns from BLT, Savanna, and Vortex repositories. This upgrade replaces basic GPU detection with comprehensive distributed training infrastructure.

### Key Improvements

**Before (Basic GPU Detection):**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

**After (Enterprise GPU Management):**
```python
from hyena_glt.distributed import init_distributed_training, create_distributed_model

device_manager = init_distributed_training()
model = create_distributed_model(
    model_cls=HyenaGLT,
    model_config=config,
    device_manager=device_manager,
    cluster_config=cluster_config,
)
```

## Quick Start

### Single Node, Multiple GPUs

```bash
# 4 GPUs on single node
torchrun --nproc_per_node=4 scripts/benchmarks/benchmark_distributed.py \
    --batch-size 32 \
    --seq-len 512 \
    --num-steps 100
```

### Multiple Nodes

```bash
# Node 0 (master)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29400 \
    scripts/benchmarks/benchmark_distributed.py

# Node 1 (worker)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29400 \
    scripts/benchmarks/benchmark_distributed.py
```

## Cluster Configurations

### SLURM Integration

Create `slurm_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=hyena_glt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29400
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/benchmarks/benchmark_distributed.py \
    --batch-size 64 \
    --use-fsdp \
    --gradient-checkpointing
```

### Kubernetes Integration

Create `k8s-job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hyena-glt-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: your-registry/hyena-glt:latest
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: 64Gi
          requests:
            nvidia.com/gpu: 4
            memory: 32Gi
        env:
        - name: NCCL_DEBUG
          value: "INFO"
        - name: PYTHONPATH
          value: "/workspace"
        command:
        - torchrun
        - --nproc_per_node=4
        - scripts/benchmarks/benchmark_distributed.py
        - --batch-size=32
        - --use-fsdp
        volumeMounts:
        - name: workspace
          mountPath: /workspace
      volumes:
      - name: workspace
        hostPath:
          path: /path/to/BLT_Hyena
      restartPolicy: Never
```

## Advanced Configuration

### Environment Variables

```bash
# Core distributed settings
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT="29400"
export WORLD_SIZE=8
export RANK=0
export LOCAL_RANK=0

# Performance tuning
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_BUFFSIZE=2097152
```

### Model Configuration

```python
from hyena_glt.distributed import GPUClusterConfig

# Large model configuration
cluster_config = GPUClusterConfig(
    nodes=4,
    gpus_per_node=8,
    backend='nccl',
    mixed_precision=True,
    gradient_checkpointing=True,
    find_unused_parameters=False,
)

model_config = {
    'vocab_size': 100000,
    'd_model': 2048,
    'num_layers': 48,
    'max_seq_len': 8192,
}
```

## Performance Optimization

### Memory Management

1. **Gradient Checkpointing**: Saves memory at cost of compute
   ```python
   cluster_config.gradient_checkpointing = True
   ```

2. **Mixed Precision**: Reduces memory usage and improves speed
   ```python
   cluster_config.mixed_precision = True
   ```

3. **FSDP for Large Models**: Shards parameters across GPUs
   ```bash
   python benchmark_distributed.py --use-fsdp
   ```

### Communication Optimization

1. **NCCL Tuning**:
   ```bash
   export NCCL_TREE_THRESHOLD=0
   export NCCL_ALGO=Tree
   ```

2. **Bandwidth Optimization**:
   ```bash
   export NCCL_MIN_NRINGS=4
   export NCCL_MAX_NRINGS=8
   ```

## Monitoring and Debugging

### Logging Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [Rank %(rank)s] %(message)s',
    handlers=[
        logging.FileHandler(f'train_rank_{os.environ.get("RANK", 0)}.log'),
        logging.StreamHandler()
    ]
)
```

### Memory Monitoring

```python
from hyena_glt.distributed import DeviceManager

device_manager = DeviceManager()

# Check memory usage
memory_info = device_manager.get_memory_info()
print(f"GPU Memory: {memory_info}")

# Clear cache if needed
device_manager.clear_memory()
```

### Performance Profiling

```bash
# Enable CUDA profiling
export CUDA_LAUNCH_BLOCKING=1

# PyTorch profiler
python -m torch.utils.bottleneck benchmark_distributed.py
```

## Troubleshooting

### Common Issues

1. **NCCL Timeout**:
   ```bash
   export NCCL_TIMEOUT=7200  # 2 hours
   ```

2. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use FSDP for large models

3. **Slow Communication**:
   - Check network bandwidth
   - Tune NCCL parameters
   - Verify InfiniBand configuration

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Test NCCL
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 4

# Network diagnostics
ib_write_bw
```

## Migration from Basic GPU Detection

### Before (Original Benchmark)
```python
# scripts/benchmarks/benchmark_blt_performance.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HyenaGLT(config).to(device)
```

### After (Cluster-Ready)
```python
# scripts/benchmarks/benchmark_distributed.py
from hyena_glt.distributed import init_distributed_training, create_distributed_model

device_manager = init_distributed_training()
model = create_distributed_model(
    model_cls=HyenaGLT,
    model_config=config,
    device_manager=device_manager,
    cluster_config=cluster_config,
)
```

## Best Practices

1. **Always use the distributed training utilities** even for single GPU to ensure consistency
2. **Profile your workload** to choose between DDP and FSDP
3. **Use mixed precision** for A100/H100 GPUs
4. **Monitor GPU memory usage** and adjust batch sizes accordingly
5. **Save checkpoints regularly** using distributed checkpointing utilities
6. **Test scaling** on smaller configurations before full cluster deployment

## Support

For cluster-specific issues:
1. Check the logs in `/tmp/hyena_glt_*.log`
2. Verify GPU topology with `nvidia-ml-py`
3. Test network with NCCL benchmarks
4. Monitor resource usage with cluster tools

This distributed training infrastructure ensures HyenaGLT will work seamlessly on any GPU cluster without code changes.
