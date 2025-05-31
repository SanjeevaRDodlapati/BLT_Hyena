"""
Cluster-ready benchmark script for HyenaGLT with distributed training support.
This upgrades the basic GPU detection to enterprise-level GPU cluster handling.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any

import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyena_glt.distributed import (
    GPUClusterConfig,
    create_distributed_model,
    init_distributed_training,
    save_checkpoint_distributed,
    setup_mixed_precision,
)
from hyena_glt.model.hyena_glt import HyenaGLT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Rank %(rank)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Get logger with rank info
def get_logger():
    logger = logging.getLogger(__name__)
    rank = int(os.environ.get("RANK", 0))

    # Add rank to logger
    class RankAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return f"[Rank {self.extra['rank']}] {msg}", kwargs

    return RankAdapter(logger, {"rank": rank})


def create_synthetic_data(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device
):
    """Create synthetic data for benchmarking."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return {"input_ids": input_ids, "labels": labels}


def benchmark_distributed_training(
    model_config: dict[str, Any],
    batch_size: int = 32,
    seq_len: int = 512,
    num_steps: int = 100,
    use_fsdp: bool = False,
    use_mixed_precision: bool = True,
    gradient_checkpointing: bool = False,
):
    """
    Benchmark distributed training with cluster-ready GPU handling.

    This function demonstrates the upgraded GPU resource management compared
    to the basic GPU detection in the original benchmark script.
    """
    logger = get_logger()

    # Initialize distributed training with enterprise-level GPU management
    device_manager = init_distributed_training(backend="nccl", timeout_minutes=30)

    # Create cluster configuration
    cluster_config = GPUClusterConfig.from_env()
    cluster_config.mixed_precision = use_mixed_precision
    cluster_config.gradient_checkpointing = gradient_checkpointing

    if device_manager.is_main_process:
        logger.info("=== Cluster-Ready HyenaGLT Benchmark ===")
        logger.info(f"Cluster config: {cluster_config.to_dict()}")
        logger.info(
            f"Device manager: rank={device_manager.rank}, local_rank={device_manager.local_rank}"
        )
        logger.info(f"GPU memory info: {device_manager.get_memory_info()}")

    # Create distributed model using enterprise patterns
    try:
        # Import the transformer layer for FSDP wrapping if needed
        from hyena_glt.model.layers import HyenaGLTBlock

        transformer_layer_cls = HyenaGLTBlock if use_fsdp else None

        model = create_distributed_model(
            model_cls=HyenaGLT,
            model_config=model_config,
            device_manager=device_manager,
            cluster_config=cluster_config,
            use_fsdp=use_fsdp,
            transformer_layer_cls=transformer_layer_cls,
        )

        if device_manager.is_main_process:
            if hasattr(model, "get_model_size"):
                size_info = model.get_model_size()
                logger.info(f"Model size: {size_info}")

    except Exception as e:
        logger.error(f"Failed to create distributed model: {e}")
        # Fallback to basic model for compatibility
        model = HyenaGLT(**model_config)
        model = device_manager.move_to_device(model)
        logger.info("Using fallback basic model")

    # Setup mixed precision training
    scaler = setup_mixed_precision(device_manager, cluster_config)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create synthetic dataset
    vocab_size = model_config.get("vocab_size", 50000)

    if device_manager.is_main_process:
        logger.info(
            f"Starting benchmark: {num_steps} steps, batch_size={batch_size}, seq_len={seq_len}"
        )

    # Benchmark training loop
    model.train()
    torch.cuda.reset_peak_memory_stats()

    step_times = []
    losses = []

    for step in range(num_steps):
        step_start_time = time.time()

        # Create batch data
        batch = create_synthetic_data(
            batch_size, seq_len, vocab_size, device_manager.device
        )

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        else:
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Synchronize for accurate timing
        device_manager.synchronize()

        step_time = time.time() - step_start_time
        step_times.append(step_time)
        losses.append(loss.item())

        if device_manager.is_main_process and step % 10 == 0:
            logger.info(
                f"Step {step}/{num_steps}: loss={loss.item():.4f}, time={step_time:.3f}s"
            )

            # Log memory usage
            memory_info = device_manager.get_memory_info()
            if memory_info["device"] != "cpu":
                logger.info(
                    f"GPU memory: {memory_info['allocated_gb']:.2f}GB allocated"
                )

    # Calculate performance metrics
    if device_manager.is_main_process:
        avg_step_time = sum(step_times[10:]) / len(step_times[10:])  # Skip warmup
        avg_loss = sum(losses[-10:]) / 10  # Last 10 steps

        tokens_per_step = batch_size * seq_len * device_manager.world_size
        tokens_per_second = tokens_per_step / avg_step_time

        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        logger.info("=== Benchmark Results ===")
        logger.info(f"Average step time: {avg_step_time:.3f}s")
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"Tokens per second: {tokens_per_second:.0f}")
        logger.info(f"Peak GPU memory: {peak_memory:.2f}GB")
        logger.info(f"World size: {device_manager.world_size}")

        # Save a checkpoint to demonstrate distributed checkpointing
        checkpoint_path = "/tmp/hyena_glt_benchmark_checkpoint.pt"
        save_checkpoint_distributed(
            model=model,
            optimizer=optimizer,
            epoch=1,
            loss=avg_loss,
            filepath=checkpoint_path,
            device_manager=device_manager,
            scaler=scaler,
            metadata={"benchmark": True, "tokens_per_second": tokens_per_second},
        )

    # Cleanup
    device_manager.clear_memory()
    device_manager.cleanup_distributed()

    return {
        "avg_step_time": avg_step_time if device_manager.is_main_process else None,
        "tokens_per_second": (
            tokens_per_second if device_manager.is_main_process else None
        ),
        "peak_memory_gb": peak_memory if device_manager.is_main_process else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Cluster-ready HyenaGLT benchmark")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument(
        "--num-steps", type=int, default=100, help="Number of training steps"
    )
    parser.add_argument("--vocab-size", type=int, default=50000, help="Vocabulary size")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument(
        "--use-fsdp", action="store_true", help="Use FSDP instead of DDP"
    )
    parser.add_argument(
        "--no-mixed-precision", action="store_true", help="Disable mixed precision"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )

    args = parser.parse_args()

    # Model configuration
    model_config = {
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
    }

    # Run benchmark
    results = benchmark_distributed_training(
        model_config=model_config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        use_fsdp=args.use_fsdp,
        use_mixed_precision=not args.no_mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    return results


if __name__ == "__main__":
    # Example usage:
    # Single GPU: python benchmark_distributed.py
    # Multi-GPU: torchrun --nproc_per_node=4 benchmark_distributed.py
    # Multi-node: torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 benchmark_distributed.py

    main()
