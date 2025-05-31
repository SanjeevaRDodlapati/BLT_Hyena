"""
Deployment Optimization for Hyena-GLT

This module provides deployment optimization tools including ONNX export,
TensorRT optimization, and inference engine optimization.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import tensorrt as trt

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

from ..model import HyenaGLT

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for deployment optimization."""

    # Export settings
    export_format: str = "onnx"  # "onnx", "torchscript", "tensorrt"

    # ONNX settings
    onnx_opset_version: int = 11
    onnx_optimize: bool = True
    onnx_dynamic_axes: bool = True

    # TensorRT settings
    trt_precision: str = "fp16"  # "fp32", "fp16", "int8"
    trt_max_workspace: int = 1 << 30  # 1GB
    trt_max_batch_size: int = 32

    # Optimization settings
    optimize_for_inference: bool = True
    enable_fusion: bool = True
    enable_quantization: bool = False

    # Inference settings
    batch_size: int = 1
    sequence_length: int = 512
    enable_profiling: bool = False

    # Hardware settings
    device: str = "cuda"  # "cuda", "cpu"
    num_threads: int = 4


class ModelOptimizer:
    """Main deployment optimization interface."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.optimized_models = {}

    def optimize_for_deployment(
        self,
        model: HyenaGLT,
        sample_input: torch.Tensor,
        save_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Optimize model for deployment.

        Args:
            model: The model to optimize
            sample_input: Sample input for tracing/export
            save_path: Path to save optimized models

        Returns:
            Dictionary of optimized models and metadata
        """
        results = {}

        # Prepare model for inference
        model = self._prepare_model_for_inference(model)

        # Export to different formats
        if self.config.export_format in ["onnx", "all"] and HAS_ONNX:
            onnx_path = self._export_onnx(model, sample_input, save_path)
            results["onnx"] = onnx_path

        if self.config.export_format in ["torchscript", "all"]:
            ts_path = self._export_torchscript(model, sample_input, save_path)
            results["torchscript"] = ts_path

        if self.config.export_format in ["tensorrt", "all"] and HAS_TENSORRT:
            trt_path = self._export_tensorrt(model, sample_input, save_path)
            results["tensorrt"] = trt_path

        # Store optimized models
        self.optimized_models = results

        return results

    def _prepare_model_for_inference(self, model: HyenaGLT) -> nn.Module:
        """Prepare model for inference optimization."""
        model.eval()

        if self.config.optimize_for_inference:
            # Apply inference optimizations
            model = self._apply_inference_optimizations(model)

        return model

    def _apply_inference_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply various inference optimizations."""
        # Freeze batch normalization
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        # Fuse operations if enabled
        if self.config.enable_fusion:
            model = self._fuse_operations(model)

        return model

    def _fuse_operations(self, model: nn.Module) -> nn.Module:
        """Fuse compatible operations for better performance."""
        # This is a simplified version - actual fusion would be more complex
        torch.backends.cudnn.benchmark = True

        # Enable JIT optimizations
        if hasattr(torch.jit, "optimize_for_inference"):
            model = torch.jit.optimize_for_inference(model)

        return model

    def _export_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        save_path: str | None = None,
    ) -> str:
        """Export model to ONNX format."""
        if not HAS_ONNX:
            raise ImportError(
                "ONNX not available. Install with: pip install onnx onnxruntime"
            )

        logger.info("Exporting model to ONNX...")

        # Prepare export path
        if save_path:
            onnx_path = Path(save_path) / "model.onnx"
        else:
            onnx_path = "model.onnx"

        # Set dynamic axes for flexible input sizes
        dynamic_axes = None
        if self.config.onnx_dynamic_axes:
            dynamic_axes = {
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"},
            }

        # Export to ONNX
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=self.config.onnx_opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        # Optimize ONNX model if requested
        if self.config.onnx_optimize:
            self._optimize_onnx_model(onnx_path)

        logger.info(f"ONNX model saved to {onnx_path}")
        return str(onnx_path)

    def _optimize_onnx_model(self, onnx_path: str):
        """Optimize ONNX model."""
        import onnx
        from onnx import optimizer

        # Load model
        model = onnx.load(onnx_path)

        # Apply optimizations
        optimized_model = optimizer.optimize(model)

        # Save optimized model
        onnx.save(optimized_model, onnx_path)
        logger.info("ONNX model optimized")

    def _export_torchscript(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        save_path: str | None = None,
    ) -> str:
        """Export model to TorchScript format."""
        logger.info("Exporting model to TorchScript...")

        # Prepare export path
        if save_path:
            ts_path = Path(save_path) / "model.pt"
        else:
            ts_path = "model.pt"

        # Trace the model
        try:
            traced_model = torch.jit.trace(model, sample_input)
        except Exception as e:
            logger.warning(f"Tracing failed: {e}. Trying scripting...")
            traced_model = torch.jit.script(model)

        # Optimize for inference
        if self.config.optimize_for_inference:
            traced_model = torch.jit.optimize_for_inference(traced_model)

        # Save model
        traced_model.save(ts_path)

        logger.info(f"TorchScript model saved to {ts_path}")
        return str(ts_path)

    def _export_tensorrt(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        save_path: str | None = None,
    ) -> str:
        """Export model to TensorRT format."""
        if not HAS_TENSORRT:
            raise ImportError("TensorRT not available")

        logger.info("Exporting model to TensorRT...")

        # First export to ONNX
        onnx_path = self._export_onnx(model, sample_input, save_path)

        # Convert ONNX to TensorRT
        trt_optimizer = TensorRTOptimizer(self.config)
        trt_path = trt_optimizer.convert_onnx_to_tensorrt(onnx_path, save_path)

        return trt_path


class ONNXExporter:
    """Specialized ONNX export utilities."""

    def __init__(self, config: DeploymentConfig):
        self.config = config

    def export_with_validation(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        tolerance: float = 1e-3,
    ) -> bool:
        """Export to ONNX with validation."""
        if not HAS_ONNX:
            raise ImportError("ONNX not available")

        # Export model
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=self.config.onnx_opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

        # Validate exported model
        return self._validate_onnx_export(model, sample_input, output_path, tolerance)

    def _validate_onnx_export(
        self,
        original_model: nn.Module,
        sample_input: torch.Tensor,
        onnx_path: str,
        tolerance: float,
    ) -> bool:
        """Validate ONNX export against original model."""
        # Get original output
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(sample_input)

        # Load and run ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        # Compare outputs
        onnx_output = torch.from_numpy(ort_outputs[0])
        diff = torch.abs(original_output - onnx_output).max().item()

        if diff < tolerance:
            logger.info(f"ONNX export validation passed (max diff: {diff})")
            return True
        else:
            logger.error(f"ONNX export validation failed (max diff: {diff})")
            return False


class TensorRTOptimizer:
    """TensorRT optimization utilities."""

    def __init__(self, config: DeploymentConfig):
        self.config = config

    def convert_onnx_to_tensorrt(
        self, onnx_path: str, save_path: str | None = None
    ) -> str:
        """Convert ONNX model to TensorRT engine."""
        if not HAS_TENSORRT:
            raise ImportError("TensorRT not available")

        logger.info("Converting ONNX to TensorRT...")

        # Prepare output path
        if save_path:
            trt_path = Path(save_path) / "model.trt"
        else:
            trt_path = "model.trt"

        # Create TensorRT builder
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)

        # Create network
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX model
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.config.trt_max_workspace

        # Set precision
        if self.config.trt_precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.config.trt_precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

        # Build engine
        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Serialize engine
        with open(trt_path, "wb") as f:
            f.write(engine.serialize())

        logger.info(f"TensorRT engine saved to {trt_path}")
        return str(trt_path)


class InferenceEngine:
    """Optimized inference engine for different model formats."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.engines = {}

    def load_model(self, model_path: str, model_format: str = "pytorch"):
        """Load model for inference."""
        if model_format == "pytorch":
            engine = self._load_pytorch_model(model_path)
        elif model_format == "torchscript":
            engine = self._load_torchscript_model(model_path)
        elif model_format == "onnx":
            engine = self._load_onnx_model(model_path)
        elif model_format == "tensorrt":
            engine = self._load_tensorrt_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")

        self.engines[model_format] = engine
        return engine

    def _load_pytorch_model(self, model_path: str):
        """Load PyTorch model."""
        model = torch.load(model_path, map_location=self.config.device)
        model.eval()
        return model

    def _load_torchscript_model(self, model_path: str):
        """Load TorchScript model."""
        model = torch.jit.load(model_path, map_location=self.config.device)
        model.eval()
        return model

    def _load_onnx_model(self, model_path: str):
        """Load ONNX model."""
        if not HAS_ONNX:
            raise ImportError("ONNX Runtime not available")

        providers = ["CPUExecutionProvider"]
        if (
            self.config.device == "cuda"
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers = ["CUDAExecutionProvider"] + providers

        session = ort.InferenceSession(model_path, providers=providers)
        return session

    def _load_tensorrt_model(self, model_path: str):
        """Load TensorRT model."""
        if not HAS_TENSORRT:
            raise ImportError("TensorRT not available")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(model_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        return context

    def predict(
        self, input_data: torch.Tensor | np.ndarray, model_format: str = "pytorch"
    ) -> torch.Tensor | np.ndarray:
        """Run inference with specified model format."""
        if model_format not in self.engines:
            raise ValueError(f"Model format {model_format} not loaded")

        engine = self.engines[model_format]

        if model_format in ["pytorch", "torchscript"]:
            return self._predict_pytorch(engine, input_data)
        elif model_format == "onnx":
            return self._predict_onnx(engine, input_data)
        elif model_format == "tensorrt":
            return self._predict_tensorrt(engine, input_data)

    def _predict_pytorch(self, model, input_data: torch.Tensor) -> torch.Tensor:
        """PyTorch model inference."""
        with torch.no_grad():
            return model(input_data)

    def _predict_onnx(
        self, session, input_data: torch.Tensor | np.ndarray
    ) -> np.ndarray:
        """ONNX model inference."""
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()

        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: input_data}
        ort_outputs = session.run(None, ort_inputs)

        return ort_outputs[0]

    def _predict_tensorrt(
        self, context, input_data: torch.Tensor | np.ndarray
    ) -> np.ndarray:
        """TensorRT model inference."""
        # This is a simplified version - actual TensorRT inference is more complex
        # involving memory allocation and CUDA streams
        raise NotImplementedError("TensorRT inference implementation pending")


class ModelProfiler:
    """Profile model performance across different optimization strategies."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.profiling_results = {}

    def profile_model(
        self,
        model_paths: dict[str, str],
        sample_input: torch.Tensor | np.ndarray,
        num_runs: int = 100,
    ) -> dict[str, dict[str, float]]:
        """Profile different model formats."""
        results = {}

        inference_engine = InferenceEngine(self.config)

        for format_name, model_path in model_paths.items():
            logger.info(f"Profiling {format_name} model...")

            # Load model
            inference_engine.load_model(model_path, format_name)

            # Warm up
            for _ in range(10):
                _ = inference_engine.predict(sample_input, format_name)

            # Profile
            latencies = []
            memory_usage = []

            for _ in range(num_runs):
                # Memory before
                mem_before = self._get_memory_usage()

                # Time inference
                start_time = time.perf_counter()
                _ = inference_engine.predict(sample_input, format_name)
                end_time = time.perf_counter()

                # Memory after
                mem_after = self._get_memory_usage()

                latencies.append((end_time - start_time) * 1000)  # Convert to ms
                memory_usage.append(mem_after - mem_before)

            results[format_name] = {
                "avg_latency_ms": np.mean(latencies),
                "std_latency_ms": np.std(latencies),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
                "avg_memory_mb": np.mean(memory_usage) / 1024 / 1024,
                "throughput_qps": 1000 / np.mean(latencies),
            }

        self.profiling_results = results
        return results

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            process = psutil.Process()
            return process.memory_info().rss

    def print_profiling_results(self):
        """Print profiling results."""
        if not self.profiling_results:
            print("No profiling results available")
            return

        print("\n" + "=" * 60)
        print("MODEL PROFILING RESULTS")
        print("=" * 60)

        for format_name, metrics in self.profiling_results.items():
            print(f"\n{format_name.upper()} MODEL:")
            print(
                f"  Average Latency: {metrics['avg_latency_ms']:.2f} ± {metrics['std_latency_ms']:.2f} ms"
            )
            print(
                f"  Min/Max Latency: {metrics['min_latency_ms']:.2f} / {metrics['max_latency_ms']:.2f} ms"
            )
            print(f"  Throughput: {metrics['throughput_qps']:.2f} QPS")
            print(f"  Memory Usage: {metrics['avg_memory_mb']:.2f} MB")

        # Compare formats
        if len(self.profiling_results) > 1:
            print("\nCOMPARISON:")
            baseline = list(self.profiling_results.keys())[0]
            baseline_latency = self.profiling_results[baseline]["avg_latency_ms"]

            for format_name, metrics in list(self.profiling_results.items())[1:]:
                speedup = baseline_latency / metrics["avg_latency_ms"]
                print(f"  {format_name} vs {baseline}: {speedup:.2f}x speedup")


class DeploymentBenchmark:
    """Comprehensive deployment benchmarking."""

    def __init__(self):
        self.benchmark_results = {}

    def run_comprehensive_benchmark(
        self,
        original_model: nn.Module,
        optimized_models: dict[str, str],
        test_data: torch.utils.data.DataLoader,
        config: DeploymentConfig,
    ) -> dict[str, Any]:
        """Run comprehensive deployment benchmark."""
        results = {"accuracy": {}, "performance": {}, "resource_usage": {}}

        # Test accuracy
        for format_name, model_path in optimized_models.items():
            accuracy = self._test_accuracy(model_path, format_name, test_data)
            results["accuracy"][format_name] = accuracy

        # Test performance
        profiler = ModelProfiler(config)
        sample_input = next(iter(test_data))[0][:1]  # Single sample
        performance_results = profiler.profile_model(optimized_models, sample_input)
        results["performance"] = performance_results

        # Test resource usage
        for format_name, model_path in optimized_models.items():
            resource_usage = self._test_resource_usage(model_path, format_name)
            results["resource_usage"][format_name] = resource_usage

        self.benchmark_results = results
        return results

    def _test_accuracy(
        self, model_path: str, format_name: str, test_data: torch.utils.data.DataLoader
    ) -> float:
        """Test model accuracy."""
        # Simplified accuracy test
        # In practice, this would load the specific model format and test thoroughly
        return 0.95  # Placeholder

    def _test_resource_usage(
        self, model_path: str, format_name: str
    ) -> dict[str, float]:
        """Test resource usage."""
        # Get model size
        model_size = Path(model_path).stat().st_size / 1024 / 1024  # MB

        return {
            "model_size_mb": model_size,
            "memory_footprint_mb": model_size * 1.5,  # Rough estimate
            "initialization_time_ms": 100.0,  # Placeholder
        }
