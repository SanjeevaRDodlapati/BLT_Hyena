# Hyena-GLT Deployment Guide

This comprehensive guide covers deploying Hyena-GLT models in production environments, from optimization to monitoring.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Model Optimization](#model-optimization)
3. [Export Formats](#export-formats)
4. [Serving Infrastructure](#serving-infrastructure)
5. [Cloud Deployment](#cloud-deployment)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Security Considerations](#security-considerations)
9. [Scaling Strategies](#scaling-strategies)
10. [Troubleshooting](#troubleshooting)

## Deployment Overview

### Deployment Workflow

```mermaid
graph LR
    A[Trained Model] --> B[Optimization]
    B --> C[Export]
    C --> D[Containerization]
    D --> E[Deployment]
    E --> F[Monitoring]
    F --> G[Scaling]
```

### Deployment Options

#### 1. Local Deployment
- Development and testing
- Single-machine inference
- Jupyter notebook environments

#### 2. Cloud Deployment
- AWS, GCP, Azure platforms
- Serverless functions
- Container orchestration

#### 3. Edge Deployment
- Mobile devices
- IoT devices
- Offline environments

#### 4. High-Performance Computing
- Cluster computing
- Multi-GPU inference
- Batch processing

## Model Optimization

### Quantization

#### Dynamic Quantization
```python
from hyena_glt.optimization import DynamicQuantizer

# Easy to use, minimal accuracy loss
quantizer = DynamicQuantizer()
quantized_model = quantizer.quantize(model)

# Performance improvement: ~2x speedup
# Memory reduction: ~50%
# Accuracy loss: <1%
```

#### Static Quantization
```python
from hyena_glt.optimization import StaticQuantizer

# Better performance, requires calibration
quantizer = StaticQuantizer()
quantized_model = quantizer.quantize(
    model, 
    calibration_dataset=calibration_data,
    num_calibration_batches=100
)

# Performance improvement: ~3x speedup
# Memory reduction: ~75%
# Accuracy loss: 1-3%
```

#### Quantization-Aware Training (QAT)
```python
from hyena_glt.optimization import QATTrainer

# Best accuracy with quantization
qat_trainer = QATTrainer(model, config)
qat_trainer.prepare_model_for_qat()
qat_trainer.train(train_dataset, num_epochs=5)
quantized_model = qat_trainer.convert_to_quantized()

# Performance improvement: ~3x speedup
# Memory reduction: ~75%
# Accuracy loss: <0.5%
```

### Pruning

#### Magnitude-Based Pruning
```python
from hyena_glt.optimization import MagnitudePruner

# Remove weights with smallest magnitudes
pruner = MagnitudePruner(sparsity=0.5)
pruned_model = pruner.prune(model)

# Model size reduction: ~50%
# Speedup: 1.5-2x (with sparse inference)
# Accuracy loss: 2-5%
```

#### Structured Pruning
```python
from hyena_glt.optimization import StructuredPruner

# Remove entire channels/layers
pruner = StructuredPruner(
    sparsity=0.3,
    pruning_method="fisher_information"
)
pruned_model = pruner.prune(model, train_dataset)

# Model size reduction: ~30%
# Speedup: ~1.5x (any hardware)
# Accuracy loss: 1-3%
```

#### Gradual Pruning
```python
from hyena_glt.optimization import GradualPruner

# Prune during training
pruner = GradualPruner(
    initial_sparsity=0.0,
    final_sparsity=0.8,
    pruning_steps=1000
)

trainer = HyenaGLTTrainer(model, config, train_dataset)
trainer.add_callback(pruner)
trainer.train()
```

### Knowledge Distillation

#### Standard Distillation
```python
from hyena_glt.optimization import KnowledgeDistiller

# Distill large teacher to small student
teacher_model = large_model  # Your trained large model
student_config = HyenaGLTConfig(
    d_model=128,    # Smaller than teacher
    n_layers=4      # Fewer layers
)

distiller = KnowledgeDistiller(
    teacher=teacher_model,
    student_config=student_config,
    temperature=3.0,
    alpha=0.7  # Balance between distillation and task loss
)

student_model = distiller.distill(
    train_dataset=train_dataset,
    num_epochs=20
)

# Model size reduction: ~10x
# Speedup: ~10x
# Accuracy retention: 90-95%
```

#### Progressive Distillation
```python
# Multi-stage distillation
stages = [
    {'d_model': 512, 'n_layers': 8},  # Intermediate model
    {'d_model': 256, 'n_layers': 6},  # Smaller model
    {'d_model': 128, 'n_layers': 4}   # Target model
]

current_teacher = large_model
for stage_config in stages:
    distiller = KnowledgeDistiller(teacher=current_teacher, 
                                 student_config=stage_config)
    current_teacher = distiller.distill(train_dataset)
```

### Optimization Pipeline

```python
from hyena_glt.optimization import OptimizationPipeline

# Complete optimization pipeline
pipeline = OptimizationPipeline([
    ('distillation', {
        'student_config': small_config,
        'temperature': 3.0
    }),
    ('quantization', {
        'method': 'static',
        'calibration_batches': 100
    }),
    ('pruning', {
        'sparsity': 0.3,
        'method': 'magnitude'
    })
])

optimized_model = pipeline.optimize(model, train_dataset)

# Final model: ~100x smaller, ~50x faster
# Accuracy retention: 85-90%
```

## Export Formats

### PyTorch Formats

#### TorchScript
```python
import torch

# Tracing (recommended for inference)
example_input = torch.randint(0, 4, (1, 1024))
traced_model = torch.jit.trace(model, example_input)
traced_model.save("hyena_glt_traced.pt")

# Scripting (for dynamic models)
scripted_model = torch.jit.script(model)
scripted_model.save("hyena_glt_scripted.pt")
```

#### Mobile Deployment
```python
# Optimize for mobile
from torch.utils.mobile_optimizer import optimize_for_mobile

mobile_model = optimize_for_mobile(traced_model)
mobile_model._save_for_lite_interpreter("hyena_glt_mobile.ptl")
```

### ONNX Export

```python
import torch.onnx

# Standard ONNX export
torch.onnx.export(
    model,
    example_input,
    "hyena_glt_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    }
)

# Verify ONNX model
import onnx
onnx_model = onnx.load("hyena_glt_model.onnx")
onnx.checker.check_model(onnx_model)
```

### TensorRT Optimization

```python
import tensorrt as trt
from hyena_glt.deployment import TensorRTConverter

# Convert ONNX to TensorRT
converter = TensorRTConverter()
trt_engine = converter.convert_onnx_to_trt(
    "hyena_glt_model.onnx",
    max_batch_size=32,
    precision="fp16",  # fp32, fp16, or int8
    max_workspace_size=1 << 30  # 1GB
)

# Save TensorRT engine
with open("hyena_glt.trt", "wb") as f:
    f.write(trt_engine.serialize())
```

### CoreML Export (for iOS)

```python
import coremltools as ct

# Convert to CoreML
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 1024), name="input_ids")],
    outputs=[ct.TensorType(name="logits")],
    compute_units=ct.ComputeUnit.ALL  # CPU_AND_GPU
)

coreml_model.save("HyenaGLT.mlmodel")
```

## Serving Infrastructure

### FastAPI REST Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
from hyena_glt.models import HyenaGLT
from hyena_glt.tokenizers import DNATokenizer

# Request/Response models
class PredictionRequest(BaseModel):
    sequence: str
    return_probabilities: bool = False

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: list = None

# FastAPI app
app = FastAPI(title="Hyena-GLT Genomic Classifier")

# Global model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model = HyenaGLT.load_from_checkpoint("model.pt")
    model.eval()
    tokenizer = DNATokenizer()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Tokenize sequence
        tokens = tokenizer.encode(request.sequence)
        input_ids = torch.tensor([tokens])
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids)
            probabilities = torch.softmax(outputs, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities)
        
        response = PredictionResponse(
            prediction=prediction.item(),
            confidence=confidence.item()
        )
        
        if request.return_probabilities:
            response.probabilities = probabilities.tolist()[0]
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Batch Processing Service

```python
from hyena_glt.serving import BatchProcessor
import asyncio
from concurrent.futures import ThreadPoolExecutor

class GenomicBatchProcessor:
    def __init__(self, model_path, batch_size=32, max_workers=4):
        self.model = HyenaGLT.load_from_checkpoint(model_path)
        self.tokenizer = DNATokenizer()
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_sequences(self, sequences):
        """Process sequences in batches."""
        results = []
        
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i:i + self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _process_batch(self, batch):
        """Process a single batch."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._predict_batch, 
            batch
        )
    
    def _predict_batch(self, sequences):
        """Synchronous batch prediction."""
        tokens = [self.tokenizer.encode(seq) for seq in sequences]
        input_ids = torch.tensor(tokens)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = torch.argmax(outputs, dim=-1)
        
        return predictions.tolist()

# Usage
processor = GenomicBatchProcessor("model.pt")
sequences = ["ATCGATCG", "GCTAGCTA", ...]  # Large list
results = await processor.process_sequences(sequences)
```

### gRPC Service

```python
# genomic_service.proto
"""
syntax = "proto3";

service GenomicClassifier {
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc PredictBatch(PredictBatchRequest) returns (PredictBatchResponse);
}

message PredictRequest {
  string sequence = 1;
}

message PredictResponse {
  int32 prediction = 1;
  float confidence = 2;
  repeated float probabilities = 3;
}
"""

# Server implementation
import grpc
from concurrent import futures
import genomic_service_pb2_grpc as pb2_grpc
import genomic_service_pb2 as pb2

class GenomicClassifierServicer(pb2_grpc.GenomicClassifierServicer):
    def __init__(self, model_path):
        self.model = HyenaGLT.load_from_checkpoint(model_path)
        self.tokenizer = DNATokenizer()
    
    def Predict(self, request, context):
        tokens = self.tokenizer.encode(request.sequence)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            probabilities = torch.softmax(outputs, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities)
        
        return pb2.PredictResponse(
            prediction=prediction.item(),
            confidence=confidence.item(),
            probabilities=probabilities.tolist()[0]
        )

# Start server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_GenomicClassifierServicer_to_server(
        GenomicClassifierServicer("model.pt"), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
```

## Cloud Deployment

### Docker Containers

#### Production Dockerfile
```dockerfile
# Multi-stage build for smaller image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy application
COPY hyena_glt/ ./hyena_glt/
COPY model.pt .
COPY serve.py .

# Set up user (security)
RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "serve.py"]
```

#### Docker Compose for Development
```yaml
version: '3.8'

services:
  hyena-glt-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.pt
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - hyena-glt-api

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### Kubernetes Deployment

#### Deployment YAML
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hyena-glt-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hyena-glt
  template:
    metadata:
      labels:
        app: hyena-glt
    spec:
      containers:
      - name: hyena-glt
        image: your-registry/hyena-glt:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: MODEL_PATH
          value: "/models/model.pt"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: hyena-glt-service
spec:
  selector:
    app: hyena-glt
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hyena-glt-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hyena-glt-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### GPU-Enabled Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hyena-glt-gpu
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hyena-glt-gpu
  template:
    metadata:
      labels:
        app: hyena-glt-gpu
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-v100
      containers:
      - name: hyena-glt
        image: your-registry/hyena-glt:gpu
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4000m"
```

### Serverless Deployment

#### AWS Lambda
```python
# lambda_handler.py
import json
import torch
from hyena_glt.models import HyenaGLT
from hyena_glt.tokenizers import DNATokenizer

# Global variables for model persistence
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        model = HyenaGLT.load_from_checkpoint('/opt/model.pt')
        model.eval()
        tokenizer = DNATokenizer()

def lambda_handler(event, context):
    load_model()
    
    try:
        # Parse request
        body = json.loads(event['body'])
        sequence = body['sequence']
        
        # Predict
        tokens = tokenizer.encode(sequence)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            outputs = model(input_ids)
            probabilities = torch.softmax(outputs, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
        
        response = {
            'prediction': prediction.item(),
            'probabilities': probabilities.tolist()[0]
        }
        
        return {
            'statusCode': 200,
            'body': json.dumps(response),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

#### Azure Functions
```python
import logging
import json
import azure.functions as func
from hyena_glt.models import HyenaGLT

app = func.FunctionApp()

@app.function_name(name="GenomicPredict")
@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Load model (cached)
        model = HyenaGLT.load_from_checkpoint('model.pt')
        
        # Process request
        req_body = req.get_json()
        sequence = req_body.get('sequence')
        
        # Predict
        result = model.predict(sequence)
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500
        )
```

### Cloud-Specific Services

#### AWS SageMaker
```python
# sagemaker_inference.py
import torch
from sagemaker.pytorch import PyTorchModel
from hyena_glt.models import HyenaGLT

class HyenaGLTPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def model_fn(self, model_dir):
        """Load model for SageMaker."""
        model = HyenaGLT.load_from_checkpoint(f"{model_dir}/model.pt")
        return model
    
    def predict_fn(self, input_data, model):
        """Make predictions."""
        sequences = input_data['sequences']
        tokenizer = DNATokenizer()
        
        predictions = []
        for sequence in sequences:
            tokens = tokenizer.encode(sequence)
            input_ids = torch.tensor([tokens])
            
            with torch.no_grad():
                outputs = model(input_ids)
                prediction = torch.argmax(outputs, dim=-1)
                predictions.append(prediction.item())
        
        return {'predictions': predictions}

# Deploy to SageMaker
pytorch_model = PyTorchModel(
    model_data='s3://your-bucket/model.tar.gz',
    role='SageMakerRole',
    entry_point='sagemaker_inference.py',
    framework_version='2.0',
    py_version='py310'
)

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge'
)
```

## Performance Optimization

### Inference Optimization

#### Model Compilation
```python
# PyTorch 2.0 compilation
compiled_model = torch.compile(model, mode="max-autotune")

# TensorRT compilation
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 1024), dtype=torch.long)],
    enabled_precisions=[torch.float16],
    workspace_size=1 << 30
)
```

#### Batching Strategies
```python
class DynamicBatcher:
    def __init__(self, model, max_batch_size=32, max_wait_time=0.1):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
    
    async def predict(self, sequence):
        # Add to batch queue
        future = asyncio.Future()
        self.pending_requests.append((sequence, future))
        
        # Trigger batch processing if needed
        if len(self.pending_requests) >= self.max_batch_size:
            await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        if not self.pending_requests:
            return
        
        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]
        
        sequences = [req[0] for req in batch]
        futures = [req[1] for req in batch]
        
        # Batch prediction
        predictions = self._batch_predict(sequences)
        
        # Set results
        for future, prediction in zip(futures, predictions):
            future.set_result(prediction)
```

#### Caching Strategies
```python
import redis
import pickle
import hashlib

class PredictionCache:
    def __init__(self, redis_client, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def get_cache_key(self, sequence):
        return hashlib.md5(sequence.encode()).hexdigest()
    
    def get_prediction(self, sequence):
        key = self.get_cache_key(sequence)
        cached = self.redis.get(key)
        if cached:
            return pickle.loads(cached)
        return None
    
    def cache_prediction(self, sequence, prediction):
        key = self.get_cache_key(sequence)
        self.redis.setex(key, self.ttl, pickle.dumps(prediction))

# Usage in service
cache = PredictionCache(redis.Redis(host='localhost'))

def predict_with_cache(sequence):
    # Check cache first
    cached_result = cache.get_prediction(sequence)
    if cached_result:
        return cached_result
    
    # Compute prediction
    result = model.predict(sequence)
    
    # Cache result
    cache.cache_prediction(sequence, result)
    
    return result
```

### Memory Optimization

#### Model Sharding
```python
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed environment
init_process_group(backend='nccl')

# Shard large model across GPUs
model = HyenaGLT(large_config)
model = DistributedDataParallel(model)

# Model parallelism for very large models
class ShardedHyenaGLT(nn.Module):
    def __init__(self, config, num_shards=4):
        super().__init__()
        self.num_shards = num_shards
        self.shards = nn.ModuleList([
            HyenaGLTShard(config, shard_idx=i) 
            for i in range(num_shards)
        ])
    
    def forward(self, x):
        for shard in self.shards:
            x = shard(x)
        return x
```

#### Gradient Checkpointing
```python
# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Custom checkpointing for specific layers
from torch.utils.checkpoint import checkpoint

class CheckpointedHyenaLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return checkpoint(self.layer, x)
```

## Monitoring and Maintenance

### Application Monitoring

#### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction latency')
MODEL_MEMORY_USAGE = Gauge('model_memory_bytes', 'Model memory usage')
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')

class MonitoredPredictor:
    def __init__(self, model):
        self.model = model
    
    def predict(self, sequence):
        ACTIVE_REQUESTS.inc()
        start_time = time.time()
        
        try:
            result = self.model.predict(sequence)
            PREDICTION_COUNTER.inc()
            return result
        finally:
            PREDICTION_DURATION.observe(time.time() - start_time)
            ACTIVE_REQUESTS.dec()

# Start metrics server
start_http_server(8080)
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Hyena-GLT Monitoring",
    "panels": [
      {
        "title": "Prediction Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(predictions_total[5m])",
            "legendFormat": "Predictions/sec"
          }
        ]
      },
      {
        "title": "Latency Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.5, prediction_duration_seconds_bucket)",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, prediction_duration_seconds_bucket)",
            "legendFormat": "p95"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "model_memory_bytes",
            "legendFormat": "Memory"
          }
        ]
      }
    ]
  }
}
```

### Model Performance Monitoring

#### Drift Detection
```python
import numpy as np
from scipy import stats

class ModelDriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold
    
    def detect_drift(self, new_data):
        # Kolmogorov-Smirnov test for distribution drift
        statistic, p_value = stats.ks_2samp(
            self.reference_data, new_data
        )
        
        drift_detected = p_value < self.threshold
        
        return {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'statistic': statistic
        }
    
    def update_baseline(self, new_reference_data):
        self.reference_data = new_reference_data

# Usage
detector = ModelDriftDetector(training_predictions)
current_predictions = get_recent_predictions()
drift_result = detector.detect_drift(current_predictions)

if drift_result['drift_detected']:
    send_alert("Model drift detected!")
```

#### A/B Testing
```python
import random

class ABTestManager:
    def __init__(self, model_a, model_b, traffic_split=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.results = {'a': [], 'b': []}
    
    def predict(self, sequence):
        # Route traffic
        use_model_a = random.random() < self.traffic_split
        
        if use_model_a:
            result = self.model_a.predict(sequence)
            self.results['a'].append(result)
            model_version = 'a'
        else:
            result = self.model_b.predict(sequence)
            self.results['b'].append(result)
            model_version = 'b'
        
        # Log for analysis
        self.log_prediction(sequence, result, model_version)
        
        return result
    
    def get_performance_metrics(self):
        # Calculate metrics for each model
        metrics_a = self.calculate_metrics(self.results['a'])
        metrics_b = self.calculate_metrics(self.results['b'])
        
        return {'model_a': metrics_a, 'model_b': metrics_b}
```

### Health Checks

#### Comprehensive Health Check
```python
import psutil
import torch

class HealthChecker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def check_health(self):
        checks = {
            'model_loaded': self._check_model(),
            'tokenizer_loaded': self._check_tokenizer(),
            'gpu_available': self._check_gpu(),
            'memory_usage': self._check_memory(),
            'inference_test': self._test_inference()
        }
        
        all_healthy = all(check['status'] == 'healthy' for check in checks.values())
        
        return {
            'overall_status': 'healthy' if all_healthy else 'unhealthy',
            'checks': checks,
            'timestamp': time.time()
        }
    
    def _check_model(self):
        try:
            return {
                'status': 'healthy' if self.model is not None else 'unhealthy',
                'details': f'Model type: {type(self.model)}'
            }
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def _check_gpu(self):
        if torch.cuda.is_available():
            return {
                'status': 'healthy',
                'details': f'GPU count: {torch.cuda.device_count()}'
            }
        return {'status': 'info', 'details': 'No GPU available'}
    
    def _test_inference(self):
        try:
            test_sequence = "ATCGATCG"
            start_time = time.time()
            result = self.predict_test(test_sequence)
            inference_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'inference_time': inference_time,
                'test_result': result
            }
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
```

## Security Considerations

### Input Validation

```python
import re
from typing import List

class SequenceValidator:
    def __init__(self, max_length=10000):
        self.max_length = max_length
        self.dna_pattern = re.compile(r'^[ATGC]+$', re.IGNORECASE)
        self.rna_pattern = re.compile(r'^[AUGC]+$', re.IGNORECASE)
        self.protein_pattern = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$', re.IGNORECASE)
    
    def validate_dna(self, sequence: str) -> dict:
        return self._validate_sequence(sequence, self.dna_pattern, "DNA")
    
    def validate_rna(self, sequence: str) -> dict:
        return self._validate_sequence(sequence, self.rna_pattern, "RNA")
    
    def validate_protein(self, sequence: str) -> dict:
        return self._validate_sequence(sequence, self.protein_pattern, "Protein")
    
    def _validate_sequence(self, sequence: str, pattern: re.Pattern, seq_type: str) -> dict:
        # Length check
        if len(sequence) > self.max_length:
            return {
                'valid': False,
                'error': f'Sequence too long: {len(sequence)} > {self.max_length}'
            }
        
        # Empty check
        if not sequence:
            return {'valid': False, 'error': 'Empty sequence'}
        
        # Pattern check
        if not pattern.match(sequence):
            return {
                'valid': False,
                'error': f'Invalid {seq_type} sequence: contains invalid characters'
            }
        
        return {'valid': True, 'length': len(sequence)}

# Usage in API
validator = SequenceValidator()

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Validate input
    validation = validator.validate_dna(request.sequence)
    if not validation['valid']:
        raise HTTPException(status_code=400, detail=validation['error'])
    
    # Proceed with prediction
    return await make_prediction(request.sequence)
```

### Rate Limiting

```python
from fastapi import HTTPException
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=1000, window_seconds=3600)

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await call_next(request)
    return response
```

### Authentication and Authorization

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        user_id = payload.get('user_id')
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    user_id: str = Depends(verify_token)
):
    # User is authenticated
    return await make_prediction(request.sequence, user_id)
```

## Scaling Strategies

### Horizontal Scaling

#### Load Balancing
```yaml
# nginx.conf for load balancing
upstream hyena_glt_backend {
    least_conn;
    server hyena-glt-1:8000 weight=1;
    server hyena-glt-2:8000 weight=1;
    server hyena-glt-3:8000 weight=1;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://hyena_glt_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://hyena_glt_backend;
    }
}
```

#### Auto-scaling
```python
# AWS Auto Scaling configuration
import boto3

autoscaling = boto3.client('autoscaling')

autoscaling.create_auto_scaling_group(
    AutoScalingGroupName='hyena-glt-asg',
    LaunchTemplate={
        'LaunchTemplateName': 'hyena-glt-template',
        'Version': '$Latest'
    },
    MinSize=2,
    MaxSize=20,
    DesiredCapacity=5,
    TargetGroupARNs=[
        'arn:aws:elasticloadbalancing:region:account:targetgroup/hyena-glt-tg'
    ],
    HealthCheckType='ELB',
    HealthCheckGracePeriod=300
)

# Scaling policies
autoscaling.put_scaling_policy(
    AutoScalingGroupName='hyena-glt-asg',
    PolicyName='scale-up',
    PolicyType='TargetTrackingScaling',
    TargetTrackingConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'ASGAverageCPUUtilization'
        }
    }
)
```

### Vertical Scaling

#### GPU Utilization
```python
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel

class MultiGPUInferenceServer:
    def __init__(self, model_path, num_gpus=None):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        
        self.num_gpus = num_gpus
        self.models = []
        
        # Load model on each GPU
        for gpu_id in range(num_gpus):
            model = HyenaGLT.load_from_checkpoint(model_path)
            model = model.cuda(gpu_id)
            model.eval()
            self.models.append(model)
        
        self.current_gpu = 0
    
    def predict(self, sequences):
        # Round-robin GPU assignment
        gpu_id = self.current_gpu % self.num_gpus
        self.current_gpu += 1
        
        model = self.models[gpu_id]
        
        # Process on specific GPU
        with torch.cuda.device(gpu_id):
            return self._predict_batch(model, sequences)
```

#### Memory-Efficient Inference
```python
class MemoryEfficientInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    
    def predict(self, sequence):
        # Lazy loading to save memory
        if self.model is None:
            self.model = HyenaGLT.load_from_checkpoint(self.model_path)
            self.model.eval()
        
        # Clear cache before inference
        torch.cuda.empty_cache()
        
        try:
            with torch.no_grad():
                result = self.model.predict(sequence)
            return result
        finally:
            # Clear cache after inference
            torch.cuda.empty_cache()
    
    def __del__(self):
        # Cleanup on destruction
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
```

## Troubleshooting

### Common Deployment Issues

#### Out of Memory Errors
```python
# Solution: Implement memory monitoring
import psutil
import gc

def check_memory_usage():
    memory = psutil.virtual_memory()
    gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    if memory.percent > 90:
        print(f"High memory usage: {memory.percent}%")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return {
        'cpu_memory_percent': memory.percent,
        'gpu_memory_mb': gpu_memory / 1024 / 1024
    }

# Call periodically
import threading
import time

def memory_monitor():
    while True:
        check_memory_usage()
        time.sleep(60)

threading.Thread(target=memory_monitor, daemon=True).start()
```

#### Slow Inference
```python
# Solution: Profile and optimize
import torch.profiler

def profile_inference(model, sample_input):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            _ = model(sample_input)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return prof

# Optimize based on profiling results
def optimize_model(model):
    # Enable optimizations
    model = torch.jit.optimize_for_inference(model)
    model = torch.compile(model, mode="max-autotune")
    return model
```

#### Model Loading Failures
```python
# Solution: Robust model loading
import os
import hashlib

def verify_model_file(model_path, expected_hash=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        raise ValueError(f"Model file is empty: {model_path}")
    
    if expected_hash:
        with open(model_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash != expected_hash:
            raise ValueError(f"Model file checksum mismatch: {file_hash} != {expected_hash}")

def robust_model_loading(model_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            verify_model_file(model_path)
            model = HyenaGLT.load_from_checkpoint(model_path)
            return model
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Model loading attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Debugging Tools

#### Request Tracing
```python
import uuid
import logging
from contextvars import ContextVar

# Request ID context
request_id_ctx: ContextVar[str] = ContextVar('request_id')

class RequestTracer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def trace_request(self, func):
        def wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())
            request_id_ctx.set(request_id)
            
            self.logger.info(f"[{request_id}] Starting request")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.logger.info(f"[{request_id}] Request completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                self.logger.error(f"[{request_id}] Request failed after {duration:.3f}s: {e}")
                raise
        
        return wrapper

# Usage
tracer = RequestTracer()

@tracer.trace_request
def predict(sequence):
    return model.predict(sequence)
```

#### Performance Diagnostics
```python
class PerformanceDiagnostics:
    def __init__(self):
        self.metrics = []
    
    def diagnose_performance(self, model, test_sequences):
        results = {
            'sequence_lengths': [],
            'inference_times': [],
            'memory_usage': [],
            'gpu_utilization': []
        }
        
        for sequence in test_sequences:
            # Measure inference time
            start_time = time.time()
            torch.cuda.synchronize()  # Ensure GPU operations complete
            
            with torch.no_grad():
                _ = model.predict(sequence)
            
            torch.cuda.synchronize()
            inference_time = time.time() - start_time
            
            # Collect metrics
            results['sequence_lengths'].append(len(sequence))
            results['inference_times'].append(inference_time)
            results['memory_usage'].append(torch.cuda.memory_allocated())
            results['gpu_utilization'].append(self._get_gpu_utilization())
        
        return self._analyze_results(results)
    
    def _analyze_results(self, results):
        import numpy as np
        
        analysis = {
            'avg_inference_time': np.mean(results['inference_times']),
            'p95_inference_time': np.percentile(results['inference_times'], 95),
            'time_vs_length_correlation': np.corrcoef(
                results['sequence_lengths'], 
                results['inference_times']
            )[0, 1],
            'memory_efficiency': np.mean(results['memory_usage']),
            'recommendations': []
        }
        
        # Generate recommendations
        if analysis['avg_inference_time'] > 1.0:
            analysis['recommendations'].append("Consider model optimization")
        
        if analysis['time_vs_length_correlation'] > 0.8:
            analysis['recommendations'].append("Consider sequence bucketing")
        
        return analysis
```

This deployment guide provides comprehensive coverage of production deployment strategies for Hyena-GLT models. It includes optimization techniques, serving infrastructure, cloud deployment options, monitoring, security, and troubleshooting guidance to ensure successful production deployments.
