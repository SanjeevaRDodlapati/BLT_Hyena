# 06 - Production Deployment

**Estimated Time:** 50 minutes  
**Prerequisites:** [05 - Evaluation](05_EVALUATION.md)  
**Next:** [07 - Advanced Topics](07_ADVANCED.md)

## Overview

This tutorial covers deploying BLT_Hyena models in production environments, from model optimization and containerization to scalable inference systems and monitoring. You'll learn to build robust, efficient, and maintainable genomic analysis services.

## What You'll Learn

- Model optimization and quantization for production
- Containerization with Docker and Kubernetes
- Building REST APIs and microservices
- Implementing batch processing pipelines
- Setting up monitoring and logging systems
- Handling scale and performance requirements
- Security and compliance considerations

## Model Optimization for Production

### Model Quantization and Optimization

```python
import torch
import torch.quantization as quantization
from torch.jit import script, trace
import onnx
import onnxruntime as ort
from hyena_glt import HyenaGLT, HyenaGLTConfig

class ModelOptimizer:
    """Optimize BLT_Hyena models for production deployment"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimized_models = {}
    
    def optimize_with_torchscript(self, example_input=None):
        """Optimize model using TorchScript"""
        
        print("Optimizing model with TorchScript...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        if example_input is None:
            # Create example input
            batch_size, seq_len = 1, 512
            example_input = torch.randint(
                0, self.config.vocab_size, 
                (batch_size, seq_len)
            )
        
        # Trace the model
        try:
            traced_model = trace(self.model, example_input)
            self.optimized_models['torchscript'] = traced_model
            print("✓ TorchScript optimization successful")
            
            # Test the traced model
            with torch.no_grad():
                original_output = self.model(example_input)
                traced_output = traced_model(example_input)
                
                # Check if outputs match
                if torch.allclose(original_output.logits, traced_output.logits, atol=1e-5):
                    print("✓ TorchScript model outputs match original")
                else:
                    print("⚠ Warning: TorchScript outputs differ from original")
                    
        except Exception as e:
            print(f"✗ TorchScript optimization failed: {e}")
            
        return self.optimized_models.get('torchscript')
    
    def optimize_with_quantization(self, calibration_loader=None):
        """Apply dynamic quantization to the model"""
        
        print("Applying dynamic quantization...")
        
        try:
            # Apply dynamic quantization
            quantized_model = quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Embedding},  # Quantize these layer types
                dtype=torch.qint8
            )
            
            self.optimized_models['quantized'] = quantized_model
            print("✓ Dynamic quantization successful")
            
            # Measure model size reduction
            original_size = self._get_model_size(self.model)
            quantized_size = self._get_model_size(quantized_model)
            reduction = (1 - quantized_size / original_size) * 100
            
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Quantized size: {quantized_size:.2f} MB")
            print(f"  Size reduction: {reduction:.1f}%")
            
        except Exception as e:
            print(f"✗ Quantization failed: {e}")
            
        return self.optimized_models.get('quantized')
    
    def export_to_onnx(self, output_path="model.onnx", example_input=None):
        """Export model to ONNX format"""
        
        print("Exporting model to ONNX...")
        
        if example_input is None:
            batch_size, seq_len = 1, 512
            example_input = torch.randint(
                0, self.config.vocab_size,
                (batch_size, seq_len)
            )
        
        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                },
                opset_version=11,
                do_constant_folding=True
            )
            
            print(f"✓ ONNX export successful: {output_path}")
            
            # Verify ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model verification passed")
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(output_path)
            self.optimized_models['onnx'] = ort_session
            
            return output_path
            
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            return None
    
    def benchmark_models(self, test_input, num_iterations=100):
        """Benchmark different model optimizations"""
        
        print("Benchmarking model optimizations...")
        
        results = {}
        
        # Benchmark original model
        results['original'] = self._benchmark_model(
            self.model, test_input, num_iterations
        )
        
        # Benchmark optimized models
        for name, model in self.optimized_models.items():
            if name == 'onnx':
                results[name] = self._benchmark_onnx_model(
                    model, test_input, num_iterations
                )
            else:
                results[name] = self._benchmark_model(
                    model, test_input, num_iterations
                )
        
        # Print results
        print("\nBenchmark Results:")
        print("-" * 50)
        for name, metrics in results.items():
            print(f"{name}:")
            print(f"  Avg inference time: {metrics['avg_time']:.4f}s")
            print(f"  Throughput: {metrics['throughput']:.1f} samples/s")
            if 'memory_mb' in metrics:
                print(f"  Memory usage: {metrics['memory_mb']:.1f} MB")
        
        return results
    
    def _get_model_size(self, model):
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _benchmark_model(self, model, test_input, num_iterations):
        """Benchmark PyTorch model"""
        import time
        import psutil
        import os
        
        model.eval()
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_input)
        
        # Benchmark
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(test_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        memory_after = process.memory_info().rss / 1024 / 1024
        
        avg_time = sum(times) / len(times)
        throughput = 1.0 / avg_time
        
        return {
            'avg_time': avg_time,
            'throughput': throughput,
            'memory_mb': memory_after - memory_before
        }
    
    def _benchmark_onnx_model(self, ort_session, test_input, num_iterations):
        """Benchmark ONNX model"""
        import time
        
        # Convert input to numpy
        input_np = test_input.numpy()
        times = []
        
        # Warmup
        for _ in range(5):
            _ = ort_session.run(None, {'input_ids': input_np})
        
        # Benchmark
        for _ in range(num_iterations):
            start_time = time.time()
            _ = ort_session.run(None, {'input_ids': input_np})
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        throughput = 1.0 / avg_time
        
        return {
            'avg_time': avg_time,
            'throughput': throughput
        }

# Example usage
config = HyenaGLTConfig(vocab_size=4, hidden_size=768, num_layers=12, use_hyena=True)
model = HyenaGLT(config)

optimizer = ModelOptimizer(model, config)

# Apply optimizations
torchscript_model = optimizer.optimize_with_torchscript()
quantized_model = optimizer.optimize_with_quantization()
onnx_path = optimizer.export_to_onnx("production_model.onnx")

# Benchmark
test_input = torch.randint(0, 4, (1, 512))
benchmark_results = optimizer.benchmark_models(test_input)
```

## Containerization and Deployment

### Docker Configuration

```dockerfile
# Dockerfile for BLT_Hyena production deployment
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn \
    fastapi \
    prometheus-client \
    redis \
    celery \
    onnxruntime-gpu

# Copy application code
COPY . .

# Install BLT_Hyena package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 bltuser && chown -R bltuser:bltuser /app
USER bltuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "api.main:app"]
```

```yaml
# docker-compose.yml for local development and testing
version: '3.8'

services:
  blt-hyena-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/production_model.pt
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  celery-worker:
    build: .
    command: celery -A api.celery_app worker --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/app/models/production_model.pt
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  flower:
    build: .
    command: celery -A api.celery_app flower
    ports:
      - "5555:5555"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

volumes:
  redis_data:
```

### FastAPI Production Service

```python
# api/main.py - Production FastAPI service
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import logging
import time
import uuid
from contextlib import asynccontextmanager
import asyncio
import redis
import json
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
INFERENCE_COUNT = Counter('inference_requests_total', 'Total inference requests')
INFERENCE_DURATION = Histogram('inference_duration_seconds', 'Inference duration')

# Global model instance
model_instance = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global model_instance, redis_client
    
    logger.info("Loading BLT_Hyena model...")
    model_instance = await load_production_model()
    
    logger.info("Connecting to Redis...")
    redis_client = redis.Redis.from_url("redis://redis:6379", decode_responses=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if redis_client:
        redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="BLT_Hyena Genomic Analysis API",
    description="Production API for BLT_Hyena genomic sequence analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request/Response models
class SequenceInput(BaseModel):
    sequence: str = Field(..., description="DNA sequence (ATGC)")
    task_type: str = Field("classification", description="Task type: classification, generation, variant_calling")
    max_length: Optional[int] = Field(4096, description="Maximum sequence length")

class ClassificationResult(BaseModel):
    predictions: List[float]
    predicted_class: int
    confidence: float
    processing_time: float

class GenerationResult(BaseModel):
    generated_sequence: str
    prompt: str
    processing_time: float

class VariantResult(BaseModel):
    variants: List[Dict[str, Any]]
    variant_count: int
    processing_time: float

class BatchRequest(BaseModel):
    sequences: List[SequenceInput]
    batch_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    gpu_available: bool

# Model loading
async def load_production_model():
    """Load optimized production model"""
    
    model_path = "/app/models/production_model.pt"
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        from hyena_glt import HyenaGLT
        model = HyenaGLT(checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        logger.info(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Dependency injection
async def get_model():
    """Get model instance"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_instance

async def get_redis():
    """Get Redis client"""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not connected")
    return redis_client

# Tokenizer (you might want to cache this as well)
def get_tokenizer():
    from hyena_glt.data import GenomicTokenizer
    return GenomicTokenizer()

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    model_loaded = model_instance is not None
    redis_connected = redis_client is not None and redis_client.ping()
    gpu_available = torch.cuda.is_available()
    
    status = "healthy" if all([model_loaded, redis_connected]) else "unhealthy"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        redis_connected=redis_connected,
        gpu_available=gpu_available
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/classify", response_model=ClassificationResult)
async def classify_sequence(
    request: SequenceInput,
    model: Any = Depends(get_model)
):
    """Classify a genomic sequence"""
    
    start_time = time.time()
    INFERENCE_COUNT.inc()
    
    try:
        # Tokenize sequence
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(request.sequence)
        
        # Truncate if necessary
        if len(tokens) > request.max_length:
            tokens = tokens[:request.max_length]
        
        # Convert to tensor
        input_tensor = torch.tensor([tokens]).to(model.device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits
            
            # Get predictions
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
        processing_time = time.time() - start_time
        INFERENCE_DURATION.observe(processing_time)
        
        return ClassificationResult(
            predictions=probabilities[0].tolist(),
            predicted_class=predicted_class,
            confidence=confidence,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GenerationResult)
async def generate_sequence(
    request: SequenceInput,
    model: Any = Depends(get_model)
):
    """Generate genomic sequence"""
    
    start_time = time.time()
    INFERENCE_COUNT.inc()
    
    try:
        # Tokenize prompt
        tokenizer = get_tokenizer()
        prompt_tokens = tokenizer.encode(request.sequence)
        
        # Convert to tensor
        input_tensor = torch.tensor([prompt_tokens]).to(model.device)
        
        # Generate
        with torch.no_grad():
            generated_tokens = model.generate(
                input_tensor,
                max_length=request.max_length,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode generated sequence
            generated_sequence = tokenizer.decode(generated_tokens[0])
        
        processing_time = time.time() - start_time
        INFERENCE_DURATION.observe(processing_time)
        
        return GenerationResult(
            generated_sequence=generated_sequence,
            prompt=request.sequence,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def process_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    redis: Any = Depends(get_redis)
):
    """Process batch of sequences asynchronously"""
    
    batch_id = request.batch_id or str(uuid.uuid4())
    
    # Store batch in Redis
    batch_data = {
        'id': batch_id,
        'status': 'pending',
        'sequences': [seq.dict() for seq in request.sequences],
        'created_at': time.time(),
        'total_sequences': len(request.sequences)
    }
    
    redis.setex(f"batch:{batch_id}", 3600, json.dumps(batch_data))  # 1 hour TTL
    
    # Queue batch processing
    background_tasks.add_task(process_batch_async, batch_id, request.sequences)
    
    return {
        'batch_id': batch_id,
        'status': 'queued',
        'total_sequences': len(request.sequences)
    }

@app.get("/batch/{batch_id}")
async def get_batch_status(
    batch_id: str,
    redis: Any = Depends(get_redis)
):
    """Get batch processing status"""
    
    batch_data = redis.get(f"batch:{batch_id}")
    if not batch_data:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    return json.loads(batch_data)

async def process_batch_async(batch_id: str, sequences: List[SequenceInput]):
    """Process batch asynchronously"""
    
    try:
        # Update status
        batch_data = json.loads(redis_client.get(f"batch:{batch_id}"))
        batch_data['status'] = 'processing'
        batch_data['started_at'] = time.time()
        redis_client.setex(f"batch:{batch_id}", 3600, json.dumps(batch_data))
        
        # Process sequences
        results = []
        for i, seq in enumerate(sequences):
            try:
                if seq.task_type == "classification":
                    result = await classify_sequence(seq, model_instance)
                elif seq.task_type == "generation":
                    result = await generate_sequence(seq, model_instance)
                else:
                    result = {"error": f"Unsupported task type: {seq.task_type}"}
                
                results.append(result.dict() if hasattr(result, 'dict') else result)
                
                # Update progress
                batch_data['processed'] = i + 1
                batch_data['progress'] = (i + 1) / len(sequences) * 100
                redis_client.setex(f"batch:{batch_id}", 3600, json.dumps(batch_data))
                
            except Exception as e:
                logger.error(f"Error processing sequence {i}: {e}")
                results.append({"error": str(e)})
        
        # Update final status
        batch_data['status'] = 'completed'
        batch_data['completed_at'] = time.time()
        batch_data['results'] = results
        redis_client.setex(f"batch:{batch_id}", 3600, json.dumps(batch_data))
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        batch_data['status'] = 'failed'
        batch_data['error'] = str(e)
        redis_client.setex(f"batch:{batch_id}", 3600, json.dumps(batch_data))

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(process_time)
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Kubernetes Deployment

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: blt-hyena

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: blt-hyena-config
  namespace: blt-hyena
data:
  MODEL_PATH: "/app/models/production_model.pt"
  LOG_LEVEL: "INFO"
  REDIS_URL: "redis://redis-service:6379"
  MAX_WORKERS: "4"

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: blt-hyena-api
  namespace: blt-hyena
  labels:
    app: blt-hyena-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: blt-hyena-api
  template:
    metadata:
      labels:
        app: blt-hyena-api
    spec:
      containers:
      - name: api
        image: blt-hyena:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          valueFrom:
            configMapKeyRef:
              name: blt-hyena-config
              key: MODEL_PATH
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: blt-hyena-config
              key: REDIS_URL
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: blt-hyena-service
  namespace: blt-hyena
spec:
  selector:
    app: blt-hyena-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: blt-hyena-hpa
  namespace: blt-hyena
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: blt-hyena-api
  minReplicas: 2
  maxReplicas: 10
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

---
# k8s/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: blt-hyena
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: blt-hyena
spec:
  selector:
    app: redis
  ports:
  - protocol: TCP
    port: 6379
    targetPort: 6379
```

## Batch Processing Pipeline

### Celery Integration for Large-Scale Processing

```python
# api/celery_app.py - Celery configuration for batch processing
from celery import Celery
import os
import logging
from hyena_glt import HyenaGLT
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    'blt_hyena_worker',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379'),
    include=['api.tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'api.tasks.process_genomic_sequence': {'queue': 'genomic'},
        'api.tasks.batch_variant_calling': {'queue': 'variants'},
        'api.tasks.generate_sequences': {'queue': 'generation'},
    },
    worker_prefetch_multiplier=1,  # Important for GPU workers
    task_acks_late=True,
    worker_disable_rate_limits=True
)

# Global model instance for workers
model_instance = None

def load_worker_model():
    """Load model in worker process"""
    global model_instance
    
    if model_instance is None:
        model_path = os.getenv('MODEL_PATH', '/app/models/production_model.pt')
        
        try:
            checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model_instance = HyenaGLT(checkpoint['model_config'])
            model_instance.load_state_dict(checkpoint['model_state_dict'])
            model_instance.eval()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_instance.to(device)
            
            logger.info(f"Worker model loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load worker model: {e}")
            raise
    
    return model_instance

# api/tasks.py - Celery tasks
from .celery_app import celery_app, load_worker_model
from hyena_glt.data import GenomicTokenizer
import torch
import time
import numpy as np

@celery_app.task(bind=True)
def process_genomic_sequence(self, sequence_data):
    """Process a single genomic sequence"""
    
    try:
        # Load model
        model = load_worker_model()
        tokenizer = GenomicTokenizer()
        
        # Extract data
        sequence = sequence_data['sequence']
        task_type = sequence_data.get('task_type', 'classification')
        max_length = sequence_data.get('max_length', 4096)
        
        # Tokenize
        tokens = tokenizer.encode(sequence)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        input_tensor = torch.tensor([tokens]).to(model.device)
        
        # Process based on task type
        start_time = time.time()
        
        with torch.no_grad():
            if task_type == 'classification':
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
                
                result = {
                    'task_type': task_type,
                    'predictions': probabilities[0].tolist(),
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'processing_time': time.time() - start_time
                }
                
            elif task_type == 'generation':
                generated_tokens = model.generate(
                    input_tensor,
                    max_length=max_length,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                generated_sequence = tokenizer.decode(generated_tokens[0])
                
                result = {
                    'task_type': task_type,
                    'generated_sequence': generated_sequence,
                    'prompt': sequence,
                    'processing_time': time.time() - start_time
                }
                
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
        
        return result
        
    except Exception as e:
        logger.error(f"Task failed: {e}")
        self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True)
def batch_variant_calling(self, sequence_batch):
    """Process batch of sequences for variant calling"""
    
    try:
        model = load_worker_model()
        tokenizer = GenomicTokenizer()
        
        results = []
        
        for seq_data in sequence_batch:
            sequence = seq_data['sequence']
            reference = seq_data.get('reference', '')
            
            # Tokenize sequences
            seq_tokens = tokenizer.encode(sequence)
            ref_tokens = tokenizer.encode(reference) if reference else None
            
            # Process with model
            input_tensor = torch.tensor([seq_tokens]).to(model.device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                # Custom variant calling logic here
                variants = detect_variants(outputs, sequence, reference)
                
                results.append({
                    'sequence_id': seq_data.get('id'),
                    'variants': variants,
                    'variant_count': len(variants)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch variant calling failed: {e}")
        self.retry(countdown=60, max_retries=3)

def detect_variants(model_outputs, sequence, reference):
    """Custom variant detection logic"""
    # Implement your variant calling algorithm
    # This is a placeholder implementation
    
    variants = []
    
    # Example: simple difference detection
    if reference:
        for i, (s, r) in enumerate(zip(sequence, reference)):
            if s != r:
                variants.append({
                    'position': i,
                    'reference': r,
                    'alternate': s,
                    'type': 'SNV',
                    'confidence': 0.95  # Placeholder confidence
                })
    
    return variants

# Workflow orchestration
@celery_app.task
def process_large_genomic_dataset(dataset_path, output_path, task_type='classification'):
    """Process large genomic dataset using workflow"""
    
    from celery import group, chord
    
    # Load dataset
    sequences = load_genomic_dataset(dataset_path)
    batch_size = 100
    
    # Create batches
    batches = [sequences[i:i+batch_size] for i in range(0, len(sequences), batch_size)]
    
    # Create batch processing jobs
    job = group(
        process_batch_sequences.s(batch, task_type) 
        for batch in batches
    )
    
    # Execute and collect results
    result = job.apply_async()
    
    # Wait for completion and aggregate results
    all_results = []
    for batch_result in result.get():
        all_results.extend(batch_result)
    
    # Save results
    save_results(all_results, output_path)
    
    return {
        'total_sequences': len(sequences),
        'batches_processed': len(batches),
        'output_path': output_path
    }

@celery_app.task
def process_batch_sequences(sequence_batch, task_type):
    """Process a batch of sequences"""
    
    results = []
    
    for seq_data in sequence_batch:
        result = process_genomic_sequence.delay(seq_data)
        results.append(result.get())
    
    return results

def load_genomic_dataset(dataset_path):
    """Load genomic dataset from file"""
    # Implement dataset loading logic
    pass

def save_results(results, output_path):
    """Save processing results"""
    import json
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
```

## Monitoring and Observability

### Prometheus Metrics and Grafana Dashboard

```python
# monitoring/metrics.py - Custom metrics collection
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
import psutil
import torch
import time
import threading

class BLTHyenaMetrics:
    """Custom metrics collector for BLT_Hyena"""
    
    def __init__(self, model=None):
        self.model = model
        self.registry = CollectorRegistry()
        
        # Application metrics
        self.inference_requests = Counter(
            'blt_hyena_inference_requests_total',
            'Total inference requests',
            ['task_type', 'status'],
            registry=self.registry
        )
        
        self.inference_duration = Histogram(
            'blt_hyena_inference_duration_seconds',
            'Inference duration',
            ['task_type'],
            registry=self.registry
        )
        
        self.sequence_length = Histogram(
            'blt_hyena_sequence_length',
            'Input sequence length',
            buckets=[100, 500, 1000, 2000, 4000, 8000],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'blt_hyena_model_accuracy',
            'Model accuracy on validation set',
            registry=self.registry
        )
        
        # System metrics
        self.gpu_memory_usage = Gauge(
            'blt_hyena_gpu_memory_bytes',
            'GPU memory usage',
            ['device'],
            registry=self.registry
        )
        
        self.gpu_utilization = Gauge(
            'blt_hyena_gpu_utilization_percent',
            'GPU utilization percentage',
            ['device'],
            registry=self.registry
        )
        
        self.model_info = Info(
            'blt_hyena_model_info',
            'Model information',
            registry=self.registry
        )
        
        # Start background metric collection
        self.start_system_metrics_collection()
        
        # Set model info
        if self.model:
            self.set_model_info()
    
    def record_inference(self, task_type, duration, sequence_length, status='success'):
        """Record inference metrics"""
        self.inference_requests.labels(task_type=task_type, status=status).inc()
        self.inference_duration.labels(task_type=task_type).observe(duration)
        self.sequence_length.observe(sequence_length)
    
    def update_model_accuracy(self, accuracy):
        """Update model accuracy metric"""
        self.model_accuracy.set(accuracy)
    
    def start_system_metrics_collection(self):
        """Start background thread for system metrics"""
        def collect_system_metrics():
            while True:
                try:
                    # GPU metrics
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            device = f"cuda:{i}"
                            
                            # Memory usage
                            memory_allocated = torch.cuda.memory_allocated(i)
                            self.gpu_memory_usage.labels(device=device).set(memory_allocated)
                            
                            # GPU utilization (requires nvidia-ml-py)
                            try:
                                import pynvml
                                pynvml.nvmlInit()
                                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                self.gpu_utilization.labels(device=device).set(utilization.gpu)
                            except ImportError:
                                pass  # pynvml not available
                    
                    time.sleep(10)  # Collect every 10 seconds
                    
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def set_model_info(self):
        """Set model information"""
        if self.model:
            info = {
                'architecture': 'BLT_Hyena',
                'num_parameters': str(sum(p.numel() for p in self.model.parameters())),
                'num_layers': str(self.model.config.num_layers),
                'hidden_size': str(self.model.config.hidden_size),
                'vocab_size': str(self.model.config.vocab_size),
                'use_hyena': str(getattr(self.model.config, 'use_hyena', False))
            }
            self.model_info.info(info)
```

```yaml
# monitoring/grafana-dashboard.json (excerpt)
{
  "dashboard": {
    "id": null,
    "title": "BLT_Hyena Production Monitoring",
    "panels": [
      {
        "title": "Inference Requests per Second",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(blt_hyena_inference_requests_total[5m])",
            "legendFormat": "{{task_type}}"
          }
        ]
      },
      {
        "title": "Inference Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, blt_hyena_inference_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, blt_hyena_inference_duration_seconds_bucket)",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "blt_hyena_gpu_memory_bytes",
            "legendFormat": "{{device}}"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "blt_hyena_model_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      }
    ]
  }
}
```

## Security and Compliance

### Security Best Practices

```python
# security/auth.py - Authentication and authorization
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
from datetime import datetime, timedelta
import hashlib
import secrets

security = HTTPBearer()
SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
ALGORITHM = "HS256"

class SecurityManager:
    """Handle authentication and authorization"""
    
    def __init__(self):
        self.api_keys = self.load_api_keys()
        self.rate_limits = {}
    
    def load_api_keys(self):
        """Load API keys from secure storage"""
        # In production, load from secure key management service
        return {
            'client1': hashlib.sha256(b'secret_key_1').hexdigest(),
            'client2': hashlib.sha256(b'secret_key_2').hexdigest()
        }
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
            
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        return encoded_jwt
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify JWT token"""
        try:
            payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
            client_id: str = payload.get("sub")
            
            if client_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
                
            return client_id
            
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    
    def verify_api_key(self, api_key: str):
        """Verify API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for client_id, stored_hash in self.api_keys.items():
            if key_hash == stored_hash:
                return client_id
                
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    def check_rate_limit(self, client_id: str, max_requests: int = 100, window_minutes: int = 60):
        """Check rate limiting"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=window_minutes)
        
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []
        
        # Remove old requests
        self.rate_limits[client_id] = [
            req_time for req_time in self.rate_limits[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.rate_limits[client_id]) >= max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Record current request
        self.rate_limits[client_id].append(now)

security_manager = SecurityManager()

# Input validation and sanitization
from pydantic import validator
import re

class SecureSequenceInput(BaseModel):
    sequence: str = Field(..., min_length=1, max_length=100000)
    task_type: str = Field("classification")
    
    @validator('sequence')
    def validate_sequence(cls, v):
        """Validate genomic sequence"""
        # Remove whitespace
        v = re.sub(r'\s+', '', v.upper())
        
        # Check for valid nucleotides only
        if not re.match(r'^[ATGCN]+$', v):
            raise ValueError('Sequence must contain only valid nucleotides (A, T, G, C, N)')
        
        # Check for suspicious patterns
        if len(set(v)) == 1 and len(v) > 100:  # All same nucleotide
            raise ValueError('Suspicious sequence pattern detected')
        
        return v
    
    @validator('task_type')
    def validate_task_type(cls, v):
        """Validate task type"""
        allowed_types = ['classification', 'generation', 'variant_calling']
        if v not in allowed_types:
            raise ValueError(f'Task type must be one of: {allowed_types}')
        return v

# Data privacy and compliance
class DataPrivacyManager:
    """Handle data privacy and compliance"""
    
    def __init__(self):
        self.encryption_key = os.getenv('ENCRYPTION_KEY', secrets.token_bytes(32))
    
    def encrypt_sequence(self, sequence: str) -> str:
        """Encrypt sequence data for storage"""
        from cryptography.fernet import Fernet
        
        f = Fernet(self.encryption_key)
        encrypted = f.encrypt(sequence.encode())
        return encrypted.decode()
    
    def decrypt_sequence(self, encrypted_sequence: str) -> str:
        """Decrypt sequence data"""
        from cryptography.fernet import Fernet
        
        f = Fernet(self.encryption_key)
        decrypted = f.decrypt(encrypted_sequence.encode())
        return decrypted.decode()
    
    def anonymize_results(self, results: dict) -> dict:
        """Remove or hash sensitive information"""
        anonymized = results.copy()
        
        # Remove raw sequences from results
        if 'sequence' in anonymized:
            anonymized['sequence_hash'] = hashlib.sha256(
                anonymized['sequence'].encode()
            ).hexdigest()[:16]
            del anonymized['sequence']
        
        return anonymized

privacy_manager = DataPrivacyManager()
```

## Performance Optimization

### Production Optimization Checklist

```python
# performance/optimization.py - Production optimization utilities
import torch
import psutil
import GPUtil
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ProductionOptimizer:
    """Optimize BLT_Hyena for production deployment"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizations_applied = []
    
    def optimize_for_inference(self):
        """Apply all inference optimizations"""
        
        optimizations = [
            self.set_inference_mode,
            self.optimize_memory_usage,
            self.enable_cudnn_optimizations,
            self.set_precision_mode,
            self.optimize_threading
        ]
        
        for optimization in optimizations:
            try:
                optimization()
                self.optimizations_applied.append(optimization.__name__)
                logger.info(f"✓ Applied {optimization.__name__}")
            except Exception as e:
                logger.warning(f"✗ Failed to apply {optimization.__name__}: {e}")
        
        return self.optimizations_applied
    
    def set_inference_mode(self):
        """Set model to inference mode"""
        self.model.eval()
        
        # Disable gradient computation globally
        torch.set_grad_enabled(False)
        
        # Set model to inference mode (if supported)
        if hasattr(self.model, 'set_inference_mode'):
            self.model.set_inference_mode(True)
    
    def optimize_memory_usage(self):
        """Optimize memory usage"""
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set memory allocation strategy
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)  # Reserve 10% for other processes
        
        # Enable memory mapping for large models
        if hasattr(self.model, 'enable_memory_mapping'):
            self.model.enable_memory_mapping()
    
    def enable_cudnn_optimizations(self):
        """Enable cuDNN optimizations"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            torch.backends.cudnn.enabled = True
    
    def set_precision_mode(self, precision='fp16'):
        """Set precision mode for inference"""
        
        if precision == 'fp16' and torch.cuda.is_available():
            # Enable automatic mixed precision
            self.model.half()
            logger.info("Enabled FP16 precision")
            
        elif precision == 'int8':
            # Apply dynamic quantization
            from torch.quantization import quantize_dynamic
            self.model = quantize_dynamic(
                self.model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            logger.info("Enabled INT8 quantization")
    
    def optimize_threading(self):
        """Optimize threading for inference"""
        
        # Set optimal number of threads
        num_cores = psutil.cpu_count(logical=False)
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(num_cores)
        
        logger.info(f"Set torch threads to {num_cores}")
    
    def benchmark_performance(self, test_inputs, num_runs=100):
        """Benchmark model performance"""
        
        import time
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(test_inputs[0])
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(test_inputs[0])
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput = 1.0 / avg_time
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
        else:
            memory_allocated = 0
            memory_cached = 0
        
        return {
            'avg_inference_time': avg_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'throughput_samples_per_sec': throughput,
            'gpu_memory_allocated_mb': memory_allocated,
            'gpu_memory_cached_mb': memory_cached,
            'optimizations_applied': self.optimizations_applied
        }
    
    def generate_optimization_report(self, benchmark_results):
        """Generate optimization report"""
        
        report = f"""
# BLT_Hyena Production Optimization Report

## System Information
- GPU Available: {torch.cuda.is_available()}
- GPU Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}
- CPU Cores: {psutil.cpu_count()}
- Available RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB

## Applied Optimizations
{chr(10).join('- ' + opt for opt in self.optimizations_applied)}

## Performance Metrics
- Average Inference Time: {benchmark_results['avg_inference_time']:.4f}s
- Throughput: {benchmark_results['throughput_samples_per_sec']:.1f} samples/sec
- GPU Memory Usage: {benchmark_results['gpu_memory_allocated_mb']:.1f} MB
- GPU Memory Cached: {benchmark_results['gpu_memory_cached_mb']:.1f} MB

## Recommendations
"""
        
        # Add recommendations based on results
        if benchmark_results['avg_inference_time'] > 1.0:
            report += "- Consider using quantization or model pruning\n"
        
        if benchmark_results['gpu_memory_allocated_mb'] > 4000:
            report += "- Consider reducing batch size or sequence length\n"
        
        if 'set_precision_mode' not in self.optimizations_applied:
            report += "- Consider enabling mixed precision training\n"
        
        return report

# Usage example
optimizer = ProductionOptimizer(model, model_config)
applied_optimizations = optimizer.optimize_for_inference()

# Benchmark performance
test_input = torch.randint(0, 4, (1, 512)).cuda() if torch.cuda.is_available() else torch.randint(0, 4, (1, 512))
benchmark_results = optimizer.benchmark_performance([test_input])

# Generate report
optimization_report = optimizer.generate_optimization_report(benchmark_results)
print(optimization_report)
```

## Key Takeaways

1. **Model Optimization**: Apply quantization, TorchScript, and ONNX for production efficiency
2. **Containerization**: Use Docker and Kubernetes for scalable deployment
3. **API Design**: Build robust REST APIs with proper error handling and monitoring
4. **Batch Processing**: Implement Celery for large-scale genomic analysis workflows
5. **Monitoring**: Set up comprehensive metrics and alerting systems
6. **Security**: Implement authentication, authorization, and data privacy measures
7. **Performance**: Optimize for inference speed and memory efficiency

## Troubleshooting

### Common Production Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use model quantization
   - Implement sequence chunking

2. **Slow Inference**
   - Enable mixed precision
   - Use TorchScript compilation
   - Optimize threading settings
   - Consider model distillation

3. **Container Startup Issues**
   - Check GPU driver compatibility
   - Verify model file accessibility
   - Review resource limits
   - Check dependency versions

## Next Steps

Continue to [07 - Advanced Topics](07_ADVANCED.md) to explore research applications, custom architectures, and cutting-edge techniques with BLT_Hyena.

## Additional Resources

- [Production Deployment Guide](../docs/PRODUCTION_DEPLOYMENT.md)
- [Security Best Practices](../docs/SECURITY.md)
- [Performance Optimization](../docs/PERFORMANCE_OPTIMIZATION.md)
- [Monitoring Setup](../docs/MONITORING.md)
