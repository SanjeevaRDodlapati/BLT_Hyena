# Python 3.12 Migration Guide

## Overview

BLT_Hyena has been updated to require Python 3.11+ (recommended: Python 3.12) to ensure compatibility with the broader ecosystem of genomic AI tools.

## Version Requirements Alignment

| Repository | Python Requirement | Reason |
|------------|-------------------|---------|
| **BLT** | `==3.12.*` | Required for core BLT functionality |
| **Savanna** | `>=3.12` | Minimum version for distributed training |
| **Evo2** | `>=3.11` | Compatible with modern Python |
| **Vortex** | `>=3.10` | Compatible |
| **BLT_Hyena** | `>=3.11` | **UPDATED** from `>=3.8` |

## Migration Steps

### 1. Environment Setup

#### Option A: Conda (Recommended)
```bash
# Create new environment with Python 3.12
conda create -n hyena-glt python=3.12
conda activate hyena-glt

# Install BLT_Hyena with all dependencies
pip install -e .
```

#### Option B: pyenv + venv
```bash
# Install Python 3.12 if not available
pyenv install 3.12.0
pyenv local 3.12.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install BLT_Hyena
pip install -e .
```

### 2. Dependency Installation

```bash
# Core installation
pip install -e .

# Development tools
pip install -e .[dev]

# GPU support (if available)
pip install -e .[gpu]

# Notebooks support  
pip install -e .[notebooks]

# All optional dependencies
pip install -e .[dev,gpu,notebooks]
```

### 3. Validation

```bash
# Test Python version
python --version  # Should show 3.11+ or 3.12+

# Test core imports
python -c "import torch; import hyena_glt; print('✅ All imports successful')"

# Test distributed training readiness
python validate_cluster_readiness.py

# Run benchmark to verify performance
python benchmark_blt_performance.py
```

## Key Benefits of Python 3.12

### Performance Improvements
- **15% faster** on average compared to Python 3.11
- **Improved memory management** for large models
- **Better GIL handling** for multi-threaded workloads

### Language Features
- **Enhanced typing** support for better code clarity
- **Improved error messages** for easier debugging
- **f-string improvements** for better string formatting

### Ecosystem Compatibility
- **Native BLT support** - no compatibility workarounds needed
- **Latest PyTorch features** work optimally
- **Future-proof** for upcoming genomic AI tools

## Package Version Updates

The following packages have been updated for Python 3.12 compatibility:

| Package | Previous | Updated | Notes |
|---------|----------|---------|-------|
| `numpy` | `>=1.21.0` | `>=1.21.0` | Compatible |
| `torch` | `>=2.0.0` | `>=2.0.0` | Compatible | 
| `transformers` | `>=4.20.0` | `>=4.20.0` | Compatible |
| `omegaconf` | _(new)_ | `>=2.3.0` | Added for BLT compatibility |
| `sentencepiece` | _(new)_ | `>=0.2.0` | Added for tokenization |
| `psutil` | _(new)_ | `>=5.8.0` | Added for system monitoring |
| `pynvml` | _(new)_ | `>=11.0.0` | Added for GPU monitoring |

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError` for core packages
```bash
# Solution: Reinstall with updated requirements
pip install --upgrade -e .
```

#### Issue: CUDA compatibility problems
```bash
# Check CUDA version compatibility
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If issues persist, reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: Flash Attention installation fails
```bash
# Flash attention requires specific CUDA versions
# On systems without compatible CUDA, it will gracefully fallback
pip install flash-attn --no-build-isolation
```

### Platform-Specific Notes

#### macOS
- Flash Attention is automatically excluded (not compatible)
- Metal Performance Shaders (MPS) backend is used for GPU acceleration
- All core functionality works without GPU-specific packages

#### Linux with CUDA
- Full GPU acceleration available
- Triton and Flash Attention supported
- Optimal performance on cluster environments

#### Windows
- CUDA support available
- Some packages may require Visual Studio Build Tools
- WSL2 recommended for best compatibility

## Validation Checklist

- [ ] Python 3.11+ or 3.12 installed
- [ ] All core dependencies installed (`pip install -e .`)
- [ ] Core imports work (`python -c "import hyena_glt"`)
- [ ] Distributed training ready (`python validate_cluster_readiness.py`)
- [ ] GPU detection works (if applicable)
- [ ] Benchmark runs successfully (`python benchmark_blt_performance.py`)

## Next Steps

1. **Update your environment** to Python 3.12
2. **Reinstall dependencies** with updated requirements
3. **Test functionality** using validation scripts
4. **Deploy to cluster** using new distributed training capabilities

The migration ensures BLT_Hyena works seamlessly with the entire genomic AI ecosystem while providing access to the latest performance improvements and features.
