# Installation Troubleshooting

## Environment Setup Issues

### Python Version Compatibility

**Problem:** Incompatible Python version

**Solution:**
```bash
# Check Python version
python --version

# BLT_Hyena requires Python 3.8+
# Install compatible Python version
conda create -n blt_hyena python=3.9
conda activate blt_hyena
```

### Virtual Environment Problems

**Problem:** Package conflicts or missing dependencies

**Solution:**
```bash
# Create clean environment
conda create -n blt_hyena_clean python=3.9
conda activate blt_hyena_clean

# Or with venv
python -m venv blt_hyena_env
source blt_hyena_env/bin/activate  # Linux/Mac
# blt_hyena_env\Scripts\activate  # Windows
```

## Dependency Conflicts

### PyTorch Installation Issues

**Problem:** CUDA version mismatch

**Diagnostic:**
```bash
# Check CUDA version
nvidia-smi
nvcc --version
```

**Solution:**
```bash
# Install PyTorch with correct CUDA version
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify installation:**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
```

### Transformers Library Conflicts

**Problem:** Incompatible transformers version

**Solution:**
```bash
# Uninstall and reinstall
pip uninstall transformers
pip install transformers>=4.20.0

# Or specific version
pip install transformers==4.30.0
```

### Flash Attention Installation

**Problem:** Flash attention compilation errors

**Solutions:**

1. **Install pre-compiled version:**
```bash
pip install flash-attn --no-build-isolation
```

2. **Install from source (if pre-compiled fails):**
```bash
# Ensure CUDA toolkit is installed
pip install packaging ninja
pip install flash-attn --no-build-isolation --no-cache-dir
```

3. **Disable flash attention if installation fails:**
```python
config = HyenaGLTConfig(
    use_flash_attention=False  # Fallback to standard attention
)
```

## CUDA Compatibility Problems

### CUDA Toolkit Mismatch

**Problem:** Different CUDA versions causing conflicts

**Check versions:**
```bash
# System CUDA
nvidia-smi

# PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# Conda CUDA (if using conda)
conda list | grep cuda
```

**Solution:**
```bash
# Option 1: Install matching PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Option 2: Install CUDA toolkit via conda
conda install cuda-toolkit=11.8 -c nvidia
```

### GPU Memory Issues During Installation

**Problem:** Out of memory during package compilation

**Solution:**
```bash
# Set memory limits
export MAX_JOBS=1
pip install package_name --no-cache-dir
```

## Package Installation Errors

### Pip Installation Failures

**Problem:** Network issues or package conflicts

**Solutions:**

1. **Update pip:**
```bash
pip install --upgrade pip setuptools wheel
```

2. **Use conda instead:**
```bash
conda install -c conda-forge package_name
```

3. **Install from source:**
```bash
git clone https://github.com/sdodlapati3/BLT_Hyena.git
cd BLT_Hyena
pip install -e .
```

### Requirements File Issues

**Problem:** Conflicting requirements

**Solution:**
```bash
# Install requirements step by step
pip install torch torchvision torchaudio
pip install transformers
pip install datasets
pip install -r requirements.txt --no-deps  # Skip dependency resolution
```

## Operating System Specific Issues

### Windows Issues

**Problem:** Long path names or permissions

**Solutions:**

1. **Enable long paths:**
```cmd
# Run as administrator
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1
```

2. **Use shorter paths:**
```bash
# Install in C:\blt instead of deep nested folder
cd C:\
git clone https://github.com/sdodlapati3/BLT_Hyena.git blt
```

3. **WSL alternative:**
```bash
# Use Windows Subsystem for Linux
wsl --install
```

### macOS Issues

**Problem:** M1/M2 chip compatibility

**Solution:**
```bash
# Use MPS backend for M1/M2
pip install torch torchvision torchaudio

# In Python
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Linux Permission Issues

**Problem:** Permission denied errors

**Solution:**
```bash
# Install in user directory
pip install --user package_name

# Or fix permissions
sudo chown -R $USER ~/.local/
```

## Verification Steps

### Complete Installation Check

```python
#!/usr/bin/env python3
"""
BLT_Hyena Installation Verification Script
"""

def verify_installation():
    print("=== BLT_Hyena Installation Verification ===\n")
    
    # Python version
    import sys
    print(f"Python version: {sys.version}")
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
    print("✓ Python version OK\n")
    
    # PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print("✓ PyTorch OK\n")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    # Transformers
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        print("✓ Transformers OK\n")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    # BLT_Hyena components
    try:
        from hyena_glt import HyenaGLT, HyenaGLTConfig
        print("✓ BLT_Hyena core components OK")
        
        # Test basic functionality
        config = HyenaGLTConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2
        )
        model = HyenaGLT(config)
        
        # Test forward pass
        inputs = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            outputs = model(inputs)
        
        print("✓ Basic model functionality OK")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Output shape: {outputs.last_hidden_state.shape}")
        
    except ImportError as e:
        print(f"✗ BLT_Hyena import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ BLT_Hyena functionality test failed: {e}")
        return False
    
    print("\n🎉 All checks passed! BLT_Hyena is properly installed.")
    return True

if __name__ == "__main__":
    verify_installation()
```

**Run verification:**
```bash
python verify_installation.py
```

## Getting Help

If you're still experiencing issues:

1. **Check system requirements:**
   - Python 3.8+
   - CUDA 11.8+ (for GPU support)
   - 8GB+ RAM
   - 10GB+ storage

2. **Collect system information:**
```bash
# Create system info
python -c "
import sys, torch, platform
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
"
```

3. **Create GitHub issue** with:
   - Complete error message
   - System information
   - Installation steps attempted
   - Environment details

4. **Community support:**
   - [GitHub Discussions](https://github.com/sdodlapati3/BLT_Hyena/discussions)
   - [Discord Server](link_to_discord)
   - [Stack Overflow](https://stackoverflow.com/questions/tagged/blt-hyena)
