# Testing Documentation for Hyena-GLT Framework

This document provides comprehensive information about the testing infrastructure for the Hyena-GLT framework.

## Overview

The testing framework provides comprehensive coverage of the Hyena-GLT system with:
- **Unit Tests**: Testing individual components in isolation
- **Integration Tests**: Testing complete workflows and component interactions
- **Performance Benchmarks**: Speed, memory, and scalability testing
- **Test Utilities**: Common fixtures, data generators, and helper functions

## Test Structure

```
tests/
├── __init__.py                 # Testing package initialization
├── utils.py                   # Testing utilities and fixtures
├── unit/                      # Unit tests
│   ├── __init__.py
│   ├── test_config.py         # Configuration system tests
│   ├── test_data.py           # Data processing tests
│   ├── test_model.py          # Model architecture tests
│   ├── test_training.py       # Training infrastructure tests
│   ├── test_evaluation.py     # Evaluation framework tests
│   └── test_optimization.py   # Optimization module tests
└── integration/               # Integration tests
    ├── __init__.py
    ├── test_workflows.py      # End-to-end workflow tests
    └── test_benchmarks.py     # Performance benchmark tests
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m benchmark     # Benchmark tests only

# Run specific test files
pytest tests/unit/test_model.py
pytest tests/integration/test_workflows.py

# Run specific test functions
pytest tests/unit/test_model.py::TestHyenaGLTModel::test_forward_pass
```

### Test Categories and Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests for workflows
- `@pytest.mark.benchmark`: Performance benchmark tests
- `@pytest.mark.slow`: Tests that take significant time
- `@pytest.mark.gpu`: Tests requiring GPU resources
- `@pytest.mark.memory_intensive`: Tests using significant memory
- `@pytest.mark.requires_data`: Tests needing external data

### Running Specific Test Categories

```bash
# Run only fast unit tests
pytest -m "unit and not slow"

# Run GPU tests (if CUDA available)
pytest -m gpu

# Run benchmarks without slow tests
pytest -m "benchmark and not slow"

# Skip memory-intensive tests
pytest -m "not memory_intensive"
```

## Test Configuration

### pytest.ini Configuration

The `pytest.ini` file contains project-wide test configuration:

```ini
[tool:pytest]
testpaths = tests
markers = 
    unit: Unit tests
    integration: Integration tests
    benchmark: Performance benchmarks
addopts = --verbose --tb=short --durations=10
timeout = 300
```

### Environment Variables

Set these environment variables for testing:

```bash
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0  # For GPU tests
export TORCH_HOME=/tmp/torch_cache
```

## Unit Tests

### Configuration Tests (`test_config.py`)

Tests the configuration system including:
- Configuration creation and validation
- Serialization to/from JSON
- Task-specific configurations
- Device compatibility

```python
def test_config_creation():
    config = HyenaGLTConfig()
    assert config.model.hidden_size > 0
    assert config.training.learning_rate > 0
```

### Data Processing Tests (`test_data.py`)

Tests data processing components:
- DNA/RNA/Protein tokenizers
- Genomic utility functions
- Dataset classes and data loaders
- Data validation and preprocessing

```python
def test_dna_tokenizer():
    tokenizer = DNATokenizer()
    sequence = "ATCGATCG"
    tokens = tokenizer.encode(sequence)
    decoded = tokenizer.decode(tokens)
    assert decoded == sequence
```

### Model Architecture Tests (`test_model.py`)

Tests model components:
- Hyena operators and convolutions
- Token merging mechanisms
- Transformer blocks
- Task-specific heads
- Full model integration

```python
def test_hyena_operator():
    config = TestConfig.get_small_config()
    operator = HyenaOperator(config)
    x = torch.randn(2, 100, 64)
    output = operator(x)
    assert output.shape == x.shape
```

### Training Tests (`test_training.py`)

Tests training infrastructure:
- Training configuration and setup
- Optimization utilities
- Multi-task learning
- Curriculum learning
- Checkpoint management

```python
def test_training_step():
    trainer = HyenaGLTTrainer(model, config, dataset)
    batch = next(iter(dataloader))
    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
```

### Evaluation Tests (`test_evaluation.py`)

Tests evaluation framework:
- Genomic and sequence metrics
- Model analysis tools
- Benchmarking utilities
- Visualization components

```python
def test_genomic_metrics():
    metrics = GenomicMetrics()
    gc_content = metrics.calculate_gc_content("ATCG")
    assert gc_content == 0.5
```

### Optimization Tests (`test_optimization.py`)

Tests optimization techniques:
- Quantization (dynamic, static, QAT)
- Pruning (magnitude, structured, gradient-based)
- Knowledge distillation
- Memory optimization
- Deployment utilities

```python
def test_dynamic_quantization():
    quantizer = DynamicQuantizer(config)
    quantized_model = quantizer.quantize(model)
    assert isinstance(quantized_model, nn.Module)
```

## Integration Tests

### Workflow Tests (`test_workflows.py`)

Tests complete end-to-end workflows:
- Training pipeline integration
- Evaluation workflow testing
- Optimization pipeline testing
- Model serialization workflows

```python
def test_complete_training_workflow():
    # Test complete pipeline: data -> model -> training -> evaluation
    model = HyenaGLTModel(config)
    trainer = HyenaGLTTrainer(model, config, dataset)
    trainer.train(num_epochs=2)
    # Verify training artifacts and model performance
```

### Performance Benchmarks (`test_benchmarks.py`)

Comprehensive performance testing:
- Speed benchmarks across input sizes
- Memory usage analysis
- Scalability testing
- Comparative benchmarks

```python
def test_forward_pass_speed():
    model = HyenaGLTModel(config)
    x = torch.randint(0, 4, (batch_size, seq_len))
    
    # Benchmark inference time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        output = model(x)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    assert avg_time < target_time
```

## Test Utilities

### Testing Configuration (`tests/utils.py`)

Provides common testing utilities:

```python
class TestConfig:
    @staticmethod
    def get_small_config():
        """Get configuration for fast testing."""
        config = HyenaGLTConfig()
        config.model.hidden_size = 64
        config.model.num_layers = 2
        return config

class DataGenerator:
    def generate_dna_sequences(self, num_sequences, min_length, max_length):
        """Generate synthetic DNA sequences for testing."""
        # Implementation here

class ModelTestUtils:
    @staticmethod
    def create_test_model(config=None):
        """Create a model for testing."""
        if config is None:
            config = TestConfig.get_small_config()
        return HyenaGLTModel(config)
```

### Fixtures and Mocks

Common pytest fixtures for testing:

```python
@pytest.fixture
def sample_config():
    return TestConfig.get_small_config()

@pytest.fixture
def sample_model(sample_config):
    return HyenaGLTModel(sample_config)

@pytest.fixture
def sample_data():
    generator = DataGenerator()
    return generator.generate_dna_sequences(100, 50, 200)
```

## Coverage and Quality

### Test Coverage

The testing framework aims for high code coverage:

```bash
# Run tests with coverage
pytest --cov=hyena_glt --cov-report=html

# View coverage report
open htmlcov/index.html
```

Target coverage levels:
- **Overall**: >80%
- **Core model components**: >90%
- **Training infrastructure**: >85%
- **Data processing**: >85%

### Code Quality Checks

Additional quality checks can be run:

```bash
# Type checking
mypy hyena_glt/

# Code formatting
black --check hyena_glt/ tests/

# Linting
flake8 hyena_glt/ tests/

# Import sorting
isort --check-only hyena_glt/ tests/
```

## Continuous Integration

### GitHub Actions Workflow

Example CI configuration (`.github/workflows/test.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests
      run: pytest --cov=hyena_glt
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Test Environments

Testing across different environments:
- **Python versions**: 3.8, 3.9, 3.10, 3.11
- **PyTorch versions**: Latest stable and LTS
- **Operating systems**: Ubuntu, macOS, Windows
- **Hardware**: CPU-only and GPU-enabled

## Performance Testing

### Benchmark Categories

1. **Speed Benchmarks**
   - Forward pass latency
   - Training throughput
   - Inference speed

2. **Memory Benchmarks**
   - Parameter memory footprint
   - Activation memory usage
   - Gradient memory requirements

3. **Scalability Benchmarks**
   - Batch size scaling
   - Sequence length scaling
   - Model size scaling

### Benchmark Execution

```bash
# Run all benchmarks
pytest -m benchmark

# Run specific benchmark categories
pytest -m "benchmark and speed"
pytest -m "benchmark and memory"
pytest -m "benchmark and scalability"

# Run benchmarks with detailed output
pytest -m benchmark -v --durations=0
```

### Performance Baselines

Establish performance baselines for regression testing:

```python
def test_performance_baseline():
    """Test against established performance baseline."""
    model = create_standard_model()
    x = create_standard_input()
    
    latency = measure_inference_latency(model, x)
    assert latency < BASELINE_LATENCY_MS
    
    memory = measure_memory_usage(model, x)
    assert memory < BASELINE_MEMORY_MB
```

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with Python debugger
pytest --pdb

# Run with verbose output and no capture
pytest -v -s

# Run single test with maximum detail
pytest -vvv -s tests/unit/test_model.py::test_specific_function
```

### Common Debugging Techniques

1. **Print Debugging**: Use `print()` statements with `-s` flag
2. **Breakpoints**: Use `import pdb; pdb.set_trace()` with `--pdb`
3. **Logging**: Enable detailed logging with `--log-cli-level=DEBUG`
4. **Profiling**: Use `pytest-profiling` for performance analysis

## Best Practices

### Writing Good Tests

1. **Descriptive Names**: Use clear, descriptive test function names
2. **Single Responsibility**: Each test should test one specific behavior
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Parameterization**: Use `@pytest.mark.parametrize` for testing multiple scenarios
5. **Fixtures**: Use fixtures for common setup and teardown
6. **Mocking**: Mock external dependencies and slow operations

### Test Organization

1. **Logical Grouping**: Group related tests in the same class
2. **Clear Documentation**: Add docstrings explaining test purpose
3. **Consistent Naming**: Follow consistent naming conventions
4. **Proper Categorization**: Use appropriate pytest markers
5. **Isolation**: Ensure tests don't depend on each other

### Performance Considerations

1. **Fast by Default**: Unit tests should run quickly
2. **Mark Slow Tests**: Use `@pytest.mark.slow` for time-consuming tests
3. **Efficient Fixtures**: Use session/module scope for expensive setup
4. **Minimal Models**: Use small models for unit tests
5. **Parallel Execution**: Design tests to run in parallel safely

## Troubleshooting

### Common Issues

1. **GPU Tests Failing**: Ensure CUDA is available or skip GPU tests
2. **Memory Errors**: Reduce batch sizes or model sizes in tests
3. **Timeout Errors**: Increase timeout or optimize slow tests
4. **Import Errors**: Check PYTHONPATH and package installation
5. **Random Failures**: Set random seeds for reproducible tests

### Getting Help

- Check test logs with `pytest --log-cli-level=DEBUG`
- Run tests in isolation to identify conflicts
- Use `pytest --collect-only` to see discovered tests
- Check CI logs for environment-specific issues

## Contributing to Tests

### Adding New Tests

1. **Identify Coverage Gaps**: Use coverage reports to find untested code
2. **Follow Conventions**: Match existing test structure and naming
3. **Add Appropriate Markers**: Categorize tests with pytest markers
4. **Update Documentation**: Document new test categories or utilities
5. **Verify Coverage**: Ensure new tests increase overall coverage

### Test Review Checklist

- [ ] Tests are properly categorized with markers
- [ ] Test names are descriptive and clear
- [ ] Tests are isolated and don't depend on each other
- [ ] Appropriate fixtures and mocks are used
- [ ] Performance tests have reasonable baselines
- [ ] Documentation is updated if needed
- [ ] All tests pass locally and in CI

This testing framework ensures the Hyena-GLT system is robust, performant, and maintainable through comprehensive automated testing.
