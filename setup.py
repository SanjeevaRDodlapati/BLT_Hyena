"""Setup configuration for Hyena-GLT package."""

import os
import platform

from setuptools import find_packages, setup

# Read README for long description
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version
def get_version():
    version_file = os.path.join("hyena_glt", "__init__.py")
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Platform-specific requirements
def get_install_requires():
    base_requirements = [
        "torch>=2.0.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "datasets>=2.0.0",
        "tokenizers>=0.13.0",
        "wandb>=0.13.0",
        "tqdm>=4.62.0",
        "einops>=0.6.0",
        "biopython>=1.79",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tensorboard>=2.8.0",
        # Add packages needed for benchmark and distributed training
        "omegaconf>=2.3.0",
        "sentencepiece>=0.2.0",
        "plotly>=5.0.0",
        "psutil>=5.8.0",
        "pynvml>=11.0.0",
    ]

    # Add platform-specific requirements
    if platform.system() != "Darwin":  # Not macOS
        base_requirements.extend([
            "triton>=2.0.0",
            "flash-attn>=2.0.0",
        ])

    return base_requirements

setup(
    name="hyena-glt",
    version=get_version(),
    author="Hyena-GLT Team",
    author_email="contact@hyena-glt.ai",
    description="Genome Language Transformer combining BLT and Hyena architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/hyena-glt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.11",
    install_requires=get_install_requires(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hyena-glt-train=hyena_glt.cli.train:main",
            "hyena-glt-eval=hyena_glt.cli.eval:main",
            "hyena-glt-preprocess=hyena_glt.cli.preprocess:main",
        ],
    },
    package_data={
        "hyena_glt": [
            "config/presets/*.json",
            "data/vocab/*.txt",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
