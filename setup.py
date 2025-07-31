#!/usr/bin/env python3
"""
BLEND Setup Configuration
Installation script for BLEND framework

Author: Raed Abdel-Sater
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    raise RuntimeError("BLEND requires Python 3.8 or later")

# Read version from __init__.py
def get_version():
    version_file = os.path.join("blend", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to find version string")

# Read long description from README
def get_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def get_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=22.0.0",
    "isort>=5.10.0", 
    "flake8>=5.0.0",
    "mypy>=0.991",
    "pre-commit>=2.20.0",
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0"
]

# Blockchain-specific requirements
blockchain_requirements = [
    "hyperledger-fabric>=2.4.0",
    "cryptography>=3.4.8",
    "web3>=6.0.0",
    "ethereum>=2.3.0",
    "ipfshttpclient>=0.8.0"
]

# Federated learning requirements
federated_requirements = [
    "flower>=1.4.0",
    "syft>=0.8.0",
    "ray>=2.0.0",
    "grpcio>=1.50.0",
    "grpcio-tools>=1.50.0"
]

# GPU-specific requirements
gpu_requirements = [
    "cupy-cuda11x>=11.0.0",
    "nvidia-ml-py3>=7.352.0"
]

# Visualization requirements
viz_requirements = [
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.10.0",
    "wandb>=0.13.0",
    "tensorboard>=2.10.0"
]

setup(
    name="blend-framework",
    version=get_version(),
    
    # Author information
    author="Raed Abdel-Sater",
    author_email="raed.abdel-sater@concordia.ca",
    
    # Package description
    description="BLEND: Blockchain-Enhanced Network Decentralisation with Large Language Models for Long-Horizon Time-Series Forecasting",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/yourusername/BLEND",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/BLEND/issues",
        "Source": "https://github.com/yourusername/BLEND",
        "Documentation": "https://blend-framework.readthedocs.io/",
        "Paper": "https://arxiv.org/abs/2025.xxxxx"
    },
    
    # Package configuration
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    package_data={
        "blend": [
            "configs/*.yaml",
            "configs/**/*.yaml",
            "data/schemas/*.json",
            "blockchain/contracts/*.sol"
        ]
    },
    include_package_data=True,
    
    # Requirements
    python_requires=">=3.8",
    install_requires=get_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": dev_requirements,
        "blockchain": blockchain_requirements,
        "federated": federated_requirements,
        "gpu": gpu_requirements,
        "viz": viz_requirements,
        "all": dev_requirements + blockchain_requirements + federated_requirements + gpu_requirements + viz_requirements
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "blend-train=blend.scripts.train_blend:main",
            "blend-eval=blend.scripts.evaluate:main",
            "blend-setup=blend.scripts.setup_blockchain:main",
            "blend-benchmark=blend.scripts.run_experiments:main"
        ]
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers", 
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords
    keywords=[
        "federated-learning",
        "blockchain", 
        "large-language-models",
        "time-series-forecasting",
        "internet-of-vehicles",
        "consensus-protocol",
        "distributed-systems",
        "machine-learning"
    ],
    
    # License
    license="MIT",
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    
    # Test configuration
    test_suite="tests",
    tests_require=[
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.21.0"
    ],
    
    # Command class for custom commands
    cmdclass={},
)