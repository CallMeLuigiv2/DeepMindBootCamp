"""Setup script for shared_utils package.

Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="shared_utils",
    version="0.1.0",
    description="Shared utilities for the Road to ML Expert course",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
    ],
)
