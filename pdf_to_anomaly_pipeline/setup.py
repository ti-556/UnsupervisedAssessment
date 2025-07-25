#!/usr/bin/env python3
"""
Setup script for PDF to Anomaly Detection Pipeline
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pdf-anomaly-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive pipeline for PDF to anomaly detection using ViT embeddings and quality metrics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pdf-anomaly-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",  # CUDA 11.8 version
        ],
    },
    entry_points={
        "console_scripts": [
            "pdf-anomaly-pipeline=main_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "pdf",
        "anomaly-detection",
        "vision-transformer",
        "computer-vision",
        "machine-learning",
        "image-processing",
        "quality-metrics",
        "isolation-forest",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pdf-anomaly-pipeline/issues",
        "Source": "https://github.com/yourusername/pdf-anomaly-pipeline",
        "Documentation": "https://github.com/yourusername/pdf-anomaly-pipeline#readme",
    },
) 