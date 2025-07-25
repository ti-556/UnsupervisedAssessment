# PDF to Anomaly Detection Pipeline

A comprehensive pipeline for downloading PDFs, converting them to images, extracting ViT embeddings, combining with quality metrics to form 71D feature vectors, and performing anomaly detection using Isolation Forest.

## Overview

This pipeline performs the following steps:

1. **PDF Download/Copy**: Download PDFs from URLs or copy from local paths
2. **PDF to Image Conversion**: Convert PDF pages to high-quality images
3. **ViT Embedding Extraction**: Extract 64-dimensional embeddings using Vision Transformer
4. **Quality Metrics Extraction**: Extract 7 quality metrics (brightness, contrast, sharpness, etc.)
5. **Feature Vector Creation**: Combine ViT embeddings and quality metrics into 71D vectors
6. **Anomaly Detection**: Use Isolation Forest to detect anomalies
7. **Analysis & Visualization**: Generate comprehensive analysis and visualizations

## Features

- **Automatic Model Download**: Downloads ViT model checkpoints if not already installed
- **Flexible Input**: Support for both local PDF files and URLs
- **Batch Processing**: Process multiple PDFs efficiently
- **Comprehensive Analysis**: Detailed anomaly detection results with visualizations
- **Modular Design**: Each component can be used independently
- **Configurable Parameters**: Customizable DPI, model selection, contamination levels, etc.

## Installation

### Prerequisites

- Python 3.8 or higher
- Linux/Ubuntu (for pdf2image with poppler-utils)

### Install Dependencies

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install poppler-utils

# Install Python dependencies
pip install -r requirements.txt
```

### Alternative Installation

If you encounter issues with `poppler-utils`, you can install it via conda:

```bash
conda install -c conda-forge poppler
```

## Quick Start

### Basic Usage

```bash
# Process local PDF files
python main_pipeline.py --pdfs presentation1.pdf presentation2.pdf --output_dir results

# Download and process PDFs from URLs
python main_pipeline.py --urls https://example.com/presentation.pdf --contamination 0.1

# Process with custom contamination level
python main_pipeline.py --pdfs presentation.pdf --contamination 0.05 --output_dir custom_results
```

### Python API Usage

```python
from main_pipeline import PDFAnomalyPipeline

# Initialize pipeline
pipeline = PDFAnomalyPipeline(output_dir="my_results")

# Run pipeline
pdf_sources = ["presentation1.pdf", "presentation2.pdf"]
results = pipeline.run_pipeline(pdf_sources, contamination=0.1)

if results:
    print(f"Processed {results['num_pdfs']} PDFs")
    print(f"Detected {results['analysis']['anomalies']} anomalies")
```

## Pipeline Components

### 1. PDF Downloader (`pdf_downloader.py`)

Downloads PDFs from URLs or copies from local paths.

```python
from pdf_downloader import PDFDownloader

downloader = PDFDownloader(output_dir="downloads")
pdf_path = downloader.download_from_url("https://example.com/presentation.pdf")
```

### 2. PDF to Image Converter (`pdf_to_images.py`)

Converts PDF pages to high-quality images.

```python
from pdf_to_images import PDFToImageConverter

converter = PDFToImageConverter(output_dir="images", dpi=300)
image_paths = converter.convert_pdf("presentation.pdf")
```

### 3. ViT Encoder (`vit_encoder.py`)

Extracts Vision Transformer embeddings from images.

```python
from vit_encoder import ViTEncoder

encoder = ViTEncoder(model_name="google/vit-base-patch16-224")
embeddings = encoder.extract_embeddings_batch(image_paths)
```

### 4. Quality Metrics Extractor (`quality_metrics.py`)

Extracts 7 quality metrics from images:
- Brightness
- Contrast
- Sharpness
- Noise Level
- Color Variety
- Edge Density
- Texture Complexity

```python
from quality_metrics import QualityMetricsExtractor

extractor = QualityMetricsExtractor()
metrics = extractor.extract_metrics_batch(image_paths)
```

### 5. Anomaly Detector (`anomaly_detector.py`)

Performs anomaly detection using Isolation Forest on 71D feature vectors.

```python
from anomaly_detector import AnomalyDetector

detector = AnomalyDetector(contamination=0.1)
feature_vectors = detector.prepare_features(vit_embeddings, quality_metrics)
results = detector.fit(feature_vectors)
```

## Output Structure

The pipeline generates the following output structure:

```
output_dir/
├── downloads/                 # Downloaded PDF files
├── images/                    # Converted images
│   └── presentation_name/
│       ├── slide_001.png
│       ├── slide_002.png
│       └── ...
├── anomaly_results/           # Anomaly detection results
│   ├── anomaly_detection_results.csv
│   ├── anomaly_analysis_overview.png
│   └── detailed_anomaly_analysis.png
├── vit_embeddings_64d.pkl     # ViT embeddings
├── quality_metrics.pkl        # Quality metrics
├── feature_vectors_71d.pkl    # Combined feature vectors
├── vit_pca_model.pkl          # PCA model for dimension reduction
├── metrics_scaler.pkl         # Metrics normalization scaler
├── anomaly_detection_model.pkl # Trained anomaly detection model
└── pipeline.log               # Pipeline execution log
```

## Configuration Options

### PDF to Image Conversion

- **DPI**: Image resolution (default: 300)
- **Format**: Output format (PNG, JPEG) (default: PNG)

### ViT Model

- **Model**: Vision Transformer model (default: google/vit-base-patch16-224)
- **Device**: CPU/GPU (auto-detected)
- **Batch Size**: Processing batch size (default: 8)

### Anomaly Detection

- **Contamination**: Expected fraction of anomalies (default: 0.1)
- **Random State**: Reproducibility seed (default: 42)
- **PCA Components**: Optional dimensionality reduction

## Examples

See `example_usage.py` for comprehensive examples:

```bash
python example_usage.py
```

### Example 1: Basic Pipeline
```python
pipeline = PDFAnomalyPipeline(output_dir="results")
results = pipeline.run_pipeline(["presentation.pdf"])
```

### Example 2: Custom Parameters
```python
converter = PDFToImageConverter(dpi=200, format='JPEG')
detector = AnomalyDetector(contamination=0.05)
pipeline = PDFAnomalyPipeline(output_dir="custom_results")
```

### Example 3: Batch Processing
```python
import glob
pdf_files = glob.glob("presentations/*.pdf")
pipeline = PDFAnomalyPipeline(output_dir="batch_results")
results = pipeline.run_pipeline(pdf_files)
```
