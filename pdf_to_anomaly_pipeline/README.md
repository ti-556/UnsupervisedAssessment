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

## Troubleshooting

### Common Issues

1. **Poppler-utils not found**:
   ```bash
   sudo apt-get install poppler-utils
   ```

2. **CUDA out of memory**:
   ```python
   vit_encoder = ViTEncoder(device='cpu')  # Force CPU usage
   ```

3. **Model download issues**:
   ```python
   # The pipeline automatically downloads models, but you can pre-download:
   from transformers import ViTImageProcessor, ViTModel
   processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
   model = ViTModel.from_pretrained("google/vit-base-patch16-224")
   ```

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster ViT processing
2. **Adjust DPI**: Lower DPI (200-300) for faster processing
3. **Batch Size**: Increase batch size for GPU processing
4. **Memory**: Monitor memory usage for large PDFs

## Model Information

### Vision Transformer (ViT)

- **Model**: google/vit-base-patch16-224
- **Input Size**: 224x224 pixels
- **Output**: 768-dimensional embeddings (reduced to 64D via PCA)
- **Download Size**: ~85MB

### Quality Metrics

The pipeline extracts 7 quality metrics:

1. **Brightness**: Average pixel intensity
2. **Contrast**: Standard deviation of pixel values
3. **Sharpness**: Laplacian variance
4. **Noise Level**: Difference from Gaussian blur
5. **Color Variety**: HSV histogram entropy
6. **Edge Density**: Canny edge detection ratio
7. **Texture Complexity**: Local patch variance

### Anomaly Detection

- **Algorithm**: Isolation Forest
- **Features**: 71D vectors (64D ViT + 7D metrics)
- **Contamination**: Configurable (default: 10%)
- **Output**: Anomaly scores and binary predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{pdf_anomaly_pipeline,
  title={PDF to Anomaly Detection Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pdf-anomaly-pipeline}
}
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples
3. Open an issue on GitHub
4. Check the pipeline.log file for detailed error messages 