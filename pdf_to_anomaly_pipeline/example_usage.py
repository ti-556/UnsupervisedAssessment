#!/usr/bin/env python3
"""
Example Usage of PDF to Anomaly Detection Pipeline
==================================================

This script demonstrates how to use the pipeline with example data.
It shows different ways to run the pipeline and handle results.
"""

import os
import sys
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_pipeline import PDFAnomalyPipeline
from pdf_downloader import PDFDownloader
from pdf_to_images import PDFToImageConverter
from vit_encoder import ViTEncoder
from quality_metrics import QualityMetricsExtractor
from anomaly_detector import AnomalyDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_basic_pipeline():
    """Example 1: Basic pipeline usage with local PDFs"""
    print("\n" + "="*60)
    print("Example 1: Basic Pipeline Usage")
    print("="*60)
    
    # Initialize pipeline
    pipeline = PDFAnomalyPipeline(output_dir="example_results_1")
    
    # Example PDF sources (replace with actual paths)
    pdf_sources = [
        # Add your PDF paths here, for example:
        # "/path/to/presentation1.pdf",
        # "/path/to/presentation2.pdf"
    ]
    
    if not pdf_sources:
        print("Please add PDF file paths to pdf_sources in the script")
        return
    
    # Run pipeline
    results = pipeline.run_pipeline(pdf_sources, contamination=0.1)
    
    if results:
        print(f"Pipeline completed successfully!")
        print(f"Processed {results['num_pdfs']} PDFs")
        print(f"Generated {results['num_images']} images")
        print(f"Detected {results['analysis']['anomalies']} anomalies")
        print(f"Results saved to: example_results_1")

def example_2_download_from_urls():
    """Example 2: Download PDFs from URLs"""
    print("\n" + "="*60)
    print("Example 2: Download PDFs from URLs")
    print("="*60)
    
    # Initialize pipeline
    pipeline = PDFAnomalyPipeline(output_dir="example_results_2")
    
    # Example URLs (replace with actual URLs)
    pdf_urls = [
        # Add your PDF URLs here, for example:
        # "https://example.com/presentation1.pdf",
        # "https://example.com/presentation2.pdf"
    ]
    
    if not pdf_urls:
        print("Please add PDF URLs to pdf_urls in the script")
        return
    
    # Run pipeline
    results = pipeline.run_pipeline(pdf_urls, contamination=0.15)
    
    if results:
        print(f"Pipeline completed successfully!")
        print(f"Downloaded and processed {results['num_pdfs']} PDFs")
        print(f"Results saved to: example_results_2")

def example_3_component_usage():
    """Example 3: Using individual components"""
    print("\n" + "="*60)
    print("Example 3: Using Individual Components")
    print("="*60)
    
    # Initialize individual components
    downloader = PDFDownloader(output_dir="component_downloads")
    converter = PDFToImageConverter(output_dir="component_images")
    vit_encoder = ViTEncoder()
    metrics_extractor = QualityMetricsExtractor()
    anomaly_detector = AnomalyDetector()
    
    # Example workflow
    pdf_path = "/path/to/your/presentation.pdf"  # Replace with actual path
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print("Please update the path in the script")
        return
    
    # Step 1: Convert PDF to images
    print("Converting PDF to images...")
    image_paths = converter.convert_pdf(pdf_path)
    
    if not image_paths:
        print("No images generated")
        return
    
    # Step 2: Extract ViT embeddings
    print("Extracting ViT embeddings...")
    vit_embeddings = vit_encoder.extract_embeddings_batch(image_paths)
    
    # Reduce to 64D
    if vit_embeddings:
        import numpy as np
        embeddings_array = np.array(list(vit_embeddings.values()))
        reduced_embeddings, pca_model = vit_encoder.reduce_dimensions(embeddings_array, target_dim=64)
        
        # Convert back to dictionary
        reduced_embeddings_dict = {}
        for i, path in enumerate(vit_embeddings.keys()):
            reduced_embeddings_dict[path] = reduced_embeddings[i]
        
        vit_embeddings = reduced_embeddings_dict
    
    # Step 3: Extract quality metrics
    print("Extracting quality metrics...")
    quality_metrics = metrics_extractor.extract_metrics_batch(image_paths)
    
    # Normalize metrics
    if quality_metrics:
        normalized_metrics, scaler = metrics_extractor.normalize_metrics(quality_metrics)
        quality_metrics = normalized_metrics
    
    # Step 4: Prepare feature vectors
    print("Preparing feature vectors...")
    if vit_embeddings and quality_metrics:
        feature_vectors = anomaly_detector.prepare_features(vit_embeddings, quality_metrics)
        
        # Step 5: Anomaly detection
        print("Performing anomaly detection...")
        results = anomaly_detector.fit(feature_vectors)
        
        # Step 6: Analysis
        print("Generating analysis...")
        analysis = anomaly_detector.analyze_results(results, output_dir="component_results")
        
        print(f"Analysis complete! Found {analysis['anomalies']} anomalies")

def example_4_batch_processing():
    """Example 4: Batch processing multiple PDFs"""
    print("\n" + "="*60)
    print("Example 4: Batch Processing Multiple PDFs")
    print("="*60)
    
    # Initialize pipeline
    pipeline = PDFAnomalyPipeline(output_dir="batch_results")
    
    # Example: Process all PDFs in a directory
    pdf_directory = "/path/to/pdf/directory"  # Replace with actual path
    
    if not os.path.exists(pdf_directory):
        print(f"Directory not found: {pdf_directory}")
        print("Please update the path in the script")
        return
    
    # Find all PDF files
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Run pipeline
    results = pipeline.run_pipeline([str(f) for f in pdf_files], contamination=0.1)
    
    if results:
        print(f"Batch processing completed!")
        print(f"Processed {results['num_pdfs']} PDFs")
        print(f"Generated {results['num_images']} images")
        print(f"Detected {results['analysis']['anomalies']} anomalies")

def example_5_custom_parameters():
    """Example 5: Using custom parameters"""
    print("\n" + "="*60)
    print("Example 5: Using Custom Parameters")
    print("="*60)
    
    # Initialize components with custom parameters
    converter = PDFToImageConverter(
        output_dir="custom_images",
        dpi=200,  # Lower DPI for faster processing
        format='JPEG'  # Use JPEG instead of PNG
    )
    
    vit_encoder = ViTEncoder(
        model_name="google/vit-base-patch16-224",  # Default model
        device='cpu'  # Force CPU usage
    )
    
    anomaly_detector = AnomalyDetector(
        contamination=0.05,  # Expect 5% anomalies
        random_state=123
    )
    
    # Create custom pipeline
    pipeline = PDFAnomalyPipeline(output_dir="custom_results")
    
    # Replace components with custom ones
    pipeline.image_converter = converter
    pipeline.vit_encoder = vit_encoder
    pipeline.anomaly_detector = anomaly_detector
    
    # Example PDF source
    pdf_source = "/path/to/your/presentation.pdf"  # Replace with actual path
    
    if not os.path.exists(pdf_source):
        print(f"PDF file not found: {pdf_source}")
        print("Please update the path in the script")
        return
    
    # Run pipeline with custom parameters
    results = pipeline.run_pipeline([pdf_source], contamination=0.05)
    
    if results:
        print(f"Custom pipeline completed!")
        print(f"Results saved to: custom_results")

def main():
    """Run all examples"""
    print("PDF to Anomaly Detection Pipeline - Examples")
    print("="*60)
    
    # Run examples (uncomment the ones you want to try)
    
    # example_1_basic_pipeline()
    # example_2_download_from_urls()
    # example_3_component_usage()
    # example_4_batch_processing()
    # example_5_custom_parameters()
    
    print("\nTo run examples:")
    print("1. Update the file paths/URLs in the example functions")
    print("2. Uncomment the example function calls above")
    print("3. Run this script: python example_usage.py")
    
    print("\nExample command line usage:")
    print("python main_pipeline.py --pdfs presentation1.pdf presentation2.pdf --output_dir results")
    print("python main_pipeline.py --urls https://example.com/presentation.pdf --contamination 0.1")

if __name__ == "__main__":
    main() 