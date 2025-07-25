#!/usr/bin/env python3
"""
Test script for PDF to Anomaly Detection Pipeline
=================================================

This script tests the pipeline components to ensure they work correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import logging
import numpy as np
from PIL import Image

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_downloader import PDFDownloader
from pdf_to_images import PDFToImageConverter
from vit_encoder import ViTEncoder
from quality_metrics import QualityMetricsExtractor
from anomaly_detector import AnomalyDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_pdf():
    """Create a simple test PDF for testing"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
        
        # Create a simple PDF with some content
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Add some text and shapes
        c.drawString(100, 750, "Test Presentation Slide 1")
        c.drawString(100, 700, "This is a test slide for the pipeline")
        c.rect(100, 600, 200, 100)
        c.circle(300, 650, 50)
        
        # Add a second page
        c.showPage()
        c.drawString(100, 750, "Test Presentation Slide 2")
        c.drawString(100, 700, "This is another test slide")
        c.rect(150, 550, 150, 80)
        
        c.save()
        
        logger.info(f"Created test PDF: {pdf_path}")
        return pdf_path
        
    except ImportError:
        logger.warning("reportlab not available, skipping PDF creation test")
        return None

def create_test_images():
    """Create test images for testing"""
    test_images = []
    
    # Create a few test images
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            img_path = tmp_file.name
        
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color=(100 + i*50, 150, 200))
        
        # Add some variation
        pixels = img.load()
        for x in range(50, 150):
            for y in range(50, 150):
                pixels[x, y] = (255, 255, 255)  # White rectangle
        
        img.save(img_path)
        test_images.append(img_path)
    
    logger.info(f"Created {len(test_images)} test images")
    return test_images

def test_pdf_downloader():
    """Test PDF downloader component"""
    print("\n" + "="*50)
    print("Testing PDF Downloader")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = PDFDownloader(output_dir=temp_dir)
        
        # Test local file copy
        test_pdf = create_test_pdf()
        if test_pdf:
            result = downloader.copy_from_local(test_pdf)
            if result:
                print("✓ PDF downloader (local copy) - PASSED")
            else:
                print("✗ PDF downloader (local copy) - FAILED")
            
            # Cleanup
            os.unlink(test_pdf)
        else:
            print("⚠ PDF downloader test skipped (no test PDF created)")

def test_pdf_to_images():
    """Test PDF to image converter component"""
    print("\n" + "="*50)
    print("Testing PDF to Image Converter")
    print("="*50)
    
    test_pdf = create_test_pdf()
    if not test_pdf:
        print("⚠ PDF to image converter test skipped (no test PDF created)")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        converter = PDFToImageConverter(output_dir=temp_dir)
        
        try:
            image_paths = converter.convert_pdf(test_pdf)
            
            if image_paths and len(image_paths) > 0:
                print(f"✓ PDF to image converter - PASSED ({len(image_paths)} images created)")
                
                # Test image info
                for img_path in image_paths[:2]:  # Test first 2 images
                    info = converter.get_image_info(img_path)
                    if info:
                        print(f"  - Image info: {info['width']}x{info['height']}, {info['format']}")
            else:
                print("✗ PDF to image converter - FAILED (no images created)")
                
        except Exception as e:
            print(f"✗ PDF to image converter - FAILED (error: {str(e)})")
    
    # Cleanup
    os.unlink(test_pdf)

def test_vit_encoder():
    """Test ViT encoder component"""
    print("\n" + "="*50)
    print("Testing ViT Encoder")
    print("="*50)
    
    test_images = create_test_images()
    if not test_images:
        print("⚠ ViT encoder test skipped (no test images created)")
        return
    
    try:
        # Initialize encoder
        encoder = ViTEncoder()
        print("✓ ViT encoder initialization - PASSED")
        
        # Test single image embedding
        embedding = encoder.extract_embedding(test_images[0])
        if embedding is not None and len(embedding) == 768:  # ViT base model output size
            print("✓ ViT encoder (single image) - PASSED")
        else:
            print("✗ ViT encoder (single image) - FAILED")
        
        # Test batch processing
        embeddings = encoder.extract_embeddings_batch(test_images)
        if embeddings and len(embeddings) == len(test_images):
            print("✓ ViT encoder (batch processing) - PASSED")
            
            # Test dimension reduction
            embeddings_array = np.array(list(embeddings.values()))
            reduced_embeddings, pca_model = encoder.reduce_dimensions(embeddings_array, target_dim=64)
            
            if reduced_embeddings.shape[1] == 64:
                print("✓ ViT encoder (dimension reduction) - PASSED")
            else:
                print("✗ ViT encoder (dimension reduction) - FAILED")
        else:
            print("✗ ViT encoder (batch processing) - FAILED")
            
    except Exception as e:
        print(f"✗ ViT encoder - FAILED (error: {str(e)})")
    
    # Cleanup
    for img_path in test_images:
        os.unlink(img_path)

def test_quality_metrics():
    """Test quality metrics extractor component"""
    print("\n" + "="*50)
    print("Testing Quality Metrics Extractor")
    print("="*50)
    
    test_images = create_test_images()
    if not test_images:
        print("⚠ Quality metrics test skipped (no test images created)")
        return
    
    try:
        extractor = QualityMetricsExtractor()
        
        # Test single image metrics
        metrics = extractor.extract_all_metrics(test_images[0])
        if metrics and len(metrics) == 7:
            print("✓ Quality metrics (single image) - PASSED")
            print(f"  - Metrics: {list(metrics.keys())}")
        else:
            print("✗ Quality metrics (single image) - FAILED")
        
        # Test batch processing
        all_metrics = extractor.extract_metrics_batch(test_images)
        if all_metrics and len(all_metrics) == len(test_images):
            print("✓ Quality metrics (batch processing) - PASSED")
            
            # Test normalization
            normalized_metrics, scaler = extractor.normalize_metrics(all_metrics)
            if normalized_metrics:
                print("✓ Quality metrics (normalization) - PASSED")
            else:
                print("✗ Quality metrics (normalization) - FAILED")
        else:
            print("✗ Quality metrics (batch processing) - FAILED")
            
    except Exception as e:
        print(f"✗ Quality metrics - FAILED (error: {str(e)})")
    
    # Cleanup
    for img_path in test_images:
        os.unlink(img_path)

def test_anomaly_detector():
    """Test anomaly detector component"""
    print("\n" + "="*50)
    print("Testing Anomaly Detector")
    print("="*50)
    
    try:
        detector = AnomalyDetector(contamination=0.1)
        
        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 100
        n_features = 71
        
        # Create normal data
        normal_data = np.random.normal(0, 1, (n_samples, n_features))
        
        # Create some anomalies
        anomaly_data = np.random.normal(5, 2, (10, n_features))
        
        # Combine data
        all_data = np.vstack([normal_data, anomaly_data])
        
        # Create feature vectors dictionary
        feature_vectors = {}
        for i in range(len(all_data)):
            feature_vectors[f"test_image_{i}.png"] = all_data[i]
        
        # Test anomaly detection
        results = detector.fit(feature_vectors)
        
        if results is not None and len(results) == len(feature_vectors):
            print("✓ Anomaly detector (fitting) - PASSED")
            
            # Check if anomalies were detected
            anomalies = results[results['is_anomaly']]
            if len(anomalies) > 0:
                print(f"✓ Anomaly detector (detection) - PASSED ({len(anomalies)} anomalies found)")
            else:
                print("⚠ Anomaly detector (detection) - WARNING (no anomalies detected)")
            
            # Test prediction on new data
            new_data = np.random.normal(0, 1, (20, n_features))
            new_feature_vectors = {f"new_image_{i}.png": new_data[i] for i in range(20)}
            
            predictions = detector.predict(new_feature_vectors)
            if predictions is not None and len(predictions) == 20:
                print("✓ Anomaly detector (prediction) - PASSED")
            else:
                print("✗ Anomaly detector (prediction) - FAILED")
                
        else:
            print("✗ Anomaly detector (fitting) - FAILED")
            
    except Exception as e:
        print(f"✗ Anomaly detector - FAILED (error: {str(e)})")

def test_full_pipeline():
    """Test the full pipeline with synthetic data"""
    print("\n" + "="*50)
    print("Testing Full Pipeline")
    print("="*50)
    
    # This test requires actual PDF files, so we'll create a minimal test
    print("⚠ Full pipeline test requires actual PDF files")
    print("   Run the following command to test with real data:")
    print("   python main_pipeline.py --pdfs your_presentation.pdf --output_dir test_results")

def main():
    """Run all tests"""
    print("PDF to Anomaly Detection Pipeline - Component Tests")
    print("="*60)
    
    # Run component tests
    test_pdf_downloader()
    test_pdf_to_images()
    test_vit_encoder()
    test_quality_metrics()
    test_anomaly_detector()
    test_full_pipeline()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print("✓ All component tests completed")
    print("⚠ Some tests may be skipped if dependencies are missing")
    print("\nTo run the full pipeline test:")
    print("1. Place a PDF file in the current directory")
    print("2. Run: python main_pipeline.py --pdfs your_file.pdf --output_dir test_results")

if __name__ == "__main__":
    main() 