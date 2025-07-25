#!/usr/bin/env python3
"""
Main Pipeline for PDF to Anomaly Detection
==========================================

This script orchestrates the complete pipeline:
1. Download PDFs from URLs or copy from local paths
2. Convert PDFs to images
3. Extract ViT embeddings (64D)
4. Extract quality metrics (7D)
5. Combine into 71D feature vectors
6. Perform anomaly detection using Isolation Forest
7. Generate analysis and visualizations

Usage:
    python main_pipeline.py --pdfs path1.pdf path2.pdf --urls url1 url2 --output_dir results
"""

import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import time

# Import our modules
from pdf_downloader import PDFDownloader
from pdf_to_images import PDFToImageConverter
from vit_encoder import ViTEncoder
from quality_metrics import QualityMetricsExtractor
from anomaly_detector import AnomalyDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PDFAnomalyPipeline:
    """Complete pipeline for PDF to anomaly detection"""
    
    def __init__(self, output_dir="pipeline_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pdf_downloader = PDFDownloader(output_dir=self.output_dir / "downloads")
        self.image_converter = PDFToImageConverter(output_dir=self.output_dir / "images")
        self.vit_encoder = ViTEncoder()
        self.metrics_extractor = QualityMetricsExtractor()
        self.anomaly_detector = AnomalyDetector()
        
        logger.info("Pipeline initialized successfully")
    
    def run_pipeline(self, pdf_sources, contamination=0.1):
        """Run the complete pipeline"""
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("Starting PDF to Anomaly Detection Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Download/Copy PDFs
        logger.info("\nStep 1: Downloading/Copying PDFs...")
        pdf_files = self._download_pdfs(pdf_sources)
        
        if not pdf_files:
            logger.error("No PDF files available. Exiting.")
            return None
        
        # Step 2: Convert PDFs to images
        logger.info("\nStep 2: Converting PDFs to images...")
        image_paths = self._convert_pdfs_to_images(pdf_files)
        
        if not image_paths:
            logger.error("No images generated. Exiting.")
            return None
        
        # Step 3: Extract ViT embeddings
        logger.info("\nStep 3: Extracting ViT embeddings...")
        vit_embeddings = self._extract_vit_embeddings(image_paths)
        
        if not vit_embeddings:
            logger.error("No ViT embeddings extracted. Exiting.")
            return None
        
        # Step 4: Extract quality metrics
        logger.info("\nStep 4: Extracting quality metrics...")
        quality_metrics = self._extract_quality_metrics(image_paths)
        
        if not quality_metrics:
            logger.error("No quality metrics extracted. Exiting.")
            return None
        
        # Step 5: Prepare feature vectors
        logger.info("\nStep 5: Preparing 71D feature vectors...")
        feature_vectors = self._prepare_feature_vectors(vit_embeddings, quality_metrics)
        
        if not feature_vectors:
            logger.error("No feature vectors prepared. Exiting.")
            return None
        
        # Step 6: Anomaly detection
        logger.info("\nStep 6: Performing anomaly detection...")
        results = self._perform_anomaly_detection(feature_vectors, contamination)
        
        if results is None:
            logger.error("Anomaly detection failed. Exiting.")
            return None
        
        # Step 7: Analysis and visualization
        logger.info("\nStep 7: Generating analysis and visualizations...")
        analysis_results = self._generate_analysis(results)
        
        # Save intermediate results
        self._save_intermediate_results(vit_embeddings, quality_metrics, feature_vectors)
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"\nPipeline completed in {total_time:.2f} seconds")
        
        return {
            'results': results,
            'analysis': analysis_results,
            'total_time': total_time,
            'num_pdfs': len(pdf_files),
            'num_images': len(image_paths),
            'num_features': len(feature_vectors)
        }
    
    def _download_pdfs(self, pdf_sources):
        """Download or copy PDFs from various sources"""
        downloaded_files = []
        
        for source in pdf_sources:
            if source.startswith(('http://', 'https://')):
                result = self.pdf_downloader.download_from_url(source)
            else:
                result = self.pdf_downloader.copy_from_local(source)
            
            if result:
                downloaded_files.append(result)
        
        logger.info(f"Successfully processed {len(downloaded_files)} PDF files")
        return downloaded_files
    
    def _convert_pdfs_to_images(self, pdf_files):
        """Convert PDFs to images"""
        all_image_paths = []
        
        for pdf_file in pdf_files:
            image_paths = self.image_converter.convert_pdf(pdf_file)
            all_image_paths.extend(image_paths)
        
        logger.info(f"Successfully converted {len(all_image_paths)} images")
        return all_image_paths
    
    def _extract_vit_embeddings(self, image_paths):
        """Extract ViT embeddings from images"""
        # Extract full embeddings first
        full_embeddings = self.vit_encoder.extract_embeddings_batch(image_paths)
        
        # Reduce to 64D using PCA
        if full_embeddings:
            embeddings_array = np.array(list(full_embeddings.values()))
            reduced_embeddings, pca_model = self.vit_encoder.reduce_dimensions(
                embeddings_array, target_dim=64
            )
            
            # Convert back to dictionary
            reduced_embeddings_dict = {}
            for i, path in enumerate(full_embeddings.keys()):
                reduced_embeddings_dict[path] = reduced_embeddings[i]
            
            # Save PCA model for future use
            import joblib
            joblib.dump(pca_model, self.output_dir / "vit_pca_model.pkl")
            
            logger.info(f"Successfully extracted 64D ViT embeddings for {len(reduced_embeddings_dict)} images")
            return reduced_embeddings_dict
        
        return None
    
    def _extract_quality_metrics(self, image_paths):
        """Extract quality metrics from images"""
        metrics = self.metrics_extractor.extract_metrics_batch(image_paths)
        
        if metrics:
            # Normalize metrics
            normalized_metrics, scaler = self.metrics_extractor.normalize_metrics(metrics)
            
            # Save scaler for future use
            import joblib
            joblib.dump(scaler, self.output_dir / "metrics_scaler.pkl")
            
            logger.info(f"Successfully extracted quality metrics for {len(normalized_metrics)} images")
            return normalized_metrics
        
        return None
    
    def _prepare_feature_vectors(self, vit_embeddings, quality_metrics):
        """Combine ViT embeddings and quality metrics into 71D feature vectors"""
        feature_vectors = self.anomaly_detector.prepare_features(vit_embeddings, quality_metrics)
        
        logger.info(f"Successfully prepared 71D feature vectors for {len(feature_vectors)} images")
        return feature_vectors
    
    def _perform_anomaly_detection(self, feature_vectors, contamination):
        """Perform anomaly detection"""
        self.anomaly_detector.contamination = contamination
        results = self.anomaly_detector.fit(feature_vectors)
        
        # Save the trained model
        self.anomaly_detector.save_model(self.output_dir / "anomaly_detection_model.pkl")
        
        return results
    
    def _generate_analysis(self, results):
        """Generate analysis and visualizations"""
        analysis_results = self.anomaly_detector.analyze_results(
            results, 
            output_dir=self.output_dir / "anomaly_results"
        )
        
        return analysis_results
    
    def _save_intermediate_results(self, vit_embeddings, quality_metrics, feature_vectors):
        """Save intermediate results for future use"""
        # Save ViT embeddings
        self.vit_encoder.save_embeddings(
            vit_embeddings, 
            self.output_dir / "vit_embeddings_64d.pkl"
        )
        
        # Save quality metrics
        self.metrics_extractor.save_metrics(
            quality_metrics, 
            self.output_dir / "quality_metrics.pkl"
        )
        
        # Save feature vectors
        import pickle
        with open(self.output_dir / "feature_vectors_71d.pkl", 'wb') as f:
            pickle.dump(feature_vectors, f)
        
        logger.info("Intermediate results saved")

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description="PDF to Anomaly Detection Pipeline")
    
    parser.add_argument(
        "--pdfs", 
        nargs="+", 
        help="Local PDF file paths"
    )
    
    parser.add_argument(
        "--urls", 
        nargs="+", 
        help="URLs to download PDFs from"
    )
    
    parser.add_argument(
        "--output_dir", 
        default="pipeline_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--contamination", 
        type=float, 
        default=0.1,
        help="Expected fraction of anomalies (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Prepare sources
    sources = []
    if args.pdfs:
        sources.extend(args.pdfs)
    if args.urls:
        sources.extend(args.urls)
    
    if not sources:
        logger.error("No PDF sources provided. Use --pdfs or --urls arguments.")
        return
    
    # Run pipeline
    pipeline = PDFAnomalyPipeline(output_dir=args.output_dir)
    results = pipeline.run_pipeline(sources, contamination=args.contamination)
    
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Summary:")
        logger.info(f"  - PDFs processed: {results['num_pdfs']}")
        logger.info(f"  - Images generated: {results['num_images']}")
        logger.info(f"  - Features extracted: {results['num_features']}")
        logger.info(f"  - Anomalies detected: {results['analysis']['anomalies']}")
        logger.info(f"  - Total time: {results['total_time']:.2f} seconds")
        logger.info(f"  - Results saved to: {args.output_dir}")
        logger.info("=" * 60)
    else:
        logger.error("Pipeline failed to complete successfully")

if __name__ == "__main__":
    import numpy as np
    main() 