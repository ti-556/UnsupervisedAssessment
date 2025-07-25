import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityMetricsExtractor:
    """Extract quality metrics from images"""
    
    def __init__(self):
        self.metrics_names = [
            'brightness',
            'contrast', 
            'sharpness',
            'noise_level',
            'color_variety',
            'edge_density',
            'texture_complexity'
        ]
    
    def extract_brightness(self, image):
        """Calculate average brightness"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return np.mean(gray)
    
    def extract_contrast(self, image):
        """Calculate contrast using standard deviation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return np.std(gray)
    
    def extract_sharpness(self, image):
        """Calculate sharpness using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def extract_noise_level(self, image):
        """Estimate noise level using median absolute deviation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to get smooth version
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Calculate difference
        diff = cv2.absdiff(gray, blurred)
        return np.mean(diff)
    
    def extract_color_variety(self, image):
        """Calculate color variety using color histogram"""
        if len(image.shape) == 3:
            # Convert to HSV and calculate histogram
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            # Normalize and calculate entropy
            hist_norm = hist / np.sum(hist)
            hist_norm = hist_norm[hist_norm > 0]  # Remove zeros
            entropy = -np.sum(hist_norm * np.log2(hist_norm))
            return entropy
        else:
            return 0.0
    
    def extract_edge_density(self, image):
        """Calculate edge density using Canny edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    def extract_texture_complexity(self, image):
        """Calculate texture complexity using GLCM-like features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate local binary pattern-like feature
        # Simple approach: calculate variance of local patches
        patch_size = 8
        h, w = gray.shape
        variances = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                variances.append(np.var(patch))
        
        return np.mean(variances) if variances else 0.0
    
    def extract_all_metrics(self, image_path):
        """Extract all quality metrics from an image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract all metrics
            metrics = {
                'brightness': self.extract_brightness(image),
                'contrast': self.extract_contrast(image),
                'sharpness': self.extract_sharpness(image),
                'noise_level': self.extract_noise_level(image),
                'color_variety': self.extract_color_variety(image),
                'edge_density': self.extract_edge_density(image),
                'texture_complexity': self.extract_texture_complexity(image)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting metrics from {image_path}: {str(e)}")
            return None
    
    def extract_metrics_batch(self, image_paths):
        """Extract metrics from multiple images"""
        all_metrics = {}
        
        for image_path in tqdm(image_paths, desc="Extracting quality metrics"):
            metrics = self.extract_all_metrics(image_path)
            if metrics:
                all_metrics[image_path] = metrics
        
        return all_metrics
    
    def normalize_metrics(self, metrics_dict):
        """Normalize metrics to 0-1 range"""
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        
        # Normalize each metric
        scaler = MinMaxScaler()
        normalized_df = pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        
        return normalized_df.to_dict('index'), scaler
    
    def save_metrics(self, metrics, output_path):
        """Save metrics to file"""
        try:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(metrics, f)
            logger.info(f"Metrics saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def load_metrics(self, input_path):
        """Load metrics from file"""
        try:
            import pickle
            with open(input_path, 'rb') as f:
                metrics = pickle.load(f)
            logger.info(f"Metrics loaded from {input_path}")
            return metrics
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")
            return None 