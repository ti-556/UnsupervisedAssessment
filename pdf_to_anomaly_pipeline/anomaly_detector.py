import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Anomaly detection using Isolation Forest on 71D feature vectors"""
    
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = None
        self.scaler = None
        self.pca = None
        self.is_fitted = False
    
    def prepare_features(self, vit_embeddings, quality_metrics):
        """Combine ViT embeddings (64D) and quality metrics (7D) into 71D feature vectors"""
        feature_vectors = {}
        
        for image_path in vit_embeddings.keys():
            if image_path in quality_metrics:
                # Get ViT embedding (64D)
                vit_embedding = vit_embeddings[image_path]
                
                # Get quality metrics (7D)
                metrics = quality_metrics[image_path]
                metrics_vector = np.array([
                    metrics['brightness'],
                    metrics['contrast'],
                    metrics['sharpness'],
                    metrics['noise_level'],
                    metrics['color_variety'],
                    metrics['edge_density'],
                    metrics['texture_complexity']
                ])
                
                # Combine into 71D vector
                combined_vector = np.concatenate([vit_embedding, metrics_vector])
                feature_vectors[image_path] = combined_vector
        
        return feature_vectors
    
    def fit(self, feature_vectors):
        """Fit the anomaly detection model"""
        logger.info("Fitting anomaly detection model...")
        
        # Convert to numpy array
        paths = list(feature_vectors.keys())
        features = np.array([feature_vectors[path] for path in paths])
        
        logger.info(f"Feature matrix shape: {features.shape}")
        
        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Optional: Apply PCA for dimensionality reduction
        if features.shape[1] > 50:
            n_components = min(50, features.shape[1] // 2)
            self.pca = PCA(n_components=n_components)
            features_reduced = self.pca.fit_transform(features_scaled)
            logger.info(f"Reduced features to {n_components} dimensions using PCA")
        else:
            features_reduced = features_scaled
        
        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        self.isolation_forest.fit(features_reduced)
        self.is_fitted = True
        
        # Calculate anomaly scores for training data
        anomaly_scores = self.isolation_forest.decision_function(features_reduced)
        predictions = self.isolation_forest.predict(features_reduced)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'image_path': paths,
            'anomaly_score': anomaly_scores,
            'is_anomaly': predictions == -1,
            'prediction': predictions
        })
        
        logger.info(f"Model fitted successfully. Found {np.sum(predictions == -1)} anomalies out of {len(predictions)} samples")
        
        return results
    
    def predict(self, feature_vectors):
        """Predict anomalies for new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy array
        paths = list(feature_vectors.keys())
        features = np.array([feature_vectors[path] for path in paths])
        
        # Apply same preprocessing as training
        features_scaled = self.scaler.transform(features)
        
        if self.pca is not None:
            features_reduced = self.pca.transform(features_scaled)
        else:
            features_reduced = features_scaled
        
        # Make predictions
        anomaly_scores = self.isolation_forest.decision_function(features_reduced)
        predictions = self.isolation_forest.predict(features_reduced)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'image_path': paths,
            'anomaly_score': anomaly_scores,
            'is_anomaly': predictions == -1,
            'prediction': predictions
        })
        
        return results
    
    def analyze_results(self, results, output_dir="anomaly_results"):
        """Analyze and visualize anomaly detection results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Basic statistics
        total_samples = len(results)
        anomalies = results[results['is_anomaly']]
        normal_samples = results[~results['is_anomaly']]
        
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/total_samples*100:.2f}%)")
        logger.info(f"Normal samples: {len(normal_samples)} ({len(normal_samples)/total_samples*100:.2f}%)")
        
        # Save detailed results
        results.to_csv(output_path / "anomaly_detection_results.csv", index=False)
        
        # Create visualizations
        self._create_visualizations(results, output_path)
        
        return {
            'total_samples': total_samples,
            'anomalies': len(anomalies),
            'normal_samples': len(normal_samples),
            'anomaly_percentage': len(anomalies)/total_samples*100
        }
    
    def _create_visualizations(self, results, output_path):
        """Create visualization plots"""
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Anomaly score distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Anomaly score histogram
        axes[0, 0].hist(results['anomaly_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Anomaly Scores')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(results['anomaly_score'].median(), color='red', linestyle='--', label='Median')
        axes[0, 0].legend()
        
        # Anomaly score by class
        normal_scores = results[~results['is_anomaly']]['anomaly_score']
        anomaly_scores = results[results['is_anomaly']]['anomaly_score']
        
        axes[0, 1].boxplot([normal_scores, anomaly_scores], labels=['Normal', 'Anomaly'])
        axes[0, 1].set_title('Anomaly Scores by Class')
        axes[0, 1].set_ylabel('Anomaly Score')
        
        # Pie chart of results
        anomaly_count = len(results[results['is_anomaly']])
        normal_count = len(results[~results['is_anomaly']])
        
        axes[1, 0].pie([normal_count, anomaly_count], 
                      labels=['Normal', 'Anomaly'], 
                      autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral'])
        axes[1, 0].set_title('Distribution of Predictions')
        
        # Anomaly score vs sample index
        axes[1, 1].scatter(range(len(results)), results['anomaly_score'], 
                          c=results['is_anomaly'], cmap='RdYlBu', alpha=0.6)
        axes[1, 1].set_title('Anomaly Scores by Sample Index')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Anomaly Score')
        
        plt.tight_layout()
        plt.savefig(output_path / "anomaly_analysis_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed anomaly analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Anomaly score distribution with threshold
        threshold = np.percentile(results['anomaly_score'], (1 - self.contamination) * 100)
        
        axes[0].hist(results['anomaly_score'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
        axes[0].set_title('Anomaly Score Distribution with Threshold')
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Cumulative distribution
        sorted_scores = np.sort(results['anomaly_score'])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        
        axes[1].plot(sorted_scores, cumulative, linewidth=2)
        axes[1].axhline(1 - self.contamination, color='red', linestyle='--', label=f'Contamination level ({self.contamination})')
        axes[1].set_title('Cumulative Distribution of Anomaly Scores')
        axes[1].set_xlabel('Anomaly Score')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "detailed_anomaly_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def save_model(self, output_path):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'pca': self.pca,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, input_path):
        """Load a trained model"""
        with open(input_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {input_path}") 