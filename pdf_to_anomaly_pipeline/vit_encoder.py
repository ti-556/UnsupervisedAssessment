import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViTEncoder:
    """Vision Transformer encoder for extracting image embeddings"""
    
    def __init__(self, model_name="google/vit-base-patch16-224", device=None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing ViT encoder with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the ViT model and processor"""
        try:
            logger.info("Loading ViT model and processor...")
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("ViT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ViT model: {str(e)}")
            raise
    
    def extract_embedding(self, image_path):
        """Extract embedding from a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding.flatten()
            
        except Exception as e:
            logger.error(f"Error extracting embedding from {image_path}: {str(e)}")
            return None
    
    def extract_embeddings_batch(self, image_paths, batch_size=8):
        """Extract embeddings from multiple images in batches"""
        embeddings = {}
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_names = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(image)
                    batch_names.append(path)
                except Exception as e:
                    logger.warning(f"Error loading image {path}: {str(e)}")
                    continue
            
            if not batch_images:
                continue
            
            try:
                # Process batch
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Store embeddings
                for j, name in enumerate(batch_names):
                    embeddings[name] = batch_embeddings[j]
                    
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
        
        return embeddings
    
    def reduce_dimensions(self, embeddings, target_dim=64, method='pca'):
        """Reduce embedding dimensions using PCA or other methods"""
        from sklearn.decomposition import PCA
        
        if method == 'pca':
            pca = PCA(n_components=target_dim)
            reduced_embeddings = pca.fit_transform(embeddings)
            logger.info(f"Reduced embeddings from {embeddings.shape[1]} to {target_dim} dimensions")
            return reduced_embeddings, pca
        else:
            logger.warning(f"Method {method} not implemented, returning original embeddings")
            return embeddings, None
    
    def save_embeddings(self, embeddings, output_path):
        """Save embeddings to file"""
        try:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Embeddings saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
    
    def load_embeddings(self, input_path):
        """Load embeddings from file"""
        try:
            import pickle
            with open(input_path, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"Embeddings loaded from {input_path}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return None 