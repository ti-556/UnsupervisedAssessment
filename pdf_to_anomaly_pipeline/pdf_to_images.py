import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFToImageConverter:
    """Convert PDF pages to images"""
    
    def __init__(self, output_dir="images", dpi=300, format='PNG'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        self.format = format
    
    def convert_pdf(self, pdf_path, output_subdir=None):
        """Convert a single PDF to images"""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return []
        
        if output_subdir is None:
            output_subdir = pdf_path.stem
        
        output_path = self.output_dir / output_subdir
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Converting {pdf_path} to images in {output_path}")
        
        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path, 
                dpi=self.dpi, 
                fmt=self.format.lower()
            )
            
            image_paths = []
            
            for i, image in enumerate(tqdm(images, desc="Converting pages")):
                image_filename = f"slide_{i+1:03d}.{self.format.lower()}"
                image_path = output_path / image_filename
                
                image.save(image_path, self.format)
                image_paths.append(image_path)
                
                logger.debug(f"Saved {image_filename}")
            
            logger.info(f"Successfully converted {len(images)} pages from {pdf_path.name}")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting {pdf_path}: {str(e)}")
            return []
    
    def convert_multiple_pdfs(self, pdf_paths, output_subdirs=None):
        """Convert multiple PDFs to images"""
        all_image_paths = {}
        
        for i, pdf_path in enumerate(pdf_paths):
            if output_subdirs and i < len(output_subdirs):
                subdir = output_subdirs[i]
            else:
                subdir = None
            
            image_paths = self.convert_pdf(pdf_path, subdir)
            if image_paths:
                all_image_paths[pdf_path] = image_paths
        
        return all_image_paths
    
    def get_image_info(self, image_path):
        """Get basic information about an image"""
        try:
            with Image.open(image_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_mb': os.path.getsize(image_path) / (1024 * 1024)
                }
        except Exception as e:
            logger.error(f"Error getting image info for {image_path}: {str(e)}")
            return None
    
    def list_converted_images(self):
        """List all converted images"""
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(self.output_dir.rglob(ext))
        return sorted(image_files) 