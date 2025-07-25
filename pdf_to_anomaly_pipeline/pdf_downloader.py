import os
import requests
import shutil
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFDownloader:
    """Download PDF files from URLs or copy from local paths"""
    
    def __init__(self, output_dir="downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def download_from_url(self, url, filename=None):
        """Download PDF from URL"""
        try:
            if filename is None:
                filename = os.path.basename(urlparse(url).path)
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            output_path = self.output_dir / filename
            
            logger.info(f"Downloading {url} to {output_path}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded {filename}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return None
    
    def copy_from_local(self, local_path, filename=None):
        """Copy PDF from local path"""
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error(f"File not found: {local_path}")
                return None
            
            if filename is None:
                filename = local_path.name
            
            output_path = self.output_dir / filename
            
            logger.info(f"Copying {local_path} to {output_path}")
            shutil.copy2(local_path, output_path)
            
            logger.info(f"Successfully copied {filename}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error copying {local_path}: {str(e)}")
            return None
    
    def download_multiple(self, sources):
        """Download multiple PDFs from list of URLs or local paths"""
        downloaded_files = []
        
        for source in sources:
            if source.startswith(('http://', 'https://')):
                result = self.download_from_url(source)
            else:
                result = self.copy_from_local(source)
            
            if result:
                downloaded_files.append(result)
        
        return downloaded_files
    
    def list_downloaded(self):
        """List all downloaded PDF files"""
        pdf_files = list(self.output_dir.glob("*.pdf"))
        return pdf_files 