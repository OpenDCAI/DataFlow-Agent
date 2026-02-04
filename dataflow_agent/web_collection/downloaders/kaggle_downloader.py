"""
Kaggle Downloader
~~~~~~~~~~~~~~~~~

Kaggle 数据集下载器
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


class KaggleDownloader:
    """Kaggle dataset downloader"""
    
    def __init__(
        self, 
        download_dir: str = "./downloads",
        username: Optional[str] = None,
        key: Optional[str] = None,
    ):
        """
        Initialize Kaggle downloader
        
        Args:
            download_dir: Directory to save downloads
            username: Kaggle username
            key: Kaggle API key
        """
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        
        # Set credentials if provided
        if username:
            os.environ["KAGGLE_USERNAME"] = username
        if key:
            os.environ["KAGGLE_KEY"] = key
        
        # Check if kaggle is available
        try:
            import kaggle
            self.kaggle = kaggle
            self.available = True
        except ImportError:
            logger.warning("kaggle not installed")
            self.kaggle = None
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to initialize kaggle: {e}")
            self.kaggle = None
            self.available = False
    
    async def search_datasets(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for datasets on Kaggle
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of dataset info dictionaries
        """
        if not self.available:
            return []
        
        try:
            datasets = self.kaggle.api.dataset_list(search=query)[:limit]
            return [
                {
                    "ref": ds.ref,
                    "title": getattr(ds, "title", ""),
                    "size": getattr(ds, "totalBytes", 0),
                    "votes": getattr(ds, "voteCount", 0),
                }
                for ds in datasets
            ]
        except Exception as e:
            logger.error(f"Failed to search Kaggle datasets: {e}")
            return []
    
    async def download_dataset(
        self,
        dataset_ref: str,
        unzip: bool = True,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Download a dataset from Kaggle
        
        Args:
            dataset_ref: Dataset reference (e.g., "username/dataset-name")
            unzip: Whether to unzip downloaded files
            
        Returns:
            Tuple of (download_path, error_message)
        """
        if not self.available:
            return None, "kaggle not installed"
        
        try:
            # Create download directory
            dataset_dir = os.path.join(
                self.download_dir, 
                dataset_ref.replace("/", "_")
            )
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Download dataset
            self.kaggle.api.dataset_download_files(
                dataset_ref,
                path=dataset_dir,
                unzip=unzip,
            )
            
            logger.info(f"Downloaded {dataset_ref} to {dataset_dir}")
            return dataset_dir, None
            
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_ref}: {e}")
            return None, str(e)
    
    async def get_dataset_info(self, dataset_ref: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a dataset
        
        Args:
            dataset_ref: Dataset reference
            
        Returns:
            Dataset info dictionary or None
        """
        if not self.available:
            return None
        
        try:
            # Get dataset metadata
            datasets = self.kaggle.api.dataset_list(search=dataset_ref)
            for ds in datasets:
                if ds.ref == dataset_ref:
                    return {
                        "ref": ds.ref,
                        "title": getattr(ds, "title", ""),
                        "size": getattr(ds, "totalBytes", 0),
                        "description": getattr(ds, "description", ""),
                    }
            return None
        except Exception as e:
            logger.error(f"Failed to get dataset info for {dataset_ref}: {e}")
            return None
