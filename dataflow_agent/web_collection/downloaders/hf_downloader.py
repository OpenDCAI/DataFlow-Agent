"""
HuggingFace Downloader
~~~~~~~~~~~~~~~~~~~~~~

HuggingFace 数据集下载器
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceDownloader:
    """HuggingFace dataset downloader"""
    
    def __init__(self, download_dir: str = "./downloads"):
        """
        Initialize HuggingFace downloader
        
        Args:
            download_dir: Directory to save downloads
        """
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        
        # Check if huggingface_hub is available
        try:
            from huggingface_hub import HfApi
            self.hf_api = HfApi()
            self.available = True
        except ImportError:
            logger.warning("huggingface_hub not installed")
            self.hf_api = None
            self.available = False
    
    async def search_datasets(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for datasets on HuggingFace
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of dataset info dictionaries
        """
        if not self.available:
            return []
        
        try:
            datasets = list(self.hf_api.list_datasets(search=query, limit=limit))
            return [
                {
                    "id": ds.id,
                    "downloads": getattr(ds, "downloads", 0),
                    "likes": getattr(ds, "likes", 0),
                    "tags": getattr(ds, "tags", []),
                }
                for ds in datasets
            ]
        except Exception as e:
            logger.error(f"Failed to search HuggingFace datasets: {e}")
            return []
    
    async def download_dataset(
        self,
        dataset_id: str,
        split: str = "train",
        max_samples: Optional[int] = 1000,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Download a dataset from HuggingFace
        
        Args:
            dataset_id: Dataset identifier (e.g., "squad")
            split: Dataset split to download
            max_samples: Maximum number of samples to download
            
        Returns:
            Tuple of (download_path, error_message)
        """
        if not self.available:
            return None, "huggingface_hub not installed"
        
        try:
            # Create download directory
            dataset_dir = os.path.join(
                self.download_dir, 
                dataset_id.replace("/", "_")
            )
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Try using datasets library
            try:
                from datasets import load_dataset
                
                # Load dataset with limit
                if max_samples:
                    ds = load_dataset(
                        dataset_id, 
                        split=f"{split}[:{max_samples}]",
                        trust_remote_code=True
                    )
                else:
                    ds = load_dataset(
                        dataset_id, 
                        split=split,
                        trust_remote_code=True
                    )
                
                # Save to JSONL
                output_path = os.path.join(dataset_dir, f"{split}.jsonl")
                ds.to_json(output_path)
                
                logger.info(f"Downloaded {dataset_id} to {output_path}")
                return dataset_dir, None
                
            except Exception as e:
                logger.warning(f"Failed to load dataset with datasets library: {e}")
                
                # Fallback: download files directly
                from huggingface_hub import hf_hub_download
                
                files = self.hf_api.list_repo_files(dataset_id, repo_type="dataset")
                downloaded_files = []
                
                for f in files[:5]:  # Limit to first 5 files
                    if f.endswith(('.json', '.jsonl', '.csv', '.parquet', '.txt')):
                        try:
                            local_path = hf_hub_download(
                                repo_id=dataset_id,
                                filename=f,
                                repo_type="dataset",
                                local_dir=dataset_dir,
                            )
                            downloaded_files.append(local_path)
                        except Exception as fe:
                            logger.warning(f"Failed to download {f}: {fe}")
                
                if downloaded_files:
                    return dataset_dir, None
                else:
                    return None, "No files downloaded"
                    
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_id}: {e}")
            return None, str(e)
    
    async def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a dataset
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dataset info dictionary or None
        """
        if not self.available:
            return None
        
        try:
            info = self.hf_api.dataset_info(dataset_id)
            return {
                "id": info.id,
                "description": getattr(info, "description", ""),
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "tags": getattr(info, "tags", []),
            }
        except Exception as e:
            logger.error(f"Failed to get dataset info for {dataset_id}: {e}")
            return None
