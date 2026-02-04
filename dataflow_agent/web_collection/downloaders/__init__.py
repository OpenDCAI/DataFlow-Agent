"""
Web Collection Downloaders
~~~~~~~~~~~~~~~~~~~~~~~~~~

数据下载器模块，包含：
- HuggingFaceDownloader: HuggingFace数据集下载
- KaggleDownloader: Kaggle数据集下载
- WebDownloader: 通用网页下载
"""

from dataflow_agent.web_collection.downloaders.hf_downloader import HuggingFaceDownloader
from dataflow_agent.web_collection.downloaders.kaggle_downloader import KaggleDownloader
from dataflow_agent.web_collection.downloaders.web_downloader import WebDownloader

__all__ = [
    "HuggingFaceDownloader",
    "KaggleDownloader",
    "WebDownloader",
]
