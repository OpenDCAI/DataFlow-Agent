"""
Web Collection Downloaders
~~~~~~~~~~~~~~~~~~~~~~~~~~

数据下载器模块，包含：
- HuggingFaceDownloader: HuggingFace数据集下载
- KaggleDownloader: Kaggle数据集下载
- WebDownloader: 通用网页下载（PlaywrightToolKit 的别名，向后兼容）
- DomFetcher: 使用 Playwright + stealth 获取动态渲染页面的 DOM
- PlaywrightToolKit: 浏览器自动化工具包
- WebAgent: 基于 LLM 的智能网页探索代理
- WebsearchResearcherAgent: 基于 LLM 的智能网页搜索研究代理
"""

from dataflow_agent.web_collection.downloaders.hf_downloader import HuggingFaceDownloader
from dataflow_agent.web_collection.downloaders.kaggle_downloader import KaggleDownloader
from dataflow_agent.web_collection.downloaders.web_downloader import (
    WebDownloader,
    DomFetcher,
    PlaywrightToolKit,
    WebAgent,
    WebsearchResearcherAgent,
    create_websearch_researcher_agent,
)

__all__ = [
    "HuggingFaceDownloader",
    "KaggleDownloader",
    "WebDownloader",
    "DomFetcher",
    "PlaywrightToolKit",
    "WebAgent",
    "WebsearchResearcherAgent",
    "create_websearch_researcher_agent",
]
