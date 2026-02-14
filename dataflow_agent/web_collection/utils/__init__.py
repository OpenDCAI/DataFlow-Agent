"""
Web Collection Utils
~~~~~~~~~~~~~~~~~~~~

工具类模块，包含：
- CategoryClassifier: 任务分类器（SFT/PT）
- ObtainQueryNormalizer: 查询标准化器
- TaskDecomposer: 任务分解器
- RAGManager: RAG向量存储管理器
- QueryGenerator: 搜索查询生成器
- SummaryAgent: 下载子任务生成代理
- URLSelector: URL智能筛选器
- WebTools: 网页搜索和读取工具
- DataConvertor: 数据格式转换器
- WebCrawlerOrchestrator: WebCrawler 爬取编排器（从网页提取代码块）
- webcrawler_dataset_generator: WebCrawler 数据集生成工具（SFT/PT）
"""

from dataflow_agent.web_collection.utils.category_classifier import (
    CategoryClassifier,
    ObtainQueryNormalizer,
    TaskDecomposer,
)
from dataflow_agent.web_collection.utils.rag_manager import RAGManager
from dataflow_agent.web_collection.utils.query_generator import QueryGenerator
from dataflow_agent.web_collection.utils.summary_agent import SummaryAgent
from dataflow_agent.web_collection.utils.url_selector import URLSelector
from dataflow_agent.web_collection.utils.web_tools import WebTools
from dataflow_agent.web_collection.utils.data_convertor import DataConvertor
from dataflow_agent.web_collection.utils.webcrawler_orchestrator import (
    WebCrawlerOrchestrator,
    CrawledContent,
    extract_code_blocks_from_markdown,
)
from dataflow_agent.web_collection.utils.webcrawler_dataset_generator import (
    generate_sft_records,
    generate_pt_records,
    generate_webpage_summary_and_relevance,
)

__all__ = [
    "CategoryClassifier",
    "ObtainQueryNormalizer",
    "TaskDecomposer",
    "RAGManager",
    "QueryGenerator",
    "SummaryAgent",
    "URLSelector",
    "WebTools",
    "DataConvertor",
    # WebCrawler 相关
    "WebCrawlerOrchestrator",
    "CrawledContent",
    "extract_code_blocks_from_markdown",
    "generate_sft_records",
    "generate_pt_records",
    "generate_webpage_summary_and_relevance",
]
