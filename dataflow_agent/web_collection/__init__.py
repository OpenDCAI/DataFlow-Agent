"""
Web Collection Module
~~~~~~~~~~~~~~~~~~~~~

Web数据收集功能模块，包含：
- utils: 工具类（分类器、任务分解器、RAG管理器）
- nodes: 工作流节点（搜索、下载、后处理、映射）
- downloaders: 数据下载器（HuggingFace、Kaggle、Web）
- research: 研究模块（Benchmark、RL训练、轨迹合成）
"""

from dataflow_agent.web_collection.utils import (
    CategoryClassifier,
    ObtainQueryNormalizer,
    TaskDecomposer,
    RAGManager,
)

from dataflow_agent.web_collection.nodes import (
    websearch_node,
    download_node,
    postprocess_node,
    mapping_node,
)

from dataflow_agent.web_collection.downloaders import (
    HuggingFaceDownloader,
    KaggleDownloader,
    WebDownloader,
)

# Research Module (延迟导入以避免循环依赖)
def get_research_module():
    """获取研究模块（延迟导入）"""
    from dataflow_agent.web_collection import research
    return research

__all__ = [
    # Utils
    "CategoryClassifier",
    "ObtainQueryNormalizer",
    "TaskDecomposer",
    "RAGManager",
    # Nodes
    "websearch_node",
    "download_node",
    "postprocess_node",
    "mapping_node",
    # Downloaders
    "HuggingFaceDownloader",
    "KaggleDownloader",
    "WebDownloader",
    # Research
    "get_research_module",
]
