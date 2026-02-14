"""
Web Collection Nodes
~~~~~~~~~~~~~~~~~~~~

工作流节点模块，包含：
- websearch_node: 网页搜索节点
- download_node: 数据下载节点
- postprocess_node: 后处理节点
- mapping_node: 格式映射节点
- webcrawler_node: WebCrawler 爬取节点（从网页提取代码块）
- webcrawler_dataset_node: WebCrawler 数据集生成节点（生成 SFT/PT）
"""

from dataflow_agent.web_collection.nodes.websearch_node import websearch_node
from dataflow_agent.web_collection.nodes.download_node import download_node
from dataflow_agent.web_collection.nodes.postprocess_node import postprocess_node
from dataflow_agent.web_collection.nodes.mapping_node import mapping_node
from dataflow_agent.web_collection.nodes.webcrawler_node import webcrawler_node
from dataflow_agent.web_collection.nodes.webcrawler_dataset_node import webcrawler_dataset_node

__all__ = [
    "websearch_node",
    "download_node",
    "postprocess_node",
    "mapping_node",
    "webcrawler_node",
    "webcrawler_dataset_node",
]
