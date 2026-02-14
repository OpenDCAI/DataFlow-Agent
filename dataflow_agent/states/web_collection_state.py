"""
WebCollection State and Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Web数据收集任务的状态定义

功能包括：
- 任务分解与规划
- 网页搜索与探索
- 数据下载与后处理
- 数据清洗与筛选
- Alpaca格式映射
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from dataflow_agent.state import MainState, MainRequest, STATICS_DIR


# ==================== WebCollection Request ====================
@dataclass
class WebCollectionRequest(MainRequest):
    """Web数据收集任务的Request，继承自MainRequest"""
    
    # === 任务配置 ===
    category: str = "PT"  # PT (Pre-Training) 或 SFT (Supervised Fine-Tuning)
    output_format: str = "alpaca"  # 目标输出格式
    download_dir: str = field(default_factory=lambda: os.path.join(STATICS_DIR, "web_collection"))
    
    # === 搜索配置 ===
    search_engine: str = "tavily"  # 搜索引擎: tavily, google, bing, duckduckgo
    max_depth: int = 2  # 爬取最大深度
    max_urls: int = 10  # 单次搜索最大处理URL数量
    concurrent_limit: int = 5  # 并发请求限制
    topk_urls: int = 5  # 保留最相关的URL数量
    url_timeout: int = 60  # URL请求超时时间(秒)
    recursion_limit: int = 50  # 递归/重试限制次数
    max_download_subtasks: Optional[int] = 5  # 每个子任务最多下载子任务数
    
    # === RAG 配置 ===
    enable_rag: bool = True
    reset_rag: bool = False
    rag_collection_name: str = "web_collection_rag"
    rag_embed_model: str = ""
    rag_api_base_url: Optional[str] = None
    rag_api_key: Optional[str] = None
    
    # === 外部 API Keys ===
    tavily_api_key: Optional[str] = None
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None
    
    # === 映射配置 ===
    default_mapping_format: str = "alpaca"  # 默认映射格式
    mapping_auto_mode: bool = True  # 自动映射模式
    
    # === 处理配置 ===
    llm_timeout: float = 120.0  # LLM调用超时时间(秒)
    max_retries: int = 3  # 最大重试次数
    max_concurrent_mapping: int = 10  # 数据映射的最大并发数
    temperature: float = 0.7  # LLM 温度参数
    debug: bool = False  # 调试模式
    
    # === WebCrawler 配置 ===
    enable_webcrawler: bool = True  # 是否启用 WebCrawler 并行爬取
    webcrawler_num_queries: int = 5  # 生成的搜索查询数量
    webcrawler_crawl_depth: int = 3  # 爬取深度
    webcrawler_max_links_per_page: int = 5  # 每页最大链接数
    webcrawler_concurrent_pages: int = 3  # 并发爬取数
    webcrawler_min_code_length: int = 50  # 最小代码长度（字符）
    webcrawler_max_records_per_page: int = 10  # 每页最大生成记录数
    min_text_length: int = 500  # 最小文本长度（字符）
    min_relevance_score: float = 0.6  # 最小相关性评分


# ==================== WebCollection State ====================
@dataclass
class WebCollectionState(MainState):
    """Web数据收集任务的State，继承自MainState"""
    
    # 重写request类型为WebCollectionRequest
    request: WebCollectionRequest = field(default_factory=WebCollectionRequest)
    
    # === 任务分解相关 ===
    user_query: str = ""  # 原始用户需求语句
    normalized_query: str = ""  # 标准化后的查询语句
    normalized_reason: str = ""  # 标准化处理的理由
    intent_type: str = ""  # 用户意图分类结果
    task_list: List[Dict[str, Any]] = field(default_factory=list)  # 分解后的任务列表
    current_task_index: int = 0  # 当前执行的任务索引
    datasets_background: str = ""  # 数据集背景描述
    
    # === 搜索探索相关 ===
    research_summary: str = ""  # 调研总结
    urls_visited: List[str] = field(default_factory=list)  # 已访问过的URL列表
    subtasks: List[Dict[str, Any]] = field(default_factory=list)  # 当前任务拆分的子任务列表
    
    # === 下载相关 ===
    download_results: Dict[str, Any] = field(default_factory=dict)  # 下载的原始结果
    
    # === 后处理相关 ===
    intermediate_data_path: str = ""  # 中间数据存储路径
    postprocess_results: Dict[str, Any] = field(default_factory=dict)  # 后处理后的结果
    
    # === 数据清洗相关 ===
    cleaning_tool_plan: List[str] = field(default_factory=list)  # 数据清洗工具执行计划列表
    cleaning_results: Dict[str, Any] = field(default_factory=dict)  # 数据清洗执行结果
    
    # === 格式映射相关 ===
    confirmed_format: Optional[Dict[str, Any]] = None  # 确认的目标格式信息
    pending_format: Optional[Dict[str, Any]] = None  # 待用户确认的格式信息
    confirmation_result: str = ""  # 格式确认结果: confirmed/restart/modify
    mapping_user_intent: str = ""  # 用户在映射流程中的意图
    mapping_selected_format_id: str = ""  # 用户选择的预设格式ID
    mapping_custom_description: str = ""  # 用户自定义格式的描述
    mapping_results: Dict[str, Any] = field(default_factory=dict)  # 数据映射执行结果
    
    # === 网页收集节点参数 ===
    webpage_collect_summary: str = ""  # 网页收集阶段的总结
    webpage_collect_urls_visited: List[str] = field(default_factory=list)  # 网页收集阶段访问的URL
    webpage_collect_data_count: int = 0  # 网页收集的数据条数
    webpage_collect_jsonl_path: str = ""  # 网页收集结果JSONL路径
    
    # === 数据集生成相关 ===
    webpage_dataset_summary: str = ""  # 数据集生成阶段的总结
    webpage_dataset_count: int = 0  # 最终生成的数据集条数
    webpage_dataset_jsonl_path: str = ""  # 最终数据集JSONL路径
    
    # === WebCrawler 相关 ===
    webcrawler_crawled_pages: List[Dict[str, Any]] = field(default_factory=list)  # WebCrawler 爬取的页面列表
    webcrawler_sft_records: List[Dict[str, Any]] = field(default_factory=list)  # WebCrawler 生成的 SFT 记录
    webcrawler_pt_records: List[Dict[str, Any]] = field(default_factory=list)  # WebCrawler 生成的 PT 记录
    webcrawler_sft_jsonl_path: str = ""  # WebCrawler SFT 数据集路径
    webcrawler_pt_jsonl_path: str = ""  # WebCrawler PT 数据集路径
    webcrawler_summary: str = ""  # WebCrawler 执行摘要
    crawled_pages: List[Dict[str, Any]] = field(default_factory=list)  # 原有 websearch 爬取的页面列表（用于合并）
    
    # === 流程控制 ===
    current_node: str = ""  # 当前执行的节点名称
    is_finished: bool = False  # 流程是否完成
    exception: str = ""  # 异常信息
    
    def reset_for_new_task(self):
        """重置任务相关状态，用于开始新子任务"""
        self.research_summary = ""
        self.urls_visited = []
        self.subtasks = []
        self.download_results = {}
        self.crawled_pages = []
        # WebCrawler 相关重置
        self.webcrawler_crawled_pages = []
        self.webcrawler_sft_records = []
        self.webcrawler_pt_records = []
        self.webcrawler_sft_jsonl_path = ""
        self.webcrawler_pt_jsonl_path = ""
        self.webcrawler_summary = ""
    
    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """获取当前待执行的任务"""
        if 0 <= self.current_task_index < len(self.task_list):
            return self.task_list[self.current_task_index]
        return None
    
    def has_more_tasks(self) -> bool:
        """检查是否还有更多任务待执行"""
        return self.current_task_index < len(self.task_list)
    
    def advance_to_next_task(self):
        """前进到下一个任务"""
        self.current_task_index += 1
        self.reset_for_new_task()
    
    def get_download_tasks(self) -> List[Dict[str, Any]]:
        """获取下载类型的子任务"""
        return [task for task in self.subtasks if task.get("type") == "download"]
    
    def get_successful_downloads(self) -> List[Dict[str, Any]]:
        """获取成功完成的下载任务"""
        return [
            task for task in self.subtasks 
            if task.get("type") == "download" and task.get("status") == "completed_successfully"
        ]
