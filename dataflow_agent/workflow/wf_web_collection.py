"""
Web Collection Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Web数据收集工作流，用于：
1. 接收收集任务
2. 自主探索网页规划下载计划
3. 下载数据
4. 总结、筛选
5. mapping到alpaca格式
6. [新增] 并行执行 WebCrawler 爬取和数据集生成

使用方法:
    dfa run --wf web_collection --target "收集机器学习问答数据集"
    
    # 启用 WebCrawler 并行爬取（默认启用）
    dfa run --wf web_collection --target "收集 Python 代码示例"
    
    # 禁用 WebCrawler
    dfa run --wf web_collection --target "收集数据集" --enable_webcrawler false
"""

from __future__ import annotations

import os
import asyncio
import copy
from typing import Dict, Any, Optional, List

from dataflow_agent.states.web_collection_state import WebCollectionState, WebCollectionRequest
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.workflow.registry import register
from dataflow_agent.web_collection.nodes import (
    websearch_node,
    download_node,
    postprocess_node,
    mapping_node,
)
from dataflow_agent.web_collection.nodes.webcrawler_node import webcrawler_node
from dataflow_agent.web_collection.nodes.webcrawler_dataset_node import webcrawler_dataset_node
from dataflow_agent.web_collection.utils import (
    CategoryClassifier,
    TaskDecomposer,
)
from dataflow_agent.agentroles.data_agents.web_collection_agent import (
    create_web_collection_agent,
)
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


@register("web_collection")
def create_web_collection_graph() -> GenericGraphBuilder:
    """
    Workflow factory: dfa run --wf web_collection
    
    创建Web数据收集工作流图
    
    工作流程（并行模式）:
        start_node -> task_decomposer -> category_classifier -> parallel_collection_node
            |
            +-- [Branch 1: 原有流程] websearch -> download
            |
            +-- [Branch 2: WebCrawler] webcrawler -> webcrawler_dataset
            |
            +-> merge -> check_more_tasks -> postprocess -> mapping -> end_node
    """
    builder = GenericGraphBuilder(
        state_model=WebCollectionState,
        entry_point="start_node"
    )

    # ==============================================================
    # PRE_TOOLS (前置工具定义)
    # ==============================================================
    
    @builder.pre_tool("user_query", "start_node")
    def _get_user_query(state: WebCollectionState):
        """获取用户查询"""
        return state.request.target or state.user_query
    
    @builder.pre_tool("category", "start_node")
    def _get_category(state: WebCollectionState):
        """获取数据类别"""
        return state.request.category

    # ==============================================================
    # NODES (节点定义)
    # ==============================================================
    
    async def start_node(state: WebCollectionState) -> WebCollectionState:
        """
        开始节点: 初始化配置
        """
        log.info("=== Web Collection Workflow: Starting ===")
        state.current_node = "start_node"
        
        # 从 request.target 初始化 user_query
        if not state.user_query and state.request.target:
            state.user_query = state.request.target
        
        # 确保下载目录存在
        os.makedirs(state.request.download_dir, exist_ok=True)
        
        log.info(f"User query: {state.user_query}")
        log.info(f"Category: {state.request.category}")
        log.info(f"Download dir: {state.request.download_dir}")
        
        return state
    
    async def task_decomposer_node(state: WebCollectionState) -> WebCollectionState:
        """
        任务分解节点: 分解用户需求为子任务
        """
        log.info("=== Task Decomposer Node ===")
        state.current_node = "task_decomposer"
        
        user_query = state.user_query
        
        if not user_query:
            log.warning("No user query, skipping task decomposition")
            state.task_list = [{"task_name": "收集数据集用于大模型微调"}]
            return state
        
        # 如果任务列表已存在，跳过分解
        if state.task_list and len(state.task_list) > 0:
            log.info(f"Task list already exists with {len(state.task_list)} tasks")
            return state
        
        try:
            # 初始化任务分解器
            decomposer = TaskDecomposer(
                model_name=state.request.model,
                base_url=state.request.chat_api_url,
                api_key=state.request.api_key,
                temperature=0.3,
            )
            
            # 分解任务
            task_list = await decomposer.decompose_tasks(user_query)
            
            # 限制任务数量
            max_tasks = 5
            if len(task_list) > max_tasks:
                log.warning(f"Limiting task list from {len(task_list)} to {max_tasks}")
                task_list = task_list[:max_tasks]
            
            state.task_list = task_list
            state.current_task_index = 0
            
            log.info(f"Decomposed into {len(task_list)} tasks")
            for i, task in enumerate(task_list):
                log.info(f"  Task {i+1}: {task.get('task_name', '')}")
                
        except Exception as e:
            log.error(f"Task decomposition failed: {e}")
            state.task_list = [{"task_name": user_query}]
            state.current_task_index = 0
        
        return state
    
    async def category_classifier_node(state: WebCollectionState) -> WebCollectionState:
        """
        分类节点: 判断当前任务是SFT还是PT
        """
        log.info("=== Category Classifier Node ===")
        state.current_node = "category_classifier"
        
        # 获取当前任务
        current_task = state.get_current_task()
        if not current_task:
            log.warning("No current task to classify")
            return state
        
        task_name = current_task.get("task_name", state.user_query)
        
        try:
            # 初始化分类器
            classifier = CategoryClassifier(
                model_name=state.request.model,
                base_url=state.request.chat_api_url,
                api_key=state.request.api_key,
                temperature=0.3,
            )
            
            # 分类
            result = await classifier.classify_category(
                user_query=task_name,
                objective=state.user_query,
            )
            
            # 更新状态 (仅当未明确指定时)
            if not state.request.category or state.request.category == "PT":
                state.request.category = result.get("category", "PT")
            
            state.datasets_background = result.get("dataset_background", task_name)
            
            log.info(f"Classified as: {state.request.category}")
            log.info(f"Dataset background: {state.datasets_background[:100]}...")
            
        except Exception as e:
            log.error(f"Category classification failed: {e}")
            # 使用关键词检测作为后备
            task_lower = task_name.lower()
            sft_keywords = ["sft", "微调", "问答", "qa", "instruction", "fine-tuning"]
            if any(kw in task_lower for kw in sft_keywords):
                state.request.category = "SFT"
            else:
                state.request.category = "PT"
            state.datasets_background = task_name
        
        return state
    
    async def websearch_node_wrapper(state: WebCollectionState) -> WebCollectionState:
        """
        网页搜索节点包装器
        """
        return await websearch_node(state)
    
    async def download_node_wrapper(state: WebCollectionState) -> WebCollectionState:
        """
        下载节点包装器
        """
        return await download_node(state)
    
    async def parallel_collection_node(state: WebCollectionState) -> WebCollectionState:
        """
        并行收集节点：同时执行原有流程和 WebCrawler 流程
        
        分支1: 原有的 websearch + download（搜索和下载数据集）
        分支2: WebCrawler 爬取 + 数据集生成（从网页提取代码块生成数据集）
        """
        log.info("=== Parallel Collection Node: Starting ===")
        state.current_node = "parallel_collection"
        
        async def original_branch(branch_state: WebCollectionState) -> Dict[str, Any]:
            """原有流程分支: websearch + download"""
            try:
                log.info("[Branch 1] 执行原有 WebSearch + Download 流程...")
                
                # 执行 websearch
                result_state = await websearch_node(branch_state)
                
                # 如果有下载任务，执行下载
                download_tasks = result_state.get_download_tasks()
                if download_tasks:
                    log.info(f"[Branch 1] 发现 {len(download_tasks)} 个下载任务")
                    result_state = await download_node(result_state)
                else:
                    log.info("[Branch 1] 无下载任务")
                
                return {
                    "success": True,
                    "research_summary": result_state.research_summary,
                    "urls_visited": result_state.urls_visited,
                    "subtasks": result_state.subtasks,
                    "download_results": result_state.download_results,
                    "crawled_pages": getattr(result_state, 'crawled_pages', []),
                }
            except Exception as e:
                log.error(f"[Branch 1] 原有流程错误: {e}", exc_info=True)
                return {"success": False, "error": str(e)}
        
        async def webcrawler_branch(branch_state: WebCollectionState) -> Dict[str, Any]:
            """WebCrawler 分支: 爬取 + 数据集生成"""
            try:
                # 检查是否启用 WebCrawler
                if not branch_state.request.enable_webcrawler:
                    log.info("[Branch 2] WebCrawler 已禁用，跳过")
                    return {"success": True, "skipped": True}
                
                log.info("[Branch 2] 执行 WebCrawler 爬取 + 数据集生成流程...")
                
                # 执行 webcrawler 爬取
                result_state = await webcrawler_node(branch_state)
                
                # 执行数据集生成
                if result_state.webcrawler_crawled_pages:
                    log.info(f"[Branch 2] 爬取了 {len(result_state.webcrawler_crawled_pages)} 个页面，生成数据集...")
                    result_state = await webcrawler_dataset_node(result_state)
                else:
                    log.info("[Branch 2] 无爬取结果")
                
                return {
                    "success": True,
                    "webcrawler_crawled_pages": result_state.webcrawler_crawled_pages,
                    "webcrawler_sft_records": result_state.webcrawler_sft_records,
                    "webcrawler_pt_records": result_state.webcrawler_pt_records,
                    "webcrawler_sft_jsonl_path": result_state.webcrawler_sft_jsonl_path,
                    "webcrawler_pt_jsonl_path": result_state.webcrawler_pt_jsonl_path,
                    "webcrawler_summary": result_state.webcrawler_summary,
                }
            except Exception as e:
                log.error(f"[Branch 2] WebCrawler 流程错误: {e}", exc_info=True)
                return {"success": False, "error": str(e)}
        
        # 并行执行两个分支
        log.info("开始并行执行两个分支...")
        
        results = await asyncio.gather(
            original_branch(state),
            webcrawler_branch(state),
            return_exceptions=True
        )
        
        # 处理结果
        original_result = results[0] if not isinstance(results[0], Exception) else {"success": False, "error": str(results[0])}
        webcrawler_result = results[1] if not isinstance(results[1], Exception) else {"success": False, "error": str(results[1])}
        
        # 合并原有流程结果
        if original_result.get("success"):
            state.research_summary = original_result.get("research_summary", "")
            state.urls_visited = original_result.get("urls_visited", [])
            state.subtasks = original_result.get("subtasks", [])
            state.download_results = original_result.get("download_results", {})
            state.crawled_pages = original_result.get("crawled_pages", [])
            log.info(f"[合并] 原有流程: {len(state.subtasks)} 个子任务")
        else:
            log.warning(f"[合并] 原有流程失败: {original_result.get('error', 'Unknown error')}")
        
        # 合并 WebCrawler 结果
        if webcrawler_result.get("success") and not webcrawler_result.get("skipped"):
            state.webcrawler_crawled_pages = webcrawler_result.get("webcrawler_crawled_pages", [])
            state.webcrawler_sft_records = webcrawler_result.get("webcrawler_sft_records", [])
            state.webcrawler_pt_records = webcrawler_result.get("webcrawler_pt_records", [])
            state.webcrawler_sft_jsonl_path = webcrawler_result.get("webcrawler_sft_jsonl_path", "")
            state.webcrawler_pt_jsonl_path = webcrawler_result.get("webcrawler_pt_jsonl_path", "")
            state.webcrawler_summary = webcrawler_result.get("webcrawler_summary", "")
            log.info(f"[合并] WebCrawler: {len(state.webcrawler_crawled_pages)} 页面, "
                    f"{len(state.webcrawler_sft_records)} SFT, {len(state.webcrawler_pt_records)} PT")
        elif webcrawler_result.get("skipped"):
            log.info("[合并] WebCrawler 已跳过")
        else:
            log.warning(f"[合并] WebCrawler 失败: {webcrawler_result.get('error', 'Unknown error')}")
        
        log.info("=== Parallel Collection Node: Completed ===")
        return state
    
    async def check_more_tasks_node(state: WebCollectionState) -> WebCollectionState:
        """
        检查是否有更多任务
        """
        log.info("=== Check More Tasks Node ===")
        state.current_node = "check_more_tasks"
        
        # 前进到下一个任务
        state.advance_to_next_task()
        
        if state.has_more_tasks():
            log.info(f"More tasks remaining: {state.current_task_index + 1}/{len(state.task_list)}")
        else:
            log.info("All tasks completed")
        
        return state
    
    async def postprocess_node_wrapper(state: WebCollectionState) -> WebCollectionState:
        """
        后处理节点包装器
        """
        return await postprocess_node(state)
    
    async def mapping_node_wrapper(state: WebCollectionState) -> WebCollectionState:
        """
        映射节点包装器
        """
        return await mapping_node(state)
    
    async def end_node(state: WebCollectionState) -> WebCollectionState:
        """
        结束节点: 生成摘要
        """
        log.info("=== Web Collection Workflow: Completed ===")
        state.current_node = "end_node"
        state.is_finished = True
        
        # 生成摘要
        summary_parts = []
        
        # 任务完成情况
        if state.task_list:
            summary_parts.append(f"共执行 {len(state.task_list)} 个数据收集任务")
        
        # 下载结果
        download_results = state.download_results
        if download_results:
            completed = download_results.get("completed", 0)
            failed = download_results.get("failed", 0)
            total = download_results.get("total", 0)
            summary_parts.append(f"下载任务: {completed}/{total} 成功, {failed} 失败")
        
        # WebCrawler 结果
        if state.webcrawler_crawled_pages or state.webcrawler_sft_records or state.webcrawler_pt_records:
            webcrawler_parts = []
            if state.webcrawler_crawled_pages:
                webcrawler_parts.append(f"爬取 {len(state.webcrawler_crawled_pages)} 个页面")
            if state.webcrawler_sft_records:
                webcrawler_parts.append(f"{len(state.webcrawler_sft_records)} 条 SFT")
            if state.webcrawler_pt_records:
                webcrawler_parts.append(f"{len(state.webcrawler_pt_records)} 条 PT")
            summary_parts.append(f"WebCrawler: {', '.join(webcrawler_parts)}")
            if state.webcrawler_sft_jsonl_path:
                summary_parts.append(f"  SFT 数据集: {state.webcrawler_sft_jsonl_path}")
            if state.webcrawler_pt_jsonl_path:
                summary_parts.append(f"  PT 数据集: {state.webcrawler_pt_jsonl_path}")
        
        # 后处理结果
        postprocess_results = state.postprocess_results
        if postprocess_results:
            total_records = postprocess_results.get("total_records", 0)
            summary_parts.append(f"后处理: {total_records} 条记录")
        
        # 映射结果
        mapping_results = state.mapping_results
        if mapping_results:
            total_mapped = mapping_results.get("total_mapped", 0)
            output_path = mapping_results.get("output_path", "")
            summary_parts.append(f"最终输出: {total_mapped} 条 {state.request.output_format} 格式记录")
            if output_path:
                summary_parts.append(f"输出路径: {output_path}")
        
        summary = "\n".join(summary_parts) if summary_parts else "工作流完成，但未生成数据"
        
        log.info(f"Summary:\n{summary}")
        
        # 添加到消息历史
        from langchain_core.messages import AIMessage
        state.messages.append(AIMessage(content=f"数据收集任务完成:\n{summary}"))
        
        return state

    # ==============================================================
    # CONDITIONAL EDGES (条件边)
    # ==============================================================
    
    def should_continue_tasks(state: WebCollectionState) -> str:
        """判断是否继续执行更多任务"""
        if state.has_more_tasks():
            return "category_classifier"
        else:
            return "postprocess_node"
    
    def has_intermediate_data(state: WebCollectionState) -> str:
        """
        判断是否有中间数据需要映射
        现在也检查 WebCrawler 生成的数据集
        """
        # 检查原有的中间数据
        if state.intermediate_data_path and os.path.exists(state.intermediate_data_path):
            return "mapping_node"
        
        # 检查 WebCrawler 生成的数据集
        if state.webcrawler_sft_jsonl_path and os.path.exists(state.webcrawler_sft_jsonl_path):
            return "mapping_node"
        if state.webcrawler_pt_jsonl_path and os.path.exists(state.webcrawler_pt_jsonl_path):
            return "mapping_node"
        
        return "end_node"

    # ==============================================================
    # 注册 nodes / edges
    # ==============================================================
    
    nodes = {
        "start_node": start_node,
        "task_decomposer": task_decomposer_node,
        "category_classifier": category_classifier_node,
        "parallel_collection": parallel_collection_node,  # 并行执行节点（替代原有的 websearch + download）
        "check_more_tasks": check_more_tasks_node,
        "postprocess_node": postprocess_node_wrapper,
        "mapping_node": mapping_node_wrapper,
        "end_node": end_node,
    }

    # 定义普通边
    edges = [
        ("start_node", "task_decomposer"),
        ("task_decomposer", "category_classifier"),
        ("category_classifier", "parallel_collection"),  # 修改: 进入并行收集节点
        ("parallel_collection", "check_more_tasks"),  # 修改: 并行收集后检查任务
        ("mapping_node", "end_node"),
    ]
    
    # 条件边字典 - GenericGraphBuilder 使用 {source_node: condition_func} 格式
    # 条件函数直接返回下一个节点的名称
    conditional_edge_dict = {
        "check_more_tasks": should_continue_tasks,
        "postprocess_node": has_intermediate_data,
    }
    
    # 添加节点
    builder.add_nodes(nodes)
    
    # 添加普通边
    builder.add_edges(edges)
    
    # 添加条件边
    builder.add_conditional_edges(conditional_edge_dict)
    
    return builder


# 便捷函数：直接运行工作流
async def run_web_collection(
    target: str,
    category: str = "PT",
    output_format: str = "alpaca",
    download_dir: str = "./web_collection_output",
    model: str = "gpt-4o",
    chat_api_url: str = "",
    api_key: str = "",
    tavily_api_key: str = "",
    **kwargs
) -> WebCollectionState:
    """
    运行Web数据收集工作流
    
    Args:
        target: 收集目标描述
        category: 数据类别 (PT/SFT)
        output_format: 输出格式 (alpaca等)
        download_dir: 下载目录
        model: 使用的模型
        chat_api_url: API URL
        api_key: API Key
        tavily_api_key: Tavily API Key
        **kwargs: 其他参数
        
    Returns:
        最终状态对象
    """
    import os
    
    # 创建请求
    request = WebCollectionRequest(
        target=target,
        category=category,
        output_format=output_format,
        download_dir=download_dir,
        model=model,
        chat_api_url=chat_api_url or os.getenv("DF_API_URL", ""),
        api_key=api_key or os.getenv("DF_API_KEY", ""),
        tavily_api_key=tavily_api_key or os.getenv("TAVILY_API_KEY", ""),
    )
    
    # 创建初始状态
    state = WebCollectionState(request=request)
    
    # 创建并编译图
    builder = create_web_collection_graph()
    graph = builder.compile()
    
    # 运行图
    final_state = await graph.ainvoke(state)
    
    return final_state
