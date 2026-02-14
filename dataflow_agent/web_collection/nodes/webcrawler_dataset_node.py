"""
WebCrawler Dataset Node
~~~~~~~~~~~~~~~~~~~~~~~

WebCrawler 数据集生成节点，执行：
1. 读取 webcrawler_node 的爬取结果
2. 对有代码块的网页生成 SFT 记录
3. 对无代码块的网页生成 PT 记录
4. 生成摘要和相关性评分
5. 保存为 JSONL 格式
"""

import asyncio
import os
import json
from typing import Dict, Any, List
from datetime import datetime

from langchain_openai import ChatOpenAI

from dataflow_agent.states.web_collection_state import WebCollectionState
from dataflow_agent.web_collection.utils.webcrawler_dataset_generator import (
    generate_sft_records,
    generate_pt_records,
    generate_webpage_summary_and_relevance,
)
from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


async def webcrawler_dataset_node(state: WebCollectionState) -> WebCollectionState:
    """
    WebCrawler 数据集生成节点
    
    从 webcrawler_node 爬取的内容生成 SFT/PT 格式的数据集：
    1. 读取爬取结果
    2. 对有代码块的网页生成 SFT 记录（代码问答对）
    3. 对无代码块的网页生成 PT 记录（文本内容）
    4. 生成网页摘要和相关性评分
    5. 保存为 JSONL 格式
    
    Args:
        state: 当前工作流状态
        
    Returns:
        更新后的状态
    """
    logger.info("=== WebCrawler Dataset Node: Starting ===")
    state.current_node = "webcrawler_dataset_node"
    
    # 检查是否有爬取结果
    crawled_pages = state.webcrawler_crawled_pages
    
    if not crawled_pages:
        logger.warning("没有爬取结果，跳过数据集生成")
        return state
    
    logger.info(f"处理 {len(crawled_pages)} 个爬取页面")
    
    try:
        # 获取配置
        model_name = state.request.model
        base_url = state.request.chat_api_url
        api_key = state.request.api_key
        temperature = state.request.temperature or 0.7
        
        if not model_name or not base_url or not api_key:
            logger.error("缺少必要的配置")
            return state
        
        # 获取用户查询
        user_query = state.user_query or state.request.target
        
        # 输出目录
        output_dir = os.path.join(state.request.download_dir, "webcrawler_dataset")
        os.makedirs(output_dir, exist_ok=True)
        
        # 运行数据集生成工作流
        result = await _webcrawler_dataset_workflow(
            user_query=user_query,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            output_dir=output_dir,
            crawled_pages=crawled_pages,
            max_records_per_page=state.request.webcrawler_max_records_per_page,
            min_relevance_score=state.request.min_relevance_score if hasattr(state.request, 'min_relevance_score') else 0.6,
            concurrent_limit=state.request.concurrent_limit,
        )
        
        # 更新状态
        state.webcrawler_sft_records = result.get("sft_records", [])
        state.webcrawler_pt_records = result.get("pt_records", [])
        state.webcrawler_sft_jsonl_path = result.get("sft_jsonl_path", "")
        state.webcrawler_pt_jsonl_path = result.get("pt_jsonl_path", "")
        
        # 更新摘要
        dataset_summary = (
            f"数据集生成完成: {len(state.webcrawler_sft_records)} 条 SFT 记录, "
            f"{len(state.webcrawler_pt_records)} 条 PT 记录"
        )
        state.webcrawler_summary = f"{state.webcrawler_summary}\n{dataset_summary}"
        
        logger.info(dataset_summary)
        
    except Exception as e:
        logger.error(f"WebCrawler 数据集节点错误: {e}", exc_info=True)
    
    logger.info("=== WebCrawler Dataset Node: Completed ===")
    return state


async def _webcrawler_dataset_workflow(
    user_query: str,
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float,
    output_dir: str,
    crawled_pages: List[Dict[str, Any]],
    max_records_per_page: int = 10,
    min_relevance_score: float = 0.6,
    concurrent_limit: int = 5,
) -> Dict[str, Any]:
    """
    WebCrawler 数据集生成工作流
    
    1. 对每个网页，根据是否有代码块决定生成 SFT 或 PT
    2. 对生成了 SFT 的网页，生成摘要和相关性评分
    3. 保存为 JSONL 格式
    """
    try:
        # 初始化 LLM
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        
        logger.info(f"处理 {len(crawled_pages)} 个网页生成数据集")
        
        # 并发限制
        semaphore = asyncio.Semaphore(concurrent_limit)
        all_sft_records = []
        all_pt_records = []
        webpages_with_sft = []
        
        async def process_webpage(webpage: Dict[str, Any], index: int) -> Dict[str, Any]:
            """处理单个网页生成数据集记录"""
            async with semaphore:
                try:
                    logger.info(f"处理网页 {index}/{len(crawled_pages)}: {webpage.get('source_url', 'N/A')}")
                    
                    content = webpage.get("text_content", "")
                    title = webpage.get("title", "")
                    url = webpage.get("source_url", "")
                    code_blocks = webpage.get("code_blocks", [])
                    
                    if not content or len(content.strip()) < 100:
                        logger.warning(f"跳过内容不足的网页: {url}")
                        return {"sft_records": [], "pt_records": []}
                    
                    # 步骤 1: 如果有代码块，尝试生成 SFT
                    sft_records = []
                    if code_blocks and len(code_blocks) > 0:
                        logger.info(f"发现 {len(code_blocks)} 个代码块，尝试生成 SFT")
                        sft_result = await generate_sft_records(
                            llm=llm,
                            user_query=user_query,
                            webpage_title=title,
                            webpage_content=content[:50000],
                            webpage_url=url,
                            code_blocks=code_blocks,
                            max_records=max_records_per_page,
                            min_relevance_score=min_relevance_score,
                        )
                        sft_records = sft_result.get("records", [])
                        
                        if sft_records:
                            logger.info(f"生成 {len(sft_records)} 条 SFT 记录")
                        else:
                            logger.info(f"SFT 生成失败: {sft_result.get('reason', '')}")
                    
                    # 步骤 2: 如果没有生成 SFT，生成 PT
                    pt_records = []
                    if not sft_records:
                        logger.info(f"生成 PT 格式")
                        pt_result = await generate_pt_records(
                            llm=llm,
                            user_query=user_query,
                            webpage_title=title,
                            webpage_content=content[:50000],
                            webpage_url=url,
                            max_records=max_records_per_page,
                            min_relevance_score=min_relevance_score,
                        )
                        pt_records = pt_result.get("records", [])
                        
                        if pt_records:
                            logger.info(f"生成 {len(pt_records)} 条 PT 记录")
                        else:
                            logger.info(f"PT 生成失败: {pt_result.get('reason', '')}")
                    
                    return {
                        "sft_records": sft_records,
                        "pt_records": pt_records,
                        "webpage_info": {
                            "url": url,
                            "title": title,
                            "content": content[:3000],  # 保留部分内容用于摘要
                            "has_sft": len(sft_records) > 0
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"处理网页 {index} 时出错: {e}")
                    return {"sft_records": [], "pt_records": []}
        
        # 并发处理所有网页
        logger.info(f"使用 {concurrent_limit} 个并发工作者处理 {len(crawled_pages)} 个网页...")
        tasks = [process_webpage(webpage, i+1) for i, webpage in enumerate(crawled_pages)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集所有记录
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"数据集生成异常: {result}")
                continue
            if result:
                all_sft_records.extend(result.get("sft_records", []))
                all_pt_records.extend(result.get("pt_records", []))
                # 记录生成了 SFT 的网页
                webpage_info = result.get("webpage_info", {})
                if webpage_info.get("has_sft", False):
                    webpages_with_sft.append(webpage_info)
        
        # 步骤 3: 为生成了 SFT 的网页生成摘要（可选）
        if webpages_with_sft:
            logger.info(f"为 {len(webpages_with_sft)} 个 SFT 网页生成摘要...")
            
            async def generate_summary_for_webpage(webpage_info: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    url = webpage_info.get("url", "")
                    title = webpage_info.get("title", "")
                    content = webpage_info.get("content", "")
                    
                    summary_result = await generate_webpage_summary_and_relevance(
                        llm=llm,
                        user_query=user_query,
                        webpage_title=title,
                        webpage_content=content,
                        webpage_url=url,
                    )
                    
                    return {
                        "url": url,
                        "title": title,
                        "summary": summary_result.get("summary", ""),
                        "relevance_score": summary_result.get("relevance_score", 0),
                    }
                except Exception as e:
                    logger.error(f"生成摘要失败 {url}: {e}")
                    return {
                        "url": url,
                        "title": title,
                        "summary": "摘要生成失败",
                        "relevance_score": 0,
                    }
            
            # 并发生成摘要
            summary_tasks = [generate_summary_for_webpage(info) for info in webpages_with_sft[:10]]  # 限制数量
            webpage_summaries = await asyncio.gather(*summary_tasks, return_exceptions=True)
            
            # 保存摘要
            valid_summaries = [s for s in webpage_summaries if isinstance(s, dict)]
            if valid_summaries:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                summary_file = os.path.join(output_dir, f"webpage_summaries_{timestamp}.jsonl")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    for summary in valid_summaries:
                        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
                logger.info(f"网页摘要已保存至: {summary_file}")
        
        # 后处理 SFT 记录：移除 null 的 system 字段
        if all_sft_records:
            for record in all_sft_records:
                if "system" in record and record["system"] is None:
                    del record["system"]
        
        # 保存到 JSONL 文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        sft_jsonl_path = ""
        if all_sft_records:
            sft_jsonl_filename = f"webcrawler_dataset_sft_{timestamp}.jsonl"
            sft_jsonl_path = os.path.join(output_dir, sft_jsonl_filename)
            with open(sft_jsonl_path, 'w', encoding='utf-8') as f:
                for record in all_sft_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"保存 {len(all_sft_records)} 条 SFT 记录到 {sft_jsonl_path}")
        
        pt_jsonl_path = ""
        if all_pt_records:
            pt_jsonl_filename = f"webcrawler_dataset_pt_{timestamp}.jsonl"
            pt_jsonl_path = os.path.join(output_dir, pt_jsonl_filename)
            with open(pt_jsonl_path, 'w', encoding='utf-8') as f:
                for record in all_pt_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"保存 {len(all_pt_records)} 条 PT 记录到 {pt_jsonl_path}")
        
        return {
            "sft_records": all_sft_records,
            "pt_records": all_pt_records,
            "sft_jsonl_path": sft_jsonl_path,
            "pt_jsonl_path": pt_jsonl_path,
        }
        
    except Exception as e:
        logger.error(f"WebCrawler 数据集工作流错误: {e}", exc_info=True)
        return {
            "sft_records": [],
            "pt_records": [],
            "sft_jsonl_path": "",
            "pt_jsonl_path": "",
        }
