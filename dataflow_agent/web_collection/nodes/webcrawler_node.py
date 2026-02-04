"""
WebCrawler Node
~~~~~~~~~~~~~~~

WebCrawler 爬取节点，执行：
1. 使用 LLM 生成针对代码/技术内容的搜索查询
2. 复用现有的 websearch 逻辑进行 BFS 深度爬取
3. 从网页内容中提取代码块
4. 将结果存储到 state 的 webcrawler 相关字段
"""

import asyncio
import os
from typing import Dict, Any, List, Optional

from dataflow_agent.states.web_collection_state import WebCollectionState
from dataflow_agent.web_collection.utils import (
    RAGManager,
    QueryGenerator,
    SummaryAgent,
    URLSelector,
    WebTools,
)
from dataflow_agent.web_collection.utils.webcrawler_orchestrator import (
    WebCrawlerOrchestrator,
    extract_code_blocks_from_markdown,
    CrawledContent,
)
from dataflow_agent.promptstemplates import PromptsTemplateGenerator
from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


async def webcrawler_node(state: WebCollectionState) -> WebCollectionState:
    """
    WebCrawler 爬取节点
    
    专门用于从网页提取代码块和技术内容的爬取流程：
    1. 使用 LLM 生成针对代码/技术内容的搜索查询
    2. 执行深度 BFS 爬取
    3. 从网页内容中提取代码块
    4. 将结果存储到 state.webcrawler_* 字段
    
    Args:
        state: 当前工作流状态
        
    Returns:
        更新后的状态
    """
    logger.info("=== WebCrawler Node: Starting ===")
    state.current_node = "webcrawler_node"
    
    # 检查是否启用 WebCrawler
    if not state.request.enable_webcrawler:
        logger.info("WebCrawler 已禁用，跳过")
        return state
    
    # 获取用户查询
    user_query = state.user_query or state.request.target
    
    if not user_query:
        logger.warning("未找到用户查询")
        return state
    
    logger.info(f"用户查询: {user_query}")
    
    try:
        # 获取配置
        model_name = state.request.model
        base_url = state.request.chat_api_url
        api_key = state.request.api_key
        temperature = state.request.temperature or 0.7
        
        if not model_name or not base_url or not api_key:
            logger.error("缺少必要的配置 (model_name, base_url, api_key)")
            return state
        
        # 获取 Tavily API key
        tavily_api_key = state.request.tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        
        # 初始化 WebCrawler 编排器
        output_dir = os.path.join(state.request.download_dir, "webcrawler_output")
        
        orchestrator = WebCrawlerOrchestrator(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            tavily_api_key=tavily_api_key,
            temperature=temperature,
            output_dir=output_dir,
            num_queries=state.request.webcrawler_num_queries,
            crawl_depth=state.request.webcrawler_crawl_depth,
            max_links_per_page=state.request.webcrawler_max_links_per_page,
            concurrent_pages=state.request.webcrawler_concurrent_pages,
            min_text_length=state.request.min_text_length if hasattr(state.request, 'min_text_length') else 500,
            min_code_length=state.request.webcrawler_min_code_length,
        )
        
        # 运行 WebCrawler 工作流
        result = await _webcrawler_workflow(
            state=state,
            orchestrator=orchestrator,
            user_query=user_query,
            tavily_api_key=tavily_api_key,
        )
        
        # 更新状态
        state.webcrawler_crawled_pages = result.get("crawled_pages", [])
        state.webcrawler_summary = result.get("summary", "")
        
        logger.info(f"WebCrawler 完成: 爬取了 {len(state.webcrawler_crawled_pages)} 个页面")
        
    except Exception as e:
        logger.error(f"WebCrawler 节点错误: {e}", exc_info=True)
        # 不设置异常，让工作流继续
    
    logger.info("=== WebCrawler Node: Completed ===")
    return state


async def _webcrawler_workflow(
    state: WebCollectionState,
    orchestrator: WebCrawlerOrchestrator,
    user_query: str,
    tavily_api_key: str = "",
) -> Dict[str, Any]:
    """
    WebCrawler 工作流
    
    1. 生成针对代码/技术内容的搜索查询
    2. 复用现有的网页搜索逻辑
    3. 从爬取的内容中提取代码块
    """
    try:
        # 步骤 1: 生成搜索查询
        logger.info("步骤 1: 生成搜索查询...")
        
        search_queries = await orchestrator.generate_search_queries(user_query)
        
        if not search_queries:
            search_queries = [user_query]
        
        logger.info(f"生成了 {len(search_queries)} 个搜索查询: {search_queries}")
        
        # 步骤 2: 执行网页搜索（复用 WebTools）
        logger.info("步骤 2: 搜索网页...")
        
        all_urls = []
        for query in search_queries:
            search_results = await WebTools.search_web(
                query,
                state.request.search_engine,
                tavily_api_key=tavily_api_key
            )
            urls = WebTools.extract_urls_from_search_results(search_results)
            all_urls.extend(urls)
            logger.info(f"查询 '{query}' 找到 {len(urls)} 个 URL")
        
        # 去重并限制数量
        unique_urls = list(dict.fromkeys(all_urls))[:state.request.max_urls * 2]  # 给 WebCrawler 更多 URL
        logger.info(f"总共 {len(unique_urls)} 个唯一 URL")
        
        # 步骤 3: 爬取网页内容
        logger.info("步骤 3: 爬取网页内容...")
        
        crawled_pages = []
        semaphore = asyncio.Semaphore(orchestrator.concurrent_pages)
        
        async def crawl_url(url: str) -> Optional[Dict[str, Any]]:
            """爬取单个 URL"""
            async with semaphore:
                try:
                    page_content = await asyncio.wait_for(
                        WebTools.read_with_jina_reader(url),
                        timeout=state.request.url_timeout
                    )
                    
                    text_content = page_content.get("text", "")
                    title = page_content.get("title", "")
                    
                    if text_content and len(text_content.strip()) >= orchestrator.min_text_length:
                        # 提取代码块
                        code_blocks = extract_code_blocks_from_markdown(text_content)
                        
                        # 过滤过短的代码块
                        if code_blocks:
                            code_blocks = [
                                cb for cb in code_blocks 
                                if cb.get('length', 0) >= orchestrator.min_code_length
                            ]
                        
                        logger.info(f"爬取成功: {url} ({len(code_blocks)} 个代码块)")
                        
                        return {
                            "source_url": url,
                            "text_content": text_content,
                            "title": title,
                            "code_blocks": code_blocks if code_blocks else [],
                            "extraction_method": "jina_reader",
                        }
                    else:
                        logger.info(f"内容不足: {url}")
                        return None
                        
                except asyncio.TimeoutError:
                    logger.warning(f"超时: {url}")
                    return None
                except Exception as e:
                    logger.error(f"爬取失败 {url}: {e}")
                    return None
        
        # 并发爬取
        tasks = [crawl_url(url) for url in unique_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and result is not None:
                crawled_pages.append(result)
        
        logger.info(f"成功爬取 {len(crawled_pages)} 个页面")
        
        # 保存爬取结果
        if crawled_pages:
            orchestrator.save_results(
                [CrawledContent(
                    url=p["source_url"],
                    title=p.get("title", ""),
                    content=p["text_content"],
                    code_blocks=p.get("code_blocks"),
                ) for p in crawled_pages],
                "webcrawler_crawled.jsonl"
            )
        
        # 生成摘要
        summary = f"WebCrawler 爬取完成: 共 {len(crawled_pages)} 个页面, " \
                  f"{sum(len(p.get('code_blocks', [])) for p in crawled_pages)} 个代码块"
        
        return {
            "crawled_pages": crawled_pages,
            "summary": summary,
            "search_queries": search_queries,
        }
        
    except Exception as e:
        logger.error(f"WebCrawler 工作流错误: {e}", exc_info=True)
        return {
            "crawled_pages": [],
            "summary": f"WebCrawler 工作流错误: {str(e)}",
            "search_queries": [],
        }
