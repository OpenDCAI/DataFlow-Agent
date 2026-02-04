"""
WebSearch Node
~~~~~~~~~~~~~~

网页搜索节点，执行：
1. 使用 QueryGenerator 生成多样化搜索查询
2. 使用 WebTools 搜索和爬取网页
3. 多层 BFS 森林探索网页 (max_depth=4)
4. 使用 URLSelector 智能筛选候选 URL
5. 存储内容到 RAG
6. 使用 SummaryAgent 从 RAG 内容生成下载子任务

从老项目 loopai/agents/Obtainer/nodes/websearch_node.py 对齐实现。
"""

import asyncio
import os
import random
import time
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from dataflow_agent.states.web_collection_state import WebCollectionState
from dataflow_agent.web_collection.utils import (
    RAGManager,
    QueryGenerator,
    SummaryAgent,
    URLSelector,
    WebTools,
)
from dataflow_agent.promptstemplates import PromptsTemplateGenerator
from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


# 需要跳过的域名列表（这些网站可能会触发 CAPTCHA 验证或有反爬虫保护）
BLOCKED_DOMAINS = [
    "stackoverflow.com",
]


def _is_blocked_url(url: str) -> bool:
    """检查 URL 是否属于被阻止的域名"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        for blocked in BLOCKED_DOMAINS:
            if blocked in domain:
                return True
        return False
    except Exception:
        return False


async def websearch_node(state: WebCollectionState) -> WebCollectionState:
    """
    Web search node that searches web, stores content in RAG, and generates download subtasks
    
    This node implements the full logic from the old ObtainerAgent:
    1. QueryGenerator - 生成多样化搜索查询
    2. WebTools.search_web() - 搜索 (Tavily)
    3. 多层 BFS 探索网页森林 (max_depth=4, concurrent=10)
    4. URLSelector - 智能筛选每层的候选 URL
    5. WebTools.read_with_jina_reader() - 爬取内容
    6. 存储内容到 RAG
    7. SummaryAgent - 从 RAG 内容生成下载子任务
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with search results
    """
    logger.info("=== WebSearch Node: Starting ===")
    state.current_node = "websearch_node"
    
    # Get user query
    user_query = state.user_query or state.request.target
    
    if not user_query:
        logger.warning("No user query found in state")
        state.exception = "No user query provided"
        return state
    
    logger.info(f"User query: {user_query}")
    
    rag_manager = None
    try:
        # Get configuration
        model_name = state.request.model
        base_url = state.request.chat_api_url
        api_key = state.request.api_key
        temperature = state.request.temperature or 0.7
        
        if not model_name or not base_url or not api_key:
            logger.error("Missing required configuration for websearch node")
            state.exception = "Missing model configuration (model_name, base_url, api_key)"
            return state
        
        # Initialize Prompt Generator
        prompt_generator = None
        try:
            prompt_generator = PromptsTemplateGenerator("pt_web_collection")
        except Exception as e:
            logger.warning(f"Failed to load prompt templates: {e}")
        
        # Initialize RAG Manager
        rag_persist_dir = os.path.join(state.request.download_dir, "rag_db")
        rag_api_base_url = state.request.rag_api_base_url or base_url
        rag_api_key = state.request.rag_api_key or api_key
        
        try:
            rag_manager = RAGManager(
                api_base_url=rag_api_base_url,
                api_key=rag_api_key,
                embed_model=state.request.rag_embed_model or None,
                persist_directory=rag_persist_dir,
                reset=state.request.reset_rag,
                collection_name=state.request.rag_collection_name,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize RAG Manager: {e}")
            rag_manager = None
        
        # Initialize QueryGenerator
        query_generator = QueryGenerator(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            prompt_generator=prompt_generator,
        )
        
        # Initialize SummaryAgent
        summary_agent = SummaryAgent(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            prompt_generator=prompt_generator,
            max_download_subtasks=state.request.max_download_subtasks,
        )
        
        # Initialize URLSelector
        url_selector = URLSelector(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=0.3,  # Lower temperature for more consistent URL selection
            prompt_generator=prompt_generator,
        )
        
        # Get Tavily API key
        tavily_api_key = state.request.tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        
        # Run async workflow
        result = await _websearch_workflow(
            user_query=user_query,
            datasets_background=state.datasets_background,
            query_generator=query_generator,
            summary_agent=summary_agent,
            url_selector=url_selector,
            rag_manager=rag_manager,
            search_engine=state.request.search_engine,
            max_urls=state.request.max_urls,
            max_depth=state.request.max_depth,
            concurrent_limit=state.request.concurrent_limit,
            topk_urls=state.request.topk_urls,
            url_timeout=state.request.url_timeout,
            tavily_api_key=tavily_api_key if tavily_api_key else None,
        )
        
        # Update state with results
        if "exception" in result:
            state.exception = result["exception"]
        else:
            state.research_summary = result.get("research_summary", "")
            state.subtasks = result.get("subtasks", [])
            state.urls_visited = result.get("urls_visited", [])
            state.crawled_pages = result.get("crawled_pages", [])
            logger.info(f"WebSearch completed: {len(result.get('subtasks', []))} subtasks generated")
        
    except Exception as e:
        logger.error(f"WebSearch node error: {e}", exc_info=True)
        state.exception = f"WebSearch error: {str(e)}"
    finally:
        # Always close RAG manager to release database connections
        try:
            if rag_manager is not None:
                logger.info("[RAG] Closing RAG Manager...")
                rag_manager.close()
        except Exception as e:
            logger.warning(f"[RAG] Error closing RAG Manager: {e}")
    
    logger.info("=== WebSearch Node: Completed ===")
    return state


async def _websearch_workflow(
    user_query: str,
    datasets_background: str,
    query_generator: QueryGenerator,
    summary_agent: SummaryAgent,
    url_selector: URLSelector,
    rag_manager: Optional[RAGManager],
    search_engine: str = "tavily",
    max_urls: int = 10,
    max_depth: int = 2,
    concurrent_limit: int = 10,
    topk_urls: int = 5,
    url_timeout: int = 60,
    tavily_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Async workflow for web search implementing old project's full logic:
    
    1. QueryGenerator 生成 3-5 个多样化搜索查询
    2. WebTools.search_web() 搜索 (Tavily)
    3. 多层 BFS 探索网页森林 (max_depth=4, concurrent=10)
    4. URLSelector 智能筛选每层的候选 URL
    5. WebTools.read_with_jina_reader() 爬取内容
    6. 存储内容到 RAG
    7. SummaryAgent 从 RAG 内容生成下载子任务
    """
    try:
        # Ensure integer parameters
        max_urls = int(max_urls) if max_urls else 10
        max_depth = int(max_depth) if max_depth else 4
        concurrent_limit = int(concurrent_limit) if concurrent_limit else 10
        topk_urls = int(topk_urls) if topk_urls else 5
        url_timeout = int(url_timeout) if url_timeout else 60
        
        # Step 1: Generate research queries using QueryGenerator
        logger.info("Step 1: Generating research queries using QueryGenerator...")
        
        queries = await query_generator.generate_queries(
            objective=user_query,
            message=datasets_background or user_query,
        )
        
        if not queries:
            queries = [user_query]  # Fallback to original query
        
        logger.info(f"Generated {len(queries)} research queries: {queries}")
        
        # Step 2: Search for URLs using WebTools
        logger.info("Step 2: Searching for URLs using WebTools...")
        
        all_urls = []
        for query in queries:
            search_results = await WebTools.search_web(
                query, 
                search_engine, 
                tavily_api_key=tavily_api_key
            )
            urls = WebTools.extract_urls_from_search_results(search_results)
            # Filter blocked domains
            urls = [u for u in urls if not _is_blocked_url(u)]
            all_urls.extend(urls)
            logger.info(f"Query '{query}' found {len(urls)} URLs")
        
        # Remove duplicates and limit
        unique_urls = list(dict.fromkeys(all_urls))[:max_urls]
        logger.info(f"Total unique URLs to visit: {len(unique_urls)}")
        
        # Step 3: Explore web forest with depth-limited BFS (max depth, concurrent limit)
        logger.info(f"Step 3: Exploring web forest (max_depth={max_depth}, concurrent={concurrent_limit}, topk={topk_urls})...")
        
        visited_urls = []
        visited_urls_set = set()
        rag_tasks = []
        rag_write_semaphore = asyncio.Semaphore(2)
        crawled_pages = []
        
        # URL queue: each item is (url, depth)
        url_queue = [(url, 0) for url in unique_urls]
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_limit)
        queue_lock = asyncio.Lock()
        
        async def explore_url(url: str, depth: int) -> Optional[Dict[str, Any]]:
            """Explore a URL: read content, store in RAG, and return candidate URLs for next layer"""
            async with semaphore:
                try:
                    # Check if URL is blocked
                    if _is_blocked_url(url):
                        logger.info(f"[Depth {depth}/{max_depth}] Skipping blocked domain: {url}")
                        return {
                            "url": url,
                            "depth": depth,
                            "candidate_urls": [],
                            "success": False,
                            "reason": "blocked_domain"
                        }
                    
                    # Add random delay before crawling (2-4 seconds)
                    delay = random.uniform(2.0, 4.0)
                    logger.info(f"[Depth {depth}/{max_depth}] Sleeping {delay:.2f}s before crawling URL: {url}")
                    await asyncio.sleep(delay)
                    
                    logger.info(f"[Depth {depth}/{max_depth}] Exploring URL: {url}")
                    
                    # Read webpage content using WebTools.read_with_jina_reader
                    page_content = await asyncio.wait_for(
                        WebTools.read_with_jina_reader(url),
                        timeout=url_timeout
                    )
                    
                    webpage_text = page_content.get("text", "")
                    candidate_urls = page_content.get("urls", [])
                    page_title = page_content.get("title", "")
                    
                    if webpage_text and len(webpage_text.strip()) > 50:
                        # Store content in RAG
                        if rag_manager:
                            async def _store_to_rag(u: str, txt: str, d: int):
                                async with rag_write_semaphore:
                                    await rag_manager.add_webpage_content(
                                        url=u,
                                        text_content=txt,
                                        metadata={
                                            "source": "websearch",
                                            "query": user_query,
                                            "depth": d
                                        }
                                    )
                            rag_tasks.append(asyncio.create_task(_store_to_rag(url, webpage_text, depth)))
                        
                        # Track visited URL
                        async with queue_lock:
                            if url not in visited_urls_set:
                                visited_urls_set.add(url)
                                visited_urls.append(url)
                                # Store page content for return
                                crawled_pages.append({
                                    "source_url": url,
                                    "text_content": webpage_text,
                                    "extraction_method": "jina_reader",
                                    "structured_content": {
                                        "title": page_title,
                                        "url": url
                                    }
                                })
                        
                        logger.info(f"[Depth {depth}] Successfully stored content from {url} ({len(candidate_urls)} links found)")
                        
                        # If not at max depth, use LLM to select topk URLs from candidate links
                        selected_urls = []
                        if depth < max_depth - 1 and candidate_urls:
                            # Filter out already visited URLs and blocked domains
                            async with queue_lock:
                                new_candidate_urls = [
                                    u for u in candidate_urls 
                                    if u not in visited_urls_set and not _is_blocked_url(u)
                                ]
                            
                            if new_candidate_urls:
                                try:
                                    # Use URLSelector to select topk most relevant URLs
                                    selected_urls = await url_selector.select_top_urls(
                                        research_objective=user_query,
                                        url_list=new_candidate_urls,
                                        webpage_content=webpage_text[:8000],
                                        topk=topk_urls,
                                    )
                                    logger.info(f"[Depth {depth}] URLSelector selected {len(selected_urls)} URLs from {len(new_candidate_urls)} candidates")
                                except Exception as e:
                                    logger.warning(f"[Depth {depth}] URL selection failed: {e}, falling back to first {topk_urls} URLs")
                                    selected_urls = new_candidate_urls[:topk_urls]
                        
                        return {
                            "url": url,
                            "depth": depth,
                            "candidate_urls": selected_urls,
                            "success": True
                        }
                    else:
                        logger.warning(f"[Depth {depth}] URL {url} has insufficient content")
                        return {
                            "url": url,
                            "depth": depth,
                            "candidate_urls": [],
                            "success": False
                        }
                except asyncio.TimeoutError:
                    logger.warning(f"[Depth {depth}] Timeout exploring URL {url}")
                    return {
                        "url": url,
                        "depth": depth,
                        "candidate_urls": [],
                        "success": False,
                        "reason": "timeout"
                    }
                except Exception as e:
                    logger.error(f"[Depth {depth}] Error exploring URL {url}: {e}")
                    return {
                        "url": url,
                        "depth": depth,
                        "candidate_urls": [],
                        "success": False,
                        "reason": str(e)
                    }
        
        # Process URLs layer by layer
        current_depth = 0
        while url_queue and current_depth < max_depth:
            # Get all URLs at current depth
            current_layer = [(url, depth) for url, depth in url_queue if depth == current_depth]
            url_queue = [(url, depth) for url, depth in url_queue if depth != current_depth]
            
            if not current_layer:
                current_depth += 1
                continue
            
            logger.info(f"[Forest Exploration] Processing depth {current_depth}: {len(current_layer)} URLs")
            
            # Process URLs at current depth in batches
            results = []
            for i in range(0, len(current_layer), concurrent_limit):
                batch = current_layer[i:i + concurrent_limit]
                tasks = [explore_url(url, depth) for url, depth in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    url, depth = current_layer[i]
                    logger.error(f"[Depth {depth}] Exception in task for {url}: {result}")
                    processed_results.append({
                        "url": url,
                        "depth": depth,
                        "candidate_urls": [],
                        "success": False,
                        "error": str(result)
                    })
                elif result is None:
                    url, depth = current_layer[i]
                    processed_results.append({
                        "url": url,
                        "depth": depth,
                        "candidate_urls": [],
                        "success": False
                    })
                else:
                    processed_results.append(result)
            
            results = processed_results
            
            # Collect candidate URLs for next layer
            async with queue_lock:
                for result in results:
                    if result and result.get("success") and result.get("candidate_urls"):
                        next_depth = current_depth + 1
                        for candidate_url in result["candidate_urls"]:
                            if candidate_url not in visited_urls_set:
                                url_already_in_queue = any(url == candidate_url for url, _ in url_queue)
                                if not url_already_in_queue:
                                    url_queue.append((candidate_url, next_depth))
            
            current_depth += 1
        
        logger.info(f"Forest exploration completed: visited {len(visited_urls)} URLs across {current_depth} layers")
        
        # Wait for all pending RAG writes
        if rag_tasks:
            logger.info(f"[RAG] Waiting for {len(rag_tasks)} pending RAG write tasks...")
            rag_results = await asyncio.gather(*rag_tasks, return_exceptions=True)
            for res in rag_results:
                if isinstance(res, Exception):
                    logger.warning(f"[RAG] Write task error: {res}")
        
        # Force persist RAG data
        if rag_manager:
            try:
                logger.info("[RAG] Force persisting all data after exploration...")
                await rag_manager.force_persist()
                logger.info("[RAG] Force persist completed")
            except Exception as e:
                logger.warning(f"[RAG] Force persist failed: {e}")
        
        # Step 4: Generate download subtasks using SummaryAgent
        logger.info("Step 4: Generating download subtasks using SummaryAgent...")
        
        # Get context from RAG
        context = ""
        if rag_manager:
            try:
                context = await rag_manager.get_context_for_single_query(
                    query=user_query,
                    max_chars=18000
                )
            except Exception as e:
                logger.warning(f"Failed to get RAG context: {e}")
        
        if not context:
            # Fallback: use crawled pages content
            context_parts = []
            for page in crawled_pages[:10]:
                content = page.get("text_content", "")
                url = page.get("source_url", "")
                if content:
                    context_parts.append(f"[Source: {url}]\n{content[:2000]}")
            context = "\n\n".join(context_parts) if context_parts else f"Visited {len(visited_urls)} URLs related to: {user_query}"
        
        # Generate subtasks using SummaryAgent
        subtask_result = await summary_agent.generate_subtasks(
            objective=user_query,
            context=context,
            existing_subtasks=[],
            message=datasets_background or user_query,
        )
        
        research_summary = subtask_result.get("summary", "")
        new_subtasks = subtask_result.get("new_sub_tasks", [])
        
        logger.info(f"Generated {len(new_subtasks)} download subtasks")
        
        return {
            "research_summary": research_summary,
            "subtasks": new_subtasks,
            "urls_visited": visited_urls,
            "crawled_pages": crawled_pages,
        }
        
    except Exception as e:
        logger.error(f"WebSearch workflow error: {e}", exc_info=True)
        return {
            "exception": f"WebSearch workflow error: {str(e)}",
            "research_summary": "",
            "subtasks": [],
            "urls_visited": [],
            "crawled_pages": [],
        }
