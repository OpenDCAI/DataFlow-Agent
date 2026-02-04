"""
Download Node
~~~~~~~~~~~~~

数据下载节点，执行：
1. 使用 LLM 决定下载方法优先顺序
2. 依次尝试 HuggingFace → Kaggle → Web
3. 使用 LLM 从搜索结果中选择最佳数据集
4. 执行下载并更新任务状态

从老项目 loopai/agents/Obtainer/nodes/download_node.py 对齐实现。
"""

import asyncio
import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from urllib.parse import urljoin

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dataflow_agent.states.web_collection_state import WebCollectionState
from dataflow_agent.web_collection.utils import WebTools
from dataflow_agent.promptstemplates import PromptsTemplateGenerator
from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


async def download_node(state: WebCollectionState) -> WebCollectionState:
    """
    Download node that executes download subtasks using LLM-decided method order
    
    This node implements the full logic from the old ObtainerAgent:
    1. DownloadMethodDecisionAgent - LLM 决定下载方法优先顺序
    2. 依次尝试 HuggingFace → Kaggle → Web
    3. HuggingFaceDecisionAgent/KaggleDecisionAgent - LLM 从搜索结果选择最佳数据集
    4. WebPageReader - LLM 分析网页寻找下载链接
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with download results
    """
    logger.info("=== Download Node: Starting ===")
    state.current_node = "download_node"
    
    # Get download subtasks
    download_tasks = state.get_download_tasks()
    
    if not download_tasks:
        logger.info("No download subtasks found, skipping download node")
        return state
    
    logger.info(f"Found {len(download_tasks)} download subtasks to execute")
    
    try:
        # Get configuration
        model_name = state.request.model
        base_url = state.request.chat_api_url
        api_key = state.request.api_key
        temperature = state.request.temperature or 0.7
        
        if not model_name or not base_url or not api_key:
            logger.error("Missing required configuration for download node")
            state.exception = "Missing model configuration for download node"
            return state
        
        # Initialize Prompt Generator
        prompt_generator = None
        try:
            prompt_generator = PromptsTemplateGenerator("pt_web_collection")
        except Exception as e:
            logger.warning(f"Failed to load prompt templates: {e}")
        
        # Output directory for downloads
        download_dir = state.request.download_dir
        os.makedirs(download_dir, exist_ok=True)
        
        # Run async workflow
        result = await _download_workflow(
            download_tasks=download_tasks,
            user_query=state.user_query,
            datasets_background=state.datasets_background,
            category=state.request.category,
            download_dir=download_dir,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            prompt_generator=prompt_generator,
            tavily_api_key=state.request.tavily_api_key,
            kaggle_username=state.request.kaggle_username,
            kaggle_key=state.request.kaggle_key,
            search_engine=state.request.search_engine,
            max_urls=state.request.max_urls,
        )
        
        # Update state with results
        if "exception" in result:
            state.exception = result["exception"]
        
        # Update subtasks with status
        completed_tasks = result.get("completed_tasks", [])
        failed_tasks = result.get("failed_tasks", [])
        
        # Update subtask statuses
        for task in state.subtasks:
            if task.get("type") == "download":
                task_objective = task.get("objective", "")
                
                # Check if completed
                for completed in completed_tasks:
                    if completed.get("objective") == task_objective:
                        task["status"] = "completed_successfully"
                        task["download_path"] = completed.get("download_path")
                        task["method_used"] = completed.get("method_used")
                        break
                else:
                    # Check if failed
                    for failed in failed_tasks:
                        if failed.get("objective") == task_objective:
                            task["status"] = "failed_to_download"
                            task["failure_reason"] = failed.get("failure_reason")
                            break
                    else:
                        task["status"] = "pending"
        
        state.download_results = {
            "completed": len(completed_tasks),
            "failed": len(failed_tasks),
            "total": len(download_tasks),
        }
        
        logger.info(
            f"Download node completed: {len(completed_tasks)} succeeded, "
            f"{len(failed_tasks)} failed out of {len(download_tasks)} total"
        )
        
    except Exception as e:
        logger.error(f"Download node error: {e}", exc_info=True)
        state.exception = f"Download error: {str(e)}"
    
    logger.info("=== Download Node: Completed ===")
    return state


async def _download_workflow(
    download_tasks: List[Dict[str, Any]],
    user_query: str,
    datasets_background: str,
    category: str,
    download_dir: str,
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.7,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
    tavily_api_key: Optional[str] = None,
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None,
    search_engine: str = "tavily",
    max_urls: int = 10,
) -> Dict[str, Any]:
    """
    Async workflow for downloading datasets with LLM-based decision making
    """
    completed_tasks = []
    failed_tasks = []
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )
    
    for task_idx, task in enumerate(download_tasks, 1):
        task_objective = task.get("objective", "")
        search_keywords = task.get("search_keywords", [])
        target_url = task.get("target_url", "")
        platform_hint = task.get("platform_hint", "")
        
        # Normalize search_keywords to string
        if isinstance(search_keywords, (list, tuple)):
            search_keywords_str = ", ".join(str(kw) for kw in search_keywords if kw)
        else:
            search_keywords_str = str(search_keywords) if search_keywords else ""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing download task {task_idx}/{len(download_tasks)}")
        logger.info(f"Objective: {task_objective}")
        logger.info(f"Search Keywords: {search_keywords_str}")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Decide download method order using LLM
            logger.info("Step 1: Deciding download method order using LLM...")
            
            decision = await _decide_download_method_order(
                llm=llm,
                user_original_request=user_query,
                current_task_objective=task_objective,
                search_keywords=search_keywords_str,
                prompt_generator=prompt_generator,
            )
            
            method_order = decision.get("method_order", ["huggingface", "kaggle", "web"])
            hf_keywords = decision.get("keywords_for_hf", search_keywords if isinstance(search_keywords, list) else [search_keywords_str])
            
            logger.info(f"Method order decided: {method_order}")
            logger.info(f"HF keywords: {hf_keywords}")
            
            # Step 2: Try each method in order
            download_success = False
            method_used = None
            download_path = None
            failure_reasons = []
            
            for method in method_order:
                logger.info(f"\nTrying download method: {method}")
                
                try:
                    if method == "huggingface":
                        result = await _try_huggingface_download(
                            task_objective=task_objective,
                            search_keywords=hf_keywords,
                            download_dir=download_dir,
                            llm=llm,
                            prompt_generator=prompt_generator,
                        )
                    elif method == "kaggle":
                        result = await _try_kaggle_download(
                            task_objective=task_objective,
                            search_keywords=search_keywords_str,
                            download_dir=download_dir,
                            kaggle_username=kaggle_username,
                            kaggle_key=kaggle_key,
                            llm=llm,
                            prompt_generator=prompt_generator,
                        )
                    elif method == "web":
                        result = await _try_web_download(
                            task_objective=task_objective,
                            search_keywords=search_keywords_str,
                            download_dir=download_dir,
                            search_engine=search_engine,
                            max_urls=max_urls,
                            tavily_api_key=tavily_api_key,
                            llm=llm,
                        )
                    else:
                        logger.warning(f"Unknown download method: {method}, skipping")
                        continue
                    
                    if result.get("success"):
                        download_success = True
                        method_used = method
                        download_path = result.get("download_path")
                        logger.info(f"✓ Download succeeded using {method}")
                        break
                    else:
                        reason = result.get("reason", "Unknown error")
                        failure_reasons.append(f"{method}: {reason}")
                        logger.info(f"✗ Download failed using {method}: {reason}")
                        
                except Exception as e:
                    failure_reasons.append(f"{method}: {str(e)}")
                    logger.error(f"Error trying {method}: {e}")
            
            # Step 3: Record result
            if download_success:
                completed_tasks.append({
                    "objective": task_objective,
                    "search_keywords": search_keywords_str,
                    "method_used": method_used,
                    "download_path": download_path,
                })
            else:
                failed_tasks.append({
                    "objective": task_objective,
                    "search_keywords": search_keywords_str,
                    "failure_reason": "; ".join(failure_reasons),
                })
                
        except Exception as e:
            logger.error(f"Error processing download task: {e}", exc_info=True)
            failed_tasks.append({
                "objective": task_objective,
                "search_keywords": search_keywords_str,
                "failure_reason": f"Task processing error: {str(e)}",
            })
    
    return {
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
    }


async def _decide_download_method_order(
    llm: ChatOpenAI,
    user_original_request: str,
    current_task_objective: str,
    search_keywords: str,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
) -> Dict[str, Any]:
    """Use LLM to decide download method priority order"""
    
    # Get prompts
    if prompt_generator:
        try:
            system_prompt = prompt_generator.templates.get("system_prompt_for_download_method_decision")
            task_prompt_template = prompt_generator.templates.get("task_prompt_for_download_method_decision")
            if system_prompt and task_prompt_template:
                human_prompt = task_prompt_template.format(
                    user_original_request=user_original_request,
                    current_task_objective=current_task_objective,
                    keywords=search_keywords,
                )
            else:
                raise KeyError("Template not found")
        except Exception as e:
            logger.warning(f"Failed to load prompt, using default: {e}")
            system_prompt = _get_default_method_decision_system_prompt()
            human_prompt = _get_default_method_decision_task_prompt(
                user_original_request, current_task_objective, search_keywords
            )
    else:
        system_prompt = _get_default_method_decision_system_prompt()
        human_prompt = _get_default_method_decision_task_prompt(
            user_original_request, current_task_objective, search_keywords
        )
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        clean_response = response.content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_response)
        
        # Validate result
        method_order = result.get("method_order", ["huggingface", "kaggle", "web"])
        keywords_for_hf = result.get("keywords_for_hf", [search_keywords])
        
        # Ensure method_order has valid methods
        valid_methods = ["huggingface", "kaggle", "web"]
        method_order = [m for m in method_order if m in valid_methods]
        if not method_order:
            method_order = ["huggingface", "kaggle", "web"]
        
        return {
            "method_order": method_order,
            "keywords_for_hf": keywords_for_hf,
            "reasoning": result.get("reasoning", ""),
        }
        
    except Exception as e:
        logger.warning(f"Failed to decide method order: {e}, using default")
        return {
            "method_order": ["huggingface", "kaggle", "web"],
            "keywords_for_hf": [search_keywords],
            "reasoning": "Default order due to LLM error",
        }


def _get_default_method_decision_system_prompt() -> str:
    """Get default method decision system prompt"""
    return """You are an intelligent download strategy decision maker. Your task is to decide the priority order of three download methods based on the user's requirements and task objective.

The three available methods are:
1. "huggingface" - Download datasets from HuggingFace Hub
2. "kaggle" - Download datasets from Kaggle
3. "web" - Download files directly from web pages

Return a JSON object with:
- "method_order": A list of three method names in priority order
- "keywords_for_hf": A list of keywords for HuggingFace search
- "reasoning": Brief explanation"""


def _get_default_method_decision_task_prompt(user_request: str, task_objective: str, keywords: str) -> str:
    """Get default method decision task prompt"""
    return f"""User's original request: {user_request}
Current task objective: {task_objective}
Search keywords: {keywords}

Please analyze the task and decide the priority order of the three download methods.
Return a JSON object with method_order, keywords_for_hf, and reasoning."""


async def _try_huggingface_download(
    task_objective: str,
    search_keywords: List[str],
    download_dir: str,
    llm: ChatOpenAI,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
) -> Dict[str, Any]:
    """Try downloading from HuggingFace with LLM-based dataset selection"""
    logger.info("[HuggingFace] Attempting download...")
    
    try:
        from huggingface_hub import HfApi, hf_hub_download, list_datasets
        
        api = HfApi()
        
        # Prepare keywords
        if isinstance(search_keywords, (list, tuple)):
            keywords = [kw for kw in search_keywords if isinstance(kw, str) and kw.strip()]
        else:
            keywords = [str(search_keywords)] if search_keywords else [task_objective]
        
        if not keywords:
            keywords = [task_objective]
        
        # Search datasets
        logger.info(f"[HuggingFace] Searching with keywords: {keywords}")
        
        all_results = {}
        for kw in keywords[:3]:  # Limit to 3 keywords
            try:
                datasets = list(api.list_datasets(search=kw, limit=5))
                for ds in datasets:
                    if ds.id not in all_results:
                        all_results[ds.id] = {
                            "id": ds.id,
                            "downloads": getattr(ds, "downloads", 0),
                            "likes": getattr(ds, "likes", 0),
                            "tags": getattr(ds, "tags", []),
                        }
            except Exception as e:
                logger.warning(f"[HuggingFace] Search error for '{kw}': {e}")
        
        if not all_results:
            return {
                "success": False,
                "reason": "No datasets found",
            }
        
        # Use LLM to select best dataset
        selected_id = await _select_best_huggingface_dataset(
            llm=llm,
            search_results=all_results,
            objective=task_objective,
            prompt_generator=prompt_generator,
        )
        
        if not selected_id:
            return {
                "success": False,
                "reason": "No suitable dataset selected by LLM",
            }
        
        logger.info(f"[HuggingFace] LLM selected dataset: {selected_id}")
        
        # Create download directory
        hf_save_dir = os.path.join(download_dir, "hf_datasets")
        dataset_dir = os.path.join(hf_save_dir, selected_id.replace("/", "_"))
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Try to download dataset
        try:
            from datasets import load_dataset
            ds = load_dataset(selected_id, split="train[:100]", trust_remote_code=True)
            sample_path = os.path.join(dataset_dir, "sample.jsonl")
            ds.to_json(sample_path)
            
            logger.info(f"[HuggingFace] Download successful: {dataset_dir}")
            return {
                "success": True,
                "download_path": dataset_dir,
                "dataset_id": selected_id,
            }
        except Exception as e:
            logger.warning(f"[HuggingFace] Failed to load dataset {selected_id}: {e}")
            
            # Try downloading files directly
            try:
                files = api.list_repo_files(selected_id, repo_type="dataset")
                downloaded = False
                for f in files[:5]:
                    if f.endswith(('.json', '.jsonl', '.csv', '.parquet', '.txt')):
                        hf_hub_download(
                            repo_id=selected_id,
                            filename=f,
                            repo_type="dataset",
                            local_dir=dataset_dir,
                        )
                        downloaded = True
                
                if downloaded:
                    return {
                        "success": True,
                        "download_path": dataset_dir,
                        "dataset_id": selected_id,
                    }
            except Exception as e2:
                logger.warning(f"[HuggingFace] Failed to download files: {e2}")
        
        return {
            "success": False,
            "reason": f"Failed to download dataset: {selected_id}",
        }
        
    except ImportError:
        return {
            "success": False,
            "reason": "huggingface_hub not installed",
        }
    except Exception as e:
        logger.error(f"[HuggingFace] Download error: {e}")
        return {
            "success": False,
            "reason": f"Error: {str(e)}",
        }


async def _select_best_huggingface_dataset(
    llm: ChatOpenAI,
    search_results: Dict[str, Any],
    objective: str,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
) -> Optional[str]:
    """Use LLM to select the best HuggingFace dataset"""
    
    # Get prompts
    if prompt_generator:
        try:
            system_prompt = prompt_generator.templates.get("system_prompt_for_huggingface_decision")
            task_prompt_template = prompt_generator.templates.get("task_prompt_for_huggingface_decision")
            if system_prompt and task_prompt_template:
                human_prompt = task_prompt_template.format(
                    objective=objective,
                    message="",
                    search_results=json.dumps(search_results, indent=2, ensure_ascii=False),
                )
            else:
                raise KeyError("Template not found")
        except Exception as e:
            logger.warning(f"Failed to load prompt: {e}")
            system_prompt = "You are a HuggingFace dataset expert. Select the best dataset from search results."
            human_prompt = f"Objective: {objective}\n\nSearch results:\n{json.dumps(search_results, indent=2)}\n\nReturn JSON with selected_dataset_id and reasoning."
    else:
        system_prompt = "You are a HuggingFace dataset expert. Select the best dataset from search results."
        human_prompt = f"Objective: {objective}\n\nSearch results:\n{json.dumps(search_results, indent=2)}\n\nReturn JSON with selected_dataset_id and reasoning."
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        clean_response = response.content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_response)
        
        selected_id = result.get("selected_dataset_id")
        logger.info(f"[HuggingFace] LLM reasoning: {result.get('reasoning', 'N/A')}")
        
        return selected_id
        
    except Exception as e:
        logger.warning(f"[HuggingFace] LLM selection failed: {e}")
        # Fallback: select dataset with most downloads
        if search_results:
            sorted_results = sorted(
                search_results.items(),
                key=lambda x: x[1].get("downloads", 0),
                reverse=True
            )
            return sorted_results[0][0] if sorted_results else None
        return None


async def _try_kaggle_download(
    task_objective: str,
    search_keywords: str,
    download_dir: str,
    kaggle_username: Optional[str],
    kaggle_key: Optional[str],
    llm: ChatOpenAI,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
) -> Dict[str, Any]:
    """Try downloading from Kaggle with LLM-based dataset selection"""
    logger.info("[Kaggle] Attempting download...")
    
    if not kaggle_username or not kaggle_key:
        return {
            "success": False,
            "reason": "Kaggle credentials not provided",
        }
    
    try:
        import kaggle
        
        # Set credentials
        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key
        
        # Search for datasets
        logger.info(f"[Kaggle] Searching with keywords: {search_keywords}")
        datasets = kaggle.api.dataset_list(search=search_keywords)
        
        if not datasets:
            return {
                "success": False,
                "reason": "No datasets found",
            }
        
        # Convert to dict for LLM
        search_results = {}
        for ds in datasets[:10]:
            search_results[ds.ref] = {
                "id": ds.ref,
                "title": getattr(ds, "title", ""),
                "size": getattr(ds, "totalBytes", 0),
                "downloads": getattr(ds, "downloadCount", 0),
            }
        
        # Use LLM to select best dataset
        selected_id = await _select_best_kaggle_dataset(
            llm=llm,
            search_results=search_results,
            objective=task_objective,
            prompt_generator=prompt_generator,
        )
        
        if not selected_id:
            # Fallback to first result
            selected_id = datasets[0].ref
        
        logger.info(f"[Kaggle] Selected dataset: {selected_id}")
        
        # Create download directory
        kaggle_save_dir = os.path.join(download_dir, "kaggle_datasets")
        dataset_dir = os.path.join(kaggle_save_dir, selected_id.replace("/", "_"))
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            selected_id,
            path=dataset_dir,
            unzip=True,
        )
        
        logger.info(f"[Kaggle] Download successful: {dataset_dir}")
        return {
            "success": True,
            "download_path": dataset_dir,
            "dataset_id": selected_id,
        }
        
    except ImportError:
        return {
            "success": False,
            "reason": "kaggle not installed",
        }
    except Exception as e:
        logger.error(f"[Kaggle] Download error: {e}")
        return {
            "success": False,
            "reason": f"Error: {str(e)}",
        }


async def _select_best_kaggle_dataset(
    llm: ChatOpenAI,
    search_results: Dict[str, Any],
    objective: str,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
) -> Optional[str]:
    """Use LLM to select the best Kaggle dataset"""
    
    # Get prompts
    if prompt_generator:
        try:
            system_prompt = prompt_generator.templates.get("system_prompt_for_kaggle_decision")
            task_prompt_template = prompt_generator.templates.get("task_prompt_for_kaggle_decision")
            if system_prompt and task_prompt_template:
                human_prompt = task_prompt_template.format(
                    objective=objective,
                    message="",
                    max_dataset_size="None",
                    search_results=json.dumps(search_results, indent=2, ensure_ascii=False),
                )
            else:
                raise KeyError("Template not found")
        except Exception as e:
            logger.warning(f"Failed to load prompt: {e}")
            system_prompt = "You are a Kaggle dataset expert. Select the best dataset from search results."
            human_prompt = f"Objective: {objective}\n\nSearch results:\n{json.dumps(search_results, indent=2)}\n\nReturn JSON with selected_dataset_id and reasoning."
    else:
        system_prompt = "You are a Kaggle dataset expert. Select the best dataset from search results."
        human_prompt = f"Objective: {objective}\n\nSearch results:\n{json.dumps(search_results, indent=2)}\n\nReturn JSON with selected_dataset_id and reasoning."
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        clean_response = response.content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_response)
        
        selected_id = result.get("selected_dataset_id")
        logger.info(f"[Kaggle] LLM reasoning: {result.get('reasoning', 'N/A')}")
        
        return selected_id
        
    except Exception as e:
        logger.warning(f"[Kaggle] LLM selection failed: {e}")
        return None


async def _try_web_download(
    task_objective: str,
    search_keywords: str,
    download_dir: str,
    search_engine: str = "tavily",
    max_urls: int = 10,
    tavily_api_key: Optional[str] = None,
    llm: Optional[ChatOpenAI] = None,
) -> Dict[str, Any]:
    """Try downloading from web using search and LLM-based link extraction"""
    logger.info("[Web] Attempting download...")
    
    try:
        # Step 1: Search for URLs
        logger.info(f"[Web] Searching with query: {search_keywords}")
        search_results = await WebTools.search_web(
            search_keywords,
            search_engine,
            tavily_api_key=tavily_api_key or "",
        )
        
        # Extract URLs
        urls = WebTools.extract_urls_from_search_results(search_results)
        if not urls:
            return {
                "success": False,
                "reason": "No URLs found in search results",
            }
        
        urls = urls[:max_urls]
        logger.info(f"[Web] Found {len(urls)} URLs to check")
        
        # Step 2: Visit URLs and look for download links
        web_save_dir = os.path.join(download_dir, "web_downloads")
        os.makedirs(web_save_dir, exist_ok=True)
        
        visited_urls = set()
        
        for url in urls[:5]:  # Process up to 5 URLs
            if url in visited_urls:
                continue
            visited_urls.add(url)
            
            logger.info(f"[Web] Analyzing URL: {url}")
            
            try:
                # Read page content
                page_content = await WebTools.read_with_jina_reader(url)
                text_content = page_content.get("text", "")
                discovered_urls = page_content.get("urls", [])
                
                # Use LLM to analyze page if available
                if llm and text_content:
                    action = await _analyze_webpage_for_download(
                        llm=llm,
                        url=url,
                        text_content=text_content[:8000],
                        discovered_urls=discovered_urls[:50],
                        objective=task_objective,
                    )
                    
                    if action.get("action") == "download":
                        download_urls = action.get("urls", [])
                        logger.info(f"[Web] LLM found {len(download_urls)} download links")
                        
                        # Try to download each link
                        for download_url in download_urls[:3]:
                            full_url = urljoin(url, download_url)
                            result = await _download_file_simple(full_url, web_save_dir)
                            if result:
                                return {
                                    "success": True,
                                    "download_path": result,
                                }
                
            except Exception as e:
                logger.warning(f"[Web] Error processing URL {url}: {e}")
                continue
        
        return {
            "success": False,
            "reason": "No download links found or all download attempts failed",
        }
        
    except Exception as e:
        logger.error(f"[Web] Download error: {e}")
        return {
            "success": False,
            "reason": f"Error: {str(e)}",
        }


async def _analyze_webpage_for_download(
    llm: ChatOpenAI,
    url: str,
    text_content: str,
    discovered_urls: List[str],
    objective: str,
) -> Dict[str, Any]:
    """Use LLM to analyze webpage and find download links"""
    
    urls_block = "\n".join(discovered_urls[:30])
    
    prompt = f"""Analyze this webpage to find download links for the objective: {objective}

URL: {url}

Discovered URLs on page:
{urls_block}

Page content (first 8000 chars):
{text_content}

Return a JSON object:
{{
    "action": "download" | "navigate" | "dead_end",
    "urls": ["url1", "url2"] (if action is download),
    "url": "url" (if action is navigate),
    "description": "brief explanation"
}}"""
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content="You are a web analysis expert. Find download links for datasets."),
            HumanMessage(content=prompt)
        ])
        
        clean_response = response.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean_response)
        
    except Exception as e:
        logger.warning(f"[Web] LLM analysis failed: {e}")
        return {"action": "dead_end", "description": str(e)}


async def _download_file_simple(url: str, save_dir: str) -> Optional[str]:
    """Simple file download using httpx"""
    try:
        import httpx
        from urllib.parse import urlparse
        
        # Check if URL looks like a download link
        url_lower = url.lower()
        data_extensions = ['.zip', '.tar', '.gz', '.csv', '.json', '.jsonl', '.parquet', '.txt']
        is_download_link = any(ext in url_lower for ext in data_extensions)
        
        if not is_download_link:
            # Check with HEAD request
            try:
                async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                    head_response = await client.head(url)
                    content_disposition = head_response.headers.get("Content-Disposition", "")
                    if "attachment" not in content_disposition.lower():
                        return None
            except Exception:
                return None
        
        # Download file
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path) or "download"
        filepath = os.path.join(save_dir, filename)
        
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.info(f"[Web] Downloaded: {filepath}")
                return filepath
        
        return None
        
    except Exception as e:
        logger.warning(f"[Web] Download failed for {url}: {e}")
        return None
