"""
Download Node
~~~~~~~~~~~~~

数据下载节点，执行：
1. 默认按固定顺序依次尝试 HuggingFace → Kaggle → Web
2. 每种方法失败后自动切换到下一种方法
3. 使用 LLM 从搜索结果中选择最佳数据集
4. 执行下载并更新任务状态


"""

import asyncio
import json
import os
import re
import shutil
import tempfile
import zipfile
import tarfile
import gzip
import bz2
import lzma
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from urllib.parse import urljoin

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dataflow_agent.states.web_collection_state import WebCollectionState
from dataflow_agent.web_collection.utils import WebTools
from dataflow_agent.web_collection.downloaders.web_downloader import PlaywrightToolKit, WebAgent
from dataflow_agent.promptstemplates import PromptsTemplateGenerator
from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


async def download_node(state: WebCollectionState) -> WebCollectionState:
    """
    Download node that executes download subtasks using fixed method order
    
    默认按固定顺序尝试下载：
    1. HuggingFace - 先尝试从 HuggingFace Hub 下载
    2. Kaggle - HuggingFace 失败后尝试从 Kaggle 下载
    3. Web - 前两者都失败后使用 Playwright WebAgent 从网页下载
    
    每种方法内部使用 LLM 从搜索结果中选择最佳数据集。
    
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
    Async workflow for downloading datasets with fixed fallback order.
    
    默认按固定顺序尝试下载：HuggingFace → Kaggle → Web。
    每种方法失败后自动切换到下一种方法。
    使用 LLM 为 HuggingFace 生成优化的搜索关键词。
    """
    completed_tasks = []
    failed_tasks = []
    
    # Fixed default download method order: HuggingFace → Kaggle → Web
    default_method_order = ["huggingface", "kaggle", "web"]
    
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
            # Use fixed default order: HuggingFace → Kaggle → Web
            method_order = list(default_method_order)
            logger.info(f"Using fixed default method order: {method_order}")
            
            # Use LLM to generate optimized keywords for HuggingFace search
            logger.info("Generating optimized HuggingFace search keywords using LLM...")
            hf_keywords = await _generate_hf_keywords(
                llm=llm,
                user_original_request=user_query,
                current_task_objective=task_objective,
                search_keywords=search_keywords_str,
                prompt_generator=prompt_generator,
            )
            logger.info(f"HF keywords: {hf_keywords}")
            
            # Try each method in fixed order
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
                        logger.info(f"✗ Download failed using {method}: {reason}, trying next method...")
                        
                except Exception as e:
                    failure_reasons.append(f"{method}: {str(e)}")
                    logger.error(f"Error trying {method}: {e}, trying next method...")
            
            # Record result
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


async def _generate_hf_keywords(
    llm: ChatOpenAI,
    user_original_request: str,
    current_task_objective: str,
    search_keywords: str,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
) -> List[str]:
    """
    Use LLM to generate optimized search keywords for HuggingFace.
    
    利用 LLM 根据用户请求和任务目标，生成更精准的 HuggingFace 搜索关键词。
    
    Args:
        llm: LLM instance
        user_original_request: 用户原始请求
        current_task_objective: 当前任务目标
        search_keywords: 原始搜索关键词
        prompt_generator: 提示模板生成器
        
    Returns:
        优化后的 HuggingFace 搜索关键词列表
    """
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
            system_prompt = _get_default_hf_keywords_system_prompt()
            human_prompt = _get_default_hf_keywords_task_prompt(
                user_original_request, current_task_objective, search_keywords
            )
    else:
        system_prompt = _get_default_hf_keywords_system_prompt()
        human_prompt = _get_default_hf_keywords_task_prompt(
            user_original_request, current_task_objective, search_keywords
        )
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        clean_response = response.content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_response)
        
        keywords_for_hf = result.get("keywords_for_hf", [search_keywords])
        
        if isinstance(keywords_for_hf, list) and keywords_for_hf:
            return keywords_for_hf
        elif isinstance(keywords_for_hf, str):
            return [keywords_for_hf]
        else:
            return [search_keywords] if search_keywords else [current_task_objective]
        
    except Exception as e:
        logger.warning(f"Failed to generate HF keywords via LLM: {e}, using original keywords")
        return [search_keywords] if search_keywords else [current_task_objective]


def _get_default_hf_keywords_system_prompt() -> str:
    """Get default system prompt for HuggingFace keyword generation"""
    return """You are an intelligent download strategy decision maker. Your task is to generate optimized search keywords for HuggingFace Hub based on the user's requirements and task objective.

Analyze the user's request and generate precise keywords that would best match relevant datasets on HuggingFace.

Return a JSON object with:
- "keywords_for_hf": A list of optimized keywords for HuggingFace search
- "reasoning": Brief explanation of your keyword choices"""


def _get_default_hf_keywords_task_prompt(user_request: str, task_objective: str, keywords: str) -> str:
    """Get default task prompt for HuggingFace keyword generation"""
    return f"""User's original request: {user_request}
Current task objective: {task_objective}
Search keywords: {keywords}

Please analyze the task and generate optimized search keywords for HuggingFace Hub.
Return a JSON object with keywords_for_hf and reasoning."""


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
    """Try downloading from Kaggle with LLM-based dataset selection (supports kagglehub)"""
    logger.info("[Kaggle] Attempting download...")
    
    # Set credentials if provided (otherwise use environment variables)
    if kaggle_username:
        os.environ["KAGGLE_USERNAME"] = kaggle_username
    if kaggle_key:
        os.environ["KAGGLE_KEY"] = kaggle_key
    
    # Check if credentials are available (from params or environment)
    env_username = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if not env_username or not env_key:
        return {
            "success": False,
            "reason": "Kaggle credentials not provided (set KAGGLE_USERNAME and KAGGLE_KEY environment variables)",
        }
    
    try:
        # Try kagglehub first, fallback to kaggle
        try:
            import kagglehub
            use_kagglehub = True
            logger.info("[Kaggle] Using kagglehub library")
        except ImportError:
            import kaggle
            use_kagglehub = False
            logger.info("[Kaggle] Using kaggle library")
        
        # Search for datasets (use kaggle API for search)
        import kaggle
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
        
        # Download dataset using kagglehub or kaggle
        if use_kagglehub:
            # kagglehub downloads to its own cache, then we can copy/link
            dataset_path = kagglehub.dataset_download(selected_id)
            logger.info(f"[Kaggle] Downloaded via kagglehub to: {dataset_path}")
            
            # Copy to our download directory for consistency
            kaggle_save_dir = os.path.join(download_dir, "kaggle_datasets")
            dataset_dir = os.path.join(kaggle_save_dir, selected_id.replace("/", "_"))
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Copy files from kagglehub cache to our directory
            import shutil
            if os.path.isdir(dataset_path):
                for item in os.listdir(dataset_path):
                    src = os.path.join(dataset_path, item)
                    dst = os.path.join(dataset_dir, item)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            else:
                shutil.copy2(dataset_path, dataset_dir)
        else:
            # Use traditional kaggle API
            kaggle_save_dir = os.path.join(download_dir, "kaggle_datasets")
            dataset_dir = os.path.join(kaggle_save_dir, selected_id.replace("/", "_"))
            os.makedirs(dataset_dir, exist_ok=True)
            
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
            "reason": "kaggle/kagglehub not installed",
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


def _is_compressed_file(file_path: str) -> bool:
    """检查文件是否为压缩包格式"""
    compressed_extensions = [
        '.zip', '.tar', '.tar.gz', '.tgz',
        '.tar.bz2', '.tbz2', '.tar.xz', '.txz',
        '.gz', '.bz2', '.xz', '.7z', '.rar'
    ]
    path_lower = file_path.lower()
    return any(path_lower.endswith(ext) for ext in compressed_extensions)


def _extract_to_temp(compressed_path: str, temp_dir: str) -> Optional[str]:
    """
    将压缩文件解压到临时目录

    Args:
        compressed_path: 压缩文件路径
        temp_dir: 临时目录的父目录

    Returns:
        解压后的目录路径，失败返回 None
    """
    if not os.path.exists(compressed_path):
        logger.error(f"压缩文件不存在: {compressed_path}")
        return None

    # 在 temp_dir 下为每个压缩包创建独立子目录
    basename = os.path.splitext(os.path.basename(compressed_path))[0]
    extract_dir = os.path.join(temp_dir, basename)
    os.makedirs(extract_dir, exist_ok=True)
    logger.info(f"正在解压 {compressed_path} 到 {extract_dir}")

    try:
        path_lower = compressed_path.lower()

        if path_lower.endswith('.zip'):
            with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            return extract_dir

        elif '.tar' in path_lower or path_lower.endswith(('.tgz', '.tbz2', '.txz')):
            with tarfile.open(compressed_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
            return extract_dir

        elif path_lower.endswith('.gz') and '.tar' not in path_lower:
            output_file = os.path.join(extract_dir, Path(compressed_path).stem)
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return extract_dir

        elif path_lower.endswith('.bz2') and '.tar' not in path_lower:
            output_file = os.path.join(extract_dir, Path(compressed_path).stem)
            with bz2.open(compressed_path, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return extract_dir

        elif path_lower.endswith('.xz') and '.tar' not in path_lower:
            output_file = os.path.join(extract_dir, Path(compressed_path).stem)
            with lzma.open(compressed_path, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return extract_dir

        else:
            logger.warning(f"不支持的压缩格式: {compressed_path}")
            return None

    except Exception as e:
        logger.error(f"解压文件失败 {compressed_path}: {e}")
        return None


# 数据文件扩展名
DATA_FILE_EXTENSIONS = ('.json', '.jsonl', '.csv', '.parquet', '.arrow', '.txt')


async def _try_web_download(
    task_objective: str,
    search_keywords: str,
    download_dir: str,
    search_engine: str = "tavily",
    max_urls: int = 10,
    tavily_api_key: Optional[str] = None,
    llm: Optional[ChatOpenAI] = None,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Try downloading from web using Playwright WebAgent
    
    使用基于 Playwright + LLM 的智能网页探索代理进行下载。
    下载完成后，自动检测压缩包并解压到临时目录以读取数据文件。
    """
    logger.info("[Web] Attempting download with WebAgent...")
    
    temp_extract_dir = None  # 跟踪临时解压目录，用于清理
    
    try:
        # 准备下载目录
        web_save_dir = os.path.join(download_dir, "web_downloads")
        os.makedirs(web_save_dir, exist_ok=True)
        
        # 配置 LLM
        if llm_config is None:
            llm_config = {
                "base_url": os.getenv("DF_API_URL", os.getenv("OPENAI_API_BASE", "http://123.129.219.111:3000/v1")),
                "api_key": os.getenv("DF_API_KEY", os.getenv("OPENAI_API_KEY", "sk-xxx")),
                "model": os.getenv("THIRD_PARTY_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o")),
            }
        
        # 配置 headless 模式和代理
        headless_mode = os.getenv("HEADLESS", "true").lower() == "true"
        use_proxy_env = os.getenv("USE_PROXY", "").lower()
        use_proxy = True if use_proxy_env == "true" else (False if use_proxy_env == "false" else None)
        
        # 构建任务描述
        task_description = f"""
Search and download dataset for: {task_objective}

Search keywords: {search_keywords}

Instructions:
1. Navigate to a search engine or directly to relevant data repositories (GitHub, Kaggle, HuggingFace, official websites)
2. Find the dataset that matches the objective
3. Download the dataset files (prefer .zip, .tar.gz, .csv, .json, .parquet formats)
4. If a direct download link is found, use download_resource to save it
5. Avoid CAPTCHA-protected sites (Google Scholar, Stack Overflow)

Goal: Successfully download at least one relevant dataset file.
"""
        
        # 创建 PlaywrightToolKit
        toolkit = PlaywrightToolKit(headless=headless_mode, use_proxy=use_proxy)
        toolkit.base_download_dir = web_save_dir
        
        try:
            await toolkit.start()
            
            # 创建 WebAgent
            agent = WebAgent(
                toolkit=toolkit,
                llm_config=llm_config,
                dom_save_dir=Path(web_save_dir) if web_save_dir else None
            )
            
            # 执行任务
            logger.info(f"[Web] Starting WebAgent with task: {task_objective}")
            final_summary = await agent.run(task_description, max_steps=15)
            
            logger.info(f"[Web] WebAgent completed: {final_summary}")
            
            # 检查是否有下载的文件
            downloaded_files = []
            if os.path.exists(web_save_dir):
                for root, dirs, files in os.walk(web_save_dir):
                    # 跳过 temp_extract 目录（避免扫描到之前的解压结果）
                    dirs[:] = [d for d in dirs if d != "temp_extract"]
                    for file in files:
                        if not file.endswith('.html') and not file.endswith('.json'):
                            filepath = os.path.join(root, file)
                            downloaded_files.append(filepath)
            
            if not downloaded_files:
                # 检查 action_history 中是否有成功的下载
                for action in agent.action_history:
                    if action.get("success") and "download_resource" in action.get("action", ""):
                        result = action.get("result", "")
                        if "Success" in result and "Downloaded" in result:
                            match = re.search(r'to\s+(.+?)\s+\(Size', result)
                            if match:
                                downloaded_path = match.group(1)
                                if os.path.exists(downloaded_path):
                                    downloaded_files.append(downloaded_path)
                
                if not downloaded_files:
                    return {
                        "success": False,
                        "reason": f"WebAgent completed but no files downloaded. Summary: {final_summary}",
                    }
            
            logger.info(f"[Web] Downloaded {len(downloaded_files)} files: {downloaded_files}")
            
            # === 压缩包检测与临时解压 ===
            has_compressed = any(_is_compressed_file(f) for f in downloaded_files)
            
            if has_compressed:
                logger.info("[Web] 检测到压缩包，创建临时解压目录...")
                temp_extract_dir = os.path.join(web_save_dir, "temp_extract")
                os.makedirs(temp_extract_dir, exist_ok=True)
                
                all_result_files = []
                for filepath in downloaded_files:
                    if _is_compressed_file(filepath):
                        logger.info(f"[Web] 正在解压压缩包: {filepath}")
                        extract_dir = _extract_to_temp(filepath, temp_extract_dir)
                        if extract_dir:
                            # 递归查找解压后的数据文件
                            extracted_data_files = []
                            for root, dirs, files in os.walk(extract_dir):
                                for f in files:
                                    if f.lower().endswith(DATA_FILE_EXTENSIONS):
                                        extracted_data_files.append(os.path.join(root, f))
                            
                            if extracted_data_files:
                                logger.info(f"[Web] 从压缩包 {os.path.basename(filepath)} 中解压出 {len(extracted_data_files)} 个数据文件")
                                all_result_files.extend(extracted_data_files)
                            else:
                                # 解压成功但没找到数据文件，也保留解压目录和原始压缩包路径
                                logger.warning(f"[Web] 压缩包 {os.path.basename(filepath)} 解压后未发现数据文件，保留原始文件路径")
                                all_result_files.append(filepath)
                        else:
                            # 解压失败，保留原始压缩包路径
                            logger.warning(f"[Web] 解压失败，保留原始压缩包路径: {filepath}")
                            all_result_files.append(filepath)
                    else:
                        all_result_files.append(filepath)
                
                logger.info(f"[Web] 解压后共有 {len(all_result_files)} 个文件可用: {all_result_files}")
                
                return {
                    "success": True,
                    "download_path": web_save_dir,
                    "downloaded_files": all_result_files,
                    "summary": final_summary,
                    "temp_extract_dir": temp_extract_dir,
                }
            else:
                return {
                    "success": True,
                    "download_path": downloaded_files[0] if len(downloaded_files) == 1 else web_save_dir,
                    "downloaded_files": downloaded_files,
                    "summary": final_summary,
                }
                
        finally:
            await toolkit.close()
        
    except Exception as e:
        logger.error(f"[Web] WebAgent download error: {e}", exc_info=True)
        return {
            "success": False,
            "reason": f"Error: {str(e)}",
        }
    finally:
        # 清理临时解压目录
        if temp_extract_dir and os.path.exists(temp_extract_dir):
            try:
                shutil.rmtree(temp_extract_dir)
                logger.info(f"[Web] 已清理临时解压目录: {temp_extract_dir}")
            except Exception as e:
                logger.warning(f"[Web] 清理临时解压目录失败: {e}")