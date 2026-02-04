"""
Postprocess Node
~~~~~~~~~~~~~~~~

数据后处理节点，执行：
1. LLM 驱动的文件发现
2. 基于数据集背景的 LLM 文件过滤
3. LLM 驱动的字段映射
4. 数据格式转换到中间格式
5. [新增] 合并 WebCrawler 生成的数据集

从老项目 loopai/agents/Constructor/nodes/postprocess_node.py 对齐实现。
"""

import asyncio
import os
import shutil
from typing import Dict, Any, List, Optional, Tuple, Set

from dataflow_agent.states.web_collection_state import WebCollectionState
from dataflow_agent.web_collection.utils import DataConvertor
from dataflow_agent.promptstemplates import PromptsTemplateGenerator
from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


async def postprocess_node(state: WebCollectionState) -> WebCollectionState:
    """
    Post-process node that converts downloaded datasets to PT/SFT intermediate format
    
    This node implements the full logic from the old ConstructorAgent:
    1. LLM-driven file discovery - 识别下载目录中的数据文件
    2. LLM-based file filtering - 基于数据集背景过滤不相关文件
    3. LLM-driven data mapping - 使用三重验证的字段映射
    4. Data conversion - 转换到中间格式 (PT/SFT)
    5. [新增] 合并 WebCrawler 生成的数据集
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with postprocess results
    """
    logger.info("=== Postprocess Node: Starting ===")
    state.current_node = "postprocess_node"
    
    # Check for successful downloads or existing files
    successful_downloads = state.get_successful_downloads()
    download_dir = state.request.download_dir
    
    has_download_files = os.path.exists(download_dir) and any(
        os.path.isfile(os.path.join(download_dir, f)) or os.path.isdir(os.path.join(download_dir, f))
        for f in os.listdir(download_dir)
        if not f.startswith('.') and f not in ['processed_output', '.tmp', '.cache', 'rag_db', 'webcrawler_output', 'webcrawler_dataset']
    ) if os.path.exists(download_dir) else False
    
    # 检查 WebCrawler 是否有数据
    has_webcrawler_data = bool(
        state.webcrawler_sft_jsonl_path or 
        state.webcrawler_pt_jsonl_path or 
        state.webcrawler_sft_records or 
        state.webcrawler_pt_records
    )
    
    if not successful_downloads and not has_download_files and not has_webcrawler_data:
        logger.info("No successful downloads, no files in download directory, and no WebCrawler data - skipping postprocess node")
        return state
    
    # 如果只有 WebCrawler 数据，直接设置中间数据路径
    if has_webcrawler_data and not successful_downloads and not has_download_files:
        logger.info("Only WebCrawler data available, using WebCrawler output as intermediate data")
        # 优先使用 SFT，其次 PT
        if state.webcrawler_sft_jsonl_path and os.path.exists(state.webcrawler_sft_jsonl_path):
            state.intermediate_data_path = os.path.dirname(state.webcrawler_sft_jsonl_path)
        elif state.webcrawler_pt_jsonl_path and os.path.exists(state.webcrawler_pt_jsonl_path):
            state.intermediate_data_path = os.path.dirname(state.webcrawler_pt_jsonl_path)
        
        state.postprocess_results = {
            "total_records_processed": len(state.webcrawler_sft_records) + len(state.webcrawler_pt_records),
            "processed_sources_count": 1,
            "output_dir": state.intermediate_data_path,
            "webcrawler_sft_count": len(state.webcrawler_sft_records),
            "webcrawler_pt_count": len(state.webcrawler_pt_records),
        }
        logger.info(f"WebCrawler data: {len(state.webcrawler_sft_records)} SFT, {len(state.webcrawler_pt_records)} PT")
        return state
    
    if successful_downloads:
        logger.info(f"Found {len(successful_downloads)} successful downloads to post-process")
    elif has_download_files:
        logger.info("No download tasks found, but download directory exists with files. Processing files in download directory.")
    
    try:
        # Get configuration
        model_name = state.request.model
        base_url = state.request.chat_api_url
        api_key = state.request.api_key
        temperature = state.request.temperature or 0.0
        category = state.request.category.upper()
        
        if category not in ["PT", "SFT"]:
            logger.warning(f"Invalid category '{category}', defaulting to PT")
            category = "PT"
        
        if not model_name or not base_url or not api_key:
            logger.error("Missing required configuration for postprocess node")
            state.exception = "Missing model configuration for postprocess node"
            return state
        
        # Initialize Prompt Generator
        prompt_generator = None
        try:
            prompt_generator = PromptsTemplateGenerator("pt_web_collection")
        except Exception as e:
            logger.warning(f"Failed to load prompt templates: {e}")
        
        # Run async workflow
        result = await _postprocess_workflow(
            download_dir=download_dir,
            user_query=state.user_query,
            category=category,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            prompt_generator=prompt_generator,
            dataset_background=state.datasets_background,
            max_concurrent_mapping=10,
        )
        
        # Update state with results
        if "exception" in result:
            state.exception = result["exception"]
        else:
            state.postprocess_results = {
                "total_records_processed": result.get("total_records_processed", 0),
                "processed_sources_count": result.get("processed_sources_count", 0),
                "output_dir": result.get("output_dir", ""),
            }
            # Save intermediate format path for mapping node
            output_dir = result.get("output_dir", "")
            if output_dir and os.path.exists(output_dir):
                state.intermediate_data_path = output_dir
                logger.info(f"Intermediate format data saved at: {output_dir}")
            logger.info(
                f"Postprocess node completed: {result.get('total_records_processed', 0)} records processed."
            )
        
        # 合并 WebCrawler 数据（如果有）
        if has_webcrawler_data:
            logger.info("Merging WebCrawler data with processed download data...")
            webcrawler_records = _merge_webcrawler_data(state, output_dir)
            if webcrawler_records > 0:
                state.postprocess_results["webcrawler_sft_count"] = len(state.webcrawler_sft_records)
                state.postprocess_results["webcrawler_pt_count"] = len(state.webcrawler_pt_records)
                state.postprocess_results["total_records_processed"] = (
                    state.postprocess_results.get("total_records_processed", 0) + webcrawler_records
                )
                logger.info(f"Merged {webcrawler_records} WebCrawler records")
        
    except Exception as e:
        logger.error(f"Postprocess node error: {e}", exc_info=True)
        state.exception = f"Postprocess error: {str(e)}"
    
    logger.info("=== Postprocess Node: Completed ===")
    return state


def _merge_webcrawler_data(state: WebCollectionState, output_dir: str) -> int:
    """
    将 WebCrawler 生成的数据合并到处理后的输出目录
    
    Args:
        state: 工作流状态
        output_dir: 后处理输出目录
        
    Returns:
        合并的记录数
    """
    total_merged = 0
    
    if not output_dir:
        output_dir = os.path.join(state.request.download_dir, "processed_output")
        os.makedirs(output_dir, exist_ok=True)
    
    # 合并 SFT 数据
    if state.webcrawler_sft_jsonl_path and os.path.exists(state.webcrawler_sft_jsonl_path):
        try:
            # 复制 WebCrawler SFT 文件到输出目录
            dest_filename = f"webcrawler_sft_{os.path.basename(state.webcrawler_sft_jsonl_path)}"
            dest_path = os.path.join(output_dir, dest_filename)
            shutil.copy2(state.webcrawler_sft_jsonl_path, dest_path)
            total_merged += len(state.webcrawler_sft_records)
            logger.info(f"Copied WebCrawler SFT data to: {dest_path}")
        except Exception as e:
            logger.error(f"Failed to copy WebCrawler SFT data: {e}")
    
    # 合并 PT 数据
    if state.webcrawler_pt_jsonl_path and os.path.exists(state.webcrawler_pt_jsonl_path):
        try:
            # 复制 WebCrawler PT 文件到输出目录
            dest_filename = f"webcrawler_pt_{os.path.basename(state.webcrawler_pt_jsonl_path)}"
            dest_path = os.path.join(output_dir, dest_filename)
            shutil.copy2(state.webcrawler_pt_jsonl_path, dest_path)
            total_merged += len(state.webcrawler_pt_records)
            logger.info(f"Copied WebCrawler PT data to: {dest_path}")
        except Exception as e:
            logger.error(f"Failed to copy WebCrawler PT data: {e}")
    
    # 更新中间数据路径
    if total_merged > 0 and not state.intermediate_data_path:
        state.intermediate_data_path = output_dir
    
    return total_merged


async def _postprocess_workflow(
    download_dir: str,
    user_query: str,
    category: str,
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
    dataset_background: str = "",
    max_concurrent_mapping: int = 10,
) -> Dict[str, Any]:
    """
    Async workflow for post-processing downloaded datasets
    
    Implements the full Constructor logic:
    1. File discovery (LLM-driven)
    2. File filtering (based on dataset background)
    3. Data loading and LLM mapping
    4. Conversion to intermediate format
    """
    try:
        # Initialize DataConvertor
        convertor = DataConvertor(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            prompt_generator=prompt_generator,
            timeout=120.0,
            max_retries=3,
        )
        
        if not os.path.exists(download_dir):
            logger.error(f"Download directory does not exist: {download_dir}")
            return {"exception": f"Download directory does not exist: {download_dir}"}
        
        # Step 1: File discovery (LLM-driven)
        logger.info("Step 1: Scanning and discovering data files...")
        
        exclude_files = [
            'PT.jsonl', 'SFT.jsonl', 'summary.txt',
            'chroma.sqlite3', 'data_level0.bin', 'header.bin', 
            'length.bin', 'link_lists.bin', 'index_metadata.pickle'
        ]
        file_list_str = convertor._get_file_list_string(download_dir, exclude_files=exclude_files)
        
        if file_list_str == "This directory is empty.":
            logger.warning(f"Directory {download_dir} is empty, no files to process.")
            return {
                "total_records_processed": 0,
                "processed_sources_count": 0,
                "output_dir": "",
            }
        
        logger.debug(f"File list:\n{file_list_str}")
        
        # Chunk file list for LLM
        chunked_file_lists = convertor._chunk_file_list_for_llm(file_list_str)
        total_chunks = len(chunked_file_lists)
        logger.info(f"File list will be split into {total_chunks} chunks for LLM file discovery.")
        
        # Process file discovery chunks with concurrent tasks
        data_file_list: List[str] = []
        seen_paths: Set[str] = set()
        failed_chunks = 0
        
        async def process_chunk(idx: int, chunk_str: str) -> Tuple[int, List[str], Optional[Exception]]:
            """Process a single chunk"""
            try:
                logger.info(f"Processing file discovery chunk {idx}/{total_chunks}")
                chunk_result = await convertor.invoke_file_discovery(chunk_str)
                logger.info(f"Chunk {idx}/{total_chunks} returned {len(chunk_result)} candidate files.")
                return (idx, chunk_result, None)
            except Exception as e:
                logger.error(f"LLM file discovery chunk {idx}/{total_chunks} failed: {e}")
                return (idx, [], e)
        
        # Use semaphore to limit concurrent file discovery tasks
        semaphore = asyncio.Semaphore(5)
        
        async def process_chunk_with_semaphore(idx: int, chunk_str: str):
            async with semaphore:
                return await process_chunk(idx, chunk_str)
        
        # Execute all discovery tasks
        tasks = [
            process_chunk_with_semaphore(idx, chunk_str)
            for idx, chunk_str in enumerate(chunked_file_lists, start=1)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                failed_chunks += 1
                logger.error(f"File discovery chunk processing failed: {result}")
            else:
                idx, chunk_result, error = result
                if error:
                    failed_chunks += 1
                else:
                    for candidate in chunk_result:
                        if isinstance(candidate, str) and candidate not in seen_paths:
                            seen_paths.add(candidate)
                            data_file_list.append(candidate)
        
        if not data_file_list:
            if failed_chunks == total_chunks:
                logger.error("All file discovery chunks failed, cannot continue.")
            else:
                logger.warning(f"LLM did not find any data files in {download_dir}.")
            return {
                "total_records_processed": 0,
                "processed_sources_count": 0,
                "output_dir": "",
            }
        
        logger.info(f"LLM identified {len(data_file_list)} data files: {data_file_list}")
        
        # Step 2: File filtering based on dataset background
        if dataset_background:
            logger.info(f"Step 2: Filtering files based on dataset background: {dataset_background[:100]}...")
            
            filtered_file_list: List[str] = []
            failed_filter_files = 0
            
            async def filter_single_file(relative_file_path: str) -> Tuple[str, bool, Optional[Exception]]:
                """Filter a single file"""
                try:
                    absolute_file_path = os.path.join(download_dir, relative_file_path)
                    
                    if not os.path.exists(absolute_file_path):
                        logger.warning(f"File does not exist: {absolute_file_path}")
                        return (relative_file_path, False, None)
                    
                    # Try to load the file for sampling
                    builder_type = convertor._get_builder_type(absolute_file_path)
                    if not builder_type:
                        logger.warning(f"Cannot determine builder type for filtering: {absolute_file_path}, keeping file")
                        return (relative_file_path, True, None)
                    
                    # Load file
                    data = await convertor._load_with_datasets(builder_type, absolute_file_path)
                    if data is None:
                        data = await convertor._load_with_fallback(builder_type, absolute_file_path)
                    
                    if data is None:
                        logger.warning(f"Could not load file for filtering: {absolute_file_path}, keeping file")
                        return (relative_file_path, True, None)
                    
                    # Sample records
                    sampled_records = []
                    for split_name, data_content in data.items():
                        if len(data_content) > 0:
                            sampled = await convertor._sample_records(data_content, num_samples=3)
                            sampled_records = sampled
                            break
                    
                    if not sampled_records:
                        return (relative_file_path, True, None)
                    
                    # Call LLM to check if file matches background
                    is_match = await convertor.invoke_file_filter(
                        file_path=relative_file_path,
                        sampled_records=sampled_records,
                        dataset_background=dataset_background
                    )
                    
                    return (relative_file_path, is_match, None)
                    
                except Exception as e:
                    logger.error(f"Error filtering file {relative_file_path}: {e}")
                    return (relative_file_path, True, e)
            
            # Filter files concurrently
            filter_semaphore = asyncio.Semaphore(max_concurrent_mapping)
            
            async def filter_file_with_semaphore(relative_file_path: str):
                async with filter_semaphore:
                    return await filter_single_file(relative_file_path)
            
            filter_tasks = [filter_file_with_semaphore(f) for f in data_file_list]
            filter_results = await asyncio.gather(*filter_tasks, return_exceptions=True)
            
            # Process filtering results
            for result in filter_results:
                if isinstance(result, Exception):
                    failed_filter_files += 1
                    continue
                
                file_path, is_match, error = result
                if error:
                    failed_filter_files += 1
                    filtered_file_list.append(file_path)
                elif is_match:
                    filtered_file_list.append(file_path)
                    logger.info(f"✓ File matches background: {file_path}")
                else:
                    logger.info(f"✗ File does not match background: {file_path}")
            
            original_count = len(data_file_list)
            filtered_count = len(filtered_file_list)
            logger.info(f"File filtering complete: {original_count} -> {filtered_count} files")
            
            data_file_list = filtered_file_list
            
            if not data_file_list:
                logger.warning("All files were filtered out, no files to process.")
                return {
                    "total_records_processed": 0,
                    "processed_sources_count": 0,
                    "output_dir": "",
                }
        else:
            logger.info("No dataset background provided, skipping file filtering step.")
        
        # Step 3: Data conversion
        logger.info("Step 3: Starting data conversion and merging...")
        
        output_dir = os.path.join(download_dir, "processed_output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_jsonl_prefix = os.path.join(output_dir, f"{category.upper()}")
        logger.info(f"Output file prefix: {output_jsonl_prefix}_*.jsonl")
        
        processed_sources_list: List[Tuple[str, int]] = []
        total_files = len(data_file_list)
        
        for file_idx, relative_file_path in enumerate(data_file_list):
            absolute_file_path = os.path.join(download_dir, relative_file_path)
            
            if not os.path.exists(absolute_file_path):
                logger.warning(f"LLM returned non-existent file path '{relative_file_path}', skipping.")
                continue
            
            logger.info(f"--- Processing file {file_idx + 1}/{total_files}: {absolute_file_path} ---")
            
            files_to_process = []
            
            # Handle compressed files
            if convertor._is_compressed_file(absolute_file_path):
                logger.info(f"Detected compressed file: {absolute_file_path}")
                extracted_dir = convertor._extract_compressed_file(absolute_file_path)
                
                if not extracted_dir:
                    logger.error(f"Extraction failed, skipping file: {absolute_file_path}")
                    continue
                
                for root, dirs, files in os.walk(extracted_dir):
                    for f in files:
                        full_path = os.path.join(root, f)
                        if any(full_path.lower().endswith(ext) for ext in 
                               ['.json', '.jsonl', '.csv', '.parquet', '.arrow', '.txt']):
                            files_to_process.append(full_path)
                
                if not files_to_process:
                    logger.warning(f"No data files found after extraction: {absolute_file_path}")
                    continue
            else:
                files_to_process = [absolute_file_path]
            
            # Load and process files
            for file_path in files_to_process:
                logger.info(f"--- Loading data file: {file_path} ---")
                
                builder_type = convertor._get_builder_type(file_path)
                if not builder_type:
                    logger.warning(f"Cannot determine builder type, skipping file: {file_path}")
                    continue
                
                # Load data
                data = await convertor._load_with_datasets(builder_type, file_path)
                if data is None:
                    data = await convertor._load_with_fallback(builder_type, file_path)
                
                if data is None:
                    logger.error(f"Failed to load file: {file_path}")
                    continue
                
                # Process each split
                file_name = os.path.basename(file_path)
                for split_name, data_content in data.items():
                    if len(data_content) == 0:
                        continue
                    
                    column_names = data_content.column_names
                    sample_record = data_content[0]
                    
                    try:
                        # Get LLM mapping
                        annotation_result = await convertor.invoke_data_mapping(
                            column_names=column_names,
                            sample_record=sample_record,
                            dataset=data_content,
                            user_target=user_query,
                            category=category
                        )
                        logger.info(f"LLM mapping result for {file_name} ({split_name}): {annotation_result}")
                        
                        # Process dataset with mapping
                        await convertor._process_dataset_with_mapping(
                            data_content,
                            file_path,
                            file_name,
                            split_name,
                            annotation_result,
                            category,
                            output_jsonl_prefix,
                            processed_sources_list
                        )
                        
                    except Exception as e:
                        logger.error(f"LLM data mapping failed for {file_name} ({split_name}): {e}")
                        continue
        
        # Cleanup
        total_records_processed = sum(count for _, count in processed_sources_list)
        logger.info(f"Data conversion complete. Total: {total_records_processed} records from {len(processed_sources_list)} sources.")
        
        convertor._cleanup_temp_dirs()
        
        return {
            "total_records_processed": total_records_processed,
            "processed_sources_count": len(processed_sources_list),
            "output_dir": os.path.abspath(output_dir),
        }
        
    except Exception as e:
        logger.error(f"Postprocess workflow error: {e}", exc_info=True)
        return {"exception": f"Postprocess workflow error: {str(e)}"}
