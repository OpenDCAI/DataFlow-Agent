"""
Data Convertor
~~~~~~~~~~~~~~

数据格式转换器，用于将下载的数据集转换为中间格式（PT/SFT）。


主要功能：
1. LLM 驱动的文件发现
2. 基于数据集背景的文件过滤
3. PT/SFT 格式的字段映射
4. 数据格式转换
"""

import asyncio
import os
import json
import re
import zipfile
import tarfile
import gzip
import bz2
import lzma
import shutil
import tempfile
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

try:
    from datasets import load_dataset, DownloadConfig
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dataflow_agent.logger import get_logger
from dataflow_agent.promptstemplates import PromptsTemplateGenerator

logger = get_logger(__name__)


class SimpleDataset:
    """A lightweight dataset container that mimics the interface from HuggingFace datasets."""

    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records
        self._column_names = list(records[0].keys()) if records else []

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._records[index]

    def __iter__(self):
        return iter(self._records)

    @property
    def column_names(self) -> List[str]:
        return self._column_names


def _build_simple_dataset(records: List[Dict[str, Any]]) -> Optional[Dict[str, "SimpleDataset"]]:
    if not records:
        return None
    return {"train": SimpleDataset(records)}


def _ensure_hf_cache_env(download_dir: Optional[str]) -> None:
    """Ensure HuggingFace related environment variables point to download directory."""
    if not download_dir:
        return

    base_dir = os.path.abspath(download_dir)
    hf_cache_root = os.path.join(base_dir, ".cache", "hf")
    hub_dir = os.path.join(hf_cache_root, "hub")
    datasets_dir = os.path.join(hf_cache_root, "datasets")

    for path in (hf_cache_root, hub_dir, datasets_dir):
        os.makedirs(path, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_cache_root)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", datasets_dir)


class DataConvertor:
    """Data converter for mapping and extracting data from downloaded datasets"""

    FIELD_TOKEN_PATTERN = re.compile(r"([^\[\]]+)(?:\[(.*?)\])?")

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_sample_length: int = 200,
        num_sample_records: int = 3,
        prompt_generator: Optional[PromptsTemplateGenerator] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize Data Convertor
        
        Args:
            model_name: LLM model name
            base_url: API base URL
            api_key: API key
            temperature: LLM temperature
            max_tokens: Max tokens for LLM response
            max_sample_length: Max length for sample values
            num_sample_records: Number of sample records for LLM
            prompt_generator: Optional PromptsTemplateGenerator
            timeout: Timeout for LLM calls
            max_retries: Max retries for LLM calls
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_sample_length = max_sample_length
        self.num_sample_records = num_sample_records
        self.prompt_generator = prompt_generator
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self._temp_dirs = []

    def _truncate_value(self, value: Any, max_length: int = None) -> Any:
        """Truncate a single value to prevent excessive length."""
        if max_length is None:
            max_length = self.max_sample_length
            
        if isinstance(value, str):
            if len(value) > max_length:
                return value[:max_length] + "..."
            return value
        elif isinstance(value, (list, tuple)):
            if len(value) > 3:
                return [self._truncate_value(v, max_length) for v in value[:3]] + ["..."]
            return [self._truncate_value(v, max_length) for v in value]
        elif isinstance(value, dict):
            if len(value) > 3:
                truncated = {k: self._truncate_value(v, max_length) for k, v in list(value.items())[:3]}
                truncated["..."] = "..."
                return truncated
            return {k: self._truncate_value(v, max_length) for k, v in value.items()}
        else:
            return value

    async def _sample_records(self, dataset: Any, num_samples: int = None) -> List[Dict[str, Any]]:
        """Sample records from dataset with truncation."""
        if num_samples is None:
            num_samples = self.num_sample_records
            
        def _get_dataset_size():
            return len(dataset)
        
        def _get_record(idx):
            return dataset[idx]
        
        dataset_size = await asyncio.to_thread(_get_dataset_size)
        if dataset_size == 0:
            return []
        
        actual_samples = min(num_samples, dataset_size)
        
        if dataset_size <= actual_samples:
            sample_indices = list(range(dataset_size))
        else:
            sample_indices = random.sample(range(dataset_size), actual_samples)
        
        async def _sample_single_record(idx: int):
            record = await asyncio.to_thread(_get_record, idx)
            truncated_record = {k: self._truncate_value(v) for k, v in record.items()}
            return truncated_record
        
        sampled_records = await asyncio.gather(*[_sample_single_record(idx) for idx in sample_indices])
        
        logger.info(f"Sampled {len(sampled_records)} records from dataset (total: {dataset_size})")
        return list(sampled_records)

    def _normalize_field_value(self, value: Any) -> Optional[str]:
        """Normalize field value to string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    def _extract_field_values(self, row: Dict[str, Any], field_spec: Optional[str]) -> List[str]:
        """Extract field values from row based on field specification."""
        if not field_spec:
            return []
        if not isinstance(field_spec, str):
            if isinstance(field_spec, dict):
                normalized = []
                for key, value in field_spec.items():
                    entry = f"{key}: {self._normalize_field_value(value)}"
                    if entry:
                        normalized.append(entry)
                return normalized
            field_spec = str(field_spec)
        
        field_spec = field_spec.strip()
        if not field_spec:
            return []
        
        # Handle simple field access
        if '.' not in field_spec and '[' not in field_spec:
            value = row.get(field_spec)
            if value is not None:
                normalized = self._normalize_field_value(value)
                return [normalized] if normalized else []
            return []
        
        # Handle nested field access with dot notation
        tokens = field_spec.split(".")
        current = row
        for token in tokens:
            if current is None:
                return []
            
            # Handle array index notation
            match = self.FIELD_TOKEN_PATTERN.match(token)
            if not match:
                return []
            name, index = match.group(1), match.group(2)
            
            if isinstance(current, dict):
                current = current.get(name)
            else:
                return []
            
            if index is not None and current is not None:
                if isinstance(current, list):
                    if index == "*" or index.lower() == "all":
                        # Return all items
                        results = []
                        for item in current:
                            normalized = self._normalize_field_value(item)
                            if normalized:
                                results.append(normalized)
                        return results
                    else:
                        try:
                            idx = int(index)
                            if 0 <= idx < len(current):
                                current = current[idx]
                            else:
                                return []
                        except ValueError:
                            return []
        
        if current is not None:
            normalized = self._normalize_field_value(current)
            return [normalized] if normalized else []
        return []

    def _field_exists_in_columns(self, field_spec: Optional[Any], column_names: List[str]) -> bool:
        """Check if field specification exists in column names."""
        if field_spec is None:
            return False
        if isinstance(field_spec, list):
            if not field_spec:
                return False
            return all(self._field_exists_in_columns(spec, column_names) for spec in field_spec)
        if not isinstance(field_spec, str):
            return False
        token = field_spec.split(".")[0]
        token = token.split("[")[0]
        return token in column_names

    def _sanitize_field_spec(self, field_spec: Optional[Any], column_names: List[str]) -> Optional[Any]:
        """Sanitize field specification."""
        if field_spec is None:
            return None
        if isinstance(field_spec, list):
            sanitized = [
                spec for spec in field_spec if self._field_exists_in_columns(spec, column_names)
            ]
            return sanitized if sanitized else None
        return field_spec if self._field_exists_in_columns(field_spec, column_names) else None

    def _extract_text_values(self, row: Dict[str, Any], field_spec: Optional[Any]) -> List[str]:
        """Extract text values from row."""
        if field_spec is None:
            return []
        if isinstance(field_spec, list):
            pieces: List[str] = []
            for spec in field_spec:
                values = self._extract_field_values(row, spec)
                if values:
                    pieces.extend(values)
            combined = "\n".join(v for v in pieces if v)
            return [combined] if combined else []
        return self._extract_field_values(row, field_spec)

    def _extract_meta_field(self, row: Dict[str, Any], field_spec: Optional[Any], column_names: List[str]) -> Optional[Any]:
        """Extract a single metadata field value from a row."""
        if field_spec is None:
            return None
        
        if isinstance(field_spec, str):
            if '.' in field_spec or '[' in field_spec:
                values = self._extract_field_values(row, field_spec)
                if values:
                    return values[0] if len(values) == 1 else " ".join(str(v) for v in values if v)
                return None
            else:
                if field_spec in column_names:
                    values = self._extract_field_values(row, field_spec)
                    if values:
                        return values[0] if len(values) == 1 else " ".join(str(v) for v in values if v)
                    return None
                else:
                    return field_spec
        
        if isinstance(field_spec, (int, float, bool)):
            return field_spec
        
        return None

    def _build_meta_dict(
        self, 
        row: Dict[str, Any], 
        annotation_result: Dict[str, Any], 
        file_path: str,
        column_names: List[str]
    ) -> Dict[str, Any]:
        """Build metadata dictionary from annotation result and row data."""
        meta_spec = annotation_result.get('meta', {}) if annotation_result else {}
        
        source = self._extract_meta_field(row, meta_spec.get('source'), column_names)
        if source is None:
            source = os.path.basename(file_path) if file_path else None
        
        meta = {
            "source": source,
            "language": self._extract_meta_field(row, meta_spec.get('language'), column_names),
            "timestamp": self._extract_meta_field(row, meta_spec.get('timestamp'), column_names),
            "token_count": self._extract_meta_field(row, meta_spec.get('token_count'), column_names),
            "quality_score": self._extract_meta_field(row, meta_spec.get('quality_score'), column_names),
            "original_id": self._extract_meta_field(row, meta_spec.get('original_id'), column_names),
        }
        
        return {k: v for k, v in meta.items() if v is not None}

    def _generate_record_id(self, file_path: str, record_index: int) -> str:
        """Generate a unique record ID."""
        import hashlib
        import time
        
        base_str = f"{file_path}:{record_index}:{time.time()}"
        hash_obj = hashlib.md5(base_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]

    def _build_intermediate_format_pt(
        self,
        row: Dict[str, Any],
        annotation_result: Dict[str, Any],
        file_path: str,
        record_index: int,
        column_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Build intermediate format record for PT (Pre-training) mode."""
        if not annotation_result or annotation_result.get('text') is None:
            return None
        
        text_field = annotation_result.get('text')
        
        if isinstance(text_field, list):
            text_fields = [self._sanitize_field_spec(field, column_names) for field in text_field]
            text_fields = [f for f in text_fields if f is not None]
        else:
            text_field = self._sanitize_field_spec(text_field, column_names)
            text_fields = [text_field] if text_field else []
        
        if not text_fields:
            return None
        
        all_text_parts = []
        for field in text_fields:
            values = self._extract_text_values(row, field)
            all_text_parts.extend(values)
        
        if not all_text_parts:
            return None
        
        merged_text = " ".join(str(part).strip() for part in all_text_parts if str(part).strip())
        
        if not merged_text:
            return None
        
        meta = self._build_meta_dict(row, annotation_result, file_path, column_names)
        record_id = self._generate_record_id(file_path, record_index)
        
        return {
            "id": record_id,
            "dataset_type": "pretrain",
            "text": merged_text,
            "meta": meta
        }

    def _build_intermediate_format_sft(
        self,
        row: Dict[str, Any],
        annotation_result: Dict[str, Any],
        file_path: str,
        record_index: int,
        column_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Build intermediate format record for SFT (Supervised Fine-Tuning) mode."""
        if not annotation_result or not annotation_result.get('messages'):
            return None
        
        messages_spec = annotation_result.get('messages', [])
        if not isinstance(messages_spec, list) or len(messages_spec) == 0:
            return None
        
        messages = []
        for msg_spec in messages_spec:
            if not isinstance(msg_spec, dict):
                continue
            
            role = msg_spec.get('role')
            content_spec = msg_spec.get('content')
            loss_mask = msg_spec.get('loss_mask')
            
            if not role or not content_spec:
                continue
            
            if isinstance(content_spec, list):
                content_parts = []
                for field in content_spec:
                    field_sanitized = self._sanitize_field_spec(field, column_names)
                    if field_sanitized:
                        values = self._extract_text_values(row, field_sanitized)
                        content_parts.extend(values)
                content = " ".join(str(part).strip() for part in content_parts if str(part).strip())
            else:
                field_sanitized = self._sanitize_field_spec(content_spec, column_names)
                if field_sanitized:
                    values = self._extract_text_values(row, field_sanitized)
                    content = " ".join(str(v).strip() for v in values if str(v).strip()) if values else None
                else:
                    content = None
            
            if not content:
                continue
            
            if loss_mask is None:
                loss_mask = (role == "assistant")
            
            messages.append({
                "role": role,
                "content": content,
                "loss_mask": loss_mask
            })
        
        if not messages:
            return None
        
        system_spec = annotation_result.get('system')
        system = None
        if system_spec:
            system = self._extract_meta_field(row, system_spec, column_names)
        
        meta = self._build_meta_dict(row, annotation_result, file_path, column_names)
        record_id = self._generate_record_id(file_path, record_index)
        
        result = {
            "id": record_id,
            "dataset_type": "sft",
            "messages": messages,
            "meta": meta
        }
        
        if system:
            result["system"] = system
        
        return result

    async def invoke_data_mapping(
        self, 
        column_names: List[str], 
        sample_record: Dict[str, Any], 
        dataset: Any = None,
        user_target: str = "",
        category: str = "PT"
    ) -> Dict[str, Any]:
        """Invoke LLM for data mapping."""
        logger.info("Starting data mapping...")
        
        # Get prompts
        if self.prompt_generator:
            try:
                system_prompt = self.prompt_generator.templates.get(f"system_prompt_for_data_conversion_{category.lower()}")
                task_prompt_template = self.prompt_generator.templates.get(f"task_prompt_for_data_conversion_{category.lower()}")
                if not system_prompt:
                    raise KeyError("System prompt not found")
            except Exception as e:
                logger.warning(f"Failed to load prompt, using default: {e}")
                system_prompt = self._get_default_system_prompt(category)
                task_prompt_template = None
        else:
            system_prompt = self._get_default_system_prompt(category)
            task_prompt_template = None
        
        # Sample records
        if dataset is not None:
            sampled_records = await self._sample_records(dataset, num_samples=3)
            sample_rows_str = json.dumps(sampled_records, indent=2, ensure_ascii=False)
        else:
            truncated_record = {k: self._truncate_value(v) for k, v in sample_record.items()}
            sample_rows_str = json.dumps([truncated_record], indent=2, ensure_ascii=False)
        
        # Build task prompt
        if task_prompt_template:
            try:
                human_prompt = task_prompt_template.format(
                    column_names=str(column_names),
                    sample_rows=sample_rows_str,
                    user_target=user_target
                )
            except Exception:
                human_prompt = self._get_default_task_prompt(column_names, sample_rows_str, user_target, category)
        else:
            human_prompt = self._get_default_task_prompt(column_names, sample_rows_str, user_target, category)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        # Call LLM with retry
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Calling LLM for data mapping (attempt {attempt}/{self.max_retries})...")
                
                answer_msg = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=self.timeout
                )
                answer_text = answer_msg.content.strip()
                logger.debug(f'LLM data mapping response: {answer_text[:200]}...')

                pattern = r'```json([\s\S]*?)```'
                match = re.search(pattern, answer_text)
                if match:
                    match_text = match.group(1).strip()
                else:
                    match_text = answer_text

                annotation_result = json.loads(match_text)
                
                # Handle case where LLM returns a list instead of dict
                if isinstance(annotation_result, list):
                    if len(annotation_result) == 1 and isinstance(annotation_result[0], dict):
                        annotation_result = annotation_result[0]
                        logger.debug("Unwrapped single-element list to dict")
                    else:
                        raise ValueError(f"Expected dict but got list with {len(annotation_result)} elements")
                
                if not isinstance(annotation_result, dict):
                    raise ValueError(f"Expected dict but got {type(annotation_result).__name__}")
                
                logger.debug(f"Data mapping result: {annotation_result}")
                return annotation_result
                
            except asyncio.TimeoutError:
                logger.warning(f"LLM call timeout on attempt {attempt}/{self.max_retries}")
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 10))
                else:
                    raise ValueError(f"LLM call timed out after {self.max_retries} attempts")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed on attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 10))
                else:
                    raise ValueError(f"Failed to parse LLM response as JSON: {e}")
                    
            except Exception as e:
                logger.error(f"LLM call failed on attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 10))
                else:
                    raise ValueError(f"Failed to get LLM response: {e}")
        
        raise ValueError("Failed to get valid LLM response")

    def _get_default_system_prompt(self, category: str) -> str:
        """Get default system prompt."""
        if category.upper() == "PT":
            return """You are an expert in dataset classification and analysis.

Your task is to identify field mappings for language model pretraining, including the main text content and metadata fields.

IMPORTANT: You MUST return a single JSON object (NOT an array/list). The response format must be:

```json
{
  "text": "<field_name>",
  "meta": {
    "source": "<field_name_or_null>",
    "language": "<field_name_or_null>"
  }
}
```

Field descriptions:
- "text" (required): Field name or array of field names containing the main text content
- "meta" (optional): Object with metadata field mappings (source, language, timestamp, etc.)"""
        else:  # SFT
            return """You are an expert in dataset classification and analysis.

Your task is to identify field mappings for supervised fine-tuning, including conversation messages, system prompts, and metadata fields.

IMPORTANT: You MUST return a single JSON object (NOT an array/list). The response format must be:

```json
{
  "messages": [
    {"role": "user", "content": "<field_name>", "loss_mask": false},
    {"role": "assistant", "content": "<field_name>", "loss_mask": true}
  ],
  "system": "<field_name_or_null>",
  "meta": {
    "source": "<field_name_or_null>"
  }
}
```

Field descriptions:
- "messages" (required): Array of message specifications, each with role, content (field path), and loss_mask
- "system" (optional): Field name for system prompt, or null if not present
- "meta" (optional): Object with metadata field mappings"""

    def _get_default_task_prompt(self, column_names: List[str], sample_rows_str: str, user_target: str, category: str) -> str:
        """Get default task prompt."""
        category_name = 'pre-training' if category.upper() == 'PT' else 'supervised fine-tuning'
        
        if category.upper() == "PT":
            example_output = '''{
  "text": "content",
  "meta": {
    "source": "url",
    "language": "lang"
  }
}'''
        else:  # SFT
            example_output = '''{
  "messages": [
    {"role": "user", "content": "instruction", "loss_mask": false},
    {"role": "assistant", "content": "output", "loss_mask": true}
  ],
  "system": "system_prompt",
  "meta": {
    "source": "source"
  }
}'''
        
        return f"""[User Requirements]
User's original request: {user_target}

[Dataset Information]
Dataset Columns: {column_names}
Sample Data: {sample_rows_str}

[Task]
Analyze the dataset and identify the field mappings for {category_name}.

[Output Requirements]
- Return ONLY a single JSON object (NOT an array/list)
- Use the exact field names from the dataset columns
- Wrap the JSON in ```json and ``` markers

Example output format:
```json
{example_output}
```

Now analyze the dataset and return the field mapping as a JSON object:"""

    async def invoke_file_discovery(self, file_list_str: str) -> List[str]:
        """Invoke LLM for file discovery."""
        logger.info("Calling LLM for file discovery...")
        
        if self.prompt_generator:
            try:
                system_prompt = self.prompt_generator.templates.get("system_prompt_for_file_discovery")
                task_prompt_template = self.prompt_generator.templates.get("task_prompt_for_file_discovery")
                if system_prompt and task_prompt_template:
                    human_prompt = task_prompt_template.format(file_list=file_list_str)
                else:
                    raise KeyError("Template not found")
            except Exception as e:
                logger.warning(f"Failed to load prompt, using default: {e}")
                system_prompt = self._get_default_file_discovery_system_prompt()
                human_prompt = f"File list:\n{file_list_str}\n\nPlease identify data files. Return a JSON array of file paths."
        else:
            system_prompt = self._get_default_file_discovery_system_prompt()
            human_prompt = f"File list:\n{file_list_str}\n\nPlease identify data files. Return a JSON array of file paths."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            answer_msg = await self.llm.ainvoke(messages)
            answer_text = answer_msg.content.strip()
            logger.info(f'LLM file discovery response: {answer_text}')

            pattern = r'```json([\s\S]*?)```'
            match = re.search(pattern, answer_text)
            if match:
                match_text = match.group(1).strip()
            else:
                match_text = answer_text

            result = json.loads(match_text)
            if isinstance(result, list) and all(isinstance(item, str) for item in result):
                return result
            else:
                raise ValueError("LLM did not return a JSON list of strings.")
        except Exception as e:
            logger.error(f"Failed to parse file discovery response: {e}")
            raise

    def _get_default_file_discovery_system_prompt(self) -> str:
        """Get default file discovery system prompt."""
        return """You are a file discovery expert. Your task is to identify which files in the provided file list are data files that should be processed.

Data files typically have extensions like: .json, .jsonl, .csv, .parquet, .arrow, .txt, .tsv, etc.
Exclude output files, summary files, cache files, and other non-data files.

Return a JSON array of file paths (relative paths from the root directory)."""

    async def invoke_file_filter(
        self,
        file_path: str,
        sampled_records: List[Dict[str, Any]],
        dataset_background: str
    ) -> bool:
        """Invoke LLM to determine if a file matches the dataset background."""
        logger.info(f"Calling LLM for file filter check: {file_path}")
        
        if self.prompt_generator:
            try:
                system_prompt = self.prompt_generator.templates.get("system_prompt_for_file_filter")
                task_prompt_template = self.prompt_generator.templates.get("task_prompt_for_file_filter")
                if system_prompt and task_prompt_template:
                    sampled_records_str = json.dumps(sampled_records, indent=2, ensure_ascii=False)
                    human_prompt = task_prompt_template.format(
                        file_path=file_path,
                        sampled_records=sampled_records_str,
                        dataset_background=dataset_background
                    )
                else:
                    raise KeyError("Template not found")
            except Exception as e:
                logger.warning(f"Failed to load file filter prompt, using default: {e}")
                system_prompt = "You are a dataset filtering expert."
                sampled_records_str = json.dumps(sampled_records, indent=2, ensure_ascii=False)
                human_prompt = f"File: {file_path}\nBackground: {dataset_background}\nSamples: {sampled_records_str}\n\nDoes this match? Return JSON with is_match boolean."
        else:
            system_prompt = "You are a dataset filtering expert."
            sampled_records_str = json.dumps(sampled_records, indent=2, ensure_ascii=False)
            human_prompt = f"File: {file_path}\nBackground: {dataset_background}\nSamples: {sampled_records_str}\n\nDoes this match? Return JSON with is_match boolean."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            answer_msg = await self.llm.ainvoke(messages)
            answer_text = answer_msg.content.strip()
            logger.info(f'LLM file filter response: {answer_text}')

            pattern = r'```json([\s\S]*?)```'
            match = re.search(pattern, answer_text)
            if match:
                match_text = match.group(1).strip()
            else:
                match_text = answer_text

            try:
                result = json.loads(match_text)
                if isinstance(result, dict):
                    is_match = result.get("is_match", False)
                elif isinstance(result, bool):
                    is_match = result
                else:
                    is_match = "true" in answer_text.lower() or "yes" in answer_text.lower()
            except json.JSONDecodeError:
                is_match = "true" in answer_text.lower() or "yes" in answer_text.lower()
            
            logger.info(f"File filter result for {file_path}: {'MATCH' if is_match else 'NOT MATCH'}")
            return is_match
        except Exception as e:
            logger.error(f"Failed to parse file filter response: {e}")
            return True  # Default to keeping the file

    # File processing methods
    def _is_compressed_file(self, file_path: str) -> bool:
        """Check if file is compressed."""
        compressed_extensions = [
            '.zip', '.tar', '.tar.gz', '.tgz', 
            '.tar.bz2', '.tbz2', '.tar.xz', '.txz',
            '.gz', '.bz2', '.xz', '.7z', '.rar'
        ]
        path_lower = file_path.lower()
        return any(path_lower.endswith(ext) for ext in compressed_extensions)

    def _extract_compressed_file(self, compressed_path: str) -> Optional[str]:
        """Extract compressed file to temporary directory."""
        if not os.path.exists(compressed_path):
            logger.error(f"Compressed file does not exist: {compressed_path}")
            return None
            
        temp_base_dir = os.getenv("DF_TEMP_DIR") or None
        if temp_base_dir is None:
            parent_dir = os.path.dirname(os.path.abspath(compressed_path))
            tmp_candidate = os.path.join(parent_dir, ".tmp")
            try:
                os.makedirs(tmp_candidate, exist_ok=True)
                temp_base_dir = tmp_candidate
            except Exception:
                temp_base_dir = None
        temp_dir = tempfile.mkdtemp(prefix="dataflow_extract_", dir=temp_base_dir)
        self._temp_dirs.append(temp_dir)
        logger.info(f"Extracting {compressed_path} to {temp_dir}")
        
        try:
            path_lower = compressed_path.lower()
            
            if path_lower.endswith('.zip'):
                with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                return temp_dir
            
            elif '.tar' in path_lower or path_lower.endswith(('.tgz', '.tbz2', '.txz')):
                with tarfile.open(compressed_path, 'r:*') as tar_ref:
                    tar_ref.extractall(temp_dir)
                return temp_dir
            
            elif path_lower.endswith('.gz') and '.tar' not in path_lower:
                output_file = os.path.join(temp_dir, Path(compressed_path).stem)
                with gzip.open(compressed_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return temp_dir
            
            elif path_lower.endswith('.bz2') and '.tar' not in path_lower:
                output_file = os.path.join(temp_dir, Path(compressed_path).stem)
                with bz2.open(compressed_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return temp_dir
            
            elif path_lower.endswith('.xz') and '.tar' not in path_lower:
                output_file = os.path.join(temp_dir, Path(compressed_path).stem)
                with lzma.open(compressed_path, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return temp_dir
            
            else:
                logger.warning(f"Unsupported compression format: {compressed_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract file {compressed_path}: {e}")
            return None

    def _cleanup_temp_dirs(self):
        """Clean up all temporary directories."""
        for temp_dir in self._temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
        self._temp_dirs.clear()

    def _get_file_list_string(self, root_path: str, exclude_files: List[str] = None) -> str:
        """Generate file list string from directory."""
        if exclude_files is None:
            exclude_files = []
        
        file_list = []
        for root, dirs, files in os.walk(root_path, topdown=True):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(('.', '__'))
                and d not in ('.cache', 'processed_output', '.tmp', 'rag_db', 'web_get')
            ]
            files = [f for f in files if not f.startswith(('.', '__'))]
            
            for f in files:
                if f in exclude_files:
                    continue
                if f.endswith('.conda'):
                    continue
                full_path = os.path.join(root, f)
                relative_path = os.path.relpath(full_path, root_path)
                file_list.append(relative_path.replace(os.sep, '/'))
        
        if not file_list:
            return "This directory is empty."
        
        return "File list:\n" + "\n".join(sorted(file_list))

    def _chunk_file_list_for_llm(self, file_list_str: str, max_chars: int = 8000) -> List[str]:
        """Split file list string into chunks for LLM."""
        if not file_list_str or len(file_list_str) <= max_chars:
            return [file_list_str]

        lines = file_list_str.splitlines()
        if not lines:
            return []

        header = lines[0] if lines[0].strip().lower().startswith("file list") else None
        content_lines = lines[1:] if header else lines

        chunks: List[List[str]] = []
        current_chunk: List[str] = []
        current_char_len = 0

        for line in content_lines:
            line_len = len(line) + 1
            if current_chunk and current_char_len + line_len > max_chars:
                chunks.append(current_chunk)
                current_chunk = [line]
                current_char_len = line_len
            else:
                current_chunk.append(line)
                current_char_len += line_len

        if current_chunk:
            chunks.append(current_chunk)

        if len(chunks) <= 1:
            return [file_list_str]

        result_chunks: List[str] = []
        for idx, chunk_lines in enumerate(chunks, start=1):
            chunk_header = f"File list (chunk {idx}/{len(chunks)})"
            chunk_str = chunk_header + "\n" + "\n".join(chunk_lines)
            result_chunks.append(chunk_str)

        return result_chunks

    def _get_builder_type(self, file_path: str) -> Optional[str]:
        """Guess builder type for load_dataset."""
        path_lower = file_path.lower()
        if '.jsonl' in path_lower or '.json' in path_lower:
            return 'json'
        if '.csv' in path_lower:
            return 'csv'
        if '.parquet' in path_lower:
            return 'parquet'
        if '.arrow' in path_lower:
            return 'arrow'
        if '.txt' in path_lower or '.md' in path_lower:
            return 'text'
        return None

    async def _load_with_datasets(self, builder_type: str, file_path: str) -> Optional[Any]:
        """Load file using datasets library."""
        if not HAS_DATASETS:
            logger.warning("datasets library not available")
            return None
            
        try:
            temp_base_dir = os.getenv("DF_TEMP_DIR") or None
            if temp_base_dir is None:
                parent_dir = os.path.dirname(os.path.abspath(file_path))
                tmp_candidate = os.path.join(parent_dir, ".tmp")
                try:
                    os.makedirs(tmp_candidate, exist_ok=True)
                    temp_base_dir = tmp_candidate
                except Exception:
                    temp_base_dir = None
            temp_cache_dir = tempfile.mkdtemp(prefix="datasets_cache_", dir=temp_base_dir)
            self._temp_dirs.append(temp_cache_dir)

            dl_config = DownloadConfig(cache_dir=temp_cache_dir)

            data = load_dataset(
                path=builder_type,
                data_files=file_path,
                cache_dir=temp_cache_dir,
                keep_in_memory=True,
                download_config=dl_config,
            )
            return data
        except Exception as e:
            logger.error(f"Error in datasets loading: {e}")
            return None

    async def _load_with_fallback(self, builder_type: str, file_path: str) -> Optional[Any]:
        """Load file using fallback methods (pandas)."""
        if pd is None:
            logger.warning("pandas not available for fallback loading")
            return None
            
        try:
            if builder_type == 'json':
                try:
                    df = pd.read_json(file_path, lines=True)
                except ValueError:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        df = pd.json_normalize(parsed)
                    elif isinstance(parsed, dict):
                        for key in ['data', 'items', 'records', 'train']:
                            if key in parsed and isinstance(parsed[key], list):
                                df = pd.json_normalize(parsed[key])
                                break
                        else:
                            df = pd.json_normalize([parsed])
                    else:
                        return None
            elif builder_type == 'csv':
                df = pd.read_csv(file_path)
            elif builder_type == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                return None
            
            if df is None or len(df) == 0:
                return None
            
            records = df.to_dict(orient="records")
            return _build_simple_dataset(records)
        except Exception as e:
            logger.error(f"Error in fallback loading: {e}")
            return None

    async def _process_dataset_with_mapping(
        self,
        data_content: Any,
        file_path: str,
        file_name: str,
        split_name: str,
        annotation_result: Dict[str, Any],
        category: str,
        output_jsonl_prefix: str,
        processed_sources_list: List[Tuple[str, int]]
    ) -> int:
        """Process a single split with pre-computed mapping result."""
        logger.info(f"--- Processing Split: '{split_name}' (from {file_name}) ---")
        
        if len(data_content) == 0:
            logger.info(f"Split '{split_name}' is empty, skipping.")
            return 0
        
        column_names = data_content.column_names
        
        split_record_count = 0
        chunk_size = 10000
        current_chunk_index = 1
        current_chunk_count = 0

        def _open_chunk_file(index: int):
            chunk_path = f"{output_jsonl_prefix}_{index:05d}.jsonl"
            return open(chunk_path, 'a', encoding='utf-8')

        f_out = _open_chunk_file(current_chunk_index)
        try:
            if category.upper() == 'PT':
                for record_index, row in enumerate(data_content):
                    intermediate_record = self._build_intermediate_format_pt(
                        row, annotation_result, file_path, record_index, column_names
                    )
                    if intermediate_record:
                        json.dump(intermediate_record, f_out, ensure_ascii=False)
                        f_out.write('\n')
                        split_record_count += 1
                        current_chunk_count += 1
                        if current_chunk_count >= chunk_size:
                            f_out.close()
                            current_chunk_index += 1
                            current_chunk_count = 0
                            f_out = _open_chunk_file(current_chunk_index)
                        
            elif category.upper() == 'SFT':
                for record_index, row in enumerate(data_content):
                    intermediate_record = self._build_intermediate_format_sft(
                        row, annotation_result, file_path, record_index, column_names
                    )
                    if intermediate_record:
                        json.dump(intermediate_record, f_out, ensure_ascii=False)
                        f_out.write('\n')
                        split_record_count += 1
                        current_chunk_count += 1
                        if current_chunk_count >= chunk_size:
                            f_out.close()
                            current_chunk_index += 1
                            current_chunk_count = 0
                            f_out = _open_chunk_file(current_chunk_index)
        finally:
            try:
                f_out.close()
            except Exception:
                pass
        
        if split_record_count > 0:
            logger.info(f"Extracted {split_record_count} records from {file_name} ({split_name}).")
            processed_sources_list.append((f"{file_name}_({split_name})", split_record_count))
        
        return split_record_count
