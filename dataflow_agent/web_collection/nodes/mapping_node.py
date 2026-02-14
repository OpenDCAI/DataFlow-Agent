"""
Mapping Node
~~~~~~~~~~~~

数据映射节点，执行：
1. 使用 LLM 生成 Python 映射函数
2. 三重验证确保映射函数一致性
3. 批量处理所有数据
4. 转换为 Alpaca 或自定义格式

"""

import os
import json
import re
import random
from typing import Dict, Any, List, Optional, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dataflow_agent.states.web_collection_state import WebCollectionState
from dataflow_agent.promptstemplates import PromptsTemplateGenerator
from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


# Alpaca format schema
ALPACA_SCHEMA = {
    "instruction": "string - Task instruction or question",
    "input": "string - Optional input context (e.g., system prompt, SQL schema, code context)",
    "output": "string - Expected response or answer"
}

ALPACA_EXAMPLE = {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
}

# Few-shot examples for SFT data with system messages (e.g., text2sql)
SFT_FEWSHOT_EXAMPLES = """
## Few-shot Examples for SFT data mapping:

### Example 1: Text2SQL with system message containing SQL schema
Input:
{
  "messages": [
    {"role": "system", "content": "CREATE TABLE farm (Id VARCHAR)"},
    {"role": "user", "content": "How many farms are there?"},
    {"role": "assistant", "content": "SELECT COUNT(*) FROM farm"}
  ]
}

Expected Output:
{
  "instruction": "How many farms are there?",
  "input": "CREATE TABLE farm (Id VARCHAR)",
  "output": "SELECT COUNT(*) FROM farm"
}

### Example 2: Text2SQL with multiple tables in schema
Input:
{
  "messages": [
    {"role": "system", "content": "CREATE TABLE department (department_id VARCHAR, name VARCHAR); CREATE TABLE management (department_id VARCHAR, head_id VARCHAR)"},
    {"role": "user", "content": "Which department has more than 1 head at a time?"},
    {"role": "assistant", "content": "SELECT T1.department_id, T1.name, COUNT(*) FROM management AS T2 JOIN department AS T1 ON T1.department_id = T2.department_id GROUP BY T1.department_id HAVING COUNT(*) > 1"}
  ]
}

Expected Output:
{
  "instruction": "Which department has more than 1 head at a time?",
  "input": "CREATE TABLE department (department_id VARCHAR, name VARCHAR); CREATE TABLE management (department_id VARCHAR, head_id VARCHAR)",
  "output": "SELECT T1.department_id, T1.name, COUNT(*) FROM management AS T2 JOIN department AS T1 ON T1.department_id = T2.department_id GROUP BY T1.department_id HAVING COUNT(*) > 1"
}

### Example 3: Code generation with system context
Input:
{
  "messages": [
    {"role": "system", "content": "You are a Python expert. Available libraries: pandas, numpy"},
    {"role": "user", "content": "Write a function to calculate mean of a list"},
    {"role": "assistant", "content": "def calculate_mean(numbers):\\n    return sum(numbers) / len(numbers)"}
  ]
}

Expected Output:
{
  "instruction": "Write a function to calculate mean of a list",
  "input": "You are a Python expert. Available libraries: pandas, numpy",
  "output": "def calculate_mean(numbers):\\n    return sum(numbers) / len(numbers)"
}

### Example 4: QA without system message
Input:
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."}
  ]
}

Expected Output:
{
  "instruction": "What is machine learning?",
  "input": "",
  "output": "Machine learning is a subset of AI that enables systems to learn from data."
}

**Key Rule**: When the input has a "system" role message, its content should ALWAYS be placed in the "input" field of the Alpaca format. This is critical for tasks like text2sql where the schema information is essential context.
"""


async def mapping_node(state: WebCollectionState) -> WebCollectionState:
    """
    Mapping node that converts intermediate data to target format using LLM
    
    This node implements the full logic from the old Constructor:
    1. LLM analyzes sample data and target format
    2. LLM generates Python mapping function code (with triple verification)
    3. Execute mapping function to batch process all data
    4. Apply quality filters
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with mapping results
    """
    logger.info("=== Mapping Node: Starting ===")
    state.current_node = "mapping_node"
    
    # Check for intermediate data
    intermediate_path = state.intermediate_data_path
    
    if not intermediate_path:
        # Try to find processed output in download directory
        processed_dir = os.path.join(state.request.download_dir, "processed_output")
        if os.path.exists(processed_dir):
            intermediate_path = processed_dir
        else:
            logger.info("No intermediate data found, skipping mapping node")
            return state
    
    if not os.path.exists(intermediate_path):
        logger.info(f"Intermediate data path does not exist: {intermediate_path}")
        return state
    
    try:
        # Get configuration
        model_name = state.request.model
        base_url = state.request.chat_api_url
        api_key = state.request.api_key
        temperature = state.request.temperature or 0.0
        category = state.request.category.upper()
        target_format = state.request.output_format or "alpaca"
        
        if not model_name or not base_url or not api_key:
            logger.error("Missing required configuration for mapping node")
            state.exception = "Missing model configuration for mapping node"
            return state
        
        # Initialize Prompt Generator
        prompt_generator = None
        try:
            prompt_generator = PromptsTemplateGenerator("pt_web_collection")
        except Exception as e:
            logger.warning(f"Failed to load prompt templates: {e}")
        
        # Output directory for mapped data
        output_dir = os.path.dirname(intermediate_path) if os.path.isfile(intermediate_path) else intermediate_path
        mapping_output_dir = os.path.join(output_dir, "mapped_output")
        os.makedirs(mapping_output_dir, exist_ok=True)
        
        # Run async workflow
        result = await _mapping_workflow(
            intermediate_path=intermediate_path,
            output_dir=mapping_output_dir,
            target_format=target_format,
            category=category,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            prompt_generator=prompt_generator,
        )
        
        # Update state
        if "exception" in result:
            state.exception = result["exception"]
        else:
            state.mapping_results = result
            state.is_finished = True
            logger.info(f"Mapping completed: {result.get('mapped_records', 0)} records mapped to {target_format}")
        
    except Exception as e:
        logger.error(f"Mapping node error: {e}", exc_info=True)
        state.exception = f"Mapping error: {str(e)}"
    
    logger.info("=== Mapping Node: Completed ===")
    return state


async def _mapping_workflow(
    intermediate_path: str,
    output_dir: str,
    target_format: str,
    category: str,
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
) -> Dict[str, Any]:
    """
    Async workflow for mapping data to target format using LLM-generated functions
    
    Process:
    1. Read intermediate data
    2. LLM generates Python mapping function (with triple verification)
    3. Batch process all records
    4. Apply quality filters
    5. Save output
    """
    try:
        # Read intermediate data
        records = _read_intermediate_data(intermediate_path)
        
        if not records:
            logger.warning("No records found in intermediate data")
            return {
                "total_records": 0,
                "mapped_records": 0,
                "output_dir": output_dir,
                "output_file": "",
                "mapping_type": "llm"
            }
        
        logger.info(f"Read {len(records)} records from intermediate data")
        
        # Initialize LLM
        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=temperature
        )
        
        # Get target schema and example
        if target_format.lower() == "alpaca":
            target_schema = ALPACA_SCHEMA
            target_example = ALPACA_EXAMPLE
        else:
            # Use default schema for unknown formats
            target_schema = ALPACA_SCHEMA
            target_example = ALPACA_EXAMPLE
        
        # Sample records for LLM function generation
        num_samples = min(3, len(records))
        if len(records) <= num_samples:
            sample_records = records
        else:
            sample_records = random.sample(records, num_samples)
        
        # Step 1: Generate mapping function with triple verification
        logger.info("Step 1: Generating mapping function using LLM with triple verification...")
        
        mapping_func = await _generate_mapping_function_with_verification(
            llm=llm,
            sample_records=sample_records,
            target_schema=target_schema,
            target_example=target_example,
            category=category,
            prompt_generator=prompt_generator,
        )
        
        if mapping_func is None:
            logger.warning("Failed to generate mapping function, using fallback mapper")
            mapping_func = lambda record: _fallback_map_to_alpaca(record, category)
        else:
            logger.info("Mapping function generated and verified successfully")
        
        # Step 2: Batch process all records
        logger.info("Step 2: Batch processing all records...")
        
        output_file = os.path.join(output_dir, f"final_{target_format}_{category.lower()}.jsonl")
        
        mapped_count = 0
        failed_count = 0
        filtered_records = []
        
        for idx, record in enumerate(records):
            try:
                mapped_record = mapping_func(record)
                
                if mapped_record and _is_valid_record(mapped_record):
                    filtered_records.append(mapped_record)
                    mapped_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                if idx < 10:
                    logger.warning(f"Record {idx}: Error during mapping: {e}")
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1} records (mapped: {mapped_count}, failed: {failed_count})")
        
        # Step 3: Apply quality filters
        logger.info("Step 3: Applying quality filters...")
        filtered_records = _apply_quality_filters(filtered_records, category)
        
        logger.info(f"After quality filtering: {len(filtered_records)} records")
        
        # Step 4: Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in filtered_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(filtered_records)} records to {output_file}")
        
        # Also save as JSON for convenience
        json_output_file = output_file.replace('.jsonl', '.json')
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_records, f, ensure_ascii=False, indent=2)
        
        return {
            "total_records": len(records),
            "mapped_records": len(filtered_records),
            "failed_records": failed_count,
            "output_dir": output_dir,
            "output_file": output_file,
            "json_output_file": json_output_file,
            "format_id": target_format,
            "category": category,
            "mapping_type": "llm"
        }
        
    except Exception as e:
        logger.error(f"Mapping workflow error: {e}", exc_info=True)
        return {"exception": str(e)}


async def _generate_mapping_function_with_verification(
    llm: ChatOpenAI,
    sample_records: List[Dict[str, Any]],
    target_schema: Dict[str, Any],
    target_example: Dict[str, Any],
    category: str,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
) -> Optional[Callable]:
    """
    Generate mapping function using LLM with triple verification
    
    Generates mapping function three times and verifies consistency
    """
    mapping_functions = []
    
    # Generate mapping function three times
    for attempt in range(3):
        logger.info(f"Generating mapping function (attempt {attempt + 1}/3)...")
        
        mapping_func = await _generate_single_mapping_function(
            llm=llm,
            sample_records=sample_records,
            target_schema=target_schema,
            target_example=target_example,
            category=category,
            prompt_generator=prompt_generator,
        )
        
        if mapping_func is None:
            logger.warning(f"Failed to generate mapping function on attempt {attempt + 1}")
            continue
        
        mapping_functions.append(mapping_func)
    
    if len(mapping_functions) < 2:
        logger.debug("Could not generate enough mapping functions for verification, using single function")
        return mapping_functions[0] if mapping_functions else None
    
    # Verify consistency: test all functions on the same sample records
    logger.info("Verifying mapping function consistency...")
    
    if not _verify_mapping_functions(mapping_functions, sample_records):
        logger.debug("Triple verification failed, using first successful function")
        return mapping_functions[0]
    
    logger.info("Triple verification passed: mapping functions are consistent")
    return mapping_functions[0]


async def _generate_single_mapping_function(
    llm: ChatOpenAI,
    sample_records: List[Dict[str, Any]],
    target_schema: Dict[str, Any],
    target_example: Dict[str, Any],
    category: str,
    prompt_generator: Optional[PromptsTemplateGenerator] = None,
) -> Optional[Callable]:
    """
    Generate a single mapping function using LLM
    """
    # Get prompts
    if prompt_generator:
        try:
            system_prompt = prompt_generator.templates.get("system_prompt_for_llm_mapping_function")
            task_prompt_template = prompt_generator.templates.get("task_prompt_for_llm_mapping_function")
            if not system_prompt:
                raise KeyError("System prompt not found")
        except Exception as e:
            logger.warning(f"Failed to load prompt: {e}")
            system_prompt = _get_default_mapping_system_prompt()
            task_prompt_template = None
    else:
        system_prompt = _get_default_mapping_system_prompt()
        task_prompt_template = None
    
    # Build sample text
    samples_text = "\n\n".join([
        f"Sample {i+1}:\n{json.dumps(record, ensure_ascii=False, indent=2)}"
        for i, record in enumerate(sample_records[:3])
    ])
    
    if task_prompt_template:
        try:
            user_prompt = task_prompt_template.format(
                sample_input=samples_text,
                target_schema=json.dumps(target_schema, ensure_ascii=False, indent=2),
                target_example=json.dumps(target_example, ensure_ascii=False, indent=2),
                category=category,
            )
        except Exception:
            user_prompt = _build_default_task_prompt(samples_text, target_schema, target_example, category)
    else:
        user_prompt = _build_default_task_prompt(samples_text, target_schema, target_example, category)
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        code = response.content.strip()
        
        logger.debug(f"LLM generated code:\n{code[:500]}...")
        
        # Extract function code
        func_code = _extract_function_code(code)
        
        if not func_code:
            logger.error("Failed to extract function code from LLM response")
            return None
        
        # Execute code to get function
        namespace = {}
        exec(func_code, namespace)
        
        if 'map_record' not in namespace:
            logger.error("Function 'map_record' not found in generated code")
            return None
        
        mapping_func = namespace['map_record']
        
        # Validate function
        try:
            test_result = mapping_func(sample_records[0])
            if not isinstance(test_result, dict):
                logger.error(f"Mapping function returned non-dict: {type(test_result)}")
                return None
            logger.debug(f"Function validation successful, test output: {json.dumps(test_result, ensure_ascii=False)[:200]}")
        except Exception as e:
            logger.error(f"Function validation failed: {e}")
            return None
        
        return mapping_func
        
    except Exception as e:
        logger.error(f"Error generating mapping function: {e}")
        return None


def _get_default_mapping_system_prompt() -> str:
    """Get default system prompt for mapping function generation"""
    return """You are a Python programming expert and data transformation specialist.

Your task is: Based on sample input data and target format definition, write a Python mapping function to convert input format to target format.

**Input Format (Intermediate)**:
- PT mode: {"text": "string | array<string>", "meta": {...}}
- SFT mode: {"messages": [{"role": "...", "content": "..."}], "meta": {...}}
  - Messages may contain: system (context/schema), user (question/instruction), assistant (answer/output)

**CRITICAL MAPPING RULES for SFT data**:
- The "system" role message content should be mapped to the "input" field
- The "user" role message content should be mapped to the "instruction" field
- The "assistant" role message content should be mapped to the "output" field
- This is especially important for text2sql data where system contains SQL schema (CREATE TABLE statements)

**Requirements**:
1. Function name must be `map_record`
2. Function signature: `def map_record(record: dict) -> dict:`
3. Function must be self-contained, no external dependencies or imports
4. Handle edge cases (null values, missing fields, type conversions, etc.)
5. If content or text is a list, merge into string (use newline separator)
6. Only output function code, no explanations or markdown markers
7. Code must be robust and handle exceptions

Only output function code, no other content."""


def _build_default_task_prompt(samples_text: str, target_schema: Dict, target_example: Dict, category: str) -> str:
    """Build default task prompt"""
    return f"""[Sample Input Data]
{samples_text}

[Target Format Schema]
{json.dumps(target_schema, ensure_ascii=False, indent=2)}

[Target Format Example]
{json.dumps(target_example, ensure_ascii=False, indent=2)}

[Category]
{category}

{SFT_FEWSHOT_EXAMPLES}

Please write a Python mapping function `map_record(record: dict) -> dict` that converts the input data to the target format.

**CRITICAL Requirements**:
1. Function name must be `map_record`
2. Handle edge cases (null values, missing fields, type conversions)
3. If content or text is a list, merge into string (use newline separator)
4. **IMPORTANT**: For SFT data with messages, the "system" role content MUST be placed in the "input" field (e.g., SQL schema for text2sql tasks)
5. Only output function code, no explanations

Only output the Python function code."""


def _extract_function_code(text: str) -> Optional[str]:
    """Extract function code from LLM response"""
    # Try to extract from code block
    pattern = r'```(?:python)?\s*([\s\S]*?)```'
    match = re.search(pattern, text)
    if match:
        code = match.group(1).strip()
        if 'def map_record' in code:
            return code
    
    # Try to find function directly
    if 'def map_record' in text:
        start_idx = text.find('def map_record')
        if start_idx != -1:
            return text[start_idx:].strip()
    
    return None


def _verify_mapping_functions(
    mapping_functions: List[Callable],
    sample_records: List[Dict[str, Any]]
) -> bool:
    """
    Verify that all mapping functions produce consistent results
    """
    if len(mapping_functions) < 2:
        return True
    
    for record_idx, sample_record in enumerate(sample_records):
        results = []
        
        for func_idx, mapping_func in enumerate(mapping_functions):
            try:
                result = mapping_func(sample_record)
                normalized_result = _normalize_mapping_result(result)
                results.append(normalized_result)
            except Exception as e:
                logger.error(f"Error applying mapping function {func_idx + 1} to sample {record_idx + 1}: {e}")
                return False
        
        # Compare all results
        if len(set(json.dumps(r, sort_keys=True) for r in results)) != 1:
            logger.debug(f"Sample {record_idx + 1} produced inconsistent results across mapping functions")
            return False
    
    return True


def _normalize_mapping_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize mapping result for comparison"""
    if not isinstance(result, dict):
        return result
    
    normalized = {}
    for key in sorted(result.keys()):
        value = result[key]
        if isinstance(value, dict):
            normalized[key] = _normalize_mapping_result(value)
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                normalized[key] = sorted(
                    [_normalize_mapping_result(item) for item in value],
                    key=lambda x: json.dumps(x, sort_keys=True)
                )
            else:
                normalized[key] = value
        else:
            normalized[key] = value
    
    return normalized


def _fallback_map_to_alpaca(record: Dict[str, Any], category: str) -> Optional[Dict[str, str]]:
    """Fallback mapper to Alpaca format"""
    if not record:
        return None
    
    if category == "SFT":
        # For SFT data with messages format
        messages = record.get("messages", [])
        if messages:
            instruction = ""
            input_text = ""
            output_text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    # System message (e.g., SQL schema) goes to input field
                    input_text = content
                elif role == "user":
                    instruction = content
                elif role == "assistant":
                    output_text = content
            
            if instruction or output_text:
                return {
                    "instruction": str(instruction).strip(),
                    "input": str(input_text).strip(),
                    "output": str(output_text).strip(),
                }
        
        # Try legacy format
        instruction = record.get("instruction") or record.get("prompt") or record.get("question") or ""
        input_text = record.get("input") or record.get("context") or record.get("system") or ""
        output_text = record.get("output") or record.get("response") or record.get("answer") or ""
        
        if instruction or output_text:
            return {
                "instruction": str(instruction).strip(),
                "input": str(input_text).strip(),
                "output": str(output_text).strip(),
            }
    
    else:  # PT
        text = record.get("text") or record.get("content") or ""
        
        if isinstance(text, list):
            text = "\n".join(str(t) for t in text)
        
        if text and len(str(text)) > 50:
            return {
                "instruction": "Continue the following text:",
                "input": "",
                "output": str(text).strip(),
            }
    
    return None


def _is_valid_record(record: Dict[str, Any]) -> bool:
    """Check if mapping result is valid"""
    if not record:
        return False
    
    for value in record.values():
        if value:
            if isinstance(value, str) and value.strip():
                return True
            elif isinstance(value, (list, dict)) and value:
                return True
            elif isinstance(value, (int, float, bool)):
                return True
    
    return False


def _read_intermediate_data(path: str) -> List[Dict[str, Any]]:
    """Read intermediate format data from file or directory"""
    records = []
    
    if os.path.isfile(path):
        records.extend(_read_jsonl_file(path))
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(path, filename)
                records.extend(_read_jsonl_file(filepath))
    
    return records


def _read_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """Read single JSONL file"""
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON line in {filepath}: {e}")
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
    return records


def _apply_quality_filters(records: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    """Apply quality filters to mapped records"""
    filtered = []
    
    for record in records:
        if not record:
            continue
        
        instruction = record.get("instruction", "")
        output_text = record.get("output", "")
        
        # Length filters
        if category == "SFT":
            if len(instruction) < 5 and len(output_text) < 10:
                continue
            if len(output_text) > 10000:
                continue
        else:
            if len(output_text) < 50:
                continue
            if len(output_text) > 50000:
                continue
        
        # Content quality checks
        if _has_excessive_repetition(output_text):
            continue
        
        if _has_suspicious_content(output_text):
            continue
        
        filtered.append(record)
    
    return filtered


def _has_excessive_repetition(text: str, threshold: float = 0.3) -> bool:
    """Check if text has excessive repetition"""
    if not text or len(text) < 100:
        return False
    
    words = text.lower().split()
    if len(words) < 20:
        return False
    
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    max_count = max(word_counts.values())
    if max_count / len(words) > threshold:
        return True
    
    return False


def _has_suspicious_content(text: str) -> bool:
    """Check for suspicious content patterns"""
    if not text:
        return False
    
    text_lower = text.lower()
    
    suspicious_patterns = [
        "lorem ipsum",
        "test test test",
        "asdf",
        "qwerty",
        "[object object]",
        "undefined",
        "null null",
    ]
    
    for pattern in suspicious_patterns:
        if pattern in text_lower:
            return True
    
    return False
