"""
Category Classifier and Task Decomposer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

数据收集任务的分类和分解工具：
- CategoryClassifier: 判断任务类型（SFT/PT）
- ObtainQueryNormalizer: 标准化用户查询
- TaskDecomposer: 分解复杂任务为子任务
"""

import json
from typing import Optional, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dataflow_agent.logger import get_logger
from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator

logger = get_logger(__name__)


class CategoryClassifier:
    """Category Classifier for determining SFT or PT task type from user query"""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.3,
        prompt_generator: Optional[PromptsTemplateGenerator] = None,
    ):
        """
        Initialize Category Classifier
        
        Args:
            model_name: LLM model name
            base_url: API base URL
            api_key: API key
            temperature: Sampling temperature
            prompt_generator: Optional prompt template generator
        """
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        self.prompt_generator = prompt_generator

    async def classify_category(
        self, user_query: str, objective: str = ""
    ) -> Dict[str, str]:
        """
        Classify the task category as SFT or PT based on user query and extract dataset background
        
        Args:
            user_query: The user's query or message
            objective: Optional objective description
            
        Returns:
            Dictionary with "category" ("SFT" or "PT") and "dataset_background" (str)
        """
        logger.info("\n--- Category Classifier ---")
        
        # Use prompt generator if available
        if self.prompt_generator:
            try:
                system_prompt = self.prompt_generator.render("system_prompt_for_category_classifier")
                task_prompt = self.prompt_generator.render(
                    "task_prompt_for_category_classifier",
                    user_query=user_query,
                    objective=objective if objective else user_query
                )
                human_prompt = task_prompt
            except Exception as e:
                logger.warning(f"Failed to load prompt, using default: {e}")
                system_prompt = self._get_default_system_prompt()
                human_prompt = self._get_default_task_prompt(user_query, objective)
        else:
            system_prompt = self._get_default_system_prompt()
            human_prompt = self._get_default_task_prompt(user_query, objective)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            logger.info(f"Category classifier raw response: {response.content}")

            # Parse response
            clean_response = (
                response.content.strip().replace("```json", "").replace("```", "").strip()
            )
            logger.debug(f"Cleaned response: {clean_response[:200]}...")
            
            # Try to parse as JSON first
            try:
                result = json.loads(clean_response)
                logger.debug(f"Successfully parsed JSON response: {type(result)}")
                
                if isinstance(result, dict):
                    category = result.get("category", "").upper()
                    dataset_background = result.get("dataset_background", "")
                    logger.info(f"[Dataset Background] Extracted - category: {category}, background length: {len(dataset_background)}")
                elif isinstance(result, str):
                    category = result.upper()
                    dataset_background = ""
                    logger.warning(f"[Dataset Background] Response is string type. Category: {category}")
                else:
                    category = clean_response.upper()
                    dataset_background = ""
                    logger.warning(f"[Dataset Background] Unexpected type ({type(result)}). Category: {category}")
            except json.JSONDecodeError as e:
                logger.warning(f"[Dataset Background] Failed to parse JSON: {e}")
                category = clean_response.upper()
                dataset_background = ""
            
            # Validate category
            if category not in ["SFT", "PT"]:
                logger.warning(f"Invalid category '{category}', defaulting to PT")
                category = "PT"
            
            # Fallback for empty dataset_background
            if not dataset_background:
                logger.info("[Dataset Background] Empty, using user_query as fallback")
                dataset_background = user_query if user_query else objective
            
            logger.info(f"Classified category: {category}")
            
            return {
                "category": category,
                "dataset_background": dataset_background
            }
                
        except Exception as e:
            logger.error(f"Error in category classification: {e}")
            logger.info("Defaulting to PT category due to classification error")
            fallback_background = user_query if user_query else objective
            return {
                "category": "PT",
                "dataset_background": fallback_background
            }

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """You are a task category classification expert. Your task is to analyze user queries and determine whether they are requesting data for:

1. **SFT (Supervised Fine-Tuning)**: Tasks that require question-answer pairs, instruction-following data, conversational data, or any structured input-output pairs for fine-tuning language models to follow instructions.

2. **PT (Pre-training)**: Tasks that require raw text data, documents, code, or any continuous text corpus for pre-training language models from scratch or continuing pre-training.

Key indicators for SFT:
- Mentions of "question", "answer", "QA", "instruction", "conversation", "dialogue", "chat", "fine-tuning", "SFT", "微调", "问答"
- Requests for structured data with input-output pairs
- Tasks involving teaching models to follow instructions

Key indicators for PT:
- Mentions of "pre-training", "PT", "corpus", "text data", "documents", "code dataset"
- Requests for raw, unstructured text data
- Tasks involving building foundational language understanding

Additionally, extract the dataset background description from the user query.

Return a JSON object with:
{
    "category": "SFT" or "PT",
    "dataset_background": "A clear description of the dataset background",
    "reasoning": "Brief explanation of why this category was chosen"
}"""

    def _get_default_task_prompt(self, user_query: str, objective: str) -> str:
        """Get default task prompt"""
        query_text = objective if objective else user_query
        return f"""User query: {user_query}

Research objective: {query_text}

Please analyze the user's query and objective to:
1. Determine if they need SFT data (question-answer pairs, instruction-following data) or PT data (raw text corpus, documents, code)
2. Extract the dataset background description from the query

Return a JSON object with "category", "dataset_background", and "reasoning" fields."""


class ObtainQueryNormalizer:
    """
    Detect evaluation-recommendation style inputs and rewrite them into
    concrete dataset collection objectives suitable for obtain workflow.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.2,
        prompt_generator: Optional[PromptsTemplateGenerator] = None,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        self.prompt_generator = prompt_generator

    async def normalize(self, user_query: str, objective: str = "") -> dict:
        """
        Identify whether the query is:
        - dataset_request: already a direct dataset collection ask
        - eval_recommendation: based on evaluation results / suggestions, needs rewrite
        Returns dict with intent_type, normalized_query, reason, raw_response.
        """
        if not user_query:
            return {}

        logger.info("\n--- Obtain Query Normalizer ---")

        system_prompt, human_prompt = self._build_prompts(user_query, objective)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            logger.info(f"Query normalizer raw response: {response.content}")
            clean = response.content.strip().replace("```json", "").replace("```", "").strip()

            result = self._parse_response(clean, user_query, objective)
            return result
        except Exception as e:
            logger.error(f"Error in query normalization: {e}")
            return {}

    def _build_prompts(self, user_query: str, objective: str):
        """Use prompt generator if available, otherwise default prompts."""
        if self.prompt_generator:
            try:
                system_prompt = self.prompt_generator.render("system_prompt_for_query_normalizer")
                task_prompt = self.prompt_generator.render(
                    "task_prompt_for_query_normalizer",
                    user_query=user_query,
                    objective=objective if objective else user_query
                )
                return system_prompt, task_prompt
            except Exception as e:
                logger.warning(f"Failed to load prompt, using default: {e}")

        # Default prompts
        system_prompt = (
            "You detect whether a request is a direct dataset collection ask, "
            "or an evaluation-based recommendation (e.g., suggests data improvements from eval results). "
            "If it is evaluation-based, rewrite it into a clear dataset collection objective "
            "that the data-obtainer can execute."
        )
        human_prompt = self._default_task_prompt(user_query, objective)
        return system_prompt, human_prompt

    def _default_task_prompt(self, user_query: str, objective: str) -> str:
        query_text = objective if objective else user_query
        return f"""User query: {user_query}

Research objective: {query_text}

Classify intent:
- dataset_request: direct ask to collect datasets for a domain/use (fine-tuning/pretraining).
- eval_recommendation: suggestions derived from model evaluation results about what data to add or improve.

If eval_recommendation, rewrite into a specific dataset collection objective the obtainer can execute.

Respond in JSON:
{{
  "intent_type": "dataset_request" | "eval_recommendation",
  "normalized_query": "<if eval_recommendation, the rewritten dataset collection objective; otherwise original>",
  "reason": "short reason",
  "confidence": 0-1
}}"""

    def _parse_response(self, content: str, user_query: str, objective: str) -> dict:
        try:
            data = json.loads(content)
            if isinstance(data, str):
                return {
                    "intent_type": data,
                    "normalized_query": user_query,
                    "reason": "",
                    "raw": content,
                }
            intent = data.get("intent_type") or data.get("intent") or ""
            normalized = data.get("normalized_query") or data.get("rewritten_query") or data.get("query") or ""
            reason = data.get("reason", "")
            confidence = data.get("confidence")
        except Exception:
            intent = "eval_recommendation" if "eval" in content.lower() else "dataset_request"
            normalized = ""
            reason = content
            confidence = None

        if not normalized:
            normalized = user_query if intent == "dataset_request" else objective or user_query

        return {
            "intent_type": intent,
            "normalized_query": normalized,
            "reason": reason,
            "confidence": confidence,
            "raw": content,
        }


class TaskDecomposer:
    """Task Decomposer for breaking down user input into multiple data collection tasks"""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.3,
        prompt_generator: Optional[PromptsTemplateGenerator] = None,
    ):
        """Initialize Task Decomposer"""
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        self.prompt_generator = prompt_generator

    async def decompose_tasks(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Decompose user input into one or more specific data collection tasks
        
        Args:
            user_input: The user's input query
            
        Returns:
            List of task dictionaries, each with "task_name" field
        """
        logger.info("\n" + "="*60)
        logger.info("--- Task Decomposer ---")
        logger.info(f"原始任务输入: {user_input}")
        logger.info("="*60)
        
        if not user_input:
            logger.warning("Empty user input, returning default single task")
            return [{"task_name": "收集数据集用于大模型微调"}]
        
        # Use prompt generator if available
        if self.prompt_generator:
            try:
                system_prompt = self.prompt_generator.render("system_prompt_for_task_decomposer")
                task_prompt = self.prompt_generator.render(
                    "task_prompt_for_task_decomposer",
                    user_query=user_input,
                    objective=user_input
                )
                human_prompt = task_prompt
            except Exception as e:
                logger.warning(f"Failed to load prompt, using default: {e}")
                system_prompt = self._get_default_system_prompt()
                human_prompt = self._get_default_task_prompt(user_input)
        else:
            system_prompt = self._get_default_system_prompt()
            human_prompt = self._get_default_task_prompt(user_input)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            logger.info(f"Task decomposer raw response: {response.content}")

            # Parse response
            clean_response = (
                response.content.strip().replace("```json", "").replace("```", "").strip()
            )
            
            # Try to parse as JSON
            try:
                result = json.loads(clean_response)
                if isinstance(result, list):
                    tasks = []
                    for task in result:
                        if isinstance(task, dict) and "task_name" in task:
                            tasks.append({"task_name": task["task_name"]})
                        elif isinstance(task, str):
                            tasks.append({"task_name": task})
                    if tasks:
                        logger.info(f"\n任务拆解成功！共拆解为 {len(tasks)} 个子任务：")
                        for idx, task in enumerate(tasks, 1):
                            logger.info(f"  子任务 {idx}: {task['task_name']}")
                        return tasks
                    else:
                        logger.warning("No valid tasks found in response, using default")
                        return [{"task_name": user_input}]
                elif isinstance(result, dict):
                    if "task_name" in result:
                        return [{"task_name": result["task_name"]}]
                    elif "task_list" in result:
                        return [{"task_name": t.get("task_name", str(t))} for t in result["task_list"]]
                    else:
                        return [{"task_name": user_input}]
                else:
                    return [{"task_name": user_input}]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {clean_response}")
                return [{"task_name": user_input}]
                
        except Exception as e:
            logger.error(f"Error in task decomposition: {e}")
            return [{"task_name": user_input}]

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """You are a task decomposition expert. Your task is to analyze user input and decompose it into one or more specific data collection tasks.

**Task Decomposition Rules:**
1. If the user input is a single, specific task, return a list with one task.
2. If the user input contains multiple related but distinct tasks, decompose it into separate tasks.
3. Each task should be specific and actionable for data collection.
4. Task names should be clear and descriptive.
5. Limit to 3-5 tasks maximum to avoid over-decomposition.

**Output Format:**
Return a JSON array where each element is a dictionary with a "task_name" field."""

    def _get_default_task_prompt(self, user_input: str) -> str:
        """Get default task prompt"""
        return f"""User input: {user_input}

Please analyze the user input and decompose it into one or more specific data collection tasks.

Return a JSON array of tasks, each with a "task_name" field. For example:
[
  {{
    "task_name": "收集text2sql数据集用于大模型微调"
  }}
]"""
