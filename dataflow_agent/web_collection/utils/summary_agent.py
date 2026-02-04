"""
Summary Agent
~~~~~~~~~~~~~

摘要代理，从 RAG 内容生成下载子任务。
从老项目 loopai/agents/Obtainer/utils/summary_agent.py 移植。
"""

import json
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dataflow_agent.logger import get_logger
from dataflow_agent.promptstemplates import PromptsTemplateGenerator

logger = get_logger(__name__)


class SummaryAgent:
    """Summary Agent for generating download subtasks from RAG content"""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.7,
        prompt_generator: Optional[PromptsTemplateGenerator] = None,
        max_download_subtasks: Optional[int] = None,
    ):
        """Initialize Summary Agent
        
        Args:
            model_name: LLM model name
            base_url: API base URL
            api_key: API key
            temperature: LLM temperature
            prompt_generator: Optional PromptsTemplateGenerator for loading prompts
            max_download_subtasks: Maximum number of download subtasks to generate
        """
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        self.prompt_generator = prompt_generator
        self.max_download_subtasks = max_download_subtasks

    async def generate_subtasks(
        self,
        objective: str,
        context: str,
        existing_subtasks: Optional[List[Dict[str, Any]]] = None,
        message: str = "",
    ) -> Dict[str, Any]:
        """Generate download subtasks from research context
        
        Args:
            objective: Research objective
            context: RAG context containing relevant information
            existing_subtasks: List of existing subtasks to avoid duplicates
            message: Additional user message
            
        Returns:
            Dict containing 'new_sub_tasks' and 'summary'
        """
        logger.info("--- Summary Agent: Generating download subtasks ---")

        existing_subtasks_str = (
            json.dumps(existing_subtasks, indent=2, ensure_ascii=False)
            if existing_subtasks
            else "[]"
        )

        # Use prompt generator if available
        if self.prompt_generator:
            try:
                system_prompt = self.prompt_generator.templates.get("system_prompt_for_summary_agent")
                task_prompt_template = self.prompt_generator.templates.get("task_prompt_for_summary_agent")
                if system_prompt and task_prompt_template:
                    human_prompt = task_prompt_template.format(
                        objective=objective,
                        message=message,
                        existing_subtasks=existing_subtasks_str,
                        context=context,
                    )
                else:
                    raise KeyError("Template not found")
            except Exception as e:
                logger.warning(f"Failed to load prompt, using default: {e}")
                system_prompt = self._get_default_system_prompt()
                human_prompt = self._get_default_task_prompt(
                    objective, message, existing_subtasks_str, context
                )
        else:
            system_prompt = self._get_default_system_prompt()
            human_prompt = self._get_default_task_prompt(
                objective, message, existing_subtasks_str, context
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        response = await self.llm.ainvoke(messages)
        logger.info(f"Summary agent raw response: {response.content}")

        try:
            clean_response = (
                response.content.strip()
                .replace("```json", "")
                .replace("```", "")
            )
            summary_plan = json.loads(clean_response)

            new_tasks = summary_plan.get("new_sub_tasks", [])
            summary_text = summary_plan.get("summary", "")

            # Apply download limit if specified
            if self.max_download_subtasks is not None:
                download_tasks = [t for t in new_tasks if t.get("type") == "download"]
                if len(download_tasks) > self.max_download_subtasks:
                    logger.info(
                        f"[Summary] Applying download limit: {len(download_tasks)} -> {self.max_download_subtasks}"
                    )
                    # Keep first max_download_subtasks download tasks
                    kept_downloads = 0
                    filtered_tasks = []
                    for task in new_tasks:
                        if task.get("type") == "download":
                            if kept_downloads >= self.max_download_subtasks:
                                continue
                            kept_downloads += 1
                        filtered_tasks.append(task)
                    new_tasks = filtered_tasks

            logger.info(f"[Summary] Generated {len(new_tasks)} new subtasks")
            if summary_text:
                logger.info(f"[Summary] Summary: {summary_text[:200]}...")

            return {
                "new_sub_tasks": new_tasks,
                "summary": summary_text,
            }
        except Exception as e:
            logger.info(f"Error parsing summary agent response: {e}\nRaw response: {response.content}")
            return {
                "new_sub_tasks": [],
                "summary": "Failed to generate summary",
            }

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """You are an AI analyst and task planner. Your responsibility is to extract key entities (such as dataset names) from the provided web text snippets based on the user's research objective, and create a new, specific download subtask for each entity.

Note: The text provided to you is the most relevant content filtered by RAG semantic search (if RAG is enabled), with each snippet annotated with source URL.
You will also receive a message from the task decomposer (a clear description of user needs), and your analysis should prioritize consistency with this message to avoid semantic drift.

Your output must be a JSON object containing:
1. `new_sub_tasks`: A list of subtasks. Each subtask dictionary must contain `type` (fixed as "download"), `objective`, and `search_keywords`.
2. `summary`: A string briefly summarizing the key information you found in the text.

If no relevant entities are found, return an empty `new_sub_tasks` list, but still provide a summary."""

    def _get_default_task_prompt(
        self, objective: str, message: str, existing_subtasks_str: str, context: str
    ) -> str:
        """Get default task prompt"""
        return f"""Research objective: '{objective}'
User description (message): '{message}'

Current download subtasks list (for reference):
{existing_subtasks_str}

Please analyze the following text snippets and generate specific download subtasks for each key dataset entity discovered:

{context[:18000]}"""
