"""
Query Generator
~~~~~~~~~~~~~~~

搜索查询生成器，使用 LLM 生成多样化的搜索查询。

"""

import json
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dataflow_agent.logger import get_logger
from dataflow_agent.promptstemplates import PromptsTemplateGenerator

logger = get_logger(__name__)


class QueryGenerator:
    """Query Generator for generating diverse search queries"""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.7,
        prompt_generator: Optional[PromptsTemplateGenerator] = None,
    ):
        """Initialize Query Generator
        
        Args:
            model_name: LLM model name
            base_url: API base URL
            api_key: API key
            temperature: LLM temperature
            prompt_generator: Optional PromptsTemplateGenerator for loading prompts
        """
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        self.prompt_generator = prompt_generator

    async def generate_queries(
        self, objective: str, message: str = ""
    ) -> List[str]:
        """Generate search queries from user requirements
        
        Args:
            objective: Research objective
            message: Additional user message
            
        Returns:
            List of search query strings
        """
        logger.info("--- Query Generator ---")
        
        # Use prompt generator if available, otherwise use default prompt
        if self.prompt_generator:
            try:
                system_prompt = self.prompt_generator.templates.get("system_prompt_for_query_generator")
                task_prompt_template = self.prompt_generator.templates.get("task_prompt_for_query_generator")
                if system_prompt and task_prompt_template:
                    human_prompt = task_prompt_template.format(objective=objective, message=message)
                else:
                    raise KeyError("Template not found")
            except Exception as e:
                logger.warning(f"Failed to load prompt, using default: {e}")
                system_prompt = self._get_default_system_prompt()
                human_prompt = self._get_default_task_prompt(objective, message)
        else:
            system_prompt = self._get_default_system_prompt()
            human_prompt = self._get_default_task_prompt(objective, message)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        logger.info(f"Query generator raw response: {response.content}")

        try:
            clean_response = (
                response.content.strip().replace("```json", "").replace("```", "")
            )
            queries = json.loads(clean_response)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                logger.info(f"Generated {len(queries)} search queries: {queries}")
                return queries
            logger.info("Query generation format error, returning empty list")
            return []
        except Exception as e:
            logger.info(f"Error parsing query generation response: {e}\nRaw response: {response.content}")
            return []

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """You are a query generation expert. Your task is to generate diverse search queries based on user requirements. Generate 3-5 search queries that cover different aspects of the research objective.

**CRITICAL: Prioritize High-Quality Data Sources**

Your search queries should prioritize finding high-quality data sources, especially:
- **Forums and Community Platforms**: Reddit, Stack Overflow, GitHub Discussions, specialized forums
- **Resource Websites**: Dataset repositories, code repositories, documentation sites, tutorial sites
- **Platforms with Rich Content**: Sites that contain detailed discussions, Q&A pairs, code examples

**Query Strategy:**
1. Include platform-specific terms when relevant (e.g., "site:reddit.com", "site:github.com")
2. Use terms that target resource-rich sites (e.g., "dataset", "repository", "tutorial", "examples")
3. Focus on finding actual content sources, not just general information pages

IMPORTANT: All search queries MUST be in English.

Return only a JSON array of query strings in English, for example: ["query1", "query2", "query3"]."""

    def _get_default_task_prompt(self, objective: str, message: str) -> str:
        """Get default task prompt"""
        return f"""Research objective: '{objective}'

User message: {message}

Please generate 3-5 diverse search queries in English that will help gather comprehensive information about the research objective.

**PRIORITY: Target High-Quality Data Sources**

Your queries should prioritize finding:
1. **Forums and Community Platforms**: Include terms like "forum", "discussion", "reddit", "stackoverflow"
2. **Resource Websites**: Include terms like "dataset", "repository", "examples", "tutorial"
3. **Q&A Sites**: Include terms like "Q&A", "question answer", "FAQ"

Return only a JSON array of English query strings."""
