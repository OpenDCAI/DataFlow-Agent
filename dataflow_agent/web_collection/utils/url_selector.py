"""
URL Selector
~~~~~~~~~~~~

URL 智能筛选器，使用 LLM 从候选 URL 中选择最相关的 top-k 个。
从老项目 loopai/agents/Obtainer/utils/url_selector.py 移植。
"""

import json
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dataflow_agent.logger import get_logger
from dataflow_agent.promptstemplates import PromptsTemplateGenerator

logger = get_logger(__name__)


class URLSelector:
    """URL Selector for selecting top-k most relevant URLs from webpage content"""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.3,
        prompt_generator: Optional[PromptsTemplateGenerator] = None,
    ):
        """Initialize URL Selector
        
        Args:
            model_name: LLM model name
            base_url: API base URL
            api_key: API key
            temperature: LLM temperature (lower for more consistent selection)
            prompt_generator: Optional PromptsTemplateGenerator for loading prompts
        """
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        self.prompt_generator = prompt_generator

    async def select_top_urls(
        self,
        research_objective: str,
        url_list: List[str],
        webpage_content: str,
        topk: int = 5,
    ) -> List[str]:
        """Select top-k most relevant URLs based on research objective
        
        Args:
            research_objective: Current research objective/subtask goal
            url_list: List of URLs found in the webpage
            webpage_content: Webpage content text (will be truncated to 8000 chars)
            topk: Number of top URLs to return
            
        Returns:
            List of top-k most relevant URLs
        """
        logger.info(f"--- URL Selector: Selecting top {topk} URLs from {len(url_list)} candidates ---")
        
        if not url_list:
            logger.info("No URLs to select from")
            return []
        
        # Truncate webpage content to 8000 characters
        truncated_content = webpage_content[:8000]
        if len(webpage_content) > 8000:
            logger.info(f"Webpage content truncated from {len(webpage_content)} to 8000 characters")
        
        # Format URL list for prompt
        url_list_str = "\n".join([f"{i+1}. {url}" for i, url in enumerate(url_list)])
        
        # Use prompt generator if available, otherwise use default prompt
        if self.prompt_generator:
            try:
                system_prompt = self.prompt_generator.templates.get("system_prompt_for_url_selector")
                task_prompt_template = self.prompt_generator.templates.get("task_prompt_for_url_selector")
                if system_prompt and task_prompt_template:
                    human_prompt = task_prompt_template.format(
                        research_objective=research_objective,
                        url_list=url_list_str,
                        webpage_content=truncated_content,
                        topk=topk
                    )
                else:
                    raise KeyError("Template not found")
            except Exception as e:
                logger.warning(f"Failed to load prompt, using default: {e}")
                system_prompt = self._get_default_system_prompt()
                human_prompt = self._get_default_task_prompt(
                    research_objective, url_list_str, truncated_content, topk
                )
        else:
            system_prompt = self._get_default_system_prompt()
            human_prompt = self._get_default_task_prompt(
                research_objective, url_list_str, truncated_content, topk
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            logger.info(f"URL selector raw response: {response.content}")

            clean_response = (
                response.content.strip().replace("```json", "").replace("```", "")
            )
            result = json.loads(clean_response)
            
            # Handle different response formats
            if isinstance(result, dict) and "urls" in result:
                selected_urls = result["urls"]
            elif isinstance(result, list):
                selected_urls = result
            else:
                logger.warning("URL selector response format error, returning empty list")
                return []
            
            # Validate URLs are in the original list
            valid_urls = [url for url in selected_urls if url in url_list]
            
            # Limit to topk
            selected_urls = valid_urls[:topk]
            
            logger.info(f"Selected {len(selected_urls)} URLs: {selected_urls}")
            return selected_urls
            
        except Exception as e:
            logger.error(f"Error in URL selection: {e}")
            # Fallback: return first topk URLs if LLM fails
            logger.info(f"Falling back to first {topk} URLs")
            return url_list[:topk]

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """You are an expert URL selector. Given a research objective, a list of candidate URLs extracted from a page, and the page content, pick the URLs most likely to contain data or information relevant to the objective.

Guidelines:
1) Favor links that look like resources (datasets, papers, code repos, forums threads, doc pages) rather than ads or navigation-only links.
2) Prefer authoritative or content-rich domains; down-rank obviously irrelevant domains.
3) Use the surrounding page content to judge relevance.
4) Return only links from the provided list.

Return JSON: {"urls": ["url1", "url2", ...]} with at most the requested top-k."""

    def _get_default_task_prompt(self, research_objective: str, url_list: str, webpage_content: str, topk: int) -> str:
        """Get default task prompt"""
        return f"""Research objective: {research_objective}

Candidate URLs (from current page):
{url_list}

Webpage content (truncated to 8000 chars):
{webpage_content}

Goal: Select up to {topk} URLs that are most likely to contain information or data relevant to the research objective.

Selection rules:
1) Prefer resource-rich links (datasets, papers, code repos, tutorials, docs, forum threads) over generic navigation/ads/login.
2) Use page context to judge relevance; avoid obviously off-topic domains.
3) Return only URLs from the provided list.

Return JSON: {{"urls": ["url1", "url2", ...]}} (length <= {topk}). If nothing is relevant, return an empty array."""
