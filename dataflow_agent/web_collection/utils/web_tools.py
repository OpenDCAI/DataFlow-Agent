"""
Web Tools
~~~~~~~~~

网页搜索和读取工具集，包含：
- Tavily 搜索
- DuckDuckGo 搜索
- Jina Reader 网页爬取


"""

import os
import re
from typing import Dict, List, Any, Optional
import httpx
import asyncio

from dataflow_agent.logger import get_logger

logger = get_logger(__name__)

try:
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    DuckDuckGoSearchRun = None


class WebTools:
    """Web search and reading tools"""
    
    @staticmethod
    def _get_proxy() -> Optional[str]:
        """Get proxy from environment variables"""
        return (
            os.getenv("HTTP_PROXY") or 
            os.getenv("HTTPS_PROXY") or 
            os.getenv("ALL_PROXY") or 
            os.getenv("http_proxy") or 
            os.getenv("https_proxy") or 
            os.getenv("all_proxy") or
            None
        )
    
    @staticmethod
    def _create_httpx_client(timeout: float = 20.0) -> httpx.AsyncClient:
        """Create httpx client with proxy support"""
        proxy = WebTools._get_proxy()
        client_kwargs = {"timeout": timeout}
        if proxy:
            client_kwargs["proxy"] = proxy
            logger.info(f"[WebTools] Using proxy: {proxy}")
        return httpx.AsyncClient(**client_kwargs)
    
    @staticmethod
    async def search_web(query: str, search_engine: str = "tavily", tavily_api_key: str = None) -> str:
        """Search the web using specified search engine
        
        Args:
            query: Search query string
            search_engine: Search engine to use ("tavily", "jina", "duckduckgo")
            tavily_api_key: Optional Tavily API key
            
        Returns:
            Formatted search results as string
        """
        if isinstance(query, (list, tuple)):
            query = ", ".join([str(x) for x in query if x])
        elif not isinstance(query, str):
            query = str(query)

        logger.info(f"[WebSearch] Using {search_engine.upper()} to search: '{query}'")

        if search_engine.lower() == "jina":
            return await WebTools._jina_search(query)
        if search_engine.lower() == "duckduckgo":
            return await WebTools._duckduckgo_search(query)
        return await WebTools._tavily_search(query, tavily_api_key=tavily_api_key)

    @staticmethod
    async def _tavily_search(query: str, tavily_api_key: str = None) -> str:
        """Search using Tavily API"""
        if not tavily_api_key:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            logger.info("[Tavily] API Key not set, falling back to DuckDuckGo")
            return await WebTools._duckduckgo_search(query)

        try:
            async with WebTools._create_httpx_client(timeout=20.0) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "include_answer": False,
                        "include_raw_content": False,
                        "max_results": 30,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])

                if not results:
                    logger.info("[Tavily] No results, falling back to DuckDuckGo")
                    return await WebTools._duckduckgo_search(query)

                formatted = [
                    "\n".join([
                        f"标题: {item.get('title', '无标题')}",
                        f"URL: {item.get('url', '')}",
                        f"摘要: {item.get('content', '')}",
                        "---",
                    ])
                    for item in results
                ]
                logger.info(f"[Tavily] Search completed, found {len(results)} results")
                return "\n".join(formatted)
        except Exception as e:
            logger.info(f"[Tavily] Error: {e}, falling back to DuckDuckGo")
            return await WebTools._duckduckgo_search(query)

    @staticmethod
    async def _duckduckgo_search(query: str) -> str:
        """Search using DuckDuckGo"""
        if DuckDuckGoSearchRun is None:
            logger.info("[DuckDuckGo] Dependency missing, returning empty result")
            return ""
        try:
            search_tool = DuckDuckGoSearchRun()
            result_text = await asyncio.to_thread(search_tool.run, query)
            logger.info("[DuckDuckGo] Search completed")
            return result_text
        except Exception as e:
            logger.info(f"[DuckDuckGo] Search error: {e}")
            return ""

    @staticmethod
    async def _jina_search(query: str) -> str:
        """Search using Jina Search API"""
        try:
            from urllib.parse import quote
            encoded_query = quote(query)
            search_url = f"https://s.jina.ai/{encoded_query}"

            logger.info(f"[Jina Search] Searching: {query}")

            async with WebTools._create_httpx_client(timeout=30.0) as client:
                resp = await client.get(
                    search_url,
                    headers={
                        "Accept": "application/json",
                        "X-Return-Format": "markdown",
                    },
                )
                resp.raise_for_status()

                try:
                    data = resp.json()
                    if isinstance(data, dict) and "data" in data:
                        results = data.get("data", [])
                        if results:
                            formatted = []
                            for item in results[:10]:
                                title = item.get("title", "无标题")
                                url = item.get("url", "")
                                content = item.get("content", "") or item.get("description", "")
                                formatted.append(
                                    f"标题: {title}\nURL: {url}\n摘要: {content}\n---"
                                )
                            logger.info(f"[Jina Search] Search completed, found {len(formatted)} results")
                            return "\n".join(formatted)
                except Exception:
                    text_content = resp.text
                    if text_content:
                        logger.info("[Jina Search] Search completed (text mode)")
                        return text_content[:15000]

                logger.info("[Jina Search] No results, falling back to DuckDuckGo")
                return await WebTools._duckduckgo_search(query)

        except Exception as e:
            logger.info(f"[Jina Search] Error: {e}, falling back to DuckDuckGo")
            return await WebTools._duckduckgo_search(query)

    @staticmethod
    async def read_with_jina_reader(url: str) -> Dict[str, Any]:
        """Read webpage content using Jina Reader
        
        Args:
            url: URL to read
            
        Returns:
            Dict containing 'urls', 'text', and 'structured_content'
        """
        logger.info(f"[Jina Reader] Extracting webpage: {url}")
        try:
            jina_url = f"https://r.jina.ai/{url}"

            client_kwargs = {"timeout": 60.0, "follow_redirects": True}
            proxy = WebTools._get_proxy()
            if proxy:
                client_kwargs["proxy"] = proxy
            
            async with httpx.AsyncClient(**client_kwargs) as client:
                resp = await client.get(
                    jina_url,
                    headers={
                        "Accept": "text/plain",
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36"
                        ),
                    },
                )
                resp.raise_for_status()

                text_response = resp.text
                structured_content = WebTools._parse_jina_text_format(text_response, url)
                markdown_content = structured_content.get("markdown", "")

                warning = structured_content.get("warning", "")
                if warning:
                    logger.info(f"[Jina Reader] Warning: {warning}")
                    if (
                        "blocked" in warning.lower()
                        or "403" in warning
                        or "forbidden" in warning.lower()
                    ):
                        logger.info("[Jina Reader] Webpage blocked, cannot extract content")
                        return {
                            "urls": [],
                            "text": f"无法访问该页面: {warning}",
                            "structured_content": structured_content,
                        }

                urls = WebTools._extract_urls_from_markdown(markdown_content)

                logger.info(
                    f"[Jina Reader] Extraction successful: {len(markdown_content)} chars, {len(urls)} links"
                )

                return {
                    "urls": urls,
                    "text": markdown_content,
                    "structured_content": structured_content,
                    "title": structured_content.get("title", ""),
                }

        except httpx.HTTPStatusError as e:
            logger.info(f"[Jina Reader] HTTP error {e.response.status_code}: {e}")
            return {
                "urls": [],
                "text": f"HTTP错误: {e.response.status_code}",
                "structured_content": None,
            }
        except Exception as e:
            logger.info(f"[Jina Reader] Extraction failed: {e}")
            return {
                "urls": [],
                "text": f"Jina Reader 错误: {str(e)}",
                "structured_content": None,
            }

    @staticmethod
    def _parse_jina_text_format(text: str, original_url: str) -> Dict[str, Any]:
        """Parse Jina Reader text format"""
        structured = {
            "title": "",
            "url_source": original_url,
            "warning": "",
            "markdown": "",
            "url": original_url,
        }

        lines = text.split("\n")
        markdown_lines: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("Title:"):
                structured["title"] = line[6:].strip()
                i += 1
                continue

            if line.startswith("URL Source:"):
                structured["url_source"] = line[11:].strip()
                i += 1
                continue

            if line.startswith("Warning:"):
                structured["warning"] = line[8:].strip()
                i += 1
                continue

            if line == "Markdown Content:":
                i += 1
                while i < len(lines):
                    markdown_lines.append(lines[i])
                    i += 1
                break

            i += 1

        structured["markdown"] = "\n".join(markdown_lines).strip()
        return structured

    @staticmethod
    def _extract_urls_from_markdown(markdown_text: str) -> List[str]:
        """Extract URLs from markdown text"""
        urls: List[str] = []

        # Extract markdown links [text](url)
        markdown_link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        markdown_links = re.findall(markdown_link_pattern, markdown_text)
        for _text, url in markdown_links:
            if url and not url.startswith("#"):
                urls.append(url)

        # Extract plain URLs
        plain_url_pattern = (
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|"
            r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        plain_urls = re.findall(plain_url_pattern, markdown_text)
        urls.extend(plain_urls)

        # Remove duplicates and limit
        unique_urls = list(dict.fromkeys(urls))[:50]
        return unique_urls

    @staticmethod
    def extract_urls_from_search_results(search_results: str) -> List[str]:
        """Extract URLs from search results text
        
        Args:
            search_results: Search results text
            
        Returns:
            List of unique URLs
        """
        urls = []
        lines = search_results.split("\n")
        for line in lines:
            if line.startswith("URL:"):
                url = line[4:].strip()
                if url:
                    urls.append(url)
        return list(dict.fromkeys(urls))  # Remove duplicates
