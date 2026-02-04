"""
WebCrawler Orchestrator
~~~~~~~~~~~~~~~~~~~~~~~

从网页内容中提取代码块的爬取编排器。
适配自 loopai/agents/WebCrawler/utils/crawl_orchestrator.py
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dataflow_agent.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CrawledContent:
    """爬取内容的数据结构"""
    url: str
    title: str
    content: str
    code_blocks: Optional[List[Dict[str, str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ai_summary: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "code_blocks": self.code_blocks,
            "metadata": self.metadata,
            "ai_summary": self.ai_summary,
        }


def extract_code_blocks_from_markdown(text: str) -> List[Dict[str, str]]:
    """
    从 Markdown 文本中提取代码块
    
    支持两种格式:
    1. 带语言标识的代码块: ```python ... ```
    2. 不带语言标识的代码块: ``` ... ```
    
    Returns:
        List[Dict[str, str]]: 每个字典包含 'language', 'code', 'length' 字段
    """
    code_blocks = []
    
    # 正则匹配 markdown 代码块
    # 匹配 ```language\n code \n``` 或 ```\n code \n```
    pattern = r'```(\w*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        language = match[0].strip() if match[0] else 'unknown'
        code = match[1].strip()
        
        # 过滤掉过短的代码块（可能是示例或占位符）
        if len(code) >= 20:  # 至少20字符
            code_blocks.append({
                'language': language,
                'code': code,
                'length': len(code)
            })
    
    # 如果没有找到，尝试匹配缩进代码块（4个空格或1个tab开头的连续行）
    if not code_blocks:
        lines = text.split('\n')
        current_block = []
        in_code_block = False
        
        for line in lines:
            # 检查是否是缩进的代码行
            if line.startswith('    ') or line.startswith('\t'):
                if not in_code_block:
                    in_code_block = True
                    current_block = []
                # 去掉缩进
                clean_line = line[4:] if line.startswith('    ') else line[1:]
                current_block.append(clean_line)
            else:
                if in_code_block and current_block:
                    code = '\n'.join(current_block).strip()
                    if len(code) >= 20:
                        code_blocks.append({
                            'language': 'unknown',
                            'code': code,
                            'length': len(code)
                        })
                    current_block = []
                in_code_block = False
        
        # 处理最后一个代码块
        if in_code_block and current_block:
            code = '\n'.join(current_block).strip()
            if len(code) >= 20:
                code_blocks.append({
                    'language': 'unknown',
                    'code': code,
                    'length': len(code)
                })
    
    return code_blocks


class WebCrawlerOrchestrator:
    """WebCrawler 爬取编排器，专门用于从网页提取代码块和技术内容"""
    
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        tavily_api_key: str = "",
        temperature: float = 0.7,
        output_dir: str = "./webcrawler_output",
        # 爬取策略参数
        num_queries: int = 5,
        crawl_depth: int = 3,
        max_links_per_page: int = 5,
        concurrent_pages: int = 3,
        # 内容过滤参数
        min_text_length: int = 500,
        min_code_length: int = 50,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.tavily_api_key = tavily_api_key
        self.temperature = temperature
        
        # 爬取策略配置
        self.num_queries = num_queries
        self.crawl_depth = crawl_depth
        self.max_links_per_page = max_links_per_page
        self.concurrent_pages = concurrent_pages
        
        # 内容过滤配置
        self.min_text_length = min_text_length
        self.min_code_length = min_code_length
        
        # 输出目录
        self.output_dir = Path(output_dir)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[WebCrawlerOrchestrator] 初始化完成")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Output dir: {self.run_dir}")
    
    async def generate_search_queries(self, task: str) -> List[str]:
        """使用 LLM 生成搜索查询"""
        prompt = f"""你是一个专业的网页信息搜寻助手。根据用户的任务描述，生成 {self.num_queries} 个搜索查询关键词，用于查找包含代码示例和技术教程的相关内容。

用户任务: {task}

你的任务:
1. 仔细分析用户需求
2. 识别出需要搜索的技术内容方向
3. 针对这些方向，生成能够找到相关高质量内容的搜索关键词

搜索关键词要求:
- 具体明确，适合搜索引擎检索
- 能够找到包含代码示例、教程、最佳实践的技术内容
- 覆盖不同的技术角度
- 优先使用英文关键词（技术内容英文资源更丰富）
- 关键词应该能够定位到详细的、可学习的内容
- 生成恰好 {self.num_queries} 个查询

请直接返回JSON格式（不要markdown代码块）:
{{
    "queries": ["query1", "query2", ..., "query{self.num_queries}"]
}}"""
        
        try:
            logger.info("  调用LLM生成搜索查询...")
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            
            content = response.content.strip()
            
            # 清理markdown代码块
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            
            data = json.loads(content)
            queries = data.get("queries", [])
            
            if not queries:
                logger.warning("LLM未返回查询，使用原始任务作为查询")
                return [task[:100]]
            
            logger.info(f"  成功生成 {len(queries)} 个查询")
            return queries
            
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON失败: {e}")
            return [task[:100]]
        except Exception as e:
            logger.error(f"生成搜索查询失败: {e}")
            return [task[:100]]
    
    def process_crawled_pages(
        self, 
        crawled_pages: List[Dict[str, Any]]
    ) -> List[CrawledContent]:
        """
        处理爬取的网页，提取代码块
        
        Args:
            crawled_pages: 爬取的网页列表，每个包含 source_url, text_content 等
            
        Returns:
            List[CrawledContent]: 处理后的内容列表
        """
        processed = []
        
        for i, page in enumerate(crawled_pages, 1):
            source_url = page.get('source_url', '')
            text_content = page.get('text_content', '')
            title = page.get('structured_content', {}).get('title', '') if isinstance(page.get('structured_content'), dict) else ''
            
            logger.info(f"[页面 {i}/{len(crawled_pages)}] 处理: {source_url}")
            logger.info(f"  内容长度: {len(text_content)} 字符")
            
            # 应用最小文本长度过滤
            if not text_content or len(text_content.strip()) < self.min_text_length:
                logger.info(f"  跳过: 内容长度 < {self.min_text_length} 字符")
                continue
            
            # 从 Markdown 内容中提取代码块
            code_blocks = extract_code_blocks_from_markdown(text_content)
            
            # 过滤过短的代码块
            if code_blocks:
                code_blocks = [
                    cb for cb in code_blocks 
                    if cb.get('length', 0) >= self.min_code_length
                ]
            
            if code_blocks:
                logger.info(f"  提取到 {len(code_blocks)} 个代码块")
            
            content = CrawledContent(
                url=source_url,
                title=title,
                content=text_content,
                code_blocks=code_blocks if code_blocks else None,
                metadata={
                    "extraction_method": page.get('extraction_method', 'unknown'),
                    "content_length": len(text_content),
                    "code_blocks_count": len(code_blocks) if code_blocks else 0
                }
            )
            
            processed.append(content)
        
        logger.info(f"处理完成: {len(processed)}/{len(crawled_pages)} 个有效页面")
        return processed
    
    def save_results(self, results: List[CrawledContent], filename: str = "crawled_results.jsonl"):
        """保存处理结果到 JSONL 文件"""
        output_path = self.run_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"结果已保存至: {output_path}")
        return str(output_path)
