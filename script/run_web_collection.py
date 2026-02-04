#!/usr/bin/env python3
"""
运行 Web Collection 工作流
~~~~~~~~~~~~~~~~~~~~~~~~~

使用方法:
    cd DataFlow-Agent
    python script/run_web_collection.py

或者直接指定参数:
    python script/run_web_collection.py --target "收集代码生成数据集" --category SFT
"""

from __future__ import annotations

import asyncio
import argparse
import os
import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dataflow_agent.states.web_collection_state import WebCollectionState, WebCollectionRequest
from dataflow_agent.workflow.wf_web_collection import create_web_collection_graph
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


async def run_workflow(
    target: str,
    category: str = "SFT",
    output_format: str = "alpaca",
    download_dir: str = "./web_collection_output",
    model: str = None,
    DF_API_URL: str = None,
    api_key: str = None,
    tavily_api_key: str = None,
    rag_api_url: str = None,
    rag_api_key: str = None,
    rag_embed_model: str = None,
    max_urls: int = 5,
    max_depth: int = 2,
    max_download_subtasks: int = 5,
    debug: bool = False,
) -> WebCollectionState:
    """
    运行 Web Collection 工作流
    
    Args:
        target: 收集目标描述
        category: 数据类别 (PT/SFT)
        output_format: 输出格式 (alpaca等)
        download_dir: 下载目录
        model: 使用的模型
        DF_API_URL: API URL
        api_key: API Key
        tavily_api_key: Tavily API Key
        rag_api_url: RAG API URL
        rag_api_key: RAG API Key
        rag_embed_model: RAG 嵌入模型
        max_urls: 最大URL数量
        max_depth: 最大爬取深度
        max_download_subtasks: 最大下载子任务数
        debug: 是否启用调试模式
        
    Returns:
        最终状态对象
    """
    # 从环境变量获取默认值
    model = model or os.getenv("CHAT_MODEL", "gpt-4o")
    DF_API_URL = DF_API_URL or os.getenv("DF_API_URL", "")
    api_key = api_key or os.getenv("DF_API_KEY", "") or os.getenv("DF_API_KEY", "")
    tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
    rag_api_url = rag_api_url or os.getenv("DF_API_URL", "")
    rag_api_key = rag_api_key or os.getenv("DF_API_KEY", "")
    rag_embed_model = rag_embed_model or os.getenv("RAG_EMB_MODEL", "text-embedding-3-large")
    
    # 验证必要参数
    if not api_key:
        print("错误: 请设置 DF_API_KEY 或 DF_API_KEY 环境变量！")
        print("示例: export DF_API_KEY=your_api_key")
        sys.exit(1)
    
    if not DF_API_URL:
        print("错误: 请设置 DF_API_URL 环境变量！")
        print("示例: export DF_API_URL=https://api.openai.com/v1")
        sys.exit(1)
    
    # 创建下载目录
    os.makedirs(download_dir, exist_ok=True)
    
    # 打印配置信息
    print("\n" + "=" * 60)
    print("Web Collection 工作流配置:")
    print("=" * 60)
    print(f"  目标: {target}")
    print(f"  数据类别: {category}")
    print(f"  输出格式: {output_format}")
    print(f"  下载目录: {download_dir}")
    print(f"  模型: {model}")
    print(f"  最大URL数: {max_urls}")
    print(f"  最大深度: {max_depth}")
    print(f"  最大下载任务: {max_download_subtasks}")
    print(f"  Tavily API: {'已配置' if tavily_api_key else '未配置'}")
    print(f"  RAG: {'已配置' if rag_api_url and rag_api_key else '未配置'}")
    print("=" * 60)
    
    # 创建请求
    request = WebCollectionRequest(
        target=target,
        category=category,
        output_format=output_format,
        download_dir=download_dir,
        model=model,
        DF_API_URL=DF_API_URL,
        api_key=api_key,
        tavily_api_key=tavily_api_key if tavily_api_key else None,
        rag_api_base_url=rag_api_url if rag_api_url else None,
        rag_api_key=rag_api_key if rag_api_key else None,
        rag_embed_model=rag_embed_model if rag_embed_model else None,
        max_urls=max_urls,
        max_depth=max_depth,
        max_download_subtasks=max_download_subtasks,
        debug=debug,
    )
    
    # 创建初始状态
    state = WebCollectionState(request=request)
    
    # 创建并编译图
    print("\n开始执行工作流...")
    print("-" * 60)
    
    builder = create_web_collection_graph()
    graph = builder.build()
    
    # 运行图
    try:
        final_state = await graph.ainvoke(state)
        
        # 打印结果
        print("\n" + "=" * 60)
        print("工作流执行完成！")
        print("=" * 60)
        
        # Handle both dict and WebCollectionState object
        if isinstance(final_state, dict):
            exception = final_state.get("exception", "")
            mapping_results = final_state.get("mapping_results", {})
            download_results = final_state.get("download_results", {})
        else:
            exception = getattr(final_state, "exception", "")
            mapping_results = getattr(final_state, "mapping_results", {})
            download_results = getattr(final_state, "download_results", {})
        
        if exception:
            print(f"警告: 执行过程中出现异常: {exception}")
        
        if mapping_results:
            output_path = mapping_results.get("output_file", "") or mapping_results.get("output_path", "")
            total_mapped = mapping_results.get("mapped_records", 0) or mapping_results.get("total_mapped", 0)
            print(f"  输出文件: {output_path}")
            print(f"  总记录数: {total_mapped}")
        else:
            print("  未生成输出数据")
        
        if download_results:
            completed = download_results.get("completed", 0)
            failed = download_results.get("failed", 0)
            total = download_results.get("total", 0)
            print(f"  下载统计: {completed}/{total} 成功, {failed} 失败")
        
        print("=" * 60)
        
        return final_state
        
    except Exception as e:
        print(f"\n错误: 工作流执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="运行 Web Collection 数据收集工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python script/run_web_collection.py --target "收集机器学习问答数据集"
  python script/run_web_collection.py --target "收集代码生成数据集" --category SFT
  python script/run_web_collection.py --target "收集Python文档" --category PT --download-dir ./my_data

环境变量:
  DF_API_URL   - LLM API URL (必需)
  DF_API_KEY   - LLM API Key (必需)
  CHAT_MODEL     - 模型名称 (默认: gpt-4o)
  TAVILY_API_KEY - Tavily 搜索 API Key (可选)
  RAG_API_URL    - RAG API URL (可选)
  RAG_API_KEY    - RAG API Key (可选)
  RAG_EMB_MODEL  - RAG 嵌入模型 (可选)
"""
    )
    
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="数据收集目标描述"
    )
    parser.add_argument(
        "--category", "-c",
        choices=["PT", "SFT"],
        default="SFT",
        help="数据类别: PT(预训练) 或 SFT(指令微调), 默认: SFT"
    )
    parser.add_argument(
        "--output-format", "-f",
        default="alpaca",
        help="输出格式, 默认: alpaca"
    )
    parser.add_argument(
        "--download-dir", "-d",
        default="./web_collection_output",
        help="下载目录, 默认: ./web_collection_output"
    )
    parser.add_argument(
        "--model", "-m",
        help="使用的模型, 默认从环境变量 CHAT_MODEL 获取"
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=10,
        help="最大URL数量, 默认: 10"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="最大爬取深度, 默认: 4"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=5,
        help="最大下载子任务数, 默认: 5"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    # 运行工作流
    asyncio.run(run_workflow(
        target=args.target,
        category=args.category,
        output_format=args.output_format,
        download_dir=args.download_dir,
        model=args.model,
        max_urls=args.max_urls,
        max_depth=args.max_depth,
        max_download_subtasks=args.max_tasks,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
