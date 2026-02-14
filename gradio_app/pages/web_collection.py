import os
import asyncio
import logging
from typing import Optional
import gradio as gr

from dataflow_agent.states.web_collection_state import WebCollectionState, WebCollectionRequest
from dataflow_agent.workflow.wf_web_collection import create_web_collection_graph


def create_web_collection():
    """子页面：网页数据采集与转换（基于 Web Collection 工作流）"""
    with gr.Blocks() as page:
        gr.Markdown("# 🌐 网页数据采集与转换")

        with gr.Row():
            # 左侧：输入区域
            with gr.Column():
                gr.Markdown("### 采集配置")
                target = gr.Textbox(
                    label="目标描述",
                    placeholder="例如：收集 Python 代码示例的数据集",
                    lines=3
                )
                category = gr.Dropdown(
                    label="数据类别",
                    choices=["PT", "SFT"],
                    value="SFT"
                )
                output_format = gr.Dropdown(
                    label="输出格式",
                    choices=["alpaca", "sharegpt"],
                    value="alpaca",
                    info="目标输出数据格式"
                )
                max_download_subtasks = gr.Number(
                    label="下载子任务上限",
                    value=5,
                    precision=0,
                    minimum=1,
                    info="每个子任务最多下载子任务数"
                )
                download_dir = gr.Textbox(
                    label="下载目录",
                    value="./web_collection_output",
                )
                language = gr.Dropdown(
                    label="提示词语言",
                    choices=["zh", "en"],
                    value="zh"
                )

                gr.Markdown("### LLM 配置")
                chat_api_url = gr.Textbox(
                    label="DF_API_URL",
                    value=os.getenv("DF_API_URL", "")
                )
                api_key = gr.Textbox(
                    label="DF_API_KEY",
                    value=os.getenv("DF_API_KEY", ""),
                    type="password"
                )
                model = gr.Textbox(
                    label="CHAT_MODEL",
                    value=os.getenv("CHAT_MODEL", "gpt-4o")
                )

                gr.Markdown("### 其他环境配置")
                hf_endpoint = gr.Textbox(
                    label="HF_ENDPOINT",
                    value=os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
                )
                kaggle_username = gr.Textbox(
                    label="KAGGLE_USERNAME",
                    value=os.getenv("KAGGLE_USERNAME", "")
                )
                kaggle_key = gr.Textbox(
                    label="KAGGLE_KEY",
                    value=os.getenv("KAGGLE_KEY", ""),
                    type="password"
                )
                tavily_api_key = gr.Textbox(
                    label="TAVILY_API_KEY",
                    value=os.getenv("TAVILY_API_KEY", ""),
                    type="password"
                )

                gr.Markdown("### RAG 配置")
                rag_embed_model = gr.Textbox(
                    label="RAG_EMB_MODEL",
                    value=os.getenv("RAG_EMB_MODEL", "text-embedding-3-large")
                )
                rag_api_url = gr.Textbox(
                    label="RAG_API_URL",
                    value=os.getenv("RAG_API_URL", "")
                )
                rag_api_key = gr.Textbox(
                    label="RAG_API_KEY",
                    value=os.getenv("RAG_API_KEY", ""),
                    type="password"
                )

                # 高级配置区域（可折叠）
                with gr.Accordion("⚙️ 高级配置", open=False):
                    gr.Markdown("### 搜索与爬取配置")
                    search_engine = gr.Dropdown(
                        label="搜索引擎",
                        choices=["tavily", "google", "bing", "duckduckgo"],
                        value="tavily",
                        info="选择用于搜索的引擎"
                    )
                    max_urls = gr.Slider(
                        label="最大 URL 数量",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=10,
                        info="单次搜索最大处理URL数量"
                    )
                    max_depth = gr.Slider(
                        label="最大爬取深度",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=2,
                        info="爬取最大深度"
                    )
                    enable_rag = gr.Checkbox(
                        label="启用 RAG 增强",
                        value=True,
                        info="是否启用 RAG 增强"
                    )
                    concurrent_pages = gr.Slider(
                        label="WebCrawler 并发爬取数",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=3,
                        info="WebCrawler 并行处理的页面数量"
                    )

                    gr.Markdown("### WebCrawler 配置")
                    enable_webcrawler = gr.Checkbox(
                        label="启用 WebCrawler",
                        value=True,
                        info="是否启用 WebCrawler 并行爬取（默认启用，agent 会根据任务描述自动评估）"
                    )
                    debug = gr.Checkbox(
                        label="调试模式",
                        value=False,
                        info="启用调试模式，输出更详细的日志"
                    )
                    disable_cache = gr.Checkbox(
                        label="禁用缓存",
                        value=True,
                        info="如果启用，将完全禁用 HuggingFace 和 Kaggle 的缓存，使用临时目录并在下载后自动清理"
                    )
                    temp_base_dir = gr.Textbox(
                        label="临时目录（可选）",
                        value="",
                        placeholder="留空则使用默认临时目录",
                        info="自定义临时目录路径，用于缓存和临时文件"
                    )

                submit_btn = gr.Button("开始网页采集与转换", variant="primary")

            # 右侧：输出区域
            with gr.Column():
                with gr.Tab("执行日志"):
                    output_log = gr.Textbox(label="日志", lines=18)
                with gr.Tab("结果摘要"):
                    output_json = gr.JSON(label="执行结果")

        async def run_pipeline(
            target_text: str,
            category_val: str,
            output_format_val: str,
            max_download_subtasks_val: float | None,
            download_dir_val: str,
            language_val: str,
            chat_api_url_val: str,
            api_key_val: str,
            model_val: str,
            hf_endpoint_val: str,
            kaggle_username_val: str,
            kaggle_key_val: str,
            tavily_api_key_val: str,
            rag_embed_model_val: str,
            rag_api_url_val: str,
            rag_api_key_val: str,
            # 高级配置参数
            search_engine_val: str,
            max_urls_val: int,
            max_depth_val: int,
            enable_rag_val: bool,
            concurrent_pages_val: int,
            enable_webcrawler_val: bool,
            debug_val: bool,
            disable_cache_val: bool,
            temp_base_dir_val: str,
        ):
            # 注入/覆盖运行所需的环境变量
            os.environ["DF_API_URL"] = chat_api_url_val or ""
            os.environ["DF_API_KEY"] = api_key_val or ""
            os.environ["CHAT_MODEL"] = model_val or ""
            os.environ["HF_ENDPOINT"] = hf_endpoint_val or ""
            os.environ["KAGGLE_USERNAME"] = kaggle_username_val or ""
            os.environ["KAGGLE_KEY"] = kaggle_key_val or ""
            os.environ["RAG_EMB_MODEL"] = rag_embed_model_val or ""
            os.environ["RAG_API_URL"] = rag_api_url_val or ""
            os.environ["RAG_API_KEY"] = rag_api_key_val or ""
            if tavily_api_key_val:
                os.environ["TAVILY_API_KEY"] = tavily_api_key_val
            else:
                os.environ.pop("TAVILY_API_KEY", None)

            # 设置高级配置相关环境变量
            if disable_cache_val:
                os.environ["DF_DISABLE_CACHE"] = "true"
            else:
                os.environ.pop("DF_DISABLE_CACHE", None)

            if temp_base_dir_val:
                os.environ["DF_TEMP_DIR"] = temp_base_dir_val
            else:
                os.environ.pop("DF_TEMP_DIR", None)

            # 规范化下载子任务上限
            max_download_subtasks_int: Optional[int] = None
            if max_download_subtasks_val is not None:
                try:
                    numeric = int(max_download_subtasks_val)
                    if numeric > 0:
                        max_download_subtasks_int = numeric
                except (TypeError, ValueError):
                    pass

            # 组装 WebCollectionRequest
            request = WebCollectionRequest(
                target=target_text,
                category=category_val,
                output_format=output_format_val,
                download_dir=download_dir_val,
                language=language_val,
                chat_api_url=chat_api_url_val,
                api_key=api_key_val,
                model=model_val,
                # 搜索配置
                search_engine=search_engine_val,
                max_urls=int(max_urls_val),
                max_depth=int(max_depth_val),
                max_download_subtasks=max_download_subtasks_int,
                # RAG 配置
                enable_rag=enable_rag_val,
                rag_embed_model=rag_embed_model_val or "",
                rag_api_base_url=rag_api_url_val or None,
                rag_api_key=rag_api_key_val or None,
                # 外部 API Keys
                tavily_api_key=tavily_api_key_val or None,
                kaggle_username=kaggle_username_val or None,
                kaggle_key=kaggle_key_val or None,
                # WebCrawler 配置
                enable_webcrawler=enable_webcrawler_val,
                webcrawler_concurrent_pages=int(concurrent_pages_val),
                # 处理配置
                debug=debug_val,
            )

            # 构建初始状态
            state = WebCollectionState(request=request)

            # 使用新版工作流图
            builder = create_web_collection_graph()
            graph = builder.build()

            header_lines = [
                "=" * 60,
                "开始执行 Web Collection 工作流",
                "=" * 60,
                f"目标: {request.target}",
                f"类别: {request.category}",
                f"输出格式: {request.output_format}",
                f"下载目录: {request.download_dir}",
                "\n【搜索与爬取配置】",
                f"  - 搜索引擎: {search_engine_val}",
                f"  - 最大 URL 数: {max_urls_val}",
                f"  - 最大爬取深度: {max_depth_val}",
                f"  - 下载子任务上限: {max_download_subtasks_int if max_download_subtasks_int is not None else '默认(5)'}",
                f"  - 启用 RAG: {'是' if enable_rag_val else '否'}",
                "\n【WebCrawler 配置】",
                f"  - 启用 WebCrawler: {'是' if enable_webcrawler_val else '否'}",
                f"  - 并发爬取数: {concurrent_pages_val}",
                f"  - 调试模式: {'是' if debug_val else '否'}",
                f"  - 禁用缓存: {'是' if disable_cache_val else '否'}",
                "=" * 60,
            ]

            log_lines: list[str] = header_lines.copy()
            log_queue: asyncio.Queue = asyncio.Queue()

            class DataflowLogFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
                    return record.name.startswith("dataflow_agent") or record.name.startswith("script")

            class GradioLogHandler(logging.Handler):
                def __init__(self, queue: asyncio.Queue[str]):
                    super().__init__(level=logging.INFO)
                    self.queue = queue
                    self.addFilter(DataflowLogFilter())
                    self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

                def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
                    try:
                        message = self.format(record)
                        loop = asyncio.get_running_loop()
                        loop.call_soon_threadsafe(self.queue.put_nowait, message)
                    except RuntimeError:
                        try:
                            self.queue.put_nowait(self.format(record))
                        except asyncio.QueueFull:
                            pass
                    except Exception:
                        pass

            handler = GradioLogHandler(log_queue)
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            original_level = root_logger.level
            if original_level == 0 or original_level > logging.INFO:
                root_logger.setLevel(logging.INFO)

            attached_loggers: set[logging.Logger] = {root_logger}

            def _attach_to_existing_loggers() -> None:
                logger_dict = logging.root.manager.loggerDict  # type: ignore[attr-defined]
                for name in list(logger_dict.keys()):
                    if isinstance(name, str) and (name.startswith("dataflow_agent") or name.startswith("script")):
                        logger_obj = logging.getLogger(name)
                        logger_obj.addHandler(handler)
                        attached_loggers.add(logger_obj)

                for name in ("dataflow_agent", "script"):
                    logger_obj = logging.getLogger(name)
                    logger_obj.addHandler(handler)
                    attached_loggers.add(logger_obj)

            _attach_to_existing_loggers()

            # 初始输出
            yield "\n".join(log_lines), gr.update(value=None)

            async def run_workflow():
                return await graph.ainvoke(state)

            workflow_task = asyncio.create_task(run_workflow())
            result_payload: Optional[dict] = None

            try:
                while True:
                    try:
                        message = await asyncio.wait_for(log_queue.get(), timeout=0.3)
                        log_lines.append(message)
                        yield "\n".join(log_lines), gr.update(value=result_payload)
                    except asyncio.TimeoutError:
                        if workflow_task.done():
                            break

                final_state = await workflow_task

                # 清空剩余日志
                while True:
                    try:
                        pending = log_queue.get_nowait()
                        log_lines.append(pending)
                    except asyncio.QueueEmpty:
                        break

                log_lines.append("流程执行完成！")

                # 从最终状态提取结果（兼容 dict 和 WebCollectionState 对象）
                if isinstance(final_state, dict):
                    exception = final_state.get("exception", "")
                    mapping_results = final_state.get("mapping_results", {})
                    download_results = final_state.get("download_results", {})
                    webcrawler_summary = final_state.get("webcrawler_summary", "")
                    webcrawler_sft_jsonl_path = final_state.get("webcrawler_sft_jsonl_path", "")
                    webcrawler_pt_jsonl_path = final_state.get("webcrawler_pt_jsonl_path", "")
                else:
                    exception = getattr(final_state, "exception", "")
                    mapping_results = getattr(final_state, "mapping_results", {})
                    download_results = getattr(final_state, "download_results", {})
                    webcrawler_summary = getattr(final_state, "webcrawler_summary", "")
                    webcrawler_sft_jsonl_path = getattr(final_state, "webcrawler_sft_jsonl_path", "")
                    webcrawler_pt_jsonl_path = getattr(final_state, "webcrawler_pt_jsonl_path", "")

                if exception:
                    log_lines.append(f"警告: 执行过程中出现异常: {exception}")

                result_payload = {
                    "download_dir": request.download_dir,
                    "category": request.category,
                    "output_format": request.output_format,
                    "language": request.language,
                    "chat_model": request.model,
                    "max_download_subtasks": request.max_download_subtasks,
                    "enable_webcrawler": request.enable_webcrawler,
                }

                if download_results:
                    completed = download_results.get("completed", 0)
                    failed = download_results.get("failed", 0)
                    total = download_results.get("total", 0)
                    result_payload["download_stats"] = f"{completed}/{total} 成功, {failed} 失败"

                if mapping_results:
                    output_path = mapping_results.get("output_file", "") or mapping_results.get("output_path", "")
                    total_mapped = mapping_results.get("mapped_records", 0) or mapping_results.get("total_mapped", 0)
                    result_payload["mapping_output_file"] = output_path
                    result_payload["mapping_total_records"] = total_mapped

                if webcrawler_sft_jsonl_path:
                    result_payload["webcrawler_sft_jsonl"] = webcrawler_sft_jsonl_path
                if webcrawler_pt_jsonl_path:
                    result_payload["webcrawler_pt_jsonl"] = webcrawler_pt_jsonl_path
                if webcrawler_summary:
                    result_payload["webcrawler_summary"] = webcrawler_summary

                yield "\n".join(log_lines), result_payload

            except Exception as exc:
                error_message = f"流程执行失败: {exc}"
                log_lines.append(error_message)
                while True:
                    try:
                        pending = log_queue.get_nowait()
                        log_lines.append(pending)
                    except asyncio.QueueEmpty:
                        break
                result_payload = {"error": str(exc)}
                yield "\n".join(log_lines), result_payload
                raise
            finally:
                for logger_obj in attached_loggers:
                    logger_obj.removeHandler(handler)
                handler.close()
                root_logger.setLevel(original_level)

        submit_btn.click(
            run_pipeline,
            inputs=[
                target,
                category,
                output_format,
                max_download_subtasks,
                download_dir,
                language,
                chat_api_url,
                api_key,
                model,
                hf_endpoint,
                kaggle_username,
                kaggle_key,
                tavily_api_key,
                rag_embed_model,
                rag_api_url,
                rag_api_key,
                # 高级配置参数
                search_engine,
                max_urls,
                max_depth,
                enable_rag,
                concurrent_pages,
                enable_webcrawler,
                debug,
                disable_cache,
                temp_base_dir,
            ],
            outputs=[output_log, output_json],
        )

    return page
