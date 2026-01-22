from __future__ import annotations

import asyncio
import os

from dataflow_agent.state import DFRequest, PromptWritingState
from dataflow_agent.workflow.wf_pipeline_prompt import create_operator_prompt_writing_graph


# ===== Example config (edit here) =====
CHAT_API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1/")
MODEL = os.getenv("DF_MODEL", "gpt-4o")
LANGUAGE = "en"

TASK_DESCRIPTION = "Write a prompt for an operator that filters missing values"
OP_NAME = "PromptedFilter"

# 这两项在算子不拥有任何一个预置提示词时才需要提供，否则会仿照已有提示词生成
OUTPUT_FORMAT = ""  # e.g. "Return JSON with keys: ..."
ARGUMENTS = []      # e.g. ["min_len=10", "drop_na=true"]

# 缓存目录，用于存储测试数据和提示词
CACHE_DIR = "./pa_cache"
DELETE_TEST_FILES = True


def _safe_setattr(obj, key, value):
    """字段不存在就跳过，兼容不同版本 DFRequest/State"""
    if hasattr(obj, key):
        setattr(obj, key, value)


async def main() -> None:
    # 读取 key（推荐统一用 DF_API_KEY）
    api_key = os.getenv("DF_API_KEY", "")

    # 关键：桥接给 OpenAI SDK（很多 client 默认只认 OPENAI_API_KEY）
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key

    if not api_key:
        print("错误: 未设置 DF_API_KEY（或 OPENAI_API_KEY）。")
        print("示例: export DF_API_KEY=your_api_key")
        return

    # ===== 显式构造 DFRequest（老师要看的重点）=====
    req = DFRequest(
        language=LANGUAGE,
        chat_api_url=CHAT_API_URL,
        api_key=api_key,
        model=MODEL,
        target=TASK_DESCRIPTION,
    )
    _safe_setattr(req, "cache_dir", CACHE_DIR)
    _safe_setattr(req, "chat_api_key", api_key)  # 兼容旧字段

    # ===== 显式构造 State（把 State 需要的东西一眼写清楚）=====
    state = PromptWritingState(request=req, messages=[])

    # prompt-writing 专属字段
    state.prompt_op_name = OP_NAME
    state.prompt_args = ARGUMENTS
    state.prompt_output_format = OUTPUT_FORMAT

    # temp_data 里的约定项
    if not hasattr(state, "temp_data") or state.temp_data is None:
        state.temp_data = {}
    state.temp_data["pipeline_file_path"] = CACHE_DIR
    state.temp_data["cache_dir"] = CACHE_DIR
    state.temp_data["delete_test_files"] = DELETE_TEST_FILES

    # ===== build + run =====
    graph = create_operator_prompt_writing_graph().build()
    final_state_any = await graph.ainvoke(state)

    # 这里不强行假设返回类型，简单打印一下
    print("Prompt writing finished.")
    # 如果你想看最终结果字段（按你们 state 结构），可以自己打印：
    # print(final_state_any)


if __name__ == "__main__":
    asyncio.run(main())