#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any

# ===== Example config (edit here) =====
LANGUAGE = "en"
CHAT_API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1/")
MODEL = os.getenv("DF_MODEL", "gpt-4o")
TARGET = "给我简单的过滤或者去重算子就好了,只需要2个算子"

NEED_DEBUG = True
MAX_DEBUG_ROUNDS = 5
CACHE_DIR = "dataflow_cache"

# JSON_FILE 依赖 PROJECT_ROOT，在 main() 内拼接
TEST_JSON_REL_PATH = "tests/test.jsonl"

# 可选：仅在 Notebook 环境展示图片
try:
    from IPython.display import Image, display  # type: ignore
except Exception:
    Image = None
    display = None

from dataflow_agent.state import DFRequest, DFState
from dataflow_agent.workflow.wf_pipeline_recommend_extract_json import create_pipeline_graph
from dataflow_agent.utils import get_project_root


# ====================== 通用工具函数 ====================== #
def to_serializable(obj: Any):
    """递归将对象转成可 JSON 序列化结构"""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    if hasattr(obj, "__dict__"):
        return to_serializable(obj.__dict__)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_final_state_json(final_state: dict, out_dir: Path, filename: str = "final_state.json") -> None:
    """
    直接把 final_state 用 json.dump 存到 <项目根>/dataflow_agent/tmps/(session_id?)/final_state.json
    遇到无法序列化的对象用 str 兜底。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_state, f, ensure_ascii=False, indent=2, default=str)
    print(f"final_state 已保存到 {out_path}")


def normalize_final_state(final_state_any: Any) -> dict:
    """将 graph.ainvoke 返回统一规范为 dict（兼容 dict / pydantic / 普通对象）"""
    if isinstance(final_state_any, dict):
        return final_state_any
    if hasattr(final_state_any, "model_dump"):
        return final_state_any.model_dump()
    if hasattr(final_state_any, "__dict__"):
        return final_state_any.__dict__
    raise TypeError(f"Unsupported final_state type: {type(final_state_any)}")


# ====================== 主函数 ====================== #
async def main() -> None:
    # -------- 基础路径与 session 处理 -------- #
    PROJECT_ROOT: Path = get_project_root()
    JSON_FILE = str(PROJECT_ROOT / TEST_JSON_REL_PATH)
    TMPS_DIR: Path = PROJECT_ROOT / "dataflow_agent" / "tmps"

    session_id = base64.urlsafe_b64encode("username=xxx".encode()).decode()
    SESSION_DIR: Path = TMPS_DIR / session_id
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    python_file_path = SESSION_DIR / "my_pipeline.py"

    api_key = os.getenv("DF_API_KEY", "")
    # 桥接给 OpenAI SDK（很多 client 默认只认 OPENAI_API_KEY）
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key

    if not api_key:
        print("错误: 未设置 DF_API_KEY（或 OPENAI_API_KEY）。")
        print("示例: export DF_API_KEY=your_api_key")
        return

    # -------- 构造请求 DFRequest -------- #
    req = DFRequest(
        language=LANGUAGE,
        chat_api_url=CHAT_API_URL,
        api_key=api_key,
        model=MODEL,
        json_file=JSON_FILE,
        target=TARGET,
        python_file_path=str(python_file_path),  # pipeline 的输出脚本路径
        need_debug=NEED_DEBUG,                   # 是否需要 Debug
        max_debug_rounds=MAX_DEBUG_ROUNDS,
        session_id=session_id,
        cache_dir=CACHE_DIR,
    )
    # 兼容旧字段
    if hasattr(req, "chat_api_key"):
        req.chat_api_key = api_key

    # -------- 初始化 DFState -------- #
    state = DFState(request=req, messages=[])
    if not hasattr(state, "temp_data") or state.temp_data is None:
        state.temp_data = {}
    state.temp_data["round"] = 0

    if hasattr(state, "debug_mode"):
        state.debug_mode = True

    # -------- 构建并运行流水线图 -------- #
    graph_builder = create_pipeline_graph()
    graph = graph_builder.build()

    # （可选）展示 mermaid 图
    try:
        png_image = graph.get_graph().draw_mermaid_png()
        if display is not None and Image is not None:
            display(Image(png_image))
        (SESSION_DIR / "graph.png").write_bytes(png_image)
        print(f"\n流水线图已保存到 {SESSION_DIR / 'graph.png'}")
    except Exception as e:
        print(f"生成 PNG 失败（可忽略）：{e}")

    # -------- 异步执行 -------- #
    final_state_any = await graph.ainvoke(state)
    final_state = normalize_final_state(final_state_any)

    # -------- 保存最终 State -------- #
    save_final_state_json(final_state=final_state, out_dir=SESSION_DIR)

    # -------- 输出执行 / 调试结果 -------- #
    if req.need_debug:
        exec_res = (final_state.get("execution_result") or {})
        if exec_res.get("success"):
            print("\n================ 最终 Pipeline 执行成功 ================\n")
            print(f"================ 可通过 python {python_file_path} 处理你的完整数据！ ================")
            print(exec_res.get("stdout", ""))
        else:
            print("\n================== 调试失败，放弃 ==================\n")
            print(exec_res)
    else:
        print(f"================== 组装完成，结果脚本位于 {python_file_path} ==================")


if __name__ == "__main__":
    asyncio.run(main())