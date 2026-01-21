#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from typing import Any

from dataflow_agent.state import DFRequest, DFState
from dataflow_agent.workflow.wf_pipeline_write import create_operator_write_graph
from dataflow_agent.utils import get_project_root

PROJDIR = get_project_root()

# ===== Example config (edit here) =====
CHAT_API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1/")
MODEL = os.getenv("DF_MODEL", "gpt-4o")
LANGUAGE = "en"

TARGET = "Create an operator that filters out missing values and keeps rows with non-empty fields."
CATEGORY = "Default"          # 兜底类别（若 classifier 未命中）
OUTPUT_PATH = ""              # e.g. "cache_local/my_operator.py"；空则不落盘
JSON_FILE = ""                # 空则使用项目自带 tests/test.jsonl

NEED_DEBUG = False
MAX_DEBUG_ROUNDS = 3

# LangGraph 默认 recursion_limit=25；need_debug 时可显式调大避免超限
RECURSION_LIMIT = 4 + 5 * MAX_DEBUG_ROUNDS + 5


def normalize_final_state(final_state_any: Any) -> dict:
    """将 graph.ainvoke 返回统一规范为 dict（兼容 dict / pydantic / 普通对象）"""
    if isinstance(final_state_any, dict):
        return final_state_any
    if hasattr(final_state_any, "model_dump"):
        return final_state_any.model_dump()
    if hasattr(final_state_any, "__dict__"):
        return final_state_any.__dict__
    raise TypeError(f"Unsupported final_state type: {type(final_state_any)}")

"""
Operator Write 示例入口脚本（非命令行工具）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用法：
  1) 修改本文件顶部的 Example config 常量（TARGET / NEED_DEBUG / MAX_DEBUG_ROUNDS / MODEL 等）
  2) 直接运行：python run_dfa_operator_write.py

说明：
  - main() 中显式构造 DFRequest / DFState
  - 运行 workflow graph
  - 打印结果摘要
"""

async def main() -> None:
    # ===== key 读取与桥接 =====
    api_key = os.getenv("DF_API_KEY", "")

    # 很多底层 client 默认只认 OPENAI_API_KEY
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key

    if not api_key:
        print("错误: 未设置 DF_API_KEY（或 OPENAI_API_KEY）。")
        print("示例: export DF_API_KEY=your_api_key")
        return

    # ===== 显式构造 request（老师要看的重点）=====
    req = DFRequest(
        language=LANGUAGE,
        chat_api_url=CHAT_API_URL,
        api_key=api_key,
        model=MODEL,
        target=TARGET,
        need_debug=bool(NEED_DEBUG),
        max_debug_rounds=int(MAX_DEBUG_ROUNDS),
        # 默认使用 dataflowagent 下的 10 条测试数据
        json_file=(JSON_FILE or f"{PROJDIR}/tests/test.jsonl"),
    )
    # 兼容旧字段（若 DFRequest 存在）
    if hasattr(req, "chat_api_key"):
        req.chat_api_key = api_key

    # ===== 显式构造 state =====
    state = DFState(request=req, messages=[])

    # temp_data 显式初始化
    if not hasattr(state, "temp_data") or state.temp_data is None:
        state.temp_data = {}

    if OUTPUT_PATH:
        state.temp_data["pipeline_file_path"] = OUTPUT_PATH
    if CATEGORY:
        state.temp_data["category"] = CATEGORY

    # 显式初始化调试轮次
    state.temp_data["round"] = 0

    graph = create_operator_write_graph().build()
    final_state_any = await graph.ainvoke(state, config={"recursion_limit": int(RECURSION_LIMIT)})
    final_state = normalize_final_state(final_state_any)

    # ---- 打印结果摘要 ----
    print("==== Match Operator Result ====")
    matched = final_state.get("matched_ops")
    if not matched:
        matched = (
            final_state.get("agent_results", {})
            .get("match_operator", {})
            .get("results", {})
            .get("match_operators", [])
        )
    print("Matched ops:", matched or [])

    print("\n==== Writer Result ====")
    code_str = (
        (final_state.get("temp_data", {}) or {}).get("pipeline_code", "")
        or final_state.get("draft_operator_code", "")
        or final_state.get("agent_results", {}).get("write_the_operator", {}).get("results", {}).get("code", "")
    )

    # 若落盘了，尝试从文件读取
    if not code_str:
        fp = (final_state.get("temp_data", {}) or {}).get("pipeline_file_path")
        if fp:
            from pathlib import Path
            p = Path(fp)
            try:
                if p.exists():
                    code_str = p.read_text(encoding="utf-8")
            except Exception:
                pass

    print(f"Code length: {len(code_str)}")
    if OUTPUT_PATH:
        print(f"Saved to: {OUTPUT_PATH}")
    else:
        preview = (code_str or "")[:1000]
        print("Code preview:\n", preview)

    # ---- Execution Result (instantiate) ----
    exec_res = final_state.get("execution_result", {}) or {}
    if not exec_res or ("success" not in exec_res):
        exec_res = final_state.get("agent_results", {}).get("operator_executor", {}).get("results", {}) or exec_res

    success = bool(exec_res.get("success"))
    print("\n==== Execution Result (instantiate) ====")
    print("Success:", success)
    if not success:
        stderr = (exec_res.get("stderr") or exec_res.get("traceback") or "")
        print("stderr preview:\n", (stderr or "")[:500])

    # ---- 调试实例化输出预览（来自 instantiate_operator_main_node） ----
    dbg = (final_state.get("temp_data") or {}).get("debug_runtime")
    if dbg:
        print("\n==== Debug Runtime Preview ==== ")
        print("input_key:", dbg.get("input_key"))
        ak = dbg.get("available_keys")
        if ak:
            print("available_keys:", ak)
        stdout_pv = (dbg.get("stdout") or "")[:1000]
        stderr_pv = (dbg.get("stderr") or "")[:1000]
        if stdout_pv:
            print("[debug stdout]\n", stdout_pv)
        if stderr_pv:
            print("[debug stderr]\n", stderr_pv)


if __name__ == "__main__":
    asyncio.run(main())
