#!/usr/bin/env python3
"""
Operator Assemble Line 示例入口脚本（非命令行工具）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
功能：
  手动定义算子序列 (Pipeline)，自动生成 Python 代码并执行。
  这是 "Operator Assemble Line" 功能的纯脚本版本。

用法：
  1) 修改 PIPELINE_STEPS 定义算子顺序和参数。
  2) 修改 INPUT_FILE 指定数据源。
  3) 运行: python run_dfa_op_assemble.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from dataflow_agent.state import DFRequest, DFState
# 引用 wf_df_op_usage.py 中定义的构图函数
from dataflow_agent.workflow.wf_df_op_usage import create_df_op_usage_graph
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

# ==============================================================================
# 1. 配置区域 (Configuration) - 请在此修改
# ==============================================================================

# [API 配置]
CHAT_API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1/")
API_KEY = os.getenv("DF_API_KEY", "")
MODEL = os.getenv("DF_MODEL", "gpt-4o")

# [环境配置]
SESSION_ID = "script_pipeline_001"
CACHE_DIR = str(PROJECT_ROOT / "dataflow_cache")
INPUT_FILE = str(PROJECT_ROOT / "tests/test.jsonl")  # 初始数据文件路径

# [Pipeline 定义]
# 这是一个列表，按顺序定义要执行的算子。
# 对应 op_assemble_line.py 中的 matched_ops_with_params
PIPELINE_STEPS = [
    {
        "op_name": "ReasoningAnswerGenerator",
        "params": {
            # __init__ 参数 (注意：在 wf_df_op_usage 中统一合并为 params)
            "prompt_template": "dataflow.prompts.reasoning.general.GeneralAnswerGeneratorPrompt",
            # run 参数
            "input_key": "raw_content",
            "output_key": "generated_cot"
        }
    },
    {
        "op_name": "ReasoningPseudoAnswerGenerator",  # 仅作示例，请替换为真实存在的算子
        "params": {
            "max_times": 3,
            "input_key": "generated_cot",
            "output_key_answer": "pseudo_answers",
            "output_key_answer_value": "pseudo_answer_value",
            "output_key_solutions": "pseudo_solutions",
            "output_key_correct_solution_example": "pseudo_correct_solution_example"
        }
    }
]

# ==============================================================================
# 2. 辅助函数
# ==============================================================================

def normalize_final_state(final_state_any: Any) -> dict:
    """将 graph.ainvoke 返回结果规范化为 dict"""
    if isinstance(final_state_any, dict):
        return final_state_any
    if hasattr(final_state_any, "model_dump"):
        return final_state_any.model_dump()
    if hasattr(final_state_any, "__dict__"):
        return final_state_any.__dict__
    return {}

def format_output(result_state: dict):
    """格式化打印执行结果"""
    print("\n" + "=" * 60)
    print("  Pipeline 执行报告")
    print("=" * 60)

    # 1. 生成信息
    gen_result = result_state.get("agent_results", {}).get("generate_pipeline", {})
    pipeline_info = result_state.get("pipeline_structure_code", {})
    
    print(f"\n[Generation]")
    print(f"Status: {gen_result.get('status', 'unknown')}")
    print(f"File Path: {pipeline_info.get('file_path', 'N/A')}")
    
    # 2. 代码预览
    code = pipeline_info.get("code", "")
    if code:
        print(f"\n[Code Preview - Top 20 lines]")
        print("-" * 30)
        print("\n".join(code.splitlines()[:20]))
        if len(code.splitlines()) > 20:
            print("... (see file for full code)")
        print("-" * 30)

    # 3. 执行结果
    exec_result = result_state.get("execution_result", {})
    status = exec_result.get("status")
    print(f"\n[Execution]")
    print(f"Status: {status}")
    print(f"Return Code: {exec_result.get('return_code')}")

    if status == "success":
        stdout = exec_result.get("stdout", "")
        # 只打印前 1000 个字符避免刷屏
        print(f"\n>> STDOUT (Preview):\n{stdout[:1000]}")
        if len(stdout) > 1000:
            print("... (truncated)")
    else:
        stderr = exec_result.get("stderr", "")
        print(f"\n>> STDERR:\n{stderr}")

    print("\n" + "=" * 60)

# ==============================================================================
# 3. 主逻辑
# ==============================================================================

async def main():
    # 0. 检查 Key
    if not API_KEY:
        log.warning("DF_API_KEY 未设置，部分涉及 LLM 的算子可能无法运行。")

    # 1. 构造 Request
    # 注意：wf_df_op_usage 强依赖 request.json_file 和 request.cache_dir
    req = DFRequest(
        language="zh",
        chat_api_url=CHAT_API_URL,
        api_key=API_KEY,
        model=MODEL,
        target="Script execution of manual pipeline",
        json_file=INPUT_FILE, 
        cache_dir=CACHE_DIR,
        session_id=SESSION_ID
    )

    # 2. 构造 State
    # 核心是将 PIPELINE_STEPS 传入 state.opname_and_params
    # 注意：wf_df_op_usage 期望的格式是 List[Dict]，包含 op_name 和 params
    # op_assemble_line.py UI 中区分了 init_params 和 run_params，
    # 但最终 build_pipeline_code_with_full_params 通常会处理合并，或者我们需要在这里预处理。
    # 根据 wf_df_op_usage.py 的逻辑，它直接读取 state.opname_and_params。
    
    # 预处理：确保 params 是扁平的或者符合 build_pipeline_code 的预期
    # 这里我们直接透传 PIPELINE_STEPS，假设用户在配置时已经填好了合并后的 params
    # 或者模仿 UI 的结构，保持 init/run 分离，取决于 pipe_tools 的实现。
    # 根据 wf_df_op_usage.py line 43: build_pipeline_code_with_full_params(opname_and_params=opname_and_params...)
    # 通常建议在 PIPELINE_STEPS 里直接写合并后的 "params"，或者根据实际工具调整。
    
    state = DFState(
        request=req,
        opname_and_params=PIPELINE_STEPS,
        messages=[]
    )

    log.info(f"正在构建图，包含 {len(PIPELINE_STEPS)} 个算子步骤...")
    
    # 3. 获取并编译图
    # 从 wf_df_op_usage.py 中获取 workflow builder
    builder = create_df_op_usage_graph()
    graph = builder.build()

    # 4. 执行
    log.info("开始执行 Workflow...")
    final_state_any = await graph.ainvoke(state)
    final_state = normalize_final_state(final_state_any)

    # 5. 输出结果
    format_output(final_state)

if __name__ == "__main__":
    asyncio.run(main())