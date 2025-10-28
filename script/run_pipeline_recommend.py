from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any

from IPython.display import Image, display

from dataflow_agent.state import DFRequest, DFState
from dataflow_agent.workflow.wf_pipeline_recommend_extract_json import (
    create_pipeline_graph,
)
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
    把 DFState 序列化写入 <项目根>/dataflow_agent/tmps/(session_id?)/final_state.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(final_state), f, ensure_ascii=False, indent=2)
    print(f"final_state 已保存到 {out_path}")


# ====================== 主函数 ====================== #
async def main() -> None:
    # -------- 基础路径与 session 处理 -------- #
    PROJECT_ROOT: Path = get_project_root()  # e.g. /mnt/DataFlow/lz/proj/dataflow-agent-kupasi
    TMPS_DIR: Path = PROJECT_ROOT / "dataflow_agent" / "tmps"

    session_id = base64.urlsafe_b64encode("username=liuzhou".encode()).decode()
    SESSION_DIR: Path = TMPS_DIR / session_id
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    # -------- 构造请求 DFRequest -------- #
    python_file_path = SESSION_DIR / "my_pipeline.py"

    req = DFRequest(
        language="en",
        chat_api_url="http://123.129.219.111:3000/v1/",
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model="gpt-4o",
        json_file=f"{PROJECT_ROOT}/tests/test.jsonl",
        target="给我随意符合逻辑的5个算子！",
        python_file_path=str(python_file_path),  # pipeline 的输出脚本路径
        need_debug= True,  # 是否需要 Debug
        max_debug_rounds= 3,
        session_id=session_id,
        cache_dir="dataflow_cache"
    )

    # -------- 初始化 DFState -------- #
    state = DFState(request=req, messages=[])
    state.temp_data["round"] = 0
    state.debug_mode = True

    # -------- 构建并运行流水线图 -------- #
    graph_builder = create_pipeline_graph()
    graph = graph_builder.build()

    # （可选）展示 mermaid 图
    try:
        png_image = graph.get_graph().draw_mermaid_png()
        display(Image(png_image))
        (SESSION_DIR / "graph.png").write_bytes(png_image)
        print(f"\n流水线图已保存到 {SESSION_DIR / 'graph.png'}")
    except Exception as e:
        print(f"生成 PNG 失败，请确保已正确安装 pygraphviz 和 Graphviz：{e}")

    # -------- 异步执行 -------- #
    final_state: DFState = await graph.ainvoke(state)

    # -------- 保存最终 State -------- #
    save_final_state_json(final_state=final_state, out_dir=SESSION_DIR)

    # -------- 输出执行 / 调试结果 -------- #
    if req.need_debug:
        if final_state.get("execution_result", {}).get("success"):
            print("\n================ 最终 Pipeline 执行成功 ================\n")
            print(f"================ 可通过 python {python_file_path} 处理你的完整数据！ ================")
            print(final_state["execution_result"]["stdout"])
        else:
            print("\n================== 调试失败，放弃 ==================\n")
            print(final_state.get("execution_result", {}))
    else:
        print(f"================== 组装完成，结果脚本位于 {python_file_path} ==================")


if __name__ == "__main__":
    asyncio.run(main())