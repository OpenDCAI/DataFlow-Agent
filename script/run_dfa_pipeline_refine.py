from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

# ===== Example config (edit here) =====
INPUT_JSON = ""  # optional: path to an input pipeline json
OUTPUT_JSON = "cache_local/pipeline_refine_result.json"  # empty string => print only
TARGET = "请将Pipeline调整为只包含3个节点，简化数据流"

LANGUAGE = "en"
CHAT_API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1/")
MODEL = os.getenv("DF_MODEL", "gpt-4o")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from dataflow_agent.state import DFRequest, DFState
from dataflow_agent.workflow.wf_pipeline_refine import create_pipeline_refine_graph

from typing import Any


def normalize_final_state(final_state_any: Any) -> dict:
    """将 graph.ainvoke 返回统一规范为 dict（兼容 dict / pydantic / 普通对象）"""
    if isinstance(final_state_any, dict):
        return final_state_any
    if hasattr(final_state_any, "model_dump"):
        return final_state_any.model_dump()
    if hasattr(final_state_any, "__dict__"):
        return final_state_any.__dict__
    raise TypeError(f"Unsupported final_state type: {type(final_state_any)}")


async def main() -> None:
    api_key = os.getenv("DF_API_KEY", "")
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key
    if not api_key:
        print("Error: DF_API_KEY environment variable not set.")
        return

    req = DFRequest(
        language=LANGUAGE,
        chat_api_url=CHAT_API_URL,
        api_key=api_key,
        model=MODEL,
        target=TARGET,
    )
    if hasattr(req, "chat_api_key"):
        setattr(req, "chat_api_key", api_key)  # backward compatibility

    state = DFState(request=req, messages=[])
    # ====== 3. 读取 pipeline 结构 ======
    if INPUT_JSON and Path(INPUT_JSON).exists():
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            state.pipeline_structure_code = json.load(f)
    elif not state.pipeline_structure_code:
        default_path = REPO_ROOT / "dataflow" / "dataflowagent" / "test_pipeline.json"
        with open(default_path, "r", encoding="utf-8") as f:
            state.pipeline_structure_code = json.load(f)
    # ====== 4. 构建并运行 workflow ======
    graph = create_pipeline_refine_graph().build()
    final_state_any = await graph.ainvoke(state)
    final_state = normalize_final_state(final_state_any)
    out_json = final_state.get("pipeline_structure_code", final_state)

    if OUTPUT_JSON:
        Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        print(f"Saved refined pipeline JSON to: {OUTPUT_JSON}")
    else:
        print(json.dumps(out_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())