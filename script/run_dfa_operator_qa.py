#!/usr/bin/env python3
"""
OperatorQA 示例入口脚本（非命令行工具）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用法：
  1) 修改本文件顶部的 Example config 常量（QUERY / INTERACTIVE / TOP_K / MODEL 等）
  2) 直接运行：python run_dfa_operator_qa.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataflow_agent.state import DFRequest, MainState
from dataflow_agent.workflow.wf_operator_qa import create_operator_qa_graph
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

# ===== Example config (edit here) =====
INTERACTIVE = True
QUERY = "我想清洗数据，应该用哪个算子？"

LANGUAGE = "zh"
SESSION_ID = "demo_operator_qa"
CACHE_DIR = "dataflow_cache"
TOP_K = 5

CHAT_API_URL = os.getenv("DF_API_URL", "http://123.129.219.111:3000/v1/")
API_KEY = os.getenv("DF_API_KEY", "")
MODEL = os.getenv("DF_MODEL", "gpt-4o")

OUTPUT_JSON = "cache_local/operator_qa_result.json"  # e.g. "cache_local/operator_qa_result.json"；空字符串表示不落盘


# 自定义JSON编码器（处理消息对象）
class MessageJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理消息对象等不可序列化类型"""
    def default(self, obj: Any) -> Any:
        # 处理消息对象
        if hasattr(obj, '__class__'):
            obj_class_name = obj.__class__.__name__
            # 匹配常见的消息对象类型
            if obj_class_name in ["SystemMessage", "HumanMessage", "AIMessage", "ChatMessage"]:
                return {
                    "type": obj_class_name,
                    "content": getattr(obj, "content", ""),
                    "role": getattr(obj, "role", ""),
                    "additional_kwargs": getattr(obj, "additional_kwargs", {}),
                    "metadata": getattr(obj, "metadata", {})
                }
        # 处理其他可转换为字典的对象
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        # 兜底：转为字符串
        return str(obj)


# 自定义消息对象转字典（单个/列表）
def message_to_dict(msg: Any) -> Dict[str, Any]:
    """将单个消息对象转换为可序列化的字典"""
    if isinstance(msg, (dict, str, int, float, bool, type(None))):
        return msg
    # 使用自定义编码器转换
    return MessageJSONEncoder().default(msg)

def messages_to_list(messages: Any) -> List[Dict[str, Any]]:
    """将消息列表转换为可序列化的字典列表"""
    if not isinstance(messages, list):
        return []
    return [message_to_dict(msg) for msg in messages]


def _safe_setattr(obj, key, value):
    """字段不存在就跳过，兼容不同版本 DFRequest"""
    if hasattr(obj, key):
        setattr(obj, key, value)

def normalize_final_state(final_state_any):
    """
    将 graph.ainvoke 的返回结果统一规范为 dict
    兼容：
      - dict
      - pydantic BaseModel (model_dump)
      - 普通对象 (__dict__)
    """
    if isinstance(final_state_any, dict):
        return final_state_any

    if hasattr(final_state_any, "model_dump"):
        return final_state_any.model_dump()

    if hasattr(final_state_any, "__dict__"):
        return final_state_any.__dict__

    raise TypeError(f"Unsupported final_state type: {type(final_state_any)}")

async def run_single_query(state: MainState, graph, query: str) -> Dict[str, Any]:
    """
    执行单次查询（复用 main() 中构造的 state/graph）
    Args:
        state: 主状态（包含 request/messages）
        graph: 已编译的 workflow graph
        query: 用户查询
    Returns:
        标准化结果 dict
    """
    # 复用同一个 state/graph；每次只更新 target
    log.info(f"正在处理查询: {query}")
    state.request.target = query

    try:
        #final_state = await graph.ainvoke(state)
        final_state_any = await graph.ainvoke(state)
        final_state_dict = normalize_final_state(final_state_any)

    except Exception as e:
        log.error(f"执行失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
        }

    # 提取结果
    # agent_result = final_state.get("agent_results", {}).get("operator_qa", {})
    agent_result = final_state_dict.get("agent_results", {}).get("operator_qa", {})

    results = agent_result.get("results", {})

    # ========== 修改：处理messages，转换为可序列化的列表 ==========
    raw_messages = final_state_dict.get("messages", [])
    serializable_messages = messages_to_list(raw_messages)

    return {
        "success": True,
        "query": query,
        "answer": results.get("answer", ""),
        "related_operators": results.get("related_operators", []),
        "code_snippet": results.get("code_snippet", ""),
        "follow_up_suggestions": results.get("follow_up_suggestions", []),
        "messages": serializable_messages,  # 使用转换后的消息列表
    }

async def interactive_mode(state: MainState, graph):
    """
    交互模式 - 多轮对话

    通过复用同一个 graph 和 state，实现真正的多轮对话。
    state.messages 会在多轮对话中累积，LLM 能看到完整的对话历史。
    """
    print("\n" + "=" * 60)
    print("  DataFlow 算子问答助手 (交互模式)")
    print("=" * 60)
    print("\n欢迎使用 DataFlow 算子问答助手！")
    print("你可以询问关于 DataFlow 算子的任何问题。")
    print("\n命令:")
    print("  - 输入问题进行查询")
    print("  - 输入 'exit' 或 'quit' 退出")
    print("  - 输入 'clear' 清除对话历史")
    print("  - 输入 'history' 查看对话历史")
    print("-" * 60 + "\n")

    # state / graph 已在 main() 中构造，这里只负责交互循环

    while True:
        try:
            # 获取用户输入
            query = input("\n🧑 你: ").strip()

            if not query:
                continue

            # 处理命令
            if query.lower() in ["exit", "quit", "q"]:
                print("\n👋 再见！")
                break

            if query.lower() == "clear":
                # 清除对话历史：重置 state.messages
                state.messages = []
                print("✅ 对话历史已清除")
                continue

            if query.lower() == "history":
                if not state.messages:
                    print("📝 对话历史为空")
                else:
                    print(f"\n📝 对话历史 ({len(state.messages)} 条消息):")
                    for i, msg in enumerate(state.messages):
                        role = "🧑 你" if msg.type == "human" else "🤖 助手" if msg.type == "ai" else f"[{msg.type}]"
                        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        print(f"  [{i+1}] {role}: {content}")
                continue

            # 更新查询
            state.request.target = query

            # 执行查询（复用同一个 state，messages 会累积）
            print("\n⏳ 正在思考...")
            try:
                # graph.ainvoke 返回的是字典，需要更新 state
                final_state_any = await graph.ainvoke(state)
                final_state_dict = normalize_final_state(final_state_any)

                # 更新 state 的 messages（用于下一轮对话）
                if "messages" in final_state_dict:
                    state.messages = final_state_dict["messages"]

                # 更新 agent_results
                if "agent_results" in final_state_dict:
                    state.agent_results = final_state_dict["agent_results"]

            except Exception as e:
                log.error(f"执行失败: {e}")
                print(f"\n❌ 查询失败: {e}")
                continue

            # 提取结果（从字典中获取）
            agent_result = final_state_dict.get("agent_results", {}).get("operator_qa", {})
            results = agent_result.get("results", {})

            if results:
                # 显示回答
                answer = results.get("answer", "")
                print(f"\n🤖 助手: {answer}")

                # 显示信息来源
                source = results.get("source_explanation", "")
                if source:
                    print(f"\n📌 信息来源: {source}")

                # 显示相关算子
                related_ops = results.get("related_operators", [])
                if related_ops:
                    print(f"\n📦 相关算子: {', '.join(related_ops)}")

                # 显示代码片段
                code_snippet = results.get("code_snippet", "")
                if code_snippet:
                    print(f"\n📄 代码片段:\n{code_snippet[:500]}...")

                # 显示后续建议
                suggestions = results.get("follow_up_suggestions", [])
                if suggestions:
                    print("\n💡 你可能还想问:")
                    for suggestion in suggestions[:3]:
                        print(f"   - {suggestion}")

                # 显示当前消息数量（调试用）
                log.debug(f"当前消息历史: {len(state.messages)} 条")
            else:
                print(f"\n❌ 未获取到有效结果")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            log.exception(f"发生错误: {e}")
            print(f"\n❌ 发生错误: {e}")


def format_result(result: Dict[str, Any]) -> str:
    """格式化输出结果"""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("  查询结果")
    lines.append("=" * 60)

    lines.append(f"\n📝 问题: {result.get('query', '')}")
    lines.append(f"\n💬 回答:\n{result.get('answer', '无回答')}")

    if result.get("related_operators"):
        lines.append(f"\n📦 相关算子: {', '.join(result['related_operators'])}")

    if result.get("code_snippet"):
        lines.append(f"\n📄 代码片段:\n{result['code_snippet']}")

    if result.get("follow_up_suggestions"):
        lines.append("\n💡 后续建议:")
        for s in result["follow_up_suggestions"]:
            lines.append(f"   - {s}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


async def main():
    """主函数"""
    # ===== 显式构造 DFRequest（示例入口的核心）=====
    req = DFRequest(
        language=LANGUAGE,
        chat_api_url=CHAT_API_URL,
        api_key=API_KEY,
        model=MODEL,
        target="",  # 每次查询前再写入
    )
    _safe_setattr(req, "chat_api_key", API_KEY)  # 兼容旧字段
    _safe_setattr(req, "top_k", TOP_K)
    _safe_setattr(req, "cache_dir", CACHE_DIR)
    _safe_setattr(req, "session_id", SESSION_ID)

    if not API_KEY:
        log.warning("DF_API_KEY 未设置，调用可能失败（示例脚本可继续运行）")

    state = MainState(request=req, messages=[])
    graph = create_operator_qa_graph().build()

    log.info(
        "OperatorQA config: model=%s url=%s session_id=%s cache_dir=%s top_k=%s",
        getattr(req, "model", ""),
        getattr(req, "chat_api_url", ""),
        getattr(req, "session_id", ""),
        getattr(req, "cache_dir", ""),
        getattr(req, "top_k", ""),
    )

    if INTERACTIVE:
        await interactive_mode(state, graph)
        return

    result = await run_single_query(state, graph, QUERY)
    
    # 使用自定义编码器写入JSON ==========
    if OUTPUT_JSON:
        Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            # 使用自定义编码器序列化
            json.dump(result, f, ensure_ascii=False, indent=2, cls=MessageJSONEncoder)
        print(f"✅ 结果已保存到: {OUTPUT_JSON}")
    else:
        print(format_result(result))


if __name__ == "__main__":
    asyncio.run(main())

