"""
States module - 各种任务的状态定义
"""

from dataflow_agent.states.test_graph_state import TestGraphState, TestGraphRequest
from dataflow_agent.states.web_collection_state import WebCollectionState, WebCollectionRequest

__all__ = [
    "TestGraphState",
    "TestGraphRequest",
    "WebCollectionState",
    "WebCollectionRequest",
]
