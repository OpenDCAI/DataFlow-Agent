"""
Data Agents - 数据处理相关的Agent集合
"""

from dataflow_agent.agentroles.data_agents.web_collection_agent import (
    WebCollectionAgent,
    TaskDecomposerAgent,
    CategoryClassifierAgent,
    WebSearchPlannerAgent,
    DownloadDecisionAgent,
    DataMappingAgent,
    create_web_collection_agent,
    create_task_decomposer_agent,
    create_category_classifier_agent,
    create_web_search_planner_agent,
    create_download_decision_agent,
    create_data_mapping_agent,
    web_collection_analysis,
)

__all__ = [
    # Agents
    "WebCollectionAgent",
    "TaskDecomposerAgent",
    "CategoryClassifierAgent",
    "WebSearchPlannerAgent",
    "DownloadDecisionAgent",
    "DataMappingAgent",
    # Factory functions
    "create_web_collection_agent",
    "create_task_decomposer_agent",
    "create_category_classifier_agent",
    "create_web_search_planner_agent",
    "create_download_decision_agent",
    "create_data_mapping_agent",
    # Async functions
    "web_collection_analysis",
]
