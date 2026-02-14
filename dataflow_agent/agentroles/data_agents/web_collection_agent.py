"""
WebCollectionAgent - Web数据收集代理

功能：
- 分析用户数据收集需求
- 规划网页探索和下载策略
- 评估数据质量并进行筛选
- 将数据转换为目标格式（如alpaca）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from dataflow_agent.states.web_collection_state import WebCollectionState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


class WebCollectionAgent(BaseAgent):
    """
    Web数据收集代理 - 继承自BaseAgent
    
    负责分析用户的数据收集需求，规划探索策略，
    并协调各个节点完成数据收集任务。
    """
    
    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs):
        """工厂方法 - 创建Agent实例"""
        return cls(tool_manager=tool_manager, **kwargs)
    
    @property
    def role_name(self) -> str:
        """角色名称"""
        return "web_collection"
    
    @property
    def system_prompt_template_name(self) -> str:
        """系统提示词模板名称"""
        return "system_prompt_for_web_collection"
    
    @property
    def task_prompt_template_name(self) -> str:
        """任务提示词模板名称"""
        return "task_prompt_for_web_collection"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取任务提示词参数
        
        Args:
            pre_tool_results: 前置工具执行结果
            
        Returns:
            提示词模板参数字典
        """
        return {
            'user_query': pre_tool_results.get('user_query', ''),
            'datasets_background': pre_tool_results.get('datasets_background', ''),
            'category': pre_tool_results.get('category', 'PT'),
            'research_summary': pre_tool_results.get('research_summary', ''),
            'urls_visited': pre_tool_results.get('urls_visited', []),
            'subtasks': pre_tool_results.get('subtasks', []),
        }
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """获取默认前置工具结果"""
        return {
            'user_query': '',
            'datasets_background': '',
            'category': 'PT',
            'research_summary': '',
            'urls_visited': [],
            'subtasks': [],
        }
    
    def update_state_result(
        self, 
        state: WebCollectionState, 
        result: Dict[str, Any], 
        pre_tool_results: Dict[str, Any]
    ):
        """
        自定义状态更新
        
        Args:
            state: 当前状态对象
            result: 执行结果
            pre_tool_results: 前置工具结果
        """
        # 更新特定字段
        if 'research_summary' in result:
            state.research_summary = result['research_summary']
        if 'subtasks' in result:
            state.subtasks = result['subtasks']
        
        # 调用父类方法
        super().update_state_result(state, result, pre_tool_results)


# ==================== 专用子Agent ====================

class TaskDecomposerAgent(BaseAgent):
    """任务分解代理 - 将用户输入分解为子任务"""
    
    @property
    def role_name(self) -> str:
        return "task_decomposer"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_task_decomposer"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_task_decomposer"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'user_query': pre_tool_results.get('user_query', ''),
            'objective': pre_tool_results.get('objective', ''),
        }


class CategoryClassifierAgent(BaseAgent):
    """分类代理 - 将任务分类为SFT或PT"""
    
    @property
    def role_name(self) -> str:
        return "category_classifier"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_category_classifier"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_category_classifier"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'user_query': pre_tool_results.get('user_query', ''),
            'objective': pre_tool_results.get('objective', ''),
        }


class WebSearchPlannerAgent(BaseAgent):
    """网页搜索规划代理 - 规划搜索策略和URL选择"""
    
    @property
    def role_name(self) -> str:
        return "web_search_planner"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_web_search_planner"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_web_search_planner"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'user_query': pre_tool_results.get('user_query', ''),
            'datasets_background': pre_tool_results.get('datasets_background', ''),
            'category': pre_tool_results.get('category', 'PT'),
            'urls_visited': pre_tool_results.get('urls_visited', []),
            'search_results': pre_tool_results.get('search_results', ''),
        }


class DownloadDecisionAgent(BaseAgent):
    """下载决策代理 - 决定下载方法和优先级"""
    
    @property
    def role_name(self) -> str:
        return "download_decision"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_download_decision"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_download_decision"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'url': pre_tool_results.get('url', ''),
            'page_content': pre_tool_results.get('page_content', ''),
            'available_methods': pre_tool_results.get('available_methods', ['huggingface', 'kaggle', 'web']),
        }


class DataMappingAgent(BaseAgent):
    """数据映射代理 - 将数据映射到目标格式"""
    
    @property
    def role_name(self) -> str:
        return "data_mapping"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_data_mapping"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_data_mapping"
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'source_data': pre_tool_results.get('source_data', ''),
            'source_schema': pre_tool_results.get('source_schema', ''),
            'target_format': pre_tool_results.get('target_format', 'alpaca'),
            'target_schema': pre_tool_results.get('target_schema', ''),
        }


# ==================== 工厂函数 ====================

async def web_collection_analysis(
    state: WebCollectionState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    use_agent: bool = False,
    **kwargs,
) -> WebCollectionState:
    """
    执行Web收集分析
    
    Args:
        state: 当前状态对象
        model_name: 模型名称
        tool_manager: 工具管理器
        temperature: 温度参数
        max_tokens: 最大token数
        use_agent: 是否使用agent模式
        **kwargs: 其他参数
        
    Returns:
        更新后的状态对象
    """
    agent = WebCollectionAgent(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await agent.execute(state, use_agent=use_agent, **kwargs)


def create_web_collection_agent(
    tool_manager: Optional[ToolManager] = None, 
    **kwargs
) -> WebCollectionAgent:
    """创建Web收集代理实例"""
    return WebCollectionAgent(tool_manager=tool_manager, **kwargs)


def create_task_decomposer_agent(
    tool_manager: Optional[ToolManager] = None, 
    **kwargs
) -> TaskDecomposerAgent:
    """创建任务分解代理实例"""
    return TaskDecomposerAgent(tool_manager=tool_manager, **kwargs)


def create_category_classifier_agent(
    tool_manager: Optional[ToolManager] = None, 
    **kwargs
) -> CategoryClassifierAgent:
    """创建分类代理实例"""
    return CategoryClassifierAgent(tool_manager=tool_manager, **kwargs)


def create_web_search_planner_agent(
    tool_manager: Optional[ToolManager] = None, 
    **kwargs
) -> WebSearchPlannerAgent:
    """创建网页搜索规划代理实例"""
    return WebSearchPlannerAgent(tool_manager=tool_manager, **kwargs)


def create_download_decision_agent(
    tool_manager: Optional[ToolManager] = None, 
    **kwargs
) -> DownloadDecisionAgent:
    """创建下载决策代理实例"""
    return DownloadDecisionAgent(tool_manager=tool_manager, **kwargs)


def create_data_mapping_agent(
    tool_manager: Optional[ToolManager] = None, 
    **kwargs
) -> DataMappingAgent:
    """创建数据映射代理实例"""
    return DataMappingAgent(tool_manager=tool_manager, **kwargs)
