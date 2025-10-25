from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_core.tools import Tool

from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import DFState
from dataflow_agent.utils import robust_parse_json
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow import get_logger

log = get_logger()

# 验证器类型定义：返回 (是否通过, 错误信息)
ValidatorFunc = Callable[[str, Dict[str, Any]], Tuple[bool, Optional[str]]]

class BaseAgent(ABC):
    """Agent基类 - 定义通用的agent执行模式"""
    
    def __init__(self, 
                 tool_manager: Optional[ToolManager] = None,
                 model_name: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 tool_mode: str = "auto",
                 react_mode: bool = False,
                 react_max_retries: int = 3):
        """
        初始化Agent
        
        Args:
            tool_manager: 工具管理器
            model_name: 模型名称
            temperature: 模型温度
            max_tokens: 最大token数
            tool_mode: 工具选择模式 ("auto", "required", "none")
            react_mode: 是否启用ReAct模式
            react_max_retries: ReAct模式最大重试次数
        """
        self.tool_manager = tool_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_mode = tool_mode
        self.react_mode = react_mode
        self.react_max_retries = react_max_retries

    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs) -> "BaseAgent":
        """
        工厂方法：保持所有 Agent 统一的创建入口。

        ----
        BaseAgent 的具体子类实例（cls）
        """
        return cls(tool_manager=tool_manager, **kwargs)
    
    @property
    @abstractmethod
    def role_name(self) -> str:
        """角色名称 - 子类必须实现"""
        pass
    
    @property
    @abstractmethod
    def system_prompt_template_name(self) -> str:
        """系统提示词模板名称 - 子类必须实现"""
        pass
    
    @property
    @abstractmethod
    def task_prompt_template_name(self) -> str:
        """任务提示词模板名称 - 子类必须实现"""
        pass
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取任务提示词参数 - 子类可重写
        
        Args:
            pre_tool_results: 前置工具结果
            
        Returns:
            提示词参数字典
        """
        return pre_tool_results
    
    def parse_result(self, content: str) -> Dict[str, Any]:
        """
        解析结果 - 子类可重写自定义解析逻辑
        
        Args:
            content: LLM输出内容
            
        Returns:
            解析后的结果
        """
        try:
            parsed = robust_parse_json(content)
            log.info(f'content是什么？？{content}')
            log.info(f"{self.role_name} 结果解析成功")
            return parsed
        except ValueError as e:
            log.warning(f"JSON解析失败: {e}")
            return {"raw": content}
        except Exception as e:
            log.warning(f"解析过程出错: {e}")
            return {"raw": content}
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """获取默认前置工具结果 - 子类可重写"""
        return {}
    
    # ==================== ReAct 模式相关方法 ====================
    
    def get_react_validators(self) -> List[ValidatorFunc]:
        """
        获取ReAct模式的验证器列表 - 子类可重写添加自定义验证器
        
        Returns:
            验证器函数列表，每个函数签名为: (content: str, parsed_result: Dict) -> (bool, Optional[str])
            - bool: 是否通过验证
            - Optional[str]: 未通过时的错误提示信息
        """
        return [
            self._default_json_validator,
            # 子类可以添加更多验证器，例如：
            # self._check_required_fields,
            # self._check_data_format,
        ]
    
    @staticmethod
    def _default_json_validator(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        默认JSON格式验证器
        
        Args:
            content: LLM原始输出
            parsed_result: parse_result()的结果
            
        Returns:
            (是否通过, 错误信息)
        """
        # 如果解析结果中只有raw字段，说明JSON解析失败
        if "raw" in parsed_result and len(parsed_result) == 1:
            return False, (
                "你返回的内容不是有效的JSON格式。请确保返回纯JSON格式的数据，"
                "不要包含其他文字说明。正确的格式示例：\n"
                '{"key1": "value1", "key2": "value2"}'
            )
        
        # 检查是否为空字典
        if not parsed_result or (isinstance(parsed_result, dict) and not parsed_result):
            return False, "你返回的JSON为空，请提供完整的结果数据。"
        
        return True, None
    
    def _run_validators(self, content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        运行所有验证器
        
        Args:
            content: LLM原始输出
            parsed_result: 解析后的结果
            
        Returns:
            (是否全部通过, 错误信息列表)
        """
        validators = self.get_react_validators()
        errors = []
        
        for i, validator in enumerate(validators):
            try:
                passed, error_msg = validator(content, parsed_result)
                if not passed:
                    validator_name = getattr(validator, '__name__', f'validator_{i}')
                    log.warning(f"验证器 {validator_name} 未通过: {error_msg}")
                    if error_msg:
                        errors.append(error_msg)
            except Exception as e:
                log.exception(f"验证器执行出错: {e}")
                errors.append(f"验证过程出错: {str(e)}")
        
        return len(errors) == 0, errors
    
    async def process_react_mode(self, state: DFState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        ReAct模式处理 - 循环调用LLM直到验证通过
        
        Args:
            state: DFState实例
            pre_tool_results: 前置工具结果
            
        Returns:
            处理结果
        """
        log.info(f"执行 {self.role_name} ReAct模式 (最大重试: {self.react_max_retries})")
        
        # 构建初始消息
        messages = self.build_messages(state, pre_tool_results)
        llm = self.create_llm(state, bind_post_tools=False)
        
        for attempt in range(self.react_max_retries + 1):
            try:
                # 调用LLM
                log.info(f"ReAct尝试 {attempt + 1}/{self.react_max_retries + 1}")
                answer_msg = await llm.ainvoke(messages)
                answer_text = answer_msg.content
                log.info(f'LLM原始输出：{answer_text[:200]}...' if len(answer_text) > 200 else f'LLM原始输出：{answer_text}')
                
                # 解析结果
                parsed_result = self.parse_result(answer_text)
                
                # 运行验证器
                all_passed, errors = self._run_validators(answer_text, parsed_result)
                
                if all_passed:
                    log.info(f"✓ {self.role_name} ReAct验证通过，共尝试 {attempt + 1} 次")
                    return parsed_result
                
                # 验证未通过
                if attempt < self.react_max_retries:
                    # 构建反馈消息
                    feedback = self._build_validation_feedback(errors)
                    log.warning(f"✗ 验证未通过 (尝试 {attempt + 1}): {feedback}")
                    
                    # 添加LLM的回复和人类的反馈到消息列表
                    messages.append(AIMessage(content=answer_text))
                    messages.append(HumanMessage(content=feedback))
                else:
                    # 达到最大重试次数
                    log.error(f"✗ {self.role_name} ReAct达到最大重试次数，验证仍未通过")
                    return {
                        "error": "ReAct验证失败",
                        "attempts": attempt + 1,
                        "last_errors": errors,
                        "last_result": parsed_result
                    }
                    
            except Exception as e:
                log.exception(f"ReAct模式LLM调用失败 (尝试 {attempt + 1}): {e}")
                if attempt >= self.react_max_retries:
                    return {"error": f"LLM调用失败: {str(e)}"}
                # 继续重试
                continue
        
        # 理论上不会到这里
        return {"error": "ReAct处理异常终止"}
    
    def _build_validation_feedback(self, errors: List[str]) -> str:
        """
        构建验证失败的反馈消息
        
        Args:
            errors: 错误信息列表
            
        Returns:
            反馈消息
        """
        if not errors:
            return "输出格式有误，请按要求重新生成。"
        
        feedback_parts = ["你的输出存在以下问题，请修正后重新生成：\n"]
        for i, error in enumerate(errors, 1):
            feedback_parts.append(f"{i}. {error}")
        
        feedback_parts.append("\n请仔细检查并重新输出正确的结果。")
        return "\n".join(feedback_parts)
    
    # ==================== 原有方法 ====================
    
    async def execute_pre_tools(self, state: DFState) -> Dict[str, Any]:
        """执行前置工具"""
        log.info(f"开始执行 {self.role_name} 的前置工具...")
        
        if not self.tool_manager:
            log.info("未提供工具管理器，使用默认值")
            return self.get_default_pre_tool_results()
        
        results = await self.tool_manager.execute_pre_tools(self.role_name)
        
        # 设置默认值
        defaults = self.get_default_pre_tool_results()
        for key, default_value in defaults.items():
            if key not in results or results[key] is None:
                results[key] = default_value
                
        log.info(f"前置工具执行完成，获得: {list(results.keys())}")
        return results
    
    def build_messages(self, 
                      state: DFState, 
                      pre_tool_results: Dict[str, Any]) -> List[BaseMessage]:
        """构建消息列表"""
        log.info("构建提示词消息...")
        
        ptg = PromptsTemplateGenerator(state.request.language)
        sys_prompt = ptg.render(self.system_prompt_template_name)
        
        task_params = self.get_task_prompt_params(pre_tool_results)
        task_prompt = ptg.render(self.task_prompt_template_name, **task_params)
        # log.info(f"系统提示词: {sys_prompt}")
        log.info(f"任务提示词: {task_prompt}")
        
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=task_prompt),
        ]
        
        log.info("提示词消息构建完成")
        return messages
    
    def create_llm(self, state: DFState, bind_post_tools: bool = False) -> ChatOpenAI:
        """创建LLM实例"""
        actual_model = self.model_name or state.request.model
        log.info(f"创建LLM实例，模型: {actual_model}")
        
        llm = ChatOpenAI(
            openai_api_base=state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=actual_model,
            temperature=self.temperature,
            # max_tokens=self.max_tokens,
        )
        
        if bind_post_tools and self.tool_manager:
            post_tools = self.get_post_tools()
            if post_tools:
                llm = llm.bind_tools(post_tools, tool_choice=self.tool_mode)
                log.info(f"为LLM绑定了 {len(post_tools)} 个后置工具: {[t.name for t in post_tools]}")
        return llm
    
    def get_post_tools(self) -> List[Tool]:
        if not self.tool_manager:
            return []
        tools = self.tool_manager.get_post_tools(self.role_name)
        uniq, seen = [], set()
        for t in tools:
            if t.name not in seen:
                uniq.append(t)
                seen.add(t.name)
        return uniq
    
    async def process_simple_mode(self, state: DFState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """简单模式处理"""
        log.info(f"执行 {self.role_name} 简单模式...")
        
        messages = self.build_messages(state, pre_tool_results)
        llm = self.create_llm(state, bind_post_tools=False)
        
        try:
            answer_msg = await llm.ainvoke(messages)
            answer_text = answer_msg.content
            log.info(f'LLM原始输出：{answer_text}')
            log.info("LLM调用成功，开始解析结果")
        except Exception as e:
            log.exception("LLM调用失败: %s", e)
            return {"error": str(e)}
        
        return self.parse_result(answer_text)
    
    async def process_with_llm_for_graph(self, messages: List[BaseMessage], state: DFState) -> BaseMessage:
        llm = self.create_llm(state, bind_post_tools=True)
        try:
            response = await llm.ainvoke(messages)
            log.info(response)
            log.info(f"{self.role_name} 图模式LLM调用成功")
            return response
        except Exception as e:
            log.exception(f"{self.role_name} 图模式LLM调用失败: {e}")
            raise
    
    def has_tool_calls(self, message: BaseMessage) -> bool:
        """检查消息是否包含工具调用"""
        return hasattr(message, 'tool_calls') and bool(getattr(message, 'tool_calls', None))
    
    def update_state_result(self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """
        更新状态结果 - 子类可重写以自定义状态更新逻辑
        
        Args:
            state: 状态对象
            result: 处理结果
            pre_tool_results: 前置工具结果
        """
        # 默认行为：将结果存储到与角色名对应的属性中
        setattr(state, self.role_name.lower(), result)
        state.agent_results[self.role_name] = {
            "pre_tool_results": pre_tool_results,
            "post_tools": [t.name for t in self.get_post_tools()],
            "results": result
        }
    
    async def execute(self, state: DFState, use_agent: bool = False, **kwargs) -> DFState:
        """
        统一执行入口
        
        Args:
            state: DFState实例
            use_agent: 是否使用代理模式（图模式）
            **kwargs: 额外参数
            
        Returns:
            更新后的DFState
        """
        log.info(f"开始执行 {self.role_name} (ReAct模式: {self.react_mode}, 图模式: {use_agent})")
        
        try:
            # 1. 执行前置工具
            pre_tool_results = await self.execute_pre_tools(state)
            
            # 2. 检查是否有后置工具
            post_tools = self.get_post_tools()
            
            # 3. 根据模式和工具情况选择处理方式
            if use_agent and post_tools:
                # 图模式 - 需要外部图构建器处理
                log.info("图模式 - 需要外部图构建器处理，暂存必要数据")
                if not hasattr(state, 'temp_data'):
                    state.temp_data = {}
                state.temp_data['pre_tool_results'] = pre_tool_results
                state.temp_data[f'{self.role_name}_instance'] = self
                log.info(f"已暂存前置工具结果和 {self.role_name} 实例，后置工具: {[t.name for t in post_tools]}")
                
            elif self.react_mode:
                # ReAct模式 - 带验证的循环调用
                log.info("ReAct模式 - 带验证循环")
                result = await self.process_react_mode(state, pre_tool_results)
                self.update_state_result(state, result, pre_tool_results)
                log.info(f"{self.role_name} ReAct模式执行完成")
                
            else:
                # 简单模式 - 单次调用
                if use_agent and not post_tools:
                    log.info("图模式无可用后置工具，回退到简单模式")
                result = await self.process_simple_mode(state, pre_tool_results)
                self.update_state_result(state, result, pre_tool_results)
                log.info(f"{self.role_name} 简单模式执行完成")
            
        except Exception as e:
            log.exception(f"{self.role_name} 执行失败: {e}")
            error_result = {"error": str(e)}
            self.update_state_result(state, error_result, {})
            
        return state
    
    def create_assistant_node_func(self, state: DFState, pre_tool_results: Dict[str, Any]):
        async def assistant_node(graph_state):
            messages = graph_state.get("messages", [])
            if not messages:
                messages = self.build_messages(state, pre_tool_results)
                log.info(f"构建 {self.role_name} 初始消息，包含前置工具结果")

            response = await self.process_with_llm_for_graph(messages, state)

            if self.has_tool_calls(response):
                log.info(f"[create_assistant_node_func]: {self.role_name} LLM选择调用工具: ...")
                return {"messages": messages + [response]}
            else:
                log.info(f"[create_assistant_node_func]: {self.role_name} LLM本次未调用工具，解析最终结果")
                result = self.parse_result(response.content)
                # **这里同步了一次 agent_results**
                state.agent_results[self.role_name.lower()] = {
                    "pre_tool_results": pre_tool_results,
                    "post_tools": [t.name for t in self.get_post_tools()],
                    "results": result
                }
                return {
                    "messages": messages + [response],
                    self.role_name.lower(): result,
                    "finished": True
                }
        return assistant_node