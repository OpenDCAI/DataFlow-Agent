"""
Web Collection Integration Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

测试 Web Collection 工作流的集成

使用方法:
    cd DataFlow-Agent
    python -m dataflow_agent.web_collection.test_integration
"""

import os
import sys
import asyncio


def test_imports():
    """测试所有模块是否能正确导入"""
    print("=" * 60)
    print("测试模块导入...")
    print("=" * 60)
    
    try:
        # 测试 State 导入
        from dataflow_agent.states.web_collection_state import (
            WebCollectionState,
            WebCollectionRequest,
        )
        print("✓ WebCollectionState, WebCollectionRequest")
        
        # 测试 Agent 导入
        from dataflow_agent.agentroles.data_agents.web_collection_agent import (
            WebCollectionAgent,
            TaskDecomposerAgent,
            CategoryClassifierAgent,
            create_web_collection_agent,
        )
        print("✓ WebCollectionAgent, TaskDecomposerAgent, etc.")
        
        # 测试 Utils 导入
        from dataflow_agent.web_collection.utils import (
            CategoryClassifier,
            TaskDecomposer,
            ObtainQueryNormalizer,
            RAGManager,
        )
        print("✓ CategoryClassifier, TaskDecomposer, RAGManager")
        
        # 测试 Nodes 导入
        from dataflow_agent.web_collection.nodes import (
            websearch_node,
            download_node,
            postprocess_node,
            mapping_node,
        )
        print("✓ websearch_node, download_node, postprocess_node, mapping_node")
        
        # 测试 Downloaders 导入
        from dataflow_agent.web_collection.downloaders import (
            HuggingFaceDownloader,
            KaggleDownloader,
            WebDownloader,
        )
        print("✓ HuggingFaceDownloader, KaggleDownloader, WebDownloader")
        
        # 测试 Workflow 导入
        from dataflow_agent.workflow.wf_web_collection import (
            create_web_collection_graph,
            run_web_collection,
        )
        print("✓ create_web_collection_graph, run_web_collection")
        
        print("\n所有模块导入成功！")
        return True
        
    except ImportError as e:
        print(f"\n✗ 导入失败: {e}")
        return False


def test_state_creation():
    """测试 State 创建"""
    print("\n" + "=" * 60)
    print("测试 State 创建...")
    print("=" * 60)
    
    try:
        from dataflow_agent.states.web_collection_state import (
            WebCollectionState,
            WebCollectionRequest,
        )
        
        # 创建 Request
        request = WebCollectionRequest(
            target="收集机器学习问答数据集用于大模型微调",
            category="SFT",
            output_format="alpaca",
            download_dir="./test_output",
        )
        print(f"✓ Request 创建成功: target={request.target[:30]}...")
        
        # 创建 State
        state = WebCollectionState(request=request)
        print(f"✓ State 创建成功")
        
        # 测试 State 方法
        state.user_query = "测试查询"
        state.task_list = [
            {"task_name": "任务1"},
            {"task_name": "任务2"},
        ]
        
        assert state.get_current_task() == {"task_name": "任务1"}
        assert state.has_more_tasks() == True
        print(f"✓ State 方法测试通过")
        
        return True
        
    except Exception as e:
        print(f"\n✗ State 创建失败: {e}")
        return False


def test_workflow_builder():
    """测试工作流构建器"""
    print("\n" + "=" * 60)
    print("测试工作流构建器...")
    print("=" * 60)
    
    try:
        from dataflow_agent.workflow.wf_web_collection import create_web_collection_graph
        
        # 创建工作流构建器
        builder = create_web_collection_graph()
        print(f"✓ 工作流构建器创建成功")
        
        # 检查节点
        # Note: GenericGraphBuilder may have different interface
        print(f"✓ 工作流入口点: {builder.entry_point}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 工作流构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_templates():
    """测试 Prompt 模板"""
    print("\n" + "=" * 60)
    print("测试 Prompt 模板...")
    print("=" * 60)
    
    try:
        from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
        
        # 创建生成器
        ptg = PromptsTemplateGenerator(output_language="zh")
        
        # 测试模板是否存在
        templates_to_check = [
            "system_prompt_for_web_collection",
            "task_prompt_for_web_collection",
            "system_prompt_for_task_decomposer",
            "system_prompt_for_category_classifier",
        ]
        
        for template_name in templates_to_check:
            if template_name in ptg.templates:
                print(f"✓ 模板存在: {template_name}")
            else:
                print(f"⚠ 模板未找到: {template_name}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Prompt 模板测试失败: {e}")
        return False


async def test_simple_workflow():
    """测试简单工作流执行（不实际调用API）"""
    print("\n" + "=" * 60)
    print("测试工作流执行（模拟）...")
    print("=" * 60)
    
    try:
        from dataflow_agent.states.web_collection_state import (
            WebCollectionState,
            WebCollectionRequest,
        )
        from dataflow_agent.web_collection.nodes import postprocess_node, mapping_node
        
        # 创建测试状态
        request = WebCollectionRequest(
            target="测试任务",
            category="SFT",
            output_format="alpaca",
            download_dir="./test_output",
        )
        state = WebCollectionState(request=request)
        state.user_query = "测试任务"
        
        # 测试 postprocess_node（空数据）- 节点现在是 async 函数
        result_state = await postprocess_node(state)
        print(f"✓ postprocess_node 执行成功（空数据场景）")
        
        # 测试 mapping_node（空数据）- 节点现在是 async 函数
        result_state = await mapping_node(state)
        print(f"✓ mapping_node 执行成功（空数据场景）")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 工作流执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Web Collection 集成测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("State 创建", test_state_creation()))
    results.append(("工作流构建", test_workflow_builder()))
    results.append(("Prompt 模板", test_prompt_templates()))
    results.append(("工作流执行", asyncio.run(test_simple_workflow())))
    
    # 打印结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
