#!/usr/bin/env python
# coding: utf-8

# # 📘 代理架构 2: 工具使用
# 
# 本脚本涵盖第二种，可以说是最具变革性的代理架构之一：**工具使用**。这种模式是连接大型语言模型推理能力与真实动态世界的桥梁。
# 
# 没有工具，LLM是一个封闭系统，受限于其训练数据中冻结的知识。它无法知道今天的天气、股票的当前价格或您公司数据库中订单的状态。通过赋予代理使用工具的能力，我们使其能够克服这一基本限制，允许它查询API、搜索数据库并访问实时信息，以提供不仅经过推理而且具有事实性、及时性和相关性的答案。

# ### 定义
# **工具使用**架构为LLM驱动的代理配备了调用外部函数或API（"工具"）的能力。代理自主决定何时用户的查询无法仅通过其内部知识回答，并确定应调用哪个工具来查找必要的信息。
# 
# ### 高级工作流程
# 
# 1. **接收查询：** 代理接收来自用户的请求。
# 2. **决策：** 代理分析查询及其可用工具。它决定是否需要工具来准确回答问题。
# 3. **行动：** 如果需要工具，代理会格式化对该工具的调用（例如，具有正确参数的特定函数）。
# 4. **观察：** 系统执行工具调用，结果（"观察"）返回给代理。
# 5. **综合：** 代理将工具的输出整合到其推理过程中，为用户生成最终的、有根据的答案。

# ## 阶段0：基础与设置

# 在构建我们的工具使用代理之前，我们需要设置我们的环境。这包括安装必要的库、导入我们的模块和配置我们的API密钥。

import os
import json

from typing import List, Annotated, TypedDict, Optional
from dotenv import load_dotenv

# LangChain组件
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import BaseModel, Field

# LangGraph组件
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

# 用于美观打印
from rich.console import Console
from rich.markdown import Markdown

# --- API密钥和追踪设置 ---
load_dotenv()

# 设置Phoenix追踪
tracer = None
phoenix_app = None
with open("phoenix_init_tool_use.log", "w", encoding="utf-8") as f:
    f.write("开始初始化Phoenix追踪...\n")
    
    # 注意：Phoenix服务器已改为外部启动（通过命令行: phoenix serve）
    try:
        import phoenix as px
        f.write("成功导入phoenix模块\n")
        
        # 使用新的OpenInference API进行追踪
        try:
            from phoenix.otel import register
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from openinference.instrumentation.openai import OpenAIInstrumentor
            f.write("成功导入新的Phoenix追踪API\n")
            
            # 获取当前文件名（不包含扩展名）作为项目名
            project_name = os.path.splitext(os.path.basename(__file__))[0]
            # 注册tracer并instrument LangChain（连接到外部Phoenix服务器）
            tracer_provider = register(project_name=project_name)
            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
            f.write("Phoenix LangChain追踪已通过OpenInference启用\n")
            
            OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
            f.write("Phoenix OpenAI追踪已通过OpenInference启用\n")
        except Exception as e:
            f.write(f"使用OpenInference API失败: {e}\n")
            import traceback
            traceback.print_exc(file=f)
        
        # 创建一个简单的标记，表示追踪已尝试初始化
        tracer = "enabled"
        f.write("Phoenix追踪初始化完成（使用外部Phoenix服务器）\n")
    except Exception as e:
        f.write(f"Phoenix 初始化失败: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        tracer = None

# 检查密钥是否已设置
for key in ["OPENAI_API_KEY", "TAVILY_API_KEY"]:
    if not os.environ.get(key):
        print(f"{key} 未找到。请创建.env文件并设置密钥。")

if tracer:
    print("环境变量已加载，Phoenix追踪设置已完成。")
else:
    print("环境变量已加载，但Phoenix追踪初始化失败。")


# ## 阶段1：定义代理的工具包

# 代理的能力取决于它可以访问的工具。在这个阶段，我们将定义并测试我们将提供给代理的特定工具：实时网络搜索。

# 初始化工具。我们可以设置最大result数以保持上下文简洁。
search_tool = TavilySearchResults(max_results=2)

# for代理提供清晰的工具名称and描述至关重要
search_tool.name = "web_search"
search_tool.description = "用于搜索互联网获取最新信息的工具，包括新闻、事件和时事。"

tools = [search_tool]

# 初始化控制台以进行漂亮打印
console = Console()

print(f"工具 '{search_tool.name}' 已创建，描述：'{search_tool.description}'")

# ## 阶段2：使用LangGraph构建工具使用代理

# 现在我们将构建代理工作流程。这包括让LLM意识到工具并创建图，允许它循环通过"思考-行动-观察"周期，这是工具使用的本质。

# 定义图状态
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

print("AgentState TypedDict已定义，用于管理对话历史。")

# 将工具绑定到LLM
llm = ChatOpenAI(model="Qwen/Qwen2.5-72B-Instruct", 
                 base_url=os.environ.get("OPENAI_API_BASE"),
                 temperature=0)

# 将工具绑定到LLM，使其具有工具意识
llm_with_tools = llm.bind_tools(tools)

print("LLM已与提供的工具绑定。")

# 定义代理节点
def agent_node(state: AgentState):
    """调用LLM决定下一步行动的主节点。"""
    console.print("--- 代理：思考中... ---")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ToolNode是LangGraph的预构建节点，用于执行工具
tool_node = ToolNode(tools)

print("Agent节点和Tool节点已定义。")

# 定义条件路由器
def router_function(state: AgentState) -> str:
    """检查代理的最后一条消息以决定下一步。"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        # 代理请求了工具调用
        console.print("--- 路由器：决定调用工具。 ---")
        return "call_tool"
    else:
        # 代理提供了最终答案
        console.print("--- 路由器：决定完成。 ---")
        return "__end__"

print("路由器函数已定义。")

# ## 阶段3：组装和运行工作流程

# 现在我们将所有组件连接在一起，形成一个完整的、可执行的图，并在一个强制代理使用其新网络搜索能力的查询上运行它。

# 构建图
graph_builder = StateGraph(AgentState)

# 添加节点
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("call_tool", tool_node)

# 设置入口点
graph_builder.set_entry_point("agent")

# 添加条件路由器
graph_builder.add_conditional_edges(
    "agent",
    router_function,
)

# 添加从工具节点返回代理的边以完成循环
graph_builder.add_edge("call_tool", "agent")

# 编译图
tool_agent_app = graph_builder.compile()

print("工具使用代理图编译成功！")

# --- Graphviz依赖检查 ---
try:
    import graphviz
    graphviz_installed = True
    print("✅ graphviz Python库已安装")
except ImportError:
    graphviz_installed = False
    print("❌ graphviz Python库未安装。如需生成PNG图像，请运行: pip install graphviz")

# 检查系统级graphviz是否可用
try:
    import subprocess
    subprocess.run(["dot", "-V"], capture_output=True, check=True)
    system_graphviz_available = True
    print("✅ 系统级graphviz (dot命令) 已安装")
except (subprocess.SubprocessError, FileNotFoundError):
    system_graphviz_available = False
    print("❌ 系统级graphviz (dot命令) 未安装。如需生成PNG图像，请访问 https://graphviz.org/download/ 下载安装")

# 可视化图 - 生成图结构文件
try:
    import os
    current_dir = os.getcwd()
    
    # 生成Mermaid格式
    mermaid_graph = tool_agent_app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(current_dir, "tool_agent_app_graph.mermaid")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_graph)
    print(f"图结构已保存为 {mermaid_path}")
    
    # 生成DOT格式
    dot_content = """digraph "Tool Use Agent Graph" {
    rankdir=TD;
    
    // 节点定义
    __start__ [shape=point];
    agent [label="agent", style=filled, fillcolor="#f2f0ff"];
    call_tool [label="call_tool", style=filled, fillcolor="#f2f0ff"];
    __end__ [label="__end__", shape=doublecircle, style=filled, fillcolor="#bfb6fc"];
    
    // 边定义
    __start__ -> agent;
    agent -> call_tool [label="需要工具"];
    agent -> __end__ [label="不需要工具"];
    call_tool -> agent;
}
"""
    dot_path = os.path.join(current_dir, "tool_agent_app_graph.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)
    print(f"图结构已保存为 {dot_path}")
    
    # 条件化生成PNG
    if graphviz_installed and system_graphviz_available:
        try:
            import graphviz
            g = graphviz.Source.from_file(dot_path)
            g.render(filename="tool_agent_app_graph", directory=current_dir, format="png", cleanup=True)
            print(f"图结构已保存为 PNG 图像: {os.path.join(current_dir, 'tool_agent_app_graph.png')}")
        except Exception as png_error:
            print(f"⚠️ 生成PNG图像时出错: {png_error}")
    else:
        print("ℹ️ graphviz依赖不完整，仅生成文本格式的图文件")
except Exception as e:
    print(f"图表可视化失败：{e}")

# 端到端执行
if __name__ == "__main__":
    console.print("\n🚀 启动工具使用工作流程，请求：'最近一周中国的热点新闻有哪些？'")
    
    # 运行代理
    result = tool_agent_app.invoke({
        "messages": [{"role": "user", "content": "最近一周中国的热点新闻有哪些？"}]
    })
    
    # 打印最终结果
    console.print("\n--- 最终结果 ---")
    console.print(result["messages"][-1].content)