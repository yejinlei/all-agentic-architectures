#!/usr/bin/env python
# coding: utf-8

# # 📘 代理架构 3: ReAct (Reason + Act)
# 
# 欢迎来到我们系列的第三个notebook。我们现在将探索**ReAct**，这是一个关键架构，它弥合了简单工具使用和复杂多步骤问题解决之间的差距。ReAct代表**推理+行动**，其核心创新在于它使代理能够动态地推理问题，根据其推理采取行动，观察结果，然后再次推理。
# 
# 这种模式将代理从静态的工具调用者转变为自适应的问题解决者。为了突出其强大功能，我们将首先构建一个**基本的单次工具使用代理**，并展示它在复杂任务上的局限性。然后，我们将构建一个完整的ReAct代理，并演示其迭代的`思考 -> 行动 -> 观察`循环如何使其在基本代理失败的地方取得成功。

# ### 定义
# **ReAct**架构是一种设计模式，其中代理将推理步骤与行动交织在一起。代理不是预先规划所有步骤，而是生成关于其下一步的想法，采取行动（如调用工具），观察结果，然后使用该新信息生成其下一个想法和行动。这创建了一个动态和自适应的循环。
# 
# ### 高级工作流程
# 
# 1. **接收目标：** 代理被赋予一个复杂的任务。
# 2. **思考（推理）：** 代理生成一个内部想法，例如：*"要回答这个问题，我首先需要找到信息X。"*
# 3. **行动：** 基于其想法，代理执行一个行动，通常是调用一个工具（例如，`search_api('X')`）。
# 4. **Observe:** 代理接收来自工具的结果。
# 5. **重复：** 代理将观察结果纳入其上下文并返回到步骤2，生成新的想法（例如，*"好的，现在我有了X，我需要用它来找到Y。"*）。这个循环持续到总体目标得到满足。
# 
# ### 何时使用/应用场景
# * **Multi-hop Question Answering:** 当回答一个问题需要按顺序查找多条信息时（例如，"制造iPhone的公司的CEO是谁？"）。
# * **Web Navigation & Research:** 代理可以搜索一个起点，阅读结果，然后根据所学内容决定新的搜索查询。
# * **Interactive Workflows:** 任何环境动态且无法预先知道完整解决方案路径的任务。
# 
# ### 优点和缺点
# * **优点：**
#  * **Adaptive & Dynamic:** 可以根据新信息即时调整其计划。
#  * **Handles Complexity:** 擅长需要链接多个依赖步骤的问题。
# * **缺点：**
#  * **Higher Latency & Cost:** 涉及多个顺序的LLM调用，使其比单次方法更慢、更昂贵。
#  * **Risk的Loops:** 引导不当的代理可能会陷入重复的、无效的思考和行动循环中。

# ## 阶段0：基础与设置
# 
# 我们将从标准设置过程开始：安装库并用于硅基流动平台、LangSmith和我们的Tavily网络搜索工具配置API密钥。

# ### 步骤0.1： 安装核心库
# 
# **我们将要做的：**
# 我们将为这个项目系列安装标准的库套件。

# In[1]:


# !pip install -q -U langchain-openai langchain langgraph rich python-dotenv tavily-python


# ### 步骤0.2： 导入库和设置密钥
# 
# **我们将要做的：**
# 我们将导入必要的模块并从`.env`文件加载我们的API密钥。
# 
# **需要执行的操作：** 在此目录中创建包含您的密钥的`.env`文件：
# ```
# OPENAI_API_KEY="your_siliconflow_api_key_here"
# LANGCHAIN_API_KEY="your_langsmith_api_key_here"
# TAVILY_API_KEY="your_tavily_api_key_here"
# ```

# In[2]:


import os 
import sys
import json
from typing import Annotated
from dotenv import load_dotenv

# PHOENIX追踪配置
import phoenix as px
from phoenix.otel import register
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
import logging

# 设置日志
logging.basicConfig(filename=f'phoenix_init_{os.path.basename(__file__)}.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain components 
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field

# LangGraph components 
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 用于美观打印 
from rich.console import Console
from rich.markdown import Markdown

# --- API密钥和追踪设置 ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - ReAct (SiliconFlow)"

# 检查密钥是否已设置
for key in ["OPENAI_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]:
    if not os.environ.get(key):
        print(f"{key} 未找到。请创建.env文件并设置密钥。")

print("环境变量已加载，追踪设置已完成。")

# 配置PHOENIX追踪
project_name = os.path.splitext(os.path.basename(__file__))[0]
try:
    logger.info(f"初始化Phoenix追踪，项目名: {project_name}")
    tracer_provider = register(project_name=project_name)
    LangchainInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info("Phoenix追踪初始化成功")
    print(f"PHOENIX追踪已配置，项目名: {project_name}")
except Exception as ex:
    logger.error(f"Phoenix追踪初始化失败: {ex}")
    print(f"警告: PHOENIX追踪初始化失败: {ex}")


# ## 阶段1： Basic Approach - 一个Single-Shot 工具使用r
# 
# 要理解为什么ReAct如此强大，我们必须首先看看没有它会发生什么。我们将构建一个"基本"代理，它可以使用工具，但只能使用一次。它将分析用户的查询，进行一次工具调用，然后尝试基于那一条信息制定最终答案。

# ### 步骤1.1： 构建基础代理
# 
# **我们将要做的：**
# 我们将定义与之前相同的工具和LLM，但我们将把它们连接到一个简单的线性图状态中。代理只有一次调用工具的机会，然后工作流程结束。没有循环。

# In[3]:


from typing import TypedDict

console = Console()

# def我们图的state
class AgentState(TypedDict):
 messages: Annotated[list[BaseMessage], add_messages]

# def工具andLLM
from langchain_core.tools import tool

# 自定义工具，可以接受字符串参数并自动转换为字典
@tool
def web_search_tool(query: str) -> str:
    """使用Tavily执行网络搜索并返回结果字符串。"""
    console.print(f"--- TOOL: Searching for '{query}'...")
    # 如果传入的是字符串，尝试解析为字典
    if isinstance(query, str):
        try:
            # 尝试直接解析JSON
            search_params = json.loads(query)
            if isinstance(search_params, dict) and 'query' in search_params:
                query = search_params['query']
        except (json.JSONDecodeError, KeyError):
            # 如果解析失败，保持原样
            pass
    
    result = search_tool.invoke({"query": query})
    return str(result)

# 原始工具用于实际搜索
search_tool = TavilySearchResults(max_results=2, name="web_search")
llm = ChatOpenAI(model="Qwen/Qwen2.5-72B-Instruct", base_url=os.environ.get("OPENAI_API_BASE"), temperature=0)
llm_with_tools = llm.bind_tools([web_search_tool])

# def基础代理的代理节点
def basic_agent_node(state: AgentState):
    console.print("--- 基本代理：思考中... ---")
    # 注意：我们提供系统提示以鼓励它一次工具调用后直接回答
    system_prompt = "你是一个有帮助的助手。你可以访问网络搜索工具。根据工具结果回答用户的问题。你必须在一次工具调用后提供最终答案。"
    messages = [("system", system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Define the basic, linear graph
basic_graph_builder = StateGraph(AgentState)
basic_graph_builder.add_node("agent", basic_agent_node)
basic_graph_builder.add_node("tools", ToolNode([web_search_tool]))

basic_graph_builder.set_entry_point("agent")
# in代理之后，它只能转到工具；in工具之后，它必须结束。
basic_graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})
basic_graph_builder.add_edge("tools", END)

basic_tool_agent_app = basic_graph_builder.compile()

print("基本单次工具使用代理编译成功。")

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

# 可视化基本工具使用代理图 - 生成图结构文件
try:
    import os
    current_dir = os.getcwd()
    
    # 生成Mermaid格式
    mermaid_graph = basic_tool_agent_app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(current_dir, "basic_tool_agent_app_graph.mermaid")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_graph)
    print(f"基本工具使用代理图结构已保存为 {mermaid_path}")
    
    # 生成DOT格式
    dot_content = """digraph "Basic Tool Agent Graph" {
    rankdir=TD;
    
    // 节点定义
    __start__ [shape=point];
    agent [label="agent", style=filled, fillcolor="#f2f0ff"];
    tools [label="tools", style=filled, fillcolor="#f2f0ff"];
    __end__ [label="__end__", shape=doublecircle, style=filled, fillcolor="#bfb6fc"];
    
    // 边定义
    __start__ -> agent;
    agent -> tools [label="需要工具"];
    agent -> __end__ [label="不需要工具"];
    tools -> __end__;
}
"""
    dot_path = os.path.join(current_dir, "basic_tool_agent_app_graph.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)
    print(f"基本工具使用代理图结构已保存为 {dot_path}")
    
    # 条件化生成PNG
    if graphviz_installed and system_graphviz_available:
        try:
            import graphviz
            g = graphviz.Source.from_file(dot_path)
            g.render(filename="basic_tool_agent_app_graph", directory=current_dir, format="png", cleanup=True)
            print(f"基本工具使用代理图结构已保存为 PNG 图像: {os.path.join(current_dir, 'basic_tool_agent_app_graph.png')}")
        except Exception as png_error:
            print(f"⚠️ 生成基本工具使用代理PNG图像时出错: {png_error}")
    else:
        print("ℹ️ graphviz依赖不完整，仅生成文本格式的图文件")
except Exception as e:
    print(f"基本工具使用代理图表可视化失败：{e}")


# ### 步骤1.2： 在多步骤问题上测试基本代理
# 
# **我们将要做的：**
# 现在我们将给基本代理一个需要多个依赖步骤才能解决的问题。这将暴露其根本弱点。

# In[4]:


multi_step_query = "创建科幻电影'沙丘'的公司的现任CEOis谁，该公司最新电影的预算is多少？"

console.print(f"[bold yellow]in多步骤查询上测试基础代理：[/bold yellow] '{multi_step_query}'\n")

basic_agent_output = basic_tool_agent_app.invoke({"messages": [("user", multi_step_query)]})

console.print("\n--- [bold red]基本代理的最终output[/bold red] ---")
console.print(Markdown(basic_agent_output['messages'][-1].content))


# **输出讨论：**
# 正如预期的那样，基本代理失败了。它的单次工具调用可能是对整个长查询的搜索。对于这样复杂的联合查询，搜索结果通常是混乱的，并且不会在一个地方包含所有必要的信息片段。
# 
# 代理的最终答案可能不完整、不正确，或声明它无法找到信息。它无法分解问题：
# 1. 找到制作'沙丘'的公司（传奇影业）。
# 2. 找到该公司的CEO（Joshua Grode）。
# 3. 找到该公司最新的电影及其预算。
# 
# 这个失败完美地说明了需要更动态的方法。代理需要一种方式来**响应**它在一个步骤中找到的信息，以指导下一步。

# ## 阶段2： 高级方法 - 实现ReAct
# 
# 现在，我们将构建真正的ReAct代理。核心区别在于图的结构：我们将引入一个循环，允许代理反复思考、行动和观察。

# ### 步骤2.1： 构建ReAct代理图
# 
# **我们将要做的：**
# 我们将定义节点和关键的路由函数，创建`思考 -> 行动`循环。关键的架构变化是将输出从`tool_node`路由*回*到`agent_node`的边，允许代理查看结果并决定其下一步。

# In[5]:


def react_agent_node(state: AgentState):
    console.print("--- REACT代理：思考中... ---")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# The ToolNode is the same as befor e
react_tool_node = ToolNode([web_search_tool])

# The router is also the same logic
def react_router(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        console.print("--- 路由器：决定调用工具。 ---")
        return "tools"
    console.print("--- 路由器：决定完成。 ---")
    return "__end__"

# 现in我们def带有关key循环的图
react_graph_builder = StateGraph(AgentState)
react_graph_builder.add_node("agent", react_agent_node)
react_graph_builder.add_node("tools", react_tool_node)

react_graph_builder.set_entry_point("agent")
react_graph_builder.add_conditional_edges("agent", react_router, {"tools": "tools", "__end__": "__end__"})

# 这is关key区别：边from工具return到代理
react_graph_builder.add_edge("tools", "agent")

react_agent_app = react_graph_builder.compile()
print("ReAct代理编译成功，带有推理循环。")

# 可视化ReAct代理图 - 生成图结构文件
try:
    import os
    current_dir = os.getcwd()
    
    # 生成Mermaid格式
    mermaid_graph = react_agent_app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(current_dir, "react_agent_app_graph.mermaid")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_graph)
    print(f"ReAct代理图结构已保存为 {mermaid_path}")
    
    # 生成DOT格式
    dot_content = """digraph "ReAct Agent Graph" {
    rankdir=TD;
    
    // 节点定义
    __start__ [shape=point];
    agent [label="agent", style=filled, fillcolor="#f2f0ff"];
    tools [label="tools", style=filled, fillcolor="#f2f0ff"];
    __end__ [label="__end__", shape=doublecircle, style=filled, fillcolor="#bfb6fc"];
    
    // 边定义
    __start__ -> agent;
    agent -> tools [label="需要工具"];
    agent -> __end__ [label="不需要工具"];
    tools -> agent;
}
"""
    dot_path = os.path.join(current_dir, "react_agent_app_graph.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)
    print(f"ReAct代理图结构已保存为 {dot_path}")
    
    # 条件化生成PNG
    if graphviz_installed and system_graphviz_available:
        try:
            import graphviz
            g = graphviz.Source.from_file(dot_path)
            g.render(filename="react_agent_app_graph", directory=current_dir, format="png", cleanup=True)
            print(f"ReAct代理图结构已保存为 PNG 图像: {os.path.join(current_dir, 'react_agent_app_graph.png')}")
        except Exception as png_error:
            print(f"⚠️ 生成ReAct代理PNG图像时出错: {png_error}")
    else:
        print("ℹ️ graphviz依赖不完整，仅生成文本格式的ReAct代理图文件")
except Exception as e:
    print(f"ReAct代理图表可视化失败：{e}")


# ## 阶段3： 正面比较
# 
# 现在我们将使用新的ReAct代理运行相同的复杂查询，并观察其过程和最终输出的差异。

# ### 步骤3.1： 在多步骤问题上测试ReAct代理
# 
# **我们将要做的：**
# 我们将使用相同的多步骤查询调用ReAct代理，并流式传输输出以查看其迭代推理过程的核心。

# In[6]:


console.print(f"[bold green]in相同的多步骤查询上测试ReAct代理：[/bold green] '{multi_step_query}'\n")

final_react_output = None
for chunk in react_agent_app.stream({"messages": [("user", multi_step_query)]}, stream_mode="values"):
    final_react_output = chunk
    console.print(f"--- [bold purple]当前state[/bold purple] ---")
    chunk['messages'][-1].pretty_print()
    console.print("\n")

console.print("\n--- [bold green]ReAct代理的最终output[/bold green] ---")
console.print(Markdown(final_react_output['messages'][-1].content))


# **输出讨论：**
# 成功！执行跟踪显示了一个完全不同且更智能的过程。您可以看到代理的逐步推理：
# 1. **想法1：** 它首先推理需要识别'沙丘'的制作公司。
# 2. **行动1：** 它使用类似"沙丘电影的制作公司"的查询调用`web_search`工具。
# 3. **观察1：** 它接收到结果："传奇影业"。
# 4. **想法2：** 现在，结合新信息，它推理出需要传奇影业的CEO。
# 5. **行动2：** 它再次使用类似"传奇影业的CEO"的查询调用`web_search`。
# 6.。..依此类推，直到它收集到所有必要的信息片段。
# 7. **综合：** 最后，它将所有收集到的事实组装成一个完整而准确的答案。
# 
# 这清楚地展示了ReAct模式对于任何不是简单的单步查找任务的优越性。

# ## 阶段4：定量评估
# 
# 为了形式化比较，我们将使用LLM作为评判者来评分基本代理和ReAct代理的最终输出，评估它们解决任务的能力。

# In[7]:


class TaskEvaluation(BaseModel):
 """评估代理完成任务能力的模式。"""
 任务完成评分: int = Field(description="对代理是否成功完成用户请求的所有部分进行1-10评分。")
 推理质量评分: int = Field(description="对代理展示的逻辑流程and推理过程进行1-10评分。")
 理由: str = Field(description="评分的简要理由。")

def evaluate_agent_output(query: str, agent_output: dict):
    trace = "\n".join([f"{m.type}: {m.content}" for m in agent_output['messages']])
    prompt = f"""你是一名专业的AI代理评判员。请在1-10的等级上评估以下代理在给定任务上的表现。10分表示任务完美完成。1分表示完全失败。
    
    **用户的任务：**
    {query}
    
    **完整的代理对话跟踪：**
    ```
    {trace}
    ```
    
    请返回以下格式的评估结果：
    任务完成评分: X
    推理质量评分: Y
    理由: Z
    
    其中X和Y是1-10的整数，Z是简要理由。"""
    
    response = llm.invoke(prompt)
    
    # 手动解析LLM响应
    try:
        lines = response.content.split('\n')
        scores = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if '任务完成评分' in key:
                    scores['任务完成评分'] = int(value)
                elif '推理质量评分' in key:
                    scores['推理质量评分'] = int(value)
                elif '理由' in key:
                    scores['理由'] = value
        
        # 如果解析失败，使用默认值
        if len(scores) < 3:
            scores = {
                '任务完成评分': 5,
                '推理质量评分': 5,
                '理由': '解析失败，使用默认值'
            }
        
        return TaskEvaluation(**scores)
    except Exception as e:
        # 如果解析失败，返回默认值
        return TaskEvaluation(
            任务完成评分=5,
            推理质量评分=5,
            理由=f'解析错误: {str(e)}'
        )

console.print("--- 评估基本代理的output ---")
basic_agent_evaluation = evaluate_agent_output(multi_step_query, basic_agent_output)
console.print(basic_agent_evaluation.model_dump())

console.print("\n--- 评估ReAct代理的output ---")
react_agent_evaluation = evaluate_agent_output(multi_step_query, final_react_output)
console.print(react_agent_evaluation.model_dump())


# **输出讨论：**
# LLM作为评判者的定量评分使差异一目了然。 
# - **基本代理**获得了非常低的`任务完成评分`，因为它未能收集所有必需的信息。它的`推理质量评分`也很低，因为其过程存在缺陷且不完整。
# - 相比之下，**ReAct代理**获得了接近完美的分数。评判者认识到其迭代过程使其能够成功完成复杂任务的所有部分。
# 
# 这种正面比较和评估提供了ReAct架构价值的确凿证据。它是解锁代理处理需要动态适应的复杂多跳问题能力的关键。

# ## 总结
# 
# 在这个notebook中，我们不仅实现了**ReAct**架构，还展示了它相对于更基本的单次方法的明显优势。通过构建一个允许代理循环进行推理和行动的工作流程，我们使其能够解决原本难以处理的复杂多步骤问题。
# 
# 观察行动结果并使用该信息来指导下一步的能力是智能行为的基本组成部分。ReAct模式提供了一种简单而深刻有效的方式来将这种能力构建到我们的AI代理中，使它们在现实世界任务中更强大、更具适应性和更有用。
