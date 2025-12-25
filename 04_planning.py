#!/usr/bin/env python
import os
import sys
from typing import Annotated

from dotenv import load_dotenv



# coding: utf-8

# # 📘 代理架构 4: 规划
# 
# 在这个notebook中，我们探索**规划**架构。这种模式在代理的推理过程中引入了关键的前瞻性层。与ReAct模型中逐步响应信息不同，规划代理首先将复杂任务分解为一系列更小、可管理的子目标。它在采取任何行动*之前*创建完整的'作战计划'。
# 
# 这种主动方法为多步骤任务带来了结构、可预测性和效率。为了突出其优势，我们将直接比较**反应式代理(ReAct)**与我们的新**规划代理**. 我们将向两者提出一个需要在执行最终计算之前收集多条信息的任务，展示预先计算的计划如何导致更稳健和直接的解决方案。

# ### 定义
# **规划**架构涉及一个代理，它明确地将复杂目标分解为详细的子任务 *之前*开始执行。这个初始规划阶段的输出是一个具体的、逐步的计划，代理然后有条不紊地遵循该计划以达到解决方案。
# 
# ### 高级工作流程
# 
# 1. **接收目标：** 代理被赋予一个复杂任务。
# 2. **规划：** 专门的'规划器'组件分析目标并生成有序的子任务列表以实现它。例如：`["查找事实A", "查找事实B", "使用A和B计算C"]`.
# 3. **执行：** '执行器'组件接受计划并按顺序执行每个子任务，根据需要使用工具。
# 4. **综合：** 一旦计划中的所有步骤完成，最终组件将执行步骤的结果综合成一个连贯的最终答案。
# 
# ### 何时使用/应用场景
# * **多步骤工作流程：** 适用于任务操作序列已知且关键的任务，例如生成需要获取数据、处理数据然后总结的报告。
# * **项目管理助手：** 将"启动新功能"等大目标分解为子任务，分配给不同的团队。
# * **教育辅导：** 创建教学计划，从基础原理到高级应用教授学生特定概念。
# 
# ### 优点和缺点
# * **优点：**
#  * **Structured & Traceable:** entire 工作流程预先布置好，使代理的过程透明且易于调试。
#  * **高效：** 对于可预测的任务，可以比ReAct更高效，因为它避免了步骤之间不必要的推理循环。
# * **缺点：**
#  * **对变化脆弱：** 如果环境在执行期间意外变化，预制计划可能会失败。它不如ReAct代理具有适应性，后者可以在每一步后改变主意。

# ## 阶段0：基础与设置
# 
# 我们将从标准设置过程开始：安装库并用于硅基流动平台、LangSmith和我们的Tavily网络搜索工具配置API密钥。

# ### 步骤0.1： 安装核心库
# 
# **我们将要做的：**
# 我们将安装标准的库套件，包括更新的`langchain-tavily`包以解决弃用警告。

# In[1]:


# !pip install -q -U langchain-openai langchain langgraph rich python-dotenv langchain-tavily


# ### 步骤0.2： 导入库和设置密钥
# 
# **我们将要做的：**
# 我们将导入必要的模块并从`.env` 文件加载我们的API密钥。
# 
# **需要执行的操作：** 在当前目录创建一个`.env`文件并设置您的密钥:
# ```
# OPENAI_API_KEY="your_siliconflow_api_key_here"
# LANGCHAIN_API_KEY="your_langsmith_api_key_here"
# TAVILY_API_KEY="your_tavily_api_key_here"
# ```

# In[ ]:


import os
import re 
from typing import List, Annotated, TypedDict, Optional
 
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

from langchain_core.messages import BaseMessage, ToolMessage 
from pydantic import BaseModel, Field
 
from langchain_core.tools import tool 
from langchain_core.messages import SystemMessage
 
from langchain_tavily import TavilySearch

# LangGraph components 
from langgraph.graph import StateGraph, END
 
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from langgraph.prebuilt import ToolNode, tools_condition

# 用于美观打印 
from rich.console import Console

from rich.markdown import Markdown

# --- API密钥和追踪设置 ---
load_dotenv()




os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Planning (SiliconFlow)"
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


# 检查密钥是否已设置
for key in ["OPENAI_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]:
    if not os.environ.get(key):
        print(f"{key} 未找到。请创建.env文件并设置密钥。")

print("环境变量已加载，追踪设置已完成。")


# ## 阶段1： 基线 - 一个Reactive Agent (ReAct)
# 
# 要理解规划的价值，我们首先需要一个基准。我们将使用在上一个notebook中构建的ReAct代理。这个代理很聪明但目光短浅——它一次一步地找出自己的路径。

# ### 步骤1.1： 重建ReAct代理
# 
# **我们将要做的：**
# 我们将快速重建ReAct代理。它的核心特性是一个循环，在每次工具调用后，代理的输出被路由回自身，允许它重新评估并决定下一步行动，基于最新信息。

# In[ ]:


console = Console()

# 定义我们图的状态
class AgentState(TypedDict):
 messages: Annotated[list[BaseMessage], add_messages]

# 1. 从tavily包定义基础工具
tavily_search_tool = TavilySearch(max_results=2)

# 2. Fix: Simplified self-defined tool. 
# The invoke() method already returns a clean string, so we just pass it through.
@tool
def web_search(query: str) -> str:
 """使用Tavily执行网络搜索并返回结果字符串。"""
 console.print(f"--- TOOL: Searching for '{query}'...")
 result= tavily_search_tool.invoke(query)
 return result

# 3. 定义LLM并将其绑定到我们的自定义工具
llm = ChatOpenAI(model="Qwen/Qwen2.5-72B-Instruct", base_url=os.environ.get("OPENAI_API_BASE"), temperature=0)

# 直接使用装饰器@tool定义的web_search函数
tools = [web_search]
llm_with_tools = llm.bind_tools(tools)

# 4. 带有系统提示的代理节点，强制一次调用一个工具
def react_agent_node(state: AgentState):
 console.print("--- 反应式代理：思考中... ---")
 
 messages_with_system_prompt = [
 SystemMessage(content="你是一个有帮助的研究助手。你必须一次只调用一个工具。不要在一次调用中调用多个工具。在收到工具结果后，你将决定下一步。")
 ] + state["messages"]

 # 使用原始的llm而不是绑定了工具的版本
 # 这样我们可以手动解析响应并创建正确格式的工具调用
 response_content = llm.invoke(messages_with_system_prompt).content
 
 # 检查响应是否包含工具调用格式
 import json
 from langchain_core.messages import AIMessage
 from langchain_core.messages.tool import ToolCall
 
 # 简单的模式匹配，寻找工具调用格式
 import re
 tool_call_pattern = r'```json\n(.*?)\n```'  # 匹配JSON代码块
 matches = re.findall(tool_call_pattern, response_content, re.DOTALL)
 
 if matches:
     try:
         # 解析工具调用
         tool_calls_json = json.loads(matches[0])
         if isinstance(tool_calls_json, list):
             # 创建正确格式的工具调用
             tool_calls = []
             for i, tool_call in enumerate(tool_calls_json):
                 if isinstance(tool_call["args"], str):
                     try:
                         args_dict = json.loads(tool_call["args"])
                     except json.JSONDecodeError:
                         args_dict = {"query": tool_call["args"]}
                 else:
                     args_dict = tool_call["args"]
                 
                 tool_calls.append(ToolCall(
                     name=tool_call["name"],
                     args=args_dict,
                     id=f"tool_{i}"
                 ))
             
             # 创建带有正确工具调用的AIMessage
             ai_message = AIMessage(
                 content="",
                 tool_calls=tool_calls
             )
         else:
             ai_message = AIMessage(content=response_content)
     except (json.JSONDecodeError, KeyError):
         # 如果解析失败，返回原始内容
         ai_message = AIMessage(content=response_content)
 else:
     ai_message = AIMessage(content=response_content)

 return {"messages": [ai_message]}

# 5. 在ToolNode中使用我们修正的自定义工具
tool_node = ToolNode([web_search])

# ReAct graphwithits characteristic loop
react_graph_builder = StateGraph(AgentState)
react_graph_builder.add_node("agent", react_agent_node)
react_graph_builder.add_node("tools", tool_node)
react_graph_builder.set_entry_point("agent")
react_graph_builder.add_conditional_edges("agent", tools_condition)
react_graph_builder.add_edge("tools", "agent")

react_agent_app = react_graph_builder.compile()
print("Reactive (ReAct)代理编译成功.")

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

# 可视化反应式代理图 - 生成图结构文件
try:
    import os
    current_dir = os.getcwd()
    
    # 生成Mermaid格式
    mermaid_graph = react_agent_app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(current_dir, "react_agent_app_graph.mermaid")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_graph)
    print(f"反应式代理图结构已保存为 {mermaid_path}")
    
    # 生成DOT格式
    dot_content = """digraph "Reactive (ReAct) Agent Graph" {
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
    print(f"反应式代理图结构已保存为 {dot_path}")
    
    # 条件化生成PNG
    if graphviz_installed and system_graphviz_available:
        try:
            import graphviz
            g = graphviz.Source.from_file(dot_path)
            g.render(filename="react_agent_app_graph", directory=current_dir, format="png", cleanup=True)
            print(f"反应式代理图结构已保存为 PNG 图像: {os.path.join(current_dir, 'react_agent_app_graph.png')}")
        except Exception as png_error:
            print(f"⚠️ 生成反应式代理PNG图像时出错: {png_error}")
    else:
        print("ℹ️ graphviz依赖不完整，仅生成文本格式的反应式代理图文件")
except Exception as e:
    print(f"反应式代理图表可视化失败：{e}")


# ### 步骤1.2： 在以规划为中心的问题上测试反应式代理
# 
# **我们将要做的：**
# 我们将给ReAct代理一个需要两个不同的数据收集步骤，然后进行最终计算的任务。这将测试它管理多步骤工作流程的能力，而无需预先计划。

# In[4]:


plan_centric_query = """
查找法国、德国和意大利首都的人口。
然后计算它们的总和。
最后，将总和与美国人口进行比较，并说明哪个更大。
"""

console.print(f"[bold yellow]测试 REACTIVE agentina plan-centric query:[/bold yellow] '{plan_centric_query}'")

final_react_output = None
for chunk in react_agent_app.stream({"messages": [("user", plan_centric_query)]}, stream_mode="values"):
 final_react_output = chunk
 console.print(f"--- [bold purple]当前状态更新[/bold purple] ---")
 chunk['messages'][-1].pretty_print()
 console.print("\n")

console.print("\n--- [bold red]反应式代理的最终输出[/bold red] ---")
console.print(Markdown(final_react_output['messages'][-1].content))


# **输出讨论：**
# ReAct代理成功完成了任务。通过观察流式输出，我们可以追踪其逐步推理过程：
# 1. 它首先决定搜索巴黎的人口。
# 2. 在接收到该结果后，它将其纳入记忆，然后决定下一步是搜索柏林的人口。
# 3. 最后，收集到两条信息后，它执行计算并提供最终答案。
# 
# 虽然它有效，但这种迭代发现过程并不总是最有效的。对于这样可预测的任务，代理在每一步之间进行额外的LLM调用来推理。这为展示规划代理的价值奠定了基础。

# ## 阶段2： 高级方法 - 一个规划 Agent
# 
# 现在，让我们构建一个在行动前思考的代理。这个代理将有一个专门的**Planner**来创建完整的任务列表，一个**Executor**来执行计划，以及一个**Synthesizer**来组装最终结果。

# ### 步骤2.1： 定义规划器、执行器和综合器节点
# 
# **我们将要做的：**
# 我们将创建新代理的核心组件：
# 1. **`Planner`:** 一个基于LLM的节点，接受用户请求并输出结构化计划。
# 2. **`Executor`:** 一个节点，接受计划，使用工具执行*下一个*步骤，并记录结果。
# 3. **`Synthesizer`:** 一个最终的基于LLM的节点，接受所有收集的结果并生成最终答案。

# In[5]:


# Pydantic模型以确保规划器的输出是结构化的步骤列表
class Plan(BaseModel):
 """执行以回答用户查询的工具调用计划。"""
 steps: List[str] = Field(description="执行后将回答查询的工具调用列表。")

# def规划代理的state
class PlanningState(TypedDict):
 user_request: str
 plan: Optional[List[str]]
 intermediate_steps: List[ToolMessage]
 final_answer: Optional[str]

def planner_node(state: PlanningState):
 """生成行动计划以回答用户的请求。"""
 console.print("--- 规划器：分解任务中... ---")
 planner_llm = llm.with_structured_output(Plan)
 
 # THE FIX: A much more explicit prompt with a clear example (few-shot prompting)
 prompt = f"""你是一名专业的规划师。你的工作是创建逐步计划来回答用户的请求。
计划中的每一步都必须是对`web_search`工具的单次调用。

**说明：**
1. 分析用户的请求。
2. 将其分解为一系列简单、合乎逻辑的搜索查询。
3. 将输出格式化为字符串列表，其中每个字符串都是单个有效的工具调用。

**示例：**
请求："法国的首都是什么，它的人口是多少？"
正确的计划输出：
[
"web_search('capital of France')",
"web_search('population of Paris')"
]

**用户的请求：**
{state['user_request']}
"""

 plan_result = planner_llm.invoke(prompt)
 # Use plan_result.steps, not plan.steps to avoid confusion with the variable name 'plan'
 console.print(f"--- 规划器：生成的计划： {plan_result.steps} ---")
 return {"plan": plan_result.steps}

def executor_node(state: PlanningState):
 """执行计划中的下一步。"""
 console.print("--- 执行器：运行下一步... ---")
 plan = state["plan"]
 next_step = plan[0]
 
 # Robust regex to handle both single and double quotes
 match = re.search(r"(\w+)\((?:\"|\')(.*?)(?:\"|\')\)", next_step)
 if not match:
    tool_name = "web_search"
    query = next_step
 else:
    tool_name, query = match.groups()[0], match.groups()[1]
 
 console.print(f"--- 执行器：调用工具 '{tool_name}' with query '{query}' ---")
 
 result= tavily_search_tool.invoke(query)
 
 # We still create a ToolMessage, but the tool call itself is now safe.
 tool_message = ToolMessage(
 content=str(result),
 name=tool_name,
 tool_call_id=f"manual-{hash(query)}"
 )
 
 return{
 "plan": plan[1:], # Pop the executed stepfromthe plan
 "intermediate_steps": state["intermediate_steps"] + [tool_message]
 }

def synthesizer_node(state: PlanningState):
 """从中间步骤综合最终答案。"""
 console.print("--- 综合器：生成最终答案中... ---")
 
 context = "\n".join([f"Tool {msg.name} returned: {msg.content}" for msg in state["intermediate_steps"]])
 
 prompt = f"""你是一名专业的综合器。基于用户的请求和收集的数据，提供全面的最终答案。

请求：{state['user_request']}
收集的数据：
{context}
"""
 final_answer = llm.invoke(prompt).content
 return {"final_answer": final_answer}

print("规划器、执行器和综合器节点已定义。")


# ### 步骤2.2： 构建规划代理图
# 
# **我们将要做的：**
# 现在我们将把新节点组装成一个图。流程将是： `Planner` -> `Executor`（循环）-> `Synthesizer`.

# In[6]:


def planning_router(state: PlanningState):
    if not state["plan"]:
        console.print("--- 路由器：计划完成。移至综合器。 ---")
        return "synthesize"
    else:
        console.print("--- 路由器：计划还有更多步骤。继续执行。 ---")
        return "execute"

planning_graph_builder = StateGraph(PlanningState)
planning_graph_builder.add_node("plan", planner_node)
planning_graph_builder.add_node("execute", executor_node)
planning_graph_builder.add_node("synthesize", synthesizer_node)

planning_graph_builder.set_entry_point("plan")
planning_graph_builder.add_conditional_edges("plan", planning_router, {"execute": "execute", "synthesize": "synthesize"}) # 规划后路由...
planning_graph_builder.add_conditional_edges("execute", planning_router, {"execute": "execute", "synthesize": "synthesize"})
planning_graph_builder.add_edge("synthesize", END)

planning_agent_app = planning_graph_builder.compile()
print("规划代理编译成功.")

# 可视化规划代理图 - 生成图结构文件
try:
    import os
    current_dir = os.getcwd()
    
    # 生成Mermaid格式
    mermaid_graph = planning_agent_app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(current_dir, "planning_agent_app_graph.mermaid")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_graph)
    print(f"规划代理图结构已保存为 {mermaid_path}")
    
    # 生成DOT格式
    dot_content = """digraph "Planning Agent Graph" {
    rankdir=TD;
    
    // 节点定义
    __start__ [shape=point];
    plan [label="plan", style=filled, fillcolor="#f2f0ff"];
    execute [label="execute", style=filled, fillcolor="#f2f0ff"];
    synthesize [label="synthesize", style=filled, fillcolor="#f2f0ff"];
    __end__ [label="__end__", shape=doublecircle, style=filled, fillcolor="#bfb6fc"];
    
    // 边定义
    __start__ -> plan;
    plan -> execute [label="有步骤需要执行"];
    plan -> synthesize [label="计划完成"];
    execute -> execute [label="继续执行"];
    execute -> synthesize [label="计划完成"];
    synthesize -> __end__;
}
"""
    dot_path = os.path.join(current_dir, "planning_agent_app_graph.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)
    print(f"规划代理图结构已保存为 {dot_path}")
    
    # 条件化生成PNG
    if graphviz_installed and system_graphviz_available:
        try:
            import graphviz
            g = graphviz.Source.from_file(dot_path)
            g.render(filename="planning_agent_app_graph", directory=current_dir, format="png", cleanup=True)
            print(f"规划代理图结构已保存为 PNG 图像: {os.path.join(current_dir, 'planning_agent_app_graph.png')}")
        except Exception as png_error:
            print(f"⚠️ 生成规划代理PNG图像时出错: {png_error}")
    else:
        print("ℹ️ graphviz依赖不完整，仅生成文本格式的规划代理图文件")
except Exception as e:
    print(f"规划代理图表可视化失败：{e}")


# ## 阶段3： 正面比较
# 
# 让我们在相同任务上运行我们的新规划代理，并将其执行流程和最终输出与反应式代理进行比较。

# In[7]:


console.print(f"[bold green]测试 PLANNING agent in the same plan-centric query:[/bold green] '{plan_centric_query}'")

# 记得正确初始化状态，特别是中间步骤的列表
initial_planning_input = {"user_request": plan_centric_query, "intermediate_steps": []}

final_planning_output = planning_agent_app.invoke(initial_planning_input)

console.print("\n--- [bold green]规划代理的最终输出[/bold green] ---")
console.print(Markdown(final_planning_output['final_answer']))


# **输出讨论：**
# 过程的差异立即显现。第一步就是`规划器`创建完整、明确的计划：`['web_search("population of Paris")', 'web_search("population of Berlin")']`. 
# 
# 代理然后有条不紊地执行这个计划，无需在步骤之间停下来思考。这个过程是：
# - **更透明：** 我们可以在代理开始之前看到其整个策略。
# - **更稳健：** 它不太可能偏离轨道，因为它遵循一套明确的指令。
# - **可能更高效：** 它避免了步骤之间推理的额外LLM调用。
# 
# 这展示了规划对于可以预先确定所需步骤的任务的强大功能。

# ## 阶段4： 定量评估
# 
# 为了正式化我们的比较，我们将使用LLM作为评判者来评分两个代理，重点关注其问题解决过程的质量和效率。

# In[8]:


class ProcessEvaluation(BaseModel):
 """评估代理问题解决过程的模式。"""
 任务完成评分: int = Field(description="对代理是否成功完成任务进行1-10评分。")
 流程效率评分: int = Field(description="对代理过程的效率和直接性进行1-10评分。更高的评分意味着更合乎逻辑且更少迂回的路径。")
评估理由: str = Field(description="评分的简要理由。")

judge_llm = llm.with_structured_output(ProcessEvaluation)

def evaluate_agent_process(query: str, final_state: dict):
    # for the ReAct agent, the trace is in 'messages'. for Planning, it's in 'intermediate_steps'.
    if 'messages' in final_state:
        trace = "\n".join([f"{m.type}: {str(m.content)}" for m in final_state['messages']])
    else:
        trace = f"""Plan: {final_state.get('plan', [])}
Steps: {final_state.get('intermediate_steps', [])}"""
 
    prompt = f"""你是一名专业的AI代理评判员。在1-10的等级上评估代理解决任务的过程。
 重点关注过程是否合乎逻辑且高效。
 
 **用户的任务：** {query}
 **完整代理跟踪：**\n```\n{trace}\n```
 """
    return judge_llm.invoke(prompt)

console.print("--- 评估 Reactive Agent's Process ---")
react_agent_evaluation = evaluate_agent_process(plan_centric_query, final_react_output)
console.print(react_agent_evaluation.model_dump())

console.print("\n--- 评估 Planning Agent's Process ---")
planning_agent_evaluation = evaluate_agent_process(plan_centric_query, final_planning_output)
console.print(planning_agent_evaluation.model_dump())


# **输出讨论：**
# 评判员的评分量化了两种方法的差异。两个代理可能都会获得高`任务完成评分`，因为它们最终都找到了答案。然而，**规划代理**将获得显著更高的`流程效率评分`。评判员的理由将强调其预先计划是解决问题的更直接和合乎逻辑的方式，相比ReAct代理的逐步探索过程。
# 
# 这个评估证实了我们的假设：对于解决路径可预测的问题，规划架构提供了更结构化、透明和高效的方法。

# ## 总结
# 
# 在这个notebook中，我们已经实现了**规划**架构，并将其与**ReAct**模式直接对比。通过强制代理在执行前首先构建全面的计划，我们在透明度、稳健性和效率方面获得了显著的好处，特别是对于定义明确的多步骤任务.
# 
# 虽然ReAct在下一步未知的探索性场景中表现出色，但当解决方案的路径可以预先规划时，规划架构就会大放异彩。理解这种权衡对系统设计者至关重要。为正确的问题选择正确的架构是构建有效和智能AI代理的关键技能。规划模式是该工具包中的重要工具，为复杂、可预测的工作流程提供所需的结构。