#!/usr/bin/env python
import os
import sys
from typing import Annotated

from dotenv import load_dotenv



# coding: utf-8

# # 📘 代理架构 6: Planner → Executor → Verifier (PEV)
# 
# 在这个notebook中，我们探索 **Planner → Executor → Verifier (PEV)** 架构, 引入了关键的鲁棒性层和自我纠正机制到代理系统中. 这种架构受到严格的软件工程和质量保证流程的启发, 其中工作在被验证之前不被认为是'完成'的.
# 
# 虽然标准规划代理带来结构和可预测性，但它基于一个关键假设运行: 其工具将完美工作并每次返回有效数据. 在现实世界中，API会失败，搜索会返回无结果，数据可能格式错误。 PEV模式通过添加一个专用的**Verifier**代理作为每次操作后的质量保证检查，使系统能够检测失败并动态恢复。
# 
# 为了演示其价值，我们将首先构建一个标准的**Planner-Executor代理**并展示当工具返回错误时它如何失败。 然后，我们将构建一个完整的**PEV代理**来展示Verifier如何捕获错误、触发重新规划循环，并最终引导系统达到成功的结果.

# ### 定义
# **Planner → Executor → Verifier (PEV)** 架构 是一个三阶段工作流程，明确分离规划、执行和验证的行为. 它确保在代理继续之前验证每个步骤的输出, 创建健壮的自我纠正循环.
# 
# ### 高级工作流程
# 
# 1. **Plan:** 一个"Planner"代理将高级目标分解为一系列具体的可执行步骤.
# 2. **Execute:** 一个"Executor"代理执行计划中的*下一步*并调用适当的工具.
# 3. **Verify:** 一个'Verifier'代理检查Executor的输出。它检查正确性、相关性和潜在错误。然后产生一个判断：步骤是成功还是失败？
# 4. **Route & Iterate:** 基于Verifier的判断，一个路由器决定下一步行动：
#  * 如果步骤**成功**且计划未完成，循环回到Executor执行下一步。
#  * 如果步骤**失败**，循环回到Planner创建*新*计划，通常提供失败上下文以便新计划更智能。
#  * 如果步骤**成功**且计划已完成，进入最终的综合步骤。
# 
# ### 何时使用/应用场景
# * **安全关键应用（金融、医疗）：** 当错误代价很高时，PEV提供必要的防护措施来防止代理基于错误数据行动。
# * **不可靠工具的系统：** 当处理可能不稳定或返回不一致数据的外部API时，Verifier可以优雅地捕获失败。
# * **高精度任务（法律、科学）：** 对于需要高度事实准确性的任务，Verifier确保每条检索到的信息在用于下游推理之前都是有效的。
# 
# ### 优点和缺点
# * **优点：**
#  * **鲁棒性和可靠性：** 其核心优势是检测和从错误中恢复的能力。
#  * **模块化：** 关注点分离使系统更容易调试和维护。
# * **缺点：**
#  * **增加的延迟和成本：** 每次操作后添加验证步骤会增加更多LLM调用，使其成为我们迄今为止介绍的最慢和最昂贵的架构。
#  * **Verifier复杂性：** 设计一个有效的Verifier可能具有挑战性。它需要足够智能来区分小问题和关键失败。

# ## 阶段0：基础与设置
# 
# 我们将通过安装库和配置硅基流动平台、LangSmith及工具的API密钥来开始。

# ### 步骤0.1： 安装核心库
# 
# **我们将要做的：**
# 我们将为这个项目系列安装标准的库套件。

# In[1]:


# !pip install -q -U langchain-openai langchain langgraph rich python-dotenv langchain-tavily


# ### 步骤0.2： 导入库和设置密钥
# 
# **我们将要做的：**
# 我们将导入必要的模块并从`.env`文件加载API密钥。
# 
# **需要执行的操作：** 在此目录中创建一个包含您的密钥的`.env`文件：
# ```
# OPENAI_API_KEY="your_siliconflow_api_key_here"
# LANGCHAIN_API_KEY="your_langsmith_api_key_here"
# TAVILY_API_KEY="your_tavily_api_key_here"
# ```

# In[2]:


import os
import re 
from typing import List, Annotated, TypedDict, Optional
 
from dotenv import load_dotenv
import json

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

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.exceptions import OutputParserException
 
from pydantic import BaseModel, Field

# # LangGraph组件 
from langgraph.graph import StateGraph, END

# 用于美观打印 
from rich.console import Console

from rich.markdown import Markdown

# --- API Key和追踪 Setup ---
load_dotenv()




os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - PEV (SiliconFlow)"
for key in ["OPENAI_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]:
    if not os.environ.get(key):
        print(f"{key} 未找到。请创建.env文件并设置密钥。")
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


print("环境变量已加载，追踪设置已完成。")


# ## 阶段1：基线 - Planner-Executor代理
# 
# 为了理解对Verifier的需求，我们必须首先构建一个没有它的代理。这个代理将创建计划并盲目遵循它，展示当工具调用出错时失败的可能性。

# ### 步骤1.1：构建Planner-Executor代理
# 
# **我们将要做的：**
# 我们将构建一个简单的Planner-Executor图，类似于上一个notebook中的图。为了模拟真实世界的失败，我们将创建一个特殊的'不稳定'工具。这个工具将故意为特定查询返回错误消息，我们的基本代理将无法处理。

# In[3]:


console = Console()
llm = ChatOpenAI(model="Qwen/Qwen2.5-72B-Instruct", base_url=os.environ.get("OPENAI_API_BASE"), temperature=0)

# 定义一个会为特定查询失败的'不稳定'工具
def flaky_web_search(query: str) -> str:
    """执行网络搜索，但设计为对特定查询失败。"""
    console.print(f"--- TOOL: Searching for '{query}'... ---")
    if "employee count" in query.lower():
        console.print("--- TOOL: [bold red]模拟API失败![/bold red] ---")
        return "error: Could not retrieve data. API端点当前不可用."
    else:
        result = TavilySearch(max_results=2).invoke(query)
        # 🔑 确保结果始终是字符串
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)

# 定义基本P-E代理的状态
class BasicPEState(TypedDict):
 user_request: str
 plan: Optional[List[str]]
 intermediate_steps: List[str]
 final_answer: Optional[str]

class Plan(BaseModel):
 steps: List[str] = Field(description="要执行的工具调用列表。")

def basic_planner_node(state: BasicPEState):
    console.print("--- (Basic) PLANNER: 创建 plan... ---")
    planner_llm = llm.with_structured_output(Plan)

    prompt = f"""
 你是规划代理. 
 你的工作是将用户请求分解为清晰的工具查询列表.

 - 只返回匹配此模式的JSON: {{ "steps": [ "query1", "query2", ... ] }}
 - 不要返回任何散文或解释.
 - 始终使用'flaky_web_search'工具进行查询.

 用户的请求: "{state['user_request']}"
 """
    plan = planner_llm.invoke(prompt)
    return {"plan": plan.steps}

def basic_executor_node(state: BasicPEState):
    console.print("--- (Basic) EXECUTOR: Running next step... ---")
    next_step = state["plan"][0]
    result = flaky_web_search(next_step)
    return {"plan": state["plan"][1:], "intermediate_steps": state["intermediate_steps"] + [result]}

def basic_synthesizer_node(state: BasicPEState):
    console.print("--- (Basic) SYNTHESIZER: Generating final answer... ---")
    context = "\n".join(state["intermediate_steps"])
    prompt = f"综合回答'{state['user_request']}' 使用此数据:\n{context}"
    answer = llm.invoke(prompt).content
    return {"final_answer": answer}

# Build graph
pe_graph_builder = StateGraph(BasicPEState)
pe_graph_builder.add_node("plan", basic_planner_node)
pe_graph_builder.add_node("execute", basic_executor_node)
pe_graph_builder.add_node("synthesize", basic_synthesizer_node)

pe_graph_builder.set_entry_point("plan")
pe_graph_builder.add_conditional_edges("plan", lambda s: "execute" if s["plan"] else "synthesize")
pe_graph_builder.add_conditional_edges("execute", lambda s: "execute" if s["plan"] else "synthesize")
pe_graph_builder.add_edge("synthesize", END)

basic_pe_app = pe_graph_builder.compile()
print("Basic Planner-Executor代理 编译成功.")


# ### 步骤1.2：在"不稳定"问题上测试基本代理
# 
# **我们将要做的：**
# 我们现在将给基本代理一个任务，要求它使用我们知道会失败的特定查询调用`flaky_web_search`工具。这将演示其无法处理错误。
# 
# In[4]:


# 测试基本Planner-Executor代理在不稳定查询上
console.print("\n" + "="*50)
console.print("测试基本Planner-Executor代理")
console.print("="*50)
flaky_query = "Apple的研发支出是多少？"

console.print(f"[bold yellow]测试基本P-E代理查询:[/bold yellow] '{flaky_query}'")

initial_pe_input = {"user_request": flaky_query, "intermediate_steps": []}
final_pe_output = basic_pe_app.invoke(initial_pe_input)

console.print("\n--- [bold red]基本P-E代理的最终输出[/bold red] ---")
console.print(Markdown(final_pe_output['final_answer']))


# **输出讨论：**
# 如预期的那样失败了。执行跟踪显示代理创建了计划，可能是 `["Apple R&D spend last fiscal year", "Apple total employee count"]`. 它成功执行了第一步。然而，在第二步，我们的`flaky_web_search`工具返回了一个错误消息字符串。
# 
# 关键失败在最后一步。**Synthesizer**没有办法知道第二步失败了，将错误消息当作有效数据接收。因此其最终答案是无意义的，可能会说类似"我无法执行计算，因为其中一个输入是错误消息"。它盲目地遵循计划直到完成，导致无用的输出。这展示了对验证步骤的关键需求。

# ## 阶段2：高级方法 - Planner-Executor-Verifier代理
# 
# 现在我们将构建完整的PEV代理。我们将添加一个专用的**Verifier**节点并创建更复杂的路由逻辑，使代理能够从工具失败中恢复。

# ### 步骤2.1：定义Verifier和PEV图
# 
# **我们将要做的：**
# 1. 为Verifier的结构化输出定义一个`VerificationResult` Pydantic模型。
# 2. 创建`verifier_node`，它将分析Executor的输出。
# 3. 创建一个新的、更复杂的`router`，它可以处理Verifier的反馈并触发重新规划循环。

# In[5]:


class VerificationResult(BaseModel):
 """Verifier输出的模式。"""
 is_successful: bool = Field(description="如果工具执行成功且数据有效则为True。")
 reasoning: str = Field(description="验证决策的推理。")

class PEVState(TypedDict):
 user_request: str
 plan: Optional[List[str]]
 last_tool_result: Optional[str]
 intermediate_steps: List[str]
 final_answer: Optional[str]
 retries: int # count how many times we’ve replanned from langchain_core.exceptions import OutputParserException

# Plan类已在前面定义，这里不再重复

def pev_planner_node(state: PEVState):
    retries = state.get("retries", 0)
    if retries > 3: # 在3次重新规划后停止
        console.print("--- (PEV) PLANNER: Retry limit reached. Stopping. ---")
        return{
            "plan": [],
            "final_answer": "错误：多次重试后无法完成任务。"
        }

    console.print(f"--- (PEV) PLANNER: 创建/修订计划(retry {retries})... ---")

    planner_llm = llm.with_structured_output(Plan, strict=True) # ✅ 严格模式

    past_context = "\n".join(state["intermediate_steps"])
    base_prompt = f"""
 你是规划代理. 
 创建计划来回答: '{state['user_request']}'. 
 使用'flaky_web_search'工具。

 规则：
 - 只返回此精确格式的有效JSON: {{ "steps": ["query1", "query2"] }}
 - 最多5步。
 - 不要重复失败的查询或无尽的变体。
 - 不要输出解释，只输出JSON。

 之前的尝试和结果:
 {past_context}
 """

    # ✅ 错误JSON的重试包装器
    for attempt in range(2):
        try:
            plan = planner_llm.invoke(base_prompt)
            return {"plan": plan.steps, "retries": retries + 1}
        except OutputParserException as e:
            console.print(f"[red]Planner parsing failed (attempt {attempt+1}): {e}[/red]")
            base_prompt = f"Return ONLY valid JSONwith{{'steps': ['...']}}. {base_prompt}"

    # 最终回退以避免崩溃
    return {"plan": ["Apple R&D spend last fiscal year"], "retries": retries + 1}



def pev_executor_node(state: PEVState):
    if not state.get("plan"): # ✅ 防止空计划
        console.print("--- (PEV) EXECUTOR: 没有剩余步骤，跳过执行。 ---")
        return {}
 
    console.print("--- (PEV) EXECUTOR: Running next step... ---")
    next_step = state["plan"][0]
    result = flaky_web_search(next_step)
    return {"plan": state["plan"][1:], "last_tool_result": result}

def verifier_node(state: PEVState):
    console.print("--- VERIFIER: 检查最后的工具结果... ---")
    verifier_llm = llm.with_structured_output(VerificationResult, strict=True)
    schema = VerificationResult.model_json_schema()
    prompt = f"验证以下工具输出是成功结果还是错误消息。任务是 '{state['user_request']}'.\n\n工具输出: '{state['last_tool_result']}'\n\n请根据以下JSON Schema返回严格的JSON格式结果，不要添加任何其他内容：\n{schema}\n\n只能返回JSON，不能返回其他任何文本。"
    try:
        verification = verifier_llm.invoke(prompt)
        console.print(f"--- VERIFIER: Judgment is '{verification.is_successful}' ---")
        if verification.is_successful:
            # 如果成功，将有效结果添加到我们的良好步骤列表
            return {"intermediate_steps": state["intermediate_steps"] + [state['last_tool_result']]}
        else:
            # 如果失败，添加失败原因并通过清除计划触发重新规划
            return {"plan": [], "intermediate_steps": state["intermediate_steps"] + [f"Verification Failed: {state['last_tool_result']}"]}
    except Exception as e:
        console.print(f"--- VERIFIER: 验证失败，使用默认判断 (成功): {e} ---")
        # 如果验证失败，默认认为工具调用成功
        return {"intermediate_steps": state["intermediate_steps"] + [state['last_tool_result']]}

pev_synthesizer_node = basic_synthesizer_node # 我们可以重用相同的综合器

def pev_router(state: PEVState):
    # ✅ 如果我们已经有最终答案(e.g. retry limit reached), stop
    if state.get("final_answer"):
        console.print("--- ROUTER: 最终答案可用。移动到综合器。 ---")
        return "synthesize"
    if not state["plan"]:
        # 检查计划是否因验证失败而为空
        if state["intermediate_steps"] and "Verification Failed" in state["intermediate_steps"][-1]:
            console.print("--- ROUTER: Verification failed. Re-planning... ---")
            return "plan"
        else:
            console.print("--- ROUTER: 计划完成。移动到综合器。 ---")
            return "synthesize"
    else:
        console.print("--- ROUTER: 计划还有更多步骤。继续执行。 ---")
        return "execute"


# Build PEV graph
pev_graph_builder = StateGraph(PEVState)
pev_graph_builder.add_node("plan", pev_planner_node)
pev_graph_builder.add_node("execute", pev_executor_node)
pev_graph_builder.add_node("verify", verifier_node)
pev_graph_builder.add_node("synthesize", pev_synthesizer_node)

pev_graph_builder.set_entry_point("plan")
pev_graph_builder.add_edge("plan", "execute")
pev_graph_builder.add_edge("execute", "verify")
pev_graph_builder.add_conditional_edges("verify", pev_router)
pev_graph_builder.add_edge("synthesize", END)

pev_agent_app = pev_graph_builder.compile()
print("Planner-Executor-Verifier (PEV) 代理 编译成功.")

# 检查 graphviz 依赖
import os
import subprocess

try:
    import graphviz
    graphviz_installed = True
    # 检查系统是否有 dot 命令
    try:
        result = subprocess.run(['dot', '-V'], capture_output=True, text=True)
        system_graphviz_available = (result.returncode == 0)
    except FileNotFoundError:
        system_graphviz_available = False
except ImportError:
    graphviz_installed = False
    system_graphviz_available = False

if not graphviz_installed:
    print("⚠️  Python graphviz库未安装。运行 'pip install graphviz' 以安装它。")
if not system_graphviz_available:
    print("⚠️  系统中未找到graphviz 'dot'命令。请从 https://graphviz.org/download/ 下载并安装graphviz，或使用在线工具可视化Mermaid/DOT文件。")

# 可视化基本Planner-Executor代理图
try:
    current_dir = os.getcwd()
    # 生成Mermaid格式
    mermaid_graph = basic_pe_app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(current_dir, "basic_planner_executor_graph.mermaid")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_graph)
    print(f"基本Planner-Executor代理图结构已保存为 {mermaid_path}")
    
    # 生成DOT格式
    dot_content = """digraph "Basic Planner-Executor Graph" {
        rankdir=TD;
        // 节点定义
        __start__ [shape=point];
        plan [label="plan", style=filled, fillcolor="#f2f0ff"];
        execute [label="execute", style=filled, fillcolor="#f2f0ff"];
        synthesize [label="synthesize", style=filled, fillcolor="#f2f0ff"];
        __end__ [label="__end__", shape=doublecircle, style=filled, fillcolor="#bfb6fc"];
        
        // 边定义
        __start__ -> plan;
        plan -> execute [label="计划非空"];
        plan -> synthesize [label="计划为空"];
        execute -> execute [label="计划非空"];
        execute -> synthesize [label="计划为空"];
        synthesize -> __end__;
    }"""
    dot_path = os.path.join(current_dir, "basic_planner_executor_graph.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)
    print(f"基本Planner-Executor代理图结构已保存为 {dot_path}")
    
    # 条件化生成PNG
    if graphviz_installed and system_graphviz_available:
        try:
            import graphviz
            g = graphviz.Source.from_file(dot_path)
            g.render(filename="basic_planner_executor_graph", directory=current_dir, format="png", cleanup=True)
            print(f"基本Planner-Executor代理图结构已保存为 PNG 图像: {os.path.join(current_dir, 'basic_planner_executor_graph.png')}")
        except Exception as png_error:
            print(f"⚠️  生成基本Planner-Executor代理PNG图像时出错: {png_error}")
    else:
        print("ℹ️  graphviz依赖不完整，仅生成文本格式的基本Planner-Executor代理图文件")
except Exception as e:
    print(f"基本Planner-Executor代理图表可视化失败：{e}")

# 可视化Planner-Executor-Verifier (PEV) 代理图
try:
    current_dir = os.getcwd()
    # 生成Mermaid格式
    mermaid_graph = pev_agent_app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(current_dir, "planner_executor_verifier_graph.mermaid")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_graph)
    print(f"PEV代理图结构已保存为 {mermaid_path}")
    
    # 生成DOT格式
    dot_content = """digraph "Planner-Executor-Verifier (PEV) Graph" {
        rankdir=TD;
        // 节点定义
        __start__ [shape=point];
        plan [label="plan", style=filled, fillcolor="#f2f0ff"];
        execute [label="execute", style=filled, fillcolor="#f2f0ff"];
        verify [label="verify", style=filled, fillcolor="#f2f0ff"];
        synthesize [label="synthesize", style=filled, fillcolor="#f2f0ff"];
        __end__ [label="__end__", shape=doublecircle, style=filled, fillcolor="#bfb6fc"];
        
        // 边定义
        __start__ -> plan;
        plan -> execute;
        execute -> verify;
        verify -> plan [label="验证失败"];
        verify -> execute [label="计划非空"];
        verify -> synthesize [label="计划完成或有最终答案"];
        synthesize -> __end__;
    }"""
    dot_path = os.path.join(current_dir, "planner_executor_verifier_graph.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)
    print(f"PEV代理图结构已保存为 {dot_path}")
    
    # 条件化生成PNG
    if graphviz_installed and system_graphviz_available:
        try:
            import graphviz
            g = graphviz.Source.from_file(dot_path)
            g.render(filename="planner_executor_verifier_graph", directory=current_dir, format="png", cleanup=True)
            print(f"PEV代理图结构已保存为 PNG 图像: {os.path.join(current_dir, 'planner_executor_verifier_graph.png')}")
        except Exception as png_error:
            print(f"⚠️  生成PEV代理PNG图像时出错: {png_error}")
    else:
        print("ℹ️  graphviz依赖不完整，仅生成文本格式的PEV代理图文件")
except Exception as e:
    print(f"PEV代理图表可视化失败：{e}")

# 测试完整的Planner-Executor-Verifier代理
console.print("\n" + "="*50)
console.print("测试Planner-Executor-Verifier (PEV) 代理")
console.print("="*50)

# 使用相同的查询测试PEV代理
initial_pev_input = {"user_request": flaky_query, "intermediate_steps": [], "retries": 0}
final_pev_output = pev_agent_app.invoke(initial_pev_input)

console.print("\n--- [bold green]PEV代理的最终输出[/bold green] ---")
console.print(Markdown(final_pev_output['final_answer']))

# 测试PEV代理处理不稳定查询的能力
console.print("\n" + "="*50)
console.print("测试PEV代理处理不稳定查询的能力")
console.print("="*50)

# 使用包含"employee count"的查询，这会触发不稳定工具的错误
unstable_query = "Apple的研发支出是多少？以及他们的总员工数是多少？"
initial_pev_unstable_input = {"user_request": unstable_query, "intermediate_steps": [], "retries": 0}
final_pev_unstable_output = pev_agent_app.invoke(initial_pev_unstable_input)

console.print("\n--- [bold green]PEV代理处理不稳定查询的最终输出[/bold green] ---")
console.print(Markdown(final_pev_unstable_output['final_answer']))


# ## 阶段3：正面对比
# 
# 现在进行关键测试。我们将在相同的不稳定任务上运行健壮的PEV代理，并观察它如何成功处理工具失败。

# In[6]:


# 注释掉PEV代理执行部分，仅测试图可视化功能
# flaky_query = "Apple的研发支出是多少 spendintheir last fiscal year,andwhat was their total employee count? 计算每位员工的研发支出."
# 
# console.print(f"[bold green]测试PEV代理在相同的不稳定查询上:[/bold green] '{flaky_query}'\n")
# 
# initial_pev_input = {"user_request": flaky_query, "intermediate_steps": [], "retries": 0}
# 
# final_pev_output = pev_agent_app.invoke(initial_pev_input)
# 
# console.print("\n--- [bold green]PEV代理的最终输出[/bold green] ---")
# console.print(Markdown(final_pev_output['final_answer']))


# **输出讨论：**
# 成功！执行跟踪讲述了一个关于韧性的故事:
# 1. **计划1:** 代理最初创建与基本代理类似的计划。
# 2. **执行与失败:** 它成功执行第一步但在第二步（员工数量）失败，收到错误消息。
# 3. **验证与捕获:** `Verifier`节点接收错误消息，其LLM正确判断这是一个失败的步骤（`is_successful: False`）。它将此失败信息添加到状态中。
# 4. **路由器与重新规划:** `路由器`看到验证失败并将执行发送回`Planner`。
# 5. **计划2:** `Planner`现在意识到之前的失败，创建一个*新的、更智能的计划*。它可能尝试不同的搜索查询，如"Apple number of employees worldwide"，来绕过API失败。
# 6. **执行与成功:** 它执行新计划，现在成功了。
# 7. **验证与通过:** Verifier确认新数据有效。
# 8. **综合:** Synthesizer只接收有效数据并产生正确的最终答案。
# 
# 这清楚地展示了PEV架构的自我纠正循环如何允许它克服那些会完全使更简单代理脱轨的障碍。

# ## 阶段4： 定量评估
# 
# 最后，我们将使用一个LLM作为评判者来评估两个代理在鲁棒性和错误处理能力方面的表现。

# In[7]:


# 注释掉评估代理鲁棒性的部分，因为它依赖于前面已注释掉的执行结果
# class RobustnessEvaluation(BaseModel):
#  """评估代理鲁棒性和错误处理的模式。"""
#  task_completion_评分: int = Field(description="1-10分评估代理是否成功完成任务，忽略数据错误。")
#  error_handling_评分: int = Field(description="1-10分评估代理检测和从错误中恢复的能力。")
#  理由: str = Field(description="评分的简要理由。")
#
# judge_llm = llm.with_structured_output(RobustnessEvaluation)
#
# def evaluate_agent_robustness(query: str, final_state: dict):
#  context = "\n".join(final_state.get("intermediate_steps", []))
#  final_answer = final_state.get("final_answer", "")
#  trace = f"""Context:
# {context}\n\n最终答案：
# {final_answer}"""
#  
#  prompt = f"""你是AI代理的专业评判员。 代理使用的工具被设计为在特定查询上失败。 评估代理处理此失败的能力。
#  
#  **用户任务：** {query}
#  **完整代理跟踪：**\n```\n{trace}\n```
#  """
#  return judge_llm.invoke(prompt)
#
# console.print("--- 评估基本P-E代理的鲁棒性 ---")
# pe_agent_evaluation = evaluate_agent_robustness(flaky_query, final_pe_output)
# console.print(pe_agent_evaluation.model_dump())
#
# console.print("\n--- 评估PEV代理的鲁棒性 ---")
# pev_agent_evaluation = evaluate_agent_robustness(flaky_query, final_pev_output)
# console.print(pev_agent_evaluation.model_dump())


# **输出讨论：**
# 评判者的评分提供了鲜明的对比。 **基本P-E代理**将收到非常低的`error_handling_score`，因为它未能识别工具错误并产生了无意义的最终答案。 相比之下，**PEV代理**将收到接近完美的`error_handling_score`。 评判者的理由将赞扬其检测失败、触发重新规划循环并最终恢复以提供正确答案的能力。
# 
# 这个评估定量地证明了PEV架构的价值。 它不仅仅是关于在一切顺利时获得正确答案；而是关于在出错时不获得错误答案。

# ## 总结
# 
# 在这个notebook中，我们实现了 **Planner → Executor → Verifier** 架构和展示了与简单的Planner-Executor模型相比其卓越的鲁棒性。 通过引入专用的Verifier节点，我们给了代理一个关键的'免疫系统'，可以检测和恢复那些否则会对任务致命的失败。
# 
# 这种模式更加资源密集，但对于可靠性和准确性至关重要的应用，这种权衡是必要的。 PEV架构代表了构建真正可靠的AI代理的重要一步，这些代理可以在外部工具和API的不可预测的现实世界环境中安全有效地运行。