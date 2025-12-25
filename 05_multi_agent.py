#!/usr/bin/env python
import os
import sys
from typing import Annotated

from dotenv import load_dotenv



# coding: utf-8

# # 📘 Agentic Architectures 5: 多代理系统
# 
# 在这个notebook中, 我们进入最 强大和灵活的 架构: **多代理系统**. 这种模式超越了单个代理的概念, 无论多么复杂, 而是建模一个专门代理团队来协作解决问题. 每个代理都有独特的角色、人格和技能集, 模仿人类专家团队的工作方式.
# 
# 这种方法允许深刻的 '分工', 其中复杂问题被分解为子任务 并分配 到agent best suited 完成工作. 为了展示其威力, 我们将进行直接比较. 首先, 我们将任务一个单一的 **单体 '通才' agent** 来创建 一个comprehensive market 分析 报告. 然后, 我们将组建一个 **专家团队**—技术分析师、新闻分析师和财务分析师—并有第四个 '管理者' 代理将他们的专家输入综合成最终报告. 质量、结构和深度的差异将立即显现.

# ### 定义
# A **多代理系统** 是一个架构，其中一组不同的专门代理协作 (or有时竞争) 以实现共同目标. 中央控制器 or定义的 工作流协议用于管理通信 并路由 代理之间的任务.
# 
# ### 高级工作流程
# 
# 1. **分解:** 主控制器 or用户提供 一个复杂任务.
# 2. **角色定义:** 系统根据定义的角色将子任务分配给专门代理 (e.g., 'Researcher', 'Coder', 'Critic', 'Writer').
# 3. **协作:** 代理执行其任务，通常并行或顺序. 它们将输出传递给彼此或中央 '黑板'.
# 4. **综合:** 最终的 '管理者' 或"综合器" 代理收集输出 从专家代理 并组装最终的综合响应.
# 
# ### 何时使用 / Applications
# * **复杂报告生成:** 创建详细的 报告 需要来自多个领域的专业知识 (例如，财务分析、科学研究).
# * **软件开发管道:** 模拟开发团队 包括程序员, 代码审查员, 测试员, 和项目 管理者.
# * **创意头脑风暴:** 具有不同的代理团队 '个性' (例如，一个乐观、一个谨慎、一个极具创造力) 可以生成更多样化的想法集合.
# 
# ### 优点和缺点
# * **优点：**
#  * **专业化和深度:** 每个代理都可以使用特定的人格和工具进行微调, 导致更高质量的 在其领域工作.
#  * **模块化和可扩展性:** 可以轻松添加、删除 or升级单个代理 而无需重新设计整个系统.
#  * **并行性:** 多个代理 可以同时处理其子任务, 可能减少总体任务时间.
# * **缺点：**
#  * **协调开销:** 管理代理之间的通信和工作流增加了系统设计的复杂性.
#  * **增加的成本和延迟:** 运行多个代理涉及更多LLM调用，这可能比单代理方法更昂贵和更慢.

# ## 阶段0：基础与设置
# 
# 我们将通过安装库和用于硅基流动平台、LangSmith和Tavily配置API密钥来开始.

# ### 步骤0.1：安装核心库
# 
# **我们将要做的:**
# 我们将安装用于这个项目系列的标准库套件.

# In[1]:


# !pip install -q -U langchain-openai langchain langgraph rich python-dotenv langchain-tavily  # 已注释 langchain-openai
# !pip install -q -U langchain-openai langchain langgraph rich python-dotenv langchain-tavily  # 使用硅基流动平台


# ### 步骤0.2：导入库和设置密钥
# 
# **我们将要做的:**
# 我们将导入必要的模块并从 `.env` 文件加载我们的API密钥.
# 
# **需要执行的操作:** 创建 `.env` 文件加载我们的API密钥 在此目录中包含您的密钥:
# ```
# OPENAI_API_KEY="your_siliconflow_api_key_here"
# LANGCHAIN_API_KEY="your_langsmith_api_key_here"
# TAVILY_API_KEY="your_tavily_api_key_here"
# ```

# In[ ]:


import os

from typing import List, Annotated, TypedDict, Optional


from dotenv import load_dotenv

# PHOENIX追踪配置
import logging
import os

# 设置日志
logging.basicConfig(filename=f'phoenix_init_{os.path.basename(__file__)}.log', level=logging.INFO, encoding='utf-8')
logger = logging.getLogger(__name__)

# LangChain components (已注释)
# 
from langchain_openai import ChatOpenAI
# 
from langchain_tavily import TavilySearch

# 硅基流动平台组件

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage


from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

# LangGraph components

from langgraph.graph import StateGraph, END

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 用于美观打印

from rich.console import Console


from rich.markdown import Markdown

# --- API密钥和追踪设置 ---
load_dotenv()




os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - 多代理 (SiliconFlow)"
# 配置PHOENIX追踪
phoenix_app = None
project_name = os.path.splitext(os.path.basename(__file__))[0]

# 将Phoenix初始化信息写入日志文件，避免编码问题
try:
    with open("phoenix_init.log", "a", encoding="utf-8") as f:
        f.write(f"\n开始初始化Phoenix追踪 - {project_name}...\n")
        
        # 1. 导入并启动 Phoenix 服务器
        try:
            import phoenix as px
            f.write("成功导入phoenix模块\n")
            
            # 使用新的OpenInference API进行追踪
            try:
                from phoenix.otel import register
                from openinference.instrumentation.langchain import LangChainInstrumentor
                f.write("成功导入新的Phoenix追踪API\n")
                
                # 注册tracer并instrument LangChain（连接到外部Phoenix服务器）
                tracer_provider = register(project_name=project_name)
                LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
                f.write("Phoenix LangChain追踪已通过OpenInference启用\n")
                
                # 尝试instrument OpenAI
                try:
                    from openinference.instrumentation.openai import OpenAIInstrumentor
                    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
                    f.write("Phoenix OpenAI追踪已通过OpenInference启用\n")
                except ImportError:
                    f.write("未找到OpenAIInstrumentor，跳过OpenAI追踪启用\n")
                
                logger.info("Phoenix追踪初始化成功")
                print(f"PHOENIX追踪已配置，项目名: {project_name}")
            except Exception as e:
                f.write(f"使用OpenInference API失败: {e}\n")
                import traceback
                traceback.print_exc(file=f)
                logger.error(f"使用OpenInference API失败: {e}")
                print(f"警告: 使用OpenInference API失败: {e}")
        except Exception as e:
            f.write(f"Phoenix 初始化失败: {e}\n")
            import traceback
            traceback.print_exc(file=f)
            logger.error(f"Phoenix追踪初始化失败: {e}")
            print(f"警告: PHOENIX追踪初始化失败: {e}")

except Exception as ex:
    logger.error(f"Phoenix追踪配置过程中发生错误: {ex}")
    print(f"警告: PHOENIX追踪配置过程中发生错误: {ex}")


for key in ["OPENAI_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]:  # 移除重复的OPENAI_API_KEY
    if not os.environ.get(key):
        print(f"{key} 未找到。请创建.env文件并设置它.")

print("环境变量已加载，追踪已设置.")


# ## 阶段1： 基线 - 单体 '通才' Agent
# 
# 为了展示价值 的 专家团队, 我们首先需要看看 单个代理如何执行 在 复杂任务. 我们将构建 一个ReAct代理 并给 it 一个广泛的提示要求它执行 多种类型 的分析 一次.

# ### 步骤1.1：构建单体代理
# 
# **我们将要做的:**
# 我们将构建一个标准的ReAct代理. 我们将为其提供网络搜索工具和一个非常通用的系统提示，要求它成为一个全面的财务分析师.

# In[3]:


console = Console()

# 为两个代理定义共享状态
class AgentState(TypedDict):
 messages: Annotated[list[BaseMessage], add_messages]

# 定义工具和LLM
search_tool = TavilySearch(max_results=3, name="web_search")
llm = ChatOpenAI(model="Qwen/Qwen2.5-72B-Instruct", base_url=os.environ.get("OPENAI_API_BASE"), temperature=0)
llm_with_tools = llm.bind_tools([search_tool])

# 定义单体代理节点
def mono_agent_node(state: AgentState):
 console.print("--- 单体代理：思考中... ---")
 response = llm_with_tools.invoke(state["messages"])
 return {"messages": [response]}

tool_node = ToolNode([search_tool])

# 为单体代理构建ReAct图
mono_graph_builder = StateGraph(AgentState)
mono_graph_builder.add_node("agent", mono_agent_node)
mono_graph_builder.add_node("tools", tool_node)
mono_graph_builder.set_entry_point("agent")

def tools_condition_with_end(state):
    result = tools_condition(state)
    if isinstance(result, str):
        # 旧版本只返回"tools"或"agent"
        return {result: "tools", "__default__": END}
    elif isinstance(result, dict):
        # 新版本返回映射
        result["__default__"] = END
        return result
    else:
        raise TypeError(f"来自tools_condition的意外类型: {type(result)}")

mono_graph_builder.add_conditional_edges("agent", tools_condition_with_end)
mono_graph_builder.add_edge("tools", "agent")

mono_agent_app = mono_graph_builder.compile()

print("单体'通才'代理编译成功。")

# --- Graphviz依赖检查 --- (单体代理)
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

# 可视化单体代理图 - 生成图结构文件
try:
    import os
    current_dir = os.getcwd()
    
    # 生成Mermaid格式
    mermaid_graph = mono_agent_app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(current_dir, "mono_agent_app_graph.mermaid")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_graph)
    print(f"单体代理图结构已保存为 {mermaid_path}")
    
    # 生成DOT格式
    dot_content = """digraph "Mono Agent Graph" {
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
    dot_path = os.path.join(current_dir, "mono_agent_app_graph.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)
    print(f"单体代理图结构已保存为 {dot_path}")
    
    # 条件化生成PNG
    if graphviz_installed and system_graphviz_available:
        try:
            import graphviz
            g = graphviz.Source.from_file(dot_path)
            g.render(filename="mono_agent_app_graph", directory=current_dir, format="png", cleanup=True)
            print(f"单体代理图结构已保存为 PNG 图像: {os.path.join(current_dir, 'mono_agent_app_graph.png')}")
        except Exception as png_error:
            print(f"⚠️ 生成单体代理PNG图像时出错: {png_error}")
    else:
        print("ℹ️ graphviz依赖不完整，仅生成文本格式的单体代理图文件")
except Exception as e:
    print(f"单体代理图表可视化失败：{e}")


# ### 步骤1.2：测试单体代理
# 
# **我们将要做的:**
# 我们将给通才代理一个复杂任务：为一家公司创建完整的市场分析报告，涵盖三个不同领域.

# In[4]:


company = "NVIDIA (NVDA)"
mono_query = f"为...创建简短但全面的市场分析报告 {company}. 报告应包括三个 sections: 1. A summary 的recent news 和市场情绪. 2. 股票的基本技术分析's price trend. 3. 查看公司最近的财务表现."

console.print(f"[bold yellow]测试单体代理在多方面任务上:[/bold yellow]\n'{mono_query}'\n")

final_mono_output = mono_agent_app.invoke({
 "messages": [
 SystemMessage(content="你是一个single, 专业财务分析师. 你必须创建全面的报告，涵盖用户请求的所有方面."),
 HumanMessage(content=mono_query)
 ]
})

console.print("\n--- [bold red]来自单体代理的最终报告[/bold red] ---")
console.print(Markdown(final_mono_output['messages'][-1].content))


# **输出讨论:**
# 单体代理生成了一个报告. 它可能执行了多次网络搜索并尽力综合信息. 然而，输出可能有一些弱点:
# - **缺乏结构:** 各部分可能混在一起，没有清晰的标题或专业格式.
# - **肤浅的分析:** 试图同时成为三个领域的专家，代理可能只提供高层次的总结，在任何单一领域都没有太多深度.
# - **通用语调:** 语言可能是通用的，缺乏每个领域真正专家的特定术语和关注点.
# 
# 这个结果是我们的基线。它是功能性的，但不是特别出色。 现在，我们将构建一个专家团队来看看我们是否能做得更好。

# ## 阶段2： 高级方法 - 多代理 专家团队
# 
# 现在我们将建立我们的团队: 新闻分析师、技术分析师和财务分析师. 每个都将是具有特定人格的自己的代理节点. 最终的报告撰写者将充当管理者，编译他们的工作.

# ### 步骤2.1：定义专家代理节点
# 
# **我们将要做的:**
# 我们将创建三个不同的代理节点. 关键区别是我们给每个代理的高度特定的系统提示. 这个提示定义了它们的人格、专业领域以及输出应采用的确切格式. 这就是我们如何强制专业化.

# In[ ]:


# 我们的多代理系统的状态将保存每个专家的输出
class MultiAgentState(TypedDict):
 user_request: str
 news_report: Optional[str]
 technical_report: Optional[str]
 financial_report: Optional[str]
 final_report: Optional[str]

def create_specialist_node(persona: str, output_key: str):
    """创建专家代理节点的工厂函数."""
    system_prompt = persona + "\n\n你可以访问网络搜索工具. 你的输出必须是一个简洁的报告部分, 使用markdown格式, 仅关注你的专业领域."

    # ✅ 构建ChatPromptTemplate而不是普通列表
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_request}")
    ])

    agent = prompt_template | llm_with_tools

    def specialist_node(state: MultiAgentState):
        console.print(f"--- CALLING {output_key.replace('_report','').upper()} ANALYST ---")
        result = agent.invoke({"user_request": state["user_request"]})
        content = result.content if result.content else f"No direct content, tool calls: {result.tool_calls}"
        return {output_key: content}

    return specialist_node


# 创建专家节点
news_analyst_node = create_specialist_node(
 "你是一个专业新闻分析师. 你的专长是搜索网络以获取最新新闻、文章和社交媒体情绪关于一家公司.",
 "news_report"
)
technical_analyst_node = create_specialist_node(
 "你是一个专业技术分析师. 你专门从事分析股票价格图表、趋势和技术指标.",
 "technical_report"
)
financial_analyst_node = create_specialist_node(
 "你是一个专业财务分析师. 你专门从事解释财务报表和绩效指标.",
 "financial_report"
)

def report_writer_node(state: MultiAgentState):
 """综合专家报告的管理者代理."""
 console.print("--- 调用报告撰写者 ---")
 prompt = f"""你是一个专业财务编辑. 你的任务是将以下专家报告合并为一个专业且连贯的市场分析报告. 添加简短的引言和结论段落.
 
 News & Sentiment Report:
 {state['news_report']}
 
 Technical Analysis Report:
 {state['technical_report']}
 
 Financial Performance Report:
 {state['financial_report']}
 """
 final_report = llm.invoke(prompt).content
 return {"final_report": final_report}

print("专家代理节点和报告撰写者节点已定义.")


# ### 步骤2.2：构建多代理图
# 
# **我们将要做的:**
# 现在我们将专家和管理者连接到图中. 对于这个任务, 专家可以独立工作, 所以我们可以按简单顺序运行它们 (在实际应用中, 这些可以并行运行). 最后一步总是报告撰写者.

# In[6]:


multi_agent_graph_builder = StateGraph(MultiAgentState)

# 添加所有节点
multi_agent_graph_builder.add_node("news_analyst", news_analyst_node)
multi_agent_graph_builder.add_node("technical_analyst", technical_analyst_node)
multi_agent_graph_builder.add_node("financial_analyst", financial_analyst_node)
multi_agent_graph_builder.add_node("report_writer", report_writer_node)

# 定义工作流序列
multi_agent_graph_builder.set_entry_point("news_analyst")
multi_agent_graph_builder.add_edge("news_analyst", "technical_analyst")
multi_agent_graph_builder.add_edge("technical_analyst", "financial_analyst")
multi_agent_graph_builder.add_edge("financial_analyst", "report_writer")
multi_agent_graph_builder.add_edge("report_writer", END)

multi_agent_app = multi_agent_graph_builder.compile()
print("多代理专家团队编译成功。")

# 可视化多代理系统图 - 生成图结构文件
try:
    import os
    current_dir = os.getcwd()
    
    # 生成Mermaid格式
    mermaid_graph = multi_agent_app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(current_dir, "multi_agent_app_graph.mermaid")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_graph)
    print(f"多代理系统图结构已保存为 {mermaid_path}")
    
    # 生成DOT格式
    dot_content = """digraph "Multi-Agent System Graph" {
    rankdir=TD;
    
    // 节点定义
    __start__ [shape=point];
    news_analyst [label="news_analyst", style=filled, fillcolor="#f2f0ff"];
    technical_analyst [label="technical_analyst", style=filled, fillcolor="#f2f0ff"];
    financial_analyst [label="financial_analyst", style=filled, fillcolor="#f2f0ff"];
    report_writer [label="report_writer", style=filled, fillcolor="#f2f0ff"];
    __end__ [label="__end__", shape=doublecircle, style=filled, fillcolor="#bfb6fc"];
    
    // 边定义
    __start__ -> news_analyst;
    news_analyst -> technical_analyst;
    technical_analyst -> financial_analyst;
    financial_analyst -> report_writer;
    report_writer -> __end__;
}
"""
    dot_path = os.path.join(current_dir, "multi_agent_app_graph.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)
    print(f"多代理系统图结构已保存为 {dot_path}")
    
    # 条件化生成PNG
    if graphviz_installed and system_graphviz_available:
        try:
            import graphviz
            g = graphviz.Source.from_file(dot_path)
            g.render(filename="multi_agent_app_graph", directory=current_dir, format="png", cleanup=True)
            print(f"多代理系统图结构已保存为 PNG 图像: {os.path.join(current_dir, 'multi_agent_app_graph.png')}")
        except Exception as png_error:
            print(f"⚠️ 生成多代理系统PNG图像时出错: {png_error}")
    else:
        print("ℹ️ graphviz依赖不完整，仅生成文本格式的多代理系统图文件")
except Exception as e:
    print(f"多代理系统图表可视化失败：{e}")


# ## 阶段3： 正面比较
# 
# Now we'll run 专家团队 在 exact same task 作为单体 agent 和compare final 报告.

# In[7]:


multi_agent_query = f"为...创建简短但全面的市场分析报告 {company}."
initial_multi_agent_input = {"user_request": multi_agent_query}

console.print(f"[bold green]测试 MULTI-AGENT TEAM in the same task:[/bold green] '{multi_agent_query}'")

final_multi_agent_output = multi_agent_app.invoke(initial_multi_agent_input)

console.print("\n--- [bold green]Final Report from多代理 Team[/bold green] ---")
console.print(Markdown(final_multi_agent_output['final_report']))


# **输出讨论:**
# 最终报告的差异是显著的. 多代理团队的输出是：
# - **高度结构化:** 它有清晰、独特的部分用于每个分析领域 因为每个部分都是由具有特定格式指令的专家生成的.
# - **更深入的分析:** 每个部分包含更详细的, 特定领域的语言和见解. 技术分析师谈论移动平均线, 新闻分析师讨论情绪, 财务分析师关注收入和盈利.
# - **更专业：** 最终报告由报告撰写者组装, 读起来像一份专业文档，有清晰的引言、正文和结论。
# 
# 这种定性比较表明，通过在专家团队中分工, 我们实现了一个单一通才代理难以复制的卓越结果.

# ## 阶段4： 定量评估
# 
# 为了正式化比较，我们将使用LLM作为评判者来评分两份报告。标准将关注我们期望在多代理方法中更好的质量，例如结构和分析深度。

# In[8]:


class ReportEvaluation(BaseModel):
 """用于评估财务报告的模式。"""
 clarity_and_structure_score: int = Field(description="1-10分，评估报告的组织、结构和清晰度.")
 analytical_depth_score: int = Field(description="1-10分，评估每个部分分析的深度和质量.")
 completeness_score: int = Field(description="1-10分，评估报告如何全面回应用户请求的所有部分.")
 justification: str = Field(description="对评分的简要理由.")

judge_llm = llm.with_structured_output(ReportEvaluation)

def evaluate_report(query: str, report: str):
 # 获取ReportEvaluation模型的JSON schema
 schema = ReportEvaluation.model_json_schema()
 
 prompt = f"""你是财务分析报告的专业评判者。请根据结构、深度和完整性，按1-10的标准评估以下报告.
 
 **评估标准：**
1. clarity_and_structure_score (1-10分): 评估报告的组织、结构和清晰度
2. analytical_depth_score (1-10分): 评估每个部分分析的深度和质量
3. completeness_score (1-10分): 评估报告如何全面回应用户请求的所有部分
4. justification: 对评分的简要理由
 
 **必须严格按照以下JSON schema返回评估结果：**
{schema}
 
 **注意：**
- 只返回JSON格式的内容，不要添加任何其他文字
- 确保所有评分都是1-10之间的整数
- justification是详细的评估理由，不是空字符串
 
 **原始用户请求：**
 {query}
 
 **待评估报告：**

 {report}
 """
 return judge_llm.invoke(prompt)

console.print("--- 评估单体代理的报告 ---")
mono_agent_evaluation = evaluate_report(mono_query, final_mono_output['messages'][-1].content)
console.print(mono_agent_evaluation.model_dump())

console.print("--- 评估多代理团队的报告 ---")
multi_agent_evaluation = evaluate_report(multi_agent_query, final_multi_agent_output['final_report'])
console.print(multi_agent_evaluation.model_dump())


# **输出讨论:**
# 评判者的评分提供了定量证明 我们的假设. **多代理团队的**报告将获得显著更高的分数, 特别是在`clarity_and_structure_score`和`analytical_depth_score`方面. 来自评判者的理由可能会赞扬清晰的分段和每个部分中详细的专家级分析, 这与单体代理更通用和混乱的输出形成对比.
# 
# 这个评估证实，对于可以分解为专业领域的复杂任务, 多代理架构是生成高质量、结构化和可靠结果的优越方法.

# ## 结论
# 
# 在这个notebook中，我们展示了**多代理系统**相对于单个单体代理在复杂、多方面任务上的明显优势。 通过创建专门代理团队，每个代理都有专注的人格和角色，以及一个管理者来综合他们的工作，我们产生了质量明显更高的最终输出。
# 
# 关键要点是**专业化**的力量。 就像在人类组织中一样，将一个大问题分解并将其部分分配给专家会产生更好的结果。 虽然这种架构在编排方面引入了更多复杂性，但最终输出在结构、深度和专业性方面的显著改进使其成为任何需要跨多个领域提供专家级性能的严肃代理应用程序不可或缺的模式。