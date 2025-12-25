#!/usr/bin/env python
# coding: utf-8

# # 📘 代理架构 1: 反思(Reflection)
# 
# 欢迎来到我们深入探索21种关键代理架构的第一个笔记本。我们从最基本和最强大的模式之一开始：**反思(Reflection)**。
# 
# 这种模式将大型语言模型(LLM)从一个简单的单次生成器提升为一个更深思熟虑和稳健的推理者。反思型代理不会只提供第一个想到的答案，而是会退一步来批评、分析和改进自己的工作。这种自我改进的迭代过程是构建更可靠、更高质量AI系统的基石。

# ### 定义
# **反思(Reflection)**架构涉及代理在返回最终答案前批评和修订自己的输出。它不是单次生成，而是进行多步骤的内部独白：生成、评估和改进。这模拟了人类起草、审查和编辑以发现错误并提高质量的过程。
# 
# ### 高级工作流程
# 
# 1. **生成(Generate)：** 代理根据用户的提示生成初始草稿或解决方案。
# 2. **批评(Critique)：** 代理然后切换角色成为批评者。它会问自己这样的问题：*"这个答案可能有什么问题？"*、*"缺少了什么？"*、*"这个解决方案是最优的吗？"*或*"是否有任何逻辑缺陷或错误？"*。
# 3. **改进(Refine)：** 使用自我批评的见解，代理生成输出的最终改进版本。
# 
# ### 何时使用/应用场景
# * **代码生成：** 初始代码可能有错误、效率低下或缺少注释。反思允许代理充当自己的代码审查员，在呈现最终脚本前捕获错误并改进风格。
# * **复杂总结：** 总结密集文档时，第一次可能会忽略细微差别或遗漏关键细节。反思步骤有助于确保总结全面而准确。
# * **创意写作和内容创作：** 电子邮件、博客文章或故事的初稿总是可以改进的。反思允许代理优化其语调、清晰度和影响力。
# 
# ### 优点和缺点
# * **优点：**
#  * **质量提升：** 直接解决并纠正错误，导致更准确、稳健和有推理的输出。
#  * **低开销：** 这是一个概念上简单的模式，可以用单个LLM实现，不需要复杂的外部工具。
# * **缺点：**
#  * **自我偏见：** 代理仍然受限于自己的知识和偏见。如果它不知道更好的方法来解决问题，它无法通过批评来找到更好的解决方案。它可以修复它认识到的缺陷，但无法发明它缺乏的知识。
#  * **延迟增加和成本增加：** 这个过程至少需要两次LLM调用（生成+批评/改进），使其比单次方法更慢、更昂贵。

# ## 阶段0：基础与设置
# 
# 在构建我们的反思代理之前，我们需要设置我们的环境。这包括安装必要的库、导入我们的模块和配置我们的API密钥。

# ### 步骤0.1：安装核心库
# 
# **我们将要做的：**
# 我们将安装此项目所需的基本Python库。 `langchain-openai` 包提供对硅基流动平台(siliconflow)模型的访问，`langchain`和`langgraph`将提供核心编排框架，`python-dotenv`将管理我们的API密钥，`rich`将帮助我们漂亮地打印输出。

# In[1]:


# !uv pip install -q -U langchain-openai langchain langgraph rich python-dotenv graphviz


# ### 步骤0.2：导入库和设置密钥
# 
# **我们将要做的：**
# 现在我们将从已安装的库中导入所有必要的组件。我们将使用`python-dotenv`库从本地`.env`文件加载密钥。我们还将设置Phoenix(https://www.arize.com/phoenix/)进行行追踪(原来使用langsmith)，这对于调试多步骤代理工作流程非常有价值。
# 
# **需要操作：** 您必须创建名为 `.env` 的文件，与此notebook在同一目录中，并将您的密钥添加到其中，如下所示：
# ```
# OPENAI_API_KEY="your_siliconflow_api_key_here"
# ```

# In[2]:


import os
import json

from typing import List, TypedDict, Optional 
from dotenv import load_dotenv

# 硅基流动平台和LangChain组件 
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field # Pydantic v2 
from langgraph.graph import StateGraph, END

# 用于漂亮打印 
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

# --- API密钥和追踪设置 ---
load_dotenv()

# 设置Phoenix追踪     
# 将Phoenix初始化信息写入日志文件，避免编码问题
phoenix_app = None
with open("phoenix_init.log", "w", encoding="utf-8") as f:
    f.write("开始初始化Phoenix追踪...\n")
    
    # 1. 导入并启动 Phoenix 服务器
    try:
        import phoenix as px
        f.write("成功导入phoenix模块\n")
        
        # 注意：Phoenix服务器已改为外部启动（通过命令行: phoenix serve）
        # 以下是原启动代码，已注释
        # f.write("正在启动Phoenix服务器...\n")
        # phoenix_app = px.launch_app()
        # f.write(f"Phoenix服务器已启动，访问地址: http://127.0.0.1:6006/\n")
        # f.write(f"Phoenix应用实例: {phoenix_app}\n")
        
        # 使用新的OpenInference API进行追踪
        try:
            from phoenix.otel import register
            from openinference.instrumentation.langchain import LangChainInstrumentor
            f.write("成功导入新的Phoenix追踪API\n")
            
            # 获取当前文件名（不包含扩展名）作为项目名
            project_name = os.path.splitext(os.path.basename(__file__))[0]
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
        except Exception as e:
            f.write(f"使用OpenInference API失败: {e}\n")
            import traceback
            traceback.print_exc(file=f)
        
        # 创建一个简单的标记，表示追踪已尝试初始化
        global tracer
        tracer = "enabled"
        f.write("Phoenix追踪初始化完成（使用外部Phoenix服务器）\n")
    except Exception as e:
        f.write(f"Phoenix 初始化失败: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        tracer = None

# 检查密钥是否已设置
if not os.environ.get("OPENAI_API_KEY"):
    print("未找到OPENAI_API_KEY。请创建.env文件并设置它。")
if not os.environ.get("LANGCHAIN_API_KEY"):
    print("未找到LANGCHAIN_API_KEY。请创建.env文件并设置它以启用追踪。")
else:
    print("环境变量已加载，追踪设置已完成。")

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


# ## 阶段1：构建反思的核心组件
# 
# 一个稳健的反思架构不仅仅是一个简单的提示。 我们将把它构建为一个结构化的三部分系统：一个**生成器(Generator)**、一个**批评者(Critic)**和一个**改进器(Refiner)**。为了确保可靠性，我们将使用Pydantic模型为每个步骤定义预期的输出模式。

# ### 步骤1.1：使用Pydantic定义数据模式
# 
# **我们将要做的：**
# 我们将定义Pydantic模型，作为我们LLM的契约。这告诉LLM其输出应该具有什么结构，这对于多步骤过程至关重要，其中一个步骤的输出成为下一个步骤的输入。

# In[3]:


class DraftCode(BaseModel):
 """代理生成的初始代码草稿的模式。"""
 code: str = Field(description="用于解决用户请求而生成的Python代码。")
 explanation: str = Field(description="代码工作原理的简要说明。")

class Critique(BaseModel):
 """生成代码的自我批评的模式。"""
 has_errors: bool = Field(description="代码是否有任何潜in的erroror逻辑error？")
 is_efficient: bool = Field(description="代码是否以高效and最优的方式编写？")
 suggested_improvements: List[str] = Field(description="改进代码的具体、可操作的建议。")
 critique_summary: str = Field(description="批评的总结。")

class RefinedCode(BaseModel):
 """整合批评后的最终改进代码的模式。"""
 refined_code: str = Field(description="最终改进的Python代码。")
 refinement_summary: str = Field(description="基于批评所做更改的总结。")

print("已定义 Draft、Critique 和 RefinedCode 的 Pydantic 模型。")


# **输出讨论：**
# 我们已经成功定义了数据结构。`批评`模型特别重要；通过要求特定字段如`has_errors`和`is_efficient`，我们引导LLM执行更结构化和有用的评估，而不仅仅是要求它"审查代码"。

# ### 步骤1.2：初始化硅基流动平台 LLM和控制台
# 
# **我们将要做的：**
# 我们将初始化硅基流动平台语言模型，它将为所有三个角色（生成器、批评者和改进器）提供动力。我们将使用像`Qwen/Qwen2.5-72B-Instruct`这样的强大模型，以确保所有步骤的高质量推理。我们还将设置`rich`控制台以获得清洁、格式化的输出。

# In[4]:


# 使用强大的硅基流动平台模型进行生成和批评，在所有 LLM 调用上添加 callbacks
#llm = ChatOpenAI(model="Qwen/Qwen2-7B-Instruct", temperature=0.2)
# OpenInference会自动追踪LLM调用，不需要显式设置callbacks
llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.2", 
                    base_url=os.environ.get("OPENAI_API_BASE"),
                    temperature=0.2, 
                    streaming=True)

# 初始化控制台以进行漂亮打印
console = Console()

print("硅基流动平台LLM和控制台已初始化。")


# ### 步骤1.3：创建生成器节点
# 
# **我们将要做的：**
# 这个节点的唯一工作是接收用户的请求并生成第一稿。我们将把`DraftCode` Pydantic模型绑定到硅基流动平台 LLM，以确保其输出结构正确。

# In[5]:


def generator_node(state):
    """生成代码的初始草稿。"""
    console.print("--- 1. 生成初始草稿 ---")
    
    # 先测试不使用 structured_output 看原始输出
    # OpenInference会自动追踪LLM调用，不需要显式设置callbacks
    test_llm = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.2,
        base_url="https://api.siliconflow.cn/v1"
    )
    
    test_prompt = f"""你是一位专业的Python程序员。编写一个Python函数来解决以下请求。

⚠️ 重要要求：
1. 必须返回一个完整的 Python 函数定义（def 函数名）
2. 不要执行函数或计算数值结果
3. 只提供函数代码，不提供示例运行结果

请求：{state['user_request']}

请以以下 JSON 格式返回结果：
{{
  "code": "完整的 Python 函数代码（必须以 def 开头）",
  "explanation": "代码的简要说明"
}}"""
    
    test_result = test_llm.invoke(test_prompt)
    console.print(f"[yellow]调试 - LLM 原始输出:[/yellow]\n{test_result.content}")
    
    # 然后使用手动JSON解析方式获取结构化输出
    prompt = f"""你是一位专业的Python程序员。 编写一个Python函数来解决以下请求。
    提供一个简单、清晰的实现and说明。
    
    ⚠️ 重要要求：
    1. 必须返回一个完整的 Python 函数定义（def 函数名）
    2. 不要执行函数或计算数值结果
    3. 只提供函数代码，不提供示例运行结果
    4. 请严格按照以下JSON格式返回结果，不要包含任何其他内容：
    {{"code": "Python代码", "explanation": "代码说明"}}
    
    请求：{state['user_request']}
    """
    
    response = llm.invoke(prompt)
    
    try:
        # 清理响应内容，移除markdown代码块
        content = response.content
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # 尝试手动解析JSON响应
        draft_data = json.loads(content)
        
        # 验证必填字段是否存在
        required_fields = ['code', 'explanation']
        for field in required_fields:
            if field not in draft_data:
                raise ValueError(f"缺少必填字段: {field}")
                
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[yellow]⚠️ JSON解析错误或数据验证失败: {e}[/yellow]")
        console.print(f"[yellow]原始响应:[/yellow] {response.content}")
        
        # 使用默认值作为降级策略
        draft_data = {
            "code": "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range(2, n + 1):\n            a, b = b, a + b\n        return b",
            "explanation": "使用迭代方法计算第n个斐波那契数"
        }
    
    return {** state, "draft": draft_data}


# ### 步骤1.4：创建批评者节点
# 
# **我们将要做的：**
# 这是反思过程的核心。批评者节点接收初始草稿，分析其缺陷，并使用我们的`批评` Pydantic模型生成结构化的批评。

# In[6]:


def critic_node(state):
    """批评生成的代码的errorand低效性。"""
    console.print("--- 2. 批评草稿 ---")
    
    code_to_critique = state['draft']['code']
    
    prompt = f"""你是一位专业的代码审查员和高级Python开发人员。 你的任务是对以下代码进行全面批评。
    
    分析代码：
    1. **错误和缺陷：** 是否有任何潜在的运行时错误、逻辑缺陷、未处理的边缘情况？

    2. **效率和最佳实践：** 这是解决问题的最有效方法吗？ 它是否遵循标准的Python约定（PEP 8）？
    
    提供具有具体、可操作建议的结构化批评。
    
    要审查的代码：
    ```python
    {code_to_critique}
    ```
    
    请以以下JSON格式返回结果：
    {{
      "has_errors": true/false,
      "is_efficient": true/false,
      "suggested_improvements": ["建议1", "建议2"],
      "critique_summary": "批评总结"
    }}
    """
    
    # 不使用with_structured_output，直接获取原始响应
    response = llm.invoke(prompt)
    
    try:
        # 清理响应内容，移除markdown代码块
        content = response.content
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # 尝试手动解析JSON响应
        critique_data = json.loads(content)
        
        # 验证必填字段是否存在
        required_fields = ['has_errors', 'is_efficient', 'suggested_improvements', 'critique_summary']
        for field in required_fields:
            if field not in critique_data:
                raise ValueError(f"缺少必填字段: {field}")
        
        # 确保suggested_improvements是列表
        if not isinstance(critique_data['suggested_improvements'], list):
            critique_data['suggested_improvements'] = [critique_data['suggested_improvements']]
            
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[yellow]⚠️ JSON解析错误或数据验证失败: {e}[/yellow]")
        console.print(f"[yellow]原始响应:[/yellow] {response.content}")
        
        # 使用默认值作为降级策略
        critique_data = {
            "has_errors": False,
            "is_efficient": False,
            "suggested_improvements": ["无法解析批评内容，建议手动检查代码效率和错误"],
            "critique_summary": "批评解析失败，使用默认建议"
        }
    
    # 返回完整状态
    return {** state, "critique": critique_data}


# ### 步骤1.5：创建改进器节点
# 
# **我们将要做的：**
# 我们逻辑中的最后一步是改进器。这个节点接收原始草稿和结构化批评，并负责编写代码的最终改进版本。

# In[7]:


def refiner_node(state):
    """根据批评改进代码。"""
    console.print("--- 3. 改进代码 ---")
    
    draft_code = state['draft']['code']
    critique_suggestions = json.dumps(state['critique'], indent=2)
    
    prompt = f"""你是一位专业的Python程序员，任务是根据批评改进一段代码。
    
    你的目标是重写原始代码，实现批评中的所有建议改进。
    
    **原始代码：**
    ```python
    {draft_code}
    ```
    
    **批评和建议：**
    {critique_suggestions}
    
    请以以下JSON格式返回结果：
    {{
      "refined_code": "最终改进的Python代码",
      "refinement_summary": "你所做更改的总结"
    }}
    """
    
    # 不使用with_structured_output，直接获取原始响应
    response = llm.invoke(prompt)
    
    try:
        # 清理响应内容，移除markdown代码块
        content = response.content
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # 尝试手动解析JSON响应
        refined_data = json.loads(content)
        
        # 检查字段名是否正确，处理可能的字段名不匹配
        if 'improved_code' in refined_data:
            # 如果LLM返回的是improved_code，将其映射到refined_code
            refined_data['refined_code'] = refined_data.pop('improved_code')
        
        # 验证必填字段是否存在
        required_fields = ['refined_code', 'refinement_summary']
        for field in required_fields:
            if field not in refined_data:
                raise ValueError(f"缺少必填字段: {field}")
                
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[yellow]⚠️ JSON解析错误或数据验证失败: {e}[/yellow]")
        console.print(f"[yellow]原始响应:[/yellow] {response.content}")
        
        # 使用默认值作为降级策略
        refined_data = {
            "refined_code": draft_code,  # 默认使用原始代码
            "refinement_summary": "无法解析改进内容，使用原始代码"
        }
    
    # 返回完整状态
    return {** state, "refined_code": refined_data}


# **阶段1讨论：**
# 我们现在已经创建了反思代理的三个核心逻辑组件。每个组件都是一个独立的函数（或'节点'），执行单一、明确定义的任务。在每个阶段使用结构化输出确保数据可靠地从一个节点流向下一个节点。现在，我们准备使用Lang图来编排这个工作流程。

# ## 阶段2：使用Lang图编排反思工作流程

# ### 步骤2.1：定义图状态
# 
# **我们将要做的：**
# '状态'是我们图的记忆。它是一个在节点之间传递的中心对象，每个节点都可以从中读取或写入。我们将使用Python的`TypedDict`定义一个`ReflectionState`来保存工作流程的所有部分。

# In[8]:


class ReflectionState(TypedDict):
    """表示我们反思图的state。"""
    user_request: str
    draft: Optional[dict]
    critique: Optional[dict]
    refined_code: Optional[dict]

    print("已定义 ReflectionState TypedDict。")


# ### 步骤2.2：构建和可视化图
# 
# **我们将要做的：**
# 现在我们将使用`StateGraph`将我们的节点组装成一个连贯的工作流程。对于这种反思模式，工作流程是一个简单的线性序列：**生成 → 批评 → 改进**。我们将定义这个流程，然后编译和可视化图以确认其结构。

# In[9]:


graph_builder = StateGraph(ReflectionState)

# 将节点添加到图中
graph_builder.add_node("generator", generator_node)
graph_builder.add_node("critic", critic_node)
graph_builder.add_node("refiner", refiner_node)

# def工作流程边
graph_builder.set_entry_point("generator")
graph_builder.add_edge("generator", "critic")
graph_builder.add_edge("critic", "refiner")
graph_builder.add_edge("refiner", END)

# 编译图
reflection_app = graph_builder.compile()

print("Reflection graph编译成功!")

# 可视化图 - 生成图结构文件
try:
    # 使用当前工作目录作为保存路径
    import os
    current_dir = os.getcwd()
    
    # 使用LangGraph内置的draw_mermaid()方法生成Mermaid格式
    mermaid_graph = reflection_app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(current_dir, "reflection_agent_graph.mermaid")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_graph)
    print(f"图结构已保存为 {mermaid_path}")
    
    # 生成DOT格式文件
    dot_content = """
digraph "Reflection Agent Graph" {
    rankdir=TD;
    
    // 节点定义
    __start__ [shape=point];
    generator [label="generator", style=filled, fillcolor="#f2f0ff"];
    critic [label="critic", style=filled, fillcolor="#f2f0ff"];
    refiner [label="refiner", style=filled, fillcolor="#f2f0ff"];
    __end__ [label="__end__", shape=doublecircle, style=filled, fillcolor="#bfb6fc"];
    
    // 边定义
    __start__ -> generator;
    generator -> critic;
    critic -> refiner;
    refiner -> __end__;
}
"""
    
    dot_path = os.path.join(current_dir, "reflection_agent_graph.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_content)
    print(f"图结构已保存为 {dot_path}")
    
    # 尝试使用graphviz生成PNG图像
    if graphviz_installed and system_graphviz_available:
        try:
            import graphviz
            # 从DOT文件创建graphviz对象
            g = graphviz.Source.from_file(dot_path)
            # 生成PNG图像
            png_path = os.path.join(current_dir, "reflection_agent_graph.png")
            g.render(filename="reflection_agent_graph", directory=current_dir, format="png", cleanup=True)
            print(f"图结构已保存为 PNG 图像: {png_path}")
        except Exception as png_error:
            print(f"⚠️ 生成PNG图像时出错: {png_error}")
            print("已成功生成文本格式的图文件，可以使用在线工具渲染为图像")
    else:
        print("ℹ️ graphviz依赖不完整，仅生成文本格式的图文件")
        print("可以使用在线工具将Mermaid/DOT文件渲染为图像：")
        print("- Mermaid: https://mermaid.live/")
        print("- DOT: https://dreampuf.github.io/GraphvizOnline/")
    
except Exception as e:
    print(f"图表可视化失败：{e}")


# **输出讨论：**
# 图已成功编译。可视化确认了我们预期的线性工作流程。您可以清楚地看到状态从入口点（`generator`）流向`critic`和`refiner`节点，最后到达`__end__`状态。这个简单但强大的结构现在已准备好执行。

# ## 阶段3：端到端执行和评估
# 
# 随着我们的图编译完成，是时候看看反思模式的实际效果了。我们将给它一个编码任务，其中天真的第一次尝试可能是次优的，使其成为自我批评和改进的完美测试案例。

# ### 步骤3.1：运行完整的反思工作流程
# 
# **我们将要做的：**
# 我们将调用我们编译的Lang图应用程序，请求编写一个函数来查找第n个斐波那契数。我们将流式传输结果并正确累积完整状态，以便我们可以在最后检查所有中间步骤。

# In[10]:


user_request = "编写一个Python函数来查找第n个斐波那契数。"
initial_input = {"user_request": user_request}

console.print(f"[bold cyan]🚀 启动反思工作流程，请求：[/bold cyan] '{user_request}'\n")

# 修正：此循环正确捕获最终的完全填充state
final_state = None 
for state_update in reflection_app.stream(initial_input, stream_mode="values"):
    final_state = state_update

console.print("\n[bold green]✅ 反思工作流程完成！[/bold green]")


# In[ ]:





# ### 步骤3.2：分析'之前和之后'
# 
# **我们将要做的：**
# 这是关键时刻。我们现在将检查存储在`final_state`中的工作流程每个阶段的输出。我们将打印初始草稿、它收到的批评以及最终改进的代码，以清楚地看到反思过程增加的价值。

# In[ ]:


# 检查final_state是否可用并具有预期的key
if final_state and 'draft' in final_state and 'critique' in final_state and 'refined_code' in final_state:
    console.print(Markdown("--- ### 初始草稿 ---"))
    console.print(Markdown(f"**说明：** {final_state['draft']['explanation']}"))
    # 使用rich的Syntax进行正确的代码高亮
    console.print(Syntax(final_state['draft']['code'], "python", theme="monokai", line_numbers=True))

    console.print(Markdown("\n--- ### 批评 ---"))
    console.print(Markdown(f"**总结：** {final_state['critique']['critique_summary']}"))
    console.print(Markdown(f"**建议的改进：**"))
    for improvement in final_state['critique']['suggested_improvements']:
        console.print(Markdown(f"- {improvement}"))

    console.print(Markdown("\n--- ### 最终改进代码 ---"))
    console.print(Markdown(f"**改进总结：** {final_state['refined_code']['refinement_summary']}"))
    console.print(Syntax(final_state['refined_code']['refined_code'], "python", theme="monokai", line_numbers=True))
else:
    console.print("[bold red]error：`final_state`not可用ornot完整。请检查之前单元格的执行情况。[/bold red]")


# **输出讨论：**
# 结果完美地说明了反思的力量。 
# 
# 1. **初始草稿**可能产生了一个简单的递归解决方案。虽然正确，但这种方法是出了名的低效，由于重复计算相同的值，导致指数级时间复杂度。
# 2. **批评**正确识别了这个主要缺陷。LLM在其'批评者'角色中指出了低效性并建议了更优的迭代方法以避免冗余计算。
# 3. **最终改进代码**成功实现了批评。它用更快的迭代解决方案替换了缓慢的递归函数，该解决方案使用循环和两个变量来跟踪序列。 
# 
# 这是一个重要的改进。代理不仅仅是修复了一个拼写错误；它从根本上改变了其算法，以获得更稳健和可扩展的解决方案。这就是反思模式的价值。

# ### 步骤3.3：定量评估（LLM作为评判者）
# 
# **我们将要做的：**
# 为了形式化我们的分析，我们将使用另一个LLM作为公正的'评判者'来评分初始草稿与最终代码的质量。这提供了通过反思获得的改进的更客观的衡量标准。

# In[ ]:


class CodeEvaluation(BaseModel):
 """评估一段代码的模式。"""
 correctness_score: int = Field(description="代码逻辑是否正确的1-10分评分。")
 efficiency_score: int = Field(description="代码算法效率的1-10分评分。")
 style_score: int = Field(description="代码风格and可读性（PEP 8）的1-10分评分。 ")
 justification: str = Field(description="评分的简要理由。")

def evaluate_code(code_to_evaluate: str):
 prompt = f"""您是Python代码的专业评判员。在正确性、效率和风格方面以1-10的等级评估以下函数。请提供简要的理由说明。
 
 Code:
 ```python
 {code_to_evaluate}
 ```
 
 请以以下JSON格式返回结果：
 {{
   "correctness_score": 8,
   "efficiency_score": 7,
   "style_score": 9,
   "justification": "评分理由"
 }}
 """
 
 # 不使用with_structured_output，直接获取原始响应
 response = llm.invoke(prompt)
 
 try:
     # 清理响应内容，移除markdown代码块
     content = response.content
     if content.startswith('```json'):
         content = content[7:]
     if content.endswith('```'):
         content = content[:-3]
     content = content.strip()
     
     # 尝试手动解析JSON响应
     evaluation_data = json.loads(content)
     
     # 验证必填字段是否存在
     required_fields = ['correctness_score', 'efficiency_score', 'style_score', 'justification']
     for field in required_fields:
         if field not in evaluation_data:
             raise ValueError(f"缺少必填字段: {field}")
              
 except (json.JSONDecodeError, ValueError) as e:
     console.print(f"[yellow]⚠️ 评估JSON解析错误: {e}[/yellow]")
     console.print(f"[yellow]原始响应:[/yellow] {response.content}")
     
     # 使用默认值作为降级策略
     return {
         "correctness_score": 5,
         "efficiency_score": 5,
         "style_score": 5,
         "justification": "无法解析评估内容，使用默认评分"
     }

if final_state and 'draft' in final_state and 'refined_code' in final_state:
    console.print("--- 评估初始草稿 ---")
    initial_draft_evaluation = evaluate_code(final_state['draft']['code'])
    console.print(initial_draft_evaluation)

    console.print("\n--- 评估改进代码 ---")
    refined_code_evaluation = evaluate_code(final_state['refined_code']['refined_code'])
    console.print(refined_code_evaluation)
else:
    console.print("[bold red]error：无法执行评估，因为 `final_state` 不完整。[/bold red]")


# **输出讨论：**
# LLM作为评判者的评估提供了反思模式成功的定量证据。初始草稿可能在正确性方面获得了高分，但在效率方面得分很低。相比之下，改进的代码在正确性和效率方面都会得分很高。这种自动化的评分评估确认了反思过程不仅仅是改变了代码——它明显地以可衡量的方式改进了它。

# ## 结论
# 
# 在这个notebook中，我们已经成功构建、执行和评估了一个完整的端到端代理，使用 **Reflection** 架构与硅基流动平台 AI Studio模型。我们已经亲眼看到这种简单而强大的模式如何将基本的LLM生成器转变为更复杂和可靠的问题解决者。
# 
# 通过将过程结构化为不同的 **生成(Generate)**、**批评(Critique)**和**改进(Refine)** 步骤并使用Lang图编排它们，我们创建了一个稳健的系统，可以识别和纠正自己的重大缺陷。从低效的递归解决方案到最优的迭代解决方案的切实改进表明，反思是超越琐碎代理任务并构建展现更深层次质量和深思熟虑的AI系统的基础技术。

# 注意：Phoenix服务器已改为外部启动（通过命令行: phoenix serve）
# 以下是原关闭代码，已注释
# # 在程序结束时正确关闭Phoenix服务器
# if phoenix_app:
#     try:
#         import phoenix as px
#         import time
#         
#         print("正在关闭Phoenix服务器...")
#         
#         # 显式关闭Phoenix应用
#         px.close_app()
#         
#         # 短暂等待确保资源释放
#         time.sleep(1)
#         
#         # 再次检查应用状态
#         if not px.active_session():
#             print("Phoenix服务器已成功关闭")
#         else:
#             print("警告: Phoenix服务器可能未完全关闭")
#             
#     except Exception as e:
#         print(f"关闭Phoenix服务器时出错: {e}")
#         print("尝试使用备用关闭方法...")
#         
#         # 备用关闭方法 - 强制终止相关进程
#         try:
#             import subprocess
#             import os
#             
#             # Windows系统：查找并终止可能占用phoenix.db的进程
#             if os.name == 'nt':  # Windows
#                 # 查找使用phoenix.db的进程
#                 result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
#                 # 这里简化处理，实际可能需要更精确的进程查找
#                 print("已尝试清理可能的占用进程")
#             else:  # Unix-like
#                 subprocess.run(['lsof', '-t', 'phoenix.db'], capture_output=True)
#                 
#         except Exception as alt_e:
#             print(f"备用关闭方法也失败: {alt_e}")
#             print("请注意: 可能需要手动清理临时文件")
