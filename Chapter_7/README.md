# 🤖 Agentic RAG — CrewAI, LlamaIndex, LangGraph & Tool Calling

![Architecture Diagram](https://raw.githubusercontent.com/paras160500/Hands-On-RAG-Full/main/Chapter_7/diagram.png)

Part of the [**Hands-On-RAG-Full**](https://github.com/paras160500/Hands-On-RAG-Full) series. Standard RAG pipelines are passive; **Agentic RAG** makes them active. This module covers how to build autonomous agents that don't just retrieve information, but reason about which tools to use, collaborate with other agents, and execute complex workflows to solve user queries.

Each framework approach offers a different level of control and abstraction:

| Framework | Primary Focus | Best For |
|---|---|---|
| 🤝 **CrewAI** | Multi-agent collaboration & role-playing | Complex workflows requiring specialized "experts" |
| 🗂️ **LlamaIndex** | Data-centric agent workflows | Deep integration with complex data indices and retrieval |
| 🦜 **LangGraph** | Fine-grained stateful agent control | Custom ReAct loops and cyclic graph-based logic |
| 🔌 **OpenAI Tools** | Native model-level tool calling | Lightweight applications using built-in model capabilities |

---

## 🗺️ What's Inside

| Notebook | Technique | Tools |
|---|---|---|
| [`crewai-multi-agent.ipynb`](#1-crewai-multi-agentipynb--multi-agent-orchestration) | Multi-Agent Roles, Task Dependencies | `crewai`, `langchain-openai` |
| [`function-agent-llamaindex.ipynb`](#2-function-agent-llamaindexipynb--agentic-workflows) | Function Calling, Agentic Workflows | `llama-index`, `tavily-python` |
| [`react_Agent_with_langchain.ipynb`](#3-react_agent_with_langchainipynb--react-pattern--langgraph) | ReAct Pattern, RAG-as-a-Tool | `langgraph`, `langchain`, `faiss-cpu` |
| [`tool_calling.ipynb`](#4-tool_callingipynb--native-openai-tool-calling) | JSON Schema, Function Definition | `openai` |

---

## 📦 Installation

```bash
pip install crewai llama-index langgraph langchain-openai faiss-cpu python-dotenv tavily-python
```

### 🔑 Environment Variables

Create a `.env` file in this folder:

```env
OPEN_AI_API=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## 🧪 How Each Notebook Works

---

### 1. `crewai-multi-agent.ipynb` — Multi-Agent Orchestration

Uses the `crewai` framework to orchestrate a team of agents. Instead of one LLM doing everything, we define specialized roles that pass work to each other.

#### Defining Specialized Agents

```python
researcher = Agent(
    role="Senior Market Research Analyst",
    goal="Find groundbreaking and emerging trends in AI.",
    backstory="You are an expert analyst with a keen eye for tech shifts...",
    llm=llm
)

writer = Agent(
    role="Technology Content Strategist",
    goal="Craft a compelling blog post based on research findings.",
    backstory="You simplify complex topics into engaging narratives...",
    llm=llm
)
```

| Component | Purpose |
|---|---|
| **Role** | Defines the agent's "personality" and expertise |
| **Goal** | The specific objective the agent is trying to achieve |
| **Backstory** | Provides context and "memory" for the agent's behavior |

---

### 2. `function-agent-llamaindex.ipynb` — Agentic Workflows

Demonstrates the new `AgentWorkflow` in LlamaIndex, which treats agentic behavior as a series of event-driven steps. This allows for highly customizable and transparent agent logic.

#### Tool Definition & Workflow

```python
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow

# Define a tool from a Python function
def search_tool(query: str):
    # Integration with Tavily or local search
    pass

agent_workflow = AgentWorkflow(
    agents=[FunctionAgent(name="research_agent", tools=[search_tool])]
)
```

---

### 3. `react_Agent_with_langchain.ipynb` — ReAct Pattern & LangGraph

Implements the **Reasoning and Acting (ReAct)** loop. The agent "thinks" about the question, decides to use a RAG tool to look up facts in a PDF, and then "acts" by executing that tool.

#### RAG-as-a-Tool

```python
@tool
def rag_gpt_tool(question: str) -> str:
    """Use this tool to answer questions about the 'diabetes' paper."""
    return rag_chain.invoke(question)

# Create the stateful graph-based agent
agent = create_react_agent(llm, tools=[rag_gpt_tool])
```

| Step | Action |
|---|---|
| **Thought** | "I need to find the cost of diabetes in Louisiana." |
| **Action** | Call `rag_gpt_tool` with the specific query. |
| **Observation** | "The cost was $231,000,000." |
| **Final Answer** | Deliver the precise figure to the user. |

---

### 4. `tool_calling.ipynb` — Native OpenAI Tool Calling

The "bare-metal" approach to agents. It shows how to define tools using JSON schemas so that the OpenAI model itself knows when and how to call them.

#### Tool Schema

```json
{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for coordinates",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": { "type": "number" },
            "longitude": { "type": "number" }
        }
    }
}
```

---

## 🧠 Choosing the Right Agent Framework

```
Need multiple agents talking to each other with minimal code?
    └── crewai-multi-agent.ipynb (CrewAI)
         ✅ Easy setup  ✅ Built-in delegation  ⚠️ Harder to customize logic

Need to build a complex, stateful, and cyclic agent?
    └── react_Agent_with_langchain.ipynb (LangGraph)
         ✅ Full control  ✅ Visualizable  ⚠️ Steeper learning curve

Need an agent that lives close to your LlamaIndex data?
    └── function-agent-llamaindex.ipynb (LlamaIndex)
         ✅ Data-native  ✅ Event-driven  ⚠️ Newer ecosystem
```

---

## ⚡ Tech Stack

| Layer | Tool |
|---|---|
| **Multi-Agent** | `CrewAI` |
| **Workflow** | `LlamaIndex AgentWorkflow` |
| **Agent Logic** | `LangGraph` (ReAct) |
| **Vector Store** | `FAISS` |
| **Embeddings** | `OpenAIEmbeddings` |
| **LLM** | `gpt-4o-mini` |

---

## 🔑 Key Learnings

- **Agents > Chains**: While chains are linear, agents can loop and use tools dynamically based on the input, making them far more flexible for real-world RAG.
- **Tools are the Agent's Hands**: A well-defined tool (with a clear description) is the difference between an agent that works and one that hallucinates.
- **Orchestration Matters**: Multi-agent systems like CrewAI allow for separation of concerns, where a "Researcher" doesn't need to know how to "Write" and vice versa.
- **State Management**: LangGraph proves that managing the "state" of an agent (its history, thoughts, and tool outputs) is critical for building reliable production systems.

---

## 🚀 Future Improvements

- Add **Human-in-the-loop** (HITL) to the LangGraph agent for sensitive tool calls.
- Implement **Memory** (Postgres/Redis) so agents remember users across sessions.
- Integrate **Multi-modal tools** (e.g., Image generation or Vision-based search).

---

Part of the [**Hands-On-RAG-Full**](https://github.com/paras160500/Hands-On-RAG-Full) repository.
