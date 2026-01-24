"""
LangGraph orchestration for Doc-Mate agent.
Defines the state machine for reasoning and tool execution.
"""

import operator
from typing import Annotated, List, TypedDict, Dict, Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.flows.agent_tools import ALL_TOOLS
from src.mcp_client.prompts import get_system_prompt
from src.llm.providers import ModelRouter

# -----------------------------------------------------------------------------
# State Definition
# -----------------------------------------------------------------------------

class AgentState(TypedDict):
    """The state passed between nodes in the graph."""
    # List of messages in the conversation. Annotated with operator.add
    # so that new messages are appended rather than overwriting.
    messages: Annotated[List[BaseMessage], operator.add]
    # Configuration passed into the graph
    provider: str
    model: str
    selected_doc_slug: str
    doc_types: set

# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------

def call_model(state: AgentState) -> Dict[str, Any]:
    """
    Reasoning node: Calls the LLM with current messages and tools.
    """
    messages = state["messages"]
    
    # 1. Initialize the correct provider using our ModelRouter logic
    router = ModelRouter()
    llm_provider = router.get_provider(state["provider"])
    llm_provider.model = state["model"]
    
    # 2. Get the underlying LangChain-compatible model
    # Note: We use the langchain_openai/community classes for tool binding
    if state["provider"] == "openai":
        model = ChatOpenAI(
            model=state["model"],
            temperature=0.1
        )
    else:
        # Local Ollama - use langchain_ollama for proper tool support
        from langchain_ollama import ChatOllama
        # Use host.docker.internal to reach native Mac Ollama (fast)
        # Fallback to 'ollama' service name if not on Mac/custom setup
        import os
        base_url = os.getenv("OLLAMA_HOST_URL", "http://host.docker.internal:11434")
        
        model = ChatOllama(
            model=state["model"],
            temperature=0,
            base_url=base_url 
        )
        
    # 3. Bind tools to the model
    model_with_tools = model.bind_tools(ALL_TOOLS)
    
    # 4. Get response
    response = model_with_tools.invoke(messages)
    
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Conditional edge: Determines if we need to call tools or finish.
    """
    last_message = state["messages"][-1]
    
    # If there are tool calls, go to the 'tools' node
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # Otherwise, stop
    return "__end__"

# -----------------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------------

def create_agent_graph():
    """Builds and compiles the LangGraph state machine."""
    
    # 1. Initialize the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # 2. Define the nodes
    workflow.add_node("agent", call_model)
    # ToolNode is a prebuilt node that runs the tools in state['messages'][-1].tool_calls
    workflow.add_node("tools", ToolNode(ALL_TOOLS))
    
    # 3. Set the entry point
    workflow.set_entry_point("agent")
    
    # 4. Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    
    # 5. Tools always loop back to the agent for next reasoning step
    workflow.add_edge("tools", "agent")
    
    # 6. Compile
    return workflow.compile()

# Global compiled graph instance
agent_graph = create_agent_graph()
