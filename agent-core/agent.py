"""LangGraph ReAct agent: Claude reasons, calls tools, then answers.

``create_react_agent`` builds a prebuilt graph: an LLM node alternates with a
tool node until the model returns text without tool calls (or the recursion
limit is hit). ``MessagesPlaceholder`` injects the conversation so the system
prompt stays fixed while chat history grows.

``build_react_agent(..., llm=...)`` accepts an optional model for tests
(``tests.helpers.ScriptedToolModel``) so CI does not call Anthropic.

``build_react_agent(..., tools=...)`` accepts an explicit tool list (same
override pattern as ``llm``). Production callers resolve the real tools from
the sandbox and mathkb MCP servers via ``mcp_client.load_mcp_tools`` and pass
them in; tests pass lightweight in-process stand-ins
(``tests.helpers.make_fake_execute_python_tool``,
``make_fake_search_math_knowledge_tool``) so the graph/checkpointer tests
don't need either MCP server running.

``build_react_agent(..., checkpointer=...)`` wires up LangGraph state
persistence so a conversation can span multiple turns. Callers must invoke the
compiled graph with ``config={"configurable": {"thread_id": ...}}`` for memory
to apply; without a checkpointer the graph stays stateless (one turn in, one
turn out), which is what the existing test suite relies on.
"""

from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from config import Settings

logger = logging.getLogger(__name__)

# Instructions to the model: favor tool use for verification; aligns with sandbox/mathkb tools.
SYSTEM_PROMPT = """You are MathForge, an expert mathematician and Python coder powered by Claude.
Your job is to solve math and coding problems using clear reasoning.
Always:
1. Think step-by-step.
2. If unsure of the right approach or a library's exact API, call
   search_math_knowledge first to check MathForge's reference notes.
3. Write clean, correct Python code.
4. Execute it with the execute_python tool.
5. Verify the result.
6. Give a friendly, educational final answer with explanations, citing the
   source (e.g. "per linear_algebra.md") whenever you used a retrieved note.
Use SymPy for symbolic math, NumPy/SciPy for numerics, Matplotlib for plots.
Never guess — always execute code to confirm."""


def build_llm(settings: Settings) -> BaseChatModel:
    """Instantiate the production Anthropic chat model from ``Settings``."""
    params: dict = {
        "model": settings.model,
        "temperature": settings.temperature,
        "api_key": settings.anthropic_api_key,
    }
    if settings.max_tokens is not None:
        params["max_tokens"] = settings.max_tokens
    return ChatAnthropic(**params)


def build_react_agent(
    settings: Settings,
    llm: BaseChatModel | None = None,
    tools: list[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile and return the LangGraph agent graph.

    Args:
        settings: Used for logging and default LLM construction.
        llm: If provided, used instead of ``ChatAnthropic`` (testing / mocking).
        tools: Tool list for the ReAct loop. Production callers resolve this
            via ``mcp_client.load_mcp_tools`` (async, so it can't default
            here); tests pass a stand-in. Required — raises if omitted.
        checkpointer: If provided, the graph persists message history per
            ``thread_id`` across ``ainvoke``/``astream`` calls (conversation
            memory). Omit for a stateless graph (each call is independent).
    """
    if tools is None:
        msg = "build_react_agent requires tools= (see mcp_client.load_mcp_tools)"
        raise ValueError(msg)

    model = llm or build_llm(settings)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            # Required name must match what create_react_agent expects for chat history.
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    graph = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer,
        debug=False,
    )
    logger.info(
        "MathForge agent built (model=%s, tools=%s, memory=%s)",
        settings.model,
        [t.name for t in tools],
        checkpointer is not None,
    )
    return graph
