"""LangGraph ReAct agent: Claude reasons, calls tools, then answers.

``create_react_agent`` builds a prebuilt graph: an LLM node alternates with a
tool node until the model returns text without tool calls (or the recursion
limit is hit). ``MessagesPlaceholder`` injects the conversation so the system
prompt stays fixed while chat history grows.

``build_react_agent(..., llm=...)`` accepts an optional model for tests
(``tests.helpers.ScriptedToolModel``) so CI does not call Anthropic.
"""

from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from config import Settings
from tools import execute_python_code

logger = logging.getLogger(__name__)

# Instructions to the model: favor tool use for verification; aligns with sandbox capabilities.
SYSTEM_PROMPT = """You are MathForge, an expert mathematician and Python coder powered by Claude.
Your job is to solve math and coding problems using clear reasoning.
Always:
1. Think step-by-step.
2. Write clean, correct Python code.
3. Execute it with the execute_python_code tool.
4. Verify the result.
5. Give a friendly, educational final answer with explanations.
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
) -> CompiledStateGraph:
    """Compile and return the LangGraph agent graph.

    Args:
        settings: Used for logging and default LLM construction.
        llm: If provided, used instead of ``ChatAnthropic`` (testing / mocking).
    """
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
        tools=[execute_python_code],
        prompt=prompt,
        debug=False,
    )
    logger.info("MathForge agent built (model=%s)", settings.model)
    return graph
