"""LangGraph ReAct agent (Claude + code execution tool)."""

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
    """Construct the chat model from settings."""
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
    """Create the compiled LangGraph agent."""
    model = llm or build_llm(settings)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
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
