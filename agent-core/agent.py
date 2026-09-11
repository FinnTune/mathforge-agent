"""Hand-rolled LangGraph agent: planner -> tools -> verifier -> responder.

Four nodes instead of a prebuilt two-node ReAct loop, each with a distinct
job:

- ``planner`` — tools-bound LLM call. Reasons about the problem and calls
  ``execute_python``/``search_math_knowledge`` as needed. Loops with
  ``tools`` (``langgraph.prebuilt.ToolNode``, reused as-is — it already runs
  every ``tool_call`` in one ``AIMessage`` concurrently, which is what
  "parallel tools" means here) until it stops requesting tools.
- ``verifier`` — separate, non-tool-bound-for-execution LLM call that reviews
  the planner's tool outputs and draft summary for correctness/completeness.
  Forces a structured decision by binding a single pydantic tool
  (``VerificationDecision``) with ``tool_choice`` set to it, then parses
  ``response.tool_calls[0]["args"]`` — the same tool-calling mechanism the
  planner already uses, deliberately not ``.with_structured_output()``.
  Routes via ``Command(goto=...)``: insufficient (and retries remain) sends
  feedback back to ``planner``; otherwise proceeds to ``responder``. Capped
  by ``settings.verification_max_attempts`` so the loop can't run forever.
- ``responder`` — separate LLM call (not tool-bound) that writes the actual
  user-facing, friendly, cited final answer from the full transcript. Its
  output is ``messages[-1]``, same as before this rewrite.

``build_agent_graph(..., llm=...)`` accepts an optional model for tests
(``tests.helpers.ScriptedToolModel``) so CI does not call Anthropic. Because
every node shares one model instance, a scripted test just needs one
``AIMessage`` per node visited, in call order (planner, [verifier, planner]*,
verifier, responder) — see ``tests/test_agent.py``.

``build_agent_graph(..., tools=...)`` accepts an explicit tool list (same
override pattern as ``llm``). Production callers resolve the real tools from
the sandbox and mathkb MCP servers via ``mcp_client.load_mcp_tools`` and pass
them in; tests pass lightweight in-process stand-ins
(``tests.helpers.make_fake_execute_python_tool``,
``make_fake_search_math_knowledge_tool``) so the graph/checkpointer tests
don't need either MCP server running.

``build_agent_graph(..., checkpointer=...)`` wires up LangGraph state
persistence so a conversation can span multiple turns (and, via
``checkpointer.build_checkpointer`` in production, survive a restart).
Callers must invoke the compiled graph with
``config={"configurable": {"thread_id": ...}}`` for memory to apply; without
a checkpointer the graph stays stateless (one turn in, one turn out), which
is what the existing test suite relies on.
"""

from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

from config import Settings

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are MathForge's planner. Your job is to work \
out how to solve the user's math or coding problem by reasoning step-by-step \
and using tools — you do NOT write the final answer shown to the user (a \
separate responder does that once your work is verified).
Always:
1. Think step-by-step about what's actually being asked.
2. If unsure of the right approach or a library's exact API, call \
search_math_knowledge to check MathForge's reference notes.
3. Write clean, correct Python and execute it with the execute_python tool. \
Never guess a numeric or symbolic result — always confirm by running code.
4. Once you have executed and confirmed everything needed, stop calling \
tools and write a short internal summary of what you found and how (this is \
a working note for the verifier, not the user-facing answer)."""

VERIFIER_SYSTEM_PROMPT = """You are MathForge's verifier. Review the \
conversation above — the user's question, the planner's tool calls and \
their results, and the planner's summary — and decide whether it is \
sufficient to answer the user's question correctly and completely.
Call the VerificationDecision tool with your decision:
- sufficient=True if the reasoning and tool outputs fully and correctly \
answer the question.
- sufficient=False if something is missing, looks wrong, or a claimed \
result was never actually confirmed by running code — set feedback to a \
specific, actionable instruction for what the planner should do next."""

RESPONDER_SYSTEM_PROMPT = """You are MathForge, writing the final answer for \
the user. Using the full conversation above (the question, tool calls and \
results, and the planner's summary), write a friendly, educational final \
answer with clear explanations. Cite the source file (e.g. "per \
linear_algebra.md") whenever you used a note retrieved via \
search_math_knowledge. Do not call any tools — just answer."""


class VerificationDecision(BaseModel):
    """Whether the planner's work is sufficient to answer the user, or needs another pass."""

    sufficient: bool = Field(
        description=(
            "True if the planner's reasoning and tool outputs fully and "
            "correctly answer the question."
        )
    )
    feedback: str = Field(
        default="",
        description=(
            "If not sufficient, specific actionable feedback for the planner. "
            "Empty if sufficient."
        ),
    )


class AgentState(MessagesState):
    """Extends the standard messages-only state with the verifier retry counter."""

    verification_attempts: int


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


def build_agent_graph(
    settings: Settings,
    llm: BaseChatModel | None = None,
    tools: list[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile and return the planner/tools/verifier/responder graph.

    Args:
        settings: Used for logging, default LLM construction, and the
            verifier's retry cap (``verification_max_attempts``).
        llm: If provided, used instead of ``ChatAnthropic`` (testing / mocking).
        tools: Tool list for the planner. Production callers resolve this via
            ``mcp_client.load_mcp_tools`` (async, so it can't default here);
            tests pass stand-ins. Required — raises if omitted.
        checkpointer: If provided, the graph persists message history per
            ``thread_id`` across ``ainvoke``/``astream`` calls (conversation
            memory). Omit for a stateless graph (each call is independent).
    """
    if tools is None:
        msg = "build_agent_graph requires tools= (see mcp_client.load_mcp_tools)"
        raise ValueError(msg)

    model = llm or build_llm(settings)
    tool_model = model.bind_tools(tools)
    verify_model = model.bind_tools([VerificationDecision], tool_choice="VerificationDecision")
    max_attempts = settings.verification_max_attempts

    async def planner(state: AgentState) -> dict:
        messages = [SystemMessage(PLANNER_SYSTEM_PROMPT), *state["messages"]]
        response = await tool_model.ainvoke(messages)
        return {"messages": [response]}

    def route_after_planner(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "verifier"

    async def verifier(state: AgentState) -> Command:
        # The transcript so far ends on the planner's own (assistant-role)
        # draft — Claude requires the last message to be user-role to
        # generate a fresh turn (it's not "prefill"-capable), so an explicit
        # trailing HumanMessage is required here, not just a nicety.
        messages = [
            SystemMessage(VERIFIER_SYSTEM_PROMPT),
            *state["messages"],
            HumanMessage("Please verify the planner's work above."),
        ]
        response = await verify_model.ainvoke(messages)
        decision = VerificationDecision(**response.tool_calls[0]["args"])
        attempts = state.get("verification_attempts", 0)

        if decision.sufficient or attempts >= max_attempts:
            return Command(goto="responder")

        feedback = HumanMessage(
            content=f"Verifier feedback — address this and continue: {decision.feedback}"
        )
        return Command(
            goto="planner",
            update={"messages": [feedback], "verification_attempts": attempts + 1},
        )

    async def responder(state: AgentState) -> dict:
        # Same reasoning as verifier: the transcript ends on the planner's
        # own draft (assistant-role), so a trailing HumanMessage is required
        # for Claude to generate a fresh turn here, not just style.
        messages = [
            SystemMessage(RESPONDER_SYSTEM_PROMPT),
            *state["messages"],
            HumanMessage("Please write the final answer for the user now."),
        ]
        response = await model.ainvoke(messages)
        # Reset the counter here (not at the top of a turn) so a fresh turn
        # sharing the same thread_id/checkpoint starts its own retry budget.
        return {"messages": [response], "verification_attempts": 0}

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("verifier", verifier, destinations=("planner", "responder"))
    graph.add_node("responder", responder)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner", route_after_planner, {"tools": "tools", "verifier": "verifier"}
    )
    graph.add_edge("tools", "planner")
    graph.add_edge("responder", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "MathForge agent graph built (model=%s, tools=%s, memory=%s, verification_max_attempts=%d)",
        settings.model,
        [t.name for t in tools],
        checkpointer is not None,
        max_attempts,
    )
    return compiled
