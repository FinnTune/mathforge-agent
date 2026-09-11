"""Tests for ``agent.build_agent_graph`` and ``build_llm``.

Uses ``ScriptedToolModel`` so no Anthropic API calls occur. Every node
(planner, verifier, responder) shares one model instance, so a scripted test
scripts one ``AIMessage`` per node visited, in call order — see the module
docstring in ``agent.py`` for why.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import build_agent_graph
from tests.helpers import (
    ScriptedToolModel,
    make_fake_execute_python_tool,
    make_fake_search_math_knowledge_tool,
)


def _verification_message(
    *, sufficient: bool, feedback: str = "", call_id: str = "v1"
) -> AIMessage:
    """A scripted verifier response forcing a VerificationDecision tool call."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "VerificationDecision",
                "args": {"sufficient": sufficient, "feedback": feedback},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


@pytest.mark.asyncio
async def test_graph_happy_path_tool_then_verify_then_respond(dummy_settings) -> None:
    model = ScriptedToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_python",
                        "args": {"code": "print(2 ** 10)"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I computed 2**10 = 1024."),  # planner done, no more tools
            _verification_message(sufficient=True),
            AIMessage(content="1024 is the answer."),  # responder's final answer
        ]
    )
    agent = build_agent_graph(dummy_settings, llm=model, tools=[make_fake_execute_python_tool()])
    result = await agent.ainvoke(
        {"messages": [HumanMessage("Compute 2**10")]},
        config={"recursion_limit": dummy_settings.recursion_limit},
    )
    assert result["messages"][-1].content == "1024 is the answer."


@pytest.mark.asyncio
async def test_graph_routes_to_search_math_knowledge(dummy_settings) -> None:
    model = ScriptedToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_math_knowledge",
                        "args": {"query": "eigenvalue of a matrix"},
                        "id": "call_kb",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Per the notes, use numpy.linalg.eig."),
            _verification_message(sufficient=True),
            AIMessage(content="Per linear_algebra.md, use numpy.linalg.eig."),
        ]
    )
    agent = build_agent_graph(
        dummy_settings,
        llm=model,
        tools=[make_fake_execute_python_tool(), make_fake_search_math_knowledge_tool()],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("How do I find eigenvalues?")]},
        config={"recursion_limit": dummy_settings.recursion_limit},
    )
    assert "linear_algebra.md" in str(result["messages"][-1].content)


@pytest.mark.asyncio
async def test_graph_verifier_sends_planner_back_with_feedback(dummy_settings) -> None:
    """Insufficient once, then sufficient: loop-back works and the counter resets after."""
    model = ScriptedToolModel(
        [
            AIMessage(content="draft one"),  # planner call 1
            _verification_message(sufficient=False, feedback="double check the sign", call_id="v1"),
            AIMessage(content="draft two, fixed the sign"),  # planner call 2 (after retry)
            _verification_message(sufficient=True, call_id="v2"),
            AIMessage(content="Final polished answer."),
        ]
    )
    agent = build_agent_graph(dummy_settings, llm=model, tools=[make_fake_execute_python_tool()])
    result = await agent.ainvoke(
        {"messages": [HumanMessage("question")], "verification_attempts": 0},
        config={"recursion_limit": dummy_settings.recursion_limit},
    )
    assert result["messages"][-1].content == "Final polished answer."
    # Responder resets the counter so the next turn on this thread starts fresh.
    assert result["verification_attempts"] == 0
    assert any("double check the sign" in str(m.content) for m in result["messages"])


@pytest.mark.asyncio
async def test_graph_verifier_cap_prevents_infinite_loop(dummy_settings) -> None:
    """Verifier stays unsatisfied forever; the attempt cap still forces termination."""
    assert dummy_settings.verification_max_attempts == 2
    model = ScriptedToolModel(
        [
            AIMessage(content="draft 1"),
            _verification_message(sufficient=False, feedback="no", call_id="v1"),
            AIMessage(content="draft 2"),
            _verification_message(sufficient=False, feedback="still no", call_id="v2"),
            AIMessage(content="draft 3"),
            _verification_message(sufficient=False, feedback="nope", call_id="v3"),
            AIMessage(content="Answer given despite imperfect verification."),
        ]
    )
    agent = build_agent_graph(dummy_settings, llm=model, tools=[make_fake_execute_python_tool()])
    result = await agent.ainvoke(
        {"messages": [HumanMessage("question")], "verification_attempts": 0},
        config={"recursion_limit": dummy_settings.recursion_limit},
    )
    assert result["messages"][-1].content == "Answer given despite imperfect verification."
    assert result["verification_attempts"] == 0


def test_build_llm_uses_max_tokens(dummy_settings) -> None:
    from agent import build_llm

    s = replace(dummy_settings, max_tokens=123)
    with patch("agent.ChatAnthropic", autospec=True) as mock_cls:
        mock_cls.return_value = MagicMock()
        build_llm(s)
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["max_tokens"] == 123


def _direct_answer_script(text: str) -> list[AIMessage]:
    """planner (no tools) -> verifier(sufficient) -> responder, for one turn."""
    return [
        AIMessage(content=f"draft: {text}"),
        _verification_message(sufficient=True),
        AIMessage(content=text),
    ]


@pytest.mark.asyncio
async def test_checkpointer_persists_history_across_turns(dummy_settings) -> None:
    """Same thread_id: the second turn's state includes both human messages."""
    from langgraph.checkpoint.memory import InMemorySaver

    model = ScriptedToolModel(
        [
            *_direct_answer_script("First answer."),
            *_direct_answer_script("Second answer, building on the first."),
        ]
    )
    agent = build_agent_graph(
        dummy_settings,
        llm=model,
        tools=[make_fake_execute_python_tool()],
        checkpointer=InMemorySaver(),
    )
    config = {
        "configurable": {"thread_id": "t1"},
        "recursion_limit": dummy_settings.recursion_limit,
    }

    await agent.ainvoke({"messages": [HumanMessage("First question")]}, config=config)
    result = await agent.ainvoke({"messages": [HumanMessage("Second question")]}, config=config)

    humans = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert len(humans) == 2
    assert "Second answer" in str(result["messages"][-1].content)


@pytest.mark.asyncio
async def test_checkpointer_isolates_different_threads(dummy_settings) -> None:
    """Different thread_id: no history bleeds from one conversation into another."""
    from langgraph.checkpoint.memory import InMemorySaver

    model = ScriptedToolModel(
        [*_direct_answer_script("Answer."), *_direct_answer_script("Answer.")]
    )
    agent = build_agent_graph(
        dummy_settings,
        llm=model,
        tools=[make_fake_execute_python_tool()],
        checkpointer=InMemorySaver(),
    )
    recursion_limit = dummy_settings.recursion_limit

    await agent.ainvoke(
        {"messages": [HumanMessage("Q1")]},
        config={"configurable": {"thread_id": "a"}, "recursion_limit": recursion_limit},
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("Q2")]},
        config={"configurable": {"thread_id": "b"}, "recursion_limit": recursion_limit},
    )

    humans = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert len(humans) == 1


@pytest.mark.asyncio
async def test_without_checkpointer_each_call_is_stateless(dummy_settings) -> None:
    """No checkpointer passed (default None): behavior matches pre-memory graphs."""
    model = ScriptedToolModel(_direct_answer_script("Answer."))
    agent = build_agent_graph(dummy_settings, llm=model, tools=[make_fake_execute_python_tool()])

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Q1")]},
        config={"recursion_limit": dummy_settings.recursion_limit},
    )
    humans = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert len(humans) == 1
