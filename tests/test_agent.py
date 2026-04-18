"""Tests for ``agent.build_react_agent`` and ``build_llm``.

Uses ``ScriptedToolModel`` so no Anthropic API calls occur. Verifies tool-then-
answer ReAct flow, streaming smoke, and that ``max_tokens`` is forwarded to
``ChatAnthropic`` when set.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import build_react_agent
from tests.helpers import ScriptedToolModel


@pytest.mark.asyncio
async def test_react_loop_tool_then_answer(dummy_settings) -> None:
    model = ScriptedToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_python_code",
                        "args": {"code": "print(2 ** 10)"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="1024 is the answer."),
        ]
    )
    agent = build_react_agent(dummy_settings, llm=model)
    result = await agent.ainvoke(
        {"messages": [HumanMessage("Compute 2**10")]},
        config={"recursion_limit": dummy_settings.recursion_limit},
    )
    texts = [m.content for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    assert any("1024" in str(t) for t in texts)


@pytest.mark.asyncio
async def test_stream_messages_yields_tool_and_ai(dummy_settings) -> None:
    model = ScriptedToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_python_code",
                        "args": {"code": "print('hi')"},
                        "id": "c2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Done."),
        ]
    )
    agent = build_react_agent(dummy_settings, llm=model)
    chunks: list = []
    async for item in agent.astream(
        {"messages": [HumanMessage("Hi")]},
        config={"recursion_limit": dummy_settings.recursion_limit},
        stream_mode="messages",
    ):
        chunks.append(item)

    assert chunks, "expected stream events"


def test_build_llm_uses_max_tokens(dummy_settings) -> None:
    from agent import build_llm

    s = replace(dummy_settings, max_tokens=123)
    with patch("agent.ChatAnthropic", autospec=True) as mock_cls:
        mock_cls.return_value = MagicMock()
        build_llm(s)
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["max_tokens"] == 123
