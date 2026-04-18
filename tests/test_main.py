"""Tests for ``main.run_turn`` and ``main.async_main`` entry behavior.

Uses stub agents that yield canned message tuples (same shapes LangGraph emits
under ``stream_mode="messages"``). ``test_async_main_missing_key`` patches
``load_dotenv`` so a developer ``.env`` file cannot satisfy the missing-key case.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

import main as main_module
from main import async_main, run_turn


class _FakeStreamAgent:
    """Minimal async agent: streaming chunks or full-message invoke."""

    async def astream(self, input_state, config=None, stream_mode=None):
        yield (AIMessageChunk(content="Hello"), {})
        yield (AIMessageChunk(content=" world."), {})

    async def ainvoke(self, input_state, config=None):
        return {"messages": [HumanMessage("q"), AIMessage(content="Full reply")]}


class _FakeToolStreamAgent:
    """Yields a tool message then a short assistant chunk (verbose-mode test)."""

    async def astream(self, input_state, config=None, stream_mode=None):
        yield (
            ToolMessage(
                content="print(1)",
                name="execute_python_code",
                tool_call_id="c1",
            ),
            {},
        )
        yield (AIMessageChunk(content="Done."), {})


@pytest.mark.asyncio
async def test_run_turn_stream(capsys: pytest.CaptureFixture[str]) -> None:
    await run_turn(
        _FakeStreamAgent(),
        "question",
        recursion_limit=5,
        stream=True,
        verbose=False,
    )
    out = capsys.readouterr().out
    assert "Hello world." in out.replace("\n", "")


@pytest.mark.asyncio
async def test_run_turn_no_stream(capsys: pytest.CaptureFixture[str]) -> None:
    await run_turn(
        _FakeStreamAgent(),
        "question",
        recursion_limit=5,
        stream=False,
        verbose=False,
    )
    assert "Full reply" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_turn_verbose_tool(capsys: pytest.CaptureFixture[str]) -> None:
    await run_turn(
        _FakeToolStreamAgent(),
        "q",
        recursion_limit=5,
        stream=True,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "[tool:execute_python_code]" in out
    assert "print(1)" in out


@pytest.mark.asyncio
async def test_async_main_missing_key(capsys: pytest.CaptureFixture[str], monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(main_module, "load_dotenv", lambda *_, **__: None)
    code = await async_main([])
    assert code == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err
