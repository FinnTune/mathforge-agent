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
                name="execute_python",
                tool_call_id="c1",
            ),
            {},
        )
        yield (AIMessageChunk(content="Done."), {})


class _CapturingAgent:
    """Records the ``config`` passed by ``run_turn`` (checks thread_id plumbing)."""

    def __init__(self) -> None:
        self.configs: list[dict | None] = []

    async def astream(self, input_state, config=None, stream_mode=None):
        self.configs.append(config)
        yield (AIMessageChunk(content="ok"), {})

    async def ainvoke(self, input_state, config=None):
        self.configs.append(config)
        return {"messages": [HumanMessage("q"), AIMessage(content="ok")]}


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
    assert "[tool:execute_python]" in out
    assert "print(1)" in out


@pytest.mark.asyncio
async def test_run_turn_streaming_passes_thread_id() -> None:
    agent = _CapturingAgent()
    await run_turn(agent, "q", recursion_limit=5, stream=True, verbose=False, thread_id="abc")
    assert agent.configs[0]["configurable"]["thread_id"] == "abc"


@pytest.mark.asyncio
async def test_run_turn_no_stream_passes_thread_id() -> None:
    agent = _CapturingAgent()
    await run_turn(agent, "q", recursion_limit=5, stream=False, verbose=False, thread_id="xyz")
    assert agent.configs[0]["configurable"]["thread_id"] == "xyz"


@pytest.mark.asyncio
async def test_run_turn_omits_configurable_without_thread_id() -> None:
    agent = _CapturingAgent()
    await run_turn(agent, "q", recursion_limit=5, stream=False, verbose=False)
    assert "configurable" not in agent.configs[0]


@pytest.mark.asyncio
async def test_async_main_reset_command_starts_new_thread(monkeypatch) -> None:
    """Typing 'reset' between two queries changes the thread_id sent to the agent."""
    from config import Settings

    agent = _CapturingAgent()
    monkeypatch.setattr(main_module, "load_dotenv", lambda *_, **__: None)
    monkeypatch.setattr(
        main_module,
        "load_settings",
        lambda: Settings(
            anthropic_api_key="test-key",
            model="claude-sonnet-4-6",
            temperature=0.0,
            max_tokens=None,
            recursion_limit=15,
            code_timeout_sec=5.0,
            workspace_root=".",
            log_level="DEBUG",
            sandbox_mcp_bin="/nonexistent/mathforge-sandbox-mcp",
            sandbox_python="python3",
            sandbox_max_memory_mb=512,
            sandbox_max_output_bytes=256_000,
            mathkb_mcp_python="/nonexistent/mathkb-python",
            mathkb_mcp_script="/nonexistent/server.py",
            qdrant_url="http://localhost:6333",
            qdrant_collection="test-notes",
            voyage_api_key="",
            voyage_model="voyage-3-lite",
        ),
    )

    async def fake_load_mcp_tools(settings):
        return []

    monkeypatch.setattr(main_module, "load_mcp_tools", fake_load_mcp_tools)
    monkeypatch.setattr(
        main_module, "build_react_agent", lambda settings, tools=None, checkpointer=None: agent
    )

    inputs = iter(["hi", "reset", "hi again", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    code = await async_main([])

    assert code == 0
    thread_ids = [c["configurable"]["thread_id"] for c in agent.configs]
    assert len(thread_ids) == 2
    assert thread_ids[0] != thread_ids[1]


@pytest.mark.asyncio
async def test_async_main_missing_key(capsys: pytest.CaptureFixture[str], monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(main_module, "load_dotenv", lambda *_, **__: None)
    code = await async_main([])
    assert code == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err
