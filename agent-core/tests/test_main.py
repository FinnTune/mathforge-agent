"""Tests for ``main.py``, the thin gRPC client.

``run_turn``/``check_server_health`` are tested against a fake stub object
(same DI spirit as the fake LangGraph agents used before this phase — see
git history) so no real network or server is needed. ``async_main`` is
tested by monkeypatching ``grpc.aio.insecure_channel`` and
``chat_pb2_grpc.MathForgeChatStub`` to return the same fake stub — the real
proto/transport wiring has its own coverage in ``test_grpc_server.py``'s
loopback test.
"""

from __future__ import annotations

import pytest

import chat_pb2
import main as main_module
from main import async_main, check_server_health, run_turn


class _FakeStub:
    """Yields canned ``ChatEvent``s / a canned ``HealthCheckResponse``."""

    def __init__(self, events=None, health_response=None, health_raises=None) -> None:
        self._events = events or []
        self._health_response = health_response
        self._health_raises = health_raises
        self.requests: list[chat_pb2.ChatRequest] = []

    async def Chat(self, request):
        self.requests.append(request)
        for event in self._events:
            yield event

    async def HealthCheck(self, request):
        if self._health_raises is not None:
            raise self._health_raises
        return self._health_response


class _FakeChannel:
    """Just enough of a ``grpc.aio.Channel`` for ``async with`` in ``async_main``."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


@pytest.mark.asyncio
async def test_check_server_health_reachable() -> None:
    stub = _FakeStub(health_response=chat_pb2.HealthCheckResponse(ok=True, model="claude-x"))
    assert await check_server_health(stub) == "claude-x"


@pytest.mark.asyncio
async def test_check_server_health_reports_not_ok_as_unreachable() -> None:
    stub = _FakeStub(health_response=chat_pb2.HealthCheckResponse(ok=False, model=""))
    assert await check_server_health(stub) is None


@pytest.mark.asyncio
async def test_check_server_health_unreachable_on_timeout() -> None:
    stub = _FakeStub(health_raises=TimeoutError())
    assert await check_server_health(stub) is None


@pytest.mark.asyncio
async def test_run_turn_stream(capsys: pytest.CaptureFixture[str]) -> None:
    stub = _FakeStub(
        events=[
            chat_pb2.ChatEvent(text_delta=chat_pb2.TextDelta(text="Hello")),
            chat_pb2.ChatEvent(text_delta=chat_pb2.TextDelta(text=" world.")),
            chat_pb2.ChatEvent(done=chat_pb2.Done()),
        ]
    )
    await run_turn(stub, "question", thread_id="t1", stream=True, verbose=False)
    out = capsys.readouterr().out
    assert "Hello world." in out.replace("\n", "")
    assert stub.requests[0].thread_id == "t1"
    assert stub.requests[0].query == "question"


@pytest.mark.asyncio
async def test_run_turn_no_stream_buffers_until_done(capsys: pytest.CaptureFixture[str]) -> None:
    stub = _FakeStub(
        events=[
            chat_pb2.ChatEvent(text_delta=chat_pb2.TextDelta(text="Full ")),
            chat_pb2.ChatEvent(text_delta=chat_pb2.TextDelta(text="reply")),
            chat_pb2.ChatEvent(done=chat_pb2.Done()),
        ]
    )
    await run_turn(stub, "question", thread_id="t1", stream=False, verbose=False)
    assert "MathForge: Full reply" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_turn_verbose_shows_tool_events(capsys: pytest.CaptureFixture[str]) -> None:
    stub = _FakeStub(
        events=[
            chat_pb2.ChatEvent(
                tool_call=chat_pb2.ToolCall(
                    tool_name="execute_python", args_json='{"code": "1+1"}'
                )
            ),
            chat_pb2.ChatEvent(
                tool_result=chat_pb2.ToolResult(tool_name="execute_python", result="2")
            ),
            chat_pb2.ChatEvent(text_delta=chat_pb2.TextDelta(text="Two.")),
            chat_pb2.ChatEvent(done=chat_pb2.Done()),
        ]
    )
    await run_turn(stub, "1+1", thread_id="t1", stream=True, verbose=True)
    out = capsys.readouterr().out
    assert "[calling execute_python]" in out
    assert "[tool:execute_python]" in out
    assert "2" in out


@pytest.mark.asyncio
async def test_run_turn_hides_tool_events_without_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub = _FakeStub(
        events=[
            chat_pb2.ChatEvent(
                tool_call=chat_pb2.ToolCall(tool_name="execute_python", args_json="{}")
            ),
            chat_pb2.ChatEvent(text_delta=chat_pb2.TextDelta(text="Answer.")),
            chat_pb2.ChatEvent(done=chat_pb2.Done()),
        ]
    )
    await run_turn(stub, "q", thread_id="t1", stream=True, verbose=False)
    out = capsys.readouterr().out
    assert "[calling" not in out
    assert "Answer." in out


@pytest.mark.asyncio
async def test_run_turn_prints_error_event(capsys: pytest.CaptureFixture[str]) -> None:
    stub = _FakeStub(
        events=[chat_pb2.ChatEvent(error=chat_pb2.Error(message="sandbox unreachable"))]
    )
    await run_turn(stub, "q", thread_id="t1", stream=True, verbose=False)
    err = capsys.readouterr().err
    assert "sandbox unreachable" in err


@pytest.mark.asyncio
async def test_async_main_missing_server(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module.grpc.aio, "insecure_channel", lambda target: _FakeChannel())
    monkeypatch.setattr(
        main_module.chat_pb2_grpc,
        "MathForgeChatStub",
        lambda channel: _FakeStub(health_raises=TimeoutError()),
    )
    code = await async_main([])
    assert code == 1
    assert "mathforge-server" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_async_main_reset_command_starts_new_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typing 'reset' between two queries changes the thread_id sent to the server."""
    stub = _FakeStub(
        health_response=chat_pb2.HealthCheckResponse(ok=True, model="claude-x"),
        events=[
            chat_pb2.ChatEvent(text_delta=chat_pb2.TextDelta(text="ok")),
            chat_pb2.ChatEvent(done=chat_pb2.Done()),
        ],
    )
    monkeypatch.setattr(main_module.grpc.aio, "insecure_channel", lambda target: _FakeChannel())
    monkeypatch.setattr(main_module.chat_pb2_grpc, "MathForgeChatStub", lambda channel: stub)

    inputs = iter(["hi", "reset", "hi again", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    code = await async_main([])

    assert code == 0
    thread_ids = [r.thread_id for r in stub.requests]
    assert len(thread_ids) == 2
    assert thread_ids[0] != thread_ids[1]
