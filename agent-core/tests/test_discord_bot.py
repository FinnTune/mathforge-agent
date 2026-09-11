"""Tests for Discord bot helper logic.

These tests intentionally avoid network/API calls and focus on deterministic
helpers used by the slash-command path. ``run_query`` is tested against a
fake gRPC stub (see ``tests/test_main.py``'s docstring for the same pattern)
— the real proto/transport wiring has its own coverage in
``test_grpc_server.py``'s loopback test.
"""

from __future__ import annotations

import pytest

import chat_pb2
from discord_bot import (
    chunk_text,
    is_channel_allowed,
    parse_allowed_channel_ids,
    run_query,
    thread_key,
)


class _FakeStub:
    def __init__(self, events) -> None:
        self._events = events

    async def Chat(self, request):
        for event in self._events:
            yield event


def test_parse_allowed_channel_ids_empty() -> None:
    assert parse_allowed_channel_ids(None) == set()
    assert parse_allowed_channel_ids("") == set()
    assert parse_allowed_channel_ids(" , ") == set()


def test_parse_allowed_channel_ids_values() -> None:
    parsed = parse_allowed_channel_ids("123, 456,789")
    assert parsed == {123, 456, 789}


def test_is_channel_allowed_allowlist_disabled() -> None:
    assert is_channel_allowed(42, set())
    assert is_channel_allowed(None, set())


def test_is_channel_allowed_allowlist_enabled() -> None:
    allowed = {100, 200}
    assert is_channel_allowed(100, allowed)
    assert not is_channel_allowed(300, allowed)
    assert not is_channel_allowed(None, allowed)


def test_chunk_text_splits_and_handles_empty() -> None:
    assert list(chunk_text("")) == ["(empty response)"]
    parts = list(chunk_text("abcdef", size=2))
    assert parts == ["ab", "cd", "ef"]


@pytest.mark.asyncio
async def test_run_query_concatenates_text_deltas() -> None:
    stub = _FakeStub(
        [
            chat_pb2.ChatEvent(text_delta=chat_pb2.TextDelta(text="Full ")),
            chat_pb2.ChatEvent(text_delta=chat_pb2.TextDelta(text="reply")),
            chat_pb2.ChatEvent(done=chat_pb2.Done()),
        ]
    )
    assert await run_query(stub, "q", "t1") == "Full reply"


@pytest.mark.asyncio
async def test_run_query_defaults_empty_reply_to_placeholder() -> None:
    stub = _FakeStub([chat_pb2.ChatEvent(done=chat_pb2.Done())])
    assert await run_query(stub, "q", "t1") == "(empty response)"


@pytest.mark.asyncio
async def test_run_query_raises_on_error_event() -> None:
    stub = _FakeStub([chat_pb2.ChatEvent(error=chat_pb2.Error(message="boom"))])
    with pytest.raises(RuntimeError, match="boom"):
        await run_query(stub, "q", "t1")


def test_thread_key_scopes_by_channel_user_and_generation() -> None:
    assert thread_key(100, 200, 0) == "100:200:0"
    assert thread_key(100, 200, 1) != thread_key(100, 200, 0)
    assert thread_key(100, 200, 0) != thread_key(101, 200, 0)
    assert thread_key(100, 200, 0) != thread_key(100, 201, 0)


def test_thread_key_handles_missing_channel() -> None:
    # DMs / channel-less interactions: channel_id may be None.
    assert thread_key(None, 200, 0) == "None:200:0"
