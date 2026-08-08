"""Tests for Discord bot helper logic.

These tests intentionally avoid network/API calls and focus on deterministic
helpers used by the slash-command path.
"""

from __future__ import annotations

from discord_bot import (
    chunk_text,
    coerce_content_to_text,
    is_channel_allowed,
    parse_allowed_channel_ids,
    thread_key,
)


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


def test_coerce_content_to_text_variants() -> None:
    assert coerce_content_to_text("hello") == "hello"
    assert coerce_content_to_text(["a", "b"]) == "ab"
    assert (
        coerce_content_to_text([{"type": "text", "text": "hi"}, {"type": "x", "text": "skip"}])
        == "hi"
    )


def test_thread_key_scopes_by_channel_user_and_generation() -> None:
    assert thread_key(100, 200, 0) == "100:200:0"
    assert thread_key(100, 200, 1) != thread_key(100, 200, 0)
    assert thread_key(100, 200, 0) != thread_key(101, 200, 0)
    assert thread_key(100, 200, 0) != thread_key(100, 201, 0)


def test_thread_key_handles_missing_channel() -> None:
    # DMs / channel-less interactions: channel_id may be None.
    assert thread_key(None, 200, 0) == "None:200:0"
