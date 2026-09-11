"""Tests for grpc_server.MathForgeChatServicer.

Most tests call ``Chat``/``HealthCheck`` directly as plain async
methods/generators (no network) against a real compiled graph backed by
``ScriptedToolModel`` — same pattern as ``tests/test_agent.py``. One test
spins up a real ``grpc.aio`` server + client on an ephemeral port to prove
the actual proto/transport wiring works, not just the servicer's Python
logic, without touching the real Anthropic API.
"""

from __future__ import annotations

import json

import grpc
import pytest
from langchain_core.messages import AIMessage

import chat_pb2
import chat_pb2_grpc
from agent import build_agent_graph
from grpc_server import MathForgeChatServicer
from tests.helpers import (
    ScriptedStreamingToolModel,
    ScriptedToolModel,
    make_fake_execute_python_tool,
    make_fake_search_math_knowledge_tool,
)


def _verification_message(*, sufficient: bool, feedback: str = "") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "VerificationDecision",
                "args": {"sufficient": sufficient, "feedback": feedback},
                "id": "v1",
                "type": "tool_call",
            }
        ],
    )


async def _collect_events(servicer: MathForgeChatServicer, request: chat_pb2.ChatRequest) -> list:
    return [event async for event in servicer.Chat(request, context=None)]


@pytest.mark.asyncio
async def test_chat_streams_tool_call_tool_result_text_delta_and_done(dummy_settings) -> None:
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
            AIMessage(content="I computed 1024."),
            _verification_message(sufficient=True),
            AIMessage(content="1024 is the answer."),
        ]
    )
    agent = build_agent_graph(dummy_settings, llm=model, tools=[make_fake_execute_python_tool()])
    servicer = MathForgeChatServicer(agent, dummy_settings)
    request = chat_pb2.ChatRequest(thread_id="t1", query="Compute 2**10")

    events = await _collect_events(servicer, request)
    kinds = [e.WhichOneof("event") for e in events]

    assert kinds == ["tool_call", "tool_result", "text_delta", "done"]
    assert events[0].tool_call.tool_name == "execute_python"
    assert "print(2 ** 10)" in events[1].tool_result.result
    assert events[2].text_delta.text == "1024 is the answer."


@pytest.mark.asyncio
async def test_chat_reconstructs_streamed_tool_call_from_chunks(dummy_settings) -> None:
    """Regression test: a real provider streams a tool call's name in one
    chunk and its args as raw JSON fragments in several more — each chunk's
    own `.tool_calls` is a misleading partial reconstruction (e.g. name
    present but args={}, or name empty) until they're all summed via `+`.
    Naively emitting a tool_call event per chunk (the original bug, caught
    live against the real Anthropic API, not by the ScriptedToolModel-based
    tests above) produced several garbled events for one real call.
    """
    model = ScriptedStreamingToolModel(
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
            AIMessage(content="I computed 1024."),
            _verification_message(sufficient=True),
            AIMessage(content="1024 is the answer."),
        ]
    )
    agent = build_agent_graph(dummy_settings, llm=model, tools=[make_fake_execute_python_tool()])
    servicer = MathForgeChatServicer(agent, dummy_settings)
    request = chat_pb2.ChatRequest(thread_id="t1s", query="Compute 2**10")

    events = await _collect_events(servicer, request)
    tool_call_events = [e for e in events if e.WhichOneof("event") == "tool_call"]

    assert len(tool_call_events) == 1
    assert tool_call_events[0].tool_call.tool_name == "execute_python"
    assert json.loads(tool_call_events[0].tool_call.args_json) == {"code": "print(2 ** 10)"}


@pytest.mark.asyncio
async def test_chat_does_not_leak_verifier_tool_call(dummy_settings) -> None:
    """The verifier's forced VerificationDecision call must never appear as a tool_call event."""
    model = ScriptedToolModel(
        [
            AIMessage(content="draft"),
            _verification_message(sufficient=True),
            AIMessage(content="Final answer."),
        ]
    )
    agent = build_agent_graph(dummy_settings, llm=model, tools=[make_fake_execute_python_tool()])
    servicer = MathForgeChatServicer(agent, dummy_settings)
    request = chat_pb2.ChatRequest(thread_id="t2", query="question")

    events = await _collect_events(servicer, request)
    kinds = [e.WhichOneof("event") for e in events]

    assert "tool_call" not in kinds
    assert kinds == ["text_delta", "done"]


@pytest.mark.asyncio
async def test_chat_routes_search_math_knowledge(dummy_settings) -> None:
    model = ScriptedToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_math_knowledge",
                        "args": {"query": "eigenvalues"},
                        "id": "call_kb",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Found notes."),
            _verification_message(sufficient=True),
            AIMessage(content="Per linear_algebra.md, use numpy.linalg.eig."),
        ]
    )
    agent = build_agent_graph(
        dummy_settings,
        llm=model,
        tools=[make_fake_execute_python_tool(), make_fake_search_math_knowledge_tool()],
    )
    servicer = MathForgeChatServicer(agent, dummy_settings)
    request = chat_pb2.ChatRequest(thread_id="t3", query="How do I find eigenvalues?")

    events = await _collect_events(servicer, request)
    text_events = [e for e in events if e.WhichOneof("event") == "text_delta"]

    assert events[0].tool_call.tool_name == "search_math_knowledge"
    assert "linear_algebra.md" in text_events[-1].text_delta.text


@pytest.mark.asyncio
async def test_chat_yields_error_event_on_failure(dummy_settings) -> None:
    class _BrokenAgent:
        async def astream(self, *args, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator

    servicer = MathForgeChatServicer(_BrokenAgent(), dummy_settings)
    request = chat_pb2.ChatRequest(thread_id="t4", query="question")

    events = await _collect_events(servicer, request)

    assert len(events) == 1
    assert events[0].WhichOneof("event") == "error"
    assert "boom" in events[0].error.message


@pytest.mark.asyncio
async def test_health_check(dummy_settings) -> None:
    servicer = MathForgeChatServicer(agent=None, settings=dummy_settings)
    response = await servicer.HealthCheck(chat_pb2.HealthCheckRequest(), context=None)
    assert response.ok is True
    assert response.model == dummy_settings.model


@pytest.mark.asyncio
async def test_full_loopback_with_real_grpc_server(dummy_settings) -> None:
    """Real sockets: proto serialization + grpc.aio server + client, no mocking."""
    model = ScriptedToolModel(
        [
            AIMessage(content="draft"),
            _verification_message(sufficient=True),
            AIMessage(content="Loopback answer."),
        ]
    )
    agent = build_agent_graph(dummy_settings, llm=model, tools=[make_fake_execute_python_tool()])
    servicer = MathForgeChatServicer(agent, dummy_settings)

    server = grpc.aio.server()
    chat_pb2_grpc.add_MathForgeChatServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = chat_pb2_grpc.MathForgeChatStub(channel)

            health = await stub.HealthCheck(chat_pb2.HealthCheckRequest())
            assert health.ok is True

            events = [
                event
                async for event in stub.Chat(chat_pb2.ChatRequest(thread_id="t5", query="hi"))
            ]
            kinds = [e.WhichOneof("event") for e in events]
            assert kinds == ["text_delta", "done"]
            assert events[0].text_delta.text == "Loopback answer."
    finally:
        await server.stop(None)
