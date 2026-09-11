"""gRPC server hosting the MathForge agent graph — see proto/chat.proto.

Builds everything once at startup (settings, MCP tools, checkpointer, the
compiled graph) and serves many client requests against the *same* agent,
replacing the old per-process pattern where ``main.py``/``discord_bot.py``
each built their own. ``Chat`` runs the same
``agent.astream(..., stream_mode="messages")`` loop and ``langgraph_node``
filtering ``main.py``'s ``run_turn`` used to do directly (see git history),
now yielding ``ChatEvent`` protos instead of printing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys

import grpc
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

import chat_pb2
import chat_pb2_grpc
from agent import build_agent_graph
from checkpointer import build_checkpointer
from config import Settings, load_settings
from mcp_client import load_mcp_tools

logger = logging.getLogger(__name__)


def _content_to_text(content: object) -> str:
    """Normalize LangChain message content (str or content-block list) to plain text."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    pieces: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            pieces.append(str(block.get("text", "")))
        elif isinstance(block, str):
            pieces.append(block)
    return "".join(pieces)


class MathForgeChatServicer(chat_pb2_grpc.MathForgeChatServicer):
    """Wraps one compiled agent graph (see ``agent.build_agent_graph``) as a gRPC service."""

    def __init__(self, agent, settings: Settings) -> None:
        self._agent = agent
        self._settings = settings

    async def Chat(self, request, context):
        input_state = {"messages": [HumanMessage(content=request.query)]}
        config = {
            "recursion_limit": self._settings.recursion_limit,
            "configurable": {"thread_id": request.thread_id},
        }
        # Tool-call args stream in as raw JSON fragments across many chunks
        # (name in the first, then only `args` deltas — LangChain surfaces
        # each chunk's own `.tool_calls` as a *partial* reconstruction, not
        # the real thing), so chunks for one LLM call must be summed via `+`
        # before a tool_call event is meaningful. Reset whenever the node
        # changes — a ToolMessage or a different langgraph_node both mean
        # the previous call's streaming is done.
        pending_chunk: AIMessageChunk | None = None
        pending_node: str | None = None

        def flush_pending_tool_calls() -> list:
            nonlocal pending_chunk, pending_node
            events = []
            # Only the planner's tool_calls are real (sandbox/mathkb) tool
            # invocations — the verifier's is a forced VerificationDecision
            # call, internal plumbing with no matching ToolResult, so
            # surfacing it would be misleading.
            if pending_chunk is not None and pending_node == "planner":
                for call in pending_chunk.tool_calls:
                    if call.get("name"):
                        events.append(
                            chat_pb2.ChatEvent(
                                tool_call=chat_pb2.ToolCall(
                                    tool_name=call["name"],
                                    args_json=json.dumps(call.get("args", {})),
                                )
                            )
                        )
            pending_chunk = None
            pending_node = None
            return events

        try:
            async for item in self._agent.astream(
                input_state, config=config, stream_mode="messages"
            ):
                if not isinstance(item, tuple) or not item:
                    continue
                message, metadata = item[0], (item[1] if len(item) > 1 else {})
                node = metadata.get("langgraph_node")

                if isinstance(message, ToolMessage):
                    for event in flush_pending_tool_calls():
                        yield event
                    yield chat_pb2.ChatEvent(
                        tool_result=chat_pb2.ToolResult(
                            tool_name=getattr(message, "name", None) or "tool",
                            result=_content_to_text(message.content),
                        )
                    )
                    continue

                if isinstance(message, AIMessageChunk) and (
                    message.tool_calls or message.tool_call_chunks
                ):
                    if pending_node is not None and pending_node != node:
                        for event in flush_pending_tool_calls():
                            yield event
                    pending_chunk = message if pending_chunk is None else pending_chunk + message
                    pending_node = node
                    continue

                # Non-chunk AIMessage with tool_calls already complete (e.g. a
                # non-streaming model, as in tests) — no accumulation needed.
                if (
                    isinstance(message, AIMessage)
                    and not isinstance(message, AIMessageChunk)
                    and message.tool_calls
                ):
                    if node == "planner":
                        for call in message.tool_calls:
                            if call.get("name"):
                                yield chat_pb2.ChatEvent(
                                    tool_call=chat_pb2.ToolCall(
                                        tool_name=call["name"],
                                        args_json=json.dumps(call.get("args", {})),
                                    )
                                )
                    continue

                if not isinstance(message, AIMessage) or node != "responder":
                    continue
                text = _content_to_text(getattr(message, "content", None))
                if text:
                    yield chat_pb2.ChatEvent(text_delta=chat_pb2.TextDelta(text=text))

            for event in flush_pending_tool_calls():
                yield event
            yield chat_pb2.ChatEvent(done=chat_pb2.Done())
        except Exception as exc:  # noqa: BLE001 — reported to the client as an Error event
            logger.exception("Chat turn failed (thread_id=%s)", request.thread_id)
            yield chat_pb2.ChatEvent(error=chat_pb2.Error(message=str(exc)))

    async def HealthCheck(self, request, context):
        return chat_pb2.HealthCheckResponse(ok=True, model=self._settings.model)


async def serve() -> int:
    """Build the agent once and serve it over gRPC until interrupted."""
    load_dotenv()
    try:
        settings = load_settings()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        tools = await load_mcp_tools(settings)
    except Exception as exc:  # noqa: BLE001 — surface setup errors to the operator
        logging.exception("Failed to load MCP tools")
        print(f"Could not start server: {exc}", file=sys.stderr)
        return 1

    async with build_checkpointer(settings) as checkpointer:
        agent = build_agent_graph(settings, tools=tools, checkpointer=checkpointer)

        server = grpc.aio.server()
        chat_pb2_grpc.add_MathForgeChatServicer_to_server(
            MathForgeChatServicer(agent, settings), server
        )
        target = f"{settings.grpc_host}:{settings.grpc_port}"
        server.add_insecure_port(target)
        await server.start()
        logger.info("MathForge gRPC server listening on %s", target)

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        await stop_event.wait()
        logger.info("Shutting down MathForge gRPC server")
        await server.stop(grace=5)
    return 0


def main() -> None:
    """Setuptools console-script entrypoint (``mathforge-server``)."""
    sys.exit(asyncio.run(serve()))


if __name__ == "__main__":
    main()
