"""Interactive CLI: a thin gRPC client for the MathForge agent server.

Flow:
1. ``load_dotenv()`` loads ``.env`` (only ``MATHFORGE_GRPC_TARGET`` matters
   here — everything else, API keys included, is the server's concern now;
   see ``grpc_server.py``).
2. Connects to ``MATHFORGE_GRPC_TARGET`` (default ``127.0.0.1:50051``) and
   calls ``HealthCheck`` once, so a server that isn't running fails with a
   clear message instead of a raw connection-refused traceback on the first
   query.
3. Each REPL turn calls the streaming ``Chat`` RPC (``proto/chat.proto``) and
   prints ``text_delta`` events live — that's the responder node's output
   only; the server already filters out the planner's internal reasoning and
   the verifier's structured decision (see ``grpc_server.py``). ``tool_call``/
   ``tool_result`` events print under ``--verbose``.
4. ``reset`` just starts a new ``thread_id`` client-side — the server needs
   no request for that, conversation memory for the old thread simply stays
   unreferenced in its SQLite DB.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid

import grpc
from dotenv import load_dotenv

import chat_pb2
import chat_pb2_grpc

HEALTH_CHECK_TIMEOUT_SEC = 5.0


def _configure_logging() -> None:
    level = os.getenv("MATHFORGE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _print_tool_call(event: chat_pb2.ToolCall, *, verbose: bool) -> None:
    if not verbose:
        return
    print(f"\n[calling {event.tool_name}] {event.args_json}\n", flush=True)


def _print_tool_result(event: chat_pb2.ToolResult, *, verbose: bool) -> None:
    if not verbose:
        return
    preview = event.result if len(event.result) < 2000 else event.result[:2000] + "…"
    print(f"\n[tool:{event.tool_name}]\n{preview}\n", flush=True)


async def check_server_health(stub: chat_pb2_grpc.MathForgeChatStub) -> str | None:
    """Return the server's reported model name, or ``None`` if unreachable."""
    try:
        response = await asyncio.wait_for(
            stub.HealthCheck(chat_pb2.HealthCheckRequest()), timeout=HEALTH_CHECK_TIMEOUT_SEC
        )
    except (grpc.aio.AioRpcError, TimeoutError):
        return None
    return response.model if response.ok else None


async def run_turn(
    stub: chat_pb2_grpc.MathForgeChatStub,
    query: str,
    *,
    thread_id: str,
    stream: bool,
    verbose: bool,
) -> None:
    """Run one turn against the gRPC server (streaming print, or buffer-then-print)."""
    request = chat_pb2.ChatRequest(thread_id=thread_id, query=query)
    buffer: list[str] = []
    if stream:
        print("MathForge: ", end="", flush=True)

    try:
        async for event in stub.Chat(request):
            kind = event.WhichOneof("event")
            if kind == "text_delta":
                if stream:
                    print(event.text_delta.text, end="", flush=True)
                else:
                    buffer.append(event.text_delta.text)
            elif kind == "tool_call":
                _print_tool_call(event.tool_call, verbose=verbose)
            elif kind == "tool_result":
                _print_tool_result(event.tool_result, verbose=verbose)
            elif kind == "error":
                print(f"\nError: {event.error.message}", file=sys.stderr)
                return
            elif kind == "done":
                break
    except grpc.aio.AioRpcError as exc:
        print(f"\nRequest failed: {exc.details()}", file=sys.stderr)
        return

    if stream:
        print("\n", flush=True)
    else:
        print(f"MathForge: {''.join(buffer)}\n", flush=True)


async def async_main(argv: list[str] | None = None) -> int:
    """Parse CLI args, connect to the gRPC server, run the REPL loop. Returns an exit code."""
    load_dotenv()
    _configure_logging()
    parser = argparse.ArgumentParser(description="MathForge — thin gRPC client")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the full reply instead of streaming tokens",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tool calls/outputs to the terminal",
    )
    args = parser.parse_args(argv)

    target = os.getenv("MATHFORGE_GRPC_TARGET", "127.0.0.1:50051")
    async with grpc.aio.insecure_channel(target) as channel:
        stub = chat_pb2_grpc.MathForgeChatStub(channel)
        model = await check_server_health(stub)
        if model is None:
            print(
                f"Could not reach MathForge gRPC server at {target}. "
                "Is `mathforge-server` running? (see README Quick Start)",
                file=sys.stderr,
            )
            return 1

        stream = not args.no_stream
        thread_id = str(uuid.uuid4())
        print(f"Welcome to MathForge (server model: {model}). Commands: exit, quit, reset.\n")
        while True:
            try:
                query = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return 0

            stripped = query.strip()
            if stripped.lower() in {"exit", "quit"}:
                print("Goodbye!")
                return 0
            if stripped.lower() == "reset":
                thread_id = str(uuid.uuid4())
                print("Conversation memory cleared.\n")
                continue
            if not stripped:
                continue

            print("", flush=True)
            await run_turn(stub, stripped, thread_id=thread_id, stream=stream, verbose=args.verbose)


def main() -> None:
    """Setuptools console-script entrypoint."""
    sys.exit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
