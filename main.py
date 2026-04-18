"""CLI entrypoint: streaming ReAct loop with optional verbose tool logging."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from agent import build_react_agent
from config import load_settings


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _print_tool_event(message: ToolMessage, *, verbose: bool) -> None:
    if not verbose:
        return
    name = getattr(message, "name", "tool")
    content = message.content
    preview = content if len(content) < 2000 else content[:2000] + "…"
    print(f"\n[tool:{name}]\n{preview}\n", flush=True)


def _emit_ai_text(message: BaseMessage) -> None:
    content = getattr(message, "content", None)
    if not content:
        return
    if isinstance(content, str):
        print(content, end="", flush=True)
        return
    # Multimodal content blocks
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            print(block.get("text", ""), end="", flush=True)
        elif isinstance(block, str):
            print(block, end="", flush=True)


async def run_turn(
    agent,
    query: str,
    *,
    recursion_limit: int,
    stream: bool,
    verbose: bool,
) -> None:
    input_state = {"messages": [HumanMessage(content=query)]}
    config = {"recursion_limit": recursion_limit}

    if not stream:
        result = await agent.ainvoke(input_state, config=config)
        final = result["messages"][-1].content
        print(f"MathForge: {final}\n", flush=True)
        return

    print("MathForge: ", end="", flush=True)
    streamed_text = False
    async for item in agent.astream(input_state, config=config, stream_mode="messages"):
        if not isinstance(item, tuple) or not item:
            continue
        message = item[0]
        if isinstance(message, ToolMessage):
            _print_tool_event(message, verbose=verbose)
        elif isinstance(message, AIMessageChunk):
            if message.content:
                streamed_text = True
            _emit_ai_text(message)
        elif isinstance(message, AIMessage) and message.content and not streamed_text:
            _emit_ai_text(message)
    print("\n", flush=True)


async def async_main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="MathForge — Claude + sandboxed Python")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the full reply instead of streaming tokens",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tool outputs to the terminal",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    os.environ["MATHFORGE_WORKSPACE_ROOT"] = settings.workspace_root
    os.environ["MATHFORGE_CODE_TIMEOUT_SEC"] = str(settings.code_timeout_sec)

    _configure_logging(settings.log_level)
    try:
        agent = build_react_agent(settings)
    except Exception as exc:  # noqa: BLE001 — surface setup errors to the user
        logging.exception("Failed to build agent")
        print(f"Could not start agent: {exc}", file=sys.stderr)
        return 1

    stream = not args.no_stream
    print("Welcome to MathForge (Claude + subprocess sandbox). Commands: exit, quit.\n")
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
        if not stripped:
            continue

        print("", flush=True)
        try:
            await run_turn(
                agent,
                stripped,
                recursion_limit=settings.recursion_limit,
                stream=stream,
                verbose=args.verbose,
            )
        except TimeoutError:
            print("Request timed out.", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Agent run failed")
            print(f"Error: {exc}", file=sys.stderr)


def main() -> None:
    sys.exit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
