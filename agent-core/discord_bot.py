"""Discord interface for MathForge via a slash command.

This module wraps the existing LangGraph agent with a Discord bot so Discord is
just another transport (CLI remains unchanged). The bot:

- registers `/mathforge query:<text> reset:<bool>`
- enforces optional channel allowlist and prompt length limits
- runs the same compiled agent used by CLI, with per-channel-per-user
  conversation memory (an in-memory LangGraph checkpointer keyed by
  ``"<channel_id>:<user_id>:<generation>"``; ``reset:true`` bumps the
  generation to start a fresh thread)
- chunks long replies to satisfy Discord message limits

Memory lives only in the bot process's RAM: it does not survive a restart and
is not shared across processes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator

import discord
from discord import app_commands
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent import build_react_agent
from config import load_settings
from mcp_client import load_sandbox_tools

logger = logging.getLogger(__name__)


def parse_allowed_channel_ids(raw: str | None) -> set[int]:
    """Parse comma-separated channel IDs from env into a set."""
    if raw is None:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        out.add(int(stripped))
    return out


def is_channel_allowed(channel_id: int | None, allowed_ids: set[int]) -> bool:
    """If allowlist is configured, require channel ID to be in it."""
    if not allowed_ids:
        return True
    if channel_id is None:
        return False
    return channel_id in allowed_ids


def chunk_text(text: str, size: int = 1900) -> Iterator[str]:
    """Split output to fit Discord's 2000-char message limit safely."""
    if not text:
        yield "(empty response)"
        return
    for i in range(0, len(text), size):
        yield text[i : i + size]


def coerce_content_to_text(content: object) -> str:
    """Normalize LangChain message content to plain text for Discord."""
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                pieces.append(str(block.get("text", "")))
            elif isinstance(block, str):
                pieces.append(block)
        if pieces:
            return "".join(pieces)
    return str(content)


def thread_key(channel_id: int | None, user_id: int, generation: int) -> str:
    """Build the checkpointer ``thread_id`` for one channel+user conversation."""
    return f"{channel_id}:{user_id}:{generation}"


async def run_query(agent, query: str, recursion_limit: int, thread_id: str) -> str:
    """Execute one query against the compiled graph and return final text."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": recursion_limit, "configurable": {"thread_id": thread_id}},
    )
    final = result["messages"][-1].content
    return coerce_content_to_text(final).strip() or "(empty response)"


async def async_main() -> int:
    """Run the Discord bot process."""
    load_dotenv()
    settings = load_settings()

    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN is required for Discord mode.")

    allowed_channels = parse_allowed_channel_ids(os.getenv("DISCORD_ALLOWED_CHANNEL_IDS"))
    max_prompt_chars = int(os.getenv("DISCORD_MAX_PROMPT_CHARS", "4000"))
    cooldown_sec = float(os.getenv("DISCORD_USER_COOLDOWN_SEC", "0"))
    dev_guild_id_raw = os.getenv("DISCORD_DEV_GUILD_ID", "").strip()
    dev_guild_id = int(dev_guild_id_raw) if dev_guild_id_raw else None

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    tools = await load_sandbox_tools(settings)
    agent = build_react_agent(settings, tools=tools, checkpointer=InMemorySaver())
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)
    last_seen_by_user: dict[int, float] = {}
    # Bumped per (channel, user) on reset; included in the thread_id so the old
    # checkpointer thread is simply abandoned rather than deleted.
    generation_by_thread: dict[tuple[int | None, int], int] = defaultdict(int)
    synced = False

    @tree.command(name="mathforge", description="Ask MathForge a math or coding question")
    @app_commands.describe(
        query="Your math/coding question (optional if reset is true)",
        reset="Clear your conversation memory in this channel before asking",
    )
    async def mathforge(
        interaction: discord.Interaction,
        query: str | None = None,
        reset: bool = False,
    ) -> None:
        channel_id = interaction.channel_id
        if not is_channel_allowed(channel_id, allowed_channels):
            await interaction.response.send_message(
                "This channel is not allowed for MathForge.",
                ephemeral=True,
            )
            return

        if interaction.user is None:
            await interaction.response.send_message("Could not identify user.", ephemeral=True)
            return
        user_id = interaction.user.id
        thread_key_id = (channel_id, user_id)

        if reset:
            generation_by_thread[thread_key_id] += 1
            if query is None:
                await interaction.response.send_message(
                    "Conversation memory cleared.", ephemeral=True
                )
                return

        if query is None:
            await interaction.response.send_message(
                "Please provide a query (or set reset:true to clear memory).",
                ephemeral=True,
            )
            return

        if len(query) > max_prompt_chars:
            await interaction.response.send_message(
                f"Query too long ({len(query)} chars). Limit is {max_prompt_chars}.",
                ephemeral=True,
            )
            return

        if cooldown_sec > 0:
            now = time.time()
            last_seen = last_seen_by_user.get(user_id, 0)
            wait = cooldown_sec - (now - last_seen)
            if wait > 0:
                await interaction.response.send_message(
                    f"Cooldown active. Please wait {wait:.1f}s and try again.",
                    ephemeral=True,
                )
                return
            last_seen_by_user[user_id] = now

        thread_id = thread_key(channel_id, user_id, generation_by_thread[thread_key_id])
        await interaction.response.defer(thinking=True)
        try:
            answer = await run_query(agent, query, settings.recursion_limit, thread_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Discord query failed")
            await interaction.followup.send(f"Error: {exc}")
            return

        for part in chunk_text(answer):
            await interaction.followup.send(part)

    @client.event
    async def on_ready() -> None:
        nonlocal synced
        if synced:
            return
        if dev_guild_id is not None:
            guild = discord.Object(id=dev_guild_id)
            # In dev mode we define commands globally, then copy to a guild for fast propagation.
            tree.copy_global_to(guild=guild)
            synced_cmds = await tree.sync(guild=guild)
            logger.info("Synced %d command(s) to guild %s", len(synced_cmds), dev_guild_id)
        else:
            synced_cmds = await tree.sync()
            logger.info("Synced %d global command(s)", len(synced_cmds))
        synced = True
        logger.info("Discord bot ready as %s", client.user)

    await client.start(token)
    return 0


def main() -> None:
    """Setuptools console-script entrypoint for the Discord bot."""
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
