"""Discord interface for MathForge via a slash command.

A thin gRPC client (see ``main.py``'s module docstring for the general shape
of that split) so Discord is just another transport in front of the one
``mathforge-server`` process — the bot itself no longer builds a LangGraph
agent, loads MCP tools, or touches the checkpointer directly. The bot:

- registers `/mathforge query:<text> reset:<bool>`
- enforces optional channel allowlist and prompt length limits
- calls the streaming ``Chat`` RPC per query and concatenates its
  ``text_delta`` events into one reply (Discord doesn't stream), with
  per-channel-per-user conversation memory via ``thread_id`` (server-side,
  SQLite-backed — see ``checkpointer.py``); ``reset:true`` bumps a
  generation counter to start a fresh thread, same as before
- chunks long replies to satisfy Discord message limits
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterator

import discord
from discord import app_commands
from dotenv import load_dotenv

import chat_pb2
import chat_pb2_grpc
from main import build_channel, check_server_health, grpc_metadata

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


def thread_key(channel_id: int | None, user_id: int, generation: int) -> str:
    """Build the gRPC ``thread_id`` for one channel+user conversation."""
    return f"{channel_id}:{user_id}:{generation}"


async def run_query(stub: chat_pb2_grpc.MathForgeChatStub, query: str, thread_id: str) -> str:
    """Run one query against the gRPC server and return the concatenated final text."""
    request = chat_pb2.ChatRequest(thread_id=thread_id, query=query)
    parts: list[str] = []
    async for event in stub.Chat(request, metadata=grpc_metadata()):
        kind = event.WhichOneof("event")
        if kind == "text_delta":
            parts.append(event.text_delta.text)
        elif kind == "error":
            raise RuntimeError(event.error.message)
        elif kind == "done":
            break
    return "".join(parts).strip() or "(empty response)"


async def async_main() -> int:
    """Run the Discord bot process."""
    load_dotenv()
    logging.basicConfig(
        level=getattr(logging, os.getenv("MATHFORGE_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN is required for Discord mode.")

    allowed_channels = parse_allowed_channel_ids(os.getenv("DISCORD_ALLOWED_CHANNEL_IDS"))
    max_prompt_chars = int(os.getenv("DISCORD_MAX_PROMPT_CHARS", "4000"))
    cooldown_sec = float(os.getenv("DISCORD_USER_COOLDOWN_SEC", "0"))
    dev_guild_id_raw = os.getenv("DISCORD_DEV_GUILD_ID", "").strip()
    dev_guild_id = int(dev_guild_id_raw) if dev_guild_id_raw else None

    target = os.getenv("MATHFORGE_GRPC_TARGET", "127.0.0.1:50051")
    # The gRPC channel must stay open for the whole bot process lifetime,
    # same reasoning as the checkpointer connection did before this phase.
    async with build_channel(target) as channel:
        stub = chat_pb2_grpc.MathForgeChatStub(channel)
        model = await check_server_health(stub)
        if model is None:
            print(
                f"Could not reach MathForge gRPC server at {target}. "
                "Is `mathforge-server` running?"
            )
            return 1
        logger.info("Connected to MathForge gRPC server at %s (model=%s)", target, model)

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
                await interaction.response.send_message(
                    "Could not identify user.", ephemeral=True
                )
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
                answer = await run_query(stub, query, thread_id)
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
                # In dev mode we define commands globally, then copy to a guild for fast sync.
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
