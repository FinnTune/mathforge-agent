# MathForge — LangGraph + Claude math and code agent

A ReAct-style agent built with **LangGraph** and **Claude (Anthropic)**. It solves math and coding tasks by generating Python, running it in a **hardened, isolated process** via an **MCP** tool server, and explaining the results.

## Architecture

MathForge is a polyglot monorepo, one directory per component:

```
agent-core/          Python — LangGraph ReAct agent, CLI, Discord bot
mcp-servers/
  sandbox-rs/         Rust — MCP server that executes model-generated Python
                       in a resource-limited subprocess (the trust boundary)
```

`agent-core` talks to `sandbox-rs` as an **MCP client** (`langchain-mcp-adapters`), over stdio — the same transport pattern Claude Desktop uses to run local MCP servers. This is in-progress toward a larger rearchitecture (RAG tool server, a hand-rolled multi-node LangGraph, a gRPC service layer); see commit history for what's landed.

## Security model

- **Not a cryptographic sandbox.** Untrusted code still runs as your user on your machine, with filesystem access under the chosen workspace and whatever the Python standard library allows.
- **Process isolation + resource limits (`mcp-servers/sandbox-rs`):** a fresh interpreter per run, a stripped environment (no `ANTHROPIC_API_KEY` in the child), and — applied via `setrlimit` in the child before `exec`, not just a wall-clock timeout — a CPU-time cap, a memory (`RLIMIT_AS`) cap, and an open-file cap. On timeout the whole process group is killed, not just the direct child, so a script's own subprocesses can't outlive it. Captured output is read with a running cap rather than buffered to EOF then truncated, bounding the parent's memory regardless of how much the child tries to write.
- **Deliberately not relying on `RLIMIT_NPROC`** as a fork-bomb guard — on Linux it caps the real UID's total process count system-wide, not the child's subtree, so it's the wrong tool here. A correct per-subtree guard needs a cgroup with `pids.max`; that's known future work, not implemented yet.
- **Stronger isolation** (containers, gVisor, remote sandboxes) is recommended if you accept **arbitrary** prompts or untrusted users.

## Quick start

```bash
git clone https://github.com/yourusername/mathforge-agent.git
cd mathforge-agent

# 1. Build the sandbox MCP server (Rust)
cargo build --release --manifest-path mcp-servers/sandbox-rs/Cargo.toml
# It needs its own Python with the scientific stack (separate from agent-core's venv):
python3 -m venv mcp-servers/sandbox-rs/.venv
mcp-servers/sandbox-rs/.venv/bin/pip install -r mcp-servers/sandbox-rs/requirements.txt

# 2. Set up agent-core (Python)
python3 -m venv .venv && source .venv/bin/activate
pip install -e "./agent-core[dev]"
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env, and point MATHFORGE_SANDBOX_PYTHON at the venv from step 1:
#   MATHFORGE_SANDBOX_PYTHON=mcp-servers/sandbox-rs/.venv/bin/python
python agent-core/main.py
```

CLI options:

- `python agent-core/main.py` — stream the assistant reply to the terminal.
- `python agent-core/main.py --no-stream` — wait for the full reply (easier for scripting).
- `python agent-core/main.py --verbose` — print tool outputs (including executed code results).

You can also run `mathforge` if the venv's `bin` is on your `PATH` (console script from `agent-core/pyproject.toml`).

### Conversation memory

The CLI and Discord bot remember earlier turns in a session (e.g. "now plot that", "solve it symbolically instead"), backed by an in-memory LangGraph checkpointer. Memory lives only in the running process — it's lost on restart and never written to disk.

- **CLI:** one conversation per run. Type `reset` at the `You:` prompt to start a fresh thread.
- **Discord:** one conversation per user per channel. Add `reset:true` to `/mathforge` to clear it (optionally combined with a new `query` in the same call).

## Example queries

After the prompt `You:`, try pasting one of these (the agent will run Python in the sandbox and explain the result):

**Symbolic (SymPy)**

- `Expand (x + y)**4 and collect terms in x.`
- `Solve x**2 - 5*x + 6 == 0 for x.`
- `Integrate sin(x)*cos(x) with respect to x.`

**Numeric (NumPy / SciPy)**

- `What is the eigenvalue with the largest magnitude of the matrix [[2,1],[1,3]]? Show the value to 6 decimal places.`
- `Use scipy to compute the definite integral of exp(-x**2) from -2 to 2.`

**Plot (Matplotlib)**

- `Plot sin(x) and cos(x) from 0 to 2*pi on the same axes, save to ./plots/trig.png, and describe the figure.`

**Short coding**

- `Write a function that returns the first 15 Fibonacci numbers as a list, run it, and print the result.`

Use `python agent-core/main.py --verbose` if you want to see tool output (stdout from executed code) in the terminal as well.

## Configuration

See `.env.example`. Notable variables:

| Variable | Role |
|----------|------|
| `ANTHROPIC_API_KEY` | Required for the chat model. |
| `MATHFORGE_MODEL` | Claude model id (default `claude-sonnet-4-6`). |
| `MATHFORGE_TEMPERATURE` | Sampling temperature. |
| `MATHFORGE_MAX_TOKENS` | Optional cap on completion tokens. |
| `MATHFORGE_RECURSION_LIMIT` | LangGraph step / recursion budget for the agent. |
| `MATHFORGE_WORKSPACE_ROOT` | Working directory for code execution and `plots/`. |
| `MATHFORGE_CODE_TIMEOUT_SEC` | Per-execution timeout (also the sandbox's CPU-time rlimit). |
| `MATHFORGE_SANDBOX_MCP_BIN` | Path to the compiled `mathforge-sandbox-mcp` binary. |
| `MATHFORGE_SANDBOX_PYTHON` | Python interpreter the sandbox invokes (needs numpy/sympy/matplotlib/scipy). |
| `MATHFORGE_SANDBOX_MAX_MEMORY_MB` | Sandbox `RLIMIT_AS` cap in MB. |
| `MATHFORGE_SANDBOX_MAX_OUTPUT_BYTES` | Cap on captured sandbox stdout+stderr. |
| `MATHFORGE_LOG_LEVEL` | `logging` level for the app. |
| `DISCORD_BOT_TOKEN` | Required only for `mathforge-discord`. |
| `DISCORD_ALLOWED_CHANNEL_IDS` | Optional comma-separated channel allowlist. |
| `DISCORD_MAX_PROMPT_CHARS` | Prompt-length guardrail for slash command input. |
| `DISCORD_USER_COOLDOWN_SEC` | Optional per-user cooldown for Discord usage. |
| `DISCORD_DEV_GUILD_ID` | Optional guild ID for faster command syncing in development. |

Optional **LangSmith** tracing: set `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` as in the LangChain docs.

## Discord integration

MathForge can run as a Discord slash-command bot using the same agent and sandbox.

1. Create a Discord app/bot in the Discord Developer Portal.
2. Copy the bot token to `.env` as `DISCORD_BOT_TOKEN=...`.
3. Invite the bot with scopes `bot` and `applications.commands`.
4. Run:

```bash
mathforge-discord
```

or:

```bash
python agent-core/discord_bot.py
```

Then call it in Discord:

```text
/mathforge query: Solve x^2 - 5x + 6 = 0 and explain each step.
```

Follow-ups in the same channel remember prior turns; use `/mathforge reset:true` to start over (see [Conversation memory](#conversation-memory)).

Recommended safety settings in `.env`:

- `DISCORD_ALLOWED_CHANNEL_IDS` to restrict usage to known channels.
- `DISCORD_MAX_PROMPT_CHARS` to cap prompt length.
- `DISCORD_USER_COOLDOWN_SEC` to avoid spam.
- `DISCORD_DEV_GUILD_ID` during development for fast command sync.

## Development

```bash
# agent-core (Python)
cd agent-core
ruff check .
ANTHROPIC_API_KEY=dummy-for-ci pytest -q

# mcp-servers/sandbox-rs (Rust) — needs MATHFORGE_SANDBOX_PYTHON set to a
# python3 with numpy/sympy/matplotlib/scipy (see its requirements.txt)
cd mcp-servers/sandbox-rs
cargo clippy --all-targets -- -D warnings
MATHFORGE_SANDBOX_PYTHON=/path/to/venv/bin/python cargo test
```

CI runs both: Ruff + pytest on Python 3.11–3.13 for `agent-core`, and `cargo clippy`/`cargo test` for `sandbox-rs`.

## Requirements

Python **3.11+** for `agent-core`, Rust (stable) for `mcp-servers/sandbox-rs`, plus a separate Python 3 with NumPy/SciPy/SymPy/Matplotlib for whatever `MATHFORGE_SANDBOX_PYTHON` points at. Dependencies are declared in each component's own manifest (`agent-core/pyproject.toml`, `mcp-servers/sandbox-rs/Cargo.toml` and `requirements.txt`).

## License

Copyright (C) 2026 Andre Teetor

This project is licensed under the GNU General Public License v2.0 —
see the [LICENSE](LICENSE) file for details.
