# MathForge — LangGraph + Claude math and code agent

A ReAct-style agent built with **LangGraph** and **Claude (Anthropic)**. It solves math and coding tasks by generating Python, running it in a **separate process** with a **minimal environment** (API keys are not passed to the child), and explaining the results.

## Security model

- **Not a cryptographic sandbox.** Untrusted code still runs as your user on your machine, with filesystem access under the chosen workspace and whatever the Python standard library allows.
- **Compared to an in-process REPL:** execution uses `subprocess` + a fresh interpreter, strict timeouts, stripped environment (no `ANTHROPIC_API_KEY` in the child), and capped captured output.
- **Stronger isolation** (containers, gVisor, remote sandboxes) is recommended if you accept **arbitrary** prompts or untrusted users.

## Quick start

```bash
git clone https://github.com/yourusername/mathforge-agent.git
cd mathforge-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env
python main.py
```

CLI options:

- `python main.py` — stream the assistant reply to the terminal.
- `python main.py --no-stream` — wait for the full reply (easier for scripting).
- `python main.py --verbose` — print tool outputs (including executed code results).

You can also run `mathforge` if the venv’s `bin` is on your `PATH` (console script from `pyproject.toml`).

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

Use `python main.py --verbose` if you want to see tool output (stdout from executed code) in the terminal as well.

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
| `MATHFORGE_CODE_TIMEOUT_SEC` | Per-execution timeout in the subprocess. |
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
python discord_bot.py
```

Then call it in Discord:

```text
/mathforge query: Solve x^2 - 5x + 6 = 0 and explain each step.
```

Recommended safety settings in `.env`:

- `DISCORD_ALLOWED_CHANNEL_IDS` to restrict usage to known channels.
- `DISCORD_MAX_PROMPT_CHARS` to cap prompt length.
- `DISCORD_USER_COOLDOWN_SEC` to avoid spam.
- `DISCORD_DEV_GUILD_ID` during development for fast command sync.

## Development

```bash
ruff check .
pytest -q
```

CI runs Ruff and pytest on Python 3.11–3.13.

## Requirements

Python **3.11+**. Dependencies are declared in `pyproject.toml` (scientific stack: NumPy, SciPy, SymPy, Matplotlib for the sandbox preamble).

## License

Copyright (C) 2026 Andre Teetor

This project is licensed under the GNU General Public License v2.0 —
see the [LICENSE](LICENSE) file for details.
