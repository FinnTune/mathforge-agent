# MathForge — LangGraph + Claude math and code agent

An agent built with **LangGraph** and **Claude (Anthropic)** using a hand-rolled planner → tools → verifier → responder graph (not just a prebuilt ReAct loop — see [Agent design](#agent-design)), served over **gRPC** (see [gRPC service](#grpc-service)) so the CLI and Discord bot are thin clients of one long-running server, optionally fronted by a **Rust gRPC gateway** for auth and rate limiting (see [Rust gRPC gateway](#rust-grpc-gateway)). It solves math and coding tasks by generating Python, running it in a **hardened, isolated process** via an **MCP** tool server, grounding answers in a **RAG** knowledge base over a second MCP server, self-checking its own work before answering, and remembering conversations across restarts.

## Architecture

MathForge is a polyglot monorepo, one directory per component:

```
agent-core/          Python — LangGraph agent (see "Agent design"); one gRPC
                      server (grpc_server.py) + two thin clients (CLI, Discord)
mcp-servers/
  sandbox-rs/         Rust MCP server — executes model-generated Python in a
                       resource-limited subprocess (the trust boundary)
  mathkb-py/          Python MCP server — search_math_knowledge, retrieval
                       over a Qdrant-backed corpus of reference notes
proto/chat.proto      gRPC service definition — agent-core's server + clients
                       AND gateway-rs generate their stubs from this one file
gateway-rs/           Rust gRPC gateway (tonic) — optional auth + rate
                       limiting in front of grpc_server.py (see below)
docker-compose.yml    Qdrant (vector store behind mathkb-py)
```

Within `agent-core`, `grpc_server.py` builds the agent graph once (loading
the MCP tools, opening the checkpointer) and serves it over gRPC
(`proto/chat.proto`); `main.py` (CLI) and `discord_bot.py` are thin clients
that just call the streaming `Chat` RPC — see [gRPC service](#grpc-service).
`agent-core` itself talks to `sandbox-rs` and `mathkb-py` as an **MCP
client** (`langchain-mcp-adapters`), over stdio — the same transport pattern
Claude Desktop uses to run local MCP servers. `gateway-rs` sits in front of
`grpc_server.py`, speaking the identical `MathForgeChat` service and
forwarding to it — clients don't need to know which one they're talking to,
see [Rust gRPC gateway](#rust-grpc-gateway).

## Security model

- **Not a cryptographic sandbox.** Untrusted code still runs as your user on your machine, with filesystem access under the chosen workspace and whatever the Python standard library allows.
- **Process isolation + resource limits (`mcp-servers/sandbox-rs`):** a fresh interpreter per run, a stripped environment (no `ANTHROPIC_API_KEY` in the child), and — applied via `setrlimit` in the child before `exec`, not just a wall-clock timeout — a CPU-time cap, a memory (`RLIMIT_AS`) cap, and an open-file cap. On timeout the whole process group is killed, not just the direct child, so a script's own subprocesses can't outlive it. Captured output is read with a running cap rather than buffered to EOF then truncated, bounding the parent's memory regardless of how much the child tries to write.
- **Deliberately not relying on `RLIMIT_NPROC`** as a fork-bomb guard — on Linux it caps the real UID's total process count system-wide, not the child's subtree, so it's the wrong tool here.
- **Fork-bomb guard via cgroup v2 `pids.max`** (`mcp-servers/sandbox-rs/src/cgroup.rs`): the correct per-subtree limit `RLIMIT_NPROC` can't provide. Opportunistic, not required — it needs the sandbox process to already be sitting in a *delegated* cgroup v2 subtree (true under a normal systemd user session; **not** guaranteed in containers without delegation, non-systemd init, or non-Linux). Falls back to the rlimits/process-group protections above alone when cgroups aren't usable on the host — never fails a run because of this. On timeout, also SIGKILLs the whole cgroup (`cgroup.kill`), which is stronger than the process-group kill alone since it catches anything that escaped the group (e.g. via `setsid()`). `MATHFORGE_SANDBOX_MAX_PROCESSES` (default 64).
- **Stronger isolation** (containers, gVisor, remote sandboxes) is recommended if you accept **arbitrary** prompts or untrusted users.

## Quick start

The fastest path is Docker (below); see [Manual setup](#manual-setup) for
running each component directly on the host instead (useful for hacking on
one piece at a time).

### Docker

```bash
git clone https://github.com/yourusername/mathforge-agent.git
cd mathforge-agent
cp .env.example .env
# Fill in ANTHROPIC_API_KEY (required), VOYAGE_API_KEY (optional, for RAG),
# and MATHFORGE_GATEWAY_API_KEYS (required — pick any string; the `cli`
# service below needs the same value in MATHFORGE_GRPC_API_KEY).

docker compose up --build -d qdrant agent gateway

# Optional: embed the RAG corpus into Qdrant (one-off; needs VOYAGE_API_KEY)
docker compose run --rm ingest

# Talk to it
docker compose run --rm cli
```

`agent` bundles agent-core + both MCP servers (they're stdio subprocesses of
`grpc_server.py`, not independent services — see `agent-core/Dockerfile`);
`gateway` is `gateway-rs` (auth + rate limiting) in front of it; `cli` is a
throwaway container running `main.py` against the gateway. `docker compose
logs -f agent` to watch the server; `docker compose down -v` to tear
everything down (including Qdrant's data volume).

### Manual setup

```bash
git clone https://github.com/yourusername/mathforge-agent.git
cd mathforge-agent

# 1. Build the sandbox MCP server (Rust)
cargo build --release --manifest-path mcp-servers/sandbox-rs/Cargo.toml
# It needs its own Python with the scientific stack (separate from agent-core's venv):
python3 -m venv mcp-servers/sandbox-rs/.venv
mcp-servers/sandbox-rs/.venv/bin/pip install -r mcp-servers/sandbox-rs/requirements.txt

# 2. Set up the RAG knowledge base server (Python) and its vector store
python3 -m venv mcp-servers/mathkb-py/.venv
mcp-servers/mathkb-py/.venv/bin/pip install -e mcp-servers/mathkb-py
docker compose up -d qdrant

# 3. Set up agent-core (Python) and generate the gRPC stubs from proto/chat.proto
python3 -m venv .venv && source .venv/bin/activate
pip install -e "./agent-core[dev]"
bash scripts/generate_proto.sh
cp .env.example .env
# Set ANTHROPIC_API_KEY and VOYAGE_API_KEY in .env, and point MATHFORGE_SANDBOX_PYTHON
# at the venv from step 1:
#   MATHFORGE_SANDBOX_PYTHON=mcp-servers/sandbox-rs/.venv/bin/python

# 4. Embed the knowledge base corpus into Qdrant (needs VOYAGE_API_KEY set above)
set -a && source .env && set +a
mcp-servers/mathkb-py/.venv/bin/python mcp-servers/mathkb-py/ingest.py

# 5. Start the gRPC server (keep this running in its own terminal)
mathforge-server
```

Steps 2 and 4 (the RAG server/knowledge base) are optional — without them the
`search_math_knowledge` tool still loads but every call returns a clear
credentials/connection error, and the agent gracefully falls back to
answering from `execute_python` alone (see [RAG knowledge base](#rag-knowledge-base)).

With the server running, in a **second terminal** (same venv):

```bash
source .venv/bin/activate
python agent-core/main.py
```

CLI options:

- `python agent-core/main.py` — stream the assistant reply to the terminal.
- `python agent-core/main.py --no-stream` — wait for the full reply (easier for scripting).
- `python agent-core/main.py --verbose` — print tool calls/outputs as the agent works.

You can also run `mathforge` (and `mathforge-server`) if the venv's `bin` is on your `PATH` (console scripts from `agent-core/pyproject.toml`). If the server isn't reachable, the CLI says so immediately (`Could not reach MathForge gRPC server at ... Is mathforge-server running?`) rather than failing on the first query.

### Conversation memory

The CLI and Discord bot remember earlier turns in a session (e.g. "now plot that", "solve it symbolically instead"), backed by a **SQLite-backed** LangGraph checkpointer (`checkpointer.py`) — conversation history survives a process restart, stored at `MATHFORGE_CHECKPOINT_DB_PATH` (default `agent-core/.mathforge/checkpoints.sqlite3`, gitignored).

- **CLI:** one conversation per run. Type `reset` at the `You:` prompt to start a fresh thread (the old thread's history stays in the DB, just unreferenced — nothing is deleted).
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

## Agent design

`agent.py` builds a hand-rolled LangGraph graph, not the prebuilt
`create_react_agent` — four nodes with distinct jobs instead of one
undifferentiated LLM-alternates-with-tools loop:

```
START → planner ⇄ tools           planner reasons and calls execute_python /
              │                    search_math_knowledge as needed; tools
              ▼                    node runs every tool_call from one turn
          verifier                 concurrently (ToolNode, reused as-is)
              │
    insufficient, retries left ──► back to planner, with feedback
              │
    sufficient, or retries exhausted
              ▼
          responder → END
```

- **planner** — reasons step-by-step, calls tools, and once done writes an internal working summary (not shown to the user).
- **verifier** — a separate LLM call that reviews the planner's tool outputs and summary for correctness/completeness, forced into a structured `sufficient`/`feedback` decision via tool-calling. If insufficient, it sends feedback back to the planner for another pass — capped at `MATHFORGE_VERIFICATION_MAX_ATTEMPTS` (default 2) so a persistently unsatisfied verifier still terminates instead of looping forever.
- **responder** — a separate LLM call, not tool-bound, that writes the actual friendly, cited, user-facing final answer from the full transcript.

This means a single user turn can involve multiple Claude calls (planner → tools → planner → verifier → \[planner again, if the verifier pushed back] → responder) before you see a reply — run with `--verbose` to watch each step.

## gRPC service

`proto/chat.proto` defines `MathForgeChat`: a streaming `Chat` RPC (one
query in, a stream of `ChatEvent`s back — `text_delta`, `tool_call`,
`tool_result`, `done`, `error`) plus a `HealthCheck` unary RPC. `grpc_server.py`
builds the agent graph **once** at startup (MCP tools, checkpointer, the
compiled graph) and serves many client requests against that same instance —
`main.py` and `discord_bot.py` no longer each build their own; they're thin
clients that call `Chat` and print/concatenate `text_delta` events.

- Only `text_delta` events carry the responder node's user-facing text — the
  server applies the same `langgraph_node` filtering that used to live in
  `main.py` (see [Agent design](#agent-design)) before this phase, so
  planner/verifier internals never reach a client.
- `tool_call`/`tool_result` events (planner's real tool calls only — the
  verifier's forced `VerificationDecision` call is internal plumbing and
  never surfaced) print under `--verbose` in the CLI.
- "Reset" is purely client-side — there's no `ResetThread` RPC. Clients just
  start sending a new `thread_id`; the old thread's history simply stays
  unreferenced in the server's SQLite DB, same as before this phase.
- **No auth on the server itself** — that's handled by the optional
  `gateway-rs` in front of it, not the server; see
  [Rust gRPC gateway](#rust-grpc-gateway).
- **Plaintext by default, TLS opt-in** — see [TLS](#tls) below.

Regenerate stubs after editing the proto: `bash scripts/generate_proto.sh`
(generated `agent-core/chat_pb2*.py` are gitignored — same "build step
required" precedent as the Rust sandbox binary and the mathkb venv).
`gateway-rs` generates its own stubs from the same file at `cargo build`
time (`build.rs`), no separate step needed.

## Rust gRPC gateway

`gateway-rs` is an optional `tonic` (Rust) gRPC gateway that implements the
*same* `MathForgeChat` service as `grpc_server.py` and transparently forwards
every call to it — but requires an API key and enforces a per-key rate limit
first. It's the systems-level counterpart to `mcp-servers/sandbox-rs`: auth
and rate limiting live here in Rust, agent orchestration stays in Python.

- **Auth:** `Chat` calls need `x-api-key` metadata matching one of
  `MATHFORGE_GATEWAY_API_KEYS` (comma-separated) — missing/invalid key →
  `UNAUTHENTICATED`, the upstream server is never even called.
  `MATHFORGE_GATEWAY_API_KEYS` is **required**; the gateway refuses to start
  without it (fail-closed — an auth gateway that's insecure by default would
  defeat the point). `HealthCheck` is deliberately left unauthenticated so a
  probe/load-balancer can check the gateway itself is up without a key.
- **Rate limiting:** per-API-key quota (`governor`), `MATHFORGE_GATEWAY_RATE_LIMIT_PER_MINUTE`
  (default 30) — checked *after* auth, so unauthenticated traffic can't burn
  a legitimate key's budget. Exceeding it → `RESOURCE_EXHAUSTED`.
- **Using it:** run `mathforge-gateway` (in `gateway-rs/`) alongside
  `mathforge-server`, pointed at it via `MATHFORGE_GATEWAY_UPSTREAM` (default
  `http://127.0.0.1:50051`). Then point clients at the gateway instead of the
  server: `MATHFORGE_GRPC_TARGET=127.0.0.1:50052` and
  `MATHFORGE_GRPC_API_KEY=<one of the configured keys>`. This is entirely
  optional — talking directly to `grpc_server.py` (the Quick Start default,
  `MATHFORGE_GRPC_API_KEY` unset) works exactly as in the no-gateway setup.
- **TLS is opt-in, on both connection surfaces independently** — see
  [TLS](#tls) below.

```bash
cd gateway-rs
cargo build --release
MATHFORGE_GATEWAY_API_KEYS=some-secret-key ./target/release/mathforge-gateway
```

## TLS

Every gRPC hop (`grpc_server.py`'s listening port, `gateway-rs`'s listening
port, `gateway-rs`'s connection to `grpc_server.py`, and every client) is
plaintext by default — unchanged from before this feature — and switches to
TLS independently per hop, based purely on whether that hop's env vars are
set. Generate a throwaway self-signed dev cert (covers `localhost`/
`127.0.0.1`, **not** a real CA — local/dev demonstration only):

```bash
./scripts/generate_dev_certs.sh   # writes certs/server.crt, certs/server.key (gitignored)
```

- **`grpc_server.py`:** set `MATHFORGE_GRPC_TLS_CERT`/`MATHFORGE_GRPC_TLS_KEY`
  (both required together) to serve TLS instead of plaintext.
- **Clients** (`main.py`/`discord_bot.py`, talking to either the server or
  the gateway): set `MATHFORGE_GRPC_TLS_CA` to the CA/cert PEM to trust.
- **`gateway-rs`'s listening side** (what clients connect to): set
  `MATHFORGE_GATEWAY_TLS_CERT`/`MATHFORGE_GATEWAY_TLS_KEY` (both required
  together).
- **`gateway-rs`'s upstream side** (gateway → `grpc_server.py`): set
  `MATHFORGE_GATEWAY_UPSTREAM_TLS_CA` to the CA/cert PEM to trust, **and**
  change `MATHFORGE_GATEWAY_UPSTREAM` to an `https://` URL — `tonic` only
  performs the TLS handshake for an `https://` endpoint; an `http://`
  endpoint with a CA configured would silently stay plaintext instead of
  failing loudly, so the gateway refuses to start in that combination.

Example: TLS on every hop, using the one generated dev cert everywhere
(a real deployment would use distinct certs per hop):

```bash
MATHFORGE_GRPC_TLS_CERT=certs/server.crt MATHFORGE_GRPC_TLS_KEY=certs/server.key mathforge-server

MATHFORGE_GATEWAY_UPSTREAM=https://127.0.0.1:50051 \
MATHFORGE_GATEWAY_UPSTREAM_TLS_CA=certs/server.crt \
MATHFORGE_GATEWAY_TLS_CERT=certs/server.crt MATHFORGE_GATEWAY_TLS_KEY=certs/server.key \
MATHFORGE_GATEWAY_API_KEYS=some-secret-key \
  ./gateway-rs/target/release/mathforge-gateway

MATHFORGE_GRPC_TARGET=127.0.0.1:50052 MATHFORGE_GRPC_API_KEY=some-secret-key \
MATHFORGE_GRPC_TLS_CA=certs/server.crt mathforge
```

## RAG knowledge base

`mcp-servers/mathkb-py` is a second MCP server exposing one tool,
`search_math_knowledge`. It retrieves from a small, **self-authored**
corpus of reference notes (`mcp-servers/mathkb-py/corpus/*.md` — linear
algebra, SciPy integration, SymPy, Matplotlib, general numerical patterns;
not scraped/copied docs, so there's no licensing question) embedded with
**Voyage AI** (`voyage-3-lite`) and stored in **Qdrant**.

- **Ingest** (run once, or after editing `corpus/`): `python mcp-servers/mathkb-py/ingest.py` — chunks the corpus, embeds it, and upserts into the `QDRANT_COLLECTION` collection. Idempotent — re-running on an unchanged corpus overwrites in place rather than duplicating.
- **Try it:** ask something close to one of the corpus notes, e.g. `What does scipy.integrate.quad return, and how do I check its accuracy?` — `--verbose` shows the raw retrieved snippet(s) with source/section/score, and the final answer cites the source file.
- **Graceful degradation:** if Qdrant is unreachable or `VOYAGE_API_KEY` is unset/invalid, `search_math_knowledge` still loads as a tool (listing tool schemas doesn't touch Voyage) but returns a clear error string when called — `langchain-mcp-adapters` surfaces this as a normal tool result, not a crash, so the model just explains it couldn't check its notes and answers from `execute_python` alone.

## Configuration

See `.env.example`. Notable variables:

| Variable | Role |
|----------|------|
| `ANTHROPIC_API_KEY` | Required for the chat model. |
| `MATHFORGE_MODEL` | Claude model id (default `claude-sonnet-4-6`). |
| `MATHFORGE_TEMPERATURE` | Sampling temperature. |
| `MATHFORGE_MAX_TOKENS` | Optional cap on completion tokens. |
| `MATHFORGE_RECURSION_LIMIT` | LangGraph node-visit budget for the whole graph. |
| `MATHFORGE_VERIFICATION_MAX_ATTEMPTS` | Cap on planner↔verifier retry loops per turn (default 2). |
| `MATHFORGE_CHECKPOINT_DB_PATH` | SQLite conversation-memory DB path. |
| `MATHFORGE_WORKSPACE_ROOT` | Working directory for code execution and `plots/`. |
| `MATHFORGE_CODE_TIMEOUT_SEC` | Per-execution timeout (also the sandbox's CPU-time rlimit). |
| `MATHFORGE_SANDBOX_MCP_BIN` | Path to the compiled `mathforge-sandbox-mcp` binary. |
| `MATHFORGE_SANDBOX_PYTHON` | Python interpreter the sandbox invokes (needs numpy/sympy/matplotlib/scipy). |
| `MATHFORGE_SANDBOX_MAX_MEMORY_MB` | Sandbox `RLIMIT_AS` cap in MB. |
| `MATHFORGE_SANDBOX_MAX_OUTPUT_BYTES` | Cap on captured sandbox stdout+stderr. |
| `MATHFORGE_SANDBOX_MAX_PROCESSES` | cgroup v2 `pids.max` fork-bomb guard (default 64; opportunistic, see Security model). |
| `VOYAGE_API_KEY` | Required to actually retrieve anything via `search_math_knowledge` / to run `ingest.py`. |
| `VOYAGE_MODEL` | Voyage embedding model (default `voyage-3-lite`). |
| `QDRANT_URL` | Qdrant REST endpoint (default `http://localhost:6333`). |
| `QDRANT_COLLECTION` | Qdrant collection name (default `mathforge-math-notes`). |
| `MATHFORGE_MATHKB_MCP_PYTHON` | Interpreter for the mathkb MCP server. |
| `MATHFORGE_MATHKB_MCP_SCRIPT` | Path to `mcp-servers/mathkb-py/server.py`. |
| `MATHFORGE_GRPC_HOST` / `MATHFORGE_GRPC_PORT` | Server bind address (default `127.0.0.1:50051`). |
| `MATHFORGE_GRPC_TARGET` | **Client-only**: server (or gateway) address to connect to. |
| `MATHFORGE_GRPC_API_KEY` | **Client-only**: required if `MATHFORGE_GRPC_TARGET` points at the gateway. |
| `MATHFORGE_GATEWAY_HOST` / `MATHFORGE_GATEWAY_PORT` | Gateway bind address (default `127.0.0.1:50052`). |
| `MATHFORGE_GATEWAY_UPSTREAM` | Server address the gateway forwards to (default `http://127.0.0.1:50051`). |
| `MATHFORGE_GATEWAY_API_KEYS` | **Required to start the gateway** — comma-separated allowed keys. |
| `MATHFORGE_GATEWAY_RATE_LIMIT_PER_MINUTE` | Per-API-key quota before `RESOURCE_EXHAUSTED` (default 30). |
| `MATHFORGE_LOG_LEVEL` | `logging` level for the app. |
| `DISCORD_BOT_TOKEN` | Required only for `mathforge-discord`. |
| `DISCORD_ALLOWED_CHANNEL_IDS` | Optional comma-separated channel allowlist. |
| `DISCORD_MAX_PROMPT_CHARS` | Prompt-length guardrail for slash command input. |
| `DISCORD_USER_COOLDOWN_SEC` | Optional per-user cooldown for Discord usage. |
| `DISCORD_DEV_GUILD_ID` | Optional guild ID for faster command syncing in development. |

Optional **LangSmith** tracing: set `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` as in the LangChain docs.

## Discord integration

MathForge can run as a Discord slash-command bot — another thin gRPC client
of the same `mathforge-server`, which must already be running.

1. Create a Discord app/bot in the Discord Developer Portal.
2. Copy the bot token to `.env` as `DISCORD_BOT_TOKEN=...`.
3. Invite the bot with scopes `bot` and `applications.commands`.
4. With `mathforge-server` running (see Quick Start), run:

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
# agent-core (Python) — regenerate gRPC stubs first (gitignored, see above)
bash scripts/generate_proto.sh
cd agent-core
ruff check .
ANTHROPIC_API_KEY=dummy-for-ci pytest -q

# mcp-servers/sandbox-rs (Rust) — needs MATHFORGE_SANDBOX_PYTHON set to a
# python3 with numpy/sympy/matplotlib/scipy (see its requirements.txt)
cd mcp-servers/sandbox-rs
cargo clippy --all-targets -- -D warnings
MATHFORGE_SANDBOX_PYTHON=/path/to/venv/bin/python cargo test

# mcp-servers/mathkb-py (Python) — no Docker/Voyage key needed, tests use
# qdrant-client's embedded (":memory:") mode and a fake embedder
cd mcp-servers/mathkb-py
ruff check .
pytest -q

# gateway-rs (Rust) — no Python/Docker needed, tests use a fake in-process
# upstream server (pure Rust); build.rs regenerates its own proto stubs
cd gateway-rs
cargo clippy --all-targets -- -D warnings
cargo test
```

CI runs four jobs: Ruff + pytest on Python 3.11–3.13 for both `agent-core` (after regenerating gRPC stubs) and `mathkb-py`, and `cargo clippy`/`cargo test` for both `sandbox-rs` and `gateway-rs`.

## Requirements

Python **3.11+** for `agent-core` and `mcp-servers/mathkb-py`, Rust (stable) for `mcp-servers/sandbox-rs` and `gateway-rs`, plus a separate Python 3 with NumPy/SciPy/SymPy/Matplotlib for whatever `MATHFORGE_SANDBOX_PYTHON` points at, and Docker (for Qdrant) if you want the RAG knowledge base live. `protoc` codegen (`grpcio-tools` for Python, `tonic-prost-build` for `gateway-rs`) comes from each component's own manifest — no separate `protoc` binary install needed. Dependencies are declared in each component's own manifest (`agent-core/pyproject.toml`, `mcp-servers/sandbox-rs/Cargo.toml` + `requirements.txt`, `mcp-servers/mathkb-py/pyproject.toml`, `gateway-rs/Cargo.toml`).

## License

Copyright (C) 2026 Andre Teetor

This project is licensed under the GNU General Public License v2.0 —
see the [LICENSE](LICENSE) file for details.
