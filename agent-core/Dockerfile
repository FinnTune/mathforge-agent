# Bundles the whole agent-core process tree into one image: agent-core
# itself, the compiled Rust sandbox MCP server, a separate Python env for the
# sandboxed interpreter (numpy/sympy/matplotlib/scipy), and the mathkb-py MCP
# server + corpus. All four are spawned as local subprocesses of grpc_server.py
# over stdio (see agent-core/mcp_client.py) rather than independent network
# services, so they have to live in one image/container, not four.
#
# Build context must be the repo root (siblings mcp-servers/, proto/,
# scripts/ are needed): docker build -f agent-core/Dockerfile .

FROM rust:1-slim-bookworm AS sandbox-builder
WORKDIR /build
COPY mcp-servers/sandbox-rs ./mcp-servers/sandbox-rs
RUN cargo build --release --manifest-path mcp-servers/sandbox-rs/Cargo.toml

FROM python:3.12-slim-bookworm

# The mathkb venv's `qdrant-client`/`voyageai` and the sandbox venv's
# numpy/scipy build/link against these; agent-core's own venv needs a C
# compiler for a couple of its transitive deps too.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sandboxed-code interpreter — deliberately a *different* venv than
# agent-core's own, same split as the non-Docker setup (see README Quick
# Start step 1).
COPY mcp-servers/sandbox-rs/requirements.txt ./mcp-servers/sandbox-rs/requirements.txt
RUN python -m venv /opt/venvs/sandbox && \
    /opt/venvs/sandbox/bin/pip install --no-cache-dir -r mcp-servers/sandbox-rs/requirements.txt

# mathkb MCP server (search_math_knowledge) — own venv, own corpus.
COPY mcp-servers/mathkb-py ./mcp-servers/mathkb-py
RUN python -m venv /opt/venvs/mathkb && \
    /opt/venvs/mathkb/bin/pip install --no-cache-dir -e ./mcp-servers/mathkb-py

# agent-core itself. `[dev]` (not just the base deps) so grpcio-tools is
# present to regenerate the gRPC stubs below — same gitignored-generated-file
# precedent as the non-Docker setup (scripts/generate_proto.sh).
COPY proto ./proto
COPY scripts ./scripts
COPY agent-core ./agent-core
RUN python -m venv /opt/venvs/agent && \
    /opt/venvs/agent/bin/pip install --no-cache-dir -e "./agent-core[dev]" && \
    PATH="/opt/venvs/agent/bin:$PATH" bash scripts/generate_proto.sh

# Compiled sandbox MCP server binary from the builder stage.
COPY --from=sandbox-builder /build/mcp-servers/sandbox-rs/target/release/mathforge-sandbox-mcp \
    /opt/mathforge-sandbox-mcp

# Point every MCP-related setting at its baked-in path — config.py's
# repo-relative defaults are for the non-Docker layout and are never used here.
ENV MATHFORGE_SANDBOX_MCP_BIN=/opt/mathforge-sandbox-mcp \
    MATHFORGE_SANDBOX_PYTHON=/opt/venvs/sandbox/bin/python \
    MATHFORGE_MATHKB_MCP_PYTHON=/opt/venvs/mathkb/bin/python \
    MATHFORGE_MATHKB_MCP_SCRIPT=/app/mcp-servers/mathkb-py/server.py \
    MATHFORGE_GRPC_HOST=0.0.0.0 \
    PATH=/opt/venvs/agent/bin:$PATH

WORKDIR /app/agent-core
EXPOSE 50051
CMD ["mathforge-server"]
