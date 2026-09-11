#!/usr/bin/env bash
# Regenerates agent-core/chat_pb2*.py from proto/chat.proto.
# Requires grpcio-tools (agent-core's `dev` extra: pip install -e "./agent-core[dev]").
# Generated files are gitignored — this script is the "build step" for them,
# the same way `cargo build` is for mcp-servers/sandbox-rs.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m grpc_tools.protoc \
  -I proto \
  --python_out=agent-core \
  --grpc_python_out=agent-core \
  --pyi_out=agent-core \
  proto/chat.proto

echo "Generated agent-core/chat_pb2.py, chat_pb2_grpc.py, chat_pb2.pyi"
