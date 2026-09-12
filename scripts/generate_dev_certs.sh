#!/usr/bin/env bash
# Generates a throwaway self-signed cert/key pair for local TLS testing of
# grpc_server.py / gateway-rs — covers localhost and 127.0.0.1, valid ~2
# years. NOT a real CA and not meant for anything beyond local/dev
# demonstration; certs/ is gitignored so nothing here is ever committed.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p certs
# `openssl req -x509` applies the `v3_ca` extensions section by default,
# which sets `basicConstraints = CA:TRUE` — rustls (what tonic/gateway-rs
# uses) correctly rejects a CA-flagged cert presented as a leaf/end-entity
# cert ("CaUsedAsEndEntity"), so CA:FALSE must be set explicitly here.
openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
  -keyout certs/server.key -out certs/server.crt \
  -subj "/CN=localhost" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

echo "Generated certs/server.crt, certs/server.key"
echo
echo "To use them:"
echo "  MATHFORGE_GRPC_TLS_CERT=certs/server.crt MATHFORGE_GRPC_TLS_KEY=certs/server.key    # grpc_server.py"
echo "  MATHFORGE_GATEWAY_TLS_CERT=certs/server.crt MATHFORGE_GATEWAY_TLS_KEY=certs/server.key  # gateway-rs"
echo "  MATHFORGE_GRPC_TLS_CA=certs/server.crt       # clients (main.py / discord_bot.py)"
