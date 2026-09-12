"""Tests for ``config.load_settings`` and ``Settings``.

Covers required API key validation, env parsing edge cases (empty strings,
whitespace), and direct dataclass construction for non-env scenarios.
"""

from __future__ import annotations

import pytest

from config import Settings, load_settings


def test_load_settings_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        load_settings()


def test_load_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("MATHFORGE_MODEL", "claude-test")
    monkeypatch.setenv("MATHFORGE_TEMPERATURE", "0.5")
    monkeypatch.setenv("MATHFORGE_RECURSION_LIMIT", "99")
    monkeypatch.setenv("MATHFORGE_CODE_TIMEOUT_SEC", "12")
    monkeypatch.setenv("MATHFORGE_WORKSPACE_ROOT", "/tmp/ws")
    monkeypatch.setenv("MATHFORGE_LOG_LEVEL", "warning")
    monkeypatch.setenv("MATHFORGE_SANDBOX_MCP_BIN", "/opt/mathforge-sandbox-mcp")
    monkeypatch.setenv("MATHFORGE_SANDBOX_PYTHON", "/opt/venv/bin/python")
    monkeypatch.setenv("MATHFORGE_SANDBOX_MAX_MEMORY_MB", "256")
    monkeypatch.setenv("MATHFORGE_SANDBOX_MAX_OUTPUT_BYTES", "1000")
    monkeypatch.setenv("MATHFORGE_MATHKB_MCP_PYTHON", "/opt/mathkb-venv/bin/python")
    monkeypatch.setenv("MATHFORGE_MATHKB_MCP_SCRIPT", "/opt/mathkb/server.py")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "custom-notes")
    monkeypatch.setenv("VOYAGE_API_KEY", "voy-test")
    monkeypatch.setenv("VOYAGE_MODEL", "voyage-3")
    monkeypatch.setenv("MATHFORGE_CHECKPOINT_DB_PATH", "/opt/mathforge/checkpoints.sqlite3")
    monkeypatch.setenv("MATHFORGE_VERIFICATION_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("MATHFORGE_GRPC_HOST", "0.0.0.0")
    monkeypatch.setenv("MATHFORGE_GRPC_PORT", "9999")
    monkeypatch.setenv("MATHFORGE_GRPC_TLS_CERT", "/opt/mathforge/certs/server.crt")
    monkeypatch.setenv("MATHFORGE_GRPC_TLS_KEY", "/opt/mathforge/certs/server.key")

    s = load_settings()
    assert s.anthropic_api_key == "sk-test"
    assert s.model == "claude-test"
    assert s.temperature == 0.5
    assert s.recursion_limit == 99
    assert s.code_timeout_sec == 12.0
    assert s.workspace_root == "/tmp/ws"
    assert s.log_level == "WARNING"
    assert s.sandbox_mcp_bin == "/opt/mathforge-sandbox-mcp"
    assert s.sandbox_python == "/opt/venv/bin/python"
    assert s.sandbox_max_memory_mb == 256
    assert s.sandbox_max_output_bytes == 1000
    assert s.mathkb_mcp_python == "/opt/mathkb-venv/bin/python"
    assert s.mathkb_mcp_script == "/opt/mathkb/server.py"
    assert s.qdrant_url == "http://qdrant.internal:6333"
    assert s.qdrant_collection == "custom-notes"
    assert s.voyage_api_key == "voy-test"
    assert s.voyage_model == "voyage-3"
    assert s.checkpoint_db_path == "/opt/mathforge/checkpoints.sqlite3"
    assert s.verification_max_attempts == 5
    assert s.grpc_host == "0.0.0.0"
    assert s.grpc_port == 9999
    assert s.tls_cert_path == "/opt/mathforge/certs/server.crt"
    assert s.tls_key_path == "/opt/mathforge/certs/server.key"


def test_load_settings_defaults_sandbox_mcp_bin_into_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("MATHFORGE_SANDBOX_MCP_BIN", raising=False)
    monkeypatch.delenv("MATHFORGE_MATHKB_MCP_PYTHON", raising=False)
    monkeypatch.delenv("MATHFORGE_CHECKPOINT_DB_PATH", raising=False)
    s = load_settings()
    assert s.sandbox_mcp_bin.endswith("mcp-servers/sandbox-rs/target/release/mathforge-sandbox-mcp")
    assert s.sandbox_python == "python3"
    assert s.mathkb_mcp_python.endswith("mcp-servers/mathkb-py/.venv/bin/python")
    assert s.mathkb_mcp_script.endswith("mcp-servers/mathkb-py/server.py")
    assert s.qdrant_url == "http://localhost:6333"
    assert s.qdrant_collection == "mathforge-math-notes"
    assert s.voyage_model == "voyage-3-lite"
    assert s.checkpoint_db_path.endswith("agent-core/.mathforge/checkpoints.sqlite3")
    assert s.verification_max_attempts == 2
    assert s.grpc_host == "127.0.0.1"
    assert s.grpc_port == 50051
    assert s.tls_cert_path is None
    assert s.tls_key_path is None


def test_settings_dataclass_instantiation() -> None:
    s = Settings(
        anthropic_api_key="k",
        model="m",
        temperature=0.0,
        max_tokens=100,
        recursion_limit=10,
        code_timeout_sec=1.0,
        workspace_root=".",
        log_level="INFO",
        sandbox_mcp_bin="/path/to/mathforge-sandbox-mcp",
        sandbox_python="python3",
        sandbox_max_memory_mb=512,
        sandbox_max_output_bytes=256_000,
        mathkb_mcp_python="/path/to/mathkb-venv/bin/python",
        mathkb_mcp_script="/path/to/server.py",
        qdrant_url="http://localhost:6333",
        qdrant_collection="mathforge-math-notes",
        voyage_api_key="",
        voyage_model="voyage-3-lite",
        checkpoint_db_path="/path/to/checkpoints.sqlite3",
        verification_max_attempts=2,
        grpc_host="127.0.0.1",
        grpc_port=50051,
        tls_cert_path=None,
        tls_key_path=None,
    )
    assert s.max_tokens == 100


def test_max_tokens_optional_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("MATHFORGE_MAX_TOKENS", "")
    s = load_settings()
    assert s.max_tokens is None


def test_strips_api_key_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-abc  ")
    s = load_settings()
    assert s.anthropic_api_key == "sk-abc"
