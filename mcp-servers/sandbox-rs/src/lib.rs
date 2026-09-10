//! Library surface for `mathforge-sandbox-mcp`, split out from `main.rs` so
//! `tests/` can exercise `runner::run_python_code_isolated` directly (the
//! actual trust boundary) without going through the MCP protocol layer.

pub mod config;
pub mod env;
pub mod limits;
pub mod runner;
