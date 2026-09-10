//! `mathforge-sandbox-mcp` — an MCP server exposing one tool, `execute_python`,
//! that runs model-generated Python in a resource-limited subprocess.
//!
//! Speaks MCP over stdio (the standard transport for locally-spawned servers —
//! the same pattern Claude Desktop uses), so it's meant to be launched as a
//! child process by an MCP client (e.g. `agent-core`'s `langchain-mcp-adapters`
//! client), not run standalone as a long-lived network service.
//!
//! stdout is reserved for MCP JSON-RPC traffic — all logging goes to stderr.

use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::ServerInfo;
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};

use mathforge_sandbox_mcp::config::Config;
use mathforge_sandbox_mcp::runner;

/// Arguments for the `execute_python` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct ExecuteArgs {
    /// Python source to execute. `sp` (SymPy), `np`, `plt` (Matplotlib, Agg
    /// backend), and `scipy` are pre-imported; save plots under `./plots/`.
    code: String,
}

#[derive(Clone)]
struct Sandbox {
    config: Config,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl Sandbox {
    fn new(config: Config) -> Self {
        Self {
            config,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "execute_python",
        description = "Execute Python in an isolated, resource-limited subprocess (fresh interpreter, no API keys in env, CPU/memory/process rlimits, wall-clock timeout). Pre-imported names: sp (SymPy), np, plt (Matplotlib, Agg backend), scipy. Save plots under ./plots/. Returns combined stdout/stderr from the child process."
    )]
    async fn execute_python(&self, Parameters(ExecuteArgs { code }): Parameters<ExecuteArgs>) -> String {
        let result = runner::run_python_code_isolated(&code, &self.config).await;
        format!(
            "Execution result:\n{result}\n(Workspace: {}; plots directory: {})",
            self.config.workspace_root.display(),
            self.config.workspace_root.join("plots").display(),
        )
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Sandbox {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.instructions = Some(
            "Executes model-generated Python for math/coding tasks in a hardened sandbox."
                .to_string(),
        );
        info
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = Config::from_env();
    tracing::info!(?config, "starting mathforge-sandbox-mcp");

    let service = Sandbox::new(config);
    let running = service.serve(rmcp::transport::stdio()).await?;
    running.waiting().await?;
    Ok(())
}
