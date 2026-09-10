//! Builds the minimal environment for the sandboxed Python child.
//!
//! Deliberately not a copy of the parent's `env::vars()` — that would leak
//! `ANTHROPIC_API_KEY` and any other credentials into model-generated code.
//! `HOME` points at the workspace so libraries that write under `$HOME` stay
//! inside the project tree; `MPLCONFIGDIR` must be writable for Matplotlib.

use std::collections::HashMap;
use std::path::Path;

/// Prepended to every run so generated code can assume these names exist.
pub const SANDBOX_PREAMBLE: &str = r#"
import os as _os
_os.makedirs("plots", exist_ok=True)

import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
"#;

pub fn minimal_env(workspace: &Path) -> std::io::Result<HashMap<String, String>> {
    let mpl_dir = workspace.join(".matplotlib");
    std::fs::create_dir_all(&mpl_dir)?;

    let mut env = HashMap::new();
    env.insert("PATH".to_string(), std::env::var("PATH").unwrap_or_default());
    env.insert("HOME".to_string(), workspace.display().to_string());
    env.insert("PYTHONHASHSEED".to_string(), "random".to_string());
    env.insert("PYTHONUNBUFFERED".to_string(), "1".to_string());
    // Prevent user site-packages from altering the interpreter profile.
    env.insert("PYTHONNOUSERSITE".to_string(), "1".to_string());
    env.insert("MPLCONFIGDIR".to_string(), mpl_dir.display().to_string());
    env.insert(
        "TMPDIR".to_string(),
        std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string()),
    );
    env.insert(
        "LANG".to_string(),
        std::env::var("LANG").unwrap_or_else(|_| "C.UTF-8".to_string()),
    );
    if let Ok(tz) = std::env::var("TZ") {
        env.insert("TZ".to_string(), tz);
    }
    // Sandboxed snippets are short, one-off computations — BLAS/OMP thread
    // pools buy nothing here and only eat into the RLIMIT_NPROC headroom
    // (each pthread_create counts against it), so pin them to 1.
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"] {
        env.insert(key.to_string(), "1".to_string());
    }
    Ok(env)
}
