//! agx-trace — Arrow-shaped agent trace store.
//!
//! Each AgentRuntime.run emits one row to an Arrow record batch:
//!   (run_id, agent_name, prompt, messages_history, tool_calls,
//!    output, latency_per_step, token_usage_per_step, timestamp)
//!
//! Configurable sinks: in-memory polars.DataFrame, parquet append,
//! DuckDB write. Helper API: agx.traces.scan(parquet_path) ->
//! polars.LazyFrame (zero-copy via Arrow IPC).
//!
//! Status: scaffolded, not yet implemented. Week 7-8 of agx_v1_plan.md.
//! The cardinality-explosion problem named by Cheng et al. AgentOps
//! (arXiv:2411.05285, 2024) is the literature charter for this crate.

use pyo3::prelude::*;

/// Register agx-trace's pyclasses into a parent Python module.
/// Currently a no-op until the Arrow emitter lands in week 7-8.
pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
