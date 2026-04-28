//! f3dx-replay - read JSONL/parquet trace bundles, replay against a
//! target config, emit diff report.
//!
//! Layered determinism modes (DiffMode):
//!   Bytes      - exact byte equality. Strictest. Used for structured-output
//!                tests where the model is supposed to emit canonical JSON.
//!   Structured - parse both as JSON, compare fields. Tolerates whitespace +
//!                key ordering. Used for tool-call extraction.
//!   Embedding  - embedding-cosine distance under a threshold. (V0.1.)
//!   Judge      - LLM-as-judge call. Most expensive. (V0.1.)
//!
//! V0 ships Bytes + Structured. Embedding + Judge land with the f3dx-cache
//! integration once the Python adapter is wired.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ReplayError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiffMode {
    Bytes,
    Structured,
    Embedding,
    Judge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRow {
    pub trace_id: Option<String>,
    pub model: Option<String>,
    pub prompt: Option<String>,
    pub system_prompt: Option<String>,
    pub output: Option<String>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    pub trace_id: Option<String>,
    pub mode: DiffMode,
    pub passed: bool,
    pub before: Option<String>,
    pub after: Option<String>,
    pub note: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DiffReport {
    pub total: u64,
    pub passed: u64,
    pub failed: u64,
    pub entries: Vec<DiffEntry>,
}

/// Read a JSONL trace file (one row per line) into a vector of TraceRow.
pub fn read_jsonl(path: impl AsRef<Path>) -> Result<Vec<TraceRow>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line)?);
    }
    Ok(out)
}

/// Compare two outputs under a DiffMode. Returns whether the comparison
/// passed and a short diagnostic note for the failure case.
pub fn diff(before: &str, after: &str, mode: DiffMode) -> (bool, Option<String>) {
    match mode {
        DiffMode::Bytes => {
            if before == after {
                (true, None)
            } else {
                (
                    false,
                    Some(format!(
                        "byte mismatch ({} vs {} bytes)",
                        before.len(),
                        after.len()
                    )),
                )
            }
        }
        DiffMode::Structured => {
            let pa: std::result::Result<serde_json::Value, _> = serde_json::from_str(before);
            let pb: std::result::Result<serde_json::Value, _> = serde_json::from_str(after);
            match (pa, pb) {
                (Ok(a), Ok(b)) => {
                    let ca = f3dx_cache::canonicalize(&a);
                    let cb = f3dx_cache::canonicalize(&b);
                    if ca == cb {
                        (true, None)
                    } else {
                        (
                            false,
                            Some("structured mismatch after canonicalization".into()),
                        )
                    }
                }
                _ => (
                    false,
                    Some("at least one side is not valid JSON; fall back to Bytes mode".into()),
                ),
            }
        }
        DiffMode::Embedding | DiffMode::Judge => (
            false,
            Some(format!(
                "{:?} mode requires the Python adapter (V0.1)",
                mode
            )),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_bytes_pass() {
        let (ok, note) = diff("abc", "abc", DiffMode::Bytes);
        assert!(ok);
        assert!(note.is_none());
    }

    #[test]
    fn diff_bytes_fail() {
        let (ok, _) = diff("abc", "abd", DiffMode::Bytes);
        assert!(!ok);
    }

    #[test]
    fn diff_structured_passes_on_key_reorder() {
        let a = r#"{"name":"alice","age":30}"#;
        let b = r#"{"age":30,"name":"alice"}"#;
        let (ok, _) = diff(a, b, DiffMode::Structured);
        assert!(ok);
    }

    #[test]
    fn diff_structured_fails_on_value_change() {
        let a = r#"{"name":"alice","age":30}"#;
        let b = r#"{"name":"alice","age":31}"#;
        let (ok, _) = diff(a, b, DiffMode::Structured);
        assert!(!ok);
    }
}
