//! Native session boundaries for f3dx.
//!
//! The journal is deliberately smaller than a general event store: each
//! record is a length-delimited JSON payload with a BLAKE3 hash chain. A
//! partially written final frame is repaired on open; a complete frame with
//! an invalid hash or JSON payload is reported as corruption.

use blake3::Hasher;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use thiserror::Error;

const MAX_FRAME_BYTES: u32 = 16 * 1024 * 1024;
const HASH_BYTES: usize = 32;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("journal I/O: {0}")]
    Io(#[from] io::Error),
    #[error("journal JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("journal corruption at byte {offset}: {reason}")]
    Corrupt { offset: u64, reason: String },
    #[error("session id must not be empty")]
    EmptySessionId,
    #[error("effect id must not be empty")]
    EmptyEffectId,
    #[error("journal mutex poisoned")]
    JournalPoisoned,
    #[error("process leases require Windows Job Objects")]
    UnsupportedProcessLease,
    #[error("Windows process lease failed: {0}")]
    ProcessLease(String),
}

pub type Result<T> = std::result::Result<T, SessionError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionRef {
    pub session_id: String,
    pub sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionEvent {
    pub sequence: u64,
    pub session_id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refs: Vec<SessionRef>,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationReport {
    pub duplicate_effect_ids: Vec<String>,
    pub cross_session_refs: Vec<SessionRef>,
    pub dangling_refs: Vec<SessionRef>,
}

impl ValidationReport {
    pub fn is_clean(&self) -> bool {
        self.duplicate_effect_ids.is_empty()
            && self.cross_session_refs.is_empty()
            && self.dangling_refs.is_empty()
    }
}

#[derive(Debug)]
struct JournalState {
    file: File,
    records: Vec<SessionEvent>,
    previous_hash: [u8; HASH_BYTES],
}

/// Append-only, hash-chained session journal.
///
/// A journal path has one writer at a time. Cross-process coordination and
/// durable effect execution remain caller-owned boundaries.
pub struct Journal {
    path: PathBuf,
    session_id: String,
    state: Mutex<JournalState>,
}

impl Journal {
    /// Open or create a journal using `default` as the writer session id.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_for_session(path, "default")
    }

    /// Open or create a journal. Existing records may belong to other
    /// sessions; newly appended records use `session_id`.
    pub fn open_for_session(path: impl AsRef<Path>, session_id: impl Into<String>) -> Result<Self> {
        let session_id = session_id.into();
        if session_id.is_empty() {
            return Err(SessionError::EmptySessionId);
        }

        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        let (records, previous_hash) = scan_and_repair(&mut file)?;
        file.seek(SeekFrom::End(0))?;

        Ok(Self {
            path,
            session_id,
            state: Mutex::new(JournalState {
                file,
                records,
                previous_hash,
            }),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Append one event and synchronise it before returning its sequence.
    pub fn append(
        &self,
        kind: impl Into<String>,
        effect_id: Option<String>,
        refs: Vec<SessionRef>,
        payload: serde_json::Value,
    ) -> Result<u64> {
        let effect_id = effect_id
            .map(|id| {
                if id.is_empty() {
                    Err(SessionError::EmptyEffectId)
                } else {
                    Ok(id)
                }
            })
            .transpose()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| SessionError::JournalPoisoned)?;
        let event = SessionEvent {
            sequence: state.records.len() as u64,
            session_id: self.session_id.clone(),
            kind: kind.into(),
            effect_id,
            refs,
            payload,
        };
        let payload_bytes = serde_json::to_vec(&event)?;
        if payload_bytes.len() > MAX_FRAME_BYTES as usize {
            return Err(SessionError::Corrupt {
                offset: state.file.stream_position()?,
                reason: "event exceeds 16 MiB frame limit".into(),
            });
        }
        let hash = chained_hash(&state.previous_hash, &payload_bytes);
        state
            .file
            .write_all(&(payload_bytes.len() as u32).to_le_bytes())?;
        state.file.write_all(&payload_bytes)?;
        state.file.write_all(&hash)?;
        state.file.sync_data()?;
        state.file.seek(SeekFrom::End(0))?;
        state.previous_hash = hash;
        state.records.push(event);
        Ok(state.records.len() as u64 - 1)
    }

    pub fn records(&self) -> Result<Vec<SessionEvent>> {
        let state = self
            .state
            .lock()
            .map_err(|_| SessionError::JournalPoisoned)?;
        Ok(state.records.clone())
    }

    pub fn validate(&self) -> Result<ValidationReport> {
        Ok(validate_records(&self.records()?))
    }
}

#[pyclass(name = "SessionJournal")]
pub struct PySessionJournal {
    inner: Journal,
}

#[pymethods]
impl PySessionJournal {
    #[new]
    #[pyo3(signature = (path, session_id = String::from("python")))]
    fn new(path: String, session_id: String) -> PyResult<Self> {
        Journal::open_for_session(path, session_id)
            .map(|inner| Self { inner })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn records_json(&self) -> PyResult<String> {
        let records = self
            .inner
            .records()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        serde_json::to_string(&records).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn validate_json(&self) -> PyResult<String> {
        serde_json::to_string(
            &self
                .inner
                .validate()
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
        )
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn append_json(
        &self,
        kind: String,
        effect_id: Option<String>,
        payload_json: String,
    ) -> PyResult<u64> {
        let payload = serde_json::from_str(&payload_json)
            .map_err(|error| PyValueError::new_err(format!("payload JSON: {error}")))?;
        self.inner
            .append(kind, effect_id, Vec::new(), payload)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}

fn chained_hash(previous_hash: &[u8; HASH_BYTES], payload: &[u8]) -> [u8; HASH_BYTES] {
    let mut hasher = Hasher::new();
    hasher.update(previous_hash);
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

fn scan_and_repair(file: &mut File) -> Result<(Vec<SessionEvent>, [u8; HASH_BYTES])> {
    file.seek(SeekFrom::Start(0))?;
    let mut records = Vec::new();
    let mut previous_hash = [0; HASH_BYTES];
    let mut last_good_offset = 0;

    loop {
        let frame_offset = file.stream_position()?;
        let mut header = [0; 4];
        match file.read_exact(&mut header) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                file.set_len(last_good_offset)?;
                break;
            }
            Err(error) => return Err(error.into()),
        }
        let payload_len = u32::from_le_bytes(header);
        if payload_len > MAX_FRAME_BYTES {
            return Err(SessionError::Corrupt {
                offset: frame_offset,
                reason: format!("frame length {payload_len} exceeds limit"),
            });
        }
        let mut payload = vec![0; payload_len as usize];
        if let Err(error) = file.read_exact(&mut payload) {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                file.set_len(last_good_offset)?;
                break;
            }
            return Err(error.into());
        }
        let mut stored_hash = [0; HASH_BYTES];
        if let Err(error) = file.read_exact(&mut stored_hash) {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                file.set_len(last_good_offset)?;
                break;
            }
            return Err(error.into());
        }
        let expected_hash = chained_hash(&previous_hash, &payload);
        if stored_hash != expected_hash {
            return Err(SessionError::Corrupt {
                offset: frame_offset,
                reason: "hash chain mismatch".into(),
            });
        }
        let event: SessionEvent =
            serde_json::from_slice(&payload).map_err(|error| SessionError::Corrupt {
                offset: frame_offset,
                reason: format!("invalid event JSON: {error}"),
            })?;
        let expected_sequence = records.len() as u64;
        if event.sequence != expected_sequence {
            return Err(SessionError::Corrupt {
                offset: frame_offset,
                reason: format!(
                    "sequence {} is not the expected {expected_sequence}",
                    event.sequence
                ),
            });
        }
        records.push(event);
        previous_hash = stored_hash;
        last_good_offset = file.stream_position()?;
    }

    Ok((records, previous_hash))
}

pub fn validate_records(records: &[SessionEvent]) -> ValidationReport {
    use std::collections::{HashMap, HashSet};

    let mut effect_counts = HashMap::<&str, usize>::new();
    let mut known_refs = HashSet::new();
    for event in records {
        if let Some(effect_id) = event.effect_id.as_deref() {
            *effect_counts.entry(effect_id).or_default() += 1;
        }
        known_refs.insert((event.session_id.as_str(), event.sequence));
    }

    let mut duplicate_effect_ids = effect_counts
        .into_iter()
        .filter_map(|(effect_id, count)| (count > 1).then_some(effect_id.to_owned()))
        .collect();
    duplicate_effect_ids.sort();
    let mut report = ValidationReport {
        duplicate_effect_ids,
        ..ValidationReport::default()
    };

    for event in records {
        for reference in &event.refs {
            if reference.session_id != event.session_id {
                report.cross_session_refs.push(reference.clone());
            }
            if !known_refs.contains(&(reference.session_id.as_str(), reference.sequence)) {
                report.dangling_refs.push(reference.clone());
            }
        }
    }
    report
}

/// A Windows Job Object lease. Closing the lease can reap every descendant
/// of the attached process when `kill_on_drop` is enabled.
#[derive(Debug)]
pub struct ProcessLease {
    #[cfg(windows)]
    job: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg_attr(windows, allow(unsafe_code))]
impl ProcessLease {
    /// Attach a raw process handle. The caller retains ownership of the
    /// process handle; the lease owns only the Job Object handle.
    pub fn attach(process_handle: usize, kill_on_drop: bool) -> Result<Self> {
        #[cfg(windows)]
        {
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
                SetInformationJobObject,
            };
            let process_handle = process_handle as windows_sys::Win32::Foundation::HANDLE;
            if process_handle.is_null() {
                return Err(SessionError::ProcessLease("null process handle".into()));
            }
            let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if job.is_null() {
                return Err(SessionError::ProcessLease(
                    io::Error::last_os_error().to_string(),
                ));
            }
            if kill_on_drop {
                let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = unsafe {
                    SetInformationJobObject(
                        job,
                        JobObjectExtendedLimitInformation,
                        (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
                            .cast::<std::ffi::c_void>(),
                        std::mem::size_of_val(&info) as u32,
                    )
                };
                if ok == 0 {
                    let _ = unsafe { CloseHandle(job) };
                    return Err(SessionError::ProcessLease(
                        io::Error::last_os_error().to_string(),
                    ));
                }
            }
            let assigned = unsafe { AssignProcessToJobObject(job, process_handle) };
            if assigned == 0 {
                let _ = unsafe { CloseHandle(job) };
                return Err(SessionError::ProcessLease(
                    io::Error::last_os_error().to_string(),
                ));
            }
            Ok(Self { job })
        }
        #[cfg(not(windows))]
        {
            let _ = (process_handle, kill_on_drop);
            Err(SessionError::UnsupportedProcessLease)
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySessionJournal>()?;
    Ok(())
}

#[cfg(windows)]
#[allow(unsafe_code)]
impl Drop for ProcessLease {
    fn drop(&mut self) {
        unsafe {
            let _ = windows_sys::Win32::Foundation::CloseHandle(self.job);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "f3dx-session-{label}-{}-{nanos}.journal",
            std::process::id()
        ))
    }

    #[test]
    fn append_reopen_and_validate() {
        let path = temp_path("reopen");
        let journal = Journal::open_for_session(&path, "run-a").unwrap();
        assert_eq!(
            journal
                .append("start", None, Vec::new(), json!({"ok": true}))
                .unwrap(),
            0
        );
        assert_eq!(
            journal
                .append("effect", Some("tool-1".into()), Vec::new(), json!({"n": 1}))
                .unwrap(),
            1
        );
        drop(journal);

        let reopened = Journal::open_for_session(&path, "run-b").unwrap();
        assert_eq!(reopened.records().unwrap().len(), 2);
        assert!(reopened.validate().unwrap().is_clean());
        assert_eq!(
            reopened
                .append("done", None, Vec::new(), json!({}))
                .unwrap(),
            2
        );
        drop(reopened);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn partial_tail_is_repaired() {
        let path = temp_path("tail");
        let journal = Journal::open_for_session(&path, "run").unwrap();
        journal
            .append("start", None, Vec::new(), json!({}))
            .unwrap();
        drop(journal);
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[8, 0, 0]).unwrap();
        drop(file);

        let reopened = Journal::open_for_session(&path, "run").unwrap();
        assert_eq!(reopened.records().unwrap().len(), 1);
        assert_eq!(
            reopened
                .append("done", None, Vec::new(), json!({}))
                .unwrap(),
            1
        );
        drop(reopened);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn validator_reports_duplicate_and_cross_session_refs() {
        let records = vec![
            SessionEvent {
                sequence: 0,
                session_id: "a".into(),
                kind: "effect".into(),
                effect_id: Some("same".into()),
                refs: vec![SessionRef {
                    session_id: "b".into(),
                    sequence: 4,
                }],
                payload: json!({}),
            },
            SessionEvent {
                sequence: 1,
                session_id: "a".into(),
                kind: "effect".into(),
                effect_id: Some("same".into()),
                refs: Vec::new(),
                payload: json!({}),
            },
        ];
        let report = validate_records(&records);
        assert_eq!(report.duplicate_effect_ids, vec!["same"]);
        assert_eq!(report.cross_session_refs.len(), 1);
        assert_eq!(report.dangling_refs.len(), 1);
    }

    #[cfg(not(windows))]
    #[test]
    fn process_lease_is_explicitly_unsupported_off_windows() {
        assert!(matches!(
            ProcessLease::attach(1, true),
            Err(SessionError::UnsupportedProcessLease)
        ));
    }
}
