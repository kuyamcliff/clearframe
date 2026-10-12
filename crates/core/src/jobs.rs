//! Job lifecycle state machine and persisted job records.

use serde::{Deserialize, Serialize};

/// Job lifecycle states.
///
/// `queued -> preparing -> (previewing) -> processing -> encoding -> validating -> completed`,
/// with `paused`, `cancelled`, `failed`, `waiting_for_resources` and `recoverable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    WaitingForResources,
    Preparing,
    Previewing,
    Processing,
    Encoding,
    Validating,
    Completed,
    Paused,
    Cancelled,
    Failed,
    Recoverable,
}

impl JobState {
    pub fn as_str(self) -> &'static str {
        match self {
            JobState::Queued => "queued",
            JobState::WaitingForResources => "waiting_for_resources",
            JobState::Preparing => "preparing",
            JobState::Previewing => "previewing",
            JobState::Processing => "processing",
            JobState::Encoding => "encoding",
            JobState::Validating => "validating",
            JobState::Completed => "completed",
            JobState::Paused => "paused",
            JobState::Cancelled => "cancelled",
            JobState::Failed => "failed",
            JobState::Recoverable => "recoverable",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "queued" => JobState::Queued,
            "waiting_for_resources" => JobState::WaitingForResources,
            "preparing" => JobState::Preparing,
            "previewing" => JobState::Previewing,
            "processing" => JobState::Processing,
            "encoding" => JobState::Encoding,
            "validating" => JobState::Validating,
            "completed" => JobState::Completed,
            "paused" => JobState::Paused,
            "cancelled" => JobState::Cancelled,
            "failed" => JobState::Failed,
            "recoverable" => JobState::Recoverable,
            _ => return None,
        })
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, JobState::Completed | JobState::Cancelled | JobState::Failed)
    }

    /// A worker is (or should be) running for the job.
    pub fn is_active(self) -> bool {
        matches!(self, JobState::Preparing | JobState::Previewing | JobState::Processing | JobState::Encoding | JobState::Validating)
    }

    /// Allowed transitions; anything else is rejected before persisting.
    pub fn can_transition_to(self, next: JobState) -> bool {
        use JobState::*;
        if self == next {
            return true;
        }
        // Crash recovery: any active state may become recoverable.
        if self.is_active() && next == Recoverable {
            return true;
        }
        match self {
            Queued => matches!(next, Preparing | WaitingForResources | Cancelled | Failed | Paused),
            WaitingForResources => matches!(next, Preparing | Queued | Cancelled | Failed | Paused),
            Preparing => matches!(next, Previewing | Processing | Encoding | Cancelled | Failed | Paused),
            Previewing => matches!(next, Processing | Encoding | Completed | Cancelled | Failed | Paused),
            Processing => matches!(next, Preparing | Encoding | Cancelled | Failed | Paused),
            Encoding => matches!(next, Validating | Cancelled | Failed),
            Validating => matches!(next, Completed | Failed),
            Paused | Recoverable => matches!(next, Queued | Preparing | Cancelled | Failed),
            Failed | Cancelled => matches!(next, Queued),
            Completed => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Render,
    Preview,
}

/// A persisted job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub kind: JobKind,
    pub name: String,
    pub state: JobState,
    pub source: String,
    pub destination: String,
    /// Serialised `JobSpec`.
    pub spec_json: String,
    pub created_at: String,
    pub updated_at: String,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub progress: f64,
    /// Last progress event (JSON).
    pub progress_json: Option<String>,
    pub detail: Option<String>,
    pub error: Option<String>,
    /// Serialised `JobReport` on success.
    pub report_json: Option<String>,
    pub queue_order: i64,
    pub attempts: i64,
    pub batch_id: Option<String>,
    pub heartbeat: Option<String>,
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitions() {
        use JobState::*;
        assert!(Queued.can_transition_to(Preparing));
        assert!(Processing.can_transition_to(Encoding));
        assert!(Processing.can_transition_to(Recoverable));
        assert!(!Completed.can_transition_to(Queued));
        assert!(!Queued.can_transition_to(Completed));
        assert!(Failed.can_transition_to(Queued));
        assert!(!Encoding.can_transition_to(Processing));
        for s in [Queued, WaitingForResources, Preparing, Previewing, Processing, Encoding, Validating, Completed, Paused, Cancelled, Failed, Recoverable] {
            assert_eq!(JobState::parse(s.as_str()), Some(s));
        }
    }
}
