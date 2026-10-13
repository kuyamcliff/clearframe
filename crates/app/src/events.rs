//! Events pushed from services to the UI.

use serde::{Deserialize, Serialize};

use clearframe_core::jobs::JobRecord;
use clearframe_models::manager::InstallProgress;
use clearframe_pipeline::runner::Progress;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppEvent {
    JobUpdated {
        job: Box<JobRecord>,
    },
    JobProgress {
        id: String,
        progress: Progress,
    },
    JobLog {
        id: String,
        level: String,
        message: String,
    },
    JobRemoved {
        id: String,
    },
    QueueState {
        paused: bool,
        running: usize,
        queued: usize,
    },
    ModelProgress {
        progress: InstallProgress,
    },
    ModelChanged {
        id: String,
    },
    ModelError {
        id: String,
        message: String,
    },
    PreviewReady {
        key: String,
        path: String,
    },
    PreviewFailed {
        key: String,
        message: String,
    },
    JobResources {
        id: String,
        sample: clearframe_pipeline::runner::ResourceSample,
    },
    StorageProgress {
        kind: String,
        bytes_done: u64,
        bytes_total: u64,
    },
    StorageDone {
        kind: String,
        ok: bool,
        message: String,
    },
    Notice {
        level: String,
        message: String,
    },
    UpdateProgress {
        done: u64,
        total: u64,
    },
    /// FFmpeg detection finished; reload `app_info`.
    ToolsChanged,
}

pub type EventSink = std::sync::Arc<dyn Fn(AppEvent) + Send + Sync>;
