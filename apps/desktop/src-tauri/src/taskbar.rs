//! Export progress on the taskbar button (Windows, and launchers that support it): a progress
//! bar while jobs run, red when one fails, gone when the queue is idle. The button flashes when
//! an export finishes while the window is in the background.

use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tauri::window::{ProgressBarState, ProgressBarStatus};
use tauri::{AppHandle, Manager};

use clearframe_app::AppEvent;
use clearframe_core::jobs::JobState;

#[derive(Default)]
pub struct Taskbar {
    state: Mutex<TaskbarState>,
}

#[derive(Default)]
struct TaskbarState {
    shown: Option<(u8, u64)>,
    last_update: Option<Instant>,
    failed_at: Option<Instant>,
}

/// Bar status as a small number so repeated identical updates can be skipped.
const NONE: u8 = 0;
const NORMAL: u8 = 1;
const PAUSED: u8 = 2;
const ERROR: u8 = 3;

impl Taskbar {
    pub fn on_event(&self, handle: &AppHandle, e: &AppEvent) {
        let Some(win) = handle.get_webview_window("main") else { return };
        let mut st = self.state.lock();
        let set = |st: &mut TaskbarState, status: u8, progress: u64| {
            if st.shown == Some((status, progress)) {
                return;
            }
            st.shown = Some((status, progress));
            st.last_update = Some(Instant::now());
            let s = match status {
                NORMAL => ProgressBarStatus::Normal,
                PAUSED => ProgressBarStatus::Paused,
                ERROR => ProgressBarStatus::Error,
                _ => ProgressBarStatus::None,
            };
            let _ = win.set_progress_bar(ProgressBarState { status: Some(s), progress: (status != NONE).then_some(progress) });
        };
        match e {
            AppEvent::JobProgress { progress, .. } => {
                let pct = (progress.fraction.clamp(0.0, 1.0) * 100.0).round() as u64;
                // At most a few updates a second; the bar only has whole percents anyway.
                let recent = st.last_update.is_some_and(|t| t.elapsed() < Duration::from_millis(250));
                if !recent || st.shown.map(|(s, _)| s) != Some(NORMAL) {
                    st.failed_at = None;
                    set(&mut st, NORMAL, pct.max(1));
                }
            }
            AppEvent::JobUpdated { job } => match job.state {
                JobState::Failed => {
                    st.failed_at = Some(Instant::now());
                    set(&mut st, ERROR, 100);
                    attention(&win);
                }
                JobState::Completed => attention(&win),
                JobState::Paused | JobState::WaitingForResources => {
                    let pct = (job.progress.clamp(0.0, 1.0) * 100.0).round() as u64;
                    set(&mut st, PAUSED, pct.max(1));
                }
                _ => {}
            },
            AppEvent::QueueState { running: 0, queued, paused } => {
                // Keep a failure visible for a while; otherwise clear the bar when idle.
                let failure_showing = st.failed_at.is_some_and(|t| t.elapsed() < Duration::from_secs(15));
                if *paused && *queued > 0 {
                    set(&mut st, PAUSED, 1);
                } else if !failure_showing {
                    set(&mut st, NONE, 0);
                }
            }
            _ => {}
        }
    }
}

/// Flash the taskbar button if the user is elsewhere.
fn attention(win: &tauri::WebviewWindow) {
    if !win.is_focused().unwrap_or(true) {
        let _ = win.request_user_attention(Some(tauri::UserAttentionType::Informational));
    }
}
