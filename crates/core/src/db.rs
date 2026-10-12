//! SQLite persistence: settings, recent files, jobs, presets, projects index,
//! model registry and performance calibration.

use std::path::Path;

use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};

use crate::jobs::{JobKind, JobRecord, JobState};
use crate::settings::Settings;

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("database error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("data error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid job state transition {from} -> {to}")]
    Transition { from: String, to: String },
    #[error("not found: {0}")]
    NotFound(String),
    #[error("the database was created by a newer version of ClearFrame (schema {0})")]
    TooNew(i64),
}

type Result<T> = std::result::Result<T, DbError>;

const MIGRATIONS: &[&str] = &[
    // v1
    r#"
    CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS recent_files (
        path TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        opened_at TEXT NOT NULL,
        meta_json TEXT
    );
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        state TEXT NOT NULL,
        source TEXT NOT NULL,
        destination TEXT NOT NULL,
        spec_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        progress REAL NOT NULL DEFAULT 0,
        progress_json TEXT,
        detail TEXT,
        error TEXT,
        report_json TEXT,
        queue_order INTEGER NOT NULL DEFAULT 0,
        attempts INTEGER NOT NULL DEFAULT 0,
        batch_id TEXT,
        heartbeat TEXT,
        warnings_json TEXT NOT NULL DEFAULT '[]'
    );
    CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state);
    CREATE TABLE IF NOT EXISTS job_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        at TEXT NOT NULL,
        level TEXT NOT NULL,
        message TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS job_events_job ON job_events(job_id);
    CREATE TABLE IF NOT EXISTS presets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT '',
        favorite INTEGER NOT NULL DEFAULT 0,
        builtin INTEGER NOT NULL DEFAULT 0,
        data_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS projects (
        path TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        source TEXT,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS models (
        id TEXT PRIMARY KEY,
        version TEXT NOT NULL,
        state TEXT NOT NULL,
        record_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS calibration (
        key TEXT PRIMARY KEY,
        megapixels_per_second REAL NOT NULL,
        samples INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    );
    "#,
];

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentFile {
    pub path: String,
    pub kind: String,
    pub title: String,
    pub opened_at: String,
    pub meta_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredPreset {
    pub id: String,
    pub name: String,
    pub category: String,
    pub favorite: bool,
    pub builtin: bool,
    pub data_json: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectEntry {
    pub path: String,
    pub name: String,
    pub source: Option<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobEvent {
    pub at: String,
    pub level: String,
    pub message: String,
}

pub struct Db {
    conn: Mutex<Connection>,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(p) = path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if version > MIGRATIONS.len() as i64 {
            return Err(DbError::TooNew(version));
        }
        for (i, m) in MIGRATIONS.iter().enumerate().skip(version as usize) {
            let tx = conn.unchecked_transaction()?;
            tx.execute_batch(m)?;
            tx.pragma_update(None, "user_version", (i + 1) as i64)?;
            tx.commit()?;
        }
        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Run a closure with the raw connection (for crate-level repositories).
    pub fn with<T>(&self, f: impl FnOnce(&Connection) -> rusqlite::Result<T>) -> Result<T> {
        let c = self.conn.lock();
        Ok(f(&c)?)
    }

    // ---- key/value & settings ----

    pub fn kv_get(&self, key: &str) -> Result<Option<String>> {
        self.with(|c| c.query_row("SELECT value FROM kv WHERE key=?1", [key], |r| r.get(0)).optional())
    }

    pub fn kv_set(&self, key: &str, value: &str) -> Result<()> {
        self.with(|c| c.execute("INSERT INTO kv(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", params![key, value]))?;
        Ok(())
    }

    pub fn load_settings(&self) -> Result<Settings> {
        match self.kv_get("settings")? {
            Some(s) => Ok(Settings::from_json(&s).unwrap_or_else(|e| {
                log::warn!("settings could not be read ({e}); using defaults");
                Settings::default()
            })),
            None => Ok(Settings::default()),
        }
    }

    pub fn save_settings(&self, s: &Settings) -> Result<()> {
        let mut s = s.clone();
        s.sanitize();
        self.kv_set("settings", &serde_json::to_string(&s)?)
    }

    // ---- recent files ----

    pub fn add_recent(&self, path: &str, kind: &str, title: &str, meta_json: Option<&str>) -> Result<()> {
        self.with(|c| {
            c.execute(
                "INSERT INTO recent_files(path,kind,title,opened_at,meta_json) VALUES(?1,?2,?3,?4,?5)
                 ON CONFLICT(path) DO UPDATE SET kind=excluded.kind,title=excluded.title,opened_at=excluded.opened_at,meta_json=excluded.meta_json",
                params![path, kind, title, now(), meta_json],
            )?;
            // Keep the list bounded.
            c.execute("DELETE FROM recent_files WHERE path NOT IN (SELECT path FROM recent_files ORDER BY opened_at DESC LIMIT 50)", [])
        })?;
        Ok(())
    }

    pub fn recent(&self, limit: usize) -> Result<Vec<RecentFile>> {
        self.with(|c| {
            let mut st = c.prepare("SELECT path,kind,title,opened_at,meta_json FROM recent_files ORDER BY opened_at DESC LIMIT ?1")?;
            let rows = st.query_map([limit as i64], |r| Ok(RecentFile { path: r.get(0)?, kind: r.get(1)?, title: r.get(2)?, opened_at: r.get(3)?, meta_json: r.get(4)? }))?;
            rows.collect()
        })
    }

    pub fn remove_recent(&self, path: &str) -> Result<()> {
        self.with(|c| c.execute("DELETE FROM recent_files WHERE path=?1", [path]))?;
        Ok(())
    }

    pub fn clear_recent(&self) -> Result<()> {
        self.with(|c| c.execute("DELETE FROM recent_files", []))?;
        Ok(())
    }

    // ---- jobs ----

    fn job_from_row(r: &Row) -> rusqlite::Result<JobRecord> {
        let kind: String = r.get("kind")?;
        let state: String = r.get("state")?;
        let warnings: String = r.get("warnings_json")?;
        Ok(JobRecord {
            id: r.get("id")?,
            kind: if kind == "preview" { JobKind::Preview } else { JobKind::Render },
            name: r.get("name")?,
            state: JobState::parse(&state).unwrap_or(JobState::Failed),
            source: r.get("source")?,
            destination: r.get("destination")?,
            spec_json: r.get("spec_json")?,
            created_at: r.get("created_at")?,
            updated_at: r.get("updated_at")?,
            started_at: r.get("started_at")?,
            finished_at: r.get("finished_at")?,
            progress: r.get("progress")?,
            progress_json: r.get("progress_json")?,
            detail: r.get("detail")?,
            error: r.get("error")?,
            report_json: r.get("report_json")?,
            queue_order: r.get("queue_order")?,
            attempts: r.get("attempts")?,
            batch_id: r.get("batch_id")?,
            heartbeat: r.get("heartbeat")?,
            warnings: serde_json::from_str(&warnings).unwrap_or_default(),
        })
    }

    pub fn insert_job(&self, j: &JobRecord) -> Result<()> {
        self.with(|c| {
            let order: i64 = c.query_row("SELECT COALESCE(MAX(queue_order),0)+1 FROM jobs", [], |r| r.get(0))?;
            c.execute(
                "INSERT INTO jobs(id,kind,name,state,source,destination,spec_json,created_at,updated_at,progress,queue_order,attempts,batch_id,warnings_json)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?8,0,?9,0,?10,'[]')",
                params![
                    j.id,
                    if j.kind == JobKind::Preview { "preview" } else { "render" },
                    j.name,
                    j.state.as_str(),
                    j.source,
                    j.destination,
                    j.spec_json,
                    j.created_at,
                    order,
                    j.batch_id
                ],
            )
        })?;
        Ok(())
    }

    pub fn job(&self, id: &str) -> Result<JobRecord> {
        self.with(|c| c.query_row("SELECT * FROM jobs WHERE id=?1", [id], Self::job_from_row).optional())?.ok_or_else(|| DbError::NotFound(format!("job {id}")))
    }

    pub fn jobs(&self, include_previews: bool) -> Result<Vec<JobRecord>> {
        self.with(|c| {
            let sql = if include_previews { "SELECT * FROM jobs ORDER BY queue_order" } else { "SELECT * FROM jobs WHERE kind='render' ORDER BY queue_order" };
            let mut st = c.prepare(sql)?;
            let rows = st.query_map([], Self::job_from_row)?;
            rows.collect()
        })
    }

    /// Validated state change. Returns the previous state.
    pub fn set_job_state(&self, id: &str, next: JobState, detail: Option<&str>) -> Result<JobState> {
        let cur = self.job(id)?.state;
        if !cur.can_transition_to(next) {
            return Err(DbError::Transition { from: cur.as_str().into(), to: next.as_str().into() });
        }
        let ts = now();
        self.with(|c| {
            c.execute("UPDATE jobs SET state=?2, updated_at=?3, detail=COALESCE(?4, detail) WHERE id=?1", params![id, next.as_str(), ts, detail])?;
            if next == JobState::Preparing && cur != JobState::Processing {
                c.execute("UPDATE jobs SET started_at=COALESCE(started_at, ?2), attempts=attempts+1, error=NULL WHERE id=?1", params![id, ts])?;
            }
            if next.is_terminal() {
                c.execute("UPDATE jobs SET finished_at=?2 WHERE id=?1", params![id, ts])?;
            }
            if next == JobState::Queued {
                c.execute("UPDATE jobs SET finished_at=NULL, error=NULL WHERE id=?1", params![id])?;
            }
            Ok(())
        })?;
        Ok(cur)
    }

    pub fn set_job_progress(&self, id: &str, fraction: f64, json: &str) -> Result<()> {
        self.with(|c| c.execute("UPDATE jobs SET progress=?2, progress_json=?3, heartbeat=?4, updated_at=?4 WHERE id=?1", params![id, fraction, json, now()]))?;
        Ok(())
    }

    pub fn set_job_error(&self, id: &str, error: &str) -> Result<()> {
        self.with(|c| c.execute("UPDATE jobs SET error=?2, updated_at=?3 WHERE id=?1", params![id, error, now()]))?;
        Ok(())
    }

    pub fn set_job_report(&self, id: &str, report_json: &str, destination: &str) -> Result<()> {
        self.with(|c| c.execute("UPDATE jobs SET report_json=?2, destination=?3, progress=1, updated_at=?4 WHERE id=?1", params![id, report_json, destination, now()]))?;
        Ok(())
    }

    pub fn set_job_spec(&self, id: &str, spec_json: &str, destination: &str) -> Result<()> {
        self.with(|c| c.execute("UPDATE jobs SET spec_json=?2, destination=?3, updated_at=?4 WHERE id=?1", params![id, spec_json, destination, now()]))?;
        Ok(())
    }

    pub fn add_job_warning(&self, id: &str, w: &str) -> Result<()> {
        let mut j = self.job(id)?;
        if !j.warnings.iter().any(|x| x == w) {
            j.warnings.push(w.to_string());
        }
        let s = serde_json::to_string(&j.warnings)?;
        self.with(|c| c.execute("UPDATE jobs SET warnings_json=?2 WHERE id=?1", params![id, s]))?;
        Ok(())
    }

    pub fn log_job_event(&self, id: &str, level: &str, message: &str) -> Result<()> {
        let msg: String = message.chars().take(4000).collect();
        self.with(|c| {
            c.execute("INSERT INTO job_events(job_id,at,level,message) VALUES(?1,?2,?3,?4)", params![id, now(), level, msg])?;
            // Bound per-job logs.
            c.execute("DELETE FROM job_events WHERE job_id=?1 AND id NOT IN (SELECT id FROM job_events WHERE job_id=?1 ORDER BY id DESC LIMIT 500)", [id])
        })?;
        Ok(())
    }

    pub fn job_events(&self, id: &str) -> Result<Vec<JobEvent>> {
        self.with(|c| {
            let mut st = c.prepare("SELECT at,level,message FROM job_events WHERE job_id=?1 ORDER BY id")?;
            let rows = st.query_map([id], |r| Ok(JobEvent { at: r.get(0)?, level: r.get(1)?, message: r.get(2)? }))?;
            rows.collect()
        })
    }

    pub fn delete_job(&self, id: &str) -> Result<()> {
        self.with(|c| {
            c.execute("DELETE FROM job_events WHERE job_id=?1", [id])?;
            c.execute("DELETE FROM jobs WHERE id=?1", [id])
        })?;
        Ok(())
    }

    /// Move a pending job to a new position in the queue.
    pub fn reorder_job(&self, id: &str, new_index: usize) -> Result<()> {
        let mut ids: Vec<String> = self.jobs(false)?.into_iter().map(|j| j.id).collect();
        let Some(pos) = ids.iter().position(|x| x == id) else { return Err(DbError::NotFound(id.into())) };
        let item = ids.remove(pos);
        ids.insert(new_index.min(ids.len()), item);
        self.with(|c| {
            for (i, jid) in ids.iter().enumerate() {
                c.execute("UPDATE jobs SET queue_order=?2 WHERE id=?1", params![jid, i as i64 + 1])?;
            }
            Ok(())
        })
    }

    /// Mark jobs that were running when the app last exited as recoverable.
    pub fn recover_interrupted(&self) -> Result<Vec<String>> {
        let ids: Vec<String> = self.jobs(true)?.into_iter().filter(|j| j.state.is_active()).map(|j| j.id).collect();
        for id in &ids {
            self.set_job_state(id, JobState::Recoverable, Some("ClearFrame closed while this job was running"))?;
        }
        Ok(ids)
    }

    pub fn clear_finished_jobs(&self) -> Result<usize> {
        let ids: Vec<String> = self.jobs(true)?.into_iter().filter(|j| j.state.is_terminal()).map(|j| j.id).collect();
        for id in &ids {
            self.delete_job(id)?;
        }
        Ok(ids.len())
    }

    // ---- presets ----

    pub fn upsert_preset(&self, p: &StoredPreset) -> Result<()> {
        self.with(|c| {
            c.execute(
                "INSERT INTO presets(id,name,category,favorite,builtin,data_json,created_at,updated_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)
                 ON CONFLICT(id) DO UPDATE SET name=excluded.name,category=excluded.category,favorite=excluded.favorite,data_json=excluded.data_json,updated_at=excluded.updated_at",
                params![p.id, p.name, p.category, p.favorite as i64, p.builtin as i64, p.data_json, p.created_at, p.updated_at],
            )
        })?;
        Ok(())
    }

    pub fn presets(&self) -> Result<Vec<StoredPreset>> {
        self.with(|c| {
            let mut st =
                c.prepare("SELECT id,name,category,favorite,builtin,data_json,created_at,updated_at FROM presets ORDER BY builtin DESC, favorite DESC, name COLLATE NOCASE")?;
            let rows = st.query_map([], |r| {
                Ok(StoredPreset {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    category: r.get(2)?,
                    favorite: r.get::<_, i64>(3)? != 0,
                    builtin: r.get::<_, i64>(4)? != 0,
                    data_json: r.get(5)?,
                    created_at: r.get(6)?,
                    updated_at: r.get(7)?,
                })
            })?;
            rows.collect()
        })
    }

    pub fn preset(&self, id: &str) -> Result<StoredPreset> {
        self.presets()?.into_iter().find(|p| p.id == id).ok_or_else(|| DbError::NotFound(format!("preset {id}")))
    }

    pub fn delete_preset(&self, id: &str) -> Result<()> {
        self.with(|c| c.execute("DELETE FROM presets WHERE id=?1 AND builtin=0", [id]))?;
        Ok(())
    }

    pub fn set_preset_favorite(&self, id: &str, fav: bool) -> Result<()> {
        self.with(|c| c.execute("UPDATE presets SET favorite=?2 WHERE id=?1", params![id, fav as i64]))?;
        Ok(())
    }

    // ---- projects index ----

    pub fn touch_project(&self, path: &str, name: &str, source: Option<&str>) -> Result<()> {
        self.with(|c| {
            c.execute(
                "INSERT INTO projects(path,name,source,updated_at) VALUES(?1,?2,?3,?4) ON CONFLICT(path) DO UPDATE SET name=excluded.name,source=excluded.source,updated_at=excluded.updated_at",
                params![path, name, source, now()],
            )
        })?;
        Ok(())
    }

    pub fn projects(&self) -> Result<Vec<ProjectEntry>> {
        self.with(|c| {
            let mut st = c.prepare("SELECT path,name,source,updated_at FROM projects ORDER BY updated_at DESC LIMIT 100")?;
            let rows = st.query_map([], |r| Ok(ProjectEntry { path: r.get(0)?, name: r.get(1)?, source: r.get(2)?, updated_at: r.get(3)? }))?;
            rows.collect()
        })
    }

    pub fn remove_project(&self, path: &str) -> Result<()> {
        self.with(|c| c.execute("DELETE FROM projects WHERE path=?1", [path]))?;
        Ok(())
    }

    // ---- model registry (records are owned by the model manager) ----

    pub fn model_put(&self, id: &str, version: &str, state: &str, record_json: &str) -> Result<()> {
        self.with(|c| {
            c.execute(
                "INSERT INTO models(id,version,state,record_json,updated_at) VALUES(?1,?2,?3,?4,?5)
                 ON CONFLICT(id) DO UPDATE SET version=excluded.version,state=excluded.state,record_json=excluded.record_json,updated_at=excluded.updated_at",
                params![id, version, state, record_json, now()],
            )
        })?;
        Ok(())
    }

    pub fn model_get(&self, id: &str) -> Result<Option<String>> {
        self.with(|c| c.query_row("SELECT record_json FROM models WHERE id=?1", [id], |r| r.get(0)).optional())
    }

    pub fn model_all(&self) -> Result<Vec<String>> {
        self.with(|c| {
            let mut st = c.prepare("SELECT record_json FROM models ORDER BY id")?;
            let rows = st.query_map([], |r| r.get(0))?;
            rows.collect()
        })
    }

    pub fn model_delete(&self, id: &str) -> Result<()> {
        self.with(|c| c.execute("DELETE FROM models WHERE id=?1", [id]))?;
        Ok(())
    }

    // ---- calibration (measured throughput, local only) ----

    /// Record a throughput sample; keeps an exponential moving average.
    pub fn record_throughput(&self, key: &str, mp_per_s: f64) -> Result<()> {
        if !mp_per_s.is_finite() || mp_per_s <= 0.0 {
            return Ok(());
        }
        self.with(|c| {
            let cur: Option<(f64, i64)> = c.query_row("SELECT megapixels_per_second, samples FROM calibration WHERE key=?1", [key], |r| Ok((r.get(0)?, r.get(1)?))).optional()?;
            let (v, n) = match cur {
                Some((v, n)) => (v * 0.7 + mp_per_s * 0.3, n + 1),
                None => (mp_per_s, 1),
            };
            c.execute(
                "INSERT INTO calibration(key,megapixels_per_second,samples,updated_at) VALUES(?1,?2,?3,?4)
                 ON CONFLICT(key) DO UPDATE SET megapixels_per_second=excluded.megapixels_per_second,samples=excluded.samples,updated_at=excluded.updated_at",
                params![key, v, n, now()],
            )
        })?;
        Ok(())
    }

    pub fn throughput(&self, key: &str) -> Result<Option<f64>> {
        self.with(|c| c.query_row("SELECT megapixels_per_second FROM calibration WHERE key=?1", [key], |r| r.get(0)).optional())
    }

    pub fn clear_calibration(&self) -> Result<()> {
        self.with(|c| c.execute("DELETE FROM calibration", []))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str) -> JobRecord {
        JobRecord {
            id: id.into(),
            kind: JobKind::Render,
            name: "n".into(),
            state: JobState::Queued,
            source: "/s.mp4".into(),
            destination: "/d.mp4".into(),
            spec_json: "{}".into(),
            created_at: now(),
            updated_at: now(),
            started_at: None,
            finished_at: None,
            progress: 0.0,
            progress_json: None,
            detail: None,
            error: None,
            report_json: None,
            queue_order: 0,
            attempts: 0,
            batch_id: None,
            heartbeat: None,
            warnings: vec![],
        }
    }

    #[test]
    fn settings_roundtrip_and_persist_across_open() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("x.sqlite3");
        {
            let db = Db::open(&p).unwrap();
            let mut s = db.load_settings().unwrap();
            s.preview_seconds = 5.0;
            db.save_settings(&s).unwrap();
        }
        let db = Db::open(&p).unwrap();
        assert_eq!(db.load_settings().unwrap().preview_seconds, 5.0);
    }

    #[test]
    fn job_lifecycle_and_recovery() {
        let db = Db::open_in_memory().unwrap();
        db.insert_job(&job("a")).unwrap();
        db.insert_job(&job("b")).unwrap();
        db.set_job_state("a", JobState::Preparing, None).unwrap();
        db.set_job_state("a", JobState::Processing, Some("x")).unwrap();
        assert!(matches!(db.set_job_state("a", JobState::Completed, None), Err(DbError::Transition { .. })));
        db.set_job_progress("a", 0.5, "{}").unwrap();
        let recovered = db.recover_interrupted().unwrap();
        assert_eq!(recovered, vec!["a".to_string()]);
        assert_eq!(db.job("a").unwrap().state, JobState::Recoverable);
        assert_eq!(db.job("a").unwrap().attempts, 1);
        db.reorder_job("b", 0).unwrap();
        let order: Vec<String> = db.jobs(false).unwrap().into_iter().map(|j| j.id).collect();
        assert_eq!(order, vec!["b", "a"]);
        db.log_job_event("a", "info", "hello").unwrap();
        assert_eq!(db.job_events("a").unwrap().len(), 1);
        db.add_job_warning("a", "w").unwrap();
        db.add_job_warning("a", "w").unwrap();
        assert_eq!(db.job("a").unwrap().warnings.len(), 1);
        db.set_job_state("a", JobState::Cancelled, None).unwrap();
        assert_eq!(db.clear_finished_jobs().unwrap(), 1);
    }

    #[test]
    fn recent_and_presets() {
        let db = Db::open_in_memory().unwrap();
        db.add_recent("/a.mp4", "video", "a", None).unwrap();
        db.add_recent("/b.mp4", "video", "b", None).unwrap();
        assert_eq!(db.recent(10).unwrap().len(), 2);
        db.remove_recent("/a.mp4").unwrap();
        assert_eq!(db.recent(10).unwrap().len(), 1);
        let p =
            StoredPreset { id: "p".into(), name: "P".into(), category: "".into(), favorite: false, builtin: true, data_json: "{}".into(), created_at: now(), updated_at: now() };
        db.upsert_preset(&p).unwrap();
        db.delete_preset("p").unwrap();
        assert_eq!(db.presets().unwrap().len(), 1, "built-in presets cannot be deleted");
        db.record_throughput("k", 2.0).unwrap();
        db.record_throughput("k", 4.0).unwrap();
        assert!((db.throughput("k").unwrap().unwrap() - 2.6).abs() < 1e-9);
    }
}
