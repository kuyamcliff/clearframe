//! Model library services.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use clearframe_core::db::Db;
use clearframe_hardware::{capability, Capability};
use clearframe_models::catalog::Catalog;
use clearframe_models::convert::KeyRules;
use clearframe_models::manager::{platform_id, CustomModelInfo, InstallControl, InstallError, ModelState, ModelStatus};
use clearframe_pipeline::plan::{ModelFacts, ModelLookup};
use clearframe_pipeline::spec::{Pipeline, ResolvedModel};

use crate::{App, AppEvent};

/// Running installs by model id.
#[derive(Default)]
pub struct Installs(Mutex<HashMap<String, InstallControl>>);

/// Use a verified remote catalog stored in the database if newer than the bundled one.
pub fn effective_catalog(db: &Db) -> Catalog {
    let bundled = Catalog::bundled();
    if let Ok(Some(s)) = db.kv_get("catalog_override") {
        if let Ok(c) = Catalog::parse(&s) {
            if c.catalog_version > bundled.catalog_version {
                return c;
            }
        }
    }
    bundled
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    #[serde(flatten)]
    pub status: ModelStatus,
    pub capability: Capability,
    pub in_progress: bool,
}

/// Planner lookup backed by the catalog and registry.
pub struct AppLookup<'a> {
    pub app: &'a App,
    pub catalog: Catalog,
}

impl ModelLookup for AppLookup<'_> {
    fn facts(&self, id: &str) -> Option<ModelFacts> {
        let installed = self.app.models.resolve(id).is_some();
        if let Some(m) = self.catalog.get(id) {
            return Some(ModelFacts {
                id: id.into(),
                display_name: m.display_name.clone(),
                scale: m.scale,
                installed,
                mb_per_megapixel: m.memory_per_megapixel_mb,
                weights_mb: m.installed_size_bytes as f64 / 1e6,
                supports_denoise_strength: m.supports_denoise_strength,
            });
        }
        let rec = self.app.models.installed(id)?;
        let info = rec.custom_info.clone()?;
        Some(ModelFacts {
            id: id.into(),
            display_name: info.display_name,
            scale: info.scale.max(1),
            installed,
            mb_per_megapixel: 1500.0,
            weights_mb: rec.installed_size as f64 / 1e6,
            supports_denoise_strength: false,
        })
    }
}

impl App {
    pub fn lookup(&self) -> AppLookup<'_> {
        AppLookup { app: self, catalog: self.catalog() }
    }

    /// Model library cards with capability levels for this machine.
    pub fn model_cards(&self) -> Vec<ModelCard> {
        let hw = self.hardware();
        let running: Vec<String> = self.installs.0.lock().keys().cloned().collect();
        self.models
            .status()
            .into_iter()
            .map(|s| {
                let cap = match &s.manifest {
                    Some(m) => capability(&hw, &m.backends, &m.supported_platforms, platform_id(), m.minimum_ram_bytes / (1 << 20)),
                    None => capability(&hw, &["cpu".into()], &[platform_id().into()], platform_id(), 0),
                };
                let id = s.manifest.as_ref().map(|m| m.id.clone()).or_else(|| s.installed.as_ref().map(|i| i.id.clone())).unwrap_or_default();
                ModelCard { in_progress: running.contains(&id), status: s, capability: cap }
            })
            .collect()
    }

    /// Resolve the models a pipeline needs (with hashes recorded for the job).
    pub fn resolve_models(&self, pipeline: &Pipeline) -> Result<BTreeMap<String, ResolvedModel>> {
        let cat = self.catalog();
        let mut out = BTreeMap::new();
        for id in pipeline.required_models() {
            let (version, files, sha256) = self.models.resolve(&id).ok_or_else(|| {
                let name = cat.get(&id).map(|m| m.display_name.clone()).unwrap_or_else(|| id.clone());
                anyhow!("{name} is not installed. Install it from the Model Library.")
            })?;
            let rec = self.models.installed(&id);
            let (architecture, scale, mbpm) = match cat.get(&id) {
                Some(m) => (m.architecture.clone(), m.scale, m.memory_per_megapixel_mb),
                None => (rec.as_ref().map(|r| r.architecture.clone()).unwrap_or_default(), rec.and_then(|r| r.custom_info).map(|c| c.scale).unwrap_or(1), 1500.0),
            };
            out.insert(id.clone(), ResolvedModel { id, version, files, sha256, architecture, scale, memory_per_megapixel_mb: mbpm });
        }
        Ok(out)
    }

    /// Start installing a model in the background.
    pub fn install_model(self: &Arc<Self>, id: &str, local_files: BTreeMap<String, PathBuf>) -> Result<()> {
        if self.catalog().get(id).is_none() {
            bail!("{id} is not in the model catalog");
        }
        let ctl = InstallControl::default();
        {
            let mut g = self.installs.0.lock();
            if g.contains_key(id) {
                bail!("this model is already being installed");
            }
            g.insert(id.to_string(), ctl.clone());
        }
        let app = self.clone();
        let id = id.to_string();
        std::thread::Builder::new()
            .name(format!("install-{id}"))
            .spawn(move || {
                let events = app.events.clone();
                let r = app.models.install(&id, &local_files, &ctl, &clearframe_pipeline::smoke::smoke_test, &mut |p| events(AppEvent::ModelProgress { progress: p }));
                app.installs.0.lock().remove(&id);
                match r {
                    Ok(_) => log::info!("installed model {id}"),
                    Err(InstallError::Paused) => log::info!("install of {id} paused"),
                    Err(InstallError::Cancelled) => {
                        app.models.discard_partial(&id);
                        log::info!("install of {id} cancelled")
                    }
                    Err(e) => {
                        log::warn!("install of {id} failed: {e}");
                        // Discard unsafe partial files for anything but network failures (which resume).
                        if !matches!(e, InstallError::Download(_)) {
                            app.models.discard_partial(&id);
                        }
                        app.emit(AppEvent::ModelError { id: id.clone(), message: e.to_string() });
                    }
                }
                app.emit(AppEvent::ModelChanged { id });
            })?;
        Ok(())
    }

    pub fn installs_running(&self) -> Vec<String> {
        self.installs.0.lock().keys().cloned().collect()
    }

    pub fn pause_install(&self, id: &str) -> bool {
        self.installs.0.lock().get(id).map(|c| c.pause.store(true, Ordering::SeqCst)).is_some()
    }

    pub fn cancel_install(&self, id: &str) -> bool {
        match self.installs.0.lock().get(id) {
            Some(c) => {
                c.cancel.store(true, Ordering::SeqCst);
                true
            }
            None => {
                self.models.discard_partial(id);
                self.emit(AppEvent::ModelChanged { id: id.to_string() });
                false
            }
        }
    }

    /// Presets and pending jobs that use a model (shown before removal).
    pub fn model_usage(&self, id: &str) -> Vec<String> {
        let mut v = Vec::new();
        for p in self.presets() {
            if p.pipeline.required_models().iter().any(|m| m == id) {
                v.push(format!("Preset: {}", p.name));
            }
        }
        for j in self.db.jobs(false).unwrap_or_default() {
            if !j.state.is_terminal() && j.spec_json.contains(&format!("\"{id}\"")) {
                v.push(format!("Job: {}", j.name));
            }
        }
        v
    }

    pub fn remove_model(&self, id: &str) -> Result<u64> {
        if self.installs.0.lock().contains_key(id) {
            bail!("cancel the running install first");
        }
        // Never delete weights while a running job or preview is using them.
        let in_use = self.queue.models_in_use();
        let freed = self.models.remove(id, &in_use).map_err(|e| anyhow!("{e}"))?;
        self.emit(AppEvent::ModelChanged { id: id.to_string() });
        Ok(freed)
    }

    pub fn verify_model(&self, id: &str) -> Result<clearframe_models::manager::VerifyReport> {
        let r = self.models.verify(id).map_err(|e| anyhow!("{e}"))?;
        self.emit(AppEvent::ModelChanged { id: id.to_string() });
        Ok(r)
    }

    /// Repair: re-verify and reinstall when broken.
    pub fn repair_model(self: &Arc<Self>, id: &str) -> Result<bool> {
        let ok = self.models.installed(id).map(|_| self.models.verify(id).map(|r| r.ok).unwrap_or(false)).unwrap_or(false);
        if ok {
            return Ok(true);
        }
        self.install_model(id, BTreeMap::new())?;
        Ok(false)
    }

    pub fn import_custom_model(&self, name: &str, license: &str, file: PathBuf, scale_hint: Option<u32>) -> Result<String> {
        let info = CustomModelInfo { display_name: name.to_string(), license: license.to_string(), source: file.to_string_lossy().into(), scale: scale_hint.unwrap_or(1) };
        let rules = KeyRules { strip_prefixes: vec!["module.".into()], drop_prefixes: vec![], wrappers: vec![] };
        let mut rec = self.models.import_custom(info, &file, &rules, &clearframe_pipeline::smoke::smoke_test).map_err(|e| anyhow!("{e}"))?;
        // Record the real scale reported by the test run ("srvgg_compact: x4, ...").
        if let Some(scale) = rec.smoke_test.as_deref().and_then(parse_scale) {
            if let Some(ci) = rec.custom_info.as_mut() {
                ci.scale = scale;
            }
            self.models.update_record(&rec).map_err(|e| anyhow!("{e}"))?;
        }
        self.emit(AppEvent::ModelChanged { id: rec.id.clone() });
        Ok(rec.id)
    }

    pub fn model_state(&self, id: &str) -> ModelState {
        self.models.status().into_iter().find(|s| s.manifest.as_ref().map(|m| m.id.as_str()) == Some(id) || s.installed.as_ref().map(|i| i.id.as_str()) == Some(id)).map(|s| s.state).unwrap_or(ModelState::NotInstalled)
    }
}

fn parse_scale(s: &str) -> Option<u32> {
    s.split(": x").nth(1)?.split(',').next()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn scale_parsing() {
        assert_eq!(super::parse_scale("srvgg_compact: x4, 32x24 -> 128x96 in 0.01s"), Some(4));
        assert_eq!(super::parse_scale("rife: 64x64"), None);
    }
}
