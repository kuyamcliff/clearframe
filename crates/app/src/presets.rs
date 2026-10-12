//! Preset services: built-in seeding, CRUD, import/export.

use std::path::Path;

use anyhow::{anyhow, bail, Result};

use clearframe_core::db::{now, Db, StoredPreset};
use clearframe_pipeline::presets::{builtin_presets, new_preset_id, Preset};

use crate::App;

/// Insert or refresh built-in presets (user presets are untouched).
pub fn seed_builtins(db: &Db) -> Result<()> {
    let existing = db.presets().map_err(|e| anyhow!("{e}"))?;
    for p in builtin_presets() {
        let fav = existing.iter().find(|e| e.id == p.id).map(|e| e.favorite).unwrap_or(false);
        let created = existing.iter().find(|e| e.id == p.id).map(|e| e.created_at.clone()).unwrap_or_else(now);
        db.upsert_preset(&StoredPreset {
            id: p.id.clone(),
            name: p.name.clone(),
            category: p.category.clone(),
            favorite: fav,
            builtin: true,
            data_json: serde_json::to_string(&p)?,
            created_at: created,
            updated_at: now(),
        })
        .map_err(|e| anyhow!("{e}"))?;
    }
    Ok(())
}

fn to_preset(s: &StoredPreset) -> Option<Preset> {
    let mut p: Preset = serde_json::from_str(&s.data_json).ok()?;
    p.favorite = s.favorite;
    p.builtin = s.builtin;
    p.id = s.id.clone();
    Some(p)
}

impl App {
    pub fn presets(&self) -> Vec<Preset> {
        self.db.presets().unwrap_or_default().iter().filter_map(to_preset).collect()
    }

    pub fn preset(&self, id: &str) -> Result<Preset> {
        to_preset(&self.db.preset(id).map_err(|e| anyhow!("{e}"))?).ok_or_else(|| anyhow!("preset {id} is damaged"))
    }

    /// Create or update a user preset. Built-in presets cannot be overwritten:
    /// saving one creates a user copy.
    pub fn save_preset(&self, mut p: Preset) -> Result<Preset> {
        if p.builtin || p.id.is_empty() || p.id.starts_with("builtin.") {
            p.id = new_preset_id();
            p.builtin = false;
        }
        if let Ok(existing) = self.db.preset(&p.id) {
            if existing.builtin {
                bail!("built-in presets cannot be changed; save a copy instead");
            }
        }
        p.validate().map_err(|e| anyhow!("{e}"))?;
        let created = self.db.preset(&p.id).map(|e| e.created_at).unwrap_or_else(|_| now());
        self.db
            .upsert_preset(&StoredPreset {
                id: p.id.clone(),
                name: p.name.trim().to_string(),
                category: p.category.clone(),
                favorite: p.favorite,
                builtin: false,
                data_json: serde_json::to_string(&p)?,
                created_at: created,
                updated_at: now(),
            })
            .map_err(|e| anyhow!("{e}"))?;
        Ok(p)
    }

    pub fn duplicate_preset(&self, id: &str) -> Result<Preset> {
        let mut p = self.preset(id)?;
        p.id = new_preset_id();
        p.builtin = false;
        p.favorite = false;
        p.name = format!("{} copy", p.name).chars().take(80).collect();
        self.save_preset(p)
    }

    pub fn rename_preset(&self, id: &str, name: &str) -> Result<Preset> {
        let mut p = self.preset(id)?;
        if p.builtin {
            bail!("built-in presets cannot be renamed; duplicate it first");
        }
        p.name = name.to_string();
        self.save_preset(p)
    }

    pub fn delete_preset(&self, id: &str) -> Result<()> {
        if self.preset(id)?.builtin {
            bail!("built-in presets cannot be deleted");
        }
        self.db.delete_preset(id).map_err(|e| anyhow!("{e}"))
    }

    pub fn set_preset_favorite(&self, id: &str, fav: bool) -> Result<()> {
        self.db.set_preset_favorite(id, fav).map_err(|e| anyhow!("{e}"))
    }

    pub fn export_preset(&self, id: &str, path: &Path) -> Result<()> {
        let p = self.preset(id)?;
        std::fs::write(path, p.export())?;
        Ok(())
    }

    /// Import a `.cfxpreset` file. Returns the preset and the models it needs that are not installed.
    pub fn import_preset(&self, path: &Path) -> Result<(Preset, Vec<String>)> {
        let meta = std::fs::metadata(path)?;
        if meta.len() > clearframe_pipeline::presets::MAX_PRESET_BYTES as u64 {
            bail!("the file is too large to be a preset");
        }
        let data = std::fs::read(path)?;
        let p = Preset::import(&data).map_err(|e| anyhow!("{e}"))?;
        let missing: Vec<String> = p.pipeline.required_models().into_iter().filter(|m| self.models.resolve(m).is_none()).collect();
        let p = self.save_preset(p)?;
        Ok((p, missing))
    }
}
