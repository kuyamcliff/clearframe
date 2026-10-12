//! User-initiated online checks (never automatic unless the user enables them).

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use clearframe_models::catalog::Catalog;
use clearframe_models::signature::{self, TrustedKeys};

use crate::App;

pub const RELEASES_API: &str = "https://api.github.com/repos/kuyamcliff/clearframe/releases/latest";
pub const CATALOG_URL: &str = "https://raw.githubusercontent.com/kuyamcliff/clearframe/main/model-catalog/catalog.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateInfo {
    pub current: String,
    pub latest: String,
    pub newer: bool,
    pub url: String,
    pub notes: String,
}

fn agent(url: &str) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(20))
        .user_agent(&format!("ClearFrame/{}", env!("CARGO_PKG_VERSION")))
        .try_proxy_from_env(clearframe_models::manager::use_proxy_for(url))
        .build()
}

fn version_tuple(v: &str) -> Vec<u64> {
    v.trim_start_matches('v').split(['.', '-']).filter_map(|p| p.parse().ok()).collect()
}

impl App {
    /// Query the latest release (no download, no install).
    pub fn check_for_update(&self) -> Result<UpdateInfo> {
        let v: serde_json::Value = agent(RELEASES_API).get(RELEASES_API).call().map_err(|e| anyhow!("{e}"))?.into_json()?;
        let latest = v["tag_name"].as_str().unwrap_or("").to_string();
        if latest.is_empty() {
            bail!("no published release was found");
        }
        let current = env!("CARGO_PKG_VERSION").to_string();
        Ok(UpdateInfo {
            newer: version_tuple(&latest) > version_tuple(&current),
            current,
            url: v["html_url"].as_str().unwrap_or("").into(),
            notes: v["body"].as_str().unwrap_or("").chars().take(4000).collect(),
            latest,
        })
    }

    /// Download a signed catalog update. Accepted only with a valid signature from a trusted key.
    pub fn update_catalog(&self) -> Result<String> {
        let keys = TrustedKeys::bundled();
        if keys.keys.is_empty() {
            bail!("This build has no trusted catalog signing key, so online catalog updates are disabled. The bundled catalog is used.");
        }
        let body = agent(CATALOG_URL).get(CATALOG_URL).call().map_err(|e| anyhow!("{e}"))?.into_string()?;
        let sig_url = format!("{CATALOG_URL}.sig");
        let sig = agent(&sig_url).get(&sig_url).call().map_err(|e| anyhow!("{e}"))?.into_string()?;
        let key = signature::verify(body.as_bytes(), &sig, &keys).map_err(|e| anyhow!("{e}"))?;
        let c = Catalog::parse(&body).map_err(|e| anyhow!("{e}"))?;
        let current = self.catalog();
        if c.catalog_version <= current.catalog_version {
            return Ok(format!("The model catalog is up to date ({}).", current.catalog_version));
        }
        self.db.kv_set("catalog_override", &body).map_err(|e| anyhow!("{e}"))?;
        self.models.set_catalog(c.clone());
        Ok(format!("Catalog updated to {} (signed by {key}).", c.catalog_version))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn versions() {
        assert!(super::version_tuple("v1.2.10") > super::version_tuple("1.2.9"));
        assert!(super::version_tuple("1.0.0") == super::version_tuple("v1.0.0"));
    }
}
