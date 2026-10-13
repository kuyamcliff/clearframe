//! Online update checks and small, signed app updates.
//!
//! Each release publishes `update.json` (signed with the release key in
//! `model-catalog/trusted_keys.json`) listing a small update package per platform: on
//! Windows only the app itself, not the bundled FFmpeg, so an update is a fraction of the
//! full installer. Nothing is downloaded until the user clicks Download, and nothing is
//! installed until they click Restart to update.

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use clearframe_models::catalog::Catalog;
use clearframe_models::signature::{self, TrustedKeys};

use crate::App;

pub const RELEASES_API: &str = "https://api.github.com/repos/kuyamcliff/clipfx-enhance/releases/latest";
pub const CATALOG_URL: &str = "https://raw.githubusercontent.com/kuyamcliff/clipfx-enhance/main/model-catalog/catalog.json";
pub const UPDATE_MANIFEST_URL: &str = "https://github.com/kuyamcliff/clipfx-enhance/releases/latest/download/update.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateInfo {
    pub current: String,
    pub latest: String,
    pub newer: bool,
    pub url: String,
    pub notes: String,
    /// Download size of the update package for this computer, if one is published.
    #[serde(default)]
    pub size: Option<u64>,
    /// The update can be downloaded and installed from inside the app.
    #[serde(default)]
    pub installable: bool,
    /// An update was already downloaded and verified, ready to install.
    #[serde(default)]
    pub downloaded: bool,
}

/// One platform's update package.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpdatePackage {
    pub url: String,
    pub sha256: String,
    pub size: u64,
    /// "nsis-patch" (Windows app-only installer) or "deb".
    pub kind: String,
}

/// The signed `update.json` published with every release.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpdateManifest {
    pub version: String,
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub notes: String,
    /// Release page, for platforms without an in-app package.
    #[serde(default)]
    pub page: String,
    pub packages: std::collections::BTreeMap<String, UpdatePackage>,
}

/// Key into [`UpdateManifest::packages`] for this build.
pub fn platform_key() -> &'static str {
    if cfg!(all(windows, target_arch = "x86_64")) {
        "windows-x64"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        // Only an installed .deb can be updated with a .deb; an AppImage or .rpm install is
        // pointed at the download page instead.
        let deb = std::env::var_os("APPIMAGE").is_none() && std::path::Path::new("/var/lib/dpkg/info/clip-fx-enhance.list").exists();
        if deb {
            "linux-x64-deb"
        } else {
            "linux-x64"
        }
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "macos-arm64"
    } else {
        "other"
    }
}

/// Verify and parse a manifest against the bundled release keys.
pub fn parse_manifest(body: &str, signature_b64: &str, keys: &TrustedKeys) -> Result<UpdateManifest> {
    signature::verify(body.as_bytes(), signature_b64, keys).map_err(|e| anyhow!("update manifest: {e}"))?;
    let m: UpdateManifest = serde_json::from_str(body)?;
    for (k, p) in &m.packages {
        if !p.url.starts_with("https://") || p.sha256.len() != 64 {
            bail!("update manifest: package {k} is malformed");
        }
    }
    Ok(m)
}

fn agent(url: &str) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(20))
        .user_agent(&format!("ClipFX-Enhance/{}", env!("CARGO_PKG_VERSION")))
        .try_proxy_from_env(clearframe_models::manager::use_proxy_for(url))
        .build()
}

fn version_tuple(v: &str) -> Vec<u64> {
    v.trim_start_matches('v').split(['.', '-']).filter_map(|p| p.parse().ok()).collect()
}

fn get_text(url: &str) -> Result<String> {
    Ok(agent(url).get(url).call().map_err(|e| anyhow!("{e}"))?.into_string()?)
}

impl App {
    fn fetch_manifest(&self) -> Result<UpdateManifest> {
        let keys = TrustedKeys::bundled();
        if keys.keys.is_empty() {
            bail!("this build has no release signing key");
        }
        let body = get_text(UPDATE_MANIFEST_URL)?;
        let sig = get_text(&format!("{UPDATE_MANIFEST_URL}.sig"))?;
        parse_manifest(&body, &sig, &keys)
    }

    fn updates_dir(&self) -> std::path::PathBuf {
        self.paths().cache_dir.join("updates")
    }

    /// Check for a newer version. Uses the signed update manifest when the release has
    /// one, otherwise the release page (no download, no install).
    pub fn check_for_update(&self) -> Result<UpdateInfo> {
        let current = env!("CARGO_PKG_VERSION").to_string();
        if let Ok(m) = self.fetch_manifest() {
            let pkg = m.packages.get(platform_key());
            let newer = version_tuple(&m.version) > version_tuple(&current);
            let downloaded = newer && pkg.is_some_and(|p| self.updates_dir().join(file_name(p)).is_file());
            return Ok(UpdateInfo {
                newer,
                url: if m.page.is_empty() { "https://www.clipfx.me/a/c8DN7xPn".into() } else { m.page.clone() },
                notes: m.notes.chars().take(4000).collect(),
                size: pkg.map(|p| p.size),
                installable: pkg.is_some(),
                downloaded,
                latest: m.version,
                current,
            });
        }
        self.check_release_page(current)
    }

    /// Download this platform's update package, verifying size and SHA-256.
    /// `progress(done, total)` is called as data arrives. Returns the verified file.
    pub fn download_update(&self, mut progress: impl FnMut(u64, u64)) -> Result<std::path::PathBuf> {
        use sha2::{Digest, Sha256};
        use std::io::{Read, Write};
        let m = self.fetch_manifest()?;
        if version_tuple(&m.version) <= version_tuple(env!("CARGO_PKG_VERSION")) {
            bail!("ClipFX Enhance is already up to date");
        }
        let pkg = m.packages.get(platform_key()).ok_or_else(|| anyhow!("no update package for this computer; download it from {}", m.page))?;
        let dir = self.updates_dir();
        std::fs::create_dir_all(&dir)?;
        let dest = dir.join(file_name(pkg));
        let part = dir.join(format!("{}.part", file_name(pkg)));
        let resp = agent(&pkg.url).get(&pkg.url).call().map_err(|e| anyhow!("{e}"))?;
        let mut reader = resp.into_reader().take(pkg.size + 1);
        let mut out = std::fs::File::create(&part)?;
        let mut hasher = Sha256::new();
        let (mut done, mut buf) = (0u64, vec![0u8; 1 << 16]);
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n])?;
            done += n as u64;
            progress(done, pkg.size);
        }
        out.sync_all()?;
        drop(out);
        let hash = hex::encode(hasher.finalize());
        if done != pkg.size || hash != pkg.sha256 {
            let _ = std::fs::remove_file(&part);
            bail!("the downloaded update failed verification; nothing was installed");
        }
        std::fs::rename(&part, &dest)?;
        Ok(dest)
    }

    /// Start installing a downloaded update. The caller should quit the app right after:
    /// the installer waits for it to close, replaces the app and starts the new version.
    pub fn install_update(&self) -> Result<()> {
        let m = self.fetch_manifest()?;
        let pkg = m.packages.get(platform_key()).ok_or_else(|| anyhow!("no update package for this computer"))?;
        let file = self.updates_dir().join(file_name(pkg));
        if !file.is_file() {
            bail!("download the update first");
        }
        launch_installer(&pkg.kind, &file)
    }

    fn check_release_page(&self, current: String) -> Result<UpdateInfo> {
        let v: serde_json::Value = agent(RELEASES_API).get(RELEASES_API).call().map_err(|e| anyhow!("{e}"))?.into_json()?;
        let latest = v["tag_name"].as_str().unwrap_or("").to_string();
        if latest.is_empty() {
            bail!("no published release was found");
        }
        Ok(UpdateInfo {
            newer: version_tuple(&latest) > version_tuple(&current),
            current,
            url: v["html_url"].as_str().unwrap_or("").into(),
            notes: v["body"].as_str().unwrap_or("").chars().take(4000).collect(),
            latest,
            size: None,
            installable: false,
            downloaded: false,
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

fn file_name(p: &UpdatePackage) -> String {
    let name = p.url.rsplit('/').next().unwrap_or("update").to_string();
    clearframe_core::fsutil::sanitize_component(&name)
}

#[cfg(windows)]
fn launch_installer(kind: &str, file: &std::path::Path) -> Result<()> {
    if kind != "nsis-patch" {
        bail!("unsupported update package {kind}");
    }
    let dir = std::env::current_exe()?.parent().ok_or_else(|| anyhow!("no install folder"))?.to_path_buf();
    // `start` goes through ShellExecute, so Windows shows the admin prompt the installer needs.
    // NSIS requires /D= to be last and unquoted.
    std::process::Command::new("cmd").args(["/C", "start", "\"\""]).arg(file).arg("/S").arg(format!("/D={}", dir.display())).spawn()?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn launch_installer(kind: &str, file: &std::path::Path) -> Result<()> {
    if kind != "deb" {
        bail!("unsupported update package {kind}");
    }
    // pkexec shows the system password prompt; apt resolves any new dependencies.
    std::process::Command::new("pkexec").args(["apt-get", "install", "-y"]).arg(file).spawn()?;
    Ok(())
}

#[cfg(not(any(windows, target_os = "linux")))]
fn launch_installer(_kind: &str, _file: &std::path::Path) -> Result<()> {
    bail!("in-app updates are not available on this platform yet")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(pk: &str) -> TrustedKeys {
        TrustedKeys { comment: String::new(), keys: vec![signature::TrustedKey { id: "test".into(), public_key_base64: pk.into() }] }
    }

    #[test]
    fn manifest_must_be_signed_by_a_trusted_key() {
        let (sk, pk) = signature::keygen();
        let body = r#"{"version":"9.9.9","packages":{"windows-x64":{"url":"https://example.org/u.exe","sha256":"0000000000000000000000000000000000000000000000000000000000000000","size":10,"kind":"nsis-patch"}}}"#;
        let sig = signature::sign(body.as_bytes(), &sk).unwrap();
        let m = parse_manifest(body, &sig, &keys(&pk)).unwrap();
        assert_eq!(m.version, "9.9.9");
        // Tampered body, wrong key and plain-http packages are all refused.
        assert!(parse_manifest(&body.replace("9.9.9", "9.9.8"), &sig, &keys(&pk)).is_err());
        let (_, other) = signature::keygen();
        assert!(parse_manifest(body, &sig, &keys(&other)).is_err());
        let http = body.replace("https://", "http://");
        let sig2 = signature::sign(http.as_bytes(), &sk).unwrap();
        assert!(parse_manifest(&http, &sig2, &keys(&pk)).is_err());
    }

    #[test]
    fn versions() {
        assert!(super::version_tuple("v1.2.10") > super::version_tuple("1.2.9"));
        assert!(super::version_tuple("1.0.0") == super::version_tuple("v1.0.0"));
    }
}
