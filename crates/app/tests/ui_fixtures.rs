//! The desktop UI's browser demo backend and unit tests use JSON fixtures in
//! apps/desktop/src/lib/fixtures. This test keeps them identical to what the
//! Rust backend really sends. Regenerate with CF_UPDATE_FIXTURES=1.

use std::path::PathBuf;

use clearframe_core::settings::Settings;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../apps/desktop/src/lib/fixtures")
}

fn check(name: &str, value: serde_json::Value) {
    let path = fixtures().join(name);
    let text = serde_json::to_string_pretty(&value).unwrap() + "\n";
    if std::env::var_os("CF_UPDATE_FIXTURES").is_some() {
        std::fs::create_dir_all(fixtures()).unwrap();
        std::fs::write(&path, text).unwrap();
        return;
    }
    // Compared as text: parsing floats back is not exact without serde_json's float_roundtrip.
    let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{} missing; run with CF_UPDATE_FIXTURES=1", path.display())).replace("\r\n", "\n");
    assert!(on_disk == text, "{} is out of date; run `CF_UPDATE_FIXTURES=1 cargo test -p clearframe-app --test ui_fixtures`", path.display());
}

#[test]
fn operation_defaults_fixture() {
    check("operation_defaults.json", clearframe_app::ipc::operation_defaults());
}

#[test]
fn settings_fixture() {
    check("settings.json", serde_json::to_value(Settings::default()).unwrap());
}

#[test]
fn catalog_and_presets_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let app = clearframe_app::App::open(Some(dir.path()), clearframe_app::WorkerCommand::current_exe(), vec![], std::sync::Arc::new(|_| {})).unwrap();
    // Machine-specific capability text is normalised so the fixture is stable.
    let mut cards = serde_json::to_value(app.model_cards()).unwrap();
    for c in cards.as_array_mut().unwrap() {
        c["capability"] = serde_json::json!({ "level": "compatible_not_certified", "backend": "cpu", "reason": "fixture" });
    }
    check("models.json", cards);
    check("presets.json", serde_json::to_value(app.presets()).unwrap());
}
