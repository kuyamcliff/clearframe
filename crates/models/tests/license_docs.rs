//! MODEL_LICENSES.md must describe every catalog model with its current license.

#[test]
fn model_licenses_doc_matches_catalog() {
    let doc = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../MODEL_LICENSES.md")).unwrap();
    let catalog = clearframe_models::catalog::Catalog::bundled();
    for m in &catalog.models {
        assert!(doc.contains(&format!("`{}`", m.id)), "MODEL_LICENSES.md does not mention {}", m.id);
        assert!(doc.contains(&format!("| {} | {} |", m.weights_license, m.code_license)), "MODEL_LICENSES.md has an outdated license for {}", m.id);
        for a in &m.artifacts {
            assert!(doc.contains(&a.sha256), "MODEL_LICENSES.md lacks the hash of {} / {}", m.id, a.filename);
        }
    }
}
