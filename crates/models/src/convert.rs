//! Conversion of verified upstream checkpoints into safetensors files.
//!
//! Conversion only rearranges tensor data: it parses checkpoints with the
//! restricted loader in [`crate::torchload`], applies declarative key rules
//! from the catalog (strip prefixes / drop unused tensors) and writes a
//! non-executable safetensors file.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::torchload::{self, StateDict};

#[derive(Debug, Error)]
pub enum ConvertError {
    #[error(transparent)]
    Load(#[from] torchload::LoadError),
    #[error("safetensors: {0}")]
    SafeTensors(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("checkpoint contains no usable tensors after applying conversion rules")]
    Empty,
    #[error("checkpoint contains non-finite values in `{0}`")]
    NonFinite(String),
}

/// Declarative tensor-name rules. Never code.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct KeyRules {
    /// Prefixes stripped from every tensor name (e.g. `module.`).
    #[serde(default)]
    pub strip_prefixes: Vec<String>,
    /// Tensors whose (stripped) name starts with any of these are dropped.
    #[serde(default)]
    pub drop_prefixes: Vec<String>,
    /// Wrapper keys to look inside, in order (e.g. `params_ema`).
    #[serde(default)]
    pub wrappers: Vec<String>,
}

/// Apply rules to a loaded state dict.
pub fn apply_rules(sd: StateDict, rules: &KeyRules) -> StateDict {
    let mut tensors = Vec::with_capacity(sd.tensors.len());
    for (mut name, t) in sd.tensors {
        for p in &rules.strip_prefixes {
            if let Some(rest) = name.strip_prefix(p.as_str()) {
                name = rest.to_string();
            }
        }
        if rules.drop_prefixes.iter().any(|p| name.starts_with(p.as_str())) {
            continue;
        }
        // BatchNorm bookkeeping counters are never needed for inference.
        if name.ends_with("num_batches_tracked") {
            continue;
        }
        tensors.push((name, t));
    }
    StateDict { tensors }
}

/// Load a checkpoint and write it as safetensors. Returns the number of tensors written.
pub fn checkpoint_to_safetensors(src: &Path, dst: &Path, rules: &KeyRules) -> Result<usize, ConvertError> {
    let wrappers: Vec<&str> = if rules.wrappers.is_empty() {
        torchload::DEFAULT_WRAPPERS.to_vec()
    } else {
        rules.wrappers.iter().map(|s| s.as_str()).collect()
    };
    let sd = torchload::load_checkpoint(src, &wrappers)?;
    let sd = apply_rules(sd, rules);
    write_safetensors(&sd, dst)
}

/// Write a state dict as an f32 safetensors file.
pub fn write_safetensors(sd: &StateDict, dst: &Path) -> Result<usize, ConvertError> {
    if sd.is_empty() {
        return Err(ConvertError::Empty);
    }
    for (name, t) in &sd.tensors {
        if t.data.iter().any(|v| !v.is_finite()) {
            return Err(ConvertError::NonFinite(name.clone()));
        }
    }
    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = sd
        .tensors
        .iter()
        .map(|(n, t)| (n.clone(), t.shape.clone(), t.data.iter().flat_map(|v| v.to_le_bytes()).collect()))
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView)> = bytes
        .iter()
        .map(|(n, shape, data)| {
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), data)
                .map(|v| (n.clone(), v))
                .map_err(|e| ConvertError::SafeTensors(e.to_string()))
        })
        .collect::<Result<_, _>>()?;
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), "pt".to_string());
    meta.insert("converted_by".to_string(), "clearframe".to_string());
    safetensors::serialize_to_file(views, &Some(meta), dst).map_err(|e| ConvertError::SafeTensors(e.to_string()))?;
    Ok(sd.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torchload::DenseTensor;

    #[test]
    fn rules_strip_and_drop() {
        let sd = StateDict {
            tensors: vec![
                ("module.block0.w".into(), DenseTensor { shape: vec![1], data: vec![1.0] }),
                ("module.teacher.w".into(), DenseTensor { shape: vec![1], data: vec![2.0] }),
                ("bn.num_batches_tracked".into(), DenseTensor { shape: vec![], data: vec![3.0] }),
            ],
        };
        let rules = KeyRules { strip_prefixes: vec!["module.".into()], drop_prefixes: vec!["teacher.".into()], wrappers: vec![] };
        let out = apply_rules(sd, &rules);
        assert_eq!(out.tensors.len(), 1);
        assert_eq!(out.tensors[0].0, "block0.w");
    }

    #[test]
    fn writes_readable_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.safetensors");
        let sd = StateDict { tensors: vec![("a.weight".into(), DenseTensor { shape: vec![2, 2], data: vec![1., 2., 3., 4.] })] };
        assert_eq!(write_safetensors(&sd, &p).unwrap(), 1);
        let bytes = std::fs::read(&p).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let t = st.tensor("a.weight").unwrap();
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn rejects_non_finite() {
        let dir = tempfile::tempdir().unwrap();
        let sd = StateDict { tensors: vec![("a".into(), DenseTensor { shape: vec![1], data: vec![f32::NAN] })] };
        assert!(matches!(write_safetensors(&sd, &dir.path().join("x")), Err(ConvertError::NonFinite(_))));
    }
}
