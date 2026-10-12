//! Named weight storage for model adapters.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

/// A flat map of parameter name -> tensor, already on the target device.
#[derive(Clone)]
pub struct Weights {
    map: HashMap<String, Tensor>,
    device: Device,
}

impl Weights {
    /// Load a `.safetensors` file produced by the model installer.
    pub fn from_safetensors(path: &Path, device: &Device) -> Result<Self> {
        let map = candle_core::safetensors::load(path, device)?;
        let map = map
            .into_iter()
            .map(|(k, v)| Ok((k, if v.dtype() == DType::F32 { v } else { v.to_dtype(DType::F32)? })))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self { map, device: device.clone() })
    }

    /// Build from raw `(name, shape, data)` triples.
    pub fn from_raw<I>(items: I, device: &Device) -> Result<Self>
    where
        I: IntoIterator<Item = (String, Vec<usize>, Vec<f32>)>,
    {
        let mut map = HashMap::new();
        for (name, shape, data) in items {
            let t = Tensor::from_vec(data, shape.as_slice(), device)?;
            map.insert(name, t);
        }
        Ok(Self { map, device: device.clone() })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.map
            .get(name)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("model weights are missing tensor `{name}`")))
    }

    pub fn opt(&self, name: &str) -> Option<Tensor> {
        self.map.get(name).cloned()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Count tensors whose name starts with `prefix` followed by a numeric index, returning
    /// the number of distinct indices (e.g. `body.` -> number of `body.N.*` entries).
    pub fn count_indexed(&self, prefix: &str) -> usize {
        let mut max: Option<usize> = None;
        for k in self.map.keys() {
            if let Some(rest) = k.strip_prefix(prefix) {
                if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                    max = Some(max.map_or(idx, |m: usize| m.max(idx)));
                }
            }
        }
        max.map_or(0, |m| m + 1)
    }

    /// Linear interpolation of two weight sets with identical structure:
    /// `alpha * self + (1 - alpha) * other` (Real-ESRGAN "DNI").
    pub fn interpolate(&self, other: &Weights, alpha: f64) -> Result<Weights> {
        let mut map = HashMap::new();
        for (k, a) in &self.map {
            let b = other.get(k)?;
            map.insert(k.clone(), ((a * alpha)? + (b * (1.0 - alpha))?)?);
        }
        Ok(Weights { map, device: self.device.clone() })
    }

    /// Convert every tensor to `dtype`.
    pub fn map_dtype(&self, dtype: DType) -> Result<Weights> {
        if dtype == DType::F32 {
            return Ok(self.clone());
        }
        let map = self.map.iter().map(|(k, v)| Ok((k.clone(), v.to_dtype(dtype)?))).collect::<Result<HashMap<_, _>>>()?;
        Ok(Weights { map, device: self.device.clone() })
    }
}
