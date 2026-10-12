//! Compute backend discovery and selection.

use candle_core::Device;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
}

impl BackendKind {
    pub fn id(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Cuda => "cuda",
            BackendKind::Metal => "metal",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            BackendKind::Cpu => "CPU",
            BackendKind::Cuda => "NVIDIA CUDA",
            BackendKind::Metal => "Apple Metal",
        }
    }
}

/// Status of a backend in this build on this machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendStatus {
    pub kind: BackendKind,
    /// Compiled into this build.
    pub compiled: bool,
    /// A device could actually be created.
    pub available: bool,
    pub detail: String,
}

/// Whether this binary was compiled with the given backend.
pub fn compiled(kind: BackendKind) -> bool {
    match kind {
        BackendKind::Cpu => true,
        BackendKind::Cuda => cfg!(feature = "cuda"),
        BackendKind::Metal => cfg!(feature = "metal"),
    }
}

/// Probe every backend.
pub fn probe() -> Vec<BackendStatus> {
    [BackendKind::Cpu, BackendKind::Cuda, BackendKind::Metal]
        .into_iter()
        .map(|kind| {
            let compiled = compiled(kind);
            if !compiled {
                return BackendStatus { kind, compiled, available: false, detail: "not included in this build".into() };
            }
            match create(kind, 0) {
                Ok(_) => BackendStatus { kind, compiled, available: true, detail: "device created".into() },
                Err(e) => BackendStatus { kind, compiled, available: false, detail: e },
            }
        })
        .collect()
}

/// Create a device for a backend.
pub fn create(kind: BackendKind, index: usize) -> Result<Device, String> {
    match kind {
        BackendKind::Cpu => Ok(Device::Cpu),
        BackendKind::Cuda => Device::new_cuda(index).map_err(|e| e.to_string()),
        BackendKind::Metal => Device::new_metal(index).map_err(|e| e.to_string()),
    }
}

/// Pick the best available backend unless one is requested.
pub fn select(requested: Option<BackendKind>) -> Result<(BackendKind, Device), String> {
    if let Some(kind) = requested {
        return create(kind, 0).map(|d| (kind, d));
    }
    for kind in [BackendKind::Cuda, BackendKind::Metal] {
        if compiled(kind) {
            if let Ok(d) = create(kind, 0) {
                return Ok((kind, d));
            }
        }
    }
    Ok((BackendKind::Cpu, Device::Cpu))
}
