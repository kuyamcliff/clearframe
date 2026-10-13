//! Compute backend discovery and selection.

use candle_core::Device;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    /// Any GPU through wgpu: Vulkan, DirectX 12 or Metal (NVIDIA, AMD, Intel, Apple).
    Gpu,
}

impl BackendKind {
    pub fn id(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Cuda => "cuda",
            BackendKind::Metal => "metal",
            BackendKind::Gpu => "gpu",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            BackendKind::Cpu => "CPU",
            BackendKind::Cuda => "NVIDIA CUDA",
            BackendKind::Metal => "Apple Metal",
            BackendKind::Gpu => "GPU",
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
        BackendKind::Gpu => cfg!(feature = "gpu"),
    }
}

/// Hardware GPU adapters usable through the cross-vendor backend, best first.
/// Software rasterisers (e.g. Mesa llvmpipe) are listed but flagged with kind "cpu".
pub fn gpu_adapters() -> Vec<GpuAdapter> {
    #[cfg(feature = "gpu")]
    {
        crate::gpu::list_adapters()
    }
    #[cfg(not(feature = "gpu"))]
    {
        Vec::new()
    }
}

#[cfg(feature = "gpu")]
pub use crate::gpu::AdapterSummary as GpuAdapter;

/// Description of a GPU adapter (empty list when the GPU backend isn't compiled in).
#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuAdapter {
    pub index: usize,
    pub name: String,
    pub vendor: String,
    pub kind: String,
    pub api: String,
    pub max_buffer_mb: u64,
}

/// Index of the first real (non-software) GPU adapter.
pub fn best_hardware_gpu() -> Option<usize> {
    gpu_adapters().into_iter().find(|a| a.kind != "cpu").map(|a| a.index)
}

/// Probe every backend.
pub fn probe() -> Vec<BackendStatus> {
    [BackendKind::Cpu, BackendKind::Gpu, BackendKind::Cuda, BackendKind::Metal]
        .into_iter()
        .map(|kind| {
            let compiled = compiled(kind);
            if !compiled {
                return BackendStatus { kind, compiled, available: false, detail: "not included in this build".into() };
            }
            if kind == BackendKind::Gpu {
                let list = gpu_adapters();
                let real: Vec<_> = list.iter().filter(|a| a.kind != "cpu").map(|a| format!("{} ({})", a.name, a.api)).collect();
                return match (real.is_empty(), list.is_empty()) {
                    (false, _) => BackendStatus { kind, compiled, available: true, detail: real.join(", ") },
                    (true, false) => BackendStatus { kind, compiled, available: false, detail: "only a software renderer was found".into() },
                    (true, true) => BackendStatus { kind, compiled, available: false, detail: "no Vulkan, DirectX 12 or Metal GPU was found".into() },
                };
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
        // Image models run through wgpu; everything else (and weight loading) stays on the CPU.
        BackendKind::Gpu => gpu_context(index).map(|_| Device::Cpu),
    }
}

/// The wgpu context for adapter `index`.
#[cfg(feature = "gpu")]
pub fn gpu_context(index: usize) -> Result<std::sync::Arc<crate::gpu::Gpu>, String> {
    crate::gpu::context(index)
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_context(_index: usize) -> Result<(), String> {
    Err("this build has no GPU backend".into())
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
    // A real GPU through wgpu; never auto-select a software renderer (it's slower than the CPU path).
    if let Some(i) = best_hardware_gpu() {
        if let Ok(d) = create(BackendKind::Gpu, i) {
            return Ok((BackendKind::Gpu, d));
        }
    }
    Ok((BackendKind::Cpu, Device::Cpu))
}
