//! Cross-vendor GPU inference on wgpu (Vulkan, DirectX 12 and Metal).
//!
//! Every network the app uses (Real-ESRGAN compact and RRDB, FFDNet, DRUNet, NAFNet,
//! RIFE, GFPGAN and RetinaFace) has a GPU implementation in [`nets`].
//! Tensors are planar NCHW `f32` buffers with a batch of one. A forward pass is
//! recorded into compute passes, and intermediate buffers come from a pool so the deep
//! networks don't allocate per layer.
//!
//! Short submissions: Windows resets the graphics driver (TDR) when one piece of GPU work runs
//! for more than about two seconds, and laptop GPUs are slow enough to hit that with one large
//! tile. So large convolutions are split into bands of rows, and work is submitted in pieces
//! of about [`TARGET_SUBMIT_SECS`], sized from the measured speed of this GPU.

pub mod nets;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

/// Aim for each GPU submission to take about this long (far below the ~2 s Windows TDR limit,
/// with room for a speed estimate that is several times too high).
pub const TARGET_SUBMIT_SECS: f64 = 0.15;
/// Longest wait for one submission before the GPU is considered hung.
const SUBMIT_TIMEOUT: Duration = Duration::from_secs(60);

use wgpu::util::DeviceExt;

/// What a GPU adapter is, for the settings UI and diagnostics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct AdapterSummary {
    pub index: usize,
    pub name: String,
    pub vendor: String,
    /// "discrete", "integrated", "virtual", "cpu" or "other".
    pub kind: String,
    /// "vulkan", "dx12", "metal" or "gl".
    pub api: String,
    /// Largest single buffer the device accepts, in megabytes.
    pub max_buffer_mb: u64,
}

fn vendor_name(id: u32) -> &'static str {
    match id {
        0x10de => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x8086 => "Intel",
        0x106b => "Apple",
        0x5143 => "Qualcomm",
        0x13b5 => "ARM",
        _ => "Other",
    }
}

fn instance() -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    // OpenGL can't run these compute kernels reliably; use the modern APIs only.
    desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL;
    wgpu::Instance::new(desc)
}

fn adapters() -> Vec<wgpu::Adapter> {
    let inst = instance();
    let mut list = pollster::block_on(inst.enumerate_adapters(wgpu::Backends::all()));
    // Prefer real GPUs; software rasterisers last. Several APIs can expose the same
    // card (e.g. Vulkan and DX12), so keep the first entry per name.
    list.sort_by_key(|a| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    });
    let mut seen = std::collections::HashSet::new();
    list.retain(|a| seen.insert(a.get_info().name));
    list
}

/// Every usable compute adapter, best first.
pub fn list_adapters() -> Vec<AdapterSummary> {
    adapters()
        .iter()
        .enumerate()
        .map(|(index, a)| {
            let i = a.get_info();
            AdapterSummary {
                index,
                name: i.name.clone(),
                vendor: vendor_name(i.vendor).to_string(),
                kind: match i.device_type {
                    wgpu::DeviceType::DiscreteGpu => "discrete",
                    wgpu::DeviceType::IntegratedGpu => "integrated",
                    wgpu::DeviceType::VirtualGpu => "virtual",
                    wgpu::DeviceType::Cpu => "cpu",
                    wgpu::DeviceType::Other => "other",
                }
                .to_string(),
                api: format!("{:?}", i.backend).to_ascii_lowercase(),
                max_buffer_mb: a.limits().max_storage_buffer_binding_size.min(a.limits().max_buffer_size) / (1 << 20),
            }
        })
        .collect()
}

/// A device, its queue, compiled kernels and a buffer pool.
pub struct Gpu {
    pub info: AdapterSummary,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    pool: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
    /// Which thread is recording and running a forward pass (see [`Gpu::turn`]), and how deeply.
    turn: Mutex<(Option<ThreadId>, u32)>,
    turn_free: Condvar,
    /// Largest storage buffer binding, in bytes.
    pub max_binding: u64,
    /// Measured throughput in multiply-adds per second; sizes submissions.
    rate: AtomicU64,
    /// Set when the driver reports the device lost (e.g. after a Windows TDR reset).
    lost: Arc<AtomicBool>,
}

/// One thread's exclusive use of a [`Gpu`] while it records, submits and waits for a forward pass.
///
/// Pooled buffers go back to the pool as soon as their tensor is dropped, while commands that use them
/// may still be unsubmitted. Within one thread that is safe (later commands are ordered after earlier
/// ones), but another thread could take such a buffer and fill it with `queue.write_buffer`, which lands
/// before the next submission of either thread. Taking turns keeps forward passes from several threads
/// (pipeline stages, frames in flight) apart; their CPU work still overlaps. Re-entrant per thread.
pub struct GpuTurn {
    gpu: Arc<Gpu>,
}

impl Drop for GpuTurn {
    fn drop(&mut self) {
        let mut t = self.gpu.turn.lock().unwrap();
        t.1 -= 1;
        if t.1 == 0 {
            t.0 = None;
            self.gpu.turn_free.notify_one();
        }
    }
}

static CONTEXTS: OnceLock<Mutex<HashMap<usize, Arc<Gpu>>>> = OnceLock::new();

/// The shared context for adapter `index` (0 = best GPU), created once.
pub fn context(index: usize) -> Result<Arc<Gpu>, String> {
    let map = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().unwrap();
    if let Some(g) = map.get(&index) {
        if !g.is_lost() {
            return Ok(g.clone());
        }
        log::warn!("GPU {} was reset by the driver; reopening it", g.info.name);
        map.remove(&index);
    }
    let g = Arc::new(Gpu::new(index)?);
    map.insert(index, g.clone());
    Ok(g)
}

impl Gpu {
    pub fn new(index: usize) -> Result<Self, String> {
        let list = adapters();
        let adapter =
            list.into_iter()
                .nth(index)
                .ok_or_else(|| if index == 0 { "no Vulkan, DirectX 12 or Metal GPU was found".to_string() } else { format!("GPU #{index} was not found") })?;
        let info = list_adapters().into_iter().nth(index).ok_or("GPU disappeared")?;
        let al = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        limits.max_storage_buffers_per_shader_stage = al.max_storage_buffers_per_shader_stage.max(limits.max_storage_buffers_per_shader_stage);
        let desc = wgpu::DeviceDescriptor { label: Some("clipfx-compute"), required_limits: limits, ..Default::default() };
        let (device, queue) = pollster::block_on(adapter.request_device(&desc)).map_err(|e| format!("could not open {}: {e}", info.name))?;
        // Errors are caught per forward pass with error scopes; anything that escapes is
        // logged instead of wgpu's default panic.
        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| log::error!("GPU error: {e}")));
        let lost = Arc::new(AtomicBool::new(false));
        let flag = lost.clone();
        let name = info.name.clone();
        device.set_device_lost_callback(move |reason, msg| {
            flag.store(true, Ordering::SeqCst);
            log::error!("GPU {name} lost ({reason:?}): {msg}");
        });
        // Starting speed estimate until the first submissions are measured: deliberately low so
        // the first pieces of work are short.
        let rate = std::env::var("CLIPFX_GPU_RATE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(match info.kind.as_str() {
            "discrete" => 100e9,
            "integrated" => 15e9,
            _ => 2e9,
        });
        let max_binding = al.max_storage_buffer_binding_size.min(al.max_buffer_size);
        let mut pipelines = HashMap::new();
        for (name, src) in KERNELS {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(format!("{PRELUDE}\n{src}").into()) });
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            pipelines.insert(*name, p);
        }
        Ok(Self {
            info,
            device,
            queue,
            pipelines,
            pool: Mutex::new(HashMap::new()),
            turn: Mutex::new((None, 0)),
            turn_free: Condvar::new(),
            max_binding,
            rate: AtomicU64::new(rate as u64),
            lost,
        })
    }

    /// Whether the driver has reset or removed this device; a new context is needed.
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    /// Multiply-adds that fit in one submission of about [`TARGET_SUBMIT_SECS`].
    fn submit_budget(&self) -> u64 {
        // Test hook: a fixed (tiny) budget exercises banding and intermediate submissions.
        static FIXED: OnceLock<Option<u64>> = OnceLock::new();
        if let Some(b) = *FIXED.get_or_init(|| std::env::var("CLIPFX_GPU_SUBMIT_MACS").ok().and_then(|v| v.parse().ok())) {
            return b;
        }
        ((self.rate.load(Ordering::Relaxed) as f64 * TARGET_SUBMIT_SECS) as u64).max(20_000_000)
    }

    /// Update the speed estimate from one measured submission. Slower results are taken at
    /// once (safety first); faster ones are blended in gradually.
    fn record_speed(&self, macs: u64, secs: f64) {
        if macs < 10_000_000 || secs <= 0.0 {
            return;
        }
        let measured = macs as f64 / secs;
        let old = self.rate.load(Ordering::Relaxed) as f64;
        let new = if measured < old { measured } else { old * 0.7 + measured * 0.3 };
        self.rate.store(new.clamp(1e8, 1e14) as u64, Ordering::Relaxed);
    }

    /// Error text for a submission that did not complete.
    fn describe_failure(&self, e: &str) -> String {
        // The lost-device callback can run during the poll that failed.
        // A submission that never ran ("not returned by a successful submission") also means
        // the device is no longer usable.
        let l = e.to_ascii_lowercase();
        if self.is_lost() || l.contains("lost") || l.contains("successful submission") {
            self.lost.store(true, Ordering::SeqCst);
            format!("GPU was reset by the driver: {e}")
        } else {
            format!("GPU did not finish: {e}")
        }
    }

    /// Current speed estimate in multiply-adds per second (for logs and tests).
    pub fn speed_estimate(&self) -> f64 {
        self.rate.load(Ordering::Relaxed) as f64
    }

    /// Upload a read-only parameter array (weights, biases, slopes).
    pub fn constant(&self, data: &[f32]) -> wgpu::Buffer {
        let data = if data.is_empty() { &[0f32][..] } else { data };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE })
    }

    fn class(bytes: u64) -> u64 {
        // Round up to one of 8 sizes per power of two (≤ 12.5% waste) so similar tiles share buffers.
        let b = bytes.max(256);
        let top = 63 - b.leading_zeros() as u64;
        if top < 3 {
            return b;
        }
        let step = 1u64 << (top - 3);
        b.div_ceil(step) * step
    }

    fn take(&self, bytes: u64) -> wgpu::Buffer {
        let size = Self::class(bytes);
        if let Some(b) = self.pool.lock().unwrap().get_mut(&size).and_then(|v| v.pop()) {
            return b;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn give(&self, b: wgpu::Buffer) {
        self.pool.lock().unwrap().entry(b.size()).or_default().push(b);
    }

    /// Wait until no other thread is running a forward pass on this GPU, then hold it until the
    /// returned guard is dropped.
    pub fn turn(self: &Arc<Self>) -> GpuTurn {
        let me = std::thread::current().id();
        let mut t = self.turn.lock().unwrap();
        while t.0.is_some_and(|o| o != me) {
            t = self.turn_free.wait(t).unwrap();
        }
        t.0 = Some(me);
        t.1 += 1;
        GpuTurn { gpu: self.clone() }
    }

    /// Free every pooled buffer.
    pub fn trim(&self) {
        self.pool.lock().unwrap().clear();
    }

    /// Bytes currently held by the pool (for diagnostics and tests).
    pub fn pooled_bytes(&self) -> u64 {
        self.pool.lock().unwrap().iter().map(|(s, v)| s * v.len() as u64).sum()
    }
}

/// A planar `f32` tensor of shape (c, h, w) living on the GPU.
pub struct GTensor {
    buf: Option<wgpu::Buffer>,
    gpu: Arc<Gpu>,
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

impl GTensor {
    pub fn new(gpu: &Arc<Gpu>, c: u32, h: u32, w: u32) -> Result<Self, String> {
        let bytes = c as u64 * h as u64 * w as u64 * 4;
        if bytes > gpu.max_binding {
            return Err(format!("a {c}×{h}×{w} tensor needs {} MB, more than the GPU allows in one buffer ({} MB); use a smaller tile", bytes >> 20, gpu.max_binding >> 20));
        }
        Ok(Self { buf: Some(gpu.take(bytes)), gpu: gpu.clone(), c, h, w })
    }

    fn buf(&self) -> &wgpu::Buffer {
        self.buf.as_ref().unwrap()
    }

    /// Whole-tensor view.
    pub fn all(&self) -> View<'_> {
        View { t: self, off: 0, len: self.c }
    }

    /// Channels `off .. off + len`.
    pub fn ch(&self, off: u32, len: u32) -> View<'_> {
        debug_assert!(off + len <= self.c);
        View { t: self, off, len }
    }

    fn hw(&self) -> u32 {
        self.h * self.w
    }
}

impl Drop for GTensor {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            self.gpu.give(b);
        }
    }
}

/// A channel range of a tensor.
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub t: &'a GTensor,
    pub off: u32,
    pub len: u32,
}

/// Activation applied after bias.
#[derive(Clone, Copy)]
pub enum GAct<'a> {
    None,
    Relu,
    Leaky(f32),
    Prelu(&'a wgpu::Buffer),
}

/// Convolution weights prepared for the GPU.
pub struct GConv {
    pub cin: u32,
    pub cout: u32,
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
    pub dilation: u32,
    pub groups: u32,
    /// `true` when the fast 3×3 kernel applies; then `weight` is in its blocked layout.
    fast: bool,
    weight: wgpu::Buffer,
    bias: wgpu::Buffer,
}

const FAST_OCB: u32 = 8;

impl GConv {
    /// `weight` is PyTorch layout [cout][cin/groups][k][k].
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu: &Gpu, weight: &[f32], bias: Option<&[f32]>, cout: u32, cin_g: u32, k: u32, stride: u32, pad: u32, dilation: u32, groups: u32) -> Self {
        let fast = k == 3 && stride == 1 && pad == 1 && dilation == 1 && groups == 1;
        let w = if fast {
            // [ocb][ic][tap][8] so each thread reads 8 output channels as two vec4s.
            let ocb = cout.div_ceil(FAST_OCB) as usize;
            let (cin, kk) = (cin_g as usize, 9usize);
            let mut out = vec![0f32; ocb * cin * kk * FAST_OCB as usize];
            for oc in 0..cout as usize {
                let (b, j) = (oc / FAST_OCB as usize, oc % FAST_OCB as usize);
                for ic in 0..cin {
                    for t in 0..kk {
                        out[((b * cin + ic) * kk + t) * FAST_OCB as usize + j] = weight[(oc * cin + ic) * kk + t];
                    }
                }
            }
            out
        } else {
            weight.to_vec()
        };
        let mut b = vec![0f32; cout.div_ceil(FAST_OCB) as usize * FAST_OCB as usize];
        if let Some(bias) = bias {
            b[..bias.len()].copy_from_slice(bias);
        }
        Self { cin: cin_g * groups, cout, k, stride, pad, dilation, groups, fast, weight: gpu.constant(&w), bias: gpu.constant(&b) }
    }

    pub fn out_size(&self, h: u32, w: u32) -> (u32, u32) {
        let f = |n: u32| (n + 2 * self.pad - self.dilation * (self.k - 1) - 1) / self.stride + 1;
        (f(h), f(w))
    }
}

/// Records a sequence of kernel dispatches into one compute pass.
pub struct Recorder {
    gpu: Arc<Gpu>,
    encoder: wgpu::CommandEncoder,
    pass: Option<wgpu::ComputePass<'static>>,
    dummy: wgpu::Buffer,
    // Popped in reverse order in `finish_read`.
    scopes: Vec<(wgpu::ErrorScopeGuard, &'static str)>,
    /// Whether any kernel has been recorded (uploads after that must be ordered copies, see `upload`).
    recorded: bool,
    /// Estimated multiply-adds recorded since the last submission.
    pending: u64,
    /// First error from an intermediate submission; reported by `finish_*`.
    failed: Option<String>,
    /// Held until the recorder is finished or dropped (declared last so it is released last).
    _turn: GpuTurn,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    a: [u32; 4],
    b: [u32; 4],
    c: [u32; 4],
    d: [u32; 4],
    e: [u32; 4],
    f: [f32; 4],
}

impl Params {
    fn zero() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

impl Recorder {
    pub fn new(gpu: &Arc<Gpu>) -> Self {
        let turn = gpu.turn();
        let scopes = Self::push_scopes(gpu);
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clipfx-forward") });
        let pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }).forget_lifetime();
        Self { gpu: gpu.clone(), encoder, pass: Some(pass), dummy: gpu.constant(&[0.0]), scopes, recorded: false, pending: 0, failed: None, _turn: turn }
    }

    fn push_scopes(gpu: &Gpu) -> Vec<(wgpu::ErrorScopeGuard, &'static str)> {
        vec![
            (gpu.device.push_error_scope(wgpu::ErrorFilter::Validation), "GPU validation error"),
            (gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory), "GPU out of memory"),
        ]
    }

    /// Submit what has been recorded so far, wait for it, and continue in a new command buffer.
    /// Keeps each piece of GPU work short and measures the GPU's speed.
    fn flush(&mut self) {
        if self.failed.is_some() || self.pending == 0 {
            return;
        }
        drop(self.pass.take());
        let next = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("clipfx-forward") });
        let enc = std::mem::replace(&mut self.encoder, next);
        let started = Instant::now();
        let idx = self.gpu.queue.submit([enc.finish()]);
        for (scope, what) in self.scopes.drain(..).rev() {
            if let Some(e) = pollster::block_on(scope.pop()) {
                self.failed.get_or_insert_with(|| format!("{what}: {e}"));
            }
        }
        self.scopes = Self::push_scopes(&self.gpu);
        if self.failed.is_none() {
            match self.gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: Some(SUBMIT_TIMEOUT) }) {
                Ok(_) => self.gpu.record_speed(self.pending, started.elapsed().as_secs_f64()),
                Err(e) => self.failed = Some(self.gpu.describe_failure(&e.to_string())),
            }
        }
        self.pending = 0;
        self.pass = Some(self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }).forget_lifetime());
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    fn dispatch(&mut self, kernel: &str, p: Params, bufs: &[&wgpu::Buffer], groups: (u32, u32, u32)) {
        // Light kernels (element-wise, resampling, pooling): a nominal cost per thread.
        let cost = groups.0.max(1) as u64 * groups.1.max(1) as u64 * groups.2.max(1) as u64 * 256 * 16;
        self.dispatch_cost(kernel, p, bufs, groups, cost);
    }

    /// Record a dispatch estimated at `cost` multiply-adds, submitting earlier work first when
    /// adding this one would make the submission too long.
    fn dispatch_cost(&mut self, kernel: &str, p: Params, bufs: &[&wgpu::Buffer], groups: (u32, u32, u32), cost: u64) {
        if self.failed.is_some() {
            return;
        }
        if self.pending > 0 && self.pending + cost > self.gpu.submit_budget() {
            self.flush();
            if self.failed.is_some() {
                return;
            }
        }
        self.pending += cost;
        let pipe = &self.gpu.pipelines[kernel];
        let ub = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM });
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() });
        }
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries });
        self.recorded = true;
        let pass = self.pass.as_mut().unwrap();
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups.0.max(1), groups.1.max(1), groups.2.max(1));
    }

    /// Upload planar data as a new tensor.
    pub fn upload(&mut self, c: u32, h: u32, w: u32, data: &[f32]) -> Result<GTensor, String> {
        let t = GTensor::new(&self.gpu, c, h, w)?;
        if !self.recorded {
            self.gpu.queue.write_buffer(t.buf(), 0, bytemuck::cast_slice(data));
            return Ok(t);
        }
        // `queue.write_buffer` lands before this recorder's commands run. The pooled buffer may still be written by an
        // already-recorded kernel (a dropped intermediate), which would then overwrite the upload, so copy in order.
        let staging = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("upload"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        drop(self.pass.take());
        self.encoder.copy_buffer_to_buffer(&staging, 0, t.buf(), 0, std::mem::size_of_val(data) as u64);
        self.pass = Some(self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }).forget_lifetime());
        Ok(t)
    }

    /// `act(conv(x) + bias) * alpha (+ res * beta)`, written into `out` channels `out.off..`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_into(&mut self, conv: &GConv, x: View<'_>, act: GAct<'_>, out: View<'_>, res: Option<(View<'_>, f32)>, alpha: f32) {
        assert_eq!(x.len, conv.cin, "conv input channels");
        assert!(out.len >= conv.cout);
        let (oh, ow) = conv.out_size(x.t.h, x.t.w);
        assert!(out.t.h == oh && out.t.w == ow, "conv output size");
        let (act_id, slope, slopes) = match act {
            GAct::None => (0, 0.0, &self.dummy),
            GAct::Relu => (1, 0.0, &self.dummy),
            GAct::Leaky(s) => (2, s, &self.dummy),
            GAct::Prelu(b) => (3, 0.0, b),
        };
        let mut p = Params::zero();
        p.a = [x.t.c, x.off, conv.cin, x.t.h];
        p.b = [x.t.w, out.t.c, out.off, conv.cout];
        p.c = [conv.k, conv.stride, conv.pad, conv.groups];
        p.d = [oh, ow, act_id, res.is_some() as u32];
        let (rv, beta) = res.map(|(v, b)| (Some(v), b)).unwrap_or((None, 0.0));
        p.e = [rv.map(|v| v.t.c).unwrap_or(0), rv.map(|v| v.off).unwrap_or(0), conv.dilation, 0];
        p.f = [slope, alpha, beta, 0.0];
        // Safety for wgpu: a buffer can't be bound both read-only and read-write in one dispatch.
        let res_buf = match rv {
            Some(v) if !std::ptr::eq(v.t, out.t) => v.t.buf(),
            Some(_) => panic!("residual and output must be different tensors"),
            None => &self.dummy,
        };
        let (wb, bb) = (&conv.weight, &conv.bias);
        let (xb, ob) = (x.t.buf(), out.t.buf());
        // Clone the handles so `dispatch` can borrow `self` mutably.
        let (xb, wb, bb, ob, rb, sb) = (xb.clone(), wb.clone(), bb.clone(), ob.clone(), res_buf.clone(), slopes.clone());
        // A large convolution is split into bands of rows (multiples of 8, the kernels' tile
        // height) so no single dispatch runs long enough to trip the driver watchdog.
        let macs_per_row = ow as u64 * conv.cout as u64 * (conv.cin / conv.groups) as u64 * (conv.k * conv.k) as u64;
        let budget = self.gpu.submit_budget();
        let rows = if macs_per_row * oh as u64 <= budget { oh } else { ((budget / macs_per_row.max(1)) as u32 / 8 * 8).max(8) };
        let mut row0 = 0;
        while row0 < oh {
            let band = rows.min(oh - row0);
            p.e[3] = row0;
            let cost = macs_per_row * band as u64;
            if conv.fast {
                self.dispatch_cost("conv3x3", p, &[&xb, &wb, &bb, &ob, &rb, &sb], (ow.div_ceil(16), band.div_ceil(8), conv.cout.div_ceil(FAST_OCB)), cost);
            } else {
                self.dispatch_cost("conv", p, &[&xb, &wb, &bb, &ob, &rb, &sb], (ow.div_ceil(8), band.div_ceil(8), conv.cout), cost);
            }
            row0 += band;
        }
    }

    /// Convolution into a new tensor.
    pub fn conv(&mut self, conv: &GConv, x: View<'_>, act: GAct<'_>) -> Result<GTensor, String> {
        let (oh, ow) = conv.out_size(x.t.h, x.t.w);
        let out = GTensor::new(&self.gpu, conv.cout, oh, ow)?;
        self.conv_into(conv, x, act, out.all(), None, 1.0);
        Ok(out)
    }

    /// Convolution plus a scaled residual into a new tensor: `conv(x) * alpha + res * beta`.
    pub fn conv_res(&mut self, conv: &GConv, x: View<'_>, act: GAct<'_>, res: View<'_>, alpha: f32, beta: f32) -> Result<GTensor, String> {
        let (oh, ow) = conv.out_size(x.t.h, x.t.w);
        let out = GTensor::new(&self.gpu, conv.cout, oh, ow)?;
        self.conv_into(conv, x, act, out.all(), Some((res, beta)), alpha);
        Ok(out)
    }

    fn shape_op(&mut self, kernel: &str, x: View<'_>, out: &GTensor, r: u32) {
        let mut p = Params::zero();
        p.a = [out.c, out.h, out.w, r];
        p.b = [x.t.c, x.off, out.c, 0];
        p.c = [x.t.h, x.t.w, 0, 0];
        let (xb, ob) = (x.t.buf().clone(), out.buf().clone());
        self.dispatch(kernel, p, &[&xb, &ob], (out.w.div_ceil(16), out.h.div_ceil(16), out.c));
    }

    /// (C·r², H, W) → (C, H·r, W·r).
    pub fn pixel_shuffle(&mut self, x: View<'_>, r: u32) -> Result<GTensor, String> {
        let out = GTensor::new(&self.gpu, x.len / (r * r), x.t.h * r, x.t.w * r)?;
        self.shape_op("pixel_shuffle", x, &out, r);
        Ok(out)
    }

    /// (C, H, W) → (C·r², H/r, W/r).
    pub fn pixel_unshuffle(&mut self, x: View<'_>, r: u32) -> Result<GTensor, String> {
        let out = GTensor::new(&self.gpu, x.len * r * r, x.t.h / r, x.t.w / r)?;
        self.shape_op("pixel_unshuffle", x, &out, r);
        Ok(out)
    }

    pub fn upsample_nearest(&mut self, x: View<'_>, r: u32) -> Result<GTensor, String> {
        let out = GTensor::new(&self.gpu, x.len, x.t.h * r, x.t.w * r)?;
        self.shape_op("upsample", x, &out, r);
        Ok(out)
    }

    /// `out[ch] = out[ch] * beta + a[ch] * alpha` in place (beta = 0 copies).
    pub fn axpy_into(&mut self, a: View<'_>, alpha: f32, out: View<'_>, beta: f32) {
        assert_eq!(a.len, out.len);
        assert!(!std::ptr::eq(a.t, out.t));
        let mut p = Params::zero();
        p.a = [out.len, out.t.h, out.t.w, 0];
        p.b = [a.t.c, a.off, out.t.c, out.off];
        p.f = [alpha, beta, 0.0, 0.0];
        let (ab, ob) = (a.t.buf().clone(), out.t.buf().clone());
        self.dispatch("axpy", p, &[&ab, &ob], (out.t.w.div_ceil(16), out.t.h.div_ceil(16), out.len));
    }

    /// `a + b` into a new tensor.
    pub fn add(&mut self, a: View<'_>, b: View<'_>) -> Result<GTensor, String> {
        let out = GTensor::new(&self.gpu, a.len, a.t.h, a.t.w)?;
        self.axpy_into(a, 1.0, out.all(), 0.0);
        self.axpy_into(b, 1.0, out.all(), 1.0);
        Ok(out)
    }

    /// Channel LayerNorm (NAFNet `LayerNorm2d`).
    pub fn layer_norm(&mut self, x: View<'_>, weight: &wgpu::Buffer, bias: &wgpu::Buffer, eps: f32) -> Result<GTensor, String> {
        let out = GTensor::new(&self.gpu, x.len, x.t.h, x.t.w)?;
        let mut p = Params::zero();
        p.a = [x.len, x.t.h, x.t.w, 0];
        p.b = [x.t.c, x.off, 0, 0];
        p.f = [eps, 0.0, 0.0, 0.0];
        let (xb, wb, bb, ob) = (x.t.buf().clone(), weight.clone(), bias.clone(), out.buf().clone());
        self.dispatch("layer_norm", p, &[&xb, &wb, &bb, &ob], (x.t.w.div_ceil(16), x.t.h.div_ceil(16), 1));
        Ok(out)
    }

    /// NAFNet `SimpleGate`: first half × second half of the channels.
    pub fn gate(&mut self, x: View<'_>) -> Result<GTensor, String> {
        let half = x.len / 2;
        let out = GTensor::new(&self.gpu, half, x.t.h, x.t.w)?;
        let mut p = Params::zero();
        p.a = [half, x.t.h, x.t.w, 0];
        p.b = [x.t.c, x.off, 0, 0];
        let (xb, ob) = (x.t.buf().clone(), out.buf().clone());
        self.dispatch("gate", p, &[&xb, &ob], (x.t.w.div_ceil(16), x.t.h.div_ceil(16), half));
        Ok(out)
    }

    /// Mean of each channel → (C, 1, 1).
    pub fn avg_pool(&mut self, x: View<'_>) -> Result<GTensor, String> {
        let out = GTensor::new(&self.gpu, x.len, 1, 1)?;
        let mut p = Params::zero();
        p.a = [x.len, x.t.h, x.t.w, 0];
        p.b = [x.t.c, x.off, 0, 0];
        let (xb, ob) = (x.t.buf().clone(), out.buf().clone());
        self.dispatch("avg_pool", p, &[&xb, &ob], (x.len, 1, 1));
        Ok(out)
    }

    /// `a[c] * s[c] (+ base)`: per-channel scale from a (C,1,1) tensor or a constant buffer.
    pub fn scale_channels(&mut self, a: View<'_>, s: &wgpu::Buffer, base: Option<View<'_>>) -> Result<GTensor, String> {
        let out = GTensor::new(&self.gpu, a.len, a.t.h, a.t.w)?;
        let mut p = Params::zero();
        p.a = [a.len, a.t.h, a.t.w, base.is_some() as u32];
        p.b = [a.t.c, a.off, base.map(|b| b.t.c).unwrap_or(0), base.map(|b| b.off).unwrap_or(0)];
        let bb = base.map(|b| b.t.buf().clone()).unwrap_or_else(|| self.dummy.clone());
        let (ab, sb, ob) = (a.t.buf().clone(), s.clone(), out.buf().clone());
        self.dispatch("scale_channels", p, &[&ab, &sb, &bb, &ob], (a.t.w.div_ceil(16), a.t.h.div_ceil(16), a.len));
        Ok(out)
    }

    /// The buffer backing a tensor, for use as a per-channel scale.
    pub fn buffer_of<'b>(&self, t: &'b GTensor) -> &'b wgpu::Buffer {
        t.buf()
    }

    /// 2×2 stride-2 transposed convolution (DRUNet up-sampling). `weight` is [cin][cout][2][2].
    pub fn conv_transpose2x2(&mut self, x: View<'_>, weight: &wgpu::Buffer, cout: u32) -> Result<GTensor, String> {
        let out = GTensor::new(&self.gpu, cout, x.t.h * 2, x.t.w * 2)?;
        let mut p = Params::zero();
        p.a = [cout, out.h, out.w, x.len];
        p.b = [x.t.c, x.off, x.t.h, x.t.w];
        let (xb, wb, ob) = (x.t.buf().clone(), weight.clone(), out.buf().clone());
        self.dispatch("conv_t2x2", p, &[&xb, &wb, &ob], (out.w.div_ceil(16), out.h.div_ceil(16), cout));
        Ok(out)
    }

    /// General transposed convolution (RIFE up-sampling). `weight` is [cin][cout][k][k], `bias` is [cout].
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose(&mut self, x: View<'_>, weight: &wgpu::Buffer, bias: &wgpu::Buffer, cout: u32, k: u32, stride: u32, pad: u32) -> Result<GTensor, String> {
        let (oh, ow) = ((x.t.h - 1) * stride + k - 2 * pad, (x.t.w - 1) * stride + k - 2 * pad);
        let out = GTensor::new(&self.gpu, cout, oh, ow)?;
        let mut p = Params::zero();
        p.a = [cout, oh, ow, x.len];
        p.b = [x.t.c, x.off, x.t.h, x.t.w];
        p.c = [k, stride, pad, 0];
        let (xb, wb, bb, ob) = (x.t.buf().clone(), weight.clone(), bias.clone(), out.buf().clone());
        self.dispatch("conv_t", p, &[&xb, &wb, &bb, &ob], (ow.div_ceil(16), oh.div_ceil(16), cout));
        Ok(out)
    }

    /// Bilinear resize by `factor` (`align_corners=False`, output `floor(size * factor)`), times `mul`.
    pub fn resize_bilinear_factor(&mut self, x: View<'_>, factor: f64, mul: f32) -> Result<GTensor, String> {
        let (oh, ow) = (((x.t.h as f64) * factor).floor() as u32, ((x.t.w as f64) * factor).floor() as u32);
        let out = GTensor::new(&self.gpu, x.len, oh, ow)?;
        let mut p = Params::zero();
        p.a = [x.len, oh, ow, 0];
        p.b = [x.t.c, x.off, x.t.h, x.t.w];
        p.f = [(1.0 / factor) as f32, (1.0 / factor) as f32, mul, 0.0];
        let (xb, ob) = (x.t.buf().clone(), out.buf().clone());
        self.dispatch("resize_bl", p, &[&xb, &ob], (ow.div_ceil(16), oh.div_ceil(16), x.len));
        Ok(out)
    }

    /// Backward warp: `out(y, x) = img(y + flow_y, x + flow_x)`, bilinear, coordinates clamped to the frame.
    pub fn warp(&mut self, img: View<'_>, flow: View<'_>) -> Result<GTensor, String> {
        assert_eq!(flow.len, 2);
        assert!(img.t.h == flow.t.h && img.t.w == flow.t.w, "warp size");
        let out = GTensor::new(&self.gpu, img.len, img.t.h, img.t.w)?;
        let mut p = Params::zero();
        p.a = [img.len, img.t.h, img.t.w, 0];
        p.b = [img.t.c, img.off, flow.t.c, flow.off];
        let (ib, fb, ob) = (img.t.buf().clone(), flow.t.buf().clone(), out.buf().clone());
        self.dispatch("warp", p, &[&ib, &fb, &ob], (img.t.w.div_ceil(16), img.t.h.div_ceil(16), img.len));
        Ok(out)
    }

    /// `leaky(a[c] * s[c] + base[c])` (RIFE's residual blocks).
    pub fn scale_add_leaky(&mut self, a: View<'_>, s: &wgpu::Buffer, base: View<'_>, slope: f32) -> Result<GTensor, String> {
        assert_eq!(a.len, base.len);
        let out = GTensor::new(&self.gpu, a.len, a.t.h, a.t.w)?;
        let mut p = Params::zero();
        p.a = [a.len, a.t.h, a.t.w, 0];
        p.b = [a.t.c, a.off, base.t.c, base.off];
        p.f = [slope, 0.0, 0.0, 0.0];
        let (ab, sb, bb, ob) = (a.t.buf().clone(), s.clone(), base.t.buf().clone(), out.buf().clone());
        self.dispatch("scale_add_leaky", p, &[&ab, &sb, &bb, &ob], (a.t.w.div_ceil(16), a.t.h.div_ceil(16), a.len));
        Ok(out)
    }

    /// `clamp(a * sigmoid(m) + b * (1 - sigmoid(m)), 0, 1)` with a one-channel mask view.
    pub fn sigmoid_blend(&mut self, a: View<'_>, b: View<'_>, m: View<'_>) -> Result<GTensor, String> {
        assert_eq!(a.len, b.len);
        assert_eq!(m.len, 1);
        let out = GTensor::new(&self.gpu, a.len, a.t.h, a.t.w)?;
        let mut p = Params::zero();
        p.a = [a.len, a.t.h, a.t.w, 0];
        p.b = [a.t.c, a.off, b.t.c, b.off];
        p.c = [m.t.c, m.off, 0, 0];
        let (ab, bb, mb, ob) = (a.t.buf().clone(), b.t.buf().clone(), m.t.buf().clone(), out.buf().clone());
        self.dispatch("sigmoid_blend", p, &[&ab, &bb, &mb, &ob], (a.t.w.div_ceil(16), a.t.h.div_ceil(16), a.len));
        Ok(out)
    }

    /// StyleGAN output stage: `act(x[c] * d[c] * gain + noise[pix] * nw + bias[c]) + skip[c]`.
    /// `noise` is a one-channel (H, W) buffer; `act` is the leaky slope (None = linear).
    #[allow(clippy::too_many_arguments)]
    pub fn style_out(
        &mut self,
        x: View<'_>,
        d: &wgpu::Buffer,
        gain: f32,
        noise: Option<(&wgpu::Buffer, f32)>,
        bias: &wgpu::Buffer,
        act: Option<f32>,
        skip: Option<View<'_>>,
    ) -> Result<GTensor, String> {
        let out = GTensor::new(&self.gpu, x.len, x.t.h, x.t.w)?;
        let mut p = Params::zero();
        p.a = [x.len, x.t.h, x.t.w, noise.is_some() as u32];
        p.b = [x.t.c, x.off, skip.map(|s| s.t.c).unwrap_or(0), skip.map(|s| s.off).unwrap_or(0)];
        p.c = [act.is_some() as u32, skip.is_some() as u32, 0, 0];
        p.f = [gain, noise.map(|n| n.1).unwrap_or(0.0), act.unwrap_or(1.0), 0.0];
        let nb = noise.map(|n| n.0.clone()).unwrap_or_else(|| self.dummy.clone());
        let sb = skip.map(|s| s.t.buf().clone()).unwrap_or_else(|| self.dummy.clone());
        let (xb, db, bb, ob) = (x.t.buf().clone(), d.clone(), bias.clone(), out.buf().clone());
        self.dispatch("style_out", p, &[&xb, &db, &nb, &bb, &sb, &ob], (x.t.w.div_ceil(16), x.t.h.div_ceil(16), x.len));
        Ok(out)
    }

    /// GFPGAN spatial feature transform: keep the first half of `x`'s channels, and replace the second half
    /// with `x * scale + shift` (scale/shift are full-resolution tensors with half the channels).
    pub fn sft(&mut self, x: View<'_>, scale: View<'_>, shift: View<'_>) -> Result<GTensor, String> {
        let half = x.len / 2;
        assert!(scale.len == half && shift.len == half);
        let out = GTensor::new(&self.gpu, x.len, x.t.h, x.t.w)?;
        let mut p = Params::zero();
        p.a = [x.len, x.t.h, x.t.w, half];
        p.b = [x.t.c, x.off, scale.t.c, scale.off];
        p.c = [shift.t.c, shift.off, 0, 0];
        let (xb, sb, hb, ob) = (x.t.buf().clone(), scale.t.buf().clone(), shift.t.buf().clone(), out.buf().clone());
        self.dispatch("sft", p, &[&xb, &sb, &hb, &ob], (x.t.w.div_ceil(16), x.t.h.div_ceil(16), x.len));
        Ok(out)
    }

    /// 3×3 / stride 2 / padding 1 max pool with zero padding (inputs are post-ReLU, so zero equals -inf padding).
    pub fn max_pool3s2(&mut self, x: View<'_>) -> Result<GTensor, String> {
        let (oh, ow) = ((x.t.h - 1) / 2 + 1, (x.t.w - 1) / 2 + 1);
        let out = GTensor::new(&self.gpu, x.len, oh, ow)?;
        let mut p = Params::zero();
        p.a = [x.len, oh, ow, 0];
        p.b = [x.t.c, x.off, x.t.h, x.t.w];
        let (xb, ob) = (x.t.buf().clone(), out.buf().clone());
        self.dispatch("max_pool3s2", p, &[&xb, &ob], (ow.div_ceil(16), oh.div_ceil(16), x.len));
        Ok(out)
    }

    /// Nearest-neighbour resize to (oh, ow) with source index `i * in / out` (matches ops::resize_nearest).
    pub fn resize_nearest(&mut self, x: View<'_>, oh: u32, ow: u32) -> Result<GTensor, String> {
        let out = GTensor::new(&self.gpu, x.len, oh, ow)?;
        let mut p = Params::zero();
        p.a = [x.len, oh, ow, 0];
        p.b = [x.t.c, x.off, x.t.h, x.t.w];
        let (xb, ob) = (x.t.buf().clone(), out.buf().clone());
        self.dispatch("resize_nearest", p, &[&xb, &ob], (ow.div_ceil(16), oh.div_ceil(16), x.len));
        Ok(out)
    }

    /// `act(a + b)` (or `act(a)` without `b`); `act` is None, ReLU or leaky.
    pub fn add_act(&mut self, a: View<'_>, b: Option<View<'_>>, act: GAct<'_>) -> Result<GTensor, String> {
        if let Some(b) = b {
            assert_eq!(a.len, b.len);
        }
        let out = GTensor::new(&self.gpu, a.len, a.t.h, a.t.w)?;
        let (id, slope) = match act {
            GAct::None => (0, 0.0),
            GAct::Relu => (1, 0.0),
            GAct::Leaky(s) => (2, s),
            GAct::Prelu(_) => return Err("add_act does not support PReLU".into()),
        };
        let mut p = Params::zero();
        p.a = [a.len, a.t.h, a.t.w, b.is_some() as u32];
        p.b = [a.t.c, a.off, b.map(|v| v.t.c).unwrap_or(0), b.map(|v| v.off).unwrap_or(0)];
        p.c = [id, 0, 0, 0];
        p.f = [slope, 0.0, 0.0, 0.0];
        let bb = b.map(|v| v.t.buf().clone()).unwrap_or_else(|| self.dummy.clone());
        let (ab, ob) = (a.t.buf().clone(), out.buf().clone());
        self.dispatch("add_act", p, &[&ab, &bb, &ob], (a.t.w.div_ceil(16), a.t.h.div_ceil(16), a.len));
        Ok(out)
    }

    /// Finish recording, run everything and read back several tensors (planar data, in order).
    pub fn finish_read_many(mut self, ts: &[&GTensor]) -> Result<Vec<Vec<f32>>, String> {
        drop(self.pass.take());
        if let Some(e) = self.failed.take() {
            return Err(e);
        }
        let stagings: Vec<(wgpu::Buffer, u64)> = ts
            .iter()
            .map(|t| {
                let bytes = t.c as u64 * t.hw() as u64 * 4;
                let st = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.encoder.copy_buffer_to_buffer(t.buf(), 0, &st, 0, bytes);
                (st, bytes)
            })
            .collect();
        let idx = self.gpu.queue.submit([self.encoder.finish()]);
        for (scope, what) in self.scopes.drain(..).rev() {
            if let Some(e) = pollster::block_on(scope.pop()) {
                return Err(format!("{what}: {e}"));
            }
        }
        let mut rxs = Vec::new();
        for (st, _) in &stagings {
            let (tx, rx) = std::sync::mpsc::channel();
            st.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rxs.push(rx);
        }
        let started = Instant::now();
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: Some(SUBMIT_TIMEOUT) }).map_err(|e| self.gpu.describe_failure(&e.to_string()))?;
        self.gpu.record_speed(self.pending, started.elapsed().as_secs_f64());
        let mut out = Vec::new();
        for ((st, _), rx) in stagings.iter().zip(rxs) {
            rx.recv().map_err(|_| "GPU readback was cancelled".to_string())?.map_err(|e| format!("GPU readback failed: {e}"))?;
            let data = {
                let view = st.slice(..).get_mapped_range().map_err(|e| format!("GPU readback failed: {e:?}"))?;
                bytemuck::cast_slice::<u8, f32>(&view).to_vec()
            };
            st.unmap();
            out.push(data);
        }
        Ok(out)
    }

    /// Finish recording, run everything and read back `t` as planar data.
    pub fn finish_read(mut self, t: &GTensor) -> Result<Vec<f32>, String> {
        drop(self.pass.take());
        if let Some(e) = self.failed.take() {
            return Err(e);
        }
        let bytes = t.c as u64 * t.hw() as u64 * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.encoder.copy_buffer_to_buffer(t.buf(), 0, &staging, 0, bytes);
        let idx = self.gpu.queue.submit([self.encoder.finish()]);
        for (scope, what) in self.scopes.drain(..).rev() {
            if let Some(e) = pollster::block_on(scope.pop()) {
                // "out of memory" in the message lets the caller retry with smaller tiles.
                return Err(format!("{what}: {e}"));
            }
        }
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let started = Instant::now();
        self.gpu.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: Some(SUBMIT_TIMEOUT) }).map_err(|e| self.gpu.describe_failure(&e.to_string()))?;
        self.gpu.record_speed(self.pending, started.elapsed().as_secs_f64());
        rx.recv().map_err(|_| "GPU readback was cancelled".to_string())?.map_err(|e| format!("GPU readback failed: {e}"))?;
        let data = {
            let view = slice.get_mapped_range().map_err(|e| format!("GPU readback failed: {e:?}"))?;
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        staging.unmap();
        Ok(data)
    }
}

const PRELUDE: &str = r#"
struct P { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32>, e: vec4<u32>, f: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;
"#;

/// Kernels. Bindings 1.. are declared per kernel in the order the Rust side passes them.
const KERNELS: &[(&str, &str)] = &[
    (
        "conv",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;
@group(0) @binding(6) var<storage, read> slopes: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let in_off = p.a.y; let cin = p.a.z; let H = p.a.w; let W = p.b.x;
  let out_off = p.b.z; let cout = p.b.w;
  let k = p.c.x; let stride = p.c.y; let pad = p.c.z; let groups = p.c.w;
  let OH = p.d.x; let OW = p.d.y; let dil = p.e.z;
  let ox = g.x; let oy = g.y + p.e.w; let oc = g.z;
  if (ox >= OW || oy >= OH || oc >= cout) { return; }
  let cin_g = cin / groups; let cout_g = cout / groups;
  let grp = oc / cout_g;
  var acc = bias[oc];
  for (var ic = 0u; ic < cin_g; ic++) {
    let c = in_off + grp * cin_g + ic;
    for (var ky = 0u; ky < k; ky++) {
      let iy = i32(oy * stride + ky * dil) - i32(pad);
      if (iy < 0 || iy >= i32(H)) { continue; }
      for (var kx = 0u; kx < k; kx++) {
        let ix = i32(ox * stride + kx * dil) - i32(pad);
        if (ix < 0 || ix >= i32(W)) { continue; }
        acc += inp[c * H * W + u32(iy) * W + u32(ix)] * wt[((oc * cin_g + ic) * k + ky) * k + kx];
      }
    }
  }
  var v = acc;
  switch p.d.z {
    case 1u: { v = max(v, 0.0); }
    case 2u: { v = select(v * p.f.x, v, v >= 0.0); }
    case 3u: { v = select(v * slopes[oc], v, v >= 0.0); }
    default: {}
  }
  v = v * p.f.y;
  let pix = oy * OW + ox;
  if (p.d.w == 1u) { v += res[(p.e.y + oc) * OH * OW + pix] * p.f.z; }
  outp[(out_off + oc) * OH * OW + pix] = v;
}
"#,
    ),
    (
        "conv3x3",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;
@group(0) @binding(6) var<storage, read> slopes: array<f32>;

const TW: u32 = 16u; const TH: u32 = 8u; const ICB: u32 = 8u;
const SW: u32 = 18u; const SH: u32 = 10u;
var<workgroup> tile: array<f32, 1440>; // ICB * SH * SW

@compute @workgroup_size(16, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let in_off = p.a.y; let cin = p.a.z; let H = p.a.w; let W = p.b.x;
  let out_off = p.b.z; let cout = p.b.w;
  let ox0 = wg.x * TW; let oy0 = wg.y * TH + p.e.w;
  let ox = ox0 + l.x; let oy = oy0 + l.y;
  let ocb = wg.z;
  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0);
  for (var ic = 0u; ic < cin; ic += ICB) {
    for (var i = li; i < ICB * SH * SW; i += TW * TH) {
      let ci = i / (SH * SW); let r = i % (SH * SW);
      let gy = i32(oy0 + r / SW) - 1; let gx = i32(ox0 + r % SW) - 1;
      var v = 0.0;
      if (ic + ci < cin && gy >= 0 && gy < i32(H) && gx >= 0 && gx < i32(W)) {
        v = inp[(in_off + ic + ci) * H * W + u32(gy) * W + u32(gx)];
      }
      tile[i] = v;
    }
    workgroupBarrier();
    let n = min(ICB, cin - ic);
    for (var ci = 0u; ci < n; ci++) {
      let base = ((ocb * cin + ic + ci) * 9u) * 2u;
      for (var t = 0u; t < 9u; t++) {
        let v = tile[ci * SH * SW + (l.y + t / 3u) * SW + l.x + t % 3u];
        a0 += v * wt[base + t * 2u];
        a1 += v * wt[base + t * 2u + 1u];
      }
    }
    workgroupBarrier();
  }
  if (ox >= W || oy >= H) { return; }
  let pix = oy * W + ox;
  for (var j = 0u; j < 8u; j++) {
    let oc = ocb * 8u + j;
    if (oc >= cout) { break; }
    var v: f32;
    if (j < 4u) { v = a0[j]; } else { v = a1[j - 4u]; }
    v += bias[oc];
    switch p.d.z {
      case 1u: { v = max(v, 0.0); }
      case 2u: { v = select(v * p.f.x, v, v >= 0.0); }
      case 3u: { v = select(v * slopes[oc], v, v >= 0.0); }
      default: {}
    }
    v = v * p.f.y;
    if (p.d.w == 1u) { v += res[(p.e.y + oc) * H * W + pix] * p.f.z; }
    outp[(out_off + oc) * H * W + pix] = v;
  }
}
"#,
    ),
    (
        "pixel_shuffle",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let OH = p.a.y; let OW = p.a.z; let r = p.a.w; let in_off = p.b.y; let IH = p.c.x; let IW = p.c.y;
  let x = g.x; let y = g.y; let c = g.z;
  if (x >= OW || y >= OH || c >= p.a.x) { return; }
  let ic = in_off + c * r * r + (y % r) * r + (x % r);
  outp[(c * OH + y) * OW + x] = inp[(ic * IH + y / r) * IW + x / r];
}
"#,
    ),
    (
        "pixel_unshuffle",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let OH = p.a.y; let OW = p.a.z; let r = p.a.w; let in_off = p.b.y; let IH = p.c.x; let IW = p.c.y;
  let x = g.x; let y = g.y; let c = g.z;
  if (x >= OW || y >= OH || c >= p.a.x) { return; }
  let ic = c / (r * r); let rem = c % (r * r);
  let iy = y * r + rem / r; let ix = x * r + rem % r;
  outp[(c * OH + y) * OW + x] = inp[((in_off + ic) * IH + iy) * IW + ix];
}
"#,
    ),
    (
        "upsample",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let OH = p.a.y; let OW = p.a.z; let r = p.a.w; let in_off = p.b.y; let IH = p.c.x; let IW = p.c.y;
  let x = g.x; let y = g.y; let c = g.z;
  if (x >= OW || y >= OH || c >= p.a.x) { return; }
  outp[(c * OH + y) * OW + x] = inp[((in_off + c) * IH + y / r) * IW + x / r];
}
"#,
    ),
    (
        "axpy",
        r#"
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let H = p.a.y; let W = p.a.z;
  let x = g.x; let y = g.y; let c = g.z;
  if (x >= W || y >= H || c >= C) { return; }
  let pix = y * W + x;
  let o = (p.b.w + c) * H * W + pix;
  var v = a[(p.b.y + c) * H * W + pix] * p.f.x;
  if (p.f.y != 0.0) { v += outp[o] * p.f.y; }
  outp[o] = v;
}
"#,
    ),
    (
        "layer_norm",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let H = p.a.y; let W = p.a.z; let off = p.b.y;
  if (g.x >= W || g.y >= H) { return; }
  let pix = g.y * W + g.x; let hw = H * W;
  var mean = 0.0;
  for (var c = 0u; c < C; c++) { mean += inp[(off + c) * hw + pix]; }
  mean /= f32(C);
  var var_ = 0.0;
  for (var c = 0u; c < C; c++) { let d = inp[(off + c) * hw + pix] - mean; var_ += d * d; }
  let inv = 1.0 / sqrt(var_ / f32(C) + p.f.x);
  for (var c = 0u; c < C; c++) {
    outp[c * hw + pix] = (inp[(off + c) * hw + pix] - mean) * inv * w[c] + b[c];
  }
}
"#,
    ),
    (
        "gate",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let H = p.a.y; let W = p.a.z; let off = p.b.y;
  if (g.x >= W || g.y >= H || g.z >= C) { return; }
  let hw = H * W; let pix = g.y * W + g.x;
  outp[g.z * hw + pix] = inp[(off + g.z) * hw + pix] * inp[(off + g.z + C) * hw + pix];
}
"#,
    ),
    (
        "avg_pool",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let hw = p.a.y * p.a.z; let c = wg.x;
  var s = 0.0;
  for (var i = li; i < hw; i += 256u) { s += inp[(p.b.y + c) * hw + i]; }
  part[li] = s;
  workgroupBarrier();
  for (var step = 128u; step > 0u; step = step / 2u) {
    if (li < step) { part[li] += part[li + step]; }
    workgroupBarrier();
  }
  if (li == 0u) { outp[c] = part[0] / f32(hw); }
}
"#,
    ),
    (
        "scale_channels",
        r#"
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> s: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let H = p.a.y; let W = p.a.z;
  if (g.x >= W || g.y >= H || g.z >= C) { return; }
  let hw = H * W; let pix = g.y * W + g.x; let c = g.z;
  var v = a[(p.b.y + c) * hw + pix] * s[c];
  if (p.a.w == 1u) { v += base[(p.b.w + c) * hw + pix]; }
  outp[c * hw + pix] = v;
}
"#,
    ),
    (
        "conv_t2x2",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let cout = p.a.x; let OH = p.a.y; let OW = p.a.z; let cin = p.a.w;
  let off = p.b.y; let IH = p.b.z; let IW = p.b.w;
  let x = g.x; let y = g.y; let oc = g.z;
  if (x >= OW || y >= OH || oc >= cout) { return; }
  let iy = y / 2u; let ix = x / 2u; let t = (y % 2u) * 2u + (x % 2u);
  var acc = 0.0;
  for (var ic = 0u; ic < cin; ic++) {
    acc += inp[((off + ic) * IH + iy) * IW + ix] * wt[(ic * cout + oc) * 4u + t];
  }
  outp[(oc * OH + y) * OW + x] = acc;
}
"#,
    ),
    (
        "conv_t",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let cout = p.a.x; let OH = p.a.y; let OW = p.a.z; let cin = p.a.w;
  let off = p.b.y; let H = i32(p.b.z); let W = i32(p.b.w);
  let k = i32(p.c.x); let s = i32(p.c.y); let pad = i32(p.c.z);
  if (g.x >= OW || g.y >= OH || g.z >= cout) { return; }
  let ox = i32(g.x); let oy = i32(g.y); let oc = g.z;
  var acc = bias[oc];
  for (var ky = 0; ky < k; ky++) {
    let ty = oy + pad - ky;
    if (ty < 0 || ty % s != 0) { continue; }
    let iy = ty / s;
    if (iy >= H) { continue; }
    for (var kx = 0; kx < k; kx++) {
      let tx = ox + pad - kx;
      if (tx < 0 || tx % s != 0) { continue; }
      let ix = tx / s;
      if (ix >= W) { continue; }
      let pix = u32(iy * W + ix); let tap = u32(ky * k + kx);
      for (var ic = 0u; ic < cin; ic++) {
        acc += inp[(off + ic) * u32(H * W) + pix] * wt[((ic * cout + oc) * u32(k * k)) + tap];
      }
    }
  }
  outp[(oc * OH + g.y) * OW + g.x] = acc;
}
"#,
    ),
    (
        "resize_bl",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let OH = p.a.y; let OW = p.a.z;
  let off = p.b.y; let H = p.b.z; let W = p.b.w;
  if (g.x >= OW || g.y >= OH || g.z >= C) { return; }
  let sy = max((f32(g.y) + 0.5) * p.f.x - 0.5, 0.0);
  let sx = max((f32(g.x) + 0.5) * p.f.y - 0.5, 0.0);
  let y0 = min(u32(floor(sy)), H - 1u); let y1 = min(y0 + 1u, H - 1u);
  let x0 = min(u32(floor(sx)), W - 1u); let x1 = min(x0 + 1u, W - 1u);
  var wy = sy - f32(y0); if (y1 == y0) { wy = 0.0; }
  var wx = sx - f32(x0); if (x1 == x0) { wx = 0.0; }
  let base = (off + g.z) * H * W;
  let a = inp[base + y0 * W + x0] * (1.0 - wy) + inp[base + y1 * W + x0] * wy;
  let b = inp[base + y0 * W + x1] * (1.0 - wy) + inp[base + y1 * W + x1] * wy;
  outp[(g.z * OH + g.y) * OW + g.x] = (a * (1.0 - wx) + b * wx) * p.f.z;
}
"#,
    ),
    (
        "warp",
        r#"
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read> flow: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let H = p.a.y; let W = p.a.z; let ioff = p.b.y; let foff = p.b.w;
  if (g.x >= W || g.y >= H || g.z >= C) { return; }
  let hw = H * W; let pix = g.y * W + g.x;
  let sx = clamp(f32(g.x) + flow[foff * hw + pix], 0.0, f32(W - 1u));
  let sy = clamp(f32(g.y) + flow[(foff + 1u) * hw + pix], 0.0, f32(H - 1u));
  let x0 = floor(sx); let y0 = floor(sy); let wx = sx - x0; let wy = sy - y0;
  let x1 = min(x0 + 1.0, f32(W - 1u)); let y1 = min(y0 + 1.0, f32(H - 1u));
  let base = (ioff + g.z) * hw;
  let v00 = img[base + u32(y0) * W + u32(x0)]; let v01 = img[base + u32(y0) * W + u32(x1)];
  let v10 = img[base + u32(y1) * W + u32(x0)]; let v11 = img[base + u32(y1) * W + u32(x1)];
  let top = v00 * (1.0 - wx) + v01 * wx; let bot = v10 * (1.0 - wx) + v11 * wx;
  outp[g.z * hw + pix] = top * (1.0 - wy) + bot * wy;
}
"#,
    ),
    (
        "scale_add_leaky",
        r#"
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> s: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let H = p.a.y; let W = p.a.z;
  if (g.x >= W || g.y >= H || g.z >= C) { return; }
  let hw = H * W; let pix = g.y * W + g.x;
  let v = a[(p.b.y + g.z) * hw + pix] * s[g.z] + base[(p.b.w + g.z) * hw + pix];
  outp[g.z * hw + pix] = max(v, v * p.f.x);
}
"#,
    ),
    (
        "sigmoid_blend",
        r#"
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let H = p.a.y; let W = p.a.z;
  if (g.x >= W || g.y >= H || g.z >= C) { return; }
  let hw = H * W; let pix = g.y * W + g.x;
  let w = 1.0 / (1.0 + exp(-m[p.c.y * hw + pix]));
  let v = a[(p.b.y + g.z) * hw + pix] * w + b[(p.b.w + g.z) * hw + pix] * (1.0 - w);
  outp[g.z * hw + pix] = clamp(v, 0.0, 1.0);
}
"#,
    ),
    (
        "style_out",
        r#"
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> d: array<f32>;
@group(0) @binding(3) var<storage, read> noise: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> skip: array<f32>;
@group(0) @binding(6) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let H = p.a.y; let W = p.a.z;
  if (g.x >= W || g.y >= H || g.z >= C) { return; }
  let hw = H * W; let pix = g.y * W + g.x;
  var v = x[(p.b.y + g.z) * hw + pix] * d[g.z] * p.f.x + bias[g.z];
  if (p.a.w == 1u) { v += noise[pix] * p.f.y; }
  if (p.c.x == 1u) { v = max(v, v * p.f.z); }
  if (p.c.y == 1u) { v += skip[(p.b.w + g.z) * hw + pix]; }
  outp[g.z * hw + pix] = v;
}
"#,
    ),
    (
        "sft",
        r#"
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> sc: array<f32>;
@group(0) @binding(3) var<storage, read> sh: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let H = p.a.y; let W = p.a.z; let half = p.a.w;
  if (g.x >= W || g.y >= H || g.z >= C) { return; }
  let hw = H * W; let pix = g.y * W + g.x;
  let v = x[(p.b.y + g.z) * hw + pix];
  if (g.z < half) { outp[g.z * hw + pix] = v; return; }
  let j = g.z - half;
  outp[g.z * hw + pix] = v * sc[(p.b.w + j) * hw + pix] + sh[(p.c.y + j) * hw + pix];
}
"#,
    ),
    (
        "max_pool3s2",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let OH = p.a.y; let OW = p.a.z; let off = p.b.y; let H = i32(p.b.z); let W = i32(p.b.w);
  if (g.x >= OW || g.y >= OH || g.z >= C) { return; }
  var m = 0.0; // zero padding; inputs are post-ReLU
  let base = (off + g.z) * u32(H * W);
  for (var dy = -1; dy <= 1; dy++) {
    let y = i32(g.y) * 2 + dy;
    if (y < 0 || y >= H) { continue; }
    for (var dx = -1; dx <= 1; dx++) {
      let x = i32(g.x) * 2 + dx;
      if (x < 0 || x >= W) { continue; }
      m = max(m, inp[base + u32(y * W + x)]);
    }
  }
  outp[(g.z * OH + g.y) * OW + g.x] = m;
}
"#,
    ),
    (
        "resize_nearest",
        r#"
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let OH = p.a.y; let OW = p.a.z; let off = p.b.y; let H = p.b.z; let W = p.b.w;
  if (g.x >= OW || g.y >= OH || g.z >= C) { return; }
  let sy = min((g.y * H) / OH, H - 1u); let sx = min((g.x * W) / OW, W - 1u);
  outp[(g.z * OH + g.y) * OW + g.x] = inp[(off + g.z) * H * W + sy * W + sx];
}
"#,
    ),
    (
        "add_act",
        r#"
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let H = p.a.y; let W = p.a.z;
  if (g.x >= W || g.y >= H || g.z >= C) { return; }
  let hw = H * W; let pix = g.y * W + g.x;
  var v = a[(p.b.y + g.z) * hw + pix];
  if (p.a.w == 1u) { v += b[(p.b.w + g.z) * hw + pix]; }
  if (p.c.x == 1u) { v = max(v, 0.0); }
  if (p.c.x == 2u) { v = select(v * p.f.x, v, v >= 0.0); }
  outp[g.z * hw + pix] = v;
}
"#,
    ),
];

/// Result of [`self_check`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct SelfCheck {
    pub device: String,
    pub api: String,
    /// Largest |GPU − CPU| over the test output.
    pub max_abs_diff: f32,
    pub passed: bool,
}

/// Run a small convolution network on the GPU and on the CPU and compare them.
/// Used by the app's `--self-test` and diagnostics to prove the device computes correctly.
pub fn self_check(index: usize) -> Result<SelfCheck, String> {
    use candle_core::{Device, Tensor};
    let gpu = context(index)?;
    let (cin, cout, h, w) = (12usize, 20usize, 37usize, 29usize);
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rnd = |n: usize, s: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * s
            })
            .collect()
    };
    let x = rnd(cin * h * w, 1.0);
    // A fast-path 3x3 layer, then a generic stride-2 layer, as in the real networks.
    let (w1, b1) = (rnd(cout * cin * 9, 0.2), rnd(cout, 0.1));
    let (w2, b2) = (rnd(3 * cout * 4, 0.2), rnd(3, 0.1));
    let err = |e: candle_core::Error| e.to_string();
    let cpu = {
        let t = Tensor::from_slice(&x, (1, cin, h, w), &Device::Cpu).map_err(err)?;
        let c1 = crate::ops::Conv2d {
            weight: Tensor::from_slice(&w1, (cout, cin, 3, 3), &Device::Cpu).map_err(err)?,
            bias: Some(Tensor::new(b1.as_slice(), &Device::Cpu).map_err(err)?),
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        };
        let c2 = crate::ops::Conv2d {
            weight: Tensor::from_slice(&w2, (3, cout, 2, 2), &Device::Cpu).map_err(err)?,
            bias: Some(Tensor::new(b2.as_slice(), &Device::Cpu).map_err(err)?),
            stride: 2,
            padding: 0,
            dilation: 1,
            groups: 1,
        };
        let y = c1.forward_act(&t, crate::ops::Act::LeakyRelu(0.2)).map_err(err)?;
        c2.forward(&y).map_err(err)?.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(err)?
    };
    let g1 = GConv::new(&gpu, &w1, Some(&b1), cout as u32, cin as u32, 3, 1, 1, 1, 1);
    let g2 = GConv::new(&gpu, &w2, Some(&b2), 3, cout as u32, 2, 2, 0, 1, 1);
    let mut rec = Recorder::new(&gpu);
    let t = rec.upload(cin as u32, h as u32, w as u32, &x)?;
    let y = rec.conv(&g1, t.all(), GAct::Leaky(0.2))?;
    let z = rec.conv(&g2, y.all(), GAct::None)?;
    drop(y);
    let out = rec.finish_read(&z)?;
    let max = cpu.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    gpu.trim();
    Ok(SelfCheck { device: gpu.info.name.clone(), api: gpu.info.api.clone(), max_abs_diff: max, passed: max < 1e-3 && out.len() >= cpu.len() })
}
