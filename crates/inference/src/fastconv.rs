//! Fast CPU convolution for stride-1, ungrouped f32 convolutions, which is
//! nearly all of the work in the super-resolution, denoising and deblurring
//! networks.
//!
//! candle's CPU conv2d materialises an im2col matrix (Cin·kh·kw × H·W), which
//! for a 64-channel 3×3 layer at 640×360 is 530 MB per call and dominates the
//! run time. Here the input is zero-padded once, and every kernel tap becomes
//! one GEMM that reads the weights and a shifted window of the padded input
//! through strides: out[Cout, h·Wp] += W[:, :, i, j] · Xp[:, off(i,j) ..]. The
//! extra columns produced at the right border are dropped when copying out.

use candle_core::{DType, Device, Result, Storage, Tensor};
use gemm::Parallelism;

/// Whether `conv2d_s1` handles this convolution.
pub fn supported(x: &Tensor, w: &Tensor, stride: usize, groups: usize) -> bool {
    stride == 1 && groups == 1 && matches!(x.device(), Device::Cpu) && x.dtype() == DType::F32 && w.dtype() == DType::F32 && x.rank() == 4 && w.rank() == 4
}

fn slice(t: &Tensor) -> Result<Vec<f32>> {
    let (st, l) = t.storage_and_layout();
    match &*st {
        Storage::Cpu(cpu) if l.is_contiguous() => {
            let s = cpu.as_slice::<f32>()?;
            Ok(s[l.start_offset()..l.start_offset() + l.shape().elem_count()].to_vec())
        }
        _ => {
            drop(st);
            t.flatten_all()?.to_vec1::<f32>()
        }
    }
}

/// Activation fused into the convolution's output pass.
#[derive(Clone, Copy, Debug)]
pub enum Act<'a> {
    None,
    Relu,
    LeakyRelu(f32),
    /// Per-output-channel slopes.
    Prelu(&'a [f32]),
}

impl Act<'_> {
    #[inline(always)]
    fn apply(&self, co: usize, v: f32) -> f32 {
        match self {
            Act::None => v,
            Act::Relu => v.max(0.0),
            Act::LeakyRelu(a) => {
                if v >= 0.0 {
                    v
                } else {
                    v * a
                }
            }
            Act::Prelu(a) => {
                if v >= 0.0 {
                    v
                } else {
                    v * a[co]
                }
            }
        }
    }
}

struct SyncPtr(*mut f32);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

/// Output columns per work item: the accumulator (Cout × CHUNK) stays in L2 across all taps.
const CHUNK: usize = 2048;

/// Stride-1 convolution of `x` [N, Cin, H, W] with `w` [Cout, Cin, kh, kw] (no bias).
pub fn conv2d_s1(x: &Tensor, w: &Tensor, padding: usize, dilation: usize) -> Result<Tensor> {
    conv2d_s1_fused(x, w, None, Act::None, padding, dilation)
}

/// Stride-1 convolution with an optional bias and activation applied while writing the output.
pub fn conv2d_s1_fused(x: &Tensor, w: &Tensor, bias: Option<&[f32]>, act: Act<'_>, padding: usize, dilation: usize) -> Result<Tensor> {
    use rayon::prelude::*;
    let (n, cin, h, wd) = x.dims4()?;
    let (cout, wcin, kh, kw) = w.dims4()?;
    if wcin != cin {
        candle_core::bail!("conv2d: input has {cin} channels, kernel expects {wcin}");
    }
    if let Some(b) = bias {
        if b.len() != cout {
            candle_core::bail!("conv2d: bias has {} values for {cout} channels", b.len());
        }
    }
    if let Act::Prelu(a) = act {
        if a.len() != cout && a.len() != 1 {
            candle_core::bail!("conv2d: PReLU has {} slopes for {cout} channels", a.len());
        }
    }
    let hp = h + 2 * padding;
    let wp = wd + 2 * padding;
    let reach_h = dilation * (kh - 1);
    let reach_w = dilation * (kw - 1);
    if hp <= reach_h || wp <= reach_w {
        candle_core::bail!("conv2d: kernel larger than the padded input");
    }
    let oh = hp - reach_h;
    let ow = wp - reach_w;
    let plane = hp * wp;
    // Slack so the last tap's window may run past the final row.
    let plane_alloc = plane + reach_w + CHUNK;
    let cols = oh * wp;
    let single_slope = matches!(act, Act::Prelu(a) if a.len() == 1);
    let xs = slice(x)?;
    let ws = slice(w)?;

    let mut out = vec![0f32; n * cout * oh * ow];
    let mut padded = vec![0f32; cin * plane_alloc];
    for b in 0..n {
        let src = &xs[b * cin * h * wd..(b + 1) * cin * h * wd];
        padded.par_chunks_mut(plane_alloc).enumerate().for_each(|(c, dst)| {
            for y in 0..h {
                let row = (y + padding) * wp + padding;
                dst[row..row + wd].copy_from_slice(&src[(c * h + y) * wd..(c * h + y + 1) * wd]);
            }
        });
        let ob = SyncPtr(out[b * cout * oh * ow..].as_mut_ptr());
        let ob = &ob;
        let padded = &padded;
        let ws = &ws;
        (0..cols.div_ceil(CHUNK)).into_par_iter().for_each_init(
            || vec![0f32; cout * CHUNK],
            |acc, ci| {
                let c0 = ci * CHUNK;
                let len = CHUNK.min(cols - c0);
                for i in 0..kh {
                    for j in 0..kw {
                        let off = i * dilation * wp + j * dilation + c0;
                        // SAFETY: lhs/rhs/dst are live buffers addressed within their extents
                        // (m = cout, n = len, k = cin); `acc` is private to this work item.
                        unsafe {
                            gemm::gemm(
                                cout,
                                len,
                                cin,
                                acc.as_mut_ptr(),
                                1,
                                CHUNK as isize,
                                !(i == 0 && j == 0),
                                ws.as_ptr().add(i * kw + j),
                                (kh * kw) as isize,
                                (cin * kh * kw) as isize,
                                padded.as_ptr().add(off),
                                1,
                                plane_alloc as isize,
                                1.0,
                                1.0,
                                false,
                                false,
                                false,
                                Parallelism::None,
                            );
                        }
                    }
                }
                for co in 0..cout {
                    let bv = bias.map_or(0.0, |b| b[co]);
                    let ac = if single_slope { 0 } else { co };
                    let row = &acc[co * CHUNK..co * CHUNK + len];
                    let mut col = c0;
                    while col < c0 + len {
                        let y = col / wp;
                        let x0 = col % wp;
                        let run = (wp - x0).min(c0 + len - col);
                        let valid = ow.saturating_sub(x0).min(run);
                        if valid > 0 {
                            // SAFETY: (co, y, x0..x0+valid) is inside this batch's output and
                            // each output element is written by exactly one work item.
                            let dst = unsafe { std::slice::from_raw_parts_mut(ob.0.add((co * oh + y) * ow + x0), valid) };
                            for (d, v) in dst.iter_mut().zip(&row[col - c0..col - c0 + valid]) {
                                *d = act.apply(ac, v + bv);
                            }
                        }
                        col += run;
                    }
                }
            },
        );
    }
    Tensor::from_vec(out, (n, cout, oh, ow), &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn check(n: usize, cin: usize, cout: usize, h: usize, w: usize, k: usize, pad: usize, dil: usize) {
        let x = Tensor::randn(0f32, 1.0, (n, cin, h, w), &Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 0.3, (cout, cin, k, k), &Device::Cpu).unwrap();
        let fast = conv2d_s1(&x, &k, pad, dil).unwrap();
        let reference = x.conv2d(&k, pad, 1, dil, 1).unwrap();
        assert_eq!(fast.dims(), reference.dims());
        let diff = (fast - reference).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-3, "n{n} cin{cin} cout{cout} {h}x{w} k pad{pad} dil{dil}: max diff {diff}");
    }

    #[test]
    fn fused_bias_and_activations() {
        let x = Tensor::randn(0f32, 1.0, (1, 8, 40, 70), &Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 0.3, (6, 8, 3, 3), &Device::Cpu).unwrap();
        let bias: Vec<f32> = (0..6).map(|i| i as f32 * 0.1 - 0.2).collect();
        let slopes: Vec<f32> = (0..6).map(|i| 0.05 * i as f32).collect();
        let base = x.conv2d(&k, 1, 1, 1, 1).unwrap().broadcast_add(&Tensor::new(bias.as_slice(), &Device::Cpu).unwrap().reshape((1, 6, 1, 1)).unwrap()).unwrap();
        let pos = base.relu().unwrap();
        let neg = base.minimum(0f64).unwrap();
        let sl = Tensor::new(slopes.as_slice(), &Device::Cpu).unwrap().reshape((1, 6, 1, 1)).unwrap();
        let cases: Vec<(Act<'_>, Tensor)> = vec![
            (Act::None, base.clone()),
            (Act::Relu, pos.clone()),
            (Act::LeakyRelu(0.2), (&pos + (&neg * 0.2).unwrap()).unwrap()),
            (Act::Prelu(&slopes), (&pos + neg.broadcast_mul(&sl).unwrap()).unwrap()),
        ];
        for (act, want) in cases {
            let got = conv2d_s1_fused(&x, &k, Some(&bias), act, 1, 1).unwrap();
            let d = (got - want).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(d < 1e-4, "{act:?}: {d}");
        }
    }

    #[test]
    fn matches_candle_conv2d() {
        check(1, 3, 8, 17, 23, 3, 1, 1);
        check(2, 16, 4, 9, 31, 3, 1, 1);
        check(1, 8, 8, 12, 10, 3, 0, 1);
        check(1, 4, 6, 15, 14, 3, 2, 2);
        check(1, 5, 7, 11, 13, 1, 0, 1);
        check(1, 6, 3, 10, 10, 5, 2, 1);
        check(1, 64, 64, 33, 47, 3, 1, 1);
        check(1, 16, 16, 70, 90, 3, 1, 1);
    }
}
