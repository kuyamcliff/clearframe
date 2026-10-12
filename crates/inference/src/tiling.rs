//! Tiled processing for image-to-image networks so large frames fit in memory.

use candle_core::Result;

use crate::image::RgbImage;

/// Process `img` in tiles of `tile` input pixels with `pad` pixels of context on
/// every side. `f` maps an input crop to an output crop scaled by `scale`.
/// Only the centre of each processed tile is kept, which removes seams when
/// `pad` covers the network's receptive field.
pub fn process_tiled<F>(img: &RgbImage, scale: usize, tile: usize, pad: usize, mut f: F) -> Result<RgbImage>
where
    F: FnMut(&RgbImage) -> Result<RgbImage>,
{
    if tile == 0 || (img.width <= tile && img.height <= tile) {
        let out = f(img)?;
        check(&out, img.width * scale, img.height * scale)?;
        return Ok(out);
    }
    let mut out = RgbImage::new(img.width * scale, img.height * scale);
    let tiles_x = img.width.div_ceil(tile);
    let tiles_y = img.height.div_ceil(tile);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = tx * tile;
            let y0 = ty * tile;
            let x1 = (x0 + tile).min(img.width);
            let y1 = (y0 + tile).min(img.height);
            let px0 = x0.saturating_sub(pad);
            let py0 = y0.saturating_sub(pad);
            let px1 = (x1 + pad).min(img.width);
            let py1 = (y1 + pad).min(img.height);
            let crop = img.crop(px0, py0, px1 - px0, py1 - py0);
            let res = f(&crop)?;
            check(&res, crop.width * scale, crop.height * scale)?;
            let keep = res.crop((x0 - px0) * scale, (y0 - py0) * scale, (x1 - x0) * scale, (y1 - y0) * scale);
            out.paste(&keep, x0 * scale, y0 * scale);
        }
    }
    Ok(out)
}

fn check(img: &RgbImage, w: usize, h: usize) -> Result<()> {
    if img.width != w || img.height != h {
        candle_core::bail!("model produced {}x{} but {}x{} was expected", img.width, img.height, w, h);
    }
    Ok(())
}

/// Pad an image to a multiple of `m` by replicating edges; returns the padded image.
pub fn pad_to_multiple(img: &RgbImage, m: usize) -> RgbImage {
    if m <= 1 || (img.width % m == 0 && img.height % m == 0) {
        return img.clone();
    }
    let w = img.width.div_ceil(m) * m;
    let h = img.height.div_ceil(m) * m;
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        let sy = y.min(img.height - 1);
        for x in 0..w {
            let sx = x.min(img.width - 1);
            for c in 0..3 {
                out.set(x, y, c, img.get(sx, sy, c));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: usize, h: usize) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    img.set(x, y, c, ((x * 7 + y * 13 + c) % 97) as f32 / 97.0);
                }
            }
        }
        img
    }

    fn nearest2x(img: &RgbImage) -> Result<RgbImage> {
        let mut out = RgbImage::new(img.width * 2, img.height * 2);
        for y in 0..out.height {
            for x in 0..out.width {
                for c in 0..3 {
                    out.set(x, y, c, img.get(x / 2, y / 2, c));
                }
            }
        }
        Ok(out)
    }

    #[test]
    fn tiled_matches_whole_for_pointwise_op() {
        let img = gradient(37, 23);
        let whole = nearest2x(&img).unwrap();
        let tiled = process_tiled(&img, 2, 8, 3, nearest2x).unwrap();
        assert_eq!(whole, tiled);
    }

    #[test]
    fn pad_multiple() {
        let img = gradient(5, 3);
        let p = pad_to_multiple(&img, 4);
        assert_eq!((p.width, p.height), (8, 4));
        assert_eq!(p.get(7, 3, 0), img.get(4, 2, 0));
    }
}
