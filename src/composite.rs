//! Composite premultiplied-alpha source over an `Rgba` raster buffer.
//!
//! The destination buffer is laid out as packed `Rgba` (RGBA byte
//! order, 4 bytes per pixel, row-major). The source is an
//! [`AlphaMask`](crate::AlphaMask) plus a function that produces the
//! source color at each pixel; coverage from the mask is multiplied
//! into the source alpha before the standard *source-over* operator
//! runs:
//!
//! ```text
//! D' = S + D * (1 - α_S)
//! α_S = α_paint * mask / 255
//! ```
//!
//! `extra_alpha` (in `0.0..=1.0`) lets the caller pre-multiply the
//! source alpha by an extra group-opacity factor without rebuilding
//! the mask.

use crate::fill::AlphaMask;
use oxideav_core::Rgba;

/// Composite a source mask + paint onto an `Rgba` byte buffer using
/// premultiplied source-over.
///
/// `dst` is the destination buffer. `dst_stride` is bytes per row
/// (typically `dst_width * 4`). `dst_width` and `dst_height` are the
/// total buffer dimensions in pixels. `(offset_x, offset_y)` is where
/// the mask's top-left corner lands in the destination buffer (may be
/// negative — the function clips automatically).
///
/// `paint_at(x, y)` is invoked for every covered pixel with the
/// pixel's destination-buffer pixel coordinate, and must return a
/// straight-alpha sRGB color.
///
/// `extra_alpha` (default `1.0`) multiplies into the source alpha
/// after the mask coverage, before the over operator. Used by group
/// opacity.
pub fn composite_rgba_premultiplied(
    dst: &mut [u8],
    dst_stride: usize,
    dst_width: u32,
    dst_height: u32,
    mask: &AlphaMask,
    offset_x: i32,
    offset_y: i32,
    extra_alpha: f32,
    mut paint_at: impl FnMut(u32, u32) -> Rgba,
) {
    if mask.is_empty() {
        return;
    }
    let extra = extra_alpha.clamp(0.0, 1.0);
    let extra_q = (extra * 255.0).round() as u32;
    if extra_q == 0 {
        return;
    }
    for my in 0..mask.height {
        let dy = offset_y + my as i32;
        if dy < 0 || dy as u32 >= dst_height {
            continue;
        }
        let row_off = (dy as usize) * dst_stride;
        let mask_row = &mask.data
            [(my as usize) * (mask.width as usize)..((my as usize) + 1) * (mask.width as usize)];
        for mx in 0..mask.width {
            let cov = mask_row[mx as usize];
            if cov == 0 {
                continue;
            }
            let dx = offset_x + mx as i32;
            if dx < 0 || dx as u32 >= dst_width {
                continue;
            }
            let pidx = row_off + (dx as usize) * 4;

            let src = paint_at(dx as u32, dy as u32);
            // src_alpha = src.a * cov * extra / (255 * 255)
            let combined =
                ((src.a as u32) * (cov as u32) * extra_q + (255 * 255 / 2)) / (255 * 255);
            if combined == 0 {
                continue;
            }
            // Pre-multiply source.
            let sa = combined as u32;
            let sr = ((src.r as u32) * sa + 127) / 255;
            let sg = ((src.g as u32) * sa + 127) / 255;
            let sb = ((src.b as u32) * sa + 127) / 255;
            // Read destination as premultiplied (we store dst in
            // straight-alpha so promote first).
            let dr0 = dst[pidx] as u32;
            let dg0 = dst[pidx + 1] as u32;
            let db0 = dst[pidx + 2] as u32;
            let da0 = dst[pidx + 3] as u32;
            let dr = (dr0 * da0 + 127) / 255;
            let dg = (dg0 * da0 + 127) / 255;
            let db = (db0 * da0 + 127) / 255;
            // Source-over in premultiplied space.
            let inv = 255 - sa;
            let or = sr + (dr * inv + 127) / 255;
            let og = sg + (dg * inv + 127) / 255;
            let ob = sb + (db * inv + 127) / 255;
            let oa = sa + (da0 * inv + 127) / 255;
            // Un-premultiply for the straight-alpha destination.
            let (or_s, og_s, ob_s) = if oa == 0 {
                (0u32, 0u32, 0u32)
            } else {
                (
                    ((or * 255 + oa / 2) / oa).min(255),
                    ((og * 255 + oa / 2) / oa).min(255),
                    ((ob * 255 + oa / 2) / oa).min(255),
                )
            };
            dst[pidx] = or_s as u8;
            dst[pidx + 1] = og_s as u8;
            dst[pidx + 2] = ob_s as u8;
            dst[pidx + 3] = oa.min(255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(c: Rgba) -> impl FnMut(u32, u32) -> Rgba {
        move |_x, _y| c
    }

    #[test]
    fn opaque_paint_with_full_mask_overwrites_dst() {
        let mut dst = vec![0u8; 16]; // 2x2 RGBA
        let mut m = AlphaMask::new(2, 2);
        m.data.fill(255);
        composite_rgba_premultiplied(
            &mut dst,
            8,
            2,
            2,
            &m,
            0,
            0,
            1.0,
            solid(Rgba::opaque(10, 20, 30)),
        );
        assert_eq!(&dst[0..4], &[10, 20, 30, 255]);
        assert_eq!(&dst[12..16], &[10, 20, 30, 255]);
    }

    #[test]
    fn empty_mask_is_no_op() {
        let mut dst = vec![1u8; 16];
        let m = AlphaMask::default();
        composite_rgba_premultiplied(
            &mut dst,
            8,
            2,
            2,
            &m,
            0,
            0,
            1.0,
            solid(Rgba::opaque(0, 0, 0)),
        );
        assert!(dst.iter().all(|&b| b == 1));
    }

    #[test]
    fn extra_alpha_zero_skips_blend() {
        let mut dst = vec![0u8; 16];
        let mut m = AlphaMask::new(2, 2);
        m.data.fill(255);
        composite_rgba_premultiplied(
            &mut dst,
            8,
            2,
            2,
            &m,
            0,
            0,
            0.0,
            solid(Rgba::opaque(255, 0, 0)),
        );
        assert!(dst.iter().all(|&b| b == 0));
    }

    #[test]
    fn negative_offset_clips_correctly() {
        let mut dst = vec![0u8; 16]; // 2x2 RGBA
        let mut m = AlphaMask::new(2, 2);
        m.data.fill(255);
        // Offset by (-1, -1) so only the bottom-right pixel of the
        // mask lands in the destination at (0, 0).
        composite_rgba_premultiplied(
            &mut dst,
            8,
            2,
            2,
            &m,
            -1,
            -1,
            1.0,
            solid(Rgba::opaque(7, 8, 9)),
        );
        assert_eq!(&dst[0..4], &[7, 8, 9, 255]);
        // Other pixels should be untouched.
        assert_eq!(&dst[4..8], &[0, 0, 0, 0]);
    }
}
