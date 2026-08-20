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

use crate::blend::{blend_over, BlendMode};
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
    paint_at: impl FnMut(u32, u32) -> Rgba,
) {
    composite_rgba_premultiplied_blend(
        dst,
        dst_stride,
        dst_width,
        dst_height,
        mask,
        offset_x,
        offset_y,
        extra_alpha,
        BlendMode::Normal,
        paint_at,
    )
}

/// Composite a source mask + paint onto an `Rgba` byte buffer using
/// the given [`BlendMode`] in the spec-defined PDF §11.3.3 / W3C
/// Compositing-1 §10 basic-compositing formula.
///
/// For [`BlendMode::Normal`] this is the same fast source-over path as
/// [`composite_rgba_premultiplied`]; non-normal modes route through the
/// per-pixel `blend_over` evaluation in [`crate::blend`]. Otherwise the
/// arguments match [`composite_rgba_premultiplied`].
pub fn composite_rgba_premultiplied_blend(
    dst: &mut [u8],
    dst_stride: usize,
    dst_width: u32,
    dst_height: u32,
    mask: &AlphaMask,
    offset_x: i32,
    offset_y: i32,
    extra_alpha: f32,
    blend: BlendMode,
    paint_at: impl FnMut(u32, u32) -> Rgba,
) {
    if mask.is_empty() {
        return;
    }
    composite_rgba_premultiplied_blend_rect(
        dst,
        dst_stride,
        dst_width,
        dst_height,
        mask,
        offset_x,
        offset_y,
        extra_alpha,
        blend,
        (0, 0, mask.width - 1, mask.height - 1),
        paint_at,
    )
}

/// [`composite_rgba_premultiplied_blend`] restricted to the inclusive
/// mask-coordinate rectangle `(x0, y0, x1, y1)`.
///
/// Every mask pixel outside the rectangle is treated as if its
/// coverage were zero — callers pass the touched-pixel bounding box
/// reported by the rasteriser, whose contract guarantees exactly that,
/// so restricting the scan changes no output byte while skipping the
/// all-zero border of a small shape on a large canvas.
#[allow(clippy::too_many_arguments)]
pub(crate) fn composite_rgba_premultiplied_blend_rect(
    dst: &mut [u8],
    dst_stride: usize,
    dst_width: u32,
    dst_height: u32,
    mask: &AlphaMask,
    offset_x: i32,
    offset_y: i32,
    extra_alpha: f32,
    blend: BlendMode,
    rect: (u32, u32, u32, u32),
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
    let (rx0, ry0, rx1, ry1) = rect;
    let rx1 = rx1.min(mask.width - 1);
    let ry1 = ry1.min(mask.height - 1);
    if rx0 > rx1 || ry0 > ry1 {
        return;
    }
    for my in ry0..=ry1 {
        let dy = offset_y + my as i32;
        if dy < 0 || dy as u32 >= dst_height {
            continue;
        }
        let row_off = (dy as usize) * dst_stride;
        let mask_row = &mask.data
            [(my as usize) * (mask.width as usize)..((my as usize) + 1) * (mask.width as usize)];
        for mx in rx0..=rx1 {
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
            if blend.is_normal() {
                if combined == 255 {
                    // Fully-opaque source pixel: the over operator
                    // reduces to a plain overwrite. Algebraically
                    // identical to the general path below — with
                    // `sa = 255` the premultiply `(c·255 + 127) / 255`
                    // is exact, `inv = 0` zeroes every destination
                    // term, and the final un-premultiply divides by
                    // `oa = 255` returning the source bytes verbatim.
                    dst[pidx] = src.r;
                    dst[pidx + 1] = src.g;
                    dst[pidx + 2] = src.b;
                    dst[pidx + 3] = 255;
                    continue;
                }
                // Fast path: standard premultiplied source-over.
                let sa = combined;
                let sr = ((src.r as u32) * sa + 127) / 255;
                let sg = ((src.g as u32) * sa + 127) / 255;
                let sb = ((src.b as u32) * sa + 127) / 255;
                let dr0 = dst[pidx] as u32;
                let dg0 = dst[pidx + 1] as u32;
                let db0 = dst[pidx + 2] as u32;
                let da0 = dst[pidx + 3] as u32;
                let dr = (dr0 * da0 + 127) / 255;
                let dg = (dg0 * da0 + 127) / 255;
                let db = (db0 * da0 + 127) / 255;
                let inv = 255 - sa;
                let or = sr + (dr * inv + 127) / 255;
                let og = sg + (dg * inv + 127) / 255;
                let ob = sb + (db * inv + 127) / 255;
                let oa = sa + (da0 * inv + 127) / 255;
                let or_s = (or * 255 + oa / 2)
                    .checked_div(oa)
                    .map_or(0, |v| v.min(255));
                let og_s = (og * 255 + oa / 2)
                    .checked_div(oa)
                    .map_or(0, |v| v.min(255));
                let ob_s = (ob * 255 + oa / 2)
                    .checked_div(oa)
                    .map_or(0, |v| v.min(255));
                dst[pidx] = or_s as u8;
                dst[pidx + 1] = og_s as u8;
                dst[pidx + 2] = ob_s as u8;
                dst[pidx + 3] = oa.min(255) as u8;
            } else {
                // Slow path: full per-pixel separable blend. The
                // backdrop is read straight-alpha; the source colour's
                // alpha is replaced with the coverage-folded `combined`
                // value so the spec's compositing formula sees the
                // already-modulated source.
                let cb = Rgba::new(dst[pidx], dst[pidx + 1], dst[pidx + 2], dst[pidx + 3]);
                let cs = Rgba::new(src.r, src.g, src.b, combined.min(255) as u8);
                let cr = blend_over(cb, cs, blend);
                dst[pidx] = cr.r;
                dst[pidx + 1] = cr.g;
                dst[pidx + 2] = cr.b;
                dst[pidx + 3] = cr.a;
            }
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
