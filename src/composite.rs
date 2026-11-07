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

use crate::blend::{BlendMode, blend_over};
use crate::fill::{AlphaMask, Coverage};
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
        // Clip the mask column range to the destination.
        let lo = (rx0 as i64).max(-(offset_x as i64)) as u32;
        let hi = (rx1 as i64).min(dst_width as i64 - 1 - offset_x as i64);
        if hi < lo as i64 {
            continue;
        }
        let hi = hi as u32;
        let dx0 = (offset_x + lo as i32) as usize;
        let n = (hi - lo + 1) as usize;
        composite_row(
            &mut dst[row_off + dx0 * 4..row_off + (dx0 + n) * 4],
            &mask_row[lo as usize..lo as usize + n],
            dx0 as u32,
            dy as u32,
            extra_q,
            blend,
            &mut paint_at,
        );
    }
}

/// `x / 255` rounded to nearest, for `x ≤ 65 535`; identical to
/// `(x + 127) / 255` there (255 is odd, so `x / 255` never ends in
/// exactly one half).
#[inline(always)]
fn div255(x: u32) -> u32 {
    let t = x + 128;
    (t + (t >> 8)) >> 8
}

/// `⌈2³² / d⌉` for `d ∈ 1..=255` (index 0 unused). With it,
/// `(n · R[d]) >> 32 == n / d` for every `n < 2¹⁶`: the product
/// overshoots `n / d` by less than `n / 2³² < 2⁻¹⁶`, while the
/// fractional part of `n / d` is at most `1 − 1/d ≤ 1 − 2⁻⁸`.
static RECIP: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut d = 1;
    while d < 256 {
        t[d] = (1u64 << 32).div_ceil(d as u64);
        d += 1;
    }
    t
};

/// Exact `n / d` for `n < 2¹⁶`, `d ∈ 1..=255`.
#[inline(always)]
fn div_small(n: u32, d: u32) -> u32 {
    ((n as u64 * RECIP[d as usize]) >> 32) as u32
}

/// Source-over of a straight-alpha source whose coverage-folded alpha
/// is `sa ∈ 1..=254` onto one straight-alpha destination pixel, with
/// the premultiplied source channels already computed. Integer
/// arithmetic identical to the reference formulation (premultiply
/// both, `S + D·(1 − αS)`, un-premultiply rounding to nearest).
#[inline(always)]
fn over_px(px: &mut [u8], sr: u32, sg: u32, sb: u32, sa: u32) {
    let da0 = px[3] as u32;
    let inv = 255 - sa;
    if da0 == 255 {
        // Opaque backdrop: the result is opaque and the
        // un-premultiply divides by 255, which is exact.
        px[0] = (sr + div255(px[0] as u32 * inv)) as u8;
        px[1] = (sg + div255(px[1] as u32 * inv)) as u8;
        px[2] = (sb + div255(px[2] as u32 * inv)) as u8;
        return;
    }
    let dr = div255(px[0] as u32 * da0);
    let dg = div255(px[1] as u32 * da0);
    let db = div255(px[2] as u32 * da0);
    let or = sr + div255(dr * inv);
    let og = sg + div255(dg * inv);
    let ob = sb + div255(db * inv);
    let oa = sa + div255(da0 * inv);
    // oa ≥ sa ≥ 1 here.
    px[0] = div_small(or * 255 + oa / 2, oa).min(255) as u8;
    px[1] = div_small(og * 255 + oa / 2, oa).min(255) as u8;
    px[2] = div_small(ob * 255 + oa / 2, oa).min(255) as u8;
    px[3] = oa.min(255) as u8;
}

/// Composite one row: `dst` holds `cov.len()` packed RGBA pixels
/// starting at device column `x0` of device row `y`.
#[inline]
pub(crate) fn composite_row(
    dst: &mut [u8],
    cov: &[u8],
    x0: u32,
    y: u32,
    extra_q: u32,
    blend: BlendMode,
    paint_at: &mut impl FnMut(u32, u32) -> Rgba,
) {
    for (k, (&c, px)) in cov.iter().zip(dst.as_chunks_mut::<4>().0).enumerate() {
        if c == 0 {
            continue;
        }
        let src = paint_at(x0 + k as u32, y);
        // src_alpha = src.a * cov * extra / (255 * 255)
        let combined = ((src.a as u32) * (c as u32) * extra_q + (255 * 255 / 2)) / (255 * 255);
        if combined == 0 {
            continue;
        }
        if blend.is_normal() {
            if combined == 255 {
                // Fully-opaque source pixel: the over operator reduces
                // to a plain overwrite.
                px.copy_from_slice(&[src.r, src.g, src.b, 255]);
                continue;
            }
            let sa = combined;
            over_px(
                px,
                div255(src.r as u32 * sa),
                div255(src.g as u32 * sa),
                div255(src.b as u32 * sa),
                sa,
            );
        } else {
            // Full per-pixel separable blend. The backdrop is read
            // straight-alpha; the source colour's alpha is replaced
            // with the coverage-folded `combined` value so the spec's
            // compositing formula sees the already-modulated source.
            let cb = Rgba::new(px[0], px[1], px[2], px[3]);
            let cs = Rgba::new(src.r, src.g, src.b, combined.min(255) as u8);
            let cr = blend_over(cb, cs, blend);
            px.copy_from_slice(&[cr.r, cr.g, cr.b, cr.a]);
        }
    }
}

/// A solid paint prepared for source-over row compositing: per
/// coverage byte, the folded source alpha and premultiplied channels
/// (bit-identical to evaluating them per pixel).
pub(crate) struct SolidSource {
    color: Rgba,
    /// `[sa, sr, sg, sb]` per coverage value.
    lut: [[u8; 4]; 256],
}

impl SolidSource {
    pub(crate) fn new(color: Rgba, extra_q: u32) -> Self {
        let mut lut = [[0u8; 4]; 256];
        for (c, e) in lut.iter_mut().enumerate() {
            let sa = ((color.a as u32) * (c as u32) * extra_q + (255 * 255 / 2)) / (255 * 255);
            *e = [
                sa as u8,
                div255(color.r as u32 * sa) as u8,
                div255(color.g as u32 * sa) as u8,
                div255(color.b as u32 * sa) as u8,
            ];
        }
        SolidSource { color, lut }
    }

    /// Source-over one row (same contract as [`composite_row`] with a
    /// constant paint and [`BlendMode::Normal`]).
    #[inline]
    pub(crate) fn over_row(&self, dst: &mut [u8], cov: &[u8]) {
        let opaque = [self.color.r, self.color.g, self.color.b, 255];
        let full = self.lut[255][0] == 255;
        let mut k = 0;
        let n = cov.len();
        while k < n {
            let c = cov[k];
            if c == 255 && full {
                // Solid run: a plain fill.
                let mut e = k + 1;
                while e < n && cov[e] == 255 {
                    e += 1;
                }
                for px in dst[k * 4..e * 4].as_chunks_mut::<4>().0 {
                    px.copy_from_slice(&opaque);
                }
                k = e;
                continue;
            }
            if c != 0 {
                let [sa, sr, sg, sb] = self.lut[c as usize];
                if sa == 255 {
                    dst[k * 4..k * 4 + 4].copy_from_slice(&opaque);
                } else if sa != 0 {
                    over_px(
                        &mut dst[k * 4..k * 4 + 4],
                        sr as u32,
                        sg as u32,
                        sb as u32,
                        sa as u32,
                    );
                }
            }
            k += 1;
        }
    }
}

/// [`composite_rgba_premultiplied_blend_rect`] for a windowed
/// [`Coverage`]: `rect` is an inclusive **device** rectangle (normally
/// the coverage's own bounds); `paint_at` receives device coordinates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn composite_coverage_rect(
    dst: &mut [u8],
    dst_stride: usize,
    dst_width: u32,
    dst_height: u32,
    cov: &Coverage,
    extra_alpha: f32,
    blend: BlendMode,
    rect: (u32, u32, u32, u32),
    paint_at: impl FnMut(u32, u32) -> Rgba,
) {
    if cov.mask.is_empty() {
        return;
    }
    let (rx0, ry0, rx1, ry1) = rect;
    if rx1 < cov.x0 || ry1 < cov.y0 {
        return;
    }
    composite_rgba_premultiplied_blend_rect(
        dst,
        dst_stride,
        dst_width,
        dst_height,
        &cov.mask,
        cov.x0 as i32,
        cov.y0 as i32,
        extra_alpha,
        blend,
        (
            rx0.saturating_sub(cov.x0),
            ry0.saturating_sub(cov.y0),
            rx1 - cov.x0,
            ry1 - cov.y0,
        ),
        paint_at,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(c: Rgba) -> impl FnMut(u32, u32) -> Rgba {
        move |_x, _y| c
    }

    #[test]
    fn div255_is_exact_rounding_division() {
        for x in 0..=65_535u32 {
            assert_eq!(div255(x), (x + 127) / 255, "x={x}");
        }
    }

    #[test]
    fn reciprocal_division_is_exact() {
        for d in 1..=255u32 {
            for n in 0..65_536u32 {
                assert_eq!(div_small(n, d), n / d, "{n}/{d}");
            }
        }
    }

    /// The reference source-over (straight-alpha in and out, both
    /// sides premultiplied, rounding divisions).
    fn reference_over(d: [u8; 4], s: [u8; 3], sa: u32) -> [u8; 4] {
        let sr = (s[0] as u32 * sa + 127) / 255;
        let sg = (s[1] as u32 * sa + 127) / 255;
        let sb = (s[2] as u32 * sa + 127) / 255;
        let da0 = d[3] as u32;
        let dr = (d[0] as u32 * da0 + 127) / 255;
        let dg = (d[1] as u32 * da0 + 127) / 255;
        let db = (d[2] as u32 * da0 + 127) / 255;
        let inv = 255 - sa;
        let or = sr + (dr * inv + 127) / 255;
        let og = sg + (dg * inv + 127) / 255;
        let ob = sb + (db * inv + 127) / 255;
        let oa = sa + (da0 * inv + 127) / 255;
        let un = |c: u32| (c * 255 + oa / 2).checked_div(oa).map_or(0, |v| v.min(255)) as u8;
        [un(or), un(og), un(ob), oa.min(255) as u8]
    }

    #[test]
    fn fast_over_matches_reference_formula() {
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let alphas = [0u8, 1, 2, 77, 128, 200, 254, 255];
        for sa in 1..=254u32 {
            for &da in &alphas {
                for _ in 0..24 {
                    let d = [next() as u8, next() as u8, next() as u8, da];
                    let s = [next() as u8, next() as u8, next() as u8];
                    let mut px = d;
                    over_px(
                        &mut px,
                        div255(s[0] as u32 * sa),
                        div255(s[1] as u32 * sa),
                        div255(s[2] as u32 * sa),
                        sa,
                    );
                    assert_eq!(px, reference_over(d, s, sa), "d={d:?} s={s:?} sa={sa}");
                }
            }
        }
    }

    #[test]
    fn solid_row_matches_generic_row() {
        let mut seed = 99u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        };
        for case in 0..64 {
            let color = Rgba::new(next(), next(), next(), [255, 0, 1, 128, next()][case % 5]);
            let extra_q = [255u32, 128, 1, 254][case % 4];
            let cov: Vec<u8> = (0..97)
                .map(|i| match i % 7 {
                    0..=2 => 255,
                    3 => 0,
                    _ => next(),
                })
                .collect();
            let dst: Vec<u8> = (0..97 * 4).map(|_| next()).collect();
            let mut a = dst.clone();
            let mut b = dst;
            SolidSource::new(color, extra_q).over_row(&mut a, &cov);
            composite_row(
                &mut b,
                &cov,
                0,
                0,
                extra_q,
                BlendMode::Normal,
                &mut |_, _| color,
            );
            assert_eq!(a, b, "case {case}");
        }
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
