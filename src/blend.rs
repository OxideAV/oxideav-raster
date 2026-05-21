//! Standard separable blend modes (PDF 32000-1:2008 §11.3.5 /
//! W3C Compositing-1 §10), evaluated per channel.
//!
//! The crate's default composite path is **source-over** ([`BlendMode::Normal`])
//! and runs through the fast-path branch in [`crate::composite`]. Any other
//! mode routes through this module, which implements the per-channel blend
//! `B(Cb, Cs)` from the spec, then folds it back into the standard PDF
//! §11.3.3 basic compositing formula:
//!
//! ```text
//!     Cr = (1 − αs/αr) · Cb + (αs/αr) · [ (1 − αb) · Cs + αb · B(Cb, Cs) ]
//!     αr = αs + αb · (1 − αs)
//! ```
//!
//! All channel-domain math is in normalised `f32` `[0, 1]`. Source and
//! backdrop colours are taken straight-alpha (the renderer caller is
//! responsible for any pre-coverage modulation of the source alpha
//! channel before the blend, just like the source-over path).
//!
//! Wall provenance: formulas transcribed from
//! `docs/document/pdf/PDF32000_2008.pdf` §11.3.3–§11.3.5 (the public
//! ISO 32000-1 PDF spec). No external blend implementation consulted.

use oxideav_core::Rgba;

/// Standard separable blend modes (PDF 32000-1:2008 Table 136).
///
/// Each variant defines a per-channel function `B(Cb, Cs)` where `Cb` is
/// the backdrop colour component (already on the destination buffer)
/// and `Cs` is the source colour component being painted, both in
/// `[0, 1]`. The result is fed into the PDF §11.3.3 / Porter–Duff over
/// operator together with the source and backdrop alphas.
///
/// `Normal` is the default and is equivalent to `B(Cb, Cs) = Cs`
/// (i.e. plain source-over). The renderer's composite path special-cases
/// it for speed so callers paying for the blend dispatch is opt-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BlendMode {
    /// `B(Cb, Cs) = Cs`. Standard Porter–Duff source-over. Default.
    #[default]
    Normal,
    /// `B(Cb, Cs) = Cb · Cs`. Multiplies backdrop and source; result is
    /// always at least as dark as either input. Black wins; white is the
    /// identity.
    Multiply,
    /// `B(Cb, Cs) = Cb + Cs − Cb·Cs`. Multiplies the complements, then
    /// complements. Result is at least as light as either input. White
    /// wins; black is the identity.
    Screen,
    /// `B(Cb, Cs) = HardLight(Cs, Cb)`. Multiply for dark backdrops,
    /// screen for light backdrops. Preserves backdrop highlights and
    /// shadows while mixing the source in.
    Overlay,
    /// `B(Cb, Cs) = min(Cb, Cs)`. Selects the darker of the two
    /// channel components.
    Darken,
    /// `B(Cb, Cs) = max(Cb, Cs)`. Selects the lighter of the two
    /// channel components.
    Lighten,
    /// Brightens the backdrop toward the source. `Cb / (1 − Cs)`
    /// (clamped to 1) for `Cs < 1`; `1` if `Cs == 1`. Black source is
    /// the identity.
    ColorDodge,
    /// Darkens the backdrop using the source. `1 − (1 − Cb) / Cs`
    /// (clamped to 0) for `Cs > 0`; `0` if `Cs == 0`. White source is
    /// the identity.
    ColorBurn,
    /// Multiply for `Cs ≤ 0.5`, screen for `Cs > 0.5`. Like shining a
    /// harsh spotlight on the backdrop.
    HardLight,
    /// Softer Overlay. Reduces contrast where the source is mid-grey
    /// and amplifies it at the extremes; uses the polynomial-vs-sqrt
    /// `D(Cb)` switch from PDF §11.3.5.
    SoftLight,
    /// `B(Cb, Cs) = |Cb − Cs|`. White source inverts the backdrop;
    /// black source is the identity.
    Difference,
    /// `B(Cb, Cs) = Cb + Cs − 2·Cb·Cs`. Like Difference but lower
    /// contrast in the midrange.
    Exclusion,
}

impl BlendMode {
    /// True when this mode is the standard pass-through source-over.
    /// Used by the composite path to dispatch to the fast branch.
    #[inline]
    pub fn is_normal(self) -> bool {
        matches!(self, BlendMode::Normal)
    }

    /// Apply this blend mode's per-channel function to the three RGB
    /// channels of `cb` (backdrop) and `cs` (source). Alpha is left
    /// untouched here; the composite formula handles alpha separately.
    /// Inputs and outputs are straight-alpha `[0, 1]`.
    #[inline]
    pub fn blend_rgb(self, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
        [
            blend_channel(self, cb[0], cs[0]),
            blend_channel(self, cb[1], cs[1]),
            blend_channel(self, cb[2], cs[2]),
        ]
    }
}

/// Per-channel `B(Cb, Cs)` for `mode`, with both inputs and the output
/// in `[0, 1]`. Out-of-range inputs are clamped first; the result is
/// also clamped on the way out per PDF §11.3.5 ("the result of the
/// mixing formula must be clamped").
#[inline]
pub fn blend_channel(mode: BlendMode, cb: f32, cs: f32) -> f32 {
    let cb = cb.clamp(0.0, 1.0);
    let cs = cs.clamp(0.0, 1.0);
    let r = match mode {
        BlendMode::Normal => cs,
        BlendMode::Multiply => cb * cs,
        BlendMode::Screen => cb + cs - cb * cs,
        BlendMode::Overlay => hard_light(cs, cb),
        BlendMode::Darken => cb.min(cs),
        BlendMode::Lighten => cb.max(cs),
        BlendMode::ColorDodge => {
            // PDF §11.3.5: min(1, Cb / (1 − Cs)) if Cs < 1; 1 if Cs == 1.
            // We also fold the W3C tweak that returns 0 when Cb == 0 so
            // a fully-black backdrop stays black under any source — both
            // specs agree on this limit even if the PDF table doesn't
            // spell it out explicitly.
            if cs >= 1.0 {
                1.0
            } else if cb <= 0.0 {
                0.0
            } else {
                (cb / (1.0 - cs)).min(1.0)
            }
        }
        BlendMode::ColorBurn => {
            // PDF §11.3.5: 1 − min(1, (1 − Cb) / Cs) if Cs > 0; 0 if
            // Cs == 0. W3C also clamps the `Cb == 1` case to 1 so a
            // fully-white backdrop stays white.
            if cs <= 0.0 {
                0.0
            } else if cb >= 1.0 {
                1.0
            } else {
                1.0 - ((1.0 - cb) / cs).min(1.0)
            }
        }
        BlendMode::HardLight => hard_light(cb, cs),
        BlendMode::SoftLight => soft_light(cb, cs),
        BlendMode::Difference => (cb - cs).abs(),
        BlendMode::Exclusion => cb + cs - 2.0 * cb * cs,
    };
    r.clamp(0.0, 1.0)
}

/// `HardLight(Cb, Cs) = Multiply(Cb, 2·Cs)` if `Cs ≤ 0.5`,
/// `Screen(Cb, 2·Cs − 1)` otherwise. PDF §11.3.5 Table 136.
#[inline]
fn hard_light(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 {
        cb * (2.0 * cs)
    } else {
        let s2 = 2.0 * cs - 1.0;
        cb + s2 - cb * s2
    }
}

/// `SoftLight(Cb, Cs)` per PDF §11.3.5 Table 136. Uses the
/// polynomial `D(x) = ((16·x − 12)·x + 4)·x` for `x ≤ 0.25` and
/// `D(x) = sqrt(x)` otherwise; for `Cs ≤ 0.5` returns
/// `Cb − (1 − 2·Cs)·Cb·(1 − Cb)`, otherwise
/// `Cb + (2·Cs − 1)·(D(Cb) − Cb)`.
#[inline]
fn soft_light(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 {
        cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
    } else {
        let d = if cb <= 0.25 {
            ((16.0 * cb - 12.0) * cb + 4.0) * cb
        } else {
            cb.sqrt()
        };
        cb + (2.0 * cs - 1.0) * (d - cb)
    }
}

/// Apply [`BlendMode`] to a backdrop colour `cb` and a source colour
/// `cs`, both straight-alpha sRGB `Rgba`, using the PDF §11.3.3 basic
/// compositing formula:
///
/// ```text
///     αr = αs + αb · (1 − αs)
///     Cr = ((1 − αs/αr) · Cb + (αs/αr) · [(1 − αb)·Cs + αb·B(Cb, Cs)]) channel-wise
/// ```
///
/// Returns a straight-alpha sRGB result. If `αr == 0` the result is
/// fully transparent black (any colour value is meaningless when alpha
/// is zero, per PDF §11.3.3 NOTE 1).
///
/// The fast `BlendMode::Normal` case is **not** dispatched here — the
/// caller should prefer the existing source-over path in
/// [`crate::composite`] which avoids the per-pixel f32 round-trip. This
/// function is intended for the modes that genuinely depend on `Cb`.
pub fn blend_over(cb: Rgba, cs: Rgba, mode: BlendMode) -> Rgba {
    let ab = cb.a as f32 / 255.0;
    let ar_s = cs.a as f32 / 255.0;
    let ar = ar_s + ab * (1.0 - ar_s);
    if ar <= 0.0 {
        return Rgba::new(0, 0, 0, 0);
    }
    let cb_rgb = [
        cb.r as f32 / 255.0,
        cb.g as f32 / 255.0,
        cb.b as f32 / 255.0,
    ];
    let cs_rgb = [
        cs.r as f32 / 255.0,
        cs.g as f32 / 255.0,
        cs.b as f32 / 255.0,
    ];
    let b = mode.blend_rgb(cb_rgb, cs_rgb);
    let mix = [
        (1.0 - ab) * cs_rgb[0] + ab * b[0],
        (1.0 - ab) * cs_rgb[1] + ab * b[1],
        (1.0 - ab) * cs_rgb[2] + ab * b[2],
    ];
    let inv_ar = 1.0 / ar;
    let s_over_r = ar_s * inv_ar;
    let b_over_r = 1.0 - s_over_r;
    let cr = [
        b_over_r * cb_rgb[0] + s_over_r * mix[0],
        b_over_r * cb_rgb[1] + s_over_r * mix[1],
        b_over_r * cb_rgb[2] + s_over_r * mix[2],
    ];
    Rgba::new(
        (cr[0] * 255.0).round().clamp(0.0, 255.0) as u8,
        (cr[1] * 255.0).round().clamp(0.0, 255.0) as u8,
        (cr[2] * 255.0).round().clamp(0.0, 255.0) as u8,
        (ar * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // --- per-channel B(Cb, Cs) spot-checks against the PDF table.

    #[test]
    fn normal_returns_source() {
        assert!(approx(
            blend_channel(BlendMode::Normal, 0.3, 0.7),
            0.7,
            1e-6
        ));
    }

    #[test]
    fn multiply_known_values() {
        // PDF Table 136: cb · cs.
        assert!(approx(
            blend_channel(BlendMode::Multiply, 0.5, 0.5),
            0.25,
            1e-6
        ));
        // White is identity.
        assert!(approx(
            blend_channel(BlendMode::Multiply, 0.4, 1.0),
            0.4,
            1e-6
        ));
        // Black wins.
        assert!(approx(
            blend_channel(BlendMode::Multiply, 0.7, 0.0),
            0.0,
            1e-6
        ));
    }

    #[test]
    fn screen_known_values() {
        // cb + cs − cb·cs at (0.5, 0.5) = 0.75.
        assert!(approx(
            blend_channel(BlendMode::Screen, 0.5, 0.5),
            0.75,
            1e-6
        ));
        // White wins.
        assert!(approx(
            blend_channel(BlendMode::Screen, 0.4, 1.0),
            1.0,
            1e-6
        ));
        // Black is identity.
        assert!(approx(
            blend_channel(BlendMode::Screen, 0.4, 0.0),
            0.4,
            1e-6
        ));
    }

    #[test]
    fn darken_and_lighten_pick_extremes() {
        assert!(approx(
            blend_channel(BlendMode::Darken, 0.2, 0.7),
            0.2,
            1e-6
        ));
        assert!(approx(
            blend_channel(BlendMode::Lighten, 0.2, 0.7),
            0.7,
            1e-6
        ));
    }

    #[test]
    fn overlay_is_hardlight_swapped() {
        // Overlay(Cb, Cs) ≡ HardLight(Cs, Cb).
        for &(cb, cs) in &[(0.2_f32, 0.7), (0.6, 0.3), (0.5, 0.5), (0.0, 1.0)] {
            let ov = blend_channel(BlendMode::Overlay, cb, cs);
            let hl = blend_channel(BlendMode::HardLight, cs, cb);
            assert!(
                approx(ov, hl, 1e-6),
                "Overlay({cb}, {cs})={ov}  HardLight(swap)={hl}"
            );
        }
    }

    #[test]
    fn hardlight_known_values() {
        // Cs = 0.25 → Multiply(Cb, 0.5) = 0.5·Cb. With Cb=0.6 → 0.3.
        assert!(approx(
            blend_channel(BlendMode::HardLight, 0.6, 0.25),
            0.3,
            1e-6
        ));
        // Cs = 0.75 → Screen(Cb, 0.5) = Cb + 0.5 − 0.5·Cb = 0.5·Cb + 0.5.
        // Cb=0.6 → 0.8.
        assert!(approx(
            blend_channel(BlendMode::HardLight, 0.6, 0.75),
            0.8,
            1e-6
        ));
    }

    #[test]
    fn softlight_known_values() {
        // Cs = 0.5 boundary: takes the first branch (Cs ≤ 0.5), with
        // (1 − 2·0.5) = 0 → result = Cb.
        assert!(approx(
            blend_channel(BlendMode::SoftLight, 0.4, 0.5),
            0.4,
            1e-6
        ));
        // Cs = 0 → Cb − Cb·(1 − Cb) = Cb² . Cb = 0.5 → 0.25.
        assert!(approx(
            blend_channel(BlendMode::SoftLight, 0.5, 0.0),
            0.25,
            1e-6
        ));
        // Cs = 1 with Cb = 0.5 (>0.25 → D = sqrt). 0.5 + 1·(√0.5 − 0.5) = √0.5.
        let expected = 0.5_f32.sqrt();
        assert!(approx(
            blend_channel(BlendMode::SoftLight, 0.5, 1.0),
            expected,
            1e-6
        ));
        // Cs = 1 with Cb = 0.1 (≤0.25 → polynomial D).
        // D(0.1) = ((16·0.1 − 12)·0.1 + 4)·0.1 = ((1.6 − 12)·0.1 + 4)·0.1
        //        = (−10.4·0.1 + 4)·0.1 = (−1.04 + 4)·0.1 = 2.96·0.1 = 0.296.
        // Result = 0.1 + 1·(0.296 − 0.1) = 0.296.
        assert!(approx(
            blend_channel(BlendMode::SoftLight, 0.1, 1.0),
            0.296,
            1e-5
        ));
    }

    #[test]
    fn colordodge_and_colorburn_limits() {
        // Dodge: Cs = 1 → 1, Cs = 0 → Cb (since 1 − 0 = 1 and Cb / 1 = Cb).
        assert!(approx(
            blend_channel(BlendMode::ColorDodge, 0.3, 1.0),
            1.0,
            1e-6
        ));
        assert!(approx(
            blend_channel(BlendMode::ColorDodge, 0.3, 0.0),
            0.3,
            1e-6
        ));
        // Cb = 0 → 0 (W3C clamp).
        assert!(approx(
            blend_channel(BlendMode::ColorDodge, 0.0, 0.5),
            0.0,
            1e-6
        ));
        // Mid case: 0.4 / (1 − 0.5) = 0.8.
        assert!(approx(
            blend_channel(BlendMode::ColorDodge, 0.4, 0.5),
            0.8,
            1e-6
        ));

        // Burn: Cs = 0 → 0, Cs = 1 → 1 − (1 − Cb) = Cb.
        assert!(approx(
            blend_channel(BlendMode::ColorBurn, 0.3, 0.0),
            0.0,
            1e-6
        ));
        assert!(approx(
            blend_channel(BlendMode::ColorBurn, 0.3, 1.0),
            0.3,
            1e-6
        ));
        // Cb = 1 → 1 (W3C clamp).
        assert!(approx(
            blend_channel(BlendMode::ColorBurn, 1.0, 0.5),
            1.0,
            1e-6
        ));
        // Mid case: 1 − (1 − 0.4)/0.5 = 1 − 1.2 → clamped to 0.
        assert!(approx(
            blend_channel(BlendMode::ColorBurn, 0.4, 0.5),
            0.0,
            1e-6
        ));
    }

    #[test]
    fn difference_and_exclusion() {
        assert!(approx(
            blend_channel(BlendMode::Difference, 0.7, 0.2),
            0.5,
            1e-6
        ));
        assert!(approx(
            blend_channel(BlendMode::Difference, 0.2, 0.7),
            0.5,
            1e-6
        ));
        // White inverts.
        assert!(approx(
            blend_channel(BlendMode::Difference, 0.3, 1.0),
            0.7,
            1e-6
        ));
        // Exclusion: 0.5 + 0.5 − 2·0.25 = 0.5.
        assert!(approx(
            blend_channel(BlendMode::Exclusion, 0.5, 0.5),
            0.5,
            1e-6
        ));
        // White inverts: 0.3 + 1 − 2·0.3 = 0.7.
        assert!(approx(
            blend_channel(BlendMode::Exclusion, 0.3, 1.0),
            0.7,
            1e-6
        ));
    }

    // --- blend_over end-to-end checks.

    #[test]
    fn blend_over_normal_matches_source_for_opaque_pair() {
        let cb = Rgba::opaque(10, 20, 30);
        let cs = Rgba::opaque(200, 100, 50);
        let r = blend_over(cb, cs, BlendMode::Normal);
        assert_eq!(r, cs);
    }

    #[test]
    fn blend_over_multiply_opaque_dark_times_white_keeps_dark() {
        // Opaque backdrop, opaque white source under Multiply: result
        // colour = B(Cb, Cs) = Cb · 1 = Cb. αr = 1.
        let cb = Rgba::opaque(80, 120, 200);
        let cs = Rgba::opaque(255, 255, 255);
        let r = blend_over(cb, cs, BlendMode::Multiply);
        assert_eq!(r, cb);
    }

    #[test]
    fn blend_over_screen_with_black_source_is_identity() {
        // Screen with black source: B(Cb, 0) = Cb; opaque source keeps
        // αr = 1 and Cr = Cb.
        let cb = Rgba::opaque(80, 120, 200);
        let cs = Rgba::opaque(0, 0, 0);
        let r = blend_over(cb, cs, BlendMode::Screen);
        assert_eq!(r, cb);
    }

    #[test]
    fn blend_over_difference_with_white_inverts_backdrop() {
        // White source under Difference inverts an opaque backdrop.
        let cb = Rgba::opaque(80, 120, 200);
        let cs = Rgba::opaque(255, 255, 255);
        let r = blend_over(cb, cs, BlendMode::Difference);
        assert_eq!(r, Rgba::opaque(175, 135, 55));
    }

    #[test]
    fn blend_over_transparent_source_returns_backdrop() {
        // αs = 0 → αr = αb, no source contribution.
        let cb = Rgba::opaque(80, 120, 200);
        let cs = Rgba::new(255, 0, 0, 0);
        let r = blend_over(cb, cs, BlendMode::Multiply);
        assert_eq!(r, cb);
    }

    #[test]
    fn blend_over_transparent_backdrop_returns_source() {
        // αb = 0 → Cr = (1 − αs/αr)·0 + (αs/αr)·[1·Cs + 0·B] = Cs.
        let cb = Rgba::new(0, 0, 0, 0);
        let cs = Rgba::opaque(200, 100, 50);
        let r = blend_over(cb, cs, BlendMode::Multiply);
        assert_eq!(r, cs);
    }

    #[test]
    fn blend_over_both_transparent_yields_transparent() {
        let cb = Rgba::new(0, 0, 0, 0);
        let cs = Rgba::new(0, 0, 0, 0);
        assert_eq!(
            blend_over(cb, cs, BlendMode::Multiply),
            Rgba::new(0, 0, 0, 0)
        );
    }
}
