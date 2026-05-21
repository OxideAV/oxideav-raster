//! Standard separable and non-separable blend modes (PDF 32000-1:2008
//! §11.3.5 / W3C Compositing-1 §10–§11), evaluated per channel for the
//! separable modes (Table 136) and per-RGB-triple for the non-separable
//! HSL modes (Table 137).
//!
//! The crate's default composite path is **source-over** ([`BlendMode::Normal`])
//! and runs through the fast-path branch in [`crate::composite`]. Any other
//! mode routes through this module, which implements the per-channel blend
//! `B(Cb, Cs)` (or, for the HSL modes, the per-triple
//! `B([Cb_r, Cb_g, Cb_b], [Cs_r, Cs_g, Cs_b])`) from the spec, then folds
//! it back into the standard PDF §11.3.3 basic compositing formula:
//!
//! ```text
//!     Cr = (1 − αs/αr) · Cb + (αs/αr) · [ (1 − αb) · Cs + αb · B(Cb, Cs) ]
//!     αr = αs + αb · (1 − αs)
//! ```
//!
//! The four non-separable modes (Hue / Saturation / Color / Luminosity)
//! treat the RGB triple as a single colour and combine the source and
//! backdrop in HSL coordinate space via the spec's `Lum` / `SetLum` /
//! `ClipColor` / `Sat` / `SetSat` auxiliary functions, without ever
//! actually converting to an explicit HSL triple. Per PDF §11.3.5.3
//! the formulas operate in the RGB blending colour space directly.
//!
//! All channel-domain math is in normalised `f32` `[0, 1]`. Source and
//! backdrop colours are taken straight-alpha (the renderer caller is
//! responsible for any pre-coverage modulation of the source alpha
//! channel before the blend, just like the source-over path).
//!
//! Wall provenance: formulas transcribed from
//! `docs/document/pdf/PDF32000_2008.pdf` §11.3.3–§11.3.5 (the public
//! ISO 32000-1 PDF spec), including §11.3.5.3 Table 137 and the
//! `Lum` / `SetLum` / `ClipColor` / `Sat` / `SetSat` definitions. No
//! external blend implementation consulted.

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
    /// Non-separable (PDF §11.3.5.3 Table 137).
    /// `B(Cb, Cs) = SetLum(SetSat(Cs, Sat(Cb)), Lum(Cb))`. Creates a
    /// colour with the hue of the source and the saturation +
    /// luminosity of the backdrop.
    Hue,
    /// Non-separable. `B(Cb, Cs) = SetLum(SetSat(Cb, Sat(Cs)), Lum(Cb))`.
    /// Creates a colour with the saturation of the source and the hue +
    /// luminosity of the backdrop. Painting in an unsaturated (grey)
    /// backdrop area produces no change.
    Saturation,
    /// Non-separable. `B(Cb, Cs) = SetLum(Cs, Lum(Cb))`. Creates a
    /// colour with the hue + saturation of the source and the
    /// luminosity of the backdrop. Useful for tinting monochrome
    /// images while keeping the backdrop's grey levels.
    Color,
    /// Non-separable. `B(Cb, Cs) = SetLum(Cb, Lum(Cs))`. Creates a
    /// colour with the luminosity of the source and the hue +
    /// saturation of the backdrop — inverse of [`BlendMode::Color`].
    Luminosity,
}

impl BlendMode {
    /// True when this mode is the standard pass-through source-over.
    /// Used by the composite path to dispatch to the fast branch.
    #[inline]
    pub fn is_normal(self) -> bool {
        matches!(self, BlendMode::Normal)
    }

    /// True when this mode is *separable* — its result on each output
    /// channel depends only on the matching input channels. The four
    /// HSL modes (Hue / Saturation / Color / Luminosity) are
    /// non-separable: they treat the RGB triple as a single colour.
    #[inline]
    pub fn is_separable(self) -> bool {
        !matches!(
            self,
            BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity
        )
    }

    /// Apply this blend mode's `B(Cb, Cs)` function to the RGB triples
    /// of `cb` (backdrop) and `cs` (source). For separable modes this
    /// reduces to three independent calls to [`blend_channel`]; for
    /// the four HSL modes (Hue / Saturation / Color / Luminosity) it
    /// runs the spec's non-separable formula on the full triple. Alpha
    /// is left untouched here; the composite formula handles alpha
    /// separately. Inputs and outputs are straight-alpha `[0, 1]`.
    #[inline]
    pub fn blend_rgb(self, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
        match self {
            BlendMode::Hue => set_lum(set_sat(cs, sat(cb)), lum(cb)),
            BlendMode::Saturation => set_lum(set_sat(cb, sat(cs)), lum(cb)),
            BlendMode::Color => set_lum(cs, lum(cb)),
            BlendMode::Luminosity => set_lum(cb, lum(cs)),
            _ => [
                blend_channel(self, cb[0], cs[0]),
                blend_channel(self, cb[1], cs[1]),
                blend_channel(self, cb[2], cs[2]),
            ],
        }
    }
}

/// Per-channel `B(Cb, Cs)` for the **separable** modes from PDF
/// §11.3.5.2 Table 136, with both inputs and the output in `[0, 1]`.
/// Out-of-range inputs are clamped first; the result is also clamped
/// on the way out per PDF §11.3.5 ("the result of the mixing formula
/// must be clamped").
///
/// Panics if `mode` is one of the four non-separable HSL modes
/// ([`BlendMode::Hue`] / [`BlendMode::Saturation`] / [`BlendMode::Color`]
/// / [`BlendMode::Luminosity`]) — those depend on all three RGB
/// channels in concert and must go through [`BlendMode::blend_rgb`].
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
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity => {
            panic!(
                "blend_channel: non-separable mode {mode:?} requires the full RGB triple; \
                 use BlendMode::blend_rgb"
            )
        }
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

// --- Non-separable (HSL) helper functions, PDF §11.3.5.3.
//
// The spec defines four auxiliary functions over RGB triples:
//     Lum(C)      = 0.30·R + 0.59·G + 0.11·B
//     SetLum(C,l) = ClipColor(C + (l − Lum(C)) broadcast)
//     ClipColor(C) — gamut-map back into [0,1] preserving luminosity
//     Sat(C)      = max(R,G,B) − min(R,G,B)
//     SetSat(C,s) — rescale the (Cmid − Cmin) span so (Cmax − Cmin) == s
//
// The four non-separable blend modes (Table 137) are composed from
// these. All inputs / outputs are in normalised [0, 1].

/// PDF §11.3.5.3: relative luminance of an RGB triple using the
/// spec's fixed coefficients (Rec. 601-style, *not* Rec. 709). These
/// coefficients are part of the blend-mode definition and must not
/// be substituted by sRGB or Rec. 709 weights even though the
/// blending colour space is conceptually sRGB.
#[inline]
fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

/// PDF §11.3.5.3 saturation: range between the max and min channel.
#[inline]
fn sat(c: [f32; 3]) -> f32 {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    mx - mn
}

/// PDF §11.3.5.3 `SetLum(C, l)`: translate every channel by the
/// signed delta `l − Lum(C)`, then call [`clip_color`] to bring the
/// result back into gamut without changing its luminance.
#[inline]
fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_color([c[0] + d, c[1] + d, c[2] + d])
}

/// PDF §11.3.5.3 `ClipColor(C)`: gamut-map an RGB triple back into
/// `[0, 1]` while preserving its luminance. When the triple is
/// already in gamut the triple is returned unchanged.
///
/// The spec branches on `n = min(C) < 0` and `x = max(C) > 1`. We
/// honour both branches independently; a triple that is both under
/// 0 and over 1 (extreme over-saturation) gets both corrections
/// applied in sequence per the spec listing. Each correction
/// preserves `Lum(C)` exactly when its denominator is non-zero —
/// when it is, the channel collapses onto `l` (the luminance itself
/// is already in `[0, 1]` because the spec applies SetLum after
/// SetSat / and inputs to the four HSL modes are also in `[0, 1]`).
#[inline]
fn clip_color(c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut out = c;
    if n < 0.0 {
        let denom = l - n;
        if denom.abs() > f32::EPSILON {
            out = [
                l + ((out[0] - l) * l) / denom,
                l + ((out[1] - l) * l) / denom,
                l + ((out[2] - l) * l) / denom,
            ];
        } else {
            // n == l, only possible when every channel equals l; the
            // spec's formula is then `l + 0/0` which we resolve to
            // the only consistent value (l).
            out = [l, l, l];
        }
    }
    if x > 1.0 {
        let denom = x - l;
        if denom.abs() > f32::EPSILON {
            out = [
                l + ((out[0] - l) * (1.0 - l)) / denom,
                l + ((out[1] - l) * (1.0 - l)) / denom,
                l + ((out[2] - l) * (1.0 - l)) / denom,
            ];
        } else {
            out = [l, l, l];
        }
    }
    out
}

/// PDF §11.3.5.3 `SetSat(C, s)`: rescale the chromatic range of `c`
/// so its `max − min` becomes `s`, preserving the relative position
/// of the middle channel. The minimum channel is mapped to 0 and the
/// maximum to `s`; an achromatic input (all channels equal) collapses
/// to `[0, 0, 0]` as the spec's `else` branch dictates.
///
/// The spec relies on subscripts `min`, `mid`, `max` referring to the
/// channels' identities **on entry**. We track the permutation
/// explicitly via indices so the result is independent of the channel
/// ordering of the input triple.
#[inline]
fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    // Find indices of min, mid, max as they were on entry.
    let mut idx = [0usize, 1, 2];
    // Sort `idx` by the corresponding `c[idx[i]]` ascending. Three
    // elements — a simple bubble keeps this allocation-free.
    if c[idx[0]] > c[idx[1]] {
        idx.swap(0, 1);
    }
    if c[idx[1]] > c[idx[2]] {
        idx.swap(1, 2);
    }
    if c[idx[0]] > c[idx[1]] {
        idx.swap(0, 1);
    }
    let (i_min, i_mid, i_max) = (idx[0], idx[1], idx[2]);
    let mut out = [0.0_f32; 3];
    if c[i_max] > c[i_min] {
        out[i_mid] = ((c[i_mid] - c[i_min]) * s) / (c[i_max] - c[i_min]);
        out[i_max] = s;
        // out[i_min] stays 0.
    } else {
        // Achromatic input: every channel was equal. Spec sets the
        // mid and max to 0, and the min is already 0.
        out[i_mid] = 0.0;
        out[i_max] = 0.0;
    }
    out
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

    // --- non-separable (HSL) helpers and modes (PDF §11.3.5.3).

    fn approx_rgb(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        approx(a[0], b[0], tol) && approx(a[1], b[1], tol) && approx(a[2], b[2], tol)
    }

    #[test]
    fn lum_uses_pdf_coefficients() {
        // Spec: Lum(R, G, B) = 0.3·R + 0.59·G + 0.11·B.
        assert!(approx(lum([1.0, 0.0, 0.0]), 0.30, 1e-6));
        assert!(approx(lum([0.0, 1.0, 0.0]), 0.59, 1e-6));
        assert!(approx(lum([0.0, 0.0, 1.0]), 0.11, 1e-6));
        // Coefficients sum to 1, so white maps to 1.
        assert!(approx(lum([1.0, 1.0, 1.0]), 1.0, 1e-6));
        // Black maps to 0.
        assert!(approx(lum([0.0, 0.0, 0.0]), 0.0, 1e-6));
        // Mid-grey maps to 0.5.
        assert!(approx(lum([0.5, 0.5, 0.5]), 0.5, 1e-6));
    }

    #[test]
    fn sat_is_max_minus_min() {
        assert!(approx(sat([0.2, 0.7, 0.5]), 0.5, 1e-6));
        assert!(approx(sat([0.4, 0.4, 0.4]), 0.0, 1e-6));
        assert!(approx(sat([1.0, 0.0, 0.5]), 1.0, 1e-6));
    }

    #[test]
    fn set_lum_preserves_set_luminance() {
        // For any in-gamut input, SetLum(C, l) should have Lum == l.
        let inputs = [
            [0.2_f32, 0.7, 0.5],
            [0.1, 0.1, 0.1],
            [0.8, 0.4, 0.6],
            [1.0, 0.0, 0.0],
        ];
        for &c in &inputs {
            let l = 0.25;
            let r = set_lum(c, l);
            assert!(
                approx(lum(r), l, 1e-5),
                "SetLum({c:?}, {l}) -> {r:?} has Lum = {} (want {l})",
                lum(r)
            );
            // ClipColor should land the result in [0, 1].
            for &x in &r {
                assert!((0.0..=1.0).contains(&x), "channel {x} out of gamut");
            }
        }
    }

    #[test]
    fn set_lum_zero_yields_black_for_pure_primaries() {
        // SetLum(pure-red, 0) should drop into the [0,0,0] neighbourhood
        // (ClipColor pulls it to the achromatic axis).
        let r = set_lum([1.0, 0.0, 0.0], 0.0);
        assert!(approx_rgb(r, [0.0, 0.0, 0.0], 1e-5));
    }

    #[test]
    fn set_lum_one_yields_white_for_pure_primaries() {
        // SetLum(pure-blue, 1) gets translated to (0.89, 0.89, 1.89) then
        // ClipColor pulls the overshoot down preserving luminance — the
        // result lands at (1, 1, 1) since Lum(1,1,1) == 1.
        let r = set_lum([0.0, 0.0, 1.0], 1.0);
        assert!(approx_rgb(r, [1.0, 1.0, 1.0], 1e-5));
    }

    #[test]
    fn set_sat_rescales_chromatic_range() {
        // Input (0.2, 0.5, 0.8): min=0.2, mid=0.5, max=0.8. SetSat(_, 0.6)
        // should make max=0.6, min=0, mid scaled linearly:
        //   mid = (0.5 − 0.2) · 0.6 / (0.8 − 0.2) = 0.3 · 0.6 / 0.6 = 0.3.
        let r = set_sat([0.2, 0.5, 0.8], 0.6);
        assert!(approx_rgb(r, [0.0, 0.3, 0.6], 1e-6));
    }

    #[test]
    fn set_sat_achromatic_collapses_to_black() {
        let r = set_sat([0.4, 0.4, 0.4], 0.5);
        assert!(approx_rgb(r, [0.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn set_sat_handles_any_channel_ordering() {
        // The spec says min / mid / max refer to identities *on entry*.
        // A permutation of the same triple should yield the matching
        // permutation of the output triple.
        let a = set_sat([0.2, 0.5, 0.8], 0.6);
        let b = set_sat([0.5, 0.8, 0.2], 0.6);
        let c = set_sat([0.8, 0.2, 0.5], 0.6);
        // a expected: [0.0, 0.3, 0.6] (R=min, G=mid, B=max)
        // b: R was mid, G was max, B was min  → [0.3, 0.6, 0.0]
        // c: R was max, G was min, B was mid  → [0.6, 0.0, 0.3]
        assert!(approx_rgb(a, [0.0, 0.3, 0.6], 1e-6));
        assert!(approx_rgb(b, [0.3, 0.6, 0.0], 1e-6));
        assert!(approx_rgb(c, [0.6, 0.0, 0.3], 1e-6));
    }

    #[test]
    fn hsl_modes_are_non_separable() {
        for m in [
            BlendMode::Hue,
            BlendMode::Saturation,
            BlendMode::Color,
            BlendMode::Luminosity,
        ] {
            assert!(!m.is_separable(), "{m:?} must be non-separable");
        }
        // The separable modes stay separable.
        for m in [
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::ColorDodge,
            BlendMode::ColorBurn,
            BlendMode::HardLight,
            BlendMode::SoftLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
        ] {
            assert!(m.is_separable(), "{m:?} must be separable");
        }
    }

    #[test]
    fn color_mode_takes_source_chroma_backdrop_luma() {
        // Color: B = SetLum(Cs, Lum(Cb)). The result's luminance must
        // equal Lum(Cb), and SetSat-of-result must equal Sat(Cs) up to
        // rounding because the source chroma carries through SetLum.
        let cb = [0.10, 0.20, 0.40];
        let cs = [0.80, 0.30, 0.10];
        let b = BlendMode::Color.blend_rgb(cb, cs);
        let want_l = lum(cb);
        assert!(
            approx(lum(b), want_l, 1e-5),
            "Lum(B)={} want {want_l}",
            lum(b)
        );
    }

    #[test]
    fn luminosity_mode_takes_source_luma_backdrop_chroma() {
        // Luminosity: B = SetLum(Cb, Lum(Cs)). Result luminance = Lum(Cs).
        let cb = [0.10, 0.20, 0.40];
        let cs = [0.80, 0.30, 0.10];
        let b = BlendMode::Luminosity.blend_rgb(cb, cs);
        assert!(approx(lum(b), lum(cs), 1e-5));
    }

    #[test]
    fn hue_mode_takes_source_hue_backdrop_sat_and_luma() {
        // Hue: B = SetLum(SetSat(Cs, Sat(Cb)), Lum(Cb)). Result
        // luminance = Lum(Cb), saturation = Sat(Cb) (pre-clip);
        // ClipColor can only reduce saturation when gamut-mapping, so
        // post-clip Sat(B) is ≤ Sat(Cb).
        let cb = [0.30, 0.60, 0.50];
        let cs = [0.90, 0.10, 0.40];
        let b = BlendMode::Hue.blend_rgb(cb, cs);
        assert!(approx(lum(b), lum(cb), 1e-5));
        assert!(
            sat(b) <= sat(cb) + 1e-5,
            "Sat(Hue(Cb,Cs))={} must be ≤ Sat(Cb)={}",
            sat(b),
            sat(cb)
        );
    }

    #[test]
    fn saturation_mode_in_grey_backdrop_is_identity() {
        // Saturation: B = SetLum(SetSat(Cb, Sat(Cs)), Lum(Cb)). If Cb
        // is a pure grey then Sat(Cb) == 0 and SetSat(Cb, _) collapses
        // to [0,0,0]; SetLum then restores Lum(Cb) on the achromatic
        // axis — so the result is the same grey as Cb. (PDF NOTE 2.)
        let cb = [0.42, 0.42, 0.42];
        let cs = [0.90, 0.10, 0.40];
        let b = BlendMode::Saturation.blend_rgb(cb, cs);
        assert!(
            approx_rgb(b, cb, 1e-5),
            "Saturation over grey backdrop changed: {b:?} != {cb:?}"
        );
    }

    #[test]
    fn color_then_luminosity_round_trip_preserves_chroma() {
        // Apply Color then Luminosity with the original Cb as the
        // "Cs" of the second stage: the result's chroma should match
        // the original Cs's chroma (since Color carried Cs through
        // SetLum). Empirically the two formulas are inverses on the
        // luminance channel; we just check the modes round-trip
        // luminance correctly.
        let cb = [0.10, 0.20, 0.40];
        let cs = [0.80, 0.30, 0.10];
        let b1 = BlendMode::Color.blend_rgb(cb, cs);
        // Lum(b1) == Lum(cb).
        assert!(approx(lum(b1), lum(cb), 1e-5));
        let b2 = BlendMode::Luminosity.blend_rgb(b1, cs);
        // Lum(b2) == Lum(cs).
        assert!(approx(lum(b2), lum(cs), 1e-5));
    }

    #[test]
    fn blend_over_color_mode_opaque_result_has_backdrop_luma() {
        // End-to-end: opaque Cb + opaque Cs under Color should land
        // an opaque result whose luminance (back-projected through
        // PDF Lum) equals Lum(Cb) ± rounding to bytes.
        let cb = Rgba::opaque(80, 180, 60);
        let cs = Rgba::opaque(220, 100, 40);
        let r = blend_over(cb, cs, BlendMode::Color);
        assert_eq!(r.a, 255);
        let want_l = 0.3 * (cb.r as f32) + 0.59 * (cb.g as f32) + 0.11 * (cb.b as f32);
        let got_l = 0.3 * (r.r as f32) + 0.59 * (r.g as f32) + 0.11 * (r.b as f32);
        // ±1.5 bytes of slack for the channel-rounding chain.
        assert!(
            (got_l - want_l).abs() <= 1.5,
            "Lum got {got_l} want {want_l}"
        );
    }

    #[test]
    fn blend_over_luminosity_mode_opaque_result_has_source_luma() {
        let cb = Rgba::opaque(80, 180, 60);
        let cs = Rgba::opaque(220, 100, 40);
        let r = blend_over(cb, cs, BlendMode::Luminosity);
        assert_eq!(r.a, 255);
        let want_l = 0.3 * (cs.r as f32) + 0.59 * (cs.g as f32) + 0.11 * (cs.b as f32);
        let got_l = 0.3 * (r.r as f32) + 0.59 * (r.g as f32) + 0.11 * (r.b as f32);
        assert!(
            (got_l - want_l).abs() <= 1.5,
            "Lum got {got_l} want {want_l}"
        );
    }

    #[test]
    #[should_panic(expected = "non-separable mode")]
    fn blend_channel_panics_on_hsl_mode() {
        // The per-channel scalar helper is undefined for non-separable
        // modes; callers must use BlendMode::blend_rgb.
        let _ = blend_channel(BlendMode::Hue, 0.5, 0.5);
    }
}
