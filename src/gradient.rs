//! Gradient evaluation — sample a [`LinearGradient`] or
//! [`RadialGradient`] at a point in pixel space and return the resulting
//! straight-alpha sRGB color.
//!
//! Supports SVG/PDF Pad / Reflect / Repeat spread methods.
//!
//! Two interpolation spaces are supported:
//!
//! * [`InterpolationSpace::Srgb`] — the default. Stops are interpolated
//!   linearly in non-linear sRGB space. Matches every SVG 1.1 / PDF
//!   renderer that doesn't explicitly opt into a color-managed pipeline.
//! * [`InterpolationSpace::LinearRgb`] — stops are converted to
//!   light-energy-linear sRGB tristimulus before interpolation, then
//!   back to non-linear sRGB for return. Implements SVG 2 §13.9
//!   `color-interpolation: linearRGB` (with the IEC 61966-2-1 transfer
//!   function — `(C + 0.055) / 1.055)^2.4` above 0.04045 / `0.0031308`,
//!   linear scale `× 12.92` / `÷ 12.92` below). Alpha is *not*
//!   gamma-encoded — it carries linear coverage and is interpolated as-is.
//!
//! Gradient coordinates are taken to be in the same pixel space the
//! caller is using for raster output (the Renderer applies the active
//! transform to the gradient endpoints before passing them in).

use oxideav_core::{LinearGradient, Point, RadialGradient, Rgba, SpreadMethod};

/// Color-space hint for gradient (and future per-pixel) interpolation.
///
/// Selects between SVG `color-interpolation: sRGB` (default — fast,
/// matches naive renderers) and `color-interpolation: linearRGB` —
/// physically-correct light blending that avoids the dark-band artefact
/// where complementary primaries cross.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InterpolationSpace {
    /// Interpolate in non-linear sRGB. The historical SVG 1.1 default
    /// (`color-interpolation: sRGB`); matches almost every SVG renderer
    /// shipped before 2010.
    #[default]
    Srgb,
    /// Interpolate in light-energy-linear sRGB (sRGB tristimulus values
    /// without the gamma transfer curve). The SVG 2 §13.9 / Filter
    /// Effects 1 §6.1 `color-interpolation: linearRGB` model. Eliminates
    /// the dark band that appears when complementary primaries (e.g.
    /// red↔green) interpolate as a midpoint in sRGB space.
    LinearRgb,
}

/// Pre-baked 256-entry stops look-up table.
///
/// Each entry is a straight-alpha sRGB color sampled at `t = i / 255`
/// for `i ∈ 0..=255`. Building the LUT pays the per-stop scan + (for
/// `LinearRgb`) the `powf` transfer-curve evaluation **once** per
/// gradient, regardless of the destination pixel count. Per-pixel
/// sampling then reduces to a clamp + index + load (no float math
/// inside the inner loop besides the `t` scaling that the geometry
/// step has to do anyway).
///
/// Practical impact: a 1024×1024 radial gradient with 4 stops in
/// `LinearRgb` mode goes from ~1M × (stops-window scan + 6× powf) to
/// 256 × (scan + powf) + 1M × (table lookup) — a ~3-5× reduction in
/// gradient-paint CPU time for non-trivial canvas sizes. The LUT is
/// also cache-friendly (1 KiB fits in L1), so the per-pixel hit is
/// dominated by `paint_at` dispatch + the over-blend, not gradient
/// evaluation.
///
/// The 256-entry resolution matches the destination byte depth: any
/// finer LUT would alias on the trip through the 8-bit per-channel
/// sRGB encoding at the end of compositing. For interpolation in
/// linear-light the LUT stores the *already-back-encoded* sRGB byte
/// values, so no per-pixel `linear_to_srgb` call is needed.
///
/// Built via [`StopsLut::build`]; queried via [`StopsLut::sample`].
#[derive(Debug, Clone)]
pub struct StopsLut {
    /// 256 RGBA entries, indexed by `(t.clamp(0,1) * 255).round() as
    /// u8`.
    entries: [Rgba; 256],
}

impl StopsLut {
    /// Build a LUT for the given stops in the requested interpolation
    /// space. `stops` must be sorted by `offset` ascending (the same
    /// contract the underlying [`sample_stops_in`] enforces).
    ///
    /// Empty stops produce an all-transparent LUT.
    pub fn build(stops: &[oxideav_core::GradientStop], space: InterpolationSpace) -> Self {
        let mut entries = [Rgba::new(0, 0, 0, 0); 256];
        if stops.is_empty() {
            return Self { entries };
        }
        for (i, slot) in entries.iter_mut().enumerate() {
            let t = i as f32 / 255.0;
            *slot = sample_stops_in(stops, t, space);
        }
        Self { entries }
    }

    /// Sample the LUT at `t ∈ [0, 1]`. Values outside the range are
    /// clamped (`Pad`-equivalent at the LUT boundary — the caller is
    /// expected to have applied [`apply_spread`] before this call,
    /// which already normalises `t` into `[0, 1]`).
    #[inline]
    pub fn sample(&self, t: f32) -> Rgba {
        if !t.is_finite() {
            return self.entries[0];
        }
        let scaled = (t * 255.0).round().clamp(0.0, 255.0) as usize;
        self.entries[scaled]
    }
}

/// Evaluate a linear gradient at pixel `(px, py)` using a pre-built
/// stops LUT. Geometry is identical to [`eval_linear_gradient_in`]; the
/// stops scan is replaced by [`StopsLut::sample`].
pub fn eval_linear_gradient_lut(g: &LinearGradient, px: f32, py: f32, lut: &StopsLut) -> Rgba {
    if g.stops.is_empty() {
        return Rgba::new(0, 0, 0, 0);
    }
    if g.stops.len() == 1 {
        return g.stops[0].color;
    }
    let dx = g.end.x - g.start.x;
    let dy = g.end.y - g.start.y;
    let denom = dx * dx + dy * dy;
    if denom <= 1e-12 {
        return g.stops.last().copied().unwrap().color;
    }
    let t = ((px - g.start.x) * dx + (py - g.start.y) * dy) / denom;
    lut.sample(apply_spread(t, g.spread))
}

/// Evaluate a radial gradient at pixel `(px, py)` using a pre-built
/// stops LUT. Geometry is identical to [`eval_radial_gradient_in`]; the
/// stops scan is replaced by [`StopsLut::sample`].
pub fn eval_radial_gradient_lut(g: &RadialGradient, px: f32, py: f32, lut: &StopsLut) -> Rgba {
    if g.stops.is_empty() {
        return Rgba::new(0, 0, 0, 0);
    }
    if g.stops.len() == 1 || g.radius <= 1e-12 {
        return g.stops[0].color;
    }
    let r = g.radius;
    let (fx, fy) = clamp_focal_inside_circle(g.focal.unwrap_or(g.center), g.center, r);
    let cdx = g.center.x - fx;
    let cdy = g.center.y - fy;
    let dx = px - fx;
    let dy = py - fy;
    let cd = cdx * dx + cdy * dy;
    let dd = dx * dx + dy * dy;
    let cc = cdx * cdx + cdy * cdy;
    let aa = cc - r * r;
    let t = if aa.abs() < 1e-12 {
        if cd.abs() < 1e-12 {
            dd.sqrt() / r
        } else {
            dd / (2.0 * cd)
        }
    } else {
        let disc = cd * cd - aa * dd;
        if disc < 0.0 {
            dd.sqrt() / r
        } else {
            (cd - disc.sqrt()) / aa
        }
    };
    lut.sample(apply_spread(t, g.spread))
}

/// Sample a linear gradient at pixel `(px, py)` in the sRGB interpolation
/// space (back-compat alias for [`eval_linear_gradient_in`]).
pub fn eval_linear_gradient(g: &LinearGradient, px: f32, py: f32) -> Rgba {
    eval_linear_gradient_in(g, px, py, InterpolationSpace::Srgb)
}

/// Sample a linear gradient at pixel `(px, py)` in the requested
/// interpolation space.
pub fn eval_linear_gradient_in(
    g: &LinearGradient,
    px: f32,
    py: f32,
    space: InterpolationSpace,
) -> Rgba {
    if g.stops.is_empty() {
        return Rgba::new(0, 0, 0, 0);
    }
    if g.stops.len() == 1 {
        return g.stops[0].color;
    }
    let dx = g.end.x - g.start.x;
    let dy = g.end.y - g.start.y;
    let denom = dx * dx + dy * dy;
    if denom <= 1e-12 {
        return g.stops.last().copied().unwrap().color;
    }
    let t = ((px - g.start.x) * dx + (py - g.start.y) * dy) / denom;
    sample_stops_in(&g.stops, apply_spread(t, g.spread), space)
}

/// Sample a radial gradient at pixel `(px, py)` in the sRGB
/// interpolation space (back-compat alias for [`eval_radial_gradient_in`]).
///
/// Implements the SVG 1.1 §13.2.4 ("Radial gradients") general
/// formula: the gradient is parameterised by `t`, where `t = 0` is
/// the focal point and `t = 1` traces the bounding circle (centre =
/// `g.center`, radius = `g.radius`). For each pixel we solve the
/// quadratic in `t`:
///
/// ```text
/// A * t^2 - 2 * (c·d) * t + (d·d) = 0
///   where  c = center - focal,  d = pixel - focal,
///          A = c·c - r^2.
/// ```
///
/// and pick the meaningful root (`t = ((c·d) - sqrt(Δ)) / A`). The
/// formula degenerates correctly when `focal == center` (gives the
/// classical `|P - centre| / r`).
///
/// SVG says the focal must lie *inside* the bounding circle. If the
/// caller supplies a focal on or outside the boundary we clamp it
/// onto the circle (just inside, by a tiny epsilon), matching the
/// browser-side normalisation step.
pub fn eval_radial_gradient(g: &RadialGradient, px: f32, py: f32) -> Rgba {
    eval_radial_gradient_in(g, px, py, InterpolationSpace::Srgb)
}

/// Sample a radial gradient at pixel `(px, py)` in the requested
/// interpolation space. See [`eval_radial_gradient`] for the geometric
/// derivation; the `space` argument only affects the per-stop
/// interpolation (geometric `t` is identical in both spaces).
pub fn eval_radial_gradient_in(
    g: &RadialGradient,
    px: f32,
    py: f32,
    space: InterpolationSpace,
) -> Rgba {
    if g.stops.is_empty() {
        return Rgba::new(0, 0, 0, 0);
    }
    if g.stops.len() == 1 || g.radius <= 1e-12 {
        return g.stops[0].color;
    }
    let r = g.radius;
    let (fx, fy) = clamp_focal_inside_circle(g.focal.unwrap_or(g.center), g.center, r);
    let cdx = g.center.x - fx;
    let cdy = g.center.y - fy;
    let dx = px - fx;
    let dy = py - fy;
    // c·d, d·d, c·c.
    let cd = cdx * dx + cdy * dy;
    let dd = dx * dx + dy * dy;
    let cc = cdx * cdx + cdy * cdy;
    let aa = cc - r * r; // <= 0 when focal is inside the circle
    let t = if aa.abs() < 1e-12 {
        // Focal on the boundary — quadratic collapses to linear:
        //   -2(c·d)*t + d·d = 0  =>  t = d·d / (2 c·d).
        if cd.abs() < 1e-12 {
            // c·d == 0: pixel sits on the line through focal
            // perpendicular to F→C, with no axis component → use
            // distance/radius as the centred fallback.
            (dd.sqrt()) / r
        } else {
            dd / (2.0 * cd)
        }
    } else {
        let disc = cd * cd - aa * dd;
        if disc < 0.0 {
            // Numerical underflow, possible only off the disc; clamp
            // to the centred fallback so the spread method still
            // runs against a meaningful t.
            dd.sqrt() / r
        } else {
            (cd - disc.sqrt()) / aa
        }
    };
    sample_stops_in(&g.stops, apply_spread(t, g.spread), space)
}

/// Pull `focal` strictly inside the bounding circle when it lies on
/// or outside the boundary. SVG normalises this so the gradient
/// equation always has a real positive root for points inside the
/// circle.
fn clamp_focal_inside_circle(focal: Point, center: Point, r: f32) -> (f32, f32) {
    let dx = focal.x - center.x;
    let dy = focal.y - center.y;
    let dist_sq = dx * dx + dy * dy;
    let r_minus_eps = (r - 1e-4).max(0.0);
    if dist_sq <= r_minus_eps * r_minus_eps {
        (focal.x, focal.y)
    } else {
        let dist = dist_sq.sqrt().max(1e-12);
        let scale = r_minus_eps / dist;
        (center.x + dx * scale, center.y + dy * scale)
    }
}

/// Map a parametric coordinate `t` through the spread method, producing
/// a value in `[0.0, 1.0]` ready for stop interpolation.
fn apply_spread(t: f32, spread: SpreadMethod) -> f32 {
    if t.is_nan() {
        return 0.0;
    }
    match spread {
        SpreadMethod::Pad => t.clamp(0.0, 1.0),
        SpreadMethod::Repeat => {
            let r = t - t.floor();
            if r < 0.0 {
                r + 1.0
            } else {
                r
            }
        }
        SpreadMethod::Reflect => {
            let two = (t * 0.5).floor() * 2.0;
            let local = t - two;
            // local is in [0, 2). Mirror around 1.
            if local <= 1.0 {
                local
            } else {
                2.0 - local
            }
        }
    }
}

/// Back-compat wrapper for the older `sample_stops` signature: always
/// interpolates in sRGB.
#[cfg(test)]
fn sample_stops(stops: &[oxideav_core::GradientStop], t: f32) -> Rgba {
    sample_stops_in(stops, t, InterpolationSpace::Srgb)
}

/// Linear interpolation across the `stops` array in the requested
/// interpolation space. `t` is in `[0, 1]`.
fn sample_stops_in(
    stops: &[oxideav_core::GradientStop],
    t: f32,
    space: InterpolationSpace,
) -> Rgba {
    if stops.is_empty() {
        return Rgba::new(0, 0, 0, 0);
    }
    if t <= stops[0].offset {
        return stops[0].color;
    }
    let last = stops[stops.len() - 1];
    if t >= last.offset {
        return last.color;
    }
    for w in stops.windows(2) {
        let (a, b) = (w[0], w[1]);
        if t >= a.offset && t <= b.offset {
            let span = (b.offset - a.offset).max(1e-9);
            let local = (t - a.offset) / span;
            return lerp_rgba(a.color, b.color, local, space);
        }
    }
    last.color
}

/// Linear interpolation between two straight-alpha sRGB colors in the
/// requested interpolation space.
fn lerp_rgba(a: Rgba, b: Rgba, t: f32, space: InterpolationSpace) -> Rgba {
    match space {
        InterpolationSpace::Srgb => {
            let l = |x: u8, y: u8| -> u8 {
                let lv = x as f32 + (y as f32 - x as f32) * t;
                lv.round().clamp(0.0, 255.0) as u8
            };
            Rgba::new(l(a.r, b.r), l(a.g, b.g), l(a.b, b.b), l(a.a, b.a))
        }
        InterpolationSpace::LinearRgb => {
            // SVG 2 §13.9 / IEC 61966-2-1: convert each RGB channel
            // through the gamma transfer function, lerp in the linear
            // domain, then transfer back.  Alpha is treated as light
            // coverage and stays linear throughout (matches CSS Filter
            // Effects 1 §6.1's note on `color-interpolation-filters`).
            let ar = srgb_to_linear(a.r);
            let ag = srgb_to_linear(a.g);
            let ab = srgb_to_linear(a.b);
            let br = srgb_to_linear(b.r);
            let bg = srgb_to_linear(b.g);
            let bb = srgb_to_linear(b.b);
            let lr = ar + (br - ar) * t;
            let lg = ag + (bg - ag) * t;
            let lb = ab + (bb - ab) * t;
            let aa = a.a as f32 + (b.a as f32 - a.a as f32) * t;
            Rgba::new(
                linear_to_srgb(lr),
                linear_to_srgb(lg),
                linear_to_srgb(lb),
                aa.round().clamp(0.0, 255.0) as u8,
            )
        }
    }
}

/// IEC 61966-2-1 sRGB → linear-light transfer function. Input is a
/// straight-alpha sRGB byte; output is the corresponding light-energy
/// value in `[0.0, 1.0]`. Matches the formula in SVG 2 §13.9
/// (`color-interpolation` linearRGB conversion).
#[inline]
fn srgb_to_linear(c: u8) -> f32 {
    let cs = c as f32 / 255.0;
    if cs <= 0.040_45 {
        cs / 12.92
    } else {
        ((cs + 0.055) / 1.055).powf(2.4)
    }
}

/// IEC 61966-2-1 linear-light → sRGB transfer function. Input is in
/// `[0.0, 1.0]` (clamped); output is the corresponding sRGB byte. The
/// inverse of [`srgb_to_linear`] and the formula required by SVG 2
/// §13.9 / IEC 61966-2-1 for the reverse transform.
#[inline]
fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let cs = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (cs * 255.0).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{GradientStop, Point};

    fn black_to_white_linear() -> LinearGradient {
        LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(10.0, 0.0),
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
            ],
            spread: SpreadMethod::Pad,
        }
    }

    #[test]
    fn linear_endpoints_yield_endpoint_colors() {
        let g = black_to_white_linear();
        assert_eq!(eval_linear_gradient(&g, 0.0, 0.0).r, 0);
        assert_eq!(eval_linear_gradient(&g, 10.0, 0.0).r, 255);
    }

    #[test]
    fn linear_midpoint_is_mid_gray() {
        let g = black_to_white_linear();
        let mid = eval_linear_gradient(&g, 5.0, 0.0);
        assert!((mid.r as i32 - 128).abs() <= 2);
    }

    #[test]
    fn linear_pad_clamps_outside_range() {
        let g = black_to_white_linear();
        // Past the end: should still be the end color.
        assert_eq!(eval_linear_gradient(&g, 100.0, 0.0).r, 255);
        // Behind the start: should still be the start color.
        assert_eq!(eval_linear_gradient(&g, -100.0, 0.0).r, 0);
    }

    #[test]
    fn linear_repeat_wraps() {
        let mut g = black_to_white_linear();
        g.spread = SpreadMethod::Repeat;
        // x=20 → t=2 → wraps to 0 → black.
        let c = eval_linear_gradient(&g, 20.0, 0.0);
        assert_eq!(c.r, 0);
    }

    #[test]
    fn linear_reflect_mirrors() {
        let mut g = black_to_white_linear();
        g.spread = SpreadMethod::Reflect;
        // x=15 → t=1.5 → reflects to 0.5 → mid-gray.
        let c = eval_linear_gradient(&g, 15.0, 0.0);
        assert!((c.r as i32 - 128).abs() <= 2);
    }

    #[test]
    fn radial_centre_is_first_stop() {
        let g = RadialGradient {
            center: Point::new(5.0, 5.0),
            radius: 5.0,
            focal: None,
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
            ],
            spread: SpreadMethod::Pad,
        };
        let c = eval_radial_gradient(&g, 5.0, 5.0);
        assert_eq!(c.r, 255);
        assert_eq!(c.b, 0);
    }

    #[test]
    fn srgb_to_linear_round_trip_within_byte_quantisation() {
        // Each sRGB byte → linear → sRGB byte should land back on the
        // original (within ±1 from float rounding).
        for c in 0u8..=255 {
            let l = srgb_to_linear(c);
            let back = linear_to_srgb(l);
            assert!(
                (c as i32 - back as i32).abs() <= 1,
                "round-trip mismatch at c={}: linear={} back={}",
                c,
                l,
                back
            );
        }
    }

    #[test]
    fn srgb_to_linear_endpoints_are_exact() {
        assert!(srgb_to_linear(0).abs() < 1e-7);
        // 255 → exactly 1.0 in light energy (within float epsilon).
        assert!((srgb_to_linear(255) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn linear_interpolation_midpoint_is_brighter_than_srgb() {
        // Black → white at t = 0.5:
        //   sRGB lerp   = 128 (midpoint between 0 and 255 in sRGB space)
        //   linear lerp = encode_srgb(0.5) ≈ 188
        // The linear-space midpoint is *brighter*; this is the canonical
        // demo of why linearRGB gradients look "right" — sRGB midpoints
        // are perceptually too dark because the encoded grey 128 is only
        // 22% reflectance.
        let g = LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(10.0, 0.0),
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
            ],
            spread: SpreadMethod::Pad,
        };
        let srgb_mid = eval_linear_gradient_in(&g, 5.0, 0.0, InterpolationSpace::Srgb);
        let lin_mid = eval_linear_gradient_in(&g, 5.0, 0.0, InterpolationSpace::LinearRgb);
        assert!(
            (srgb_mid.r as i32 - 128).abs() <= 2,
            "sRGB midpoint should be ~128, got {}",
            srgb_mid.r
        );
        assert!(
            lin_mid.r >= 185 && lin_mid.r <= 192,
            "linearRGB midpoint should be ~188, got {}",
            lin_mid.r
        );
    }

    #[test]
    fn linear_interpolation_red_to_green_avoids_dark_midpoint() {
        // The classic "ugly grey midpoint" sRGB artefact: red → green
        // through (188, 188, 0) in sRGB but through (~188, ~188, 0)
        // *brighter* in linearRGB. The point: both channels should be
        // at the linear midpoint of fully-on, not the sRGB midpoint of
        // fully-on (= 128 each).
        let g = LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(10.0, 0.0),
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(0, 255, 0)),
            ],
            spread: SpreadMethod::Pad,
        };
        let srgb_mid = eval_linear_gradient_in(&g, 5.0, 0.0, InterpolationSpace::Srgb);
        let lin_mid = eval_linear_gradient_in(&g, 5.0, 0.0, InterpolationSpace::LinearRgb);
        // sRGB midpoint: each channel at ~128.
        assert!(
            (srgb_mid.r as i32 - 128).abs() <= 2,
            "sRGB red channel at midpoint should be ~128, got {}",
            srgb_mid.r
        );
        // linearRGB midpoint: each channel at ~188 (encoded from 0.5
        // light energy).
        assert!(
            lin_mid.r >= 185 && lin_mid.r <= 192,
            "linearRGB red channel at midpoint should be ~188, got {}",
            lin_mid.r
        );
        assert!(
            lin_mid.g >= 185 && lin_mid.g <= 192,
            "linearRGB green channel at midpoint should be ~188, got {}",
            lin_mid.g
        );
        assert_eq!(lin_mid.b, 0, "blue should stay 0 throughout");
    }

    #[test]
    fn legacy_sample_stops_alias_matches_srgb_path() {
        // The deprecated `sample_stops` helper must still produce the
        // same output as the new explicit-sRGB form (so existing tests
        // keep their semantics).
        let stops = vec![
            GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
        ];
        for t in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let legacy = sample_stops(&stops, t);
            let explicit = sample_stops_in(&stops, t, InterpolationSpace::Srgb);
            assert_eq!(legacy, explicit, "mismatch at t={}", t);
        }
    }

    #[test]
    fn stops_lut_endpoint_exact() {
        let stops = vec![
            GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
        ];
        let lut = StopsLut::build(&stops, InterpolationSpace::Srgb);
        assert_eq!(lut.sample(0.0), Rgba::opaque(255, 0, 0));
        assert_eq!(lut.sample(1.0), Rgba::opaque(0, 0, 255));
    }

    #[test]
    fn stops_lut_empty_stops_returns_transparent() {
        let lut = StopsLut::build(&[], InterpolationSpace::Srgb);
        // Every entry must be (0, 0, 0, 0) — the documented behaviour
        // for an empty stop set.
        for t in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(lut.sample(t), Rgba::new(0, 0, 0, 0));
        }
    }

    #[test]
    fn stops_lut_radial_eval_lut_matches_per_pixel() {
        let g = RadialGradient {
            center: Point::new(0.0, 0.0),
            radius: 10.0,
            focal: None,
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
            ],
            spread: SpreadMethod::Pad,
        };
        let lut = StopsLut::build(&g.stops, InterpolationSpace::Srgb);
        // Sample a 21×21 grid spanning the gradient's bounding circle
        // and assert per-pixel and LUT outputs agree within ±1 LSB.
        for iy in -10..=10 {
            for ix in -10..=10 {
                let px = ix as f32;
                let py = iy as f32;
                let p = eval_radial_gradient(&g, px, py);
                let q = eval_radial_gradient_lut(&g, px, py, &lut);
                for (pa, qa) in [(p.r, q.r), (p.g, q.g), (p.b, q.b), (p.a, q.a)] {
                    assert!((pa as i32 - qa as i32).abs() <= 1);
                }
            }
        }
    }

    #[test]
    fn stops_lut_linear_eval_lut_matches_per_pixel() {
        let g = black_to_white_linear();
        let lut = StopsLut::build(&g.stops, InterpolationSpace::Srgb);
        for x in 0..=10 {
            let p = eval_linear_gradient(&g, x as f32, 0.0);
            let q = eval_linear_gradient_lut(&g, x as f32, 0.0, &lut);
            for (pa, qa) in [(p.r, q.r), (p.g, q.g), (p.b, q.b), (p.a, q.a)] {
                assert!((pa as i32 - qa as i32).abs() <= 1);
            }
        }
    }

    #[test]
    fn radial_at_radius_is_last_stop() {
        let g = RadialGradient {
            center: Point::new(0.0, 0.0),
            radius: 10.0,
            focal: None,
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
            ],
            spread: SpreadMethod::Pad,
        };
        let c = eval_radial_gradient(&g, 10.0, 0.0);
        assert_eq!(c.b, 255);
        assert_eq!(c.r, 0);
    }
}
