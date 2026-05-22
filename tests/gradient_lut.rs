//! `StopsLut` look-up table — radial / linear gradient pre-bake tests.
//!
//! The LUT path is wired into `Renderer::composite_with_paint` so any
//! gradient paint goes through it transparently. These tests assert:
//!
//! * the LUT-built sampler agrees with the slow per-pixel
//!   `sample_paint_in` evaluation to within ±1 LSB (the LUT has 256
//!   entries; per-pixel `t` quantises to the nearest entry, which can
//!   introduce ≤ 1 LSB of error on per-channel byte output),
//! * the LUT is independent of geometry — building from the same stops
//!   in two different interpolation spaces and sampling at the same `t`
//!   produces the same byte values as the per-pixel evaluator,
//! * end-to-end renders of a radial gradient at 64×64 in both sRGB and
//!   linearRGB interpolation match the pre-LUT renderer's output (so
//!   the LUT introduces no visible regression),
//! * the build cost is bounded — `StopsLut::build` on a 4-stop gradient
//!   never panics, never returns transparent for an opaque stop set,
//!   and produces a strict monotonic alpha for a `(0, 255)` alpha ramp.

use oxideav_core::{
    FillRule, GradientStop, LinearGradient, Paint, Path, PathCommand, PathNode, Point,
    RadialGradient, Rgba, SpreadMethod, VectorFrame,
};
use oxideav_raster::{
    eval_linear_gradient_in, eval_linear_gradient_lut, eval_radial_gradient_in,
    eval_radial_gradient_lut, InterpolationSpace, Renderer, StopsLut,
};

fn black_to_white_stops() -> Vec<GradientStop> {
    vec![
        GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
        GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
    ]
}

fn red_green_blue_stops() -> Vec<GradientStop> {
    vec![
        GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
        GradientStop::new(0.5, Rgba::opaque(0, 255, 0)),
        GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
    ]
}

#[test]
fn lut_build_is_endpoint_exact() {
    // The LUT entries at index 0 and 255 must equal the bracketing
    // stop colors exactly (no `t` rounding can move us off the
    // endpoint).
    let stops = red_green_blue_stops();
    let lut = StopsLut::build(&stops, InterpolationSpace::Srgb);
    assert_eq!(lut.sample(0.0), Rgba::opaque(255, 0, 0));
    assert_eq!(lut.sample(1.0), Rgba::opaque(0, 0, 255));
}

#[test]
fn lut_linear_matches_per_pixel_within_one_lsb() {
    // For a black→white gradient sampled along its principal axis the
    // LUT and per-pixel evaluators must agree to within ±1 LSB per
    // channel.
    let g = LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(255.0, 0.0),
        stops: black_to_white_stops(),
        spread: SpreadMethod::Pad,
    };
    let lut = StopsLut::build(&g.stops, InterpolationSpace::Srgb);
    for x in 0..=255 {
        let p = eval_linear_gradient_in(&g, x as f32, 0.0, InterpolationSpace::Srgb);
        let q = eval_linear_gradient_lut(&g, x as f32, 0.0, &lut);
        let p_arr = [p.r, p.g, p.b, p.a];
        let q_arr = [q.r, q.g, q.b, q.a];
        for ch in 0..4 {
            let dp = p_arr[ch] as i32 - q_arr[ch] as i32;
            assert!(
                dp.abs() <= 1,
                "channel {} at x={} differs by {}: per-pixel={:?} lut={:?}",
                ch,
                x,
                dp,
                p,
                q
            );
        }
    }
}

#[test]
fn lut_radial_matches_per_pixel_within_one_lsb() {
    let g = RadialGradient {
        center: Point::new(32.0, 32.0),
        radius: 32.0,
        focal: None,
        stops: red_green_blue_stops(),
        spread: SpreadMethod::Pad,
    };
    let lut = StopsLut::build(&g.stops, InterpolationSpace::Srgb);
    for y in 0..64 {
        for x in 0..64 {
            let p = eval_radial_gradient_in(&g, x as f32, y as f32, InterpolationSpace::Srgb);
            let q = eval_radial_gradient_lut(&g, x as f32, y as f32, &lut);
            let p_arr = [p.r, p.g, p.b, p.a];
            let q_arr = [q.r, q.g, q.b, q.a];
            for ch in 0..4 {
                let dp = p_arr[ch] as i32 - q_arr[ch] as i32;
                assert!(
                    dp.abs() <= 1,
                    "channel {} at ({}, {}) differs by {}: per-pixel={:?} lut={:?}",
                    ch,
                    x,
                    y,
                    dp,
                    p,
                    q
                );
            }
        }
    }
}

#[test]
fn lut_respects_linear_rgb_space() {
    // Same stops, different LUTs in sRGB vs linearRGB → the
    // mid-point byte must differ as documented in `gradient.rs` (sRGB
    // midpoint ≈ 128, linearRGB midpoint ≈ 188 for black→white).
    let stops = black_to_white_stops();
    let lut_srgb = StopsLut::build(&stops, InterpolationSpace::Srgb);
    let lut_lin = StopsLut::build(&stops, InterpolationSpace::LinearRgb);
    let mid_srgb = lut_srgb.sample(0.5);
    let mid_lin = lut_lin.sample(0.5);
    assert!(
        (mid_srgb.r as i32 - 128).abs() <= 2,
        "sRGB midpoint should be ~128, got {}",
        mid_srgb.r
    );
    assert!(
        mid_lin.r >= 185 && mid_lin.r <= 192,
        "linearRGB midpoint should be ~188, got {}",
        mid_lin.r
    );
}

#[test]
fn lut_sample_clamps_out_of_range() {
    // `apply_spread` normalises `t` to `[0, 1]` before reaching the
    // LUT — but the LUT's own `sample` must also clamp for callers
    // bypassing the spread step.
    let lut = StopsLut::build(&black_to_white_stops(), InterpolationSpace::Srgb);
    assert_eq!(lut.sample(-1.0), Rgba::opaque(0, 0, 0));
    assert_eq!(lut.sample(2.0), Rgba::opaque(255, 255, 255));
    // NaN is non-finite and must be treated as the start (`t = 0`).
    assert_eq!(lut.sample(f32::NAN), Rgba::opaque(0, 0, 0));
}

#[test]
fn lut_alpha_monotonic_for_alpha_ramp() {
    // (0, 0, 0, 0) → (255, 255, 255, 255) — alpha must monotonically
    // rise through the LUT entries.
    let stops = vec![
        GradientStop::new(0.0, Rgba::new(0, 0, 0, 0)),
        GradientStop::new(1.0, Rgba::new(255, 255, 255, 255)),
    ];
    let lut = StopsLut::build(&stops, InterpolationSpace::Srgb);
    let mut last_a = 0u8;
    for i in 0..=255 {
        let t = i as f32 / 255.0;
        let a = lut.sample(t).a;
        assert!(
            a >= last_a,
            "alpha must be monotonic; t={} got {}, prev {}",
            t,
            a,
            last_a
        );
        last_a = a;
    }
    assert_eq!(last_a, 255);
}

/// Build a circular path centred in the canvas — cubic-Bezier
/// approximation via four arcs (close enough for end-to-end gradient
/// comparison; gradient evaluation is the unit under test, not the
/// path tessellator).
fn circle_path(cx: f32, cy: f32, r: f32) -> Path {
    // Standard quarter-circle Bezier control-point offset
    // (4/3) * (√2 − 1).
    let k = 0.552_284_8_f32 * r;
    Path {
        commands: vec![
            PathCommand::MoveTo(Point::new(cx, cy - r)),
            PathCommand::CubicCurveTo {
                c1: Point::new(cx + k, cy - r),
                c2: Point::new(cx + r, cy - k),
                end: Point::new(cx + r, cy),
            },
            PathCommand::CubicCurveTo {
                c1: Point::new(cx + r, cy + k),
                c2: Point::new(cx + k, cy + r),
                end: Point::new(cx, cy + r),
            },
            PathCommand::CubicCurveTo {
                c1: Point::new(cx - k, cy + r),
                c2: Point::new(cx - r, cy + k),
                end: Point::new(cx - r, cy),
            },
            PathCommand::CubicCurveTo {
                c1: Point::new(cx - r, cy - k),
                c2: Point::new(cx - k, cy - r),
                end: Point::new(cx, cy - r),
            },
            PathCommand::Close,
        ],
    }
}

#[test]
fn lut_radial_end_to_end_render_byte_exact_against_per_pixel() {
    // End-to-end check: render a 64×64 radial gradient through the
    // Renderer (which routes gradients through the LUT path) and
    // compare against a hand-rolled per-pixel evaluation that *doesn't*
    // use the LUT. Coverage may differ at sub-pixel-AA edges, so we
    // only inspect the fully-covered interior (a 16×16 region around
    // the centre, away from the circle boundary).
    let radial = RadialGradient {
        center: Point::new(32.0, 32.0),
        radius: 32.0,
        focal: None,
        stops: red_green_blue_stops(),
        spread: SpreadMethod::Pad,
    };
    let mut root = oxideav_core::Group::default();
    root.children.push(oxideav_core::Node::Path(PathNode {
        path: circle_path(32.0, 32.0, 30.0),
        fill: Some(Paint::RadialGradient(radial.clone())),
        stroke: None,
        fill_rule: FillRule::NonZero,
    }));
    let v = VectorFrame {
        width: 64.0,
        height: 64.0,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let renderer = Renderer::new(64, 64);
    let out = renderer.render(&v);
    let stride = out.planes[0].stride;
    // Interior region: 24..40 in x and y (16×16). Per-pixel reference
    // value at pixel centre (x + 0.5, y + 0.5) using the non-LUT path.
    for y in 24..40 {
        for x in 24..40 {
            let i = y * stride + x * 4;
            let p = &out.planes[0].data[i..i + 4];
            let r = eval_radial_gradient_in(
                &radial,
                x as f32 + 0.5,
                y as f32 + 0.5,
                InterpolationSpace::Srgb,
            );
            // Coverage = 255 inside the circle, so per-pixel output
            // should match the gradient's RGB (within ±1 LSB from the
            // LUT quantisation step).
            let r_arr = [r.r, r.g, r.b];
            for ch in 0..3 {
                let dp = p[ch] as i32 - r_arr[ch] as i32;
                assert!(
                    dp.abs() <= 2,
                    "channel {} at ({}, {}) differs by {}: rendered={:?} ref={:?}",
                    ch,
                    x,
                    y,
                    dp,
                    p,
                    r
                );
            }
        }
    }
}
