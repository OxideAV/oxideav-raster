//! Integration coverage for [`oxideav_raster::blend_filter`] — the
//! SVG 1.1 §15.9 `<feBlend>` filter primitive.
//!
//! The unit tests inside `src/filter.rs::blend_filter_tests` cover the
//! per-pixel arithmetic (mode formulas, shared result-alpha, transparent
//! collapse, mode-divergence sanity, typed-pixel round trip, panic
//! guards). This file is the consumer-facing API exercise — driving the
//! public re-exports through the cases a real `<filter>` element would
//! target.

use oxideav_core::Rgba;
use oxideav_raster::{blend_filter, blend_filter_pixels, BlendFilterMode, CompositeOp};

/// Build a packed-RGBA buffer of `w·h` pixels coloured by `f`.
fn build<F: FnMut(u32, u32) -> Rgba>(w: u32, h: u32, mut f: F) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let c = f(x, y);
            v.extend_from_slice(&[c.r, c.g, c.b, c.a]);
        }
    }
    v
}

#[test]
fn normal_mode_default_is_normal() {
    // Default of the mode enum should be `Normal`, matching the
    // §15.9 attribute table ("If attribute `mode` is not specified,
    // then the effect is as if a value of normal were specified").
    assert_eq!(BlendFilterMode::default(), BlendFilterMode::Normal);
}

#[test]
fn normal_mode_with_transparent_top_returns_bottom() {
    // §15.9 normal formula collapses to `cr = cb`, `qr = qb` when the
    // top input has `qa = 0` ⇒ the bottom passes through verbatim.
    let top = build(8, 8, |_, _| Rgba::new(255, 255, 0, 0));
    let bot = build(8, 8, |x, y| {
        Rgba::new((x * 32) as u8, (y * 32) as u8, 64, 255)
    });
    let out = blend_filter(&top, &bot, 8, 8, BlendFilterMode::Normal);
    assert_eq!(out, bot);
}

#[test]
fn normal_mode_with_opaque_top_returns_top() {
    // §15.9 normal formula collapses to `cr = ca`, `qr = 1` when the
    // top input is fully opaque.
    let top = build(8, 8, |x, y| {
        Rgba::new((x * 32) as u8, (y * 32) as u8, 200, 255)
    });
    let bot = build(8, 8, |_, _| Rgba::new(0, 0, 0, 200));
    let out = blend_filter(&top, &bot, 8, 8, BlendFilterMode::Normal);
    assert_eq!(out, top);
}

#[test]
fn normal_mode_agrees_with_composite_over_for_opaque_inputs() {
    // §15.9: "normal blend mode is equivalent to operator='over' on
    // the feComposite filter primitive". When both operands are fully
    // opaque, this equivalence is bit-exact for both the colour and
    // alpha channels (no `q = 0` un-premultiply edge case).
    let a = build(4, 4, |x, y| {
        Rgba::new((x * 60) as u8, (y * 60) as u8, 100, 255)
    });
    let b = build(4, 4, |x, y| {
        Rgba::new(40, (x * 50) as u8, (y * 70) as u8, 255)
    });
    let blended = blend_filter(&a, &b, 4, 4, BlendFilterMode::Normal);
    let composed = oxideav_raster::composite_filter(&a, &b, 4, 4, CompositeOp::Over);
    assert_eq!(blended, composed);
}

#[test]
fn multiply_with_opaque_inputs_matches_componentwise_product() {
    // Both inputs opaque ⇒ §15.9 multiply formula reduces to
    //   cr = ca · cb
    // (the two `(1 − qX) · cY` extension terms vanish). So the
    // output is the component-wise product of the straight-alpha
    // inputs, evaluated in `[0, 1]` float and re-quantised. Within
    // a ULP of `(a · b) / 255`.
    let a = build(4, 4, |x, _| Rgba::new(64 + (x * 32) as u8, 128, 200, 255));
    let b = build(4, 4, |_, y| Rgba::new(255, 64 + (y * 32) as u8, 50, 255));
    let out = blend_filter(&a, &b, 4, 4, BlendFilterMode::Multiply);
    for (i, px) in out.chunks_exact(4).enumerate() {
        let ai = &a[i * 4..i * 4 + 4];
        let bi = &b[i * 4..i * 4 + 4];
        for c in 0..3 {
            let prod = (ai[c] as f32 / 255.0) * (bi[c] as f32 / 255.0);
            let expected = (prod * 255.0 + 0.5) as i32;
            assert!(
                (px[c] as i32 - expected).abs() <= 1,
                "channel {c} at pixel {i}: got {} expected {} (a={} b={})",
                px[c],
                expected,
                ai[c],
                bi[c]
            );
        }
        assert_eq!(px[3], 255);
    }
}

#[test]
fn screen_with_opaque_inputs_is_screen_formula() {
    // Both opaque ⇒ §15.9 screen formula: `cr = ca + cb − ca·cb`.
    let a = build(4, 4, |x, _| Rgba::new((x * 50) as u8, 128, 200, 255));
    let b = build(4, 4, |_, y| Rgba::new(255, (y * 50) as u8, 50, 255));
    let out = blend_filter(&a, &b, 4, 4, BlendFilterMode::Screen);
    for (i, px) in out.chunks_exact(4).enumerate() {
        let ai = &a[i * 4..i * 4 + 4];
        let bi = &b[i * 4..i * 4 + 4];
        for c in 0..3 {
            let af = ai[c] as f32 / 255.0;
            let bf = bi[c] as f32 / 255.0;
            let cr = af + bf - af * bf;
            let expected = (cr.clamp(0.0, 1.0) * 255.0 + 0.5) as i32;
            assert!(
                (px[c] as i32 - expected).abs() <= 1,
                "channel {c} at pixel {i}: got {} expected {} (a={} b={})",
                px[c],
                expected,
                ai[c],
                bi[c]
            );
        }
        assert_eq!(px[3], 255);
    }
}

#[test]
fn darken_lighten_are_componentwise_min_max_on_opaque_inputs() {
    // For opaque pairs both formula branches in the §15.9 darken /
    // lighten rows reduce to `Min(ca, cb)` / `Max(ca, cb)`.
    let a = build(4, 4, |x, _| Rgba::new((x * 60) as u8, 200, 100, 255));
    let b = build(4, 4, |_, y| Rgba::new(50, (y * 60) as u8, 220, 255));
    let dk = blend_filter(&a, &b, 4, 4, BlendFilterMode::Darken);
    let lt = blend_filter(&a, &b, 4, 4, BlendFilterMode::Lighten);
    for (i, (d, l)) in dk.chunks_exact(4).zip(lt.chunks_exact(4)).enumerate() {
        let ai = &a[i * 4..i * 4 + 4];
        let bi = &b[i * 4..i * 4 + 4];
        for c in 0..3 {
            let lo = ai[c].min(bi[c]) as i32;
            let hi = ai[c].max(bi[c]) as i32;
            assert!(
                (d[c] as i32 - lo).abs() <= 1,
                "darken pixel {i} chan {c}: got {} expected ≈ {lo}",
                d[c]
            );
            assert!(
                (l[c] as i32 - hi).abs() <= 1,
                "lighten pixel {i} chan {c}: got {} expected ≈ {hi}",
                l[c]
            );
        }
        assert_eq!(d[3], 255);
        assert_eq!(l[3], 255);
    }
}

#[test]
fn shared_alpha_invariant_holds_across_modes() {
    // §15.9 specifies the result-alpha formula `qr = 1 − (1 − qa)·(1 − qb)`
    // independently of the chosen mode. Any pair of inputs must therefore
    // produce the same output alpha across all five modes.
    let a = build(4, 4, |x, _| {
        Rgba::new((x * 60) as u8, 50, 100, 100 + (x * 30) as u8)
    });
    let b = build(4, 4, |_, y| {
        Rgba::new(220, (y * 30) as u8, 50, 200 - (y * 30) as u8)
    });
    let modes = [
        BlendFilterMode::Normal,
        BlendFilterMode::Multiply,
        BlendFilterMode::Screen,
        BlendFilterMode::Darken,
        BlendFilterMode::Lighten,
    ];
    let outs: Vec<Vec<u8>> = modes
        .iter()
        .map(|&m| blend_filter(&a, &b, 4, 4, m))
        .collect();
    // Reference: result alpha per pixel computed directly from the spec
    // formula.
    let n = 4 * 4;
    for i in 0..n {
        let qa = a[i * 4 + 3] as f32 / 255.0;
        let qb = b[i * 4 + 3] as f32 / 255.0;
        let qr = 1.0 - (1.0 - qa) * (1.0 - qb);
        let expected = (qr.clamp(0.0, 1.0) * 255.0 + 0.5) as i32;
        for (m_idx, out) in outs.iter().enumerate() {
            assert!(
                (out[i * 4 + 3] as i32 - expected).abs() <= 1,
                "mode {:?} pixel {i}: got alpha {} expected ≈ {expected}",
                modes[m_idx],
                out[i * 4 + 3]
            );
        }
    }
}

#[test]
fn typed_pixel_wrapper_matches_byte_path_across_modes() {
    let w = 5u32;
    let h = 3u32;
    let bytes_a = build(w, h, |x, y| {
        Rgba::new((x * 40) as u8, (y * 80) as u8, 100, 220)
    });
    let bytes_b = build(w, h, |x, y| {
        Rgba::new(30, (x * 50) as u8, (y * 50) as u8, 180)
    });
    let pixels_a: Vec<Rgba> = bytes_a
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let pixels_b: Vec<Rgba> = bytes_b
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    for mode in [
        BlendFilterMode::Normal,
        BlendFilterMode::Multiply,
        BlendFilterMode::Screen,
        BlendFilterMode::Darken,
        BlendFilterMode::Lighten,
    ] {
        let bytes_out = blend_filter(&bytes_a, &bytes_b, w, h, mode);
        let pixels_out = blend_filter_pixels(&pixels_a, &pixels_b, w, h, mode);
        let bytes_from_typed: Vec<u8> = pixels_out
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(bytes_out, bytes_from_typed, "mode {mode:?} mismatch");
    }
}

#[test]
fn fully_transparent_inputs_collapse_to_zero() {
    // `qa = qb = 0` ⇒ `qr = 0`; the §15.7.3 "transparent black" rule
    // selects `(0, 0, 0, 0)` as the output regardless of the colour
    // bytes carried in the input (which are meaningless once the
    // alpha is zero in a straight-alpha buffer).
    let a = build(4, 4, |_, _| Rgba::new(250, 100, 50, 0));
    let b = build(4, 4, |_, _| Rgba::new(10, 200, 220, 0));
    for mode in [
        BlendFilterMode::Normal,
        BlendFilterMode::Multiply,
        BlendFilterMode::Screen,
        BlendFilterMode::Darken,
        BlendFilterMode::Lighten,
    ] {
        let out = blend_filter(&a, &b, 4, 4, mode);
        assert!(
            out.iter().all(|&b| b == 0),
            "mode {mode:?} did not collapse to fully-transparent black"
        );
    }
}

#[test]
fn modes_diverge_on_general_pair() {
    // A representative mixed-alpha mixed-colour pair: every mode
    // must produce a distinct output. Catches accidental aliasing of
    // two enum arms onto the same arithmetic path through the match.
    let a = build(2, 2, |_, _| Rgba::new(200, 100, 50, 200));
    let b = build(2, 2, |_, _| Rgba::new(30, 200, 90, 150));
    let outs = [
        blend_filter(&a, &b, 2, 2, BlendFilterMode::Normal),
        blend_filter(&a, &b, 2, 2, BlendFilterMode::Multiply),
        blend_filter(&a, &b, 2, 2, BlendFilterMode::Screen),
        blend_filter(&a, &b, 2, 2, BlendFilterMode::Darken),
        blend_filter(&a, &b, 2, 2, BlendFilterMode::Lighten),
    ];
    for i in 0..outs.len() {
        for j in (i + 1)..outs.len() {
            assert_ne!(
                outs[i], outs[j],
                "outputs {i} and {j} collapsed onto the same arithmetic path"
            );
        }
    }
}

#[test]
fn empty_extent_returns_empty_buffer() {
    let empty: Vec<u8> = Vec::new();
    let out = blend_filter(&empty, &empty, 0, 0, BlendFilterMode::Multiply);
    assert!(out.is_empty());
}
