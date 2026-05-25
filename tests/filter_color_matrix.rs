//! Integration coverage for `oxideav_raster::color_matrix` — the
//! SVG 1.1 §15.10 `<feColorMatrix>` primitive.
//!
//! Unit tests inside `src/filter.rs` cover the algorithmic contracts
//! (identity, saturate(1) ≈ identity, saturate(0) collapses to
//! luminance, hueRotate(0) ≈ identity, grey-axis preservation under
//! rotation, 360° round-trip, luminanceToAlpha math, op dispatch,
//! channel clamping, bias column, wrapper round-trip, zero-area,
//! panic-on-bad-length). This file is the public-API black-box
//! exercise — the same shape as `tests/filter_morphology.rs`.

use oxideav_core::Rgba;
use oxideav_raster::{
    color_matrix, color_matrix_op, color_matrix_pixels, ColorMatrix, ColorMatrixOp,
};

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
fn public_api_identity_round_trips_bytes() {
    // The 4×5 identity matrix must reproduce its input byte-for-byte.
    let img = build(6, 4, |x, y| {
        Rgba::new((x * 41) as u8, (y * 71) as u8, ((x + y) * 23) as u8, 200)
    });
    let out = color_matrix(&img, 6, 4, &ColorMatrix::identity());
    assert_eq!(out, img, "identity matrix changed the buffer");
}

#[test]
fn public_api_saturate_zero_makes_image_greyscale() {
    // For every pixel in the buffer, fully-desaturated output must
    // satisfy R == G == B (within 1 LSB rounding noise — the
    // saturate(0) matrix has three identical rows).
    let img = build(8, 5, |x, y| {
        Rgba::new((x * 30) as u8, (y * 50) as u8, ((x ^ y) * 10) as u8, 255)
    });
    let out = color_matrix_op(
        &img,
        8,
        5,
        ColorMatrixOp::Saturate(0.0),
        &ColorMatrix::identity(),
    );
    for (i, chunk) in out.chunks_exact(4).enumerate() {
        let dr = (chunk[0] as i32 - chunk[1] as i32).abs();
        let dg = (chunk[1] as i32 - chunk[2] as i32).abs();
        assert!(
            dr <= 1 && dg <= 1,
            "pixel {i}: R={} G={} B={}",
            chunk[0],
            chunk[1],
            chunk[2]
        );
        assert_eq!(
            chunk[3], 255,
            "pixel {i}: alpha must pass through saturation"
        );
    }
}

#[test]
fn public_api_hue_rotate_preserves_alpha_and_total_brightness_on_grey() {
    // §15.10 hue rotation is around the achromatic axis ⇒ grey
    // pixels are eigenvectors with eigenvalue 1 (up to rounding) on
    // the RGB block and the alpha channel is the identity row.
    let img = build(3, 3, |_, _| Rgba::new(100, 100, 100, 180));
    for theta in [10.0_f32, 45.0, 90.0, 135.0, 200.0, 359.0] {
        let out = color_matrix_op(
            &img,
            3,
            3,
            ColorMatrixOp::HueRotate(theta),
            &ColorMatrix::identity(),
        );
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[3], 180, "θ={theta}: alpha must pass through");
            // R, G, B should all be 100 ±2 LSBs.
            for (i, b) in chunk[..3].iter().enumerate() {
                assert!(
                    (*b as i32 - 100).abs() <= 2,
                    "θ={theta}: channel {i} drifted to {b}"
                );
            }
        }
    }
}

#[test]
fn public_api_luminance_to_alpha_paints_transparent_black_with_luma_alpha() {
    // Fixed §15.10 matrix: out RGB = 0; out alpha = BT.709 luminance
    // of the input (using the §15.10 coefficient set
    // (0.2125, 0.7154, 0.0721), distinct from the saturate set).
    let pixels = [
        (Rgba::new(0, 0, 0, 255), 0u8),       // black → α = 0
        (Rgba::new(255, 255, 255, 200), 255), // white → α = 255 (input α is ignored)
        (Rgba::new(255, 0, 0, 100), 54),      // 0.2125·255 ≈ 54.19 → 54
        (Rgba::new(0, 255, 0, 100), 182),     // 0.7154·255 ≈ 182.43 → 182
        (Rgba::new(0, 0, 255, 100), 18),      // 0.0721·255 ≈ 18.39 → 18
    ];
    for (input, expected_alpha) in pixels {
        let buf = build(1, 1, |_, _| input);
        let out = color_matrix_op(
            &buf,
            1,
            1,
            ColorMatrixOp::LuminanceToAlpha,
            &ColorMatrix::identity(),
        );
        assert_eq!(out[0], 0, "input {input:?}: R must clear");
        assert_eq!(out[1], 0, "input {input:?}: G must clear");
        assert_eq!(out[2], 0, "input {input:?}: B must clear");
        let d = (out[3] as i32 - expected_alpha as i32).abs();
        assert!(
            d <= 1,
            "input {input:?}: α drift {d} (got {} expected {expected_alpha})",
            out[3]
        );
    }
}

#[test]
fn public_api_typed_pixel_wrapper_matches_byte_api() {
    // Build matched buffers and verify the wrapper's output is
    // byte-equal to a regroup of the byte-API output across every
    // operator family. (Same shape as the morphology test —
    // safety-net for accidental future divergence.)
    let w = 5u32;
    let h = 3u32;
    let mut bytes = Vec::with_capacity((w * h * 4) as usize);
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for i in 0..(w * h) as u8 {
        let r = i.wrapping_mul(11).wrapping_add(2);
        let g = i.wrapping_mul(29).wrapping_add(7);
        let b = i.wrapping_mul(53).wrapping_add(13);
        let a = 200u8;
        bytes.extend_from_slice(&[r, g, b, a]);
        pixels.push(Rgba::new(r, g, b, a));
    }
    for op in [
        ColorMatrixOp::Matrix,
        ColorMatrixOp::Saturate(0.7),
        ColorMatrixOp::HueRotate(33.0),
        ColorMatrixOp::LuminanceToAlpha,
    ] {
        let m = ColorMatrix::from_op(op, &ColorMatrix::identity());
        let via_bytes = color_matrix(&bytes, w, h, &m);
        let via_pixels = color_matrix_pixels(&pixels, w, h, &m);
        let regrouped: Vec<u8> = via_pixels
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(
            via_bytes, regrouped,
            "{op:?}: byte API / pixel wrapper diverged"
        );
    }
}

#[test]
fn public_api_op_matrix_uses_user_supplied_matrix() {
    // A user-supplied matrix that *halves every channel* exercises the
    // ColorMatrixOp::Matrix branch — confirms the dispatch consults
    // the user buffer (the other three variants build their matrix
    // from the operator parameters and would silently ignore it).
    let user = ColorMatrix([
        [0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0],
    ]);
    let img = build(2, 2, |_, _| Rgba::new(200, 100, 60, 240));
    let out = color_matrix_op(&img, 2, 2, ColorMatrixOp::Matrix, &user);
    for chunk in out.chunks_exact(4) {
        // 0.5·v rounded to nearest integer: 200→100, 100→50, 60→30, 240→120.
        assert!((chunk[0] as i32 - 100).abs() <= 1);
        assert!((chunk[1] as i32 - 50).abs() <= 1);
        assert!((chunk[2] as i32 - 30).abs() <= 1);
        assert!((chunk[3] as i32 - 120).abs() <= 1);
    }
}

#[test]
fn public_api_hue_rotate_180_swaps_complementary_axis() {
    // Apply hueRotate(180) to a pure-red pixel and verify the result is
    // the §15.10 row-evaluation outcome — not the input. Specifically:
    // for R = 1, G = B = 0 and θ = 180°, cos = −1, sin = 0, so each
    // row is `const_row − cos_row` applied to the R coordinate. Verify
    // R drops to non-saturated and G/B pick up the rotated luma.
    let img = build(1, 1, |_, _| Rgba::new(255, 0, 0, 255));
    let out = color_matrix_op(
        &img,
        1,
        1,
        ColorMatrixOp::HueRotate(180.0),
        &ColorMatrix::identity(),
    );
    // Spec algebra: row 0 of the rotated matrix at θ=180 is
    // (0.213 - 0.787, 0.715 + 0.715, 0.072 + 0.072) = (-0.574, 1.430, 0.144).
    // Applied to (1, 0, 0): R' = -0.574, clamped to 0.
    assert_eq!(out[0], 0, "R must clamp to 0 at θ=180 against (255,0,0)");
    // Row 1: (0.213 + 0.213, 0.715 - 0.285, 0.072 + 0.072) = (0.426, 0.430, 0.144).
    // Applied to (1, 0, 0): G' = 0.426 ⇒ ~109.
    let expected_g = ((0.426_f32 * 255.0).round()) as u8;
    let d = (out[1] as i32 - expected_g as i32).abs();
    assert!(
        d <= 2,
        "G at θ=180 against red: drift {d} (got {} expected {expected_g})",
        out[1]
    );
    assert_eq!(out[3], 255, "alpha must survive hue rotation");
}

#[test]
fn public_api_zero_area_image_is_empty_buffer() {
    let img: Vec<u8> = Vec::new();
    for op in [
        ColorMatrixOp::Matrix,
        ColorMatrixOp::Saturate(0.5),
        ColorMatrixOp::HueRotate(45.0),
        ColorMatrixOp::LuminanceToAlpha,
    ] {
        let out = color_matrix_op(&img, 0, 0, op, &ColorMatrix::identity());
        assert!(
            out.is_empty(),
            "{op:?}: zero-area must produce empty output"
        );
    }
}
