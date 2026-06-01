//! Integration coverage for the public SVG 1.1 §15.13 `<feConvolveMatrix>`
//! entry points — `convolve_matrix`, `convolve_matrix_pixels`,
//! `ConvolveMatrix`, `ConvolveEdgeMode`.
//!
//! These tests exercise the public re-exports through `lib.rs` (the unit
//! suite inside `src/filter.rs` covers the algorithm in detail; this file
//! verifies the public surface is exposed correctly and the documented
//! semantics hold for callers consuming the published API).

use oxideav_core::Rgba;
use oxideav_raster::{convolve_matrix, convolve_matrix_pixels, ConvolveEdgeMode, ConvolveMatrix};

fn solid_bytes(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        v.extend_from_slice(&rgba);
    }
    v
}

#[test]
fn public_identity_kernel_roundtrips_arbitrary_input() {
    let w = 6;
    let h = 4;
    let mut src = Vec::with_capacity((w * h * 4) as usize);
    for i in 0..(w * h) {
        src.push(((i * 29 + 5) % 256) as u8);
        src.push(((i * 47 + 13) % 256) as u8);
        src.push(((i * 61 + 21) % 256) as u8);
        src.push(((i * 19 + 39) % 256) as u8);
    }
    #[rustfmt::skip]
    let cm = ConvolveMatrix::new(3, 3, vec![
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ]);
    let out = convolve_matrix(&src, w, h, &cm);
    assert_eq!(out, src);
}

#[test]
fn public_box_blur_solid_image_is_identity() {
    let w = 5;
    let h = 4;
    let colour = [80, 160, 240, 200];
    let src = solid_bytes(w, h, colour);
    let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]); // divisor = 9
    let out = convolve_matrix(&src, w, h, &cm);
    assert_eq!(out, src);
}

#[test]
fn public_spec_15_13_worked_example_byte_exact() {
    // Re-runs the spec's §15.13 worked example through the public
    // entry point: a 5×5 grey image with row values
    // {0,20,40,235,235}, {100,120,140,235,235}, …
    let w = 5u32;
    let h = 5u32;
    #[rustfmt::skip]
    let grey: Vec<u8> = vec![
          0,  20,  40, 235, 235,
        100, 120, 140, 235, 235,
        200, 220, 240, 235, 235,
        225, 225, 255, 255, 255,
        225, 225, 255, 255, 255,
    ];
    let mut src = Vec::with_capacity((w * h * 4) as usize);
    for v in &grey {
        src.extend_from_slice(&[*v, *v, *v, 255]);
    }
    #[rustfmt::skip]
    let kernel = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let cm = ConvolveMatrix::new(3, 3, kernel);
    let out = convolve_matrix(&src, w, h, &cm);
    let idx = ((w + 1) * 4) as usize;
    // Per §15.13 (kernel is 180°-rotated): 3480 / 45 = 77.333… → 77
    // after round-half-up clamp. The spec lists the individual
    // products as 9·0 + 8·20 + 7·40 + 6·100 + 5·120 + 4·140 + 3·200 +
    // 2·220 + 1·240 = 3480.
    assert_eq!(out[idx], 77);
}

#[test]
fn public_edge_modes_compile_through_re_export() {
    let src = solid_bytes(2, 2, [10, 20, 30, 255]);
    for mode in [
        ConvolveEdgeMode::Duplicate,
        ConvolveEdgeMode::Wrap,
        ConvolveEdgeMode::None,
    ] {
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_edge_mode(mode);
        // No panic — just exercise each variant through the public API.
        let _ = convolve_matrix(&src, 2, 2, &cm);
    }
}

#[test]
fn public_preserve_alpha_passes_alpha_through() {
    let w = 5;
    let h = 5;
    let mut src = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let alpha = if (x + y) % 2 == 0 { 255 } else { 64 };
            src.extend_from_slice(&[120, 80, 200, alpha]);
        }
    }
    let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_preserve_alpha(true);
    let out = convolve_matrix(&src, w, h, &cm);
    for i in 0..(w * h) as usize {
        assert_eq!(out[i * 4 + 3], src[i * 4 + 3]);
    }
}

#[test]
fn public_typed_pixel_wrapper_matches_byte_api() {
    let w = 5;
    let h = 4;
    let mut src_b = Vec::with_capacity((w * h * 4) as usize);
    let mut src_p = Vec::with_capacity((w * h) as usize);
    for i in 0..(w * h) {
        let p = Rgba::new(
            ((i * 37) % 256) as u8,
            ((i * 51) % 256) as u8,
            ((i * 71) % 256) as u8,
            ((i * 11 + 60) % 256) as u8,
        );
        src_b.extend_from_slice(&[p.r, p.g, p.b, p.a]);
        src_p.push(p);
    }
    #[rustfmt::skip]
    let cm = ConvolveMatrix::new(3, 3, vec![
        -1.0, -1.0, -1.0,
        -1.0,  9.0, -1.0,
        -1.0, -1.0, -1.0,
    ])
    // The sharpen kernel sums to 1 → default divisor would pick 1.0
    // already, but make it explicit for clarity.
    .with_divisor(1.0)
    .with_edge_mode(ConvolveEdgeMode::Duplicate);
    let via_bytes = convolve_matrix(&src_b, w, h, &cm);
    let via_typed = convolve_matrix_pixels(&src_p, w, h, &cm);
    let typed_bytes: Vec<u8> = via_typed
        .iter()
        .flat_map(|p| [p.r, p.g, p.b, p.a])
        .collect();
    assert_eq!(via_bytes, typed_bytes);
}
