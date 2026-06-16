//! `color-interpolation-filters` working-space conversion
//! (Filter Effects Module Level 1 §10).
//!
//! These tests pin the spec contract that filter colour operations
//! default to the linearRGB working space: the source is linearised
//! before a primitive runs and re-encoded to sRGB afterwards, alpha is
//! never touched, and `sRGB` skips the conversion entirely.

use oxideav_core::Rgba;
use oxideav_raster::{
    in_filter_space, in_filter_space_pixels, linear_to_srgb_f32, srgb_to_linear_f32, to_linear_rgb,
    to_linear_rgb_pixels, to_srgb, to_srgb_pixels, FilterColorSpace,
};

/// The property's initial value is `linearRGB`, and `auto` resolves to
/// it. `sRGB` resolves to itself.
#[test]
fn default_and_resolution() {
    assert_eq!(FilterColorSpace::default(), FilterColorSpace::LinearRgb);
    assert_eq!(
        FilterColorSpace::Auto.resolve(),
        FilterColorSpace::LinearRgb
    );
    assert_eq!(FilterColorSpace::Srgb.resolve(), FilterColorSpace::Srgb);
    assert_eq!(
        FilterColorSpace::LinearRgb.resolve(),
        FilterColorSpace::LinearRgb
    );

    assert!(FilterColorSpace::LinearRgb.needs_linearisation());
    assert!(FilterColorSpace::Auto.needs_linearisation());
    assert!(!FilterColorSpace::Srgb.needs_linearisation());
}

/// sRGB 0 and 255 map to linear 0.0 and 1.0 exactly; the curve is
/// monotone and the round trip is identity within byte quantisation.
#[test]
fn transfer_curve_endpoints_and_round_trip() {
    assert_eq!(srgb_to_linear_f32(0.0), 0.0);
    assert!((srgb_to_linear_f32(1.0) - 1.0).abs() < 1e-6);
    assert_eq!(linear_to_srgb_f32(0.0), 0.0);
    assert!((linear_to_srgb_f32(1.0) - 1.0).abs() < 1e-6);

    // sRGB mid-grey (0.5) is well below linear 0.5 (a key reason
    // filters must work in linear light): 0.5 sRGB ≈ 0.214 linear.
    let mid = srgb_to_linear_f32(0.5);
    assert!(mid > 0.20 && mid < 0.23, "0.5 sRGB linearised = {mid}");

    // Full-precision round trip is the identity.
    for i in 0..=255u32 {
        let s = i as f32 / 255.0;
        let back = linear_to_srgb_f32(srgb_to_linear_f32(s));
        assert!((back - s).abs() < 1e-5, "round trip drifted at {i}");
    }
}

/// Byte-buffer round trip is identity within byte quantisation, and
/// alpha is never modified.
#[test]
fn byte_buffer_round_trip_preserves_alpha() {
    let original: Vec<u8> = vec![
        0, 0, 0, 255, // black opaque
        255, 255, 255, 128, // white half-alpha
        128, 64, 200, 0, // colour, fully transparent
        18, 200, 7, 77, // arbitrary
    ];

    let mut buf = original.clone();
    to_linear_rgb(&mut buf);

    // Alpha bytes untouched after linearisation.
    for (i, chunk) in buf.chunks_exact(4).enumerate() {
        assert_eq!(
            chunk[3],
            original[i * 4 + 3],
            "alpha changed by linearisation at pixel {i}"
        );
    }

    to_srgb(&mut buf);

    // Colour channels recover closely; alpha exact. An 8-bit *linear*
    // intermediate cannot represent every sRGB code distinctly (sRGB
    // spends more codes in the dark end than linear light does), so a
    // byte round trip drifts by a few codes in the very dark range —
    // the maximum over all 256 values is 6 bytes (near sRGB byte 6).
    // This is the inherent cost of an 8-bit linear buffer; the f32
    // transfer functions are exact.
    for (i, (got, want)) in buf
        .chunks_exact(4)
        .zip(original.chunks_exact(4))
        .enumerate()
    {
        for c in 0..3 {
            let d = got[c] as i32 - want[c] as i32;
            assert!(d.abs() <= 6, "channel {c} drift {d} at pixel {i}");
        }
        assert_eq!(got[3], want[3], "alpha drift at pixel {i}");
    }
}

/// `Rgba`-pixel path agrees with the byte path and preserves alpha.
#[test]
fn pixel_path_matches_byte_path() {
    let pixels = [
        Rgba::new(10, 20, 30, 200),
        Rgba::new(200, 100, 50, 255),
        Rgba::new(255, 0, 128, 0),
    ];

    // Byte equivalent.
    let mut bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
    to_linear_rgb(&mut bytes);

    let mut px = pixels.to_vec();
    to_linear_rgb_pixels(&mut px);

    for (i, p) in px.iter().enumerate() {
        assert_eq!(p.r, bytes[i * 4], "r mismatch pixel {i}");
        assert_eq!(p.g, bytes[i * 4 + 1], "g mismatch pixel {i}");
        assert_eq!(p.b, bytes[i * 4 + 2], "b mismatch pixel {i}");
        assert_eq!(p.a, pixels[i].a, "alpha changed pixel {i}");
    }

    // Round trip recovers the original pixels (within the 8-bit linear
    // intermediate's dark-end precision loss; see the byte round-trip
    // test for the 6-byte bound).
    to_srgb_pixels(&mut px);
    for (i, (got, want)) in px.iter().zip(pixels.iter()).enumerate() {
        assert!(
            (got.r as i32 - want.r as i32).abs() <= 6,
            "r drift pixel {i}"
        );
        assert!(
            (got.g as i32 - want.g as i32).abs() <= 6,
            "g drift pixel {i}"
        );
        assert!(
            (got.b as i32 - want.b as i32).abs() <= 6,
            "b drift pixel {i}"
        );
        assert_eq!(got.a, want.a, "alpha drift pixel {i}");
    }
}

/// `in_filter_space` under `sRGB` runs the op on the raw bytes with no
/// conversion; under `linearRGB`/`auto` it linearises, runs, re-encodes.
///
/// The chosen op is an identity copy, so under `sRGB` the output equals
/// the input exactly, while under `linearRGB` it equals the linearise →
/// re-encode round trip (identity within byte quantisation).
#[test]
fn in_filter_space_wraps_op_in_working_space() {
    let src: Vec<u8> = vec![64, 128, 192, 255, 0, 255, 13, 100];

    // sRGB: identity op leaves bytes untouched.
    let out_srgb = in_filter_space(FilterColorSpace::Srgb, &src, |b| b.to_vec());
    assert_eq!(out_srgb, src, "sRGB space must not convert");

    // linearRGB: identity op still returns the round trip of src.
    let out_lin = in_filter_space(FilterColorSpace::LinearRgb, &src, |b| b.to_vec());
    let mut expected = src.clone();
    to_linear_rgb(&mut expected);
    to_srgb(&mut expected);
    assert_eq!(out_lin, expected, "linearRGB round trip mismatch");

    // auto resolves to linearRGB.
    let out_auto = in_filter_space(FilterColorSpace::Auto, &src, |b| b.to_vec());
    assert_eq!(out_auto, expected, "auto must resolve to linearRGB");
}

/// The op genuinely sees *linear* samples under `linearRGB`. A 50%
/// average of black and white, computed in the working space, gives a
/// very different sRGB result depending on the space: linear-light
/// averaging yields sRGB ~188 (the perceptually-correct mid value),
/// while naive sRGB averaging yields 128.
#[test]
fn averaging_differs_between_spaces() {
    // Two pixels: black and white, opaque.
    let src: Vec<u8> = vec![0, 0, 0, 255, 255, 255, 255, 255];

    // Op: replace both pixels with their per-channel average.
    let avg = |b: &[u8]| -> Vec<u8> {
        let mut out = b.to_vec();
        for c in 0..4 {
            let m = ((b[c] as u16 + b[4 + c] as u16) / 2) as u8;
            out[c] = m;
            out[4 + c] = m;
        }
        out
    };

    let srgb_avg = in_filter_space(FilterColorSpace::Srgb, &src, avg);
    // Naive sRGB average of 0 and 255 is 127.
    assert!(
        (srgb_avg[0] as i32 - 127).abs() <= 1,
        "sRGB average = {}",
        srgb_avg[0]
    );

    let lin_avg = in_filter_space(FilterColorSpace::LinearRgb, &src, avg);
    // Linear-light average (linear 0.0 and 1.0 → 0.5 → sRGB ~188).
    assert!(
        lin_avg[0] >= 185 && lin_avg[0] <= 192,
        "linearRGB average = {} (expected ~188)",
        lin_avg[0]
    );

    // The two spaces must produce visibly different results.
    assert!(
        lin_avg[0] as i32 - srgb_avg[0] as i32 > 50,
        "linearRGB and sRGB averaging should differ markedly"
    );
}

/// `in_filter_space_pixels` mirrors the byte wrapper.
#[test]
fn in_filter_space_pixels_wraps_op() {
    let src = vec![Rgba::new(64, 128, 192, 255), Rgba::new(0, 255, 13, 100)];

    let out_srgb = in_filter_space_pixels(FilterColorSpace::Srgb, &src, |b| b.to_vec());
    assert_eq!(out_srgb, src);

    let out_lin = in_filter_space_pixels(FilterColorSpace::LinearRgb, &src, |b| b.to_vec());
    let mut expected = src.clone();
    to_linear_rgb_pixels(&mut expected);
    to_srgb_pixels(&mut expected);
    assert_eq!(out_lin, expected);
}
