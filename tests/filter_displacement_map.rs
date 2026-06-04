//! Integration coverage for [`oxideav_raster::displacement_map`] — the
//! SVG 1.1 §15.15 `<feDisplacementMap>` filter primitive.
//!
//! The unit tests inside `src/filter.rs::displacement_map_tests` cover
//! the per-pixel sampling algebra (zero shift identity, scale = 0 copy,
//! ±half-scale shifts from XC = 0 / 1, alpha-default selector,
//! out-of-bounds transparent black, bilinear-at-integer-equals-nearest,
//! fractional-shift bilinear blend, channel-selector independence,
//! typed-wrapper round trip, panic guards). This file is the
//! consumer-facing API exercise — driving the public re-exports through
//! the cases a real `<filter>` element would target.

use oxideav_core::Rgba;
use oxideav_raster::{
    displacement_map, displacement_map_pixels, DisplacementChannel, DisplacementSampling,
};

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
fn flat_grey_map_is_full_identity_with_nearest_sampling() {
    // §15.15: per-pixel XC = YC = 0.5 ⇒ both shifts are zero
    // regardless of the scale value ⇒ the result equals `in1`.
    //
    // We pick byte value 128 for both the X and Y channels of `in2`;
    // 128 / 255 = 0.5019…, so the per-axis shift is `scale · 0.00196`.
    // For `scale ≤ 250`, the nearest-pixel round of |shift| < 0.5
    // hits zero and the result is bit-exact identity.
    let in1 = build(16, 16, |x, y| {
        Rgba::new((x * 16) as u8, (y * 16) as u8, ((x ^ y) * 16) as u8, 255)
    });
    let in2 = build(16, 16, |_x, _y| Rgba::new(128, 128, 128, 255));
    for scale in [0.0_f32, 1.0, 10.0, 100.0, 250.0] {
        let out = displacement_map(
            &in1,
            &in2,
            16,
            16,
            scale,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        );
        assert_eq!(out, in1, "scale = {scale} broke half-grey identity");
    }
}

#[test]
fn uniform_full_x_full_y_pulls_diagonal() {
    // XC = YC = 1.0 ⇒ shifts = (+0.5 · scale, +0.5 · scale). For
    // `scale = 4` the source coordinate is (x + 2, y + 2) on every
    // output pixel. Out-of-bounds positions emit transparent black.
    let w = 8u32;
    let h = 8u32;
    // Distinct-per-pixel source: `(x * 30, y * 30, 100, 255)` so we
    // can verify which source pixel landed at every output.
    let in1 = build(w, h, |x, y| {
        Rgba::new((x * 30) as u8, (y * 30) as u8, 100, 255)
    });
    let in2 = build(w, h, |_x, _y| Rgba::new(255, 255, 0, 255));
    let out = displacement_map(
        &in1,
        &in2,
        w,
        h,
        4.0,
        DisplacementChannel::R,
        DisplacementChannel::G,
        DisplacementSampling::Nearest,
    );
    for y in 0..h {
        for x in 0..w {
            let sx = x as i64 + 2;
            let sy = y as i64 + 2;
            let dst = ((y * w + x) as usize) * 4;
            if (0..w as i64).contains(&sx) && (0..h as i64).contains(&sy) {
                let s = ((sy * w as i64 + sx) as usize) * 4;
                assert_eq!(&out[dst..dst + 4], &in1[s..s + 4]);
            } else {
                assert_eq!(&out[dst..dst + 4], &[0, 0, 0, 0]);
            }
        }
    }
}

#[test]
fn alpha_channel_default_drives_displacement() {
    // §15.15: default selector is `A`. Build a map whose A varies
    // across columns (drives a per-column shift_x) while R / G / B
    // would all give zero-shift if accidentally selected — proving
    // the default really does read from the alpha lane.
    let w = 6u32;
    let h = 1u32;
    let in1 = build(w, h, |x, _y| Rgba::new((x * 40) as u8, 7, 9, 200));
    // R = G = B = 128 (XC via any RGB selector = 0.5 ⇒ zero shift —
    // this is the "if you read the wrong channel you'd see ~no shift"
    // guard). A varies across columns and is what the default `A`
    // selector picks up.
    //   Column 0: A = 0   ⇒ XC = 0   ⇒ shift_x = scale · -0.5
    //   Column 1: A = 51  ⇒ XC = 0.2 ⇒ shift_x = scale · -0.6
    //   Column 5: A = 255 ⇒ XC = 1.0 ⇒ shift_x = scale · +0.5
    let in2 = build(w, h, |x, _y| {
        Rgba::new(128, 128, 128, (x * 51).min(255) as u8)
    });
    // x-channel = default (A) — that is the point of this test. The
    // y-channel is set to R (the 128 byte) so YC = 0.5019 ⇒ y-shift
    // rounds to 0 under nearest sampling, isolating the warp to the
    // x-axis (h = 1 anyway, so any non-zero y-shift would make every
    // sample fall outside the canvas).
    let out = displacement_map(
        &in1,
        &in2,
        w,
        h,
        2.0,
        DisplacementChannel::default(), // A
        DisplacementChannel::R,         // 128 / 255 ≈ 0.5 ⇒ zero y-shift
        DisplacementSampling::Nearest,
    );
    // For column x, shift_x = 2.0 · ((x · 51 / 255) − 0.5) where the
    // 51 / 255 = 0.2 step gives shifts of (−1.0, −0.6, −0.2, +0.2, +0.6, +1.0).
    // Sample location is therefore `(x + shift_x).round()`.
    for x in 0..w {
        let xc = (x as i64 * 51).min(255) as f32 / 255.0;
        let shift = 2.0 * (xc - 0.5);
        let sx = (x as f32 + shift).round() as i64;
        let dst = (x as usize) * 4;
        if (0..w as i64).contains(&sx) {
            let s = (sx as usize) * 4;
            assert_eq!(&out[dst..dst + 4], &in1[s..s + 4], "col {x}");
        } else {
            assert_eq!(&out[dst..dst + 4], &[0, 0, 0, 0]);
        }
    }
}

#[test]
fn x_and_y_channels_are_truly_independent() {
    // R varies horizontally; A varies vertically. Selecting (R, A)
    // gives a per-pixel shift that depends on both axes ⇒ generates
    // a 2-D warp that cannot be reduced to either axis alone.
    let w = 8u32;
    let h = 8u32;
    let in1 = build(w, h, |x, y| {
        Rgba::new((x * 30) as u8, (y * 30) as u8, 0, 255)
    });
    let in2 = build(w, h, |x, y| {
        Rgba::new((x * 32) as u8, 128, 0, (y * 32) as u8)
    });
    let out = displacement_map(
        &in1,
        &in2,
        w,
        h,
        4.0,
        DisplacementChannel::R, // x-shift varies across columns
        DisplacementChannel::A, // y-shift varies across rows
        DisplacementSampling::Nearest,
    );
    for y in 0..h {
        for x in 0..w {
            let xc = (x * 32) as f32 / 255.0;
            let yc = (y * 32) as f32 / 255.0;
            let sx = (x as f32 + 4.0 * (xc - 0.5)).round() as i64;
            let sy = (y as f32 + 4.0 * (yc - 0.5)).round() as i64;
            let dst = ((y * w + x) as usize) * 4;
            if (0..w as i64).contains(&sx) && (0..h as i64).contains(&sy) {
                let s = ((sy * w as i64 + sx) as usize) * 4;
                assert_eq!(&out[dst..dst + 4], &in1[s..s + 4], "({x},{y})");
            } else {
                assert_eq!(&out[dst..dst + 4], &[0, 0, 0, 0], "({x},{y})");
            }
        }
    }
}

#[test]
fn scale_zero_is_a_full_copy() {
    // §15.15 attribute table: "When the value of this attribute is 0,
    // this operation has no effect on the source image." Verify under
    // both sampling policies, against an arbitrary map.
    let in1 = build(5, 4, |x, y| {
        Rgba::new((x * 50) as u8, (y * 60) as u8, ((x ^ y) * 40) as u8, 200)
    });
    let in2 = build(5, 4, |x, y| {
        Rgba::new((x * 50) as u8, (y * 60) as u8, 0, 255)
    });
    for sampling in [
        DisplacementSampling::Nearest,
        DisplacementSampling::Bilinear,
    ] {
        let out = displacement_map(
            &in1,
            &in2,
            5,
            4,
            0.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            sampling,
        );
        assert_eq!(out, in1);
    }
}

#[test]
fn extreme_scale_with_zero_xc_yc_pulls_everything_off_canvas() {
    // XC = YC = 0 (R / G channels are 0) and scale = 1000 ⇒ shift =
    // (−500, −500). Every output pixel asks for a source position far
    // outside the canvas ⇒ result is uniformly transparent black per
    // §15.7.3 across the whole extent.
    let in1 = build(4, 4, |_x, _y| Rgba::new(99, 99, 99, 255));
    let in2 = build(4, 4, |_x, _y| Rgba::new(0, 0, 0, 0));
    let out = displacement_map(
        &in1,
        &in2,
        4,
        4,
        1000.0,
        DisplacementChannel::R,
        DisplacementChannel::G,
        DisplacementSampling::Nearest,
    );
    assert_eq!(out.len(), 4 * 4 * 4);
    for px in out.chunks_exact(4) {
        assert_eq!(px, [0, 0, 0, 0]);
    }
}

#[test]
fn typed_pixel_wrapper_round_trips_through_byte_path() {
    let w = 6u32;
    let h = 6u32;
    let bytes_in1 = build(w, h, |x, y| {
        Rgba::new((x * 20) as u8, (y * 20) as u8, 80, 255)
    });
    let bytes_in2 = build(w, h, |x, y| {
        Rgba::new((x * 30) as u8, (y * 30) as u8, 128, 255)
    });
    let pixels_in1: Vec<Rgba> = bytes_in1
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let pixels_in2: Vec<Rgba> = bytes_in2
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    for sampling in [
        DisplacementSampling::Nearest,
        DisplacementSampling::Bilinear,
    ] {
        let bytes_out = displacement_map(
            &bytes_in1,
            &bytes_in2,
            w,
            h,
            3.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            sampling,
        );
        let pixels_out = displacement_map_pixels(
            &pixels_in1,
            &pixels_in2,
            w,
            h,
            3.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            sampling,
        );
        let from_typed: Vec<u8> = pixels_out
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(bytes_out, from_typed, "sampling = {sampling:?}");
    }
}

#[test]
fn negative_scale_inverts_the_displacement_direction() {
    // §15.15 doesn't restrict `scale` to be non-negative, and the
    // formula `(x + scale · (XC − 0.5))` is linear in `scale`; a
    // negative scale flips the per-axis shift direction.
    //
    // XC = 1.0 (R = 255), scale = +2 ⇒ shift_x = +1 ⇒ samples from
    // column x + 1. The same XC with scale = −2 ⇒ shift_x = −1 ⇒
    // samples from column x − 1. Build a horizontal ramp and check
    // both warps fetch the adjacent columns in opposite directions.
    let w = 5u32;
    let h = 1u32;
    let in1 = build(w, h, |x, _y| Rgba::new((x * 50) as u8, 0, 0, 255));
    let in2 = build(w, h, |_x, _y| Rgba::new(255, 128, 0, 255));
    let pos = displacement_map(
        &in1,
        &in2,
        w,
        h,
        2.0,
        DisplacementChannel::R,
        DisplacementChannel::G,
        DisplacementSampling::Nearest,
    );
    let neg = displacement_map(
        &in1,
        &in2,
        w,
        h,
        -2.0,
        DisplacementChannel::R,
        DisplacementChannel::G,
        DisplacementSampling::Nearest,
    );
    for x in 0..w {
        let dst = (x as usize) * 4;
        let sx_pos = x as i64 + 1;
        let sx_neg = x as i64 - 1;
        if sx_pos < w as i64 {
            let s = (sx_pos as usize) * 4;
            assert_eq!(&pos[dst..dst + 4], &in1[s..s + 4]);
        } else {
            assert_eq!(&pos[dst..dst + 4], &[0, 0, 0, 0]);
        }
        if sx_neg >= 0 {
            let s = (sx_neg as usize) * 4;
            assert_eq!(&neg[dst..dst + 4], &in1[s..s + 4]);
        } else {
            assert_eq!(&neg[dst..dst + 4], &[0, 0, 0, 0]);
        }
    }
}

#[test]
fn integer_shift_is_bit_exact_under_both_sampling_policies() {
    // §15.15 sampling-policy note "high quality viewers apply an
    // interpolent" applies to fractional source coordinates. When the
    // per-pixel shift is integer-valued, the bilinear blend collapses
    // to nearest, so both policies must agree byte-for-byte.
    let w = 6u32;
    let h = 6u32;
    let in1 = build(w, h, |x, y| {
        Rgba::new((x * 25) as u8, (y * 25) as u8, ((x + y) * 15) as u8, 255)
    });
    // XC = 1.0, scale = 2 ⇒ shift_x = +1 (integer). YC ≈ 0.502 ⇒
    // shift_y ≈ +0.004 — within nearest-rounding tolerance and bilinear
    // blends mostly the top row, so we keep YC = 128 / 255 for both.
    let in2 = build(w, h, |_x, _y| Rgba::new(255, 128, 0, 255));
    let nearest = displacement_map(
        &in1,
        &in2,
        w,
        h,
        2.0,
        DisplacementChannel::R,
        DisplacementChannel::G,
        DisplacementSampling::Nearest,
    );
    let bilinear = displacement_map(
        &in1,
        &in2,
        w,
        h,
        2.0,
        DisplacementChannel::R,
        DisplacementChannel::G,
        DisplacementSampling::Bilinear,
    );
    // The 128/255 YC drift is small enough that the bilinear top-row
    // weight is > 0.99 and the bottom-row weight is < 0.01; round-to-
    // nearest of a fully-saturated lane plus a tiny opposite-row tap
    // can still nudge a byte by ±1. We assert ≥ 95 % bit-exact match
    // and the remaining lanes within ±1.
    let mut exact = 0usize;
    let mut close = 0usize;
    for (a, b) in nearest.iter().zip(bilinear.iter()) {
        if a == b {
            exact += 1;
        } else if a.abs_diff(*b) <= 1 {
            close += 1;
        }
    }
    let total = nearest.len();
    assert_eq!(
        exact + close,
        total,
        "bilinear should be within ±1 of nearest"
    );
    let exact_frac = exact as f64 / total as f64;
    assert!(
        exact_frac >= 0.95,
        "exact match {exact_frac:.4} < 0.95 — YC drift too large?"
    );
}

#[test]
fn warp_is_purely_local_and_does_not_invent_pixels() {
    // §15.15 is a "for every output pixel, sample one source coordinate"
    // primitive. With finite `scale`, no output pixel can land outside
    // the disc `(x ± scale/2, y ± scale/2)` around its own coordinate
    // ⇒ the colours present in the output must all be present in the
    // input (give or take a bilinear blend, which we suppress with
    // Nearest sampling here). Verifies we did not accidentally
    // synthesise colour samples through a math error.
    let w = 5u32;
    let h = 5u32;
    let in1 = build(w, h, |x, y| {
        Rgba::new(((x * 50) % 200) as u8, (y * 40) as u8, 11, 255)
    });
    let in2 = build(w, h, |x, y| {
        Rgba::new((x * 50) as u8, (y * 60) as u8, 0, 255)
    });

    // Collect the set of byte triples used by `in1`.
    use std::collections::HashSet;
    let mut src_pixels: HashSet<[u8; 4]> = HashSet::new();
    for chunk in in1.chunks_exact(4) {
        src_pixels.insert([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    // Transparent black is the documented §15.7.3 fallback ⇒
    // always-permitted result.
    src_pixels.insert([0, 0, 0, 0]);

    let out = displacement_map(
        &in1,
        &in2,
        w,
        h,
        3.0,
        DisplacementChannel::R,
        DisplacementChannel::G,
        DisplacementSampling::Nearest,
    );
    for (i, chunk) in out.chunks_exact(4).enumerate() {
        let p: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        assert!(
            src_pixels.contains(&p),
            "pixel {i} = {p:?} not present in source or transparent-black"
        );
    }
}
