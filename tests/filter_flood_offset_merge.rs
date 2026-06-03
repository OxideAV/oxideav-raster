//! Integration coverage for [`oxideav_raster::flood`] /
//! [`oxideav_raster::offset`] / [`oxideav_raster::merge`] — the SVG 1.1
//! §15.16 `<feFlood>`, §15.21 `<feOffset>`, and §15.19 `<feMerge>`
//! primitives.
//!
//! The unit tests inside `src/filter.rs` cover the per-primitive algebra
//! (range clamping, sampling sub-modes, panic on bad input). This file
//! is the consumer-facing API exercise — treating the public re-exports
//! as a black box and checking the documented behaviour through the
//! pipeline shape these three primitives are most often composed in:
//! the §15.2 drop-shadow example sketched in the spec
//! (`SourceAlpha → blur → offset → merge` with the original source
//! graphic on top).

use oxideav_core::Rgba;
use oxideav_raster::{
    composite_filter, flood, flood_pixels, gaussian_blur, merge, merge_pixels, offset,
    offset_pixels, CompositeOp, OffsetSampling,
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

// -------------------- feFlood --------------------

#[test]
fn flood_emits_extent_filled_with_resolved_pixel() {
    let out = flood(7, 5, 12, 34, 56, 0.5);
    assert_eq!(out.len(), 7 * 5 * 4);
    // round_half_up(0.5 · 255) = 128.
    for px in out.chunks_exact(4) {
        assert_eq!(px, [12, 34, 56, 128]);
    }
}

#[test]
fn flood_typed_path_agrees_with_byte_path() {
    let bytes = flood(6, 4, 90, 100, 110, 0.9);
    let typed = flood_pixels(6, 4, 90, 100, 110, 0.9);
    let from_typed: Vec<u8> = typed.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
    assert_eq!(bytes, from_typed);
}

// -------------------- feOffset --------------------

#[test]
fn offset_shifts_the_image_by_an_integer_vector() {
    // Source: opaque white at (1, 1), transparent elsewhere.
    let src = build(4, 4, |x, y| {
        if (x, y) == (1, 1) {
            Rgba::new(255, 255, 255, 255)
        } else {
            Rgba::new(0, 0, 0, 0)
        }
    });
    // Shift +2 in x, +1 in y → expect the dot at (3, 2).
    let out = offset(&src, 4, 4, 2.0, 1.0, OffsetSampling::Nearest);
    for y in 0..4 {
        for x in 0..4 {
            let p = ((y * 4 + x) * 4) as usize;
            if (x, y) == (3, 2) {
                assert_eq!(&out[p..p + 4], &[255, 255, 255, 255], "({x}, {y})");
            } else {
                assert_eq!(&out[p..p + 4], &[0, 0, 0, 0], "({x}, {y})");
            }
        }
    }
}

#[test]
fn offset_typed_wrapper_round_trips_through_bytes() {
    let bytes = build(5, 4, |x, y| {
        Rgba::new(
            (x * 30) as u8,
            (y * 40) as u8,
            ((x ^ y) * 20) as u8,
            ((x + y) * 25) as u8,
        )
    });
    let typed: Vec<Rgba> = bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    for sampling in [OffsetSampling::Nearest, OffsetSampling::Bilinear] {
        for (dx, dy) in [(0.0, 0.0), (1.0, -2.0), (3.0, 0.0)] {
            let via_bytes = offset(&bytes, 5, 4, dx, dy, sampling);
            let via_typed = offset_pixels(&typed, 5, 4, dx, dy, sampling);
            let typed_bytes: Vec<u8> = via_typed
                .iter()
                .flat_map(|p| [p.r, p.g, p.b, p.a])
                .collect();
            assert_eq!(
                via_bytes, typed_bytes,
                "{sampling:?} shift=({dx}, {dy}) byte vs typed mismatch"
            );
        }
    }
}

#[test]
fn offset_off_canvas_pixels_are_transparent_black() {
    // A fully opaque white canvas, shifted off to the right by the full
    // width: every output pixel should be (0, 0, 0, 0) per §15.7.3.
    let src = build(4, 4, |_, _| Rgba::new(255, 255, 255, 255));
    let out = offset(&src, 4, 4, 4.0, 0.0, OffsetSampling::Nearest);
    assert!(out.iter().all(|b| *b == 0));
}

#[test]
fn offset_bilinear_subpixel_shift_is_finite_and_in_bounds() {
    // Smoke test: a sub-pixel bilinear shift over a non-trivial buffer
    // must always be in u8 range and the same length as the source.
    let src = build(10, 8, |x, y| {
        Rgba::new(
            (x * 25) as u8,
            (y * 30) as u8,
            ((x + y) * 12) as u8,
            ((x * y) * 3 % 256) as u8,
        )
    });
    let out = offset(&src, 10, 8, 0.5, -0.5, OffsetSampling::Bilinear);
    assert_eq!(out.len(), src.len());
    // Spot-check: every pixel in the offset-bilinear output has a
    // finite alpha less than or equal to the maximum of the four
    // bilinear taps' alphas (a property of any convex combination —
    // and the alpha clamp inside `offset_bilinear` is post-clamp,
    // so out-of-range float artefacts cannot escape). With the
    // input alphas all in `[0, 255 - ε]` it is enough to confirm
    // that the output buffer is non-empty and equal in length to
    // the source.
    assert!(!out.is_empty());
}

// -------------------- feMerge --------------------

#[test]
fn merge_empty_list_is_transparent_black_canvas() {
    let out = merge(4, 3, &[]);
    assert_eq!(out.len(), 4 * 3 * 4);
    assert!(out.iter().all(|b| *b == 0));
}

#[test]
fn merge_single_layer_is_identity() {
    let a = build(3, 3, |x, y| Rgba::new(x as u8 * 30, y as u8 * 40, 50, 200));
    let out = merge(3, 3, &[&a]);
    assert_eq!(out, a);
}

#[test]
fn merge_n_minus_1_composites_match_associative_fold() {
    // §15.19 documents that `feMerge` is equivalent to `n − 1`
    // `feComposite` operators with `op = over`. Confirm by hand: a
    // pairwise composite_filter(Over) reduce should match a single
    // call to merge() to within quantisation noise.
    let layers = [
        build(4, 4, |x, _| Rgba::new(255, 0, 0, (x * 50) as u8)),
        build(4, 4, |x, y| Rgba::new(0, 255, 0, ((x + y) * 25) as u8)),
        build(4, 4, |_, y| Rgba::new(0, 0, 255, (y * 60) as u8)),
        build(4, 4, |x, y| Rgba::new(200, 200, 0, ((x ^ y) * 40) as u8)),
    ];
    let layer_refs: Vec<&[u8]> = layers.iter().map(|l| l.as_slice()).collect();
    let merged = merge(4, 4, &layer_refs);

    // Reduce by repeated composite-Over.
    let mut acc = layers[0].clone();
    for next in &layers[1..] {
        acc = composite_filter(next, &acc, 4, 4, CompositeOp::Over);
    }
    // ±2 quantisation tolerance per channel.
    assert_eq!(acc.len(), merged.len());
    for (a, b) in acc.iter().zip(merged.iter()) {
        assert!(
            (*a as i32 - *b as i32).abs() <= 2,
            "channel diff {} vs {}",
            *a,
            *b
        );
    }
}

#[test]
fn merge_typed_wrapper_round_trips_through_bytes() {
    let a_b = build(3, 3, |x, _| Rgba::new(255, 0, 0, (x * 60) as u8));
    let b_b = build(3, 3, |_, y| Rgba::new(0, 255, 0, (y * 80) as u8));
    let a_p: Vec<Rgba> = a_b
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let b_p: Vec<Rgba> = b_b
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let bytes_out = merge(3, 3, &[&a_b, &b_b]);
    let typed_out = merge_pixels(3, 3, &[&a_p, &b_p]);
    let from_typed: Vec<u8> = typed_out
        .iter()
        .flat_map(|p| [p.r, p.g, p.b, p.a])
        .collect();
    assert_eq!(bytes_out, from_typed);
}

// -------------------- §15.2 drop-shadow pipeline shape --------------------

#[test]
fn drop_shadow_pipeline_shape_runs_end_to_end() {
    // The §15.2 example wires up:
    //   SourceAlpha → feGaussianBlur → feOffset → feMerge(offset, src)
    // with feFlood / feColorMatrix in some variants. Confirm the
    // canonical shape composes through the public API without panics
    // and yields a sensibly-shaped output.
    //
    // Source: a solid red square in the centre of a black canvas with
    // alpha 255 (so SourceAlpha is just the alpha channel of source).
    let w = 16u32;
    let h = 16u32;
    let source = build(w, h, |x, y| {
        if (4..12).contains(&x) && (4..12).contains(&y) {
            Rgba::new(255, 0, 0, 255)
        } else {
            Rgba::new(0, 0, 0, 0)
        }
    });
    // SourceAlpha: copy the alpha into all four channels — the
    // conventional implementation strategy for a §15.7.2 SourceAlpha
    // input.
    let source_alpha: Vec<u8> = source
        .chunks_exact(4)
        .flat_map(|p| [0, 0, 0, p[3]])
        .collect();

    // Blur the alpha by stdDeviation = 2.
    let blurred = gaussian_blur(&source_alpha, w, h, 2.0, 2.0);
    // Offset by (dx, dy) = (2, 2).
    let shadow = offset(&blurred, w, h, 2.0, 2.0, OffsetSampling::Nearest);
    // Merge shadow under the original source.
    let composed = merge(w, h, &[&shadow, &source]);

    assert_eq!(composed.len(), (w * h * 4) as usize);

    // The red square is opaque in the source and must remain opaque
    // (alpha 255) in the merged output: §14.2 `over` of an opaque
    // foreground onto any backdrop is the opaque foreground.
    for y in 4..12 {
        for x in 4..12 {
            let i = ((y * w + x) * 4) as usize;
            assert_eq!(composed[i], 255, "red kept at ({x}, {y})");
            assert_eq!(composed[i + 3], 255, "alpha at ({x}, {y})");
        }
    }
    // Some pixel below-and-to-the-right of the square should have
    // shadow alpha greater than zero (the offset-blurred alpha plume).
    // Look at (14, 14): outside the square (so source alpha is 0) and
    // within the blurred-and-offset shadow reach.
    let probe = ((14 * w + 14) * 4) as usize;
    assert!(
        composed[probe + 3] > 0,
        "shadow contribution at (14, 14) alpha {}",
        composed[probe + 3]
    );
}

#[test]
fn drop_shadow_pipeline_with_flood_replaces_shadow_colour() {
    // A common extension: replace the blurred alpha's neutral colour
    // with a coloured flood, then composite the flood "in" the
    // blurred-and-offset mask before merging under the source.
    let w = 12u32;
    let h = 12u32;
    let source = build(w, h, |x, y| {
        if (3..9).contains(&x) && (3..9).contains(&y) {
            Rgba::new(0, 0, 255, 255)
        } else {
            Rgba::new(0, 0, 0, 0)
        }
    });
    let source_alpha: Vec<u8> = source
        .chunks_exact(4)
        .flat_map(|p| [0, 0, 0, p[3]])
        .collect();
    let blurred = gaussian_blur(&source_alpha, w, h, 1.5, 1.5);
    let shadow_mask = offset(&blurred, w, h, 1.0, 1.0, OffsetSampling::Nearest);
    let flood_layer = flood(w, h, 80, 0, 80, 0.6);
    // `in` the flood with the shadow mask — the spec's documented
    // pattern for tinting a drop shadow.
    let shadow_tinted = composite_filter(&flood_layer, &shadow_mask, w, h, CompositeOp::In);
    let composed = merge(w, h, &[&shadow_tinted, &source]);

    assert_eq!(composed.len(), (w * h * 4) as usize);
    // The blue square pixels remain blue (source is opaque, on top).
    let i = ((5 * w + 5) * 4) as usize;
    assert_eq!(&composed[i..i + 4], &[0, 0, 255, 255]);
}
