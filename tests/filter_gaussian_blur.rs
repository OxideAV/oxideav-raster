//! Integration coverage for [`oxideav_raster::gaussian_blur`] — the
//! SVG 1.1 §15.17 `<feGaussianBlur>` filter primitive.
//!
//! The unit tests inside `src/filter.rs` exercise the algorithmic
//! contracts of both implementation branches (direct discrete
//! separable convolution for `s < 2.0`, the spec's three-box-blur
//! approximation for `s ≥ 2.0`) — kernel normalisation, separability,
//! axis-only invariance, monotonicity, panic semantics, and the
//! `box_sizes_for_std` table. This file is the consumer-facing API
//! exercise — the same style as `tests/filter_morphology.rs` /
//! `tests/filter_color_matrix.rs`. Math is sourced verbatim from
//! `docs/image/svg/svg11-second-edition.pdf` §15.17.

use oxideav_core::Rgba;
use oxideav_raster::{gaussian_blur, gaussian_blur_pixels, GAUSSIAN_BLUR_BOX_THRESHOLD};

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
fn public_api_zero_stddev_is_identity() {
    let img = build(7, 5, |x, y| {
        Rgba::new((x * 13) as u8, (y * 37) as u8, 17, 200)
    });
    assert_eq!(gaussian_blur(&img, 7, 5, 0.0, 0.0), img);
}

#[test]
fn box_threshold_constant_is_two_point_zero() {
    // Documented threshold: callers should be able to inspect the
    // numeric value so they can reason about which branch a given
    // stdDeviation will activate.
    assert_eq!(GAUSSIAN_BLUR_BOX_THRESHOLD, 2.0);
}

#[test]
fn solid_image_is_pixel_exact_for_a_grid_of_stddevs() {
    // Spec §15.17 implies a normalised kernel; on a constant image
    // that means the output is bit-exactly the input, regardless of
    // which implementation branch is taken. Sweep both branches
    // (s < 2.0 and s ≥ 2.0) and a mix of axis-only configurations.
    let img = build(13, 9, |_, _| Rgba::new(123, 200, 50, 255));
    let cases = [
        (0.3, 0.0),
        (0.0, 0.7),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.5, 1.99),
        (2.0, 2.0),
        (3.0, 0.0),
        (0.0, 4.0),
        (5.0, 7.0),
    ];
    for (sx, sy) in cases {
        let out = gaussian_blur(&img, 13, 9, sx, sy);
        assert_eq!(
            out, img,
            "constant image must be invariant; (sx, sy) = ({sx}, {sy})"
        );
    }
}

#[test]
fn alpha_only_input_stays_alpha_only_under_blur() {
    // §15.17 calls out the SourceAlpha case explicitly. Our function
    // doesn't special-case it but it does run per-channel — so an
    // input with R = G = B = 0 must remain R = G = B = 0 after the
    // blur (and only the alpha channel diffuses).
    let w = 9u32;
    let h = 9u32;
    let mut img = build(w, h, |_, _| Rgba::new(0, 0, 0, 0));
    let off = ((h as usize / 2) * w as usize + w as usize / 2) * 4;
    img[off + 3] = 255; // single alpha-only impulse

    for &(sx, sy) in &[(0.8, 0.8), (1.5, 1.5), (3.0, 3.0)] {
        let out = gaussian_blur(&img, w, h, sx, sy);
        for (i, chunk) in out.chunks_exact(4).enumerate() {
            assert_eq!(
                chunk[0], 0,
                "R diffused at pixel {i} for (sx, sy) = ({sx}, {sy})"
            );
            assert_eq!(chunk[1], 0, "G diffused at pixel {i}");
            assert_eq!(chunk[2], 0, "B diffused at pixel {i}");
        }
        // Alpha must have at least diffused somewhere (we put in
        // 255 at the centre, the centre will drop below 255).
        let centre_alpha = out[off + 3];
        assert!(
            centre_alpha < 255,
            "alpha did not diffuse for (sx, sy) = ({sx}, {sy})"
        );
    }
}

#[test]
fn impulse_response_is_four_fold_symmetric_under_isotropic_blur() {
    // A single bright pixel at the canvas centre, blurred with
    // sx = sy, must produce a result that's symmetric under both
    // x-mirror and y-mirror. This is a direct consequence of the
    // separable Gaussian being isotropic for equal stdDeviation.
    let w = 11u32;
    let h = 11u32;
    let cx = 5usize;
    let cy = 5usize;
    let mut img = build(w, h, |_, _| Rgba::new(0, 0, 0, 0));
    let off = (cy * w as usize + cx) * 4;
    img[off..off + 4].copy_from_slice(&[255, 255, 255, 255]);

    for s in [0.8f32, 1.5, 2.5, 3.5] {
        let out = gaussian_blur(&img, w, h, s, s);
        // x-mirror
        for y in 0..h as usize {
            for x in 0..(w as usize / 2) {
                let mirror_x = w as usize - 1 - x;
                let a = &out[(y * w as usize + x) * 4..(y * w as usize + x) * 4 + 4];
                let b = &out[(y * w as usize + mirror_x) * 4..(y * w as usize + mirror_x) * 4 + 4];
                assert_eq!(a, b, "x-mirror broke at s={s} y={y} x={x}");
            }
        }
        // y-mirror
        for y in 0..(h as usize / 2) {
            let mirror_y = h as usize - 1 - y;
            for x in 0..w as usize {
                let a = &out[(y * w as usize + x) * 4..(y * w as usize + x) * 4 + 4];
                let b = &out[(mirror_y * w as usize + x) * 4..(mirror_y * w as usize + x) * 4 + 4];
                assert_eq!(a, b, "y-mirror broke at s={s} y={y} x={x}");
            }
        }
    }
}

#[test]
fn impulse_response_approximate_mass_conservation() {
    // Sum of all alpha channels (the impulse colour was opaque
    // white, alpha == 255 in only one cell) after the blur should
    // be reasonably close to 255: a normalised analytical Gaussian
    // preserves total mass exactly, but `u8` per-pass quantisation
    // truncates the far-tail samples to zero, biasing the integer
    // sum slightly *downward*. We assert mass loss is bounded —
    // not zero. (A bug in the kernel normalisation would either
    // double-count or zero out the centre; either failure mode
    // would blow this bound spectacularly.)
    let w = 25u32;
    let h = 25u32;
    let mut img = build(w, h, |_, _| Rgba::new(0, 0, 0, 0));
    let off = ((h as usize / 2) * w as usize + w as usize / 2) * 4;
    img[off + 3] = 255;
    img[off] = 255;
    img[off + 1] = 255;
    img[off + 2] = 255;

    for s in [1.0f32, 1.5, 2.5] {
        let out = gaussian_blur(&img, w, h, s, s);
        let alpha_sum: u32 = out.chunks_exact(4).map(|c| c[3] as u32).sum();
        // The Gaussian footprint is well-contained inside a
        // 25×25 canvas for s ≤ 2.5 (3·s ≤ 7.5 ≪ 12), so virtually
        // no analytical mass leaks off the edge. The per-pass `u8`
        // quantisation rounds far-tail samples to zero — that's
        // physically expected (we're working with 8-bit pixels)
        // but it caps the integer alpha sum *below* the true 255.
        // Anything inside 50–255 is consistent with "mass
        // diffused, no integral blew up by a factor".
        assert!(
            (200..=265).contains(&(alpha_sum as i32)),
            "alpha mass off the bell-shape envelope at s={s}: sum = {alpha_sum}"
        );
    }
}

#[test]
fn pixels_wrapper_matches_byte_api_across_branches() {
    let w = 8u32;
    let h = 6u32;
    let pixels: Vec<Rgba> = (0..(w * h))
        .map(|i| Rgba::new((i * 7) as u8, (i * 11) as u8, (i * 13) as u8, 255))
        .collect();
    let bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
    for &(sx, sy) in &[(0.0, 0.0), (0.7, 0.7), (1.5, 0.0), (2.5, 2.5)] {
        let from_bytes = gaussian_blur(&bytes, w, h, sx, sy);
        let from_pixels = gaussian_blur_pixels(&pixels, w, h, sx, sy);
        let repacked: Vec<u8> = from_pixels
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(
            repacked, from_bytes,
            "wrapper drift at (sx, sy) = ({sx}, {sy})"
        );
    }
}

#[test]
fn axis_only_blur_along_x_does_not_smear_y() {
    // Take a buffer with two distinct rows — row 0 is bright red,
    // row 1+ is transparent black. An X-only blur (sy = 0) must
    // leave the row-1+ pixels untouched (no vertical energy flow).
    let w = 9u32;
    let h = 5u32;
    let mut img = build(w, h, |_, _| Rgba::new(0, 0, 0, 0));
    for x in 0..w as usize {
        let off = x * 4;
        img[off..off + 4].copy_from_slice(&[200, 100, 50, 255]);
    }
    let out = gaussian_blur(&img, w, h, 1.5, 0.0);
    for y in 1..h as usize {
        for x in 0..w as usize {
            let off = (y * w as usize + x) * 4;
            assert_eq!(
                &out[off..off + 4],
                &[0, 0, 0, 0],
                "vertical smear at y={y} x={x} despite sy=0"
            );
        }
    }
}

#[test]
fn box_branch_blur_smooths_step_edge_monotonically() {
    // A vertical step edge: left half of every row is opaque white,
    // right half is opaque black. After a horizontal Gaussian blur
    // the brightness profile should rise monotonically left-to-right
    // through the transition zone (if you measure top-to-bottom on
    // any one row; equivalently, intensity decreases monotonically
    // from left edge to right edge when the bright side is on the
    // left). This is a fundamental property of low-pass filtering.
    let w = 21u32;
    let h = 5u32;
    let mut img = build(w, h, |_, _| Rgba::new(0, 0, 0, 255));
    for y in 0..h as usize {
        for x in 0..(w as usize / 2) {
            let off = (y * w as usize + x) * 4;
            img[off..off + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    let out = gaussian_blur(&img, w, h, 2.5, 0.0);
    for y in 0..h as usize {
        let mut prev = 255u8;
        for x in 0..w as usize {
            let v = out[(y * w as usize + x) * 4];
            assert!(
                v <= prev,
                "non-monotone step-edge response at y={y} x={x}: {prev} -> {v}"
            );
            prev = v;
        }
    }
}
