//! Differential gates for the r449 filter-kernel optimisations —
//! each optimised kernel is pinned byte-identical to the direct
//! formulation it replaced, on randomised inputs.

use oxideav_raster::{
    component_transfer, morphology, to_linear_rgb, to_srgb, ComponentTransfer, MorphologyOp,
    TransferFunc,
};

fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        })
        .collect()
}

/// Naive per-pixel sliding-window 2-D morphology (clamp-to-edge), the
/// direct formulation the van Herk / Gil-Werman decomposition
/// replaced. Min / max over the same window is order-independent, so
/// the optimised path must match byte-for-byte.
fn morphology_naive(
    src: &[u8],
    width: u32,
    height: u32,
    rx: u32,
    ry: u32,
    op: MorphologyOp,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; src.len()];
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(rx as usize);
            let x1 = (x + rx as usize).min(w - 1);
            let y0 = y.saturating_sub(ry as usize);
            let y1 = (y + ry as usize).min(h - 1);
            let mut acc = match op {
                MorphologyOp::Erode => [255u8; 4],
                MorphologyOp::Dilate => [0u8; 4],
            };
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let i = (yy * w + xx) * 4;
                    for c in 0..4 {
                        acc[c] = match op {
                            MorphologyOp::Erode => acc[c].min(src[i + c]),
                            MorphologyOp::Dilate => acc[c].max(src[i + c]),
                        };
                    }
                }
            }
            out[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&acc);
        }
    }
    out
}

#[test]
fn morphology_matches_naive_sliding_window() {
    for (w, h, seed) in [(17u32, 13u32, 0xAAu64), (32, 32, 0xBB), (5, 40, 0xCC)] {
        let img = lcg_bytes((w * h * 4) as usize, seed);
        for op in [MorphologyOp::Erode, MorphologyOp::Dilate] {
            for (rx, ry) in [(0, 1), (1, 0), (1, 1), (3, 2), (7, 15), (40, 3), (100, 100)] {
                let got = morphology(&img, w, h, rx, ry, op);
                let want = morphology_naive(&img, w, h, rx, ry, op);
                assert_eq!(
                    got, want,
                    "morphology {op:?} r=({rx},{ry}) {w}x{h} diverged"
                );
            }
        }
    }
}

#[test]
fn component_transfer_matches_direct_per_pixel_formula() {
    // The direct formulation: normalise, apply, quantise per pixel
    // (mirroring the spec formula the LUT build replicates).
    fn quantise_unit(v: f32) -> u8 {
        if v.is_nan() {
            return 0;
        }
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    }
    let ct = ComponentTransfer::identity()
        .with_r(TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 2.2,
            offset: 0.0,
        })
        .with_g(TransferFunc::Linear {
            slope: 1.2,
            intercept: -0.1,
        })
        .with_b(TransferFunc::Table(vec![0.0, 0.6, 0.4, 1.0]))
        .with_a(TransferFunc::Discrete(vec![0.1, 0.5, 0.9]));
    let img = lcg_bytes(64 * 64 * 4, 0xD1FF);
    let got = component_transfer(&img, 64, 64, &ct);
    let funcs = [&ct.r, &ct.g, &ct.b, &ct.a];
    let want: Vec<u8> = img
        .iter()
        .enumerate()
        .map(|(i, &b)| quantise_unit(funcs[i % 4].apply(b as f32 / 255.0)))
        .collect();
    assert_eq!(got, want);
}

#[test]
fn colorspace_byte_luts_match_direct_transfer_functions() {
    use oxideav_raster::{linear_to_srgb_f32, srgb_to_linear_f32};
    let img = lcg_bytes(4096, 0xF0F0);

    let mut lin = img.clone();
    to_linear_rgb(&mut lin);
    let mut srgb = img.clone();
    to_srgb(&mut srgb);

    for (i, &b) in img.iter().enumerate() {
        if i % 4 == 3 {
            assert_eq!(lin[i], b, "alpha byte must pass through");
            assert_eq!(srgb[i], b, "alpha byte must pass through");
            continue;
        }
        let want_lin = (srgb_to_linear_f32(b as f32 / 255.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;
        let want_srgb = (linear_to_srgb_f32(b as f32 / 255.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;
        assert_eq!(lin[i], want_lin, "to_linear_rgb byte {b} diverged");
        assert_eq!(srgb[i], want_srgb, "to_srgb byte {b} diverged");
    }
}
