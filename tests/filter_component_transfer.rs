//! Integration tests for the SVG 1.1 §15.11 `<feComponentTransfer>`
//! public API exposed by `oxideav-raster`. The unit-level tests live in
//! `src/filter.rs::component_transfer_tests` and cover the per-mode
//! branches; this file exercises the **public** call shape (byte API
//! plus typed-pixel wrapper) on whole-frame buffers and verifies the
//! four `type=` modes called out in the round brief end-to-end.

use oxideav_core::Rgba;
use oxideav_raster::{
    component_transfer, component_transfer_pixels, ComponentTransfer, TransferFunc,
};

fn checker_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let on = ((x + y) & 1) == 0;
            let r = if on { 200 } else { 50 };
            let g = if on { 100 } else { 150 };
            v.extend_from_slice(&[r as u8, g as u8, 60, 220]);
        }
    }
    v
}

#[test]
fn public_api_all_identity_is_byte_passthrough() {
    let img = checker_rgba(8, 8);
    let ct = ComponentTransfer::identity();
    let out = component_transfer(&img, 8, 8, &ct);
    assert_eq!(out, img);
}

#[test]
fn public_api_linear_invert_on_red_only() {
    // R' = 1 − R; G, B, A unchanged.
    let img = checker_rgba(4, 4);
    let ct = ComponentTransfer::identity().with_r(TransferFunc::Linear {
        slope: -1.0,
        intercept: 1.0,
    });
    let out = component_transfer(&img, 4, 4, &ct);
    for i in (0..img.len()).step_by(4) {
        let want_r = 255 - img[i] as i32;
        let got_r = out[i] as i32;
        assert!(
            (got_r - want_r).abs() <= 1,
            "R got {got_r} want {want_r} at pixel {}",
            i / 4
        );
        assert_eq!(out[i + 1], img[i + 1]);
        assert_eq!(out[i + 2], img[i + 2]);
        assert_eq!(out[i + 3], img[i + 3]);
    }
}

#[test]
fn public_api_table_two_entry_zero_one_is_identity() {
    // §15.11 Table(N=1) with v = [0, 1] is the identity function.
    let img = checker_rgba(6, 6);
    let ct = ComponentTransfer::identity().with_g(TransferFunc::Table(vec![0.0, 1.0]));
    let out = component_transfer(&img, 6, 6, &ct);
    for i in 0..img.len() {
        let d = (out[i] as i32 - img[i] as i32).abs();
        assert!(d <= 1, "table-identity differs by {d} at byte {i}");
    }
}

#[test]
fn public_api_discrete_two_entry_threshold_at_half() {
    // Discrete(vec![0.0, 1.0]) with N=2 partitions [0, 1] into two
    // half-open buckets meeting at C = 0.5 (== 127.5 in 8-bit). The
    // floor-style break point sits at byte 128. Test a hand-picked
    // pair of samples that straddle the break and confirm the step.
    let pixels: Vec<u8> = vec![0, 100, 127, 128, 200, 255];
    let mut img = Vec::new();
    for r in &pixels {
        img.extend_from_slice(&[*r, 0, 0, 0]);
    }
    let ct = ComponentTransfer::identity().with_r(TransferFunc::Discrete(vec![0.0, 1.0]));
    let out = component_transfer(&img, 6, 1, &ct);
    let want: [u8; 6] = [0, 0, 0, 255, 255, 255];
    for (idx, &w) in want.iter().enumerate() {
        assert_eq!(
            out[idx * 4],
            w,
            "discrete-half[{idx}] R for input {} got {}, want {w}",
            pixels[idx],
            out[idx * 4]
        );
    }
}

#[test]
fn public_api_gamma_two_two_round_trips_with_inverse() {
    // Forward gamma 2.2 then inverse gamma 1/2.2 must reproduce every
    // pixel within ±1 LSB (two u8→f32→u8 round-trips, ≤½ LSB each).
    let img = checker_rgba(5, 5);
    let fwd = ComponentTransfer {
        r: TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 2.2,
            offset: 0.0,
        },
        g: TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 2.2,
            offset: 0.0,
        },
        b: TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 2.2,
            offset: 0.0,
        },
        a: TransferFunc::Identity,
    };
    let inv = ComponentTransfer {
        r: TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 1.0 / 2.2,
            offset: 0.0,
        },
        g: TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 1.0 / 2.2,
            offset: 0.0,
        },
        b: TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 1.0 / 2.2,
            offset: 0.0,
        },
        a: TransferFunc::Identity,
    };
    let bent = component_transfer(&img, 5, 5, &fwd);
    let back = component_transfer(&bent, 5, 5, &inv);
    for i in 0..img.len() {
        let d = (back[i] as i32 - img[i] as i32).abs();
        assert!(
            d <= 1,
            "gamma round-trip differs by {d} at byte {i} ({} → {} → {})",
            img[i],
            bent[i],
            back[i]
        );
    }
}

#[test]
fn public_api_pixels_wrapper_matches_byte_api() {
    // The Vec<Rgba> wrapper must produce a result that is the
    // typed-pixel translation of the byte API output for the same
    // ComponentTransfer.
    let bytes = checker_rgba(4, 4);
    let pixels: Vec<Rgba> = bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let ct = ComponentTransfer {
        r: TransferFunc::Linear {
            slope: 0.5,
            intercept: 0.25,
        },
        g: TransferFunc::Discrete(vec![0.0, 0.5, 1.0]),
        b: TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 1.5,
            offset: 0.0,
        },
        a: TransferFunc::Identity,
    };
    let via_bytes = component_transfer(&bytes, 4, 4, &ct);
    let via_pixels = component_transfer_pixels(&pixels, 4, 4, &ct);
    let want: Vec<Rgba> = via_bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    assert_eq!(via_pixels, want);
}

#[test]
fn public_api_default_is_identity() {
    // ComponentTransfer::default() == ComponentTransfer::identity();
    // the byte path is an exact copy.
    let img = checker_rgba(3, 3);
    let ct = ComponentTransfer::default();
    let out = component_transfer(&img, 3, 3, &ct);
    assert_eq!(out, img);
}

#[test]
fn public_api_linear_intercept_one_saturates_to_full() {
    // Linear { slope: 0, intercept: 1 } on every channel forces every
    // output pixel to (255, 255, 255, 255) regardless of input —
    // tests that the output clamp is applied per-channel.
    let img = checker_rgba(2, 2);
    let ct = ComponentTransfer {
        r: TransferFunc::Linear {
            slope: 0.0,
            intercept: 1.0,
        },
        g: TransferFunc::Linear {
            slope: 0.0,
            intercept: 1.0,
        },
        b: TransferFunc::Linear {
            slope: 0.0,
            intercept: 1.0,
        },
        a: TransferFunc::Linear {
            slope: 0.0,
            intercept: 1.0,
        },
    };
    let out = component_transfer(&img, 2, 2, &ct);
    for byte in out {
        assert_eq!(byte, 255);
    }
}

#[test]
fn public_api_linear_intercept_minus_one_clamps_to_zero() {
    // Linear { slope: 1, intercept: -1 } sends every input C ∈ [0, 1]
    // to C - 1 ≤ 0; the output clamp drops everything to 0.
    let img = checker_rgba(2, 2);
    let ct = ComponentTransfer {
        r: TransferFunc::Linear {
            slope: 1.0,
            intercept: -1.0,
        },
        g: TransferFunc::Linear {
            slope: 1.0,
            intercept: -1.0,
        },
        b: TransferFunc::Linear {
            slope: 1.0,
            intercept: -1.0,
        },
        a: TransferFunc::Linear {
            slope: 1.0,
            intercept: -1.0,
        },
    };
    let out = component_transfer(&img, 2, 2, &ct);
    for byte in out {
        assert_eq!(byte, 0);
    }
}
