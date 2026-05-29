//! Integration coverage for [`oxideav_raster::composite_filter`] — the
//! SVG 1.1 §15.12 `<feComposite>` primitive.
//!
//! The unit tests inside `src/filter.rs` cover the per-operator algebra
//! (over/in/out/atop/xor blend factors, the arithmetic formula, range
//! clamping, panic on bad input). This file is the consumer-facing API
//! exercise — treating the public re-exports as a black box and checking
//! the documented behaviour, the same shape as the existing
//! `tests/filter_morphology.rs` / `tests/filter_color_matrix.rs`.

use oxideav_core::Rgba;
use oxideav_raster::{composite_filter, composite_filter_pixels, CompositeOp};

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
fn over_with_transparent_in2_returns_in1() {
    // §14.2 `over` with an empty backdrop (in2) is the identity on the
    // foreground (in1): Cr' = (1 - Ea)*0 + Er = Er, αo = αa.
    let a = build(4, 3, |x, y| {
        Rgba::new((x * 30) as u8, (y * 40) as u8, 200, ((x + y) * 20) as u8)
    });
    let b = build(4, 3, |_, _| Rgba::new(99, 99, 99, 0));
    let out = composite_filter(&a, &b, 4, 3, CompositeOp::Over);
    // Where in1 is itself transparent the colour is normalised to
    // (0,0,0,0); compare against that normalisation of in1.
    let want: Vec<u8> = a
        .chunks_exact(4)
        .flat_map(|c| {
            if c[3] == 0 {
                [0, 0, 0, 0]
            } else {
                [c[0], c[1], c[2], c[3]]
            }
        })
        .collect();
    assert_eq!(out, want);
}

#[test]
fn in_then_out_partition_alpha() {
    // For any in1 with opaque in2, (in1 IN in2).alpha + (in1 OUT in2)
    // .alpha == in1.alpha, because αb = 1 ⇒ IN keeps αa·1 and OUT keeps
    // αa·(1-1) = 0; with αb = 0 the roles swap. Test the αb = 1 case.
    let a = build(3, 1, |x, _| Rgba::new(10, 20, 30, (x * 90) as u8));
    let b = build(3, 1, |_, _| Rgba::new(0, 0, 0, 255));
    let inp = composite_filter(&a, &b, 3, 1, CompositeOp::In);
    let outp = composite_filter(&a, &b, 3, 1, CompositeOp::Out);
    for px in 0..3 {
        let i = px * 4 + 3;
        // IN keeps the full source alpha, OUT is fully removed.
        assert_eq!(inp[i], a[i], "IN alpha pixel {px}");
        assert_eq!(outp[i], 0, "OUT alpha pixel {px}");
    }
}

#[test]
fn arithmetic_average_of_two_opaque_inputs() {
    // result = 0.5*i1 + 0.5*i2 (k1=k4=0, k2=k3=0.5). Opaque red and
    // opaque blue average to (~128, 0, ~128) at full alpha.
    let a = build(1, 1, |_, _| Rgba::new(255, 0, 0, 255));
    let b = build(1, 1, |_, _| Rgba::new(0, 0, 255, 255));
    let out = composite_filter(
        &a,
        &b,
        1,
        1,
        CompositeOp::Arithmetic {
            k1: 0.0,
            k2: 0.5,
            k3: 0.5,
            k4: 0.0,
        },
    );
    assert!((out[0] as i32 - 128).abs() <= 2, "r {}", out[0]);
    assert_eq!(out[1], 0);
    assert!((out[2] as i32 - 128).abs() <= 2, "b {}", out[2]);
    assert_eq!(out[3], 255);
}

#[test]
fn typed_and_byte_paths_agree_across_operators() {
    let a_b = build(6, 5, |x, y| {
        Rgba::new(
            (x * 40) as u8,
            (y * 50) as u8,
            ((x ^ y) * 30) as u8,
            ((x + y) * 22) as u8,
        )
    });
    let b_b = build(6, 5, |x, y| {
        Rgba::new(13, (y * 45) as u8, (x * 35) as u8, (200 - x * 20) as u8)
    });
    let a_p: Vec<Rgba> = a_b
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let b_p: Vec<Rgba> = b_b
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    for op in [
        CompositeOp::Over,
        CompositeOp::In,
        CompositeOp::Out,
        CompositeOp::Atop,
        CompositeOp::Xor,
        CompositeOp::Arithmetic {
            k1: 0.3,
            k2: 0.4,
            k3: 0.2,
            k4: 0.05,
        },
    ] {
        let via_bytes = composite_filter(&a_b, &b_b, 6, 5, op);
        let via_typed = composite_filter_pixels(&a_p, &b_p, 6, 5, op);
        let typed_bytes: Vec<u8> = via_typed
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(via_bytes, typed_bytes, "byte vs typed mismatch for {op:?}");
    }
}
