//! Integration coverage for [`oxideav_raster::drop_shadow`] — the
//! Filter Effects Module Level 1 §9.12 `<feDropShadow>` shorthand
//! primitive.
//!
//! §9.12 defines `feDropShadow` purely as an equivalent primitive chain:
//!
//! ```text
//! <feGaussianBlur in="alpha-channel-of-feDropShadow-in" stdDeviation="…"/>
//! <feOffset dx="…" dy="…" result="offsetblur"/>
//! <feFlood flood-color="…" flood-opacity="…"/>
//! <feComposite in2="offsetblur" operator="in"/>
//! <feMerge>
//!   <feMergeNode/>
//!   <feMergeNode in="in-of-feDropShadow"/>
//! </feMerge>
//! ```
//!
//! These tests treat the public re-export as a black box and verify it
//! is byte-for-byte identical to evaluating the spec's equivalent chain
//! through the individual primitive entry points, plus the documented
//! structural behaviour (source graphic on top, shadow offset/coloured
//! per the attributes).

use oxideav_core::Rgba;
use oxideav_raster::{
    composite_filter, drop_shadow, drop_shadow_pixels, flood, gaussian_blur, merge, offset,
    CompositeOp, OffsetSampling,
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

/// Evaluate the §9.12 equivalent chain by hand through the individual
/// primitive entry points. The `drop_shadow` convenience function must
/// reproduce this exactly.
#[allow(clippy::too_many_arguments)]
fn manual_chain(
    src: &[u8],
    w: u32,
    h: u32,
    sx: f32,
    sy: f32,
    dx: f32,
    dy: f32,
    fr: u8,
    fg: u8,
    fb: u8,
    fo: f32,
    sampling: OffsetSampling,
) -> Vec<u8> {
    // Step 1: alpha channel of input (RGB zeroed), then blur.
    let mut alpha_only = vec![0u8; src.len()];
    for (d, s) in alpha_only.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        d[3] = s[3];
    }
    let blurred = gaussian_blur(&alpha_only, w, h, sx, sy);
    // Step 2: feOffset → "offsetblur".
    let offsetblur = offset(&blurred, w, h, dx, dy, sampling);
    // Step 3: feFlood.
    let flooded = flood(w, h, fr, fg, fb, fo);
    // Step 4: feComposite(flood in2="offsetblur" operator="in").
    let shadow = composite_filter(&flooded, &offsetblur, w, h, CompositeOp::In);
    // Step 5: feMerge — shadow below, source graphic on top.
    merge(w, h, &[&shadow, src])
}

#[test]
fn matches_the_spec_equivalent_chain_exactly() {
    // A small opaque white square on a transparent field.
    let src = build(9, 9, |x, y| {
        if (3..6).contains(&x) && (3..6).contains(&y) {
            Rgba::new(255, 255, 255, 255)
        } else {
            Rgba::new(0, 0, 0, 0)
        }
    });
    let got = drop_shadow(
        &src,
        9,
        9,
        1.5,
        1.5,
        2.0,
        2.0,
        0,
        0,
        0,
        1.0,
        OffsetSampling::Nearest,
    );
    let want = manual_chain(
        &src,
        9,
        9,
        1.5,
        1.5,
        2.0,
        2.0,
        0,
        0,
        0,
        1.0,
        OffsetSampling::Nearest,
    );
    assert_eq!(got, want);
}

#[test]
fn coloured_shadow_uses_flood_color() {
    // Opaque source pixel at (1,1); zero blur + zero offset means the
    // shadow lands exactly under the source, but the source (drawn on
    // top by the final feMerge) wins where it is opaque. A red flood
    // colour must appear wherever the offset shadow extends beyond the
    // source — here, with dx=dy=0 and no blur, the shadow is fully
    // covered, so we shift it so the shadow peeks out.
    let src = build(5, 5, |x, y| {
        if (x, y) == (1, 1) {
            Rgba::new(20, 200, 20, 255) // green
        } else {
            Rgba::new(0, 0, 0, 0)
        }
    });
    // No blur, offset by (+2, +2) so the shadow sits at (3,3); red flood.
    let out = drop_shadow(
        &src,
        5,
        5,
        0.0,
        0.0,
        2.0,
        2.0,
        255,
        0,
        0,
        1.0,
        OffsetSampling::Nearest,
    );
    let at = |x: usize, y: usize| {
        let i = (y * 5 + x) * 4;
        [out[i], out[i + 1], out[i + 2], out[i + 3]]
    };
    // Source graphic is preserved on top at (1,1).
    assert_eq!(at(1, 1), [20, 200, 20, 255]);
    // Red, fully-opaque shadow appears at the offset position (3,3).
    assert_eq!(at(3, 3), [255, 0, 0, 255]);
    // Everywhere else is transparent.
    assert_eq!(at(0, 0), [0, 0, 0, 0]);
}

#[test]
fn fully_transparent_input_yields_transparent_output() {
    // The shadow derives from the input alpha; an all-transparent input
    // has an all-zero alpha channel, so blur → offset → composite-in all
    // stay transparent, and the merge of two transparent buffers is
    // transparent.
    let src = vec![0u8; 6 * 4 * 4];
    let out = drop_shadow(
        &src,
        6,
        4,
        2.0,
        2.0,
        2.0,
        2.0,
        255,
        128,
        64,
        1.0,
        OffsetSampling::Nearest,
    );
    assert!(out.iter().all(|&b| b == 0));
}

#[test]
fn source_graphic_is_drawn_on_top() {
    // An opaque blue source must survive verbatim where it is opaque,
    // regardless of the shadow underneath it.
    let src = build(7, 7, |_, _| Rgba::new(0, 0, 255, 255));
    let out = drop_shadow(
        &src,
        7,
        7,
        2.0,
        2.0,
        3.0,
        3.0,
        0,
        0,
        0,
        1.0,
        OffsetSampling::Nearest,
    );
    // Every pixel: opaque source over (anything) = opaque source.
    for px in out.chunks_exact(4) {
        assert_eq!(px, [0, 0, 255, 255]);
    }
}

#[test]
fn flood_opacity_scales_shadow_alpha() {
    // A lone opaque pixel, no blur, offset out from under the source so
    // the shadow is visible; flood-opacity 0.5 must halve the shadow's
    // alpha (round_half_up(0.5·255) = 128) where the offset alpha is
    // fully opaque.
    let src = build(5, 5, |x, y| {
        if (x, y) == (0, 0) {
            Rgba::new(255, 255, 255, 255)
        } else {
            Rgba::new(0, 0, 0, 0)
        }
    });
    let out = drop_shadow(
        &src,
        5,
        5,
        0.0,
        0.0,
        2.0,
        2.0,
        10,
        20,
        30,
        0.5,
        OffsetSampling::Nearest,
    );
    let i = (2 * 5 + 2) * 4; // (2,2)
    assert_eq!(&out[i..i + 4], &[10, 20, 30, 128]);
}

#[test]
fn typed_path_agrees_with_byte_path() {
    let bytes = build(8, 6, |x, y| {
        Rgba::new((x * 30) as u8, (y * 40) as u8, ((x + y) * 20) as u8, 200)
    });
    let typed: Vec<Rgba> = bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let byte_out = drop_shadow(
        &bytes,
        8,
        6,
        1.2,
        2.4,
        1.5,
        -1.5,
        200,
        100,
        50,
        0.8,
        OffsetSampling::Bilinear,
    );
    let typed_out = drop_shadow_pixels(
        &typed,
        8,
        6,
        1.2,
        2.4,
        1.5,
        -1.5,
        200,
        100,
        50,
        0.8,
        OffsetSampling::Bilinear,
    );
    let from_typed: Vec<u8> = typed_out
        .iter()
        .flat_map(|p| [p.r, p.g, p.b, p.a])
        .collect();
    assert_eq!(byte_out, from_typed);
}

#[test]
fn anisotropic_std_deviation_blurs_each_axis_independently() {
    // A single opaque pixel blurred with std_x large, std_y zero, no
    // offset: the shadow must spread horizontally but not vertically.
    let src = build(11, 11, |x, y| {
        if (x, y) == (5, 5) {
            Rgba::new(255, 255, 255, 255)
        } else {
            Rgba::new(0, 0, 0, 0)
        }
    });
    let out = drop_shadow(
        &src,
        11,
        11,
        2.5,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        1.0,
        OffsetSampling::Nearest,
    );
    let alpha = |x: usize, y: usize| out[(y * 11 + x) * 4 + 3];
    // Horizontal neighbour of the centre row picks up shadow alpha.
    assert!(alpha(3, 5) > 0, "horizontal spread expected");
    assert!(alpha(7, 5) > 0, "horizontal spread expected");
    // Vertical neighbour off the centre row stays transparent (no y blur,
    // and the source pixel at (5,5) is drawn on top there only at (5,5)).
    assert_eq!(alpha(5, 3), 0, "no vertical spread expected");
    assert_eq!(alpha(5, 7), 0, "no vertical spread expected");
}

#[test]
#[should_panic(expected = "stdDeviation must be non-negative")]
fn negative_std_deviation_panics() {
    let src = vec![0u8; 4 * 4 * 4];
    let _ = drop_shadow(
        &src,
        4,
        4,
        -1.0,
        1.0,
        0.0,
        0.0,
        0,
        0,
        0,
        1.0,
        OffsetSampling::Nearest,
    );
}

#[test]
fn zero_extent_returns_empty() {
    let out = drop_shadow(
        &[],
        0,
        0,
        2.0,
        2.0,
        2.0,
        2.0,
        0,
        0,
        0,
        1.0,
        OffsetSampling::Nearest,
    );
    assert!(out.is_empty());
}
