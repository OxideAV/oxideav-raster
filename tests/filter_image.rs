//! Integration coverage for [`oxideav_raster::image_source`] — the
//! SVG 1.1 §15.18 `<feImage>` filter primitive.
//!
//! The unit tests inside `src/filter.rs::image_source_tests` cover the
//! §7.8 fitting algebra at small extents (default value, align-none
//! stretch, meet bands, slice crop, bilinear weights, degenerate
//! extents, panic guards). This file exercises the public re-exports
//! as a consumer would: realistic raster sizes, every alignment
//! anchor, and composition with the other filter primitives in the
//! chain.

use oxideav_core::Rgba;
use oxideav_raster::{
    image_source, image_source_pixels, merge, AspectRatioAlign, ImageSourceSampling, MeetOrSlice,
    PreserveAspectRatio,
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

fn px(out: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * w + x) * 4) as usize;
    [out[i], out[i + 1], out[i + 2], out[i + 3]]
}

#[test]
fn meet_band_geometry_at_realistic_extents() {
    // 32×32 opaque source into a 96×48 region (§15.18 default
    // xMidYMid meet). Uniform scale = min(3, 1.5) = 1.5 → the fitted
    // image is 48×48, centred horizontally at tx = (96 − 48)/2 = 24.
    // Columns 24..72 carry the image, the rest is the §15.7.3
    // transparent black; rows are fully covered.
    let src = build(32, 32, |_, _| Rgba::new(200, 150, 100, 255));
    let out = image_source(
        &src,
        32,
        32,
        96,
        48,
        PreserveAspectRatio::default(),
        ImageSourceSampling::Nearest,
    );
    assert_eq!(out.len(), 96 * 48 * 4);
    for y in [0u32, 24, 47] {
        for x in 0..96u32 {
            let expect = if (24..72).contains(&x) {
                [200, 150, 100, 255]
            } else {
                [0, 0, 0, 0]
            };
            assert_eq!(px(&out, 96, x, y), expect, "({x},{y})");
        }
    }
}

#[test]
fn slice_geometry_covers_every_pixel_at_realistic_extents() {
    // 32×32 source into 96×48 with xMidYMid slice: scale = max(3,
    // 1.5) = 3 → the fitted image is 96×96 and the vertical overhang
    // is cropped symmetrically. No output pixel may stay transparent.
    let src = build(32, 32, |x, y| {
        Rgba::new((x * 8) as u8, (y * 8) as u8, 0, 255)
    });
    let par = PreserveAspectRatio {
        align: AspectRatioAlign::XMidYMid,
        meet_or_slice: MeetOrSlice::Slice,
    };
    let out = image_source(&src, 32, 32, 96, 48, par, ImageSourceSampling::Nearest);
    for y in 0..48u32 {
        for x in 0..96u32 {
            assert_eq!(px(&out, 96, x, y)[3], 255, "uncovered pixel at ({x},{y})");
        }
    }
    // Centred crop: ty = (48 − 96)/2 = −24, so output row 0 reads
    // source row round((0.5 + 24)/3 − 0.5) = round(7.666) = 8.
    assert_eq!(px(&out, 96, 0, 0)[1], 8 * 8);
    // And the bottom row reads source row round((47.5 + 24)/3 − 0.5)
    // = round(23.333) = 23.
    assert_eq!(px(&out, 96, 0, 47)[1], 23 * 8);
}

#[test]
fn all_nine_uniform_anchors_place_the_image_in_distinct_positions() {
    // 10×10 source into 30×30 under meet — scale 3 fills the target
    // exactly, so anchors coincide. Use a 30×12 target instead:
    // scale = min(3, 1.2) = 1.2 → fitted size 12×12, X anchor picks
    // tx ∈ {0, 9, 18}, Y is fully covered. Then a 12×30 target for
    // the Y anchors.
    let src = build(10, 10, |_, _| Rgba::new(255, 255, 255, 255));
    let x_cases = [
        (AspectRatioAlign::XMinYMin, 0u32),
        (AspectRatioAlign::XMidYMid, 9),
        (AspectRatioAlign::XMaxYMax, 18),
    ];
    for (align, x_origin) in x_cases {
        let par = PreserveAspectRatio {
            align,
            meet_or_slice: MeetOrSlice::Meet,
        };
        let out = image_source(&src, 10, 10, 30, 12, par, ImageSourceSampling::Nearest);
        for x in 0..30u32 {
            let inside = (x_origin..x_origin + 12).contains(&x);
            assert_eq!(
                px(&out, 30, x, 6)[3],
                if inside { 255 } else { 0 },
                "{align:?} column {x}"
            );
        }
    }
    let y_cases = [
        (AspectRatioAlign::XMinYMin, 0u32),
        (AspectRatioAlign::XMidYMid, 9),
        (AspectRatioAlign::XMaxYMax, 18),
    ];
    for (align, y_origin) in y_cases {
        let par = PreserveAspectRatio {
            align,
            meet_or_slice: MeetOrSlice::Meet,
        };
        let out = image_source(&src, 10, 10, 12, 30, par, ImageSourceSampling::Nearest);
        for y in 0..30u32 {
            let inside = (y_origin..y_origin + 12).contains(&y);
            assert_eq!(
                px(&out, 12, 6, y)[3],
                if inside { 255 } else { 0 },
                "{align:?} row {y}"
            );
        }
    }
}

#[test]
fn align_none_upscale_keeps_blocks_constant_under_nearest() {
    // A 2×2 block image stretched ×8 with align none: nearest keeps
    // each 8×8 output block constant at its source-pixel colour.
    let src = build(2, 2, |x, y| {
        Rgba::new((x * 255) as u8, (y * 255) as u8, 128, 255)
    });
    let par = PreserveAspectRatio {
        align: AspectRatioAlign::None,
        meet_or_slice: MeetOrSlice::Meet,
    };
    let out = image_source(&src, 2, 2, 16, 16, par, ImageSourceSampling::Nearest);
    for oy in 0..16u32 {
        for ox in 0..16u32 {
            // u = (ox + 0.5)/8 − 0.5 rounds to ox / 8 for every
            // column of the block (and symmetrically in y).
            let sx = ox / 8;
            let sy = oy / 8;
            assert_eq!(
                px(&out, 16, ox, oy),
                [(sx * 255) as u8, (sy * 255) as u8, 128, 255],
                "block pixel ({ox},{oy})"
            );
        }
    }
}

#[test]
fn align_none_upscale_bilinear_hits_the_exact_midpoint_between_blocks() {
    // Same ×8 stretch, bilinear: the output column whose mapped
    // coordinate is exactly halfway between the two source pixels
    // (u = 0.5 at ox = 7.5 ± — use ox = 7: u = (7.5)/8 − 0.5 =
    // 0.4375; ox = 8: u = 0.5625) brackets the 50% blend. Check a
    // pure-arithmetic case instead at scale ×2: ox = 0 maps to
    // u = −0.25 (blend 25% toward the out-of-range transparent
    // neighbour is avoided by using a 4-wide source so all samples
    // stay interior at ox ≥ 1).
    let src = build(4, 1, |x, _| Rgba::new((x * 60) as u8, 0, 0, 255));
    let par = PreserveAspectRatio {
        align: AspectRatioAlign::None,
        meet_or_slice: MeetOrSlice::Meet,
    };
    let out = image_source(&src, 4, 1, 8, 1, par, ImageSourceSampling::Bilinear);
    // ox = 1 → u = (1.5)/2 − 0.5 = 0.25 → 0.75·src[0] + 0.25·src[1]
    //   = 0.25 · 60 = 15.
    assert_eq!(px(&out, 8, 1, 0), [15, 0, 0, 255]);
    // ox = 2 → u = 0.75 → 0.25·src[0] + 0.75·src[1] = 45.
    assert_eq!(px(&out, 8, 2, 0), [45, 0, 0, 255]);
    // ox = 4 → u = 1.75 → 0.25·60 + 0.75·120 = 105.
    assert_eq!(px(&out, 8, 4, 0), [105, 0, 0, 255]);
}

#[test]
fn image_source_feeds_the_merge_primitive_like_a_filter_chain() {
    // A §15.18 result is just another filter-chain buffer: place a
    // small opaque image centred via meet, then `feMerge` it over a
    // flood-like opaque backdrop. Outside the fitted image the §15.7.3
    // transparent band must let the backdrop through unchanged.
    let backdrop = build(24, 8, |_, _| Rgba::new(0, 0, 255, 255));
    let icon = build(8, 8, |_, _| Rgba::new(255, 0, 0, 255));
    let placed = image_source(
        &icon,
        8,
        8,
        24,
        8,
        PreserveAspectRatio::default(),
        ImageSourceSampling::Nearest,
    );
    let merged = merge(24, 8, &[&backdrop, &placed]);
    for x in 0..24u32 {
        let expect = if (8..16).contains(&x) {
            [255, 0, 0, 255] // icon (top layer) where placed
        } else {
            [0, 0, 255, 255] // backdrop shows through the meet band
        };
        assert_eq!(px(&merged, 24, x, 4), expect, "column {x}");
    }
}

#[test]
fn typed_and_byte_paths_agree_on_an_asymmetric_slice() {
    let src_bytes = build(9, 6, |x, y| {
        Rgba::new((x * 28) as u8, (y * 42) as u8, ((x * y) % 256) as u8, 230)
    });
    let src_typed: Vec<Rgba> = src_bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let par = PreserveAspectRatio {
        align: AspectRatioAlign::XMinYMax,
        meet_or_slice: MeetOrSlice::Slice,
    };
    let bytes_out = image_source(&src_bytes, 9, 6, 20, 17, par, ImageSourceSampling::Bilinear);
    let typed_out =
        image_source_pixels(&src_typed, 9, 6, 20, 17, par, ImageSourceSampling::Bilinear);
    let from_typed: Vec<u8> = typed_out
        .iter()
        .flat_map(|p| [p.r, p.g, p.b, p.a])
        .collect();
    assert_eq!(bytes_out, from_typed);
}
