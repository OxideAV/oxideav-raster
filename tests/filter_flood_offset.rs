//! Integration coverage for [`oxideav_raster::flood`] (SVG 1.1 §15.16
//! `<feFlood>`) and [`oxideav_raster::offset_filter`] (SVG 1.1 §15.21
//! `<feOffset>`).
//!
//! Unit-level operator algebra lives in `src/filter.rs`. This file is
//! the public-API smoke exercise — same shape as the existing
//! `filter_composite` / `filter_morphology` integration files.

use oxideav_core::Rgba;
use oxideav_raster::{flood, flood_pixels, offset_filter, offset_filter_pixels, OffsetEdge};

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

// ----- feFlood --------------------------------------------------------------

#[test]
fn flood_fills_buffer_with_constant_tuple() {
    // Every pixel is the same straight-alpha tuple; size is W*H*4 bytes.
    let w = 8u32;
    let h = 5u32;
    let color = Rgba::new(33, 88, 144, 255);
    let out = flood(w, h, color, 0.5);
    assert_eq!(out.len(), (w * h * 4) as usize);
    for chunk in out.chunks_exact(4) {
        assert_eq!(chunk[0], 33);
        assert_eq!(chunk[1], 88);
        assert_eq!(chunk[2], 144);
        // alpha = round(0.5*255) = 128 (half-up rounding).
        assert_eq!(chunk[3], 128);
    }
}

#[test]
fn flood_opacity_clamped_to_unit_range() {
    // §15.16 references the `<opacity-value>` syntax — out-of-range
    // values are clamped (we deliberately do not error so a caller
    // animating opacity past 1.0 keeps producing useful frames).
    let big = flood(1, 1, Rgba::new(10, 20, 30, 100), 9.0);
    assert_eq!(big, vec![10, 20, 30, 255]);
    let neg = flood(1, 1, Rgba::new(10, 20, 30, 100), -3.0);
    assert_eq!(neg, vec![10, 20, 30, 0]);
}

#[test]
fn flood_typed_wrapper_round_trips_through_bytes() {
    let bytes = flood(3, 2, Rgba::new(100, 0, 200, 255), 0.25);
    let pixels = flood_pixels(3, 2, Rgba::new(100, 0, 200, 255), 0.25);
    let flat: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
    assert_eq!(bytes, flat);
}

// ----- feOffset -------------------------------------------------------------

#[test]
fn offset_zero_is_identity() {
    let src = build(5, 4, |x, y| {
        Rgba::new((x * 30) as u8, (y * 40) as u8, 7, 255)
    });
    let out = offset_filter(&src, 5, 4, 0, 0, OffsetEdge::TransparentBlack);
    assert_eq!(out, src);
}

#[test]
fn offset_positive_dx_shifts_right_with_transparent_left_strip() {
    // Solid opaque square shifted by (+2, 0) leaves a 2-column
    // transparent-black strip on the left and pushes 2 columns out the
    // right edge.
    let src = build(4, 1, |_, _| Rgba::new(255, 0, 0, 255));
    let out = offset_filter(&src, 4, 1, 2, 0, OffsetEdge::TransparentBlack);
    assert_eq!(&out[0..4], &[0, 0, 0, 0]);
    assert_eq!(&out[4..8], &[0, 0, 0, 0]);
    assert_eq!(&out[8..12], &[255, 0, 0, 255]);
    assert_eq!(&out[12..16], &[255, 0, 0, 255]);
}

#[test]
fn offset_clamp_replicates_border_pixel() {
    // A unique sentinel in column 0 must propagate to the vacated cells
    // under clamp-to-edge for any positive dx.
    let src = build(4, 1, |x, _| {
        if x == 0 {
            Rgba::new(11, 22, 33, 255)
        } else {
            Rgba::new(200, 200, 200, 255)
        }
    });
    let out = offset_filter(&src, 4, 1, 2, 0, OffsetEdge::ClampToEdge);
    // out[0..2] sample sx ∈ {-2, -1} → clamp to 0 → sentinel.
    assert_eq!(&out[0..4], &[11, 22, 33, 255]);
    assert_eq!(&out[4..8], &[11, 22, 33, 255]);
    // out[2] = src[0] = sentinel; out[3] = src[1] = grey.
    assert_eq!(&out[8..12], &[11, 22, 33, 255]);
    assert_eq!(&out[12..16], &[200, 200, 200, 255]);
}

#[test]
fn offset_typed_wrapper_matches_byte_path() {
    let bytes = build(4, 3, |x, y| {
        Rgba::new((x * 30) as u8, (y * 40) as u8, 11, ((x + y) * 25) as u8)
    });
    let pixels: Vec<Rgba> = bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    for &(dx, dy) in &[(0, 0), (1, 0), (-2, 1), (3, -1), (10, 10)] {
        for edge in [OffsetEdge::TransparentBlack, OffsetEdge::ClampToEdge] {
            let via_bytes = offset_filter(&bytes, 4, 3, dx, dy, edge);
            let via_typed = offset_filter_pixels(&pixels, 4, 3, dx, dy, edge);
            let typed_bytes: Vec<u8> = via_typed
                .iter()
                .flat_map(|p| [p.r, p.g, p.b, p.a])
                .collect();
            assert_eq!(via_bytes, typed_bytes);
        }
    }
}

// ----- pipeline composition ------------------------------------------------

#[test]
fn flood_then_offset_pushes_solid_block_in_place() {
    // A flood produces a uniform buffer; offsetting it must produce a
    // (different-extent-aware) uniform interior with the vacated strip
    // following the requested edge mode. This is the typical "make a
    // coloured backdrop, slide it" recipe.
    let w = 5;
    let h = 4;
    let filled = flood(w, h, Rgba::new(50, 100, 150, 255), 1.0);
    let shifted = offset_filter(&filled, w, h, 2, 1, OffsetEdge::TransparentBlack);
    // Interior pixels (x ≥ 2 && y ≥ 1) are the original colour.
    for y in 1..h {
        for x in 2..w {
            let off = ((y * w + x) * 4) as usize;
            assert_eq!(
                &shifted[off..off + 4],
                &[50, 100, 150, 255],
                "interior pixel ({x}, {y}) lost flood colour"
            );
        }
    }
    // Vacated strip is transparent black.
    for x in 0..2 {
        let off = (x * 4) as usize;
        assert_eq!(&shifted[off..off + 4], &[0, 0, 0, 0]);
    }
    for x in 0..w {
        // Top row (y = 0) is entirely vacated by the dy = +1 shift.
        let off = (x * 4) as usize;
        assert_eq!(&shifted[off..off + 4], &[0, 0, 0, 0]);
    }
}
