//! Integration coverage for [`oxideav_raster::tile`] — the SVG 1.1
//! §15.23 `<feTile>` primitive.
//!
//! The unit tests inside `src/filter.rs::tile_tests` cover the
//! per-primitive placement algebra (origin shift, negative tile
//! origin, periodicity, degenerate extents). This file exercises the
//! public re-export and a couple of larger-shape integration scenarios
//! the §15.23 description sketches: filling a target rectangle from a
//! smaller reference tile, lining up the §15.23 `(x, y)` origin to an
//! arbitrary offset of the target rectangle, and composing
//! `tile → offset` into the "shifted-then-tiled" pipeline a §15.23
//! `feTile` is typically combined with on top of `feOffset`.

use oxideav_core::Rgba;
use oxideav_raster::{offset, tile, tile_pixels, OffsetSampling};

/// Build a packed-RGBA buffer of `w · h` pixels coloured by `f`.
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
fn tile_fills_a_4x_larger_target_from_a_checker_tile() {
    // 2×2 checker reference tile: white at (0,0)+(1,1), black
    // at (0,1)+(1,0).
    let src = build(2, 2, |x, y| {
        if (x + y) % 2 == 0 {
            Rgba::opaque(255, 255, 255)
        } else {
            Rgba::opaque(0, 0, 0)
        }
    });
    let out = tile(&src, 2, 2, 0, 0, 8, 8);
    assert_eq!(out.len(), 8 * 8 * 4);
    // The output must remain a 2-wide checkerboard everywhere.
    for y in 0..8 {
        for x in 0..8 {
            let p = (y * 8 + x) * 4;
            let expected = if (x + y) % 2 == 0 { 255 } else { 0 };
            assert_eq!(out[p], expected, "px ({x}, {y}) r");
            assert_eq!(out[p + 1], expected);
            assert_eq!(out[p + 2], expected);
            assert_eq!(out[p + 3], 255);
        }
    }
}

#[test]
fn tile_origin_offset_phase_shifts_the_pattern() {
    // Horizontal stripe tile: red row on top, blue row below.
    let src = build(1, 2, |_, y| {
        if y == 0 {
            Rgba::opaque(255, 0, 0)
        } else {
            Rgba::opaque(0, 0, 255)
        }
    });
    // With tile_y = 0 the target's row 0 is red, row 1 is blue,
    // row 2 is red, etc. (the §15.23 base case).
    let base = tile(&src, 1, 2, 0, 0, 1, 4);
    assert_eq!(&base[0..4], [255, 0, 0, 255]);
    assert_eq!(&base[4..8], [0, 0, 255, 255]);
    assert_eq!(&base[8..12], [255, 0, 0, 255]);
    assert_eq!(&base[12..16], [0, 0, 255, 255]);

    // Shifting the tile origin down by one row swaps the two
    // alternating colours (the §15.23 placement rule applied with
    // `tile_y = 1` lines source row 0 up with target row 1, etc.).
    let shifted = tile(&src, 1, 2, 0, 1, 1, 4);
    assert_eq!(&shifted[0..4], [0, 0, 255, 255]);
    assert_eq!(&shifted[4..8], [255, 0, 0, 255]);
    assert_eq!(&shifted[8..12], [0, 0, 255, 255]);
    assert_eq!(&shifted[12..16], [255, 0, 0, 255]);
}

#[test]
fn tile_pixels_round_trips_through_typed_wrapper() {
    let src = build(3, 3, |x, y| Rgba::opaque(x as u8 * 32, y as u8 * 32, 200));
    let typed: Vec<Rgba> = src
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let bytes_out = tile(&src, 3, 3, -2, 5, 7, 4);
    let typed_out = tile_pixels(&typed, 3, 3, -2, 5, 7, 4);
    assert_eq!(typed_out.len(), 7 * 4);
    let from_typed: Vec<u8> = typed_out
        .iter()
        .flat_map(|p| [p.r, p.g, p.b, p.a])
        .collect();
    assert_eq!(bytes_out, from_typed);
}

#[test]
fn tile_then_offset_composes_into_a_shifted_periodic_field() {
    // §15.23 + §15.21 composition: tile a small "L" reference into a
    // wide rectangle, then offset the result by an integer shift.
    // For integer shifts the §15.21 `Nearest` policy is exact, so
    // the composite must agree pixel-for-pixel with a directly
    // tiled buffer whose tile origin has been moved by the same
    // shift.
    let src = build(2, 2, |x, y| {
        if (x, y) == (0, 0) {
            Rgba::opaque(200, 50, 50)
        } else if (x, y) == (1, 0) {
            Rgba::opaque(50, 200, 50)
        } else if (x, y) == (0, 1) {
            Rgba::opaque(50, 50, 200)
        } else {
            Rgba::opaque(100, 100, 100)
        }
    });
    let tiled = tile(&src, 2, 2, 0, 0, 8, 4);
    let shifted_by_offset = offset(&tiled, 8, 4, 1.0, -1.0, OffsetSampling::Nearest);
    // The §15.21 `Nearest` shift leaves an out-of-bounds region
    // along the destination's top row (sy = y + 1 = 0 + 1 falls
    // outside the tiled buffer) and one column on the left of the
    // first row, so we only assert the body of the result equals
    // the directly tile-shifted buffer.
    let direct_shift = tile(&src, 2, 2, 1, -1, 8, 4);
    // Compare rows 1..3, columns 1..8. (Row 0 is the §15.21 OOB
    // region; column 0 of row 0 is also OOB.)
    for y in 1..3 {
        for x in 1..8 {
            let p = (y * 8 + x) * 4;
            assert_eq!(
                &shifted_by_offset[p..p + 4],
                &direct_shift[p..p + 4],
                "px ({x}, {y})"
            );
        }
    }
}

#[test]
fn tile_preserves_alpha_channel_unchanged() {
    // A semi-transparent reference tile must tile out to a buffer
    // whose alpha matches the source's alpha at the resolved
    // source coordinate, untouched by the §15.23 placement rule.
    let src = build(2, 2, |x, _| {
        if x == 0 {
            Rgba::new(10, 20, 30, 128)
        } else {
            Rgba::new(40, 50, 60, 64)
        }
    });
    let out = tile(&src, 2, 2, 0, 0, 6, 3);
    for y in 0..3 {
        for x in 0..6 {
            let p = (y * 6 + x) * 4;
            let expected_a = if x % 2 == 0 { 128 } else { 64 };
            assert_eq!(out[p + 3], expected_a, "px ({x}, {y}) alpha");
        }
    }
}
