//! Integration coverage for [`oxideav_raster::tile`] — the SVG 1.1
//! §15.23 `<feTile>` filter primitive.
//!
//! The unit tests inside `src/filter.rs::tile_tests` cover the per-
//! pixel modular sampling algebra (identity, partial periods, 1×1
//! constant fill, target-smaller-than-source crop, panic guards).
//! This file is the consumer-facing API exercise — treating the
//! public re-exports as a black box and checking the documented
//! periodic-tiling behaviour at the larger extent sizes a real
//! `<filter>` element would target.

use oxideav_core::Rgba;
use oxideav_raster::{tile, tile_pixels};

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
fn tile_8x8_pattern_into_64x64_target_is_periodic() {
    // 8×8 gradient block — distinct value at every source pixel.
    let src = build(8, 8, |x, y| {
        Rgba::new((x * 32) as u8, (y * 32) as u8, ((x ^ y) * 32) as u8, 255)
    });
    let out = tile(&src, 8, 8, 64, 64);
    assert_eq!(out.len(), 64 * 64 * 4);

    // Every output pixel at (ox, oy) must equal the source pixel
    // at (ox % 8, oy % 8).
    for oy in 0..64u32 {
        for ox in 0..64u32 {
            let sx = (ox % 8) as usize;
            let sy = (oy % 8) as usize;
            let s = (sy * 8 + sx) * 4;
            let d = (oy as usize * 64 + ox as usize) * 4;
            assert_eq!(
                &out[d..d + 4],
                &src[s..s + 4],
                "periodicity broken at ({ox},{oy})"
            );
        }
    }
}

#[test]
fn tile_target_extent_not_a_multiple_of_source_is_still_periodic() {
    // 7×5 source into a 20×13 target — neither extent is a multiple
    // of the source. Every output column / row must still satisfy
    // the mod rule.
    let src = build(7, 5, |x, y| {
        Rgba::new((x * 36) as u8, (y * 51) as u8, ((x + y) * 25) as u8, 200)
    });
    let out = tile(&src, 7, 5, 20, 13);
    assert_eq!(out.len(), 20 * 13 * 4);
    for oy in 0..13u32 {
        for ox in 0..20u32 {
            let sx = (ox % 7) as usize;
            let sy = (oy % 5) as usize;
            let s = (sy * 7 + sx) * 4;
            let d = (oy as usize * 20 + ox as usize) * 4;
            assert_eq!(&out[d..d + 4], &src[s..s + 4]);
        }
    }
}

#[test]
fn tile_alpha_is_preserved_per_pixel() {
    // Diagonal alpha ramp on the source — copied through verbatim,
    // since §15.23 is a sampling operation and does not touch alpha.
    let src = build(4, 4, |x, y| Rgba::new(10, 20, 30, (x * 16 + y * 4) as u8));
    let out = tile(&src, 4, 4, 4, 4);
    assert_eq!(out, src);
}

#[test]
fn tile_one_dimensional_strip_extends_correctly() {
    // 1×4 vertical strip → 6×4 target. Every output column reads the
    // same one-column source.
    let src = build(1, 4, |_x, y| Rgba::new(0, 0, 0, (60 + y * 30) as u8));
    let out = tile(&src, 1, 4, 6, 4);
    assert_eq!(out.len(), 6 * 4 * 4);
    for y in 0..4u32 {
        let s = (y * 4) as usize;
        for x in 0..6u32 {
            let d = ((y * 6 + x) * 4) as usize;
            assert_eq!(&out[d..d + 4], &src[s..s + 4]);
        }
    }
}

#[test]
fn tile_typed_path_matches_byte_path_at_a_realistic_size() {
    let src_bytes = build(16, 11, |x, y| {
        Rgba::new((x * 15) as u8, (y * 22) as u8, ((x * y) % 256) as u8, 240)
    });
    let src_typed: Vec<Rgba> = src_bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let bytes_out = tile(&src_bytes, 16, 11, 100, 80);
    let typed_out = tile_pixels(&src_typed, 16, 11, 100, 80);
    let from_typed: Vec<u8> = typed_out
        .iter()
        .flat_map(|p| [p.r, p.g, p.b, p.a])
        .collect();
    assert_eq!(bytes_out, from_typed);
}

#[test]
fn tile_then_offset_round_trip_recovers_the_origin_seam_alignment() {
    // §15.23 places the tile's i=0, j=0 copy at the target origin.
    // If we offset the source by (sw, sh) first, the result is the
    // i=1, j=1 copy at the target origin — but since the tiling is
    // periodic with period (sw, sh), the visible output is unchanged.
    // This documents the "shift-then-tile is tile" invariant.
    let src = build(4, 4, |x, y| {
        Rgba::new((x * 60) as u8, (y * 60) as u8, 0, 255)
    });
    let direct = tile(&src, 4, 4, 12, 12);
    // Build a shifted source where every pixel is rotated by (1, 1):
    // (sx, sy) in the shifted source carries the original
    // ((sx + 3) % 4, (sy + 3) % 4) pixel. Tiling the shifted source
    // must produce a tiling that is a (1, 1) rotation of `direct`.
    let mut shifted = vec![0u8; 4 * 4 * 4];
    for sy in 0..4u32 {
        for sx in 0..4u32 {
            let nx = (sx + 3) % 4;
            let ny = (sy + 3) % 4;
            let s = ((ny * 4 + nx) * 4) as usize;
            let d = ((sy * 4 + sx) * 4) as usize;
            shifted[d..d + 4].copy_from_slice(&src[s..s + 4]);
        }
    }
    let from_shifted = tile(&shifted, 4, 4, 12, 12);
    // Every output pixel of from_shifted at (ox, oy) must equal
    // `direct` at ((ox + 11) % 12, (oy + 11) % 12) — a (1, 1)
    // rotation through the 12-period tiled grid.
    for oy in 0..12u32 {
        for ox in 0..12u32 {
            let rx = (ox + 11) % 12;
            let ry = (oy + 11) % 12;
            let a = ((oy * 12 + ox) * 4) as usize;
            let b = ((ry * 12 + rx) * 4) as usize;
            assert_eq!(&from_shifted[a..a + 4], &direct[b..b + 4]);
        }
    }
}
