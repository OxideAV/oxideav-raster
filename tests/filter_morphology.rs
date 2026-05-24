//! Integration coverage for [`oxideav_raster::morphology`] — the SVG
//! 1.1 §15.20 `<feMorphology>` erode / dilate primitive.
//!
//! The unit tests inside `src/filter.rs` cover the algorithmic
//! contracts (separability vs naive 2-D, duality, extensivity,
//! solid-image invariance, zero-radius identity, single-pixel
//! dilation footprint, panic on bad input). This file is the
//! consumer-facing API exercise — the same kind of "treat the public
//! re-exports as a black box and check the documented behaviour"
//! coverage the existing `tests/image_lanczos3.rs` etc. provide.

use oxideav_core::Rgba;
use oxideav_raster::{morphology, morphology_pixels, MorphologyOp};

/// Helper — build a packed-RGBA buffer of `w·h` pixels coloured by
/// the supplied closure.
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
fn public_api_zero_radius_identity() {
    let img = build(5, 4, |x, y| {
        Rgba::new((x * 13) as u8, (y * 37) as u8, 17, 200)
    });
    let dilated = morphology(&img, 5, 4, 0, 0, MorphologyOp::Dilate);
    let eroded = morphology(&img, 5, 4, 0, 0, MorphologyOp::Erode);
    assert_eq!(dilated, img, "0-radius dilate must be identity");
    assert_eq!(eroded, img, "0-radius erode must be identity");
}

#[test]
fn dilate_isolated_pixel_paints_full_axis_aligned_rectangle() {
    // SVG 1.1 §15.20 spec: "The dilation kernel is a rectangle with
    // a width of 2*x-radius and a height of 2*y-radius."  Our
    // discrete realisation is (2·rx + 1) × (2·ry + 1) inclusive of
    // the centre. Verify the dilated footprint of a single bright
    // pixel hits exactly that rectangle for a non-square radius.
    let w = 11u32;
    let h = 9u32;
    let bg = Rgba::new(0, 0, 0, 0);
    let fg = Rgba::new(255, 255, 255, 255);
    let mut img = build(w, h, |_, _| bg);
    let cx = 5usize;
    let cy = 4usize;
    let off = (cy * w as usize + cx) * 4;
    img[off..off + 4].copy_from_slice(&[fg.r, fg.g, fg.b, fg.a]);

    let rx = 3u32;
    let ry = 1u32;
    let out = morphology(&img, w, h, rx, ry, MorphologyOp::Dilate);

    // Expected: (2·3 + 1) × (2·1 + 1) = 7 × 3 = 21 bright pixels.
    let mut bright = 0usize;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = &out[(y * w as usize + x) * 4..(y * w as usize + x) * 4 + 4];
            if p == [255, 255, 255, 255] {
                bright += 1;
                let dx = (x as isize - cx as isize).unsigned_abs();
                let dy = (y as isize - cy as isize).unsigned_abs();
                assert!(
                    dx <= rx as usize && dy <= ry as usize,
                    "stray bright pixel at ({x},{y})"
                );
            }
        }
    }
    assert_eq!(
        bright, 21,
        "expected 7×3 = 21 bright pixels after non-square dilate"
    );
}

#[test]
fn erode_eats_through_thin_strokes_first() {
    // Build a 1-px-wide vertical line through the middle of an 11×11
    // canvas. erode(rx = 1, ry = 0) over a 1-px line should remove
    // it entirely (3-px horizontal min reaches the background on
    // both sides).
    let w = 11u32;
    let h = 11u32;
    let bg = Rgba::new(0, 0, 0, 0);
    let mut img = build(w, h, |_, _| bg);
    for y in 0..h as usize {
        let off = (y * w as usize + 5) * 4;
        img[off..off + 4].copy_from_slice(&[255, 255, 255, 255]);
    }

    let out = morphology(&img, w, h, 1, 0, MorphologyOp::Erode);
    for byte in &out {
        assert_eq!(*byte, 0, "1-px line must be eroded to transparent black");
    }
}

#[test]
fn dilate_then_erode_is_a_closing_idempotent_on_a_solid_block() {
    // Morphological closing = dilate then erode. On a solid block
    // that already covers its closed form, closing is a no-op for
    // any radius ≤ the block's interior. We use a 9×9 opaque red
    // block in an 11×11 canvas with rx = ry = 1 to verify.
    let w = 11u32;
    let h = 11u32;
    let bg = Rgba::new(0, 0, 0, 0);
    let img = build(w, h, |x, y| {
        if (1..10).contains(&x) && (1..10).contains(&y) {
            Rgba::new(200, 50, 50, 255)
        } else {
            bg
        }
    });

    let dilated = morphology(&img, w, h, 1, 1, MorphologyOp::Dilate);
    let closed = morphology(&dilated, w, h, 1, 1, MorphologyOp::Erode);

    // Interior 7×7 of the original block must be preserved
    // pixel-exact (the boundary effect is bounded by the radius +
    // clamp-to-edge handling at the canvas edge, which is the only
    // place closing can differ from the input on this fixture).
    for y in 2..9usize {
        for x in 2..9usize {
            let off = (y * w as usize + x) * 4;
            assert_eq!(
                &closed[off..off + 4],
                &[200, 50, 50, 255],
                "interior pixel ({x},{y}) drifted under closing"
            );
        }
    }
}

#[test]
fn erode_then_dilate_is_an_opening_removes_thin_features() {
    // Morphological opening = erode then dilate. Opening with a
    // 3×3 SE annihilates structures thinner than 3 pixels and
    // preserves the rest. We build a canvas with a thick 5×5 block
    // *and* a 1-px noise pixel detached from it; verify the noise
    // pixel is gone after opening, the block survives.
    let w = 13u32;
    let h = 11u32;
    let bg = Rgba::new(0, 0, 0, 0);
    let mut img = build(w, h, |x, y| {
        if (2..7).contains(&x) && (2..7).contains(&y) {
            Rgba::new(255, 255, 255, 255)
        } else {
            bg
        }
    });
    // Detached noise pixel.
    let off = (w as usize + 10) * 4;
    img[off..off + 4].copy_from_slice(&[255, 255, 255, 255]);

    let eroded = morphology(&img, w, h, 1, 1, MorphologyOp::Erode);
    let opened = morphology(&eroded, w, h, 1, 1, MorphologyOp::Dilate);

    // Noise pixel and its 3×3 neighbourhood must be background.
    for y in 0..3usize {
        for x in 9..12usize {
            if x >= w as usize || y >= h as usize {
                continue;
            }
            let off = (y * w as usize + x) * 4;
            assert_eq!(
                &opened[off..off + 4],
                &[0, 0, 0, 0],
                "noise neighbourhood ({x},{y}) survived opening"
            );
        }
    }
    // 3×3 interior of the original 5×5 block must still be bright
    // (centre of the block is untouched by erosion at radius 1).
    for y in 3..6usize {
        for x in 3..6usize {
            let off = (y * w as usize + x) * 4;
            assert_eq!(
                &opened[off..off + 4],
                &[255, 255, 255, 255],
                "block interior ({x},{y}) erased by opening"
            );
        }
    }
}

#[test]
fn axis_decoupling_horizontal_only_does_not_smear_vertically() {
    // rx > 0, ry = 0 ⇒ only the horizontal pass runs. A single bright
    // pixel must dilate into a horizontal line, not a square.
    let w = 7u32;
    let h = 5u32;
    let mut img = build(w, h, |_, _| Rgba::new(0, 0, 0, 0));
    let cx = 3usize;
    let cy = 2usize;
    let off = (cy * w as usize + cx) * 4;
    img[off..off + 4].copy_from_slice(&[100, 200, 50, 255]);

    let out = morphology(&img, w, h, 2, 0, MorphologyOp::Dilate);

    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = &out[(y * w as usize + x) * 4..(y * w as usize + x) * 4 + 4];
            let lit = p == [100, 200, 50, 255];
            let in_band = y == cy && (x as isize - cx as isize).unsigned_abs() <= 2;
            assert_eq!(lit, in_band, "({x},{y}) lit={lit} in_band={in_band}");
        }
    }
}

#[test]
fn pixels_wrapper_round_trips_through_typed_buffer() {
    let w = 6u32;
    let h = 5u32;
    let pixels: Vec<Rgba> = (0..(w * h) as u8)
        .map(|i| {
            Rgba::new(
                i.wrapping_mul(11),
                i.wrapping_mul(29).wrapping_add(7),
                i.wrapping_mul(53).wrapping_add(13),
                255,
            )
        })
        .collect();
    let bytes: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();

    for op in [MorphologyOp::Erode, MorphologyOp::Dilate] {
        for (rx, ry) in [(1u32, 1u32), (2, 0), (0, 2), (2, 1)] {
            let pix = morphology_pixels(&pixels, w, h, rx, ry, op);
            let byt = morphology(&bytes, w, h, rx, ry, op);
            let rebuilt: Vec<u8> = pix.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
            assert_eq!(byt, rebuilt, "wrapper mismatch op={op:?} rx={rx} ry={ry}");
        }
    }
}
