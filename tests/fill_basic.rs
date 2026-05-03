//! Integration tests for the scanline AA fill.

use oxideav_core::{FillRule, Path, Point, Transform2D};
use oxideav_raster::{flatten_path, rasterize_fill};

fn rect_path(x: f32, y: f32, w: f32, h: f32) -> Path {
    let mut p = Path::new();
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
    p
}

#[test]
fn axis_aligned_rect_fills_interior_opaque() {
    let p = rect_path(2.0, 2.0, 4.0, 4.0);
    let cs = flatten_path(&p.commands, &Transform2D::identity());
    let m = rasterize_fill(&cs, 8, 8, FillRule::NonZero, 4);
    // 4×4 interior should all be 255.
    for y in 2..6 {
        for x in 2..6 {
            assert_eq!(m.get(x, y), 255, "pixel ({}, {})", x, y);
        }
    }
    // Corners outside should still be 0.
    assert_eq!(m.get(0, 0), 0);
    assert_eq!(m.get(7, 7), 0);
}

#[test]
fn supersampling_levels_all_produce_full_interior() {
    let p = rect_path(2.0, 2.0, 4.0, 4.0);
    let cs = flatten_path(&p.commands, &Transform2D::identity());
    for &ss in &[1u8, 2, 4, 8] {
        let m = rasterize_fill(&cs, 8, 8, FillRule::NonZero, ss);
        // Centre is fully covered at every supersample level.
        assert_eq!(m.get(3, 3), 255, "ss = {}", ss);
    }
}

#[test]
fn even_odd_donut_is_hollow() {
    let outer = rect_path(1.0, 1.0, 6.0, 6.0);
    let inner = rect_path(3.0, 3.0, 2.0, 2.0);
    let mut cs_outer = flatten_path(&outer.commands, &Transform2D::identity());
    let cs_inner = flatten_path(&inner.commands, &Transform2D::identity());
    cs_outer.extend(cs_inner);
    let m = rasterize_fill(&cs_outer, 8, 8, FillRule::EvenOdd, 4);
    // Inner-square centre cancels out.
    assert_eq!(m.get(3, 3), 0);
    assert_eq!(m.get(4, 4), 0);
    // Donut ring (just inside the outer, just outside the inner) is filled.
    assert_eq!(m.get(2, 2), 255);
}
