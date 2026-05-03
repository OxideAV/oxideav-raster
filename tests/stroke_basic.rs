//! Integration tests for stroke geometry.

use oxideav_core::{LineCap, LineJoin, Paint, Path, Point, Rgba, Stroke, Transform2D};
use oxideav_raster::{flatten_path, stroke_to_fill_path};

fn basic_stroke(width: f32) -> Stroke {
    Stroke {
        width,
        paint: Paint::Solid(Rgba::opaque(0, 0, 0)),
        cap: LineCap::Butt,
        join: LineJoin::Miter,
        miter_limit: 4.0,
        dash: None,
    }
}

#[test]
fn horizontal_line_stroked_yields_rectangle_geometry() {
    let mut p = Path::new();
    p.move_to(Point::new(0.0, 5.0))
        .line_to(Point::new(10.0, 5.0));
    let cs = flatten_path(&p.commands, &Transform2D::identity());
    let s = basic_stroke(2.0);
    let g = stroke_to_fill_path(&cs[0], &s, 2.0);
    assert_eq!(g.len(), 1);
    assert!(g[0].closed);
    // Geometry should bound the line within ±1 pixel vertically.
    let ys: Vec<f32> = g[0].points.iter().map(|p| p.1).collect();
    let y_min = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!((y_min - 4.0).abs() < 1e-3, "y_min = {}", y_min);
    assert!((y_max - 6.0).abs() < 1e-3, "y_max = {}", y_max);
}

#[test]
fn closed_square_stroke_emits_outer_and_inner_loops() {
    let mut p = Path::new();
    p.move_to(Point::new(2.0, 2.0))
        .line_to(Point::new(8.0, 2.0))
        .line_to(Point::new(8.0, 8.0))
        .line_to(Point::new(2.0, 8.0))
        .close();
    let cs = flatten_path(&p.commands, &Transform2D::identity());
    let s = basic_stroke(2.0);
    let g = stroke_to_fill_path(&cs[0], &s, 2.0);
    assert_eq!(g.len(), 2);
}

#[test]
fn round_cap_adds_extra_vertices() {
    let mut p = Path::new();
    p.move_to(Point::new(0.0, 5.0))
        .line_to(Point::new(10.0, 5.0));
    let cs = flatten_path(&p.commands, &Transform2D::identity());
    let mut s = basic_stroke(4.0);
    s.cap = LineCap::Round;
    let g = stroke_to_fill_path(&cs[0], &s, 4.0);
    let butt = stroke_to_fill_path(&cs[0], &basic_stroke(4.0), 4.0);
    // Round caps emit ~7 polyline vertices each → ~14 more vertices
    // than the equivalent butt-cap rectangle.
    assert!(
        g[0].points.len() > butt[0].points.len(),
        "round cap should produce more polyline vertices than butt cap"
    );
}

#[test]
fn round_join_softens_corners() {
    let mut p = Path::new();
    p.move_to(Point::new(0.0, 0.0))
        .line_to(Point::new(10.0, 0.0))
        .line_to(Point::new(10.0, 10.0));
    let cs = flatten_path(&p.commands, &Transform2D::identity());
    let mut s = basic_stroke(2.0);
    s.join = LineJoin::Round;
    let g = stroke_to_fill_path(&cs[0], &s, 2.0);
    let mut bevel = basic_stroke(2.0);
    bevel.join = LineJoin::Bevel;
    let gb = stroke_to_fill_path(&cs[0], &bevel, 2.0);
    // Round join inserts an arc polyline at the corner; bevel only
    // emits the two offset endpoints.
    assert!(g[0].points.len() > gb[0].points.len());
}
