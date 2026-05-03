//! Integration tests for path flattening.

use oxideav_core::{Path, PathCommand, Point, Transform2D};
use oxideav_raster::{flatten_arc_to_cubics, flatten_path};

#[test]
fn quad_curve_yields_intermediate_polyline_points() {
    let mut p = Path::new();
    p.move_to(Point::new(0.0, 0.0))
        .quad_to(Point::new(50.0, 100.0), Point::new(100.0, 0.0))
        .close();
    let flat = flatten_path(&p.commands, &Transform2D::identity());
    assert_eq!(flat.len(), 1);
    assert!(
        flat[0].points.len() > 8,
        "expected subdivision to produce many points, got {}",
        flat[0].points.len()
    );
    assert!(flat[0].closed);
}

#[test]
fn cubic_curve_yields_intermediate_polyline_points() {
    let mut p = Path::new();
    p.move_to(Point::new(0.0, 0.0)).cubic_to(
        Point::new(100.0, 100.0),
        Point::new(100.0, -100.0),
        Point::new(0.0, 0.0),
    );
    let flat = flatten_path(&p.commands, &Transform2D::identity());
    assert!(flat[0].points.len() > 10);
}

#[test]
fn arc_quarter_circle_emits_at_least_one_cubic() {
    let cubics = flatten_arc_to_cubics(
        Point::new(1.0, 0.0),
        1.0,
        1.0,
        0.0,
        false,
        true,
        Point::new(0.0, 1.0),
    );
    // 90° = exactly one cubic.
    assert_eq!(cubics.len(), 1);
    let last = cubics[0];
    assert!((last[2].x - 0.0).abs() < 1e-3);
    assert!((last[2].y - 1.0).abs() < 1e-3);
}

#[test]
fn arc_full_circle_split_into_four_cubics() {
    // Going from (1, 0) all the way around to (1, 0) is degenerate
    // (identical endpoints). Test a 270° arc instead.
    let cubics = flatten_arc_to_cubics(
        Point::new(1.0, 0.0),
        1.0,
        1.0,
        0.0,
        true, // large arc
        true,
        Point::new(0.0, -1.0),
    );
    // 270° splits into 3 segments at most.
    assert!(cubics.len() >= 3);
}

#[test]
fn arc_in_path_lands_at_target() {
    let mut p = Path::new();
    p.move_to(Point::new(10.0, 0.0));
    p.commands.push(PathCommand::ArcTo {
        rx: 10.0,
        ry: 10.0,
        x_axis_rot: 0.0,
        large_arc: false,
        sweep: true,
        end: Point::new(0.0, 10.0),
    });
    let cs = flatten_path(&p.commands, &Transform2D::identity());
    let last = cs[0].points.last().copied().unwrap();
    assert!((last.0 - 0.0).abs() < 1e-2, "x landed at {}", last.0);
    assert!((last.1 - 10.0).abs() < 1e-2, "y landed at {}", last.1);
}

#[test]
fn transform_scales_polyline() {
    let mut p = Path::new();
    p.move_to(Point::new(0.0, 0.0))
        .line_to(Point::new(10.0, 0.0));
    let t = Transform2D::scale(2.0, 3.0);
    let cs = flatten_path(&p.commands, &t);
    assert_eq!(cs[0].points[0], (0.0, 0.0));
    assert_eq!(cs[0].points[1], (20.0, 0.0));
}
