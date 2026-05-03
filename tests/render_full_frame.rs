//! End-to-end render of a small VectorFrame (3 rects + a circle-ish
//! path made from 4 cubic Beziers).

use oxideav_core::{
    FillRule, Group, Node, Paint, Path, PathCommand, PathNode, Point, Rgba, VectorFrame,
};
use oxideav_raster::Renderer;

fn rect_node(x: f32, y: f32, w: f32, h: f32, fill: Rgba) -> Node {
    let mut p = Path::new();
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
    Node::Path(PathNode {
        path: p,
        fill: Some(Paint::Solid(fill)),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
}

fn circle_node(cx: f32, cy: f32, r: f32, fill: Rgba) -> Node {
    // Approximate a circle from 4 cubic Bezier quadrants, using
    // SVG's standard kappa = 4/3 * (sqrt(2) - 1) ≈ 0.5522847.
    let k = 0.5522847f32 * r;
    let mut p = Path::new();
    p.move_to(Point::new(cx + r, cy));
    // Top quarter (right → top):
    p.cubic_to(
        Point::new(cx + r, cy - k),
        Point::new(cx + k, cy - r),
        Point::new(cx, cy - r),
    );
    // Top → left:
    p.cubic_to(
        Point::new(cx - k, cy - r),
        Point::new(cx - r, cy - k),
        Point::new(cx - r, cy),
    );
    // Left → bottom:
    p.cubic_to(
        Point::new(cx - r, cy + k),
        Point::new(cx - k, cy + r),
        Point::new(cx, cy + r),
    );
    // Bottom → right:
    p.cubic_to(
        Point::new(cx + k, cy + r),
        Point::new(cx + r, cy + k),
        Point::new(cx + r, cy),
    );
    p.commands.push(PathCommand::Close);
    Node::Path(PathNode {
        path: p,
        fill: Some(Paint::Solid(fill)),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
}

#[test]
fn render_3_rects_and_a_circle() {
    // A 32×32 canvas with three coloured rects and a green circle in
    // the centre.
    let mut root = Group::default();
    root.children
        .push(rect_node(2.0, 2.0, 8.0, 8.0, Rgba::opaque(255, 0, 0)));
    root.children
        .push(rect_node(12.0, 2.0, 8.0, 8.0, Rgba::opaque(0, 255, 0)));
    root.children
        .push(rect_node(22.0, 2.0, 8.0, 8.0, Rgba::opaque(0, 0, 255)));
    root.children
        .push(circle_node(16.0, 22.0, 6.0, Rgba::opaque(255, 255, 0)));

    let r = Renderer::new(32, 32);
    let v = VectorFrame {
        width: 32.0,
        height: 32.0,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    // Centre of the red rect: (6, 6) → red.
    let p = &out.planes[0].data[6 * stride + 6 * 4..][..4];
    assert_eq!(p, &[255, 0, 0, 255]);
    // Centre of the green rect: (16, 6) → green.
    let p = &out.planes[0].data[6 * stride + 16 * 4..][..4];
    assert_eq!(p, &[0, 255, 0, 255]);
    // Centre of the blue rect: (26, 6) → blue.
    let p = &out.planes[0].data[6 * stride + 26 * 4..][..4];
    assert_eq!(p, &[0, 0, 255, 255]);
    // Circle centre: (16, 22) → yellow.
    let p = &out.planes[0].data[22 * stride + 16 * 4..][..4];
    assert_eq!(p, &[255, 255, 0, 255]);
    // Outside the circle: top-left corner of the canvas should still
    // be transparent.
    let p = &out.planes[0].data[0..4];
    assert_eq!(p, &[0, 0, 0, 0]);
}

#[test]
fn rasterize_convenience_at_natural_size() {
    let mut root = Group::default();
    root.children
        .push(rect_node(0.0, 0.0, 4.0, 4.0, Rgba::opaque(128, 64, 32)));
    let v = VectorFrame {
        width: 4.0,
        height: 4.0,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    match oxideav_raster::rasterize(&v) {
        oxideav_core::Frame::Video(vf) => {
            assert_eq!(vf.planes[0].stride, 16);
            assert_eq!(vf.planes[0].data[0], 128);
            assert_eq!(vf.planes[0].data[1], 64);
            assert_eq!(vf.planes[0].data[2], 32);
            assert_eq!(vf.planes[0].data[3], 255);
        }
        _ => panic!("expected Frame::Video"),
    }
}
