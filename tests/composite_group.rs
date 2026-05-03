//! Group opacity composite test.

use oxideav_core::{FillRule, Group, Node, Paint, Path, PathNode, Point, Rgba, VectorFrame};
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

#[test]
fn group_opacity_half_blends_with_background() {
    // Background: solid blue (renderer's background fill).
    // Foreground: a red rect inside a group with opacity 0.5.
    let mut group = Group::default();
    group.opacity = 0.5;
    group
        .children
        .push(rect_node(0.0, 0.0, 4.0, 4.0, Rgba::opaque(255, 0, 0)));

    let mut root = Group::default();
    root.children.push(Node::Group(group));

    let mut r = Renderer::new(4, 4);
    r.background = Rgba::opaque(0, 0, 255);
    let v = VectorFrame {
        width: 4.0,
        height: 4.0,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    let i = stride + 4; // pixel (1, 1)
    let p = &out.planes[0].data[i..i + 4];
    // Half-opacity red over solid blue:
    //   src_premul = (128, 0, 0, 128)
    //   over (0, 0, 255, 255) → (128 + 0 * 127/255, 0, 127 + 255*127/255, 255)
    //   ~= (128, 0, 127, 255) before un-premul → after un-premul roughly the same.
    assert!(
        (p[0] as i32 - 128).abs() < 5,
        "red channel ~128, got {}",
        p[0]
    );
    assert_eq!(p[1], 0);
    assert!(
        (p[2] as i32 - 127).abs() < 5,
        "blue channel ~127, got {}",
        p[2]
    );
    assert_eq!(p[3], 255);
}

#[test]
fn nested_group_opacities_multiply() {
    // outer.opacity = 0.5, inner.opacity = 0.5 → effective alpha 0.25
    // for the red rect over a transparent background.
    let mut inner = Group::default();
    inner.opacity = 0.5;
    inner
        .children
        .push(rect_node(0.0, 0.0, 2.0, 2.0, Rgba::opaque(255, 0, 0)));

    let mut outer = Group::default();
    outer.opacity = 0.5;
    outer.children.push(Node::Group(inner));

    let mut root = Group::default();
    root.children.push(Node::Group(outer));

    let r = Renderer::new(2, 2);
    let v = VectorFrame {
        width: 2.0,
        height: 2.0,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let out = r.render(&v);
    let p = &out.planes[0].data[0..4];
    // 0.25 alpha over transparent → final alpha ~64.
    assert!(
        (p[3] as i32 - 64).abs() < 5,
        "expected alpha ~64, got {}",
        p[3]
    );
}
