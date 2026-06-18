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
    let mut group = Group {
        opacity: 0.5,
        ..Default::default()
    };
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
    let mut inner = Group {
        opacity: 0.5,
        ..Default::default()
    };
    inner
        .children
        .push(rect_node(0.0, 0.0, 2.0, 2.0, Rgba::opaque(255, 0, 0)));

    let mut outer = Group {
        opacity: 0.5,
        ..Default::default()
    };
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

#[test]
fn group_opacity_does_not_double_darken_overlapping_children() {
    // SVG 2 §3.4 group opacity / "Simple Alpha Compositing": a group with
    // `opacity = 0.5` is rendered into an offscreen image at full opacity
    // and *then* blended onto the canvas uniformly. Two overlapping
    // opaque children inside that group must therefore produce the SAME
    // final pixel in their overlap as in the region only one child
    // covers — opacity is a post-process on the composited group, not a
    // per-child alpha. (The pre-fix direct path multiplied each child by
    // 0.5 then over-composited them, so the overlap accumulated to alpha
    // ~192 while the single-cover region stayed ~128 — a visible seam.)
    let mut group = Group {
        opacity: 0.5,
        ..Default::default()
    };
    // Two opaque red rects whose union spans (0,0)..(6,4) and whose
    // overlap is the central column band x in [2,4).
    group
        .children
        .push(rect_node(0.0, 0.0, 4.0, 4.0, Rgba::opaque(255, 0, 0)));
    group
        .children
        .push(rect_node(2.0, 0.0, 4.0, 4.0, Rgba::opaque(255, 0, 0)));

    let mut root = Group::default();
    root.children.push(Node::Group(group));

    // Transparent background so we read the composited group alpha
    // directly rather than blended into an opaque colour.
    let r = Renderer::new(6, 4);
    let v = VectorFrame {
        width: 6.0,
        height: 4.0,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    let px = |x: usize, y: usize| {
        let i = y * stride + x * 4;
        let d = &out.planes[0].data[i..i + 4];
        [d[0], d[1], d[2], d[3]]
    };
    // Single-cover sample (only the first rect): x = 1.
    let single = px(1, 2);
    // Overlap sample (both rects): x = 3.
    let overlap = px(3, 2);

    // Opaque red at group opacity 0.5 over transparent → alpha ~128,
    // red 255, in BOTH regions.
    assert!(
        (single[3] as i32 - 128).abs() <= 2,
        "single-cover alpha should be ~128, got {}",
        single[3]
    );
    assert!(
        (overlap[3] as i32 - 128).abs() <= 2,
        "overlap alpha should be ~128 (NOT double-darkened ~192), got {}",
        overlap[3]
    );
    // The whole union must be a uniform colour+alpha — no seam.
    assert_eq!(
        single, overlap,
        "overlap pixel {overlap:?} must equal single-cover pixel {single:?}"
    );
    // Red channel is opaque red in both.
    assert_eq!(overlap[0], 255, "overlap red channel");
    assert_eq!(overlap[1], 0, "overlap green channel");
    assert_eq!(overlap[2], 0, "overlap blue channel");
}
