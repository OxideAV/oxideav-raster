//! Gradient paints follow the painted node's current transformation
//! matrix (CTM).
//!
//! `oxideav-core` defines [`LinearGradient`] / [`RadialGradient`]
//! geometry in the painted element's *user space* — SVG 2 §14.2.2.1 /
//! §14.2.3.1 `gradientUnits="userSpaceOnUse"`: "the user coordinate
//! system for the element referencing the gradient element via a
//! `fill` or `stroke` property". The renderer must therefore map every
//! device pixel back through the inverse CTM before evaluating the
//! gradient, so the gradient geometry picks up the `viewBox` → viewport
//! scale, every ancestor group `transform` and the element's own
//! transform exactly like the path outline does.
//!
//! Each test builds the scene twice where that helps (with / without
//! the transform) and checks pixel values against the closed-form
//! `t` the mapping predicts. Tolerance: ±2 levels on 8-bit channels
//! (the 256-entry stops LUT interpolates between entries, ±1; the
//! pixel-centre `t` rounds to a byte, ±1).

use oxideav_core::{
    FillRule, GradientStop, Group, LinearGradient, Node, Paint, Path, PathNode, Point,
    RadialGradient, Rgba, SpreadMethod, Stroke, Transform2D, VectorFrame, VideoFrame, ViewBox,
};
use oxideav_raster::Renderer;

fn rect_path(x: f32, y: f32, w: f32, h: f32) -> Path {
    let mut p = Path::new();
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
    p
}

fn black_to_white(start: Point, end: Point) -> Paint {
    Paint::LinearGradient(LinearGradient {
        start,
        end,
        stops: vec![
            GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
        ],
        spread: SpreadMethod::Pad,
    })
}

fn red_to_blue_radial(center: Point, radius: f32) -> Paint {
    Paint::RadialGradient(RadialGradient {
        center,
        radius,
        focal: None,
        stops: vec![
            GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
        ],
        spread: SpreadMethod::Pad,
    })
}

fn filled(path: Path, paint: Paint) -> Node {
    Node::Path(PathNode {
        path,
        fill: Some(paint),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
}

fn frame(w: f32, h: f32, root: Group) -> VectorFrame {
    VectorFrame::new(w, h).with_root(root)
}

fn px(out: &VideoFrame, x: u32, y: u32) -> [u8; 4] {
    let plane = &out.planes[0];
    let i = (y as usize) * plane.stride + (x as usize) * 4;
    [
        plane.data[i],
        plane.data[i + 1],
        plane.data[i + 2],
        plane.data[i + 3],
    ]
}

fn assert_close(got: u8, want: f32, what: &str) {
    let want_b = want.round().clamp(0.0, 255.0) as i32;
    assert!(
        (got as i32 - want_b).abs() <= 2,
        "{what}: got {got}, expected ≈{want_b}"
    );
}

/// `viewBox="0 0 100 100"` rendered into a 50×50 canvas: user space is
/// scaled by 0.5. A black→white gradient along user `x ∈ [0, 100]`
/// must therefore sweep the full canvas width — device column `x`
/// samples user `2x + 1` at its centre, `t = (2x + 1) / 100`.
#[test]
fn view_box_scale_maps_gradient_with_the_shape() {
    let root = Group::new().with_child(filled(
        rect_path(0.0, 0.0, 100.0, 100.0),
        black_to_white(Point::new(0.0, 0.0), Point::new(100.0, 0.0)),
    ));
    let v = frame(50.0, 50.0, root).with_view_box(ViewBox::new(0.0, 0.0, 100.0, 100.0));
    let out = Renderer::new(50, 50).render(&v);
    for x in 0..50u32 {
        let t = (2.0 * x as f32 + 1.0) / 100.0;
        let p = px(&out, x, 25);
        assert_close(p[0], 255.0 * t, &format!("column {x}"));
        assert_eq!(p[0], p[1]);
        assert_eq!(p[1], p[2]);
        assert_eq!(p[3], 255);
    }
}

/// A translated group: the gradient must move with the rectangle it
/// paints. Rect `[0, 40]` under `translate(20, 0)` lands on device
/// `[20, 60)`; its gradient start must sit at device x = 20, not at
/// device x = 0.
#[test]
fn group_translate_moves_gradient_with_the_shape() {
    let root = Group::new().with_child(Node::Group(
        Group::new()
            .with_transform(Transform2D::translate(20.0, 0.0))
            .with_child(filled(
                rect_path(0.0, 0.0, 40.0, 40.0),
                black_to_white(Point::new(0.0, 0.0), Point::new(40.0, 0.0)),
            )),
    ));
    let out = Renderer::new(64, 64).render(&frame(64.0, 64.0, root));
    // Left column of the rect: user x = 0.5 → t = 0.0125.
    assert_close(px(&out, 20, 20)[0], 255.0 * 0.5 / 40.0, "left edge");
    // Middle: device 40 → user 20.5 → t = 0.5125.
    assert_close(px(&out, 40, 20)[0], 255.0 * 20.5 / 40.0, "middle");
    // Right column: device 59 → user 39.5 → t = 0.9875.
    assert_close(px(&out, 59, 20)[0], 255.0 * 39.5 / 40.0, "right edge");
    // Outside the rect nothing is painted.
    assert_eq!(px(&out, 10, 20)[3], 0);
    assert_eq!(px(&out, 62, 20)[3], 0);
}

/// Without any transform the same scene paints the gradient at the
/// same user coordinates — a regression guard that the inverse-CTM
/// mapping is the identity when the CTM is.
#[test]
fn identity_ctm_is_unchanged() {
    let root = Group::new().with_child(filled(
        rect_path(0.0, 0.0, 40.0, 40.0),
        black_to_white(Point::new(0.0, 0.0), Point::new(40.0, 0.0)),
    ));
    let out = Renderer::new(64, 64).render(&frame(64.0, 64.0, root));
    for x in 0..40u32 {
        assert_close(
            px(&out, x, 10)[0],
            255.0 * (x as f32 + 0.5) / 40.0,
            &format!("column {x}"),
        );
    }
}

/// A 90° rotation turns a user-x gradient into a device-y gradient.
/// `translate(64, 0) ∘ rotate(90°)` maps user `(x, y)` → device
/// `(64 − y, x)`: the gradient along user x now runs down the canvas.
#[test]
fn rotation_turns_the_gradient_axis() {
    let t = Transform2D::translate(64.0, 0.0)
        .compose(&Transform2D::rotate(std::f32::consts::FRAC_PI_2));
    let root = Group::new().with_child(Node::Group(Group::new().with_transform(t).with_child(
        filled(
            rect_path(0.0, 0.0, 64.0, 64.0),
            black_to_white(Point::new(0.0, 0.0), Point::new(64.0, 0.0)),
        ),
    )));
    let out = Renderer::new(64, 64).render(&frame(64.0, 64.0, root));
    for y in 0..64u32 {
        let want = 255.0 * (y as f32 + 0.5) / 64.0;
        // Every column in the row carries the same value.
        for x in [1u32, 20, 40, 62] {
            assert_close(px(&out, x, y)[0], want, &format!("({x}, {y})"));
        }
    }
}

/// A radial gradient under a non-uniform scale renders elliptical
/// rings: SVG 2 §14.2.3.1 — "the rings … will render as elliptical due
/// to application of the non-uniform scaling transformation". Under
/// `scale(2, 1)` a user circle of radius 16 becomes a 32 × 16 ellipse,
/// so the `t = 0.5` iso-contour passes through device `(cx + 16, cy)`
/// and `(cx, cy + 8)`.
#[test]
fn radial_under_non_uniform_scale_is_elliptical() {
    let root = Group::new().with_child(Node::Group(
        Group::new()
            .with_transform(Transform2D::scale(2.0, 1.0))
            .with_child(filled(
                rect_path(0.0, 0.0, 32.0, 64.0),
                red_to_blue_radial(Point::new(16.0, 32.0), 16.0),
            )),
    ));
    let out = Renderer::new(64, 64).render(&frame(64.0, 64.0, root));
    // Device centre (32, 32) ↔ user (16, 32): t ≈ 0 → red.
    let c = px(&out, 32, 32);
    assert!(c[0] >= 240 && c[2] <= 15, "centre should be red, got {c:?}");
    // Device (47, 32) → user (23.75, 32): t = 7.75 / 16 = 0.484.
    // Device (32, 39) → user (16, 39.5): t = 7.5 / 16 = 0.469.
    let east = px(&out, 47, 32);
    let south = px(&out, 32, 39);
    assert_close(east[2], 255.0 * 7.75 / 16.0, "east blue");
    assert_close(south[2], 255.0 * 7.5 / 16.0, "south blue");
    // Device (32, 47) → user (16, 47.5): t = 15.5 / 16 → almost pure
    // blue — the ring that is at t ≈ 0.48 eastwards is at t ≈ 0.97
    // southwards, which is exactly the elliptical squash.
    let far_south = px(&out, 32, 47);
    assert_close(far_south[2], 255.0 * 15.5 / 16.0, "far-south blue");
}

/// Stroke paint goes through the same mapping as fill paint.
#[test]
fn stroke_gradient_follows_the_ctm() {
    let mut line = Path::new();
    line.move_to(Point::new(0.0, 8.0))
        .line_to(Point::new(40.0, 8.0));
    let stroke = Stroke::new(
        4.0,
        black_to_white(Point::new(0.0, 0.0), Point::new(40.0, 0.0)),
    );
    let node = Node::Path(PathNode::new(line).with_stroke(stroke));
    let root = Group::new().with_child(Node::Group(
        Group::new()
            .with_transform(Transform2D::translate(20.0, 10.0))
            .with_child(node),
    ));
    let out = Renderer::new(64, 64).render(&frame(64.0, 64.0, root));
    // The stroke covers device y ∈ [16, 20), x ∈ [20, 60).
    assert_close(px(&out, 20, 18)[0], 255.0 * 0.5 / 40.0, "stroke start");
    assert_close(px(&out, 40, 18)[0], 255.0 * 20.5 / 40.0, "stroke middle");
    assert_close(px(&out, 59, 18)[0], 255.0 * 39.5 / 40.0, "stroke end");
    assert_eq!(px(&out, 40, 18)[3], 255);
}

/// Group opacity renders through an isolated offscreen buffer and a
/// cached group through the bitmap cache — both must carry the CTM
/// into the gradient sampler. The translucent result is the opaque
/// result at half alpha, and the cached second render is byte-identical
/// to the uncached one.
#[test]
fn offscreen_and_cached_groups_keep_the_ctm() {
    let inner = |opacity: f32, cache_key: Option<u64>| {
        let mut g = Group::new()
            .with_transform(Transform2D::translate(20.0, 0.0))
            .with_opacity(opacity)
            .with_child(filled(
                rect_path(0.0, 0.0, 40.0, 40.0),
                black_to_white(Point::new(0.0, 0.0), Point::new(40.0, 0.0)),
            ));
        if let Some(k) = cache_key {
            g = g.with_cache_key(k);
        }
        Group::new().with_child(Node::Group(g))
    };
    let r = Renderer::new(64, 64);
    let opaque = r.render(&frame(64.0, 64.0, inner(1.0, None)));
    let half = r.render(&frame(64.0, 64.0, inner(0.5, None)));
    for x in [20u32, 40, 59] {
        let o = px(&opaque, x, 20);
        let h = px(&half, x, 20);
        // Straight-alpha colour is preserved; alpha halves.
        assert!(
            (o[0] as i32 - h[0] as i32).abs() <= 2,
            "x={x}: {o:?} vs {h:?}"
        );
        assert!((h[3] as i32 - 128).abs() <= 1, "x={x}: alpha {}", h[3]);
    }
    let cached_scene = frame(64.0, 64.0, inner(1.0, Some(0xC0FFEE)));
    let first = r.render(&cached_scene);
    let second = r.render(&cached_scene);
    assert!(
        r.cache_stats().hits >= 1,
        "second render should hit the cache"
    );
    assert_eq!(first.planes[0].data, opaque.planes[0].data);
    assert_eq!(second.planes[0].data, opaque.planes[0].data);
}

/// `viewBox` scale and a nested group transform compose: with
/// `viewBox="0 0 200 200"` into 100×100 (scale 0.5) and a group
/// `translate(40, 0)`, a user gradient over `x ∈ [0, 120]` on a rect
/// at user `[0, 120]` paints device `[20, 80)`.
#[test]
fn view_box_and_group_transform_compose() {
    let root = Group::new().with_child(Node::Group(
        Group::new()
            .with_transform(Transform2D::translate(40.0, 0.0))
            .with_child(filled(
                rect_path(0.0, 0.0, 120.0, 200.0),
                black_to_white(Point::new(0.0, 0.0), Point::new(120.0, 0.0)),
            )),
    ));
    let v = frame(100.0, 100.0, root).with_view_box(ViewBox::new(0.0, 0.0, 200.0, 200.0));
    let out = Renderer::new(100, 100).render(&v);
    assert_eq!(px(&out, 10, 50)[3], 0, "left of the rect is empty");
    assert_eq!(px(&out, 90, 50)[3], 0, "right of the rect is empty");
    for x in [20u32, 35, 50, 65, 79] {
        // device x → user (2x + 1) − 40 → t = that / 120.
        let user_x = (2.0 * x as f32 + 1.0) - 40.0;
        assert_close(
            px(&out, x, 50)[0],
            255.0 * user_x / 120.0,
            &format!("column {x}"),
        );
    }
}
