//! `GradientPaint` — gradients with their own `gradientUnits` /
//! `gradientTransform` (SVG 2 §14.2) painted through the direct
//! renderer endpoints.

use oxideav_core::{
    FillRule, GradientStop, Group, LinearGradient, Node, Paint, Path, PathNode, Point,
    RadialGradient, Rgba, Stroke, Transform2D, VectorFrame, VideoFrame,
};
use oxideav_raster::{GradientPaint, GradientUnits, Renderer, path_bounding_box};

const BLACK: Rgba = Rgba::opaque(0, 0, 0);
const WHITE: Rgba = Rgba::opaque(255, 255, 255);

fn rect(x: f32, y: f32, w: f32, h: f32) -> Path {
    let mut p = Path::new();
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
    p
}

fn bw_linear(x1: f32, y1: f32, x2: f32, y2: f32) -> LinearGradient {
    LinearGradient::new(Point::new(x1, y1), Point::new(x2, y2))
        .with_stop(GradientStop::new(0.0, BLACK))
        .with_stop(GradientStop::new(1.0, WHITE))
}

fn px(out: &VideoFrame, x: u32, y: u32) -> [u8; 4] {
    let p = &out.planes[0];
    let i = (y as usize) * p.stride + (x as usize) * 4;
    [p.data[i], p.data[i + 1], p.data[i + 2], p.data[i + 3]]
}

fn close(got: u8, want: f32, what: &str) {
    let w = want.round() as i32;
    assert!((got as i32 - w).abs() <= 2, "{what}: got {got}, want ≈{w}");
}

/// A user-space `GradientPaint` through the direct endpoint renders
/// byte-identically to the same gradient as a `Paint` inside a scene.
#[test]
fn user_space_gradient_paint_matches_scene_paint() {
    let g = bw_linear(0.0, 0.0, 40.0, 0.0);
    let ctm = Transform2D::translate(10.0, 5.0).compose(&Transform2D::rotate(0.4));
    let r = Renderer::new(64, 64);
    let direct = r.fill_path_with_gradient(
        &rect(0.0, 0.0, 40.0, 30.0),
        FillRule::NonZero,
        &GradientPaint::linear(g.clone()),
        ctm,
    );
    let scene = VectorFrame::new(64.0, 64.0).with_root(Group::new().with_child(Node::Group(
        Group::new().with_transform(ctm).with_child(Node::Path(
            PathNode::new(rect(0.0, 0.0, 40.0, 30.0)).with_fill(Paint::LinearGradient(g)),
        )),
    )));
    let via_scene = r.render(&scene);
    assert_eq!(direct.planes[0].data, via_scene.planes[0].data);
}

/// `objectBoundingBox` units: a 0 → 1 gradient spans exactly the
/// rectangle regardless of where / how big it is, and follows the CTM.
#[test]
fn bbox_units_span_the_shape() {
    let gp = GradientPaint::linear(bw_linear(0.0, 0.0, 1.0, 0.0))
        .with_units(GradientUnits::ObjectBoundingBox);
    let r = Renderer::new(64, 64);
    // Rect 10..50 in user space, CTM scale 1.
    let out = r.fill_path_with_gradient(
        &rect(10.0, 20.0, 40.0, 8.0),
        FillRule::NonZero,
        &gp,
        Transform2D::identity(),
    );
    close(px(&out, 10, 24)[0], 255.0 * 0.5 / 40.0, "left");
    close(px(&out, 30, 24)[0], 255.0 * 20.5 / 40.0, "middle");
    close(px(&out, 49, 24)[0], 255.0 * 39.5 / 40.0, "right");
    // Same rect under a CTM scale of 0.5: device 5..25.
    let out = r.fill_path_with_gradient(
        &rect(10.0, 20.0, 40.0, 8.0),
        FillRule::NonZero,
        &gp,
        Transform2D::scale(0.5, 0.5),
    );
    close(px(&out, 5, 12)[0], 255.0 * 1.0 / 40.0, "scaled left");
    close(px(&out, 24, 12)[0], 255.0 * 39.0 / 40.0, "scaled right");
    assert_eq!(px(&out, 30, 12)[3], 0);
}

/// §8.11: an `objectBoundingBox` gradient on a shape whose box has no
/// width or height paints nothing — here a zero-height rect (which has
/// no area anyway) and a horizontal stroked line (whose stroke *does*
/// have area, yet must not be painted).
#[test]
fn bbox_units_on_degenerate_box_paint_nothing() {
    let gp = GradientPaint::linear(bw_linear(0.0, 0.0, 1.0, 0.0))
        .with_units(GradientUnits::ObjectBoundingBox);
    let r = Renderer::new(32, 32);
    let mut line = Path::new();
    line.move_to(Point::new(4.0, 16.0))
        .line_to(Point::new(28.0, 16.0));
    let stroked = r.stroke_path_with_gradient(
        &line,
        &Stroke::solid(6.0, BLACK),
        &gp,
        Transform2D::identity(),
    );
    assert!(stroked.planes[0].data.iter().all(|&b| b == 0));
    // The same stroke in user-space units paints.
    let us = GradientPaint::linear(bw_linear(4.0, 0.0, 28.0, 0.0));
    let stroked = r.stroke_path_with_gradient(
        &line,
        &Stroke::solid(6.0, BLACK),
        &us,
        Transform2D::identity(),
    );
    assert_eq!(px(&stroked, 16, 16)[3], 255);
    close(px(&stroked, 16, 16)[0], 255.0 * 12.5 / 24.0, "stroke mid");
}

/// Stroke paint uses the *fill* geometry's bounding box (§8.11), not
/// the stroked outline's.
#[test]
fn stroke_bbox_is_the_fill_geometry_box() {
    let gp = GradientPaint::linear(bw_linear(0.0, 0.0, 1.0, 0.0))
        .with_units(GradientUnits::ObjectBoundingBox);
    let r = Renderer::new(64, 64);
    // Rect 16..48 stroked 8 wide: the outline spans 12..52 but the
    // gradient must run 16 → 48 (black at x=16, white at x=48), so the
    // outer stroke band left of 16 is padded black and right of 48 white.
    let out = r.stroke_path_with_gradient(
        &rect(16.0, 16.0, 32.0, 32.0),
        &Stroke::solid(8.0, BLACK),
        &gp,
        Transform2D::identity(),
    );
    assert_eq!(px(&out, 13, 32)[0], 0, "pad left of the box is black");
    assert_eq!(px(&out, 50, 32)[0], 255, "pad right of the box is white");
    close(px(&out, 32, 14)[0], 255.0 * 16.5 / 32.0, "top band middle");
}

/// `gradientTransform` composes after the bounding-box matrix: a
/// rotation by 90° in bbox space turns the horizontal 0 → 1 ramp into a
/// vertical one.
#[test]
fn gradient_transform_rotates_in_bbox_space() {
    let rot90 = Transform2D::translate(0.5, 0.5)
        .compose(&Transform2D::rotate(std::f32::consts::FRAC_PI_2))
        .compose(&Transform2D::translate(-0.5, -0.5));
    let gp = GradientPaint::linear(bw_linear(0.0, 0.0, 1.0, 0.0))
        .with_units(GradientUnits::ObjectBoundingBox)
        .with_transform(rot90);
    let out = Renderer::new(64, 64).fill_path_with_gradient(
        &rect(8.0, 8.0, 48.0, 32.0),
        FillRule::NonZero,
        &gp,
        Transform2D::identity(),
    );
    // Rows vary, columns don't: (x, y) → bbox ((x−8)/48, (y−8)/32) →
    // rotated 90° about the centre → t = (y − 8) / 32 … up to direction.
    let top = px(&out, 32, 8)[0] as i32;
    let bottom = px(&out, 32, 39)[0] as i32;
    assert!((top - bottom).abs() > 200, "ramp should run vertically");
    assert_eq!(px(&out, 10, 20)[0], px(&out, 53, 20)[0]);
}

/// Radial `objectBoundingBox` gradient on a non-square rect: rings are
/// ellipses with the box's aspect ratio.
#[test]
fn radial_bbox_units_render_elliptical_rings() {
    let rg = RadialGradient::new(Point::new(0.5, 0.5), 0.5)
        .with_stop(GradientStop::new(0.0, BLACK))
        .with_stop(GradientStop::new(1.0, WHITE));
    let gp = GradientPaint::radial(rg).with_units(GradientUnits::ObjectBoundingBox);
    let out = Renderer::new(64, 64).fill_path_with_gradient(
        &rect(0.0, 16.0, 64.0, 32.0),
        FillRule::NonZero,
        &gp,
        Transform2D::identity(),
    );
    // Centre (32, 32); the 0.5 iso-ring passes through device (48, 32)
    // and (32, 40). Pixel centres sit at bbox offsets (16.5/64, 0.5/32)
    // and (0.5/64, 8.5/32) from the centre: t = |offset| / 0.5.
    let t_east = ((16.5f32 / 64.0).powi(2) + (0.5f32 / 32.0).powi(2)).sqrt() / 0.5;
    let t_south = ((0.5f32 / 64.0).powi(2) + (8.5f32 / 32.0).powi(2)).sqrt() / 0.5;
    close(px(&out, 48, 32)[0], 255.0 * t_east, "east");
    close(px(&out, 32, 40)[0], 255.0 * t_south, "south");
    // A circular (non-elliptical) gradient would put (32, 40) at
    // t = 8.5 / 32 ≈ 0.27 instead — far from the elliptical value.
    assert!(px(&out, 32, 40)[0] > 120);
}

#[test]
fn path_bounding_box_follows_curves() {
    let mut p = Path::new();
    p.move_to(Point::new(0.0, 10.0))
        .quad_to(Point::new(10.0, -10.0), Point::new(20.0, 10.0))
        .close();
    let r = path_bounding_box(&p).unwrap();
    // Quadratic apex is at y = 0 (midpoint of the control polygon).
    assert!((r.y - 0.0).abs() < 0.05, "y = {}", r.y);
    assert!((r.height - 10.0).abs() < 0.05);
    assert_eq!((r.x, r.width), (0.0, 20.0));
}
