//! Black-box pixel comparison against `rsvg-convert` (librsvg / cairo).
//!
//! `tests/fixtures/svg/*.svg` are small hand-written scenes; the
//! matching `tests/fixtures/ref/<name>_64x64.rgba` files are their
//! 64×64 rasters produced by `rsvg-convert` and dumped to raw
//! straight-alpha RGBA bytes with ImageMagick (`tests/fixtures/gen.sh`).
//! Each test rebuilds the same scene as a [`VectorFrame`] by hand (this
//! crate has no SVG parser) and compares the renderer's output.
//!
//! What is compared, and the tolerances (8-bit levels), as *measured*
//! against librsvg 2.62 / cairo 1.18 on these fixtures:
//!
//! * **interior pixels** (fully covered in both rasters): straight
//!   colour must agree within `INTERIOR_TOL` = 3. This is the gradient
//!   geometry check of issue #6 — the radial-under-non-uniform-scale
//!   fixture is byte-identical, the linear ones differ by ≤ 1.
//! * **edge pixels** (partially covered in either raster): alpha within
//!   `EDGE_TOL` = 16 per pixel and `EDGE_MEAN_TOL` = 7 on average, and
//!   premultiplied colour (`c · a / 255`) within `EDGE_TOL`. Straight
//!   colour is not compared at edges: a one-level alpha difference at a
//!   nearly transparent pixel would unpremultiply into a large, invisible
//!   colour difference.
//!
//! Why 16 and not "a level or two": the exact-area kernel reproduces the
//! closed-form pixel area to ±1 (see the unit tests in `fill.rs`), but
//! the reference raster is *not* exact area — along a near-horizontal
//! edge rsvg's coverage climbs in steps of 8.5 levels (≈ 1/30) with a
//! lateral offset of up to a pixel, a black-box observation on
//! `aa_near_horizontal`. Its deviation from the true area is therefore
//! up to ~12 levels on single edges and ~15 on thin strokes, and that is
//! what these numbers bound. The 4× supersampled kernel deviates by up
//! to 39 levels on the same fixture.
//!
//! Gradient fixtures exercise issue #6 (gradient geometry under the
//! node CTM); `aa_*` fixtures exercise the exact-area coverage kernel.

use oxideav_core::{
    FillRule, GradientStop, Group, LinearGradient, Node, Paint, Path, PathNode, Point,
    RadialGradient, Rgba, SpreadMethod, Stroke, Transform2D, VectorFrame, VideoFrame, ViewBox,
};
use oxideav_raster::{CoverageMode, GradientPaint, GradientUnits, Renderer};

const W: u32 = 64;
const H: u32 = 64;

/// Per-pixel cap on straight colour at fully covered pixels.
const INTERIOR_TOL: i32 = 3;
/// Per-pixel cap on |Δalpha| and |Δ premultiplied colour| at edge pixels.
const EDGE_TOL: i32 = 16;
/// Cap on the mean |Δalpha| over the edge pixels.
const EDGE_MEAN_TOL: f64 = 7.0;

fn load_ref(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/ref/{name}_{W}x{H}.rgba",
        env!("CARGO_MANIFEST_DIR")
    );
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(data.len(), (W * H * 4) as usize, "{path}: unexpected size");
    data
}

fn frame(root: Group) -> VectorFrame {
    VectorFrame::new(W as f32, H as f32).with_root(root)
}

fn render(v: &VectorFrame) -> Vec<u8> {
    let out: VideoFrame = Renderer::new(W, H).render(v);
    let plane = &out.planes[0];
    let mut flat = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H as usize {
        flat.extend_from_slice(&plane.data[y * plane.stride..y * plane.stride + (W as usize) * 4]);
    }
    flat
}

struct Diff {
    max_alpha: i32,
    max_edge_premul: i32,
    max_interior: i32,
    mean_edge_alpha: f64,
    edge_pixels: usize,
    edge_over: usize,
    interior_over: usize,
}

fn diff(got: &[u8], want: &[u8]) -> Diff {
    let mut d = Diff {
        max_alpha: 0,
        max_edge_premul: 0,
        max_interior: 0,
        mean_edge_alpha: 0.0,
        edge_pixels: 0,
        edge_over: 0,
        interior_over: 0,
    };
    let mut edge_sum = 0i64;
    for (g, w) in got.as_chunks::<4>().0.iter().zip(want.as_chunks::<4>().0) {
        let da = (g[3] as i32 - w[3] as i32).abs();
        d.max_alpha = d.max_alpha.max(da);
        if g[3] == 255 && w[3] == 255 {
            let worst = (0..3)
                .map(|c| (g[c] as i32 - w[c] as i32).abs())
                .max()
                .unwrap();
            d.max_interior = d.max_interior.max(worst);
            if worst > INTERIOR_TOL {
                d.interior_over += 1;
            }
        } else if g[3] != 0 || w[3] != 0 {
            d.edge_pixels += 1;
            edge_sum += da as i64;
            let mut worst = da;
            for c in 0..3 {
                let pg = (g[c] as i32 * g[3] as i32 + 127) / 255;
                let pw = (w[c] as i32 * w[3] as i32 + 127) / 255;
                let dp = (pg - pw).abs();
                d.max_edge_premul = d.max_edge_premul.max(dp);
                worst = worst.max(dp);
            }
            if worst > EDGE_TOL {
                d.edge_over += 1;
            }
        }
    }
    if d.edge_pixels > 0 {
        d.mean_edge_alpha = edge_sum as f64 / d.edge_pixels as f64;
    }
    d
}

fn check(name: &str, v: &VectorFrame) {
    let got = render(v);
    let want = load_ref(name);
    let d = diff(&got, &want);
    eprintln!(
        "{name}: interior max|Δc| = {}; edge max|Δα| = {}, max|Δpremul| = {}, mean|Δα| = {:.3} over {} px",
        d.max_interior, d.max_alpha, d.max_edge_premul, d.mean_edge_alpha, d.edge_pixels
    );
    assert!(
        d.interior_over == 0,
        "{name}: {} interior pixel(s) differ by more than {INTERIOR_TOL} (max {})",
        d.interior_over,
        d.max_interior
    );
    assert!(
        d.edge_over == 0,
        "{name}: {} edge pixel(s) deviate by more than {EDGE_TOL} (max |Δα| {}, max |Δpremul| {})",
        d.edge_over,
        d.max_alpha,
        d.max_edge_premul
    );
    assert!(
        d.mean_edge_alpha <= EDGE_MEAN_TOL,
        "{name}: mean edge alpha deviation {:.3} > {EDGE_MEAN_TOL}",
        d.mean_edge_alpha
    );
}

fn black() -> Paint {
    Paint::Solid(Rgba::opaque(0, 0, 0))
}

fn polygon(points: &[(f32, f32)]) -> Path {
    let mut p = Path::new();
    p.move_to(Point::new(points[0].0, points[0].1));
    for &(x, y) in &points[1..] {
        p.line_to(Point::new(x, y));
    }
    p.close();
    p
}

fn rect(x: f32, y: f32, w: f32, h: f32) -> Path {
    polygon(&[(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
}

/// Circle as four cubic Béziers (`κ = 4(√2 − 1)/3`).
fn circle(cx: f32, cy: f32, r: f32) -> Path {
    let k = 0.552_284_8 * r;
    let mut p = Path::new();
    p.move_to(Point::new(cx + r, cy))
        .cubic_to(
            Point::new(cx + r, cy + k),
            Point::new(cx + k, cy + r),
            Point::new(cx, cy + r),
        )
        .cubic_to(
            Point::new(cx - k, cy + r),
            Point::new(cx - r, cy + k),
            Point::new(cx - r, cy),
        )
        .cubic_to(
            Point::new(cx - r, cy - k),
            Point::new(cx - k, cy - r),
            Point::new(cx, cy - r),
        )
        .cubic_to(
            Point::new(cx + k, cy - r),
            Point::new(cx + r, cy - k),
            Point::new(cx + r, cy),
        )
        .close();
    p
}

fn fill(path: Path, paint: Paint, rule: FillRule) -> Node {
    Node::Path(PathNode::new(path).with_fill(paint).with_fill_rule(rule))
}

fn rotate_about(deg: f32, cx: f32, cy: f32) -> Transform2D {
    Transform2D::translate(cx, cy)
        .compose(&Transform2D::rotate(deg.to_radians()))
        .compose(&Transform2D::translate(-cx, -cy))
}

#[test]
fn aa_circle() {
    let v =
        frame(Group::new().with_child(fill(circle(32.3, 31.7, 20.0), black(), FillRule::NonZero)));
    check("aa_circle", &v);
}

#[test]
fn aa_rotated_square() {
    let v = frame(
        Group::new().with_child(Node::Group(
            Group::new()
                .with_transform(rotate_about(30.0, 32.0, 32.0))
                .with_child(fill(
                    rect(16.0, 16.0, 32.0, 32.0),
                    black(),
                    FillRule::NonZero,
                )),
        )),
    );
    check("aa_rotated_square", &v);
}

#[test]
fn aa_near_horizontal() {
    let v = frame(Group::new().with_child(fill(
        polygon(&[(2.0, 20.0), (62.0, 21.5), (62.0, 44.3), (2.0, 42.0)]),
        black(),
        FillRule::NonZero,
    )));
    check("aa_near_horizontal", &v);
}

#[test]
fn aa_thin_lines() {
    let line = |x1: f32, y1: f32, x2: f32, y2: f32, w: f32| {
        let mut p = Path::new();
        p.move_to(Point::new(x1, y1)).line_to(Point::new(x2, y2));
        Node::Path(PathNode::new(p).with_stroke(Stroke::solid(w, Rgba::opaque(0, 0, 0))))
    };
    let v = frame(
        Group::new()
            .with_child(line(4.0, 8.0, 60.0, 20.0, 0.7))
            .with_child(line(4.0, 24.0, 60.0, 52.0, 1.3))
            .with_child(line(6.0, 60.0, 58.0, 30.0, 2.2)),
    );
    check("aa_thin_lines", &v);
}

const STAR: [(f32, f32); 5] = [
    (32.0, 4.0),
    (43.5, 54.7),
    (8.4, 22.1),
    (55.6, 22.1),
    (20.5, 54.7),
];

#[test]
fn aa_star_evenodd() {
    let v = frame(Group::new().with_child(fill(polygon(&STAR), black(), FillRule::EvenOdd)));
    check("aa_star_evenodd", &v);
}

#[test]
fn aa_star_nonzero() {
    let v = frame(Group::new().with_child(fill(polygon(&STAR), black(), FillRule::NonZero)));
    check("aa_star_nonzero", &v);
}

fn linear(x1: f32, y1: f32, x2: f32, y2: f32, a: Rgba, b: Rgba) -> Paint {
    Paint::LinearGradient(LinearGradient {
        start: Point::new(x1, y1),
        end: Point::new(x2, y2),
        stops: vec![GradientStop::new(0.0, a), GradientStop::new(1.0, b)],
        spread: SpreadMethod::Pad,
    })
}

const RED: Rgba = Rgba::opaque(255, 0, 0);
const BLUE: Rgba = Rgba::opaque(0, 0, 255);
const BLACK: Rgba = Rgba::opaque(0, 0, 0);
const WHITE: Rgba = Rgba::opaque(255, 255, 255);

#[test]
fn grad_linear_viewbox() {
    let v = frame(Group::new().with_child(fill(
        rect(10.0, 10.0, 80.0, 80.0),
        linear(20.0, 0.0, 80.0, 0.0, RED, BLUE),
        FillRule::NonZero,
    )))
    .with_view_box(ViewBox::new(0.0, 0.0, 100.0, 100.0));
    check("grad_linear_viewbox", &v);
}

#[test]
fn grad_linear_group_transform() {
    // transform="translate(12 6) rotate(20) scale(1.1 1)"
    let t = Transform2D::translate(12.0, 6.0)
        .compose(&Transform2D::rotate(20f32.to_radians()))
        .compose(&Transform2D::scale(1.1, 1.0));
    let v = frame(
        Group::new().with_child(Node::Group(Group::new().with_transform(t).with_child(
            fill(
                rect(0.0, 0.0, 40.0, 30.0),
                linear(0.0, 0.0, 40.0, 0.0, BLACK, WHITE),
                FillRule::NonZero,
            ),
        ))),
    );
    check("grad_linear_group_transform", &v);
}

#[test]
fn grad_radial_nonuniform() {
    let radial = Paint::RadialGradient(RadialGradient {
        center: Point::new(16.0, 32.0),
        radius: 16.0,
        focal: None,
        stops: vec![GradientStop::new(0.0, RED), GradientStop::new(1.0, BLUE)],
        spread: SpreadMethod::Pad,
    });
    let v = frame(
        Group::new().with_child(Node::Group(
            Group::new()
                .with_transform(Transform2D::scale(2.0, 1.0))
                .with_child(fill(rect(0.0, 0.0, 32.0, 64.0), radial, FillRule::NonZero)),
        )),
    );
    check("grad_radial_nonuniform", &v);
}

/// Rebuild the harness render through a direct gradient endpoint.
fn render_gradient_fill(path: Path, gradient: &GradientPaint) -> Vec<u8> {
    let out = Renderer::new(W, H).fill_path_with_gradient(
        &path,
        FillRule::NonZero,
        gradient,
        Transform2D::identity(),
    );
    let plane = &out.planes[0];
    let mut flat = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H as usize {
        flat.extend_from_slice(&plane.data[y * plane.stride..y * plane.stride + (W as usize) * 4]);
    }
    flat
}

fn check_bytes(name: &str, got: &[u8]) {
    let want = load_ref(name);
    let d = diff(got, &want);
    eprintln!(
        "{name}: interior max|Δc| = {}; edge max|Δα| = {}, max|Δpremul| = {}, mean|Δα| = {:.3} over {} px",
        d.max_interior, d.max_alpha, d.max_edge_premul, d.mean_edge_alpha, d.edge_pixels
    );
    assert!(
        d.interior_over == 0,
        "{name}: {} interior px over tol (max {})",
        d.interior_over,
        d.max_interior
    );
    assert!(
        d.edge_over == 0,
        "{name}: {} edge px over tol (max |Δα| {})",
        d.edge_over,
        d.max_alpha
    );
    assert!(
        d.mean_edge_alpha <= EDGE_MEAN_TOL,
        "{name}: mean edge |Δα| {:.3}",
        d.mean_edge_alpha
    );
}

/// `objectBoundingBox` linear gradient (SVG defaults x1 = 0%, x2 =
/// 100%) with `gradientTransform="rotate(45 0.5 0.5)"` on a 48 × 24
/// rect: the non-uniform box turns the 45° bbox-space rotation into a
/// skewed user-space gradient whose normal is not perpendicular to its
/// vector (SVG 2 §14.2.2.1).
#[test]
fn grad_linear_bbox_transform() {
    let g = LinearGradient::new(Point::new(0.0, 0.0), Point::new(1.0, 0.0))
        .with_stop(GradientStop::new(0.0, BLACK))
        .with_stop(GradientStop::new(1.0, WHITE));
    let gp = GradientPaint::linear(g)
        .with_units(GradientUnits::ObjectBoundingBox)
        .with_transform(rotate_about(45.0, 0.5, 0.5));
    check_bytes(
        "grad_linear_bbox_transform",
        &render_gradient_fill(rect(8.0, 12.0, 48.0, 24.0), &gp),
    );
}

/// `objectBoundingBox` radial gradient with an off-centre focus and
/// `gradientTransform="translate(0.1 0)"` on a 56 × 28 rect: elliptical
/// rings (§14.2.3.1), focus displaced inside them.
#[test]
fn grad_radial_bbox_ellipse() {
    let g = RadialGradient::new(Point::new(0.4, 0.5), 0.5)
        .with_focal(Point::new(0.3, 0.5))
        .with_stop(GradientStop::new(0.0, RED))
        .with_stop(GradientStop::new(1.0, BLUE));
    let gp = GradientPaint::radial(g)
        .with_units(GradientUnits::ObjectBoundingBox)
        .with_transform(Transform2D::translate(0.1, 0.0));
    check_bytes(
        "grad_radial_bbox_ellipse",
        &render_gradient_fill(rect(4.0, 16.0, 56.0, 28.0), &gp),
    );
}

/// The supersampled kernel is kept available (it is what the issue
/// measured against); its 4-sub-scanline vertical quantisation puts it
/// more than twice outside the exact-area tolerance on the
/// near-horizontal edge — recorded here so the gain stays visible.
#[test]
fn supersampled_mode_is_coarser_than_analytic() {
    let v = frame(Group::new().with_child(fill(
        polygon(&[(2.0, 20.0), (62.0, 21.5), (62.0, 44.3), (2.0, 42.0)]),
        black(),
        FillRule::NonZero,
    )));
    let want = load_ref("aa_near_horizontal");
    let analytic = diff(&render(&v), &want);
    let mut r = Renderer::new(W, H).with_coverage(CoverageMode::Supersampled);
    r.supersampling = 4;
    let out = r.render(&v);
    let plane = &out.planes[0];
    let mut flat = Vec::new();
    for y in 0..H as usize {
        flat.extend_from_slice(&plane.data[y * plane.stride..y * plane.stride + (W as usize) * 4]);
    }
    let ss4 = diff(&flat, &want);
    eprintln!(
        "near-horizontal edge: analytic max|Δα| {} vs 4× supersampled max|Δα| {}",
        analytic.max_alpha, ss4.max_alpha
    );
    assert!(analytic.max_alpha <= EDGE_TOL);
    assert!(
        ss4.max_alpha > 2 * EDGE_TOL,
        "ss4 max |Δα| {}",
        ss4.max_alpha
    );
}
