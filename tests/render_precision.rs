//! Round 484 regression tests: curve-flattening accuracy, huge
//! coordinates, off-canvas culling, and user-space stroking under
//! rotation / non-uniform scale / dashing.

use oxideav_core::{
    DashPattern, FillRule, Group, LineCap, LineJoin, Node, Paint, Path, PathNode, Point, Rgba,
    Stroke, Transform2D, VectorFrame,
};
use oxideav_raster::{Renderer, flatten_path, rasterize_fill_analytic};

fn frame(w: u32, h: u32, children: Vec<Node>) -> VectorFrame {
    VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root: Group {
            children,
            ..Group::default()
        },
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

fn alpha_plane(r: &Renderer, f: &VectorFrame) -> Vec<u8> {
    let out = r.render(f);
    out.planes[0].data.chunks(4).map(|p| p[3]).collect()
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next() * (hi - lo)
    }
}

fn cubic_at(p: [(f64, f64); 4], t: f64) -> (f64, f64) {
    let u = 1.0 - t;
    let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    (
        a * p[0].0 + b * p[1].0 + c * p[2].0 + d * p[3].0,
        a * p[0].1 + b * p[1].1 + c * p[2].1 + d * p[3].1,
    )
}

fn dist_to_polyline(q: (f64, f64), pts: &[(f32, f32)]) -> f64 {
    let mut best = f64::INFINITY;
    for w in pts.windows(2) {
        let (ax, ay) = (w[0].0 as f64, w[0].1 as f64);
        let (bx, by) = (w[1].0 as f64, w[1].1 as f64);
        let (dx, dy) = (bx - ax, by - ay);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 {
            (((q.0 - ax) * dx + (q.1 - ay) * dy) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (px, py) = (ax + t * dx, ay + t * dy);
        best = best.min(((q.0 - px).powi(2) + (q.1 - py).powi(2)).sqrt());
    }
    best
}

/// Property: the flattened polyline of any cubic stays within the
/// documented 1/32 px of the true curve (at every zoom level), so
/// large-scale curves never facet.
#[test]
fn flattening_error_is_bounded_in_device_pixels() {
    let mut rng = Lcg(0xF1A7);
    for case in 0..40 {
        let scale = [1.0f32, 7.5, 120.0, 3000.0][case % 4];
        let pts: Vec<Point> = (0..4)
            .map(|_| Point::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)))
            .collect();
        let mut p = Path::new();
        p.move_to(pts[0]).cubic_to(pts[1], pts[2], pts[3]);
        let t = Transform2D::scale(scale, scale);
        let cs = flatten_path(&p.commands, &t);
        let poly = &cs[0].points;
        let dev: [(f64, f64); 4] = std::array::from_fn(|i| {
            (
                pts[i].x as f64 * scale as f64,
                pts[i].y as f64 * scale as f64,
            )
        });
        let mut worst = 0.0f64;
        for k in 0..=2000 {
            let q = cubic_at(dev, k as f64 / 2000.0);
            worst = worst.max(dist_to_polyline(q, poly));
        }
        // 1/32 px plus f32 rounding of the emitted vertices.
        let slack = 1.0 / 32.0 + 2e-7 * scale as f64 * 4.0;
        assert!(
            worst <= slack,
            "case {case} scale {scale}: error {worst} px > {slack}"
        );
    }
}

/// The flattener is curvature-driven, not chord-length-driven: a
/// radius-1000 px circle needs a few hundred segments (the historical
/// 0.5 px chord rule produced ~12 600), and a radius-4 px one a handful.
#[test]
fn flattening_density_follows_curvature() {
    let circle = |r: f32| {
        let k = 0.552_284_7 * r;
        let mut p = Path::new();
        p.move_to(Point::new(r, 0.0));
        p.cubic_to(Point::new(r, k), Point::new(k, r), Point::new(0.0, r));
        p.cubic_to(Point::new(-k, r), Point::new(-r, k), Point::new(-r, 0.0));
        p.cubic_to(Point::new(-r, -k), Point::new(-k, -r), Point::new(0.0, -r));
        p.cubic_to(Point::new(k, -r), Point::new(r, -k), Point::new(r, 0.0));
        p.close();
        flatten_path(&p.commands, &Transform2D::identity())[0]
            .points
            .len()
    };
    let big = circle(1000.0);
    assert!((150..=1500).contains(&big), "r=1000: {big} points");
    let small = circle(4.0);
    assert!((8..=60).contains(&small), "r=4: {small} points");
}

/// Exact coverage of pixel `(i, j)` by the half-plane `y ≥ line(x)`.
fn half_plane_cov(i: u32, j: u32, line: impl Fn(f64) -> f64) -> f64 {
    let n = 4096;
    let mut acc = 0.0;
    for k in 0..n {
        let x = i as f64 + (k as f64 + 0.5) / n as f64;
        let lo = line(x).max(j as f64);
        acc += (j as f64 + 1.0 - lo).clamp(0.0, 1.0);
    }
    acc / n as f64
}

/// A polygon whose vertices lie ~3·10⁶ px off-canvas: the edge set-up
/// is done in double precision after clipping, so the shallow edge
/// crossing the canvas lands exactly where its endpoints put it. The
/// historical f32 `x0 + (y − y0)·slope` evaluation carried the
/// endpoint magnitude's rounding (~0.25 px here) into the canvas.
#[test]
fn far_off_canvas_vertices_keep_subpixel_precision() {
    let (ax, ay) = (-3_000_000.7f32, -999_993.1f32);
    let (bx, by) = (3_000_017.3f32, 1_000_011.9f32);
    let mut p = Path::new();
    p.move_to(Point::new(ax, ay))
        .line_to(Point::new(bx, by))
        .line_to(Point::new(ax, by))
        .close();
    let r = Renderer::new(16, 16);
    let a = alpha_plane(
        &r,
        &frame(
            16,
            16,
            vec![Node::Path(PathNode {
                path: p,
                fill: Some(Paint::Solid(Rgba::opaque(0, 0, 0))),
                stroke: None,
                fill_rule: FillRule::NonZero,
            })],
        ),
    );
    let (ax, ay, bx, by) = (ax as f64, ay as f64, bx as f64, by as f64);
    let line = |x: f64| ay + (x - ax) * (by - ay) / (bx - ax);
    let mut covered = 0;
    for j in 0..16 {
        for i in 0..16 {
            let want = (half_plane_cov(i, j, line) * 255.0).round() as i32;
            let got = a[(j * 16 + i) as usize] as i32;
            covered += (want > 0) as i32;
            assert!((got - want).abs() <= 1, "({i},{j}) got {got} want {want}");
        }
    }
    assert!(covered > 20 && covered < 256, "line must cross the canvas");
}

/// Replacing an off-canvas curve by its chord is exact for a fill: a
/// shape with huge lobes far outside the canvas renders exactly as the
/// unculled flattening does.
#[test]
fn off_canvas_curve_culling_is_exact() {
    let mut p = Path::new();
    p.move_to(Point::new(4.3, 3.1))
        .cubic_to(
            Point::new(-900.0, -40.0),
            Point::new(-700.0, 90.0),
            Point::new(5.2, 27.7),
        )
        .cubic_to(
            Point::new(30.0, 900.0),
            Point::new(900.0, 400.0),
            Point::new(28.6, 20.2),
        )
        .cubic_to(
            Point::new(400.0, -800.0),
            Point::new(20.0, -300.0),
            Point::new(4.3, 3.1),
        )
        .close();
    let r = Renderer::new(32, 32);
    let got = alpha_plane(
        &r,
        &frame(
            32,
            32,
            vec![Node::Path(PathNode {
                path: p.clone(),
                fill: Some(Paint::Solid(Rgba::opaque(0, 0, 0))),
                stroke: None,
                fill_rule: FillRule::NonZero,
            })],
        ),
    );
    let full = rasterize_fill_analytic(
        &flatten_path(&p.commands, &Transform2D::identity()),
        32,
        32,
        FillRule::NonZero,
    );
    let mut max_d = 0;
    for (g, w) in got.iter().zip(&full.data) {
        max_d = max_d.max((*g as i32 - *w as i32).abs());
    }
    assert!(max_d <= 1, "culled vs unculled differ by {max_d}");
    assert!(full.nonzero_pixel_count() > 100);
}

fn stroked_line(from: Point, to: Point, width: f32, dash: Option<DashPattern>) -> PathNode {
    let mut p = Path::new();
    p.move_to(from).line_to(to);
    PathNode {
        path: p,
        fill: None,
        stroke: Some(Stroke {
            width,
            paint: Paint::Solid(Rgba::opaque(0, 0, 0)),
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: 4.0,
            dash,
        }),
        fill_rule: FillRule::NonZero,
    }
}

fn wrap(t: Transform2D, node: PathNode) -> Node {
    Node::Group(Group {
        transform: t,
        children: vec![Node::Path(node)],
        ..Group::default()
    })
}

/// A 90° rotation used to collapse the stroke width to zero (the width
/// scale was the mean of the matrix's diagonal, which a quarter turn
/// zeroes). The stroke must keep its width under any rotation.
#[test]
fn stroke_width_survives_rotation() {
    let r = Renderer::new(32, 32);
    // Rotate 90° about the origin, then move into view: the user-space
    // horizontal line y = 0, x ∈ [4, 28] becomes the vertical device
    // line x = 16, y ∈ [4, 28].
    let t = Transform2D {
        a: 0.0,
        b: 1.0,
        c: -1.0,
        d: 0.0,
        e: 16.0,
        f: 0.0,
    };
    let a = alpha_plane(
        &r,
        &frame(
            32,
            32,
            vec![wrap(
                t,
                stroked_line(Point::new(4.0, 0.0), Point::new(28.0, 0.0), 4.0, None),
            )],
        ),
    );
    for y in 6..26 {
        let row: Vec<u8> = (0..32).map(|x| a[y * 32 + x]).collect();
        assert_eq!(&row[14..18], &[255, 255, 255, 255], "row {y}: {row:?}");
        assert_eq!(row[13], 0, "row {y}");
        assert_eq!(row[18], 0, "row {y}");
    }
    // 45°: width 4 must stay 4 (the mean-diagonal factor gave 2.83).
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let t45 = Transform2D {
        a: s,
        b: s,
        c: -s,
        d: s,
        e: 4.0,
        f: 4.0,
    };
    let a = alpha_plane(
        &r,
        &frame(
            32,
            32,
            vec![wrap(
                t45,
                stroked_line(Point::new(4.0, 0.0), Point::new(24.0, 0.0), 4.0, None),
            )],
        ),
    );
    let area: f32 = a.iter().map(|&v| v as f32 / 255.0).sum();
    assert!(
        (area - 80.0).abs() < 0.5,
        "45° stroke area {area}, want 4·20"
    );
}

/// SVG strokes are user-space geometry: under `scale(1, 3)` a width-2
/// horizontal line is 6 px tall, and a vertical one stays 2 px wide.
#[test]
fn non_uniform_scale_stretches_the_pen() {
    let r = Renderer::new(32, 32);
    let t = Transform2D::scale(1.0, 3.0);
    let a = alpha_plane(
        &r,
        &frame(
            32,
            32,
            vec![
                wrap(
                    t,
                    stroked_line(Point::new(2.0, 2.0), Point::new(14.0, 2.0), 2.0, None),
                ),
                wrap(
                    t,
                    stroked_line(Point::new(24.0, 1.0), Point::new(24.0, 9.0), 2.0, None),
                ),
            ],
        ),
    );
    // Horizontal: device y ∈ [3, 9].
    let col: Vec<u8> = (0..32).map(|y| a[y * 32 + 8]).collect();
    assert_eq!(&col[3..9], &[255; 6], "{col:?}");
    assert_eq!(col[2], 0);
    assert_eq!(col[9], 0);
    // Vertical: device x ∈ [23, 25].
    let row: Vec<u8> = (0..32).map(|x| a[15 * 32 + x]).collect();
    assert_eq!(&row[23..25], &[255, 255], "{row:?}");
    assert_eq!(row[22], 0);
    assert_eq!(row[25], 0);
}

/// Dash lengths are user units: under `scale(2)` a `10 10` pattern has
/// a 40 px device period.
#[test]
fn dash_pattern_scales_with_the_ctm() {
    let r = Renderer::new(128, 8);
    let a = alpha_plane(
        &r,
        &frame(
            128,
            8,
            vec![wrap(
                Transform2D::scale(2.0, 2.0),
                stroked_line(
                    Point::new(0.0, 2.0),
                    Point::new(64.0, 2.0),
                    2.0,
                    Some(DashPattern {
                        array: vec![10.0, 10.0],
                        offset: 0.0,
                    }),
                ),
            )],
        ),
    );
    let row: Vec<u8> = (0..128).map(|x| a[4 * 128 + x]).collect();
    for x in 0..128 {
        let on = (x / 20) % 2 == 0;
        assert_eq!(row[x], if on { 255 } else { 0 }, "x={x}: {row:?}");
    }
}
