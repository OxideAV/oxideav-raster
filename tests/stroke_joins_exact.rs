//! Round 484: bevel / miter joins (butt caps) at every turning angle,
//! up to near-reversals, against a brute-force point-membership
//! reference: the union of the segment rectangles and, at each vertex,
//! the join polygon on the outer side (SVG 2 §13.5.5: bevel triangle,
//! or the miter quadrilateral while `1/sin(θ/2) ≤ miter_limit`).

use oxideav_core::{
    FillRule, Group, LineCap, LineJoin, Node, Paint, Path, PathNode, Point, Rgba, Stroke,
    VectorFrame,
};
use oxideav_raster::Renderer;

type P = (f64, f64);

fn inside_convex(q: P, poly: &[P]) -> bool {
    // Either orientation.
    let mut pos = false;
    let mut neg = false;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let c = (b.0 - a.0) * (q.1 - a.1) - (b.1 - a.1) * (q.0 - a.0);
        pos |= c > 0.0;
        neg |= c < 0.0;
    }
    !(pos && neg)
}

fn pieces(pts: &[P], half: f64, join: LineJoin, ml: f64) -> Vec<Vec<P>> {
    let mut out = Vec::new();
    let normal = |a: P, b: P| {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let l = (dx * dx + dy * dy).sqrt();
        (dy / l, -dx / l)
    };
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let n = normal(a, b);
        out.push(vec![
            (a.0 + n.0 * half, a.1 + n.1 * half),
            (b.0 + n.0 * half, b.1 + n.1 * half),
            (b.0 - n.0 * half, b.1 - n.1 * half),
            (a.0 - n.0 * half, a.1 - n.1 * half),
        ]);
    }
    for i in 1..pts.len() - 1 {
        let (p, c, nx) = (pts[i - 1], pts[i], pts[i + 1]);
        let n0 = normal(p, c);
        let n1 = normal(c, nx);
        // Outer side: opposite to the turn.
        let turn_right = n0.0 * (nx.0 - c.0) + n0.1 * (nx.1 - c.1) > 0.0;
        let s = if turn_right { -1.0 } else { 1.0 };
        let o0 = (c.0 + n0.0 * half * s, c.1 + n0.1 * half * s);
        let o1 = (c.0 + n1.0 * half * s, c.1 + n1.1 * half * s);
        let mut poly = vec![c, o0];
        if join == LineJoin::Miter {
            // Miter apex: intersection of the two outer offset lines.
            let d0 = (c.0 - p.0, c.1 - p.1);
            let d1 = (nx.0 - c.0, nx.1 - c.1);
            let den = d0.0 * d1.1 - d0.1 * d1.0;
            if den.abs() > 1e-12 {
                let t = ((o1.0 - o0.0) * d1.1 - (o1.1 - o0.1) * d1.0) / den;
                let m = (o0.0 + d0.0 * t, o0.1 + d0.1 * t);
                let ml_len = ((m.0 - c.0).powi(2) + (m.1 - c.1).powi(2)).sqrt();
                if ml_len <= ml * half {
                    poly.push(m);
                }
            }
        }
        poly.push(o1);
        out.push(poly);
    }
    out
}

fn check(pts: &[(f32, f32)], width: f32, join: LineJoin, n: u32) -> f64 {
    let mut p = Path::new();
    p.move_to(Point::new(pts[0].0, pts[0].1));
    for q in &pts[1..] {
        p.line_to(Point::new(q.0, q.1));
    }
    let f = VectorFrame {
        width: n as f32,
        height: n as f32,
        view_box: None,
        root: Group {
            children: vec![Node::Path(PathNode {
                path: p,
                fill: None,
                stroke: Some(Stroke {
                    width,
                    paint: Paint::Solid(Rgba::opaque(0, 0, 0)),
                    cap: LineCap::Butt,
                    join,
                    miter_limit: 4.0,
                    dash: None,
                }),
                fill_rule: FillRule::NonZero,
            })],
            ..Group::default()
        },
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let a: Vec<u8> = Renderer::new(n, n).render(&f).planes[0]
        .data
        .chunks(4)
        .map(|p| p[3])
        .collect();
    let p64: Vec<P> = pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    let polys = pieces(&p64, width as f64 / 2.0, join, 4.0);
    let s = 12;
    let mut worst = 0.0f64;
    for y in 0..n {
        for x in 0..n {
            let mut hit = 0;
            for sy in 0..s {
                for sx in 0..s {
                    let q = (
                        x as f64 + (sx as f64 + 0.5) / s as f64,
                        y as f64 + (sy as f64 + 0.5) / s as f64,
                    );
                    if polys.iter().any(|poly| inside_convex(q, poly)) {
                        hit += 1;
                    }
                }
            }
            let want = hit as f64 / (s * s) as f64 * 255.0;
            worst = worst.max((a[(y * n + x) as usize] as f64 - want).abs());
        }
    }
    worst
}

/// Every turning angle from a gentle bend to a near-reversal (179°),
/// for thin and thick strokes, both joins.
#[test]
fn bevel_and_miter_joins_at_every_angle() {
    for join in [LineJoin::Bevel, LineJoin::Miter] {
        for deg in [10.0f32, 45.0, 90.0, 120.0, 150.0, 170.0, 179.0] {
            for width in [2.0f32, 7.0] {
                let t = deg.to_radians();
                let (cx, cy) = (20.0, 20.0);
                let pts = [
                    (cx - 14.0, cy + 0.3),
                    (cx, cy),
                    (cx + 14.0 * t.cos(), cy + 14.0 * t.sin()),
                ];
                let err = check(&pts, width, join, 40);
                assert!(err <= 12.0, "{join:?} {deg}° w={width}: error {err}");
            }
        }
    }
}

/// Short zig-zags with sharp turns and thick strokes (segments shorter
/// than the half-width) — the offset outline's worst case.
#[test]
fn thick_zigzag_with_short_segments() {
    for join in [LineJoin::Bevel, LineJoin::Miter] {
        let pts: Vec<(f32, f32)> = (0..9)
            .map(|i| (8.0 + i as f32 * 3.0, if i % 2 == 0 { 18.0 } else { 22.0 }))
            .collect();
        let err = check(&pts, 9.0, join, 40);
        assert!(err <= 12.0, "{join:?}: error {err}");
    }
}

fn stroke_alpha(
    p: Path,
    width: f32,
    cap: LineCap,
    dash: Option<oxideav_core::DashPattern>,
    w: u32,
    h: u32,
) -> Vec<f32> {
    let f = VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root: Group {
            children: vec![Node::Path(PathNode {
                path: p,
                fill: None,
                stroke: Some(Stroke {
                    width,
                    paint: Paint::Solid(Rgba::opaque(0, 0, 0)),
                    cap,
                    join: LineJoin::Round,
                    miter_limit: 4.0,
                    dash,
                }),
                fill_rule: FillRule::NonZero,
            })],
            ..Group::default()
        },
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    Renderer::new(w, h).render(&f).planes[0]
        .data
        .chunks(4)
        .map(|p| p[3] as f32 / 255.0)
        .collect()
}

/// SVG 2 §13.5.4: zero-length subpaths, open (`M L` to the same point)
/// or closed (`M Z`), paint a full circle with `round` caps and a
/// stroke-width square with `square` caps, nothing with `butt`. They
/// used to paint nothing at all.
#[test]
fn zero_length_subpaths_paint_their_cap() {
    let mut open = Path::new();
    open.move_to(Point::new(10.3, 10.6))
        .line_to(Point::new(10.3, 10.6));
    let mut closed = Path::new();
    closed.move_to(Point::new(20.5, 10.2));
    closed.close();
    for p in [open, closed] {
        let round: f32 = stroke_alpha(p.clone(), 8.0, LineCap::Round, None, 32, 24)
            .iter()
            .sum();
        assert!(
            (round - std::f32::consts::PI * 16.0).abs() < 0.3,
            "round dot area {round}"
        );
        let square: f32 = stroke_alpha(p.clone(), 8.0, LineCap::Square, None, 32, 24)
            .iter()
            .sum();
        assert!((square - 64.0).abs() < 0.05, "square dot area {square}");
        let butt: f32 = stroke_alpha(p, 8.0, LineCap::Butt, None, 32, 24)
            .iter()
            .sum();
        assert_eq!(butt, 0.0);
    }
}

/// A `0 10` dash array with round caps is a dotted line: one dot per
/// period (zero-length dashes paint their caps).
#[test]
fn zero_length_dashes_make_dots() {
    let mut p = Path::new();
    p.move_to(Point::new(5.0, 8.0))
        .line_to(Point::new(56.0, 8.0));
    let a = stroke_alpha(
        p,
        4.0,
        LineCap::Round,
        Some(oxideav_core::DashPattern {
            array: vec![0.0, 10.0],
            offset: 0.0,
        }),
        64,
        16,
    );
    // Dots centred at x = 15, 25, 35, 45, 55 (and possibly 5).
    for cx in [15usize, 25, 35, 45, 55] {
        assert!(a[8 * 64 + cx] > 0.9, "no dot at {cx}");
        assert!(a[8 * 64 + cx + 5] < 0.01, "gap after {cx} not empty");
    }
    let total: f32 = a.iter().sum();
    let dot = std::f32::consts::PI * 4.0;
    assert!(total > 4.5 * dot && total < 6.5 * dot, "total {total}");
}
