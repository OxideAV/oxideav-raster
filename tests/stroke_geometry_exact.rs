//! Round 484 stroke-geometry regression tests: a round-joined,
//! round-capped stroke is the set of points within half the width of
//! the polyline (the Minkowski sum with a disk), so its coverage can be
//! checked against a brute-force distance field — including the
//! thick-on-tight cases (stroke wider than the curvature radius, short
//! segments between sharp turns, tiny closed loops) where the classic
//! offset outline cancels part of the stroke.

use oxideav_core::{
    FillRule, Group, LineCap, LineJoin, Node, Paint, Path, PathNode, Point, Rgba, Stroke,
    VectorFrame,
};
use oxideav_raster::Renderer;

fn render(path: Path, width: f32, cap: LineCap, join: LineJoin, n: u32) -> Vec<u8> {
    let f = VectorFrame {
        width: n as f32,
        height: n as f32,
        view_box: None,
        root: Group {
            children: vec![Node::Path(PathNode {
                path,
                fill: None,
                stroke: Some(Stroke {
                    width,
                    paint: Paint::Solid(Rgba::opaque(0, 0, 0)),
                    cap,
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
    Renderer::new(n, n).render(&f).planes[0]
        .data
        .chunks(4)
        .map(|p| p[3])
        .collect()
}

fn seg_dist(q: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 {
        (((q.0 - a.0) * dx + (q.1 - a.1) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    ((q.0 - a.0 - t * dx).powi(2) + (q.1 - a.1 - t * dy).powi(2)).sqrt()
}

/// Max |rendered − distance-field| coverage, in levels.
fn max_error(pts: &[(f32, f32)], closed: bool, width: f32, n: u32) -> (f64, (u32, u32)) {
    let mut p = Path::new();
    p.move_to(Point::new(pts[0].0, pts[0].1));
    for q in &pts[1..] {
        p.line_to(Point::new(q.0, q.1));
    }
    if closed {
        p.close();
    }
    let a = render(p, width, LineCap::Round, LineJoin::Round, n);
    let k = pts.len();
    let mut segs: Vec<((f64, f64), (f64, f64))> = pts
        .windows(2)
        .map(|w| {
            (
                (w[0].0 as f64, w[0].1 as f64),
                (w[1].0 as f64, w[1].1 as f64),
            )
        })
        .collect();
    if closed {
        segs.push((
            (pts[k - 1].0 as f64, pts[k - 1].1 as f64),
            (pts[0].0 as f64, pts[0].1 as f64),
        ));
    }
    let half = width as f64 / 2.0;
    let mut worst = (0.0f64, (0, 0));
    let s = 12;
    for y in 0..n {
        for x in 0..n {
            let mut hit = 0;
            for sy in 0..s {
                for sx in 0..s {
                    let q = (
                        x as f64 + (sx as f64 + 0.5) / s as f64,
                        y as f64 + (sy as f64 + 0.5) / s as f64,
                    );
                    if segs.iter().any(|&(a, b)| seg_dist(q, a, b) <= half) {
                        hit += 1;
                    }
                }
            }
            let want = hit as f64 / (s * s) as f64 * 255.0;
            let d = (a[(y * n + x) as usize] as f64 - want).abs();
            if d > worst.0 {
                worst = (d, (x, y));
            }
        }
    }
    worst
}

/// A tiny closed triangle stroked 15 px wide: the inner offset loop
/// turns inside out and used to cancel the outer one, leaving a hole
/// (a 233-level error) in the middle of what must be a solid blob.
#[test]
fn thick_stroke_around_a_tiny_loop_has_no_hole() {
    let tri = [(33.5, 20.9), (26.6, 19.6), (20.4, 24.8)];
    let (err, at) = max_error(&tri, true, 15.0, 48);
    assert!(err <= 12.0, "error {err} at {at:?}");
    // And a small circle (polygonised) under a stroke wider than its
    // diameter.
    let circle: Vec<(f32, f32)> = (0..24)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / 24.0;
            (24.0 + 3.0 * a.cos(), 24.0 + 3.0 * a.sin())
        })
        .collect();
    let (err, at) = max_error(&circle, true, 14.0, 48);
    assert!(err <= 12.0, "circle: error {err} at {at:?}");
}

/// Property: random open / closed polylines with widths from hairline
/// to very thick agree with the distance field everywhere (12 levels
/// covers the 12×12 sampling of the reference and the 1/32 px
/// tessellation of joins and caps).
#[test]
fn round_strokes_match_the_distance_field() {
    let mut seed = 7u64;
    let mut rnd = |lo: f32, hi: f32| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + ((seed >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
    };
    for case in 0..24 {
        let k = 3 + case % 5;
        let pts: Vec<(f32, f32)> = (0..k).map(|_| (rnd(10.0, 30.0), rnd(10.0, 30.0))).collect();
        let closed = case % 3 == 0;
        let width = [1.5f32, 4.0, 9.0, 15.0][case % 4];
        let (err, at) = max_error(&pts, closed, width, 40);
        assert!(
            err <= 12.0,
            "case {case} closed={closed} w={width}: error {err} at {at:?} pts={pts:?}"
        );
    }
}

/// Thick round caps used to be tessellated to a 0.2 px sagitta, which
/// under-covers the chord midpoints by up to 51 levels; the 1/32 px
/// tolerance keeps a single fat dot within the distance-field bound.
#[test]
fn fat_round_cap_is_round() {
    let (err, at) = max_error(&[(20.0, 20.0), (20.5, 20.0)], false, 30.0, 40);
    assert!(err <= 12.0, "error {err} at {at:?}");
}
