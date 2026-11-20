//! Stroke geometry at its extremes, against a reference that needs no
//! outline: a point is inside the stroke of a polyline when it lies in
//! the rectangle of a segment, in the join piece of a vertex (bevel
//! triangle, the miter quadrilateral while `1 / sin(θ/2)` does not
//! exceed the limit — SVG 2 §13.5.5 — or the round join's disc), or in
//! a cap. Each pixel's coverage is estimated with a 512-point rank-1
//! lattice (area error well under one 8-bit level).
//!
//! Covered: turning angles from 0.5° to 179.5°, miter limits from 1 to
//! 100 including values just either side of the limit a given angle
//! needs, pens from 5 px down to hairlines of 1/50 px, and all three
//! caps.

use oxideav_core::{
    FillRule, Group, LineCap, LineJoin, Node, Paint, Path, PathNode, Point, Rgba, Stroke,
    VectorFrame,
};
use oxideav_raster::Renderer;

type P = (f64, f64);

fn inside_convex(q: P, poly: &[P]) -> bool {
    let (mut pos, mut neg) = (false, false);
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let c = (b.0 - a.0) * (q.1 - a.1) - (b.1 - a.1) * (q.0 - a.0);
        pos |= c > 0.0;
        neg |= c < 0.0;
    }
    !(pos && neg)
}

/// The stroke as convex polygons plus discs `(centre, radius)`.
struct Shape {
    polys: Vec<Vec<P>>,
    discs: Vec<(P, f64)>,
}

impl Shape {
    fn contains(&self, q: P) -> bool {
        self.discs
            .iter()
            .any(|&(c, r)| (q.0 - c.0).powi(2) + (q.1 - c.1).powi(2) <= r * r)
            || self.polys.iter().any(|p| inside_convex(q, p))
    }
}

fn reference(pts: &[P], half: f64, join: LineJoin, limit: f64, cap: LineCap) -> Shape {
    let mut s = Shape {
        polys: Vec::new(),
        discs: Vec::new(),
    };
    let unit = |a: P, b: P| {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let l = (dx * dx + dy * dy).sqrt();
        (dx / l, dy / l)
    };
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let d = unit(a, b);
        let n = (d.1 * half, -d.0 * half);
        s.polys.push(vec![
            (a.0 + n.0, a.1 + n.1),
            (b.0 + n.0, b.1 + n.1),
            (b.0 - n.0, b.1 - n.1),
            (a.0 - n.0, a.1 - n.1),
        ]);
    }
    for i in 1..pts.len() - 1 {
        let (p, c, nx) = (pts[i - 1], pts[i], pts[i + 1]);
        let (d0, d1) = (unit(p, c), unit(c, nx));
        let cross = d0.0 * d1.1 - d0.1 * d1.0;
        // Outer side normal sign: away from the turn.
        let sgn = if cross > 0.0 { 1.0 } else { -1.0 };
        let o0 = (c.0 + d0.1 * half * sgn, c.1 - d0.0 * half * sgn);
        let o1 = (c.0 + d1.1 * half * sgn, c.1 - d1.0 * half * sgn);
        match join {
            LineJoin::Round => s.discs.push((c, half)),
            _ => {
                let mut poly = vec![c, o0];
                if join == LineJoin::Miter {
                    // cos of the turning angle φ; the angle between the
                    // segments is θ = π − φ, so sin(θ/2) = cos(φ/2).
                    let cos_phi = (d0.0 * d1.0 + d0.1 * d1.1).clamp(-1.0, 1.0);
                    let cos_half = ((1.0 + cos_phi) * 0.5).sqrt();
                    if cos_half > 1e-9 && 1.0 / cos_half <= limit {
                        // Apex on the bisector, half / cos(φ/2) from the vertex.
                        let bis = (o0.0 + o1.0 - 2.0 * c.0, o0.1 + o1.1 - 2.0 * c.1);
                        let bl = (bis.0 * bis.0 + bis.1 * bis.1).sqrt();
                        let reach = half / cos_half;
                        poly.push((c.0 + bis.0 / bl * reach, c.1 + bis.1 / bl * reach));
                    }
                }
                poly.push(o1);
                s.polys.push(poly);
            }
        }
    }
    for (end, toward) in [(pts[0], pts[1]), (pts[pts.len() - 1], pts[pts.len() - 2])] {
        let d = unit(toward, end); // pointing out of the path
        let n = (d.1 * half, -d.0 * half);
        match cap {
            LineCap::Round => s.discs.push((end, half)),
            LineCap::Square => s.polys.push(vec![
                (end.0 + n.0, end.1 + n.1),
                (end.0 + n.0 + d.0 * half, end.1 + n.1 + d.1 * half),
                (end.0 - n.0 + d.0 * half, end.1 - n.1 + d.1 * half),
                (end.0 - n.0, end.1 - n.1),
            ]),
            _ => {}
        }
    }
    s
}

/// Worst per-pixel error (8-bit levels) and the summed coverage error
/// relative to the reference area — after allowing half a level of
/// quantisation per touched pixel, which does not average out when a
/// straight hairline gives every pixel of a row the same fraction.
fn compare(pts: &[P], width: f64, join: LineJoin, limit: f64, cap: LineCap, n: u32) -> (f64, f64) {
    let mut p = Path::new();
    p.move_to(Point::new(pts[0].0 as f32, pts[0].1 as f32));
    for q in &pts[1..] {
        p.line_to(Point::new(q.0 as f32, q.1 as f32));
    }
    let frame = VectorFrame {
        width: n as f32,
        height: n as f32,
        view_box: None,
        root: Group {
            children: vec![Node::Path(PathNode {
                path: p,
                fill: None,
                stroke: Some(Stroke {
                    width: width as f32,
                    paint: Paint::Solid(Rgba::opaque(0, 0, 0)),
                    cap,
                    join,
                    miter_limit: limit as f32,
                    dash: None,
                }),
                fill_rule: FillRule::NonZero,
            })],
            ..Group::default()
        },
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let out = Renderer::new(n, n).render(&frame);
    let alpha: Vec<u8> = out.planes[0].data.chunks(4).map(|p| p[3]).collect();
    // The reference sees the coordinates the renderer was given.
    let pts: Vec<P> = pts
        .iter()
        .map(|p| (p.0 as f32 as f64, p.1 as f32 as f64))
        .collect();
    let shape = reference(
        &pts,
        width as f32 as f64 / 2.0,
        join,
        limit as f32 as f64,
        cap,
    );
    const SAMPLES: usize = 512;
    let (mut worst, mut got_sum, mut want_sum, mut touched) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..n {
        for x in 0..n {
            let mut hit = 0usize;
            for i in 0..SAMPLES {
                let u = (i as f64 + 0.5) / SAMPLES as f64;
                let v = (i as f64 * 0.618_033_988_749_894_9).fract();
                if shape.contains((x as f64 + u, y as f64 + v)) {
                    hit += 1;
                }
            }
            let want = hit as f64 / SAMPLES as f64 * 255.0;
            let got = alpha[(y * n + x) as usize] as f64;
            worst = worst.max((got - want).abs());
            got_sum += got;
            want_sum += want;
            if hit != 0 {
                touched += 1.0;
            }
        }
    }
    let excess = ((got_sum - want_sum).abs() - 0.5 * touched).max(0.0);
    (worst, excess / want_sum.max(1.0))
}

fn elbow(deg: f64, len: f64) -> [P; 3] {
    // Interior angle `deg` at the middle vertex.
    let t = (180.0 - deg).to_radians();
    let (cx, cy) = (14.3, 13.6);
    [
        (cx - len, cy + 0.21),
        (cx, cy),
        (cx + len * t.cos(), cy + len * t.sin()),
    ]
}

#[test]
fn joins_at_extreme_angles_and_miter_limits() {
    let mut worst_seen = 0.0f64;
    for deg in [0.5f64, 7.0, 29.0, 60.0, 91.0, 135.0, 179.5] {
        // The limit this angle needs, and limits around it.
        let need = 1.0 / (deg.to_radians() / 2.0).sin();
        for limit in [1.0, need * 0.98, need * 1.02, 100.0] {
            for width in [0.6f64, 5.0] {
                for join in [LineJoin::Miter, LineJoin::Bevel, LineJoin::Round] {
                    // A miter spike may reach limit · width / 2: keep it
                    // on the canvas (off-canvas parts are not compared).
                    let pts = elbow(deg, 8.0);
                    let (worst, rel) = compare(&pts, width, join, limit, LineCap::Butt, 28);
                    worst_seen = worst_seen.max(worst);
                    // A round join is a polygon whose chords stay within
                    // 1/32 px of the circle.
                    let tolerance = if join == LineJoin::Round { 4.5 } else { 2.5 };
                    assert!(
                        worst <= tolerance,
                        "{join:?} join, {deg}° corner, limit {limit:.3} (needs {need:.3}), \
                         width {width}: worst pixel off by {worst:.2} levels, area by {:.2} %",
                        rel * 100.0
                    );
                }
            }
        }
    }
    println!("joins: worst pixel error {worst_seen:.2} levels");
}

#[test]
fn hairlines_and_caps() {
    let mut worst_seen = 0.0f64;
    for width in [1.0f64, 0.5, 0.1, 0.02] {
        for cap in [LineCap::Butt, LineCap::Square, LineCap::Round] {
            for (i, deg) in [3.0f64, 44.0, 90.0, 133.0, 178.0].into_iter().enumerate() {
                let pts = elbow(deg, 7.5 + i as f64 * 0.37);
                let (worst, rel) = compare(&pts, width, LineJoin::Miter, 4.0, cap, 28);
                worst_seen = worst_seen.max(worst);
                assert!(
                    // (The lattice estimate of a band 1/50 px wide is
                    // itself only good to a few percent.)
                    worst <= if cap == LineCap::Round { 4.5 } else { 2.5 }
                        && rel <= if width < 0.5 { 0.04 } else { 0.01 },
                    "{cap:?} cap, width {width}, {deg}° corner: worst pixel off by {worst:.2} \
                     levels, area by {:.2} %",
                    rel * 100.0
                );
            }
        }
    }
    println!("hairlines: worst pixel error {worst_seen:.2} levels");
}
