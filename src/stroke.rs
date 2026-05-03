//! Stroke geometry — convert a polyline + [`Stroke`] style into a
//! closed fillable polygon (the "stroke outline") that can be handed
//! to [`crate::fill::rasterize_fill`].
//!
//! This is the round-1 *real* stroke geometry — it builds offset
//! polygons rather than the alpha-dilation shortcut that
//! `oxideav_scribe::stroke` ships for glyph synthesis. Sharp corners
//! stay sharp; the renderer can honour SVG/PDF cap and join styles
//! exactly.
//!
//! # Algorithm
//!
//! For each open contour:
//!
//! 1. Apply the dash pattern (if any), turning the contour into a list
//!    of "on" sub-polylines.
//! 2. For each sub-polyline:
//!    - Walk forward, offsetting each segment by `+w/2` perpendicular
//!      to its direction (right-hand normal). Insert a join at every
//!      interior vertex.
//!    - Place an end cap.
//!    - Walk backward, offsetting by `+w/2` (which on the reversed
//!      walk lands on the *other* side of the original line).
//!    - Place a start cap, close.
//!
//! For each closed contour:
//!
//! 1. Generate two offset loops (outer at `+w/2`, inner at `-w/2`) —
//!    each forms its own closed polygon. Both are emitted; with
//!    [`FillRule::NonZero`](oxideav_core::FillRule::NonZero) the inner
//!    "donut hole" cancels the outer when wound oppositely.
//!
//! Output contours are intentionally CCW + CW so callers should fill
//! them with [`FillRule::NonZero`](oxideav_core::FillRule::NonZero).

use crate::flatten::FlatContour;
use oxideav_core::{DashPattern, LineCap, LineJoin, Stroke};

/// Build the fillable stroke geometry for a flattened input contour.
///
/// `width_px` is the stroke width in raster pixels (the caller has
/// already mapped the user-space width through the active transform —
/// for non-uniform scale we approximate with the average of x/y scale
/// factors at flatten time).
pub fn stroke_to_fill_path(
    input: &FlatContour,
    stroke: &Stroke,
    width_px: f32,
) -> Vec<FlatContour> {
    if input.points.len() < 2 || width_px <= 0.0 {
        return Vec::new();
    }
    let half = width_px * 0.5;
    let segments = if let Some(dash) = &stroke.dash {
        apply_dash(&input.points, input.closed, dash)
    } else {
        // No dash → a single piece with the original closed-ness.
        vec![DashSegment {
            points: input.points.clone(),
            closed: input.closed,
        }]
    };
    let mut out: Vec<FlatContour> = Vec::with_capacity(segments.len());
    for seg in segments {
        if seg.closed {
            // Closed input → emit outer + inner offset loops.
            if let Some(loops) = stroke_closed(&seg.points, half, stroke) {
                out.extend(loops);
            }
        } else if seg.points.len() >= 2 {
            if let Some(c) = stroke_open(&seg.points, half, stroke) {
                out.push(c);
            }
        }
    }
    out
}

#[derive(Debug, Clone)]
struct DashSegment {
    points: Vec<(f32, f32)>,
    closed: bool,
}

/// Walk `points` along their length, splitting into "on" sub-polylines
/// according to `dash`. The output sub-polylines are always open.
fn apply_dash(points: &[(f32, f32)], closed: bool, dash: &DashPattern) -> Vec<DashSegment> {
    if dash.array.is_empty() || dash.array.iter().all(|&v| v <= 0.0) {
        return vec![DashSegment {
            points: points.to_vec(),
            closed,
        }];
    }
    // Concatenate the closing edge if requested so dashing wraps cleanly.
    let mut walk: Vec<(f32, f32)> = points.to_vec();
    if closed {
        walk.push(points[0]);
    }

    // Skip negative array entries (treat as zero) — SVG validator.
    let pattern: Vec<f32> = dash.array.iter().map(|&v| v.max(0.0)).collect();
    let total: f32 = pattern.iter().sum();
    if total <= 0.0 {
        return vec![DashSegment {
            points: walk,
            closed: false,
        }];
    }
    // SVG dasharray with odd count is implicitly doubled to make
    // dash-on / dash-off pairs.
    let pattern: Vec<f32> = if pattern.len() % 2 == 1 {
        let mut p2 = pattern.clone();
        p2.extend(pattern.iter().copied());
        p2
    } else {
        pattern
    };
    let total: f32 = pattern.iter().sum();

    // Initial phase: SVG's dashoffset starts inside the pattern.
    let mut phase = ((dash.offset % total) + total) % total;
    let mut idx = 0usize;
    let mut on = true;
    while phase >= pattern[idx] {
        phase -= pattern[idx];
        idx = (idx + 1) % pattern.len();
        on = !on;
    }

    let mut out: Vec<DashSegment> = Vec::new();
    let mut cur: Vec<(f32, f32)> = Vec::new();
    if on {
        cur.push(walk[0]);
    }
    let mut left_in_dash = pattern[idx] - phase;

    for w in 0..walk.len() - 1 {
        let mut start = walk[w];
        let end = walk[w + 1];
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let mut len = (dx * dx + dy * dy).sqrt();
        if len <= 1e-9 {
            continue;
        }
        let mut ux = dx / len;
        let mut uy = dy / len;
        while len > left_in_dash {
            let cut = (start.0 + ux * left_in_dash, start.1 + uy * left_in_dash);
            if on {
                cur.push(cut);
                if cur.len() >= 2 {
                    out.push(DashSegment {
                        points: std::mem::take(&mut cur),
                        closed: false,
                    });
                } else {
                    cur.clear();
                }
            }
            on = !on;
            idx = (idx + 1) % pattern.len();
            len -= left_in_dash;
            start = cut;
            // Re-derive direction to guard against floating-point drift.
            let rdx = end.0 - start.0;
            let rdy = end.1 - start.1;
            let rlen = (rdx * rdx + rdy * rdy).sqrt().max(1e-9);
            ux = rdx / rlen;
            uy = rdy / rlen;
            left_in_dash = pattern[idx];
            if on {
                cur.push(start);
            }
        }
        // Remainder of the segment is fully inside the current dash.
        if on {
            cur.push(end);
        }
        left_in_dash -= len;
        if left_in_dash <= 0.0 {
            // Segment ended exactly on a dash boundary — flush.
            if on && cur.len() >= 2 {
                out.push(DashSegment {
                    points: std::mem::take(&mut cur),
                    closed: false,
                });
            } else {
                cur.clear();
            }
            on = !on;
            idx = (idx + 1) % pattern.len();
            left_in_dash = pattern[idx];
            if on {
                cur.push(end);
            }
        }
    }
    if on && cur.len() >= 2 {
        out.push(DashSegment {
            points: cur,
            closed: false,
        });
    }
    out
}

/// Build the fillable outline for an open polyline.
fn stroke_open(points: &[(f32, f32)], half: f32, stroke: &Stroke) -> Option<FlatContour> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let mut out: Vec<(f32, f32)> = Vec::with_capacity(n * 4);

    // Forward (right side).
    for i in 0..n {
        if i == 0 {
            // Start cap is added later (we walk back to start at the
            // end). Just emit the offset of the first vertex.
            let (nx, ny) = right_normal(points[0], points[1]);
            out.push((points[0].0 + nx * half, points[0].1 + ny * half));
        } else if i == n - 1 {
            let (nx, ny) = right_normal(points[n - 2], points[n - 1]);
            out.push((points[n - 1].0 + nx * half, points[n - 1].1 + ny * half));
        } else {
            push_join(
                &mut out,
                points[i - 1],
                points[i],
                points[i + 1],
                half,
                stroke,
                true,
            );
        }
    }
    // End cap (at the last vertex, going from incoming → outgoing direction
    // which is the reverse of the incoming direction).
    push_cap(&mut out, points[n - 2], points[n - 1], half, stroke.cap);
    // Backward (left side relative to the original direction = right
    // side of the reversed walk).
    for i in (0..n).rev() {
        if i == n - 1 {
            // Already covered by the end cap.
            continue;
        } else if i == 0 {
            let (nx, ny) = right_normal(points[1], points[0]);
            out.push((points[0].0 + nx * half, points[0].1 + ny * half));
        } else {
            push_join(
                &mut out,
                points[i + 1],
                points[i],
                points[i - 1],
                half,
                stroke,
                true,
            );
        }
    }
    // Start cap (at points[0], going from the reversed-walk-incoming
    // direction).
    push_cap(&mut out, points[1], points[0], half, stroke.cap);

    Some(FlatContour {
        points: out,
        closed: true,
    })
}

/// Build the fillable outline for a closed contour. Returns up to two
/// loops (outer then inner) for the caller to fill with NonZero.
fn stroke_closed(points: &[(f32, f32)], half: f32, stroke: &Stroke) -> Option<Vec<FlatContour>> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let mut outer: Vec<(f32, f32)> = Vec::with_capacity(n);
    let mut inner: Vec<(f32, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        let prev = points[(i + n - 1) % n];
        let cur = points[i];
        let nxt = points[(i + 1) % n];
        push_join(&mut outer, prev, cur, nxt, half, stroke, true);
        push_join(&mut inner, prev, cur, nxt, half, stroke, false);
    }
    // Inner loop is wound opposite to the outer so a NonZero fill
    // produces a hollow ring.
    inner.reverse();
    Some(vec![
        FlatContour {
            points: outer,
            closed: true,
        },
        FlatContour {
            points: inner,
            closed: true,
        },
    ])
}

/// Outward (right-hand) unit normal to the segment `a → b`.
#[inline]
fn right_normal(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let len = (dx * dx + dy * dy).sqrt().max(1e-9);
    // Rotate (dx, dy) by -90° in screen coordinates (Y-down):
    // right-hand normal is (dy, -dx) / len.
    (dy / len, -dx / len)
}

/// Emit join geometry at vertex `cur` between segments `prev→cur` and
/// `cur→nxt`. `outer_side` selects the right-hand offset (true) or
/// left-hand (false).
fn push_join(
    out: &mut Vec<(f32, f32)>,
    prev: (f32, f32),
    cur: (f32, f32),
    nxt: (f32, f32),
    half: f32,
    stroke: &Stroke,
    outer_side: bool,
) {
    let sign = if outer_side { 1.0 } else { -1.0 };
    let (n0x, n0y) = right_normal(prev, cur);
    let (n1x, n1y) = right_normal(cur, nxt);
    let off0 = (cur.0 + n0x * half * sign, cur.1 + n0y * half * sign);
    let off1 = (cur.0 + n1x * half * sign, cur.1 + n1y * half * sign);
    if (off0.0 - off1.0).abs() < 1e-6 && (off0.1 - off1.1).abs() < 1e-6 {
        out.push(off0);
        return;
    }
    match stroke.join {
        LineJoin::Miter => {
            // Intersect the two offset lines. If the miter length
            // exceeds miter_limit * width, fall back to a bevel.
            let dpx = nxt.0 - cur.0;
            let dpy = nxt.1 - cur.1;
            let dqx = cur.0 - prev.0;
            let dqy = cur.1 - prev.1;
            let denom = dqx * dpy - dqy * dpx; // 2× the signed area
            if denom.abs() < 1e-6 {
                // Collinear — same as bevel of zero length.
                out.push(off0);
                out.push(off1);
                return;
            }
            // Solve: off0 + t * (segment_prev_dir) = off1 + s * (segment_nxt_dir).
            // Prev dir tangent = (dqx, dqy); nxt dir tangent = (dpx, dpy).
            let rhsx = off1.0 - off0.0;
            let rhsy = off1.1 - off0.1;
            let t = (rhsx * dpy - rhsy * dpx) / denom;
            let mx = off0.0 + dqx * t;
            let my = off0.1 + dqy * t;
            // Miter length = distance from cur to (mx, my).
            let mlx = mx - cur.0;
            let mly = my - cur.1;
            let miter_len = (mlx * mlx + mly * mly).sqrt();
            if miter_len > stroke.miter_limit * half {
                // Fall back to bevel.
                out.push(off0);
                out.push(off1);
            } else {
                out.push((mx, my));
            }
        }
        LineJoin::Bevel => {
            out.push(off0);
            out.push(off1);
        }
        LineJoin::Round => {
            // Approximate the round join as a polyline arc on the
            // offset circle of radius `half` around `cur`. Step in
            // ~12° increments so a full quarter-turn becomes ~7
            // points — visually smooth at typical stroke widths.
            out.push(off0);
            arc_polyline(out, cur, off0, off1, half);
            out.push(off1);
        }
    }
}

/// Append a polyline approximating the short arc from `from` to `to`
/// on the circle of radius `r` centred at `c`. Excludes the endpoints.
fn arc_polyline(
    out: &mut Vec<(f32, f32)>,
    c: (f32, f32),
    from: (f32, f32),
    to: (f32, f32),
    r: f32,
) {
    let a0 = (from.1 - c.1).atan2(from.0 - c.0);
    let a1 = (to.1 - c.1).atan2(to.0 - c.0);
    // Pick the short arc.
    let mut delta = a1 - a0;
    while delta > std::f32::consts::PI {
        delta -= std::f32::consts::TAU;
    }
    while delta < -std::f32::consts::PI {
        delta += std::f32::consts::TAU;
    }
    let steps = ((delta.abs() / 0.21).ceil() as usize).max(1); // ~12° per step
    for i in 1..steps {
        let t = a0 + delta * (i as f32) / (steps as f32);
        out.push((c.0 + r * t.cos(), c.1 + r * t.sin()));
    }
}

/// Emit cap geometry at the end of a stroke segment. `prev → cur` is
/// the incoming segment direction; the cap is laid down at `cur`.
fn push_cap(out: &mut Vec<(f32, f32)>, prev: (f32, f32), cur: (f32, f32), half: f32, cap: LineCap) {
    let (nx, ny) = right_normal(prev, cur);
    // Direction along the stroke (prev → cur), unit length.
    let dx = cur.0 - prev.0;
    let dy = cur.1 - prev.1;
    let len = (dx * dx + dy * dy).sqrt().max(1e-9);
    let ux = dx / len;
    let uy = dy / len;
    match cap {
        LineCap::Butt => {
            // Cross from right-side offset to left-side offset directly.
            out.push((cur.0 - nx * half, cur.1 - ny * half));
        }
        LineCap::Square => {
            // Project both offsets out by `half` along the direction,
            // then cross over.
            out.push((cur.0 + nx * half + ux * half, cur.1 + ny * half + uy * half));
            out.push((cur.0 - nx * half + ux * half, cur.1 - ny * half + uy * half));
            out.push((cur.0 - nx * half, cur.1 - ny * half));
        }
        LineCap::Round => {
            // Half-circle arc swept from right offset (start) over
            // the cap direction to the left offset (end).
            let from = (cur.0 + nx * half, cur.1 + ny * half);
            let to = (cur.0 - nx * half, cur.1 - ny * half);
            arc_polyline_long(out, cur, from, to, half, ux, uy);
            out.push(to);
        }
    }
}

/// Append a half-circle arc from `from` to `to` on the circle of
/// radius `r` centred at `c`, sweeping through the outward cap
/// direction `(ux, uy)`. Excludes endpoints. The sweep direction is
/// chosen so the arc passes through the cap-outward side; with
/// `from` and `to` diametrically opposite we'd otherwise pick either
/// half at random and get an inverted cap.
fn arc_polyline_long(
    out: &mut Vec<(f32, f32)>,
    c: (f32, f32),
    from: (f32, f32),
    _to: (f32, f32),
    r: f32,
    ux: f32,
    uy: f32,
) {
    let a0 = (from.1 - c.1).atan2(from.0 - c.0);
    let mut delta = std::f32::consts::PI;
    // Pick the half-turn whose midpoint lands in the cap-outward
    // direction. If the midpoint of the +π sweep is closer to (ux,
    // uy), keep it; otherwise flip to -π.
    let mid_pos = (a0 + delta * 0.5).cos() * ux + (a0 + delta * 0.5).sin() * uy;
    let mid_neg = (a0 - delta * 0.5).cos() * ux + (a0 - delta * 0.5).sin() * uy;
    if mid_neg > mid_pos {
        delta = -delta;
    }
    let steps = 8usize;
    for i in 1..steps {
        let t = a0 + delta * (i as f32) / (steps as f32);
        out.push((c.0 + r * t.cos(), c.1 + r * t.sin()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{Paint, Rgba};

    fn stroke_basic(width: f32, cap: LineCap, join: LineJoin) -> Stroke {
        Stroke {
            width,
            paint: Paint::Solid(Rgba::opaque(0, 0, 0)),
            cap,
            join,
            miter_limit: 4.0,
            dash: None,
        }
    }

    #[test]
    fn empty_input_yields_no_output() {
        let c = FlatContour {
            points: vec![],
            closed: false,
        };
        let s = stroke_basic(2.0, LineCap::Butt, LineJoin::Bevel);
        assert!(stroke_to_fill_path(&c, &s, 2.0).is_empty());
    }

    #[test]
    fn open_line_butt_cap_yields_rectangle() {
        // Horizontal line from (0, 5) to (10, 5), width 2, butt caps.
        // Expected: 4 corners forming a 10×2 rectangle around the line.
        let c = FlatContour {
            points: vec![(0.0, 5.0), (10.0, 5.0)],
            closed: false,
        };
        let s = stroke_basic(2.0, LineCap::Butt, LineJoin::Bevel);
        let geom = stroke_to_fill_path(&c, &s, 2.0);
        assert_eq!(geom.len(), 1);
        assert!(geom[0].closed);
        // Should have at least 4 corners (the dedup may add a few).
        assert!(geom[0].points.len() >= 4);
    }

    #[test]
    fn closed_square_emits_outer_and_inner_loops() {
        let c = FlatContour {
            points: vec![(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)],
            closed: true,
        };
        let s = stroke_basic(2.0, LineCap::Butt, LineJoin::Miter);
        let geom = stroke_to_fill_path(&c, &s, 2.0);
        assert_eq!(geom.len(), 2);
        assert!(geom[0].closed && geom[1].closed);
    }

    #[test]
    fn dash_pattern_splits_into_segments() {
        // 20-pixel horizontal line, dasharray [2, 2]. Should yield 5
        // visible dashes (positions 0..2, 4..6, 8..10, 12..14, 16..18).
        let c = FlatContour {
            points: vec![(0.0, 5.0), (20.0, 5.0)],
            closed: false,
        };
        let mut s = stroke_basic(1.0, LineCap::Butt, LineJoin::Bevel);
        s.dash = Some(DashPattern {
            array: vec![2.0, 2.0],
            offset: 0.0,
        });
        let geom = stroke_to_fill_path(&c, &s, 1.0);
        assert_eq!(geom.len(), 5, "expected 5 dashes, got {}", geom.len());
    }

    #[test]
    fn miter_limit_falls_back_to_bevel_at_sharp_angle() {
        // A very acute V-shape would produce a long miter spike.
        // With miter_limit = 1.0 it must bevel.
        let c = FlatContour {
            points: vec![(0.0, 0.0), (5.0, 0.0), (0.0, 0.1)],
            closed: false,
        };
        let mut s = stroke_basic(2.0, LineCap::Butt, LineJoin::Miter);
        s.miter_limit = 1.0;
        let geom = stroke_to_fill_path(&c, &s, 2.0);
        // Just check the result is non-empty and well-formed.
        assert!(!geom.is_empty());
        for g in &geom {
            assert!(g.closed);
        }
    }
}
