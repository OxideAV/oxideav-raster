//! Path flattening — turn a [`Path`](oxideav_core::Path)'s command list
//! into one polyline per contour, in destination raster pixels.
//!
//! Implements:
//!
//! * line / quad / cubic Bezier de Casteljau subdivision (the standard
//!   "split + recurse until the chord length is below a tolerance"
//!   algorithm; same approach used in [`oxideav_scribe::outline`] —
//!   generalised here for arbitrary user-space paths rather than glyph
//!   outlines),
//! * SVG 1.1 Appendix F.6.5 elliptic-arc → cubic-Bezier conversion.
//!
//! All inputs are in *user space* (the path's local coordinate system);
//! the caller passes a 2D affine [`Transform2D`](oxideav_core::Transform2D)
//! that maps user space → raster space (Y-down pixels). The transform is
//! applied per-emitted-point so curves stay smooth under non-uniform
//! scale, rotation, and shear.

use oxideav_core::{PathCommand, Point, Transform2D};

/// Maximum distance, in device pixels, between a curve and the
/// polyline that replaces it.
///
/// Coverage is an area integral, so moving an edge by `d` pixels moves
/// the coverage of the pixels along it by at most `d` (of a full
/// pixel). At 1/32 px the worst pixel on a flattened curve is off by at
/// most 8 of 255 levels (typically 2–3: the chord error is a parabola
/// that peaks only at the chord midpoint), so faceting is invisible at
/// every scale — the bound is in *device* pixels, after the transform,
/// so a curve zoomed 1000× gets proportionally more segments, and a
/// tiny one gets very few.
pub(crate) const FLATTEN_TOLERANCE_PX: f64 = 1.0 / 32.0;

/// A cubic whose uniform-step segment count exceeds this is split in
/// half (de Casteljau) and each half re-measured, which makes the
/// segment density follow the local curvature and lets off-canvas
/// halves be culled.
const MAX_UNIFORM_SEGMENTS: u32 = 16;

/// Work bound for one filled curve (see [`flatten_cubic`]).
const MAX_SEGMENTS_PER_CURVE: u32 = 65_536;

/// Work bound for one stroked curve. Every chord of a stroked curve
/// becomes an outline quad plus a join, all of which overlap when the
/// pen is much wider than the curve, so the scan converter's work grows
/// faster than linearly in the chord count. 4 096 chords keep the
/// offset outline of a full-turn curve within the flattening tolerance
/// for pens up to ~10⁵ device pixels wide; beyond that the tolerance
/// scales with the pen.
const MAX_SEGMENTS_PER_STROKED_CURVE: u32 = 4_096;

/// Hard cap on subdivision depth — protects against pathological
/// control points (and bounds the work for astronomically large
/// on-canvas curves).
const MAX_SUBDIV_DEPTH: u8 = 16;

/// Internal flattening controls.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FlattenOptions {
    /// Maximum curve-to-polyline distance in device pixels.
    pub tolerance: f64,
    /// Optional device-space rectangle `(x0, y0, x1, y1)`. A curve (or
    /// sub-curve) whose control hull lies entirely beyond one side of
    /// it is replaced by its chord. For a **fill** this is exact
    /// inside the rectangle: the closed loop formed by the curve and
    /// its chord lies in the outside half-plane, so it winds zero
    /// times around every point of the rectangle. Strokes pass a
    /// rectangle grown by the stroke's reach.
    pub cull: Option<(f64, f64, f64, f64)>,
    /// Stroke half-width in device pixels (0 for fills). The offset
    /// curve of a polyline deviates from the true offset by roughly
    /// `half · δ² / 8` for a turn of `δ` radians per segment, so a
    /// non-zero value also bounds the per-segment turning angle.
    pub stroke_half: f64,
}

impl Default for FlattenOptions {
    fn default() -> Self {
        FlattenOptions {
            tolerance: FLATTEN_TOLERANCE_PX,
            cull: None,
            stroke_half: 0.0,
        }
    }
}

/// One flattened contour: a list of points forming an open or closed
/// polyline in raster (Y-down, pixel-units) coordinates.
///
/// `closed` is `true` when the source contour ended with `Close` (or
/// implicitly closed at the next `MoveTo`). The fill rasterizer uses
/// this to decide whether to emit the wrap-around edge from the last
/// point back to the first.
#[derive(Debug, Clone, Default)]
pub struct FlatContour {
    /// Raster-space points in pixel units, Y-down.
    pub points: Vec<(f32, f32)>,
    /// Whether the source contour was closed.
    pub closed: bool,
}

/// Flatten a path, applying `transform` to every point on its way
/// through. Returns one [`FlatContour`] per source subpath
/// (sub-paths are introduced by every `MoveTo` after the first).
///
/// Curves are replaced by polylines whose distance to the true curve
/// is at most 1/32 device pixel (see the module docs), with the
/// segment count derived from a rigorous bound on the curve's second
/// derivative — so the density adapts to curvature and to the zoom.
///
/// Empty paths return an empty `Vec`.
pub fn flatten_path(commands: &[PathCommand], transform: &Transform2D) -> Vec<FlatContour> {
    flatten_path_with(commands, transform, &FlattenOptions::default())
}

type P64 = (f64, f64);

/// [`flatten_path`] with explicit [`FlattenOptions`].
pub(crate) fn flatten_path_with(
    commands: &[PathCommand],
    transform: &Transform2D,
    opts: &FlattenOptions,
) -> Vec<FlatContour> {
    let mut out: Vec<FlatContour> = Vec::new();
    let mut cur: FlatContour = FlatContour::default();
    let mut pen = Point::new(0.0, 0.0);
    let mut subpath_start = Point::new(0.0, 0.0);
    let mut have_subpath = false;

    // The affine map is evaluated in f64 so a large translation that
    // cancels a large user-space coordinate keeps its fractional part.
    let (ta, tb, tc, td, te, tf) = (
        transform.a as f64,
        transform.b as f64,
        transform.c as f64,
        transform.d as f64,
        transform.e as f64,
        transform.f as f64,
    );
    let map = |p: Point| -> P64 {
        let (x, y) = (p.x as f64, p.y as f64);
        (ta * x + tc * y + te, tb * x + td * y + tf)
    };
    let to32 = |p: P64| -> (f32, f32) { (p.0 as f32, p.1 as f32) };

    for cmd in commands {
        match *cmd {
            PathCommand::MoveTo(p) => {
                // Flush any in-flight subpath before starting a new one.
                if have_subpath && cur.points.len() >= 2 {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.points.clear();
                    cur.closed = false;
                }
                pen = p;
                subpath_start = p;
                have_subpath = true;
                cur.points.push(to32(map(p)));
                cur.closed = false;
            }
            PathCommand::LineTo(p) => {
                if !have_subpath {
                    // Implicit start at (0, 0) per SVG 1.1 §8.3.2.
                    cur.points.push(to32(map(Point::new(0.0, 0.0))));
                    pen = Point::new(0.0, 0.0);
                    subpath_start = pen;
                    have_subpath = true;
                }
                cur.points.push(to32(map(p)));
                pen = p;
            }
            PathCommand::QuadCurveTo { control, end } => {
                if !have_subpath {
                    cur.points.push(to32(map(Point::new(0.0, 0.0))));
                    pen = Point::new(0.0, 0.0);
                    subpath_start = pen;
                    have_subpath = true;
                }
                // Exact degree elevation: the quadratic is the cubic
                // with controls at 2/3 of the way to its control point.
                let p0 = map(pen);
                let q = map(control);
                let p3 = map(end);
                let p1 = (
                    p0.0 + (q.0 - p0.0) * (2.0 / 3.0),
                    p0.1 + (q.1 - p0.1) * (2.0 / 3.0),
                );
                let p2 = (
                    p3.0 + (q.0 - p3.0) * (2.0 / 3.0),
                    p3.1 + (q.1 - p3.1) * (2.0 / 3.0),
                );
                flatten_cubic(
                    &mut cur.points,
                    p0,
                    p1,
                    p2,
                    p3,
                    opts,
                    0,
                    &mut curve_budget(opts),
                );
                pen = end;
            }
            PathCommand::CubicCurveTo { c1, c2, end } => {
                if !have_subpath {
                    cur.points.push(to32(map(Point::new(0.0, 0.0))));
                    pen = Point::new(0.0, 0.0);
                    subpath_start = pen;
                    have_subpath = true;
                }
                flatten_cubic(
                    &mut cur.points,
                    map(pen),
                    map(c1),
                    map(c2),
                    map(end),
                    opts,
                    0,
                    &mut curve_budget(opts),
                );
                pen = end;
            }
            PathCommand::ArcTo {
                rx,
                ry,
                x_axis_rot,
                large_arc,
                sweep,
                end,
            } => {
                if !have_subpath {
                    cur.points.push(to32(map(Point::new(0.0, 0.0))));
                    pen = Point::new(0.0, 0.0);
                    subpath_start = pen;
                    have_subpath = true;
                }
                // Arc → cubics in user space, then each cubic is
                // flattened in device space against the tolerance.
                let cubics = flatten_arc_to_cubics(pen, rx, ry, x_axis_rot, large_arc, sweep, end);
                for [c1, c2, e] in &cubics {
                    flatten_cubic(
                        &mut cur.points,
                        map(pen),
                        map(*c1),
                        map(*c2),
                        map(*e),
                        opts,
                        0,
                        &mut curve_budget(opts),
                    );
                    pen = *e;
                }
            }
            PathCommand::Close if have_subpath => {
                cur.closed = true;
                pen = subpath_start;
                if cur.points.len() == 1 {
                    // `M x y Z` is a zero-length *closed* subpath: it
                    // has no area, but SVG 2 §13.5.4 paints its round /
                    // square line cap, so keep it as a degenerate
                    // two-point contour.
                    let p = cur.points[0];
                    cur.points.push(p);
                }
                if cur.points.len() >= 2 {
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.points.clear();
                    cur.closed = false;
                }
                have_subpath = false;
            }
            // `PathCommand` is #[non_exhaustive]; future shorthand
            // variants (smooth-curve, etc.) become a no-op until
            // explicitly handled.
            _ => {}
        }
    }
    if have_subpath && cur.points.len() >= 2 {
        out.push(cur);
    }
    out
}

/// A fresh chord budget for one curve.
#[inline]
fn curve_budget(opts: &FlattenOptions) -> u32 {
    if opts.stroke_half > 0.0 {
        MAX_SEGMENTS_PER_STROKED_CURVE
    } else {
        MAX_SEGMENTS_PER_CURVE
    }
}

#[inline]
fn hypot64(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

/// Turning angle (radians, ≥ 0) between the directions `a → b` and
/// `b → c`; zero when either leg is degenerate.
#[inline]
fn turn(a: P64, b: P64, c: P64) -> f64 {
    let (ux, uy) = (b.0 - a.0, b.1 - a.1);
    let (vx, vy) = (c.0 - b.0, c.1 - b.1);
    if (ux == 0.0 && uy == 0.0) || (vx == 0.0 && vy == 0.0) {
        return 0.0;
    }
    (ux * vy - uy * vx).atan2(ux * vx + uy * vy).abs()
}

/// Flatten the cubic `p0 p1 p2 p3` (device space), appending every
/// output point after `p0` (so `p3` is always the last one pushed).
///
/// # Segment count
///
/// Split `[0, 1]` into `n` equal parameter steps and join the samples
/// with chords. On each step of length `h = 1/n` the chord is the
/// linear interpolant of `B`, whose error is bounded by
/// `h² / 8 · max |B''|` (the Green's-function bound for linear
/// interpolation; it holds for the vector-valued curve in Euclidean
/// norm). `B''` is the linear interpolation of
/// `6 (p0 − 2p1 + p2)` and `6 (p1 − 2p2 + p3)`, so with
/// `M = max(|p0 − 2p1 + p2|, |p1 − 2p2 + p3|)` the distance to the
/// curve stays below `tol` when `n ≥ √(3M / (4 tol))`.
///
/// For strokes the per-step turning angle `δ` is also bounded so the
/// stroke's offset outline (which turns with the polyline) stays within
/// `tol` of the true offset curve: the offset edge at distance `half`
/// sags by `half · δ² / 8`, giving `δ ≤ √(8 tol / half)`. The curve's
/// total turning is bounded by its control polygon's (variation
/// diminishing).
///
/// When more than [`MAX_UNIFORM_SEGMENTS`] steps would be needed the
/// curve is split at `t = ½` instead and each half re-measured: the
/// second differences shrink 4× per split, so the recursion converges
/// on the locally required density.
fn flatten_cubic(
    out: &mut Vec<(f32, f32)>,
    p0: P64,
    p1: P64,
    p2: P64,
    p3: P64,
    opts: &FlattenOptions,
    depth: u8,
    budget: &mut u32,
) {
    let finite = p0.0.is_finite()
        && p0.1.is_finite()
        && p1.0.is_finite()
        && p1.1.is_finite()
        && p2.0.is_finite()
        && p2.1.is_finite()
        && p3.0.is_finite()
        && p3.1.is_finite();
    if !finite {
        out.push((p3.0 as f32, p3.1 as f32));
        return;
    }
    if let Some((cx0, cy0, cx1, cy1)) = opts.cull {
        let min_x = p0.0.min(p1.0).min(p2.0).min(p3.0);
        let max_x = p0.0.max(p1.0).max(p2.0).max(p3.0);
        let min_y = p0.1.min(p1.1).min(p2.1).min(p3.1);
        let max_y = p0.1.max(p1.1).max(p2.1).max(p3.1);
        if max_x < cx0 || min_x > cx1 || max_y < cy0 || min_y > cy1 {
            out.push((p3.0 as f32, p3.1 as f32));
            return;
        }
    }
    let mut tol = opts.tolerance.max(1e-6);
    let d1 = hypot64(p0.0 - 2.0 * p1.0 + p2.0, p0.1 - 2.0 * p1.1 + p2.1);
    let d2 = hypot64(p1.0 - 2.0 * p2.0 + p3.0, p1.1 - 2.0 * p2.1 + p3.1);
    let m = d1.max(d2);
    // Per-step turning-angle bound for strokes (see below).
    let total_turn = if opts.stroke_half > 0.0 {
        turn(p0, p1, p2) + turn(p1, p2, p3)
            // Degenerate middle leg: measure across it.
            + if p1 == p2 { turn(p0, p1, p3) } else { 0.0 }
    } else {
        0.0
    };
    let turn_steps = |tol: f64| -> f64 {
        if opts.stroke_half > 0.0 {
            total_turn / (8.0 * tol / opts.stroke_half).sqrt()
        } else {
            0.0
        }
    };
    if depth == 0 {
        // Bound the work for one curve: a curve that would need more
        // than its chord budget at full precision (a
        // control polygon millions of pixels across with no culling
        // rectangle to discard it, or a stroke pen of astronomical
        // width) is flattened at the tolerance that budget affords —
        // both step counts scale as tol^(-1/2), and a sub-pixel
        // accuracy claim is meaningless at that scale anyway.
        let need = (0.75 * m / tol).sqrt().max(turn_steps(tol));
        let cap = *budget as f64;
        if need > cap {
            tol *= (need / cap).powi(2);
            let capped = FlattenOptions {
                tolerance: tol,
                ..*opts
            };
            flatten_cubic(out, p0, p1, p2, p3, &capped, depth + 1, budget);
            return;
        }
    }
    let mut n = (0.75 * m / tol).sqrt().max(turn_steps(tol)).ceil();
    if !n.is_finite() || n < 1.0 {
        n = 1.0;
    }
    if n > MAX_UNIFORM_SEGMENTS as f64 && depth < MAX_SUBDIV_DEPTH && *budget > MAX_UNIFORM_SEGMENTS
    {
        let q0 = mid(p0, p1);
        let q1 = mid(p1, p2);
        let q2 = mid(p2, p3);
        let r0 = mid(q0, q1);
        let r1 = mid(q1, q2);
        let s = mid(r0, r1);
        flatten_cubic(out, p0, q0, r0, s, opts, depth + 1, budget);
        flatten_cubic(out, s, r1, q2, p3, opts, depth + 1, budget);
        return;
    }
    // The analytic step counts bound the work only in exact arithmetic:
    // on a control polygon ~10¹⁹ px across, rounding noise in the
    // sub-curves' turning angles dwarfs the capped per-step angle, and
    // every leaf of the subdivision would ask for the maximum again.
    // The chord budget makes the per-curve bound unconditional.
    let n = (n as u32).min(1 << 12).min((*budget).max(1));
    *budget = budget.saturating_sub(n);
    // Power-basis form B(t) = p0 + t·a + t²·b + t³·c.
    let a = (3.0 * (p1.0 - p0.0), 3.0 * (p1.1 - p0.1));
    let b = (
        3.0 * (p0.0 - 2.0 * p1.0 + p2.0),
        3.0 * (p0.1 - 2.0 * p1.1 + p2.1),
    );
    let c = (
        p3.0 - p0.0 + 3.0 * (p1.0 - p2.0),
        p3.1 - p0.1 + 3.0 * (p1.1 - p2.1),
    );
    let inv = 1.0 / n as f64;
    for i in 1..n {
        let t = i as f64 * inv;
        let x = p0.0 + t * (a.0 + t * (b.0 + t * c.0));
        let y = p0.1 + t * (a.1 + t * (b.1 + t * c.1));
        out.push((x as f32, y as f32));
    }
    out.push((p3.0 as f32, p3.1 as f32));
}

#[inline]
fn mid(a: P64, b: P64) -> P64 {
    ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
}

/// Convert an SVG elliptic-arc segment into a sequence of cubic-Bezier
/// segments. Returns each as `[c1, c2, end]` (the start point is the
/// caller's pen position).
///
/// Implements SVG 1.1 Appendix F.6.5 + F.6.6:
///
/// 1. Endpoint → center parameterization (F.6.5).
/// 2. Split the arc at angles where each piece spans ≤ 90° so the cubic
///    approximation stays under one part in 10000 (F.6.6).
/// 3. Each ≤ 90° piece becomes one cubic Bezier using the standard
///    `α = (4/3) * tan(Δθ/4)` control-point factor.
///
/// Special cases:
/// - Identical start / end → empty arc (no cubics emitted).
/// - `rx == 0` or `ry == 0` → degenerate, emitted as a single straight
///   line (one cubic with control points on the chord).
pub fn flatten_arc_to_cubics(
    start: Point,
    rx: f32,
    ry: f32,
    x_axis_rot: f32,
    large_arc: bool,
    sweep: bool,
    end: Point,
) -> Vec<[Point; 3]> {
    // F.6.2: degenerate → straight line.
    if (start.x - end.x).abs() < 1e-6 && (start.y - end.y).abs() < 1e-6 {
        return Vec::new();
    }
    let mut rx = rx.abs();
    let mut ry = ry.abs();
    if rx < 1e-6 || ry < 1e-6 {
        // Treat as a straight line: one degenerate cubic.
        return vec![[start, end, end]];
    }

    // Step 1: F.6.5.1 — half the chord vector, rotated into the
    // ellipse's local frame.
    let (sin_phi, cos_phi) = x_axis_rot.sin_cos();
    let dx = (start.x - end.x) * 0.5;
    let dy = (start.y - end.y) * 0.5;
    let x1p = cos_phi * dx + sin_phi * dy;
    let y1p = -sin_phi * dx + cos_phi * dy;

    // Step 2: F.6.6.2 — clamp radii so the arc actually exists.
    let lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
    if lambda > 1.0 {
        let s = lambda.sqrt();
        rx *= s;
        ry *= s;
    }

    // Step 3: F.6.5.2 — center in the local frame.
    let rx2 = rx * rx;
    let ry2 = ry * ry;
    let x1p2 = x1p * x1p;
    let y1p2 = y1p * y1p;
    let denom = rx2 * y1p2 + ry2 * x1p2;
    let factor_sq = ((rx2 * ry2 - denom) / denom).max(0.0);
    let factor = factor_sq.sqrt() * if large_arc == sweep { -1.0 } else { 1.0 };
    let cxp = factor * (rx * y1p) / ry;
    let cyp = factor * -(ry * x1p) / rx;

    // Step 4: F.6.5.3 — center in user space.
    let cx = cos_phi * cxp - sin_phi * cyp + (start.x + end.x) * 0.5;
    let cy = sin_phi * cxp + cos_phi * cyp + (start.y + end.y) * 0.5;

    // Step 5: F.6.5.4-6 — angles.
    fn angle((ux, uy): (f32, f32), (vx, vy): (f32, f32)) -> f32 {
        let dot = ux * vx + uy * vy;
        let len = ((ux * ux + uy * uy) * (vx * vx + vy * vy)).sqrt();
        let mut c = dot / len.max(1e-12);
        c = c.clamp(-1.0, 1.0);
        let sign = if ux * vy - uy * vx < 0.0 { -1.0 } else { 1.0 };
        sign * c.acos()
    }

    let v1 = ((x1p - cxp) / rx, (y1p - cyp) / ry);
    let v2 = ((-x1p - cxp) / rx, (-y1p - cyp) / ry);
    let theta1 = angle((1.0, 0.0), v1);
    let mut delta = angle(v1, v2);
    if !sweep && delta > 0.0 {
        delta -= std::f32::consts::TAU;
    } else if sweep && delta < 0.0 {
        delta += std::f32::consts::TAU;
    }

    // Step 6: split into segments of ≤ 90°.
    let n_segs = ((delta.abs() / std::f32::consts::FRAC_PI_2).ceil() as usize).max(1);
    let dtheta = delta / n_segs as f32;
    let alpha = (4.0 / 3.0) * (dtheta * 0.25).tan();

    let mut out: Vec<[Point; 3]> = Vec::with_capacity(n_segs);
    let mut t1 = theta1;
    let mut p_prev = start;
    let (mut sin_t1, mut cos_t1) = t1.sin_cos();
    for _ in 0..n_segs {
        let t2 = t1 + dtheta;
        let (sin_t2, cos_t2) = t2.sin_cos();

        // End point of this sub-arc in the local-rotated frame.
        let ex_local = rx * cos_t2;
        let ey_local = ry * sin_t2;
        // Tangent at t1 (derivative of the parametric ellipse, scaled
        // by alpha to land at the cubic control distance).
        let dx1 = -rx * sin_t1 * alpha;
        let dy1 = ry * cos_t1 * alpha;
        // Tangent at t2 (negated because the control point at the end
        // sits *behind* the tangent direction).
        let dx2 = -rx * sin_t2 * alpha;
        let dy2 = ry * cos_t2 * alpha;

        // Rotate-and-translate the local-frame end point.
        let ex = cos_phi * ex_local - sin_phi * ey_local + cx;
        let ey = sin_phi * ex_local + cos_phi * ey_local + cy;
        // Control points: c1 sits one alpha-tangent ahead of p_prev,
        // c2 sits one alpha-tangent behind the new end.
        let c1x = p_prev.x + cos_phi * dx1 - sin_phi * dy1;
        let c1y = p_prev.y + sin_phi * dx1 + cos_phi * dy1;
        let c2x = ex - (cos_phi * dx2 - sin_phi * dy2);
        let c2y = ey - (sin_phi * dx2 + cos_phi * dy2);

        out.push([
            Point::new(c1x, c1y),
            Point::new(c2x, c2y),
            Point::new(ex, ey),
        ]);
        p_prev = Point::new(ex, ey);
        t1 = t2;
        sin_t1 = sin_t2;
        cos_t1 = cos_t2;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{Path, Transform2D};

    fn close_to(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn line_to_emits_two_points() {
        let mut p = Path::new();
        p.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(10.0, 0.0));
        let cs = flatten_path(&p.commands, &Transform2D::identity());
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].points.len(), 2);
        assert_eq!(cs[0].points[0], (0.0, 0.0));
        assert_eq!(cs[0].points[1], (10.0, 0.0));
        assert!(!cs[0].closed);
    }

    #[test]
    fn quadratic_curve_subdivides() {
        let mut p = Path::new();
        p.move_to(Point::new(0.0, 0.0))
            .quad_to(Point::new(50.0, 100.0), Point::new(100.0, 0.0));
        let cs = flatten_path(&p.commands, &Transform2D::identity());
        assert_eq!(cs.len(), 1);
        assert!(
            cs[0].points.len() > 5,
            "expected subdivided curve, got {}",
            cs[0].points.len()
        );
    }

    #[test]
    fn cubic_curve_subdivides() {
        let mut p = Path::new();
        p.move_to(Point::new(0.0, 0.0)).cubic_to(
            Point::new(0.0, 100.0),
            Point::new(100.0, 100.0),
            Point::new(100.0, 0.0),
        );
        let cs = flatten_path(&p.commands, &Transform2D::identity());
        assert!(cs[0].points.len() > 5);
    }

    #[test]
    fn close_marks_contour_closed() {
        let mut p = Path::new();
        p.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(10.0, 0.0))
            .line_to(Point::new(10.0, 10.0))
            .close();
        let cs = flatten_path(&p.commands, &Transform2D::identity());
        assert_eq!(cs.len(), 1);
        assert!(cs[0].closed);
    }

    #[test]
    fn multiple_subpaths_split_by_moveto() {
        let mut p = Path::new();
        p.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(10.0, 0.0));
        p.move_to(Point::new(20.0, 0.0))
            .line_to(Point::new(30.0, 0.0));
        let cs = flatten_path(&p.commands, &Transform2D::identity());
        assert_eq!(cs.len(), 2);
    }

    #[test]
    fn transform_translate_applies_to_all_points() {
        let mut p = Path::new();
        p.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(10.0, 0.0));
        let t = Transform2D::translate(5.0, 7.0);
        let cs = flatten_path(&p.commands, &t);
        assert_eq!(cs[0].points[0], (5.0, 7.0));
        assert_eq!(cs[0].points[1], (15.0, 7.0));
    }

    #[test]
    fn arc_quarter_circle_lands_at_end_point() {
        // 90° quarter-circle from (1, 0) to (0, 1) on the unit circle.
        let cubics = flatten_arc_to_cubics(
            Point::new(1.0, 0.0),
            1.0,
            1.0,
            0.0,
            false,
            true,
            Point::new(0.0, 1.0),
        );
        assert!(!cubics.is_empty());
        let last = cubics[cubics.len() - 1];
        assert!(close_to(last[2].x, 0.0, 1e-4));
        assert!(close_to(last[2].y, 1.0, 1e-4));
    }

    #[test]
    fn arc_degenerate_zero_radius_emits_line() {
        let cubics = flatten_arc_to_cubics(
            Point::new(0.0, 0.0),
            0.0,
            0.0,
            0.0,
            false,
            true,
            Point::new(10.0, 0.0),
        );
        assert_eq!(cubics.len(), 1);
        assert_eq!(cubics[0][2], Point::new(10.0, 0.0));
    }

    #[test]
    fn arc_identical_endpoints_emits_nothing() {
        let cubics = flatten_arc_to_cubics(
            Point::new(5.0, 5.0),
            1.0,
            1.0,
            0.0,
            false,
            true,
            Point::new(5.0, 5.0),
        );
        assert!(cubics.is_empty());
    }

    #[test]
    fn arc_in_path_flattens_to_polyline() {
        // Use a radius large enough that the chord-length subdivision
        // tolerance (FLATTEN_TOLERANCE_PX = 0.5 px) actually splits the
        // 90° cubic into many polyline segments. A 1-px-radius arc has
        // a ~1.4 px chord and produces only the single-cubic endpoints
        // (~3-5 points), which is geometrically correct but too few to
        // observe as a "polyline". Use radius 50 instead — a 90°
        // quarter-arc on r=50 is ~78 px long and subdivides plenty.
        let mut p = Path::new();
        p.move_to(Point::new(50.0, 0.0));
        p.commands.push(PathCommand::ArcTo {
            rx: 50.0,
            ry: 50.0,
            x_axis_rot: 0.0,
            large_arc: false,
            sweep: true,
            end: Point::new(0.0, 50.0),
        });
        let cs = flatten_path(&p.commands, &Transform2D::identity());
        assert_eq!(cs.len(), 1);
        assert!(
            cs[0].points.len() > 5,
            "expected the 90°-r=50 arc to subdivide to many points, got {}",
            cs[0].points.len()
        );
        let last = cs[0].points.last().copied().unwrap();
        assert!(close_to(last.0, 0.0, 1e-2));
        assert!(close_to(last.1, 50.0, 1e-2));
    }
}
