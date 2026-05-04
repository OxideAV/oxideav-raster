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

/// Maximum chord length (in raster pixels) we tolerate before splitting
/// a Bezier segment further. Matches `oxideav_scribe`'s default — at
/// 0.5 px the residual approximation error is below the 4× supersample
/// AA threshold so it's invisible after the box average.
const FLATTEN_TOLERANCE_PX: f32 = 0.5;

/// Hard cap on subdivision depth — protects against pathological
/// control points. 16 levels = 65536 intermediate samples, way past
/// anything sensible.
const MAX_SUBDIV_DEPTH: u8 = 16;

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
/// Empty paths return an empty `Vec`.
pub fn flatten_path(commands: &[PathCommand], transform: &Transform2D) -> Vec<FlatContour> {
    let mut out: Vec<FlatContour> = Vec::new();
    let mut cur: FlatContour = FlatContour::default();
    let mut pen = Point::new(0.0, 0.0);
    let mut subpath_start = Point::new(0.0, 0.0);
    let mut have_subpath = false;

    let map = |p: Point| -> (f32, f32) {
        let q = transform.apply(p);
        (q.x, q.y)
    };

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
                cur.points.push(map(p));
                cur.closed = false;
            }
            PathCommand::LineTo(p) => {
                if !have_subpath {
                    // Implicit start at (0, 0) per SVG 1.1 §8.3.2.
                    cur.points.push(map(Point::new(0.0, 0.0)));
                    pen = Point::new(0.0, 0.0);
                    subpath_start = pen;
                    have_subpath = true;
                }
                cur.points.push(map(p));
                pen = p;
            }
            PathCommand::QuadCurveTo { control, end } => {
                if !have_subpath {
                    cur.points.push(map(Point::new(0.0, 0.0)));
                    pen = Point::new(0.0, 0.0);
                    subpath_start = pen;
                    have_subpath = true;
                }
                let p0 = map(pen);
                let p1 = map(control);
                let p2 = map(end);
                subdivide_quad(&mut cur.points, p0, p1, p2, 0);
                pen = end;
            }
            PathCommand::CubicCurveTo { c1, c2, end } => {
                if !have_subpath {
                    cur.points.push(map(Point::new(0.0, 0.0)));
                    pen = Point::new(0.0, 0.0);
                    subpath_start = pen;
                    have_subpath = true;
                }
                let p0 = map(pen);
                let p1 = map(c1);
                let p2 = map(c2);
                let p3 = map(end);
                subdivide_cubic(&mut cur.points, p0, p1, p2, p3, 0);
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
                    cur.points.push(map(Point::new(0.0, 0.0)));
                    pen = Point::new(0.0, 0.0);
                    subpath_start = pen;
                    have_subpath = true;
                }
                // Arc → cubics in user space (so the per-cubic
                // tolerance-driven subdivision below sees the user-
                // space curvature), then map each to raster space.
                let cubics = flatten_arc_to_cubics(pen, rx, ry, x_axis_rot, large_arc, sweep, end);
                for [c1, c2, e] in &cubics {
                    let p0 = map(pen);
                    let p1 = map(*c1);
                    let p2 = map(*c2);
                    let p3 = map(*e);
                    subdivide_cubic(&mut cur.points, p0, p1, p2, p3, 0);
                    pen = *e;
                }
            }
            PathCommand::Close if have_subpath => {
                cur.closed = true;
                pen = subpath_start;
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

/// Recursively subdivide a quadratic Bezier (`p0`, `p1`, `p2`). Pushes
/// output points (excluding `p0`, including `p2`).
fn subdivide_quad(
    out: &mut Vec<(f32, f32)>,
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    depth: u8,
) {
    let dx = p2.0 - p0.0;
    let dy = p2.1 - p0.1;
    let chord_sq = dx * dx + dy * dy;
    let d_pdx = p1.0 - p0.0;
    let d_pdy = p1.1 - p0.1;
    let cross = d_pdx * dy - d_pdy * dx;
    let chord_len = chord_sq.sqrt();
    let perp = if chord_len > 1e-6 {
        (cross / chord_len).abs()
    } else {
        (d_pdx * d_pdx + d_pdy * d_pdy).sqrt()
    };

    if depth >= MAX_SUBDIV_DEPTH
        || (chord_sq <= FLATTEN_TOLERANCE_PX * FLATTEN_TOLERANCE_PX && perp <= FLATTEN_TOLERANCE_PX)
    {
        out.push(p2);
        return;
    }

    let m01 = ((p0.0 + p1.0) * 0.5, (p0.1 + p1.1) * 0.5);
    let m12 = ((p1.0 + p2.0) * 0.5, (p1.1 + p2.1) * 0.5);
    let m = ((m01.0 + m12.0) * 0.5, (m01.1 + m12.1) * 0.5);

    subdivide_quad(out, p0, m01, m, depth + 1);
    subdivide_quad(out, m, m12, p2, depth + 1);
}

/// Recursively subdivide a cubic Bezier (`p0`, `p1`, `p2`, `p3`).
fn subdivide_cubic(
    out: &mut Vec<(f32, f32)>,
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    depth: u8,
) {
    let dx = p3.0 - p0.0;
    let dy = p3.1 - p0.1;
    let chord_sq = dx * dx + dy * dy;
    let chord_len = chord_sq.sqrt();
    let perp = |p: (f32, f32)| -> f32 {
        let dpx = p.0 - p0.0;
        let dpy = p.1 - p0.1;
        let cross = dpx * dy - dpy * dx;
        if chord_len > 1e-6 {
            (cross / chord_len).abs()
        } else {
            (dpx * dpx + dpy * dpy).sqrt()
        }
    };
    let max_perp = perp(p1).max(perp(p2));

    if depth >= MAX_SUBDIV_DEPTH
        || (chord_sq <= FLATTEN_TOLERANCE_PX * FLATTEN_TOLERANCE_PX
            && max_perp <= FLATTEN_TOLERANCE_PX)
    {
        out.push(p3);
        return;
    }

    let q0 = ((p0.0 + p1.0) * 0.5, (p0.1 + p1.1) * 0.5);
    let q1 = ((p1.0 + p2.0) * 0.5, (p1.1 + p2.1) * 0.5);
    let q2 = ((p2.0 + p3.0) * 0.5, (p2.1 + p3.1) * 0.5);
    let r0 = ((q0.0 + q1.0) * 0.5, (q0.1 + q1.1) * 0.5);
    let r1 = ((q1.0 + q2.0) * 0.5, (q1.1 + q2.1) * 0.5);
    let s = ((r0.0 + r1.0) * 0.5, (r0.1 + r1.1) * 0.5);

    subdivide_cubic(out, p0, q0, r0, s, depth + 1);
    subdivide_cubic(out, s, r1, q2, p3, depth + 1);
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
