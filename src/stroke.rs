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

/// The full SVG 2 §13.5.5 `stroke-linejoin` value set.
///
/// `oxideav_core::LineJoin` carries the three SVG 1.1 / PDF 1.4 values
/// (`Miter`, `Round`, `Bevel`); the two **new in SVG 2** values
/// (`miter-clip` and `arcs`) have no core enum variant, so this crate
/// models them locally — the same extension pattern
/// [`crate::FocalRadialGradient`] uses for the SVG 2 `fr` focal radius.
///
/// Drive the SVG 2 joins through [`ExtendedStroke`] +
/// [`stroke_to_fill_path_ext`]; the core [`stroke_to_fill_path`] keeps
/// taking a plain [`Stroke`] and maps its join through
/// [`ExtendedLineJoin::from_core`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtendedLineJoin {
    /// Sharp corner; collapses to a bevel past the miter limit
    /// (SVG 1.1 / SVG 2 `miter`).
    Miter,
    /// Sharp corner; past the miter limit the apex is *clipped* by a
    /// line perpendicular to the corner bisector at
    /// `miter_limit/2 · stroke-width` from the join, rather than
    /// collapsing to a bevel (SVG 2 `miter-clip`).
    MiterClip,
    /// Circular-sector corner (`round`).
    Round,
    /// Triangular bevel corner (`bevel`).
    Bevel,
    /// Arcs corner built from circles matching the incident edge
    /// curvature (SVG 2 `arcs`). On the flattened polyline the renderer
    /// feeds the stroker, both edges have zero curvature, so per the
    /// spec this falls through to [`MiterClip`](Self::MiterClip).
    Arcs,
}

impl ExtendedLineJoin {
    /// Lift a core [`LineJoin`] into the extended set (lossless: the
    /// three core values map onto their namesakes).
    #[inline]
    pub fn from_core(join: LineJoin) -> Self {
        match join {
            LineJoin::Miter => ExtendedLineJoin::Miter,
            LineJoin::Round => ExtendedLineJoin::Round,
            LineJoin::Bevel => ExtendedLineJoin::Bevel,
        }
    }
}

/// A [`Stroke`] whose join is an [`ExtendedLineJoin`], so SVG 2's
/// `miter-clip` and `arcs` corners can be requested.
///
/// Every other stroke attribute (width, cap, miter limit, dash) comes
/// from the wrapped [`Stroke`]; only [`Stroke::join`] is overridden.
/// Build one with [`ExtendedStroke::new`] or
/// [`ExtendedStroke::with_join`].
#[derive(Debug, Clone)]
pub struct ExtendedStroke {
    /// The base stroke. Its `join` field is ignored in favour of
    /// [`ExtendedStroke::join`].
    pub base: Stroke,
    /// The SVG 2 join shape.
    pub join: ExtendedLineJoin,
}

impl ExtendedStroke {
    /// Wrap `base`, taking its core join as the initial extended join.
    pub fn new(base: Stroke) -> Self {
        let join = ExtendedLineJoin::from_core(base.join);
        ExtendedStroke { base, join }
    }

    /// Wrap `base` and override the join with an [`ExtendedLineJoin`].
    pub fn with_join(base: Stroke, join: ExtendedLineJoin) -> Self {
        ExtendedStroke { base, join }
    }
}

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
    let join = ExtendedLineJoin::from_core(stroke.join);
    build_segments(segments, half, stroke, join)
}

/// SVG 2 variant of [`stroke_to_fill_path`] that honours the full
/// `stroke-linejoin` value set via [`ExtendedStroke`], including the two
/// new-in-SVG-2 joins `miter-clip` and `arcs`.
///
/// Behaviour is identical to [`stroke_to_fill_path`] for the three
/// SVG 1.1 joins; the only difference is the corner shape selected at
/// each interior vertex.
pub fn stroke_to_fill_path_ext(
    input: &FlatContour,
    stroke: &ExtendedStroke,
    width_px: f32,
) -> Vec<FlatContour> {
    if input.points.len() < 2 || width_px <= 0.0 {
        return Vec::new();
    }
    let half = width_px * 0.5;
    let base = &stroke.base;
    let segments = if let Some(dash) = &base.dash {
        apply_dash(&input.points, input.closed, dash)
    } else {
        vec![DashSegment {
            points: input.points.clone(),
            closed: input.closed,
        }]
    };
    build_segments(segments, half, base, stroke.join)
}

/// Shared outline builder for both entry points: turn dash-split
/// sub-polylines into fillable contours using the given join shape.
fn build_segments(
    segments: Vec<DashSegment>,
    half: f32,
    stroke: &Stroke,
    join: ExtendedLineJoin,
) -> Vec<FlatContour> {
    let mut out: Vec<FlatContour> = Vec::with_capacity(segments.len());
    for seg in segments {
        if seg.closed {
            // Closed input → emit outer + inner offset loops.
            if let Some(loops) = stroke_closed(&seg.points, half, stroke, join) {
                out.extend(loops);
            }
        } else if seg.points.len() >= 2 {
            if let Some(c) = stroke_open(&seg.points, half, stroke, join) {
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
/// according to `dash`.
///
/// For an open contour every output sub-polyline is open and each gets
/// its own caps.
///
/// For a **closed** contour the pattern wraps continuously around the
/// loop: SVG 1.1 §11.4 / SVG 2 §13.5.6 require the stroke (and therefore
/// the dash pattern) to start at the path's first point and progress
/// along one continuous outline. When the dash pattern is in its "on"
/// phase as it crosses the start/end seam, the trailing dash and the
/// leading dash are the *same* dash and must be rendered as one
/// continuous, *joined* piece — not two separately *capped* pieces.
/// This function detects that case and splices the leading and trailing
/// sub-polylines into a single segment that passes through the seam
/// vertex (so the caller's join logic, not a cap, is applied there).
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
    // Whether the pattern is "on" exactly at the seam (path position 0).
    // If so the first emitted sub-polyline begins at the seam vertex.
    let started_on = on;
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
    // `ended_on` records that the walk finished mid-dash at the seam:
    // the still-open `cur` reaches the loop's final point (== the start
    // vertex for a closed contour).
    let ended_on = on && cur.len() >= 2;
    if ended_on {
        out.push(DashSegment {
            points: cur,
            closed: false,
        });
    } else if !cur.is_empty() {
        // Drop a dangling single-point run.
        cur.clear();
    }

    // Closed-contour seam splice: when the dash pattern is "on" across
    // the start/end seam, the trailing dash (last segment, ending at the
    // seam vertex) and the leading dash (first segment, starting at the
    // seam vertex) are one continuous dash. Merge them so the seam
    // vertex carries a join instead of two abutting caps.
    if closed && started_on && ended_on {
        if out.len() >= 2 {
            let tail = out.pop().expect("ended_on implies a trailing segment");
            // `tail` ends at the seam vertex; `out[0]` starts at the same
            // seam vertex. Concatenate, dropping the duplicated seam point.
            let head = &mut out[0].points;
            let mut merged = tail.points;
            // The shared seam vertex is `merged.last()` == `head[0]`; skip
            // the duplicate when appending the head's interior + tail points.
            merged.extend_from_slice(&head[1..]);
            out[0].points = merged;
        } else if out.len() == 1 {
            // A single "on" run spans the whole loop (e.g. a zero-length
            // gap): it is effectively an undashed closed contour. Drop
            // the duplicated seam point and mark it closed so both offset
            // loops are emitted with joins all round.
            let seg = &mut out[0];
            if seg.points.len() >= 2 && seg.points.first() == seg.points.last() {
                seg.points.pop();
            }
            seg.closed = true;
        }
    }
    out
}

/// Build the fillable outline for an open polyline.
fn stroke_open(
    points: &[(f32, f32)],
    half: f32,
    stroke: &Stroke,
    join: ExtendedLineJoin,
) -> Option<FlatContour> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let mut out: Vec<(f32, f32)> = Vec::with_capacity(n * 4);
    let ml = stroke.miter_limit;

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
            push_join_kind(
                &mut out,
                points[i - 1],
                points[i],
                points[i + 1],
                half,
                join,
                ml,
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
            push_join_kind(
                &mut out,
                points[i + 1],
                points[i],
                points[i - 1],
                half,
                join,
                ml,
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
fn stroke_closed(
    points: &[(f32, f32)],
    half: f32,
    stroke: &Stroke,
    join: ExtendedLineJoin,
) -> Option<Vec<FlatContour>> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let ml = stroke.miter_limit;
    let mut outer: Vec<(f32, f32)> = Vec::with_capacity(n);
    let mut inner: Vec<(f32, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        let prev = points[(i + n - 1) % n];
        let cur = points[i];
        let nxt = points[(i + 1) % n];
        push_join_kind(&mut outer, prev, cur, nxt, half, join, ml, true);
        push_join_kind(&mut inner, prev, cur, nxt, half, join, ml, false);
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

/// Kind-parametrised join emitter shared by the SVG 1.1
/// ([`stroke_to_fill_path`]) and SVG 2 ([`stroke_to_fill_path_ext`])
/// entry points.
///
/// `join` selects the corner shape and `miter_limit` is the
/// `stroke-miterlimit` ratio (a multiple of `stroke-width`). `half` is
/// half the stroke width. The function appends the outer-boundary
/// vertices of the corner at `cur` for the offset side chosen by
/// `outer_side` (true = right-hand offset, false = left-hand).
///
/// The geometry follows SVG 2 §13.5.5 / the "line join shape"
/// construction in the stroke implementation notes: the bevel triangle
/// is `P, P1, P2` (here `cur`, `off0`, `off1`); the miter point `P3` is
/// the intersection of the two offset edges. The new SVG 2 values are:
///
/// * **`MiterClip`** — identical to `Miter` while
///   `1/sin(θ/2) ≤ miter_limit`. When the limit is exceeded the miter
///   region is *clipped* by a line perpendicular to the corner bisector
///   at distance `miter_limit/2 · stroke-width` from `cur`, rather than
///   collapsing all the way to a bevel. This gives a flat-topped corner
///   whose width is stable across a multi-join path.
/// * **`Arcs`** — the arcs corner is built from circles tangent to the
///   stroke edges with the same curvature as those edges at the join.
///   The curvatures are defined in user space "before any transforms
///   are applied"; the renderer flattens every curve to a polyline
///   before stroking, so both incident edges have zero curvature here.
///   Per the spec's "If both curvatures are zero fall through to
///   miter-clip" rule, `Arcs` therefore evaluates exactly as
///   `MiterClip` on polyline input. Callers that retain curvature can
///   pre-bend the incident segments; the fall-through stays
///   spec-faithful for the flattened path.
#[allow(clippy::too_many_arguments)]
fn push_join_kind(
    out: &mut Vec<(f32, f32)>,
    prev: (f32, f32),
    cur: (f32, f32),
    nxt: (f32, f32),
    half: f32,
    join: ExtendedLineJoin,
    miter_limit: f32,
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
    // `Arcs` falls through to `MiterClip` on zero-curvature polylines.
    let join = match join {
        ExtendedLineJoin::Arcs => ExtendedLineJoin::MiterClip,
        other => other,
    };
    match join {
        ExtendedLineJoin::Miter | ExtendedLineJoin::MiterClip => {
            // Intersect the two offset lines to find the miter apex P3.
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
            // Miter length = distance from cur to the apex (mx, my).
            let mlx = mx - cur.0;
            let mly = my - cur.1;
            let miter_len = (mlx * mlx + mly * mly).sqrt();
            if miter_len <= miter_limit * half {
                // Within the limit: full miter for both Miter and
                // MiterClip.
                out.push((mx, my));
            } else if join == ExtendedLineJoin::Miter {
                // `miter` collapses to a bevel when the limit is hit.
                out.push(off0);
                out.push(off1);
            } else {
                // `miter-clip`: keep the miter direction but clip the
                // apex by a line perpendicular to the corner bisector
                // at distance `miter_limit/2 · stroke-width` from `cur`.
                push_miter_clip(out, cur, off0, off1, (mx, my), miter_limit * half);
            }
        }
        ExtendedLineJoin::Bevel => {
            out.push(off0);
            out.push(off1);
        }
        ExtendedLineJoin::Round => {
            // Approximate the round join as a polyline arc on the
            // offset circle of radius `half` around `cur`. Step in
            // ~12° increments so a full quarter-turn becomes ~7
            // points — visually smooth at typical stroke widths.
            out.push(off0);
            arc_polyline(out, cur, off0, off1, half);
            out.push(off1);
        }
        ExtendedLineJoin::Arcs => unreachable!("Arcs is rewritten to MiterClip above"),
    }
}

/// Emit a clipped-miter corner (SVG 2 `stroke-linejoin: miter-clip`).
///
/// The full miter would extend from `off0` to the apex `apex` to
/// `off1`. When the miter limit is exceeded we instead clip the region
/// with a line perpendicular to the corner bisector at distance
/// `clip_dist` (= `miter_limit/2 · stroke-width`) from the join point
/// `cur`. The emitted boundary walks `off0 → c0 → c1 → off1`, where
/// `c0` and `c1` are the points where the two miter edges cross the
/// clip line. If an edge never reaches the clip line within its miter
/// span (the apex is already inside the clip distance), the
/// corresponding clip point coincides with `apex`, degrading
/// gracefully to the plain miter.
fn push_miter_clip(
    out: &mut Vec<(f32, f32)>,
    cur: (f32, f32),
    off0: (f32, f32),
    off1: (f32, f32),
    apex: (f32, f32),
    clip_dist: f32,
) {
    // Corner bisector: unit vector from `cur` toward the apex.
    let bx = apex.0 - cur.0;
    let by = apex.1 - cur.1;
    let blen = (bx * bx + by * by).sqrt();
    if blen < 1e-6 {
        out.push(off0);
        out.push(off1);
        return;
    }
    let ux = bx / blen;
    let uy = by / blen;
    // The clip line passes through `clip_pt` and is perpendicular to the
    // bisector (normal = (ux, uy)). A point Q is clipped out when its
    // signed distance along the bisector exceeds `clip_dist`.
    let clip_x = cur.0 + ux * clip_dist;
    let clip_y = cur.1 + uy * clip_dist;
    // Intersect the segment off0→apex with the clip line.
    let c0 = clip_segment(off0, apex, (clip_x, clip_y), (ux, uy));
    let c1 = clip_segment(off1, apex, (clip_x, clip_y), (ux, uy));
    out.push(off0);
    out.push(c0);
    out.push(c1);
    out.push(off1);
}

/// Clip the segment `from → to` against the half-plane bounded by the
/// line through `plane_pt` with unit `normal` (the kept side is
/// `dot(P − plane_pt, normal) ≤ 0`). Returns the boundary crossing
/// point, or `to` when the far endpoint is already inside the kept
/// half-plane (no crossing within the span — the apex stands).
fn clip_segment(
    from: (f32, f32),
    to: (f32, f32),
    plane_pt: (f32, f32),
    normal: (f32, f32),
) -> (f32, f32) {
    let d_from = (from.0 - plane_pt.0) * normal.0 + (from.1 - plane_pt.1) * normal.1;
    let d_to = (to.0 - plane_pt.0) * normal.0 + (to.1 - plane_pt.1) * normal.1;
    if d_to <= 0.0 || (d_to - d_from).abs() < 1e-9 {
        // Far endpoint inside, or the segment runs parallel to the clip
        // line: no crossing to compute, keep the far endpoint.
        return to;
    }
    // Crossing parameter where the signed distance reaches zero, clamped
    // into the segment span.
    let t = ((0.0 - d_from) / (d_to - d_from)).clamp(0.0, 1.0);
    (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t)
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

    /// Count the "on" sub-polylines produced by the dash walker for a
    /// given contour + dash, before any caps/joins are applied.
    fn dash_segment_count(c: &FlatContour, dash: &DashPattern) -> usize {
        apply_dash(&c.points, c.closed, dash).len()
    }

    #[test]
    fn closed_dash_on_at_seam_does_not_split_into_two() {
        // 10×10 closed square, perimeter 40, dasharray [10,10], offset 5.
        // The walk: on 0..5, off 5..15, on 15..25, off 25..35,
        // on 35..40 then wrapping to 0..5. The trailing on-run (35..40)
        // and the leading on-run (0..5) are the SAME dash across the
        // seam and must be merged into a single segment.
        let c = FlatContour {
            points: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            closed: true,
        };
        let dash = DashPattern {
            array: vec![10.0, 10.0],
            offset: 5.0,
        };
        // Naively this would be 3 runs (0..5, 15..25, 35..40); after the
        // seam splice the 0..5 and 35..40 runs merge → 2 runs.
        let segs = apply_dash(&c.points, c.closed, &dash);
        assert_eq!(
            segs.len(),
            2,
            "seam-crossing dash must merge: got {} segments",
            segs.len()
        );
        // Some merged segment must pass *through* the seam vertex (0,0):
        // it contains (0,0) as an interior point, not an endpoint, so the
        // caller applies a join there rather than two caps.
        let seam = (0.0_f32, 0.0_f32);
        let interior_seam = segs.iter().any(|s| {
            s.points.len() >= 3
                && s.points[1..s.points.len() - 1]
                    .iter()
                    .any(|&p| (p.0 - seam.0).abs() < 1e-4 && (p.1 - seam.1).abs() < 1e-4)
        });
        assert!(
            interior_seam,
            "seam vertex must be interior (joined), not a capped endpoint"
        );
    }

    /// True if any sub-polyline carries the seam vertex `(0,0)` as an
    /// interior point (i.e. a join, not a cap, lands there).
    fn seam_is_joined(segs: &[DashSegment]) -> bool {
        let seam = (0.0_f32, 0.0_f32);
        segs.iter().any(|s| {
            s.closed
                || (s.points.len() >= 3
                    && s.points[1..s.points.len() - 1]
                        .iter()
                        .any(|&p| (p.0 - seam.0).abs() < 1e-4 && (p.1 - seam.1).abs() < 1e-4))
        })
    }

    #[test]
    fn closed_dash_off_at_seam_is_unaffected() {
        // Same square, offset 0: on 0..10, off 10..20, on 20..30,
        // off 30..40. The seam (pos 0 / pos 40) sits between an off-run
        // ending at 40 and an on-run starting at 0 — NO merge expected.
        let c = FlatContour {
            points: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            closed: true,
        };
        let dash = DashPattern {
            array: vec![10.0, 10.0],
            offset: 0.0,
        };
        assert_eq!(dash_segment_count(&c, &dash), 2);
    }

    #[test]
    fn closed_dash_seam_joined_open_dash_capped() {
        // Same geometry + dash. For the CLOSED contour the seam dash is
        // joined (seam vertex is interior); for the OPEN contour the
        // seam vertex is necessarily a capped endpoint.
        let pts = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let dash = DashPattern {
            array: vec![10.0, 10.0],
            offset: 5.0,
        };
        let open = apply_dash(&pts, false, &dash);
        let closed = apply_dash(&pts, true, &dash);
        assert!(
            seam_is_joined(&closed),
            "closed seam dash must be joined through the start vertex"
        );
        assert!(
            !seam_is_joined(&open),
            "open contour has no seam to join across"
        );
    }

    #[test]
    fn closed_dash_run_longer_than_perimeter_is_closed_loop() {
        // A single dash longer than the whole perimeter (40) with a gap:
        // the "on" run spans the entire loop and re-enters the seam, so
        // it must be reported as one CLOSED segment (both offset loops,
        // joins all round — no caps).
        let c = FlatContour {
            points: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            closed: true,
        };
        let dash = DashPattern {
            array: vec![100.0, 10.0],
            offset: 0.0,
        };
        let segs = apply_dash(&c.points, c.closed, &dash);
        assert_eq!(segs.len(), 1, "a perimeter-spanning dash is one run");
        assert!(
            segs[0].closed,
            "a full-loop dash run must be marked closed, not open-capped"
        );
    }

    #[test]
    fn open_dash_pattern_still_splits_unchanged() {
        // Regression guard: the original open-contour behaviour is intact.
        let c = FlatContour {
            points: vec![(0.0, 5.0), (20.0, 5.0)],
            closed: false,
        };
        let dash = DashPattern {
            array: vec![2.0, 2.0],
            offset: 0.0,
        };
        assert_eq!(dash_segment_count(&c, &dash), 5);
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

    // ---- SVG 2 §13.5.5 extended joins (miter-clip / arcs) ----

    /// A right-angle corner: incoming +x, vertex at origin, outgoing +y.
    /// With half = 1 the outer offset apex sits at (1, -1) so the miter
    /// length is √2 ≈ 1.414. Returns (prev, cur, nxt).
    fn right_angle_corner() -> ((f32, f32), (f32, f32), (f32, f32)) {
        ((-10.0, 0.0), (0.0, 0.0), (0.0, 10.0))
    }

    fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
    }

    #[test]
    fn from_core_round_trips_the_three_svg11_joins() {
        assert_eq!(
            ExtendedLineJoin::from_core(LineJoin::Miter),
            ExtendedLineJoin::Miter
        );
        assert_eq!(
            ExtendedLineJoin::from_core(LineJoin::Round),
            ExtendedLineJoin::Round
        );
        assert_eq!(
            ExtendedLineJoin::from_core(LineJoin::Bevel),
            ExtendedLineJoin::Bevel
        );
    }

    #[test]
    fn miter_clip_equals_miter_within_the_limit() {
        // miter_limit = 4 → clip distance 4·half = 4 ≥ √2, so both
        // `Miter` and `MiterClip` emit the single apex vertex (1, -1).
        let (p, c, n) = right_angle_corner();
        let mut miter = Vec::new();
        push_join_kind(&mut miter, p, c, n, 1.0, ExtendedLineJoin::Miter, 4.0, true);
        let mut clip = Vec::new();
        push_join_kind(
            &mut clip,
            p,
            c,
            n,
            1.0,
            ExtendedLineJoin::MiterClip,
            4.0,
            true,
        );
        assert_eq!(miter.len(), 1, "plain miter within limit = one apex");
        assert!(dist(miter[0], (1.0, -1.0)) < 1e-4);
        assert_eq!(clip, miter, "miter-clip within the limit equals miter");
    }

    #[test]
    fn miter_clip_clips_the_apex_past_the_limit_instead_of_bevelling() {
        // Set the limit between half (1) and the apex distance √2 so the
        // limit is exceeded. clip distance = 1.2·1 = 1.2.
        let (p, c, n) = right_angle_corner();
        let limit = 1.2_f32;
        let mut miter = Vec::new();
        push_join_kind(
            &mut miter,
            p,
            c,
            n,
            1.0,
            ExtendedLineJoin::Miter,
            limit,
            true,
        );
        let mut clip = Vec::new();
        push_join_kind(
            &mut clip,
            p,
            c,
            n,
            1.0,
            ExtendedLineJoin::MiterClip,
            limit,
            true,
        );
        // `miter` collapses to a bevel: two offset vertices, no apex.
        assert_eq!(miter.len(), 2);
        assert!(dist(miter[0], (0.0, -1.0)) < 1e-4);
        assert!(dist(miter[1], (1.0, 0.0)) < 1e-4);
        // `miter-clip` keeps the miter direction but trims the apex:
        // off0 → c0 → c1 → off1 (four vertices).
        assert_eq!(clip.len(), 4);
        assert!(dist(clip[0], (0.0, -1.0)) < 1e-4, "starts at off0");
        assert!(dist(clip[3], (1.0, 0.0)) < 1e-4, "ends at off1");
        // The two clip points lie on the bisector-perpendicular line at
        // distance `clip` from the vertex, strictly nearer than the
        // √2 apex and farther than the bevel chord.
        for cp in &clip[1..3] {
            let d = dist(*cp, (0.0, 0.0));
            assert!(d < std::f32::consts::SQRT_2, "clipped nearer than apex");
            assert!(d > 1.0, "but past the offset ring (flat top, not bevel)");
        }
        // The clip line is perpendicular to the bisector: both clip
        // points share the same bisector-projection = clip distance 1.2.
        let (ux, uy) = (
            1.0_f32 / std::f32::consts::SQRT_2,
            -1.0 / std::f32::consts::SQRT_2,
        );
        for cp in &clip[1..3] {
            let proj = cp.0 * ux + cp.1 * uy;
            assert!((proj - 1.2).abs() < 1e-3, "clip point on the limit line");
        }
    }

    #[test]
    fn arcs_falls_through_to_miter_clip_on_polylines() {
        // Per the spec, `arcs` with both edge curvatures zero falls
        // through to miter-clip. Our flattened polylines always have
        // zero curvature, so the two must be byte-identical here.
        let (p, c, n) = right_angle_corner();
        for limit in [4.0_f32, 1.2, 1.0] {
            let mut clip = Vec::new();
            push_join_kind(
                &mut clip,
                p,
                c,
                n,
                1.0,
                ExtendedLineJoin::MiterClip,
                limit,
                true,
            );
            let mut arcs = Vec::new();
            push_join_kind(&mut arcs, p, c, n, 1.0, ExtendedLineJoin::Arcs, limit, true);
            assert_eq!(arcs, clip, "arcs ≡ miter-clip at miter_limit {limit}");
        }
    }

    #[test]
    fn extended_path_matches_core_path_for_shared_joins() {
        // stroke_to_fill_path_ext with a core-equivalent join must
        // reproduce stroke_to_fill_path exactly.
        let c = FlatContour {
            points: vec![(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)],
            closed: true,
        };
        for (core, ext) in [
            (LineJoin::Miter, ExtendedLineJoin::Miter),
            (LineJoin::Round, ExtendedLineJoin::Round),
            (LineJoin::Bevel, ExtendedLineJoin::Bevel),
        ] {
            let s = stroke_basic(2.0, LineCap::Butt, core);
            let base = stroke_to_fill_path(&c, &s, 2.0);
            let es = ExtendedStroke::with_join(s.clone(), ext);
            let extended = stroke_to_fill_path_ext(&c, &es, 2.0);
            assert_eq!(base.len(), extended.len());
            for (a, b) in base.iter().zip(extended.iter()) {
                assert_eq!(a.points, b.points, "{ext:?} matches {core:?}");
            }
        }
    }

    #[test]
    fn extended_stroke_new_lifts_core_join() {
        let s = stroke_basic(2.0, LineCap::Round, LineJoin::Bevel);
        let es = ExtendedStroke::new(s);
        assert_eq!(es.join, ExtendedLineJoin::Bevel);
        assert_eq!(es.base.cap, LineCap::Round);
    }

    #[test]
    fn miter_clip_produces_a_well_formed_closed_outline() {
        // Sharp corner on a real path; miter-clip must keep the outline
        // closed and non-empty (no collapse, no NaN spike).
        let c = FlatContour {
            points: vec![(0.0, 0.0), (20.0, 0.0), (0.0, 2.0)],
            closed: false,
        };
        let mut s = stroke_basic(4.0, LineCap::Butt, LineJoin::Miter);
        s.miter_limit = 2.0;
        let es = ExtendedStroke::with_join(s, ExtendedLineJoin::MiterClip);
        let geom = stroke_to_fill_path_ext(&c, &es, 4.0);
        assert!(!geom.is_empty());
        for g in &geom {
            assert!(g.closed);
            for pt in &g.points {
                assert!(pt.0.is_finite() && pt.1.is_finite());
            }
        }
    }
}
