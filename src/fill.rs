//! Scanline anti-aliased fill rasterizer.
//!
//! Two coverage kernels share this module:
//!
//! * [`rasterize_fill_analytic`] — **exact-area coverage** (the
//!   default of [`Renderer`](crate::Renderer)). Every pixel's alpha is
//!   the exact area of the pixel square covered by the filled region
//!   (for a non-self-overlapping outline), computed in closed form from
//!   the signed trapezoids each edge cuts out of the pixel cells it
//!   crosses. There is no sampling grid, so the number of distinct
//!   coverage levels is only limited by the 8-bit output. See the
//!   function docs for the derivation.
//! * [`rasterize_fill`] — the historical **supersampled** kernel:
//!   analytic horizontal coverage along each scanline, with 1 / 2 / 4 /
//!   8 vertical sub-scanlines per pixel row averaged down. Kept for
//!   callers who want its exact bytes; its vertical quantisation
//!   (`4·ss + 1` coverage levels on a near-horizontal edge) is what the
//!   analytic kernel removes.
//!
//! Both kernels implement the SVG 2 §13.4.1 fill rule for open
//! subpaths — "the fill operation fills open subpaths by performing the
//! fill operation as if an additional 'closepath' command were added"
//! — so an open [`FlatContour`] is closed implicitly before filling.
//! Mirrors the algorithm in `oxideav_scribe::rasterizer` (which
//! inspired the supersampled kernel) — generalised here for arbitrary
//! user-space contours and both even-odd and non-zero fill rules.
//!
//! # Supersampled algorithm
//!
//! For a given output `width × height` and a set of [`FlatContour`]s
//! in pixel space:
//!
//! 1. Build an *edge table*: one entry per non-horizontal segment with
//!    `y_min`, `y_max`, `x_at_y_min`, slope `dx/dy`, and the winding
//!    sign (+1 if `y0 < y1`, -1 otherwise — needed for the non-zero
//!    rule).
//! 2. Walk the supersampled scanline grid (`height * supersample`
//!    rows, sampled at row centre).
//! 3. Maintain an *active edge list*: edges whose `y_min..y_max` spans
//!    the current scanline.
//! 4. Sort the active list by current `x`.
//! 5. Fill spans:
//!    * Even-odd: every pair of crossings is a fill.
//!    * Non-zero: track running winding sum; non-zero stretches are
//!      filled.
//!
//!    Each filled span `[x0, x1)` contributes **fractional** horizontal
//!    coverage: interior pixels gain `1.0`, and the two boundary pixels
//!    gain the exact fraction of their area the span overlaps. This is
//!    analytic coverage along the scanline, so near-vertical edges are
//!    anti-aliased in X by the span geometry itself — the Y supersample
//!    only has to resolve the vertical direction.
//! 6. After all supersample rows for a destination row are accumulated,
//!    average them down to one 0..255 alpha byte.

use crate::flatten::FlatContour;
use oxideav_core::FillRule;

/// Which anti-aliasing coverage kernel a [`Renderer`](crate::Renderer)
/// rasterizes with.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoverageMode {
    /// Exact pixel-area coverage ([`rasterize_fill_analytic`]). No
    /// sampling grid: every edge pixel receives the closed-form area of
    /// the pixel inside the outline, so coverage is continuous up to
    /// the 8-bit quantisation and matches reference renderers that
    /// compute exact area. The default.
    #[default]
    Analytic,
    /// Vertical supersampling with analytic horizontal coverage
    /// ([`rasterize_fill`]), `Renderer::supersampling` sub-scanlines
    /// per pixel row. Kept for callers who depend on its exact bytes;
    /// a near-horizontal edge quantises to `2·ss + 1` levels.
    Supersampled,
}

/// A grayscale alpha mask. Row-major, stride = width.
#[derive(Debug, Clone, Default)]
pub struct AlphaMask {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height` bytes; each is 0..255.
    pub data: Vec<u8>,
}

impl AlphaMask {
    /// Allocate an empty (all-zero) mask of the given size.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; (width as usize) * (height as usize)],
        }
    }

    /// `true` when the mask has zero pixels.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Read the alpha at `(x, y)`. Out-of-range reads return 0.
    pub fn get(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.data[(y * self.width + x) as usize]
    }

    /// Pixel count where alpha != 0 — handy for tests.
    pub fn nonzero_pixel_count(&self) -> usize {
        self.data.iter().filter(|&&b| b != 0).count()
    }
}

/// Rasterize a set of flattened contours into an alpha mask of size
/// `(width, height)`. `supersampling` is the per-axis-Y oversampling
/// factor (1, 2, 4, 8). Anything else is clamped to the nearest valid
/// value.
pub fn rasterize_fill(
    contours: &[FlatContour],
    width: u32,
    height: u32,
    fill_rule: FillRule,
    supersampling: u8,
) -> AlphaMask {
    rasterize_fill_with_bounds(contours, width, height, fill_rule, supersampling).0
}

/// Like [`rasterize_fill`], but additionally reports the inclusive
/// destination-pixel bounding box `(min_x, min_y, max_x, max_y)` of
/// every pixel column/row any span touched, or `None` when no span was
/// emitted at all. Every pixel outside the reported box is guaranteed
/// to be `0` in the mask, so downstream compositors can restrict their
/// scan to the box without changing any output byte.
pub(crate) fn rasterize_fill_with_bounds(
    contours: &[FlatContour],
    width: u32,
    height: u32,
    fill_rule: FillRule,
    supersampling: u8,
) -> (AlphaMask, Option<(u32, u32, u32, u32)>) {
    if width == 0 || height == 0 || contours.is_empty() {
        return (AlphaMask::new(width, height), None);
    }
    let ss = match supersampling {
        0 | 1 => 1u32,
        2 => 2,
        3 | 4 => 4,
        _ => 8,
    };
    let ss_h = height.saturating_mul(ss);
    if ss_h == 0 {
        return (AlphaMask::new(width, height), None);
    }

    let mut edges: Vec<Edge> = Vec::with_capacity(contours.iter().map(|c| c.points.len()).sum());
    for c in contours {
        if c.points.len() < 2 {
            continue;
        }
        let n = c.points.len();
        // SVG 2 §13.4.1: fill implicitly closes every subpath, so the
        // closing segment is always emitted (`closed` only records
        // whether the source path spelled it out — strokes care).
        for i in 0..n {
            let (x0, y0) = c.points[i];
            let (x1, y1) = c.points[(i + 1) % n];
            if (x0 - x1).abs() < 1e-6 && (y0 - y1).abs() < 1e-6 {
                continue;
            }
            let yss0 = y0 * ss as f32;
            let yss1 = y1 * ss as f32;
            if (yss0 - yss1).abs() < 1e-6 {
                continue; // horizontal — no parity flip / winding contribution
            }
            // Winding sign: SVG's non-zero rule counts the crossing
            // direction. Fill rule's "down-going" edge contributes
            // +1 to the winding sum, "up-going" -1. In our Y-down
            // coordinate system "down" is increasing y.
            let winding = if y1 > y0 { 1i32 } else { -1 };
            let (mx0, my0, mx1, my1) = if yss0 < yss1 {
                (x0, yss0, x1, yss1)
            } else {
                (x1, yss1, x0, yss0)
            };
            let dxdy = (mx1 - mx0) / (my1 - my0);
            edges.push(Edge {
                y_min: my0,
                y_max: my1,
                x_at_y_min: mx0,
                dxdy,
                winding,
            });
        }
    }

    if edges.is_empty() {
        return (AlphaMask::new(width, height), None);
    }

    edges.sort_by(|a, b| {
        a.y_min
            .partial_cmp(&b.y_min)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Fractional coverage accumulator for ONE destination row's block
    // of `ss` supersample rows (row-major, stride = width). Each cell
    // holds the horizontal area coverage in `[0.0, 1.0]` for that
    // supersample row (boundary pixels carry partial coverage).
    //
    // Working one destination-row block at a time (instead of a full
    // `width × height × ss` grid) keeps the accumulator resident in
    // cache and makes the allocation O(width) — the per-supersample-row
    // fills and the `s = 0..ss` averaging below visit the exact same
    // values in the exact same order as the full-grid formulation, so
    // the output bytes are identical.
    let w = width as usize;
    let mut block: Vec<f32> = vec![0.0; w * (ss as usize)];

    let mut mask = AlphaMask::new(width, height);
    let inv_ss = 1.0 / ss as f32;

    let mut active: Vec<ActiveEdge> = Vec::new();
    let mut next_edge = 0usize;

    // Union of every span's touched pixel-column/row range, in
    // destination pixels (inclusive).
    let mut bounds: Option<(u32, u32, u32, u32)> = None;

    for dest_y in 0..height {
        // Touched column range within this block (inclusive), tracked
        // so the averaging + re-zeroing pass below only walks columns a
        // span actually wrote. Untouched columns hold coverage 0.0,
        // which averages to alpha 0 — the value the mask already holds.
        let mut block_lo = w;
        let mut block_hi = 0usize;

        for s in 0..ss {
            let ss_y = dest_y * ss + s;
            let y = ss_y as f32 + 0.5;

            while next_edge < edges.len() && edges[next_edge].y_min <= y {
                let e = &edges[next_edge];
                if e.y_max > y {
                    let x = e.x_at_y_min + (y - e.y_min) * e.dxdy;
                    active.push(ActiveEdge {
                        x,
                        y_max: e.y_max,
                        dxdy: e.dxdy,
                        winding: e.winding,
                    });
                }
                next_edge += 1;
            }
            active.retain(|e| e.y_max > y);

            if active.is_empty() {
                continue;
            }

            active.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

            let row = &mut block[(s as usize) * w..(s as usize + 1) * w];

            let mut touched = |range: Option<(usize, usize)>| {
                if let Some((lo, hi)) = range {
                    block_lo = block_lo.min(lo);
                    block_hi = block_hi.max(hi);
                }
            };

            match fill_rule {
                FillRule::EvenOdd => {
                    let n = active.len();
                    let mut i = 0;
                    while i + 1 < n {
                        touched(fill_span(row, active[i].x, active[i + 1].x, width));
                        i += 2;
                    }
                }
                FillRule::NonZero => {
                    // Walk the active edges left-to-right. After
                    // crossing edge `w`, the winding number on the
                    // right side of that edge is
                    // `winding + active[w].winding`. A span is filled
                    // when that running sum is non-zero.
                    let mut winding = 0i32;
                    for e in 0..active.len().saturating_sub(1) {
                        winding += active[e].winding;
                        if winding != 0 {
                            touched(fill_span(row, active[e].x, active[e + 1].x, width));
                        }
                    }
                }
            }

            for e in &mut active {
                e.x += e.dxdy;
            }
        }

        if block_lo > block_hi {
            continue; // no span touched this destination row
        }

        // Average the ss supersample rows into the destination row over
        // the touched column range, then re-zero the block for reuse.
        // Each supersample cell is already a fractional [0, 1]
        // horizontal coverage; the mean over the ss rows is the pixel's
        // combined 2D coverage.
        let mask_row = (dest_y as usize) * w;
        for x in block_lo..=block_hi {
            let mut sum = 0.0f32;
            for s in 0..ss {
                sum += block[(s as usize) * w + x];
            }
            // Mean coverage in [0, 1] → 0..255, rounded to nearest.
            let alpha = (sum * inv_ss * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            mask.data[mask_row + x] = alpha;
        }
        for s in 0..ss {
            block[(s as usize) * w + block_lo..(s as usize) * w + block_hi + 1].fill(0.0);
        }

        bounds = Some(match bounds {
            None => (block_lo as u32, dest_y, block_hi as u32, dest_y),
            Some((bx0, by0, bx1, by1)) => (
                bx0.min(block_lo as u32),
                by0.min(dest_y),
                bx1.max(block_hi as u32),
                by1.max(dest_y),
            ),
        });
    }

    (mask, bounds)
}

/// Rasterize a set of flattened contours into an alpha mask of size
/// `(width, height)` with **exact-area anti-aliasing**.
///
/// # Coverage model
///
/// The outline partitions the plane into regions of constant winding
/// number; the fill rule (SVG 2 §13.4.2) maps each region to inside or
/// outside. A pixel's coverage is the exact area of the pixel square
/// that lies in an *inside* region:
///
/// ```text
///   coverage(P) = area( P ∩ inside ) / area(P)
/// ```
///
/// There is no sampling grid in either axis, so the result is a
/// continuous function of the geometry up to the 8-bit quantisation of
/// the output byte, and it is exact for self-intersecting and
/// self-overlapping outlines too (a stroke outline is one: the
/// per-segment quads and join wedges overlap, and the union must not
/// be over-covered where two of them share an anti-aliased boundary).
///
/// # Algorithm — sparse exact scan conversion
///
/// 1. Build an edge table (one entry per non-horizontal segment, with
///    its winding direction; subpaths are implicitly closed per SVG 2
///    §13.4.1), sorted by top `y`.
/// 2. For each pixel row, gather the active edges and cut the row into
///    **sub-strips** at every edge endpoint inside the row. Inside a
///    sub-strip every active edge spans its full height, so each edge
///    is a straight segment from `x(y_top)` to `x(y_bot)`.
/// 3. Sort the strip's edges by `x`. If two neighbours swap order
///    between the top and bottom of the strip they intersect inside
///    it: the strip is split again at the lowest such crossing (and
///    re-examined), so every strip that is finally processed has a
///    consistent left-to-right edge order and therefore constant
///    winding between consecutive edges.
/// 4. Walk the ordered edges with the fill rule exactly as a classic
///    scanline filler does — even-odd pairs crossings, non-zero tracks
///    the running winding sum — producing *trapezoid spans*: regions
///    bounded above and below by the strip and left / right by two
///    straight edges.
/// 5. Integrate each trapezoid span over the pixel cells it touches in
///    closed form. For a boundary line `x(y)` sweeping linearly from
///    `x_a` to `x_b` across a strip of height `h`, the area of the
///    column `[c, c + 1)` that lies to the **left** of the line is
///
///    ```text
///      A(c) = h · mean_{x ∈ [x_a, x_b]} clamp(x − c, 0, 1)
///           = h · (F(x_b) − F(x_a)) / (x_b − x_a),   F' = clamp(· − c, 0, 1)
///      F(x) = 0                     for x ≤ c
///           = (x − c)² / 2          for c ≤ x ≤ c + 1
///           = (x − c) − 1/2         for x ≥ c + 1
///    ```
///
///    (the mean is exact because `x(y)` is linear in `y`, so `y`
///    sweeps `x` uniformly). The span's contribution to a cell is
///    `A_right(c) − A_left(c)`; cells wholly between the two edges get
///    `h`. Spans within one strip are disjoint and strips tile the row,
///    so contributions simply add; an edge shared by two adjacent spans
///    cancels exactly.
/// 6. Quantise the accumulated per-cell area to a byte (round to
///    nearest).
///
/// Degenerate input is harmless: non-finite points are skipped, a strip
/// thinner than `1e-7` px is dropped, and the crossing split is capped
/// per row so hostile geometry cannot make a row's work unbounded (past
/// the cap the strip is processed in top-ordering, the only situation
/// in which the result is an approximation).
pub fn rasterize_fill_analytic(
    contours: &[FlatContour],
    width: u32,
    height: u32,
    fill_rule: FillRule,
) -> AlphaMask {
    rasterize_fill_analytic_with_bounds(contours, width, height, fill_rule).0
}

/// Hard cap on crossing-induced strip splits in one pixel row.
const MAX_CROSSING_SPLITS_PER_ROW: usize = 512;

/// Two strip edges whose top x differ by no more than this are treated
/// as meeting at the top of the strip (ordered by their bottom x).
const COINCIDENT_EPS: f32 = 1e-4;

/// Like [`rasterize_fill_analytic`], additionally reporting the
/// inclusive touched-pixel bounding box (see
/// [`rasterize_fill_with_bounds`] for the contract).
pub(crate) fn rasterize_fill_analytic_with_bounds(
    contours: &[FlatContour],
    width: u32,
    height: u32,
    fill_rule: FillRule,
) -> (AlphaMask, Option<(u32, u32, u32, u32)>) {
    if width == 0 || height == 0 || contours.is_empty() {
        return (AlphaMask::new(width, height), None);
    }
    let w = width as usize;
    let hf = height as f32;

    // Edge table in pixel space.
    let mut edges: Vec<Edge> = Vec::with_capacity(contours.iter().map(|c| c.points.len()).sum());
    for c in contours {
        if c.points.len() < 2 {
            continue;
        }
        let n = c.points.len();
        for i in 0..n {
            let (x0, y0) = c.points[i];
            let (x1, y1) = c.points[(i + 1) % n];
            if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
                continue;
            }
            if y0 == y1 {
                continue; // horizontal: contributes no area boundary
            }
            let (top_y, bot_y) = if y0 < y1 { (y0, y1) } else { (y1, y0) };
            if bot_y <= 0.0 || top_y >= hf {
                continue;
            }
            let winding = if y1 > y0 { 1i32 } else { -1 };
            let (mx0, my0, mx1, my1) = if y0 < y1 {
                (x0, y0, x1, y1)
            } else {
                (x1, y1, x0, y0)
            };
            let dxdy = (mx1 - mx0) / (my1 - my0);
            if !dxdy.is_finite() {
                continue;
            }
            edges.push(Edge {
                y_min: my0,
                y_max: my1,
                x_at_y_min: mx0,
                dxdy,
                winding,
            });
        }
    }
    if edges.is_empty() {
        return (AlphaMask::new(width, height), None);
    }
    edges.sort_by(|a, b| {
        a.y_min
            .partial_cmp(&b.y_min)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mask = AlphaMask::new(width, height);
    // Per-row accumulators: `area` holds boundary-cell areas written
    // directly, `run` is a difference array for the fully-covered
    // interior runs of each span (prefix-summed at row end). Both have
    // one spare slot so a run ending at the canvas edge can close.
    let mut area: Vec<f32> = vec![0.0; w + 1];
    let mut run: Vec<f32> = vec![0.0; w + 1];
    let mut active: Vec<Edge> = Vec::new();
    let mut next_edge = 0usize;
    let mut events: Vec<f32> = Vec::new();
    let mut strip: Vec<StripEdge> = Vec::new();
    let mut bounds: Option<(u32, u32, u32, u32)> = None;

    for py in 0..height {
        let row_top = py as f32;
        let row_bot = row_top + 1.0;

        while next_edge < edges.len() && edges[next_edge].y_min < row_bot {
            active.push(edges[next_edge]);
            next_edge += 1;
        }
        active.retain(|e| e.y_max > row_top);
        if active.is_empty() {
            continue;
        }

        // Sub-strip boundaries: the row edges plus every edge endpoint
        // strictly inside the row.
        events.clear();
        events.push(row_top);
        events.push(row_bot);
        for e in &active {
            if e.y_min > row_top && e.y_min < row_bot {
                events.push(e.y_min);
            }
            if e.y_max > row_top && e.y_max < row_bot {
                events.push(e.y_max);
            }
        }
        events.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        events.dedup();

        let mut touched = Touched { lo: w, hi: 0 };
        let mut splits = 0usize;
        let mut i = 0usize;
        while i + 1 < events.len() {
            let ya = events[i];
            let yb = events[i + 1];
            let h = yb - ya;
            if h <= 1e-7 {
                i += 1;
                continue;
            }
            // Edges spanning this whole strip, with their x at top and
            // bottom.
            strip.clear();
            for e in &active {
                if e.y_min <= ya && e.y_max >= yb {
                    strip.push(StripEdge {
                        xa: e.x_at_y_min + (ya - e.y_min) * e.dxdy,
                        xb: e.x_at_y_min + (yb - e.y_min) * e.dxdy,
                        winding: e.winding,
                    });
                }
            }
            if strip.len() < 2 {
                i += 1;
                continue;
            }
            // Order by x at the top of the strip. Edges that meet at the
            // top (a vertex, or the crossing a previous split was made
            // at — equal up to rounding) diverge downwards, so they are
            // ordered by their bottom x instead; without the tolerance
            // rounding noise could put them in the wrong order with a
            // crossing too close to the top to split on.
            strip.sort_by(|p, q| {
                let key = if (p.xa - q.xa).abs() <= COINCIDENT_EPS {
                    p.xb.partial_cmp(&q.xb)
                } else {
                    p.xa.partial_cmp(&q.xa)
                };
                key.unwrap_or(std::cmp::Ordering::Equal)
            });
            // Neighbours that swap order between top and bottom cross
            // inside the strip: split at the highest crossing and redo.
            if splits < MAX_CROSSING_SPLITS_PER_ROW {
                let mut first_cross: Option<f32> = None;
                for pair in strip.windows(2) {
                    let (p, q) = (&pair[0], &pair[1]);
                    if p.xb > q.xb + 1e-5 {
                        let da = q.xa - p.xa; // ≥ 0 by the sort
                        let db = p.xb - q.xb; // > 0 by the test
                        let t = da / (da + db);
                        let yc = ya + t * h;
                        if yc > ya + 1e-6 && yc < yb - 1e-6 {
                            first_cross = Some(first_cross.map_or(yc, |c: f32| c.min(yc)));
                        }
                    }
                }
                if let Some(yc) = first_cross {
                    events.insert(i + 1, yc);
                    splits += 1;
                    continue;
                }
            }
            // Fill-rule span walk over the ordered strip edges.
            match fill_rule {
                FillRule::EvenOdd => {
                    let mut k = 0;
                    while k + 1 < strip.len() {
                        accumulate_trapezoid(
                            &mut area,
                            &mut run,
                            &mut touched,
                            w,
                            h,
                            &strip[k],
                            &strip[k + 1],
                        );
                        k += 2;
                    }
                }
                FillRule::NonZero => {
                    let mut winding = 0i32;
                    for k in 0..strip.len() - 1 {
                        winding += strip[k].winding;
                        if winding != 0 {
                            accumulate_trapezoid(
                                &mut area,
                                &mut run,
                                &mut touched,
                                w,
                                h,
                                &strip[k],
                                &strip[k + 1],
                            );
                        }
                    }
                }
            }
            i += 1;
        }

        if touched.lo > touched.hi {
            continue;
        }
        // `touched.hi` may be `w` (a run marker in the spare slot).
        let lo = touched.lo;
        let hi = touched.hi.min(w - 1);
        let row_off = (py as usize) * w;
        let mut running = 0.0f32;
        for x in lo..=hi {
            running += run[x];
            let cov = area[x] + running;
            mask.data[row_off + x] = (cov.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
        area[lo..=touched.hi].fill(0.0);
        run[lo..=touched.hi].fill(0.0);

        bounds = Some(match bounds {
            None => (lo as u32, py, hi as u32, py),
            Some((bx0, by0, bx1, by1)) => (
                bx0.min(lo as u32),
                by0.min(py),
                bx1.max(hi as u32),
                by1.max(py),
            ),
        });
    }

    (mask, bounds)
}

/// One active edge restricted to a sub-strip: its x at the strip's top
/// and bottom, and its winding direction.
#[derive(Debug, Clone, Copy)]
struct StripEdge {
    xa: f32,
    xb: f32,
    winding: i32,
}

/// Inclusive touched-column range of one row.
struct Touched {
    lo: usize,
    hi: usize,
}

impl Touched {
    #[inline]
    fn add(&mut self, lo: usize, hi: usize) {
        if lo < self.lo {
            self.lo = lo;
        }
        if hi > self.hi {
            self.hi = hi;
        }
    }
}

/// Antiderivative of `clamp(x − c, 0, 1)` in `x` (up to a constant).
#[inline]
fn clamp_integral(x: f32, c: f32) -> f32 {
    let u = x - c;
    if u <= 0.0 {
        0.0
    } else if u >= 1.0 {
        u - 0.5
    } else {
        0.5 * u * u
    }
}

/// Area of column `[c, c + 1)` over a strip of height `h` that lies to
/// the left of the straight line sweeping from `xa` (top) to `xb`
/// (bottom).
#[inline]
fn area_left_of_line(xa: f32, xb: f32, c: f32, h: f32) -> f32 {
    let d = xb - xa;
    if d.abs() < 1e-6 {
        h * (xa - c).clamp(0.0, 1.0)
    } else {
        h * (clamp_integral(xb, c) - clamp_integral(xa, c)) / d
    }
}

/// Add the exact per-cell area of the trapezoid bounded by `left` /
/// `right` over a strip of height `h` to the row accumulators.
#[inline]
fn accumulate_trapezoid(
    area: &mut [f32],
    run: &mut [f32],
    touched: &mut Touched,
    w: usize,
    h: f32,
    left: &StripEdge,
    right: &StripEdge,
) {
    let (lmin, lmax) = if left.xa <= left.xb {
        (left.xa, left.xb)
    } else {
        (left.xb, left.xa)
    };
    let (rmin, rmax) = if right.xa <= right.xb {
        (right.xa, right.xb)
    } else {
        (right.xb, right.xa)
    };
    let wf = w as f32;
    if rmax <= 0.0 || lmin >= wf || rmax <= lmin {
        return;
    }
    // Cells the left boundary band touches: [floor(lmin), ceil(lmax)).
    // Cells the right band touches: [floor(rmin), ceil(rmax)).
    // Cells strictly between the bands are fully covered.
    let c_first = lmin.floor().max(0.0) as usize;
    let c_last_excl = (rmax.ceil().min(wf)) as usize; // exclusive
    if c_last_excl <= c_first {
        return;
    }
    let l_band_end = (lmax.ceil().max(0.0).min(wf)) as usize; // exclusive
    let r_band_start = (rmin.floor().max(0.0).min(wf)) as usize; // inclusive

    // Interior run (fully covered cells), if the bands do not overlap.
    // The closing marker may sit one slot past the last touched cell
    // (a right edge exactly on an integer x); the touched range covers
    // it so the row reset clears it.
    if r_band_start > l_band_end {
        run[l_band_end] += h;
        run[r_band_start] -= h;
        touched.add(c_first, r_band_start.max(c_last_excl - 1));
    } else {
        touched.add(c_first, c_last_excl - 1);
    }
    // Boundary cells: everything not strictly inside the interior run.
    let mut c = c_first;
    while c < c_last_excl {
        if c >= l_band_end && c < r_band_start {
            c = r_band_start;
            continue;
        }
        let cf = c as f32;
        let a_r = if cf + 1.0 <= rmin {
            h
        } else if cf >= rmax {
            0.0
        } else {
            area_left_of_line(right.xa, right.xb, cf, h)
        };
        let a_l = if cf + 1.0 <= lmin {
            h
        } else if cf >= lmax {
            0.0
        } else {
            area_left_of_line(left.xa, left.xb, cf, h)
        };
        area[c] += a_r - a_l;
        c += 1;
    }
}

/// Add the horizontal coverage of the span `[x0, x1)` to `row`.
///
/// Interior pixels — those wholly inside the span — gain `1.0`. The two
/// boundary pixels (the ones the span partly overlaps) gain the exact
/// fraction of their unit width the span covers. This is analytic
/// scanline coverage: it anti-aliases near-vertical edges in X without
/// any horizontal supersampling. Spans within one scanline are produced
/// non-overlapping and left-to-right by the parity / winding walk, so
/// the additive accumulation never double-counts a pixel.
///
/// Returns the inclusive `(first, last)` pixel-column range the span
/// touched, or `None` when the clipped span was empty.
#[inline]
fn fill_span(row: &mut [f32], x0: f32, x1: f32, width: u32) -> Option<(usize, usize)> {
    // Clip the span to the raster width.
    let x0 = x0.max(0.0);
    let x1 = x1.min(width as f32);
    if x1 <= x0 {
        return None;
    }
    let lo = x0.floor() as usize;
    // `hi` is the index of the last pixel the span touches.
    let hi_f = x1.ceil();
    let hi = (hi_f as usize).min(width as usize);
    if hi <= lo {
        return None;
    }
    if lo + 1 == hi {
        // The whole span sits inside a single pixel column.
        row[lo] += (x1 - x0).clamp(0.0, 1.0);
        return Some((lo, lo));
    }
    // Left boundary pixel `lo`: covered from x0 to its right edge (lo+1).
    row[lo] += ((lo as f32 + 1.0) - x0).clamp(0.0, 1.0);
    // Interior pixels are fully covered.
    for px in &mut row[lo + 1..hi - 1] {
        *px += 1.0;
    }
    // Right boundary pixel `hi-1`: covered from its left edge (hi-1) to x1.
    row[hi - 1] += (x1 - (hi as f32 - 1.0)).clamp(0.0, 1.0);
    Some((lo, hi - 1))
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    y_min: f32,
    y_max: f32,
    x_at_y_min: f32,
    dxdy: f32,
    winding: i32,
}

#[derive(Debug, Clone, Copy)]
struct ActiveEdge {
    x: f32,
    y_max: f32,
    dxdy: f32,
    winding: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect_contour(x: f32, y: f32, w: f32, h: f32) -> FlatContour {
        FlatContour {
            points: vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            closed: true,
        }
    }

    #[test]
    fn empty_contours_yield_empty_mask() {
        let m = rasterize_fill(&[], 4, 4, FillRule::NonZero, 4);
        assert_eq!(m.nonzero_pixel_count(), 0);
    }

    #[test]
    fn filled_rectangle_covers_interior_pixels() {
        // Pixel-aligned 4x4 rectangle inside a 8x8 buffer.
        let c = rect_contour(2.0, 2.0, 4.0, 4.0);
        let m = rasterize_fill(&[c], 8, 8, FillRule::NonZero, 4);
        // Centre pixel should be opaque (255), corner should be empty.
        assert_eq!(m.get(3, 3), 255);
        assert_eq!(m.get(0, 0), 0);
        // 4x4 fill = 16 fully-opaque pixels.
        let opaque = m.data.iter().filter(|&&b| b == 255).count();
        assert_eq!(opaque, 16);
    }

    #[test]
    fn non_zero_vs_even_odd_donut() {
        // Outer 6x6 square wound CCW + inner 2x2 square wound CCW
        // (same direction). Even-odd fills the donut (the interior of
        // the inner square cancels the outer); non-zero fills the
        // entire outer square (windings add to 2 inside the inner,
        // which is still non-zero).
        let outer = rect_contour(1.0, 1.0, 6.0, 6.0);
        let inner = rect_contour(3.0, 3.0, 2.0, 2.0);
        let m_even = rasterize_fill(&[outer.clone(), inner.clone()], 8, 8, FillRule::EvenOdd, 4);
        let m_nz = rasterize_fill(&[outer, inner], 8, 8, FillRule::NonZero, 4);
        // Inner-square-centre pixel:
        // - even-odd: 0 (cancelled)
        // - non-zero (both CCW = same sign so they add to 2): 255
        assert_eq!(m_even.get(3, 3), 0);
        assert_eq!(m_nz.get(3, 3), 255);
    }

    #[test]
    fn supersampling_factor_is_clamped() {
        // 7 → falls into the 5+ bucket → 8.
        let c = rect_contour(0.0, 0.0, 4.0, 4.0);
        let m = rasterize_fill(&[c], 4, 4, FillRule::NonZero, 7);
        assert_eq!(m.nonzero_pixel_count(), 16);
    }

    #[test]
    fn half_pixel_span_yields_half_coverage() {
        // A 1px-tall, 0.5px-wide rectangle covering the LEFT half of
        // pixel column 0 on row 0. With analytic horizontal coverage the
        // pixel must read ~128 (half of 255), not 0 or 255. The old
        // binary fill_span rounded this to a full pixel.
        let c = rect_contour(0.0, 0.0, 0.5, 1.0);
        // supersampling = 1 isolates the *horizontal* coverage path (no
        // vertical averaging in play).
        let m = rasterize_fill(&[c], 4, 1, FillRule::NonZero, 1);
        let v = m.get(0, 0);
        assert!(
            (120..=136).contains(&v),
            "0.5px-wide span should give ~half coverage, got {v}"
        );
        // The neighbouring full-width column is untouched.
        assert_eq!(m.get(1, 0), 0);
    }

    #[test]
    fn fractional_span_boundaries_split_between_two_pixels() {
        // Rectangle from x=0.75 to x=2.25 on a single row: pixel 0 gets
        // 0.25 coverage, pixel 1 full, pixel 2 gets 0.25. Confirms the
        // boundary fractions land on the correct pixels and the interior
        // is solid.
        let c = rect_contour(0.75, 0.0, 1.5, 1.0);
        let m = rasterize_fill(&[c], 4, 1, FillRule::NonZero, 1);
        let p0 = m.get(0, 0);
        let p1 = m.get(1, 0);
        let p2 = m.get(2, 0);
        assert!((58..=70).contains(&p0), "left fraction ~64, got {p0}");
        assert_eq!(p1, 255, "interior pixel full, got {p1}");
        assert!((58..=70).contains(&p2), "right fraction ~64, got {p2}");
        assert_eq!(m.get(3, 0), 0);
    }

    #[test]
    fn adjacent_spans_sharing_a_fractional_boundary_do_not_over_cover() {
        // Two contiguous spans that meet at a fractional x (the winding
        // walk can emit a single nonzero region as two adjacent sub-spans
        // when an interior edge sits inside it). The pixel straddling the
        // shared boundary must total exactly 1.0 coverage, not more.
        let mut row = vec![0.0f32; 8];
        fill_span(&mut row, 2.3, 3.6, 8);
        fill_span(&mut row, 3.6, 5.0, 8);
        // Pixel 3 is fully inside the union [2.3, 5.0]; coverage = 1.0.
        assert!(
            (row[3] - 1.0).abs() < 1e-5,
            "shared-boundary pixel over/under-covered: {}",
            row[3]
        );
        // No pixel may ever exceed full coverage.
        for (i, &v) in row.iter().enumerate() {
            assert!(v <= 1.0 + 1e-5, "pixel {i} over-covered: {v}");
        }
        // End fractions land on the outer columns.
        assert!((row[2] - 0.7).abs() < 1e-5);
        assert!((row[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn near_vertical_edge_is_antialiased_in_x() {
        // A tall thin parallelogram whose right edge slants by one pixel
        // across its height. At a single supersample (ss=1) each row's
        // right edge lands at a fractional x, so the boundary column must
        // show a graded ramp of partial-coverage values rather than the
        // hard 0/255 step the old binary fill produced.
        let c = FlatContour {
            points: vec![(0.0, 0.0), (4.0, 0.0), (5.0, 8.0), (0.0, 8.0)],
            closed: true,
        };
        let m = rasterize_fill(&[c], 8, 8, FillRule::NonZero, 1);
        // Collect coverage values in the slanted boundary column band.
        let mut partials = 0;
        for y in 0..8 {
            for x in 4..6 {
                let v = m.get(x, y);
                if v > 0 && v < 255 {
                    partials += 1;
                }
            }
        }
        assert!(
            partials >= 4,
            "slanted edge should produce several partial-coverage pixels, \
             got {partials}"
        );
    }

    #[test]
    fn open_contour_is_implicitly_closed_for_fill() {
        // Three points forming an open "L": SVG 2 §13.4.1 fills it as
        // if a closepath joined (6,6) back to (2,2) — the triangle's
        // interior is painted, in both kernels.
        let c = FlatContour {
            points: vec![(2.0, 2.0), (6.0, 2.0), (6.0, 6.0)],
            closed: false,
        };
        let mut closed = c.clone();
        closed.closed = true;
        for (m, name) in [
            (
                rasterize_fill(std::slice::from_ref(&c), 8, 8, FillRule::NonZero, 4),
                "ss",
            ),
            (
                rasterize_fill_analytic(std::slice::from_ref(&c), 8, 8, FillRule::NonZero),
                "analytic",
            ),
        ] {
            // (5, 3) is well inside the triangle; (2, 5) is outside it.
            assert_eq!(m.get(5, 3), 255, "{name}: interior should be full");
            assert_eq!(m.get(2, 5), 0, "{name}: exterior should be empty");
        }
        // And the open contour renders byte-identically to its closed
        // spelling.
        assert_eq!(
            rasterize_fill(std::slice::from_ref(&c), 8, 8, FillRule::NonZero, 4).data,
            rasterize_fill(&[closed.clone()], 8, 8, FillRule::NonZero, 4).data
        );
        assert_eq!(
            rasterize_fill_analytic(&[c], 8, 8, FillRule::NonZero).data,
            rasterize_fill_analytic(&[closed], 8, 8, FillRule::NonZero).data
        );
    }

    // ---- exact-area kernel -------------------------------------------

    #[test]
    fn analytic_pixel_aligned_rectangle_is_exact() {
        let c = rect_contour(2.0, 2.0, 4.0, 4.0);
        let m = rasterize_fill_analytic(&[c], 8, 8, FillRule::NonZero);
        assert_eq!(m.data.iter().filter(|&&b| b == 255).count(), 16);
        assert_eq!(m.nonzero_pixel_count(), 16);
    }

    #[test]
    fn analytic_fractional_rectangle_has_exact_edge_coverage() {
        // Rect from (1.25, 1.5) to (3.75, 3.25): every boundary pixel's
        // coverage is the product of its x- and y-overlap.
        let c = rect_contour(1.25, 1.5, 2.5, 1.75);
        let m = rasterize_fill_analytic(&[c], 6, 6, FillRule::NonZero);
        let want = |x: u32, y: u32| -> u8 {
            let ox = (3.75f32.min(x as f32 + 1.0) - 1.25f32.max(x as f32)).max(0.0);
            let oy = (3.25f32.min(y as f32 + 1.0) - 1.5f32.max(y as f32)).max(0.0);
            (ox * oy * 255.0 + 0.5) as u8
        };
        for y in 0..6 {
            for x in 0..6 {
                let g = m.get(x, y);
                let e = want(x, y);
                assert!(
                    (g as i32 - e as i32).abs() <= 1,
                    "({x},{y}) got {g} want {e}"
                );
            }
        }
    }

    #[test]
    fn analytic_triangle_total_coverage_equals_its_area() {
        // Area of this triangle: ½ · 6 · 5 = 15 pixels².
        let c = FlatContour {
            points: vec![(1.3, 1.2), (7.3, 1.2), (4.1, 6.2)],
            closed: true,
        };
        let m = rasterize_fill_analytic(&[c], 10, 10, FillRule::NonZero);
        let total: f32 = m.data.iter().map(|&b| b as f32 / 255.0).sum();
        assert!((total - 15.0).abs() < 0.05, "total coverage {total}");
    }

    #[test]
    fn analytic_edge_coverage_is_not_quantised_to_sub_scanlines() {
        // A near-horizontal top edge rising 1 px over 64 px: the
        // boundary row must show a smooth ramp with (almost) every
        // distinct value, not the 5 levels a 4× supersampler yields.
        let c = FlatContour {
            points: vec![(0.0, 2.0), (64.0, 1.0), (64.0, 10.0), (0.0, 10.0)],
            closed: true,
        };
        let m = rasterize_fill_analytic(std::slice::from_ref(&c), 64, 12, FillRule::NonZero);
        let mut distinct = std::collections::BTreeSet::new();
        for x in 0..64 {
            distinct.insert(m.get(x, 1));
        }
        assert!(
            distinct.len() >= 48,
            "only {} distinct levels",
            distinct.len()
        );
        let ss = rasterize_fill(&[c], 64, 12, FillRule::NonZero, 4);
        let mut ss_distinct = std::collections::BTreeSet::new();
        for x in 0..64 {
            ss_distinct.insert(ss.get(x, 1));
        }
        assert!(
            ss_distinct.len() <= 5,
            "ss4 has {} levels",
            ss_distinct.len()
        );
    }

    #[test]
    fn analytic_matches_8x_supersampling_closely() {
        // Against the finest supersampler the two kernels must agree to
        // within the 8-sample vertical quantisation (≤ 1/16 ≈ 16 levels
        // worst case, typically far less).
        let c = FlatContour {
            points: vec![(3.3, 2.1), (27.6, 5.4), (20.2, 29.7), (6.1, 22.8)],
            closed: true,
        };
        let a = rasterize_fill_analytic(std::slice::from_ref(&c), 32, 32, FillRule::NonZero);
        let s = rasterize_fill(&[c], 32, 32, FillRule::NonZero, 8);
        let mut max_d = 0i32;
        for (pa, ps) in a.data.iter().zip(&s.data) {
            max_d = max_d.max((*pa as i32 - *ps as i32).abs());
        }
        assert!(max_d <= 17, "max deviation {max_d}");
        let ta: f32 = a.data.iter().map(|&b| b as f32).sum();
        let ts: f32 = s.data.iter().map(|&b| b as f32).sum();
        assert!((ta - ts).abs() / ts < 0.002, "totals {ta} vs {ts}");
    }

    #[test]
    fn analytic_even_odd_donut_and_non_zero_overlap() {
        let outer = rect_contour(1.0, 1.0, 6.0, 6.0);
        let inner = rect_contour(3.0, 3.0, 2.0, 2.0);
        let eo = rasterize_fill_analytic(&[outer.clone(), inner.clone()], 8, 8, FillRule::EvenOdd);
        let nz = rasterize_fill_analytic(&[outer, inner], 8, 8, FillRule::NonZero);
        assert_eq!(eo.get(3, 3), 0);
        assert_eq!(eo.get(2, 2), 255);
        assert_eq!(nz.get(3, 3), 255);
        // Winding 2 saturates at full coverage; the ring stays full.
        assert_eq!(nz.get(2, 2), 255);
        assert_eq!(nz.nonzero_pixel_count(), 36);
    }

    #[test]
    fn analytic_clips_off_canvas_geometry() {
        // A shape far wider than the canvas fills the whole row it
        // crosses (left and right excursions), and one entirely to the
        // left paints nothing.
        let wide = rect_contour(-100.0, 2.0, 1000.0, 3.0);
        let m = rasterize_fill_analytic(&[wide], 8, 8, FillRule::NonZero);
        for x in 0..8 {
            assert_eq!(m.get(x, 3), 255, "x={x}");
            assert_eq!(m.get(x, 0), 0);
        }
        let left = rect_contour(-100.0, 2.0, 50.0, 3.0);
        let m = rasterize_fill_analytic(&[left], 8, 8, FillRule::NonZero);
        assert_eq!(m.nonzero_pixel_count(), 0);
        // Right-running shape: the residual winding fills to the edge.
        let right = rect_contour(5.5, 2.0, 1000.0, 3.0);
        let m = rasterize_fill_analytic(&[right], 8, 8, FillRule::NonZero);
        assert_eq!(m.get(4, 3), 0);
        assert!((m.get(5, 3) as i32 - 128).abs() <= 1);
        assert_eq!(m.get(7, 3), 255);
        // Nothing above/below leaks.
        assert_eq!(m.get(7, 1), 0);
        assert_eq!(m.get(7, 5), 0);
    }

    #[test]
    fn analytic_bounds_cover_every_nonzero_pixel() {
        let c = FlatContour {
            points: vec![(2.2, 1.1), (14.9, 3.3), (9.0, 12.7)],
            closed: true,
        };
        let (m, b) = rasterize_fill_analytic_with_bounds(&[c], 16, 16, FillRule::NonZero);
        let (x0, y0, x1, y1) = b.expect("some coverage");
        for y in 0..16 {
            for x in 0..16 {
                if m.get(x, y) != 0 {
                    assert!(
                        x >= x0 && x <= x1 && y >= y0 && y <= y1,
                        "({x},{y}) outside bounds"
                    );
                }
            }
        }
    }

    /// Brute-force reference: the exact fill-rule insideness evaluated
    /// on a dense `n × n` point grid per pixel (SVG 2 §13.4.2 ray
    /// casting, exact winding per sample).
    fn point_sampled_coverage(
        contours: &[FlatContour],
        width: u32,
        height: u32,
        rule: FillRule,
        n: u32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; (width * height) as usize];
        let inside = |px: f32, py: f32| -> bool {
            let mut winding = 0i32;
            for c in contours {
                let m = c.points.len();
                for i in 0..m {
                    let (x0, y0) = c.points[i];
                    let (x1, y1) = c.points[(i + 1) % m];
                    if (y0 <= py) != (y1 <= py) {
                        let t = (py - y0) / (y1 - y0);
                        let x = x0 + t * (x1 - x0);
                        if x < px {
                            winding += if y1 > y0 { 1 } else { -1 };
                        }
                    }
                }
            }
            match rule {
                FillRule::NonZero => winding != 0,
                FillRule::EvenOdd => winding % 2 != 0,
            }
        };
        for y in 0..height {
            for x in 0..width {
                let mut hits = 0u32;
                for sy in 0..n {
                    for sx in 0..n {
                        let px = x as f32 + (sx as f32 + 0.5) / n as f32;
                        let py = y as f32 + (sy as f32 + 0.5) / n as f32;
                        if inside(px, py) {
                            hits += 1;
                        }
                    }
                }
                out[(y * width + x) as usize] = hits as f32 / (n * n) as f32;
            }
        }
        out
    }

    fn assert_matches_point_sampling(contours: &[FlatContour], rule: FillRule, what: &str) {
        let (w, h) = (24u32, 24u32);
        let m = rasterize_fill_analytic(contours, w, h, rule);
        let reference = point_sampled_coverage(contours, w, h, rule, 32);
        for (i, (&got, &want)) in m.data.iter().zip(&reference).enumerate() {
            let want_b = (want * 255.0).round();
            assert!(
                (got as f32 - want_b).abs() <= 4.0,
                "{what}: pixel ({}, {}) got {got}, point-sampled {want_b}",
                i as u32 % w,
                i as u32 / w
            );
        }
    }

    #[test]
    fn analytic_self_intersecting_star_matches_point_sampling() {
        // A pentagram: every crossing pixel holds regions of winding
        // 0, 1 and 2 at once, which the exact strip splitting resolves.
        let star = FlatContour {
            points: vec![
                (12.0, 1.5),
                (16.3, 20.5),
                (3.2, 8.3),
                (20.8, 8.3),
                (7.7, 20.5),
            ],
            closed: true,
        };
        assert_matches_point_sampling(
            std::slice::from_ref(&star),
            FillRule::EvenOdd,
            "star even-odd",
        );
        assert_matches_point_sampling(&[star], FillRule::NonZero, "star non-zero");
    }

    #[test]
    fn analytic_overlapping_outlines_do_not_over_cover_shared_edges() {
        // Two rotated rectangles overlapping like consecutive stroke
        // quads at a join: the union's anti-aliased boundary must carry
        // the union's area, not the sum of both pieces' areas.
        let quad = |cx: f32, cy: f32, ang: f32, len: f32, wid: f32| {
            let (s, c) = ang.sin_cos();
            let (ux, uy) = (c * len / 2.0, s * len / 2.0);
            let (vx, vy) = (-s * wid / 2.0, c * wid / 2.0);
            FlatContour {
                points: vec![
                    (cx - ux - vx, cy - uy - vy),
                    (cx + ux - vx, cy + uy - vy),
                    (cx + ux + vx, cy + uy + vy),
                    (cx - ux + vx, cy - uy + vy),
                ],
                closed: true,
            }
        };
        let a = quad(9.0, 12.3, 0.3, 14.0, 3.7);
        let b = quad(15.0, 12.3, 0.3, 14.0, 3.7); // overlaps `a` by 8 px along the axis
        assert_matches_point_sampling(&[a, b], FillRule::NonZero, "overlapping quads");
    }

    #[test]
    fn analytic_random_polygons_match_point_sampling() {
        // Deterministic LCG polygons (self-intersecting, partly
        // off-canvas) under both rules.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move |lo: f32, hi: f32| -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((seed >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
        };
        for case in 0..12 {
            let n = 3 + (case % 5);
            let pts: Vec<(f32, f32)> = (0..n)
                .map(|_| (next(-4.0, 28.0), next(-4.0, 28.0)))
                .collect();
            let c = FlatContour {
                points: pts,
                closed: true,
            };
            for rule in [FillRule::NonZero, FillRule::EvenOdd] {
                assert_matches_point_sampling(
                    std::slice::from_ref(&c),
                    rule,
                    &format!("case {case} {rule:?}"),
                );
            }
        }
    }

    #[test]
    fn analytic_non_finite_and_degenerate_inputs_are_harmless() {
        let bad = FlatContour {
            points: vec![(f32::NAN, 0.0), (4.0, f32::INFINITY), (2.0, 2.0)],
            closed: true,
        };
        let m = rasterize_fill_analytic(&[bad], 8, 8, FillRule::NonZero);
        assert_eq!(m.width, 8);
        let m = rasterize_fill_analytic(&[], 0, 0, FillRule::NonZero);
        assert!(m.is_empty());
        let one = FlatContour {
            points: vec![(1.0, 1.0)],
            closed: true,
        };
        assert_eq!(
            rasterize_fill_analytic(&[one], 4, 4, FillRule::EvenOdd).nonzero_pixel_count(),
            0
        );
    }
}
