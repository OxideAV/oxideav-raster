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
    let mut mask = AlphaMask::new(width, height);
    let Some(table) = EdgeTable::build(contours, width, height, None) else {
        return (mask, None);
    };
    let w = width as usize;
    let mut bounds = BoundsAcc::default();
    table.scan(fill_rule, |y, x0, row| {
        let off = y as usize * w + x0 as usize;
        mask.data[off..off + row.len()].copy_from_slice(row);
        bounds.add(y, x0, row.len());
    });
    (mask, bounds.get())
}

/// The coverage of one painted element, stored only over the device
/// window its geometry can touch (a small shape on a 4K canvas costs a
/// mask the size of the shape, not of the canvas).
///
/// `mask` is `window`-sized; `bounds` is the inclusive device-pixel box
/// of every row range the scan emitted (every pixel outside it is 0).
#[derive(Debug, Clone)]
pub(crate) struct Coverage {
    pub mask: AlphaMask,
    pub x0: u32,
    pub y0: u32,
    pub bounds: Option<(u32, u32, u32, u32)>,
}

impl Coverage {
    /// An empty coverage (nothing painted).
    pub(crate) fn empty() -> Self {
        Coverage {
            mask: AlphaMask::default(),
            x0: 0,
            y0: 0,
            bounds: None,
        }
    }

    /// Exact-area coverage of `contours` on a `width × height` canvas.
    pub(crate) fn analytic(
        contours: &[FlatContour],
        width: u32,
        height: u32,
        fill_rule: FillRule,
    ) -> Self {
        let Some(table) = EdgeTable::build(contours, width, height, None) else {
            return Coverage::empty();
        };
        let win = table.window();
        let ww = (win.x1 - win.x0) as usize;
        let mut mask = AlphaMask::new(win.x1 - win.x0, win.y1 - win.y0);
        let mut bounds = BoundsAcc::default();
        table.scan(fill_rule, |y, x0, row| {
            let off = (y - win.y0) as usize * ww + (x0 - win.x0) as usize;
            mask.data[off..off + row.len()].copy_from_slice(row);
            bounds.add(y, x0, row.len());
        });
        Coverage {
            mask,
            x0: win.x0,
            y0: win.y0,
            bounds: bounds.get(),
        }
    }

    /// Wrap a canvas-sized mask, cropping it to `bounds`.
    pub(crate) fn from_canvas_mask(full: AlphaMask, bounds: Option<(u32, u32, u32, u32)>) -> Self {
        let Some((bx0, by0, bx1, by1)) = bounds else {
            return Coverage::empty();
        };
        let (cw, ch) = (bx1 - bx0 + 1, by1 - by0 + 1);
        let mut mask = AlphaMask::new(cw, ch);
        for y in 0..ch {
            let src = ((by0 + y) * full.width + bx0) as usize;
            let dst = (y * cw) as usize;
            mask.data[dst..dst + cw as usize].copy_from_slice(&full.data[src..src + cw as usize]);
        }
        Coverage {
            mask,
            x0: bx0,
            y0: by0,
            bounds,
        }
    }

    /// Expand to a canvas-sized mask.
    pub(crate) fn to_canvas_mask(&self, width: u32, height: u32) -> AlphaMask {
        let mut full = AlphaMask::new(width, height);
        let mw = self.mask.width as usize;
        for y in 0..self.mask.height {
            let dy = self.y0 + y;
            if dy >= height {
                break;
            }
            let n = (self.mask.width).min(width.saturating_sub(self.x0)) as usize;
            let src = y as usize * mw;
            let dst = (dy * width + self.x0) as usize;
            full.data[dst..dst + n].copy_from_slice(&self.mask.data[src..src + n]);
        }
        full
    }

    /// Multiply in (per-pixel minimum with) a canvas-sized clip mask.
    pub(crate) fn intersect_clip(mut self, clip: &AlphaMask) -> Self {
        let mw = self.mask.width as usize;
        for y in 0..self.mask.height {
            let dy = self.y0 + y;
            if dy >= clip.height {
                self.mask.data[y as usize * mw..(y as usize + 1) * mw].fill(0);
                continue;
            }
            let crow = (dy * clip.width) as usize;
            let row = &mut self.mask.data[y as usize * mw..(y as usize + 1) * mw];
            for (x, v) in row.iter_mut().enumerate() {
                let dx = self.x0 as usize + x;
                let c = if dx < clip.width as usize {
                    clip.data[crow + dx]
                } else {
                    0
                };
                *v = (*v).min(c);
            }
        }
        self
    }
}

/// Running union of touched row ranges (inclusive pixel bounds).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct BoundsAcc(Option<(u32, u32, u32, u32)>);

impl BoundsAcc {
    #[inline]
    pub(crate) fn add(&mut self, y: u32, x0: u32, len: usize) {
        if len == 0 {
            return;
        }
        let x1 = x0 + len as u32 - 1;
        self.0 = Some(match self.0 {
            None => (x0, y, x1, y),
            Some((a, b, c, d)) => (a.min(x0), b.min(y), c.max(x1), d.max(y)),
        });
    }

    #[inline]
    pub(crate) fn get(&self) -> Option<(u32, u32, u32, u32)> {
        self.0
    }
}

/// Half-open device-pixel rectangle `x0..x1 × y0..y1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Window {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl Window {
    /// The whole `width × height` canvas.
    pub(crate) fn canvas(width: u32, height: u32) -> Self {
        Window {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        }
    }

    /// Intersection, or `None` when empty.
    pub(crate) fn intersect(&self, o: &Window) -> Option<Window> {
        let w = Window {
            x0: self.x0.max(o.x0),
            y0: self.y0.max(o.y0),
            x1: self.x1.min(o.x1),
            y1: self.y1.min(o.y1),
        };
        (w.x0 < w.x1 && w.y0 < w.y1).then_some(w)
    }
}

/// An edge table clipped to a device window, in window-local
/// coordinates — the set-up half of the exact-area scan converter.
///
/// # Edge set-up in double precision
///
/// Every segment is clipped in `f64` against the window before it is
/// stored in `f32`, so the stored coordinates never exceed the window
/// size and keep ~2⁻²⁴ · size precision no matter how far off-canvas
/// the source vertices lie (an `f32` edge running from x = −10⁷ to
/// +10⁷ would otherwise carry a whole-pixel error into the canvas).
///
/// * Rows: the part of a segment above / below the window contributes
///   no area inside it and is dropped.
/// * Columns: the part of a segment left of the window is replaced by
///   its projection onto the window's left side (a vertical edge with
///   the same y-extent and direction), and likewise on the right. A
///   horizontal ray from any point inside the window crosses the
///   projection exactly when it crossed the original part, so every
///   winding number inside the window — and therefore every coverage
///   value — is unchanged.
pub(crate) struct EdgeTable {
    edges: Vec<Edge>,
    win: Window,
}

impl EdgeTable {
    /// Build the table for `contours` on a `width × height` canvas,
    /// restricted to `limit` when given. The window is shrunk to the
    /// contours' bounding box; `None` when nothing can be covered.
    pub(crate) fn build(
        contours: &[FlatContour],
        width: u32,
        height: u32,
        limit: Option<Window>,
    ) -> Option<EdgeTable> {
        if width == 0 || height == 0 {
            return None;
        }
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut count = 0usize;
        for c in contours {
            if c.points.len() < 2 {
                continue;
            }
            count += c.points.len();
            for &(x, y) in &c.points {
                if x.is_finite() && y.is_finite() {
                    let (x, y) = (x as f64, y as f64);
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }
        if !(min_x <= max_x && min_y <= max_y) {
            return None;
        }
        let clamp = |v: f64, hi: u32| -> u32 { v.clamp(0.0, hi as f64) as u32 };
        let bbox = Window {
            x0: clamp(min_x.floor(), width),
            y0: clamp(min_y.floor(), height),
            x1: clamp(max_x.ceil(), width),
            y1: clamp(max_y.ceil(), height),
        };
        let mut win = bbox.intersect(&Window::canvas(width, height))?;
        if let Some(l) = limit {
            win = win.intersect(&l)?;
        }
        let (wx0, wy0, wx1, wy1) = (win.x0 as f64, win.y0 as f64, win.x1 as f64, win.y1 as f64);

        let mut edges: Vec<Edge> = Vec::with_capacity(count + 8);
        let mut push = |xa: f64, ya: f64, xb: f64, yb: f64, winding: i32| {
            let (lya, lyb) = ((ya - wy0) as f32, (yb - wy0) as f32);
            if lya >= lyb {
                return;
            }
            let (lxa, lxb) = ((xa - wx0) as f32, (xb - wx0) as f32);
            let dxdy = ((xb - xa) / (yb - ya)) as f32;
            if !dxdy.is_finite() {
                return;
            }
            edges.push(Edge {
                y_min: lya,
                y_max: lyb,
                x_at_y_min: lxa,
                dxdy: if lxa == lxb { 0.0 } else { dxdy },
                winding,
            });
        };
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
                let winding = if y1 > y0 { 1i32 } else { -1 };
                let (mut xa, mut ya, mut xb, mut yb) = if y0 < y1 {
                    (x0 as f64, y0 as f64, x1 as f64, y1 as f64)
                } else {
                    (x1 as f64, y1 as f64, x0 as f64, y0 as f64)
                };
                if yb <= wy0 || ya >= wy1 {
                    continue;
                }
                let slope = (xb - xa) / (yb - ya);
                if ya < wy0 {
                    xa += (wy0 - ya) * slope;
                    ya = wy0;
                }
                if yb > wy1 {
                    xb -= (yb - wy1) * slope;
                    yb = wy1;
                }
                let inside = |x: f64| (wx0..=wx1).contains(&x);
                if inside(xa) && inside(xb) {
                    push(xa, ya, xb, yb, winding);
                    continue;
                }
                // Split where the segment crosses the window's left /
                // right sides; clamping each piece's endpoints then
                // projects the outside pieces onto the sides.
                let mut ts = [0.0f64, 1.0, 1.0, 1.0];
                let mut nt = 1usize;
                for side in [wx0, wx1] {
                    if (xa - side) * (xb - side) < 0.0 {
                        ts[nt] = (side - xa) / (xb - xa);
                        nt += 1;
                    }
                }
                ts[nt] = 1.0;
                ts[1..nt].sort_unstable_by(f64::total_cmp);
                for k in 0..nt {
                    let (t0, t1) = (ts[k], ts[k + 1]);
                    let pya = ya + (yb - ya) * t0;
                    let pyb = if k + 1 == nt { yb } else { ya + (yb - ya) * t1 };
                    let pxa = (xa + (xb - xa) * t0).clamp(wx0, wx1);
                    let pxb = if k + 1 == nt {
                        xb.clamp(wx0, wx1)
                    } else {
                        (xa + (xb - xa) * t1).clamp(wx0, wx1)
                    };
                    if pyb > pya {
                        push(pxa, pya, pxb, pyb, winding);
                    }
                }
            }
        }
        if edges.is_empty() {
            return None;
        }
        edges.sort_unstable_by(|a, b| a.y_min.total_cmp(&b.y_min));
        Some(EdgeTable { edges, win })
    }

    /// The device window the table was clipped to.
    #[inline]
    pub(crate) fn window(&self) -> Window {
        self.win
    }

    /// Run the exact-area scan, calling `emit(y, x0, coverage)` once
    /// per device row that received any coverage, with the 8-bit
    /// coverage of the touched columns `x0 .. x0 + coverage.len()`
    /// (device coordinates). Rows are emitted top to bottom.
    pub(crate) fn scan(&self, fill_rule: FillRule, mut emit: impl FnMut(u32, u32, &[u8])) {
        let w = (self.win.x1 - self.win.x0) as usize;
        let height = self.win.y1 - self.win.y0;
        let edges = &self.edges;
        // Per-row accumulators: `area` holds boundary-cell areas written
        // directly, `run` is a difference array for the fully-covered
        // interior runs of each span (prefix-summed at row end). Both have
        // one spare slot so a run ending at the window edge can close.
        let mut area: Vec<f32> = vec![0.0; w + 1];
        let mut run: Vec<f32> = vec![0.0; w + 1];
        let mut out_row: Vec<u8> = vec![0; w];
        let mut active: Vec<Edge> = Vec::new();
        let mut next_edge = 0usize;
        let mut events: Vec<f32> = Vec::new();
        let mut strip: Vec<StripEdge> = Vec::new();

        for py in 0..height {
            let row_top = py as f32;
            let row_bot = row_top + 1.0;

            while next_edge < edges.len() && edges[next_edge].y_min < row_bot {
                active.push(edges[next_edge]);
                next_edge += 1;
            }
            active.retain(|e| e.y_max > row_top);
            if active.is_empty() {
                if next_edge >= edges.len() {
                    break;
                }
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
            if events.len() > 2 {
                events.sort_unstable_by(f32::total_cmp);
                events.dedup();
            }

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
                sort_strip(&mut strip);
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
            let mut running = 0.0f32;
            for x in lo..=hi {
                running += run[x];
                let cov = area[x] + running;
                out_row[x] = (cov.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            area[lo..=touched.hi].fill(0.0);
            run[lo..=touched.hi].fill(0.0);
            emit(self.win.y0 + py, self.win.x0 + lo as u32, &out_row[lo..=hi]);
        }
    }
}

/// Sort a strip's edges left to right (see the call site).
///
/// The order must be a genuine total order (the standard library sorts
/// may panic otherwise), so the tolerance rule is applied in two
/// passes: sort by top `x`, then re-sort every maximal run of
/// neighbours whose top `x` values chain within [`COINCIDENT_EPS`] by
/// their bottom `x`.
#[inline]
fn sort_strip(strip: &mut [StripEdge]) {
    fn insertion_by(s: &mut [StripEdge], key: impl Fn(&StripEdge) -> f32) {
        for i in 1..s.len() {
            let mut j = i;
            while j > 0 && key(&s[j]) < key(&s[j - 1]) {
                s.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    // Strips are short and usually already ordered from the previous
    // strip, so a straight insertion sort is the fast path.
    if strip.len() <= 24 {
        insertion_by(strip, |e| e.xa);
    } else {
        strip.sort_unstable_by(|p, q| p.xa.total_cmp(&q.xa));
    }
    let n = strip.len();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && strip[j].xa - strip[j - 1].xa <= COINCIDENT_EPS {
            j += 1;
        }
        if j - i > 1 {
            if j - i <= 24 {
                insertion_by(&mut strip[i..j], |e| e.xb);
            } else {
                strip[i..j].sort_unstable_by(|p, q| p.xb.total_cmp(&q.xb));
            }
        }
        i = j;
    }
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
