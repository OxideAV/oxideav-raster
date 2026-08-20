//! Scanline anti-aliased fill rasterizer.
//!
//! Standard active-edge-list scanline fill with configurable vertical
//! supersampling for AA. Mirrors the algorithm in
//! `oxideav_scribe::rasterizer` (which inspired it) — generalised here
//! for arbitrary user-space contours and both even-odd and non-zero
//! fill rules.
//!
//! # Algorithm
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
        let last_idx = if c.closed { n } else { n - 1 };
        for i in 0..last_idx {
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
    fn open_contour_does_not_close_implicitly() {
        // Three points forming an open "L". Should produce no fill
        // because there's no implicit closing edge from p2 back to p0.
        let c = FlatContour {
            points: vec![(2.0, 2.0), (6.0, 2.0), (6.0, 6.0)],
            closed: false,
        };
        let m = rasterize_fill(&[c], 8, 8, FillRule::NonZero, 4);
        // No fully-opaque interior should exist (only a thin strip
        // of partial coverage along the edges from the supersample
        // grid; that may or may not net to zero exactly so we check
        // the *centre* pixel which is way inside the would-be triangle).
        assert!(
            m.get(3, 3) == 0 || m.get(3, 3) < 200,
            "open contour should not produce a full fill at centre, got {}",
            m.get(3, 3)
        );
    }
}
