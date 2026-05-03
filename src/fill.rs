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
    if width == 0 || height == 0 || contours.is_empty() {
        return AlphaMask::new(width, height);
    }
    let ss = match supersampling {
        0 | 1 => 1u32,
        2 => 2,
        3 | 4 => 4,
        _ => 8,
    };
    let ss_h = height.saturating_mul(ss);
    if ss_h == 0 {
        return AlphaMask::new(width, height);
    }

    let mut edges: Vec<Edge> = Vec::new();
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
        return AlphaMask::new(width, height);
    }

    edges.sort_by(|a, b| {
        a.y_min
            .partial_cmp(&b.y_min)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // One-bit-per-pixel coverage for the supersampled grid.
    let mut coverage: Vec<u8> = vec![0; (width as usize) * (ss_h as usize)];

    let mut active: Vec<ActiveEdge> = Vec::new();
    let mut next_edge = 0usize;

    for ss_y in 0..ss_h {
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

        let row = &mut coverage
            [(ss_y as usize) * (width as usize)..(ss_y as usize + 1) * (width as usize)];

        match fill_rule {
            FillRule::EvenOdd => {
                let n = active.len();
                let mut i = 0;
                while i + 1 < n {
                    fill_span(row, active[i].x, active[i + 1].x, width);
                    i += 2;
                }
            }
            FillRule::NonZero => {
                // Walk the active edges left-to-right. After crossing
                // edge `w`, the winding number on the right side of
                // that edge is `winding + active[w].winding`. A span
                // is filled when that running sum is non-zero.
                let mut winding = 0i32;
                for w in 0..active.len().saturating_sub(1) {
                    winding += active[w].winding;
                    if winding != 0 {
                        fill_span(row, active[w].x, active[w + 1].x, width);
                    }
                }
            }
        }

        for e in &mut active {
            e.x += e.dxdy;
        }
    }

    // Average ss rows into one destination row.
    let mut mask = AlphaMask::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0u32;
            for s in 0..ss {
                let row = (y * ss + s) as usize;
                let idx = row * (width as usize) + (x as usize);
                sum += coverage[idx] as u32;
            }
            let alpha = (sum * 255 + (ss / 2)) / ss;
            mask.data[(y * width + x) as usize] = alpha.min(255) as u8;
        }
    }
    mask
}

#[inline]
fn fill_span(row: &mut [u8], x0: f32, x1: f32, width: u32) {
    let lo = x0.max(0.0).floor() as i64;
    let hi = x1.min(width as f32).ceil() as i64;
    if hi <= lo {
        return;
    }
    let lo = lo.max(0) as usize;
    let hi = (hi as usize).min(width as usize);
    if hi <= lo {
        return;
    }
    for px in &mut row[lo..hi] {
        *px = 1;
    }
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
