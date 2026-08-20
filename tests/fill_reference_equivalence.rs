//! Differential gate for the scanline fill rasteriser.
//!
//! `rasterize_fill` processes one destination-row block of supersample
//! rows at a time (r449) instead of materialising the full
//! `width × height × ss` coverage grid. This test pins the optimised
//! kernel byte-identical to the straightforward full-grid reference
//! formulation below on randomised geometry — every accumulation and
//! averaging step is performed in the same order on the same values,
//! so equality must be exact, not approximate.

use oxideav_core::FillRule;
use oxideav_raster::{rasterize_fill, AlphaMask, FlatContour};

/// The full-grid reference formulation: identical edge table, active
/// edge list, span walk, and averaging arithmetic — but accumulating
/// into one `width × height × ss` coverage grid and averaging it in a
/// second pass, the way the pre-r449 kernel did.
fn rasterize_fill_reference(
    contours: &[FlatContour],
    width: u32,
    height: u32,
    fill_rule: FillRule,
    supersampling: u8,
) -> AlphaMask {
    #[derive(Clone, Copy)]
    struct Edge {
        y_min: f32,
        y_max: f32,
        x_at_y_min: f32,
        dxdy: f32,
        winding: i32,
    }
    #[derive(Clone, Copy)]
    struct ActiveEdge {
        x: f32,
        y_max: f32,
        dxdy: f32,
        winding: i32,
    }
    fn fill_span(row: &mut [f32], x0: f32, x1: f32, width: u32) {
        let x0 = x0.max(0.0);
        let x1 = x1.min(width as f32);
        if x1 <= x0 {
            return;
        }
        let lo = x0.floor() as usize;
        let hi = (x1.ceil() as usize).min(width as usize);
        if hi <= lo {
            return;
        }
        if lo + 1 == hi {
            row[lo] += (x1 - x0).clamp(0.0, 1.0);
            return;
        }
        row[lo] += ((lo as f32 + 1.0) - x0).clamp(0.0, 1.0);
        for px in &mut row[lo + 1..hi - 1] {
            *px += 1.0;
        }
        row[hi - 1] += (x1 - (hi as f32 - 1.0)).clamp(0.0, 1.0);
    }

    let mut mask = AlphaMask::new(width, height);
    if width == 0 || height == 0 || contours.is_empty() {
        return mask;
    }
    let ss = match supersampling {
        0 | 1 => 1u32,
        2 => 2,
        3 | 4 => 4,
        _ => 8,
    };
    let ss_h = height.saturating_mul(ss);
    if ss_h == 0 {
        return mask;
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
                continue;
            }
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
        return mask;
    }
    edges.sort_by(|a, b| {
        a.y_min
            .partial_cmp(&b.y_min)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut coverage: Vec<f32> = vec![0.0; (width as usize) * (ss_h as usize)];
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
    let inv_ss = 1.0 / ss as f32;
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for s in 0..ss {
                let row = (y * ss + s) as usize;
                let idx = row * (width as usize) + (x as usize);
                sum += coverage[idx];
            }
            let alpha = (sum * inv_ss * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            mask.data[(y * width + x) as usize] = alpha;
        }
    }
    mask
}

/// Simple LCG for deterministic pseudo-random geometry.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / ((1u64 << 24) as f32);
        lo + unit * (hi - lo)
    }
    fn next_usize(&mut self, lo: usize, hi: usize) -> usize {
        self.next_f32(lo as f32, hi as f32).floor() as usize
    }
}

/// Random star-ish closed blob around a centre, with occasional
/// off-canvas excursions so clipping paths get exercised too.
fn random_blob(rng: &mut Lcg, w: f32, h: f32) -> FlatContour {
    let cx = rng.next_f32(-10.0, w + 10.0);
    let cy = rng.next_f32(-10.0, h + 10.0);
    let n = rng.next_usize(3, 24);
    let base_r = rng.next_f32(0.5, w.max(h) * 0.6);
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let a = std::f32::consts::TAU * (i as f32) / (n as f32);
        let r = base_r * rng.next_f32(0.3, 1.0);
        points.push((cx + r * a.cos(), cy + r * a.sin()));
    }
    FlatContour {
        points,
        closed: rng.next_f32(0.0, 1.0) > 0.15,
    }
}

#[test]
fn optimised_fill_matches_reference_bytes_exactly() {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    for case in 0..200 {
        let width = rng.next_usize(1, 96) as u32;
        let height = rng.next_usize(1, 96) as u32;
        let n_contours = rng.next_usize(1, 5);
        let contours: Vec<FlatContour> = (0..n_contours)
            .map(|_| random_blob(&mut rng, width as f32, height as f32))
            .collect();
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            for ss in [1u8, 2, 4, 8] {
                let got = rasterize_fill(&contours, width, height, rule, ss);
                let want = rasterize_fill_reference(&contours, width, height, rule, ss);
                assert_eq!(
                    got.data, want.data,
                    "case {case}: {width}x{height} rule {rule:?} ss {ss} diverged"
                );
            }
        }
    }
}

#[test]
fn degenerate_inputs_match_reference() {
    // Empty contour list, zero-sized canvas, sub-2-point contours.
    let empty: Vec<FlatContour> = Vec::new();
    let tiny = vec![FlatContour {
        points: vec![(1.0, 1.0)],
        closed: true,
    }];
    for (contours, w, h) in [(&empty, 8u32, 8u32), (&tiny, 8, 8), (&empty, 0, 4)] {
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            let got = rasterize_fill(contours, w, h, rule, 4);
            let want = rasterize_fill_reference(contours, w, h, rule, 4);
            assert_eq!(got.data, want.data);
        }
    }
}
