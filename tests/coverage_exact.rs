//! Anti-aliasing accuracy of the exact-area kernel against a closed-form
//! ground truth, on the shapes where sampling kernels go wrong: slivers
//! thinner than a pixel, edges within a hair of horizontal or vertical,
//! shapes far smaller than a pixel, vertices on pixel corners, and the
//! same shapes far from the origin.
//!
//! The reference is not a supersampler: for a simple polygon the
//! coverage of a pixel is the area of the polygon clipped to the pixel
//! square, computed here in `f64` by clipping against the four sides
//! and applying the surveyor's formula. An 8-bit coverage value can be
//! at most half a level away from it; the tests allow that plus a
//! little arithmetic slack.

use oxideav_core::FillRule;
use oxideav_raster::{FlatContour, rasterize_fill_analytic};

type P = (f64, f64);

/// Clip polygon `poly` to the half-plane `a·x + b·y ≤ c`.
fn clip_half_plane(poly: &[P], a: f64, b: f64, c: f64) -> Vec<P> {
    let mut out = Vec::with_capacity(poly.len() + 4);
    let side = |p: P| a * p.0 + b * p.1 - c;
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        let (sp, sq) = (side(p), side(q));
        if sp <= 0.0 {
            out.push(p);
        }
        if (sp < 0.0 && sq > 0.0) || (sp > 0.0 && sq < 0.0) {
            let t = sp / (sp - sq);
            out.push((p.0 + (q.0 - p.0) * t, p.1 + (q.1 - p.1) * t));
        }
    }
    out
}

/// Signed area of a polygon (surveyor's formula).
fn signed_area(poly: &[P]) -> f64 {
    let mut s = 0.0;
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        s += p.0 * q.1 - q.0 * p.1;
    }
    0.5 * s
}

/// Exact coverage of pixel `(x, y)` by the simple polygon `poly`.
fn exact_coverage(poly: &[P], x: u32, y: u32) -> f64 {
    let (x, y) = (x as f64, y as f64);
    let mut p = clip_half_plane(poly, -1.0, 0.0, -x);
    p = clip_half_plane(&p, 1.0, 0.0, x + 1.0);
    p = clip_half_plane(&p, 0.0, -1.0, -y);
    p = clip_half_plane(&p, 0.0, 1.0, y + 1.0);
    signed_area(&p).abs()
}

/// Largest |rendered − exact| over the canvas, in 8-bit levels, and the
/// summed coverage against the polygon's area (both in pixels).
fn compare(poly: &[P], w: u32, h: u32) -> (f64, f64, f64) {
    let contour = FlatContour {
        points: poly.iter().map(|p| (p.0 as f32, p.1 as f32)).collect(),
        closed: true,
    };
    // The reference uses the coordinates the rasteriser was given.
    let poly: Vec<P> = contour
        .points
        .iter()
        .map(|p| (p.0 as f64, p.1 as f64))
        .collect();
    let mask = rasterize_fill_analytic(&[contour], w, h, FillRule::NonZero);
    let (mut worst, mut total, mut want_total) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            let want = exact_coverage(&poly, x, y) * 255.0;
            let got = mask.get(x, y) as f64;
            worst = worst.max((got - want).abs());
            total += got / 255.0;
            want_total += want / 255.0;
        }
    }
    (worst, total, want_total)
}

/// Half a level of quantisation plus single-precision slack.
const TOLERANCE: f64 = 0.5 + 0.02;

fn check(name: &str, poly: &[P], w: u32, h: u32) {
    let (worst, total, want) = compare(poly, w, h);
    assert!(
        worst <= TOLERANCE,
        "{name}: worst pixel error {worst:.3} levels (area {total:.4} vs {want:.4})"
    );
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

#[test]
fn slivers_thinner_than_a_pixel() {
    let mut rng = Lcg(1);
    for i in 0..400 {
        // A needle: two long edges a fraction of a pixel apart, at any
        // angle, anywhere on the lattice.
        let (ax, ay) = (rng.range(2.0, 60.0), rng.range(2.0, 44.0));
        let (bx, by) = (rng.range(2.0, 60.0), rng.range(2.0, 44.0));
        let width = [0.9, 0.3, 0.05, 0.004, 0.0003][i % 5];
        let (dx, dy) = (bx - ax, by - ay);
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        let (nx, ny) = (-dy / len * width, dx / len * width);
        check(
            &format!("needle {i} (width {width})"),
            &[(ax, ay), (bx, by), (bx + nx, by + ny)],
            64,
            48,
        );
        check(
            &format!("ribbon {i} (width {width})"),
            &[(ax, ay), (bx, by), (bx + nx, by + ny), (ax + nx, ay + ny)],
            64,
            48,
        );
    }
}

#[test]
fn edges_a_hair_from_horizontal_and_vertical() {
    for (i, eps) in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6].into_iter().enumerate() {
        // A wide quadrilateral whose top and bottom edges rise by `eps`
        // over 50 pixels, and its transpose.
        let q = [
            (3.25, 10.5),
            (53.75, 10.5 + eps),
            (53.75, 20.25 + eps),
            (3.25, 20.25 - eps),
        ];
        check(&format!("near-horizontal {i}"), &q, 64, 32);
        let t: Vec<P> = q.iter().rev().map(|p| (p.1, p.0)).collect();
        check(&format!("near-vertical {i}"), &t, 32, 64);
        // Edges exactly on pixel boundaries except for the hair.
        let b = [
            (4.0, 8.0 - eps),
            (40.0, 8.0),
            (40.0 + eps, 24.0),
            (4.0, 24.0),
        ];
        check(&format!("boundary-hugging {i}"), &b, 48, 32);
    }
}

#[test]
fn shapes_much_smaller_than_a_pixel() {
    let mut rng = Lcg(7);
    for i in 0..300 {
        let (cx, cy) = (rng.range(1.0, 15.0), rng.range(1.0, 15.0));
        let r = [0.4, 0.1, 0.02, 0.003][i % 4];
        // A small convex polygon (random angles, sorted).
        let n = 3 + i % 5;
        let mut angles: Vec<f64> = (0..n)
            .map(|_| rng.range(0.0, std::f64::consts::TAU))
            .collect();
        angles.sort_by(f64::total_cmp);
        let poly: Vec<P> = angles
            .iter()
            .map(|a| (cx + r * a.cos(), cy + r * a.sin()))
            .collect();
        check(&format!("speck {i} (radius {r})"), &poly, 16, 16);
    }
}

#[test]
fn random_simple_polygons_at_every_lattice_phase() {
    let mut rng = Lcg(99);
    for i in 0..300 {
        // Star-shaped around a centre: simple by construction, with
        // reflex vertices and spikes.
        let (cx, cy) = (rng.range(14.0, 34.0), rng.range(14.0, 34.0));
        let n = 3 + i % 17;
        let poly: Vec<P> = (0..n)
            .map(|k| {
                let a = (k as f64 + rng.range(0.05, 0.95)) / n as f64 * std::f64::consts::TAU;
                let r = rng.range(0.3, 13.0);
                (cx + r * a.cos(), cy + r * a.sin())
            })
            .collect();
        check(&format!("star {i}"), &poly, 48, 48);
    }
}

#[test]
fn vertices_on_pixel_corners_and_centres() {
    // Every vertex on the half-pixel lattice: boundary cases of the
    // column / row classification.
    let mut rng = Lcg(5);
    for i in 0..300 {
        let n = 3 + i % 6;
        let (cx, cy) = (16.0, 16.0);
        let poly: Vec<P> = (0..n)
            .map(|k| {
                let a = (k as f64 + rng.range(0.1, 0.9)) / n as f64 * std::f64::consts::TAU;
                let r = rng.range(1.0, 14.0);
                (
                    ((cx + r * a.cos()) * 2.0).round() / 2.0,
                    ((cy + r * a.sin()) * 2.0).round() / 2.0,
                )
            })
            .collect();
        if signed_area(&poly).abs() < 1e-9 {
            continue;
        }
        check(&format!("lattice polygon {i}"), &poly, 32, 32);
    }
}

#[test]
fn steep_edges_at_arbitrary_offsets() {
    // Long edges that drift by a small fraction of a pixel per row, at
    // offsets that are not round numbers — the per-row boundary area is
    // then the difference of two nearly equal quantities unless it is
    // evaluated in a cancellation-free form. Includes edges that creep
    // across a pixel-column boundary.
    let mut rng = Lcg(42);
    let mut worst_seen = 0.0f64;
    for i in 0..600 {
        let drift = [3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6][i % 6] * rng.range(0.3, 1.0);
        let x_left = if i % 2 == 0 {
            rng.range(3.0, 20.0)
        } else {
            // Straddling an integer column.
            rng.range(3.0, 20.0).round() - drift * rng.range(0.0, 60.0)
        };
        let x_right = x_left + rng.range(0.6, 9.0);
        let (y0, y1) = (rng.range(1.0, 3.0), rng.range(58.0, 62.0));
        // Both long edges lean the same way (the right one a little
        // more), so the quadrilateral stays simple.
        let poly = [
            (x_left, y0),
            (x_right, y0),
            (x_right + drift * 1.37 * (y1 - y0), y1),
            (x_left + drift * (y1 - y0), y1),
        ];
        let (worst, total, want) = compare(&poly, 64, 64);
        worst_seen = worst_seen.max(worst);
        assert!(
            worst <= TOLERANCE,
            "steep edge {i} (drift {drift:e}/row): worst pixel error {worst:.3} levels \
             (area {total:.4} vs {want:.4})"
        );
    }
    println!("steep edges: worst pixel error {worst_seen:.3} levels");
}

/// Curve flattening at large scales: a cubic magnified until the canvas
/// shows a 96-pixel stretch of a curve whose device-space extent is up
/// to 10⁷ pixels. The reference is the true curve — evaluated in `f64`
/// at 0.1 px steps across the window — clipped per pixel; the
/// renderer's chords may deviate from it by the flattening tolerance
/// (1/32 px, i.e. at most 8 of 255 levels where a chord's worst point
/// sits in a pixel), whatever the zoom.
#[test]
fn flattening_error_does_not_grow_with_scale() {
    use oxideav_core::{
        Group, Node, Paint, Path, PathNode, Point, Rgba, TimeBase, Transform2D, VectorFrame,
    };
    use oxideav_raster::Renderer;

    const N: u32 = 96;
    // Control points in user space (an S-bend with unequal handles).
    let ctrl: [P; 4] = [(0.05, 0.9), (0.45, -0.35), (0.6, 1.3), (0.95, 0.15)];
    let bezier = |t: f64| -> P {
        let u = 1.0 - t;
        let b = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
        (
            b.iter().zip(&ctrl).map(|(w, c)| w * c.0).sum(),
            b.iter().zip(&ctrl).map(|(w, c)| w * c.1).sum(),
        )
    };
    let mut report = Vec::new();
    for (scale, t_mid) in [
        (60.0f64, 0.5),
        (2.0e3, 0.21),
        (5.0e4, 0.37),
        (1.0e6, 0.64),
        (1.0e7, 0.83),
    ] {
        // Put the curve point at `t_mid` on the canvas centre.
        let c = bezier(t_mid);
        let t = Transform2D {
            a: scale as f32,
            b: 0.0,
            c: 0.0,
            d: scale as f32,
            e: (N as f64 / 2.0 - scale * c.0) as f32,
            f: (N as f64 / 2.0 - scale * c.1) as f32,
        };
        // The device map exactly as the renderer evaluates it (f32
        // coefficients, f64 arithmetic).
        let dev = |p: P| -> P { (t.a as f64 * p.0 + t.e as f64, t.d as f64 * p.1 + t.f as f64) };
        // The path: the curve, then far away below-left and back, so
        // that the window sees "everything on one side of the curve".
        let far: P = (-3.0, 4.0);
        let pt = |p: P| Point::new(p.0 as f32, p.1 as f32);
        let mut path = Path::new();
        path.move_to(pt(ctrl[0]))
            .cubic_to(pt(ctrl[1]), pt(ctrl[2]), pt(ctrl[3]))
            .line_to(pt((ctrl[3].0, far.1)))
            .line_to(pt(far))
            .close();
        let frame = VectorFrame {
            width: N as f32,
            height: N as f32,
            view_box: None,
            root: Group {
                transform: t,
                children: vec![Node::Path(PathNode {
                    path,
                    fill: Some(Paint::Solid(Rgba::opaque(0, 0, 0))),
                    stroke: None,
                    fill_rule: FillRule::NonZero,
                })],
                ..Group::default()
            },
            pts: None,
            time_base: TimeBase::new(1, 1),
        };
        let out = Renderer::new(N, N).render(&frame);
        let alpha: Vec<u8> = out.planes[0].data.chunks(4).map(|p| p[3]).collect();

        // Reference polygon: the true curve sampled densely where it is
        // near the window, coarsely elsewhere, closed the same way. The
        // control points are the f32 values the path holds.
        let ctrl32: Vec<P> = ctrl
            .iter()
            .map(|p| (p.0 as f32 as f64, p.1 as f32 as f64))
            .collect();
        let bez32 = |t: f64| -> P {
            let u = 1.0 - t;
            let b = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
            (
                b.iter().zip(&ctrl32).map(|(w, c)| w * c.0).sum(),
                b.iter().zip(&ctrl32).map(|(w, c)| w * c.1).sum(),
            )
        };
        let near =
            |p: P| p.0 > -200.0 && p.0 < N as f64 + 200.0 && p.1 > -200.0 && p.1 < N as f64 + 200.0;
        let mut poly: Vec<P> = Vec::new();
        let coarse = 4000;
        for k in 0..coarse {
            let (ta, tb) = (k as f64 / coarse as f64, (k + 1) as f64 / coarse as f64);
            let (pa, pb) = (dev(bez32(ta)), dev(bez32(tb)));
            poly.push(pa);
            if near(pa) || near(pb) {
                let len = ((pb.0 - pa.0).powi(2) + (pb.1 - pa.1).powi(2)).sqrt();
                let fine = ((len / 0.1).ceil() as usize).clamp(1, 400_000);
                for j in 1..fine {
                    poly.push(dev(bez32(ta + (tb - ta) * j as f64 / fine as f64)));
                }
            }
        }
        poly.push(dev(bez32(1.0)));
        poly.push(dev((ctrl32[3].0, far.1 as f32 as f64)));
        poly.push(dev((far.0 as f32 as f64, far.1 as f32 as f64)));
        // Only the part of the polygon near the window matters for the
        // clip; keep the far vertices (they close it) and clip.
        let (mut worst, mut sum_abs, mut edge_px) = (0.0f64, 0.0f64, 0usize);
        for y in 0..N {
            // Clip to the pixel row once, then to each of its columns.
            let row = clip_half_plane(&poly, 0.0, -1.0, -(y as f64));
            let row = clip_half_plane(&row, 0.0, 1.0, y as f64 + 1.0);
            for x in 0..N {
                let cell = clip_half_plane(&row, -1.0, 0.0, -(x as f64));
                let cell = clip_half_plane(&cell, 1.0, 0.0, x as f64 + 1.0);
                let want = signed_area(&cell).abs() * 255.0;
                let got = alpha[(y * N + x) as usize] as f64;
                let e = (got - want).abs();
                worst = worst.max(e);
                if want > 0.5 && want < 254.5 {
                    sum_abs += e;
                    edge_px += 1;
                }
            }
        }
        assert!(edge_px > 60, "scale {scale:e}: the curve misses the window");
        let mean = sum_abs / edge_px as f64;
        report.push(format!("{scale:e}: worst {worst:.2}, mean {mean:.2}"));
        assert!(
            // A chord between two points of a smooth curve deviates
            // parabolically: two thirds of its worst deviation on
            // average, so the mean over boundary pixels can approach
            // 2/3 · 255/32 ≈ 5.3 levels where chords span many pixels.
            worst <= 8.5 && mean <= 5.8,
            "scale {scale:e}: boundary pixels off by up to {worst:.2} levels (mean {mean:.2})"
        );
    }
    println!("flattening error by scale — {}", report.join("; "));
}
