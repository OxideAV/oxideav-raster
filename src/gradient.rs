//! Gradient evaluation — sample a [`LinearGradient`] or
//! [`RadialGradient`] at a point in pixel space and return the resulting
//! straight-alpha sRGB color.
//!
//! Supports SVG/PDF Pad / Reflect / Repeat spread methods. Stops are
//! interpolated linearly in non-linear sRGB space (a future round-2
//! item: linear-light blending for color-managed pipelines).
//!
//! Gradient coordinates are taken to be in the same pixel space the
//! caller is using for raster output (the Renderer applies the active
//! transform to the gradient endpoints before passing them in).

use oxideav_core::{LinearGradient, Point, RadialGradient, Rgba, SpreadMethod};

/// Sample a linear gradient at pixel `(px, py)`.
pub fn eval_linear_gradient(g: &LinearGradient, px: f32, py: f32) -> Rgba {
    if g.stops.is_empty() {
        return Rgba::new(0, 0, 0, 0);
    }
    if g.stops.len() == 1 {
        return g.stops[0].color;
    }
    let dx = g.end.x - g.start.x;
    let dy = g.end.y - g.start.y;
    let denom = dx * dx + dy * dy;
    if denom <= 1e-12 {
        return g.stops.last().copied().unwrap().color;
    }
    let t = ((px - g.start.x) * dx + (py - g.start.y) * dy) / denom;
    sample_stops(&g.stops, apply_spread(t, g.spread))
}

/// Sample a radial gradient at pixel `(px, py)`.
///
/// Implements the SVG 1.1 §13.2.4 ("Radial gradients") general
/// formula: the gradient is parameterised by `t`, where `t = 0` is
/// the focal point and `t = 1` traces the bounding circle (centre =
/// `g.center`, radius = `g.radius`). For each pixel we solve the
/// quadratic in `t`:
///
/// ```text
/// A * t^2 - 2 * (c·d) * t + (d·d) = 0
///   where  c = center - focal,  d = pixel - focal,
///          A = c·c - r^2.
/// ```
///
/// and pick the meaningful root (`t = ((c·d) - sqrt(Δ)) / A`). The
/// formula degenerates correctly when `focal == center` (gives the
/// classical `|P - centre| / r`).
///
/// SVG says the focal must lie *inside* the bounding circle. If the
/// caller supplies a focal on or outside the boundary we clamp it
/// onto the circle (just inside, by a tiny epsilon), matching the
/// browser-side normalisation step.
pub fn eval_radial_gradient(g: &RadialGradient, px: f32, py: f32) -> Rgba {
    if g.stops.is_empty() {
        return Rgba::new(0, 0, 0, 0);
    }
    if g.stops.len() == 1 || g.radius <= 1e-12 {
        return g.stops[0].color;
    }
    let r = g.radius;
    let (fx, fy) = clamp_focal_inside_circle(g.focal.unwrap_or(g.center), g.center, r);
    let cdx = g.center.x - fx;
    let cdy = g.center.y - fy;
    let dx = px - fx;
    let dy = py - fy;
    // c·d, d·d, c·c.
    let cd = cdx * dx + cdy * dy;
    let dd = dx * dx + dy * dy;
    let cc = cdx * cdx + cdy * cdy;
    let aa = cc - r * r; // <= 0 when focal is inside the circle
    let t = if aa.abs() < 1e-12 {
        // Focal on the boundary — quadratic collapses to linear:
        //   -2(c·d)*t + d·d = 0  =>  t = d·d / (2 c·d).
        if cd.abs() < 1e-12 {
            // c·d == 0: pixel sits on the line through focal
            // perpendicular to F→C, with no axis component → use
            // distance/radius as the centred fallback.
            (dd.sqrt()) / r
        } else {
            dd / (2.0 * cd)
        }
    } else {
        let disc = cd * cd - aa * dd;
        if disc < 0.0 {
            // Numerical underflow, possible only off the disc; clamp
            // to the centred fallback so the spread method still
            // runs against a meaningful t.
            dd.sqrt() / r
        } else {
            (cd - disc.sqrt()) / aa
        }
    };
    sample_stops(&g.stops, apply_spread(t, g.spread))
}

/// Pull `focal` strictly inside the bounding circle when it lies on
/// or outside the boundary. SVG normalises this so the gradient
/// equation always has a real positive root for points inside the
/// circle.
fn clamp_focal_inside_circle(focal: Point, center: Point, r: f32) -> (f32, f32) {
    let dx = focal.x - center.x;
    let dy = focal.y - center.y;
    let dist_sq = dx * dx + dy * dy;
    let r_minus_eps = (r - 1e-4).max(0.0);
    if dist_sq <= r_minus_eps * r_minus_eps {
        (focal.x, focal.y)
    } else {
        let dist = dist_sq.sqrt().max(1e-12);
        let scale = r_minus_eps / dist;
        (center.x + dx * scale, center.y + dy * scale)
    }
}

/// Map a parametric coordinate `t` through the spread method, producing
/// a value in `[0.0, 1.0]` ready for stop interpolation.
fn apply_spread(t: f32, spread: SpreadMethod) -> f32 {
    if t.is_nan() {
        return 0.0;
    }
    match spread {
        SpreadMethod::Pad => t.clamp(0.0, 1.0),
        SpreadMethod::Repeat => {
            let r = t - t.floor();
            if r < 0.0 {
                r + 1.0
            } else {
                r
            }
        }
        SpreadMethod::Reflect => {
            let two = (t * 0.5).floor() * 2.0;
            let local = t - two;
            // local is in [0, 2). Mirror around 1.
            if local <= 1.0 {
                local
            } else {
                2.0 - local
            }
        }
    }
}

/// Linear interpolation across the `stops` array. `t` is in `[0, 1]`.
fn sample_stops(stops: &[oxideav_core::GradientStop], t: f32) -> Rgba {
    if stops.is_empty() {
        return Rgba::new(0, 0, 0, 0);
    }
    if t <= stops[0].offset {
        return stops[0].color;
    }
    let last = stops[stops.len() - 1];
    if t >= last.offset {
        return last.color;
    }
    for w in stops.windows(2) {
        let (a, b) = (w[0], w[1]);
        if t >= a.offset && t <= b.offset {
            let span = (b.offset - a.offset).max(1e-9);
            let local = (t - a.offset) / span;
            return lerp_rgba(a.color, b.color, local);
        }
    }
    last.color
}

/// Linear interpolation between two straight-alpha sRGB colors.
fn lerp_rgba(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let l = |x: u8, y: u8| -> u8 {
        let lv = x as f32 + (y as f32 - x as f32) * t;
        lv.round().clamp(0.0, 255.0) as u8
    };
    Rgba::new(l(a.r, b.r), l(a.g, b.g), l(a.b, b.b), l(a.a, b.a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{GradientStop, Point};

    fn black_to_white_linear() -> LinearGradient {
        LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(10.0, 0.0),
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
            ],
            spread: SpreadMethod::Pad,
        }
    }

    #[test]
    fn linear_endpoints_yield_endpoint_colors() {
        let g = black_to_white_linear();
        assert_eq!(eval_linear_gradient(&g, 0.0, 0.0).r, 0);
        assert_eq!(eval_linear_gradient(&g, 10.0, 0.0).r, 255);
    }

    #[test]
    fn linear_midpoint_is_mid_gray() {
        let g = black_to_white_linear();
        let mid = eval_linear_gradient(&g, 5.0, 0.0);
        assert!((mid.r as i32 - 128).abs() <= 2);
    }

    #[test]
    fn linear_pad_clamps_outside_range() {
        let g = black_to_white_linear();
        // Past the end: should still be the end color.
        assert_eq!(eval_linear_gradient(&g, 100.0, 0.0).r, 255);
        // Behind the start: should still be the start color.
        assert_eq!(eval_linear_gradient(&g, -100.0, 0.0).r, 0);
    }

    #[test]
    fn linear_repeat_wraps() {
        let mut g = black_to_white_linear();
        g.spread = SpreadMethod::Repeat;
        // x=20 → t=2 → wraps to 0 → black.
        let c = eval_linear_gradient(&g, 20.0, 0.0);
        assert_eq!(c.r, 0);
    }

    #[test]
    fn linear_reflect_mirrors() {
        let mut g = black_to_white_linear();
        g.spread = SpreadMethod::Reflect;
        // x=15 → t=1.5 → reflects to 0.5 → mid-gray.
        let c = eval_linear_gradient(&g, 15.0, 0.0);
        assert!((c.r as i32 - 128).abs() <= 2);
    }

    #[test]
    fn radial_centre_is_first_stop() {
        let g = RadialGradient {
            center: Point::new(5.0, 5.0),
            radius: 5.0,
            focal: None,
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
            ],
            spread: SpreadMethod::Pad,
        };
        let c = eval_radial_gradient(&g, 5.0, 5.0);
        assert_eq!(c.r, 255);
        assert_eq!(c.b, 0);
    }

    #[test]
    fn radial_at_radius_is_last_stop() {
        let g = RadialGradient {
            center: Point::new(0.0, 0.0),
            radius: 10.0,
            focal: None,
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
            ],
            spread: SpreadMethod::Pad,
        };
        let c = eval_radial_gradient(&g, 10.0, 0.0);
        assert_eq!(c.b, 255);
        assert_eq!(c.r, 0);
    }
}
