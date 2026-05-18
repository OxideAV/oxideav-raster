//! Paint dispatch — given a [`Paint`] and a pixel position, return
//! the straight-alpha sRGB color to composite at that pixel.
//!
//! For solid paints this is trivial; for gradients it delegates to
//! [`crate::gradient::eval_linear_gradient_in`] /
//! [`crate::gradient::eval_radial_gradient_in`] in the requested
//! interpolation space.
//!
//! Gradient endpoints are passed through the active transform by the
//! [`Renderer`](crate::Renderer) before reaching this layer, so the
//! gradient evaluators see the gradient in raster pixel space.

use oxideav_core::{Paint, Rgba};

use crate::gradient::{eval_linear_gradient_in, eval_radial_gradient_in, InterpolationSpace};

/// Sample `paint` at pixel `(x, y)` (in raster pixel coordinates),
/// using the default sRGB interpolation space (back-compat alias).
pub fn sample_paint(paint: &Paint, x: f32, y: f32) -> Rgba {
    sample_paint_in(paint, x, y, InterpolationSpace::Srgb)
}

/// Sample `paint` at pixel `(x, y)` in the requested interpolation
/// space. Solid paints are unaffected by `space`; gradients route
/// through [`eval_linear_gradient_in`] / [`eval_radial_gradient_in`].
pub fn sample_paint_in(paint: &Paint, x: f32, y: f32, space: InterpolationSpace) -> Rgba {
    match paint {
        Paint::Solid(c) => *c,
        Paint::LinearGradient(g) => eval_linear_gradient_in(g, x, y, space),
        Paint::RadialGradient(g) => eval_radial_gradient_in(g, x, y, space),
        // `Paint` is #[non_exhaustive]; future paint servers
        // (patterns, ICC named colors) fall back to transparent.
        _ => Rgba::new(0, 0, 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{GradientStop, LinearGradient, Point, SpreadMethod};

    #[test]
    fn solid_returns_color_unchanged() {
        let p = Paint::Solid(Rgba::opaque(10, 20, 30));
        assert_eq!(sample_paint(&p, 5.0, 5.0), Rgba::opaque(10, 20, 30));
    }

    #[test]
    fn linear_dispatches_to_gradient_evaluator() {
        let p = Paint::LinearGradient(LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(10.0, 0.0),
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
            ],
            spread: SpreadMethod::Pad,
        });
        assert_eq!(sample_paint(&p, 0.0, 0.0).r, 0);
        assert_eq!(sample_paint(&p, 10.0, 0.0).r, 255);
    }

    #[test]
    fn solid_is_unaffected_by_interpolation_space() {
        let p = Paint::Solid(Rgba::opaque(10, 20, 30));
        let s = sample_paint_in(&p, 5.0, 5.0, InterpolationSpace::Srgb);
        let l = sample_paint_in(&p, 5.0, 5.0, InterpolationSpace::LinearRgb);
        assert_eq!(s, l);
    }

    #[test]
    fn gradient_midpoint_differs_between_spaces() {
        let p = Paint::LinearGradient(LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(10.0, 0.0),
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
                GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
            ],
            spread: SpreadMethod::Pad,
        });
        let s = sample_paint_in(&p, 5.0, 0.0, InterpolationSpace::Srgb);
        let l = sample_paint_in(&p, 5.0, 0.0, InterpolationSpace::LinearRgb);
        assert!(
            (s.r as i32 - 128).abs() <= 2,
            "sRGB midpoint should be ~128, got {}",
            s.r
        );
        assert!(
            l.r >= 185 && l.r <= 192,
            "linearRGB midpoint should be ~188, got {}",
            l.r
        );
    }
}
