//! Paint dispatch — given a [`Paint`] and a pixel position, return
//! the straight-alpha sRGB color to composite at that pixel.
//!
//! For solid paints this is trivial; for gradients it delegates to
//! [`crate::gradient::eval_linear_gradient`] /
//! [`crate::gradient::eval_radial_gradient`].
//!
//! Gradient endpoints are passed through the active transform by the
//! [`Renderer`](crate::Renderer) before reaching this layer, so the
//! gradient evaluators see the gradient in raster pixel space.

use oxideav_core::{Paint, Rgba};

use crate::gradient::{eval_linear_gradient, eval_radial_gradient};

/// Sample `paint` at pixel `(x, y)` (in raster pixel coordinates).
pub fn sample_paint(paint: &Paint, x: f32, y: f32) -> Rgba {
    match paint {
        Paint::Solid(c) => *c,
        Paint::LinearGradient(g) => eval_linear_gradient(g, x, y),
        Paint::RadialGradient(g) => eval_radial_gradient(g, x, y),
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
}
