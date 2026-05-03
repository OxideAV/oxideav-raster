//! Integration tests for linear gradient evaluation.

use oxideav_core::{GradientStop, LinearGradient, Point, Rgba, SpreadMethod};
use oxideav_raster::eval_linear_gradient;

#[test]
fn left_to_right_black_to_white_midpoint_is_mid_gray() {
    let g = LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(10.0, 0.0),
        stops: vec![
            GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
        ],
        spread: SpreadMethod::Pad,
    };
    let c = eval_linear_gradient(&g, 5.0, 0.0);
    assert!((c.r as i32 - 128).abs() <= 2, "got r = {}", c.r);
    assert_eq!(c.g, c.r);
    assert_eq!(c.b, c.r);
}

#[test]
fn three_stop_gradient_picks_correct_segment() {
    let g = LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(100.0, 0.0),
        stops: vec![
            GradientStop::new(0.0, Rgba::opaque(255, 0, 0)), // red at t=0
            GradientStop::new(0.5, Rgba::opaque(0, 255, 0)), // green at t=0.5
            GradientStop::new(1.0, Rgba::opaque(0, 0, 255)), // blue at t=1
        ],
        spread: SpreadMethod::Pad,
    };
    // Sample at t=0.25 → halfway between red and green → yellow-ish.
    let c = eval_linear_gradient(&g, 25.0, 0.0);
    // Red component should be roughly halfway (128 ± 2), green similarly.
    assert!((c.r as i32 - 128).abs() <= 2);
    assert!((c.g as i32 - 128).abs() <= 2);
    assert_eq!(c.b, 0);
}

#[test]
fn pad_clamps_outside_range() {
    let g = LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(10.0, 0.0),
        stops: vec![
            GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
        ],
        spread: SpreadMethod::Pad,
    };
    assert_eq!(eval_linear_gradient(&g, -5.0, 0.0).r, 255);
    assert_eq!(eval_linear_gradient(&g, 100.0, 0.0).b, 255);
}

#[test]
fn reflect_mirrors_at_boundaries() {
    let g = LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(10.0, 0.0),
        stops: vec![
            GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
        ],
        spread: SpreadMethod::Reflect,
    };
    // x=15 → t=1.5 → reflects to 0.5 → mid gray.
    let c = eval_linear_gradient(&g, 15.0, 0.0);
    assert!((c.r as i32 - 128).abs() <= 2);
    // x=20 → t=2.0 → wraps back to 0 → black.
    let c = eval_linear_gradient(&g, 20.0, 0.0);
    assert_eq!(c.r, 0);
}

#[test]
fn repeat_wraps_periodically() {
    let g = LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(10.0, 0.0),
        stops: vec![
            GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
        ],
        spread: SpreadMethod::Repeat,
    };
    // x=10 wraps to 0 → black.
    assert_eq!(eval_linear_gradient(&g, 10.0, 0.0).r, 0);
    // x=15 wraps to 0.5 → mid gray.
    let c = eval_linear_gradient(&g, 15.0, 0.0);
    assert!((c.r as i32 - 128).abs() <= 2);
}
