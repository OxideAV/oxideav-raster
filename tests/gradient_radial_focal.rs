//! Off-centre focal radial gradient (SVG `radialGradient` `fx`/`fy`
//! distinct from `cx`/`cy`).

use oxideav_core::{GradientStop, Point, RadialGradient, Rgba, SpreadMethod};
use oxideav_raster::eval_radial_gradient;

fn red_to_blue(g_center: Point, radius: f32, focal: Option<Point>) -> RadialGradient {
    RadialGradient {
        center: g_center,
        radius,
        focal,
        stops: vec![
            GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
        ],
        spread: SpreadMethod::Pad,
    }
}

#[test]
fn focal_at_centre_matches_centred_formula() {
    // Off-focal sampler with focal == center must be bit-identical to
    // the classical centred-radial — sanity check that the rewrite
    // didn't regress the common case.
    let g_explicit = red_to_blue(Point::new(10.0, 10.0), 5.0, Some(Point::new(10.0, 10.0)));
    let g_implicit = red_to_blue(Point::new(10.0, 10.0), 5.0, None);
    for &(px, py) in &[(10.0, 10.0), (12.0, 10.0), (15.0, 10.0), (5.0, 10.0)] {
        let a = eval_radial_gradient(&g_explicit, px, py);
        let b = eval_radial_gradient(&g_implicit, px, py);
        assert_eq!(a, b, "off-focal vs centred at ({}, {})", px, py);
    }
}

#[test]
fn focal_offset_returns_first_stop_at_focal_point() {
    // With focal = (cx + r/4, cy), sampling at the focal point must
    // land exactly on the first stop (red).
    let radius = 8.0;
    let center = Point::new(20.0, 20.0);
    let focal = Point::new(center.x + radius / 4.0, center.y);
    let g = red_to_blue(center, radius, Some(focal));
    let c = eval_radial_gradient(&g, focal.x, focal.y);
    assert_eq!(
        c,
        Rgba::opaque(255, 0, 0),
        "at focal must be the first stop"
    );
}

#[test]
fn focal_offset_far_edge_is_last_stop() {
    // The "far edge" along the focal→center direction (i.e. the
    // bounding-circle point on the far side of centre from the focal)
    // is t = 1 and must be the last stop (blue).
    let radius = 8.0;
    let center = Point::new(20.0, 20.0);
    let focal = Point::new(center.x + radius / 4.0, center.y);
    let g = red_to_blue(center, radius, Some(focal));
    // Pick the far edge: along the negative-x direction from focal,
    // crossing centre, hitting the bounding circle at (cx - r, cy).
    let far = Point::new(center.x - radius, center.y);
    let c = eval_radial_gradient(&g, far.x, far.y);
    assert_eq!(c, Rgba::opaque(0, 0, 255), "far edge must be the last stop");
}

#[test]
fn focal_offset_pattern_is_asymmetric() {
    // When the focal is shifted off-centre, the gradient is no
    // longer rotationally symmetric. Sample two points at equal
    // distance from the centre but on opposite sides along the
    // focal-to-centre axis: the colours must differ noticeably.
    let radius = 10.0;
    let center = Point::new(20.0, 20.0);
    let focal = Point::new(center.x + radius / 4.0, center.y);
    let g = red_to_blue(center, radius, Some(focal));
    // Symmetric points around the centre on the X axis.
    let left = eval_radial_gradient(&g, center.x - 3.0, center.y);
    let right = eval_radial_gradient(&g, center.x + 3.0, center.y);
    let dr = (left.r as i32 - right.r as i32).abs();
    let db = (left.b as i32 - right.b as i32).abs();
    assert!(
        dr + db > 30,
        "off-focal gradient must be asymmetric — got left={:?} right={:?}",
        left,
        right
    );
}

#[test]
fn focal_outside_circle_clamped_inside() {
    // Pathological input: focal placed outside the bounding circle
    // gets clamped onto the boundary. The result must still be
    // well-defined — pixel values stay in 0..=255 with no NaN
    // bleed-through.
    let radius = 5.0;
    let center = Point::new(10.0, 10.0);
    let focal = Point::new(50.0, 50.0); // way outside
    let g = red_to_blue(center, radius, Some(focal));
    for &(px, py) in &[(10.0, 10.0), (12.0, 10.0), (8.0, 11.0), (15.0, 10.0)] {
        let c = eval_radial_gradient(&g, px, py);
        // Channels must be the gradient stops' interpolation domain
        // (red <-> blue, no green leak, fully opaque) — no NaN /
        // out-of-domain bleed.
        assert_eq!(c.g, 0, "no green leak at ({}, {}), got {:?}", px, py, c);
        assert_eq!(c.a, 255, "alpha must stay 255 at ({}, {})", px, py);
        // r + b should be within ±1 of 255 once the integer rounding
        // closes (each channel is a u8 so the bounds are implicit).
        let sum = c.r as i32 + c.b as i32;
        assert!(
            (sum - 255).abs() <= 2,
            "red + blue should sum to ~255 at ({}, {}) got {}+{}={}",
            px,
            py,
            c.r,
            c.b,
            sum
        );
    }
}
