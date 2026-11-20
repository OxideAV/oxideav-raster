//! Conic (sweep) gradients: the parameter is the angle around the
//! centre, `t = ((θ − start) mod 2π) / sweep`, with `θ` measured from
//! the positive x axis towards the positive y axis of gradient space.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use oxideav_core::{FillRule, GradientStop, Path, Point, Rgba, SpreadMethod, Transform2D};
use oxideav_raster::{
    ConicGradient, GradientPaint, GradientUnits, InterpolationSpace, Renderer, StopsLut,
    eval_conic_gradient, eval_conic_gradient_in, eval_conic_gradient_lut,
};

fn stop(offset: f32, r: u8, g: u8, b: u8) -> GradientStop {
    GradientStop {
        offset,
        color: Rgba::opaque(r, g, b),
    }
}

/// Black at t = 0 to white at t = 1.
fn ramp(center: Point, start: f32) -> ConicGradient {
    ConicGradient::new(center, start).with_stops(vec![stop(0.0, 0, 0, 0), stop(1.0, 255, 255, 255)])
}

fn at(g: &ConicGradient, angle: f32, radius: f32) -> u8 {
    eval_conic_gradient(
        g,
        g.center.x + radius * angle.cos(),
        g.center.y + radius * angle.sin(),
    )
    .r
}

#[test]
fn the_parameter_is_the_angle_from_the_start_direction() {
    let g = ramp(Point::new(40.0, 30.0), 0.0);
    for (k, frac) in [0.125f32, 0.25, 0.5, 0.75, 0.9].into_iter().enumerate() {
        for radius in [0.5f32, 7.0, 300.0] {
            let v = at(&g, frac * TAU, radius);
            let want = (frac * 255.0).round() as i32;
            assert!(
                (v as i32 - want).abs() <= 1,
                "case {k}: {v} vs {want} at radius {radius}"
            );
        }
    }
    // +y is a quarter turn after +x.
    assert!((eval_conic_gradient(&g, 40.0, 50.0).r as i32 - 64).abs() <= 1);
    // A different start direction shifts the whole ramp.
    let g = ramp(Point::new(0.0, 0.0), FRAC_PI_2);
    assert!(at(&g, FRAC_PI_2 + 0.01, 10.0) < 2);
    assert!((at(&g, PI, 10.0) as i32 - 64).abs() <= 1);
    assert!((at(&g, 0.0, 10.0) as i32 - 191).abs() <= 1);
    // The centre itself takes θ = 0.
    assert_eq!(
        eval_conic_gradient(&g, 0.0, 0.0),
        eval_conic_gradient(&g, 5.0, 0.0)
    );
}

#[test]
fn the_seam_is_a_hard_edge_at_the_start_ray_and_never_wraps_early() {
    let g = ramp(Point::new(0.0, 0.0), 0.0);
    // Just after the start ray: the beginning of the ramp; just before
    // it: the end. No sample anywhere near the ray shows the other end.
    for radius in [1.0f32, 10.0, 1000.0, 1.0e6] {
        for eps in [1e-3f32, 1e-5, 1e-7] {
            assert!(
                at(&g, eps, radius) <= 1,
                "after the seam, radius {radius}, eps {eps}"
            );
            let before = eval_conic_gradient(&g, radius, -radius * eps).r;
            assert!(
                before >= 254,
                "before the seam, radius {radius}, eps {eps}: {before}"
            );
        }
    }
    // Equal end stops: no seam.
    let closed = ConicGradient::new(Point::new(0.0, 0.0), 1.0).with_stops(vec![
        stop(0.0, 200, 10, 10),
        stop(0.5, 10, 10, 200),
        stop(1.0, 200, 10, 10),
    ]);
    let a = eval_conic_gradient(&closed, 1.0f32.cos() * 9.0, (1.0f32 + 1e-4).sin() * 9.0);
    let b = eval_conic_gradient(&closed, 1.0f32.cos() * 9.0, (1.0f32 - 1e-4).sin() * 9.0);
    assert!(a.r.abs_diff(b.r) <= 1 && a.b.abs_diff(b.b) <= 1);
}

#[test]
fn a_partial_sweep_is_completed_by_the_spread_method() {
    let base = ramp(Point::new(0.0, 0.0), 0.0).with_sweep(FRAC_PI_2);
    // Inside the quarter-turn wedge: the whole ramp.
    assert!((at(&base, FRAC_PI_2 * 0.5, 20.0) as i32 - 128).abs() <= 1);
    // Outside it, per spread method.
    let pad = base.clone().with_spread(SpreadMethod::Pad);
    assert_eq!(at(&pad, PI, 20.0), 255);
    assert_eq!(at(&pad, 1.9 * PI, 20.0), 255);
    let repeat = base.clone().with_spread(SpreadMethod::Repeat);
    assert!((at(&repeat, FRAC_PI_2 * 1.5, 20.0) as i32 - 128).abs() <= 1);
    assert!((at(&repeat, FRAC_PI_2 * 3.25, 20.0) as i32 - 64).abs() <= 1);
    let reflect = base.with_spread(SpreadMethod::Reflect);
    assert!((at(&reflect, FRAC_PI_2 * 1.25, 20.0) as i32 - 191).abs() <= 1);
    assert!((at(&reflect, FRAC_PI_2 * 2.25, 20.0) as i32 - 64).abs() <= 1);
    // A degenerate sweep falls back to a full turn.
    for sweep in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        let g = ramp(Point::new(0.0, 0.0), 0.0).with_sweep(sweep);
        assert!((at(&g, PI, 3.0) as i32 - 128).abs() <= 1, "sweep {sweep}");
    }
}

#[test]
fn stops_interpolation_space_and_lut_agree_with_the_exact_evaluator() {
    let g = ConicGradient::new(Point::new(12.0, -7.0), 0.4).with_stops(vec![
        stop(0.0, 255, 0, 0),
        stop(0.3, 0, 255, 0),
        stop(0.3, 0, 0, 255),
        stop(1.0, 255, 255, 0),
    ]);
    for space in [InterpolationSpace::Srgb, InterpolationSpace::LinearRgb] {
        let lut = StopsLut::build(&g.stops, space);
        for k in 0..400 {
            let a = k as f32 * 0.0157;
            // Keep clear of the hard stop and the seam, where a table
            // necessarily smooths.
            let t = ((a - 0.4).rem_euclid(TAU)) / TAU;
            if (t - 0.3).abs() < 0.01 || !(0.01..=0.99).contains(&t) {
                continue;
            }
            let (x, y) = (12.0 + 30.0 * a.cos(), -7.0 + 30.0 * a.sin());
            let exact = eval_conic_gradient_in(&g, x, y, space);
            let fast = eval_conic_gradient_lut(&g, x, y, &lut);
            for (e, f) in [(exact.r, fast.r), (exact.g, fast.g), (exact.b, fast.b)] {
                assert!(
                    e.abs_diff(f) <= 2,
                    "{space:?} angle {a}: {exact:?} vs {fast:?}"
                );
            }
        }
    }
    // Degenerate stop lists.
    let none = ConicGradient::new(Point::new(0.0, 0.0), 0.0);
    assert_eq!(eval_conic_gradient(&none, 3.0, 4.0), Rgba::new(0, 0, 0, 0));
    let one = none.with_stops(vec![stop(0.5, 9, 8, 7)]);
    assert_eq!(eval_conic_gradient(&one, 3.0, 4.0), Rgba::opaque(9, 8, 7));
}

fn disc(cx: f32, cy: f32, r: f32) -> Path {
    let k = 0.552_284_7 * r;
    let mut p = Path::new();
    p.move_to(Point::new(cx + r, cy))
        .cubic_to(
            Point::new(cx + r, cy + k),
            Point::new(cx + k, cy + r),
            Point::new(cx, cy + r),
        )
        .cubic_to(
            Point::new(cx - k, cy + r),
            Point::new(cx - r, cy + k),
            Point::new(cx - r, cy),
        )
        .cubic_to(
            Point::new(cx - r, cy - k),
            Point::new(cx - k, cy - r),
            Point::new(cx, cy - r),
        )
        .cubic_to(
            Point::new(cx + k, cy - r),
            Point::new(cx + r, cy - k),
            Point::new(cx + r, cy),
        )
        .close();
    p
}

#[test]
fn painted_through_the_gradient_paint_route_it_follows_every_transform() {
    let px = |data: &[u8], w: u32, x: u32, y: u32| -> [u8; 4] {
        let i = ((y * w + x) * 4) as usize;
        [data[i], data[i + 1], data[i + 2], data[i + 3]]
    };
    let r = Renderer::new(96, 96);
    // User space: a disc of radius 40 swept from black to white.
    let paint = GradientPaint::conic(ramp(Point::new(48.0, 48.0), 0.0));
    let out = r.fill_path_with_gradient(
        &disc(48.0, 48.0, 40.0),
        FillRule::NonZero,
        &paint,
        Transform2D::identity(),
    );
    let d = &out.planes[0].data;
    // Right of centre, just below the seam: start of the ramp; below
    // centre: a quarter; left: half; above: three quarters.
    assert!(px(d, 96, 78, 49)[0] <= 6);
    assert!((px(d, 96, 48, 78)[0] as i32 - 64).abs() <= 3);
    assert!((px(d, 96, 18, 48)[0] as i32 - 128).abs() <= 3);
    assert!((px(d, 96, 48, 18)[0] as i32 - 191).abs() <= 3);
    assert_eq!(px(d, 96, 2, 2), [0, 0, 0, 0]);

    // The CTM rotates the gradient with the shape: a quarter turn of
    // the canvas about its centre moves the start ray to +y.
    let quarter = Transform2D::translate(48.0, 48.0)
        .compose(&Transform2D::rotate(FRAC_PI_2))
        .compose(&Transform2D::translate(-48.0, -48.0));
    let out =
        r.fill_path_with_gradient(&disc(48.0, 48.0, 40.0), FillRule::NonZero, &paint, quarter);
    let d = &out.planes[0].data;
    assert!(px(d, 96, 47, 78)[0] <= 6, "{:?}", px(d, 96, 47, 78));
    assert!((px(d, 96, 18, 48)[0] as i32 - 64).abs() <= 3);

    // Bounding-box units: centre (½, ½) of an 80 × 40 ellipse's box; the
    // angles are those of the unit square, so the diagonal of the box
    // is the eighth-turn ray.
    let mut wide = Path::new();
    wide.move_to(Point::new(8.0, 28.0))
        .line_to(Point::new(88.0, 28.0))
        .line_to(Point::new(88.0, 68.0))
        .line_to(Point::new(8.0, 68.0))
        .close();
    let boxed = GradientPaint::conic(ramp(Point::new(0.5, 0.5), 0.0))
        .with_units(GradientUnits::ObjectBoundingBox);
    let out = r.fill_path_with_gradient(&wide, FillRule::NonZero, &boxed, Transform2D::identity());
    let d = &out.planes[0].data;
    // Towards the bottom-right corner of the box: 1/8.
    assert!(
        (px(d, 96, 78, 63)[0] as i32 - 32).abs() <= 3,
        "{:?}",
        px(d, 96, 78, 63)
    );
    // Straight below the centre: 1/4, straight left: 1/2.
    assert!((px(d, 96, 48, 64)[0] as i32 - 64).abs() <= 3);
    assert!((px(d, 96, 12, 48)[0] as i32 - 128).abs() <= 3);
}
