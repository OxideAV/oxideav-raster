//! Round 484: inputs found by the `render_paths` fuzz target that used
//! to exhaust memory (a stroke pen of astronomical width made the curve
//! flattener emit hundreds of millions of points) or time (a sub-pixel
//! dash pattern under a strongly anisotropic CTM produced 10⁵ dashes
//! with 4 096-chord caps). The decoder is the fuzz target's.

use oxideav_core::{
    DashPattern, ExecutionContext, FillRule, Group, LineCap, LineJoin, Node, Paint, Path, PathNode,
    Point, Rgba, Stroke, Transform2D, VectorFrame,
};
use oxideav_raster::{CoverageMode, InterpolationSpace, Renderer};

fn f32_at(d: &[u8], i: usize) -> f32 {
    d.get(i..i + 4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .unwrap_or(0.0)
}

fn run(data: &[u8]) {
    if data.len() < 39 {
        return;
    }
    let w = (data[0] % 40) as u32 + 1;
    let h = (data[1] % 40) as u32 + 1;
    let flags = data[2];
    let t = Transform2D {
        a: f32_at(data, 15),
        b: f32_at(data, 19),
        c: f32_at(data, 23),
        d: f32_at(data, 27),
        e: f32_at(data, 31),
        f: f32_at(data, 35),
    };
    let mut path = Path::new();
    let mut i = 39;
    let pt = |i: usize| Point::new(f32_at(data, i), f32_at(data, i + 4));
    let mut ops = 0;
    while i < data.len() && ops < 64 {
        let op = data[i] % 5;
        i += 1;
        match op {
            0 => {
                path.move_to(pt(i));
                i += 8;
            }
            1 => {
                path.line_to(pt(i));
                i += 8;
            }
            2 => {
                path.quad_to(pt(i), pt(i + 8));
                i += 16;
            }
            3 => {
                path.cubic_to(pt(i), pt(i + 8), pt(i + 16));
                i += 24;
            }
            _ => {
                path.close();
            }
        }
        ops += 1;
    }
    let stroke = (flags & 2 != 0).then(|| Stroke {
        width: f32_at(data, 3),
        paint: Paint::Solid(Rgba::new(10, 200, 30, 180)),
        cap: [
            LineCap::Butt,
            LineCap::Round,
            LineCap::Square,
            LineCap::Round,
        ][((flags >> 3) & 3) as usize],
        join: [
            LineJoin::Miter,
            LineJoin::Round,
            LineJoin::Bevel,
            LineJoin::Miter,
        ][((flags >> 5) & 3) as usize],
        miter_limit: 4.0,
        dash: (flags & 4 != 0).then(|| DashPattern {
            array: vec![f32_at(data, 7), f32_at(data, 11)],
            offset: 0.0,
        }),
    });
    let node = Node::Path(PathNode {
        path,
        fill: Some(Paint::Solid(Rgba::new(200, 40, 90, 200))),
        stroke,
        fill_rule: if flags & 1 != 0 {
            FillRule::EvenOdd
        } else {
            FillRule::NonZero
        },
    });
    let frame = VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root: Group {
            transform: t,
            children: vec![node],
            ..Group::default()
        },
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let space = if flags & 0x80 != 0 {
        InterpolationSpace::LinearRgb
    } else {
        InterpolationSpace::Srgb
    };
    let r = Renderer::new(w, h).with_compositing_space(space);
    let serial = r.render(&frame);
    assert_eq!(serial.planes[0].data.len(), (w * h * 4) as usize);
    let par = r
        .clone()
        .with_execution_context(&ExecutionContext::with_threads(3))
        .render(&frame);
    assert_eq!(serial.planes[0].data, par.planes[0].data);
    let ss = Renderer::new(w, h)
        .with_coverage(CoverageMode::Supersampled)
        .render(&frame);
    assert_eq!(ss.planes[0].data.len(), (w * h * 4) as usize);
}

#[test]
fn astronomical_pen_is_bounded() {
    let t = std::time::Instant::now();
    run(&[
        0x00, 0x7f, 0xff, 0xff, 0x00, 0x00, 0x7f, 0x00, 0xff, 0x3d, 0xff, 0x00, 0x00, 0x00, 0x05,
        0x00, 0x00, 0x00, 0xe7, 0xff, 0xff, 0x2a, 0xdf, 0x7f, 0xff, 0xff, 0x00, 0x00, 0x7f, 0x00,
        0xff, 0x3d, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0xff, 0x12, 0x00, 0x00, 0x2b, 0x0a, 0x0a,
        0x20, 0xcc, 0x00,
    ]);
    assert!(t.elapsed().as_secs() < 20);
}

#[test]
fn sub_pixel_dashes_under_anisotropic_ctm_are_bounded() {
    let t = std::time::Instant::now();
    run(&[
        0x0a, 0xff, 0xff, 0x00, 0xde, 0xfb, 0x09, 0x0a, 0x00, 0x22, 0xf7, 0x00, 0x06, 0xee, 0x00,
        0x00, 0x37, 0x00, 0x00, 0xa7, 0x00, 0x00, 0xf7, 0x07, 0x2b, 0x7f, 0x7f, 0x01, 0x00, 0x7f,
        0x00, 0x26, 0xde, 0xff, 0x22, 0x0a, 0x00, 0x00, 0x2d, 0x01, 0x00, 0xff, 0xff, 0x06, 0x7e,
        0x00, 0x3e,
    ]);
    assert!(t.elapsed().as_secs() < 20);
}

/// Scheduled-fuzz crash: a path mixing `f32::MAX`-scale and NaN
/// coordinates produced NaN edge keys in the supersampled kernel, and
/// the standard library's sort rejected the resulting non-total order.
#[test]
fn nan_and_overflowing_coordinates_keep_the_edge_order_total() {
    run(&[
        0x13, 0x13, 0x2a, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x00, 0x00,
        0x40, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x90, 0x41, 0x00, 0x00, 0x00, 0x80,
        0x3f, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x48, 0x00, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0x7f, 0x00, 0x00, 0x90, 0x41, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb,
        0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0x41, 0x04,
    ]);
}

/// Every non-finite value an edge list can carry, in both coverage
/// kernels and at every supersampling factor: the mask comes back the
/// right size and the finite part of the outline still fills.
#[test]
fn non_finite_contour_points_never_reach_the_edge_sort() {
    use oxideav_raster::{FlatContour, rasterize_fill, rasterize_fill_analytic};
    let bad = [
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        3.0e38,
        -3.0e38,
    ];
    let square = FlatContour {
        points: vec![(4.0, 4.0), (20.0, 4.0), (20.0, 20.0), (4.0, 20.0)],
        closed: true,
    };
    for (k, &u) in bad.iter().enumerate() {
        for (j, &v) in bad.iter().enumerate() {
            // A long zig-zag (more points than the small-sort
            // threshold) whose every third vertex is hostile.
            let mut pts = Vec::new();
            for i in 0..96u32 {
                let (x, y) = ((i * 7 % 23) as f32, (i * 5 % 19) as f32 + 0.25);
                pts.push(match i % 3 {
                    0 => (x, y),
                    1 => (u, y + (j as f32)),
                    _ => (x + (k as f32), v),
                });
            }
            let hostile = FlatContour {
                points: pts,
                closed: true,
            };
            for rule in [FillRule::NonZero, FillRule::EvenOdd] {
                for ss in [1u8, 2, 4, 8] {
                    let m = rasterize_fill(&[hostile.clone(), square.clone()], 24, 24, rule, ss);
                    assert_eq!(m.data.len(), 24 * 24);
                }
                let m = rasterize_fill_analytic(&[hostile.clone(), square.clone()], 24, 24, rule);
                assert_eq!(m.data.len(), 24 * 24);
            }
        }
    }
    // The hostile points alone paint nothing; the square alone is exact.
    let nan_only = FlatContour {
        points: vec![(f32::NAN, 1.0), (2.0, f32::INFINITY), (f32::NAN, f32::NAN)],
        closed: true,
    };
    for ss in [1u8, 2, 4, 8] {
        let m = rasterize_fill(
            std::slice::from_ref(&nan_only),
            24,
            24,
            FillRule::NonZero,
            ss,
        );
        assert!(m.data.iter().all(|&a| a == 0));
        let m = rasterize_fill(
            &[nan_only.clone(), square.clone()],
            24,
            24,
            FillRule::NonZero,
            ss,
        );
        assert_eq!(m.get(10, 10), 255);
        assert_eq!(m.get(1, 1), 0);
        assert_eq!(m.get(22, 22), 0);
    }
}

/// The typed rejection at the API boundary: `try_render` names the part
/// of the scene that carries a non-finite value, and accepts the same
/// scene once the value is finite.
#[test]
fn try_render_rejects_non_finite_scenes_with_a_typed_error() {
    use oxideav_raster::{GeometryPart, RenderError};
    let build = |x: f32, e: f32, width: f32, opacity: f32| {
        let mut path = Path::new();
        path.move_to(Point::new(1.0, 1.0))
            .line_to(Point::new(x, 1.0))
            .line_to(Point::new(9.0, 9.0))
            .close();
        VectorFrame {
            width: 16.0,
            height: 16.0,
            view_box: None,
            root: Group {
                transform: Transform2D {
                    e,
                    ..Transform2D::identity()
                },
                opacity,
                children: vec![Node::Path(PathNode {
                    path,
                    fill: Some(Paint::Solid(Rgba::new(1, 2, 3, 255))),
                    stroke: Some(Stroke::solid(width, Rgba::new(9, 9, 9, 255))),
                    fill_rule: FillRule::NonZero,
                })],
                ..Group::default()
            },
            pts: None,
            time_base: oxideav_core::time::TimeBase::new(1, 1),
        }
    };
    let r = Renderer::new(16, 16);
    let ok = r.try_render(&build(9.0, 0.0, 1.0, 1.0)).expect("finite");
    assert_eq!(
        ok.planes[0].data,
        r.render(&build(9.0, 0.0, 1.0, 1.0)).planes[0].data
    );
    let err = |f: &VectorFrame| r.try_render(f).map(|_| ()).unwrap_err();
    assert_eq!(
        err(&build(f32::NAN, 0.0, 1.0, 1.0)),
        RenderError::NonFiniteGeometry(GeometryPart::PathPoint)
    );
    assert_eq!(
        err(&build(9.0, f32::INFINITY, 1.0, 1.0)),
        RenderError::NonFiniteGeometry(GeometryPart::Transform)
    );
    assert_eq!(
        err(&build(9.0, 0.0, f32::NEG_INFINITY, 1.0)),
        RenderError::NonFiniteGeometry(GeometryPart::Stroke)
    );
    assert_eq!(
        err(&build(9.0, 0.0, 1.0, f32::NAN)),
        RenderError::NonFiniteGeometry(GeometryPart::Opacity)
    );
    // The total entry point still renders each of them without panicking.
    for f in [
        build(f32::NAN, 0.0, 1.0, 1.0),
        build(9.0, f32::INFINITY, 1.0, 1.0),
        build(9.0, 0.0, f32::NEG_INFINITY, 1.0),
        build(9.0, 0.0, 1.0, f32::NAN),
    ] {
        assert_eq!(r.render(&f).planes[0].data.len(), 16 * 16 * 4);
    }
}

/// A pen 10⁷⁶ px wide over curves 10¹⁹ px across: the per-curve chord
/// cap held only in exact arithmetic (rounding noise in the sub-curves'
/// turning angles re-inflated every subdivision leaf to the maximum
/// step count, 2²⁸ points in all). The chord budget is now explicit.
#[test]
fn turning_angle_noise_cannot_defeat_the_chord_budget() {
    let t = std::time::Instant::now();
    run(&[
        0x00, 0x7f, 0xff, 0xff, 0x00, 0x00, 0x7f, 0x00, 0xff, 0x3d, 0xff, 0x5d, 0x88, 0x00, 0x00,
        0xe7, 0xff, 0xff, 0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0xff, 0x12, 0x00, 0x00,
        0x2b, 0xff, 0x00, 0x2b, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbc, 0xbb, 0xbb,
        0xbb, 0xbb, 0xab, 0xbb, 0xbc, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb,
        0xbb, 0xbb, 0xbb, 0x00, 0x0a, 0x12, 0x0a, 0x0a, 0x20, 0xcc, 0x00, 0x0a, 0x0a, 0x20, 0xcc,
        0x00,
    ]);
    assert!(t.elapsed().as_secs() < 20);
}

/// A round-joined stroke with a 10¹⁰ px pen around an 8 px cubic: at
/// the fill chord budget the curve became 65 536 mutually overlapping
/// outline quads, each covering the whole canvas (18 s for 20 × 20
/// pixels). Stroked curves have their own, smaller chord budget.
#[test]
fn huge_round_pen_on_a_small_curve_is_bounded() {
    let t = std::time::Instant::now();
    run(&[
        0x13, 0x13, 0x2a, 0x00, 0x24, 0x40, 0x50, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x00, 0x40,
        0x00, 0x10, 0x80, 0x3f, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab,
        0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0xab, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f,
        0xab, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x03, 0x00,
        0x00, 0x00, 0x41, 0x00, 0x53, 0x00, 0x40, 0xd0, 0x03, 0x00, 0x00, 0xd0, 0x41, 0x00, 0x40,
        0x00, 0x53, 0x00, 0x36, 0x00, 0x80, 0x41, 0x04,
    ]);
    assert!(t.elapsed().as_secs() < 20);
}
