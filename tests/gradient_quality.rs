//! Round 484 regression tests for gradient quality: hard stops stay
//! hard at any gradient length, the row evaluator matches the exact
//! per-pixel evaluator, and the optional ordered dither removes
//! banding without biasing the mean.

use oxideav_core::{
    FillRule, GradientStop, Group, LinearGradient, Node, Paint, Path, PathNode, Point,
    RadialGradient, Rgba, SpreadMethod, VectorFrame,
};
use oxideav_raster::{GradientDither, InterpolationSpace, Renderer, sample_paint_in};

fn full_rect(w: u32, h: u32, paint: Paint) -> VectorFrame {
    let mut p = Path::new();
    p.move_to(Point::new(0.0, 0.0))
        .line_to(Point::new(w as f32, 0.0))
        .line_to(Point::new(w as f32, h as f32))
        .line_to(Point::new(0.0, h as f32))
        .close();
    VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root: Group {
            children: vec![Node::Path(PathNode {
                path: p,
                fill: Some(paint),
                stroke: None,
                fill_rule: FillRule::NonZero,
            })],
            ..Group::default()
        },
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

fn px(data: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * w + x) * 4) as usize;
    [data[i], data[i + 1], data[i + 2], data[i + 3]]
}

/// A hard colour stop (two stops at one offset) on a 2 560 px wide
/// gradient must switch colour between two adjacent pixels. The
/// 256-entry stop table interpolated across it, smearing the edge over
/// ~10 px of purple.
#[test]
fn hard_stop_stays_hard_on_a_wide_gradient() {
    let (w, h) = (2560u32, 2u32);
    let red = Rgba::opaque(255, 0, 0);
    let blue = Rgba::opaque(0, 0, 255);
    let paint = Paint::LinearGradient(LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(w as f32, 0.0),
        stops: vec![
            GradientStop::new(0.0, red),
            GradientStop::new(0.5, red),
            GradientStop::new(0.5, blue),
            GradientStop::new(1.0, blue),
        ],
        spread: SpreadMethod::Pad,
    });
    let out = Renderer::new(w, h).render(&full_rect(w, h, paint));
    let d = &out.planes[0].data;
    for x in 0..w {
        let want = if x < 1280 {
            [255, 0, 0, 255]
        } else {
            [0, 0, 255, 255]
        };
        assert_eq!(px(d, w, x, 1), want, "x={x}");
    }
}

/// Without dither the row evaluator rounds the exact stop
/// interpolation: it agrees with the public per-pixel evaluator to the
/// level, in both interpolation spaces, for linear and radial
/// gradients under a rotated, scaled CTM.
#[test]
fn row_evaluator_matches_exact_per_pixel_evaluation() {
    let (w, h) = (96u32, 64u32);
    let stops = vec![
        GradientStop::new(0.0, Rgba::new(250, 10, 30, 255)),
        GradientStop::new(0.3, Rgba::new(20, 200, 90, 255)),
        GradientStop::new(1.0, Rgba::new(10, 40, 230, 255)),
    ];
    let paints = [
        Paint::LinearGradient(LinearGradient {
            start: Point::new(3.0, 5.0),
            end: Point::new(40.0, 31.0),
            stops: stops.clone(),
            spread: SpreadMethod::Reflect,
        }),
        Paint::RadialGradient(RadialGradient {
            center: Point::new(30.0, 20.0),
            radius: 22.0,
            focal: Some(Point::new(24.0, 18.0)),
            stops,
            spread: SpreadMethod::Repeat,
        }),
    ];
    // CTM: scale 1.7 then rotate ~20° and translate (applied through
    // the frame's root group).
    let (s, c) = (0.342f32, 0.94f32);
    let t = oxideav_core::Transform2D {
        a: 1.7 * c,
        b: 1.7 * s,
        c: -1.7 * s,
        d: 1.7 * c,
        e: 20.0,
        f: -4.0,
    };
    let inv = {
        let det = t.a * t.d - t.b * t.c;
        oxideav_core::Transform2D {
            a: t.d / det,
            b: -t.b / det,
            c: -t.c / det,
            d: t.a / det,
            e: (t.c * t.f - t.d * t.e) / det,
            f: (t.b * t.e - t.a * t.f) / det,
        }
    };
    for space in [InterpolationSpace::Srgb, InterpolationSpace::LinearRgb] {
        for paint in &paints {
            let mut f = full_rect(w, h, paint.clone());
            // Draw a huge rectangle in user space so it covers the canvas
            // under the CTM.
            if let Node::Path(pn) = &mut f.root.children[0] {
                let mut p = Path::new();
                p.move_to(Point::new(-500.0, -500.0))
                    .line_to(Point::new(500.0, -500.0))
                    .line_to(Point::new(500.0, 500.0))
                    .line_to(Point::new(-500.0, 500.0))
                    .close();
                pn.path = p;
            }
            f.root.transform = t;
            let mut r = Renderer::new(w, h);
            r.color_interpolation = space;
            let out = r.render(&f);
            let d = &out.planes[0].data;
            let mut identical = 0;
            for y in 0..h {
                for x in 0..w {
                    let u = inv.apply(Point::new(x as f32 + 0.5, y as f32 + 0.5));
                    let e = sample_paint_in(paint, u.x, u.y, space);
                    let g = px(d, w, x, y);
                    for (k, (gv, ev)) in g.iter().zip([e.r, e.g, e.b, e.a]).enumerate() {
                        assert!(
                            (*gv as i32 - ev as i32).abs() <= 1,
                            "{space:?} ({x},{y}) ch{k}: got {g:?} exact {e:?}"
                        );
                    }
                    identical += (g == [e.r, e.g, e.b, e.a]) as u32;
                }
            }
            assert!(
                identical * 100 >= w * h * 97,
                "{space:?}: only {identical} of {} pixels identical",
                w * h
            );
        }
    }
}

/// A wide, low-contrast ramp (gray 100 → 104 over 1 024 px) bands into
/// five flat steps when rounded. With [`GradientDither::Ordered`] every
/// 8×8 block's mean tracks the exact ramp (within ~1/16 level), the
/// output never strays more than one level from the exact value, and
/// rounding stays the default.
#[test]
fn ordered_dither_tracks_the_exact_ramp() {
    let (w, h) = (1024u32, 8u32);
    let paint = Paint::LinearGradient(LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(w as f32, 0.0),
        stops: vec![
            GradientStop::new(0.0, Rgba::opaque(100, 100, 100)),
            GradientStop::new(1.0, Rgba::opaque(104, 104, 104)),
        ],
        spread: SpreadMethod::Pad,
    });
    let frame = full_rect(w, h, paint);
    assert_eq!(Renderer::new(w, h).gradient_dither, GradientDither::None);
    let plain = Renderer::new(w, h).render(&frame);
    let dith = Renderer::new(w, h)
        .with_gradient_dither(GradientDither::Ordered)
        .render(&frame);
    let exact = |x: u32| 100.0 + 4.0 * (x as f64 + 0.5) / w as f64;
    let mut plain_err = 0.0f64;
    let mut dith_err = 0.0f64;
    for bx in 0..w / 8 {
        let (mut sp, mut sd, mut se) = (0.0, 0.0, 0.0);
        for y in 0..8 {
            for x in bx * 8..bx * 8 + 8 {
                let p = px(&plain.planes[0].data, w, x, y)[0] as f64;
                let d = px(&dith.planes[0].data, w, x, y)[0] as f64;
                assert!((d - exact(x)).abs() < 1.0, "x={x}: dithered {d}");
                sp += p;
                sd += d;
                se += exact(x);
            }
        }
        plain_err = plain_err.max((sp - se).abs() / 64.0);
        dith_err = dith_err.max((sd - se).abs() / 64.0);
    }
    assert!(dith_err < 0.07, "dithered block mean error {dith_err}");
    assert!(
        plain_err > 0.4,
        "rounded ramp should band, error {plain_err}"
    );
}
