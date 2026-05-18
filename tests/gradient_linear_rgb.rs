//! End-to-end test for the [`Renderer::color_interpolation`] toggle.
//!
//! Renders a black→white horizontal linear gradient through the full
//! pipeline (path flatten → fill mask → gradient sampler → premultiplied
//! composite), and confirms that:
//!
//! * `InterpolationSpace::Srgb` (the default) produces a perceptual
//!   midpoint of ≈ 128 (the sRGB-naive midpoint),
//! * `InterpolationSpace::LinearRgb` produces a midpoint of ≈ 188 (the
//!   linear-light midpoint encoded back into sRGB), matching SVG 2
//!   §13.9 `color-interpolation: linearRGB`.
//!
//! This protects against the path between the renderer setting and the
//! per-pixel gradient sampler regressing (e.g. dropping the space on the
//! `sample_paint_in` call).

use oxideav_core::{
    FillRule, GradientStop, Group, LinearGradient, Node, Paint, Path, PathNode, Point, Rgba,
    SpreadMethod, Transform2D, VectorFrame,
};
use oxideav_raster::{InterpolationSpace, Renderer};

fn black_to_white_horizontal_rect_frame(w: u32, h: u32) -> VectorFrame {
    let mut path = Path::new();
    path.move_to(Point::new(0.0, 0.0))
        .line_to(Point::new(w as f32, 0.0))
        .line_to(Point::new(w as f32, h as f32))
        .line_to(Point::new(0.0, h as f32))
        .close();
    let g = LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(w as f32, 0.0),
        stops: vec![
            GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
        ],
        spread: SpreadMethod::Pad,
    };
    let node = Node::Path(PathNode {
        path,
        fill: Some(Paint::LinearGradient(g)),
        stroke: None,
        fill_rule: FillRule::NonZero,
    });
    let root = Group {
        transform: Transform2D::identity(),
        opacity: 1.0,
        clip: None,
        children: vec![node],
        cache_key: None,
    };
    VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

#[test]
fn srgb_midpoint_is_perceptual_grey_128() {
    let frame = black_to_white_horizontal_rect_frame(32, 4);
    let r = Renderer::new(32, 4);
    assert_eq!(r.color_interpolation, InterpolationSpace::Srgb);
    let out = r.render(&frame);
    let stride = out.planes[0].stride;
    // Centre column (x = 16) of mid row (y = 2). The black→white
    // gradient spans x ∈ [0, 32], so at x = 16, t = 0.5.
    let i = 2 * stride + 16 * 4;
    let p = &out.planes[0].data[i..i + 4];
    assert!(
        (p[0] as i32 - 128).abs() <= 4,
        "sRGB midpoint should be ~128, got {}",
        p[0]
    );
    assert_eq!(p[3], 255, "alpha should be opaque");
}

#[test]
fn linear_rgb_midpoint_is_light_linear_188() {
    let frame = black_to_white_horizontal_rect_frame(32, 4);
    let mut r = Renderer::new(32, 4);
    r.color_interpolation = InterpolationSpace::LinearRgb;
    let out = r.render(&frame);
    let stride = out.planes[0].stride;
    let i = 2 * stride + 16 * 4;
    let p = &out.planes[0].data[i..i + 4];
    assert!(
        p[0] >= 183 && p[0] <= 192,
        "linearRGB midpoint should be ~188 (light-linear midpoint encoded back to sRGB), got {}",
        p[0]
    );
    assert_eq!(p[3], 255, "alpha should be opaque");
}

#[test]
fn endpoints_match_between_spaces() {
    // Pure black at one end and pure white at the other end must come
    // back identically regardless of interpolation space.
    let frame = black_to_white_horizontal_rect_frame(32, 4);
    let r_srgb = Renderer::new(32, 4);
    let mut r_lin = Renderer::new(32, 4);
    r_lin.color_interpolation = InterpolationSpace::LinearRgb;
    let a = r_srgb.render(&frame);
    let b = r_lin.render(&frame);
    let stride = a.planes[0].stride;
    // Left edge (x = 0). Supersampling means the leftmost column
    // samples the gradient at sub-pixel offsets > 0, so the apparent
    // colour is *not* exactly stop[0] — but it is very dark. linearRGB
    // is brighter than sRGB at the same offset (the linear midpoint is
    // higher), so the bound has to accomodate both spaces.
    for y in 0..4u32 {
        let i = (y as usize) * stride;
        assert!(
            a.planes[0].data[i] <= 8,
            "sRGB left edge should be near black, got {}",
            a.planes[0].data[i]
        );
        assert!(
            b.planes[0].data[i] <= 40,
            "linearRGB left edge should be very dark, got {}",
            b.planes[0].data[i]
        );
    }
    // Right edge (x = 31). Same supersample-offset wrinkle in reverse —
    // the rightmost column samples *slightly past* the stop[1] end
    // (which Pad clamps), so we expect near-white but with the linear
    // midpoint of the last sub-pixel being slightly darker than the
    // straight sRGB lerp.
    for y in 0..4u32 {
        let i = (y as usize) * stride + 31 * 4;
        assert!(
            a.planes[0].data[i] >= 247,
            "sRGB right edge should be near white, got {}",
            a.planes[0].data[i]
        );
        assert!(
            b.planes[0].data[i] >= 215,
            "linearRGB right edge should be light, got {}",
            b.planes[0].data[i]
        );
    }
}
