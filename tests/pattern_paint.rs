//! Integration tests for the `<pattern>` paint server (SVG 2 §14.3) —
//! tiled fill / stroke through [`Renderer::fill_path_with_pattern`] and
//! [`Renderer::stroke_path_with_pattern`].

use oxideav_core::{
    FillRule, Node, Paint, Path, PathNode, Point, Rgba, Stroke, Transform2D, VideoFrame,
};
use oxideav_raster::{ImageFilter, Pattern, Renderer};

const RED: Rgba = Rgba::opaque(255, 0, 0);
const BLUE: Rgba = Rgba::opaque(0, 0, 255);

fn rect_path(x: f32, y: f32, w: f32, h: f32) -> Path {
    let mut p = Path::new();
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
    p
}

fn rect_node(x: f32, y: f32, w: f32, h: f32, color: Rgba) -> Node {
    Node::Path(PathNode {
        path: rect_path(x, y, w, h),
        fill: Some(Paint::Solid(color)),
        fill_rule: FillRule::NonZero,
        stroke: None,
    })
}

/// A 10×10 tile of two 5-wide vertical stripes: red on the left,
/// blue on the right (tile-origin-relative coordinates).
fn stripe_pattern() -> Pattern {
    Pattern::new(0.0, 0.0, 10.0, 10.0)
        .with_child(rect_node(0.0, 0.0, 5.0, 10.0, RED))
        .with_child(rect_node(5.0, 0.0, 5.0, 10.0, BLUE))
}

/// An exact-pixel renderer: no supersampling, nearest tile sampling.
fn exact_renderer(w: u32, h: u32) -> Renderer {
    let mut r = Renderer::new(w, h);
    r.supersampling = 1;
    r.image_filter = ImageFilter::Nearest;
    r
}

fn pixel(frame: &VideoFrame, x: u32, y: u32) -> Rgba {
    let plane = &frame.planes[0];
    let i = (y as usize) * plane.stride + (x as usize) * 4;
    Rgba::new(
        plane.data[i],
        plane.data[i + 1],
        plane.data[i + 2],
        plane.data[i + 3],
    )
}

#[test]
fn fill_tiles_periodically_across_the_canvas() {
    let r = exact_renderer(40, 20);
    let out = r.fill_path_with_pattern(
        &rect_path(0.0, 0.0, 40.0, 20.0),
        FillRule::NonZero,
        &stripe_pattern(),
        Transform2D::identity(),
    );
    // Stripe parity: x in [0,5) red, [5,10) blue, then repeat.
    for y in [0u32, 9, 19] {
        for x in 0..40u32 {
            let expect = if (x / 5) % 2 == 0 { RED } else { BLUE };
            assert_eq!(pixel(&out, x, y), expect, "pixel ({x}, {y})");
        }
    }
    // Strict periodicity: every pixel equals its tile-translated twin.
    for y in 0..20u32 {
        for x in 0..30u32 {
            assert_eq!(
                pixel(&out, x, y),
                pixel(&out, x + 10, y),
                "period violation at ({x}, {y})"
            );
        }
    }
}

#[test]
fn paint_is_confined_to_the_filled_path() {
    let r = exact_renderer(40, 20);
    let out = r.fill_path_with_pattern(
        &rect_path(10.0, 5.0, 10.0, 10.0),
        FillRule::NonZero,
        &stripe_pattern(),
        Transform2D::identity(),
    );
    // Inside: painted with the stripe phase of the *canvas* (pattern
    // space is user space, not path-local), so x=12 is red (12/5 = 2,
    // even stripe).
    assert_eq!(pixel(&out, 12, 10), RED);
    assert_eq!(pixel(&out, 17, 10), BLUE);
    // Outside the path: untouched (transparent background).
    assert_eq!(pixel(&out, 2, 10), Rgba::new(0, 0, 0, 0));
    assert_eq!(pixel(&out, 30, 10), Rgba::new(0, 0, 0, 0));
    assert_eq!(pixel(&out, 12, 2), Rgba::new(0, 0, 0, 0));
}

#[test]
fn degenerate_tile_paints_nothing() {
    let r = exact_renderer(20, 20);
    for (w, h) in [(0.0f32, 10.0f32), (10.0, 0.0), (-5.0, 10.0)] {
        let pat = Pattern::new(0.0, 0.0, w, h).with_child(rect_node(0.0, 0.0, 5.0, 5.0, RED));
        let out = r.fill_path_with_pattern(
            &rect_path(0.0, 0.0, 20.0, 20.0),
            FillRule::NonZero,
            &pat,
            Transform2D::identity(),
        );
        for y in 0..20u32 {
            for x in 0..20u32 {
                assert_eq!(
                    pixel(&out, x, y),
                    Rgba::new(0, 0, 0, 0),
                    "({w}×{h}) tile painted ({x}, {y})"
                );
            }
        }
    }
}

#[test]
fn tile_origin_shifts_the_phase() {
    // §14.3: tiles start at (x + m·width, y + n·height). Moving the
    // tile origin to x=5 swaps the stripe parity on the canvas.
    let r = exact_renderer(40, 10);
    let mut pat = stripe_pattern();
    pat.x = 5.0;
    let out = r.fill_path_with_pattern(
        &rect_path(0.0, 0.0, 40.0, 10.0),
        FillRule::NonZero,
        &pat,
        Transform2D::identity(),
    );
    // Canvas x=2 → pattern-local u = (2.5 − 5) mod 10 = 7.5 → blue.
    assert_eq!(pixel(&out, 2, 5), BLUE);
    // Canvas x=7 → u = 2.5 → red.
    assert_eq!(pixel(&out, 7, 5), RED);
}

#[test]
fn pattern_transform_translation_shifts_the_phase() {
    let r = exact_renderer(40, 10);
    let pat = stripe_pattern().with_transform(Transform2D::translate(5.0, 0.0));
    let out = r.fill_path_with_pattern(
        &rect_path(0.0, 0.0, 40.0, 10.0),
        FillRule::NonZero,
        &pat,
        Transform2D::identity(),
    );
    // Same phase swap as moving the tile origin.
    assert_eq!(pixel(&out, 2, 5), BLUE);
    assert_eq!(pixel(&out, 7, 5), RED);
}

#[test]
fn pattern_transform_rotation_turns_stripes() {
    // Rotating the pattern by 90° turns vertical stripes into
    // horizontal ones: constant along x, alternating along y.
    let r = exact_renderer(40, 40);
    let pat = stripe_pattern().with_transform(Transform2D::rotate(std::f32::consts::FRAC_PI_2));
    let out = r.fill_path_with_pattern(
        &rect_path(0.0, 0.0, 40.0, 40.0),
        FillRule::NonZero,
        &pat,
        Transform2D::identity(),
    );
    for y in [2u32, 7, 12, 22] {
        let row_color = pixel(&out, 0, y);
        for x in 1..40u32 {
            assert_eq!(
                pixel(&out, x, y),
                row_color,
                "row {y} not constant at x={x}"
            );
        }
    }
    // Adjacent stripe rows differ.
    assert_ne!(pixel(&out, 10, 2), pixel(&out, 10, 7));
    // Full period along y.
    for y in 0..30u32 {
        assert_eq!(
            pixel(&out, 10, y),
            pixel(&out, 10, y + 10),
            "y-period at {y}"
        );
    }
}

#[test]
fn device_scale_magnifies_the_tile_and_doubles_the_period() {
    let r = exact_renderer(40, 10);
    let out = r.fill_path_with_pattern(
        &rect_path(0.0, 0.0, 20.0, 5.0),
        FillRule::NonZero,
        &stripe_pattern(),
        Transform2D::scale(2.0, 2.0),
    );
    // User-space 20×5 rect covers the whole 40×10 canvas at 2×; the
    // 10-unit tile spans 20 device pixels, so stripes are 10 px wide.
    for x in 0..40u32 {
        let expect = if (x / 10) % 2 == 0 { RED } else { BLUE };
        assert_eq!(pixel(&out, x, 5), expect, "pixel ({x}, 5)");
    }
}

#[test]
fn stroke_paints_pattern_only_on_stroke_coverage() {
    let r = exact_renderer(40, 20);
    let mut line = Path::new();
    line.move_to(Point::new(0.0, 10.0))
        .line_to(Point::new(40.0, 10.0));
    let stroke = Stroke::solid(8.0, Rgba::opaque(9, 9, 9)); // paint ignored
    let out =
        r.stroke_path_with_pattern(&line, &stroke, &stripe_pattern(), Transform2D::identity());
    // On the stroke band (y in [6, 14)): stripe colors in canvas phase.
    assert_eq!(pixel(&out, 2, 10), RED);
    assert_eq!(pixel(&out, 7, 10), BLUE);
    assert_eq!(pixel(&out, 12, 10), RED);
    // Off the band: untouched.
    assert_eq!(pixel(&out, 2, 2), Rgba::new(0, 0, 0, 0));
    assert_eq!(pixel(&out, 2, 18), Rgba::new(0, 0, 0, 0));
}

#[test]
fn bilinear_filtering_keeps_a_solid_tile_uniform() {
    // A tile filled edge-to-edge with one color must tile into a
    // perfectly uniform field — wrap-around bilinear taps always read
    // the same color, so no seam darkening/lightening can appear.
    let mut r = Renderer::new(40, 20);
    r.supersampling = 1;
    r.image_filter = ImageFilter::Bilinear;
    let pat = Pattern::new(0.0, 0.0, 10.0, 10.0).with_child(rect_node(0.0, 0.0, 10.0, 10.0, RED));
    let out = r.fill_path_with_pattern(
        &rect_path(0.0, 0.0, 40.0, 20.0),
        FillRule::NonZero,
        &pat,
        Transform2D::identity(),
    );
    for y in 0..20u32 {
        for x in 0..40u32 {
            assert_eq!(pixel(&out, x, y), RED, "seam artifact at ({x}, {y})");
        }
    }
}

#[test]
fn tile_content_overflow_is_clipped_to_the_tile() {
    // §14.3.2: overflow is hidden — content past the tile rectangle
    // must not leak into neighbouring tiles. A red rect spilling 5
    // units past the 10-unit tile would otherwise cover the area that
    // stays transparent here.
    let r = exact_renderer(40, 10);
    let pat = Pattern::new(0.0, 0.0, 10.0, 10.0)
        // 15-wide content in a 10-wide tile: the overflowing 5 units
        // are clipped, NOT painted into the next tile.
        .with_child(rect_node(0.0, 0.0, 15.0, 4.0, RED));
    let out = r.fill_path_with_pattern(
        &rect_path(0.0, 0.0, 40.0, 10.0),
        FillRule::NonZero,
        &pat,
        Transform2D::identity(),
    );
    // Within each tile: top-left band red, rest transparent.
    assert_eq!(pixel(&out, 2, 2), RED);
    assert_eq!(pixel(&out, 12, 2), RED);
    // y below the band: transparent everywhere (no leak).
    assert_eq!(pixel(&out, 2, 7), Rgba::new(0, 0, 0, 0));
    assert_eq!(pixel(&out, 12, 7), Rgba::new(0, 0, 0, 0));
}
