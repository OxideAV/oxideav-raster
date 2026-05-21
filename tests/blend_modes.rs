//! End-to-end integration tests for the PDF 32000-1:2008 §11.3.5 /
//! W3C Compositing-1 §10 standard separable blend modes wired into
//! [`oxideav_raster::Renderer::blend_mode`].
//!
//! Each test stacks two opaque rectangles (red backdrop on a transparent
//! canvas, then a paint-coloured rectangle on top) and checks the
//! resulting pixel against the spec formula evaluated by hand. The
//! per-channel `B(Cb, Cs)` math is unit-tested inside `src/blend.rs`;
//! these tests guard the renderer integration — that the field on
//! `Renderer` actually reaches the composite path, and that the formula
//! sees the expected backdrop pixel after the prior paint pass.

use oxideav_core::{FillRule, Group, Node, Paint, Path, PathNode, Point, Rgba, VectorFrame};
use oxideav_raster::{BlendMode, Renderer};

fn frame_with(w: u32, h: u32, root: Group) -> VectorFrame {
    VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

fn rect_node(x: f32, y: f32, w: f32, h: f32, fill: Rgba) -> Node {
    let mut p = Path::new();
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
    Node::Path(PathNode {
        path: p,
        fill: Some(Paint::Solid(fill)),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
}

fn stacked_scene(backdrop: Rgba, source: Rgba) -> VectorFrame {
    // Two fully-overlapping 8×8 rectangles centred well away from the
    // canvas edge so the centre pixel is the pure interior of both fills
    // (no AA contribution to the byte we read).
    let mut root = Group::default();
    root.children
        .push(rect_node(0.0, 0.0, 16.0, 16.0, backdrop));
    root.children.push(rect_node(0.0, 0.0, 16.0, 16.0, source));
    frame_with(16, 16, root)
}

/// Read the centre pixel of the rendered canvas.
fn centre_pixel(r: &Renderer, scene: &VectorFrame) -> Rgba {
    let out = r.render(scene);
    let stride = out.planes[0].stride;
    let i = 8 * stride + 8 * 4;
    let d = &out.planes[0].data;
    Rgba::new(d[i], d[i + 1], d[i + 2], d[i + 3])
}

#[test]
fn normal_overwrites_backdrop_for_opaque_source() {
    // Default mode: source-over. Opaque source replaces opaque backdrop.
    let r = Renderer::new(16, 16);
    let scene = stacked_scene(Rgba::opaque(80, 120, 200), Rgba::opaque(220, 40, 60));
    assert_eq!(centre_pixel(&r, &scene), Rgba::opaque(220, 40, 60));
}

#[test]
fn multiply_darkens_to_per_channel_product() {
    // Multiply with αb = αs = 1 simplifies to B(Cb, Cs) = Cb · Cs;
    // αr = 1 and Cr = Cb · Cs per channel after rounding to bytes.
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::Multiply;
    let cb = Rgba::opaque(200, 100, 50);
    let cs = Rgba::opaque(128, 128, 128);
    let scene = stacked_scene(cb, cs);
    let got = centre_pixel(&r, &scene);
    // Expected channel values, rounded the same way the renderer does
    // (round-to-nearest from the f32 path).
    let er = ((cb.r as f32 / 255.0) * (cs.r as f32 / 255.0) * 255.0).round() as u8;
    let eg = ((cb.g as f32 / 255.0) * (cs.g as f32 / 255.0) * 255.0).round() as u8;
    let eb = ((cb.b as f32 / 255.0) * (cs.b as f32 / 255.0) * 255.0).round() as u8;
    assert_eq!(got, Rgba::opaque(er, eg, eb));
    // Sanity: the result is darker than either input on every channel.
    assert!(got.r <= cb.r.min(cs.r) + 1);
    assert!(got.g <= cb.g.min(cs.g) + 1);
    assert!(got.b <= cb.b.min(cs.b) + 1);
}

#[test]
fn screen_with_opaque_white_becomes_white() {
    // Screen(Cb, 1) = Cb + 1 − Cb·1 = 1. Opaque white source over any
    // opaque backdrop yields opaque white.
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::Screen;
    let scene = stacked_scene(Rgba::opaque(80, 120, 200), Rgba::opaque(255, 255, 255));
    assert_eq!(centre_pixel(&r, &scene), Rgba::opaque(255, 255, 255));
}

#[test]
fn screen_with_opaque_black_is_identity() {
    // Screen(Cb, 0) = Cb. Opaque black source must leave the backdrop
    // unchanged.
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::Screen;
    let cb = Rgba::opaque(80, 120, 200);
    let scene = stacked_scene(cb, Rgba::opaque(0, 0, 0));
    assert_eq!(centre_pixel(&r, &scene), cb);
}

#[test]
fn darken_picks_channelwise_min() {
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::Darken;
    let cb = Rgba::opaque(200, 50, 100);
    let cs = Rgba::opaque(100, 200, 50);
    let scene = stacked_scene(cb, cs);
    assert_eq!(centre_pixel(&r, &scene), Rgba::opaque(100, 50, 50));
}

#[test]
fn lighten_picks_channelwise_max() {
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::Lighten;
    let cb = Rgba::opaque(200, 50, 100);
    let cs = Rgba::opaque(100, 200, 50);
    let scene = stacked_scene(cb, cs);
    assert_eq!(centre_pixel(&r, &scene), Rgba::opaque(200, 200, 100));
}

#[test]
fn difference_with_opaque_white_inverts_backdrop() {
    // Difference(Cb, 1) = 1 − Cb; opaque white over an opaque backdrop
    // gives the channelwise inverse.
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::Difference;
    let cb = Rgba::opaque(80, 120, 200);
    let scene = stacked_scene(cb, Rgba::opaque(255, 255, 255));
    assert_eq!(centre_pixel(&r, &scene), Rgba::opaque(175, 135, 55));
}

#[test]
fn exclusion_midgrey_over_midgrey_is_half() {
    // Exclusion(0.5, 0.5) = 0.5 + 0.5 − 2·0.25 = 0.5.
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::Exclusion;
    let scene = stacked_scene(Rgba::opaque(128, 128, 128), Rgba::opaque(128, 128, 128));
    let got = centre_pixel(&r, &scene);
    // Allow ±1 due to integer rounding.
    assert!((got.r as i32 - 128).abs() <= 1);
    assert!((got.g as i32 - 128).abs() <= 1);
    assert!((got.b as i32 - 128).abs() <= 1);
    assert_eq!(got.a, 255);
}

#[test]
fn hardlight_mid_grey_source_returns_backdrop_times_two_minus_one_capped() {
    // HardLight(Cb, 0.5) lies on the boundary; the implementation
    // takes the ≤ 0.5 branch → Multiply(Cb, 1) = Cb. Opaque backdrop +
    // opaque mid-grey source under HardLight must equal the backdrop.
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::HardLight;
    let cb = Rgba::opaque(60, 180, 120);
    let scene = stacked_scene(cb, Rgba::opaque(128, 128, 128));
    let got = centre_pixel(&r, &scene);
    // The exact spec answer is Multiply(Cb, 2·0.5) = Cb, but the
    // rounding through f32 may slip one channel by a count.
    assert!((got.r as i32 - cb.r as i32).abs() <= 1);
    assert!((got.g as i32 - cb.g as i32).abs() <= 1);
    assert!((got.b as i32 - cb.b as i32).abs() <= 1);
}

#[test]
fn softlight_with_mid_grey_source_is_identity() {
    // SoftLight(Cb, 0.5) sits on the boundary; the ≤ 0.5 branch is
    // Cb − (1 − 1)·… = Cb. Opaque mid-grey source must therefore be
    // the identity over an opaque backdrop.
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::SoftLight;
    let cb = Rgba::opaque(60, 180, 120);
    let scene = stacked_scene(cb, Rgba::opaque(128, 128, 128));
    let got = centre_pixel(&r, &scene);
    assert!((got.r as i32 - cb.r as i32).abs() <= 1);
    assert!((got.g as i32 - cb.g as i32).abs() <= 1);
    assert!((got.b as i32 - cb.b as i32).abs() <= 1);
}

#[test]
fn colordodge_white_source_becomes_white() {
    // ColorDodge(Cb, 1) = 1 for any Cb > 0.
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::ColorDodge;
    let scene = stacked_scene(Rgba::opaque(40, 80, 200), Rgba::opaque(255, 255, 255));
    assert_eq!(centre_pixel(&r, &scene), Rgba::opaque(255, 255, 255));
}

#[test]
fn colorburn_black_source_becomes_black() {
    // ColorBurn(Cb, 0) = 0 for any Cb < 1.
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::ColorBurn;
    let scene = stacked_scene(Rgba::opaque(40, 80, 200), Rgba::opaque(0, 0, 0));
    assert_eq!(centre_pixel(&r, &scene), Rgba::opaque(0, 0, 0));
}

#[test]
fn overlay_equals_hardlight_with_swapped_args() {
    // Overlay(Cb, Cs) ≡ HardLight(Cs, Cb): the renderer applied with
    // BlendMode::Overlay at (cb, cs) must match the renderer applied
    // with BlendMode::HardLight at (cs, cb) — pixel-for-pixel.
    let cb = Rgba::opaque(60, 180, 120);
    let cs = Rgba::opaque(220, 100, 40);

    let mut r_ov = Renderer::new(16, 16);
    r_ov.blend_mode = BlendMode::Overlay;
    let scene_ov = stacked_scene(cb, cs);
    let p_ov = centre_pixel(&r_ov, &scene_ov);

    let mut r_hl = Renderer::new(16, 16);
    r_hl.blend_mode = BlendMode::HardLight;
    // Swap the order of the two rectangles to flip the Cb/Cs identity.
    let scene_hl = stacked_scene(cs, cb);
    let p_hl = centre_pixel(&r_hl, &scene_hl);

    // Allow ±1 per channel for integer-rounding drift.
    assert!((p_ov.r as i32 - p_hl.r as i32).abs() <= 1);
    assert!((p_ov.g as i32 - p_hl.g as i32).abs() <= 1);
    assert!((p_ov.b as i32 - p_hl.b as i32).abs() <= 1);
}

#[test]
fn semi_transparent_source_alpha_blends_correctly_under_multiply() {
    // αb = 1, αs = 0.5, Multiply → αr = 1, Cr = 0.5·Cs + 0.5·(Cb·Cs).
    // Plug in Cb = (0.8, 0.4, 0.2), Cs = (0.5, 0.5, 0.5):
    //   B = Cb·Cs = (0.40, 0.20, 0.10)
    //   Cr = 0.5·(0.5,0.5,0.5) + 0.5·(0.40,0.20,0.10)
    //      = (0.25, 0.25, 0.25) + (0.20, 0.10, 0.05)
    //      = (0.45, 0.35, 0.30)  →  bytes (115, 89, 77)
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::Multiply;
    let cb = Rgba::opaque(204, 102, 51); // 0.8 / 0.4 / 0.2 approx
    let cs = Rgba::new(128, 128, 128, 128); // 0.5 alpha
    let scene = stacked_scene(cb, cs);
    let got = centre_pixel(&r, &scene);

    // Recompute the expected value the exact same way the
    // implementation does (float-domain, rounded to bytes) so the test
    // tracks the formula rather than a hand-rounded snapshot. The byte
    // backdrop after a Normal paint of `cb` is exactly `cb`. The source
    // pixel reaches composite with combined alpha ≈ 128/255.
    let cb_f = [
        cb.r as f32 / 255.0,
        cb.g as f32 / 255.0,
        cb.b as f32 / 255.0,
    ];
    let cs_f = [
        cs.r as f32 / 255.0,
        cs.g as f32 / 255.0,
        cs.b as f32 / 255.0,
    ];
    let ab = 1.0_f32;
    let as_ = 128.0 / 255.0;
    let ar = as_ + ab * (1.0 - as_);
    let b_ch = [cb_f[0] * cs_f[0], cb_f[1] * cs_f[1], cb_f[2] * cs_f[2]];
    let mix = [
        (1.0 - ab) * cs_f[0] + ab * b_ch[0],
        (1.0 - ab) * cs_f[1] + ab * b_ch[1],
        (1.0 - ab) * cs_f[2] + ab * b_ch[2],
    ];
    let s_over_r = as_ / ar;
    let cr_f = [
        (1.0 - s_over_r) * cb_f[0] + s_over_r * mix[0],
        (1.0 - s_over_r) * cb_f[1] + s_over_r * mix[1],
        (1.0 - s_over_r) * cb_f[2] + s_over_r * mix[2],
    ];
    let er = (cr_f[0] * 255.0).round() as i32;
    let eg = (cr_f[1] * 255.0).round() as i32;
    let eb = (cr_f[2] * 255.0).round() as i32;
    assert!((got.r as i32 - er).abs() <= 1);
    assert!((got.g as i32 - eg).abs() <= 1);
    assert!((got.b as i32 - eb).abs() <= 1);
    assert_eq!(got.a, 255);
}

#[test]
fn non_normal_blend_does_not_affect_isolated_pixels() {
    // A blend mode only affects pixels where both the backdrop and
    // source paints touch. Where only the source touches (no backdrop
    // underneath), the result must equal `(1 − αb)·Cs + αb·B = Cs`
    // — same as Normal — so a single rectangle painted with Multiply
    // onto a transparent canvas should be the same opaque source.
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::Multiply;
    let mut root = Group::default();
    root.children
        .push(rect_node(0.0, 0.0, 16.0, 16.0, Rgba::opaque(200, 100, 50)));
    let scene = frame_with(16, 16, root);
    let out = r.render(&scene);
    let stride = out.planes[0].stride;
    let i = 8 * stride + 8 * 4;
    let p = &out.planes[0].data[i..i + 4];
    assert_eq!(p, &[200, 100, 50, 255]);
}

#[test]
fn changing_blend_mode_changes_pixel_value() {
    // Sanity guard: the same scene rendered with Normal vs Multiply
    // must differ when the source is not white (which would be a
    // multiply identity).
    let scene = stacked_scene(Rgba::opaque(200, 100, 50), Rgba::opaque(128, 128, 128));

    let r_normal = Renderer::new(16, 16);
    let p_normal = centre_pixel(&r_normal, &scene);

    let mut r_multiply = Renderer::new(16, 16);
    r_multiply.blend_mode = BlendMode::Multiply;
    let p_multiply = centre_pixel(&r_multiply, &scene);

    assert_ne!(p_normal, p_multiply, "blend mode must change the result");
    assert_eq!(p_normal, Rgba::opaque(128, 128, 128));
}

#[test]
fn blend_mode_field_clones_through_renderer() {
    // The Renderer is Clone; a cloned renderer must carry the same
    // blend mode (the cache is shared but the configuration is
    // copied).
    let mut r = Renderer::new(16, 16);
    r.blend_mode = BlendMode::SoftLight;
    let r2 = r.clone();
    assert_eq!(r2.blend_mode, BlendMode::SoftLight);
}
