//! Property-based rendering-correctness tests.
//!
//! These exercise *invariants* of the renderer that must hold across a
//! large family of randomly-constructed scenes, rather than pinning a
//! single golden pixel. No external property-testing crate is used: a
//! small deterministic LCG (so failures are reproducible and CI is
//! flake-free) drives the scene generation, and each generated scene is
//! checked against an invariant derived from the SVG / PDF rendering
//! model.
//!
//! The invariants checked here:
//!
//! 1. **Opaque-fill bound.** Painting an opaque colour anywhere only
//!    ever *raises* a destination pixel's alpha; it never lowers it.
//!    (Premultiplied source-over: `ar = as + ad·(1 − as) ≥ ad`.)
//! 2. **Group-opacity alpha scaling.** A single opaque rectangle inside
//!    a group of opacity α produces a covered-pixel alpha ≈ 255·α.
//! 3. **Clip containment.** A clipped fill never paints a pixel whose
//!    centre lies outside the clip rectangle.
//! 4. **`meet` letterbox containment.** Under `xMidYMid meet`, a
//!    full-viewBox opaque fill never paints a pixel outside the
//!    uniformly-scaled, centred content rectangle.
//! 5. **`slice` full coverage.** Under any `slice` alignment, a
//!    full-viewBox opaque fill covers every pixel of the canvas.
//! 6. **`none` full coverage.** Under `none`, a full-viewBox opaque
//!    fill covers every pixel of the canvas (legacy stretch).
//! 7. **Determinism.** Rendering the same scene twice is byte-identical.

use oxideav_core::{
    FillRule, Group, Node, Paint, Path, PathNode, Point, Rgba, Transform2D, VectorFrame, ViewBox,
};
use oxideav_raster::{AspectRatioAlign, MeetOrSlice, PreserveAspectRatio, Renderer};

/// Minimal reproducible PRNG (a 64-bit LCG, Numerical-Recipes constants).
/// Deterministic by seed so a failing property prints a reproducible
/// scene and CI never flakes.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform `f32` in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let t = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        lo + t * (hi - lo)
    }
    /// Uniform integer in `[lo, hi]`.
    fn int(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as u32
    }
}

fn rect_path(x: f32, y: f32, w: f32, h: f32) -> Path {
    let mut p = Path::new();
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
    p
}

fn rect_node(x: f32, y: f32, w: f32, h: f32, fill: Rgba) -> Node {
    Node::Path(PathNode {
        path: rect_path(x, y, w, h),
        fill: Some(Paint::Solid(fill)),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
}

fn frame(w: u32, h: u32, root: Group, view_box: Option<ViewBox>) -> VectorFrame {
    VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

fn alpha_at(out: &oxideav_core::VideoFrame, x: u32, y: u32) -> u8 {
    let stride = out.planes[0].stride;
    out.planes[0].data[(y as usize) * stride + (x as usize) * 4 + 3]
}

/// 1. Opaque painting only ever raises a pixel's alpha. Stack several
///    random opaque rects on a random background and confirm every
///    pixel's alpha is ≥ the background alpha.
#[test]
fn opaque_fill_never_lowers_alpha() {
    let mut rng = Lcg::new(0x0102_0304);
    for _ in 0..64 {
        let (w, h) = (rng.int(8, 40), rng.int(8, 40));
        let bg_a = rng.int(0, 255) as u8;
        let mut r = Renderer::new(w, h);
        r.background = Rgba::new(
            rng.int(0, 255) as u8,
            rng.int(0, 255) as u8,
            rng.int(0, 255) as u8,
            bg_a,
        );
        let mut root = Group::default();
        for _ in 0..rng.int(1, 5) {
            let rx = rng.range(0.0, w as f32);
            let ry = rng.range(0.0, h as f32);
            let rw = rng.range(1.0, w as f32);
            let rh = rng.range(1.0, h as f32);
            root.children.push(rect_node(
                rx,
                ry,
                rw,
                rh,
                Rgba::opaque(
                    rng.int(0, 255) as u8,
                    rng.int(0, 255) as u8,
                    rng.int(0, 255) as u8,
                ),
            ));
        }
        let out = r.render(&frame(w, h, root, None));
        for y in 0..h {
            for x in 0..w {
                assert!(
                    alpha_at(&out, x, y) >= bg_a,
                    "opaque paint lowered alpha at ({x},{y}): {} < bg {bg_a}",
                    alpha_at(&out, x, y)
                );
            }
        }
    }
}

/// 2. Group opacity scales covered-pixel alpha. An opaque rect that
///    fully covers the canvas, wrapped in a group of opacity α, must
///    produce alpha ≈ round(255·α) everywhere.
#[test]
fn group_opacity_scales_alpha() {
    let mut rng = Lcg::new(0x55aa_55aa);
    for _ in 0..48 {
        let (w, h) = (rng.int(4, 24), rng.int(4, 24));
        let alpha = rng.range(0.0, 1.0);
        let inner = rect_node(0.0, 0.0, w as f32, h as f32, Rgba::opaque(200, 100, 50));
        let group = Group::default().with_opacity(alpha).with_child(inner);
        let mut root = Group::default();
        root.children.push(Node::Group(group));
        let out = Renderer::new(w, h).render(&frame(w, h, root, None));
        let expect = (alpha * 255.0).round() as i32;
        // Sample the centre (fully covered, away from any AA edge).
        let got = alpha_at(&out, w / 2, h / 2) as i32;
        assert!(
            (got - expect).abs() <= 1,
            "group opacity {alpha}: centre alpha {got} != expected ~{expect}"
        );
    }
}

/// 3. A clip rectangle confines the fill: no pixel whose centre lies
///    strictly outside the clip rect is ever painted.
#[test]
fn clip_confines_fill_to_clip_rect() {
    let mut rng = Lcg::new(0xfeed_face);
    for _ in 0..48 {
        let (w, h) = (rng.int(16, 40), rng.int(16, 40));
        // A clip rect somewhere inside the canvas.
        let cx = rng.range(2.0, w as f32 * 0.5);
        let cy = rng.range(2.0, h as f32 * 0.5);
        let cw = rng.range(4.0, w as f32 * 0.5);
        let ch = rng.range(4.0, h as f32 * 0.5);
        // The fill covers the whole canvas; only the clip should show.
        let inner = rect_node(0.0, 0.0, w as f32, h as f32, Rgba::opaque(255, 0, 0));
        let group = Group::default()
            .with_clip(rect_path(cx, cy, cw, ch))
            .with_child(inner);
        let mut root = Group::default();
        root.children.push(Node::Group(group));
        let out = Renderer::new(w, h).render(&frame(w, h, root, None));
        for y in 0..h {
            for x in 0..w {
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;
                // A pixel more than one pixel outside the clip rect must
                // be untouched (the +1 guard band tolerates AA fringe on
                // the clip boundary).
                let outside =
                    px < cx - 1.0 || px > cx + cw + 1.0 || py < cy - 1.0 || py > cy + ch + 1.0;
                if outside {
                    assert_eq!(
                        alpha_at(&out, x, y),
                        0,
                        "clip leaked paint at ({x},{y}) outside rect \
                         ({cx},{cy},{cw},{ch})"
                    );
                }
            }
        }
    }
}

/// 4. Under `xMidYMid meet`, a full-viewBox opaque fill never paints a
///    pixel outside the uniformly-scaled, centred content rectangle.
#[test]
fn meet_letterbox_paints_only_inside_content_rect() {
    let mut rng = Lcg::new(0xabcd_1234);
    for _ in 0..64 {
        let (cw, ch) = (rng.int(8, 48), rng.int(8, 48));
        let (vbw, vbh) = (rng.range(4.0, 60.0), rng.range(4.0, 60.0));
        // Fill the entire viewBox opaque.
        let mut root = Group::default();
        root.children
            .push(rect_node(0.0, 0.0, vbw, vbh, Rgba::opaque(0, 0, 255)));
        let vb = ViewBox::new(0.0, 0.0, vbw, vbh);
        let out = Renderer::new(cw, ch).render(&frame(cw, ch, root, Some(vb)));
        // Compute the meet content rectangle in device pixels.
        let s = (cw as f32 / vbw).min(ch as f32 / vbh);
        let content_w = vbw * s;
        let content_h = vbh * s;
        let off_x = (cw as f32 - content_w) / 2.0;
        let off_y = (ch as f32 - content_h) / 2.0;
        for y in 0..ch {
            for x in 0..cw {
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;
                // Guard band of 1 px for AA on the content edge.
                let outside = px < off_x - 1.0
                    || px > off_x + content_w + 1.0
                    || py < off_y - 1.0
                    || py > off_y + content_h + 1.0;
                if outside {
                    assert_eq!(
                        alpha_at(&out, x, y),
                        0,
                        "meet painted outside content rect at ({x},{y}) \
                         (canvas {cw}x{ch}, viewBox {vbw}x{vbh}, \
                         content {content_w}x{content_h} @ {off_x},{off_y})"
                    );
                }
            }
        }
    }
}

/// 5. Under any `slice` alignment, a full-viewBox opaque fill covers
///    every pixel of the canvas (the content is scaled to the larger
///    factor, overspilling and cropping at the edge).
#[test]
fn slice_covers_every_canvas_pixel() {
    let mut rng = Lcg::new(0x7777_3333);
    let aligns = [
        AspectRatioAlign::XMinYMin,
        AspectRatioAlign::XMidYMid,
        AspectRatioAlign::XMaxYMax,
    ];
    for _ in 0..48 {
        let (cw, ch) = (rng.int(8, 40), rng.int(8, 40));
        let (vbw, vbh) = (rng.range(4.0, 50.0), rng.range(4.0, 50.0));
        let align = aligns[(rng.int(0, 2)) as usize];
        let mut root = Group::default();
        root.children
            .push(rect_node(0.0, 0.0, vbw, vbh, Rgba::opaque(10, 200, 10)));
        let vb = ViewBox::new(0.0, 0.0, vbw, vbh);
        let r = Renderer::new(cw, ch).with_preserve_aspect_ratio(PreserveAspectRatio {
            align,
            meet_or_slice: MeetOrSlice::Slice,
        });
        let out = r.render(&frame(cw, ch, root, Some(vb)));
        for y in 0..ch {
            for x in 0..cw {
                assert_eq!(
                    alpha_at(&out, x, y),
                    255,
                    "slice left a gap at ({x},{y}) (canvas {cw}x{ch}, \
                     viewBox {vbw}x{vbh}, align {align:?})"
                );
            }
        }
    }
}

/// 6. Under `none`, a full-viewBox opaque fill covers every pixel
///    (legacy non-uniform stretch fills the whole canvas exactly).
#[test]
fn none_stretch_covers_every_canvas_pixel() {
    let mut rng = Lcg::new(0x2468_ace0);
    for _ in 0..48 {
        let (cw, ch) = (rng.int(4, 40), rng.int(4, 40));
        let (vbw, vbh) = (rng.range(4.0, 50.0), rng.range(4.0, 50.0));
        let mut root = Group::default();
        root.children
            .push(rect_node(0.0, 0.0, vbw, vbh, Rgba::opaque(200, 200, 0)));
        let vb = ViewBox::new(0.0, 0.0, vbw, vbh);
        let r = Renderer::new(cw, ch).with_preserve_aspect_ratio(PreserveAspectRatio {
            align: AspectRatioAlign::None,
            meet_or_slice: MeetOrSlice::Meet,
        });
        let out = r.render(&frame(cw, ch, root, Some(vb)));
        for y in 0..ch {
            for x in 0..cw {
                assert_eq!(
                    alpha_at(&out, x, y),
                    255,
                    "none stretch left a gap at ({x},{y})"
                );
            }
        }
    }
}

/// 7. Rendering the same scene twice yields byte-identical output —
///    the renderer is a pure function of the scene + config (the shared
///    bitmap cache must not perturb results across calls).
#[test]
fn render_is_deterministic() {
    let mut rng = Lcg::new(0x1357_9bdf);
    for _ in 0..32 {
        let (w, h) = (rng.int(8, 40), rng.int(8, 40));
        let mut root = Group::default();
        for _ in 0..rng.int(1, 6) {
            // Mix plain fills and cache-keyed groups to exercise the
            // memoisation path too.
            let node = rect_node(
                rng.range(0.0, w as f32),
                rng.range(0.0, h as f32),
                rng.range(1.0, w as f32),
                rng.range(1.0, h as f32),
                Rgba::new(
                    rng.int(0, 255) as u8,
                    rng.int(0, 255) as u8,
                    rng.int(0, 255) as u8,
                    rng.int(0, 255) as u8,
                ),
            );
            if rng.int(0, 1) == 1 {
                let g = Group::default()
                    .with_cache_key(rng.next_u64())
                    .with_transform(Transform2D::translate(rng.range(-2.0, 2.0), 0.0))
                    .with_child(node);
                root.children.push(Node::Group(g));
            } else {
                root.children.push(node);
            }
        }
        let v = frame(w, h, root, None);
        let r = Renderer::new(w, h);
        let a = r.render(&v);
        let b = r.render(&v);
        assert_eq!(
            a.planes[0].data, b.planes[0].data,
            "render is non-deterministic for a {w}x{h} scene"
        );
    }
}
