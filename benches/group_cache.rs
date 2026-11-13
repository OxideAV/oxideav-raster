//! Criterion benchmarks for `Group::cache_key` rendering.
//!
//! The headline shape is a page of text as a glyph producer emits it:
//! ~300 glyph-sized groups (16 px em), each keyed by its glyph identity
//! and placed by a `translate(pen) ∘ scale(size / units_per_em)`
//! transform with fractional pen positions, on a 1206 × 2622 canvas.
//! It is measured cold (first frame, empty cache), warm (same renderer
//! again), across per-frame renderers that share one cache, and with
//! the keys stripped (the unkeyed baseline a keyed miss must not lose
//! to). A second scene does the same for a few large groups.
//!
//! Run with: `cargo bench -p oxideav-raster --bench group_cache`

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use oxideav_core::{
    FillRule, Group, Node, Paint, Path, PathNode, Point, Rgba, Stroke, Transform2D, VectorFrame,
};
use oxideav_raster::{GroupCache, Renderer};

const CANVAS_W: u32 = 1206;
const CANVAS_H: u32 = 2622;
const UNITS_PER_EM: f32 = 1000.0;
const EM_PX: f32 = 16.0;
const GLYPHS: usize = 300;
const DISTINCT_GLYPHS: u32 = 40;

/// A glyph-like outline in font units (Y up): an outer blob of
/// quadratic segments with a counter cut out of it, the proportions
/// varied by `id` so the page holds [`DISTINCT_GLYPHS`] different
/// shapes.
fn glyph_outline(id: u32) -> Path {
    let mut p = Path::new();
    let w = 380.0 + (id % 7) as f32 * 35.0;
    let h = 480.0 + (id % 5) as f32 * 55.0;
    let ring = |p: &mut Path, cx: f32, cy: f32, rx: f32, ry: f32, n: u32, reverse: bool| {
        for k in 0..=n {
            let k = if reverse { n - k } else { k };
            let a = k as f32 / n as f32 * std::f32::consts::TAU;
            let wobble = 1.0 + 0.08 * ((id + 3) as f32 * a).sin();
            let end = Point::new(cx + rx * wobble * a.cos(), cy + ry * wobble * a.sin());
            if (k == 0 && !reverse) || (k == n && reverse) {
                p.move_to(end);
            } else {
                let am = a + if reverse { 0.5 } else { -0.5 } * std::f32::consts::TAU / n as f32;
                let c = Point::new(cx + rx * 1.12 * am.cos(), cy + ry * 1.12 * am.sin());
                p.quad_to(c, end);
            }
        }
        p.close();
    };
    ring(&mut p, w * 0.5 + 40.0, h * 0.5, w * 0.5, h * 0.5, 10, false);
    ring(&mut p, w * 0.5 + 40.0, h * 0.5, w * 0.27, h * 0.3, 8, true);
    p
}

/// The text page. `keyed` tags every glyph group with its glyph id;
/// `shift` moves every pen position (a scroll between frames).
fn text_page(keyed: bool, shift: (f32, f32)) -> VectorFrame {
    let scale = EM_PX / UNITS_PER_EM;
    let mut root = Group::default();
    let (mut pen_x, mut baseline) = (24.0f32, 40.0f32);
    for i in 0..GLYPHS {
        let id = (i as u32 * 17 + (i as u32 / 7) * 5) % DISTINCT_GLYPHS;
        let advance = (460.0 + (id % 7) as f32 * 35.0) * scale;
        if pen_x + advance > CANVAS_W as f32 - 24.0 {
            pen_x = 24.0;
            baseline += 22.0;
        }
        let transform = Transform2D {
            a: scale,
            b: 0.0,
            c: 0.0,
            d: -scale,
            e: pen_x + shift.0,
            f: baseline + shift.1,
        };
        root.children.push(Node::Group(Group {
            transform,
            cache_key: keyed.then_some(0x6c79_7068_0000_0000 | id as u64),
            children: vec![Node::Path(PathNode {
                path: glyph_outline(id),
                fill: Some(Paint::Solid(Rgba::opaque(20, 20, 24))),
                stroke: None,
                fill_rule: FillRule::NonZero,
            })],
            ..Group::default()
        }));
        pen_x += advance;
    }
    frame(root)
}

/// Four ~600 px groups, each a filled panel, a stroked frame and a
/// dozen discs.
fn large_groups(keyed: bool) -> VectorFrame {
    let mut root = Group::default();
    for g in 0..4u32 {
        let mut children = Vec::new();
        let mut panel = Path::new();
        panel
            .move_to(Point::new(0.0, 40.0))
            .quad_to(Point::new(0.0, 0.0), Point::new(40.0, 0.0))
            .line_to(Point::new(520.0, 0.0))
            .quad_to(Point::new(560.0, 0.0), Point::new(560.0, 40.0))
            .line_to(Point::new(560.0, 560.0))
            .quad_to(Point::new(560.0, 600.0), Point::new(520.0, 600.0))
            .line_to(Point::new(40.0, 600.0))
            .quad_to(Point::new(0.0, 600.0), Point::new(0.0, 560.0))
            .close();
        children.push(Node::Path(PathNode {
            path: panel,
            fill: Some(Paint::Solid(Rgba::new(40 + 50 * g as u8, 90, 160, 255))),
            stroke: Some(Stroke::solid(6.0, Rgba::opaque(250, 250, 250))),
            fill_rule: FillRule::NonZero,
        }));
        for k in 0..12u32 {
            let (cx, cy, r) = (
                70.0 + (k % 4) as f32 * 140.0,
                90.0 + (k / 4) as f32 * 180.0,
                48.0,
            );
            let mut disc = Path::new();
            disc.move_to(Point::new(cx + r, cy));
            for q in 1..=8 {
                let a = q as f32 / 8.0 * std::f32::consts::TAU;
                let am = a - std::f32::consts::TAU / 16.0;
                disc.quad_to(
                    Point::new(cx + r * 1.0824 * am.cos(), cy + r * 1.0824 * am.sin()),
                    Point::new(cx + r * a.cos(), cy + r * a.sin()),
                );
            }
            disc.close();
            children.push(Node::Path(PathNode {
                path: disc,
                fill: Some(Paint::Solid(Rgba::new(255, 200, 40 + 15 * k as u8, 210))),
                stroke: None,
                fill_rule: FillRule::NonZero,
            }));
        }
        root.children.push(Node::Group(Group {
            transform: Transform2D::translate(
                20.5 + (g % 2) as f32 * 590.0,
                30.25 + (g / 2) as f32 * 640.0,
            ),
            cache_key: keyed.then_some(0xb16_0000 + g as u64),
            children,
            ..Group::default()
        }));
    }
    frame(root)
}

/// 200 small groups, each clipped to its own disc, inside a root group
/// clipped to a rounded panel (so every inner clip is intersected with
/// an inherited one), half of them translucent.
fn clipped_groups() -> VectorFrame {
    let disc = |cx: f32, cy: f32, r: f32| {
        let mut p = Path::new();
        p.move_to(Point::new(cx + r, cy));
        for q in 1..=8 {
            let a = q as f32 / 8.0 * std::f32::consts::TAU;
            let am = a - std::f32::consts::TAU / 16.0;
            p.quad_to(
                Point::new(cx + r * 1.0824 * am.cos(), cy + r * 1.0824 * am.sin()),
                Point::new(cx + r * a.cos(), cy + r * a.sin()),
            );
        }
        p.close();
        p
    };
    let mut panel = Path::new();
    panel
        .move_to(Point::new(60.0, 20.0))
        .line_to(Point::new(1146.0, 20.0))
        .quad_to(Point::new(1186.0, 20.0), Point::new(1186.0, 60.0))
        .line_to(Point::new(1186.0, 2560.0))
        .quad_to(Point::new(1186.0, 2600.0), Point::new(1146.0, 2600.0))
        .line_to(Point::new(60.0, 2600.0))
        .quad_to(Point::new(20.0, 2600.0), Point::new(20.0, 2560.0))
        .line_to(Point::new(20.0, 60.0))
        .quad_to(Point::new(20.0, 20.0), Point::new(60.0, 20.0))
        .close();
    let mut root = Group {
        clip: Some(panel),
        ..Group::default()
    };
    for i in 0..200u32 {
        let (x, y) = (
            30.0 + (i % 10) as f32 * 115.0,
            30.0 + (i / 10) as f32 * 128.0,
        );
        let mut square = Path::new();
        square
            .move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(64.0, 0.0))
            .line_to(Point::new(64.0, 64.0))
            .line_to(Point::new(0.0, 64.0))
            .close();
        root.children.push(Node::Group(Group {
            transform: Transform2D::translate(x, y),
            opacity: if i % 2 == 0 { 1.0 } else { 0.7 },
            clip: Some(disc(32.0, 32.0, 30.0)),
            children: vec![Node::Path(PathNode {
                path: square,
                fill: Some(Paint::Solid(Rgba::opaque(200, 60 + (i % 5) as u8 * 30, 90))),
                stroke: None,
                fill_rule: FillRule::NonZero,
            })],
            ..Group::default()
        }));
    }
    frame(root)
}

fn frame(root: Group) -> VectorFrame {
    VectorFrame {
        width: CANVAS_W as f32,
        height: CANVAS_H as f32,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

/// Time only the scene walk: the canvas clear (12.6 MB) is the same in
/// every variant, so each group is rendered through `render_node` on a
/// renderer whose canvas buffer cost is reported separately by
/// `canvas_clear_only`.
fn bench_group_cache(c: &mut Criterion) {
    let mut g = c.benchmark_group("group_cache");
    g.sample_size(20);

    let empty = frame(Group::default());
    g.bench_function("canvas_clear_only", |b| {
        let r = Renderer::new(CANVAS_W, CANVAS_H);
        b.iter(|| black_box(r.render(black_box(&empty))))
    });

    for (name, keyed_scene, plain_scene) in [
        (
            "glyphs300",
            text_page(true, (0.0, 0.0)),
            text_page(false, (0.0, 0.0)),
        ),
        ("large4", large_groups(true), large_groups(false)),
    ] {
        g.bench_function(format!("{name}_unkeyed"), |b| {
            let r = Renderer::new(CANVAS_W, CANVAS_H);
            b.iter(|| black_box(r.render(black_box(&plain_scene))))
        });
        // First frame: a renderer built for this frame, nothing cached.
        g.bench_function(format!("{name}_keyed_first_frame"), |b| {
            b.iter(|| {
                let r = Renderer::new(CANVAS_W, CANVAS_H);
                black_box(r.render(black_box(&keyed_scene)))
            })
        });
        // Steady state, long-lived renderer.
        g.bench_function(format!("{name}_keyed_steady_same_renderer"), |b| {
            let r = Renderer::new(CANVAS_W, CANVAS_H);
            let _ = r.render(&keyed_scene);
            b.iter(|| black_box(r.render(black_box(&keyed_scene))))
        });
        // Steady state, a renderer built per frame around one cache.
        g.bench_function(format!("{name}_keyed_steady_shared_cache"), |b| {
            let cache = GroupCache::new();
            let _ = Renderer::new(CANVAS_W, CANVAS_H)
                .with_group_cache(cache.clone())
                .render(&keyed_scene);
            b.iter(|| {
                let r = Renderer::new(CANVAS_W, CANVAS_H).with_group_cache(cache.clone());
                black_box(r.render(black_box(&keyed_scene)))
            })
        });
    }

    // Clip masks: 200 clipped groups under an inherited clip.
    g.bench_function("clipped_groups200_nested", |b| {
        let scene = clipped_groups();
        let r = Renderer::new(CANVAS_W, CANVAS_H);
        b.iter(|| black_box(r.render(black_box(&scene))))
    });

    // Steady state while the page scrolls: every frame moves every pen
    // position by a whole number of pixels vertically.
    g.bench_function("glyphs300_keyed_steady_scrolling", |b| {
        let frames: Vec<VectorFrame> = (0..8)
            .map(|k| text_page(true, (0.0, k as f32 * 3.0)))
            .collect();
        let r = Renderer::new(CANVAS_W, CANVAS_H);
        for f in &frames {
            let _ = r.render(f);
        }
        let mut k = 0usize;
        b.iter(|| {
            k = (k + 1) % frames.len();
            black_box(r.render(black_box(&frames[k])))
        })
    });
    g.finish();
}

criterion_group!(benches, bench_group_cache);
criterion_main!(benches);
