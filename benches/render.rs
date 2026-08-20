//! Criterion benchmarks for the core rasterisation hot loops.
//!
//! Companion to `benches/filter.rs` (see its header for the round-401
//! rationale). Scenarios:
//!
//!   - **flatten_star_128pt**: de Casteljau flattening of a 128-point
//!     star polygon with cubic edges — the geometry front-end.
//!   - **fill_star_256_ss1 / _ss4**: the active-edge-list scanline
//!     fill with analytic horizontal AA, at 1× and 4× vertical
//!     supersampling.
//!   - **render_scene_256**: `Renderer::render` end-to-end on a
//!     procedurally built `VectorFrame` (16 solid rects + 8 cubic
//!     "circles"), measuring the walk → flatten → fill → paint →
//!     composite pipeline.
//!
//! Round 449 additions (the surfaces the round-401 harness left
//! unbenched):
//!
//!   - **stroke_star_256**: stroke-geometry build (round joins/caps) +
//!     NonZero fill of a 64-point cubic star outline.
//!   - **render_glyphlike_400_256**: 400 small (~9 px) cubic blobs on a
//!     256×256 canvas — a caption-density scene where per-shape
//!     overheads (edge-table build, mask alloc, composite setup)
//!     dominate over per-pixel work.
//!   - **render_softmask_256**: one full-canvas `Node::SoftMask`
//!     (luminance kind) — two offscreen subtree renders + coverage
//!     conversion + modulated blit.
//!   - **render_gradient_linear_256 / _radial_256**: full-canvas
//!     gradient fills through the stops-LUT paint path.
//!   - **render_cached_group_hit_256**: a `cache_key`-tagged group
//!     re-rendered with a warm bitmap cache — measures the
//!     lookup-plus-blit fast path.
//!
//! Run with: `cargo bench -p oxideav-raster --bench render`

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use oxideav_core::{
    FillRule, GradientStop, Group, LineCap, LineJoin, LinearGradient, MaskKind, Node, Paint, Path,
    PathNode, Point, RadialGradient, Rgba, SpreadMethod, Stroke, Transform2D, VectorFrame,
};
use oxideav_raster::{flatten_path, rasterize_fill, Renderer};

/// An `n`-point star polygon path centred at (cx, cy) with cubic-bulge
/// edges, giving the flattener and the edge list something non-trivial.
fn star_path(cx: f32, cy: f32, r_outer: f32, r_inner: f32, n: u32) -> Path {
    let mut p = Path::new();
    for i in 0..(2 * n) {
        let r = if i % 2 == 0 { r_outer } else { r_inner };
        let a = std::f32::consts::PI * i as f32 / n as f32;
        let pt = Point::new(cx + r * a.cos(), cy + r * a.sin());
        if i == 0 {
            p.move_to(pt);
        } else {
            // Cubic with slightly off-chord control points so the
            // flattener actually subdivides.
            let prev_a = std::f32::consts::PI * (i - 1) as f32 / n as f32;
            let prev_r = if (i - 1) % 2 == 0 { r_outer } else { r_inner };
            let prev = Point::new(cx + prev_r * prev_a.cos(), cy + prev_r * prev_a.sin());
            let mid = Point::new(
                (prev.x + pt.x) / 2.0 + (pt.y - prev.y) * 0.15,
                (prev.y + pt.y) / 2.0 - (pt.x - prev.x) * 0.15,
            );
            p.cubic_to(mid, mid, pt);
        }
    }
    p.close();
    p
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

fn circle_node(cx: f32, cy: f32, r: f32, fill: Rgba) -> Node {
    let k = 0.5522847f32 * r;
    let mut p = Path::new();
    p.move_to(Point::new(cx + r, cy));
    p.cubic_to(
        Point::new(cx + r, cy - k),
        Point::new(cx + k, cy - r),
        Point::new(cx, cy - r),
    );
    p.cubic_to(
        Point::new(cx - k, cy - r),
        Point::new(cx - r, cy - k),
        Point::new(cx - r, cy),
    );
    p.cubic_to(
        Point::new(cx - r, cy + k),
        Point::new(cx - k, cy + r),
        Point::new(cx, cy + r),
    );
    p.cubic_to(
        Point::new(cx + k, cy + r),
        Point::new(cx + r, cy + k),
        Point::new(cx + r, cy),
    );
    p.close();
    Node::Path(PathNode {
        path: p,
        fill: Some(Paint::Solid(fill)),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
}

fn scene(width: f32, height: f32) -> VectorFrame {
    let mut root = Group::default();
    for i in 0..16u32 {
        let x = (i % 4) as f32 * width / 4.0 + 4.0;
        let y = (i / 4) as f32 * height / 4.0 + 4.0;
        root.children.push(rect_node(
            x,
            y,
            width / 5.0,
            height / 5.0,
            Rgba::opaque((i * 15) as u8, 255 - (i * 12) as u8, (i * 7) as u8),
        ));
    }
    for i in 0..8u32 {
        root.children.push(circle_node(
            width * (0.15 + 0.1 * i as f32),
            height * 0.5,
            width / 12.0,
            Rgba::new((i * 30) as u8, 100, 200, 180),
        ));
    }
    VectorFrame {
        width,
        height,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

fn bench_render(c: &mut Criterion) {
    let star = star_path(128.0, 128.0, 120.0, 48.0, 64);
    let identity = Transform2D::default();

    c.bench_function("flatten_star_128pt", |b| {
        b.iter(|| flatten_path(black_box(&star.commands), black_box(&identity)))
    });

    let contours = flatten_path(&star.commands, &identity);
    c.bench_function("fill_star_256_ss1", |b| {
        b.iter(|| rasterize_fill(black_box(&contours), 256, 256, FillRule::NonZero, 1))
    });
    c.bench_function("fill_star_256_ss4", |b| {
        b.iter(|| rasterize_fill(black_box(&contours), 256, 256, FillRule::NonZero, 4))
    });

    let frame = scene(256.0, 256.0);
    c.bench_function("render_scene_256", |b| {
        let r = Renderer::new(256, 256);
        b.iter(|| r.render(black_box(&frame)))
    });

    // --- Round 449 additions ---

    // Stroked star outline: stroke-geometry build + NonZero fill.
    let stroked_star = {
        let mut root = Group::default();
        root.children.push(Node::Path(PathNode {
            path: star_path(128.0, 128.0, 120.0, 48.0, 64),
            fill: None,
            stroke: Some(Stroke {
                width: 3.0,
                paint: Paint::Solid(Rgba::opaque(20, 40, 200)),
                cap: LineCap::Round,
                join: LineJoin::Round,
                miter_limit: 4.0,
                dash: None,
            }),
            fill_rule: FillRule::NonZero,
        }));
        VectorFrame {
            width: 256.0,
            height: 256.0,
            view_box: None,
            root,
            pts: None,
            time_base: oxideav_core::time::TimeBase::new(1, 1),
        }
    };
    c.bench_function("stroke_star_256", |b| {
        let r = Renderer::new(256, 256);
        b.iter(|| r.render(black_box(&stroked_star)))
    });

    // Caption-density scene: 400 small cubic blobs (~9 px each).
    let glyphlike = {
        let mut root = Group::default();
        for i in 0..400u32 {
            let gx = (i % 20) as f32 * 12.0 + 8.0;
            let gy = (i / 20) as f32 * 12.0 + 8.0;
            root.children.push(circle_node(
                gx,
                gy,
                4.5,
                Rgba::opaque(((i * 7) % 200) as u8 + 30, 40, 90),
            ));
        }
        VectorFrame {
            width: 256.0,
            height: 256.0,
            view_box: None,
            root,
            pts: None,
            time_base: oxideav_core::time::TimeBase::new(1, 1),
        }
    };
    c.bench_function("render_glyphlike_400_256", |b| {
        let r = Renderer::new(256, 256);
        b.iter(|| r.render(black_box(&glyphlike)))
    });

    // Full-canvas luminance soft mask over a full-canvas fill.
    let softmask = {
        let content = circle_node(128.0, 128.0, 120.0, Rgba::opaque(255, 30, 30));
        let mask = circle_node(100.0, 100.0, 110.0, Rgba::opaque(255, 255, 255));
        let mut root = Group::default();
        root.children.push(Node::SoftMask {
            mask: Box::new(mask),
            mask_kind: MaskKind::Luminance,
            content: Box::new(content),
        });
        VectorFrame {
            width: 256.0,
            height: 256.0,
            view_box: None,
            root,
            pts: None,
            time_base: oxideav_core::time::TimeBase::new(1, 1),
        }
    };
    c.bench_function("render_softmask_256", |b| {
        let r = Renderer::new(256, 256);
        b.iter(|| r.render(black_box(&softmask)))
    });

    // Full-canvas gradient fills through the stops-LUT paint path.
    let grad_stops = vec![
        GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
        GradientStop::new(0.5, Rgba::opaque(0, 255, 0)),
        GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
    ];
    let full_rect_with = |paint: Paint| -> VectorFrame {
        let mut p = Path::new();
        p.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(256.0, 0.0))
            .line_to(Point::new(256.0, 256.0))
            .line_to(Point::new(0.0, 256.0))
            .close();
        let mut root = Group::default();
        root.children.push(Node::Path(PathNode {
            path: p,
            fill: Some(paint),
            stroke: None,
            fill_rule: FillRule::NonZero,
        }));
        VectorFrame {
            width: 256.0,
            height: 256.0,
            view_box: None,
            root,
            pts: None,
            time_base: oxideav_core::time::TimeBase::new(1, 1),
        }
    };
    let linear = full_rect_with(Paint::LinearGradient(LinearGradient {
        start: Point::new(0.0, 0.0),
        end: Point::new(256.0, 256.0),
        stops: grad_stops.clone(),
        spread: SpreadMethod::Pad,
    }));
    c.bench_function("render_gradient_linear_256", |b| {
        let r = Renderer::new(256, 256);
        b.iter(|| r.render(black_box(&linear)))
    });
    let radial = full_rect_with(Paint::RadialGradient(RadialGradient {
        center: Point::new(128.0, 128.0),
        radius: 128.0,
        focal: None,
        stops: grad_stops,
        spread: SpreadMethod::Reflect,
    }));
    c.bench_function("render_gradient_radial_256", |b| {
        let r = Renderer::new(256, 256);
        b.iter(|| r.render(black_box(&radial)))
    });

    // Warm bitmap-cache hit: cache_key-tagged group, second-and-later
    // renders reuse the cached crop (lookup + blit only).
    let cached = {
        let mut root = Group::default();
        root.children.push(Node::Group(Group {
            cache_key: Some(0xC0FFEE),
            children: vec![circle_node(128.0, 128.0, 100.0, Rgba::opaque(10, 200, 90))],
            ..Group::default()
        }));
        VectorFrame {
            width: 256.0,
            height: 256.0,
            view_box: None,
            root,
            pts: None,
            time_base: oxideav_core::time::TimeBase::new(1, 1),
        }
    };
    c.bench_function("render_cached_group_hit_256", |b| {
        let r = Renderer::new(256, 256);
        let _warm = r.render(&cached); // populate the cache
        b.iter(|| r.render(black_box(&cached)))
    });
}

criterion_group!(benches, bench_render);
criterion_main!(benches);
