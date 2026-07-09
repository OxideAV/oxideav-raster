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
//! Run with: `cargo bench -p oxideav-raster --bench render`

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use oxideav_core::{
    FillRule, Group, Node, Paint, Path, PathNode, Point, Rgba, Transform2D, VectorFrame,
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
}

criterion_group!(benches, bench_render);
criterion_main!(benches);
