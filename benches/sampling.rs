//! Criterion benchmarks for the image-resampling and pattern-paint
//! hot loops.
//!
//! Round 449 depth work: the `filter` / `render` harnesses (round 401)
//! cover the filter-primitive kernels and the scanline fill, but the
//! `Node::Image` sampling family (nearest / bilinear / Lanczos2 /
//! Lanczos3 / Mitchell / Catmull-Rom / B-spline), the pattern paint
//! server, and the soft-mask composite had no baselines. Every
//! scenario is self-contained (LCG pixel soup + procedural tiles —
//! no fixtures).
//!
//!   - **image_<filter>_up_64_to_256**: a 64×64 source drawn to a
//!     256×256 canvas (4× magnification) through each of the seven
//!     [`ImageFilter`] kernels — the per-pixel sampling cost dominates.
//!   - **image_<filter>_down_512_to_256**: a 512×512 source drawn to a
//!     256×256 canvas (2× minification) for the two most common
//!     quality kernels (bilinear, Lanczos3) — same tap count per
//!     output pixel, different cache behaviour on the source walk.
//!   - **pattern_fill_256_bilinear / _nearest**: a 40×40 canvas-space
//!     two-stripe tile filled across 256×256 through
//!     `fill_path_with_pattern` (tile rasterisation + periodic
//!     inverse-mapped sampling).
//!
//! Run with: `cargo bench -p oxideav-raster --bench sampling`

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use oxideav_core::{
    FillRule, Group, ImageRef, Node, Paint, Path, PathNode, Point, Rect, Rgba, Transform2D,
    VectorFrame, VideoFrame, VideoPlane,
};
use oxideav_raster::{ImageFilter, Pattern, Renderer};

fn lcg_frame(width: u32, height: u32, seed: u64) -> VideoFrame {
    let mut state = seed;
    let data: Vec<u8> = (0..width as usize * height as usize * 4)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        })
        .collect();
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (width as usize) * 4,
            data,
        }],
    }
}

fn image_scene(src: VideoFrame, canvas: f32) -> VectorFrame {
    let img_node = Node::Image(ImageRef {
        frame: Box::new(src),
        bounds: Rect {
            x: 0.0,
            y: 0.0,
            width: canvas,
            height: canvas,
        },
        transform: Transform2D::identity(),
    });
    let mut root = Group::default();
    root.children.push(img_node);
    VectorFrame {
        width: canvas,
        height: canvas,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
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

fn rect_node(x: f32, y: f32, w: f32, h: f32, color: Rgba) -> Node {
    Node::Path(PathNode {
        path: rect_path(x, y, w, h),
        fill: Some(Paint::Solid(color)),
        fill_rule: FillRule::NonZero,
        stroke: None,
    })
}

fn stripe_pattern() -> Pattern {
    Pattern::new(0.0, 0.0, 40.0, 40.0)
        .with_child(rect_node(0.0, 0.0, 20.0, 40.0, Rgba::opaque(255, 0, 0)))
        .with_child(rect_node(20.0, 0.0, 20.0, 40.0, Rgba::opaque(0, 0, 255)))
}

fn bench_sampling(c: &mut Criterion) {
    let filters = [
        ("nearest", ImageFilter::Nearest),
        ("bilinear", ImageFilter::Bilinear),
        ("lanczos2", ImageFilter::Lanczos2),
        ("lanczos3", ImageFilter::Lanczos3),
        ("mitchell", ImageFilter::Mitchell),
        ("catmull_rom", ImageFilter::CatmullRom),
        ("b_spline", ImageFilter::BSpline),
    ];

    for (name, filter) in filters {
        let scene = image_scene(lcg_frame(64, 64, 0xfeed_beef), 256.0);
        c.bench_function(&format!("image_{name}_up_64_to_256"), |b| {
            let mut r = Renderer::new(256, 256);
            r.image_filter = filter;
            b.iter(|| r.render(black_box(&scene)))
        });
    }

    for (name, filter) in [
        ("bilinear", ImageFilter::Bilinear),
        ("lanczos3", ImageFilter::Lanczos3),
    ] {
        let scene = image_scene(lcg_frame(512, 512, 0xdead_cafe), 256.0);
        c.bench_function(&format!("image_{name}_down_512_to_256"), |b| {
            let mut r = Renderer::new(256, 256);
            r.image_filter = filter;
            b.iter(|| r.render(black_box(&scene)))
        });
    }

    let pat = stripe_pattern();
    let full = rect_path(0.0, 0.0, 256.0, 256.0);
    for (name, filter) in [
        ("bilinear", ImageFilter::Bilinear),
        ("nearest", ImageFilter::Nearest),
    ] {
        c.bench_function(&format!("pattern_fill_256_{name}"), |b| {
            let mut r = Renderer::new(256, 256);
            r.image_filter = filter;
            b.iter(|| {
                r.fill_path_with_pattern(
                    black_box(&full),
                    FillRule::NonZero,
                    black_box(&pat),
                    Transform2D::identity(),
                )
            })
        });
    }
}

criterion_group!(benches, bench_sampling);
criterion_main!(benches);
