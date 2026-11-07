//! Criterion benchmarks for realistic full-scene workloads (round 484).
//!
//! The `render` harness measures the kernels on small 256×256 scenes;
//! these scenarios measure what a document / SVG renderer actually
//! feeds the crate:
//!
//!   - **paths_1080p**: a 1920×1080 illustration — 2 400 curved filled
//!     blobs (radius 3–120 px, translucent), 600 stroked open cubic
//!     curves (1–6 px, round joins / caps, a third dashed) and 1 800
//!     glyph-like outlines (ring + stem, even-odd, ~14 px tall).
//!   - **gradient_linear_4k / gradient_radial_4k**: one full 3840×2160
//!     rectangle filled with a 3-stop gradient.
//!   - **many_small_4k**: 20 000 small (3–10 px) rects and circles
//!     scattered over a 3840×2160 canvas.
//!   - **big_circle_4k**: one radius-1000 circle, filled and stroked
//!     (24 px) on a 3840×2160 canvas — curve flattening + long edges.
//!   - **text_page_1080p**: 6 000 tiny glyph-like outlines laid out in
//!     lines like a page of body text.
//!   - **…_mt4**: the same scenes under a 4-worker `ExecutionContext`.
//!
//! Run with: `cargo bench -p oxideav-raster --bench scenes`

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

use oxideav_core::{
    DashPattern, FillRule, GradientStop, Group, LineCap, LineJoin, LinearGradient, Node, Paint,
    Path, PathNode, Point, RadialGradient, Rgba, SpreadMethod, Stroke, VectorFrame,
};
use oxideav_raster::Renderer;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next() * (hi - lo)
    }
    fn byte(&mut self) -> u8 {
        (self.next() * 255.0) as u8
    }
}

fn frame(w: f32, h: f32, root: Group) -> VectorFrame {
    VectorFrame {
        width: w,
        height: h,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

fn fill_node(path: Path, paint: Paint, rule: FillRule) -> Node {
    Node::Path(PathNode {
        path,
        fill: Some(paint),
        stroke: None,
        fill_rule: rule,
    })
}

/// A wobbly closed blob: `n` cubic segments around (cx, cy).
fn blob(p: &mut Path, rng: &mut Lcg, cx: f32, cy: f32, r: f32, n: u32) {
    let pts: Vec<(f32, f32)> = (0..n)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / n as f32;
            let rr = r * rng.range(0.7, 1.1);
            (cx + rr * a.cos(), cy + rr * a.sin())
        })
        .collect();
    p.move_to(Point::new(pts[0].0, pts[0].1));
    for i in 0..n as usize {
        let a = pts[i];
        let b = pts[(i + 1) % n as usize];
        let k = 0.35;
        // Tangent-ish control points perpendicular to the radius.
        let (ax, ay) = (a.0 - cx, a.1 - cy);
        let (bx, by) = (b.0 - cx, b.1 - cy);
        p.cubic_to(
            Point::new(a.0 - ay * k, a.1 + ax * k),
            Point::new(b.0 + by * k, b.1 - bx * k),
            Point::new(b.0, b.1),
        );
    }
    p.close();
}

fn circle(p: &mut Path, cx: f32, cy: f32, r: f32) {
    let k = 0.552_284_7 * r;
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
}

fn rect(p: &mut Path, x: f32, y: f32, w: f32, h: f32) {
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
}

/// A "glyph": an even-odd ring (outer + inner ellipse) plus a stem —
/// the topology of a lowercase `b` / `d` / `p`.
fn glyph(x: f32, y: f32, size: f32, color: Rgba) -> Node {
    let mut p = Path::new();
    let r = size * 0.3;
    circle(&mut p, x + r, y + size - r, r);
    circle(&mut p, x + r, y + size - r, r * 0.55);
    rect(&mut p, x + 2.0 * r - size * 0.12, y, size * 0.12, size);
    fill_node(p, Paint::Solid(color), FillRule::EvenOdd)
}

fn paths_scene(w: f32, h: f32) -> VectorFrame {
    let mut rng = Lcg(0x5EED_1234);
    let mut root = Group::default();
    for _ in 0..2400 {
        let r = if rng.next() < 0.9 {
            rng.range(3.0, 30.0)
        } else {
            rng.range(30.0, 120.0)
        };
        let mut p = Path::new();
        let (cx, cy) = (rng.range(0.0, w), rng.range(0.0, h));
        blob(&mut p, &mut rng, cx, cy, r, 6);
        let c = Rgba::new(rng.byte(), rng.byte(), rng.byte(), 128 + rng.byte() / 2);
        root.children
            .push(fill_node(p, Paint::Solid(c), FillRule::NonZero));
    }
    for i in 0..600 {
        let mut p = Path::new();
        let x0 = rng.range(0.0, w);
        let y0 = rng.range(0.0, h);
        p.move_to(Point::new(x0, y0));
        let (mut x, mut y) = (x0, y0);
        for _ in 0..4 {
            let nx = x + rng.range(-150.0, 150.0);
            let ny = y + rng.range(-150.0, 150.0);
            p.cubic_to(
                Point::new(x + rng.range(-80.0, 80.0), y + rng.range(-80.0, 80.0)),
                Point::new(nx + rng.range(-80.0, 80.0), ny + rng.range(-80.0, 80.0)),
                Point::new(nx, ny),
            );
            x = nx;
            y = ny;
        }
        let dash = (i % 3 == 0).then(|| DashPattern {
            array: vec![12.0, 6.0],
            offset: 0.0,
        });
        root.children.push(Node::Path(PathNode {
            path: p,
            fill: None,
            stroke: Some(Stroke {
                width: rng.range(1.0, 6.0),
                paint: Paint::Solid(Rgba::new(rng.byte(), rng.byte(), rng.byte(), 230)),
                cap: LineCap::Round,
                join: LineJoin::Round,
                miter_limit: 4.0,
                dash,
            }),
            fill_rule: FillRule::NonZero,
        }));
    }
    for i in 0..1800u32 {
        let gx = (i % 90) as f32 * 20.0 + 10.0;
        let gy = (i / 90) as f32 * 50.0 + 20.0;
        root.children
            .push(glyph(gx, gy, 14.0, Rgba::opaque(20, 20, (i % 200) as u8)));
    }
    frame(w, h, root)
}

fn text_page(w: f32, h: f32) -> VectorFrame {
    let mut root = Group::default();
    let mut i = 0u32;
    'outer: for line in 0.. {
        let y = 30.0 + line as f32 * 16.0;
        if y > h - 20.0 {
            break;
        }
        for col in 0..240 {
            let x = 20.0 + col as f32 * 7.8;
            if x > w - 20.0 {
                break;
            }
            root.children
                .push(glyph(x, y, 10.0, Rgba::opaque(10, 10, 10)));
            i += 1;
            if i >= 6000 {
                break 'outer;
            }
        }
    }
    frame(w, h, root)
}

fn gradient_scene(w: f32, h: f32, paint: Paint) -> VectorFrame {
    let mut p = Path::new();
    rect(&mut p, 0.0, 0.0, w, h);
    let mut root = Group::default();
    root.children.push(fill_node(p, paint, FillRule::NonZero));
    frame(w, h, root)
}

fn many_small(w: f32, h: f32) -> VectorFrame {
    let mut rng = Lcg(0xABCD_EF01);
    let mut root = Group::default();
    for i in 0..20_000 {
        let mut p = Path::new();
        let x = rng.range(0.0, w);
        let y = rng.range(0.0, h);
        let s = rng.range(3.0, 10.0);
        if i % 2 == 0 {
            rect(&mut p, x, y, s, s * rng.range(0.5, 1.5));
        } else {
            circle(&mut p, x, y, s * 0.5);
        }
        let c = Rgba::new(rng.byte(), rng.byte(), rng.byte(), 255);
        root.children
            .push(fill_node(p, Paint::Solid(c), FillRule::NonZero));
    }
    frame(w, h, root)
}

fn big_circle(w: f32, h: f32) -> VectorFrame {
    let mut p = Path::new();
    circle(&mut p, w * 0.5, h * 0.5, 1000.0);
    let mut root = Group::default();
    root.children.push(Node::Path(PathNode {
        path: p,
        fill: Some(Paint::Solid(Rgba::new(40, 120, 220, 200))),
        stroke: Some(Stroke {
            width: 24.0,
            paint: Paint::Solid(Rgba::opaque(10, 10, 10)),
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: 4.0,
            dash: None,
        }),
        fill_rule: FillRule::NonZero,
    }));
    frame(w, h, root)
}

fn bench_scenes(c: &mut Criterion) {
    let mut g = c.benchmark_group("scenes");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(6));
    g.warm_up_time(Duration::from_secs(1));

    let stops = vec![
        GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
        GradientStop::new(0.5, Rgba::opaque(0, 255, 0)),
        GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
    ];

    let s = paths_scene(1920.0, 1080.0);
    let r = Renderer::new(1920, 1080);
    g.bench_function("paths_1080p", |b| b.iter(|| r.render(black_box(&s))));

    let s = text_page(1920.0, 1080.0);
    g.bench_function("text_page_1080p", |b| b.iter(|| r.render(black_box(&s))));

    let r4 = Renderer::new(3840, 2160);
    let s = gradient_scene(
        3840.0,
        2160.0,
        Paint::LinearGradient(LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(3840.0, 2160.0),
            stops: stops.clone(),
            spread: SpreadMethod::Pad,
        }),
    );
    g.bench_function("gradient_linear_4k", |b| {
        b.iter(|| r4.render(black_box(&s)))
    });

    let s = gradient_scene(
        3840.0,
        2160.0,
        Paint::RadialGradient(RadialGradient {
            center: Point::new(1920.0, 1080.0),
            radius: 1500.0,
            focal: None,
            stops,
            spread: SpreadMethod::Reflect,
        }),
    );
    g.bench_function("gradient_radial_4k", |b| {
        b.iter(|| r4.render(black_box(&s)))
    });

    let s = many_small(3840.0, 2160.0);
    g.bench_function("many_small_4k", |b| b.iter(|| r4.render(black_box(&s))));

    let s = big_circle(3840.0, 2160.0);
    g.bench_function("big_circle_4k", |b| b.iter(|| r4.render(black_box(&s))));

    // Same workloads with a 4-worker execution context (row bands of
    // large shapes; byte-identical output).
    let ctx = oxideav_core::ExecutionContext::with_threads(4);
    let r = Renderer::new(1920, 1080).with_execution_context(&ctx);
    let s = paths_scene(1920.0, 1080.0);
    g.bench_function("paths_1080p_mt4", |b| b.iter(|| r.render(black_box(&s))));
    let r4 = Renderer::new(3840, 2160).with_execution_context(&ctx);
    let s = gradient_scene(
        3840.0,
        2160.0,
        Paint::LinearGradient(LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(3840.0, 2160.0),
            stops: vec![
                GradientStop::new(0.0, Rgba::opaque(255, 0, 0)),
                GradientStop::new(0.5, Rgba::opaque(0, 255, 0)),
                GradientStop::new(1.0, Rgba::opaque(0, 0, 255)),
            ],
            spread: SpreadMethod::Pad,
        }),
    );
    g.bench_function("gradient_linear_4k_mt4", |b| {
        b.iter(|| r4.render(black_box(&s)))
    });
    let s = big_circle(3840.0, 2160.0);
    g.bench_function("big_circle_4k_mt4", |b| b.iter(|| r4.render(black_box(&s))));

    g.finish();
}

criterion_group!(benches, bench_scenes);
criterion_main!(benches);
