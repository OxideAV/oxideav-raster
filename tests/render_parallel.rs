//! Round 484: band-parallel rendering under an `ExecutionContext` is
//! byte-identical to the serial render.

use oxideav_core::{
    ExecutionContext, FillRule, GradientStop, Group, LineCap, LineJoin, LinearGradient, Node,
    Paint, Path, PathNode, Point, RadialGradient, Rgba, SpreadMethod, Stroke, VectorFrame,
};
use oxideav_raster::{BlendMode, CoverageMode, Renderer};

fn blob(cx: f32, cy: f32, r: f32, wobble: f32) -> Path {
    let mut p = Path::new();
    let n = 9;
    let pt = |i: u32| {
        let a = std::f32::consts::TAU * i as f32 / n as f32;
        let rr = r * (1.0 + wobble * (3.0 * a).sin());
        Point::new(cx + rr * a.cos(), cy + rr * a.sin())
    };
    p.move_to(pt(0));
    for i in 0..n {
        let (a, b) = (pt(i), pt(i + 1));
        p.cubic_to(
            Point::new(a.x - (a.y - cy) * 0.3, a.y + (a.x - cx) * 0.3),
            Point::new(b.x + (b.y - cy) * 0.3, b.y - (b.x - cx) * 0.3),
            b,
        );
    }
    p.close();
    p
}

fn scene(w: f32, h: f32) -> VectorFrame {
    let stops = vec![
        GradientStop::new(0.0, Rgba::new(255, 0, 0, 255)),
        GradientStop::new(0.4, Rgba::new(0, 200, 40, 160)),
        GradientStop::new(1.0, Rgba::new(0, 0, 255, 255)),
    ];
    let mut root = Group::default();
    root.children.push(Node::Path(PathNode {
        path: blob(w * 0.5, h * 0.5, w * 0.4, 0.2),
        fill: Some(Paint::LinearGradient(LinearGradient {
            start: Point::new(0.0, 0.0),
            end: Point::new(w, h * 0.3),
            stops: stops.clone(),
            spread: SpreadMethod::Reflect,
        })),
        stroke: Some(Stroke {
            width: 17.0,
            paint: Paint::Solid(Rgba::new(20, 20, 20, 200)),
            cap: LineCap::Round,
            join: LineJoin::Round,
            miter_limit: 4.0,
            dash: None,
        }),
        fill_rule: FillRule::EvenOdd,
    }));
    root.children.push(Node::Group(Group {
        clip: Some(blob(w * 0.4, h * 0.6, w * 0.3, 0.1)),
        children: vec![Node::Path(PathNode {
            path: blob(w * 0.6, h * 0.4, w * 0.45, 0.3),
            fill: Some(Paint::RadialGradient(RadialGradient {
                center: Point::new(w * 0.5, h * 0.5),
                radius: w * 0.3,
                focal: Some(Point::new(w * 0.45, h * 0.45)),
                stops,
                spread: SpreadMethod::Repeat,
            })),
            stroke: None,
            fill_rule: FillRule::NonZero,
        })],
        ..Group::default()
    }));
    VectorFrame {
        width: w,
        height: h,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

#[test]
fn band_parallel_render_is_byte_identical() {
    let (w, h) = (700u32, 650u32);
    let f = scene(w as f32, h as f32);
    for blend in [BlendMode::Normal, BlendMode::Multiply] {
        let mut serial = Renderer::new(w, h);
        serial.blend_mode = blend;
        serial.background = Rgba::new(240, 230, 200, 255);
        let want = serial.render(&f);
        for threads in [2usize, 3, 7] {
            let r = serial
                .clone()
                .with_execution_context(&ExecutionContext::with_threads(threads));
            assert_eq!(r.threads, threads);
            let got = r.render(&f);
            assert!(
                got.planes[0].data == want.planes[0].data,
                "{blend:?}: {threads} workers differ from serial"
            );
        }
    }
    // The supersampled kernel is unaffected by the budget.
    let ss = Renderer::new(w, h).with_coverage(CoverageMode::Supersampled);
    let ss_mt = ss
        .clone()
        .with_execution_context(&ExecutionContext::with_threads(4));
    assert!(ss.render(&f).planes[0].data == ss_mt.render(&f).planes[0].data);
}

/// `render_node` has no scene to band: a large shape is scanned in
/// concurrent row bands of its own window instead.
#[test]
fn single_node_shape_bands_are_byte_identical() {
    let (w, h) = (700u32, 650u32);
    let f = scene(w as f32, h as f32);
    for (i, node) in f.root.children.iter().enumerate() {
        for coverage in [CoverageMode::Analytic, CoverageMode::Supersampled] {
            let serial = Renderer::new(w, h).with_coverage(coverage);
            let t = oxideav_core::Transform2D::translate(13.5, -41.25);
            let want = serial.render_node(node, t);
            assert!(want.planes[0].data.iter().any(|&b| b != 0));
            for threads in [2usize, 5, 9] {
                let r = serial
                    .clone()
                    .with_execution_context(&ExecutionContext::with_threads(threads));
                assert!(
                    r.render_node(node, t).planes[0].data == want.planes[0].data,
                    "node {i}, {coverage:?}, {threads} workers"
                );
            }
        }
    }
}

#[test]
fn default_renderer_is_serial() {
    assert_eq!(Renderer::new(4, 4).threads, 1);
}

// ---------------------------------------------------------------------
// Scene bands: every node kind, at every budget.
// ---------------------------------------------------------------------

mod scene_bands {
    use oxideav_core::{
        DashPattern, ExecutionContext, FillRule, GradientStop, Group, ImageRef, LineCap, LineJoin,
        LinearGradient, MaskKind, Node, Paint, Path, PathNode, Point, Rect, Rgba, SpreadMethod,
        Stroke, Transform2D, VectorFrame, VideoFrame, VideoPlane,
    };
    use oxideav_raster::{BlendMode, CoverageMode, GradientDither, InterpolationSpace, Renderer};

    struct Lcg(u32);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 8) as f32 / (1u32 << 24) as f32
        }
        fn byte(&mut self) -> u8 {
            (self.next() * 255.0) as u8
        }
    }

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Path {
        let mut p = Path::new();
        p.move_to(Point::new(x, y))
            .line_to(Point::new(x + w, y))
            .line_to(Point::new(x + w, y + h))
            .line_to(Point::new(x, y + h))
            .close();
        p
    }

    fn blob(s: f32) -> Path {
        let mut p = Path::new();
        p.move_to(Point::new(1.3 * s, 7.1 * s))
            .quad_to(Point::new(0.7 * s, 0.9 * s), Point::new(7.2 * s, 1.1 * s))
            .cubic_to(
                Point::new(13.9 * s, 1.4 * s),
                Point::new(13.1 * s, 9.7 * s),
                Point::new(12.2 * s, 12.6 * s),
            )
            .quad_to(Point::new(6.1 * s, 15.3 * s), Point::new(1.3 * s, 7.1 * s))
            .close();
        p
    }

    fn painted(rng: &mut Lcg, path: Path, i: u32) -> Node {
        let fill = if i % 4 == 1 {
            Paint::LinearGradient(LinearGradient {
                start: Point::new(0.0, 0.0),
                end: Point::new(40.0, 25.0),
                stops: vec![
                    GradientStop::new(0.0, Rgba::new(rng.byte(), 40, 200, 255)),
                    GradientStop::new(1.0, Rgba::new(250, rng.byte(), 30, 180)),
                ],
                spread: SpreadMethod::Reflect,
            })
        } else {
            Paint::Solid(Rgba::new(rng.byte(), rng.byte(), 200, 150 + rng.byte() / 3))
        };
        let stroke = (i % 3 != 2).then(|| Stroke {
            width: 0.75 + 4.0 * rng.next(),
            paint: Paint::Solid(Rgba::new(10, 10, rng.byte(), 220)),
            cap: [LineCap::Butt, LineCap::Round, LineCap::Square][(i % 3) as usize],
            join: [LineJoin::Miter, LineJoin::Round, LineJoin::Bevel][(i / 3 % 3) as usize],
            miter_limit: 4.0,
            dash: i.is_multiple_of(5).then(|| DashPattern {
                array: vec![5.0, 3.0, 1.0, 3.0],
                offset: 1.5,
            }),
        });
        Node::Path(PathNode {
            path,
            fill: Some(fill),
            stroke,
            fill_rule: if i.is_multiple_of(2) {
                FillRule::NonZero
            } else {
                FillRule::EvenOdd
            },
        })
    }

    fn image() -> Node {
        let mut data = Vec::new();
        for i in 0..64u32 {
            data.extend_from_slice(&[
                (i * 4) as u8,
                (255 - i * 3) as u8,
                (i * 37 % 256) as u8,
                230,
            ]);
        }
        Node::Image(ImageRef {
            frame: Box::new(VideoFrame {
                pts: None,
                planes: vec![VideoPlane { stride: 32, data }],
            }),
            bounds: Rect {
                x: 0.0,
                y: 0.0,
                width: 40.0,
                height: 40.0,
            },
            transform: Transform2D::identity(),
        })
    }

    /// Nested groups with every feature the scene walk has: clips,
    /// translucent and keyed groups, soft masks, images (axis-aligned
    /// and rotated), gradients, dashed strokes.
    fn tree(rng: &mut Lcg, depth: u32, w: f32, h: f32) -> Group {
        let mut g = Group::default();
        for i in 0..7u32 {
            let (x, y) = (rng.next() * w - 20.0, rng.next() * h - 15.0);
            let turn = if i % 2 == 0 { rng.next() * 2.0 } else { 0.0 };
            let mut child = Group {
                transform: Transform2D::translate(x, y).compose(&Transform2D::rotate(turn)),
                clip: (i % 3 == 0).then(|| blob(3.0 + rng.next() * 4.0)),
                opacity: if i % 3 == 1 { 0.55 } else { 1.0 },
                cache_key: (i % 4 == 3).then_some(depth as u64 * 16 + (i % 2) as u64),
                ..Group::default()
            };
            if child.cache_key.is_none() {
                let r = rect(
                    -6.0,
                    -4.0,
                    30.0 + rng.next() * 60.0,
                    20.0 + rng.next() * 50.0,
                );
                child.children.push(painted(rng, r, i));
                let b = blob(2.0 + rng.next() * 3.0);
                child.children.push(painted(rng, b, i + 1));
            } else {
                // Keyed content must be a function of its key (the
                // random stream is not consumed, so it is).
                let mut fixed = Lcg(child.cache_key.unwrap_or(0) as u32 + 7);
                // (The clip is content too.)
                child.clip = child.clip.is_some().then(|| blob(4.0));
                child.children.push(painted(&mut fixed, blob(3.5), 4));
            }
            if depth > 0 && child.cache_key.is_none() {
                child
                    .children
                    .push(Node::Group(tree(rng, depth - 1, w * 0.5, h * 0.5)));
                if i == 2 {
                    child.children.push(image());
                }
                if i == 4 {
                    child.children.push(Node::SoftMask {
                        mask: Box::new(painted(rng, blob(6.0), 1)),
                        mask_kind: if depth.is_multiple_of(2) {
                            MaskKind::Alpha
                        } else {
                            MaskKind::Luminance
                        },
                        content: Box::new(Node::Group(tree(rng, 0, 80.0, 60.0))),
                    });
                }
            }
            g.children.push(Node::Group(child));
        }
        g
    }

    fn frame(seed: u32, w: u32, h: u32) -> VectorFrame {
        let mut rng = Lcg(seed.wrapping_mul(2654435761) | 1);
        let mut root = tree(&mut rng, 2, w as f32, h as f32);
        if seed % 2 == 1 {
            root.clip = Some(blob(w as f32 / 14.0));
        }
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
    fn scene_band_render_is_byte_identical_at_every_budget() {
        type Configure = Box<dyn Fn(Renderer) -> Renderer>;
        let configs: Vec<(&str, Configure)> = vec![
            ("default", Box::new(|r| r)),
            (
                "supersampled",
                Box::new(|r| r.with_coverage(CoverageMode::Supersampled)),
            ),
            (
                "linear, dithered, multiply",
                Box::new(|mut r| {
                    r.blend_mode = BlendMode::Multiply;
                    r.background = Rgba::new(250, 240, 230, 255);
                    r.with_compositing_space(InterpolationSpace::LinearRgb)
                        .with_gradient_dither(GradientDither::Ordered)
                }),
            ),
            (
                "scene supersampling",
                Box::new(|r| r.with_scene_supersampling(2)),
            ),
            (
                "exact sub-pixel cache",
                Box::new(|r| r.with_cache_subpixel_grid(0)),
            ),
        ];
        for seed in 0..3u32 {
            let (w, h) = (230 + seed * 17, 170 + seed * 23);
            let f = frame(seed, w, h);
            for (name, make) in &configs {
                let want = make(Renderer::new(w, h))
                    .render(&f)
                    .planes
                    .swap_remove(0)
                    .data;
                assert!(want.iter().any(|&b| b != 0));
                for threads in [2usize, 3, 7, 16] {
                    let r = make(Renderer::new(w, h))
                        .with_execution_context(&ExecutionContext::with_threads(threads));
                    let got = r.render(&f).planes.swap_remove(0).data;
                    if got != want {
                        let first = got.iter().zip(&want).position(|(a, b)| a != b).unwrap();
                        panic!(
                            "seed {seed}, {name}, {threads} workers: first difference at \
                             pixel ({}, {})",
                            first / 4 % w as usize,
                            first / 4 / w as usize
                        );
                    }
                    // A warm cache gives the same picture again.
                    if threads == 3 {
                        assert!(
                            r.render(&f).planes[0].data == want,
                            "{name}: warm, {threads} workers"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tiny_canvases_and_huge_budgets_are_fine() {
        for (w, h) in [
            (1u32, 1u32),
            (3, 15),
            (40, 16),
            (40, 31),
            (40, 33),
            (5, 200),
        ] {
            let f = frame(9, w, h);
            let want = Renderer::new(w, h).render(&f).planes.swap_remove(0).data;
            for threads in [2usize, 64] {
                let r = Renderer::new(w, h)
                    .with_execution_context(&ExecutionContext::with_threads(threads));
                assert!(
                    r.render(&f).planes[0].data == want,
                    "{w}×{h}, {threads} workers"
                );
            }
        }
    }
}
