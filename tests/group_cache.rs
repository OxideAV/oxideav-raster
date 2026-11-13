//! Group cache behaviour: translation-independent keys, the sub-pixel
//! grid and its pixel tolerance, partly / wholly off-canvas groups,
//! size caps, the byte budget, cache sharing between renderers, and
//! the bounded offscreen windows of group opacity and soft masks.

use oxideav_core::{
    FillRule, Group, ImageRef, LineCap, LineJoin, MaskKind, Node, Paint, Path, PathCommand,
    PathNode, Point, Rect, Rgba, Stroke, Transform2D, VectorFrame, VideoFrame, VideoPlane,
};
use oxideav_raster::{BlendMode, CoverageMode, GroupCache, InterpolationSpace, Renderer};

fn frame(w: u32, h: u32, root: Group) -> VectorFrame {
    VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

fn render(r: &Renderer, root: &Group) -> Vec<u8> {
    let out = r.render(&frame(r.width, r.height, root.clone()));
    assert_eq!(out.planes[0].data.len(), (r.width * r.height * 4) as usize);
    out.planes.into_iter().next().unwrap().data
}

fn max_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

fn fill(path: Path, color: Rgba) -> Node {
    Node::Path(PathNode {
        path,
        fill: Some(Paint::Solid(color)),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
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

/// A polygon with sloped edges whose vertices are multiples of 1/8 —
/// exactly representable wherever a whole-pixel or grid translation
/// puts it.
fn kite() -> Path {
    let mut p = Path::new();
    p.move_to(Point::new(1.125, 6.5))
        .line_to(Point::new(6.25, 0.375))
        .line_to(Point::new(11.875, 5.75))
        .line_to(Point::new(7.5, 13.625))
        .close();
    p
}

/// A glyph-like outline of curves (an outer blob and a counter).
fn blob() -> Path {
    let mut p = Path::new();
    p.move_to(Point::new(1.3, 7.1))
        .quad_to(Point::new(0.7, 0.9), Point::new(7.2, 1.1))
        .cubic_to(
            Point::new(13.9, 1.4),
            Point::new(13.1, 9.7),
            Point::new(12.2, 12.6),
        )
        .quad_to(Point::new(6.1, 15.3), Point::new(1.3, 7.1))
        .close();
    p.move_to(Point::new(5.1, 6.9))
        .line_to(Point::new(8.3, 9.4))
        .line_to(Point::new(9.6, 5.2))
        .close();
    p
}

/// `children` in a group translated to `(x, y)`, keyed or not.
fn placed(key: Option<u64>, x: f32, y: f32, children: Vec<Node>) -> Node {
    Node::Group(Group {
        transform: Transform2D::translate(x, y),
        cache_key: key,
        children,
        ..Group::default()
    })
}

fn root_of(nodes: Vec<Node>) -> Group {
    Group {
        children: nodes,
        ..Group::default()
    }
}

const INK: Rgba = Rgba::new(30, 60, 200, 255);

#[test]
fn one_bitmap_serves_every_whole_pixel_position() {
    let positions = [
        (3.0, 2.0),
        (20.0, 2.0),
        (41.0, 7.0),
        (5.0, 30.0),
        (33.0, 44.0),
        (50.0, 21.0),
    ];
    let scene = |key: Option<u64>| {
        root_of(
            positions
                .iter()
                .map(|&(x, y)| placed(key, x, y, vec![fill(kite(), INK)]))
                .collect(),
        )
    };
    let r = Renderer::new(64, 64);
    let keyed = render(&r, &scene(Some(7)));
    let s = r.cache_stats();
    assert_eq!((s.misses, s.hits, s.entries), (1, 5, 1));
    // The bitmap is the group's bounds, not the canvas.
    assert!(s.bytes < 16 * 16 * 4 + 512, "{} bytes cached", s.bytes);
    let plain = render(&Renderer::new(64, 64), &scene(None));
    assert_eq!(keyed, plain);
}

#[test]
fn positions_on_the_sub_pixel_grid_render_exactly_like_unkeyed_groups() {
    let mut positions = Vec::new();
    for (i, fx) in [0.0f32, 0.25, 0.5, 0.75].into_iter().enumerate() {
        for (j, fy) in [0.0f32, 0.5].into_iter().enumerate() {
            positions.push((2.0 + 15.0 * i as f32 + fx, 3.0 + 30.0 * j as f32 + fy));
            // The same phase again, elsewhere.
            positions.push((2.0 + 15.0 * i as f32 + fx, 17.0 + 30.0 * j as f32 + fy));
        }
    }
    let scene = |key: Option<u64>| {
        root_of(
            positions
                .iter()
                .map(|&(x, y)| placed(key, x, y, vec![fill(kite(), INK)]))
                .collect(),
        )
    };
    let r = Renderer::new(64, 64);
    let keyed = render(&r, &scene(Some(7)));
    let s = r.cache_stats();
    assert_eq!((s.misses, s.hits, s.entries), (8, 8, 8));
    assert_eq!(keyed, render(&Renderer::new(64, 64), &scene(None)));
}

#[test]
fn off_grid_positions_stay_within_the_documented_tolerance() {
    // Opaque background, so a channel difference is a coverage
    // difference. Moving a shape by (dx, dy) changes a pixel's coverage
    // by at most |dx| + |dy|; snapping moves it by at most 1 / (2·grid)
    // per axis, i.e. 1 / grid of full scale in total.
    let positions: Vec<(f32, f32)> = (0..12)
        .map(|i| {
            (
                3.0 + (i % 4) as f32 * 15.0 + i as f32 * 0.137,
                2.0 + (i / 4) as f32 * 20.0 + i as f32 * 0.291,
            )
        })
        .collect();
    let scene = |key: Option<u64>, at: &dyn Fn(f32) -> f32| {
        root_of(
            positions
                .iter()
                .map(|&(x, y)| placed(key, at(x), at(y), vec![fill(blob(), INK)]))
                .collect(),
        )
    };
    let white = |r: Renderer| {
        let mut r = r;
        r.background = Rgba::opaque(255, 255, 255);
        r
    };
    let exact = render(&white(Renderer::new(72, 64)), &scene(None, &|v| v));
    for grid in [1u8, 2, 4, 8, 16] {
        let r = white(Renderer::new(72, 64)).with_cache_subpixel_grid(grid);
        let keyed = render(&r, &scene(Some(1), &|v| v));
        let bound = ((255.0 / grid as f32).ceil() as u8).saturating_add(1);
        let d = max_diff(&keyed, &exact);
        println!("grid {grid}: max difference {d} of 255 (bound {bound})");
        assert!(d <= bound, "grid {grid}: max difference {d} > {bound}");
        // What the cache draws is the unkeyed group at the snapped
        // position, up to f32 rounding of the device coordinates.
        let g = grid as f32;
        let snapped = render(
            &white(Renderer::new(72, 64)),
            &scene(None, &|v| (v * g).round() / g),
        );
        assert!(max_diff(&keyed, &snapped) <= 1, "grid {grid}");
    }
    // The default grid is a quarter pixel.
    assert_eq!(Renderer::new(1, 1).cache_subpixel_grid, 4);
    // Grid 0 keeps the exact fraction: only f32 rounding remains.
    let r = white(Renderer::new(72, 64)).with_cache_subpixel_grid(0);
    assert!(max_diff(&render(&r, &scene(Some(1), &|v| v)), &exact) <= 1);
    assert_eq!(r.cache_stats().entries, positions.len());
}

#[test]
fn groups_partly_outside_the_canvas_are_cached_whole() {
    let content = || {
        vec![
            fill(rect(0.0, 0.0, 12.0, 10.0), INK),
            fill(kite(), Rgba::new(250, 80, 20, 200)),
        ]
    };
    // Straddling every edge and corner of a 40 × 32 canvas.
    let straddle = [
        (-6.0, -5.0),
        (14.0, -4.0),
        (34.0, -3.0),
        (-7.0, 11.0),
        (33.0, 12.0),
        (-5.0, 27.0),
        (15.0, 26.0),
        (35.0, 28.0),
    ];
    let scene = |key: Option<u64>, at: &[(f32, f32)]| {
        root_of(
            at.iter()
                .map(|&(x, y)| placed(key, x, y, content()))
                .collect(),
        )
    };
    let r = Renderer::new(40, 32);
    let keyed = render(&r, &scene(Some(3), &straddle));
    let plain = render(&Renderer::new(40, 32), &scene(None, &straddle));
    // Rectangles are exact; the sloped edges of the kite are clipped at
    // a different place in the two paths.
    assert!(max_diff(&keyed, &plain) <= 1);
    let s = r.cache_stats();
    assert_eq!((s.misses, s.entries), (1, 1));
    // The entry made while partly off-canvas is complete: fully inside
    // the canvas it is a hit and matches the unkeyed render exactly.
    let inside = [(10.0, 9.0)];
    let keyed = render(&r, &scene(Some(3), &inside));
    assert_eq!(r.cache_stats().misses, 1);
    assert_eq!(keyed, render(&Renderer::new(40, 32), &scene(None, &inside)));
}

#[test]
fn groups_wholly_outside_the_canvas_cost_nothing() {
    let r = Renderer::new(32, 32);
    let far = [
        (-100.0, 4.0),
        (4.0, 400.0),
        (1.0e7, -1.0e7),
        (3.0e12, 0.0),
        (f32::NAN, 2.0),
        (f32::INFINITY, f32::NEG_INFINITY),
    ];
    let scene = root_of(
        far.iter()
            .map(|&(x, y)| placed(Some(9), x, y, vec![fill(kite(), INK)]))
            .collect(),
    );
    let out = render(&r, &scene);
    assert!(out.iter().all(|&b| b == 0));
    assert_eq!(r.cache_stats().entries, 0);
}

#[test]
fn bitmaps_over_a_quarter_of_the_budget_are_drawn_directly() {
    let scene = |key: Option<u64>| {
        root_of(vec![
            // 60 × 50 px → 12 000 bytes and more with the margin.
            placed(key, 2.0, 3.0, vec![fill(rect(0.0, 0.0, 60.0, 50.0), INK)]),
            placed(key.map(|k| k + 1), 70.0, 3.0, vec![fill(kite(), INK)]),
        ])
    };
    let r = Renderer::new(96, 64).with_cache_bytes(16_384);
    let keyed = render(&r, &scene(Some(1)));
    let s = r.cache_stats();
    assert_eq!(s.max_bytes, 16_384);
    assert_eq!(
        s.entries, 1,
        "only the small group fits a quarter of the budget"
    );
    assert!(s.bytes <= s.max_bytes);
    assert_eq!(keyed, render(&Renderer::new(96, 64), &scene(None)));
    // A zero budget disables the cache altogether.
    let off = Renderer::new(96, 64).with_cache_bytes(0);
    assert_eq!(render(&off, &scene(Some(1))), keyed);
    assert_eq!(off.cache_stats().entries, 0);
}

#[test]
fn a_mostly_off_canvas_group_is_drawn_directly() {
    // 1500 × 1500 px of which a 20 px corner is visible: rasterising
    // all of it for the cache would cost far more than drawing the
    // visible part.
    let scene = |key: Option<u64>| {
        root_of(vec![placed(
            key,
            -1480.0,
            -1480.0,
            vec![
                fill(rect(0.0, 0.0, 1500.0, 1500.0), INK),
                fill(rect(1470.0, 1470.0, 20.0, 20.0), Rgba::opaque(255, 0, 0)),
            ],
        )])
    };
    let r = Renderer::new(48, 48);
    let keyed = render(&r, &scene(Some(5)));
    assert_eq!(r.cache_stats().entries, 0);
    assert_eq!(keyed, render(&Renderer::new(48, 48), &scene(None)));
    assert_eq!(&keyed[0..4], &[255, 0, 0, 255]);
}

#[test]
fn a_cache_handle_outlives_and_spans_renderers() {
    let scene = |key: Option<u64>| {
        root_of(
            (0..10)
                .map(|i| {
                    placed(
                        key.map(|k| k + i % 3),
                        4.0 + 5.0 * i as f32,
                        6.0 + 3.5 * i as f32,
                        vec![fill(if i % 3 == 0 { kite() } else { blob() }, INK)],
                    )
                })
                .collect(),
        )
    };
    let cache = GroupCache::with_max_bytes(1 << 20);
    let first = {
        // A renderer that lives for one frame.
        let r = Renderer::new(72, 56).with_group_cache(cache.clone());
        assert!(r.group_cache().shares_with(&cache));
        render(&r, &scene(Some(100)))
    };
    let after_first = cache.stats();
    assert!(after_first.entries > 0);
    assert_eq!(after_first.max_bytes, 1 << 20);
    // The next frame's renderer starts warm.
    let r2 = Renderer::new(72, 56).with_group_cache(cache.clone());
    let second = render(&r2, &scene(Some(100)));
    let after_second = cache.stats();
    assert_eq!(
        after_second.misses, after_first.misses,
        "second frame is all hits"
    );
    assert_eq!(after_second.hits, after_first.hits + 10);
    assert_eq!(first, second);
    // So does a renderer with another canvas size.
    let r3 = Renderer::new(200, 90).with_group_cache(cache.clone());
    let third = render(&r3, &scene(Some(100)));
    assert_eq!(cache.stats().misses, after_first.misses);
    let fresh = render(&Renderer::new(200, 90), &scene(Some(100)));
    assert_eq!(third, fresh);
    // Clearing drops the bitmaps; a renderer's own default cache is
    // not shared with anything.
    cache.clear();
    assert_eq!(cache.stats().entries, 0);
    assert_eq!(cache.stats().bytes, 0);
    assert!(!Renderer::new(8, 8).group_cache().shares_with(&cache));
    cache.set_max_bytes(0);
    let _ = render(&r3, &scene(Some(100)));
    assert_eq!(cache.stats().entries, 0);
}

#[test]
fn renderer_settings_never_share_an_entry() {
    let scene = root_of(vec![
        fill(rect(0.0, 0.0, 48.0, 40.0), Rgba::opaque(200, 220, 90)),
        placed(
            Some(1),
            5.25,
            4.5,
            vec![fill(blob(), Rgba::new(30, 60, 200, 170))],
        ),
        placed(
            Some(1),
            25.25,
            19.5,
            vec![fill(blob(), Rgba::new(30, 60, 200, 170))],
        ),
    ]);
    let variants: Vec<Box<dyn Fn() -> Renderer>> = vec![
        Box::new(|| Renderer::new(48, 40)),
        Box::new(|| Renderer::new(48, 40).with_coverage(CoverageMode::Supersampled)),
        Box::new(|| {
            let mut r = Renderer::new(48, 40).with_coverage(CoverageMode::Supersampled);
            r.supersampling = 8;
            r
        }),
        Box::new(|| {
            let mut r = Renderer::new(48, 40);
            r.blend_mode = BlendMode::Multiply;
            r
        }),
        Box::new(|| Renderer::new(48, 40).with_compositing_space(InterpolationSpace::LinearRgb)),
    ];
    let shared = GroupCache::new();
    let mut outputs = Vec::new();
    for make in &variants {
        let alone = render(&make(), &scene);
        let sharing = render(&make().with_group_cache(shared.clone()), &scene);
        assert_eq!(alone, sharing);
        outputs.push(alone);
    }
    assert_eq!(shared.stats().entries, variants.len());
    assert_eq!(shared.stats().hits, variants.len() as u64);
    // The settings really do produce different bitmaps.
    assert_ne!(outputs[0], outputs[1]);
    // Same key, different linear transform: separate entries too.
    let scaled = root_of(vec![Node::Group(Group {
        transform: Transform2D::scale(2.0, 2.0),
        children: vec![placed(Some(1), 2.0, 2.0, vec![fill(blob(), INK)])],
        ..Group::default()
    })]);
    let r = Renderer::new(48, 40).with_group_cache(shared.clone());
    let before = shared.stats().entries;
    let _ = render(&r, &scaled);
    assert_eq!(shared.stats().entries, before + 1);
}

#[test]
fn opacity_and_inherited_clip_are_applied_when_blitting() {
    let scene = |key: Option<u64>, opacity: f32| {
        let inner = Node::Group(Group {
            transform: Transform2D::translate(6.0, 5.0),
            opacity,
            clip: Some(rect(1.0, 1.0, 9.0, 11.0)),
            cache_key: key,
            children: vec![
                fill(kite(), INK),
                fill(rect(4.0, 4.0, 8.0, 8.0), Rgba::opaque(255, 40, 40)),
            ],
        });
        root_of(vec![
            fill(rect(0.0, 0.0, 32.0, 32.0), Rgba::opaque(240, 240, 240)),
            Node::Group(Group {
                clip: Some(rect(8.0, 0.0, 6.0, 32.0)),
                opacity: 0.75,
                children: vec![inner],
                ..Group::default()
            }),
        ])
    };
    let r = Renderer::new(32, 32);
    for (i, opacity) in [1.0f32, 0.5, 0.2].into_iter().enumerate() {
        let keyed = render(&r, &scene(Some(4), opacity));
        let plain = render(&Renderer::new(32, 32), &scene(None, opacity));
        assert_eq!(keyed, plain, "opacity {opacity}");
        // One bitmap for every opacity.
        let s = r.cache_stats();
        assert_eq!((s.misses, s.hits, s.entries), (1, i as u64, 1));
    }
}

#[test]
fn nested_keyed_groups_compose() {
    let word = |key: Option<u64>| {
        (0..4)
            .map(|i| {
                placed(
                    key.map(|k| k + (i % 2) as u64),
                    13.0 * i as f32,
                    0.0,
                    vec![fill(kite(), INK)],
                )
            })
            .collect::<Vec<_>>()
    };
    let scene = |key: Option<u64>| {
        root_of(vec![
            placed(key.map(|k| k + 10), 3.0, 4.0, word(key)),
            placed(key.map(|k| k + 10), 9.0, 30.0, word(key)),
        ])
    };
    let r = Renderer::new(72, 48);
    let keyed = render(&r, &scene(Some(1)));
    assert_eq!(keyed, render(&Renderer::new(72, 48), &scene(None)));
    // Two glyph bitmaps and one word bitmap; the second word is a hit.
    let s = r.cache_stats();
    assert_eq!(s.entries, 3);
    assert!(s.hits >= 3);
}

#[test]
fn the_byte_budget_bounds_the_cache_and_evicts_least_recently_used() {
    let budget = 24_000;
    let r = Renderer::new(64, 64).with_cache_bytes(budget);
    let scene = |keys: std::ops::Range<u64>| {
        root_of(
            keys.map(|k| {
                placed(
                    Some(k),
                    (k % 5) as f32 * 9.0,
                    (k % 4) as f32 * 11.0,
                    vec![fill(kite(), INK)],
                )
            })
            .collect(),
        )
    };
    let _ = render(&r, &scene(0..200));
    let s = r.cache_stats();
    assert_eq!(s.misses, 200);
    assert!(s.bytes <= budget, "{} > {budget}", s.bytes);
    assert!(s.entries >= 8 && s.entries < 200, "{} entries", s.entries);
    assert_eq!(s.evictions as usize, 200 - s.entries);
    assert_eq!(s.capacity, usize::MAX);
    // The most recent keys survived; the oldest were evicted.
    r.reset_cache_stats();
    let _ = render(&r, &scene(196..200));
    assert_eq!(r.cache_stats().hits, 4);
    let _ = render(&r, &scene(0..2));
    assert_eq!(r.cache_stats().misses, 2);
    // The entry-count form still works alongside the byte budget.
    let r = Renderer::with_cache_capacity(64, 64, 3);
    let _ = render(&r, &scene(0..10));
    assert_eq!(r.cache_stats().entries, 3);
    assert_eq!(r.cache_stats().capacity, 3);
}

// ---------------------------------------------------------------------
// Bounded offscreen windows (group opacity, soft masks).
// ---------------------------------------------------------------------

/// The renderer's group blit over a transparent destination, for one
/// straight-alpha pixel: alpha scaled by `cov / 255 · extra / 255`.
fn over_transparent(px: [u8; 4], cov: u32, extra: u32) -> [u8; 4] {
    let sa = (px[3] as u32 * cov * extra + 255 * 255 / 2) / (255 * 255);
    if px[3] == 0 || cov == 0 || sa == 0 {
        return [0; 4];
    }
    if sa == 255 {
        return [px[0], px[1], px[2], 255];
    }
    let c = |v: u8| {
        let p = (v as u32 * sa + 127) / 255;
        ((p * 255 + sa / 2) / sa).min(255) as u8
    };
    [c(px[0]), c(px[1]), c(px[2]), sa as u8]
}

/// Content that reaches as far from its path as anything the renderer
/// draws: miter spikes, square and round caps, dashes, an arc, a
/// rotated image.
fn spiky_content() -> Vec<Node> {
    let mut zig = Path::new();
    zig.move_to(Point::new(10.0, 40.0))
        .line_to(Point::new(30.0, 12.0))
        .line_to(Point::new(34.0, 40.0))
        .line_to(Point::new(60.0, 14.0));
    let mut miter = Stroke::solid(5.0, Rgba::opaque(200, 30, 30));
    miter.join = LineJoin::Miter;
    miter.miter_limit = 10.0;
    miter.cap = LineCap::Square;
    let mut arc = Path::new();
    arc.move_to(Point::new(20.0, 70.0));
    arc.commands.push(PathCommand::ArcTo {
        rx: 30.0,
        ry: 14.0,
        x_axis_rot: 0.5,
        large_arc: true,
        sweep: true,
        end: Point::new(58.0, 66.0),
    });
    let mut round = Stroke::solid(7.0, Rgba::new(20, 160, 60, 220));
    round.cap = LineCap::Round;
    round.join = LineJoin::Round;
    let mut data = Vec::new();
    for i in 0..16u8 {
        data.extend_from_slice(&[i * 16, 255 - i * 16, 90, 255]);
    }
    let image = Node::Image(ImageRef {
        frame: Box::new(VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 16, data }],
        }),
        bounds: Rect {
            x: 0.0,
            y: 0.0,
            width: 18.0,
            height: 18.0,
        },
        transform: Transform2D::translate(70.0, 30.0).compose(&Transform2D::rotate(0.6)),
    });
    vec![
        Node::Path(PathNode {
            path: zig,
            fill: None,
            stroke: Some(miter),
            fill_rule: FillRule::NonZero,
        }),
        Node::Path(PathNode {
            path: arc,
            fill: Some(Paint::Solid(Rgba::new(250, 220, 40, 160))),
            stroke: Some(round),
            fill_rule: FillRule::EvenOdd,
        }),
        image,
        fill(blob(), INK),
    ]
}

#[test]
fn group_opacity_window_holds_everything_the_group_paints() {
    // The isolated group, blended at opacity 128/255 over a transparent
    // canvas, must equal its direct render with every alpha scaled —
    // anywhere the offscreen window were too small, content would be
    // cut off. Checked under transforms that move, rotate and scale the
    // content, including partly off the canvas.
    let transforms = [
        Transform2D::translate(40.0, 30.0),
        Transform2D::translate(-20.0, -18.0),
        Transform2D::translate(150.0, 96.0).compose(&Transform2D::rotate(2.4)),
        Transform2D::translate(30.0, 20.0).compose(&Transform2D::scale(1.7, 0.6)),
        Transform2D::translate(96.5, 64.25).compose(&Transform2D::rotate(-0.9)),
    ];
    for (i, t) in transforms.iter().enumerate() {
        let group = |opacity: f32| {
            root_of(vec![Node::Group(Group {
                transform: *t,
                opacity,
                children: spiky_content(),
                ..Group::default()
            })])
        };
        let r = Renderer::new(192, 128);
        let direct = render(&r, &group(1.0));
        let isolated = render(&r, &group(128.0 / 255.0));
        assert!(direct.iter().any(|&b| b != 0));
        let mut worst = 0u8;
        for (d, o) in direct
            .as_chunks::<4>()
            .0
            .iter()
            .zip(isolated.as_chunks::<4>().0)
        {
            let want = over_transparent(*d, 255, 128);
            // The window is rasterised at another origin (f32 rounding
            // of device coordinates: one coverage level at most), and
            // un-premultiplying at half alpha doubles a colour step.
            for c in 0..3 {
                if want[3] > 8 && o[3] > 8 {
                    worst = worst.max(want[c].abs_diff(o[c]) / 2);
                }
            }
            worst = worst.max(want[3].abs_diff(o[3]));
        }
        assert!(worst <= 1, "transform {i}: off by {worst}");
    }
}

#[test]
fn soft_mask_window_holds_the_masked_content() {
    let content = Node::Group(Group {
        transform: Transform2D::translate(40.0, 20.0),
        children: spiky_content(),
        ..Group::default()
    });
    let mut ring = rect(50.0, 30.0, 70.0, 60.0);
    ring.commands.extend(rect(70.0, 45.0, 20.0, 20.0).commands);
    let mask = Node::Path(PathNode {
        path: ring,
        fill: Some(Paint::Solid(Rgba::new(255, 255, 255, 200))),
        stroke: Some(Stroke::solid(9.0, Rgba::new(255, 255, 255, 90))),
        fill_rule: FillRule::EvenOdd,
    });
    let r = Renderer::new(192, 128);
    let content_px = render(&r, &root_of(vec![content.clone()]));
    let mask_px = render(&r, &root_of(vec![mask.clone()]));
    let masked = render(
        &r,
        &root_of(vec![Node::SoftMask {
            mask: Box::new(mask),
            mask_kind: MaskKind::Alpha,
            content: Box::new(content),
        }]),
    );
    let mut painted = 0usize;
    for ((c, m), o) in content_px
        .as_chunks::<4>()
        .0
        .iter()
        .zip(mask_px.as_chunks::<4>().0)
        .zip(masked.as_chunks::<4>().0)
    {
        let want = over_transparent(*c, m[3] as u32, 255);
        painted += (want[3] != 0) as usize;
        assert!(want[3].abs_diff(o[3]) <= 2, "{want:?} vs {o:?}");
        if want[3] > 16 && o[3] > 16 {
            for k in 0..3 {
                assert!(want[k].abs_diff(o[k]) <= 12, "{want:?} vs {o:?}");
            }
        }
    }
    assert!(painted > 500);
    // A mask that misses the content entirely paints nothing.
    let disjoint = Node::SoftMask {
        mask: Box::new(fill(rect(150.0, 100.0, 20.0, 20.0), INK)),
        mask_kind: MaskKind::Alpha,
        content: Box::new(fill(rect(2.0, 2.0, 20.0, 20.0), INK)),
    };
    assert!(render(&r, &root_of(vec![disjoint])).iter().all(|&b| b == 0));
}

#[test]
fn group_opacity_is_position_independent() {
    // The same translucent group at several whole-pixel positions of a
    // large canvas (its offscreen window starts at a different origin
    // each time) renders the same pixels as on a small canvas.
    let group = |x: f32, y: f32| {
        root_of(vec![Node::Group(Group {
            transform: Transform2D::translate(x, y),
            opacity: 0.6,
            clip: Some(rect(0.0, 0.0, 13.0, 12.0)),
            children: vec![
                fill(kite(), INK),
                fill(rect(3.0, 2.0, 7.0, 6.0), Rgba::new(250, 80, 20, 200)),
            ],
            ..Group::default()
        })])
    };
    let small = render(&Renderer::new(16, 16), &group(1.0, 1.0));
    let big = Renderer::new(300, 200);
    for (x, y) in [(1.0, 1.0), (7.0, 9.0), (100.0, 37.0), (283.0, 183.0)] {
        let out = render(&big, &group(x, y));
        for row in 0..16usize {
            for col in 0..16usize {
                let s = (row * 16 + col) * 4;
                let (bx, by) = (col + x as usize - 1, row + y as usize - 1);
                if bx >= 300 || by >= 200 {
                    continue;
                }
                let b = (by * 300 + bx) * 4;
                assert_eq!(
                    &small[s..s + 4],
                    &out[b..b + 4],
                    "at ({x}, {y}) + ({col}, {row})"
                );
            }
        }
    }
}

#[test]
fn a_cold_cache_costs_about_as_much_as_no_cache() {
    // 300 glyph-sized keyed groups on a 1206 × 2622 canvas, every one a
    // miss (distinct keys). Each miss used to zero and scan a
    // canvas-sized buffer — three orders of magnitude over the unkeyed
    // draw. The bound here is deliberately loose (shared CI runners):
    // it only has to tell "per-group work" from "per-canvas work".
    let scene = |keyed: bool| {
        root_of(
            (0..300u64)
                .map(|i| {
                    placed(
                        keyed.then_some(i),
                        20.0 + (i % 60) as f32 * 19.3,
                        30.0 + (i / 60) as f32 * 41.7,
                        vec![fill(blob(), INK)],
                    )
                })
                .collect(),
        )
    };
    let time = |keyed: bool| {
        (0..3)
            .map(|_| {
                let r = Renderer::new(1206, 2622);
                let s = scene(keyed);
                let t = std::time::Instant::now();
                let out = render(&r, &s);
                let dt = t.elapsed();
                assert!(out.iter().any(|&b| b != 0));
                if keyed {
                    let st = r.cache_stats();
                    assert_eq!((st.misses, st.entries), (300, 300));
                    // 300 glyph bitmaps, not 300 canvases.
                    assert!(st.bytes < 300 * 2048, "{} bytes", st.bytes);
                }
                dt
            })
            .min()
            .unwrap()
    };
    let (plain, keyed) = (time(false), time(true));
    println!("unkeyed {plain:?}, keyed cold {keyed:?}");
    assert!(
        keyed < plain * 10 + std::time::Duration::from_millis(20),
        "cold keyed frame {keyed:?} vs unkeyed {plain:?}"
    );
}
