//! Scene set for the `pipeline` harness: one workload per stage of the
//! rasteriser (edge building, coverage accumulation, span compositing,
//! gradient evaluation, stroking, image sampling, clip / mask / group
//! offscreens, the keyed-group cache). Everything is procedural.

use oxideav_core::{
    DashPattern, FillRule, GradientStop, Group, ImageRef, LineCap, LineJoin, LinearGradient,
    MaskKind, Node, Paint, Path, PathNode, Point, RadialGradient, Rect, Rgba, SpreadMethod, Stroke,
    TimeBase, Transform2D, VectorFrame, VideoFrame, VideoPlane,
};
use oxideav_raster::{BlendMode, GroupCache, Pattern, Renderer};

pub const W: u32 = 1920;
pub const H: u32 = 1080;

/// One benchmark case: a name and a closure rendering one frame and
/// returning a byte of the output (so the work cannot be elided).
pub struct Case {
    pub name: &'static str,
    pub run: Box<dyn FnMut() -> u8>,
}

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
    fn byte(&mut self) -> u8 {
        (self.next() * 255.99) as u8
    }
}

fn frame(root: Group) -> VectorFrame {
    VectorFrame {
        width: W as f32,
        height: H as f32,
        view_box: None,
        root,
        pts: None,
        time_base: TimeBase::new(1, 1),
    }
}

fn fill_node(path: Path, paint: Paint) -> Node {
    Node::Path(PathNode {
        path,
        fill: Some(paint),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
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

fn circle_path(cx: f32, cy: f32, r: f32) -> Path {
    let k = 0.552_284_7 * r;
    let mut p = Path::new();
    p.move_to(Point::new(cx + r, cy));
    p.cubic_to(
        Point::new(cx + r, cy + k),
        Point::new(cx + k, cy + r),
        Point::new(cx, cy + r),
    );
    p.cubic_to(
        Point::new(cx - k, cy + r),
        Point::new(cx - r, cy + k),
        Point::new(cx - r, cy),
    );
    p.cubic_to(
        Point::new(cx - r, cy - k),
        Point::new(cx - k, cy - r),
        Point::new(cx, cy - r),
    );
    p.cubic_to(
        Point::new(cx + k, cy - r),
        Point::new(cx + r, cy - k),
        Point::new(cx + r, cy),
    );
    p.close();
    p
}

fn stops(n: usize, alpha: u8) -> Vec<GradientStop> {
    let mut rng = Lcg(0x57_0b5 + n as u32);
    (0..n)
        .map(|i| GradientStop {
            offset: i as f32 / (n - 1) as f32,
            color: Rgba::new(rng.byte(), rng.byte(), rng.byte(), alpha),
        })
        .collect()
}

// ---------------------------------------------------------------- text

pub const GLYPH_KINDS: u32 = 96;
const UNITS_PER_EM: f32 = 1000.0;

/// A glyph-like outline in font units (Y up): an outer contour of
/// quadratic segments with a counter cut out of it.
fn glyph_outline(id: u32) -> Path {
    let mut p = Path::new();
    let w = 380.0 + (id % 7) as f32 * 35.0;
    let h = 480.0 + (id % 5) as f32 * 55.0;
    let ring = |p: &mut Path, rx: f32, ry: f32, n: u32, reverse: bool| {
        let (cx, cy) = (w * 0.5 + 40.0, h * 0.5);
        for k in 0..=n {
            let k = if reverse { n - k } else { k };
            let a = k as f32 / n as f32 * std::f32::consts::TAU;
            let wobble = 1.0 + 0.08 * ((id + 3) as f32 * a).sin();
            let end = Point::new(cx + rx * wobble * a.cos(), cy + ry * wobble * a.sin());
            if (k == 0 && !reverse) || (k == n && reverse) {
                p.move_to(end);
            } else {
                let am = a + if reverse { 0.5 } else { -0.5 } * std::f32::consts::TAU / n as f32;
                p.quad_to(
                    Point::new(cx + rx * 1.12 * am.cos(), cy + ry * 1.12 * am.sin()),
                    end,
                );
            }
        }
        p.close();
    };
    ring(&mut p, w * 0.5, h * 0.5, 10, false);
    ring(&mut p, w * 0.27, h * 0.3, 8, true);
    p
}

/// A page of `count` glyph groups in four sizes, set line by line with
/// fractional advances (so pen positions take every sub-pixel phase in
/// x) and fractional line spacing. `keyed` tags each group with
/// (glyph, size); `scroll` shifts the whole page.
pub fn text_page(count: usize, keyed: bool, scroll: (f32, f32)) -> VectorFrame {
    let sizes = [11.0f32, 13.0, 16.0, 28.0];
    let mut root = Group::default();
    let (mut pen_x, mut baseline, mut line) = (12.0f32, 24.0f32, 0u32);
    for i in 0..count {
        let size_ix = (line as usize / 3) % sizes.len();
        let em = sizes[size_ix];
        let scale = em / UNITS_PER_EM;
        let id = (i as u32 * 17 + (i as u32 / 7) * 5) % GLYPH_KINDS;
        let advance = (455.0 + (id % 7) as f32 * 35.0) * scale;
        if pen_x + advance > W as f32 - 12.0 {
            pen_x = 12.0;
            baseline += em * 1.23;
            line += 1;
            if baseline > H as f32 - 8.0 {
                baseline = 24.0 + (i % 5) as f32 * 0.37;
            }
        }
        root.children.push(Node::Group(Group {
            transform: Transform2D {
                a: scale,
                b: 0.0,
                c: 0.0,
                d: -scale,
                e: pen_x + scroll.0,
                f: baseline + scroll.1,
            },
            cache_key: keyed.then_some(((size_ix as u64) << 32) | id as u64),
            children: vec![fill_node(
                glyph_outline(id),
                Paint::Solid(Rgba::opaque(20, 20, 24)),
            )],
            ..Group::default()
        }));
        pen_x += advance;
    }
    frame(root)
}

// --------------------------------------------------------------- fills

/// One closed outline of `n` straight edges: a ring whose radius is
/// modulated at several frequencies, filling most of the canvas.
fn many_edge_fill(n: u32) -> VectorFrame {
    let mut p = Path::new();
    let (cx, cy) = (W as f32 * 0.5, H as f32 * 0.5);
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        let r =
            1.0 + 0.18 * (a * 37.0).sin() + 0.07 * (a * 611.0).sin() + 0.02 * (a * 4099.0).cos();
        let pt = Point::new(
            cx + 900.0 * r * a.cos() * 0.8,
            cy + 500.0 * r * a.sin() * 0.8,
        );
        if i == 0 {
            p.move_to(pt);
        } else {
            p.line_to(pt);
        }
    }
    p.close();
    let mut root = Group::default();
    root.children
        .push(fill_node(p, Paint::Solid(Rgba::new(30, 90, 200, 230))));
    frame(root)
}

/// 3 000 translucent curved blobs between 4 and 120 px.
fn blobs() -> VectorFrame {
    let mut rng = Lcg(0x5EED_1234);
    let mut root = Group::default();
    for _ in 0..3000 {
        let r = if rng.next() < 0.9 {
            rng.range(4.0, 30.0)
        } else {
            rng.range(30.0, 120.0)
        };
        let (cx, cy) = (rng.range(0.0, W as f32), rng.range(0.0, H as f32));
        let c = Rgba::new(rng.byte(), rng.byte(), rng.byte(), 128 + rng.byte() / 2);
        root.children
            .push(fill_node(circle_path(cx, cy, r), Paint::Solid(c)));
    }
    frame(root)
}

/// 20 000 opaque shapes of 3–10 px.
fn tiny_shapes() -> VectorFrame {
    let mut rng = Lcg(0xABCD_EF01);
    let mut root = Group::default();
    for i in 0..20_000 {
        let (x, y) = (rng.range(0.0, W as f32), rng.range(0.0, H as f32));
        let s = rng.range(3.0, 10.0);
        let p = if i % 2 == 0 {
            rect_path(x, y, s, s * rng.range(0.5, 1.5))
        } else {
            circle_path(x, y, s * 0.5)
        };
        let c = Rgba::opaque(rng.byte(), rng.byte(), rng.byte());
        root.children.push(fill_node(p, Paint::Solid(c)));
    }
    frame(root)
}

// ------------------------------------------------------------- strokes

fn polyline(rng: &mut Lcg, segments: u32, step: f32, curved: bool) -> Path {
    let mut p = Path::new();
    let (mut x, mut y) = (
        rng.range(40.0, W as f32 - 40.0),
        rng.range(40.0, H as f32 - 40.0),
    );
    let mut heading = rng.range(0.0, std::f32::consts::TAU);
    p.move_to(Point::new(x, y));
    for _ in 0..segments {
        heading += rng.range(-2.4, 2.4);
        let (nx, ny) = (x + step * heading.cos(), y + step * heading.sin());
        if curved {
            let bend = rng.range(-0.6, 0.6) * step;
            p.quad_to(
                Point::new(
                    (x + nx) * 0.5 - bend * heading.sin(),
                    (y + ny) * 0.5 + bend * heading.cos(),
                ),
                Point::new(nx, ny),
            );
        } else {
            p.line_to(Point::new(nx, ny));
        }
        (x, y) = (nx, ny);
    }
    p
}

/// `count` open polylines stroked `width` px wide, cycling through the
/// join and cap kinds; `dash` adds a dash array.
fn strokes(count: u32, width: f32, dash: Option<Vec<f32>>, curved: bool) -> VectorFrame {
    let mut rng = Lcg(0x57_20CE);
    let mut root = Group::default();
    for i in 0..count {
        let step = rng.range(18.0, 70.0);
        let path = polyline(&mut rng, 24, step, curved);
        let stroke = Stroke {
            width,
            paint: Paint::Solid(Rgba::new(rng.byte(), rng.byte(), rng.byte(), 220)),
            cap: [LineCap::Butt, LineCap::Round, LineCap::Square][(i % 3) as usize],
            join: [LineJoin::Miter, LineJoin::Round, LineJoin::Bevel][((i / 3) % 3) as usize],
            miter_limit: 4.0,
            dash: dash.clone().map(|array| DashPattern {
                array,
                offset: i as f32 * 0.7,
            }),
        };
        root.children.push(Node::Path(PathNode {
            path,
            fill: None,
            stroke: Some(stroke),
            fill_rule: FillRule::NonZero,
        }));
    }
    frame(root)
}

// ----------------------------------------------------------- gradients

fn linear(spread: SpreadMethod, n: usize, alpha: u8) -> Paint {
    Paint::LinearGradient(LinearGradient {
        start: Point::new(200.0, 100.0),
        end: Point::new(1500.0, 900.0),
        stops: stops(n, alpha),
        spread,
    })
}

fn radial(spread: SpreadMethod, focal: Option<Point>, n: usize) -> Paint {
    Paint::RadialGradient(RadialGradient {
        center: Point::new(960.0, 540.0),
        radius: 700.0,
        focal,
        stops: stops(n, 255),
        spread,
    })
}

fn full_canvas(paint: Paint) -> VectorFrame {
    let mut root = Group::default();
    root.children
        .push(fill_node(rect_path(0.0, 0.0, W as f32, H as f32), paint));
    frame(root)
}

/// 2 000 small shapes, each with its own gradient (the per-paint setup
/// cost: stop tables, inverse transforms).
fn small_gradients() -> VectorFrame {
    let mut rng = Lcg(0x6AD_1E47);
    let mut root = Group::default();
    for i in 0..2000 {
        let (cx, cy) = (rng.range(0.0, W as f32), rng.range(0.0, H as f32));
        let r = rng.range(6.0, 28.0);
        let s = vec![
            GradientStop {
                offset: 0.0,
                color: Rgba::opaque(rng.byte(), rng.byte(), rng.byte()),
            },
            GradientStop {
                offset: 0.6,
                color: Rgba::opaque(rng.byte(), rng.byte(), rng.byte()),
            },
            GradientStop {
                offset: 1.0,
                color: Rgba::new(rng.byte(), rng.byte(), rng.byte(), 120),
            },
        ];
        let paint = if i % 2 == 0 {
            Paint::LinearGradient(LinearGradient {
                start: Point::new(cx - r, cy - r),
                end: Point::new(cx + r, cy + r),
                stops: s,
                spread: SpreadMethod::Pad,
            })
        } else {
            Paint::RadialGradient(RadialGradient {
                center: Point::new(cx, cy),
                radius: r,
                focal: None,
                stops: s,
                spread: SpreadMethod::Pad,
            })
        };
        root.children.push(fill_node(circle_path(cx, cy, r), paint));
    }
    frame(root)
}

// -------------------------------------------------------------- images

fn noise_image(w: u32, h: u32, seed: u32) -> VideoFrame {
    let mut rng = Lcg(seed);
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let a = if (x / 16 + y / 16) % 5 == 0 { 140 } else { 255 };
            data.extend_from_slice(&[rng.byte(), (x * 255 / w) as u8, (y * 255 / h) as u8, a]);
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (w * 4) as usize,
            data,
        }],
    }
}

/// 24 images of 256 × 256, placed upright (`rotate == false`) or under
/// rotation + non-uniform scale.
fn images(rotate: bool) -> VectorFrame {
    let mut root = Group::default();
    for i in 0..24u32 {
        let (col, row) = (i % 6, i / 6);
        let place = Transform2D::translate(40.0 + col as f32 * 310.0, 20.0 + row as f32 * 262.0);
        let transform = if rotate {
            place
                .compose(&Transform2D::translate(128.0, 128.0))
                .compose(&Transform2D::rotate(0.15 + i as f32 * 0.21))
                .compose(&Transform2D::scale(1.1, 0.8))
                .compose(&Transform2D::translate(-128.0, -128.0))
        } else {
            place.compose(&Transform2D::scale(1.13, 0.94))
        };
        root.children.push(Node::Image(ImageRef {
            frame: Box::new(noise_image(256, 256, 77 + i)),
            bounds: Rect::new(0.0, 0.0, 256.0, 256.0),
            transform,
        }));
    }
    frame(root)
}

fn tile_pattern() -> Pattern {
    Pattern::new(0.0, 0.0, 48.0, 36.0)
        .with_transform(Transform2D::rotate(0.4).compose(&Transform2D::scale(1.3, 0.9)))
        .with_child(fill_node(
            rect_path(0.0, 0.0, 48.0, 36.0),
            Paint::Solid(Rgba::opaque(250, 240, 220)),
        ))
        .with_child(fill_node(
            circle_path(24.0, 18.0, 13.0),
            Paint::Solid(Rgba::new(200, 40, 60, 200)),
        ))
}

// ------------------------------------------- clips, masks, group stacks

fn discs(rng: &mut Lcg, cx: f32, cy: f32, spread: f32, n: u32) -> Vec<Node> {
    (0..n)
        .map(|_| {
            let c = Rgba::new(rng.byte(), rng.byte(), rng.byte(), 200);
            fill_node(
                circle_path(
                    cx + rng.range(-spread, spread),
                    cy + rng.range(-spread, spread),
                    rng.range(8.0, spread * 0.5),
                ),
                Paint::Solid(c),
            )
        })
        .collect()
}

/// 150 groups, each clipped by a disc, inside one page-sized rounded
/// clip (two clip levels on every shape).
fn clipped() -> VectorFrame {
    let mut rng = Lcg(0xC11_9ED);
    let mut outer = Group {
        clip: Some(circle_path(960.0, 540.0, 1000.0)),
        ..Group::default()
    };
    for _ in 0..150 {
        let (cx, cy) = (rng.range(80.0, 1840.0), rng.range(80.0, 1000.0));
        outer.children.push(Node::Group(Group {
            clip: Some(circle_path(cx, cy, 70.0)),
            children: discs(&mut rng, cx, cy, 80.0, 8),
            ..Group::default()
        }));
    }
    let mut root = Group::default();
    root.children.push(Node::Group(outer));
    frame(root)
}

/// 60 soft-masked groups (luminance masks made of a radial ramp).
fn soft_masks() -> VectorFrame {
    let mut rng = Lcg(0x50F7_3A5C);
    let mut root = Group::default();
    for i in 0..60 {
        let (cx, cy) = (rng.range(120.0, 1800.0), rng.range(120.0, 960.0));
        let mask = fill_node(
            circle_path(cx, cy, 110.0),
            Paint::RadialGradient(RadialGradient {
                center: Point::new(cx, cy),
                radius: 110.0,
                focal: None,
                stops: vec![
                    GradientStop {
                        offset: 0.0,
                        color: Rgba::opaque(255, 255, 255),
                    },
                    GradientStop {
                        offset: 1.0,
                        color: Rgba::opaque(0, 0, 0),
                    },
                ],
                spread: SpreadMethod::Pad,
            }),
        );
        root.children.push(Node::SoftMask {
            mask: Box::new(mask),
            mask_kind: if i % 2 == 0 {
                MaskKind::Luminance
            } else {
                MaskKind::Alpha
            },
            content: Box::new(Node::Group(Group {
                children: discs(&mut rng, cx, cy, 100.0, 6),
                ..Group::default()
            })),
        });
    }
    frame(root)
}

/// 120 three-level group stacks, every level translucent.
fn nested_opacity() -> VectorFrame {
    let mut rng = Lcg(0x0FAC_17E5);
    let mut root = Group::default();
    for _ in 0..120 {
        let (cx, cy) = (rng.range(100.0, 1820.0), rng.range(100.0, 980.0));
        let leaf = Group {
            opacity: 0.8,
            children: discs(&mut rng, cx, cy, 50.0, 5),
            ..Group::default()
        };
        let mut mid = Group {
            opacity: 0.7,
            children: discs(&mut rng, cx, cy, 80.0, 4),
            ..Group::default()
        };
        mid.children.push(Node::Group(leaf));
        let mut top = Group {
            opacity: 0.6,
            children: discs(&mut rng, cx, cy, 110.0, 3),
            ..Group::default()
        };
        top.children.push(Node::Group(mid));
        root.children.push(Node::Group(top));
    }
    frame(root)
}

fn first_byte(f: &VideoFrame) -> u8 {
    let d = &f.planes[0].data;
    d[d.len() / 2 + 1]
}

fn render_case(name: &'static str, r: Renderer, f: VectorFrame) -> Case {
    Case {
        name,
        run: Box::new(move || first_byte(&r.render(&f))),
    }
}

/// Every case of the harness, in report order.
pub fn cases() -> Vec<Case> {
    let r = || Renderer::new(W, H);
    let mut v = Vec::new();

    // Text: thousands of glyph groups at every sub-pixel phase.
    v.push(render_case(
        "text5000_unkeyed",
        r(),
        text_page(5000, false, (0.0, 0.0)),
    ));
    {
        let f = text_page(5000, true, (0.0, 0.0));
        v.push(Case {
            name: "text5000_keyed_cold",
            run: Box::new(move || first_byte(&Renderer::new(W, H).render(&f))),
        });
    }
    {
        let f = text_page(5000, true, (0.0, 0.0));
        let rr = r();
        rr.render(&f);
        v.push(render_case("text5000_keyed_warm", rr, f));
    }
    {
        // A renderer per frame around one cache, the page scrolling by
        // a fractional amount each frame.
        let pages: Vec<VectorFrame> = (0..8)
            .map(|k| text_page(5000, true, (k as f32 * 0.37, k as f32 * 3.3)))
            .collect();
        let cache = GroupCache::new();
        let mut k = 0usize;
        v.push(Case {
            name: "text5000_keyed_scrolling",
            run: Box::new(move || {
                k += 1;
                let rr = Renderer::new(W, H).with_group_cache(cache.clone());
                first_byte(&rr.render(&pages[k % pages.len()]))
            }),
        });
    }

    // Fills.
    v.push(render_case("fill_60k_edges", r(), many_edge_fill(60_000)));
    v.push(render_case("fill_blobs3000", r(), blobs()));
    v.push(render_case("fill_tiny20000", r(), tiny_shapes()));

    // Strokes.
    v.push(render_case(
        "stroke_thin600",
        r(),
        strokes(600, 1.5, None, false),
    ));
    v.push(render_case(
        "stroke_hairline600",
        r(),
        strokes(600, 0.35, None, false),
    ));
    v.push(render_case(
        "stroke_wide_curved300",
        r(),
        strokes(300, 9.0, None, true),
    ));
    v.push(render_case(
        "stroke_dashed600",
        r(),
        strokes(600, 2.0, Some(vec![7.0, 3.0, 2.0, 3.0]), false),
    ));

    // Gradients.
    v.push(render_case(
        "grad_linear_pad",
        r(),
        full_canvas(linear(SpreadMethod::Pad, 6, 255)),
    ));
    v.push(render_case(
        "grad_linear_reflect_alpha",
        r(),
        full_canvas(linear(SpreadMethod::Reflect, 6, 160)),
    ));
    v.push(render_case(
        "grad_radial_pad",
        r(),
        full_canvas(radial(SpreadMethod::Pad, None, 6)),
    ));
    v.push(render_case(
        "grad_radial_focal_repeat",
        r(),
        full_canvas(radial(
            SpreadMethod::Repeat,
            Some(Point::new(700.0, 400.0)),
            6,
        )),
    ));
    v.push(render_case("grad_small2000", r(), small_gradients()));

    // Images and patterns.
    v.push(render_case("image_upright24", r(), images(false)));
    v.push(render_case("image_rotated24", r(), images(true)));
    {
        let (rr, pat) = (r(), tile_pattern());
        let path = circle_path(960.0, 540.0, 520.0);
        v.push(Case {
            name: "pattern_fill_rotated",
            run: Box::new(move || {
                first_byte(&rr.fill_path_with_pattern(
                    &path,
                    FillRule::NonZero,
                    &pat,
                    Transform2D::identity(),
                ))
            }),
        });
    }

    // Clips, masks, group stacks, blend modes.
    v.push(render_case("clip_nested150", r(), clipped()));
    v.push(render_case("softmask60", r(), soft_masks()));
    v.push(render_case("opacity_nested120", r(), nested_opacity()));
    {
        let mut rr = r();
        rr.blend_mode = BlendMode::Multiply;
        rr.background = Rgba::opaque(255, 255, 255);
        v.push(render_case("blend_multiply_blobs3000", rr, blobs()));
    }
    {
        let mut rr = r();
        rr.blend_mode = BlendMode::Luminosity;
        rr.background = Rgba::opaque(200, 180, 90);
        v.push(render_case("blend_luminosity_blobs3000", rr, blobs()));
    }
    v
}
