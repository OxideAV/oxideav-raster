//! Round 484: conflation (seams between abutting shapes), linear-light
//! compositing, and hairline coverage.

use oxideav_core::{
    FillRule, Group, LineCap, LineJoin, Node, Paint, Path, PathNode, Point, Rgba, Stroke,
    VectorFrame,
};
use oxideav_raster::{InterpolationSpace, Renderer};

fn frame(w: u32, h: u32, children: Vec<Node>) -> VectorFrame {
    VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root: Group {
            children,
            ..Group::default()
        },
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32, c: Rgba) -> Node {
    let mut p = Path::new();
    p.move_to(Point::new(x0, y0))
        .line_to(Point::new(x1, y0))
        .line_to(Point::new(x1, y1))
        .line_to(Point::new(x0, y1))
        .close();
    Node::Path(PathNode {
        path: p,
        fill: Some(Paint::Solid(c)),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
}

/// Two black rectangles meeting at x = 10.3 on white: the seam column
/// should be black. Per-shape compositing lets 0.3·0.7 = 21 % of the
/// white through (a visible light seam); whole-scene supersampling
/// bounds the bleed by 1/(4k).
#[test]
fn scene_supersampling_closes_conflation_seams() {
    let black = Rgba::opaque(0, 0, 0);
    let f = frame(
        24,
        4,
        vec![
            rect(0.0, 0.0, 10.3, 4.0, black),
            rect(10.3, 0.0, 24.0, 4.0, black),
        ],
    );
    let seam = |r: Renderer| -> u8 {
        let out = r.render(&f);
        let d = &out.planes[0].data;
        // Every other column is solid black.
        for x in (0..24).filter(|&x| x != 10) {
            assert_eq!(&d[(24 + x) * 4..(24 + x) * 4 + 3], &[0, 0, 0], "x={x}");
        }
        d[(24 + 10) * 4]
    };
    let white = Rgba::opaque(255, 255, 255);
    let plain = seam({
        let mut r = Renderer::new(24, 4);
        r.background = white;
        r
    });
    assert!(plain > 40, "per-shape AA seam should show: {plain}");
    for k in [4u8, 8] {
        let mut r = Renderer::new(24, 4).with_scene_supersampling(k);
        r.background = white;
        let v = seam(r);
        let bound = (255.0 / (4.0 * k as f32)).ceil() as u8 + 1;
        assert!(v <= bound, "k={k}: seam {v} > {bound}");
    }
    // k = 1 is the default and changes nothing.
    assert_eq!(Renderer::new(1, 1).scene_supersampling, 1);
}

/// A half-covered black-on-white edge pixel: sRGB-space compositing
/// mixes the encoded values (128), linear-light compositing mixes light
/// (sRGB encoding of 0.5 = 188). A 50 % red over opaque green likewise.
#[test]
fn linear_light_compositing_mixes_light() {
    let f = frame(4, 1, vec![rect(0.0, 0.0, 1.5, 1.0, Rgba::opaque(0, 0, 0))]);
    let px = |space: InterpolationSpace| {
        let mut r = Renderer::new(4, 1).with_compositing_space(space);
        r.background = Rgba::opaque(255, 255, 255);
        let out = r.render(&f);
        [out.planes[0].data[4], out.planes[0].data[7]]
    };
    let srgb = px(InterpolationSpace::Srgb);
    assert!(
        (srgb[0] as i32 - 128).abs() <= 1 && srgb[1] == 255,
        "{srgb:?}"
    );
    let lin = px(InterpolationSpace::LinearRgb);
    assert!((lin[0] as i32 - 188).abs() <= 1, "{lin:?}");
    assert_eq!(lin[1], 255);
    assert_eq!(
        Renderer::new(1, 1).compositing_space,
        InterpolationSpace::Srgb
    );

    let f = frame(
        1,
        1,
        vec![rect(0.0, 0.0, 1.0, 1.0, Rgba::new(255, 0, 0, 128))],
    );
    let mix = |space: InterpolationSpace| {
        let mut r = Renderer::new(1, 1).with_compositing_space(space);
        r.background = Rgba::opaque(0, 255, 0);
        r.render(&f).planes[0].data[..4].to_vec()
    };
    let m = mix(InterpolationSpace::Srgb);
    assert!(
        (m[0] as i32 - 128).abs() <= 1 && (m[1] as i32 - 127).abs() <= 1,
        "{m:?}"
    );
    let m = mix(InterpolationSpace::LinearRgb);
    assert!(
        (m[0] as i32 - 188).abs() <= 1 && (m[1] as i32 - 187).abs() <= 1,
        "{m:?}"
    );
}

/// Hairlines never drop out: a 0.2 px wide line at any angle and
/// sub-pixel offset deposits exactly its area (exact-area coverage),
/// and every pixel column it crosses receives some coverage.
#[test]
fn hairlines_keep_their_area_at_any_offset() {
    let n = 48u32;
    for case in 0..16 {
        let ang = case as f32 * 0.37;
        let off = case as f32 * 0.173;
        let (s, c) = ang.sin_cos();
        let (cx, cy) = (24.0 + off, 24.0 - off * 0.5);
        let len = 30.0;
        let a = Point::new(cx - c * len / 2.0, cy - s * len / 2.0);
        let b = Point::new(cx + c * len / 2.0, cy + s * len / 2.0);
        let mut p = Path::new();
        p.move_to(a).line_to(b);
        let f = frame(
            n,
            n,
            vec![Node::Path(PathNode {
                path: p,
                fill: None,
                stroke: Some(Stroke {
                    width: 0.2,
                    paint: Paint::Solid(Rgba::opaque(0, 0, 0)),
                    cap: LineCap::Butt,
                    join: LineJoin::Miter,
                    miter_limit: 4.0,
                    dash: None,
                }),
                fill_rule: FillRule::NonZero,
            })],
        );
        let out = Renderer::new(n, n).render(&f);
        let alpha: Vec<f32> = out.planes[0]
            .data
            .chunks(4)
            .map(|p| p[3] as f32 / 255.0)
            .collect();
        let total: f32 = alpha.iter().sum();
        assert!(
            (total - 0.2 * len).abs() < 0.15,
            "case {case}: area {total}, want {}",
            0.2 * len
        );
        if c.abs() > 0.5 {
            // Shallow line: each crossed column is touched.
            let (x0, x1) = (a.x.min(b.x).ceil() as u32, b.x.max(a.x).floor() as u32);
            for x in x0..x1 {
                let col: f32 = (0..n).map(|y| alpha[(y * n + x) as usize]).sum();
                assert!(col > 0.1, "case {case}: column {x} dropped out ({col})");
            }
        }
    }
}

/// Linear-light compositing applies to isolated groups too: a group's
/// offscreen image, a cached bitmap and a soft-masked subtree are laid
/// over the canvas by mixing light, so the same glyph has the same
/// weight whether it is drawn directly, through a translucent group or
/// from the keyed-group cache. (The blits used to mix encoded values:
/// a half-covered black-on-white edge pixel then read 128 in a keyed
/// glyph and 188 in an unkeyed one.)
#[test]
fn linear_light_compositing_covers_group_and_cache_blits() {
    let ink = |key: Option<u64>, opacity: f32| {
        let mut p = Path::new();
        // A slanted bar: plenty of partially covered pixels.
        p.move_to(Point::new(6.0, 4.0))
            .line_to(Point::new(14.5, 4.0))
            .line_to(Point::new(27.5, 28.0))
            .line_to(Point::new(19.0, 28.0))
            .close();
        Group {
            children: vec![Node::Group(Group {
                cache_key: key,
                opacity,
                children: vec![Node::Path(PathNode {
                    path: p,
                    fill: Some(Paint::Solid(Rgba::opaque(0, 0, 0))),
                    stroke: None,
                    fill_rule: FillRule::NonZero,
                })],
                ..Group::default()
            })],
            ..Group::default()
        }
    };
    let render = |root: Group, space: InterpolationSpace| {
        let mut r = Renderer::new(32, 32).with_compositing_space(space);
        r.background = Rgba::opaque(255, 255, 255);
        r.render(&frame(32, 32, root.children))
            .planes
            .remove(0)
            .data
    };
    let worst = |a: &[u8], b: &[u8]| a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
    for space in [InterpolationSpace::LinearRgb, InterpolationSpace::Srgb] {
        let direct = render(ink(None, 1.0), space);
        let keyed = render(ink(Some(7), 1.0), space);
        assert!(
            worst(&direct, &keyed) <= 1,
            "{space:?}: keyed differs from direct by {}",
            worst(&direct, &keyed)
        );
    }
    // The two spaces do differ on the edge pixels, keyed or not.
    let lin = render(ink(Some(7), 1.0), InterpolationSpace::LinearRgb);
    let srgb = render(ink(Some(7), 1.0), InterpolationSpace::Srgb);
    assert!(worst(&lin, &srgb) > 40);
    // A translucent group over white in linear light: 50 % black is
    // half the light, sRGB 188 — not 128.
    let half = render(ink(None, 128.0 / 255.0), InterpolationSpace::LinearRgb);
    let inside = ((16 * 32 + 17) * 4) as usize;
    assert!(half[inside].abs_diff(188) <= 2, "interior {}", half[inside]);
}
