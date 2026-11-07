//! Round 484: inputs found by the `render_paths` fuzz target that used
//! to exhaust memory (a stroke pen of astronomical width made the curve
//! flattener emit hundreds of millions of points) or time (a sub-pixel
//! dash pattern under a strongly anisotropic CTM produced 10⁵ dashes
//! with 4 096-chord caps). The decoder is the fuzz target's.

use oxideav_core::{
    DashPattern, ExecutionContext, FillRule, Group, LineCap, LineJoin, Node, Paint, Path, PathNode,
    Point, Rgba, Stroke, Transform2D, VectorFrame,
};
use oxideav_raster::{CoverageMode, InterpolationSpace, Renderer};

fn f32_at(d: &[u8], i: usize) -> f32 {
    d.get(i..i + 4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .unwrap_or(0.0)
}

fn run(data: &[u8]) {
    if data.len() < 39 {
        return;
    }
    let w = (data[0] % 40) as u32 + 1;
    let h = (data[1] % 40) as u32 + 1;
    let flags = data[2];
    let t = Transform2D {
        a: f32_at(data, 15),
        b: f32_at(data, 19),
        c: f32_at(data, 23),
        d: f32_at(data, 27),
        e: f32_at(data, 31),
        f: f32_at(data, 35),
    };
    let mut path = Path::new();
    let mut i = 39;
    let pt = |i: usize| Point::new(f32_at(data, i), f32_at(data, i + 4));
    let mut ops = 0;
    while i < data.len() && ops < 64 {
        let op = data[i] % 5;
        i += 1;
        match op {
            0 => {
                path.move_to(pt(i));
                i += 8;
            }
            1 => {
                path.line_to(pt(i));
                i += 8;
            }
            2 => {
                path.quad_to(pt(i), pt(i + 8));
                i += 16;
            }
            3 => {
                path.cubic_to(pt(i), pt(i + 8), pt(i + 16));
                i += 24;
            }
            _ => {
                path.close();
            }
        }
        ops += 1;
    }
    let stroke = (flags & 2 != 0).then(|| Stroke {
        width: f32_at(data, 3),
        paint: Paint::Solid(Rgba::new(10, 200, 30, 180)),
        cap: [
            LineCap::Butt,
            LineCap::Round,
            LineCap::Square,
            LineCap::Round,
        ][((flags >> 3) & 3) as usize],
        join: [
            LineJoin::Miter,
            LineJoin::Round,
            LineJoin::Bevel,
            LineJoin::Miter,
        ][((flags >> 5) & 3) as usize],
        miter_limit: 4.0,
        dash: (flags & 4 != 0).then(|| DashPattern {
            array: vec![f32_at(data, 7), f32_at(data, 11)],
            offset: 0.0,
        }),
    });
    let node = Node::Path(PathNode {
        path,
        fill: Some(Paint::Solid(Rgba::new(200, 40, 90, 200))),
        stroke,
        fill_rule: if flags & 1 != 0 {
            FillRule::EvenOdd
        } else {
            FillRule::NonZero
        },
    });
    let frame = VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root: Group {
            transform: t,
            children: vec![node],
            ..Group::default()
        },
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let space = if flags & 0x80 != 0 {
        InterpolationSpace::LinearRgb
    } else {
        InterpolationSpace::Srgb
    };
    let r = Renderer::new(w, h).with_compositing_space(space);
    let serial = r.render(&frame);
    assert_eq!(serial.planes[0].data.len(), (w * h * 4) as usize);
    let par = r
        .clone()
        .with_execution_context(&ExecutionContext::with_threads(3))
        .render(&frame);
    assert_eq!(serial.planes[0].data, par.planes[0].data);
    let ss = Renderer::new(w, h)
        .with_coverage(CoverageMode::Supersampled)
        .render(&frame);
    assert_eq!(ss.planes[0].data.len(), (w * h * 4) as usize);
}

#[test]
fn astronomical_pen_is_bounded() {
    let t = std::time::Instant::now();
    run(&[
        0x00, 0x7f, 0xff, 0xff, 0x00, 0x00, 0x7f, 0x00, 0xff, 0x3d, 0xff, 0x00, 0x00, 0x00, 0x05,
        0x00, 0x00, 0x00, 0xe7, 0xff, 0xff, 0x2a, 0xdf, 0x7f, 0xff, 0xff, 0x00, 0x00, 0x7f, 0x00,
        0xff, 0x3d, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0xff, 0x12, 0x00, 0x00, 0x2b, 0x0a, 0x0a,
        0x20, 0xcc, 0x00,
    ]);
    assert!(t.elapsed().as_secs() < 20);
}

#[test]
fn sub_pixel_dashes_under_anisotropic_ctm_are_bounded() {
    let t = std::time::Instant::now();
    run(&[
        0x0a, 0xff, 0xff, 0x00, 0xde, 0xfb, 0x09, 0x0a, 0x00, 0x22, 0xf7, 0x00, 0x06, 0xee, 0x00,
        0x00, 0x37, 0x00, 0x00, 0xa7, 0x00, 0x00, 0xf7, 0x07, 0x2b, 0x7f, 0x7f, 0x01, 0x00, 0x7f,
        0x00, 0x26, 0xde, 0xff, 0x22, 0x0a, 0x00, 0x00, 0x2d, 0x01, 0x00, 0xff, 0xff, 0x06, 0x7e,
        0x00, 0x3e,
    ]);
    assert!(t.elapsed().as_secs() < 20);
}
