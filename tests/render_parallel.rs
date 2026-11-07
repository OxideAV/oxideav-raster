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

#[test]
fn default_renderer_is_serial() {
    assert_eq!(Renderer::new(4, 4).threads, 1);
}
