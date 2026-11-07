#![no_main]

//! Scene-rendering fuzz target: fuzzer bytes become filled and stroked
//! paths (coordinates, stroke widths, dash lengths and the transform
//! all from raw `f32` bits, so NaN / ±inf / denormal / 1e38 values
//! reach the flattener, the stroker and the scan converter) rendered
//! through every coverage / compositing option on a small canvas. The
//! contract is a panic-free render returning an exactly
//! `width × height × 4` buffer, byte-identical between the serial and
//! the band-parallel paths.
//!
//! ## Wire framing
//!
//! * byte 0 — canvas width, clamped to 1..=40
//! * byte 1 — canvas height, clamped to 1..=40
//! * byte 2 — flags: bit 0 even-odd, bit 1 stroke, bit 2 dash,
//!   bits 3-4 cap, bits 5-6 join, bit 7 linear-light compositing
//! * bytes 3..7 — stroke width (f32 bits)
//! * bytes 7..15 — two dash lengths (f32 bits)
//! * bytes 15..39 — transform a b c d e f (f32 bits)
//! * bytes 39.. — path records, 1 opcode byte + up to three points of
//!   8 bytes each (`op % 5`: move, line, quad, cubic, close)

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    DashPattern, ExecutionContext, FillRule, Group, LineCap, LineJoin, Node, Paint, Path,
    PathNode, Point, Rgba, Stroke, Transform2D, VectorFrame,
};
use oxideav_raster::{CoverageMode, InterpolationSpace, Renderer};

fn f32_at(d: &[u8], i: usize) -> f32 {
    d.get(i..i + 4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .unwrap_or(0.0)
}

fuzz_target!(|data: &[u8]| {
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
        cap: [LineCap::Butt, LineCap::Round, LineCap::Square, LineCap::Round]
            [((flags >> 3) & 3) as usize],
        join: [LineJoin::Miter, LineJoin::Round, LineJoin::Bevel, LineJoin::Miter]
            [((flags >> 5) & 3) as usize],
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
});
