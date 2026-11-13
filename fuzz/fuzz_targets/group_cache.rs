#![no_main]

//! Group-cache fuzz target: a scene of keyed groups (few distinct keys,
//! many placements, nesting, clips, opacities, quarter-turn and
//! power-of-two transforms) rendered through caches in every state —
//! cold, warm, starved of bytes, disabled, and shared with a renderer
//! of different settings. Whether a group is blitted from a bitmap,
//! rasterised afresh or rendered through a transient window (too large
//! for the budget, or mostly off-canvas) must never change a byte, so
//! every render of one renderer configuration has to be identical; an
//! entry reused for different content (a key collision) would show up
//! as a difference, because every key paints its own colour and shape.
//!
//! All geometry is axis-aligned with coordinates on a 1/16 px lattice,
//! which the rasteriser handles exactly at any position — that is what
//! makes byte equality (rather than a tolerance) the right oracle.
//!
//! ## Wire framing
//!
//! * byte 0 — canvas width, 1..=48; byte 1 — canvas height, 1..=48
//! * byte 2 — sub-pixel grid selector (0, 1, 2, 4, 8, 16)
//! * byte 3 — byte budget of the starved cache (256 · (1 + b) bytes)
//! * then 6-byte placement records: key selector, x (i8, 1/2 px) and
//!   y (i8, 1/2 px), sub-pixel bits (x in the low nibble, y in the
//!   high, 1/16 px), transform selector, flags (bits 0-1 opacity,
//!   bit 2 nest inside the previous placement, bit 3 unkeyed)

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    FillRule, Group, Node, Paint, Path, PathNode, Point, Rgba, Transform2D, VectorFrame,
};
use oxideav_raster::{CoverageMode, GroupCache, Renderer};

fn rect(x: f32, y: f32, w: f32, h: f32) -> Path {
    let mut p = Path::new();
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
    p
}

fn fill(path: Path, color: Rgba) -> Node {
    Node::Path(PathNode {
        path,
        fill: Some(Paint::Solid(color)),
        stroke: None,
        fill_rule: FillRule::EvenOdd,
    })
}

/// The content a key stands for — a pure function of the key, as the
/// cache contract requires of producers.
fn content(key: u8) -> (Vec<Node>, Option<Path>) {
    let k = key as f32;
    let color = Rgba::new(40 + 25 * key, 250 - 30 * key, 17 * key, 255 - 20 * (key & 3));
    let mut nodes = vec![fill(rect(0.5 * k, 0.25 * k, 3.0 + k, 2.5 + 0.5 * k), color)];
    if key & 1 != 0 {
        // A frame: outer and inner rectangle under the even-odd rule.
        let mut ring = rect(-1.0, -1.0, 9.0 + k, 7.0);
        ring.commands
            .extend(rect(0.75, 0.5, 5.0, 3.0 + 0.125 * k).commands);
        nodes.push(fill(ring, Rgba::new(250, 10 * key, 200, 140)));
    }
    let clip = (key & 2 != 0).then(|| rect(0.25, -0.5, 6.0 + 0.5 * k, 5.5));
    (nodes, clip)
}

fn linear(sel: u8) -> Transform2D {
    match sel % 6 {
        0 => Transform2D::identity(),
        1 => Transform2D::scale(2.0, 2.0),
        2 => Transform2D::scale(0.5, 0.5),
        3 => Transform2D {
            a: 0.0,
            b: 1.0,
            c: -1.0,
            d: 0.0,
            e: 0.0,
            f: 0.0,
        },
        4 => Transform2D::scale(-1.0, 1.0),
        _ => Transform2D::scale(4.0, 0.25),
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let w = (data[0] % 48) as u32 + 1;
    let h = (data[1] % 48) as u32 + 1;
    let grid = [0u8, 1, 2, 4, 8, 16][(data[2] % 6) as usize];
    let starved_bytes = 256 * (1 + data[3] as usize);
    let mut root = Group::default();
    for rec in data[4..].chunks_exact(6).take(48) {
        let key = rec[0] % 8;
        let (children, clip) = content(key);
        let x = (rec[1] as i8) as f32 * 0.5 + (rec[3] & 15) as f32 / 16.0;
        let y = (rec[2] as i8) as f32 * 0.5 + (rec[3] >> 4) as f32 / 16.0;
        let group = Group {
            transform: Transform2D::translate(x, y).compose(&linear(rec[4])),
            opacity: [1.0, 0.5, 0.25, 0.0][(rec[5] & 3) as usize],
            clip,
            children,
            // The key names the content only; the transform, phase and
            // renderer settings are the cache's business.
            cache_key: (rec[5] & 8 == 0).then_some(key as u64),
        };
        match root.children.last_mut() {
            Some(Node::Group(prev)) if rec[5] & 4 != 0 => {
                // The outer group now has other content than its key
                // promises, so it stops being cacheable.
                prev.cache_key = None;
                prev.children.push(Node::Group(group));
            }
            _ => root.children.push(Node::Group(group)),
        }
    }
    let frame = VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let base = || Renderer::new(w, h).with_cache_subpixel_grid(grid);
    let px = |r: &Renderer| r.render(&frame).planes.swap_remove(0).data;

    let cold = base();
    let reference = px(&cold);
    assert_eq!(reference.len(), (w * h * 4) as usize);
    // Warm: every lookup that missed is now a hit.
    assert_eq!(px(&cold), reference, "warm cache changed the picture");
    // Disabled: every keyed group through a transient window.
    assert_eq!(px(&base().with_cache_bytes(0)), reference, "uncached draw differs");
    // Starved: constant eviction, bitmaps over a quarter of it uncached.
    let starved = base().with_cache_bytes(starved_bytes);
    assert_eq!(px(&starved), reference, "starved cache changed the picture");
    assert_eq!(px(&starved), reference);
    let s = starved.cache_stats();
    assert!(s.bytes <= s.max_bytes);
    // Shared with a renderer whose settings differ, and with one on a
    // different canvas: neither may pick up the other's bitmaps.
    let shared: GroupCache = cold.group_cache();
    let other = || base().with_coverage(CoverageMode::Supersampled);
    let other_alone = px(&other());
    assert_eq!(px(&other().with_group_cache(shared.clone())), other_alone);
    assert_eq!(px(&cold), reference, "foreign entries leaked in");
    let wide = Renderer::new(w + 9, h + 5).with_cache_subpixel_grid(grid);
    let wide_alone = px(&wide);
    assert_eq!(px(&wide.clone().with_group_cache(shared.clone())), wide_alone);
});
