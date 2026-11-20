//! A page of text through the keyed-group cache: thousands of
//! glyph-sized groups at arbitrary sub-pixel pen positions on a large
//! canvas.
//!
//! Three properties are pinned down, each by measurement rather than by
//! timing:
//!
//! * **a miss allocates a glyph-sized buffer.** A counting allocator
//!   records every allocation made while a cold page renders: apart
//!   from the output canvas itself nothing comes within two orders of
//!   magnitude of the canvas size.
//! * **keys are independent of position up to the sub-pixel phase.**
//!   The number of bitmaps a page needs is bounded by (distinct glyphs)
//!   × (phases of the sub-pixel grid), whatever the number of glyph
//!   instances and wherever the page is scrolled to.
//! * **the cache is bounded in bytes**, evicts least recently used
//!   bitmaps, and the picture never depends on what it holds.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use oxideav_core::{
    FillRule, Group, Node, Paint, Path, PathNode, Point, Rgba, TimeBase, Transform2D, VectorFrame,
};
use oxideav_raster::{GroupCache, Renderer};

/// Allocations of at least `THRESHOLD` bytes made while `RECORDING`.
struct Counting;

static RECORDING: AtomicBool = AtomicBool::new(false);
static THRESHOLD: AtomicUsize = AtomicUsize::new(usize::MAX);
static BIG_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static LARGEST_BELOW: AtomicUsize = AtomicUsize::new(0);
static TOTAL_BYTES: AtomicUsize = AtomicUsize::new(0);

fn note(size: usize) {
    if RECORDING.load(Ordering::Relaxed) {
        TOTAL_BYTES.fetch_add(size, Ordering::Relaxed);
        if size >= THRESHOLD.load(Ordering::Relaxed) {
            BIG_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        } else {
            LARGEST_BELOW.fetch_max(size, Ordering::Relaxed);
        }
    }
}

// SAFETY: defers every request to the system allocator unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The counters are process-wide: tests in this file take turns.
static TURN: Mutex<()> = Mutex::new(());

struct Recorded {
    /// Allocations of at least the threshold.
    big: usize,
    /// Largest allocation below the threshold.
    largest_below: usize,
    /// Sum of every requested size.
    total: usize,
}

fn record<T>(threshold: usize, f: impl FnOnce() -> T) -> (T, Recorded) {
    THRESHOLD.store(threshold, Ordering::Relaxed);
    BIG_ALLOCATIONS.store(0, Ordering::Relaxed);
    LARGEST_BELOW.store(0, Ordering::Relaxed);
    TOTAL_BYTES.store(0, Ordering::Relaxed);
    RECORDING.store(true, Ordering::Relaxed);
    let out = f();
    RECORDING.store(false, Ordering::Relaxed);
    (
        out,
        Recorded {
            big: BIG_ALLOCATIONS.load(Ordering::Relaxed),
            largest_below: LARGEST_BELOW.load(Ordering::Relaxed),
            total: TOTAL_BYTES.load(Ordering::Relaxed),
        },
    )
}

const W: u32 = 1920;
const H: u32 = 1080;
const GLYPH_KINDS: u32 = 96;
const SIZES: [f32; 4] = [11.0, 13.0, 16.0, 28.0];

/// A glyph-like outline in font units (Y up): a wobbly ring with a
/// counter.
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

/// `count` glyph groups set in lines with fractional advances and line
/// spacing, the whole page moved by `scroll`.
fn text_page(count: usize, keyed: bool, scroll: (f32, f32)) -> VectorFrame {
    let mut root = Group::default();
    let (mut pen_x, mut baseline, mut line) = (12.0f32, 24.0f32, 0u32);
    for i in 0..count {
        let size_ix = (line as usize / 3) % SIZES.len();
        let em = SIZES[size_ix];
        let scale = em / 1000.0;
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
            children: vec![Node::Path(PathNode {
                path: glyph_outline(id),
                fill: Some(Paint::Solid(Rgba::opaque(20, 20, 24))),
                stroke: None,
                fill_rule: FillRule::NonZero,
            })],
            ..Group::default()
        }));
        pen_x += advance;
    }
    VectorFrame {
        width: W as f32,
        height: H as f32,
        view_box: None,
        root,
        pts: None,
        time_base: TimeBase::new(1, 1),
    }
}

fn pixels(r: &Renderer, f: &VectorFrame) -> Vec<u8> {
    r.render(f).planes.remove(0).data
}

const GLYPHS: usize = 6000;
const CANVAS_BYTES: usize = (W as usize) * (H as usize) * 4;
/// The largest glyph box on the page is under 40 × 40 px (6 400 bytes
/// of RGBA); anything of this size or more is "not glyph-sized".
const NOT_GLYPH_SIZED: usize = CANVAS_BYTES / 100;

#[test]
fn a_cold_page_allocates_one_canvas_and_glyph_sized_buffers() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let page = text_page(GLYPHS, true, (0.0, 0.0));
    let r = Renderer::new(W, H);
    // A first frame grows the cache's index tables (a few hundred
    // kilobytes for thousands of entries, kept by `clear`); the frame
    // measured is the second cold one.
    pixels(&r, &page);
    r.group_cache().clear();
    r.reset_cache_stats();
    let (out, rec) = record(NOT_GLYPH_SIZED, || pixels(&r, &page));
    assert!(out.iter().any(|&b| b != 0));
    let st = r.cache_stats();
    assert_eq!(st.misses, st.entries as u64);
    assert!(st.misses > 2000, "{} misses", st.misses);
    println!(
        "cold page: {} misses / {} hits, {} cached bytes; allocations of {} bytes or more: {}, \
         largest below that: {} bytes, total requested: {} bytes (canvas: {CANVAS_BYTES})",
        st.misses, st.hits, st.bytes, NOT_GLYPH_SIZED, rec.big, rec.largest_below, rec.total
    );
    // Exactly one large allocation: the output canvas.
    assert_eq!(rec.big, 1, "a miss allocated a canvas-scale buffer");
    // Everything else is glyph-sized: the largest glyph box on the
    // page is under 48 × 48 px.
    assert!(
        rec.largest_below <= 48 * 48 * 4,
        "largest non-canvas allocation: {} bytes",
        rec.largest_below
    );
    // All allocations of the frame together stay within a small
    // multiple of the canvas — 6 000 canvas-sized buffers would be
    // 50 GB.
    assert!(
        rec.total < 4 * CANVAS_BYTES,
        "{} bytes requested",
        rec.total
    );
    // The cached bitmaps are glyph boxes.
    assert!(st.bytes / st.entries.max(1) < 8 << 10);
}

#[test]
fn bitmaps_are_shared_across_positions_up_to_the_sub_pixel_phase() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let kinds = (GLYPH_KINDS as usize) * SIZES.len();
    for (grid, phases) in [(1u8, 1usize), (2, 4), (4, 16)] {
        let cache = GroupCache::new();
        let page = text_page(GLYPHS, true, (0.0, 0.0));
        let r = Renderer::new(W, H)
            .with_group_cache(cache.clone())
            .with_cache_subpixel_grid(grid);
        pixels(&r, &page);
        let cold = cache.stats();
        assert_eq!(cold.hits + cold.misses, GLYPHS as u64);
        assert_eq!(cold.misses, cold.entries as u64);
        assert!(
            cold.entries <= kinds * phases,
            "grid {grid}: {} bitmaps for {kinds} glyphs × {phases} phases",
            cold.entries
        );
        // The same page again: every glyph is a hit.
        cache.reset_stats();
        pixels(&r, &page);
        let warm = cache.stats();
        assert_eq!((warm.misses, warm.hits), (0, GLYPHS as u64));
        // Scrolled by whole pixels: still no miss, by a new renderer.
        cache.reset_stats();
        let moved = text_page(GLYPHS, true, (37.0, -12.0));
        let r2 = Renderer::new(W, H)
            .with_group_cache(cache.clone())
            .with_cache_subpixel_grid(grid);
        pixels(&r2, &moved);
        let scrolled = cache.stats();
        assert_eq!(scrolled.misses, 0, "grid {grid}: whole-pixel scroll missed");
        // Scrolled by arbitrary fractions, many times: the cache
        // converges on at most one bitmap per glyph and phase.
        for k in 1..=24 {
            let f = text_page(GLYPHS, true, (k as f32 * 0.377, k as f32 * 1.913));
            pixels(&r2, &f);
        }
        let after = cache.stats();
        assert!(
            after.entries <= kinds * phases,
            "grid {grid}: {} bitmaps after scrolling",
            after.entries
        );
        println!(
            "grid {grid}: cold {} misses for {GLYPHS} glyphs; after 24 fractional scrolls \
             {} bitmaps (bound {}), {} bytes, hit rate {:.1} %",
            cold.misses,
            after.entries,
            kinds * phases,
            after.bytes,
            100.0 * after.hits as f64 / (after.hits + after.misses) as f64
        );
    }
}

#[test]
fn a_small_byte_budget_bounds_the_cache_without_changing_the_picture() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let page = text_page(GLYPHS, true, (0.3, 0.6));
    let roomy = Renderer::new(W, H);
    let reference = pixels(&roomy, &page);
    let full = roomy.cache_stats();
    assert_eq!(full.evictions, 0);
    // A budget an order of magnitude below what the page wants.
    let budget = full.bytes / 12;
    let cache = GroupCache::with_max_bytes(budget);
    let tight = Renderer::new(W, H).with_group_cache(cache.clone());
    for round in 0..3 {
        let got = pixels(&tight, &page);
        let nd = got.iter().zip(&reference).filter(|(a, b)| a != b).count();
        let first = got.iter().zip(&reference).position(|(a, b)| a != b);
        if let Some(i) = first {
            println!(
                "round {round}: {nd} differ; first at px ({}, {}) ch {}: {:?} vs {:?}",
                (i / 4) % W as usize,
                i / 4 / W as usize,
                i % 4,
                &got[i / 4 * 4..i / 4 * 4 + 4],
                &reference[i / 4 * 4..i / 4 * 4 + 4]
            );
        }
        assert!(got == reference, "round {round}");
        let st = cache.stats();
        assert!(
            st.bytes <= budget,
            "{} bytes cached over a {budget} budget",
            st.bytes
        );
        assert!(st.evictions > 0);
    }
    // No budget at all: every glyph is drawn directly, same picture.
    let none = Renderer::new(W, H).with_cache_bytes(0);
    assert!(pixels(&none, &page) == reference);
    assert_eq!(none.cache_stats().entries, 0);
    println!(
        "page wants {} bytes in {} bitmaps; a {budget}-byte budget holds {} of them",
        full.bytes,
        full.entries,
        cache.stats().entries
    );
}

#[test]
fn large_and_partly_visible_groups_render_the_same_at_every_budget() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    // Curved keyed panels of 300–400 px at fractional positions far from
    // the origin, some hanging off the canvas edges, plus a line of
    // glyphs. With the default budget they are all cached; with a small
    // one only the glyphs are; with none nothing is. Whatever is not
    // stored is rasterised in the frame the stored bitmap would have
    // had, so every budget — and every thread budget — produces the
    // same bytes.
    let mut root = text_page(400, true, (0.4, 0.7)).root;
    for i in 0..9u32 {
        let (px, py) = (
            -90.3 + (i % 3) as f32 * 811.37,
            -60.9 + (i / 3) as f32 * 447.61,
        );
        let mut panel = Path::new();
        let (w, h) = (300.0 + i as f32 * 11.0, 330.0 - i as f32 * 7.0);
        panel
            .move_to(Point::new(0.0, h * 0.5))
            .quad_to(Point::new(0.0, 0.0), Point::new(w * 0.5, 0.0))
            .quad_to(Point::new(w, 0.0), Point::new(w, h * 0.5))
            .quad_to(Point::new(w, h), Point::new(w * 0.5, h))
            .quad_to(Point::new(0.0, h), Point::new(0.0, h * 0.5))
            .close();
        root.children.push(Node::Group(Group {
            transform: Transform2D::translate(px, py)
                .compose(&Transform2D::rotate(0.07 * i as f32)),
            cache_key: Some(0xbead_0000 + i as u64),
            opacity: 0.85,
            children: vec![Node::Path(PathNode {
                path: panel,
                fill: Some(Paint::Solid(Rgba::new(40 + 20 * i as u8, 120, 200, 210))),
                stroke: None,
                fill_rule: FillRule::NonZero,
            })],
            ..Group::default()
        }));
    }
    let frame = VectorFrame {
        width: W as f32,
        height: H as f32,
        view_box: None,
        root,
        pts: None,
        time_base: TimeBase::new(1, 1),
    };
    let roomy = Renderer::new(W, H);
    let reference = pixels(&roomy, &frame);
    let all = roomy.cache_stats().entries;
    let small = Renderer::new(W, H).with_cache_bytes(256 << 10);
    assert!(pixels(&small, &frame) == reference, "256 KiB budget");
    let kept = small.cache_stats().entries;
    assert!(kept > 0 && kept < all, "{kept} of {all} bitmaps kept");
    assert!(
        pixels(&Renderer::new(W, H).with_cache_bytes(0), &frame) == reference,
        "no cache"
    );
    let threaded = Renderer::new(W, H)
        .with_cache_bytes(0)
        .with_execution_context(&oxideav_core::ExecutionContext::with_threads(4));
    assert!(
        pixels(&threaded, &frame) == reference,
        "no cache, 4 workers"
    );
}

/// A page of 60 distinct keyed glyph groups whose keys start at `base`,
/// spread over the whole canvas height.
fn keyed_set(base: u64) -> VectorFrame {
    let mut root = Group::default();
    for i in 0..60u32 {
        root.children.push(Node::Group(Group {
            transform: Transform2D {
                a: 0.02,
                b: 0.0,
                c: 0.0,
                d: -0.02,
                e: 30.0 + (i % 10) as f32 * 180.0,
                f: 60.0 + (i / 10) as f32 * 190.0,
            },
            cache_key: Some(base + i as u64),
            children: vec![Node::Path(PathNode {
                path: glyph_outline(i),
                fill: Some(Paint::Solid(Rgba::opaque(20, 20, 24))),
                stroke: None,
                fill_rule: FillRule::NonZero,
            })],
            ..Group::default()
        }));
    }
    VectorFrame {
        width: W as f32,
        height: H as f32,
        view_box: None,
        root,
        pts: None,
        time_base: TimeBase::new(1, 1),
    }
}

#[test]
fn a_thread_budget_keeps_the_statistics_and_the_recency_order() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let threaded = |cache: &GroupCache| {
        Renderer::new(W, H)
            .with_group_cache(cache.clone())
            .with_execution_context(&oxideav_core::ExecutionContext::with_threads(4))
    };
    // A warm page under four scene bands: the same picture, no miss,
    // and every instance still counted as a hit.
    let page = text_page(GLYPHS, true, (0.0, 0.0));
    let cache = GroupCache::new();
    let serial = Renderer::new(W, H).with_group_cache(cache.clone());
    let reference = pixels(&serial, &page);
    cache.reset_stats();
    assert!(pixels(&threaded(&cache), &page) == reference);
    let st = cache.stats();
    assert_eq!(st.misses, 0);
    assert!(st.hits >= GLYPHS as u64, "{} hits", st.hits);
    // A cold page under four bands, too.
    let cold = GroupCache::new();
    assert!(pixels(&threaded(&cold), &page) == reference);
    assert_eq!(cold.stats().entries, cache.stats().entries);

    // Recency: sets A then B fill a 120-entry cache; A is then used by
    // a threaded frame, so a third set must push out B, not A.
    let lru = GroupCache::new().with_max_entries(120);
    let (a, b, c) = (keyed_set(1000), keyed_set(2000), keyed_set(3000));
    let plain = Renderer::new(W, H).with_group_cache(lru.clone());
    pixels(&plain, &a);
    pixels(&plain, &b);
    assert_eq!(lru.stats().entries, 120);
    pixels(&threaded(&lru), &a);
    pixels(&plain, &c);
    assert_eq!(lru.stats().entries, 120);
    lru.reset_stats();
    pixels(&plain, &a);
    assert_eq!(
        lru.stats().misses,
        0,
        "the set used by the threaded frame was evicted"
    );
    lru.reset_stats();
    pixels(&plain, &b);
    assert_eq!(
        lru.stats().misses,
        60,
        "the unused set should have been evicted"
    );
}
