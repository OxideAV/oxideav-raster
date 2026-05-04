//! Bitmap cache hits / misses for `Group::cache_key`.
//!
//! Renders the same cached subtree twice and asserts the second call
//! reuses the cached bitmap (cache hit), while a different `cache_key`
//! or different effective transform produces a fresh miss.

use oxideav_core::{
    FillRule, Group, Node, Paint, Path, PathNode, Point, Rgba, Transform2D, VectorFrame,
};
use oxideav_raster::Renderer;

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

fn red_rect_node(x: f32, y: f32, w: f32, h: f32) -> Node {
    let mut p = Path::new();
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
    Node::Path(PathNode {
        path: p,
        fill: Some(Paint::Solid(Rgba::opaque(255, 0, 0))),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
}

fn cached_glyph_group(cache_key: u64) -> Group {
    Group {
        cache_key: Some(cache_key),
        children: vec![red_rect_node(2.0, 2.0, 4.0, 4.0)],
        ..Group::default()
    }
}

#[test]
fn second_render_of_same_cache_key_hits_cache() {
    let r = Renderer::new(8, 8);
    // First render — cold cache, expect miss.
    let mut root1 = Group::default();
    root1.children.push(Node::Group(cached_glyph_group(0x1234)));
    let _ = r.render(&frame(8, 8, root1));
    let s1 = r.cache_stats();
    assert_eq!(s1.misses, 1);
    assert_eq!(s1.hits, 0);
    assert_eq!(s1.entries, 1);

    // Second render of the same scene — hot cache, expect hit.
    let mut root2 = Group::default();
    root2.children.push(Node::Group(cached_glyph_group(0x1234)));
    let _ = r.render(&frame(8, 8, root2));
    let s2 = r.cache_stats();
    assert_eq!(s2.misses, 1);
    assert_eq!(s2.hits, 1);
    assert_eq!(s2.entries, 1);
}

#[test]
fn different_cache_keys_produce_distinct_entries() {
    let r = Renderer::new(8, 8);
    let mut root = Group::default();
    root.children.push(Node::Group(cached_glyph_group(0xAAAA)));
    root.children.push(Node::Group(cached_glyph_group(0xBBBB)));
    let _ = r.render(&frame(8, 8, root));
    let s = r.cache_stats();
    // Two misses (one per fresh key), no hits, two entries.
    assert_eq!(s.misses, 2);
    assert_eq!(s.hits, 0);
    assert_eq!(s.entries, 2);
}

#[test]
fn different_transform_invalidates_cache() {
    let r = Renderer::new(16, 16);
    // First call: glyph at identity transform.
    let mut root1 = Group::default();
    root1.children.push(Node::Group(cached_glyph_group(0x4444)));
    let _ = r.render(&frame(16, 16, root1));
    // Second call: same cache_key, but child group is wrapped in an
    // outer group with a non-identity transform, so the effective
    // local transform differs → cache miss.
    let outer = Group {
        transform: Transform2D::scale(2.0, 2.0),
        children: vec![Node::Group(cached_glyph_group(0x4444))],
        ..Group::default()
    };
    let mut root2 = Group::default();
    root2.children.push(Node::Group(outer));
    let _ = r.render(&frame(16, 16, root2));
    let s = r.cache_stats();
    // Two misses (different transforms → different composite keys).
    assert_eq!(s.misses, 2);
    assert_eq!(s.hits, 0);
}

#[test]
fn cache_key_none_bypasses_cache() {
    let r = Renderer::new(8, 8);
    let mut root = Group::default();
    // Group with cache_key = None — should NOT be cached.
    let inner = Group {
        cache_key: None,
        children: vec![red_rect_node(0.0, 0.0, 4.0, 4.0)],
        ..Group::default()
    };
    root.children.push(Node::Group(inner));
    let _ = r.render(&frame(8, 8, root));
    let s = r.cache_stats();
    assert_eq!(s.entries, 0);
    assert_eq!(s.hits, 0);
    assert_eq!(s.misses, 0);
}

#[test]
fn cache_capacity_evicts_oldest_entry() {
    let r = Renderer::with_cache_capacity(8, 8, 2);
    // Three distinct cached groups → first one evicted on the third
    // insert. Then re-render the first → miss again.
    for k in [0xAAA, 0xBBB, 0xCCC] {
        let mut root = Group::default();
        root.children.push(Node::Group(cached_glyph_group(k)));
        let _ = r.render(&frame(8, 8, root));
    }
    let s = r.cache_stats();
    assert_eq!(s.entries, 2);
    assert_eq!(s.misses, 3);

    // Re-render 0xAAA — should miss again (evicted).
    let mut root = Group::default();
    root.children.push(Node::Group(cached_glyph_group(0xAAA)));
    let _ = r.render(&frame(8, 8, root));
    let s = r.cache_stats();
    assert_eq!(s.misses, 4);
    assert_eq!(s.hits, 0);
}

#[test]
fn cached_subtree_stores_bbox_crop_not_full_canvas() {
    // Render a 4×4 cached glyph at the upper-left of a 1024×1024 canvas
    // and verify the cached bitmap is the bbox crop, not the full
    // canvas — saves significant memory for tiny glyphs.
    let r = Renderer::new(1024, 1024);
    let mut root = Group::default();
    // Tiny rect: 4×4 at (10, 10).
    let inner = Group {
        cache_key: Some(0xC0DE_CAFE),
        children: vec![red_rect_node(10.0, 10.0, 4.0, 4.0)],
        ..Group::default()
    };
    root.children.push(Node::Group(inner));
    let _ = r.render(&frame(1024, 1024, root));

    // The cache must report exactly one entry.
    let s = r.cache_stats();
    assert_eq!(s.entries, 1);

    // We can't read the entry directly through the public API, but we
    // can re-render: the second pass hits the cache and must produce
    // pixel-identical output to a fresh renderer that did NOT cache
    // anything. The cache's correctness is the load-bearing property.
    let mut root2 = Group::default();
    root2.children.push(Node::Group(Group {
        cache_key: Some(0xC0DE_CAFE),
        children: vec![red_rect_node(10.0, 10.0, 4.0, 4.0)],
        ..Group::default()
    }));
    let cached_out = r.render(&frame(1024, 1024, root2));

    let r_uncached = Renderer::new(1024, 1024);
    let mut root3 = Group::default();
    root3.children.push(Node::Group(Group {
        cache_key: None,
        children: vec![red_rect_node(10.0, 10.0, 4.0, 4.0)],
        ..Group::default()
    }));
    let uncached_out = r_uncached.render(&frame(1024, 1024, root3));

    assert_eq!(cached_out.planes[0].data, uncached_out.planes[0].data);
}

#[test]
fn cached_render_produces_pixel_identical_output() {
    // Cached path + uncached path of the same scene must produce the
    // same pixels (the cache must not change the rendered output).
    let r_cached = Renderer::new(8, 8);
    let r_uncached = Renderer::new(8, 8);

    let mut cached_root = Group::default();
    cached_root
        .children
        .push(Node::Group(cached_glyph_group(0xDEAD)));

    let mut uncached_root = Group::default();
    uncached_root.children.push(Node::Group(Group {
        cache_key: None,
        children: vec![red_rect_node(2.0, 2.0, 4.0, 4.0)],
        ..Group::default()
    }));

    // Render twice through the cached renderer (first miss, then hit)
    // and once through the uncached renderer.
    let _ = r_cached.render(&frame(8, 8, cached_root.clone()));
    let cached_out = r_cached.render(&frame(8, 8, cached_root));
    let uncached_out = r_uncached.render(&frame(8, 8, uncached_root));

    let s = r_cached.cache_stats();
    assert!(s.hits >= 1, "second render should hit cache");
    assert_eq!(cached_out.planes[0].data, uncached_out.planes[0].data);
}
