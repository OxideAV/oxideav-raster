//! Bitmap cache for memoised group-subtree rasterisation.
//!
//! When a [`Group`](oxideav_core::Group) carries a `cache_key`, the
//! [`Renderer`](crate::Renderer) memoises the rasterised output of its
//! children under a composite key derived from the user-supplied
//! `cache_key` and the *effective* transform at render time, so the
//! same glyph rendered twice — at the same position and scale — only
//! gets rasterised once.
//!
//! The cache is a plain capacity-bounded LRU (doubly-linked list +
//! hash map) implemented in-tree to avoid pulling in `lru` /
//! `parking_lot` dependencies. Default capacity is 256 entries.
//!
//! Producers that don't want caching simply leave `cache_key = None`
//! (the default); the cache is bypassed in that case.
//!
//! # Cache key
//!
//! The composite key is `mix64(group_cache_key, transform_signature)`:
//!
//! * `group_cache_key` is the producer-supplied `Group::cache_key`
//!   (e.g. a deterministic hash of `(face_id, glyph_id, size_q8,
//!   subpixel_x)` for scribe-shaped glyphs);
//! * `transform_signature` is a 64-bit hash of the effective
//!   `Transform2D` at render time, computed from `f32::to_bits()` of
//!   each of the 6 matrix entries.
//!
//! `mix64` is a single round of the SplitMix64 finaliser — fast,
//! avalanching, no dependencies, no false positives between adjacent
//! key tuples.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use oxideav_core::Transform2D;

/// One cached rasterised subtree. Captures the rasterised output of a
/// group's children at a specific `transform_at_cache_time` so a future
/// hit can blit the same pixels straight into the destination buffer.
///
/// Round 3 stores only the *touched-pixel bounding box* of the subtree,
/// not the full canvas — a 16 px glyph in a 4096 px canvas now consumes
/// ~1 KB instead of 64 MB. `(offset_x, offset_y)` is the destination
/// position of the bitmap's top-left corner; `width × height` are the
/// crop's dimensions in pixels. `rgba` is `width * height * 4` bytes of
/// packed straight-alpha RGBA.
///
/// A subtree that paints nothing is stored with `width = height = 0`;
/// blits of empty subtrees are no-ops.
#[derive(Debug, Clone)]
pub struct RasterizedSubtree {
    /// Bitmap width in pixels (post-crop).
    pub width: u32,
    /// Bitmap height in pixels (post-crop).
    pub height: u32,
    /// X offset of the bitmap's top-left corner in destination pixels.
    pub offset_x: u32,
    /// Y offset of the bitmap's top-left corner in destination pixels.
    pub offset_y: u32,
    /// Packed straight-alpha RGBA, `width * height * 4` bytes.
    pub rgba: Vec<u8>,
    /// Effective transform at the moment the bitmap was rasterised —
    /// kept so debug tooling can confirm the cache key really matches
    /// the geometry that produced the bitmap.
    pub transform_at_cache_time: Transform2D,
}

/// Hits / misses counter, observable through
/// [`crate::Renderer::cache_stats`] for testing and tuning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of `get` calls that returned a cached entry.
    pub hits: u64,
    /// Number of `get` calls that did not find an entry (and the
    /// associated put-on-miss).
    pub misses: u64,
    /// Current number of entries in the cache.
    pub entries: usize,
    /// Maximum number of entries the cache will hold before evicting
    /// the LRU.
    pub capacity: usize,
}

/// Capacity-bounded LRU mapping `u64 → RasterizedSubtree`.
///
/// The list head is the *most* recently used; the tail is evicted on
/// insert when the cache is full. We use a small `VecDeque`-style
/// doubly-linked list backed by a `Vec<Node>` with `Option<usize>`
/// prev/next pointers — a single `HashMap<u64, usize>` indexes into
/// it in O(1). No external crates.
#[derive(Debug)]
pub(crate) struct LruCache {
    nodes: Vec<Slot>,
    free: Vec<usize>,
    head: Option<usize>,
    tail: Option<usize>,
    map: HashMap<u64, usize>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

#[derive(Debug)]
struct Slot {
    key: u64,
    value: RasterizedSubtree,
    prev: Option<usize>,
    next: Option<usize>,
}

impl LruCache {
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            nodes: Vec::with_capacity(cap),
            free: Vec::new(),
            head: None,
            tail: None,
            map: HashMap::with_capacity(cap),
            capacity: cap,
            hits: 0,
            misses: 0,
        }
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.map.len(),
            capacity: self.capacity,
        }
    }

    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    /// Look up `key`, promoting it to MRU on hit. Returns a clone of
    /// the cached value (so the caller can release the lock before the
    /// blit — and so the cached entry can't be invalidated mid-blit).
    pub fn get(&mut self, key: u64) -> Option<RasterizedSubtree> {
        let idx = match self.map.get(&key) {
            Some(&i) => i,
            None => {
                self.misses += 1;
                return None;
            }
        };
        self.hits += 1;
        self.move_to_head(idx);
        Some(self.nodes[idx].value.clone())
    }

    /// Insert `value` under `key`. If the key already exists its value
    /// is overwritten and it's promoted to MRU; if the cache is full
    /// the LRU entry is evicted first.
    pub fn put(&mut self, key: u64, value: RasterizedSubtree) {
        if let Some(&idx) = self.map.get(&key) {
            self.nodes[idx].value = value;
            self.move_to_head(idx);
            return;
        }
        if self.map.len() >= self.capacity {
            self.evict_lru();
        }
        let idx = self.alloc_slot(Slot {
            key,
            value,
            prev: None,
            next: self.head,
        });
        if let Some(h) = self.head {
            self.nodes[h].prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
        self.map.insert(key, idx);
    }

    fn evict_lru(&mut self) {
        let tail_idx = match self.tail {
            Some(t) => t,
            None => return,
        };
        let prev = self.nodes[tail_idx].prev;
        let key = self.nodes[tail_idx].key;
        if let Some(p) = prev {
            self.nodes[p].next = None;
        } else {
            self.head = None;
        }
        self.tail = prev;
        self.map.remove(&key);
        self.free_slot(tail_idx);
    }

    fn move_to_head(&mut self, idx: usize) {
        if Some(idx) == self.head {
            return;
        }
        // Detach.
        let prev = self.nodes[idx].prev;
        let next = self.nodes[idx].next;
        if let Some(p) = prev {
            self.nodes[p].next = next;
        }
        if let Some(n) = next {
            self.nodes[n].prev = prev;
        }
        if Some(idx) == self.tail {
            self.tail = prev;
        }
        // Re-insert at head.
        self.nodes[idx].prev = None;
        self.nodes[idx].next = self.head;
        if let Some(h) = self.head {
            self.nodes[h].prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
    }

    fn alloc_slot(&mut self, slot: Slot) -> usize {
        if let Some(idx) = self.free.pop() {
            self.nodes[idx] = slot;
            idx
        } else {
            self.nodes.push(slot);
            self.nodes.len() - 1
        }
    }

    fn free_slot(&mut self, idx: usize) {
        // Leave a placeholder; we'll overwrite it on next alloc.
        // Avoid re-shrinking the vec to keep indices stable.
        self.nodes[idx].prev = None;
        self.nodes[idx].next = None;
        self.free.push(idx);
    }
}

/// Shared cache handle. Cheap to clone — wraps an `Arc<Mutex<LruCache>>`.
#[derive(Debug, Clone)]
pub(crate) struct SharedCache {
    inner: Arc<Mutex<LruCache>>,
}

impl SharedCache {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LruCache::with_capacity(capacity))),
        }
    }

    pub fn get(&self, key: u64) -> Option<RasterizedSubtree> {
        self.inner.lock().ok().and_then(|mut g| g.get(key))
    }

    pub fn put(&self, key: u64, value: RasterizedSubtree) {
        if let Ok(mut g) = self.inner.lock() {
            g.put(key, value);
        }
    }

    pub fn stats(&self) -> CacheStats {
        self.inner.lock().map(|g| g.stats()).unwrap_or_default()
    }

    pub fn reset_stats(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.reset_stats();
        }
    }
}

/// Build the composite cache key from a producer-supplied `cache_key`
/// and the effective transform.
///
/// The transform signature is built from the bit pattern of each of
/// the 6 affine entries; `mix64` is a single SplitMix64 finalisation
/// round which gives strong bit diffusion in 5 multiplications.
pub(crate) fn composite_key(group_key: u64, t: &Transform2D) -> u64 {
    let sig = transform_signature(t);
    mix64(
        group_key
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(sig),
    )
}

/// 64-bit hash of an affine transform. Two transforms that produce
/// different output (different scale, rotation, or translation)
/// hash to different values with overwhelming probability.
pub(crate) fn transform_signature(t: &Transform2D) -> u64 {
    // Quantise to canonicalise tiny floating-point noise (a transform
    // assembled in two different orders may differ in the last bit
    // after compose() but should hit the same cache slot).
    fn q(v: f32) -> u64 {
        // 1e-4 user-space units rounding — well below 1 pixel at any
        // realistic scale, but coarse enough to absorb composition
        // drift.
        let scaled = (v * 1e4) as i64;
        scaled as u64
    }
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for v in [t.a, t.b, t.c, t.d, t.e, t.f] {
        h = mix64(h ^ q(v).wrapping_mul(0x100000001B3));
    }
    h
}

/// SplitMix64 finaliser. Single round; fast and well-mixed.
#[inline]
pub(crate) fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(width: u32, height: u32) -> RasterizedSubtree {
        RasterizedSubtree {
            width,
            height,
            offset_x: 0,
            offset_y: 0,
            rgba: vec![0; (width as usize) * (height as usize) * 4],
            transform_at_cache_time: Transform2D::identity(),
        }
    }

    #[test]
    fn put_and_get_basic() {
        let mut c = LruCache::with_capacity(2);
        c.put(1, dummy(2, 2));
        let got = c.get(1).expect("cached value present");
        assert_eq!(got.width, 2);
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 0);
    }

    #[test]
    fn get_misses_when_absent() {
        let mut c = LruCache::with_capacity(2);
        assert!(c.get(99).is_none());
        assert_eq!(c.stats().misses, 1);
    }

    #[test]
    fn capacity_evicts_lru_entry() {
        let mut c = LruCache::with_capacity(2);
        c.put(1, dummy(1, 1));
        c.put(2, dummy(1, 1));
        c.put(3, dummy(1, 1)); // evicts 1
        assert!(c.get(1).is_none());
        assert!(c.get(2).is_some());
        assert!(c.get(3).is_some());
    }

    #[test]
    fn get_promotes_to_mru_so_it_survives_eviction() {
        let mut c = LruCache::with_capacity(2);
        c.put(1, dummy(1, 1));
        c.put(2, dummy(1, 1));
        // Touch 1 → it becomes MRU; 2 is LRU now.
        let _ = c.get(1);
        c.put(3, dummy(1, 1)); // evicts 2
        assert!(c.get(1).is_some());
        assert!(c.get(2).is_none());
    }

    #[test]
    fn re_insert_overwrites_and_promotes() {
        let mut c = LruCache::with_capacity(2);
        c.put(1, dummy(1, 1));
        c.put(2, dummy(2, 2));
        c.put(1, dummy(3, 3)); // overwrite 1
        let got = c.get(1).unwrap();
        assert_eq!(got.width, 3);
    }

    #[test]
    fn composite_key_distinguishes_transforms() {
        let id = Transform2D::identity();
        let sc = Transform2D::scale(2.0, 2.0);
        assert_ne!(composite_key(0xABCD, &id), composite_key(0xABCD, &sc));
    }

    #[test]
    fn composite_key_distinguishes_group_keys() {
        let id = Transform2D::identity();
        assert_ne!(composite_key(1, &id), composite_key(2, &id));
    }

    #[test]
    fn composite_key_stable_for_equal_inputs() {
        let t = Transform2D::translate(3.0, -1.0);
        assert_eq!(composite_key(42, &t), composite_key(42, &t));
    }

    #[test]
    fn shared_cache_is_thread_safe() {
        let s = SharedCache::with_capacity(4);
        s.put(1, dummy(2, 2));
        let s2 = s.clone();
        let h = std::thread::spawn(move || s2.get(1).is_some());
        assert!(h.join().unwrap());
    }
}
