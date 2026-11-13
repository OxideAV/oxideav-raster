//! Bitmap cache for memoised group-subtree rasterisation.
//!
//! When a [`Group`](oxideav_core::Group) carries a `cache_key`, the
//! [`Renderer`](crate::Renderer) rasterises the group once into a
//! bitmap just large enough for the group's own bounds and afterwards
//! blits that bitmap wherever the group is drawn again.
//!
//! # What a key matches
//!
//! A cached bitmap is reused when **all** of these agree, compared
//! field by field (there is no hashed composite that two different
//! inputs could collide on):
//!
//! * the producer's `Group::cache_key`;
//! * the linear part of the effective transform (`a b c d`, bit for
//!   bit) — scale, rotation and shear;
//! * the **sub-pixel phase** of the translation, i.e. the fractional
//!   part of `(e, f)` snapped to the renderer's sub-pixel grid
//!   ([`Renderer::cache_subpixel_grid`](crate::Renderer::cache_subpixel_grid),
//!   ¼ px by default);
//! * every renderer setting that changes the pixels of a subtree
//!   (coverage kernel, supersampling factor, image filter, gradient
//!   interpolation space and dither, blend mode, compositing space).
//!
//! The whole-pixel part of the translation is **not** part of the key:
//! the bitmap is blitted at that integer offset, so one glyph bitmap
//! serves every pen position that shares its sub-pixel phase.
//!
//! # Size
//!
//! The cache is bounded by the **bytes** of pixel data it holds
//! ([`DEFAULT_CACHE_BYTES`] unless configured), evicting the least
//! recently used entries first. A single bitmap larger than a quarter
//! of the budget is never cached — the group is then drawn directly,
//! exactly as if it had no key.
//!
//! # Sharing
//!
//! [`GroupCache`] is a cheap, clonable handle. A long-lived renderer
//! keeps its cache across `render` calls by itself; a renderer built
//! per frame is handed the previous frame's cache with
//! [`Renderer::with_group_cache`](crate::Renderer::with_group_cache) so
//! it does not start cold.

use std::collections::HashMap;
use std::mem::Discriminant;
use std::sync::{Arc, Mutex, MutexGuard};

use oxideav_core::Transform2D;

use crate::blend::BlendMode;
use crate::fill::CoverageMode;
use crate::gradient::InterpolationSpace;
use crate::gradient_row::GradientDither;
use crate::renderer::ImageFilter;

/// Default budget of a [`GroupCache`]: 32 MiB of cached pixel data
/// (about 20 000 glyph bitmaps at a 16 px em).
pub const DEFAULT_CACHE_BYTES: usize = 32 << 20;

/// Bookkeeping bytes charged per entry on top of its pixel data, so a
/// flood of empty or one-pixel bitmaps is bounded too.
const ENTRY_OVERHEAD_BYTES: usize = 128;

/// A rasterised subtree record.
///
/// Retained for API compatibility: the renderer's cache now stores its
/// bitmaps relative to the group's own origin (see the module docs) and
/// no public entry point produces this type.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// Effective transform at the moment the bitmap was rasterised.
    pub transform_at_cache_time: Transform2D,
}

/// One cached group bitmap: straight-alpha RGBA covering the group's
/// bounds, positioned relative to the whole-pixel part of the
/// translation it is drawn with.
#[derive(Debug)]
pub(crate) struct CachedBitmap {
    pub width: u32,
    pub height: u32,
    /// Where the bitmap's top-left pixel sits relative to the integer
    /// pen offset (negative for content left of / above the origin).
    pub dx: i64,
    pub dy: i64,
    /// `width * height * 4` bytes.
    pub rgba: Vec<u8>,
}

impl CachedBitmap {
    fn cost(&self) -> usize {
        self.rgba.len() + ENTRY_OVERHEAD_BYTES
    }
}

/// The renderer settings that change a subtree's pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RenderSettings {
    pub coverage: Discriminant<CoverageMode>,
    pub supersampling: u8,
    pub image_filter: Discriminant<ImageFilter>,
    pub color_interpolation: Discriminant<InterpolationSpace>,
    pub blend_mode: Discriminant<BlendMode>,
    pub gradient_dither: Discriminant<GradientDither>,
    pub compositing_space: Discriminant<InterpolationSpace>,
}

/// Full identity of a cached bitmap. Equality is structural, so two
/// different (key, transform, settings) tuples can never share an
/// entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BitmapKey {
    group_key: u64,
    /// Bit patterns of `a b c d` (with `-0.0` folded onto `0.0`).
    linear: [u32; 4],
    /// Bit patterns of the snapped fractional translation.
    phase: [u32; 2],
    settings: RenderSettings,
}

impl BitmapKey {
    /// `snapped` is the transform the bitmap is rasterised with: the
    /// group's linear part and the snapped sub-pixel phase as its
    /// translation.
    pub fn new(group_key: u64, snapped: &Transform2D, settings: RenderSettings) -> Self {
        // `+ 0.0` maps −0.0 to +0.0 and leaves every other value alone.
        let bits = |v: f32| (v + 0.0).to_bits();
        Self {
            group_key,
            linear: [
                bits(snapped.a),
                bits(snapped.b),
                bits(snapped.c),
                bits(snapped.d),
            ],
            phase: [bits(snapped.e), bits(snapped.f)],
            settings,
        }
    }
}

/// Split a translation component into its whole-pixel part and its
/// fractional part snapped to a `1 / grid` lattice (`grid == 0` keeps
/// the fraction exactly). `None` for values a pixel offset cannot
/// represent.
pub(crate) fn snap_axis(v: f32, grid: u8) -> Option<(i64, f32)> {
    if !(v.is_finite() && v.abs() < 1.0e9) {
        return None;
    }
    let whole = v.floor();
    // Exact: `whole` and `v` are within a factor of two of each other
    // (or `v` is a sub-unit value and `whole` is 0 or −1).
    let frac = v - whole;
    let mut whole = whole as i64;
    if grid == 0 {
        return Some((whole, frac));
    }
    let g = grid as f32;
    let mut steps = (frac * g).round();
    if steps >= g {
        steps = 0.0;
        whole += 1;
    }
    Some((whole, steps / g))
}

/// Hits / misses counter, observable through
/// [`crate::Renderer::cache_stats`] and [`GroupCache::stats`] for
/// testing and tuning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CacheStats {
    /// Number of lookups that returned a cached bitmap.
    pub hits: u64,
    /// Number of lookups that did not find one.
    pub misses: u64,
    /// Current number of entries in the cache.
    pub entries: usize,
    /// Maximum number of entries (`usize::MAX` when only the byte
    /// budget applies, which is the default).
    pub capacity: usize,
    /// Bytes currently charged to the cache (pixel data plus a small
    /// fixed overhead per entry).
    pub bytes: usize,
    /// Byte budget; least recently used entries are evicted to stay
    /// within it.
    pub max_bytes: usize,
    /// Entries evicted so far to make room.
    pub evictions: u64,
}

/// Byte- and entry-bounded LRU over [`BitmapKey`].
///
/// The list head is the most recently used entry; the tail is evicted
/// first. Nodes live in a `Vec` with index links and a free list, and a
/// `HashMap` keyed by the full [`BitmapKey`] finds them in O(1).
#[derive(Debug)]
struct LruCache {
    nodes: Vec<Slot>,
    free: Vec<usize>,
    head: Option<usize>,
    tail: Option<usize>,
    map: HashMap<BitmapKey, usize>,
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

#[derive(Debug)]
struct Slot {
    key: BitmapKey,
    /// `None` only while the slot sits on the free list.
    value: Option<Arc<CachedBitmap>>,
    prev: Option<usize>,
    next: Option<usize>,
}

impl LruCache {
    fn new(max_bytes: usize, max_entries: usize) -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            head: None,
            tail: None,
            map: HashMap::new(),
            max_entries: max_entries.max(1),
            max_bytes,
            bytes: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.map.len(),
            capacity: self.max_entries,
            bytes: self.bytes,
            max_bytes: self.max_bytes,
            evictions: self.evictions,
        }
    }

    fn get(&mut self, key: &BitmapKey) -> Option<Arc<CachedBitmap>> {
        let Some(&idx) = self.map.get(key) else {
            self.misses += 1;
            return None;
        };
        self.hits += 1;
        self.move_to_head(idx);
        self.nodes[idx].value.clone()
    }

    /// Largest bitmap (in bytes of pixel data) worth caching.
    fn max_entry_bytes(&self) -> usize {
        self.max_bytes / 4
    }

    fn put(&mut self, key: BitmapKey, value: Arc<CachedBitmap>) {
        let cost = value.cost();
        if value.rgba.len() > self.max_entry_bytes() || cost > self.max_bytes {
            return;
        }
        if let Some(&idx) = self.map.get(&key) {
            let old = self.nodes[idx].value.replace(value);
            self.bytes = self.bytes - old.map_or(0, |o| o.cost()) + cost;
            self.move_to_head(idx);
        } else {
            let slot = Slot {
                key,
                value: Some(value),
                prev: None,
                next: self.head,
            };
            let idx = match self.free.pop() {
                Some(i) => {
                    self.nodes[i] = slot;
                    i
                }
                None => {
                    self.nodes.push(slot);
                    self.nodes.len() - 1
                }
            };
            if let Some(h) = self.head {
                self.nodes[h].prev = Some(idx);
            }
            self.head = Some(idx);
            if self.tail.is_none() {
                self.tail = Some(idx);
            }
            self.map.insert(key, idx);
            self.bytes += cost;
        }
        self.trim();
    }

    /// Evict from the cold end until both limits hold. The entry just
    /// inserted is the head, and it fits the budget on its own, so it
    /// survives.
    fn trim(&mut self) {
        while (self.bytes > self.max_bytes || self.map.len() > self.max_entries)
            && self.map.len() > 1
        {
            self.evict_lru();
        }
        if self.bytes > self.max_bytes {
            self.evict_lru();
        }
    }

    fn evict_lru(&mut self) {
        let Some(tail_idx) = self.tail else {
            return;
        };
        let prev = self.nodes[tail_idx].prev;
        match prev {
            Some(p) => self.nodes[p].next = None,
            None => self.head = None,
        }
        self.tail = prev;
        let key = self.nodes[tail_idx].key;
        self.map.remove(&key);
        if let Some(v) = self.nodes[tail_idx].value.take() {
            self.bytes -= v.cost();
        }
        self.nodes[tail_idx].prev = None;
        self.nodes[tail_idx].next = None;
        self.free.push(tail_idx);
        self.evictions += 1;
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.free.clear();
        self.map.clear();
        self.head = None;
        self.tail = None;
        self.bytes = 0;
    }

    fn move_to_head(&mut self, idx: usize) {
        if Some(idx) == self.head {
            return;
        }
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
}

/// A shareable cache of rasterised `Group::cache_key` subtrees.
///
/// Cloning the handle shares the cache (it is reference counted and
/// internally locked, so clones may live on different threads). Every
/// [`Renderer`](crate::Renderer) owns one; hand the same handle to
/// several renderers — or to the renderer built for the next frame —
/// with [`Renderer::with_group_cache`](crate::Renderer::with_group_cache):
///
/// ```
/// use oxideav_raster::{GroupCache, Renderer};
///
/// let cache = GroupCache::with_max_bytes(8 << 20);
/// for _frame in 0..3 {
///     // A renderer per frame, one cache for all of them.
///     let renderer = Renderer::new(640, 480).with_group_cache(cache.clone());
///     # let _ = renderer;
/// }
/// assert_eq!(cache.stats().max_bytes, 8 << 20);
/// ```
///
/// Renderers with different settings or canvas sizes can share a cache
/// safely: the settings that affect pixels are part of every entry's
/// identity, and bitmaps do not depend on the canvas they were first
/// drawn on.
#[derive(Debug, Clone)]
pub struct GroupCache {
    inner: Arc<Mutex<LruCache>>,
}

impl Default for GroupCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupCache {
    /// A cache with the default byte budget ([`DEFAULT_CACHE_BYTES`])
    /// and no limit on the number of entries.
    pub fn new() -> Self {
        Self::with_max_bytes(DEFAULT_CACHE_BYTES)
    }

    /// A cache holding at most `max_bytes` of pixel data (plus a small
    /// fixed overhead per entry). `0` disables caching: every keyed
    /// group is drawn directly.
    pub fn with_max_bytes(max_bytes: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LruCache::new(max_bytes, usize::MAX))),
        }
    }

    /// Additionally bound the number of entries (at least 1). Returns
    /// the same handle for chaining.
    pub fn with_max_entries(self, max_entries: usize) -> Self {
        {
            let mut g = self.lock();
            g.max_entries = max_entries.max(1);
            g.trim();
        }
        self
    }

    /// Change the byte budget, evicting least recently used entries if
    /// the cache is now over it.
    pub fn set_max_bytes(&self, max_bytes: usize) {
        let mut g = self.lock();
        g.max_bytes = max_bytes;
        g.trim();
    }

    /// Hit / miss / occupancy snapshot.
    pub fn stats(&self) -> CacheStats {
        self.lock().stats()
    }

    /// Zero the hit / miss / eviction counters, keeping the entries.
    pub fn reset_stats(&self) {
        let mut g = self.lock();
        g.hits = 0;
        g.misses = 0;
        g.evictions = 0;
    }

    /// Drop every cached bitmap (the counters are kept).
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// `true` when `other` is a handle to this same cache.
    pub fn shares_with(&self, other: &GroupCache) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// A panic while the lock is held cannot leave the LRU links half
    /// updated in a way a later reader would trip on (every mutation is
    /// index bookkeeping over owned data), so a poisoned lock is simply
    /// taken over.
    fn lock(&self) -> MutexGuard<'_, LruCache> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn get(&self, key: &BitmapKey) -> Option<Arc<CachedBitmap>> {
        self.lock().get(key)
    }

    pub(crate) fn put(&self, key: BitmapKey, value: Arc<CachedBitmap>) {
        self.lock().put(key, value);
    }

    pub(crate) fn max_entry_bytes(&self) -> usize {
        self.lock().max_entry_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> RenderSettings {
        RenderSettings {
            coverage: std::mem::discriminant(&CoverageMode::Analytic),
            supersampling: 4,
            image_filter: std::mem::discriminant(&ImageFilter::Bilinear),
            color_interpolation: std::mem::discriminant(&InterpolationSpace::Srgb),
            blend_mode: std::mem::discriminant(&BlendMode::Normal),
            gradient_dither: std::mem::discriminant(&GradientDither::None),
            compositing_space: std::mem::discriminant(&InterpolationSpace::Srgb),
        }
    }

    fn key(k: u64) -> BitmapKey {
        BitmapKey::new(k, &Transform2D::identity(), settings())
    }

    fn bitmap(width: u32, height: u32) -> Arc<CachedBitmap> {
        Arc::new(CachedBitmap {
            width,
            height,
            dx: 0,
            dy: 0,
            rgba: vec![0; (width as usize) * (height as usize) * 4],
        })
    }

    #[test]
    fn put_and_get_basic() {
        let mut c = LruCache::new(1 << 20, 2);
        c.put(key(1), bitmap(2, 2));
        let got = c.get(&key(1)).expect("cached value present");
        assert_eq!(got.width, 2);
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 0);
        assert_eq!(c.stats().bytes, 16 + ENTRY_OVERHEAD_BYTES);
    }

    #[test]
    fn get_misses_when_absent() {
        let mut c = LruCache::new(1 << 20, 2);
        assert!(c.get(&key(99)).is_none());
        assert_eq!(c.stats().misses, 1);
    }

    #[test]
    fn entry_limit_evicts_lru_entry() {
        let mut c = LruCache::new(1 << 20, 2);
        c.put(key(1), bitmap(1, 1));
        c.put(key(2), bitmap(1, 1));
        c.put(key(3), bitmap(1, 1)); // evicts 1
        assert!(c.get(&key(1)).is_none());
        assert!(c.get(&key(2)).is_some());
        assert!(c.get(&key(3)).is_some());
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn get_promotes_to_mru_so_it_survives_eviction() {
        let mut c = LruCache::new(1 << 20, 2);
        c.put(key(1), bitmap(1, 1));
        c.put(key(2), bitmap(1, 1));
        let _ = c.get(&key(1));
        c.put(key(3), bitmap(1, 1)); // evicts 2
        assert!(c.get(&key(1)).is_some());
        assert!(c.get(&key(2)).is_none());
    }

    #[test]
    fn re_insert_overwrites_and_promotes() {
        let mut c = LruCache::new(1 << 20, 2);
        c.put(key(1), bitmap(1, 1));
        c.put(key(2), bitmap(2, 2));
        c.put(key(1), bitmap(3, 3));
        assert_eq!(c.get(&key(1)).unwrap().width, 3);
        assert_eq!(c.stats().bytes, 36 + 16 + 2 * ENTRY_OVERHEAD_BYTES);
    }

    #[test]
    fn byte_budget_evicts_cold_entries_and_tracks_bytes() {
        // Each 16×16 bitmap costs 1024 + overhead; four fit in 5000.
        let per = 1024 + ENTRY_OVERHEAD_BYTES;
        let mut c = LruCache::new(4 * per + 10, usize::MAX);
        for k in 0..4 {
            c.put(key(k), bitmap(16, 16));
        }
        assert_eq!(c.stats().entries, 4);
        assert_eq!(c.stats().bytes, 4 * per);
        let _ = c.get(&key(0)); // 1 is now coldest
        c.put(key(4), bitmap(16, 16));
        assert_eq!(c.stats().entries, 4);
        assert!(c.get(&key(1)).is_none());
        assert!(c.get(&key(0)).is_some());
        assert!(c.stats().bytes <= c.stats().max_bytes);
        // Shrinking the budget evicts down to it.
        c.max_bytes = per;
        c.trim();
        assert_eq!(c.stats().entries, 1);
        assert_eq!(c.stats().bytes, per);
        c.clear();
        assert_eq!(c.stats().bytes, 0);
        assert_eq!(c.stats().entries, 0);
        c.max_bytes = 8 * per;
        c.put(key(7), bitmap(16, 16));
        assert_eq!(c.stats().entries, 1);
    }

    #[test]
    fn oversized_bitmaps_are_not_cached() {
        let mut c = LruCache::new(4096, usize::MAX);
        c.put(key(1), bitmap(32, 32)); // 4096 bytes > budget / 4
        assert_eq!(c.stats().entries, 0);
        c.put(key(2), bitmap(8, 8));
        assert_eq!(c.stats().entries, 1);
        let mut zero = LruCache::new(0, usize::MAX);
        zero.put(key(1), bitmap(0, 0));
        assert_eq!(zero.stats().entries, 0);
    }

    #[test]
    fn keys_are_structural() {
        let s = settings();
        let id = Transform2D::identity();
        let sc = Transform2D::scale(2.0, 2.0);
        assert_ne!(BitmapKey::new(1, &id, s), BitmapKey::new(1, &sc, s));
        assert_ne!(BitmapKey::new(1, &id, s), BitmapKey::new(2, &id, s));
        assert_eq!(BitmapKey::new(42, &sc, s), BitmapKey::new(42, &sc, s));
        // −0.0 and +0.0 draw the same pixels.
        let neg = Transform2D {
            b: -0.0,
            c: -0.0,
            ..id
        };
        assert_eq!(BitmapKey::new(1, &id, s), BitmapKey::new(1, &neg, s));
        // A setting that changes pixels separates entries.
        let other = RenderSettings {
            supersampling: 8,
            ..s
        };
        assert_ne!(BitmapKey::new(1, &id, s), BitmapKey::new(1, &id, other));
        let phase = Transform2D { e: 0.25, ..id };
        assert_ne!(BitmapKey::new(1, &id, s), BitmapKey::new(1, &phase, s));
    }

    #[test]
    fn snap_axis_splits_whole_and_phase() {
        assert_eq!(snap_axis(10.0, 4), Some((10, 0.0)));
        assert_eq!(snap_axis(10.25, 4), Some((10, 0.25)));
        assert_eq!(snap_axis(10.3, 4), Some((10, 0.25)));
        assert_eq!(snap_axis(10.4, 4), Some((10, 0.5)));
        assert_eq!(snap_axis(10.9, 4), Some((11, 0.0)));
        assert_eq!(snap_axis(-0.1, 4), Some((0, 0.0)));
        assert_eq!(snap_axis(-0.3, 4), Some((-1, 0.75)));
        assert_eq!(snap_axis(-3.0, 4), Some((-3, 0.0)));
        assert_eq!(snap_axis(7.6, 1), Some((8, 0.0)));
        assert_eq!(snap_axis(7.4, 1), Some((7, 0.0)));
        assert_eq!(snap_axis(5.375, 0), Some((5, 0.375)));
        assert_eq!(snap_axis(f32::NAN, 4), None);
        assert_eq!(snap_axis(f32::INFINITY, 4), None);
        assert_eq!(snap_axis(3.0e12, 4), None);
        // The snapped position is never further than half a grid step.
        for i in -4000..4000 {
            let v = i as f32 * 0.013_7;
            for grid in [1u8, 2, 4, 8, 16] {
                let (w, f) = snap_axis(v, grid).unwrap();
                assert!((0.0..1.0).contains(&f));
                assert!((w as f32 + f - v).abs() <= 0.5 / grid as f32 + 1e-4);
            }
        }
    }

    #[test]
    fn shared_cache_is_thread_safe() {
        let s = GroupCache::new();
        s.put(key(1), bitmap(2, 2));
        let s2 = s.clone();
        assert!(s.shares_with(&s2));
        assert!(!s.shares_with(&GroupCache::new()));
        let h = std::thread::spawn(move || s2.get(&key(1)).is_some());
        assert!(h.join().unwrap());
    }
}
