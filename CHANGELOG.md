# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Bitmap cache for memoised group subtrees. When a
  [`Group`](oxideav_core::Group) carries `cache_key: Some(k)`, the
  rendered children are stored under `mix64(k, transform_signature)`
  so a re-render at the same effective transform returns the cached
  bitmap. Capacity-bounded LRU (default 256 entries; configurable via
  `Renderer::with_cache_capacity`); thread-safe behind `Arc<Mutex<…>>`
  so the cache survives `Renderer::clone()`. Cache hits / misses /
  occupancy are observable through `Renderer::cache_stats()`. Designed
  to consume scribe's `Group { cache_key: Some(deterministic_hash(face,
  glyph, size, subpixel)) }` glyph wrappers (#357).

### Initial release

- Initial release: pure-Rust vector→raster rendering kernel.
- Path flattening: quadratic + cubic de Casteljau subdivision; SVG
  elliptic-arc → cubic-Bezier conversion (Appendix F.6.5).
- Scanline AA fill: active-edge list, even-odd / non-zero rules,
  configurable 1/2/4/8× supersampling (default 4×).
- Stroke geometry: real offset polygon (not dilation) — Butt / Round /
  Square caps, Miter / Round / Bevel joins, dash pattern.
- Gradient evaluation: linear + radial, Pad / Reflect / Repeat spread.
- Single-path clip; group opacity; premultiplied-alpha composite onto
  an `Rgba` `VideoFrame`.
- `Renderer { width, height, supersampling, background }` + free
  `rasterize(&VectorFrame)` convenience function.
