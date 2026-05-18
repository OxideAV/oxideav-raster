# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Mitchell–Netravali bicubic image resampling (`ImageFilter::Mitchell`).
  Standard `B = 1/3, C = 1/3` parameter pair from Mitchell & Netravali's
  1988 reconstruction-filter paper; 4×4 separable kernel sampled in
  premultiplied-alpha space. Smoother than Lanczos2 (no visible halo
  banding on hard step edges) and sharper than bilinear; useful as the
  default for downscaling photographic content. Joins the existing
  Nearest / Bilinear / Lanczos2 entries on `Renderer::image_filter`.
- `Renderer::color_interpolation` selects between SVG 2 §13.9
  `color-interpolation: sRGB` (the default, matches the historic SVG 1.1
  default) and `color-interpolation: linearRGB`. Linear-RGB
  interpolation converts each gradient stop through the IEC 61966-2-1
  sRGB transfer function (`(C + 0.055) / 1.055)^2.4` above 0.04045 /
  linear `÷ 12.92` below), interpolates in the light-energy-linear
  domain, and encodes back to sRGB for the output frame. Eliminates the
  "ugly grey midpoint" artefact where complementary primaries (e.g.
  black↔white, red↔green) appear too dark in the middle of a naive
  sRGB-space gradient. Wired through `sample_paint_in` →
  `eval_{linear,radial}_gradient_in` so consumers can also evaluate
  paints standalone.



### Fixed

- *(clippy)* inline Group opacity into struct literal
- *(clippy)* rewrite stroke algorithm doc as nested bullets
- *(clippy)* clear the 1.95 lint sweep (cast/checked/match/dead/doc)

### Other

- Lanczos2 (4×4 windowed sinc) image resampling
- store touched-pixel bbox crop instead of full canvas
- enlarge unit-circle arc test radius so polyline assertion holds
- re-land Node::SoftMask render path
- Revert "renderer: implement Node::SoftMask render path"
- implement Node::SoftMask render path
- off-centre focal radial gradients
- bilinear image resampling for Node::Image
- bitmap cache for Group::cache_key subtrees

### Added

- `Node::SoftMask` render path — luminance + alpha mask kinds. The
  `mask` subtree is rasterised to its own offscreen RGBA buffer and
  reduced to a per-pixel coverage byte (BT.709 luminance for
  `MaskKind::Luminance`, the mask's own alpha for `MaskKind::Alpha`);
  the `content` subtree is then blitted onto the destination with
  that coverage as a per-pixel alpha modulator. Covers SVG `<mask>`
  and PDF `SMask` (subtype `Luminosity` vs. `Alpha`). Requires the
  `Node::SoftMask` variant + `MaskKind` enum landed alongside in
  `oxideav-core`.
- Off-centre focal radial gradients (SVG 1.1 §13.2.4 / PDF Type 3
  axial radial). `eval_radial_gradient` now solves the standard
  quadratic in `t` for the unique circle (centre on the focal→centre
  line, parameterised by `t`) that contains the sample point;
  degenerates correctly to the centred-radial formula when
  `focal == center`. SVG-mandated focal-clamp-inside-boundary is
  applied so the equation always has a real positive root.
- Bilinear image resampling for `Node::Image`. Selectable via
  `Renderer::image_filter: ImageFilter`; defaults to
  `ImageFilter::Bilinear` (matching CSS `image-rendering: auto` and the
  SVG spec). `ImageFilter::Nearest` retains the round-1 block-replication
  behaviour for callers that want pixel-perfect sprite rendering.
  Bilinear samples in premultiplied-alpha space and uses clamp-to-edge
  for the 2×2 footprint at texture boundaries. Lanczos2 is queued for
  round 3+.
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
