# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
