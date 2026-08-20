# oxideav-raster

[![CI](https://github.com/OxideAV/oxideav-raster/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-raster/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-raster.svg)](https://crates.io/crates/oxideav-raster) [![docs.rs](https://docs.rs/oxideav-raster/badge.svg)](https://docs.rs/oxideav-raster) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust vector→raster rendering kernel for the
[`oxideav`](https://github.com/OxideAV/oxideav) framework. Consumes a
resolution-independent [`VectorFrame`](https://docs.rs/oxideav-core/latest/oxideav_core/vector/struct.VectorFrame.html)
(SVG 1.1 / PDF 1.4 intersection: paths, paints, strokes, gradients,
groups with opacity / clip / transform, embedded raster images) and
produces a packed `Rgba` `VideoFrame`.

## Status

| Capability                                | Status               |
| ----------------------------------------- | -------------------- |
| Path flattening (line/quad/cubic)         | done                 |
| SVG elliptic-arc → cubic Bezier           | done                 |
| Scanline AA fill (even-odd / NZ)          | done — analytic horizontal coverage (fractional span boundaries) + vertical supersampling, so near-vertical edges anti-alias in X |
| Configurable supersampling 1/2/4/8×       | done                 |
| Stroke geometry (caps + joins)            | done — round joins/caps tessellate with a radius-adaptive sagitta-bounded chord count (sub-pixel-smooth at every width) |
| SVG 2 `stroke-linejoin: miter-clip` / `arcs` (§13.5.5) | done — `ExtendedStroke` + `stroke_to_fill_path_ext`; miter-clip trims the apex at `miterlimit/2 · width` instead of bevelling; `arcs` falls through to miter-clip on flattened (zero-curvature) input per spec |
| Dash patterns                             | done — closed-subpath seam dash joins continuously (SVG 1.1 §11.4 / SVG 2 §13.5.6) |
| Linear + radial gradients                 | done                 |
| Off-centre focal radial                   | done                 |
| Two-circle radial (`fr` focal radius)     | done (SVG 2 §13.2.4 / CSS Images 3) |
| Pad / Reflect / Repeat spread             | done                 |
| `color-interpolation: sRGB` (default)     | done                 |
| `color-interpolation: linearRGB`          | done                 |
| Gradient stops LUT (256-entry pre-bake)   | done — sub-entry linear interpolation between LUT entries (no 256-step banding) |
| Single-path clip                          | done                 |
| Top-level `viewBox` → viewport `preserveAspectRatio` (SVG 1.1 §7.8 / §8.2 / §8.6) | done — all 10 alignments × meet/slice + `none`; default `xMidYMid meet`; slice overflow cropped at canvas edge; degenerate viewBox disables render |
| Group opacity                             | done — isolated offscreen composite (SVG 2 §3.4); overlapping children no longer double-darken |
| Soft masks (luminance / alpha)            | done                 |
| Bitmap cache (`Group::cache_key`)         | done (bbox)          |
| Image resample: nearest / bilinear        | done                 |
| Image resample: Lanczos2 / Lanczos3       | done                 |
| Image resample: Mitchell–Netravali / Catmull–Rom bicubic | done   |
| Image resample: cubic B-spline (B=1,C=0)  | done                 |
| 12 separable blend modes (PDF §11.3.5.2)  | done                 |
| 4 non-separable HSL modes (PDF §11.3.5.3) | done                 |
| Filters — all 17 §15 / Filter-Effects-1 primitives | done — morphology, color-matrix, Gaussian blur (incl. Filter-Effects-1 §9.14 `edgeMode` duplicate/wrap/none), component-transfer, composite, convolve-matrix, turbulence, flood, offset, merge, tile, displacement-map, blend, diffuse-lighting, specular-lighting, image, drop-shadow |
| `<pattern>` tiled fill/stroke paint (SVG2 §14.3) | done — user-space tile rect + patternTransform + wrap-around sampling; lacks a core `Paint` variant |
| Pattern `viewBox` + `preserveAspectRatio` (SVG2 §14.3.2 / §8.2) | done — all 10 alignments × meet/slice + `align=none`, slice overhang tile-clipped |
| `color-interpolation-filters` working space (Filter Effects 1 §10) | done — `linearRGB` (initial) / `sRGB` / `auto`; linearise→run→re-encode wrapper, alpha preserved |
| Filter primitive subregion clip (Filter Effects 1 §9.4) | done — `x`/`y`/`width`/`height` hard-clip rectangle on a primitive result; keeps partly-intersecting pixels, disables the primitive on non-positive extent |
| `feComposite` Porter-Duff operators (Filter Effects 1 §9.8) | done — `over` / `in` / `out` / `atop` / `xor` / `lighter` + `arithmetic`; `lighter` (`Fa=Fb=1`) unions disjoint coverage and saturates overlap |
| `feBlend` extended mode set (Filter Effects 1 §9.13) | done — all 16 Compositing-1 `<blend-mode>` values; the 5 SVG 1.1 modes keep their byte-stable §15.9 closed form, the 11 extensions (overlay / color-dodge / color-burn / hard-light / soft-light / difference / exclusion / hue / saturation / color / luminosity) route through the shared `blend_over` mix-with-Source-Over path with `in`=Cs, `in2`=Cb |
| Filter primitive tree (Filter Effects 1 §9.2 / §9.3) | done — `FilterGraph` evaluates ordered primitives with `in`/`in2` resolution (`SourceGraphic` / `SourceAlpha` / named `result` back-references, closest-preceding rule, forward-reference degradation), per-step §9.4 subregion clip, one-pass linearRGB working space, anchored §9.20 `feTile` |
| CSS filter shorthand functions (Filter Effects 1 §6 / §13.1) | done — all 10 `<filter-function>`s expanded to their §13.1 primitive equivalents (exact grayscale/sepia matrices, table/linear transfer forms, blur with `edgeMode` `none`); `apply_filter_functions` runs a `<filter-value-list>` pinned to sRGB per §10 |
| CSS `filter` value parser (Filter Effects 1 §5 / §6.1) | done — `<filter-value-list>` → `FilterFunction`s with §6.1 defaults, negative/unit/overflow rejection, `drop-shadow()` `<color>? && <length>{2,3}`, `currentColor` substitution; panic-free on hostile input, byte-offset errors |
| Fuzz + benches                            | done — cargo-fuzz targets on the CSS parser (~17M execs) + filter graph (~4.5M execs) with daily CI; three Criterion harnesses (`render` / `filter` / `sampling`) covering fill, stroke, scenes, all seven resampling kernels, patterns, soft masks, gradients, cache hits, and the filter kernels — see `BENCHMARKS.md` |
| r449 perf pass (zero behaviour change)    | done — 1.79× geo-mean over 32 benches, byte-exact by differential tests + 31-scene frame-hash A/B; caption-density scene 32.5×, small-shape composite O(shape) not O(canvas), Lanczos ~2–2.4×, morphology O(1)/px, component-transfer 4.2× |
| Full ICC color-managed pipeline           | not yet              |

## Usage

```rust
use oxideav_raster::Renderer;
use oxideav_core::VectorFrame;

let frame: VectorFrame = /* … decoded by oxideav-svg / oxideav-pdf */;
let renderer = Renderer::new(800, 600);
let video = renderer.render(&frame);
// `video` is a packed Rgba VideoFrame, 800×600, ready to composite
// onto a player canvas or encode through any RGBA-input codec.
```

For a one-shot render at the frame's natural pixel size:

```rust
use oxideav_raster::rasterize;

let frame_out = rasterize(&frame); // returns oxideav_core::Frame::Video(...)
```

## Clean-room policy

Algorithms are implemented from first principles against the SVG 1.1 +
PDF 1.4 specifications. The de Casteljau flattening and active-edge-list
fill are also used (in glyph-specific form) by
[`oxideav-scribe`](https://github.com/OxideAV/oxideav-scribe); the
implementations here are independent and shape-general.

## License

MIT — see [LICENSE](LICENSE).
