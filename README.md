# oxideav-raster

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
| Scanline AA fill (even-odd / NZ)          | done                 |
| Configurable supersampling 1/2/4/8×       | done                 |
| Stroke geometry (caps + joins)            | done                 |
| Dash patterns                             | done — closed-subpath seam dash joins continuously (SVG 1.1 §11.4 / SVG 2 §13.5.6) |
| Linear + radial gradients                 | done                 |
| Off-centre focal radial                   | done                 |
| Two-circle radial (`fr` focal radius)     | done (SVG 2 §13.2.4 / CSS Images 3) |
| Pad / Reflect / Repeat spread             | done                 |
| `color-interpolation: sRGB` (default)     | done                 |
| `color-interpolation: linearRGB`          | done                 |
| Gradient stops LUT (256-entry pre-bake)   | done                 |
| Single-path clip                          | done                 |
| Group opacity                             | done                 |
| Soft masks (luminance / alpha)            | done                 |
| Bitmap cache (`Group::cache_key`)         | done (bbox)          |
| Image resample: nearest / bilinear        | done                 |
| Image resample: Lanczos2 / Lanczos3       | done                 |
| Image resample: Mitchell–Netravali / Catmull–Rom bicubic | done   |
| Image resample: cubic B-spline (B=1,C=0)  | done                 |
| 12 separable blend modes (PDF §11.3.5.2)  | done                 |
| 4 non-separable HSL modes (PDF §11.3.5.3) | done                 |
| Filters — all 17 §15 / Filter-Effects-1 primitives | done — morphology, color-matrix, Gaussian blur, component-transfer, composite, convolve-matrix, turbulence, flood, offset, merge, tile, displacement-map, blend, diffuse-lighting, specular-lighting, image, drop-shadow |
| `<pattern>` tiled fill/stroke paint (SVG2 §14.3) | done — user-space tile rect + patternTransform + wrap-around sampling; lacks a core `Paint` variant |
| Pattern `viewBox` + `preserveAspectRatio` (SVG2 §14.3.2 / §8.2) | done — all 10 alignments × meet/slice + `align=none`, slice overhang tile-clipped |
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
