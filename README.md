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
| Dash patterns                             | done                 |
| Linear + radial gradients                 | done                 |
| Off-centre focal radial                   | done                 |
| Pad / Reflect / Repeat spread             | done                 |
| `color-interpolation: sRGB` (default)     | done                 |
| `color-interpolation: linearRGB`          | done (round 4)       |
| Gradient stops LUT (256-entry pre-bake)   | done (round 7)       |
| Single-path clip                          | done                 |
| Group opacity                             | done                 |
| Soft masks (luminance / alpha)            | done (round 2)       |
| Bitmap cache (`Group::cache_key`)         | done (round 3, bbox) |
| Image resample: nearest                   | done                 |
| Image resample: bilinear                  | done (round 2)       |
| Image resample: Lanczos2 (4×4 sinc)       | done (round 3)       |
| Image resample: Lanczos3 (6×6 sinc)       | done (round 10)      |
| Image resample: Mitchell–Netravali bicubic| done (round 4)       |
| Image resample: Catmull–Rom bicubic       | done (round 8)       |
| Image resample: cubic B-spline (B=1,C=0)   | done (round 9)       |
| 12 separable blend modes (PDF §11.3.5.2)  | done (round 5)       |
| 4 non-separable HSL modes (PDF §11.3.5.3) | done (round 6)       |
| `feMorphology` erode / dilate (SVG §15.20)| done (round 11)      |
| `feColorMatrix` matrix/sat/hue/luma (§15.10)| done (round 12)    |
| `feGaussianBlur` separable Gaussian (§15.17)| done (round 13)    |
| `feComponentTransfer` 5-mode per-channel (§15.11)| done (round 14)|
| `feComposite` Porter-Duff + arithmetic (§15.12)| done (round 188) |
| `feConvolveMatrix` arbitrary 2-D convolution (§15.13)| done (round 206) |
| Filters (drop-shadow, feTurbulence, lighting, …)| partial (morphology + color-matrix + Gaussian blur + component-transfer + composite + convolve-matrix) |
| `<pattern>` paint                         | pending              |
| Full ICC color-managed pipeline           | pending              |

## Usage

```rust
use oxideav_raster::Renderer;
use oxideav_core::VectorFrame;

let frame: VectorFrame = /* … decoded by oxideav-svg / oxideav-pdf */;
let renderer = Renderer::new(800, 600);
let video = renderer.render(&frame);
// `video` is a packed Rgba VideoFrame, 800×600, ready for
// composite onto a player canvas or encode through any RGBA-input
// codec.
```

For a one-shot render at the frame's natural pixel size:

```rust
use oxideav_raster::rasterize;

let frame_out = rasterize(&frame); // returns oxideav_core::Frame::Video(...)
```

## Workspace policy

No `lyon` / `tiny-skia` / `resvg` / `kurbo` / `pathfinder` source has
been used. Algorithms are implemented from first principles against
SVG 1.1 + PDF 1.4 specs. The de Casteljau flattening and active-edge-
list fill are also used (in glyph-specific form) by
[`oxideav-scribe`](https://github.com/OxideAV/oxideav-scribe); the
implementations here are independent and shape-general. A future task
(#354 r1) will refactor scribe to depend on this crate.
