# oxideav-raster

Pure-Rust vector→raster rendering kernel for the
[`oxideav`](https://github.com/OxideAV/oxideav) framework. Consumes a
resolution-independent [`VectorFrame`](https://docs.rs/oxideav-core/latest/oxideav_core/vector/struct.VectorFrame.html)
(SVG 1.1 / PDF 1.4 intersection: paths, paints, strokes, gradients,
groups with opacity / clip / transform, embedded raster images) and
produces a packed `Rgba` `VideoFrame`.

## Round 1

| Capability                          | Status     |
| ----------------------------------- | ---------- |
| Path flattening (line/quad/cubic)   | done       |
| SVG elliptic-arc → cubic Bezier     | done       |
| Scanline AA fill (even-odd / NZ)    | done       |
| Configurable supersampling 1/2/4/8× | done       |
| Stroke geometry (caps + joins)      | done       |
| Dash patterns                       | done       |
| Linear + radial gradients           | done       |
| Pad / Reflect / Repeat spread       | done       |
| Single-path clip                    | done       |
| Group opacity                       | done       |
| Embedded raster image (nearest)     | done       |
| Filters (blur, drop-shadow, …)      | round 2    |
| `<pattern>` paint                   | round 2    |
| Soft masks (luminance / alpha)      | round 2    |
| Color-managed pipeline (linear sRGB)| round 2    |
| Bilinear image resampling           | round 2    |

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
