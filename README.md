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
| Two-circle radial (`fr` focal radius)     | done (round 297) — SVG 2 §13.2.4 / CSS Images 3 |
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
| `feTurbulence` Perlin noise / fractalNoise (§15.24)| done (round 212) |
| `feFlood` solid-colour source primitive (§15.16)| done (round 217) |
| `feOffset` integer + bilinear-fractional shift (§15.21)| done (round 217) |
| `feMerge` N-layer `over` composite (§15.19)| done (round 217) |
| `feTile` periodic tiled-pattern (§15.23)   | done (round 224)     |
| `feDisplacementMap` channel-driven warp (§15.15)| done (round 230) |
| `feBlend` 5-mode pixel-wise blend (§15.9)  | done (round 233)     |
| `feDiffuseLighting` Phong-diffuse + 3 light sources (§15.14)| done (round 247) |
| `feSpecularLighting` Phong-specular + 3 light sources (§15.22)| done (round 259) |
| `feImage` raster placement + §7.8 preserveAspectRatio (§15.18)| done (round 268) |
| `feDropShadow` shorthand drop shadow (Filter Effects 1 §9.12)| done (round 288) — spec equivalent chain `feGaussianBlur(alpha) → feOffset → feFlood → feComposite(in) → feMerge(source)` |
| Filters — all 17 §15 / Filter-Effects-1 primitives | done — morphology + color-matrix + Gaussian blur + component-transfer + composite + convolve-matrix + turbulence + flood + offset + merge + tile + displacement-map + blend + diffuse-lighting + specular-lighting + image + drop-shadow |
| `<pattern>` tiled fill/stroke paint (SVG2 §14.3)| done (round 276) — user-space tile rect + patternTransform + wrap-around sampling; lacks a core `Paint` variant |
| Pattern `viewBox` + `preserveAspectRatio` fitting (SVG2 §14.3.2 / §8.2)| done (round 280) — all 10 alignments × meet/slice + `align=none`, slice overhang tile-clipped |
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
