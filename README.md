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
| Path flattening (line/quad/cubic)         | done — segment count from a rigorous 1/32 device-px error bound (no faceting at any zoom, a few hundred segments for a radius-1000 px circle); off-canvas curves culled to chords (exact for fills); transform in f64 |
| SVG elliptic-arc → cubic Bezier           | done                 |
| Exact-area AA fill (even-odd / NZ)        | done — default `CoverageMode::Analytic`: y-monotone-chain scan conversion, closed-form per-boundary area integration on provably ordered strips, vertex + crossing subdivision only where chains cross; exact for self-intersecting / overlapping outlines; windowed to the shape's bbox with f64 edge clipping (precise at 10⁶ px off-canvas); each row streamed straight into the compositor as spans of covered pixels (the gaps between the pieces of a shape are never visited); boundary-cell areas in a cancellation-free form — within the 0.5-level quantisation limit of clipped-polygon ground truth on needles, near-axis edges and specks |
| Supersampled AA fill 1/2/4/8×             | done — `CoverageMode::Supersampled`: analytic horizontal coverage + vertical sub-scanlines; kept for byte-stable callers |
| Open subpaths implicitly closed for fill (SVG 2 §13.4.1) | done |
| Stroke geometry (caps + joins)            | done — built in user space (exact under rotation / non-uniform scale, SVG 2 §13.5); offset outline where provably exact, positively wound union of pieces where it is not (thick strokes on tight curves have no holes); round joins / caps / zero-length dots straddle the circle at 1/32 px; matches a brute-force distance field |
| SVG 2 `stroke-linejoin: miter-clip` / `arcs` (§13.5.5) | done — `ExtendedStroke` + `stroke_to_fill_path_ext`; miter-clip trims the apex at `miterlimit/2 · width` instead of bevelling; `arcs` falls through to miter-clip on flattened (zero-curvature) input per spec |
| Dash patterns                             | done — user-space lengths; closed-subpath seam dash joins continuously (SVG 1.1 §11.4 / SVG 2 §13.5.6); zero-length dashes paint their caps (dotted lines) |
| Linear + radial gradients                 | done — geometry in the node's user space, sampled through the inverse CTM (viewBox scale, group / element transforms, elliptical radial rings under non-uniform scale — SVG 2 §14.2.2.1 / §14.2.3.1) |
| Off-centre focal radial                   | done                 |
| Two-circle radial (`fr` focal radius)     | done (SVG 2 §13.2.4 / CSS Images 3) |
| Conic (sweep) gradient                    | done — `ConicGradient` through `GradientPaint` (start angle, partial sweep completed by the spread method, bbox units, gradientTransform); lacks a core `Paint` variant |
| Pad / Reflect / Repeat spread             | done                 |
| `color-interpolation: sRGB` (default)     | done                 |
| `color-interpolation: linearRGB`          | done                 |
| Gradient evaluation                       | done — row-wise, `t` affine in device position (f64 set-up), stops evaluated exactly per pixel (hard stops stay 1 px sharp at any length); optional 8×8 Bayer `GradientDither::Ordered` |
| Single-path clip                          | done                 |
| Top-level `viewBox` → viewport `preserveAspectRatio` (SVG 1.1 §7.8 / §8.2 / §8.6) | done — all 10 alignments × meet/slice + `none`; default `xMidYMid meet`; slice overflow cropped at canvas edge; degenerate viewBox disables render |
| Group opacity                             | done — isolated offscreen composite (SVG 2 §3.4); overlapping children no longer double-darken; the offscreen image covers only the group's analytic bounds ∩ canvas |
| Soft masks (luminance / alpha)            | done — both subtrees rasterised into a window of content bounds ∩ mask bounds ∩ canvas |
| Group cache (`Group::cache_key`)         | done — one bitmap per (key, linear transform, ¼ px sub-pixel phase, pixel-affecting settings), blitted at any whole-pixel position; bitmap sized to the group's analytic bounds; byte-budgeted LRU (32 MiB default); `GroupCache` handle shared across frames / renderers; lookups are concurrent, scene-band workers read a published snapshot; the picture is bit-identical at every budget — see *Group cache* below |
| Image resample: nearest / bilinear        | done                 |
| Image resample: Lanczos2 / Lanczos3       | done                 |
| Image resample: Mitchell–Netravali / Catmull–Rom bicubic | done   |
| Image resample: cubic B-spline (B=1,C=0)  | done                 |
| 12 separable blend modes (PDF §11.3.5.2)  | done                 |
| 4 non-separable HSL modes (PDF §11.3.5.3) | done                 |
| Filters — all 17 §15 / Filter-Effects-1 primitives | done — morphology, color-matrix, Gaussian blur (incl. Filter-Effects-1 §9.14 `edgeMode` duplicate/wrap/none; on premultiplied samples per §9.1, as are the bilinear samplers of offset / displacement-map / image), component-transfer, composite, convolve-matrix, turbulence, flood, offset, merge, tile, displacement-map, blend, diffuse-lighting, specular-lighting, image, drop-shadow |
| Gradient paint servers with `gradientUnits` + `gradientTransform` (SVG 2 §14.2 / §8.11) | done — `GradientPaint` record (linear / radial / two-circle focal radial, `objectBoundingBox` or `userSpaceOnUse`, post-multiplied transform) resolved against the node CTM + bbox by `GradientPaint::sampler`; `fill_path_with_gradient` / `stroke_path_with_gradient`; rsvg-convert interiors within 1 level; lacks a core `Paint` variant — see the contract below |
| `<pattern>` tiled fill/stroke paint (SVG2 §14.3) | done — user-space tile rect + patternTransform + wrap-around sampling; lacks a core `Paint` variant |
| Pattern `viewBox` + `preserveAspectRatio` (SVG2 §14.3.2 / §8.2) | done — all 10 alignments × meet/slice + `align=none`, slice overhang tile-clipped |
| `color-interpolation-filters` working space (Filter Effects 1 §10) | done — `linearRGB` (initial) / `sRGB` / `auto`; linearise→run→re-encode wrapper, alpha preserved |
| Filter primitive subregion clip (Filter Effects 1 §9.4) | done — `x`/`y`/`width`/`height` hard-clip rectangle on a primitive result; keeps partly-intersecting pixels, disables the primitive on non-positive extent |
| `feComposite` Porter-Duff operators (Filter Effects 1 §9.8) | done — `over` / `in` / `out` / `atop` / `xor` / `lighter` + `arithmetic`; `lighter` (`Fa=Fb=1`) unions disjoint coverage and saturates overlap |
| `feBlend` extended mode set (Filter Effects 1 §9.13) | done — all 16 Compositing-1 `<blend-mode>` values; the 5 SVG 1.1 modes keep their byte-stable §15.9 closed form, the 11 extensions (overlay / color-dodge / color-burn / hard-light / soft-light / difference / exclusion / hue / saturation / color / luminosity) route through the shared `blend_over` mix-with-Source-Over path with `in`=Cs, `in2`=Cb |
| Filter primitive tree (Filter Effects 1 §9.2 / §9.3) | done — `FilterGraph` evaluates ordered primitives with `in`/`in2` resolution (`SourceGraphic` / `SourceAlpha` / named `result` back-references, closest-preceding rule, forward-reference degradation), per-step §9.4 subregion clip, one-pass linearRGB working space, anchored §9.20 `feTile` |
| CSS filter shorthand functions (Filter Effects 1 §6 / §13.1) | done — all 10 `<filter-function>`s expanded to their §13.1 primitive equivalents (exact grayscale/sepia matrices, table/linear transfer forms, blur with `edgeMode` `none`); `apply_filter_functions` runs a `<filter-value-list>` pinned to sRGB per §10 |
| CSS `filter` value parser (Filter Effects 1 §5 / §6.1) | done — `<filter-value-list>` → `FilterFunction`s with §6.1 defaults, negative/unit/overflow rejection, `drop-shadow()` `<color>? && <length>{2,3}`, `currentColor` substitution; panic-free on hostile input, byte-offset errors |
| Fuzz + benches                            | done — cargo-fuzz targets on the CSS parser (~17M execs), filter graph (~4.5M execs) scene rendering (`render_paths`: raw-f32 paths / strokes / dashes / transforms, serial = parallel) and the group cache (`group_cache`: cold = warm = starved = disabled = shared, byte for byte) with daily CI; six Criterion harnesses (`render` / `filter` / `sampling` / `scenes` / `group_cache` / `pipeline`) covering fill, stroke, scenes, all seven resampling kernels, patterns, soft masks, gradients, cache hits, and the filter kernels — see `BENCHMARKS.md` |
| r449 perf pass (zero behaviour change)    | done — 1.79× geo-mean over 32 benches, byte-exact by differential tests + 31-scene frame-hash A/B; caption-density scene 32.5×, small-shape composite O(shape) not O(canvas), Lanczos ~2–2.4×, morphology O(1)/px, component-transfer 4.2× |
| Rendering options                         | done — `with_execution_context` (scene-band parallel: each worker walks the scene over its own rows of the canvas, byte-identical to serial at every budget; 1080p illustration 4 workers ≈ 2.7×), `with_compositing_space(LinearRgb)` (linear-light AA + translucency, including group / soft-mask / cached-bitmap blits; default sRGB per SVG / CSS), `with_scene_supersampling(k)` (bounds conflation seams between abutting shapes by 1/(4k)) |
| r484 quality + performance pass           | done — 1080p illustration 769 → 166 ms, 20 000 small shapes on 4K 731 → 29 ms, 4K gradients 55 / 63 → 33 / 47 ms (4 workers: 9 ms), 256² star stroke 131 → 1.9 ms — see `BENCHMARKS.md` |
| r488 group cache + scene bands           | done — 300 keyed glyph groups on 1206 × 2622: first frame 1 245 → 1.1 ms, steady 1 227 → 0.26 ms; group opacity / soft masks / clips in bounded windows; 1080p illustration with 4 workers 208 → 68 ms — see *Group cache* and `BENCHMARKS.md` |
| r491 throughput + accuracy pass           | done — span-wise coverage delivery, strip order kept across rows, per-thread scratch reuse, per-row blend dispatch, contention-free cache lookups: 1.2–1.7× on fills / strokes / groups, warm keyed text with 4 workers 8.5 → 1.0 ms; steep-edge coverage error 5 → 0.5 levels; premultiplied blur — see *Throughput by pipeline stage* and *Accuracy* below |
| Full ICC color-managed pipeline           | not yet              |

## Usage

```rust
use oxideav_raster::Renderer;
use oxideav_core::VectorFrame;

// Normally decoded by oxideav-svg / oxideav-pdf; an empty 800×600 canvas here.
let frame = VectorFrame::new(800.0, 600.0);
let renderer = Renderer::new(800, 600);
let video = renderer.render(&frame);
// `video` is a packed Rgba VideoFrame, 800×600, ready to composite
// onto a player canvas or encode through any RGBA-input codec.
```

For a one-shot render at the frame's natural pixel size:

```rust
use oxideav_raster::rasterize;

# let frame = oxideav_core::VectorFrame::new(800.0, 600.0);
let frame_out = rasterize(&frame); // returns oxideav_core::Frame::Video(...)
```

## Group cache

A [`Group`](https://docs.rs/oxideav-core/latest/oxideav_core/vector/struct.Group.html)
with a `cache_key` (a glyph, an icon, a decorative subtree — the key
names the *content*) is rasterised once and blitted from then on:

```rust
use oxideav_raster::{GroupCache, Renderer};

# let frames: Vec<oxideav_core::VectorFrame> = Vec::new();
// One cache for the whole session; a renderer per frame starts warm.
let cache = GroupCache::with_max_bytes(64 << 20); // default: 32 MiB
for frame in &frames {
    let renderer = Renderer::new(1206, 2622).with_group_cache(cache.clone());
    let _video = renderer.render(frame);
}
println!("{:?}", cache.stats()); // hits, misses, entries, bytes, evictions
```

* **What is shared.** An entry is identified by the producer's key, the
  linear part of the effective transform (`a b c d`), the sub-pixel
  phase of its translation, and every renderer setting that changes
  pixels (coverage kernel, image filter, gradient space / dither, blend
  mode, compositing space) — compared field by field, never through a
  combined hash. The whole-pixel part of the translation is *not* part
  of it: the bitmap is blitted at that offset, so one glyph bitmap
  serves every pen position with the same phase, on any canvas.
* **Sub-pixel grid.** A keyed group is drawn with its translation
  snapped to `1 / Renderer::cache_subpixel_grid` px (default 4 → ¼ px;
  `with_cache_subpixel_grid`). It therefore sits at most `1 / (2·grid)`
  px per axis from where the same group without a key would be drawn,
  and an anti-aliased edge pixel differs by at most `1 / grid` of full
  scale (measured on a curved outline over 12 arbitrary positions:
  44 / 255 at grid 4, 19 at 8, 10 at 16). On the grid there is no
  displacement: the pixels equal the unkeyed group's, up to one 8-bit
  rounding step where the bitmap's extra quantisation or the f32
  rounding of device coordinates shows. Grid `0` keeps the exact
  fraction (sharing only between bit-identical phases); grid `1` snaps
  to whole pixels.
* **Size.** The offscreen bitmap covers the group's own bounds —
  computed from the geometry (control-point hull, stroke reach `half
  width · max(miter limit, √2)` under the transform's largest singular
  value, clip intersection) plus one pixel — never the canvas. The
  cache holds at most its byte budget, evicting least recently used
  entries. A bitmap over a quarter of the budget is not stored, but it
  is still rasterised in the stored bitmap's own frame (a transient
  bitmap of the rows on the canvas), so the pixels are the same at
  every budget, including none. A group more than 4× larger than its
  on-canvas part is never cached; it is rendered through a transient
  window of bounds ∩ canvas.
* **Semantics.** A keyed group is an *isolated* group whether or not
  its bitmap is cached at the moment: its children composite among
  themselves (with the renderer's blend mode), and the result is laid
  over the canvas with source-over under the group's opacity and the
  inherited clip — both applied at blit time, so neither invalidates an
  entry. The picture never depends on what the cache happens to hold,
  nor on its budget. Ordered gradient dither inside a cached bitmap is
  anchored to the bitmap, not the canvas.
* **Threads.** Lookups take a shared lock and stamp the entry from an
  atomic clock; eviction takes the least recently stamped entry. Under
  a thread budget the scene-band workers of a frame read the entries
  that existed when the frame began from a read-only copy of the map
  (no lock, nothing written that another worker reads) and keep what
  the frame adds in a private map, so a warm page costs no more with
  workers than without.

The reported shape — 300 keyed glyph groups (16 px em, 40 distinct
glyphs, fractional pen positions) on a 1206 × 2622 canvas, plus four
~600 px keyed panels (`cargo bench --bench group_cache`, medians,
Apple M5 Max, quietest of five passes on a shared machine):

| scene | unkeyed | keyed, first frame | keyed, steady (same renderer) | keyed, steady (renderer per frame, shared cache) |
| ----- | ------: | -----------------: | ----------------------------: | -----------------------------------------------: |
| 300 glyphs — before | 1.24 ms | 1 245 ms | 1 227 ms | n/a (no shareable cache) |
| 300 glyphs — after  | 1.2–1.7 ms | **1.12 ms** | **0.26 ms** | **0.18 ms** |
| 300 glyphs, page scrolled 3 px per frame — before → after | | | 1 486 ms → **0.18 ms** | |
| 4 large panels — before | 6.5 ms | 24.1 ms | 2.2 ms | n/a |
| 4 large panels — after  | 6.4 ms | **6.9 ms** | **1.1 ms** | **1.3 ms** |

A keyed miss now costs about what the unkeyed draw does (the first text
frame is already faster than unkeyed, because repeated glyphs hit within
the frame), and a hit is a clipped blit.

The three properties a text renderer needs from the cache are pinned by
`tests/text_cache.rs` on a page of 6 000 glyph groups (96 glyphs × 4
sizes, arbitrary fractional pen positions and line spacing, 1920 × 1080),
by counting rather than timing:

| property | measured |
| --- | --- |
| a miss allocates a glyph-sized buffer | cold frame (2 870 misses, 3 130 hits within the frame) under a counting allocator: **1** allocation of 83 kB or more — the 8.3 MB output canvas — and none above **2 100 bytes** otherwise; 10.3 MB requested in total (canvas + 2.2 MB of bitmaps), against 23.7 MB before the scratch reuse of this round |
| keys are position-independent up to the sub-pixel phase | bitmaps for the page ≤ glyphs × phases: 384 at grid 1, 1 536 at grid 2, 6 144 at the default ¼ px grid, however often the page is scrolled by fractions (24 scrolls: 100 % / 99.8 % / 97.8 % hits); a whole-pixel scroll or a second frame: **0** misses |
| the cache is bounded in bytes | a budget one twelfth of what the page wants (207 kB): never over budget, least recently used evicted, **identical bytes** to the unbounded render; zero budget (cache off): identical bytes too |

## Throughput by pipeline stage

`cargo bench --bench pipeline` renders one workload per stage of the
rasteriser on a 1920 × 1080 canvas. Before → after this round (baseline
`38b72ba`; Apple M5 Max, release; the two builds run alternately seven
times on a machine shared with nine other builds, five frames per case
and run — each figure is the median over the runs of the run's median
frame; rows whose code path did not change show the noise floor):

| scene (`pipeline/…`) | what it is | before | after | × |
| --- | --- | ---: | ---: | ---: |
| `text5000_unkeyed` | 5 000 glyph groups (4 sizes, fractional pens), no keys | 16.0 ms | 12.4 ms | 1.29 |
| `text5000_keyed_cold` | … keyed, empty cache (first frame) | 11.2 ms | 8.52 ms | 1.31 |
| `text5000_keyed_warm` | … keyed, warm cache, same renderer | 1.77 ms | 1.14 ms | 1.55 |
| `text5000_keyed_scrolling` | … keyed, renderer per frame around one cache, page scrolled by fractions | 6.65 ms | 4.96 ms | 1.34 |
| `fill_60k_edges` | one outline of 60 000 edges across the canvas | 30.6 ms | 20.8 ms | 1.47 |
| `fill_blobs3000` | 3 000 translucent discs, 4–120 px | 48.1 ms | 41.4 ms | 1.16 |
| `fill_tiny20000` | 20 000 opaque shapes of 3–10 px | 23.6 ms | 19.2 ms | 1.23 |
| `stroke_thin600` | 600 polylines × 24 segments, 1.5 px, all joins / caps | 59.0 ms | 40.7 ms | 1.45 |
| `stroke_hairline600` | … 0.35 px hairlines | 54.9 ms | 37.6 ms | 1.46 |
| `stroke_wide_curved300` | 300 curved polylines, 9 px | 116.3 ms | 94.7 ms | 1.23 |
| `stroke_dashed600` | 600 polylines, 2 px, dash 7-3-2-3 | 173.6 ms | 138.3 ms | 1.26 |
| `grad_linear_pad` | full-canvas linear gradient, 6 stops, pad | 8.67 ms | 8.48 ms | 1.02 |
| `grad_linear_reflect_alpha` | … translucent stops, reflect | 16.3 ms | 9.30 ms | 1.75 |
| `grad_radial_pad` | full-canvas radial gradient, pad | 10.5 ms | 10.4 ms | 1.01 |
| `grad_radial_focal_repeat` | … off-centre focus, repeat | 11.3 ms | 11.2 ms | 1.01 |
| `grad_small2000` | 2 000 small shapes, each with its own gradient | 25.2 ms | 20.8 ms | 1.21 |
| `image_upright24` | 24 images 256², scaled | 13.7 ms | 13.2 ms | 1.04 |
| `image_rotated24` | … rotated + non-uniform scale | 17.8 ms | 17.2 ms | 1.04 |
| `pattern_fill_rotated` | 1 040 px disc filled with a rotated pattern tile | 8.80 ms | 8.71 ms | 1.01 |
| `clip_nested150` | 150 disc-clipped groups inside a page clip | 19.0 ms | 16.4 ms | 1.16 |
| `softmask60` | 60 soft-masked groups (radial luminance / alpha masks) | 23.8 ms | 18.7 ms | 1.28 |
| `opacity_nested120` | 120 three-level translucent group stacks | 40.7 ms | 24.7 ms | 1.65 |
| `blend_multiply_blobs3000` | 3 000 discs under `BlendMode::Multiply` | 86.7 ms | 53.9 ms | 1.61 |
| `blend_luminosity_blobs3000` | 3 000 discs under `BlendMode::Luminosity` | 74.1 ms | 70.2 ms | 1.06 |

Under a budget of four workers (scene bands; byte-identical to the
serial render):

| scene, 4 workers | before | after | × |
| --- | ---: | ---: | ---: |
| `mt4_text5000_unkeyed` | 13.4 ms | 8.03 ms | 1.67 |
| `mt4_text5000_keyed_warm` | 8.46 ms | 1.04 ms | 8.12 |
| `mt4_fill_60k_edges` | 11.1 ms | 7.50 ms | 1.47 |
| `mt4_fill_blobs3000` | 17.9 ms | 14.6 ms | 1.23 |
| `mt4_stroke_thin600` | 20.6 ms | 14.4 ms | 1.43 |
| `mt4_stroke_dashed600` | 62.9 ms | 51.6 ms | 1.22 |
| `mt4_grad_small2000` | 8.97 ms | 6.93 ms | 1.29 |
| `mt4_clip_nested150` | 12.9 ms | 10.4 ms | 1.25 |
| `mt4_softmask60` | 12.5 ms | 9.38 ms | 1.33 |
| `mt4_opacity_nested120` | 16.1 ms | 9.55 ms | 1.68 |

What changed: the scan delivers each row as spans of covered pixels
instead of one range from the leftmost to the rightmost edge; the
chains crossing a strip keep their left-to-right order from strip to
strip and row to row instead of being collected and sorted per strip;
the scan's accumulators, the edge table and the flattened contours are
recycled per thread instead of allocated per path; the blend mode and
compositing space are resolved per row, and group / cache / mask blits
share the path compositor's kernel; cache lookups no longer serialise
the workers (the 4-worker warm page was almost 5× *slower* than its
serial form).

## Accuracy

Checked against references that are not other rasterisers:

| what | reference | result |
| --- | --- | --- |
| exact-area coverage on slivers down to 3 · 10⁻⁴ px wide, edges 10⁻¹ … 10⁻⁶ from horizontal / vertical, sub-pixel specks, star and half-pixel-lattice polygons | polygon clipped to each pixel, surveyor's formula, f64 (`tests/coverage_exact.rs`) | within **0.52** of 255 levels (0.5 is the quantisation limit) |
| long edges drifting 10⁻¹ … 10⁻⁶ px per row | same | before this round up to **5.0** levels off (the cell integral cancelled six digits in f32); now 0.50 |
| miter / bevel / round joins, 0.5° – 179.5°, miter limits 1 – 100 and 2 % either side of each angle's own limit, pens 5 px – 0.6 px | point membership in segment rectangles ∪ join pieces, 512-point lattice per pixel (`tests/stroke_extremes.rs`) | worst pixel **3.4** levels (round joins: 1/32 px chords); miter and bevel within 2.5 |
| hairlines 1 – 1/50 px, butt / square / round caps | same | worst pixel **2.3** levels; area within quantisation and the lattice's own noise |
| curve flattening at 60× … 10⁷× zoom (device extents to 10⁷ px) | the true cubic clipped per pixel | worst boundary pixel 4.8 / 7.1 / 7.9 / 6.7 / 4.8 levels, mean 0.8 – 3.6: the 1/32 px chord tolerance, not growing with scale |
| a glyph drawn directly vs from the keyed-group cache, linear-light compositing | each other | within 1 level (the cache blit used to mix encoded values: 128 vs 188 on a half-covered edge) |
| source-over onto a transparent pixel | the source colour | exact (was rounded through an 8-bit premultiplied value: (20, 20, 24) at alpha 10 → (26, 26, 26)) |
| Gaussian blur of a coloured shape on transparent | Filter Effects 1 §9.1 (premultiplied samples) | colour preserved, only alpha spreads (edge was (85, 13, 0) at alpha 85 for an orange (255, 40, 0)) |

## Gradient paint contract

Where a gradient's coordinates live, and who applies which transform:

* **`oxideav_core::Paint::LinearGradient` / `RadialGradient`** carry
  geometry in the painted node's **user space** (SVG 2 §14.2.2.1 /
  §14.2.3.1 `gradientUnits="userSpaceOnUse"`). The renderer maps every
  device pixel centre through the inverse of the node's current
  transformation matrix (viewBox → viewport fit ∘ ancestor group
  transforms ∘ the element's own transform) before evaluating the
  gradient — the same mapping the outline goes through. A decoder must
  therefore **not** pre-transform gradient points into device space, and
  a radial gradient under a non-uniform CTM renders elliptical rings by
  construction.
* **`oxideav_raster::GradientPaint`** is the additive record for
  gradients that have a coordinate system of their own: `geometry`
  (`GradientGeometry::Linear` / `Radial` / `FocalRadial`), `units`
  (`GradientUnits::UserSpaceOnUse` | `ObjectBoundingBox`) and
  `transform` (`gradientTransform`). Gradient space maps to user space
  by `B · transform` with `B = [bw 0 0 bh bx by]` for `objectBoundingBox`
  (SVG 2 §8.11) and the identity otherwise — `gradientTransform` is
  post-multiplied, exactly as §14.2.2.1 words it. `GradientPaint::sampler(&ctm, bbox, space)`
  resolves it to a `GradientSampler` (device → gradient inverse + stops
  LUT); it returns `None` when nothing is to be painted: an
  `objectBoundingBox` gradient on a box with zero width or height
  (§8.11), or a singular matrix. `path_bounding_box` yields the §8.11
  geometry box (fill geometry only — stroke width does not contribute,
  so stroke paint uses the same box as fill paint).
* `Renderer::fill_path_with_gradient` / `stroke_path_with_gradient`
  paint a single path with a `GradientPaint` under a given CTM (the same
  shape as the pattern endpoints). `GradientPaint::from_paint` lifts a
  user-space `Paint` gradient so a decoder can attach the transform it
  could not express in the core IR (the typical case is a radial
  gradient under a non-conformal mapping).
* All of the above is `#[non_exhaustive]` with `new()` / `with_*`
  builders; a future `oxideav-core` `Paint` variant carrying the same
  three fields is the intended end state, at which point the scene walk
  will route through `GradientPaint::sampler` directly.

## Clean-room policy

Algorithms are implemented from first principles against the SVG 1.1 +
PDF 1.4 specifications. The de Casteljau flattening and active-edge-list
fill are also used (in glyph-specific form) by
[`oxideav-scribe`](https://github.com/OxideAV/oxideav-scribe); the
implementations here are independent and shape-general.

## License

MIT — see [LICENSE](LICENSE).
