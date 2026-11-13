# oxideav-raster benchmarks

Criterion benchmarks for the rasterisation kernel hot paths, split
across three harnesses:

```sh
cargo bench --bench render    # flatten / scanline fill / scene walks
cargo bench --bench filter    # filter-primitive kernels (r401)
cargo bench --bench sampling  # image resampling + pattern paint (r449)
cargo bench --bench scenes    # realistic 1080p / 4K workloads (r484)
cargo bench --bench group_cache  # keyed groups: cold / warm / shared cache (r488)
```

All inputs are synthesised in the setup step (LCG pixel soup +
procedural scenes); no fixture files.

## Results (r488 group cache)

Apple M5 Max, rustc stable, `--release`, canvas 1206 × 2622. "before" is
commit `5699a65`; the machine was shared with other builds, so every
figure is the median of the quietest of five passes and the unkeyed rows
(whose code path did not change) show the noise floor.

| bench (`group_cache/…`) | before | after |
| ----------------------- | -----: | ----: |
| `canvas_clear_only` (empty scene) | 55 µs | 55–97 µs |
| `glyphs300_unkeyed` | 1.24 ms | 1.2–1.7 ms |
| `glyphs300_keyed_first_frame` | 1 245 ms | 1.12 ms |
| `glyphs300_keyed_steady_same_renderer` | 1 227 ms | 0.26 ms |
| `glyphs300_keyed_steady_shared_cache` (renderer per frame) | n/a | 0.18 ms |
| `glyphs300_keyed_steady_scrolling` | 1 486 ms | 0.18 ms |
| `large4_unkeyed` | 6.5 ms | 6.4 ms |
| `large4_keyed_first_frame` | 24.1 ms | 6.9 ms |
| `large4_keyed_steady_same_renderer` | 2.2 ms | 1.1 ms |
| `large4_keyed_steady_shared_cache` | n/a | 1.3 ms |
| `clipped_groups200_nested` (windowed clip masks) | 19.4 ms | 9.7 ms |

What changed: a cache miss used to zero a canvas-sized buffer (12.6 MB
here), rasterise into it and scan all of it for the touched bounding
box; and it missed almost always, because the key included the
translation and the default capacity (256 entries) was below one screen
of glyphs. Now the bitmap is the size of the group's analytic bounds,
the key excludes the whole-pixel translation, the cache is bounded by
bytes, a hit shares the bitmap instead of copying it, and the blit runs
on row slices. Group opacity and soft masks use the same bounded
offscreen windows.

## Results (r484 quality + performance pass)

Apple Silicon (aarch64-apple-darwin), rustc stable, `--release`.
"before" is commit `69253a5` (start of the round), "after" the end of
the round. Unlike r449 this pass changes output within justified
tolerances (curves flattened to a 1/32 px bound instead of a 0.5 px
chord rule, exact stroke geometry, exact gradient stops) — see the
CHANGELOG for every behaviour change and the regression test pinning
it. Band-parallel rendering is byte-identical to serial.

### `scenes` — realistic workloads (new)

| bench | before | after | speedup |
|---|---|---|---|
| `paths_1080p` (2 400 blobs, 600 strokes — a third dashed — 1 800 glyphs) | 768.7 ms | 166.1 ms | **4.6×** |
| `text_page_1080p` (6 000 glyph outlines) | 91.2 ms | 14.2 ms | **6.4×** |
| `gradient_linear_4k` (3-stop, full canvas) | 54.6 ms | 32.8 ms | 1.7× |
| `gradient_radial_4k` (3-stop reflect, full canvas) | 63.0 ms | 46.7 ms | 1.3× |
| `many_small_4k` (20 000 shapes of 3–10 px) | 731.4 ms | 28.8 ms | **25×** |
| `big_circle_4k` (r = 1000 fill + 24 px stroke) | 27.3 ms | 17.2 ms | 1.6× |
| `gradient_linear_4k_mt4` (4-worker `ExecutionContext`) | — | 8.9 ms | 6.1× vs before |
| `big_circle_4k_mt4` | — | 5.4 ms | 5.0× vs before |
| `paths_1080p_mt4` (shapes too small to band) | — | 165.6 ms | — |

### `render` — 256×256 kernels (2 s measurement)

| bench | before | after | speedup |
|---|---|---|---|
| `flatten_star_128pt` | 131.4 µs | 9.2 µs | 14× (curvature-driven segment count) |
| `fill_star_256_ss1` | 705.7 µs | 142.4 µs | 5.0× (fewer input edges) |
| `fill_star_256_ss4` | 925.6 µs | 281.7 µs | 3.3× |
| `render_scene_256` | 315.4 µs | 149.3 µs | 2.1× |
| `stroke_star_256` | 131.3 ms | 1.93 ms | **68×** |
| `render_glyphlike_400_256` | 1.47 ms | 0.55 ms | 2.7× |
| `render_softmask_256` | 398.9 µs | 247.2 µs | 1.6× |
| `render_gradient_linear_256` | 445.3 µs | 263.9 µs | 1.7× |
| `render_gradient_radial_256` | 523.3 µs | 392.3 µs | 1.3× |
| `render_cached_group_hit_256` | 46.4 µs | 46.3 µs | 1.0× |

Where the time went: a canvas-sized coverage mask per shape (the
small-shape scenes), 0.5 px-chord over-flattening (≈ 50× more edges
than a 1/32 px error bound needs on large curves), polyline vertices
splitting every scan row into sub-strips (now y-monotone chains), the
pivot loops of inner round joins (now trimmed), per-pixel divisions
in source-over (now exact shift / reciprocal forms), and a 256-entry
gradient table lookup per pixel (now row-wise exact evaluation).

## Results (r449 optimisation pass)

Apple Silicon (aarch64-apple-darwin), rustc stable, `--release`.
"before" is the r449 baseline measured at the start of the round;
"after" is the same harness on the optimised kernels. Every
optimisation is **byte-exact** — proven by differential tests
(`tests/fill_reference_equivalence.rs`,
`tests/filter_kernel_equivalence.rs`), the existing pixel-pinning
suites, and a 31-scene whole-frame hash A/B against the pre-round
commit.

**Geometric-mean speedup: 1.79× across all 32 benches.**

### `render` — geometry, scanline fill, scene walks

| bench | before | after | speedup |
|---|---|---|---|
| `flatten_star_128pt` | 137.1 µs | 141.8 µs | 0.97× (untouched code — noise band) |
| `fill_star_256_ss1` | 744.6 µs | 788.7 µs | 0.94× (touched-range bookkeeping; see note) |
| `fill_star_256_ss4` | 998.0 µs | 995.2 µs | 1.00× |
| `render_scene_256` | 3.735 ms | 398.3 µs | **9.4×** |
| `stroke_star_256` | 5.113 ms | 5.314 ms | 0.96× (fill-dominated; noise band) |
| `render_glyphlike_400_256` | 58.90 ms | 1.812 ms | **32.5×** |
| `render_softmask_256` | 950.4 µs | 504.6 µs | 1.88× |
| `render_gradient_linear_256` | 1.108 ms | 589.1 µs | 1.88× |
| `render_gradient_radial_256` | 1.202 ms | 674.1 µs | 1.78× |
| `render_cached_group_hit_256` | 147.6 µs | 51.7 µs | 2.86× |

Note: the two sub-1.0× rows sit in the same ±5 % band the *untouched*
`flatten_star_128pt` drifted across runs; the isolated full-canvas
star fill pays a few percent of touched-range bookkeeping in the
ss = 1 worst case, which the scene-level wins above repay by 1–2
orders of magnitude.

### `sampling` — image resampling + pattern paint (new in r449)

| bench | before | after | speedup |
|---|---|---|---|
| `image_nearest_up_64_to_256` | 734.0 µs | 550.1 µs | 1.33× |
| `image_bilinear_up_64_to_256` | 1.311 ms | 942.5 µs | 1.39× |
| `image_lanczos2_up_64_to_256` | 4.041 ms | 1.659 ms | **2.44×** |
| `image_lanczos3_up_64_to_256` | 6.564 ms | 3.234 ms | 2.03× |
| `image_mitchell_up_64_to_256` | 2.790 ms | 1.655 ms | 1.69× |
| `image_catmull_rom_up_64_to_256` | 2.767 ms | 1.679 ms | 1.65× |
| `image_b_spline_up_64_to_256` | 2.872 ms | 1.676 ms | 1.71× |
| `image_bilinear_down_512_to_256` | 1.494 ms | 964.8 µs | 1.55× |
| `image_lanczos3_down_512_to_256` | 7.787 ms | 3.343 ms | **2.33×** |
| `pattern_fill_256_bilinear` | 1.572 ms | 658.4 µs | 2.39× |
| `pattern_fill_256_nearest` | 829.1 µs | 287.1 µs | **2.89×** |

### `filter` — filter-primitive kernels (r401 harness)

| bench | before | after | speedup |
|---|---|---|---|
| `gaussian_small_kernel_256` | 949.1 µs | 905.4 µs | 1.05× |
| `gaussian_box_256` | 1.745 ms | 1.736 ms | 1.01× |
| `gaussian_box_huge_std_256` | 1.677 ms | 1.683 ms | 1.00× |
| `morphology_dilate_r15_256` | 2.277 ms | 633.5 µs | **3.59×** |
| `color_matrix_saturate_256` | 336.1 µs | 338.3 µs | 0.99× |
| `component_transfer_256` | 547.8 µs | 131.6 µs | **4.16×** |
| `composite_arithmetic_256` | 431.2 µs | 422.2 µs | 1.02× |
| `blend_soft_light_256` | 1.509 ms | 1.473 ms | 1.02× |
| `turbulence_fractal_128` | 1.110 ms | 1.097 ms | 1.01× |
| `graph_chain_128` | 643.4 µs | 475.4 µs | 1.35× |
| `css_parse_apply_64` | 224.9 µs | 214.7 µs | 1.05× |

New r449 baselines for the previously unbenched primitive kernels
(no "before" column — these exist so the next depth round can A/B):

| bench | r449 baseline |
|---|---|
| `convolve_matrix_3x3_256` | 1.131 ms |
| `displacement_map_bilinear_256` | 518.5 µs |
| `diffuse_lighting_distant_256` | 851.6 µs |
| `specular_lighting_point_256` | 2.175 ms |
| `drop_shadow_256` | 2.669 ms |

## What changed (r449)

1. **Scanline fill**: coverage accumulates in one
   `width × supersample` row block reused per destination row instead
   of a full `width × height × supersample` grid, and the rasteriser
   reports the touched-pixel bounding box.
2. **Composite / blit**: mask scans restricted to the touched box;
   fully-opaque source pixels take a plain-overwrite shortcut.
3. **Image sampling**: axis-aligned transforms cache per-column /
   per-row filter taps (positions + weights), eliminating per-pixel
   kernel evaluation (incl. the Lanczos `sin` calls); the general
   path no longer deep-copies the source frame per draw.
4. **Pattern paint**: the same per-axis caching for the wrapped tile
   samplers under axis-aligned transforms.
5. **Soft masks**: coverage conversion + modulated blit restricted to
   the intersection of the content's and mask's alpha bounding boxes.
6. **Filter kernels**: `feComponentTransfer` via per-channel 256-entry
   LUTs; `feMorphology` via the block prefix/suffix extremum
   decomposition (`O(1)` per pixel regardless of radius, hostile radii
   bounded at `O(len)` per line); linearRGB entry/exit via byte→byte
   LUTs; small-`stdDeviation` Gaussian interior pixels skip per-tap
   edge resolution.
