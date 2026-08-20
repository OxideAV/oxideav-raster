# oxideav-raster benchmarks

Criterion benchmarks for the rasterisation kernel hot paths, split
across three harnesses:

```sh
cargo bench --bench render    # flatten / scanline fill / scene walks
cargo bench --bench filter    # filter-primitive kernels (r401)
cargo bench --bench sampling  # image resampling + pattern paint (r449)
```

All inputs are synthesised in the setup step (LCG pixel soup +
procedural scenes); no fixture files.

## Results (r449)

Apple Silicon (aarch64-apple-darwin), rustc stable, `--release`.
"before" is the r449 baseline measured prior to the optimisation
pass; "after" is the same harness on the optimised kernels.

### `render` — geometry, scanline fill, scene walks

| bench | before | after | delta |
|---|---|---|---|
| `flatten_star_128pt` | 137.1 µs | | |
| `fill_star_256_ss1` | 744.6 µs | | |
| `fill_star_256_ss4` | 998.0 µs | | |
| `render_scene_256` | 3.735 ms | | |
| `stroke_star_256` | 5.113 ms | | |
| `render_glyphlike_400_256` | 58.90 ms | | |
| `render_softmask_256` | 950.4 µs | | |
| `render_gradient_linear_256` | 1.108 ms | | |
| `render_gradient_radial_256` | 1.202 ms | | |
| `render_cached_group_hit_256` | 147.6 µs | | |

### `sampling` — image resampling + pattern paint (new in r449)

| bench | before | after | delta |
|---|---|---|---|
| `image_nearest_up_64_to_256` | 734.0 µs | | |
| `image_bilinear_up_64_to_256` | 1.311 ms | | |
| `image_lanczos2_up_64_to_256` | 4.041 ms | | |
| `image_lanczos3_up_64_to_256` | 6.564 ms | | |
| `image_mitchell_up_64_to_256` | 2.790 ms | | |
| `image_catmull_rom_up_64_to_256` | 2.767 ms | | |
| `image_b_spline_up_64_to_256` | 2.872 ms | | |
| `image_bilinear_down_512_to_256` | 1.494 ms | | |
| `image_lanczos3_down_512_to_256` | 7.787 ms | | |
| `pattern_fill_256_bilinear` | 1.572 ms | | |
| `pattern_fill_256_nearest` | 829.1 µs | | |

### `filter` — filter-primitive kernels (r401 harness)

| bench | before | after | delta |
|---|---|---|---|
| `gaussian_small_kernel_256` | 949.1 µs | | |
| `gaussian_box_256` | 1.745 ms | | |
| `gaussian_box_huge_std_256` | 1.677 ms | | |
| `morphology_dilate_r15_256` | 2.277 ms | | |
| `color_matrix_saturate_256` | 336.1 µs | | |
| `component_transfer_256` | 547.8 µs | | |
| `composite_arithmetic_256` | 431.2 µs | | |
| `blend_soft_light_256` | 1.509 ms | | |
| `turbulence_fractal_128` | 1.110 ms | | |
| `graph_chain_128` | 643.4 µs | | |
| `css_parse_apply_64` | 224.9 µs | | |
