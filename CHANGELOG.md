# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `feImage` filter primitive (SVG 1.1 §15.18) — places an
  already-decoded straight-alpha RGBA raster into the filter chain,
  fitted to the filter-primitive subregion with the full SVG 1.1 §7.8
  `preserveAspectRatio` algebra. This completes all 16 `<feXxx>`
  filter primitives of SVG 1.1 Chapter 15.
  - `image_source(src, sw, sh, ow, oh, par, sampling) -> Vec<u8>` plus
    the typed-pixel wrapper `image_source_pixels(...) -> Vec<Rgba>`.
    New types: `PreserveAspectRatio { align, meet_or_slice }` (with
    `Default` = `xMidYMid meet` per the §15.18 attribute table),
    `AspectRatioAlign` (all ten §7.8 `<align>` values),
    `MeetOrSlice::{Meet, Slice}`, and
    `ImageSourceSampling::{Nearest, Bilinear}` (the bilinear route
    follows the §15.18 "high quality viewers make use of appropriate
    interpolation techniques, for example bilinear or bicubic" note).
  - Fitting per §7.8: `align = none` scales each axis independently
    so the source exactly fills the target; the nine uniform
    alignments force a single scale factor — per-axis minimum under
    `meet`, maximum under `slice` — then anchor the scaled source at
    the min / midpoint / max of each axis. Target pixels the fitted
    source does not cover emit the §15.7.3 transparent-black value.
  - 12 unit tests (default attribute value, identity copy under both
    samplers, non-uniform `none` stretch, matching-aspect upscale,
    centred meet bands, min/max anchors, centred slice crop, bilinear
    interior weights, degenerate zero-extent source and target, panic
    guard, typed/byte-path parity) + 7 integration tests
    (`tests/filter_image.rs`: realistic-extent meet/slice geometry,
    all anchor placements, ×8 nearest block upscale, bilinear
    midpoint arithmetic, `feMerge` chain composition, typed parity).
- `feSpecularLighting` filter primitive (SVG 1.1 §15.22) — Phong-specular
  reflectance computed from the same §15.14 surface-normal Sobel kernel
  and §15.8 light-source descriptor as `feDiffuseLighting`.
  - `specular_lighting(src, width, height, &params) -> Vec<u8>` plus
    the typed-pixel wrapper
    `specular_lighting_pixels(...) -> Vec<Rgba>`. Parameters bundled in
    `SpecularLighting { surface_scale, specular_constant,
    specular_exponent, kernel_unit_length, light_color, light_source }`
    with a `Default` impl encoding the §15.22 attribute defaults
    (`surface_scale = 1`, `specular_constant = 1`,
    `specular_exponent = 1`, `kernel_unit_length = (1, 1)`,
    `light_color = white`, distant light at
    `azimuth = 0` / `elevation = 0`).
  - Algorithm follows §15.22 verbatim: form
    `H = (L + E) / Norm(L + E)` with the constant eye vector
    `E = (0, 0, 1)`, then emit
    `Sc = ks · pow(max(N·H, 0), specularExponent) · L_color` per RGB
    channel with `Sa = max(Sr, Sg, Sb)` (the §15.22 "non-opaque
    image" property distinguishing specular from diffuse, where
    diffuse sets `Da = 1.0` everywhere). The `max(N·H, 0)` clamp
    handles the §15.22 degenerate case `L = −E` and surfaces facing
    away from the halfway direction.
  - 15 unit tests covering: overhead-distant ⇒ full-white + full
    alpha; grazing distant ⇒ half-intensity; zero
    `specular_constant` ⇒ transparent black; coloured light ⇒
    `Sa = max-channel` (not 255); position-invariance under distant
    lights; back-of-surface light ⇒ zero; exponent monotonicity at a
    grazing angle; point-light directly overhead; spot-light
    outside the §15.8.4 limiting cone ⇒ zero; default-attribute
    smoke test; three negative-input panic paths; typed-pixel-
    wrapper equivalence with the byte-buffer path.

- `feDiffuseLighting` filter primitive (SVG 1.1 §15.14) — Phong-diffuse
  lighting of an image's alpha channel treated as a bump-map height
  field `Z(x, y) = surfaceScale · I(x, y)`. Supports all three §15.8
  light-source kinds.
  - `diffuse_lighting(src, width, height, &params) -> Vec<u8>` plus
    the typed-pixel wrapper `diffuse_lighting_pixels(...) -> Vec<Rgba>`.
    Parameters are bundled in `DiffuseLighting { surface_scale,
    diffuse_constant, kernel_unit_length, light_color, light_source }`
    with a `Default` impl encoding the §15.14 / §15.8.2 attribute
    defaults (`surface_scale = 1`, `diffuse_constant = 1`,
    `kernel_unit_length = (1, 1)`, `light_color = white`, distant
    light at `azimuth = 0`, `elevation = 0`).
  - `LightSource` enum mirrors the three SVG 1.1 light-source
    elements: `Distant { azimuth_deg, elevation_deg }` for §15.8.2,
    `Point { x, y, z }` for §15.8.3, and `Spot { x, y, z,
    points_at_x, points_at_y, points_at_z, specular_exponent,
    limiting_cone_angle_deg }` for §15.8.4 — the cone-angle field is
    `Option<f32>` matching §15.8.4's "if no value is supplied, there
    is no limiting cone".
  - Surface-normal computation follows §15.14's nine-region Sobel
    table verbatim: one kernel variant per corner / edge / interior
    cell, with the §15.14 `FACTORx / FACTORy` normalising constants
    tracked per region (`2/(3·d)`, `1/(2·d)`, `1/(3·d)`, `1/(4·d)`).
    The kernel returns `N = (Nx, Ny, 1) / Norm(Nx, Ny, 1)` per the
    §15.14 unit-vector formula.
  - Light-direction `L` is computed per the §15.14 closed forms: a
    constant `(cos(az)·cos(el), sin(az)·cos(el), sin(el))` for
    distant sources; `(light_pos − sample_pos) / Norm(…)` for point
    and spot sources where `Z(x, y) = surfaceScale · I(x, y)` is the
    sample's height-field value. Spot sources additionally scale the
    inherited `lighting-color` by `pow(−L·S, specularExponent)` per
    §15.14, where `S` is the unit vector from the spot source toward
    its `pointsAt` target; the §15.14 cut-offs (`−L·S ≤ 0` ⇒ no
    light; `−L·S < cos(limitingConeAngle)` ⇒ no light) zero the
    emission outside the cone.
  - Output formula `D = kd · max(N · L, 0) · L_color` reproduces
    §15.14's `D = kd · (N · L) · Light` with the standard Phong
    `max(·, 0)` clamp implied by "the standard diffuse component of
    the Phong lighting model"; `Da = 1.0` everywhere per the §15.14
    "alpha = 1.0 everywhere" statement.
  - `kernel_unit_length.0 <= 0` and `kernel_unit_length.1 <= 0` panic
    per §15.14's "A negative or zero value is an error".
  - 22 new tests across `src/filter.rs::diffuse_lighting_tests` and
    `tests/filter_diffuse_lighting.rs` covering: flat surface
    overhead-distant produces the light colour; flat surface grazing
    distant emits opaque black; `diffuse_constant = 0` collapses
    every pixel to opaque black; distant light is position-invariant
    on a flat surface and quantitatively matches `light_color ·
    sin(elevation)`; a step-function bump-map lights its facing side
    only; point light is brightest directly under the source on a
    flat plate; spot light with 5° cone illuminates the on-axis
    centre and dark-clamps the corners; `specular_exponent`
    concentrates emission toward the axis; no-cone spot illuminates
    the full hemisphere; default parameters match the §15.14 /
    §15.8.2 attribute defaults; `light_color` and `diffuse_constant`
    scale the output linearly; typed-pixel wrapper byte-equivalence;
    `Da = 1.0` invariant across all three light-source kinds; empty
    extent short-circuit; length-mismatch panic; `kernel_unit_length`
    zero / negative panics.
- `feBlend` filter primitive (SVG 1.1 §15.9) — per-pixel combination of
  two equally-sized inputs through one of five spec-listed modes.
  - `blend_filter(in1, in2, width, height, mode) -> Vec<u8>` plus the
    typed-pixel wrapper `blend_filter_pixels(...) -> Vec<Rgba>`. The
    `mode` argument takes a `BlendFilterMode` (`Normal | Multiply |
    Screen | Darken | Lighten`); the §15.9 attribute table lists
    `Normal` as the default (`BlendFilterMode::default()` returns it).
  - Result alpha follows the §15.9 mode-independent rule
    `qr = 1 − (1 − qa) · (1 − qb)` — evaluated once per pixel and
    shared across the three colour channels.
  - The five per-mode colour formulas operate on premultiplied RGB,
    matching the §15.9 statement that the `ca` / `cb` quantities are
    "premultiplied" in the formula table:
    `normal: cr = (1 − qa) · cb + ca`;
    `multiply: cr = (1 − qa) · cb + (1 − qb) · ca + ca · cb`;
    `screen: cr = cb + ca − ca · cb`;
    `darken: cr = Min((1 − qa) · cb + ca, (1 − qb) · ca + cb)`;
    `lighten: cr = Max((1 − qa) · cb + ca, (1 − qb) · ca + cb)`.
    The straight-alpha `u8` inputs are converted to premultiplied
    `[0, 1]` floats at the buffer boundary and the premultiplied
    result is converted back to straight-alpha bytes for output —
    the convention used elsewhere in the filter module
    (`composite_filter`, `merge`).
  - When `qr <= 0` (both operands fully transparent) the §15.7.3
    transparent-black value `(0, 0, 0, 0)` is emitted directly to
    avoid the undefined `0 / 0` un-premultiply division. The
    `cr / qr` colour division is otherwise clamped to `[0, 1]`
    before quantisation to absorb floating-point drift on the
    `screen` and `lighten` paths.
  - Per the §15.9 note, `normal` mode is the pixel-wise equivalent
    of `feComposite` operator `"over"`; the integration suite
    cross-checks the two routines on opaque pairs and confirms
    bit-exact agreement.
  - 22 new tests across `src/filter.rs::blend_filter_tests` and
    `tests/filter_blend.rs` covering: default mode is `Normal`;
    transparent-top yields bottom under `Normal`; opaque-top yields
    top under `Normal`; `Normal` opaque-pair matches `composite_filter`
    `Over`; `Multiply` opaque-pair is the component-wise product;
    `Screen` opaque-pair follows the `ca + cb − ca·cb` formula;
    `Darken` / `Lighten` opaque-pair reduce to component-wise min /
    max; result-alpha invariance across all five modes; typed-pixel
    wrapper byte-equivalence; full-transparent collapse to
    transparent black across modes; modes diverge on a general
    mixed-alpha pair (no enum-arm aliasing); empty-extent
    short-circuit; length-mismatch panics on `in1`, `in2`.
- `feDisplacementMap` filter primitive (SVG 1.1 §15.15) — per-pixel
  channel-driven warp of `in1` by the §15.15 algorithm
  `P'(x, y) = P(x + scale · (XC(x, y) − 0.5),
                y + scale · (YC(x, y) − 0.5))`,
  where `XC` / `YC` are the `[0, 1]` channel values of `in2` selected
  by `xChannelSelector` / `yChannelSelector`.
  - `displacement_map(in1, in2, width, height, scale, x_channel,
    y_channel, sampling) -> Vec<u8>` plus the typed-pixel wrapper
    `displacement_map_pixels(...) -> Vec<Rgba>`. The `XC` / `YC`
    arguments take a `DisplacementChannel` (`R | G | B | A`); both
    attributes default to `A` per the §15.15 attribute table
    (`DisplacementChannel::default()` returns `A`).
  - The §15.15 paragraph "the calculations using the pixel values
    from `in2` are performed using non-premultiplied color values"
    matches the straight-alpha convention used throughout the
    filter module, so the selected channel byte divided by `255`
    is `XC` / `YC` directly without an explicit un-premultiply.
  - Two sample-reconstruction policies are wired in
    (`DisplacementSampling`): `Nearest` rounds the warped source
    coordinate to the nearest integer and copies the source byte
    triple verbatim (round-half-away-from-zero, matching the
    `OffsetSampling::Nearest` convention so the two integer-rounding
    policies stay consistent across the filter module); `Bilinear`
    resamples the four-pixel neighbourhood around the fractional
    source coordinate, the §15.15 "high quality viewers apply an
    interpolent on the surrounding pixels, for example bilinear or
    bicubic" route. Out-of-bounds source positions and out-of-bounds
    members of the bilinear footprint emit the §15.7.3
    transparent-black value `(0, 0, 0, 0)`, naturally fading the
    warped image at the displacement-induced edges. The bilinear
    path coincides bit-exactly with the nearest path when the
    per-pixel warped coordinate is integer-valued.
  - `scale = 0` reduces the operation to a copy of `in1`
    (§15.15 attribute table: "When the value of this attribute is 0,
    this operation has no effect on the source image").
    Negative `scale` is permitted — the formula is linear in `scale`
    so a sign flip inverts the per-axis shift direction, which is a
    common authoring shortcut for the §15.15 example pipelines.
  - 24 new tests across `src/filter.rs::displacement_map_tests` and
    `tests/filter_displacement_map.rs` covering: `scale = 0` identity
    under both sampling policies; half-grey (XC = YC ≈ 0.5) shift
    rounds to zero under nearest; XC = 1.0 / scale = 10 pulls source
    column `x + 5`; XC = 0.0 / scale = 10 pulls column `x − 5`; the
    §15.7.3 OOB → transparent-black rule for partial and full extent
    misses; `DisplacementChannel::default()` reads the alpha channel;
    `x_channel` and `y_channel` are independently wired (swapping
    them produces a different warp); negative `scale` inverts the
    direction; uniform-channel warps respect the analytic per-axis
    shift formula; the warp is purely local — no synthesised
    colours; integer-shift bilinear-equals-nearest invariant on a
    multi-row source; fractional-shift bilinear blends adjacent
    pixels with the correct interpolation weights; typed-pixel
    wrapper byte-equivalence under both sampling policies and an
    arbitrary 6×6 map; empty-extent (`0×N` / `N×0`) short-circuit to
    an empty `Vec`; length-mismatch panics on `in1`, `in2`;
    NaN-`scale` panic guard.
- `feTile` filter primitive (SVG 1.1 §15.23) — periodic replication of
  a reference tile across a target rectangle.
  - `tile(src, src_width, src_height, out_width, out_height) -> Vec<u8>`
    plus the typed-pixel wrapper `tile_pixels(...) -> Vec<Rgba>`. The
    §15.23 statement "i and j can be any integer value" reduces, per
    output pixel `(ox, oy)`, to the Euclidean remainder lookup
    `(ox mod src_width, oy mod src_height)` into the source — the
    analytical form of the integer-shift family applied at every
    output sample. Source and target rectangles are aligned at the
    origin (the i = 0, j = 0 copy of the tile lands with its top-left
    at the target top-left); callers that need a different alignment
    shift the input through `offset` before tiling.
  - 15 new tests across `src/filter.rs::tile_tests` and
    `tests/filter_tile.rs` covering: identity when extents match;
    4×4 from 2×2 verbatim replication; 3×2 partial-period truncation
    on the trailing column; 1×1 constant-fill degenerate case; 2×2
    crop from a 4×4 reference; zero-extent target shortcut; 9×9
    triple replication of a 3×3 reference; typed-wrapper round-trip
    parity; wrong-source-length panic; empty-source-with-non-empty-
    target panic; empty-source-with-empty-target shortcut; integration
    tests for 64×64 tiling of an 8×8 gradient, 20×13 from a 7×5
    non-multiple source, alpha preservation, 1×4 strip extension, and
    the "shift-then-tile is tile" invariant.
- `feFlood` / `feOffset` / `feMerge` filter primitives (SVG 1.1
  §15.16 / §15.21 / §15.19) — the three "drop-shadow building
  blocks" that compose into the §15.2 example pipeline.
  - `flood(width, height, r, g, b, flood_opacity) -> Vec<u8>` plus
    the typed-pixel wrapper `flood_pixels(...) -> Vec<Rgba>`. Emits
    a `width × height` packed-RGBA buffer in which every pixel
    carries the resolved `flood-color` triple modulated by
    `flood-opacity` (straight-alpha form, with `flood_opacity`
    clamped to `[0, 1]` and quantised half-up to `u8`).
  - `offset(src, width, height, dx, dy, sampling) -> Vec<u8>` plus
    the typed-pixel wrapper `offset_pixels(...) -> Vec<Rgba>`.
    Translates the packed-RGBA source by `(dx, dy)`; output pixels
    whose source position falls outside the input extent are set to
    transparent black per the §15.7.3 "undefined pixels are set to
    transparent black" rule. Two `OffsetSampling` policies are
    wired in: `Nearest` (rounds `(dx, dy)` to the nearest integer
    via `f32::round` and copies source pixels verbatim — the
    default, exact for the common case of integer shifts produced
    by user-space `dx` / `dy` attributes on an untransformed
    primitive) and `Bilinear` (resamples the four-pixel
    neighbourhood for fractional shifts — the §15.21 "high quality
    viewer should make use of appropriate interpolation techniques,
    for example bilinear or bicubic" route; out-of-bounds samples
    contribute transparent black, naturally fading the offset
    image at the edges). The bilinear path coincides exactly with
    the nearest path for integer `(dx, dy)`.
  - `merge(width, height, &[&[u8]]) -> Vec<u8>` plus the
    typed-pixel wrapper `merge_pixels(width, height, &[&[Rgba]])
    -> Vec<Rgba>`. Composites N input layers bottom-to-top using
    the §14.2 simple-alpha-compositing `over` algebra
    `αo = αs + αd · (1 − αs)`,
    `co = (cs · αs + cd · αd · (1 − αs)) / αo`. With zero layers
    the result is transparent black (the filter-effects-region
    initial value per §15.1). With a single layer the result is
    that layer verbatim. With N ≥ 2 layers the canonical n − 1
    `feComposite` reduction is recovered transparently, matching
    `((L0 over L1) … over LN)` composition.
  - 31 new tests across `src/filter.rs::flood_offset_merge_tests`
    and `tests/filter_flood_offset_merge.rs` covering: flood
    extent / opacity quantisation / range clamping / NaN-panic
    guard / empty-extent shortcut; offset integer shift in both
    directions / undefined-pixel transparent-black rule / zero
    identity / nearest rounding / bilinear half-pixel split /
    bilinear-integer-equals-nearest identity / NaN-panic guard /
    length mismatch panic; merge zero-layer transparent-black
    canvas / single-layer identity / opaque-over-anything /
    transparent-over-anything / half-alpha 50/50 average /
    three-layer associativity / typed wrapper round-trip /
    extent mismatch panic. The end-to-end §15.2 drop-shadow
    pipeline shape (`SourceAlpha → feGaussianBlur → feOffset →
    feMerge(shadow, source)`) is exercised, including the
    `feFlood + feComposite-In` shadow-tint variant.
- `feTurbulence` filter primitive (SVG 1.1 §15.24) — Perlin-noise
  source primitive, exposed as
  `pub fn turbulence_filter(width, height, &params) -> Vec<u8>` plus
  the typed-pixel wrapper `turbulence_filter_pixels(width, height,
  &params) -> Vec<Rgba>`. Implements the §15.24 algorithm verbatim:
  the Park–Miller minimum-standard LCG (`a = 16807`, `m = 2^31 − 1`)
  seeds a permutation table and four channel-specific gradient
  tables; `noise2(x, y)` applies a `t² · (3 − 2t)` smoothstep to a
  bilinear interpolation between four surrounding lattice gradients;
  the `turbulence` accumulator sums `num_octaves` `noise2` terms with
  geometrically-doubling frequency and halving amplitude. Both
  `type` arms are wired in: `Turbulence` (`SUM |noise2|`, aim
  `[0, 1]`, quantised as `value · 255`) and `FractalNoise`
  (`SUM noise2`, aim `[-1, 1]`, quantised as `(value · 255 + 255) /
  2`). The `Turbulence` parameter block exposes the full §15.24
  attribute set: `base_frequency_x` / `base_frequency_y`,
  `num_octaves`, `seed`, `kind` (`TurbulenceType`), `stitch_tiles`
  (`StitchTiles`), and the `(tile_x, tile_y, tile_width,
  tile_height)` filter-primitive subregion. `Turbulence::new(fx, fy)`
  applies the §15.24 defaults (`num_octaves = 1`, `seed = 0`,
  `kind = Turbulence`, `stitch_tiles = NoStitch`, zero-extent tile);
  builder-style `with_num_octaves` / `with_seed` / `with_kind` /
  `with_stitch_tiles` / `with_tile_region` set individual attributes.
  `setupSeed` from §15.24 — `if (lSeed <= 0) lSeed = -(lSeed %
  (RAND_M − 1)) + 1; if (lSeed > RAND_M − 1) lSeed = RAND_M − 1;` —
  is implemented as `turb_setup_seed` and clamps non-positive and
  out-of-range inputs back into the positive `[1, RAND_M − 1]`
  window. The `stitchTiles="stitch"` path rounds the base
  frequencies to the nearer of `floor(width · frequency) / width`
  vs. `ceil(width · frequency) / width` (and the same for `height`)
  so the tile rectangle contains an integral number of first-octave
  Perlin tiles, and wraps the lattice indices at the right and
  bottom edges of the active area so the output tiles seamlessly.
  Each octave doubles `stitchWidth` / `stitchHeight` so the
  higher-octave lattices wrap on their own period. Output channels
  are clamped to `[0, 255]` and re-quantised half-up. Complexity is
  `O(W · H · num_octaves)`. 19 new tests in the `src/filter.rs`
  `turbulence_tests` suite, including the §15.24 self-check
  (`turb_setup_seed(1)` followed by 10000 LCG iterations yields
  `1043618065`), the `setupSeed` clamp for positive, zero, and
  negative inputs, deterministic byte-for-byte output for repeated
  invocations at the same seed, divergent output for different
  seeds, divergent output across the four channels (independent
  gradient tables), `turbulence` aim into `[0, 255]` (per-channel
  mean strictly inside `(1, 254)`), `fractalNoise` aim around the
  signed centre 128 (per-channel mean within ±48 of 128 over a
  48×48 grid), zero base-frequency produces a per-channel constant,
  the typed-pixel wrapper agrees with the byte API, empty extent
  returns an empty buffer, output length matches `width · height ·
  4`, `num_octaves = 1` differs from `num_octaves > 1`, the
  `stitchTiles="stitch"` path produces seam-continuous output across
  the tile boundary, and 3 `#[should_panic]` guards for non-positive
  base-frequency components and `num_octaves = 0`. Math cited to
  `docs/image/svg/svg11-second-edition.pdf` §15.24.

- `feConvolveMatrix` filter primitive (SVG 1.1 §15.13) — general 2-D
  matrix convolution over a packed-RGBA buffer, exposed as
  `pub fn convolve_matrix(src, width, height, &cm) -> Vec<u8>` plus the
  typed-pixel wrapper `convolve_matrix_pixels(&[Rgba], …)`. Implements
  the §15.13 formula verbatim:
  `COLOR_{X,Y} = (SUM_I SUM_J SOURCE_{X−targetX+J, Y−targetY+I} ·
  kernelMatrix[orderX−J−1, orderY−I−1]) / divisor + bias`,
  including the spec-mandated 180° kernel rotation ("the values in
  the kernel matrix are applied such that the kernel matrix is rotated
  180 degrees relative to the source and destination images"). The
  `ConvolveMatrix` parameter block exposes the full spec attribute set:
  `order_x` × `order_y` kernel dimensions, `kernel` row-major weights,
  `divisor`, `bias`, `target_x` / `target_y` anchor positioning,
  `edge_mode`, and `preserve_alpha`. `ConvolveMatrix::new(ox, oy, k)`
  applies the §15.13 defaults — `divisor = sum(k)` (falling back to
  `1.0` when the sum is zero), `bias = 0`, `target = (floor(ox/2),
  floor(oy/2))`, `edge_mode = Duplicate`, `preserve_alpha = false`.
  Builder-style setters `with_bias` / `with_divisor` / `with_target` /
  `with_edge_mode` / `with_preserve_alpha` apply the corresponding spec
  attributes. All three §15.13 edge-mode policies are implemented in
  the new `ConvolveEdgeMode` enum: `Duplicate` clamps out-of-bounds
  reads to the nearest in-range coordinate; `Wrap` uses toroidal
  addressing (`x.rem_euclid(width)` / `y.rem_euclid(height)`); `None`
  returns `(0, 0, 0, 0)` for out-of-bounds samples. `preserve_alpha =
  true` (§15.13 wording: "the filter will temporarily unpremultiply the
  color component values, apply the kernel, and then re-premultiply at
  the end") convolves only the RGB channels of the straight-alpha
  source — alpha is taken from the un-convolved target pixel
  (`ALPHA_{X,Y} = SOURCE_{X,Y}` per the spec's preserveAlpha=true
  branch). Output channels are clamped to `[0, 255]` and re-quantised
  half-up. Complexity is `O(W · H · orderX · orderY)` — `feConvolveMatrix`
  is the catch-all arbitrary-2-D-kernel primitive; the existing
  separable Gaussian path (`feGaussianBlur`) stays the right choice
  for that common case. 17 new tests: 16 unit tests in the `src/filter.rs`
  `convolve_matrix_tests` suite (identity 3×3 kernel byte-for-byte
  round-trip on a pseudo-random pattern; box-blur 3×3 solid-image
  invariance across all three edge modes; the spec's §15.13 worked
  example reproduces the spec-listed `(9·0 + 8·20 + 7·40 + 6·100 +
  5·120 + 4·140 + 3·200 + 2·220 + 1·240) / 45 = 3480 / 45 = 77.333…`
  to a byte (77) at the (1, 1) target pixel, demonstrating the 180°
  kernel rotation; `preserve_alpha = true` leaves the
  source alpha byte-identical while the RGB channels blur;
  `edge_mode = "none"` zeroes out-of-bounds reads exactly; 1×1 image
  under any kernel + `Duplicate` is identity; `Wrap` smoke test on a
  solid 2×2; `bias = 0.25` shifts every channel by `+64`; `bias =
  -1.0` clamps every output to 0; the Sobel-X kernel produces a
  byte-exact-zero high-pass response on a constant image; the
  §15.13 180° kernel-rotation property is asserted by an asymmetric
  delta-impulse fixture; the typed-pixel wrapper agrees with the
  byte API on an asymmetric kernel + bias + wrap edge mode; empty
  image returns empty buffer; non-square 5×1 kernel only blurs the X
  axis; non-default `target` shifts the kernel anchor; 5 `#[should_panic]`
  guards for wrong input length, zero divisor, wrong kernel length,
  zero order, and out-of-range target) plus 1 integration test in
  `tests/filter_convolve_matrix.rs` exercising the public re-exports.
  Math cited to `docs/image/svg/svg11-second-edition.pdf` §15.13; no
  `image` / `imageproc` / `opencv` / `cairo` / `skia` / `resvg` /
  `librsvg` source consulted.

## [0.1.2](https://github.com/OxideAV/oxideav-raster/compare/v0.1.1...v0.1.2) - 2026-05-29

### Other

- feComposite Porter-Duff + arithmetic (SVG 1.1 §15.12)
- feComponentTransfer 5-mode per-channel transfer function (SVG 1.1 §15.11)
- feGaussianBlur separable Gaussian blur (SVG 1.1 §15.17)
- feColorMatrix (SVG 1.1 §15.10) — matrix / saturate / hueRotate / luminanceToAlpha
- feMorphology erode/dilate filter primitive (SVG 1.1 §15.20)
- Catmull–Rom round-trip + PSNR regression coverage
- add ImageFilter::Lanczos3 (6×6 windowed-sinc image resampling)
- add ImageFilter::BSpline cubic B-spline image resampling
- Catmull–Rom bicubic image filter (B=0, C=1/2)
- pre-baked 256-entry stops LUT for linear / radial paints
- 4 non-separable HSL blend modes (PDF §11.3.5.3 / W3C §11)
- 11 PDF §11.3.5 standard separable blend modes
- Mitchell–Netravali bicubic image filter + linear-RGB gradient interpolation

### Added

- `feComposite` filter primitive (SVG 1.1 §15.12) — pixel-wise
  combination of two equal-sized packed-RGBA buffers, exposed as
  `pub fn composite_filter(in1, in2, width, height, op) -> Vec<u8>` plus
  the typed-pixel wrapper `composite_filter_pixels(&[Rgba], &[Rgba], …)`,
  selected by the new `CompositeOp` enum. `in1` maps to the spec's `in`
  / `i1` operand and `in2` to `in2` / `i2`. The compositing algebra runs
  in the premultiplied-alpha domain fixed by §14.2 ("all color values
  use premultiplied alpha"): each straight-alpha input is converted to
  premultiplied `[0, 1]`, the operator is evaluated, then the result is
  un-premultiplied back to straight-alpha `u8`.
  - The five Porter–Duff operators (`Over` / `In` / `Out` / `Atop` /
    `Xor`, referenced by §15.12) are expressed through the standard
    blend-factor pair `(Fa, Fb)`: `co = ca·Fa + cb·Fb`,
    `αo = αa·Fa + αb·Fb`, derived from the §14.2 simple-alpha formula
    `Cr' = (1 − Ea)·Cr + Er` already used by `composite_rgba_premultiplied`.
    Factors: `over` `(1, 1−αa)`; `in` `(αb, 0)`; `out` `(1−αb, 0)`;
    `atop` `(αb, 1−αa)`; `xor` `(1−αb, 1−αa)`.
  - The `Arithmetic { k1, k2, k3, k4 }` operator evaluates the §15.12
    per-channel formula `result = k1·i1·i2 + k2·i1 + k3·i2 + k4` on the
    premultiplied channels, clamping each result channel to `[0, 1]`.
  - Re-quantisation uses the same half-up `round(x · 255)` rule
    (`quantise_unit`) as the rest of the filter module; a fully
    transparent result normalises to `(0, 0, 0, 0)`.
- `feComponentTransfer` filter primitive (SVG 1.1 §15.11) — per-pixel,
  per-channel transfer function over a packed-RGBA buffer, exposed as
  `pub fn component_transfer(src, width, height, &ct) -> Vec<u8>` plus
  the typed-pixel wrapper `component_transfer_pixels(&[Rgba], …)`. The
  §15.11 spec defines five `type=` modes per channel, applied
  independently to R / G / B / A on un-premultiplied channel values
  normalised to `[0, 1]`, with the result clamped to `[0, 1]` before
  re-quantising back to `u8`:
  - `identity` — `C' = C` (the no-op default).
  - `table` — piecewise-linear interpolation across `N + 1`
    `tableValues`. For `C ∈ [0, 1]` let `s = C · N`, `k = floor(s)`,
    `f = s - k`; then `C' = v_k + f · (v_{k+1} - v_k)`. At the `C = 1`
    boundary the interpolation pins to `v_N`. An empty
    `tableValues` (or a list of fewer than 2 entries) collapses to
    `identity` per §15.11's fall-back wording.
  - `discrete` — step function across `N` `tableValues`. For
    `C ∈ [0, 1]` let `k = floor(C · N)` clamped to `[0, N-1]`; then
    `C' = v_k`. An empty list collapses to `identity`.
  - `linear` — affine transform `C' = slope · C + intercept`.
  - `gamma` — `C' = amplitude · C^exponent + offset`. `exponent` must
    be strictly positive (§15.11 error processing); a non-positive or
    NaN exponent panics. Negative channel inputs are floored to zero
    before the `powf` to keep the spec's "evaluate then clamp"
    ordering well-defined.
  Public surface: `pub enum TransferFunc { Identity, Table(Vec<f32>),
  Discrete(Vec<f32>), Linear { slope, intercept }, Gamma { amplitude,
  exponent, offset } }` with an inherent `apply(c) -> f32` helper, plus
  `pub struct ComponentTransfer { r, g, b, a }` of four
  `TransferFunc` values with an `::identity()` constructor and
  `with_r` / `with_g` / `with_b` / `with_a` builder-style setters.
  `ComponentTransfer::default()` is the identity. All-identity hits a
  byte-copy fast path that skips the f32 round-trip entirely. 15 new
  tests in the `src/filter.rs` unit-suite (all-identity fast-path
  passthrough; `Linear { slope: -1, intercept: 1 }` inverts R only;
  `Gamma { exp: 2.2 }` ∘ `Gamma { exp: 1/2.2 }` round-trips within
  ±1 LSB across an 8×8 sampler; `Discrete(vec![0, 0.5, 1])` step
  boundaries at `1/3` and `2/3` of the unit interval;
  `Table(vec![0, 1])` reduces to identity; `Table(vec![0, 0.5])`
  halves the value; empty image returns empty buffer; solid image
  under identity is invariant; `Linear` on alpha leaves RGB
  byte-exact; `Linear { slope: 0, intercept: 1 }` saturates to 255;
  typed-pixel wrapper agrees with byte API across a mixed-mode
  configuration; empty `Table` / `Discrete` fall back to identity;
  `#[should_panic]` guards for wrong input length, zero-exponent
  gamma, and negative-exponent gamma) plus 8 in
  `tests/filter_component_transfer.rs` (public-API all-identity
  byte passthrough; `Linear` invert on R only; `Table(vec![0, 1])`
  identity on G; `Discrete(vec![0, 1])` threshold at `0.5`; gamma
  2.2 round-trip via inverse on R/G/B; pixels-wrapper / byte-API
  agreement on a mixed-mode configuration; `default()` ==
  `identity()`; `Linear { 0, 1 }` saturates every byte to 255;
  `Linear { 1, -1 }` clamps every byte to 0). Math cited to
  `docs/image/svg/svg11-second-edition.pdf` §15.11; no `image` /
  `imageproc` / `opencv` / `cairo` / `skia` / `resvg` / `librsvg`
  source consulted.

- `feGaussianBlur` filter primitive (SVG 1.1 §15.17) — separable
  Gaussian blur over a packed-RGBA buffer, exposed as
  `pub fn gaussian_blur(src, width, height, std_x, std_y) -> Vec<u8>`
  plus the typed-pixel wrapper
  `gaussian_blur_pixels(&[Rgba], ...)`. The spec defines the kernel as
  the normalised separable Gaussian `G(x, y) = H(x) · I(y)` with
  `H(x) = exp(-x²/(2s²)) / sqrt(2π·s²)` and `I(y) = exp(-y²/(2t²)) /
  sqrt(2π·t²)`, and explicitly permits a "separable convolution"
  implementation. Two branches mirror the two performance regimes
  the spec calls out:
  - **Direct discrete kernel** for `std_x < GAUSSIAN_BLUR_BOX_THRESHOLD`
    (= `2.0`): build the 1-D Gaussian on `[-ceil(3·s) .. ceil(3·s)]`
    (capturing ≈99.7% of the analytical mass), renormalise so the
    discrete row sum equals exactly `1.0` in `f32` (removes the DC
    drift that would otherwise creep in from truncating the tails),
    then run one X-pass over rows and one Y-pass over columns with
    clamp-to-edge boundary handling.
  - **Three-box-blur approximation** for `s >= 2.0`: the spec's
    `d = floor(s · 3·sqrt(2π)/4 + 0.5)` formula picks the box size;
    if `d` is odd we compose three centred box-blurs of size `d`, if
    `d` is even we compose two box-blurs of size `d` (centred half a
    pixel to the left and to the right respectively, per the spec
    paragraph) with one centred box-blur of size `d + 1`. Each box
    pass is an O(W·H) rolling-sum per channel.
  Both branches are pure separable (X-pass then Y-pass) so the public
  `(std_x, std_y)` argument pair selects per-axis blur independently;
  zero on either axis disables the effect on that axis; zero on both
  returns the input unchanged; negative or NaN panics per the spec's
  error-processing rules. Mass conservation is exact at the
  analytical level and bounded-loss at `u8` quantisation (far-tail
  samples round to zero). 21 new tests: 12 in the `src/filter.rs`
  unit suite (zero-stddev identity; solid-image invariance across
  nine `(sx, sy)` configurations spanning both branches; impulse-
  response symmetry / monotonicity / centre-dominance on a small
  canvas for the direct branch; impulse-response axis-aligned
  monotonicity on a 21×21 canvas for the box branch; separability
  identity `blur(sx, sy) ≡ blur(sx, 0) ∘ blur(0, sy)` on
  pseudo-random content for both branches; axis-only blur preserves
  the orthogonal axis exactly; typed-pixel wrapper agreement with
  the byte API across both branches; empty image returns empty
  buffer; `box_sizes_for_std` table reproduces the spec formula at
  `s = 2.0 / 3.0 / 4.0 / 5.0`; kernel normalisation sum, centre
  dominance, and symmetry for the direct branch; `#[should_panic]`
  guards for negative `std_x` and wrong input length) plus 9 in
  `tests/filter_gaussian_blur.rs` (public-API zero-stddev identity;
  `GAUSSIAN_BLUR_BOX_THRESHOLD` constant value; solid-image
  byte-exact invariance across nine `(sx, sy)` configurations;
  SourceAlpha-style input keeps RGB at zero; impulse-response
  four-fold mirror symmetry at four standard deviations; bounded
  mass conservation; pixels-wrapper / byte-API agreement across
  branches; X-only blur leaves Y-axis untouched; box-branch blur is
  monotone across a vertical step edge). Math cited to
  `docs/image/svg/svg11-second-edition.pdf` §15.17; no `image` /
  `imageproc` / `opencv` / `cairo` / `skia` source consulted.
- `feColorMatrix` filter primitive (SVG 1.1 §15.10) — per-pixel
  4×5 colour-matrix transform on packed-RGBA buffers. The §15.10
  spec defines the operation on un-premultiplied normalised channel
  values: `(R', G', B', A')ᵀ = M · (R, G, B, A, 1)ᵀ`, with the
  trailing `1` acting as a bias-column multiplier so a matrix row
  can encode a constant offset. Output is clamped to `[0, 1]` and
  re-quantised to `u8`. Exposed as
  `pub fn color_matrix(src, w, h, &m) -> Vec<u8>` plus the typed-
  pixel convenience wrapper `color_matrix_pixels(&[Rgba], ...)`,
  the `ColorMatrix([[f32; 5]; 4])` row-major 4×5 matrix type, and
  the `color_matrix_op(src, w, h, op, &user)` dispatcher driven by
  the `ColorMatrixOp { Matrix, Saturate(f32), HueRotate(f32),
  LuminanceToAlpha }` enum (mirroring the SVG `type=` attribute).
  The three parameterised matrices are reproduced verbatim from
  §15.10's coefficient tables:
  - `ColorMatrix::saturate(s)` — desaturation/saturation factor `s`
    (clamped to `[0, ∞)`; `s = 1` is identity, `s = 0` collapses
    RGB to the spec's `(0.213, 0.715, 0.072)` BT.709 luminance
    scalar).
  - `ColorMatrix::hue_rotate(degrees)` — rotation around the
    achromatic axis, built as
    `const_matrix + cos(θ)·cos_matrix + sin(θ)·sin_matrix` from the
    three §15.10 coefficient blocks; alpha row stays identity.
  - `ColorMatrix::luminance_to_alpha()` — fixed matrix producing
    transparent black RGB and alpha equal to
    `0.2125·R + 0.7154·G + 0.0721·B` (the §15.10 luminance triple,
    distinct from the rounded saturate set above — we follow the
    spec literally for each operator). Used as the matrix back-end
    for SVG `mask-type="luminance"` / PDF `SMask` `Luminosity`.
  - `ColorMatrix::identity()` and `ColorMatrix::from_op(op, user)`
    constructors for callers that want to inspect / cache /
    compose the matrix before applying it.
  21 new tests: 13 in the `src/filter.rs` unit suite (identity
  byte-for-byte round-trip on a pseudo-random pattern; saturate(1)
  ≈ identity within 1 LSB; saturate(0) collapses RGB to the
  analytic luminance scalar; hueRotate(0) ≈ identity; hueRotate
  preserves grey-axis pixels across 6 angles; hueRotate(360°) ≈
  identity; luminanceToAlpha zeros RGB and writes the §15.10
  luminance to alpha at the analytic value; `color_matrix_op`
  dispatches byte-exactly against `ColorMatrix::from_op`; positive
  + negative out-of-gamut matrix entries clamp at 255/0 rather than
  wrap; bias column adds a constant offset across all pixels;
  typed-pixel wrapper round-trips through the byte API across all
  four operator families; zero-area image returns an empty buffer;
  `#[should_panic]` guard for wrong-length input) plus 8 integration
  tests in `tests/filter_color_matrix.rs` (public-API identity
  byte-exact; saturate(0) greyscale invariant across every pixel of
  an 8×5 buffer; hueRotate preserves alpha and grey-axis brightness
  across 6 angles; luminanceToAlpha analytic check across 5
  representative pixels including pure primaries; typed-pixel
  wrapper agreement across all four operator families; user-supplied
  matrix branch halves every channel correctly; hueRotate(180°) on
  pure red clamps R to 0 and lifts G to the analytic `0.426·255`
  value; zero-area image returns empty across every operator). Math
  cited to `docs/image/svg/svg11-second-edition.pdf` §15.10; no
  `image` / `imageproc` / `opencv` / `cairo` / `skia` source
  consulted.
- `feMorphology` filter primitive (SVG 1.1 §15.20) — erosion and
  dilation of a packed-RGBA buffer by an axis-aligned rectangular
  structuring element of half-extents `(rx, ry)`, exposed as
  `pub fn morphology(src, w, h, rx, ry, op) -> Vec<u8>` plus the
  typed-pixel convenience wrapper `morphology_pixels(&[Rgba], ...)`
  and the `MorphologyOp { Erode, Dilate }` enum. Per the spec the
  output is the component-wise min (erode) or max (dilate) of the
  RGBA values inside the kernel rectangle, with the discrete kernel
  realised as a `(2·rx + 1) × (2·ry + 1)` window inclusive of the
  centre (the standard "ball of radius r" in discrete morphology).
  Boundary handling is clamp-to-edge. The implementation exploits the
  classical separability of flat rectangular structuring elements
  (`f ⊖ B = (f ⊖ Bx) ⊖ By`, `f ⊕ B = (f ⊕ Bx) ⊕ By` — Serra,
  *Image Analysis and Mathematical Morphology*, 1982 §I.4 Theorem 4.1
  / Gonzalez & Woods, *Digital Image Processing*, 3rd ed. 2008 §9.4.1)
  to run one horizontal 1-D sliding-window pass followed by one
  vertical 1-D pass, dropping per-pixel work from `O(rx · ry)` to
  `O(rx + ry)`. 16 new tests: 9 in the `src/filter.rs` unit suite
  (zero-radius identity, solid-image invariance under both operators
  across 6 radii, dilation footprint of an isolated bright pixel
  matches the 25-pixel 5×5 kernel exactly, erosion shaves a 7×7
  block down to 5×5 at `rx = ry = 1`, the duality
  `erode(f) = ¬dilate(¬f)` byte-exact across 5 radii on a noisy
  pattern, the separable H-then-V pipeline matches the naive 2-D
  reference across 5 radii × 2 operators on a 11×9 pseudo-random
  field, per-pixel extensivity
  `erode(f) ≤ f ≤ dilate(f)` across 5 radii, the typed-pixel wrapper
  round-trips through the byte API, and a `#[should_panic]` guard
  for wrong-length input) plus 7 integration tests in
  `tests/filter_morphology.rs` (public-API zero-radius identity,
  isolated-pixel dilation paints exactly 7×3 = 21 bright pixels for
  `rx = 3, ry = 1`, a 1-pixel-wide line is fully erased by
  `rx = 1, ry = 0` erosion, closing is idempotent on the interior of
  a 9×9 block, opening with a 3×3 SE removes a detached noise pixel
  while preserving a 5×5 block, axis-decoupled `ry = 0` dilation
  does not smear vertically, and the typed-pixel wrapper matches the
  byte API across 4 radii × 2 operators). Math cited to
  `docs/image/svg/svg11-second-edition.pdf` §15.20 + Serra (1982) +
  Gonzalez & Woods (2008); no `image` / `imageproc` / `opencv` /
  `cairo` / `skia` source consulted.
- Catmull–Rom round-trip regression tests in `tests/image_catmull_rom.rs`:
  a 1:1-scale identity check on a smooth synthetic plate (asserts
  PSNR ≈ ∞ — the interpolation property `k(0)=1, k(±n)=0` implies
  pixel-perfect reproduction at unit scale), a downsample→upsample
  chain bound (32×32 → 16×16 → 32×32 must clear 30 dB; measured
  56.33 dB at HEAD), and a head-to-head sanity assertion that
  Catmull–Rom beats nearest-neighbour by >5 dB on the same chain
  (measured margin 24 dB: 56.33 vs 32.32). Cites Catmull & Rom
  (1974) and Mitchell & Netravali (1988) for the underlying math —
  no external image-library source consulted.
- `ImageFilter::Lanczos3` — windowed-sinc image resampling at `a = 3`,
  the wider sibling of the existing `Lanczos2`. The kernel is
  `lanczos3(x) = sinc(π·x) · sinc(π·x/3)` for `|x| < 3` (zero elsewhere)
  — the standard high-quality reconstruction filter of the image-
  processing literature (Duchon, "Lanczos filtering in one and two
  dimensions", 1979; Turkowski, "Filters for Common Resampling Tasks",
  1990). The 2-D filter is the separable product `lanczos3(x) * lanczos3(y)`
  evaluated over a 6×6 footprint (36 taps vs Lanczos2's 16), so it
  captures more of the underlying sinc main lobe and a piece of the
  first negative side-lobe — sharper than Lanczos2 with a less abrupt
  impulse-response truncation, at the cost of 2.25× the per-pixel
  work and a stronger secondary side-lobe that can overshoot further.
  Identical bookkeeping to the Lanczos2 path: premultiplied-alpha
  accumulation, clamp-to-edge for footprints that extend past the
  texture boundary, per-axis weight re-normalisation, and a final
  per-channel `[0, 255]` clamp before the un-premultiply. The
  refactor extracts a shared const-generic `sample_image_lanczos_a::<N, W>`
  (`N` is the half-window in source pixels, `W = 2·N` the per-axis
  tap count) so the Lanczos2 and Lanczos3 samplers are byte-identical
  apart from `a` and the kernel pointer. Selectable via
  `Renderer::image_filter`; the renderer default stays `Bilinear`. 10
  new tests (4 kernel unit tests in `renderer.rs` covering unit-at-zero
  / vanishing at non-zero integers / even symmetry / negative side-lobe
  presence vs the cubic B-spline / wider window than Lanczos2, plus 6
  integration tests in `tests/image_lanczos3.rs` covering solid-colour
  centre + clamped-corner preservation, seam blending, in-gamut clamp,
  sharper-than-Lanczos2 step edge, and the unchanged renderer default).
- `ImageFilter::BSpline` — cubic B-spline image resampling, the `B = 1,
  C = 0` *approximating* member of the Mitchell–Netravali (1988) BC
  reconstruction-filter family and the third canonical point alongside
  Mitchell (`B = C = 1/3`) and Catmull–Rom (`B = 0, C = 1/2`). It is the
  smoothest of the three: the kernel is everywhere non-negative, so it
  cannot ring or overshoot (nothing to clamp), at the cost of the
  strongest blur. Unlike Catmull–Rom it does not pass through the source
  samples (`k(0) = 4/6`, `k(±1) = 1/6`), so it smooths rather than
  interpolates — the right choice for halo-free, monotone-preserving
  *downscaling* / minification. Reuses the shared `sample_image_bc_cubic`
  4×4 separable premultiplied-alpha sampler via a new `b_spline(x)`
  kernel wrapper over the existing generic `bc_cubic(x, B, C)` evaluator,
  so the boundary clamp, weight re-normalisation, and un-premultiply are
  byte-identical across the family. Selectable via
  `Renderer::image_filter`; the renderer default stays `Bilinear`. 10 new
  tests (5 kernel unit tests in `renderer.rs` covering the approximating
  property `k(0) = 4/6` / even symmetry / everywhere-non-negative (no
  ringing) / partition-of-unity / generic-vs-named agreement, plus 5
  integration tests in `tests/image_b_spline.rs` covering solid-colour
  preservation, seam blending, pixel-aligned smoothing, in-gamut step
  edges, and the unchanged default).
- `ImageFilter::CatmullRom` — Catmull–Rom bicubic image resampling, the
  `B = 0, C = 1/2` interpolating member of the Mitchell–Netravali (1988)
  BC reconstruction-filter family. Unlike Mitchell (`B = C = 1/3`),
  Catmull–Rom passes exactly through the source samples (`k(0) = 1`,
  `k(±1) = k(±2) = 0`), so it preserves crisp edges with no blur term —
  a better default for *upscaling* line art / UI than Mitchell, which
  trades sharpness for reduced ringing. The 4×4 separable Mitchell
  sampler was refactored into a shared `sample_image_bc_cubic` taking a
  per-axis kernel function and a generic `bc_cubic(x, B, C)` evaluator;
  `mitchell_netravali` and the new `catmull_rom` are thin wrappers, so
  the premultiplied-alpha accumulation, boundary clamp, weight
  re-normalisation, and un-premultiply are byte-identical across the
  family. Selectable via `Renderer::image_filter`; the renderer default
  stays `Bilinear`. 10 new tests (5 kernel unit tests in `renderer.rs`
  covering interpolation property / even symmetry / partition-of-unity /
  negative side-lobe / generic-vs-named agreement, plus 5 integration
  tests in `tests/image_catmull_rom.rs` covering solid-colour
  preservation, seam blending, pixel-aligned exact reproduction,
  in-gamut step edges, and the unchanged default).
- Pre-baked stops look-up table for gradient paints. `StopsLut::build`
  pre-computes a 256-entry `Rgba` table once per fill in the active
  `InterpolationSpace`; per-pixel evaluation reduces to a clamped index
  + load (no `srgb_to_linear` / `linear_to_srgb` `powf` evaluation in
  the hot loop, no stops-window scan). `eval_linear_gradient_lut` and
  `eval_radial_gradient_lut` are the LUT-aware sampler entry points;
  the existing `eval_*_in` per-pixel evaluators remain for callers
  that don't have a LUT in hand. The `Renderer::composite_with_paint`
  fast path builds a LUT once per gradient fill, then dispatches the
  per-pixel closure through `sample_paint_with_lut` — gradient renders
  in `LinearRgb` mode see the largest improvement (the LUT collapses
  6× `powf` per pixel down to one read), but `sRGB` interpolation
  also benefits from the eliminated stops scan. Wired transparently;
  no public-facing API change for callers using `Renderer::render`.
  13 new tests (6 `gradient.rs` unit + 7 integration in
  `tests/gradient_lut.rs`) verify LUT/per-pixel agreement within ±1
  LSB, endpoint exactness, empty-stops transparency, NaN-safety, and
  end-to-end Renderer parity.
- 4 non-separable HSL blend modes from PDF 32000-1:2008 §11.3.5.3 /
  W3C Compositing-1 §11: Hue, Saturation, Color, Luminosity. Composed
  from the spec's auxiliary helpers `Lum(C) = 0.30·R + 0.59·G + 0.11·B`
  (PDF coefficients — *not* Rec. 709), `SetLum`, `ClipColor`,
  `Sat = max(R,G,B) − min(R,G,B)`, and `SetSat`. The four formulas
  from Table 137 are wired into [`BlendMode::blend_rgb`] which now
  dispatches non-separable modes per-RGB-triple and falls through to
  the existing per-channel `blend_channel` path for the 12 separable
  modes. `BlendMode::is_separable()` is added for introspection;
  `blend_channel` panics with a clear message if called with a
  non-separable mode (those require the full triple). Composite path
  is unchanged: any non-`Normal` mode routes through `blend_over`,
  which now correctly evaluates the HSL modes. Brings the
  `Renderer::blend_mode` enum to the full 16-mode PDF / W3C surface.
- 11 standard separable blend modes from PDF 32000-1:2008 §11.3.5 /
  W3C Compositing-1 §10: Normal (default, fast source-over path
  unchanged), Multiply, Screen, Overlay, Darken, Lighten, ColorDodge,
  ColorBurn, HardLight, SoftLight, Difference, Exclusion. Selected via
  the new `Renderer::blend_mode` field (defaults to `BlendMode::Normal`,
  so existing renders are bit-identical); applied uniformly to fill,
  stroke, and image paints. Per-pixel `B(Cb, Cs)` evaluated in
  normalised f32 then folded into the spec's basic compositing formula
  `Cr = (1 − αs/αr)·Cb + (αs/αr)·[(1 − αb)·Cs + αb·B(Cb, Cs)]`. The
  Normal path retains the existing premultiplied-integer fast path so
  the dispatch cost is one branch per pixel for legacy callers. A
  free-standing `blend_over(cb, cs, mode)` helper and the lower-level
  `blend_channel` / `composite_rgba_premultiplied_blend` entry points
  are also re-exported. Per-element SVG-2 `mix-blend-mode` override on
  individual nodes deferred to a future round (would need a
  `Node::BlendMode` enum on `oxideav-core`).
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
