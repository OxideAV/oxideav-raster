# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
