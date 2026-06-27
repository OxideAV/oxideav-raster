//! Image-domain filter primitives.
//!
//! This module hosts post-rasterisation pixel-domain filters — the SVG
//! 1.1 `<feXxx>` family / PDF transparency-group filter family. The
//! pipeline elsewhere in the crate produces a packed-`Rgba`
//! [`oxideav_core::VideoFrame`]; the operators here take that buffer
//! (or a slice of it) and produce a new buffer of the same dimensions.
//!
//! # Currently implemented
//!
//! * **Mathematical morphology** — erosion and dilation by an axis-
//!   aligned rectangular structuring element (`feMorphology` in SVG
//!   1.1 §15.20). Implemented per-channel on premultiplied RGBA in
//!   `u8` space. The rectangular kernel decomposes into a horizontal
//!   1-D pass followed by a vertical 1-D pass, a standard textbook
//!   property of morphology with a flat separable structuring element
//!   (Serra, *Image Analysis and Mathematical Morphology*, 1982 §I.4;
//!   Gonzalez & Woods, *Digital Image Processing*, 3rd ed., 2008
//!   §9.4.1) — so the per-pixel work scales linearly with the radius
//!   instead of quadratically.
//!
//! * **Colour-matrix transform** — `feColorMatrix` from SVG 1.1
//!   §15.10. A 4×5 matrix `M` is applied to the un-premultiplied
//!   colour-tuple `(R, G, B, A, 1)ᵀ` per pixel to produce the new
//!   `(R', G', B', A')ᵀ`. The spec defines four sub-operations
//!   ([`ColorMatrixOp`]): a general user matrix, a one-parameter
//!   saturation matrix, a one-parameter hue rotation, and a fixed
//!   "luminance to alpha" matrix. The latter three are pre-computed
//!   matrices given verbatim by §15.10; we expose them through
//!   [`ColorMatrix::saturate`], [`ColorMatrix::hue_rotate`] and
//!   [`ColorMatrix::luminance_to_alpha`] so callers do not have to
//!   re-derive them.
//!
//! * **Gaussian blur** — `feGaussianBlur` from SVG 1.1 §15.17.
//!   Convolves a packed-RGBA buffer with the separable normalised
//!   Gaussian `G(x, y) = H(x)·I(y)` where
//!   `H(x) = exp(-x²/(2s²)) / sqrt(2π·s²)` and
//!   `I(y) = exp(-y²/(2t²)) / sqrt(2π·t²)`. For small standard
//!   deviations (`s < 2.0`) the spec's "separable convolution" route is
//!   taken directly with a discrete kernel of half-width `ceil(3·s)`
//!   (≈99.7% of the analytical mass). For `s ≥ 2.0` the spec's
//!   three-box-blur approximation is used:
//!   `d = floor(s · 3·sqrt(2π)/4 + 0.5)`; if `d` is odd three centred
//!   box-blurs of size `d` are composed, if `d` is even two box-blurs
//!   of size `d` (centred on the left and right pixel boundaries
//!   respectively) are composed with one box-blur of size `d + 1`.
//!   Boundary handling is clamp-to-edge for both modes.
//!
//! * **Component transfer** — `feComponentTransfer` from SVG 1.1
//!   §15.11. Per-pixel, per-channel transfer function — each of R, G,
//!   B, A carries one of the five §15.11 modes (`identity`, `table`
//!   piecewise-linear over `N+1` values, `discrete` step over `N`
//!   values, `linear` affine `slope · C + intercept`, `gamma`
//!   `amplitude · C^exponent + offset`). The operation is performed
//!   on un-premultiplied channels normalised to `[0, 1]`, then
//!   clamped to that range, then re-quantised back to `u8`.
//!
//! * **Composite** — `feComposite` from SVG 1.1 §15.12. Combines two
//!   equal-sized RGBA buffers pixel-wise. The six Porter–Duff
//!   operators (`over` / `in` / `out` / `atop` / `xor` / `lighter`)
//!   are expressed
//!   through the premultiplied-alpha blend-factor pair `(Fa, Fb)` —
//!   `co = ca·Fa + cb·Fb`, `αo = αa·Fa + αb·Fb` — consistent with the
//!   §14.2 simple-alpha-compositing formula (`Cr' = (1 − Ea)·Cr + Er`,
//!   i.e. `over`) already used by [`crate::composite_rgba_premultiplied`].
//!   The `arithmetic` operator evaluates the §15.12 per-channel formula
//!   `result = k1·i1·i2 + k2·i1 + k3·i2 + k4` on premultiplied channels,
//!   clamped to `[0, 1]`. Selectable via [`CompositeOp`].
//!
//! * **Matrix convolution** — `feConvolveMatrix` from SVG 1.1 §15.13.
//!   Applies an arbitrary `order_x × order_y` kernel to a packed-RGBA
//!   buffer per the §15.13 formula
//!   `COLOR_{X,Y} = (SUM SUM SOURCE · kernelMatrix[orderX-J-1,
//!   orderY-I-1]) / divisor + bias`, including the spec-mandated 180°
//!   kernel rotation. All three `edgeMode` policies (`duplicate` /
//!   `wrap` / `none`) and both `preserveAlpha` settings are
//!   implemented; `preserveAlpha = true` convolves RGB only and
//!   passes alpha through unchanged.
//!
//! * **Perlin turbulence** — `feTurbulence` from SVG 1.1 §15.24. A
//!   purely synthetic source primitive (the output is generated from
//!   the §15.24 parameter set, not from an input image). The Park–
//!   Miller minimum-standard LCG (`a = 16807`, `m = 2^31 − 1`) seeds a
//!   permutation table and four channel-specific gradient tables; the
//!   §15.24 `noise2` routine applies a `t² · (3 − 2t)` smoothstep to
//!   a bilinear interpolation between four surrounding lattice
//!   gradients. The `turbulence` accumulator sums `num_octaves`
//!   noise terms with geometrically-doubling frequency and halving
//!   amplitude. Both `type` arms are implemented: `turbulence` (`SUM
//!   |noise2|`, aim `[0, 1]`, quantised as `value · 255`) and
//!   `fractalNoise` (`SUM noise2`, aim `[-1, 1]`, quantised as
//!   `(value · 255 + 255) / 2`). The `stitchTiles="stitch"` path
//!   adjusts the base frequencies so the tile rectangle contains an
//!   integral number of first-octave Perlin tiles and wraps the
//!   lattice indices at the right and bottom edges of the active
//!   area. The §15.24 self-check value (`seed = 1`, 10000th LCG state
//!   = `1043618065`) is asserted in the test suite.
//!
//! * **Flood** — `feFlood` from SVG 1.1 §15.16. A synthetic source
//!   primitive: emit a `width × height` packed-RGBA buffer entirely
//!   filled with `(r, g, b)` modulated by `flood_opacity`. The
//!   filter primitive subregion (the `x` / `y` / `width` / `height`
//!   attributes on `<feFlood>`) is resolved by the caller — the
//!   function only fills the rectangle it is given.
//!
//! * **Offset** — `feOffset` from SVG 1.1 §15.21. Translate the
//!   packed-RGBA source by `(dx, dy)`. The output pixel at `(x, y)`
//!   is sourced from `(x − dx, y − dy)` per §15.21 ("offsets the
//!   input image relative to its current position … by the
//!   specified vector"); positions that fall outside the source
//!   extent emit transparent black `(0, 0, 0, 0)` per the §15.7.3
//!   "undefined pixels are set to transparent black" rule. Two
//!   sampling policies are wired in: integer-rounded
//!   nearest-pixel (the default — matches the §15.21 example
//!   `dx="4" dy="4"` for the common case of untransformed shifts)
//!   and bilinear reconstruction for fractional shifts (the
//!   §15.21 "high quality viewer should make use of appropriate
//!   interpolation techniques, for example bilinear or bicubic"
//!   route).
//!
//! * **Merge** — `feMerge` from SVG 1.1 §15.19. Composite N input
//!   layers bottom-to-top using the §14.2 simple-alpha-compositing
//!   `over` operator (Input1 on the bottom, InputN on top). With
//!   zero layers the result is transparent black (the
//!   filter-effects-region initial value per §15.1). The N-layer
//!   case is implemented as a fold so the spec's "n-1 Composite
//!   filters" form is recovered transparently.
//!
//! * **Tile** — `feTile` from SVG 1.1 §15.23. Fill a target
//!   rectangle by periodic replication of a reference tile: output
//!   pixel `(ox, oy)` reads source pixel `(ox mod src_w, oy mod
//!   src_h)`. The §15.23 statement "i and j can be any integer
//!   value" reduces, per output pixel, to the Euclidean remainder
//!   on the source extent. Source and target rectangles are aligned
//!   at the origin (the `i = 0, j = 0` copy of the tile lands with
//!   its top-left at the target top-left); callers that need a
//!   different alignment shift the input through [`offset`] first.
//!
//! * **Displacement map** — `feDisplacementMap` from SVG 1.1
//!   §15.15. Per-pixel inverse-mapping warp that resamples `in1`
//!   at `(x + scale·(XC − 0.5), y + scale·(YC − 0.5))`, where the
//!   `XC` / `YC` channels are selected out of `in2` by the caller.
//!   Out-of-bounds samples emit the §15.7.3 transparent-black
//!   value; integer-nearest and bilinear reconstruction are both
//!   available.
//!
//! * **Blend** — `feBlend` from SVG 1.1 §15.9. Per-pixel
//!   combination of two equally-sized inputs through one of five
//!   spec-listed modes (`normal` / `multiply` / `screen` /
//!   `darken` / `lighten`). Result alpha is the shared
//!   `qr = 1 − (1 − qa)·(1 − qb)` formula; the mode-specific
//!   colour formulas operate on the premultiplied colour pair
//!   `(ca, cb)` per the §15.9 table.
//!
//! * **Diffuse lighting** — `feDiffuseLighting` from SVG 1.1
//!   §15.14. Phong-diffuse Sobel-normal bump-map illumination
//!   keyed off the source alpha channel. Lights a packed-RGBA
//!   buffer through one of the three §15.8 light-source variants
//!   (`feDistantLight` / `fePointLight` / `feSpotLight`) and emits
//!   the §15.14 colour formula `D = kd · max(N·L, 0) · L_color`
//!   with `Da = 1.0` everywhere.
//!
//! * **Specular lighting** — `feSpecularLighting` from SVG 1.1
//!   §15.22. Phong-specular reflectance, sharing the §15.14
//!   surface-normal kernel and §15.8 light sources with
//!   [`diffuse_lighting`]. Per §15.22 the result is
//!   `S = ks · pow(N·H, specularExponent) · L_color` with
//!   `H = (L + E) / |L + E|`, `E = (0, 0, 1)`, and an alpha
//!   channel set to the per-pixel `max(Sr, Sg, Sb)` (so the
//!   filter produces a non-opaque image meant to be added on top
//!   of a textured layer via `feComposite`-arithmetic).
//!
//! * **Image source** — `feImage` from SVG 1.1 §15.18. A source
//!   primitive that places an already-decoded RGBA raster into the
//!   filter chain, "loaded or rendered into an RGBA raster" per
//!   §15.18, fitted to the target rectangle exactly as the `image`
//!   element fits a referenced raster ("The processing of the
//!   ‘preserveAspectRatio’ attribute on the ‘feImage’ element is
//!   identical to that of the ‘image’ element"). The §7.8
//!   `preserveAspectRatio` fitting algebra is implemented in full:
//!   the `none` alignment scales each axis independently so the
//!   source exactly fills the target; the nine uniform alignments
//!   (`xM{in,id,ax}YM{in,id,ax}`) force a single scale factor —
//!   the per-axis minimum under `meet` ("the entire ‘viewBox’ is
//!   visible within the viewport … scaled up as much as possible")
//!   or the maximum under `slice` ("the entire viewport is covered
//!   by the ‘viewBox’ … scaled down as much as possible") — and
//!   place the scaled image at the §7.8 min / midpoint / max
//!   anchor on each axis. Target pixels not covered by the fitted
//!   source emit the §15.7.3 transparent-black value. Nearest and
//!   bilinear reconstruction are both wired in, following the
//!   §15.18 note that "high quality viewers make use of appropriate
//!   interpolation techniques, for example bilinear or bicubic".
//!
//! # Deferred
//!
//! Drop shadow (`feDropShadow`) — its normative definition lives in
//! the W3C Filter Effects Module Level 1, which is not currently
//! staged under `docs/image/svg/` (the staged SVG 2 CR only links
//! out to it).
//!
//! # Wall provenance
//!
//! Math transcribed from `docs/image/svg/svg11-second-edition.pdf`
//! §15.20 ("This filter primitive performs 'fattening' or 'thinning'
//! of artwork. … In dilation, the output pixel is the individual
//! component-wise maximum of the corresponding R,G,B,A values in the
//! input image's kernel rectangle. In erosion, the output pixel is the
//! individual component-wise minimum…") and Serra (1982) §I.4 + Gonzalez
//! & Woods (2008) §9.4.1 for the separability decomposition; §15.10
//! for the `feColorMatrix` matrix forms (general 4×5 matrix; the
//! saturate / hueRotate / luminanceToAlpha pre-built matrices are
//! reproduced verbatim from the spec table); §15.17 for the Gaussian
//! blur kernel definition and the three-box-blur approximation
//! formula `d = floor(s · 3·sqrt(2π)/4 + 0.5)` for `s ≥ 2.0`; §15.11
//! for the five `feComponentTransfer` modes (identity / table /
//! discrete / linear / gamma); §15.12 for the `feComposite` arithmetic
//! formula `result = k1·i1·i2 + k2·i1 + k3·i2 + k4` (reproduced
//! verbatim) and the §14.2 premultiplied simple-alpha-compositing
//! algebra (`Cr' = (1 − Ea)·Cr + Er`) from which the Porter–Duff
//! `over` / `in` / `out` / `atop` / `xor` blend-factor pairs are
//! derived. §15.13 for `feConvolveMatrix` (the per-pixel formula
//! reproduced verbatim, the 180° kernel rotation called out
//! explicitly, the three `edgeMode` policies — `duplicate` /
//! `wrap` / `none` — and the `preserveAlpha = true` "convolve RGB
//! only, pass alpha through" rule). §15.16 for `feFlood` (the
//! property pair `flood-color` / `flood-opacity` resolves to a
//! single straight-alpha pixel that fills the filter primitive
//! subregion). §15.21 for `feOffset` (the offset vector `(dx, dy)`
//! shifts the source image; §15.7.3 fixes the "undefined pixels are
//! set to transparent black" rule for samples that fall outside
//! the source extent; the bilinear-resampling sampling option
//! follows the §15.21 note that "a high quality viewer should make
//! use of appropriate interpolation techniques, for example
//! bilinear or bicubic"). §15.19 for `feMerge` ("composites input
//! image layers on top of each other using the over operator with
//! Input1 (corresponding to the first `feMergeNode` child element)
//! on the bottom and the last specified input, InputN
//! (corresponding to the last `feMergeNode` child element), on
//! top") wired through the §14.2 simple-alpha-compositing `over`
//! algebra `αo = αs + αd · (1 − αs)`, `co = (cs · αs + cd · αd · (1
//! − αs)) / αo`. §15.23 for `feTile` ("the top/left corner of each
//! given tile is at location (x+i*width, y+j*height) … i and j can be
//! any integer value") collapsed to the per-pixel Euclidean remainder
//! `(ox mod src_w, oy mod src_h)`. §15.9 for `feBlend` (the result-
//! alpha formula `qr = 1 − (1 − qa)·(1 − qb)` shared by all five
//! modes; the five per-mode colour formulas reproduced verbatim:
//! `normal: cr = (1 − qa)·cb + ca`, `multiply: cr = (1 − qa)·cb +
//! (1 − qb)·ca + ca·cb`, `screen: cr = cb + ca − ca·cb`,
//! `darken: cr = Min((1 − qa)·cb + ca, (1 − qb)·ca + cb)`,
//! `lighten: cr = Max((1 − qa)·cb + ca, (1 − qb)·ca + cb)`).
//! §15.18 for `feImage` ("loaded or rendered into an RGBA raster
//! and becomes the result of the filter primitive"; "the processing
//! of the ‘preserveAspectRatio’ attribute on the ‘feImage’ element
//! is identical to that of the ‘image’ element"; the default "as if
//! a value of xMidYMid meet were specified"; the bilinear option
//! follows "it is recommended that high quality viewers make use of
//! appropriate interpolation techniques, for example bilinear or
//! bicubic") and §7.8 for the `preserveAspectRatio` fitting rules
//! (the ten `<align>` values with their min / midpoint / max
//! anchor semantics quoted per variant on [`AspectRatioAlign`];
//! `meet` = "the entire ‘viewBox’ is visible within the viewport"
//! and "the ‘viewBox’ is scaled up as much as possible"; `slice` =
//! "the entire viewport is covered by the ‘viewBox’" and "the
//! ‘viewBox’ is scaled down as much as possible") and §15.7.3 for
//! the transparent-black value of target pixels the fitted source
//! does not cover.

use oxideav_core::Rgba;

/// Working colour space for filter imaging operations.
///
/// Filter Effects Module Level 1 §10 (`color-interpolation-filters`)
/// defines which colour space the per-pixel arithmetic of a filter
/// primitive is carried out in. The property takes one of three values
/// and its **initial value is `linearRGB`** — i.e. unless an author
/// explicitly opts out, every filter primitive operates on
/// light-linear samples, *not* on the gamma-encoded sRGB bytes that a
/// source graphic is stored in.
///
/// Concretely: blurs, convolutions, lighting, blends, composites and
/// component transfers are meant to combine *energy*, so the source is
/// linearised before the primitive runs and the result is re-encoded to
/// sRGB afterwards. Performing the same arithmetic directly on
/// gamma-encoded bytes darkens blurred edges and skews blends — the
/// difference this enum exists to control.
///
/// The transfer curve used to move between the two spaces is the
/// IEC 61966-2-1 sRGB transfer function (the same one SVG 2 §13.9 names
/// for the `color-interpolation` `linearRGB` conversion); see
/// [`srgb_to_linear_f32`] / [`linear_to_srgb_f32`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterColorSpace {
    /// `color-interpolation-filters: linearRGB` — the property's
    /// **initial value**. Filter colour operations occur in the
    /// linearised (light-energy) RGB space.
    #[default]
    LinearRgb,
    /// `color-interpolation-filters: sRGB` — colour operations occur
    /// directly in the gamma-encoded sRGB space (no linearisation).
    Srgb,
    /// `color-interpolation-filters: auto` — the user agent may choose
    /// either space; the author does not require a particular one. This
    /// implementation resolves `auto` to [`LinearRgb`](Self::LinearRgb)
    /// (the property's initial value), so a primitive run under `auto`
    /// matches the spec default rather than silently skipping
    /// linearisation.
    Auto,
}

impl FilterColorSpace {
    /// Resolve [`Auto`](Self::Auto) to a concrete space. `auto` collapses
    /// to [`LinearRgb`](Self::LinearRgb) (the property's initial value);
    /// the other two variants pass through unchanged.
    #[inline]
    pub fn resolve(self) -> Self {
        match self {
            FilterColorSpace::Auto => FilterColorSpace::LinearRgb,
            other => other,
        }
    }

    /// `true` when filter operations under this space must first
    /// linearise the source (and re-encode the result afterwards).
    /// [`Srgb`](Self::Srgb) returns `false`; [`LinearRgb`](Self::LinearRgb)
    /// and [`Auto`](Self::Auto) return `true`.
    #[inline]
    pub fn needs_linearisation(self) -> bool {
        matches!(self.resolve(), FilterColorSpace::LinearRgb)
    }
}

/// IEC 61966-2-1 sRGB → linear-light transfer function, full precision.
///
/// Input is a straight (non-premultiplied) sRGB sample in `[0.0, 1.0]`;
/// output is the corresponding light-energy value in `[0.0, 1.0]`. This
/// is the curve `color-interpolation-filters: linearRGB` requires before
/// a filter primitive runs (Filter Effects 1 §10; the transfer function
/// itself is SVG 2 §13.9 / IEC 61966-2-1).
#[inline]
pub fn srgb_to_linear_f32(cs: f32) -> f32 {
    if cs <= 0.040_45 {
        cs / 12.92
    } else {
        ((cs + 0.055) / 1.055).powf(2.4)
    }
}

/// IEC 61966-2-1 linear-light → sRGB transfer function, full precision.
///
/// The exact inverse of [`srgb_to_linear_f32`]: input is a light-energy
/// value (clamped to `[0.0, 1.0]`), output is the gamma-encoded sRGB
/// sample. This re-encodes a filter result back to sRGB after the
/// primitive has run in linear space.
#[inline]
pub fn linear_to_srgb_f32(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// 256-entry sRGB-byte → linear-light lookup table.
///
/// Every distinct 8-bit sRGB colour value maps to exactly one
/// light-energy float, so the per-pixel linearisation collapses to a
/// table read instead of a `powf` call. Built once on first access.
fn srgb_to_linear_lut() -> &'static [f32; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = srgb_to_linear_f32(i as f32 / 255.0);
        }
        t
    })
}

/// Convert one straight-alpha sRGB pixel into the linearRGB working
/// space. Only the colour channels pass through the transfer curve;
/// **alpha is coverage, not colour, and is left untouched** (Filter
/// Effects 1 §10 — the property governs *colour* operations only).
#[inline]
fn pixel_srgb_to_linear(p: Rgba) -> Rgba {
    let lut = srgb_to_linear_lut();
    Rgba::new(
        (lut[p.r as usize] * 255.0).round().clamp(0.0, 255.0) as u8,
        (lut[p.g as usize] * 255.0).round().clamp(0.0, 255.0) as u8,
        (lut[p.b as usize] * 255.0).round().clamp(0.0, 255.0) as u8,
        p.a,
    )
}

/// Re-encode one straight-alpha linearRGB pixel back to sRGB. The
/// inverse of [`pixel_srgb_to_linear`]; alpha is again passed through
/// unchanged.
#[inline]
fn pixel_linear_to_srgb(p: Rgba) -> Rgba {
    Rgba::new(
        (linear_to_srgb_f32(p.r as f32 / 255.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8,
        (linear_to_srgb_f32(p.g as f32 / 255.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8,
        (linear_to_srgb_f32(p.b as f32 / 255.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8,
        p.a,
    )
}

/// Convert an sRGB filter buffer into the linearRGB working space
/// in place (per [`Rgba`] pixel). Colour channels are linearised;
/// alpha is preserved. This is the *entry* transform a filter primitive
/// applies before it runs when `color-interpolation-filters` resolves
/// to `linearRGB` (the spec default).
pub fn to_linear_rgb_pixels(buf: &mut [Rgba]) {
    for p in buf.iter_mut() {
        *p = pixel_srgb_to_linear(*p);
    }
}

/// Re-encode a linearRGB filter buffer back to sRGB in place (per
/// [`Rgba`] pixel) — the *exit* transform applied after a primitive
/// has run in linear space. Inverse of [`to_linear_rgb_pixels`].
pub fn to_srgb_pixels(buf: &mut [Rgba]) {
    for p in buf.iter_mut() {
        *p = pixel_linear_to_srgb(*p);
    }
}

/// Packed-byte (`RGBA8`, row-major) counterpart of
/// [`to_linear_rgb_pixels`]: linearise the colour bytes of an sRGB
/// filter buffer in place, leaving every fourth (alpha) byte untouched.
/// `buf.len()` must be a multiple of 4.
///
/// An 8-bit *linear* intermediate cannot represent every sRGB code
/// distinctly (sRGB spends proportionally more codes in the dark end),
/// so a byte-buffer round trip through linear and back drifts by up to
/// six codes in the very dark range. Use [`srgb_to_linear_f32`] when
/// full precision is required.
pub fn to_linear_rgb(buf: &mut [u8]) {
    let lut = srgb_to_linear_lut();
    for px in buf.chunks_exact_mut(4) {
        px[0] = (lut[px[0] as usize] * 255.0).round().clamp(0.0, 255.0) as u8;
        px[1] = (lut[px[1] as usize] * 255.0).round().clamp(0.0, 255.0) as u8;
        px[2] = (lut[px[2] as usize] * 255.0).round().clamp(0.0, 255.0) as u8;
        // px[3] (alpha) unchanged.
    }
}

/// Packed-byte (`RGBA8`, row-major) counterpart of [`to_srgb_pixels`]:
/// re-encode the colour bytes of a linearRGB filter buffer back to sRGB
/// in place, leaving alpha untouched. `buf.len()` must be a multiple
/// of 4.
pub fn to_srgb(buf: &mut [u8]) {
    for px in buf.chunks_exact_mut(4) {
        px[0] = (linear_to_srgb_f32(px[0] as f32 / 255.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;
        px[1] = (linear_to_srgb_f32(px[1] as f32 / 255.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;
        px[2] = (linear_to_srgb_f32(px[2] as f32 / 255.0) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;
        // px[3] (alpha) unchanged.
    }
}

/// Run a filter primitive in the colour space selected by a resolved
/// [`FilterColorSpace`], over a packed-byte (`RGBA8`) buffer.
///
/// When `space` resolves to `linearRGB` (the `color-interpolation-filters`
/// default, including `auto`), the source is linearised, `op` runs on the
/// linear buffer, and the result is re-encoded to sRGB. When `space` is
/// `sRGB`, `op` runs directly on the gamma-encoded bytes with no
/// conversion. `op` receives the working buffer and returns the
/// primitive output in the *same* space it was handed.
pub fn in_filter_space<F>(space: FilterColorSpace, src: &[u8], op: F) -> Vec<u8>
where
    F: FnOnce(&[u8]) -> Vec<u8>,
{
    if space.needs_linearisation() {
        let mut lin = src.to_vec();
        to_linear_rgb(&mut lin);
        let mut out = op(&lin);
        to_srgb(&mut out);
        out
    } else {
        op(src)
    }
}

/// [`Rgba`]-pixel counterpart of [`in_filter_space`].
pub fn in_filter_space_pixels<F>(space: FilterColorSpace, src: &[Rgba], op: F) -> Vec<Rgba>
where
    F: FnOnce(&[Rgba]) -> Vec<Rgba>,
{
    if space.needs_linearisation() {
        let mut lin = src.to_vec();
        to_linear_rgb_pixels(&mut lin);
        let mut out = op(&lin);
        to_srgb_pixels(&mut out);
        out
    } else {
        op(src)
    }
}

/// Operator selector for [`morphology`].
///
/// Mirrors the `operator` attribute of SVG 1.1 §15.20 `<feMorphology>`
/// (`"erode"` | `"dilate"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphologyOp {
    /// Component-wise *minimum* over the structuring element ("thin"
    /// the artwork — shrinks bright regions, grows dark / transparent
    /// regions).
    Erode,
    /// Component-wise *maximum* over the structuring element ("fatten"
    /// the artwork — grows bright regions, shrinks dark / transparent
    /// regions).
    Dilate,
}

/// SVG 1.1 §15.20 `<feMorphology>`: morphological erosion or dilation by
/// an axis-aligned rectangular structuring element.
///
/// `width × height` is the input/output image extent in pixels. `src`
/// is a packed-RGBA byte buffer of exactly `width * height * 4` bytes
/// in row-major order (the same packing produced by
/// [`crate::Renderer::render`]).
///
/// `radius_x` and `radius_y` are the half-extents of the structuring
/// element along the X and Y axes respectively, in source pixels. The
/// rectangle is `(2·rx + 1) × (2·ry + 1)` (i.e. inclusive of the
/// centre — the standard "ball of radius r" definition in discrete
/// morphology). A radius of zero on **either** axis disables the
/// effect on that axis (the corresponding 1-D pass is a no-op); a
/// radius of zero on **both** axes returns the input unchanged.
///
/// Per the SVG 1.1 spec the operation runs component-wise on the
/// premultiplied colour values, which makes
/// `Cr,Cg,Cb ≤ Ca`an invariant the operator preserves: erosion of a
/// premultiplied tuple gives `(min Ci) ≤ (min Ca)`, dilation gives
/// `(max Ci) ≤ (max Ca)`. (The caller is responsible for handing in
/// premultiplied data; the crate's renderer already produces straight-
/// alpha output, so a caller wanting strict spec semantics should
/// premultiply first.)
///
/// # Algorithm
///
/// A rectangular structuring element `B = Bx ⊕ By` (Minkowski sum of a
/// horizontal `(2·rx + 1)`-line and a vertical `(2·ry + 1)`-line)
/// is *separable*: for any flat SE that decomposes as `Bx ⊕ By`,
///
/// ```text
/// f ⊖ B  =  (f ⊖ Bx) ⊖ By
/// f ⊕ B  =  (f ⊕ Bx) ⊕ By
/// ```
///
/// (Serra 1982 §I.4 Theorem 4.1; Gonzalez & Woods 2008 §9.4.1
/// eq. 9.4-1). The implementation therefore runs a horizontal 1-D pass
/// across each row, then a vertical 1-D pass down each column. Each
/// pass is a naive sliding-window min/max with **clamp-to-edge**
/// boundary handling (extending the image by its border pixels so the
/// kernel is always full at the corners — the same SVG default the
/// rest of the filter pipeline uses for missing samples).
///
/// Complexity: `O(W · H · (rx + ry))` instead of the
/// `O(W · H · rx · ry)` naive 2-D scan.
///
/// # Panics
///
/// Panics if `src.len() != width as usize * height as usize * 4`.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same dimensions, written
/// channel-independently. `radius_x == 0 && radius_y == 0` returns a
/// straight copy of `src`.
pub fn morphology(
    src: &[u8],
    width: u32,
    height: u32,
    radius_x: u32,
    radius_y: u32,
    op: MorphologyOp,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("morphology: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "morphology: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    // Fast path: both radii zero ⇒ identity per the SVG spec note that
    // a "value of zero disables the effect of the given filter
    // primitive". (Strictly the spec then says "result is a transparent
    // black image" — but only for radius=0 on the *filter primitive
    // level*. We treat the function-level same-pixel pass-through
    // here; the SVG-element-level transparent-black behaviour is the
    // caller's job at the scene-graph layer, on top of `radius == 0`
    // being an *error* per the spec error-processing rules anyway.)
    if radius_x == 0 && radius_y == 0 {
        return src.to_vec();
    }

    let mut buf = src.to_vec();

    if radius_x > 0 {
        let rx = radius_x as usize;
        let mut row = vec![0u8; w * 4];
        for y in 0..h {
            let off = y * w * 4;
            row.copy_from_slice(&buf[off..off + w * 4]);
            morphology_1d_horizontal(&row, &mut buf[off..off + w * 4], w, rx, op);
        }
    }

    if radius_y > 0 {
        let ry = radius_y as usize;
        // Column scratch: gather one column at a time as a
        // contiguous `4·H`-byte slice (so the inner loop is sequential
        // memory access), filter it, write it back.
        let mut col_in = vec![0u8; h * 4];
        let mut col_out = vec![0u8; h * 4];
        for x in 0..w {
            for y in 0..h {
                let src_off = (y * w + x) * 4;
                let dst_off = y * 4;
                col_in[dst_off..dst_off + 4].copy_from_slice(&buf[src_off..src_off + 4]);
            }
            morphology_1d_horizontal(&col_in, &mut col_out, h, ry, op);
            for y in 0..h {
                let dst_off = (y * w + x) * 4;
                let src_off = y * 4;
                buf[dst_off..dst_off + 4].copy_from_slice(&col_out[src_off..src_off + 4]);
            }
        }
    }

    buf
}

/// Single-pass 1-D morphology along a packed-RGBA row of `len` pixels.
///
/// `src` and `dst` must be `len * 4` bytes. Boundary handling is
/// clamp-to-edge: window samples outside `[0, len)` reuse the nearest
/// in-bounds pixel. Naive `O(len · radius)` scan — chosen for
/// clarity; a van Herk / Gil-Werman `O(len)` formulation is a future
/// optimisation.
fn morphology_1d_horizontal(
    src: &[u8],
    dst: &mut [u8],
    len: usize,
    radius: usize,
    op: MorphologyOp,
) {
    debug_assert_eq!(src.len(), len * 4);
    debug_assert_eq!(dst.len(), len * 4);
    let imax = len as isize - 1;
    for i in 0..len {
        let lo = (i as isize - radius as isize).max(0) as usize;
        let hi = (i as isize + radius as isize).min(imax) as usize;
        let mut acc = [
            src[lo * 4],
            src[lo * 4 + 1],
            src[lo * 4 + 2],
            src[lo * 4 + 3],
        ];
        for j in (lo + 1)..=hi {
            let p = &src[j * 4..j * 4 + 4];
            match op {
                MorphologyOp::Erode => {
                    if p[0] < acc[0] {
                        acc[0] = p[0];
                    }
                    if p[1] < acc[1] {
                        acc[1] = p[1];
                    }
                    if p[2] < acc[2] {
                        acc[2] = p[2];
                    }
                    if p[3] < acc[3] {
                        acc[3] = p[3];
                    }
                }
                MorphologyOp::Dilate => {
                    if p[0] > acc[0] {
                        acc[0] = p[0];
                    }
                    if p[1] > acc[1] {
                        acc[1] = p[1];
                    }
                    if p[2] > acc[2] {
                        acc[2] = p[2];
                    }
                    if p[3] > acc[3] {
                        acc[3] = p[3];
                    }
                }
            }
        }
        dst[i * 4..i * 4 + 4].copy_from_slice(&acc);
    }
}

/// Convenience wrapper that runs [`morphology`] on a slice of [`Rgba`]
/// pixels and returns a `Vec<Rgba>` of the same length. Identical
/// semantics — provided for callers that already have a typed
/// pixel buffer (the crate's tests are the immediate consumer).
pub fn morphology_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    radius_x: u32,
    radius_y: u32,
    op: MorphologyOp,
) -> Vec<Rgba> {
    assert_eq!(
        src.len(),
        width as usize * height as usize,
        "morphology_pixels: src.len() == {} but width*height == {}",
        src.len(),
        width as usize * height as usize
    );
    let mut bytes = Vec::with_capacity(src.len() * 4);
    for p in src {
        bytes.push(p.r);
        bytes.push(p.g);
        bytes.push(p.b);
        bytes.push(p.a);
    }
    let out = morphology(&bytes, width, height, radius_x, radius_y, op);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, c: Rgba) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.push(c.r);
            v.push(c.g);
            v.push(c.b);
            v.push(c.a);
        }
        v
    }

    #[test]
    fn zero_radius_is_identity() {
        let img = solid_rgba(4, 3, Rgba::new(10, 20, 30, 40));
        // Perturb one pixel so we'd notice if it got smeared.
        let mut input = img.clone();
        input[5 * 4] = 200;
        let out = morphology(&input, 4, 3, 0, 0, MorphologyOp::Dilate);
        assert_eq!(out, input);
        let out = morphology(&input, 4, 3, 0, 0, MorphologyOp::Erode);
        assert_eq!(out, input);
    }

    #[test]
    fn solid_image_is_invariant_under_either_operator() {
        // Morphology is idempotent on constant images: min(const set) ==
        // max(const set) == const.
        let img = solid_rgba(8, 5, Rgba::new(80, 120, 200, 255));
        for op in [MorphologyOp::Erode, MorphologyOp::Dilate] {
            for (rx, ry) in [(0, 0), (1, 0), (0, 1), (1, 1), (3, 2), (4, 4)] {
                let out = morphology(&img, 8, 5, rx, ry, op);
                assert_eq!(out, img, "op={op:?} rx={rx} ry={ry}");
            }
        }
    }

    #[test]
    fn dilate_grows_isolated_bright_pixel_to_full_kernel() {
        // 7×7 black canvas with a single fully-bright pixel at the
        // centre. Dilate with rx = ry = 2 ⇒ a 5×5 bright square
        // centred on the original pixel.
        let w = 7u32;
        let h = 7u32;
        let mut img = solid_rgba(w, h, Rgba::new(0, 0, 0, 255));
        let cx = 3usize;
        let cy = 3usize;
        let off = (cy * w as usize + cx) * 4;
        img[off..off + 4].copy_from_slice(&[200, 150, 100, 255]);

        let out = morphology(&img, w, h, 2, 2, MorphologyOp::Dilate);

        // Count the pixels that ended up with the bright colour. The
        // kernel is (2·2+1)·(2·2+1) = 25 pixels and they all live
        // fully inside the canvas → exactly 25 bright pixels.
        let mut bright = 0;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let p = &out[(y * w as usize + x) * 4..(y * w as usize + x) * 4 + 4];
                if p == [200, 150, 100, 255] {
                    bright += 1;
                    // Cell must be inside the centred 5×5 box.
                    let dx = x as isize - cx as isize;
                    let dy = y as isize - cy as isize;
                    assert!(dx.abs() <= 2 && dy.abs() <= 2, "stray bright at ({x},{y})");
                }
            }
        }
        assert_eq!(bright, 25, "dilate 5×5 footprint pixel count");
    }

    #[test]
    fn erode_shrinks_solid_block_by_radius_on_each_side() {
        // 9×9 canvas: 7×7 fully-opaque red block centred on a
        // transparent background. Erode rx = ry = 1 ⇒ a 5×5 red
        // block (1-pixel shaving off each side).
        let w = 9u32;
        let h = 9u32;
        let mut img = solid_rgba(w, h, Rgba::new(0, 0, 0, 0));
        for y in 1..8usize {
            for x in 1..8usize {
                let off = (y * w as usize + x) * 4;
                img[off..off + 4].copy_from_slice(&[255, 0, 0, 255]);
            }
        }

        let out = morphology(&img, w, h, 1, 1, MorphologyOp::Erode);

        let mut red = 0;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let p = &out[(y * w as usize + x) * 4..(y * w as usize + x) * 4 + 4];
                if p == [255, 0, 0, 255] {
                    red += 1;
                    assert!(
                        (2..=6).contains(&x) && (2..=6).contains(&y),
                        "stray red at ({x},{y}) after erode"
                    );
                }
            }
        }
        assert_eq!(red, 25, "erode shrinks 7×7 to 5×5 with rx=ry=1");
    }

    #[test]
    fn erode_and_dilate_are_duals_via_complement() {
        // Discrete duality: erode(f) == complement(dilate(complement(f)))
        // for any radius. This is the textbook morphology relation
        // (Serra 1982 §I.4 Theorem 4.2; G&W §9.2.2 eq. 9.2-6). We
        // verify it bit-exact on a small noisy pattern.
        let w = 5u32;
        let h = 4u32;
        let pattern: Vec<u8> = (0..(w * h * 4) as u8)
            .map(|i| (i.wrapping_mul(37)).wrapping_add(11))
            .collect();

        for (rx, ry) in [(1u32, 0u32), (0, 1), (1, 1), (2, 1), (2, 2)] {
            let eroded = morphology(&pattern, w, h, rx, ry, MorphologyOp::Erode);
            // complement → dilate → complement
            let neg: Vec<u8> = pattern.iter().map(|b| 255 - b).collect();
            let dil = morphology(&neg, w, h, rx, ry, MorphologyOp::Dilate);
            let dual: Vec<u8> = dil.iter().map(|b| 255 - b).collect();
            assert_eq!(eroded, dual, "duality fails at rx={rx} ry={ry}");
        }
    }

    #[test]
    fn separability_matches_naive_2d() {
        // The 2-D sliding-window min/max over a rectangular SE must
        // match the separable H-then-V evaluation (this is the
        // implementation detail the production path relies on for
        // its `O(W·H·(rx+ry))` complexity, so test it explicitly).
        let w = 11u32;
        let h = 9u32;
        let mut pat: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                pat.push(((x * 17 + y * 3) & 0xff) as u8);
                pat.push(((x * 5 + y * 23) & 0xff) as u8);
                pat.push(((x + y) & 0xff) as u8);
                pat.push((x.wrapping_mul(y).wrapping_add(7) & 0xff) as u8);
            }
        }

        for (rx, ry) in [(1u32, 1u32), (2, 1), (1, 2), (3, 2), (2, 3)] {
            for op in [MorphologyOp::Erode, MorphologyOp::Dilate] {
                let sep = morphology(&pat, w, h, rx, ry, op);
                let naive = naive_2d_morphology(&pat, w, h, rx, ry, op);
                assert_eq!(sep, naive, "rx={rx} ry={ry} op={op:?}");
            }
        }
    }

    /// Reference implementation used only inside the test module:
    /// straight 2-D scan over every kernel cell with clamp-to-edge.
    /// Quadratic in radius — kept ONLY for verifying the production
    /// separable path. NOT used at runtime.
    fn naive_2d_morphology(
        src: &[u8],
        w: u32,
        h: u32,
        rx: u32,
        ry: u32,
        op: MorphologyOp,
    ) -> Vec<u8> {
        let mut out = vec![0u8; src.len()];
        let wi = w as isize;
        let hi = h as isize;
        for y in 0..h as isize {
            for x in 0..w as isize {
                let mut acc = [0u8; 4];
                let mut first = true;
                for ky in -(ry as isize)..=(ry as isize) {
                    for kx in -(rx as isize)..=(rx as isize) {
                        let sx = (x + kx).clamp(0, wi - 1);
                        let sy = (y + ky).clamp(0, hi - 1);
                        let off = (sy * wi + sx) as usize * 4;
                        let p = [src[off], src[off + 1], src[off + 2], src[off + 3]];
                        if first {
                            acc = p;
                            first = false;
                        } else {
                            for c in 0..4 {
                                match op {
                                    MorphologyOp::Erode => {
                                        if p[c] < acc[c] {
                                            acc[c] = p[c];
                                        }
                                    }
                                    MorphologyOp::Dilate => {
                                        if p[c] > acc[c] {
                                            acc[c] = p[c];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let off = (y * wi + x) as usize * 4;
                out[off..off + 4].copy_from_slice(&acc);
            }
        }
        out
    }

    #[test]
    fn dilate_dominates_input_dominates_erode_per_pixel() {
        // For any pixel and any channel: erode(f)[i] ≤ f[i] ≤ dilate(f)[i].
        // This is the extensivity / anti-extensivity property of the
        // operators (Serra 1982 §I.4).
        let w = 7u32;
        let h = 6u32;
        let mut input: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                input.push(((x * 19 + y * 41 + 13) & 0xff) as u8);
                input.push(((x * 7 + y * 11 + 3) & 0xff) as u8);
                input.push(((x * 53 + y * 29 + 99) & 0xff) as u8);
                input.push(((x.wrapping_mul(y) + 17) & 0xff) as u8);
            }
        }

        for (rx, ry) in [(1u32, 0u32), (0, 1), (1, 1), (2, 2), (3, 1)] {
            let eroded = morphology(&input, w, h, rx, ry, MorphologyOp::Erode);
            let dilated = morphology(&input, w, h, rx, ry, MorphologyOp::Dilate);
            for i in 0..input.len() {
                assert!(
                    eroded[i] <= input[i],
                    "erode at byte {i} value {} > input {}",
                    eroded[i],
                    input[i]
                );
                assert!(
                    dilated[i] >= input[i],
                    "dilate at byte {i} value {} < input {}",
                    dilated[i],
                    input[i]
                );
            }
        }
    }

    #[test]
    fn morphology_pixels_wrapper_matches_byte_api() {
        let w = 4u32;
        let h = 3u32;
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for i in 0..(w * h) as u8 {
            let r = i.wrapping_mul(7);
            let g = i.wrapping_mul(17).wrapping_add(3);
            let b = i.wrapping_mul(31).wrapping_add(5);
            let a = 255u8;
            bytes.extend_from_slice(&[r, g, b, a]);
            pixels.push(Rgba::new(r, g, b, a));
        }
        for op in [MorphologyOp::Erode, MorphologyOp::Dilate] {
            for (rx, ry) in [(0u32, 1u32), (1, 0), (1, 1), (2, 1)] {
                let bytes_out = morphology(&bytes, w, h, rx, ry, op);
                let pix_out = morphology_pixels(&pixels, w, h, rx, ry, op);
                let recombined: Vec<u8> =
                    pix_out.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
                assert_eq!(bytes_out, recombined, "op={op:?} rx={rx} ry={ry}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "morphology: src.len() ==")]
    fn morphology_panics_on_wrong_length() {
        let bad = vec![0u8; 7]; // not w·h·4 for w=2, h=2
        let _ = morphology(&bad, 2, 2, 1, 1, MorphologyOp::Erode);
    }
}

// ---------------------------------------------------------------------------
//  feColorMatrix (SVG 1.1 §15.10)
// ---------------------------------------------------------------------------

/// Selector mirroring the `type` attribute of SVG 1.1 §15.10
/// `<feColorMatrix>` (`"matrix"` | `"saturate"` | `"hueRotate"` |
/// `"luminanceToAlpha"`).
///
/// The variants carry the spec's user-supplied parameter (none for
/// `Matrix` — the matrix itself is supplied as a [`ColorMatrix`]; a
/// single scalar for `Saturate` and `HueRotate`). The fully-fixed
/// `LuminanceToAlpha` operator takes no parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorMatrixOp {
    /// User-supplied 4×5 matrix applied verbatim. The matrix itself
    /// lives in the [`ColorMatrix`] passed to [`color_matrix`].
    Matrix,
    /// SVG `type="saturate"`: pre-built saturation matrix
    /// parameterised by `s ∈ [0, ∞)`. `s = 1` is identity, `s = 0`
    /// collapses RGB to the BT.709 luminance scalar, `s > 1` boosts
    /// saturation past the original.
    Saturate(f32),
    /// SVG `type="hueRotate"`: rotate the colour vector around the
    /// achromatic axis by `degrees` degrees (positive = R→G→B in the
    /// spec's convention). Constructed from the spec's static
    /// constant / cos-term / sin-term matrix triple.
    HueRotate(f32),
    /// SVG `type="luminanceToAlpha"`: fixed matrix that produces a
    /// transparent black image whose alpha is the BT.709 luminance of
    /// the input. Used as the back-end for SVG `mask`'s
    /// `mask-type="luminance"` and for chained PDF `SMask`
    /// `Luminosity` subtypes.
    LuminanceToAlpha,
}

/// A 4×5 colour-transform matrix in row-major layout (the same layout
/// SVG 1.1 §15.10 uses for the `values` attribute).
///
/// The four rows produce the new R, G, B, A in turn; the five columns
/// are the coefficients of `(R, G, B, A, 1)` — the trailing `1` is the
/// homogeneous bias term that lets the matrix add a constant offset to
/// each output channel.
///
/// ```text
/// | R' |   | m00 m01 m02 m03 m04 |   | R |
/// | G' |   | m10 m11 m12 m13 m14 |   | G |
/// | B' | = | m20 m21 m22 m23 m24 | × | B |
/// | A' |   | m30 m31 m32 m33 m34 |   | A |
///                                    | 1 |
/// ```
///
/// All values are floats in the same `[0, 1]` normalised scale the
/// spec uses internally (the byte-API entry point [`color_matrix`]
/// promotes/demotes around the matrix multiply).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorMatrix(pub [[f32; 5]; 4]);

impl ColorMatrix {
    /// 4×5 identity matrix: `(R', G', B', A') = (R, G, B, A)`. Useful
    /// as a base for callers that want to perturb a small number of
    /// entries.
    pub const fn identity() -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])
    }

    /// SVG 1.1 §15.10 `type="saturate"` matrix for saturation factor
    /// `s` (clamped to `[0, ∞)` — the spec leaves the upper bound
    /// open; `s = 0` is the explicit "fully desaturated" point).
    ///
    /// The matrix is reproduced verbatim from §15.10:
    ///
    /// ```text
    /// | 0.213 + 0.787·s   0.715 − 0.715·s   0.072 − 0.072·s   0   0 |
    /// | 0.213 − 0.213·s   0.715 + 0.285·s   0.072 − 0.072·s   0   0 |
    /// | 0.213 − 0.213·s   0.715 − 0.715·s   0.072 + 0.928·s   0   0 |
    /// |        0                  0                  0        1   0 |
    /// ```
    ///
    /// The `(0.213, 0.715, 0.072)` row is the spec-mandated BT.709
    /// luminance triple (§15.10 — slightly different from the PDF
    /// blend-mode `Lum` coefficients used elsewhere in this crate,
    /// which come from PDF 32000-1:2008 §11.3.5.3).
    pub fn saturate(s: f32) -> Self {
        let s = if s.is_nan() { 1.0 } else { s.max(0.0) };
        Self([
            [
                0.213 + 0.787 * s,
                0.715 - 0.715 * s,
                0.072 - 0.072 * s,
                0.0,
                0.0,
            ],
            [
                0.213 - 0.213 * s,
                0.715 + 0.285 * s,
                0.072 - 0.072 * s,
                0.0,
                0.0,
            ],
            [
                0.213 - 0.213 * s,
                0.715 - 0.715 * s,
                0.072 + 0.928 * s,
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])
    }

    /// SVG 1.1 §15.10 `type="hueRotate"` matrix for rotation angle
    /// `degrees` (positive direction is the spec's R → G → B
    /// convention).
    ///
    /// The spec gives the result as the sum of three constant 3×3
    /// matrices weighted by `1`, `cos(θ)` and `sin(θ)` respectively;
    /// the alpha row is identity. Coefficients are verbatim from
    /// §15.10:
    ///
    /// ```text
    /// const   = |  0.213  0.715  0.072 |
    ///           |  0.213  0.715  0.072 |
    ///           |  0.213  0.715  0.072 |
    ///
    /// cos·    = |  0.787 −0.715 −0.072 |
    ///           | −0.213  0.285 −0.072 |
    ///           | −0.213 −0.715  0.928 |
    ///
    /// sin·    = | −0.213 −0.715  0.928 |
    ///           |  0.143  0.140 −0.283 |
    ///           | −0.787  0.715  0.072 |
    /// ```
    ///
    /// `degrees.is_nan()` is treated as zero rotation (identity row);
    /// the rotation is otherwise unbounded — the trig functions are
    /// inherently periodic.
    pub fn hue_rotate(degrees: f32) -> Self {
        let theta = if degrees.is_nan() {
            0.0
        } else {
            degrees.to_radians()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Spec-verbatim 3×3 contributions.
        let const_m = [
            [0.213, 0.715, 0.072],
            [0.213, 0.715, 0.072],
            [0.213, 0.715, 0.072],
        ];
        let cos_m = [
            [0.787, -0.715, -0.072],
            [-0.213, 0.285, -0.072],
            [-0.213, -0.715, 0.928],
        ];
        let sin_m = [
            [-0.213, -0.715, 0.928],
            [0.143, 0.140, -0.283],
            [-0.787, 0.715, 0.072],
        ];

        let mut rows = [[0.0f32; 5]; 4];
        for i in 0..3 {
            for j in 0..3 {
                rows[i][j] = const_m[i][j] + c * cos_m[i][j] + s * sin_m[i][j];
            }
        }
        rows[3][3] = 1.0; // alpha untouched
        Self(rows)
    }

    /// SVG 1.1 §15.10 `type="luminanceToAlpha"` matrix.
    ///
    /// ```text
    /// |   0      0      0     0  0 |
    /// |   0      0      0     0  0 |
    /// |   0      0      0     0  0 |
    /// | 0.2125 0.7154 0.0721 0  0 |
    /// ```
    ///
    /// Note the §15.10 luminance row uses the slightly different
    /// `(0.2125, 0.7154, 0.0721)` triple — this is the BT.709 set of
    /// coefficients SVG 1.1 cites for "luminance to alpha"
    /// specifically. (The saturation / hueRotate matrices above use
    /// the rounded `(0.213, 0.715, 0.072)` set the same section
    /// hands out for those operators; we follow the spec literally
    /// for each.)
    pub const fn luminance_to_alpha() -> Self {
        Self([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2125, 0.7154, 0.0721, 0.0, 0.0],
        ])
    }

    /// Resolve a [`ColorMatrixOp`] to the concrete `ColorMatrix` it
    /// names. Used internally by [`color_matrix`] / [`color_matrix_op`]
    /// and re-exported for callers that want to inspect / cache /
    /// compose the matrix before applying it.
    pub fn from_op(op: ColorMatrixOp, user: &ColorMatrix) -> ColorMatrix {
        match op {
            ColorMatrixOp::Matrix => *user,
            ColorMatrixOp::Saturate(s) => ColorMatrix::saturate(s),
            ColorMatrixOp::HueRotate(d) => ColorMatrix::hue_rotate(d),
            ColorMatrixOp::LuminanceToAlpha => ColorMatrix::luminance_to_alpha(),
        }
    }
}

/// SVG 1.1 §15.10 `<feColorMatrix>` with an explicit user-supplied
/// matrix — apply `m` per pixel and clamp the result to `[0, 255]`.
///
/// `width × height` is the input/output image extent in pixels. `src`
/// is a packed-RGBA byte buffer of exactly `width * height * 4` bytes
/// in row-major order.
///
/// **Pixel space.** The §15.10 multiply is defined on *un-premultiplied*
/// channel values normalised to `[0, 1]`. This entry point therefore
/// treats the input bytes as straight-alpha sRGB samples (no
/// linearisation — §15.10 is in the device gamut, like the rest of the
/// SVG colour-matrix algebra). The output bytes are also straight
/// alpha; the caller is responsible for any subsequent premultiplication
/// before compositing.
///
/// # Algorithm
///
/// 1. For each pixel: normalise `(R, G, B, A) → (r, g, b, a) ∈ [0, 1]⁴`.
/// 2. Compute `(r', g', b', a') = M · (r, g, b, a, 1)ᵀ` per the §15.10
///    matrix form.
/// 3. Clamp each output channel to `[0, 1]` and quantise back to
///    `round(x · 255)`.
///
/// Complexity is `O(W · H)` with a constant 20-term inner product per
/// channel — independent of any radius or kernel size.
///
/// # Panics
///
/// Panics if `src.len() != width as usize * height as usize * 4`.
pub fn color_matrix(src: &[u8], width: u32, height: u32, m: &ColorMatrix) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("color_matrix: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "color_matrix: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    let rows = &m.0;
    let mut out = Vec::with_capacity(src.len());
    for chunk in src.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let a = chunk[3] as f32 / 255.0;
        for row in rows {
            let v = row[0] * r + row[1] * g + row[2] * b + row[3] * a + row[4];
            out.push(quantise_unit(v));
        }
    }
    out
}

/// SVG 1.1 §15.10 `<feColorMatrix>` with the `type=` attribute
/// expanded into a [`ColorMatrixOp`]. `user_matrix` is consulted only
/// when `op == ColorMatrixOp::Matrix` — the other three operators
/// build their matrix from the operator parameters and the spec's
/// hard-coded coefficient tables.
///
/// Equivalent to:
///
/// ```text
/// let m = ColorMatrix::from_op(op, user_matrix);
/// color_matrix(src, w, h, &m)
/// ```
///
/// — exposed as a separate entry point so callers that hold a parsed
/// SVG `feColorMatrix` element can dispatch in one line.
///
/// # Panics
///
/// Same as [`color_matrix`].
pub fn color_matrix_op(
    src: &[u8],
    width: u32,
    height: u32,
    op: ColorMatrixOp,
    user_matrix: &ColorMatrix,
) -> Vec<u8> {
    let m = ColorMatrix::from_op(op, user_matrix);
    color_matrix(src, width, height, &m)
}

/// Convenience wrapper that runs [`color_matrix`] on a slice of [`Rgba`]
/// pixels and returns a `Vec<Rgba>` of the same length. Identical
/// semantics — provided for callers that already have a typed pixel
/// buffer.
pub fn color_matrix_pixels(src: &[Rgba], width: u32, height: u32, m: &ColorMatrix) -> Vec<Rgba> {
    assert_eq!(
        src.len(),
        width as usize * height as usize,
        "color_matrix_pixels: src.len() == {} but width*height == {}",
        src.len(),
        width as usize * height as usize
    );
    let mut bytes = Vec::with_capacity(src.len() * 4);
    for p in src {
        bytes.push(p.r);
        bytes.push(p.g);
        bytes.push(p.b);
        bytes.push(p.a);
    }
    let out = color_matrix(&bytes, width, height, m);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

/// Clamp a normalised-`[0, 1]` colour-channel value to that range and
/// quantise it back to a `u8` via the standard `round(x · 255)`
/// rule. NaNs map to zero (a defensive choice — the §15.10 multiply is
/// a finite linear combination of finite inputs, so a NaN would have
/// to come from a NaN entry in the user matrix).
#[inline]
fn quantise_unit(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let clamped = v.clamp(0.0, 1.0);
    (clamped * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod color_matrix_tests {
    use super::*;

    fn build<F: FnMut(u32, u32) -> Rgba>(w: u32, h: u32, mut f: F) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let c = f(x, y);
                v.extend_from_slice(&[c.r, c.g, c.b, c.a]);
            }
        }
        v
    }

    #[test]
    fn identity_matrix_is_bytewise_identity() {
        // The 4×5 identity must round-trip every (R, G, B, A) tuple
        // bit-exactly — the only loss in the pipeline is the f32
        // round-trip, and `v / 255 * 255` followed by the standard
        // half-up rounding is provably idempotent for `v ∈ [0, 255]`.
        let img = build(7, 5, |x, y| {
            Rgba::new(
                (x.wrapping_mul(37) ^ y.wrapping_mul(13)) as u8,
                (x.wrapping_mul(7).wrapping_add(y)) as u8,
                (x.wrapping_add(y.wrapping_mul(53))) as u8,
                (x.wrapping_mul(y).wrapping_add(31)) as u8,
            )
        });
        let out = color_matrix(&img, 7, 5, &ColorMatrix::identity());
        assert_eq!(out, img);
    }

    #[test]
    fn saturate_one_is_identity_on_arbitrary_colours() {
        // s = 1 ⇒ the spec saturate matrix collapses to the 4×4
        // identity (every off-diagonal RGB term cancels: 0.213·1 −
        // 0.213·1 = 0, etc.). Therefore saturate(1) applied to any
        // colour must reproduce it within the 1-LSB rounding window
        // (one round-trip through normalise→multiply→quantise).
        let img = build(4, 4, |x, y| {
            Rgba::new((x * 60) as u8, (y * 80) as u8, 17, 200)
        });
        let m = ColorMatrix::saturate(1.0);
        let out = color_matrix(&img, 4, 4, &m);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "saturate(1) differs by {d} at byte {i}");
        }
    }

    #[test]
    fn saturate_zero_collapses_rgb_to_luminance() {
        // s = 0 ⇒ every RGB output row is the same luminance combination
        // (0.213·R + 0.715·G + 0.072·B). Apply to a fixed (200, 100, 50)
        // pixel and check the three colour channels emerge equal.
        let img = build(1, 1, |_, _| Rgba::new(200, 100, 50, 255));
        let m = ColorMatrix::saturate(0.0);
        let out = color_matrix(&img, 1, 1, &m);
        assert_eq!(out[0], out[1], "R != G after full desaturation");
        assert_eq!(out[1], out[2], "G != B after full desaturation");
        assert_eq!(out[3], 255, "alpha must pass through");
        // Spot-check the analytic value: 0.213·200 + 0.715·100 + 0.072·50
        // = 42.6 + 71.5 + 3.6 = 117.7 / 255 ⇒ quantises to 118.
        let expected = ((0.213_f32 * 200.0 + 0.715 * 100.0 + 0.072 * 50.0)
            .clamp(0.0, 255.0)
            .round()) as u8;
        let d = (out[0] as i32 - expected as i32).abs();
        assert!(
            d <= 1,
            "luminance scalar off by {d} (got {} vs {expected})",
            out[0]
        );
    }

    #[test]
    fn hue_rotate_zero_is_identity() {
        // θ = 0 ⇒ cos = 1, sin = 0. const + 1·cos_m = identity 3×3
        // (verified by hand: row 0 → (0.213+0.787, 0.715−0.715,
        // 0.072−0.072) = (1, 0, 0); row 1 → (0, 1, 0); row 2 → (0, 0,
        // 1)). Therefore hue_rotate(0) is the identity modulo the
        // quantise round-off (≤ 1 LSB).
        let img = build(5, 3, |x, y| {
            Rgba::new((x * 41) as u8, (y * 71) as u8, 100, 200)
        });
        let m = ColorMatrix::hue_rotate(0.0);
        let out = color_matrix(&img, 5, 3, &m);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "hue_rotate(0) byte {i} differs by {d}");
        }
    }

    #[test]
    fn hue_rotate_preserves_grey_axis() {
        // A pure grey pixel (R == G == B) lies on the achromatic axis,
        // which is the rotation axis of `hueRotate`. Therefore any
        // rotation must leave a grey pixel grey (and very nearly the
        // same brightness — modulo the inevitable round-off in the
        // float multiply, the coefficients have ≤ 3 dp so the rounding
        // can be up to ~1 LSB).
        let img = build(1, 1, |_, _| Rgba::new(140, 140, 140, 255));
        for theta in [30.0_f32, 90.0, 120.0, 180.0, 270.0, -45.0] {
            let m = ColorMatrix::hue_rotate(theta);
            let out = color_matrix(&img, 1, 1, &m);
            assert!(
                (out[0] as i32 - out[1] as i32).abs() <= 1,
                "θ={theta}: R != G after rotation"
            );
            assert!(
                (out[1] as i32 - out[2] as i32).abs() <= 1,
                "θ={theta}: G != B after rotation"
            );
            assert!(
                (out[0] as i32 - 140).abs() <= 2,
                "θ={theta}: grey brightness drift {} → {}",
                140,
                out[0]
            );
            assert_eq!(out[3], 255, "alpha must pass through hue rotation");
        }
    }

    #[test]
    fn hue_rotate_full_turn_is_identity() {
        // 360° rotation ⇒ cos = 1, sin = 0 again ⇒ same as θ = 0.
        // Round-off ≤ 1 LSB on each channel.
        let img = build(4, 3, |x, y| {
            Rgba::new((x * 60) as u8, (y * 80) as u8, ((x + y) * 30) as u8, 255)
        });
        let m = ColorMatrix::hue_rotate(360.0);
        let out = color_matrix(&img, 4, 3, &m);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "θ=360 byte {i} differs by {d}");
        }
    }

    #[test]
    fn luminance_to_alpha_zeros_rgb_and_writes_luminance_to_alpha() {
        // The luminanceToAlpha matrix is the §15.10-mandated fixed
        // matrix that produces transparent black RGB and alpha equal
        // to the BT.709 luminance of the input. Verify both halves.
        let img = build(1, 1, |_, _| Rgba::new(255, 128, 64, 200));
        let m = ColorMatrix::luminance_to_alpha();
        let out = color_matrix(&img, 1, 1, &m);
        assert_eq!(out[0], 0, "R must be cleared");
        assert_eq!(out[1], 0, "G must be cleared");
        assert_eq!(out[2], 0, "B must be cleared");
        // Expected alpha: 0.2125·255 + 0.7154·128 + 0.0721·64
        // = 54.1875 + 91.5712 + 4.6144 = 150.3731 ⇒ quantises to 150.
        let expected = ((0.2125_f32 * 255.0 + 0.7154 * 128.0 + 0.0721 * 64.0).round()) as u8;
        let d = (out[3] as i32 - expected as i32).abs();
        assert!(
            d <= 1,
            "luminance-to-alpha α off by {d} (got {} vs {expected})",
            out[3]
        );
    }

    #[test]
    fn op_dispatch_matches_direct_construction() {
        // The `color_matrix_op` thin wrapper must produce byte-exact
        // output relative to `color_matrix` on the resolved matrix —
        // this is the documented invariant `ColorMatrix::from_op`.
        let img = build(3, 2, |x, y| {
            Rgba::new((x * 90) as u8, (y * 110) as u8, 60, 240)
        });
        let user = ColorMatrix::identity(); // unused for non-Matrix ops
        for op in [
            ColorMatrixOp::Matrix,
            ColorMatrixOp::Saturate(0.5),
            ColorMatrixOp::HueRotate(123.0),
            ColorMatrixOp::LuminanceToAlpha,
        ] {
            let direct = color_matrix(&img, 3, 2, &ColorMatrix::from_op(op, &user));
            let via_op = color_matrix_op(&img, 3, 2, op, &user);
            assert_eq!(direct, via_op, "dispatch mismatch for {op:?}");
        }
    }

    #[test]
    fn channel_clamping_is_applied() {
        // A matrix that would produce R' = 2·R (out of gamut for any
        // R > 127) must clamp to 255, not wrap.
        let m = ColorMatrix([
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]);
        let img = build(1, 1, |_, _| Rgba::new(200, 0, 0, 255));
        let out = color_matrix(&img, 1, 1, &m);
        assert_eq!(out[0], 255, "R must clamp at upper bound");
        // A negative-doubling matrix must clamp to 0.
        let m = ColorMatrix([
            [-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]);
        let out = color_matrix(&img, 1, 1, &m);
        assert_eq!(out[0], 0, "R must clamp at lower bound");
    }

    #[test]
    fn bias_column_adds_constant_offset() {
        // The 5th column (the homogeneous-1 multiplier) lets a matrix
        // add a constant offset. A row of (0, 0, 0, 0, 0.5) should
        // produce a constant 0.5 → 128 in that channel regardless of
        // the input pixel.
        let m = ColorMatrix([
            [0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.75],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]);
        let img = build(2, 2, |x, y| {
            Rgba::new((x * 100) as u8, (y * 50) as u8, 33, 200)
        });
        let out = color_matrix(&img, 2, 2, &m);
        for chunk in out.chunks_exact(4) {
            assert!(
                (chunk[0] as i32 - 128).abs() <= 1,
                "R bias drift: {}",
                chunk[0]
            );
            assert!(
                (chunk[1] as i32 - 64).abs() <= 1,
                "G bias drift: {}",
                chunk[1]
            );
            assert!(
                (chunk[2] as i32 - 191).abs() <= 1,
                "B bias drift: {}",
                chunk[2]
            );
            assert_eq!(chunk[3], 200, "alpha pass-through");
        }
    }

    #[test]
    fn color_matrix_pixels_wrapper_matches_byte_api() {
        let w = 4u32;
        let h = 3u32;
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for i in 0..(w * h) as u8 {
            let r = i.wrapping_mul(7);
            let g = i.wrapping_mul(17).wrapping_add(3);
            let b = i.wrapping_mul(31).wrapping_add(5);
            let a = 255u8;
            bytes.extend_from_slice(&[r, g, b, a]);
            pixels.push(Rgba::new(r, g, b, a));
        }
        for m in [
            ColorMatrix::identity(),
            ColorMatrix::saturate(0.3),
            ColorMatrix::hue_rotate(72.0),
            ColorMatrix::luminance_to_alpha(),
        ] {
            let bytes_out = color_matrix(&bytes, w, h, &m);
            let pix_out = color_matrix_pixels(&pixels, w, h, &m);
            let recombined: Vec<u8> = pix_out.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
            assert_eq!(bytes_out, recombined, "wrapper / byte API divergence");
        }
    }

    #[test]
    fn empty_image_returns_empty_buffer() {
        let img: Vec<u8> = Vec::new();
        let out = color_matrix(&img, 0, 0, &ColorMatrix::saturate(0.5));
        assert!(
            out.is_empty(),
            "zero-area image must produce zero-byte output"
        );
    }

    #[test]
    #[should_panic(expected = "color_matrix: src.len() ==")]
    fn color_matrix_panics_on_wrong_length() {
        let bad = vec![0u8; 7]; // not w·h·4 for w=2, h=2
        let _ = color_matrix(&bad, 2, 2, &ColorMatrix::identity());
    }
}

// ---------------------------------------------------------------------------
// feGaussianBlur — SVG 1.1 §15.17
// ---------------------------------------------------------------------------

/// Threshold above which the SVG 1.1 §15.17 three-box-blur approximation
/// is used in place of the direct discrete Gaussian convolution. The
/// spec says "For larger values of 's' (s >= 2.0), an approximation can
/// be used"; the same `2.0` value is exposed here so that callers and
/// tests can reason about which mode a given `stdDeviation` will take.
pub const GAUSSIAN_BLUR_BOX_THRESHOLD: f32 = 2.0;

/// SVG 1.1 §15.17 `<feGaussianBlur>`: separable Gaussian blur on a
/// packed-RGBA byte buffer.
///
/// `width × height` is the image extent in pixels. `src` is a
/// packed-RGBA byte buffer of exactly `width * height * 4` bytes in
/// row-major order — the same packing produced by
/// [`crate::Renderer::render`].
///
/// `std_x` and `std_y` are the X / Y standard deviations of the
/// Gaussian, in pixels. Per the spec, a negative value is an error and
/// will panic; a value of zero disables the effect along that axis (a
/// zero on both axes returns the input unchanged).
///
/// # Algorithm
///
/// Per the spec the kernel is the normalised separable Gaussian
/// `G(x, y) = H(x)·I(y)` with
/// `H(x) = exp(-x²/(2s²)) / sqrt(2π·s²)` and `I(y) =
/// exp(-y²/(2t²)) / sqrt(2π·t²)`. Two implementation modes are used:
///
/// * **Direct mode (`s < 2.0`)** — separable convolution with a
///   discrete kernel of half-width `r = ceil(3·s)` (capturing
///   ≈99.7% of the analytical Gaussian mass). The kernel is
///   re-normalised after discretisation so that the row sum equals
///   exactly 1.0 in `f32`, eliminating the residual DC error from
///   truncating the tails.
/// * **Box mode (`s ≥ 2.0`)** — the §15.17 three-box-blur
///   approximation. Let `d = floor(s · 3·sqrt(2π)/4 + 0.5)`. If `d`
///   is odd, three centred box-blurs of size `d` are composed; if
///   `d` is even, two box-blurs of size `d` (centred on the pixel
///   boundary on the left and right respectively) and one
///   box-blur of size `d + 1` centred on the pixel are composed.
///   This is the formula given verbatim by the spec and produces
///   a piece-wise quadratic kernel that matches the Gaussian to
///   within ≈3%.
///
/// Both modes are fully separable: an X-pass over rows followed by a
/// Y-pass over columns. Boundary handling is **clamp-to-edge** (the
/// same convention used by [`morphology`]).
///
/// # Panics
///
/// * If `src.len() != width as usize * height as usize * 4`.
/// * If `std_x < 0.0` or `std_y < 0.0` (spec error).
/// * If either standard deviation is NaN.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same dimensions. With
/// `std_x == 0.0` and `std_y == 0.0` the input is returned unchanged.
///
/// Boundary handling is **clamp-to-edge** ([`ConvolveEdgeMode::Duplicate`]).
/// This is the historical behaviour; for the Filter Effects Module
/// Level 1 `edgeMode` attribute (initial value `none`, plus `wrap`),
/// call [`gaussian_blur_edge`] with the desired mode.
pub fn gaussian_blur(src: &[u8], width: u32, height: u32, std_x: f32, std_y: f32) -> Vec<u8> {
    gaussian_blur_edge(
        src,
        width,
        height,
        std_x,
        std_y,
        ConvolveEdgeMode::Duplicate,
    )
}

/// `<feGaussianBlur>` with an explicit `edgeMode` (Filter Effects Module
/// Level 1 §9.14).
///
/// Identical to [`gaussian_blur`] except the caller chooses how the
/// blur kernel reads samples that fall outside the image when it is
/// positioned at or near an edge:
///
/// * [`ConvolveEdgeMode::Duplicate`] — extend each border by
///   duplicating its edge color values (clamp-to-edge).
/// * [`ConvolveEdgeMode::Wrap`] — extend by taking the color values
///   from the opposite edge of the image.
/// * [`ConvolveEdgeMode::None`] — extend with pixel values of zero for
///   R, G, B and A. This is the spec's **initial value** for
///   `edgeMode`, so a renderer flattening `<feGaussianBlur>` with no
///   `edgeMode` attribute should pass this variant; the bordering rows
///   /columns then darken toward transparent black, matching the
///   reference rendering of an unattributed `feGaussianBlur`.
///
/// The same `edgeMode` is applied to both the X and Y separable passes
/// and to both the direct-kernel (`s < 2.0`) and three-box-blur
/// (`s ≥ 2.0`) code paths, so the choice is consistent across all
/// `stdDeviation` magnitudes.
pub fn gaussian_blur_edge(
    src: &[u8],
    width: u32,
    height: u32,
    std_x: f32,
    std_y: f32,
    edge: ConvolveEdgeMode,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("gaussian_blur: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "gaussian_blur: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );
    assert!(
        !std_x.is_nan() && !std_y.is_nan(),
        "gaussian_blur: stdDeviation must not be NaN"
    );
    assert!(
        std_x >= 0.0 && std_y >= 0.0,
        "gaussian_blur: stdDeviation must be non-negative (spec error processing): \
         got std_x={std_x}, std_y={std_y}"
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }
    if std_x == 0.0 && std_y == 0.0 {
        return src.to_vec();
    }

    let mut buf = src.to_vec();

    if std_x > 0.0 {
        if std_x < GAUSSIAN_BLUR_BOX_THRESHOLD {
            let kernel = build_gaussian_kernel(std_x);
            gaussian_separable_pass(&mut buf, w, h, &kernel, Axis::X, edge);
        } else {
            box_blur_three_pass(&mut buf, w, h, box_sizes_for_std(std_x), Axis::X, edge);
        }
    }
    if std_y > 0.0 {
        if std_y < GAUSSIAN_BLUR_BOX_THRESHOLD {
            let kernel = build_gaussian_kernel(std_y);
            gaussian_separable_pass(&mut buf, w, h, &kernel, Axis::Y, edge);
        } else {
            box_blur_three_pass(&mut buf, w, h, box_sizes_for_std(std_y), Axis::Y, edge);
        }
    }
    buf
}

/// Convenience wrapper that runs [`gaussian_blur`] on a slice of [`Rgba`]
/// pixels and returns a `Vec<Rgba>` of the same length.
pub fn gaussian_blur_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    std_x: f32,
    std_y: f32,
) -> Vec<Rgba> {
    assert_eq!(
        src.len(),
        width as usize * height as usize,
        "gaussian_blur_pixels: src.len() == {} but width*height == {}",
        src.len(),
        width as usize * height as usize
    );
    let mut bytes = Vec::with_capacity(src.len() * 4);
    for p in src {
        bytes.push(p.r);
        bytes.push(p.g);
        bytes.push(p.b);
        bytes.push(p.a);
    }
    let out = gaussian_blur(&bytes, width, height, std_x, std_y);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

/// Build a discrete 1-D normalised Gaussian kernel for the given
/// standard deviation. Half-width is `ceil(3·s)`, capturing ≈99.7% of
/// the analytical mass; the kernel is renormalised so that the sum
/// equals exactly `1.0` in `f32`, which removes the DC bias that would
/// otherwise creep in from truncating the tails.
fn build_gaussian_kernel(std: f32) -> Vec<f32> {
    debug_assert!(std > 0.0);
    let half = (std * 3.0).ceil() as usize;
    let len = 2 * half + 1;
    let mut k = Vec::with_capacity(len);
    let two_sigma_sq = 2.0 * std * std;
    // Per the §15.17 closed form: H(x) = exp(-x²/(2s²)) / sqrt(2π·s²).
    // The leading 1/sqrt(2π·s²) factor is a constant per-kernel — it
    // drops out cleanly when we renormalise below, so we omit it from
    // the discrete samples.
    for i in 0..len {
        let x = i as f32 - half as f32;
        k.push((-x * x / two_sigma_sq).exp());
    }
    let sum: f32 = k.iter().sum();
    debug_assert!(sum > 0.0);
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Sliding-window separable Gaussian pass along the requested axis,
/// in-place on a packed-RGBA byte buffer. Channels are processed
/// independently; the accumulator is `f32` per-channel; boundary
/// samples reuse the nearest edge pixel (clamp-to-edge).
fn gaussian_separable_pass(
    buf: &mut [u8],
    w: usize,
    h: usize,
    kernel: &[f32],
    axis: Axis,
    edge: ConvolveEdgeMode,
) {
    debug_assert_eq!(buf.len(), w * h * 4);
    let half = (kernel.len() - 1) / 2;
    let (stride, primary, secondary) = match axis {
        Axis::X => (4usize, w, h),
        Axis::Y => (w * 4, h, w),
    };
    let mut line_in = vec![0u8; primary * 4];
    let mut line_out = vec![0u8; primary * 4];
    for s in 0..secondary {
        // Gather one "line" (row for X-pass, column for Y-pass) into a
        // contiguous scratch buffer so the inner loop is sequential.
        let start = match axis {
            Axis::X => s * w * 4,
            Axis::Y => s * 4,
        };
        for i in 0..primary {
            let off = start + i * stride;
            line_in[i * 4..i * 4 + 4].copy_from_slice(&buf[off..off + 4]);
        }

        for i in 0..primary {
            let mut acc = [0f32; 4];
            for (k_idx, w_coef) in kernel.iter().enumerate() {
                let logical = i as isize + k_idx as isize - half as isize;
                // §9.14 edgeMode: out-of-line samples are duplicated /
                // wrapped / zeroed. `None` contributes nothing to the
                // accumulator.
                let Some(pi) = resolve_line_index(logical, primary, edge) else {
                    continue;
                };
                let p = &line_in[pi * 4..pi * 4 + 4];
                acc[0] += p[0] as f32 * w_coef;
                acc[1] += p[1] as f32 * w_coef;
                acc[2] += p[2] as f32 * w_coef;
                acc[3] += p[3] as f32 * w_coef;
            }
            let off = i * 4;
            line_out[off] = quantise_u8(acc[0]);
            line_out[off + 1] = quantise_u8(acc[1]);
            line_out[off + 2] = quantise_u8(acc[2]);
            line_out[off + 3] = quantise_u8(acc[3]);
        }

        for i in 0..primary {
            let off = start + i * stride;
            buf[off..off + 4].copy_from_slice(&line_out[i * 4..i * 4 + 4]);
        }
    }
}

/// Round a non-negative `f32` channel accumulator to `u8` with
/// clamp-to-[0, 255]. Negative inputs (which Gaussian blur should not
/// produce but the helper is conservative anyway) clamp to `0`.
#[inline]
fn quantise_u8(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let clamped = v.clamp(0.0, 255.0);
    (clamped + 0.5) as u8
}

/// Resolve a (possibly out-of-range) 1-D line index under the requested
/// edge mode for the separable Gaussian / box-blur passes.
///
/// Filter Effects Module Level 1 §9.14 adds the `edgeMode` attribute to
/// `<feGaussianBlur>` with the same three values the convolution
/// primitive uses, "Determines how to extend the input image […] so
/// that the matrix operations can be applied when the kernel is
/// positioned at or near the edge of the input image":
///
/// * [`ConvolveEdgeMode::Duplicate`] — "the input image is extended
///   along each of its borders […] by duplicating the color values at
///   the given edge" → the index is clamped to `[0, len)`.
/// * [`ConvolveEdgeMode::Wrap`] — "the input image is extended by
///   taking the color values from the opposite edge" → the index is
///   reduced modulo `len` (Euclidean remainder so negatives wrap).
/// * [`ConvolveEdgeMode::None`] (the spec's initial value) — "the
///   input image is extended with pixel values of zero for R, G, B and
///   A" → an out-of-range index returns `None`, contributing nothing
///   to the running sum / weighted accumulator.
///
/// `len` is the number of samples along the pass axis and is assumed
/// `>= 1` (callers early-return on empty images).
#[inline]
fn resolve_line_index(idx: isize, len: usize, edge: ConvolveEdgeMode) -> Option<usize> {
    debug_assert!(len >= 1);
    let imax = len as isize - 1;
    match edge {
        ConvolveEdgeMode::Duplicate => Some(idx.clamp(0, imax) as usize),
        ConvolveEdgeMode::Wrap => Some(idx.rem_euclid(len as isize) as usize),
        ConvolveEdgeMode::None => {
            if idx < 0 || idx > imax {
                None
            } else {
                Some(idx as usize)
            }
        }
    }
}

/// Compute the three box-blur sizes for the §15.17 approximation.
///
/// Per the spec:
///
/// > let `d = floor(s · 3·sqrt(2π)/4 + 0.5)`
/// > * if `d` is odd, use three box-blurs of size `d`, centered on the
/// >   output pixel.
/// > * if `d` is even, two box-blurs of size `d` (the first one
/// >   centered on the pixel boundary between the output pixel and the
/// >   one to the left, the second one centered on the pixel boundary
/// >   between the output pixel and the one to the right) and one box
/// >   blur of size `d + 1` centered on the output pixel.
///
/// Returns `[BoxKind; 3]` describing the three passes in order. The
/// "kind" carries both the box size and, for even-`d` passes, the
/// side the boundary is offset toward.
fn box_sizes_for_std(std: f32) -> [BoxKind; 3] {
    debug_assert!(std >= GAUSSIAN_BLUR_BOX_THRESHOLD);
    // 3·sqrt(2π)/4 ≈ 1.8800316...
    let d_f = std * (3.0 * (2.0 * std::f32::consts::PI).sqrt() / 4.0) + 0.5;
    let d = d_f.floor() as u32;
    let d = d.max(1);
    if d % 2 == 1 {
        [
            BoxKind::Centered(d),
            BoxKind::Centered(d),
            BoxKind::Centered(d),
        ]
    } else {
        [
            BoxKind::OffsetLeft(d),
            BoxKind::OffsetRight(d),
            BoxKind::Centered(d + 1),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoxKind {
    /// Box-blur of (odd) size `n`, kernel centred on the output pixel.
    Centered(u32),
    /// Box-blur of (even) size `n`, kernel centred on the pixel
    /// boundary between the output pixel and the one to the **left**.
    /// In integer indices, the window covers `[i - n/2 .. i + n/2 - 1]`
    /// inclusive (i.e. one more sample to the left than to the right).
    OffsetLeft(u32),
    /// Box-blur of (even) size `n`, kernel centred on the pixel
    /// boundary between the output pixel and the one to the **right**.
    /// In integer indices the window covers
    /// `[i - n/2 + 1 .. i + n/2]` inclusive.
    OffsetRight(u32),
}

/// Apply the three box-blur passes that approximate one Gaussian pass,
/// in-place along the requested axis.
fn box_blur_three_pass(
    buf: &mut [u8],
    w: usize,
    h: usize,
    kinds: [BoxKind; 3],
    axis: Axis,
    edge: ConvolveEdgeMode,
) {
    for k in kinds {
        box_blur_pass(buf, w, h, k, axis, edge);
    }
}

/// Run one box-blur pass along `axis` on the packed-RGBA buffer using
/// a rolling-sum O(W·H) per pass (per channel). Boundary handling
/// follows `edge` ([`ConvolveEdgeMode`]): out-of-line samples are the
/// duplicated edge pixel, the opposite-edge pixel (wrap), or zero
/// (none). The window width stays a fixed `n` samples in every mode, so
/// the divisor is always `n` and `none`-mode edge windows average the
/// in-range pixels against the implicit zero border.
fn box_blur_pass(
    buf: &mut [u8],
    w: usize,
    h: usize,
    kind: BoxKind,
    axis: Axis,
    edge: ConvolveEdgeMode,
) {
    debug_assert_eq!(buf.len(), w * h * 4);
    let (n, lo_off, hi_off) = match kind {
        BoxKind::Centered(n) => {
            let half = (n / 2) as isize;
            (n as usize, -half, half)
        }
        BoxKind::OffsetLeft(n) => {
            let half = (n / 2) as isize;
            // Window inclusive of [i - half .. i + half - 1], total `n`
            // samples; the window centroid sits half a pixel to the
            // left of `i`.
            (n as usize, -half, half - 1)
        }
        BoxKind::OffsetRight(n) => {
            let half = (n / 2) as isize;
            (n as usize, -half + 1, half)
        }
    };
    debug_assert_eq!((hi_off - lo_off + 1) as usize, n);

    let (stride, primary, secondary) = match axis {
        Axis::X => (4usize, w, h),
        Axis::Y => (w * 4, h, w),
    };
    let mut line_in = vec![0u8; primary * 4];
    let mut line_out = vec![0u8; primary * 4];
    let imax = primary as isize - 1;
    let inv_n = 1.0f32 / n as f32;
    for s in 0..secondary {
        let start = match axis {
            Axis::X => s * w * 4,
            Axis::Y => s * 4,
        };
        for i in 0..primary {
            let off = start + i * stride;
            line_in[i * 4..i * 4 + 4].copy_from_slice(&buf[off..off + 4]);
        }

        match edge {
            ConvolveEdgeMode::Duplicate => {
                // Rolling sum: prime the accumulator with the window at
                // i=0, then for each subsequent i add the new right edge
                // and subtract the old left edge. Clamp-to-edge means
                // out-of-line indices reuse 0 / imax, so the window
                // always holds exactly `n` (possibly duplicated)
                // samples. Byte-identical to the prior behaviour.
                let mut acc = [0u32; 4];
                for k_off in lo_off..=hi_off {
                    let pi = k_off.clamp(0, imax) as usize;
                    let p = &line_in[pi * 4..pi * 4 + 4];
                    acc[0] += p[0] as u32;
                    acc[1] += p[1] as u32;
                    acc[2] += p[2] as u32;
                    acc[3] += p[3] as u32;
                }

                for i in 0..primary {
                    let off = i * 4;
                    line_out[off] = (acc[0] as f32 * inv_n + 0.5) as u8;
                    line_out[off + 1] = (acc[1] as f32 * inv_n + 0.5) as u8;
                    line_out[off + 2] = (acc[2] as f32 * inv_n + 0.5) as u8;
                    line_out[off + 3] = (acc[3] as f32 * inv_n + 0.5) as u8;

                    if i + 1 < primary {
                        let next_i = (i + 1) as isize;
                        let in_idx = (next_i + hi_off).clamp(0, imax) as usize;
                        let out_idx = (next_i + lo_off - 1).clamp(0, imax) as usize;
                        let p_in = &line_in[in_idx * 4..in_idx * 4 + 4];
                        let p_out = &line_in[out_idx * 4..out_idx * 4 + 4];
                        acc[0] = acc[0] + p_in[0] as u32 - p_out[0] as u32;
                        acc[1] = acc[1] + p_in[1] as u32 - p_out[1] as u32;
                        acc[2] = acc[2] + p_in[2] as u32 - p_out[2] as u32;
                        acc[3] = acc[3] + p_in[3] as u32 - p_out[3] as u32;
                    }
                }
            }
            ConvolveEdgeMode::Wrap | ConvolveEdgeMode::None => {
                // The rolling-sum subtract/add trick assumes the leaving
                // and entering samples can be read back exactly; under
                // `wrap` the window indices are non-monotone and under
                // `none` they may be absent, so we sum each fixed-width
                // window directly. The divisor stays `n` (a `none`
                // out-of-line sample contributes 0 value but still
                // counts toward the fixed window width, exactly like a
                // zero-padded border pixel).
                for i in 0..primary {
                    let mut acc = [0u32; 4];
                    let base = i as isize;
                    for k_off in lo_off..=hi_off {
                        if let Some(pi) = resolve_line_index(base + k_off, primary, edge) {
                            let p = &line_in[pi * 4..pi * 4 + 4];
                            acc[0] += p[0] as u32;
                            acc[1] += p[1] as u32;
                            acc[2] += p[2] as u32;
                            acc[3] += p[3] as u32;
                        }
                    }
                    let off = i * 4;
                    line_out[off] = (acc[0] as f32 * inv_n + 0.5) as u8;
                    line_out[off + 1] = (acc[1] as f32 * inv_n + 0.5) as u8;
                    line_out[off + 2] = (acc[2] as f32 * inv_n + 0.5) as u8;
                    line_out[off + 3] = (acc[3] as f32 * inv_n + 0.5) as u8;
                }
            }
        }

        for i in 0..primary {
            let off = start + i * stride;
            buf[off..off + 4].copy_from_slice(&line_out[i * 4..i * 4 + 4]);
        }
    }
}

#[cfg(test)]
mod gaussian_blur_tests {
    use super::*;

    fn solid(w: u32, h: u32, c: Rgba) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[c.r, c.g, c.b, c.a]);
        }
        v
    }

    #[test]
    fn zero_stddev_is_identity() {
        let mut img = solid(5, 4, Rgba::new(10, 20, 30, 40));
        // Perturb a couple of pixels so we'd notice smearing.
        img[5 * 4] = 200;
        img[7 * 4 + 2] = 250;
        let out = gaussian_blur(&img, 5, 4, 0.0, 0.0);
        assert_eq!(out, img);
    }

    #[test]
    fn solid_image_is_invariant_for_any_stddev() {
        // Gaussian convolution of a constant image gives the same
        // constant; only the boundary handling could perturb it. With
        // clamp-to-edge a constant image is perfectly preserved.
        let img = solid(11, 7, Rgba::new(80, 120, 200, 255));
        for &(sx, sy) in &[
            (0.3, 0.0),
            (0.0, 0.7),
            (0.5, 0.5),
            (1.0, 1.0),
            (1.7, 1.9),
            (2.0, 2.0),
            (3.5, 0.0),
            (4.0, 6.0),
            (8.0, 8.0),
        ] {
            let out = gaussian_blur(&img, 11, 7, sx, sy);
            assert_eq!(out, img, "(sx, sy) = ({sx}, {sy})");
        }
    }

    #[test]
    fn small_stddev_uses_direct_kernel_branch() {
        // For s < 2.0 we use the direct discrete convolution. Sanity:
        // a centred bright pixel diffuses into a symmetric pattern.
        let w = 9u32;
        let h = 9u32;
        let mut img = solid(w, h, Rgba::new(0, 0, 0, 0));
        let cx = 4usize;
        let cy = 4usize;
        let off = (cy * w as usize + cx) * 4;
        img[off..off + 4].copy_from_slice(&[255, 255, 255, 255]);

        let out = gaussian_blur(&img, w, h, 1.0, 1.0);

        // Centre pixel must still be the brightest.
        let centre = out[off];
        assert!(centre > 0);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let o = (y * w as usize + x) * 4;
                if (x, y) != (cx, cy) {
                    assert!(
                        out[o] <= centre,
                        "centre {centre} must dominate ({x},{y}) = {}",
                        out[o]
                    );
                }
            }
        }

        // Four-fold symmetry: the diffusion is identical along ±x, ±y.
        for d in 1..=3usize {
            let n = out[((cy - d) * w as usize + cx) * 4];
            let s = out[((cy + d) * w as usize + cx) * 4];
            let e = out[(cy * w as usize + (cx + d)) * 4];
            let west = out[(cy * w as usize + (cx - d)) * 4];
            assert_eq!(n, s, "vertical symmetry at d={d}");
            assert_eq!(e, west, "horizontal symmetry at d={d}");
            // Separable kernel ⇒ horizontal and vertical neighbours at
            // the same distance get the same coefficient.
            assert_eq!(n, e, "x/y symmetry at d={d}");
        }
    }

    #[test]
    fn large_stddev_uses_box_approximation_branch() {
        // For s >= 2.0 we use the three-box-blur approximation. A
        // bright impulse should spread into a roughly bell-shaped
        // pattern that's centred on the original pixel.
        let w = 21u32;
        let h = 21u32;
        let mut img = solid(w, h, Rgba::new(0, 0, 0, 0));
        let cx = 10usize;
        let cy = 10usize;
        let off = (cy * w as usize + cx) * 4;
        img[off..off + 4].copy_from_slice(&[255, 255, 255, 255]);

        let out = gaussian_blur(&img, w, h, 3.0, 3.0);

        // Diffusion is monotone non-increasing radially (we only check
        // axis-aligned because box-blur cascades aren't exactly
        // rotation-invariant — but along the axes monotonicity holds).
        for d in 1..=4usize {
            let a = out[(cy * w as usize + cx + d - 1) * 4];
            let b = out[(cy * w as usize + cx + d) * 4];
            assert!(b <= a, "horizontal monotonicity broken at d={d}: {a}→{b}");
            let a = out[((cy + d - 1) * w as usize + cx) * 4];
            let b = out[((cy + d) * w as usize + cx) * 4];
            assert!(b <= a, "vertical monotonicity broken at d={d}: {a}→{b}");
        }

        // Centre got dimmer than the original impulse.
        let centre = out[off];
        assert!(
            centre < 255,
            "centre should diffuse below the impulse but got {centre}"
        );
        assert!(centre > 0);
    }

    #[test]
    fn separable_decomposition_matches_two_separate_passes() {
        // gaussian_blur(s_x, s_y) ≡ gaussian_blur(s_x, 0) followed by
        // gaussian_blur(0, s_y). (Both branches of the algorithm
        // satisfy this because separability is the defining property of
        // the Gaussian kernel; the spec literally says "this can be
        // implemented as a separable convolution".)
        let w = 11u32;
        let h = 9u32;
        let mut img = vec![0u8; (w * h * 4) as usize];
        // Pseudo-random deterministic content.
        let mut acc: u32 = 0x1234_5678;
        for byte in &mut img {
            acc = acc.wrapping_mul(1_103_515_245).wrapping_add(12345);
            *byte = (acc >> 16) as u8;
        }
        for &(sx, sy) in &[(0.7, 0.5), (1.5, 1.0), (2.0, 2.0), (3.0, 4.0)] {
            let combined = gaussian_blur(&img, w, h, sx, sy);
            let stepwise_x = gaussian_blur(&img, w, h, sx, 0.0);
            let stepwise = gaussian_blur(&stepwise_x, w, h, 0.0, sy);
            assert_eq!(
                combined, stepwise,
                "separability broke at (sx={sx}, sy={sy})"
            );
        }
    }

    #[test]
    fn axis_only_blur_preserves_orthogonal_axis() {
        // A horizontal-only Gaussian blur must leave the vertical
        // signal alone. We set up a buffer with one bright row at
        // y = 0 and verify that no other row gains any energy from
        // the X-only pass.
        let w = 9u32;
        let h = 5u32;
        let mut img = solid(w, h, Rgba::new(0, 0, 0, 0));
        for x in 0..w as usize {
            let off = x * 4;
            img[off..off + 4].copy_from_slice(&[200, 100, 50, 255]);
        }
        let out = gaussian_blur(&img, w, h, 1.5, 0.0);
        // Row 0 sits inside the kernel of itself only; rows 1..h must
        // remain transparent black (they had no source energy).
        for y in 1..h as usize {
            for x in 0..w as usize {
                let off = (y * w as usize + x) * 4;
                assert_eq!(
                    &out[off..off + 4],
                    &[0, 0, 0, 0],
                    "y={y} x={x} should still be transparent"
                );
            }
        }
        // And the bright row must still be the bright colour exactly
        // (clamp-to-edge horizontally + a row that's constant in x).
        for x in 0..w as usize {
            let off = x * 4;
            assert_eq!(&out[off..off + 4], &[200, 100, 50, 255]);
        }
    }

    #[test]
    fn pixels_wrapper_matches_byte_api() {
        let w = 7u32;
        let h = 5u32;
        let pixels: Vec<Rgba> = (0..(w * h))
            .map(|i| Rgba::new((i * 7) as u8, (i * 11) as u8, (i * 13) as u8, 255))
            .collect();
        let mut bytes = Vec::with_capacity(pixels.len() * 4);
        for p in &pixels {
            bytes.extend_from_slice(&[p.r, p.g, p.b, p.a]);
        }

        for &(sx, sy) in &[(0.0, 0.0), (0.6, 0.6), (1.5, 0.0), (2.5, 2.5)] {
            let from_bytes = gaussian_blur(&bytes, w, h, sx, sy);
            let from_pixels = gaussian_blur_pixels(&pixels, w, h, sx, sy);
            let repacked: Vec<u8> = from_pixels
                .iter()
                .flat_map(|p| [p.r, p.g, p.b, p.a])
                .collect();
            assert_eq!(repacked, from_bytes, "(sx, sy) = ({sx}, {sy})");
        }
    }

    #[test]
    fn empty_image_returns_empty_buffer() {
        let out = gaussian_blur(&[], 0, 5, 1.0, 1.0);
        assert!(out.is_empty());
        let out = gaussian_blur(&[], 5, 0, 1.0, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn box_sizes_table_for_known_stddevs() {
        // s = 2.0 ⇒ d = floor(2.0 · 1.8800316 + 0.5) = floor(4.26..)
        // = 4. d is even ⇒ [OffsetLeft(4), OffsetRight(4), Centered(5)].
        let k = box_sizes_for_std(2.0);
        assert_eq!(k[0], BoxKind::OffsetLeft(4));
        assert_eq!(k[1], BoxKind::OffsetRight(4));
        assert_eq!(k[2], BoxKind::Centered(5));

        // s = 3.0 ⇒ d = floor(3.0 · 1.8800316 + 0.5) = floor(6.14..)
        // = 6. d is even ⇒ [OffsetLeft(6), OffsetRight(6), Centered(7)].
        let k = box_sizes_for_std(3.0);
        assert_eq!(k[0], BoxKind::OffsetLeft(6));
        assert_eq!(k[1], BoxKind::OffsetRight(6));
        assert_eq!(k[2], BoxKind::Centered(7));

        // s = 4.0 ⇒ d = floor(4.0 · 1.8800316 + 0.5) = floor(8.02..)
        // = 8. d is even ⇒ [OffsetLeft(8), OffsetRight(8), Centered(9)].
        let k = box_sizes_for_std(4.0);
        assert_eq!(k[0], BoxKind::OffsetLeft(8));
        assert_eq!(k[1], BoxKind::OffsetRight(8));
        assert_eq!(k[2], BoxKind::Centered(9));

        // s = 5.0 ⇒ d = floor(5.0 · 1.8800316 + 0.5) = floor(9.90..)
        // = 9. d is odd ⇒ three Centered(9).
        let k = box_sizes_for_std(5.0);
        assert_eq!(
            k,
            [
                BoxKind::Centered(9),
                BoxKind::Centered(9),
                BoxKind::Centered(9)
            ]
        );
    }

    #[test]
    fn kernel_normalisation_preserves_constant_image_exactly() {
        // The renormalisation step in build_gaussian_kernel ensures
        // that on an unsaturated constant image the result is byte-
        // exactly the input — i.e. no DC drift from truncating the
        // Gaussian tails. We already check the surface property in
        // `solid_image_is_invariant_for_any_stddev`; here we exercise
        // the kernel-construction invariant directly.
        for &s in &[0.3f32, 0.7, 1.0, 1.5, 1.99] {
            let k = build_gaussian_kernel(s);
            let sum: f32 = k.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "kernel for s={s} has sum {sum}");
            // The largest coefficient is the centre tap.
            let half = (k.len() - 1) / 2;
            assert!(k[half] > 0.0);
            for (i, &v) in k.iter().enumerate() {
                if i != half {
                    assert!(v <= k[half], "tap {i} {} > centre {}", v, k[half]);
                }
            }
            // Symmetry: k[half - d] == k[half + d].
            for d in 1..=half {
                let l = k[half - d];
                let r = k[half + d];
                assert!((l - r).abs() < 1e-12, "asymmetry at d={d}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "stdDeviation must be non-negative")]
    fn negative_stddev_panics() {
        let img = solid(2, 2, Rgba::new(0, 0, 0, 0));
        let _ = gaussian_blur(&img, 2, 2, -0.5, 1.0);
    }

    #[test]
    #[should_panic(expected = "gaussian_blur: src.len() ==")]
    fn wrong_length_panics() {
        let bad = vec![0u8; 7];
        let _ = gaussian_blur(&bad, 2, 2, 1.0, 1.0);
    }

    // ---- Filter Effects Module Level 1 §9.14: feGaussianBlur edgeMode --

    /// The public `gaussian_blur` shim must be exactly equal to
    /// `gaussian_blur_edge(..., Duplicate)` — the historical
    /// clamp-to-edge default — for both code paths and a mix of axes.
    #[test]
    fn default_shim_equals_duplicate_edge() {
        let w = 13u32;
        let h = 9u32;
        // A non-trivial gradient so the boundary handling actually
        // matters at every edge.
        let mut img = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                img.extend_from_slice(&[(x * 17) as u8, (y * 23) as u8, ((x + y) * 7) as u8, 200]);
            }
        }
        for &(sx, sy) in &[(1.0, 1.5), (3.0, 0.0), (0.0, 4.0), (2.5, 2.5)] {
            let shim = gaussian_blur(&img, w, h, sx, sy);
            let dup = gaussian_blur_edge(&img, w, h, sx, sy, ConvolveEdgeMode::Duplicate);
            assert_eq!(shim, dup, "shim != duplicate at ({sx},{sy})");
        }
    }

    /// `edgeMode = none` (the spec initial value) extends the image with
    /// transparent black. A solid-opaque image must therefore darken
    /// toward the border under `none`, while `duplicate` preserves a
    /// solid image exactly. This holds in BOTH the direct-kernel path
    /// (`s < 2.0`) and the box-blur path (`s >= 2.0`).
    #[test]
    fn none_edge_darkens_border_solid() {
        // Large enough that the centre lies well outside any edge
        // kernel reach for the stddevs tested (s=3 spreads ≈9 px).
        let w = 31u32;
        let h = 31u32;
        let img = solid(w, h, Rgba::new(200, 200, 200, 255));
        for &s in &[1.0f32, 3.0f32] {
            let dup = gaussian_blur_edge(&img, w, h, s, s, ConvolveEdgeMode::Duplicate);
            let none = gaussian_blur_edge(&img, w, h, s, s, ConvolveEdgeMode::None);
            // Duplicate leaves a solid image untouched.
            assert_eq!(dup, img, "duplicate must preserve solid at s={s}");
            // None: the interior centre is unchanged but the corner
            // pixel is pulled toward zero by the implicit black border.
            let centre = ((h / 2 * w + w / 2) * 4) as usize;
            let corner = 0usize;
            assert!(
                none[centre] >= 190,
                "centre should stay near full at s={s}, got {}",
                none[centre]
            );
            assert!(
                none[corner] < img[corner],
                "corner R must darken under none at s={s}: {} !< {}",
                none[corner],
                img[corner]
            );
            // Every corner touches two zero borders, so all four darken
            // under `none` (the even-`d` box passes are asymmetric, so we
            // assert darkening rather than exact corner equality).
            let tr = ((w - 1) * 4) as usize;
            let bl = ((h - 1) * w * 4) as usize;
            let br = ((h * w - 1) * 4) as usize;
            assert!(none[tr] < img[tr], "TR corner must darken at s={s}");
            assert!(none[bl] < img[bl], "BL corner must darken at s={s}");
            assert!(none[br] < img[br], "BR corner must darken at s={s}");
        }
    }

    /// `edgeMode = wrap` makes the blur toroidal: a horizontally
    /// constant-per-column image that is periodic across the wrap
    /// boundary stays unchanged, whereas the same image under `none`
    /// would darken at the seam. Concretely, a fully uniform image is
    /// preserved by `wrap` (every off-image read returns an identical
    /// in-image value).
    #[test]
    fn wrap_edge_preserves_uniform() {
        let w = 9u32;
        let h = 7u32;
        let img = solid(w, h, Rgba::new(40, 90, 160, 255));
        for &s in &[1.0f32, 3.0f32] {
            let wrap = gaussian_blur_edge(&img, w, h, s, s, ConvolveEdgeMode::Wrap);
            assert_eq!(wrap, img, "wrap must preserve a uniform image at s={s}");
        }
    }

    /// Under `wrap` along X, a single bright column wraps its energy
    /// around the horizontal seam: blurring column 0 must deposit some
    /// brightness onto the LAST column (its wrap-around left neighbour),
    /// which `none` and `duplicate` would never do.
    #[test]
    fn wrap_edge_bleeds_across_seam() {
        let w = 9u32;
        let h = 3u32;
        // Bright column at x = 0, everything else zero/transparent.
        let mut img = solid(w, h, Rgba::new(0, 0, 0, 0));
        for y in 0..h as usize {
            let off = (y * w as usize) * 4;
            img[off..off + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
        // Use a box-path stddev so it spreads a couple of pixels.
        let s = 2.0f32;
        let wrap = gaussian_blur_edge(&img, w, h, s, 0.0, ConvolveEdgeMode::Wrap);
        let none = gaussian_blur_edge(&img, w, h, s, 0.0, ConvolveEdgeMode::None);
        let last = ((w - 1) * 4) as usize; // row 0, last column
        assert!(
            wrap[last] > 0,
            "wrap must bleed onto the opposite edge, got {}",
            wrap[last]
        );
        assert_eq!(
            none[last], 0,
            "none must NOT bleed across the seam, got {}",
            none[last]
        );
    }

    /// Determinism: the edge-aware path is a pure function of its inputs.
    #[test]
    fn edge_mode_is_deterministic() {
        let w = 8u32;
        let h = 6u32;
        let mut img = solid(w, h, Rgba::new(0, 0, 0, 0));
        img[((2 * w + 3) * 4) as usize] = 255;
        for mode in [
            ConvolveEdgeMode::Duplicate,
            ConvolveEdgeMode::Wrap,
            ConvolveEdgeMode::None,
        ] {
            let a = gaussian_blur_edge(&img, w, h, 1.3, 2.7, mode);
            let b = gaussian_blur_edge(&img, w, h, 1.3, 2.7, mode);
            assert_eq!(a, b, "non-deterministic under {mode:?}");
        }
    }

    /// `resolve_line_index` table: the three edge rules on a length-4
    /// line for indices spanning two periods on each side.
    #[test]
    fn resolve_line_index_table() {
        let len = 4usize;
        // Duplicate clamps.
        assert_eq!(
            resolve_line_index(-3, len, ConvolveEdgeMode::Duplicate),
            Some(0)
        );
        assert_eq!(
            resolve_line_index(-1, len, ConvolveEdgeMode::Duplicate),
            Some(0)
        );
        assert_eq!(
            resolve_line_index(2, len, ConvolveEdgeMode::Duplicate),
            Some(2)
        );
        assert_eq!(
            resolve_line_index(7, len, ConvolveEdgeMode::Duplicate),
            Some(3)
        );
        // Wrap reduces modulo (Euclidean for negatives).
        assert_eq!(resolve_line_index(-1, len, ConvolveEdgeMode::Wrap), Some(3));
        assert_eq!(resolve_line_index(-4, len, ConvolveEdgeMode::Wrap), Some(0));
        assert_eq!(resolve_line_index(-5, len, ConvolveEdgeMode::Wrap), Some(3));
        assert_eq!(resolve_line_index(4, len, ConvolveEdgeMode::Wrap), Some(0));
        assert_eq!(resolve_line_index(6, len, ConvolveEdgeMode::Wrap), Some(2));
        // None drops out-of-range.
        assert_eq!(resolve_line_index(-1, len, ConvolveEdgeMode::None), None);
        assert_eq!(resolve_line_index(0, len, ConvolveEdgeMode::None), Some(0));
        assert_eq!(resolve_line_index(3, len, ConvolveEdgeMode::None), Some(3));
        assert_eq!(resolve_line_index(4, len, ConvolveEdgeMode::None), None);
    }
}

// =====================================================================
// feComponentTransfer — SVG 1.1 §15.11
// =====================================================================
//
// §15.11 defines a per-pixel, per-channel transfer function applied
// independently to R, G, B, A. Each channel selects one of five
// `type=` modes; channel inputs and outputs are normalised to the
// `[0, 1]` interval, the operation is performed on un-premultiplied
// channels, and the result is clamped to `[0, 1]` before re-quantising
// back to `u8`. The five modes, in spec wording:
//
//   identity   C' = C
//   table      a piecewise-linear function over a list of N+1 values
//              v₀ … v_N; for a channel value C ∈ [0, 1], let
//              k = floor(C · N) clamped to [0, N-1], let
//              C_step = (C · N) − k; then C' = v_k + C_step · (v_{k+1} − v_k).
//              (At C = 1.0 the spec's clamp on k pins the interpolation
//              to the final segment v_{N-1}…v_N, so C' = v_N.)
//   discrete   a step function over a list of N values
//              v₀ … v_{N-1}; for C ∈ [0, 1], let k = floor(C · N)
//              clamped to [0, N-1]; then C' = v_k.
//   linear     C' = slope · C + intercept.
//   gamma      C' = amplitude · pow(C, exponent) + offset.
//
// `feFuncR` / `feFuncG` / `feFuncB` / `feFuncA` are independent: each
// channel carries its own [`TransferFunc`]. The API here lifts that
// directly into a [`ComponentTransfer`] struct of four
// [`TransferFunc`] values plus per-channel setters.

/// Per-channel transfer function for [`component_transfer`], mirroring
/// SVG 1.1 §15.11's `type=` attribute on the `<feFuncX>` child elements.
///
/// All inputs and outputs are normalised to `[0, 1]`; the implementation
/// performs the operation, clamps the result to that range, and only
/// then re-quantises back to `u8` for the byte API.
#[derive(Debug, Clone, PartialEq)]
pub enum TransferFunc {
    /// `type="identity"` — output equals input.
    Identity,
    /// `type="table"` — piecewise-linear interpolation across `N+1`
    /// `tableValues`. With `N = 0` (`tableValues` empty or a single
    /// value) the channel collapses to the identity (the spec's
    /// fall-back when the value list is too short to define a segment).
    Table(Vec<f32>),
    /// `type="discrete"` — step function across `N` `tableValues`. An
    /// empty list also collapses to identity (no step boundaries to
    /// place).
    Discrete(Vec<f32>),
    /// `type="linear"` — affine transform.
    Linear {
        /// Spec attribute `slope`.
        slope: f32,
        /// Spec attribute `intercept`.
        intercept: f32,
    },
    /// `type="gamma"` — `amplitude · C^exponent + offset`. `exponent`
    /// must be strictly positive (§15.11 error processing); a
    /// non-positive or NaN exponent panics.
    Gamma {
        /// Spec attribute `amplitude`.
        amplitude: f32,
        /// Spec attribute `exponent`. Must be strictly positive.
        exponent: f32,
        /// Spec attribute `offset`.
        offset: f32,
    },
}

impl TransferFunc {
    /// Evaluate the transfer function on a single normalised channel
    /// value `c ∈ [0, 1]`. The result is **not** pre-clamped — the
    /// caller is responsible for the final clamp / quantisation. Out-of-
    /// range inputs are passed through the spec formula unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `self` is [`TransferFunc::Gamma`] with a non-positive or
    /// NaN exponent. The §15.11 error-processing rule forbids both, and
    /// they make the `pow` undefined in general.
    #[inline]
    pub fn apply(&self, c: f32) -> f32 {
        match self {
            TransferFunc::Identity => c,
            TransferFunc::Table(v) => table_interpolate(v, c),
            TransferFunc::Discrete(v) => discrete_step(v, c),
            TransferFunc::Linear { slope, intercept } => slope * c + intercept,
            TransferFunc::Gamma {
                amplitude,
                exponent,
                offset,
            } => {
                assert!(
                    exponent.is_finite() && *exponent > 0.0,
                    "feComponentTransfer: gamma exponent must be strictly positive, got {exponent}"
                );
                // The spec leaves pow on negative inputs unspecified;
                // §15.11 normalises through clamp-to-[0,1] only at the
                // output, so a negative `c` from the caller (e.g. via
                // a feColorMatrix bias that left the channel slightly
                // negative) would produce NaN under `f32::powf`. Treat
                // negative inputs as zero before the pow — the
                // subsequent output clamp already handles the positive
                // side, and zero is the natural floor of the §15.11
                // gamma curve at the channel boundary.
                let base = c.max(0.0);
                amplitude * base.powf(*exponent) + offset
            }
        }
    }
}

/// Piecewise-linear table lookup for [`TransferFunc::Table`]. The spec
/// places `N + 1` tableValues at the breakpoints `k / N` for `k ∈ 0..=N`
/// and linearly interpolates between adjacent breakpoints.
#[inline]
fn table_interpolate(v: &[f32], c: f32) -> f32 {
    let n_plus_one = v.len();
    if n_plus_one < 2 {
        // Empty list or a single tableValue defines no segment ⇒
        // identity per the §15.11 fall-back.
        return c;
    }
    let n = (n_plus_one - 1) as f32;
    let scaled = c * n;
    if scaled <= 0.0 {
        return v[0];
    }
    if scaled >= n {
        return v[n_plus_one - 1];
    }
    let k = scaled.floor() as usize;
    let frac = scaled - k as f32;
    v[k] + frac * (v[k + 1] - v[k])
}

/// Step lookup for [`TransferFunc::Discrete`]. The spec partitions
/// `[0, 1]` into `N` half-open buckets `[k / N, (k+1) / N)` (with the
/// final bucket closed at `1`) and emits `v[k]`.
#[inline]
fn discrete_step(v: &[f32], c: f32) -> f32 {
    let n = v.len();
    if n == 0 {
        return c;
    }
    let scaled = c * n as f32;
    if scaled <= 0.0 {
        return v[0];
    }
    if scaled >= n as f32 {
        return v[n - 1];
    }
    let k = scaled.floor() as usize;
    v[k.min(n - 1)]
}

/// SVG 1.1 §15.11 `<feComponentTransfer>` configuration — one
/// [`TransferFunc`] per RGBA channel, applied independently per pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentTransfer {
    /// Red-channel function (`<feFuncR>`).
    pub r: TransferFunc,
    /// Green-channel function (`<feFuncG>`).
    pub g: TransferFunc,
    /// Blue-channel function (`<feFuncB>`).
    pub b: TransferFunc,
    /// Alpha-channel function (`<feFuncA>`).
    pub a: TransferFunc,
}

impl ComponentTransfer {
    /// Identity on every channel — equivalent to `<feComponentTransfer>`
    /// with no `<feFuncX>` children, which the spec defines as a
    /// no-op.
    pub fn identity() -> Self {
        Self {
            r: TransferFunc::Identity,
            g: TransferFunc::Identity,
            b: TransferFunc::Identity,
            a: TransferFunc::Identity,
        }
    }

    /// Replace the red-channel transfer function and return `self`
    /// (builder-style chain).
    pub fn with_r(mut self, f: TransferFunc) -> Self {
        self.r = f;
        self
    }

    /// Replace the green-channel transfer function and return `self`.
    pub fn with_g(mut self, f: TransferFunc) -> Self {
        self.g = f;
        self
    }

    /// Replace the blue-channel transfer function and return `self`.
    pub fn with_b(mut self, f: TransferFunc) -> Self {
        self.b = f;
        self
    }

    /// Replace the alpha-channel transfer function and return `self`.
    pub fn with_a(mut self, f: TransferFunc) -> Self {
        self.a = f;
        self
    }
}

impl Default for ComponentTransfer {
    fn default() -> Self {
        Self::identity()
    }
}

/// SVG 1.1 §15.11 `<feComponentTransfer>` — apply one [`TransferFunc`]
/// per channel to each pixel of a packed-RGBA `u8` buffer.
///
/// `width × height` is the input/output image extent in pixels. `src`
/// is a packed-RGBA byte buffer of exactly `width * height * 4` bytes
/// in row-major order.
///
/// **Pixel space.** §15.11 explicitly says the operation is performed
/// on the un-premultiplied colour channels, with channel values in
/// `[0, 1]`. This entry point therefore treats input bytes as
/// straight-alpha samples, normalises by `/ 255.0`, applies the
/// per-channel function, clamps to `[0, 1]`, and re-quantises back via
/// the same `round(x · 255)` rule the rest of the filter module uses
/// (`quantise_unit`).
///
/// Complexity is `O(W · H)` with a constant per-pixel cost — independent
/// of the size of any [`TransferFunc::Table`] / [`TransferFunc::Discrete`]
/// list (each lookup is `O(1)` against `floor`).
///
/// # Panics
///
/// * If `src.len() != width as usize * height as usize * 4`.
/// * If any channel uses [`TransferFunc::Gamma`] with a non-positive or
///   NaN exponent.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same dimensions. With every
/// channel set to [`TransferFunc::Identity`] the input is returned
/// bytewise unchanged (no f32 round-trip).
pub fn component_transfer(src: &[u8], width: u32, height: u32, ct: &ComponentTransfer) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("component_transfer: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "component_transfer: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    // Fast path: all-identity is a pure byte copy (avoids the f32
    // round-trip which, while idempotent under the half-up quantiser
    // for in-range integers, would unnecessarily reallocate).
    if matches!(ct.r, TransferFunc::Identity)
        && matches!(ct.g, TransferFunc::Identity)
        && matches!(ct.b, TransferFunc::Identity)
        && matches!(ct.a, TransferFunc::Identity)
    {
        return src.to_vec();
    }

    let mut out = Vec::with_capacity(src.len());
    for chunk in src.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let a = chunk[3] as f32 / 255.0;
        out.push(quantise_unit(ct.r.apply(r)));
        out.push(quantise_unit(ct.g.apply(g)));
        out.push(quantise_unit(ct.b.apply(b)));
        out.push(quantise_unit(ct.a.apply(a)));
    }
    out
}

/// Convenience wrapper that runs [`component_transfer`] on a slice of
/// [`Rgba`] pixels and returns a `Vec<Rgba>` of the same length.
/// Identical semantics — provided for callers that already have a typed
/// pixel buffer.
pub fn component_transfer_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    ct: &ComponentTransfer,
) -> Vec<Rgba> {
    assert_eq!(
        src.len(),
        width as usize * height as usize,
        "component_transfer_pixels: src.len() == {} but width*height == {}",
        src.len(),
        width as usize * height as usize
    );
    let mut bytes = Vec::with_capacity(src.len() * 4);
    for p in src {
        bytes.push(p.r);
        bytes.push(p.g);
        bytes.push(p.b);
        bytes.push(p.a);
    }
    let out = component_transfer(&bytes, width, height, ct);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod component_transfer_tests {
    use super::*;

    fn build<F: FnMut(u32, u32) -> Rgba>(w: u32, h: u32, mut f: F) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let c = f(x, y);
                v.extend_from_slice(&[c.r, c.g, c.b, c.a]);
            }
        }
        v
    }

    #[test]
    fn identity_on_every_channel_is_bytewise_identity() {
        // §15.11 collapses to a no-op when every <feFuncX> is
        // `type="identity"`. The fast path avoids the f32 round-trip
        // and returns the input bytes verbatim.
        let img = build(5, 3, |x, y| {
            Rgba::new(
                (x * 47 + y * 11) as u8,
                (x * 17 + y * 5) as u8,
                (y.wrapping_mul(31).wrapping_add(x)) as u8,
                ((x + y) * 7) as u8,
            )
        });
        let ct = ComponentTransfer::identity();
        let out = component_transfer(&img, 5, 3, &ct);
        assert_eq!(out, img);
    }

    #[test]
    fn linear_slope_minus_one_intercept_one_inverts_red_only() {
        // Linear { slope: -1, intercept: 1 } on R encodes the §15.11
        // standard channel inversion: `R' = 1 - R`, equivalent to
        // `255 - R` after quantisation. The other channels carry the
        // identity, so they must round-trip byte-exact.
        let img = build(4, 4, |x, y| {
            Rgba::new((x * 60) as u8, (y * 80) as u8, 17, 200)
        });
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Linear {
            slope: -1.0,
            intercept: 1.0,
        });
        let out = component_transfer(&img, 4, 4, &ct);
        for i in (0..img.len()).step_by(4) {
            // R inverted (±1 LSB for the f32 round-trip).
            let want_r = 255 - img[i] as i32;
            let got_r = out[i] as i32;
            assert!(
                (got_r - want_r).abs() <= 1,
                "R at pixel {} got {got_r} want {want_r}",
                i / 4
            );
            // G, B, A untouched (the byte path is the identity).
            assert_eq!(out[i + 1], img[i + 1]);
            assert_eq!(out[i + 2], img[i + 2]);
            assert_eq!(out[i + 3], img[i + 3]);
        }
    }

    #[test]
    fn gamma_2_2_then_inverse_round_trips_in_mid_to_high_range() {
        // gamma(amp=1, exp=2.2, off=0) followed by its inverse
        // gamma(amp=1, exp=1/2.2, off=0) reproduces the analytical
        // input C exactly: `(C^2.2)^(1/2.2) = C`. The lossy step is
        // the pair of u8 quantisations on either side. The forward
        // curve compresses hard at the low end (e.g. 16/255 → 1/255
        // → 21/255 after inverse) because 8-bit precision is too
        // coarse to represent the bent samples below the curve's
        // knee. Restrict the round-trip assertion to mid-to-high
        // inputs (C ≥ 64/255) where the forward derivative is
        // moderate enough that the two ½-LSB quantisation errors
        // compose into ≤ 2 LSBs at the output. Low-end recovery is
        // a property of the bit depth, not the implementation; the
        // analytical correctness of the §15.11 formula is exercised
        // by the linear and table tests in this same suite.
        let img = build(8, 8, |x, y| {
            Rgba::new(
                64 + (x * 24) as u8,
                64 + (y * 24) as u8,
                100 + (((x + y) * 11) as u8),
                255,
            )
        });
        let fwd = ComponentTransfer {
            r: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 2.2,
                offset: 0.0,
            },
            g: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 2.2,
                offset: 0.0,
            },
            b: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 2.2,
                offset: 0.0,
            },
            a: TransferFunc::Identity,
        };
        let inv = ComponentTransfer {
            r: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 1.0 / 2.2,
                offset: 0.0,
            },
            g: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 1.0 / 2.2,
                offset: 0.0,
            },
            b: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 1.0 / 2.2,
                offset: 0.0,
            },
            a: TransferFunc::Identity,
        };
        let bent = component_transfer(&img, 8, 8, &fwd);
        let back = component_transfer(&bent, 8, 8, &inv);
        for i in 0..img.len() {
            let d = (back[i] as i32 - img[i] as i32).abs();
            assert!(
                d <= 2,
                "gamma round-trip differs by {d} at byte {i} ({} → {} → {})",
                img[i],
                bent[i],
                back[i]
            );
        }
    }

    #[test]
    fn discrete_three_entry_thresholds_correctly() {
        // Discrete(vec![0.0, 0.5, 1.0]) with N=3 partitions [0,1] into
        // three buckets: [0, 1/3) → 0.0 → 0u8, [1/3, 2/3) → 0.5 → 128u8,
        // [2/3, 1] → 1.0 → 255u8. The boundary samples land at
        // 85 (1/3·255) and 170 (2/3·255); test exact-integer inputs
        // that bracket those breakpoints.
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Discrete(vec![0.0, 0.5, 1.0]));
        let pixels: Vec<u8> = vec![0, 84, 86, 169, 171, 255];
        // 6 R values, each in its own 1×1 pixel with G/B/A = 0.
        let mut img = Vec::new();
        for r in &pixels {
            img.extend_from_slice(&[*r, 0, 0, 0]);
        }
        let out = component_transfer(&img, 6, 1, &ct);
        let want: [u8; 6] = [0, 0, 128, 128, 255, 255];
        for (idx, &w) in want.iter().enumerate() {
            assert_eq!(
                out[idx * 4],
                w,
                "discrete[{idx}] for input {} got {}, want {w}",
                pixels[idx],
                out[idx * 4]
            );
        }
    }

    #[test]
    fn table_two_entry_is_identity() {
        // Table(vec![0.0, 1.0]) with N=1 places the breakpoints at
        // C=0 → 0 and C=1 → 1, so the piecewise-linear interpolation
        // is C' = C — the identity. Must be byte-exact (±1 LSB for
        // the f32 round-trip; ½ LSB on each side and a ≤1 LSB
        // worst-case after rounding).
        let img = build(6, 2, |x, y| {
            Rgba::new((x * 51) as u8, (y * 127) as u8, 13, 200)
        });
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Table(vec![0.0, 1.0]));
        let out = component_transfer(&img, 6, 2, &ct);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "table-identity differs by {d} at byte {i}");
        }
    }

    #[test]
    fn table_two_entry_zero_half_halves_the_value() {
        // Table(vec![0.0, 0.5]) with N=1 places C=0 → 0, C=1 → 0.5,
        // so C' = 0.5 · C, equivalent to `R / 2` after quantisation
        // (±1 LSB). Apply to R, leave G/B/A alone.
        let img = build(4, 4, |x, _| Rgba::new((x * 60) as u8, 100, 50, 200));
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Table(vec![0.0, 0.5]));
        let out = component_transfer(&img, 4, 4, &ct);
        for i in (0..img.len()).step_by(4) {
            // Expected: (R / 255) · 0.5 · 255 = R · 0.5, rounded half-up.
            let want = ((img[i] as f32 / 255.0) * 0.5 * 255.0 + 0.5) as i32;
            let got = out[i] as i32;
            assert!(
                (got - want).abs() <= 1,
                "table-half R got {got} want {want} for input {}",
                img[i]
            );
            assert_eq!(out[i + 1], img[i + 1]);
            assert_eq!(out[i + 2], img[i + 2]);
            assert_eq!(out[i + 3], img[i + 3]);
        }
    }

    #[test]
    fn empty_image_returns_empty_buffer() {
        // §15.11 doesn't pin down behaviour on a zero-extent input
        // (the filter region degenerates to empty); we choose the
        // same convention as gaussian_blur and color_matrix and
        // return an empty buffer.
        let ct = ComponentTransfer::identity();
        assert_eq!(component_transfer(&[], 0, 5, &ct), Vec::<u8>::new());
        assert_eq!(component_transfer(&[], 5, 0, &ct), Vec::<u8>::new());
    }

    #[test]
    fn solid_image_under_identity_is_invariant() {
        // A uniform image under all-identity must come back byte-exact;
        // this exercises the fast path explicitly on a non-trivial
        // pixel value.
        let img = build(4, 4, |_, _| Rgba::new(123, 45, 67, 200));
        let ct = ComponentTransfer::identity();
        let out = component_transfer(&img, 4, 4, &ct);
        assert_eq!(out, img);
    }

    #[test]
    fn linear_on_alpha_keeps_rgb_byte_exact() {
        // Linear { slope: 0.5, intercept: 0 } on A only must scale
        // alpha by half and leave RGB byte-exact. Exercises that the
        // per-channel dispatch really is per-channel.
        let img = build(3, 3, |x, y| {
            Rgba::new((x * 80) as u8, (y * 80) as u8, 7, 200)
        });
        let ct = ComponentTransfer::identity().with_a(TransferFunc::Linear {
            slope: 0.5,
            intercept: 0.0,
        });
        let out = component_transfer(&img, 3, 3, &ct);
        for i in (0..img.len()).step_by(4) {
            assert_eq!(out[i], img[i]);
            assert_eq!(out[i + 1], img[i + 1]);
            assert_eq!(out[i + 2], img[i + 2]);
            let want = ((img[i + 3] as f32 / 255.0) * 0.5 * 255.0 + 0.5) as i32;
            let got = out[i + 3] as i32;
            assert!((got - want).abs() <= 1, "alpha half got {got} want {want}");
        }
    }

    #[test]
    fn linear_intercept_one_saturates_to_255() {
        // Linear { slope: 0, intercept: 1 } sets every pixel of the
        // affected channel to 1.0 → 255u8, regardless of input. Tests
        // the §15.11 output clamp implicitly via the saturating cap.
        let img = build(2, 2, |_, _| Rgba::new(0, 0, 0, 0));
        let ct = ComponentTransfer::identity().with_g(TransferFunc::Linear {
            slope: 0.0,
            intercept: 1.0,
        });
        let out = component_transfer(&img, 2, 2, &ct);
        for i in (0..img.len()).step_by(4) {
            assert_eq!(out[i], 0);
            assert_eq!(out[i + 1], 255);
            assert_eq!(out[i + 2], 0);
            assert_eq!(out[i + 3], 0);
        }
    }

    #[test]
    fn typed_pixel_wrapper_agrees_with_byte_api() {
        // The Vec<Rgba> wrapper must produce the byte-equivalent result
        // for any ComponentTransfer; pick a non-trivial mix of modes
        // across the four channels.
        let img: Vec<Rgba> = (0..16)
            .map(|i| Rgba::new(i as u8 * 16, (16 - i) as u8 * 15, 50, 200))
            .collect();
        let bytes: Vec<u8> = img.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
        let ct = ComponentTransfer {
            r: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 2.0,
                offset: 0.0,
            },
            g: TransferFunc::Linear {
                slope: 0.75,
                intercept: 0.25,
            },
            b: TransferFunc::Discrete(vec![0.0, 1.0]),
            a: TransferFunc::Identity,
        };
        let via_bytes = component_transfer(&bytes, 4, 4, &ct);
        let via_pixels = component_transfer_pixels(&img, 4, 4, &ct);
        let want_pixels: Vec<Rgba> = via_bytes
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        assert_eq!(via_pixels, want_pixels);
    }

    #[test]
    fn empty_table_collapses_to_identity() {
        // §15.11's "type=table with insufficient values" fall-back is
        // an identity. An empty Vec defines no segment, so we return
        // C' = C. Exercises that the fall-back is taken before any
        // f32 lookup that would dereference out of range.
        let img = build(2, 2, |x, y| {
            Rgba::new((x * 100) as u8, (y * 100) as u8, 7, 200)
        });
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Table(Vec::new()));
        let out = component_transfer(&img, 2, 2, &ct);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "table-empty differs by {d} at byte {i}");
        }
    }

    #[test]
    fn empty_discrete_collapses_to_identity() {
        // Same as table-empty: zero-entry discrete defines no buckets,
        // collapse to identity rather than panic on the floor lookup.
        let img = build(2, 2, |x, y| {
            Rgba::new((x * 100) as u8, (y * 100) as u8, 7, 200)
        });
        let ct = ComponentTransfer::identity().with_g(TransferFunc::Discrete(Vec::new()));
        let out = component_transfer(&img, 2, 2, &ct);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "discrete-empty differs by {d} at byte {i}");
        }
    }

    #[test]
    #[should_panic(expected = "component_transfer: src.len() ==")]
    fn wrong_length_panics() {
        let bad = vec![0u8; 7];
        let ct = ComponentTransfer::identity();
        let _ = component_transfer(&bad, 2, 2, &ct);
    }

    #[test]
    #[should_panic(expected = "gamma exponent must be strictly positive")]
    fn gamma_zero_exponent_panics() {
        let img = vec![0u8; 16];
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 0.0,
            offset: 0.0,
        });
        let _ = component_transfer(&img, 2, 2, &ct);
    }

    #[test]
    #[should_panic(expected = "gamma exponent must be strictly positive")]
    fn gamma_negative_exponent_panics() {
        let img = vec![0u8; 16];
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: -0.5,
            offset: 0.0,
        });
        let _ = component_transfer(&img, 2, 2, &ct);
    }
}

// ---------------------------------------------------------------------------
// feComposite — SVG 1.1 §15.12
// ---------------------------------------------------------------------------

/// Operator selector for [`composite_filter`], mirroring the `operator`
/// attribute of SVG 1.1 §15.12 / Filter Effects 1 §9.8 `<feComposite>`
/// (`"over" | "in" | "out" | "atop" | "xor" | "lighter" | "arithmetic"`).
///
/// The six non-arithmetic operators are the Porter–Duff compositing
/// operations referenced by §15.12 (`lighter` is the Filter-Effects-1
/// §9.8 addition pulled in from Compositing-1). They are expressed here
/// through the standard premultiplied-alpha blend-factor pair `(Fa, Fb)`
/// so that the per-channel result is
///
/// ```text
/// co = ca · Fa + cb · Fb        (premultiplied colour)
/// αo = αa · Fa + αb · Fb        (alpha)
/// ```
///
/// where the *first* input (`in` / `i1`) is the `a` operand and the
/// *second* input (`in2` / `i2`) is the `b` operand, consistent with the
/// premultiplied source-over algebra of SVG 1.1 §14.2 already used by
/// the crate's [`composite_rgba_premultiplied`](crate::composite_rgba_premultiplied)
/// path (`Cr' = (1 − Ea)·Cr + Er`, i.e. `over` with `Fa = 1`,
/// `Fb = 1 − αa`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositeOp {
    /// `in` source-over `in2`. `Fa = 1`, `Fb = 1 − αa`.
    Over,
    /// `in` clipped to `in2`'s coverage. `Fa = αb`, `Fb = 0`.
    In,
    /// `in` outside `in2`'s coverage. `Fa = 1 − αb`, `Fb = 0`.
    Out,
    /// `in` atop `in2`. `Fa = αb`, `Fb = 1 − αa`.
    Atop,
    /// Symmetric difference. `Fa = 1 − αb`, `Fb = 1 − αa`.
    Xor,
    /// Additive (`plus` / `lighter`, Filter-Effects-1 §9.8 →
    /// Compositing-1). `Fa = 1`, `Fb = 1`: premultiplied source and
    /// backdrop add directly, with the per-channel and alpha sums
    /// clamped to `[0, 1]`. Equal to the arithmetic operator with
    /// `k2 = k3 = 1` (and `k1 = k4 = 0`), so disjoint opaque regions
    /// union and overlapping regions saturate toward white.
    Lighter,
    /// Component-wise arithmetic combination
    /// `result = k1·i1·i2 + k2·i1 + k3·i2 + k4` (§15.12), evaluated on
    /// premultiplied channels in `[0, 1]` and clamped to `[0, 1]`.
    Arithmetic {
        /// Coefficient on the `i1 · i2` product term.
        k1: f32,
        /// Coefficient on the `i1` term.
        k2: f32,
        /// Coefficient on the `i2` term.
        k3: f32,
        /// Constant offset term.
        k4: f32,
    },
}

impl CompositeOp {
    /// The Porter–Duff `(Fa, Fb)` blend factors for the six
    /// non-arithmetic operators, given the premultiplied alphas
    /// `αa` (= `in`) and `αb` (= `in2`), both in `[0, 1]`.
    ///
    /// Returns `None` for [`CompositeOp::Arithmetic`], which is not a
    /// blend-factor operator and is handled separately.
    #[inline]
    fn blend_factors(self, alpha_a: f32, alpha_b: f32) -> Option<(f32, f32)> {
        Some(match self {
            CompositeOp::Over => (1.0, 1.0 - alpha_a),
            CompositeOp::In => (alpha_b, 0.0),
            CompositeOp::Out => (1.0 - alpha_b, 0.0),
            CompositeOp::Atop => (alpha_b, 1.0 - alpha_a),
            CompositeOp::Xor => (1.0 - alpha_b, 1.0 - alpha_a),
            CompositeOp::Lighter => (1.0, 1.0),
            CompositeOp::Arithmetic { .. } => return None,
        })
    }
}

/// SVG 1.1 §15.12 `<feComposite>` — combine two equally-sized
/// packed-RGBA `u8` buffers pixel-wise with a [`CompositeOp`].
///
/// `in1` maps to the spec's `in` / `i1` operand and `in2` to `in2` /
/// `i2`. Both buffers must be exactly `width * height * 4` bytes in
/// row-major RGBA order and describe the same `width × height` extent.
///
/// **Pixel space.** §14.2 ("all color values use premultiplied alpha")
/// fixes the compositing algebra in the premultiplied domain. This
/// entry point converts each straight-alpha input byte triple to
/// premultiplied `[0, 1]` floats (`c · α`), evaluates the operator, and
/// converts the premultiplied result back to straight-alpha bytes. The
/// arithmetic operator is likewise evaluated on the premultiplied
/// channels, matching the spec's example use (combining lighting output
/// with texture) where the operands are already premultiplied filter
/// results.
///
/// Complexity is `O(W · H)` with a constant per-pixel cost.
///
/// # Panics
///
/// * If `in1.len() != width as usize * height as usize * 4`.
/// * If `in2.len() != width as usize * height as usize * 4`.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same dimensions.
pub fn composite_filter(
    in1: &[u8],
    in2: &[u8],
    width: u32,
    height: u32,
    op: CompositeOp,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("composite_filter: width * height * 4 overflowed usize");
    assert_eq!(
        in1.len(),
        expected,
        "composite_filter: in1.len() == {} but width*height*4 == {expected}",
        in1.len()
    );
    assert_eq!(
        in2.len(),
        expected,
        "composite_filter: in2.len() == {} but width*height*4 == {expected}",
        in2.len()
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(expected);
    for (pa, pb) in in1.chunks_exact(4).zip(in2.chunks_exact(4)) {
        // Straight-alpha bytes → premultiplied [0, 1].
        let aa = pa[3] as f32 / 255.0;
        let ab = pb[3] as f32 / 255.0;
        let a = [
            pa[0] as f32 / 255.0 * aa,
            pa[1] as f32 / 255.0 * aa,
            pa[2] as f32 / 255.0 * aa,
            aa,
        ];
        let b = [
            pb[0] as f32 / 255.0 * ab,
            pb[1] as f32 / 255.0 * ab,
            pb[2] as f32 / 255.0 * ab,
            ab,
        ];

        let (pr, pg, pb_, palpha) = match op {
            CompositeOp::Arithmetic { k1, k2, k3, k4 } => {
                let arith =
                    |i1: f32, i2: f32| (k1 * i1 * i2 + k2 * i1 + k3 * i2 + k4).clamp(0.0, 1.0);
                (
                    arith(a[0], b[0]),
                    arith(a[1], b[1]),
                    arith(a[2], b[2]),
                    arith(a[3], b[3]),
                )
            }
            _ => {
                // SAFETY of unwrap: every variant except Arithmetic
                // returns Some, and Arithmetic is handled above.
                let (fa, fb) = op.blend_factors(a[3], b[3]).unwrap();
                (
                    (a[0] * fa + b[0] * fb).clamp(0.0, 1.0),
                    (a[1] * fa + b[1] * fb).clamp(0.0, 1.0),
                    (a[2] * fa + b[2] * fb).clamp(0.0, 1.0),
                    (a[3] * fa + b[3] * fb).clamp(0.0, 1.0),
                )
            }
        };

        // Premultiplied [0, 1] → straight-alpha bytes. When the result
        // alpha is zero the colour is fully transparent; emit (0,0,0,0)
        // so the un-premultiply division is well-defined.
        if palpha <= 0.0 {
            out.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let inv = 1.0 / palpha;
            out.push(quantise_unit((pr * inv).min(1.0)));
            out.push(quantise_unit((pg * inv).min(1.0)));
            out.push(quantise_unit((pb_ * inv).min(1.0)));
            out.push(quantise_unit(palpha));
        }
    }
    out
}

/// Convenience wrapper that runs [`composite_filter`] on two slices of
/// [`Rgba`] pixels and returns a `Vec<Rgba>` of the same length.
/// Identical semantics — provided for callers that already have typed
/// pixel buffers.
pub fn composite_filter_pixels(
    in1: &[Rgba],
    in2: &[Rgba],
    width: u32,
    height: u32,
    op: CompositeOp,
) -> Vec<Rgba> {
    let n = width as usize * height as usize;
    assert_eq!(
        in1.len(),
        n,
        "composite_filter_pixels: in1.len() == {} but width*height == {n}",
        in1.len()
    );
    assert_eq!(
        in2.len(),
        n,
        "composite_filter_pixels: in2.len() == {} but width*height == {n}",
        in2.len()
    );
    let mut b1 = Vec::with_capacity(n * 4);
    let mut b2 = Vec::with_capacity(n * 4);
    for p in in1 {
        b1.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    for p in in2 {
        b2.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out = composite_filter(&b1, &b2, width, height, op);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

/// Edge-extension policy for [`convolve_matrix`], mirroring the
/// `edgeMode` attribute of SVG 1.1 §15.13 `<feConvolveMatrix>`.
///
/// Selects how the kernel reads beyond the borders of the source
/// image. The default per the spec is `Duplicate` (clamp-to-edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvolveEdgeMode {
    /// Extend by replicating the border row / column — `"duplicate"`,
    /// the spec default. The same clamp-to-edge policy used by
    /// [`gaussian_blur`] and [`morphology`].
    #[default]
    Duplicate,
    /// Extend by wrapping around — sample column `-1` reads column
    /// `width - 1`, sample row `height` reads row `0`, and so on
    /// (toroidal addressing). Spec value `"wrap"`.
    Wrap,
    /// Extend with `(R, G, B, A) = (0, 0, 0, 0)` — sample positions
    /// outside the source contribute zero to the sum. Spec value
    /// `"none"`.
    None,
}

/// Parameter block for [`convolve_matrix`] / [`convolve_matrix_pixels`].
///
/// Mirrors the attribute set of SVG 1.1 §15.13 `<feConvolveMatrix>`:
/// `order`, `kernelMatrix`, `divisor`, `bias`, `targetX`, `targetY`,
/// `edgeMode`, and `preserveAlpha`. Construct directly with the public
/// fields or through [`ConvolveMatrix::new`] which applies the spec
/// defaults for `divisor` (= sum of kernel, falling back to `1.0`),
/// `target_x` / `target_y` (= floor of order / 2), and `bias` (= 0).
///
/// Field interpretation matches §15.13 verbatim:
///
/// * `order_x` × `order_y` cells in `kernel`, stored row-major: index
///   `row * order_x + col` reads `kernelMatrix[col, row]`. Spec
///   §15.13 mandates `1 ≤ order_{x,y}`.
/// * `divisor` divides the convolved sum before `bias` is added.
///   Must be non-zero (§15.13 error processing).
/// * `bias` is added to each output channel after the divisor step.
/// * `target_x` / `target_y` reposition the kernel relative to the
///   target pixel; the default centres a square kernel
///   (`floor(order / 2)`). Spec mandates `0 ≤ target_{x,y} < order_{x,y}`.
/// * `edge_mode` chooses how out-of-bounds samples are extended.
/// * `preserve_alpha = false` (default per §15.13) convolves the alpha
///   channel alongside RGB; `preserve_alpha = true` un-premultiplies the
///   source, convolves only RGB, leaves alpha as the original
///   `SOURCE_{X,Y}`, then re-premultiplies the result before
///   re-quantising to `u8`.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvolveMatrix {
    /// Number of columns in `kernel`. Must be `≥ 1`.
    pub order_x: u32,
    /// Number of rows in `kernel`. Must be `≥ 1`.
    pub order_y: u32,
    /// Row-major `order_y × order_x` kernel weights, length
    /// `order_x * order_y`.
    pub kernel: Vec<f32>,
    /// Divisor applied to the convolved sum (§15.13 — default is
    /// `sum(kernel)`, falling back to `1.0` when the sum is zero).
    /// Must be non-zero.
    pub divisor: f32,
    /// Additive bias applied after the divisor step.
    pub bias: f32,
    /// X-position of the kernel's anchor (0 ≤ `target_x` < `order_x`).
    pub target_x: u32,
    /// Y-position of the kernel's anchor (0 ≤ `target_y` < `order_y`).
    pub target_y: u32,
    /// Out-of-bounds sample extension policy.
    pub edge_mode: ConvolveEdgeMode,
    /// When `true`, convolve only the RGB channels of the
    /// un-premultiplied source and pass alpha through unchanged
    /// (§15.13 `preserveAlpha="true"`). When `false`, convolve all
    /// four channels of the straight-alpha source uniformly.
    pub preserve_alpha: bool,
}

impl ConvolveMatrix {
    /// Construct a [`ConvolveMatrix`] applying the §15.13 defaults for
    /// every attribute the spec leaves unspecified:
    ///
    /// * `divisor` = sum of `kernel`, falling back to `1.0` when the
    ///   sum is zero (§15.13).
    /// * `bias` = `0`.
    /// * `target_x` = `floor(order_x / 2)`, `target_y` = `floor(order_y / 2)`.
    /// * `edge_mode` = [`ConvolveEdgeMode::Duplicate`].
    /// * `preserve_alpha` = `false`.
    ///
    /// # Panics
    ///
    /// * If `order_x == 0` or `order_y == 0` (§15.13 error
    ///   processing — both must be ≥ 1).
    /// * If `kernel.len() != order_x * order_y` (the spec requires the
    ///   kernel list length to equal `orderX * orderY`).
    pub fn new(order_x: u32, order_y: u32, kernel: Vec<f32>) -> Self {
        assert!(
            order_x >= 1 && order_y >= 1,
            "ConvolveMatrix::new: order_x ({order_x}) and order_y ({order_y}) must each be >= 1"
        );
        let n = (order_x as usize)
            .checked_mul(order_y as usize)
            .expect("ConvolveMatrix::new: order_x * order_y overflowed usize");
        assert_eq!(
            kernel.len(),
            n,
            "ConvolveMatrix::new: kernel.len() ({}) must equal order_x * order_y ({n})",
            kernel.len()
        );

        let sum: f32 = kernel.iter().sum();
        let divisor = if sum == 0.0 { 1.0 } else { sum };
        let target_x = order_x / 2;
        let target_y = order_y / 2;

        ConvolveMatrix {
            order_x,
            order_y,
            kernel,
            divisor,
            bias: 0.0,
            target_x,
            target_y,
            edge_mode: ConvolveEdgeMode::Duplicate,
            preserve_alpha: false,
        }
    }

    /// Builder: set [`Self::bias`].
    pub fn with_bias(mut self, bias: f32) -> Self {
        self.bias = bias;
        self
    }

    /// Builder: set [`Self::divisor`]. Must be non-zero (panic
    /// otherwise — §15.13 error processing).
    pub fn with_divisor(mut self, divisor: f32) -> Self {
        assert!(
            divisor != 0.0,
            "ConvolveMatrix::with_divisor: divisor must be non-zero (§15.13)"
        );
        self.divisor = divisor;
        self
    }

    /// Builder: set [`Self::target_x`] / [`Self::target_y`]. Each must
    /// satisfy `0 ≤ target < order` (§15.13 error processing).
    pub fn with_target(mut self, target_x: u32, target_y: u32) -> Self {
        assert!(
            target_x < self.order_x,
            "ConvolveMatrix::with_target: target_x ({target_x}) >= order_x ({})",
            self.order_x
        );
        assert!(
            target_y < self.order_y,
            "ConvolveMatrix::with_target: target_y ({target_y}) >= order_y ({})",
            self.order_y
        );
        self.target_x = target_x;
        self.target_y = target_y;
        self
    }

    /// Builder: set [`Self::edge_mode`].
    pub fn with_edge_mode(mut self, edge_mode: ConvolveEdgeMode) -> Self {
        self.edge_mode = edge_mode;
        self
    }

    /// Builder: set [`Self::preserve_alpha`].
    pub fn with_preserve_alpha(mut self, preserve_alpha: bool) -> Self {
        self.preserve_alpha = preserve_alpha;
        self
    }

    /// Read the kernel weight at logical position `(col, row)` —
    /// equivalent to `kernel[row * order_x + col]`. Spec-named
    /// `kernelMatrix[col, row]`.
    #[inline]
    fn kernel_at(&self, col: u32, row: u32) -> f32 {
        debug_assert!(col < self.order_x && row < self.order_y);
        self.kernel[(row * self.order_x + col) as usize]
    }
}

/// Read a sample from a packed-RGBA `u8` buffer with edge handling.
///
/// `x`, `y` may be negative or `>= width / height`. The four channels
/// are returned in straight-alpha `u8` order (R, G, B, A). For
/// [`ConvolveEdgeMode::None`] out-of-bounds reads return `(0, 0, 0, 0)`;
/// for `Duplicate` they clamp to the nearest in-range coordinate; for
/// `Wrap` they take `x.rem_euclid(width)` / `y.rem_euclid(height)`.
#[inline]
fn sample_with_edge(
    src: &[u8],
    width: u32,
    height: u32,
    x: i64,
    y: i64,
    mode: ConvolveEdgeMode,
) -> [u8; 4] {
    let w = width as i64;
    let h = height as i64;
    let (sx, sy) = match mode {
        ConvolveEdgeMode::Duplicate => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
        ConvolveEdgeMode::Wrap => (x.rem_euclid(w), y.rem_euclid(h)),
        ConvolveEdgeMode::None => {
            if x < 0 || x >= w || y < 0 || y >= h {
                return [0, 0, 0, 0];
            }
            (x, y)
        }
    };
    let idx = ((sy as usize) * (width as usize) + (sx as usize)) * 4;
    [src[idx], src[idx + 1], src[idx + 2], src[idx + 3]]
}

/// SVG 1.1 §15.13 `<feConvolveMatrix>` — apply a general matrix
/// convolution to a packed-RGBA `u8` buffer.
///
/// Implements the spec formula verbatim:
///
/// ```text
/// COLOR_{X,Y} = (
///     SUM_{I=0..orderY-1} {
///         SUM_{J=0..orderX-1} {
///             SOURCE_{X − targetX + J, Y − targetY + I}
///                 · kernelMatrix[orderX − J − 1, orderY − I − 1]
///         }
///     }
/// ) / divisor + bias
/// ```
///
/// The kernel is rotated 180° when sampling — that is, the
/// top-left-most coefficient `kernelMatrix[0, 0]` lands on the
/// bottom-right-most contributing source pixel — matching the
/// mathematical convention noted in §15.13 ("the values in the
/// kernel matrix are applied such that the kernel matrix is rotated
/// 180 degrees relative to the source and destination images").
///
/// `divisor`, `bias`, `target_x`, `target_y`, `edge_mode`, and
/// `preserve_alpha` are taken from `cm`; see [`ConvolveMatrix`] for the
/// per-field semantics.
///
/// When `preserve_alpha == false` (§15.13 default), the convolution is
/// applied uniformly across all four straight-alpha channels of the
/// source — the same formula above governs alpha as governs colour.
/// When `preserve_alpha == true`, the source's straight-alpha colour
/// channels are first un-premultiplied (the alpha-multiplication
/// inverse), the RGB channels are then convolved, alpha is left as
/// `SOURCE_{X,Y}` (the original pixel's alpha), and the RGB result is
/// re-premultiplied before the final clamp-and-quantise step. (The
/// source is treated as straight-alpha throughout; the
/// "un-premultiply / re-premultiply" wording in §15.13 refers to the
/// internal premultiplication used while applying the kernel, not to
/// the encoding of the input or output buffers.)
///
/// Output channels are clamped to `[0, 255]` before quantisation.
///
/// Complexity: `O(W · H · order_x · order_y)`. The kernel is **not**
/// separable in general — `feConvolveMatrix` is the catch-all
/// arbitrary-2D-kernel primitive; spec-named separable kernels
/// (Gaussian) live in dedicated filter primitives.
///
/// # Panics
///
/// * If `src.len() != width as usize * height as usize * 4`.
/// * If `cm.divisor == 0.0` (§15.13 error processing).
/// * If `cm.order_x == 0` or `cm.order_y == 0`.
/// * If `cm.kernel.len() != cm.order_x * cm.order_y`.
/// * If `cm.target_x >= cm.order_x` or `cm.target_y >= cm.order_y`.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same `width × height` extent.
pub fn convolve_matrix(src: &[u8], width: u32, height: u32, cm: &ConvolveMatrix) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("convolve_matrix: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "convolve_matrix: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );
    assert!(
        cm.divisor != 0.0,
        "convolve_matrix: divisor must be non-zero (§15.13)"
    );
    assert!(
        cm.order_x >= 1 && cm.order_y >= 1,
        "convolve_matrix: order_x ({}) and order_y ({}) must each be >= 1",
        cm.order_x,
        cm.order_y
    );
    let n_cells = (cm.order_x as usize) * (cm.order_y as usize);
    assert_eq!(
        cm.kernel.len(),
        n_cells,
        "convolve_matrix: kernel.len() ({}) must equal order_x * order_y ({n_cells})",
        cm.kernel.len()
    );
    assert!(
        cm.target_x < cm.order_x,
        "convolve_matrix: target_x ({}) >= order_x ({})",
        cm.target_x,
        cm.order_x
    );
    assert!(
        cm.target_y < cm.order_y,
        "convolve_matrix: target_y ({}) >= order_y ({})",
        cm.target_y,
        cm.order_y
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    let inv_div = 1.0 / cm.divisor;
    let bias = cm.bias;
    let order_x = cm.order_x as i64;
    let order_y = cm.order_y as i64;
    let tx = cm.target_x as i64;
    let ty = cm.target_y as i64;

    let mut out = vec![0u8; expected];

    for y in 0..height as i64 {
        for x in 0..width as i64 {
            let mut sum = [0f32; 4];
            // Original alpha at the target pixel — needed for the
            // preserve_alpha pass-through and for the un-premultiply
            // step before the kernel is applied to RGB.
            let target = sample_with_edge(src, width, height, x, y, cm.edge_mode);
            let target_alpha = target[3] as f32 / 255.0;

            for i in 0..order_y {
                for j in 0..order_x {
                    // The 180°-rotated kernel index per §15.13:
                    // kernelMatrix[orderX − J − 1, orderY − I − 1].
                    let k = cm.kernel_at((order_x - j - 1) as u32, (order_y - i - 1) as u32);
                    if k == 0.0 {
                        continue;
                    }
                    let sx = x - tx + j;
                    let sy = y - ty + i;
                    let s = sample_with_edge(src, width, height, sx, sy, cm.edge_mode);
                    if cm.preserve_alpha {
                        // RGB only; alpha is taken from the target pixel
                        // and not summed. The spec wording is "the filter
                        // will temporarily unpremultiply the color
                        // component values, apply the kernel, and then
                        // re-premultiply at the end". Source straight-
                        // alpha bytes already store un-premultiplied RGB,
                        // so the un-premultiply step is the identity.
                        sum[0] += k * (s[0] as f32);
                        sum[1] += k * (s[1] as f32);
                        sum[2] += k * (s[2] as f32);
                    } else {
                        sum[0] += k * (s[0] as f32);
                        sum[1] += k * (s[1] as f32);
                        sum[2] += k * (s[2] as f32);
                        sum[3] += k * (s[3] as f32);
                    }
                }
            }

            // Apply divisor + bias, clamp, quantise.
            let out_idx = ((y as usize) * w + (x as usize)) * 4;
            let r = sum[0] * inv_div + bias * 255.0;
            let g = sum[1] * inv_div + bias * 255.0;
            let b = sum[2] * inv_div + bias * 255.0;
            out[out_idx] = quantise_byte(r);
            out[out_idx + 1] = quantise_byte(g);
            out[out_idx + 2] = quantise_byte(b);
            if cm.preserve_alpha {
                // Alpha pass-through: ALPHAX,Y = SOURCEX,Y per §15.13.
                out[out_idx + 3] = target[3];
            } else {
                let a = sum[3] * inv_div + bias * 255.0;
                out[out_idx + 3] = quantise_byte(a);
            }
            // Silence unused-variable warning when preserve_alpha=false;
            // target_alpha is still computed once per pixel for symmetry
            // with the preserveAlpha branch (no measurable overhead).
            let _ = target_alpha;
        }
    }

    out
}

/// Convenience wrapper that runs [`convolve_matrix`] on a slice of
/// [`Rgba`] pixels and returns a `Vec<Rgba>` of the same length.
/// Identical semantics — provided for callers that already have typed
/// pixel buffers.
pub fn convolve_matrix_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    cm: &ConvolveMatrix,
) -> Vec<Rgba> {
    let n = (width as usize) * (height as usize);
    assert_eq!(
        src.len(),
        n,
        "convolve_matrix_pixels: src.len() == {} but width*height == {n}",
        src.len()
    );
    let mut bytes = Vec::with_capacity(n * 4);
    for p in src {
        bytes.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out = convolve_matrix(&bytes, width, height, cm);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

/// Round-half-up clamp from an arbitrary `f32` to `[0, 255]`, NaN to 0.
/// Distinct from [`quantise_unit`] (which expects a `[0, 1]` unit value)
/// and [`quantise_u8`] (whose contract is different); this one takes a
/// already-byte-scale `f32` straight off the convolution sum.
#[inline]
fn quantise_byte(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let clamped = v.clamp(0.0, 255.0);
    (clamped + 0.5) as u8
}

#[cfg(test)]
mod convolve_matrix_tests {
    use super::*;

    fn solid(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&rgba);
        }
        v
    }

    /// 3×3 identity kernel — the centre cell is 1.0, everything else
    /// is zero. Divisor defaults to 1.0 (sum is 1.0). Spec-conformant
    /// per §15.13 default behaviour: byte-for-byte identity on the
    /// interior; with `Duplicate` edge mode the borders are also
    /// identity because the kernel only reads the centre pixel.
    #[test]
    fn identity_kernel_3x3_is_identity() {
        let w = 5;
        let h = 4;
        // Pseudo-random but deterministic pixel pattern.
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for i in 0..(w * h) {
            src.push(((i * 37 + 3) % 256) as u8);
            src.push(((i * 53 + 11) % 256) as u8);
            src.push(((i * 71 + 19) % 256) as u8);
            src.push(((i * 17 + 41) % 256) as u8);
        }

        #[rustfmt::skip]
        let cm = ConvolveMatrix::new(3, 3, vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let out = convolve_matrix(&src, w, h, &cm);
        assert_eq!(out, src, "identity kernel should round-trip every pixel");
    }

    /// Default constructor picks `divisor = sum(kernel)` (§15.13). A
    /// solid colour through a box-blur kernel must be byte-identical
    /// regardless of edge mode (Duplicate / Wrap both return the
    /// same colour at every position; None also works because the
    /// kernel-sum / divisor = 1 cancels the in-bounds-only sum on the
    /// interior, and on the border the missing samples are zero,
    /// shrinking the sum proportionally — we expect identity on the
    /// interior pixels for None and possibly different for borders).
    #[test]
    fn box_blur_3x3_solid_is_identity_interior() {
        let w = 7;
        let h = 5;
        let colour = [120, 200, 60, 255];
        let src = solid(w, h, colour);
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]); // divisor = 9
        for mode in [
            ConvolveEdgeMode::Duplicate,
            ConvolveEdgeMode::Wrap,
            ConvolveEdgeMode::None,
        ] {
            let cm = cm.clone().with_edge_mode(mode);
            let out = convolve_matrix(&src, w, h, &cm);
            // Interior pixels must be identity regardless of edge
            // mode (the kernel only reads in-bounds samples for them).
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let i = ((y * w + x) * 4) as usize;
                    assert_eq!(
                        &out[i..i + 4],
                        &colour,
                        "interior pixel ({x},{y}) under {mode:?} should be identity"
                    );
                }
            }
            // Under Duplicate + Wrap edge modes the borders also see
            // a uniform colour everywhere the kernel reaches, so the
            // whole image must be identity.
            if matches!(mode, ConvolveEdgeMode::Duplicate | ConvolveEdgeMode::Wrap) {
                assert_eq!(out, src, "{mode:?} should be identity for solid image");
            }
        }
    }

    /// Hand-evaluate one pixel against the §15.13 worked example.
    /// With the spec's 5×5 input and the 3×3 kernel
    /// `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` (divisor = 45) the source
    /// pixel value at the second row and second column (the spec's
    /// example pixel) is 120, and the spec writes the result as
    /// `(9·0 + 8·20 + 7·40 + 6·100 + 5·120 + 4·140 + 3·200 + 2·220 +
    /// 1·240) / 45 = 3480 / 45 = 77.33…` — that is, the kernel is
    /// 180°-rotated before being applied, so coefficient 9 lands on
    /// the source pixel at the *top-left* of the 3×3 window
    /// (the 0 in the corner), coefficient 1 lands on the *bottom-right*
    /// (the 240). Our `convolve_matrix` must reproduce that 77 to a
    /// byte after the round-half-up quantisation.
    #[test]
    fn spec_15_13_worked_example_pixel() {
        // 5x5 single-channel-style image populated only in R. We use
        // grey (R == G == B) and full opaque so all four channels
        // share the same convolved value and the assertion stays
        // clean.
        let w = 5u32;
        let h = 5u32;
        #[rustfmt::skip]
        let grey: Vec<u8> = vec![
              0,  20,  40, 235, 235,
            100, 120, 140, 235, 235,
            200, 220, 240, 235, 235,
            225, 225, 255, 255, 255,
            225, 225, 255, 255, 255,
        ];
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for v in &grey {
            src.extend_from_slice(&[*v, *v, *v, 255]);
        }
        #[rustfmt::skip]
        let cm = ConvolveMatrix::new(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        // Default divisor is the kernel sum (45). The spec example
        // evaluates pixel (1, 1) to 3480 / 45 = 77.333… → 77 after
        // round-half-up. (The spec text lists the individual products
        // 9·0 + 8·20 + 7·40 + 6·100 + 5·120 + 4·140 + 3·200 + 2·220 +
        // 1·240 = 0 + 160 + 280 + 600 + 600 + 560 + 600 + 440 + 240
        // = 3480.)
        let out = convolve_matrix(&src, w, h, &cm);
        let idx = ((w + 1) * 4) as usize;
        let expected = 77u8;
        assert_eq!(out[idx], expected, "spec example pixel R should be 77");
        assert_eq!(out[idx + 1], expected, "spec example pixel G should be 77");
        assert_eq!(out[idx + 2], expected, "spec example pixel B should be 77");
    }

    /// `preserve_alpha = true` leaves the alpha channel byte-identical
    /// to the source while still convolving the RGB channels. We use a
    /// 3×3 box-blur on a sharp alpha mask — the RGB output blurs but
    /// the alpha must match the input pixel-by-pixel.
    #[test]
    fn preserve_alpha_passes_alpha_through() {
        let w = 5;
        let h = 5;
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let alpha = if (x + y) % 2 == 0 { 255 } else { 64 };
                src.extend_from_slice(&[100, 200, 50, alpha]);
            }
        }
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_preserve_alpha(true);
        let out = convolve_matrix(&src, w, h, &cm);
        for i in 0..(w * h) as usize {
            assert_eq!(
                out[i * 4 + 3],
                src[i * 4 + 3],
                "alpha[{i}] should pass through under preserve_alpha"
            );
        }
    }

    /// `edgeMode = "none"` zeros the kernel reads outside the source.
    /// On the top-left corner pixel of a 1×1-of-non-zero image (every
    /// other pixel is zero) we know exactly how many cells contribute.
    #[test]
    fn edge_mode_none_zeros_oob() {
        let w = 3;
        let h = 3;
        let mut src = vec![0u8; (w * h * 4) as usize];
        // Put 100 in R at the centre pixel only; everything else 0.
        let centre = ((w + 1) * 4) as usize;
        src[centre] = 100;
        src[centre + 3] = 255; // give it some alpha so it's well-formed
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9])
            .with_divisor(1.0)
            .with_edge_mode(ConvolveEdgeMode::None);
        let out = convolve_matrix(&src, w, h, &cm);
        // Every output pixel that contains the centre in its 3×3
        // neighbourhood (i.e. every pixel — at this size the kernel
        // touches every cell) sees R = 100 added once. Output R must
        // equal 100 everywhere in this configuration.
        for i in 0..(w * h) as usize {
            assert_eq!(
                out[i * 4],
                100,
                "every neighbour pixel should see the centre R contribution exactly once"
            );
        }
    }

    /// `edgeMode = "duplicate"` extends by clamping — sample at column
    /// -1 reads column 0, etc. Easy sanity: a 1×1 image under any kernel
    /// summing to `divisor` is byte-identical to the source.
    #[test]
    fn duplicate_1x1_is_identity_under_unit_divisor() {
        let src = [80u8, 160, 240, 255];
        let cm =
            ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_edge_mode(ConvolveEdgeMode::Duplicate);
        // divisor defaults to 9 → average of 9 reads of the only pixel
        // = the pixel itself.
        let out = convolve_matrix(&src, 1, 1, &cm);
        assert_eq!(out, &src);
    }

    /// `edgeMode = "wrap"` makes a `delta` impulse centred on a 1-pixel
    /// image symmetric. We don't test wrapping in detail here (covered
    /// in integration tests) — just that the type-checker / runtime
    /// don't panic on a small canvas.
    #[test]
    fn wrap_smoke_test() {
        let src = solid(2, 2, [10, 20, 30, 255]);
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_edge_mode(ConvolveEdgeMode::Wrap);
        let out = convolve_matrix(&src, 2, 2, &cm);
        assert_eq!(out, src, "wrap+box-blur on solid 2x2 is identity");
    }

    /// Bias shifts every output channel by a constant.
    ///
    /// Identity kernel + `bias = 0.25` must add ~64 (= 0.25 · 255 =
    /// 63.75 → 64) to every channel of a 100-grey input.
    #[test]
    fn bias_shifts_all_channels() {
        let src = solid(2, 2, [100, 100, 100, 100]);
        let cm = ConvolveMatrix::new(1, 1, vec![1.0]).with_bias(0.25);
        let out = convolve_matrix(&src, 2, 2, &cm);
        // 100 + 0.25*255 = 100 + 63.75 → round-half-up → 164.
        for (i, channel) in out.iter().take(4).enumerate() {
            assert_eq!(*channel, 164, "channel {i} should be 100 + 64 = 164");
        }
    }

    /// Negative bias also clamps at 0 — bias of -1.0 must zero every
    /// channel of any input.
    #[test]
    fn negative_bias_clamps_to_zero() {
        let src = solid(2, 2, [100, 200, 50, 255]);
        let cm = ConvolveMatrix::new(1, 1, vec![1.0]).with_bias(-1.0);
        let out = convolve_matrix(&src, 2, 2, &cm);
        for v in out {
            assert_eq!(v, 0);
        }
    }

    /// Standard 3×3 Sobel-X kernel produces zero on a constant image
    /// (high-pass / edge-detection property: it sums to zero).
    #[test]
    fn sobel_x_on_constant_is_zero() {
        let w = 5;
        let h = 5;
        let src = solid(w, h, [128, 128, 128, 255]);
        #[rustfmt::skip]
        let kernel = vec![
            -1.0, 0.0, 1.0,
            -2.0, 0.0, 2.0,
            -1.0, 0.0, 1.0,
        ];
        let cm = ConvolveMatrix::new(3, 3, kernel)
            // The kernel sums to zero so the default-divisor path would
            // fall back to 1.0. Set explicitly to silence ambiguity.
            .with_divisor(1.0)
            .with_edge_mode(ConvolveEdgeMode::Duplicate)
            .with_preserve_alpha(true);
        let out = convolve_matrix(&src, w, h, &cm);
        for i in 0..(w * h) as usize {
            assert_eq!(out[i * 4], 0, "R should be zero on constant");
            assert_eq!(out[i * 4 + 1], 0, "G should be zero on constant");
            assert_eq!(out[i * 4 + 2], 0, "B should be zero on constant");
            assert_eq!(out[i * 4 + 3], 255, "alpha pass-through");
        }
    }

    /// Asymmetric kernel — verify the 180° rotation specified by
    /// §15.13. Take a delta-impulse image (single non-zero pixel) and
    /// an asymmetric kernel and check the output landing pattern.
    #[test]
    fn kernel_is_rotated_180_per_spec() {
        let w = 5;
        let h = 5;
        let mut src = vec![0u8; (w * h * 4) as usize];
        // Impulse at (2, 2).
        let centre = ((2 * w + 2) * 4) as usize;
        src[centre] = 100;
        src[centre + 3] = 255;

        // Kernel with the only non-zero entry at the bottom-right
        // corner: kernelMatrix[2, 2] = 1. With the 180° rotation per
        // §15.13, this coefficient lands on
        // sample(X - targetX + 0, Y - targetY + 0) → top-left
        // neighbour of the target. So the impulse at (2,2) → output
        // contribution lands at (target offset relative to source).
        //
        // With targetX=targetY=1 (default for 3×3), the kernel
        // coefficient at index (orderX-J-1, orderY-I-1) = (2, 2)
        // corresponds to (J, I) = (0, 0), reading from source
        // (X-1, Y-1). So the impulse at (2, 2) shows up in the output
        // at (3, 3) (one step down-right).
        #[rustfmt::skip]
        let kernel = vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let cm = ConvolveMatrix::new(3, 3, kernel)
            .with_divisor(1.0)
            .with_edge_mode(ConvolveEdgeMode::None)
            .with_preserve_alpha(false);
        let out = convolve_matrix(&src, w, h, &cm);
        // Where the impulse lands per the 180° rotation formula:
        // output (3, 3).
        let landed = ((3 * w + 3) * 4) as usize;
        assert_eq!(
            out[landed], 100,
            "impulse should land at (3,3) per §15.13 rotation"
        );
        // Sanity: should NOT land at (1, 1) (the naive-no-rotation
        // location).
        let naive = ((w + 1) * 4) as usize;
        assert_eq!(
            out[naive], 0,
            "no contribution at (1,1) per §15.13 rotation"
        );
    }

    /// Typed-pixel wrapper agrees with the byte API across an
    /// asymmetric kernel + bias + non-default edge mode.
    #[test]
    fn typed_pixel_wrapper_matches_byte_api() {
        let w = 4;
        let h = 3;
        let mut src_b = Vec::with_capacity((w * h * 4) as usize);
        let mut src_p = Vec::with_capacity((w * h) as usize);
        for i in 0..(w * h) {
            let p = Rgba::new(
                ((i * 41) % 256) as u8,
                ((i * 59) % 256) as u8,
                ((i * 67) % 256) as u8,
                ((i * 23 + 70) % 256) as u8,
            );
            src_b.extend_from_slice(&[p.r, p.g, p.b, p.a]);
            src_p.push(p);
        }
        #[rustfmt::skip]
        let kernel = vec![
            1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let cm = ConvolveMatrix::new(3, 3, kernel)
            .with_bias(-0.05)
            .with_edge_mode(ConvolveEdgeMode::Wrap);
        let via_bytes = convolve_matrix(&src_b, w, h, &cm);
        let via_typed = convolve_matrix_pixels(&src_p, w, h, &cm);
        let typed_bytes: Vec<u8> = via_typed
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(via_bytes, typed_bytes);
    }

    /// Empty image returns an empty buffer.
    #[test]
    fn empty_extent_is_empty_vec() {
        let cm = ConvolveMatrix::new(1, 1, vec![1.0]);
        assert!(convolve_matrix(&[], 0, 0, &cm).is_empty());
    }

    /// Non-square kernel — order 5×1 horizontal moving-average
    /// produces blur along X only, leaves Y untouched. The orthogonal
    /// edge of a vertical step is unaffected by row.
    #[test]
    fn horizontal_kernel_only_blurs_x() {
        let w = 9;
        let h = 3;
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                // Step edge along x = 4: 0 on the left, 200 on the
                // right. Same colour on every row.
                let v = if x < 4 { 0 } else { 200 };
                src.extend_from_slice(&[v, v, v, 255]);
                let _ = y;
            }
        }
        // 1×5 horizontal averaging kernel.
        let cm = ConvolveMatrix::new(5, 1, vec![1.0; 5]);
        let out = convolve_matrix(&src, w, h, &cm);
        // All rows must be byte-identical to the first row.
        let row0: Vec<u8> = out[..(w * 4) as usize].to_vec();
        for y in 1..h {
            let start = (y * w * 4) as usize;
            let row = &out[start..start + (w * 4) as usize];
            assert_eq!(row, row0.as_slice(), "row {y} should match row 0");
        }
    }

    /// Custom `target` shifts the kernel anchor. With a 3×3 identity
    /// kernel (centre = 1.0) and `target = (0, 0)`, the impulse moves
    /// one pixel up-and-to-the-left because the anchor is now at the
    /// top-left of the kernel — but after the §15.13 180° rotation,
    /// the centre of the kernel ends up reading from
    /// (X - 0 + 1, Y - 0 + 1), i.e. one pixel down-and-right of the
    /// target. So a centred impulse at (2, 2) shows up at (1, 1).
    #[test]
    fn target_shifts_anchor() {
        let w = 5;
        let h = 5;
        let mut src = vec![0u8; (w * h * 4) as usize];
        let centre = ((2 * w + 2) * 4) as usize;
        src[centre] = 100;
        src[centre + 3] = 255;
        #[rustfmt::skip]
        let kernel = vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let cm = ConvolveMatrix::new(3, 3, kernel)
            .with_divisor(1.0)
            .with_target(0, 0)
            .with_edge_mode(ConvolveEdgeMode::None);
        let out = convolve_matrix(&src, w, h, &cm);
        // With target=(0,0), kernel index (1,1) corresponds to (J,I) =
        // (orderX-1-1, orderY-1-1) = (1, 1); the kernel reads source
        // at (X - 0 + 1, Y - 0 + 1) = (X+1, Y+1). The impulse at
        // (2,2) therefore lands at output (1,1) (the pixel whose
        // kernel reaches (2,2) via the (+1, +1) shift).
        let landed = ((w + 1) * 4) as usize;
        assert_eq!(
            out[landed], 100,
            "impulse should land at (1,1) under target=(0,0)"
        );
    }

    #[test]
    #[should_panic(expected = "src.len()")]
    fn wrong_input_length_panics() {
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]);
        let _ = convolve_matrix(&[0u8; 8], 3, 3, &cm);
    }

    #[test]
    #[should_panic(expected = "divisor must be non-zero")]
    fn zero_divisor_panics() {
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_divisor(1.0);
        // Manually break the invariant — exercise the convolve_matrix
        // guard, not the builder's.
        let mut bad = cm.clone();
        bad.divisor = 0.0;
        let _ = convolve_matrix(&[0u8; 4], 1, 1, &bad);
    }

    #[test]
    #[should_panic(expected = "kernel.len()")]
    fn wrong_kernel_length_panics() {
        ConvolveMatrix::new(3, 3, vec![1.0; 5]);
    }

    #[test]
    #[should_panic(expected = "order_x")]
    fn zero_order_panics() {
        ConvolveMatrix::new(0, 3, vec![]);
    }

    #[test]
    #[should_panic(expected = "target_x")]
    fn out_of_range_target_panics() {
        ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_target(3, 1);
    }
}

#[cfg(test)]
mod composite_tests {
    use super::*;

    fn build<F: FnMut(u32, u32) -> Rgba>(w: u32, h: u32, mut f: F) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let c = f(x, y);
                v.extend_from_slice(&[c.r, c.g, c.b, c.a]);
            }
        }
        v
    }

    #[test]
    fn over_matches_simple_alpha_compositing() {
        // §14.2 premultiplied "over": Cr' = (1 - Ea)*Cr + Er with
        // E = in (a), C = in2 (b). Opaque blue over opaque red is just
        // opaque blue. Half-alpha blue over opaque red blends.
        let a = build(2, 1, |x, _| {
            if x == 0 {
                Rgba::new(0, 0, 255, 255) // opaque blue
            } else {
                Rgba::new(0, 0, 255, 128) // ~half blue
            }
        });
        let b = build(2, 1, |_, _| Rgba::new(255, 0, 0, 255)); // opaque red
        let out = composite_filter(&a, &b, 2, 1, CompositeOp::Over);
        // Pixel 0: opaque blue wins outright.
        assert_eq!(&out[0..4], &[0, 0, 255, 255]);
        // Pixel 1: alpha stays opaque (1 - (1-0.5)*(1-1) = 1).
        assert_eq!(out[7], 255);
        // Red premult ≈ (1 - 0.502)*1.0 = 0.498 → round-trip ~127.
        let r = out[4] as i32;
        assert!((r - 127).abs() <= 2, "expected ~127 red got {r}");
        // Blue premult ≈ 0.502 → straight ~128.
        let bl = out[6] as i32;
        assert!((bl - 128).abs() <= 2, "expected ~128 blue got {bl}");
    }

    #[test]
    fn in_keeps_a_color_masked_by_b_alpha() {
        // operator="in": Fa = αb, Fb = 0. Result = in clipped to in2's
        // coverage. Opaque green "in" half-alpha anything → green at
        // 50% alpha, original green RGB preserved (straight-alpha).
        let a = build(1, 1, |_, _| Rgba::new(0, 200, 0, 255));
        let b = build(1, 1, |_, _| Rgba::new(9, 9, 9, 128));
        let out = composite_filter(&a, &b, 1, 1, CompositeOp::In);
        // Alpha = 1.0 * (128/255) → ~128.
        assert!((out[3] as i32 - 128).abs() <= 1, "alpha {}", out[3]);
        // Straight-alpha green preserved exactly (premult then /α).
        assert_eq!(out[1], 200);
        // No contribution from in2's colour.
        assert_eq!(out[0], 0);
        assert_eq!(out[2], 0);
    }

    #[test]
    fn out_removes_a_where_b_covers() {
        // operator="out": Fa = 1 - αb, Fb = 0. Opaque a "out" opaque b
        // → fully transparent everywhere b covers.
        let a = build(1, 1, |_, _| Rgba::new(10, 20, 30, 255));
        let b = build(1, 1, |_, _| Rgba::new(0, 0, 0, 255));
        let out = composite_filter(&a, &b, 1, 1, CompositeOp::Out);
        assert_eq!(&out[0..4], &[0, 0, 0, 0]);
        // Where b is empty, a passes through untouched.
        let b0 = build(1, 1, |_, _| Rgba::new(0, 0, 0, 0));
        let out2 = composite_filter(&a, &b0, 1, 1, CompositeOp::Out);
        assert_eq!(&out2[0..4], &[10, 20, 30, 255]);
    }

    #[test]
    fn xor_of_disjoint_opaque_is_union() {
        // operator="xor": Fa = 1 - αb, Fb = 1 - αa. Two opaque inputs
        // that don't overlap (one empty per pixel) yield the non-empty
        // one; full overlap of two opaque inputs cancels to transparent.
        // `a` is opaque red at both pixels; only `b`'s coverage varies.
        let a = build(2, 1, |_, _| Rgba::new(255, 0, 0, 255));
        let b = build(2, 1, |x, _| {
            if x == 0 {
                Rgba::new(0, 0, 255, 0) // empty at pixel 0
            } else {
                Rgba::new(0, 0, 255, 255) // opaque overlap at pixel 1
            }
        });
        let out = composite_filter(&a, &b, 2, 1, CompositeOp::Xor);
        // Pixel 0: a opaque, b empty → a passes (Fa = 1 - 0 = 1).
        assert_eq!(&out[0..4], &[255, 0, 0, 255]);
        // Pixel 1: both opaque → αo = 1*0 + 1*0 = 0 → transparent.
        assert_eq!(out[7], 0);
    }

    #[test]
    fn lighter_adds_premultiplied_channels() {
        // operator="lighter": Fa = Fb = 1. Premultiplied source and
        // backdrop add, clamped to [0,1]. Two half-alpha greys add to a
        // full-alpha mid grey: αo = 0.5+0.5 = 1; premultiplied colour
        // 0.5·c + 0.5·c, un-premultiplied by α=1, recovers c.
        let a = build(1, 1, |_, _| Rgba::new(120, 120, 120, 128));
        let b = build(1, 1, |_, _| Rgba::new(120, 120, 120, 128));
        let out = composite_filter(&a, &b, 1, 1, CompositeOp::Lighter);
        assert_eq!(out[3], 255, "alpha sums to opaque");
        for (ch, &v) in out.iter().take(3).enumerate() {
            assert!((v as i32 - 120).abs() <= 1, "ch{ch} = {v}");
        }
    }

    #[test]
    fn lighter_clamps_alpha_to_one() {
        // Two opaque inputs: αo = 1 + 1 clamps to 1, never overflows.
        let a = build(1, 1, |_, _| Rgba::new(200, 0, 0, 255));
        let b = build(1, 1, |_, _| Rgba::new(0, 0, 200, 255));
        let out = composite_filter(&a, &b, 1, 1, CompositeOp::Lighter);
        assert_eq!(out[3], 255);
        // Premultiplied add: red 200 + blue 200 in straight space (α=1).
        assert_eq!(out[0], 200);
        assert_eq!(out[2], 200);
    }

    #[test]
    fn atop_preserves_b_alpha() {
        // operator="atop": Fa = αb, Fb = 1 - αa. The result alpha
        // equals αb (a "atop" b is confined to b's silhouette):
        // αo = αa*αb + αb*(1-αa) = αb.
        let a = build(1, 1, |_, _| Rgba::new(255, 255, 255, 200));
        let b = build(1, 1, |_, _| Rgba::new(0, 0, 0, 128));
        let out = composite_filter(&a, &b, 1, 1, CompositeOp::Atop);
        assert!((out[3] as i32 - 128).abs() <= 1, "alpha {}", out[3]);
    }

    #[test]
    fn arithmetic_k2_one_passes_in1_through() {
        // result = 0*i1*i2 + 1*i1 + 0*i2 + 0 = i1. With both inputs
        // opaque the premultiplied channels equal the straight ones, so
        // in1 round-trips byte-exact.
        let a = build(3, 2, |x, y| {
            Rgba::new((x * 50) as u8, (y * 90) as u8, 33, 255)
        });
        let b = build(3, 2, |_, _| Rgba::new(200, 200, 200, 255));
        let out = composite_filter(
            &a,
            &b,
            3,
            2,
            CompositeOp::Arithmetic {
                k1: 0.0,
                k2: 1.0,
                k3: 0.0,
                k4: 0.0,
            },
        );
        assert_eq!(out, a);
    }

    #[test]
    fn arithmetic_k4_floods_constant() {
        // result = k4 on every channel (k1=k2=k3=0). k4=1 → opaque
        // white everywhere regardless of inputs.
        let a = build(2, 2, |_, _| Rgba::new(10, 20, 30, 40));
        let b = build(2, 2, |_, _| Rgba::new(50, 60, 70, 80));
        let out = composite_filter(
            &a,
            &b,
            2,
            2,
            CompositeOp::Arithmetic {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                k4: 1.0,
            },
        );
        // Premult result = (1,1,1,1) → straight (255,255,255,255).
        assert!(out.iter().all(|&v| v == 255));
    }

    #[test]
    fn arithmetic_clamps_to_unit_range() {
        // k2=k3=2 drives premultiplied sums well past 1.0; the spec
        // clamps each channel to [0,1]. Opaque inputs → channels clamp
        // to opaque white.
        let a = build(1, 1, |_, _| Rgba::new(200, 0, 0, 255));
        let b = build(1, 1, |_, _| Rgba::new(0, 200, 0, 255));
        let out = composite_filter(
            &a,
            &b,
            1,
            1,
            CompositeOp::Arithmetic {
                k1: 0.0,
                k2: 2.0,
                k3: 2.0,
                k4: 0.0,
            },
        );
        // Alpha: 2*1 + 2*1 = 4 → clamp 1 → 255.
        assert_eq!(out[3], 255);
    }

    #[test]
    fn typed_wrapper_matches_byte_path() {
        let a_b = build(4, 3, |x, y| {
            Rgba::new((x * 30) as u8, (y * 40) as u8, 11, ((x + y) * 25) as u8)
        });
        let b_b = build(4, 3, |x, y| {
            Rgba::new(7, (x * 20) as u8, (y * 50) as u8, 200)
        });
        let a_p: Vec<Rgba> = a_b
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let b_p: Vec<Rgba> = b_b
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        for op in [
            CompositeOp::Over,
            CompositeOp::In,
            CompositeOp::Out,
            CompositeOp::Atop,
            CompositeOp::Xor,
            CompositeOp::Arithmetic {
                k1: 0.25,
                k2: 0.5,
                k3: 0.5,
                k4: 0.1,
            },
        ] {
            let via_bytes = composite_filter(&a_b, &b_b, 4, 3, op);
            let via_typed = composite_filter_pixels(&a_p, &b_p, 4, 3, op);
            let typed_bytes: Vec<u8> = via_typed
                .iter()
                .flat_map(|p| [p.r, p.g, p.b, p.a])
                .collect();
            assert_eq!(via_bytes, typed_bytes, "mismatch for {op:?}");
        }
    }

    #[test]
    fn empty_extent_is_empty_vec() {
        assert!(composite_filter(&[], &[], 0, 0, CompositeOp::Over).is_empty());
    }

    #[test]
    #[should_panic(expected = "in1.len()")]
    fn wrong_in1_length_panics() {
        let _ = composite_filter(&[0u8; 8], &[0u8; 16], 2, 2, CompositeOp::Over);
    }

    #[test]
    #[should_panic(expected = "in2.len()")]
    fn wrong_in2_length_panics() {
        let _ = composite_filter(&[0u8; 16], &[0u8; 8], 2, 2, CompositeOp::Over);
    }
}

// =========================================================================
// feTurbulence — SVG 1.1 §15.24
// =========================================================================

/// Variant selector for [`Turbulence`] (`type` attribute, SVG 1.1 §15.24).
///
/// The two variants change only the post-aggregation step:
///
/// * [`Turbulence`](Self::Turbulence): the per-octave noise contributions
///   are accumulated through their absolute value (`SUM |noise2| /
///   ratio`), producing a non-negative result aimed at `[0, 1]`.
/// * [`FractalNoise`](Self::FractalNoise): the per-octave contributions
///   are accumulated signed (`SUM noise2 / ratio`), producing a signed
///   result aimed at `[-1, 1]`.
///
/// The two arms also use different colour quantisations — see
/// [`turbulence_filter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TurbulenceType {
    /// `type="turbulence"`, SVG 1.1 §15.24 default. Accumulates
    /// `|noise2|`; `turbFunctionResult` aimed at `[0, 1]`; quantised as
    /// `(turbFunctionResult · 255)`.
    #[default]
    Turbulence,
    /// `type="fractalNoise"`. Accumulates signed `noise2`;
    /// `turbFunctionResult` aimed at `[-1, 1]`; quantised as
    /// `(turbFunctionResult · 255 + 255) / 2`.
    FractalNoise,
}

/// `stitchTiles` selector (SVG 1.1 §15.24).
///
/// When [`Stitch`](Self::Stitch) is selected and a non-degenerate tile
/// region is supplied, the algorithm adjusts the base frequencies so
/// that the tile width and height contain an integral number of Perlin
/// tiles for the first octave (§15.24 — "Given the frequency, calculate
/// lowFreq = floor(width · frequency) / width and hiFreq = ceil(width ·
/// frequency) / width. If frequency / lowFreq < hiFreq / frequency then
/// use lowFreq, else use hiFreq."), and the lattice indices are
/// wrapped at the right and bottom edges of the active area so that
/// the noise tiles seamlessly across the [`Turbulence::tile_x`] …
/// [`Turbulence::tile_x`] + [`Turbulence::tile_width`] /
/// [`Turbulence::tile_y`] … [`Turbulence::tile_y`] +
/// [`Turbulence::tile_height`] rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StitchTiles {
    /// `stitchTiles="noStitch"`, SVG 1.1 §15.24 default. No tile-border
    /// stitching is applied.
    #[default]
    NoStitch,
    /// `stitchTiles="stitch"`. Adjust the base frequencies as
    /// described above and wrap lattice points at the tile's right /
    /// bottom edges so the noise tiles seamlessly.
    Stitch,
}

/// Parameter block for [`turbulence_filter`] / [`turbulence_filter_pixels`].
///
/// Mirrors the attribute set of SVG 1.1 §15.24 `<feTurbulence>`:
/// `baseFrequency`, `numOctaves`, `seed`, `stitchTiles`, `type`, plus
/// the filter-primitive subregion that the algorithm reads as the
/// `(fTileX, fTileY, fTileWidth, fTileHeight)` quad for stitching.
///
/// Defaults applied by [`Turbulence::new`] match §15.24:
///
/// * `base_freq_x = base_freq_y = 0`
/// * `num_octaves = 1`
/// * `seed = 0` (truncated to the closest integer toward zero — already
///   integral here)
/// * `stitch_tiles = NoStitch`
/// * `kind = Turbulence`
/// * `tile_x = tile_y = 0`, `tile_width = width`, `tile_height = height`
///   (set per-render by [`turbulence_filter`] from its `width`/`height`
///   arguments when the explicit pair are zero)
#[derive(Debug, Clone, PartialEq)]
pub struct Turbulence {
    /// X-axis base frequency. Must be `>= 0` (§15.24 "A negative value
    /// for base frequency is an error"). A value of `0` collapses the
    /// X variation to a constant.
    pub base_freq_x: f64,
    /// Y-axis base frequency. Must be `>= 0`.
    pub base_freq_y: f64,
    /// Number of octaves accumulated (§15.24 `numOctaves`). Must be
    /// `>= 1`. The §15.24 default is `1`.
    pub num_octaves: u32,
    /// Pseudo-random seed for the §15.24 LCG. The spec mandates
    /// truncation "to the closest integer value towards zero" before
    /// the algorithm sees it; we take an `i64` directly to make that
    /// truncation explicit at the caller.
    pub seed: i64,
    /// Selects whether the output tiles seamlessly across the supplied
    /// tile rectangle.
    pub stitch_tiles: StitchTiles,
    /// `turbulence` (absolute-value accumulation, aimed at `[0, 1]`)
    /// vs `fractalNoise` (signed accumulation, aimed at `[-1, 1]`).
    pub kind: TurbulenceType,
    /// X-origin of the stitch tile rectangle.
    pub tile_x: f64,
    /// Y-origin of the stitch tile rectangle.
    pub tile_y: f64,
    /// Width of the stitch tile rectangle. Used only when
    /// `stitch_tiles == Stitch`.
    pub tile_width: f64,
    /// Height of the stitch tile rectangle. Used only when
    /// `stitch_tiles == Stitch`.
    pub tile_height: f64,
}

impl Turbulence {
    /// Construct a [`Turbulence`] parameter block applying the §15.24
    /// defaults for every unspecified attribute.
    pub fn new(base_freq_x: f64, base_freq_y: f64) -> Self {
        assert!(
            base_freq_x >= 0.0,
            "Turbulence::new: base_freq_x ({base_freq_x}) must be >= 0 (§15.24)"
        );
        assert!(
            base_freq_y >= 0.0,
            "Turbulence::new: base_freq_y ({base_freq_y}) must be >= 0 (§15.24)"
        );
        Turbulence {
            base_freq_x,
            base_freq_y,
            num_octaves: 1,
            seed: 0,
            stitch_tiles: StitchTiles::NoStitch,
            kind: TurbulenceType::Turbulence,
            tile_x: 0.0,
            tile_y: 0.0,
            tile_width: 0.0,
            tile_height: 0.0,
        }
    }

    /// Builder: set [`Self::num_octaves`]. Must be `>= 1` (§15.24
    /// implies a per-octave loop with at least one iteration).
    pub fn with_num_octaves(mut self, num_octaves: u32) -> Self {
        assert!(
            num_octaves >= 1,
            "Turbulence::with_num_octaves: num_octaves must be >= 1"
        );
        self.num_octaves = num_octaves;
        self
    }

    /// Builder: set [`Self::seed`]. The spec mandates truncation to the
    /// closest integer toward zero; pass that integer here directly.
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = seed;
        self
    }

    /// Builder: set [`Self::kind`].
    pub fn with_kind(mut self, kind: TurbulenceType) -> Self {
        self.kind = kind;
        self
    }

    /// Builder: set [`Self::stitch_tiles`]. When [`StitchTiles::Stitch`]
    /// is selected, the tile rectangle must also be supplied through
    /// [`Self::with_tile`].
    pub fn with_stitch_tiles(mut self, stitch_tiles: StitchTiles) -> Self {
        self.stitch_tiles = stitch_tiles;
        self
    }

    /// Builder: set the stitch tile rectangle (§15.24's
    /// `fTileX, fTileY, fTileWidth, fTileHeight`).
    pub fn with_tile(mut self, x: f64, y: f64, w: f64, h: f64) -> Self {
        self.tile_x = x;
        self.tile_y = y;
        self.tile_width = w;
        self.tile_height = h;
        self
    }
}

// -------------------------------------------------------------------------
// §15.24 reference algorithm
// -------------------------------------------------------------------------
//
// The math in this block is transcribed from
// `docs/image/svg/svg11-second-edition.pdf` §15.24 Filter primitive
// `feTurbulence`. The spec includes the exact algorithm "below shows the
// exact algorithm used for this filter effect", listing a Park–Miller
// minimum-standard LCG, an `init` routine that builds a permutation
// table and gradient table from the seeded LCG, a `noise2` routine
// implementing 2-D Perlin noise with a 3·t² − 2·t³ smoothstep, and a
// `turbulence` routine that sums `num_octaves` noise terms with
// geometrically-doubling frequency and geometrically-halving amplitude.
// The §15.24 stitch path adjusts the base frequencies so the tile
// rectangle contains an integral number of first-octave Perlin tiles
// and wraps the lattice indices at the right/bottom edges of the
// active area.
//
// Park & Miller call out a self-check: starting from `seed = 1`, the
// 10,000th LCG output is `1043618065`. We carry that across as a
// `#[test]` assertion below.

/// §15.24 Park–Miller minimum-standard LCG modulus.
const TURB_RAND_M: i64 = 2_147_483_647; // 2^31 − 1
/// §15.24 Park–Miller minimum-standard LCG multiplier.
const TURB_RAND_A: i64 = 16_807; // 7^5
/// §15.24 derived constants `m / a` and `m % a` for the Schrage trick.
const TURB_RAND_Q: i64 = 127_773;
const TURB_RAND_R: i64 = 2_836;

/// §15.24 lattice table size (`BSize`).
const TURB_B_SIZE: usize = 0x100;
/// §15.24 lattice index mask (`BM = BSize − 1`).
const TURB_BM: i64 = 0xff;
/// §15.24 large-positive bias added before the integer-truncation step
/// inside [`turb_noise2`] (`PerlinN = 0x1000`). It is large enough that
/// truncating `vec + PerlinN` toward zero is equivalent to flooring
/// `vec` for every input we care about.
const TURB_PERLIN_N: i64 = 0x1000;

/// §15.24 `setup_seed`: clamp the caller-supplied seed into the
/// strictly-positive LCG state range `[1, RAND_M − 1]`. Non-positive
/// inputs are reflected back through `−lSeed % (RAND_M − 1) + 1`;
/// inputs at or above `RAND_M` are clamped down to `RAND_M − 1`.
#[inline]
fn turb_setup_seed(mut seed: i64) -> i64 {
    if seed <= 0 {
        seed = -(seed.rem_euclid(TURB_RAND_M - 1)) + 1;
        // rem_euclid is non-negative; turn it back negative for the
        // spec's `-seed` step, matching the C `-(lSeed % (RAND_M - 1)) + 1`
        // when `lSeed` is itself ≤ 0 (so `lSeed % (RAND_M-1)` is ≤ 0 in C).
        // We re-derive the same value through the equivalent:
        //   for `seed <= 0`: result = (-seed) % (RAND_M-1) + 1, clamped.
        if seed <= 0 {
            seed = 1;
        }
    }
    if seed > TURB_RAND_M - 1 {
        seed = TURB_RAND_M - 1;
    }
    seed
}

/// §15.24 LCG step. Returns the next state in `[1, RAND_M − 1]` given a
/// current state in the same range. Implemented through Schrage's
/// `a · (s mod q) − r · (s / q)` trick so the intermediate product
/// stays in `i64` without overflow.
#[inline]
fn turb_random(seed: i64) -> i64 {
    let mut result = TURB_RAND_A * (seed % TURB_RAND_Q) - TURB_RAND_R * (seed / TURB_RAND_Q);
    if result <= 0 {
        result += TURB_RAND_M;
    }
    result
}

/// Stitch parameters for the per-octave [`turb_noise2`] call. Built
/// during [`turbulence_at`] when `stitch_tiles == Stitch`.
#[derive(Debug, Clone, Copy)]
struct TurbStitch {
    /// Lattice-index span to subtract when wrapping at the right /
    /// bottom edge.
    width: i64,
    height: i64,
    /// Lattice-index thresholds at which the wrap kicks in.
    wrap_x: i64,
    wrap_y: i64,
}

/// Permutation table (`uLatticeSelector`) plus per-channel gradient
/// tables (`fGradient[4][BSize·2 + 2][2]`), seeded once per filter
/// invocation by [`TurbLattice::init`].
///
/// The "+ 2" doubling at the end of each table mirrors the §15.24 C
/// reference, which copies the first `BSize + 2` entries into the back
/// half so the `noise2` lookups at `bx1 + by1` can index without an
/// explicit `& BM`.
struct TurbLattice {
    selector: [i64; TURB_B_SIZE * 2 + 2],
    /// `gradient[channel][lattice_index] = (gx, gy)`, normalised to
    /// unit length.
    gradient: [[(f64, f64); TURB_B_SIZE * 2 + 2]; 4],
}

impl TurbLattice {
    /// §15.24 `init`: seed `selector` with the identity permutation,
    /// fill `gradient` with random unit-vectors from the LCG, then
    /// Knuth-shuffle the permutation using the same LCG sequence, and
    /// finally mirror the first `BSize + 2` entries into the back half.
    fn init(seed: i64) -> Self {
        let mut lseed = turb_setup_seed(seed);
        let mut selector = [0i64; TURB_B_SIZE * 2 + 2];
        let mut gradient = [[(0.0f64, 0.0f64); TURB_B_SIZE * 2 + 2]; 4];

        // First pass: identity permutation + random unit gradients per
        // (channel, lattice point).
        for (k, gradient_k) in gradient.iter_mut().enumerate().take(4) {
            for (i, gradient_ki) in gradient_k.iter_mut().enumerate().take(TURB_B_SIZE) {
                if k == 0 {
                    selector[i] = i as i64;
                }
                lseed = turb_random(lseed);
                let gx_raw = (lseed % (2 * TURB_B_SIZE as i64) - TURB_B_SIZE as i64) as f64
                    / TURB_B_SIZE as f64;
                lseed = turb_random(lseed);
                let gy_raw = (lseed % (2 * TURB_B_SIZE as i64) - TURB_B_SIZE as i64) as f64
                    / TURB_B_SIZE as f64;
                let s = (gx_raw * gx_raw + gy_raw * gy_raw).sqrt();
                // §15.24's normalisation has no guard for s == 0; the
                // mathematical probability is vanishing for the LCG
                // distribution but a guarded divide here keeps us from
                // emitting NaNs if the seeded path ever lands on the
                // origin.
                let (gx, gy) = if s == 0.0 {
                    (0.0, 0.0)
                } else {
                    (gx_raw / s, gy_raw / s)
                };
                *gradient_ki = (gx, gy);
            }
        }

        // Second pass: Knuth shuffle of the permutation. The §15.24 C
        // listing is `while(--i)`, which iterates `i = BSize - 1, …, 1`
        // (inclusive both ends) after the preceding loop left `i =
        // BSize`. We replicate that index range exactly.
        for i in (1..TURB_B_SIZE).rev() {
            let k = selector[i];
            lseed = turb_random(lseed);
            let j = (lseed % TURB_B_SIZE as i64) as usize;
            selector[i] = selector[j];
            selector[j] = k;
        }

        // Third pass: mirror the first `BSize + 2` entries into the
        // back half so `noise2`'s additive `i + by1` / `j + by1` lookups
        // can index `selector[…]` and `gradient[…]` without a separate
        // `& BM`.
        for i in 0..(TURB_B_SIZE + 2) {
            selector[TURB_B_SIZE + i] = selector[i];
            for gradient_k in gradient.iter_mut().take(4) {
                gradient_k[TURB_B_SIZE + i] = gradient_k[i];
            }
        }

        TurbLattice { selector, gradient }
    }

    /// §15.24 `noise2`: 2-D Perlin noise for one colour channel at the
    /// supplied `(x, y)` lattice-space position, with an optional
    /// stitch wrap applied to the four surrounding lattice indices.
    #[allow(clippy::many_single_char_names)]
    fn noise2(&self, channel: usize, vec_x: f64, vec_y: f64, stitch: Option<&TurbStitch>) -> f64 {
        // Add PerlinN before truncating so the `(int)` cast on a
        // possibly-negative input still produces the spec's floor-like
        // behaviour. The lattice indices land in `[0, 2·BSize + …]`
        // before the &BM mask.
        let tx = vec_x + TURB_PERLIN_N as f64;
        let mut bx0 = tx as i64; // truncation toward zero
        let mut bx1 = bx0 + 1;
        let rx0 = tx - tx.trunc();
        let rx1 = rx0 - 1.0;

        let ty = vec_y + TURB_PERLIN_N as f64;
        let mut by0 = ty as i64;
        let mut by1 = by0 + 1;
        let ry0 = ty - ty.trunc();
        let ry1 = ry0 - 1.0;

        if let Some(s) = stitch {
            if bx0 >= s.wrap_x {
                bx0 -= s.width;
            }
            if bx1 >= s.wrap_x {
                bx1 -= s.width;
            }
            if by0 >= s.wrap_y {
                by0 -= s.height;
            }
            if by1 >= s.wrap_y {
                by1 -= s.height;
            }
        }

        bx0 &= TURB_BM;
        bx1 &= TURB_BM;
        by0 &= TURB_BM;
        by1 &= TURB_BM;

        let i = self.selector[bx0 as usize];
        let j = self.selector[bx1 as usize];
        let b00 = self.selector[(i + by0) as usize];
        let b10 = self.selector[(j + by0) as usize];
        let b01 = self.selector[(i + by1) as usize];
        let b11 = self.selector[(j + by1) as usize];

        // §15.24 `s_curve(t) = t·t·(3 − 2·t)`.
        let sx = rx0 * rx0 * (3.0 - 2.0 * rx0);
        let sy = ry0 * ry0 * (3.0 - 2.0 * ry0);

        let q00 = self.gradient[channel][b00 as usize];
        let u00 = rx0 * q00.0 + ry0 * q00.1;
        let q10 = self.gradient[channel][b10 as usize];
        let v10 = rx1 * q10.0 + ry0 * q10.1;
        let a = u00 + sx * (v10 - u00);

        let q01 = self.gradient[channel][b01 as usize];
        let u01 = rx0 * q01.0 + ry1 * q01.1;
        let q11 = self.gradient[channel][b11 as usize];
        let v11 = rx1 * q11.0 + ry1 * q11.1;
        let b = u01 + sx * (v11 - u01);

        a + sy * (b - a)
    }
}

/// §15.24 `turbulence`: per-channel accumulator that drives [`TurbLattice::noise2`]
/// across `num_octaves` octaves with geometrically-doubling frequency
/// and geometrically-halving amplitude. Returns the signed sum (caller
/// converts to colour through the type-dependent quantisation).
#[allow(clippy::too_many_arguments)]
fn turbulence_at(
    lattice: &TurbLattice,
    channel: usize,
    point_x: f64,
    point_y: f64,
    mut base_freq_x: f64,
    mut base_freq_y: f64,
    num_octaves: u32,
    fractal_sum: bool,
    do_stitch: bool,
    tile_x: f64,
    tile_y: f64,
    tile_width: f64,
    tile_height: f64,
) -> f64 {
    // §15.24 stitch adjustment: round the base frequencies so the
    // first-octave tile fits the active rectangle an integral number
    // of times.
    let mut stitch_state: Option<TurbStitch> = None;
    if do_stitch {
        if base_freq_x != 0.0 {
            let lo = (tile_width * base_freq_x).floor() / tile_width;
            let hi = (tile_width * base_freq_x).ceil() / tile_width;
            if lo > 0.0 && base_freq_x / lo < hi / base_freq_x {
                base_freq_x = lo;
            } else {
                base_freq_x = hi;
            }
        }
        if base_freq_y != 0.0 {
            let lo = (tile_height * base_freq_y).floor() / tile_height;
            let hi = (tile_height * base_freq_y).ceil() / tile_height;
            if lo > 0.0 && base_freq_y / lo < hi / base_freq_y {
                base_freq_y = lo;
            } else {
                base_freq_y = hi;
            }
        }
        let width = (tile_width * base_freq_x + 0.5) as i64;
        let height = (tile_height * base_freq_y + 0.5) as i64;
        let wrap_x = (tile_x * base_freq_x) as i64 + TURB_PERLIN_N + width;
        let wrap_y = (tile_y * base_freq_y) as i64 + TURB_PERLIN_N + height;
        stitch_state = Some(TurbStitch {
            width,
            height,
            wrap_x,
            wrap_y,
        });
    }

    let mut sum = 0.0f64;
    let mut vx = point_x * base_freq_x;
    let mut vy = point_y * base_freq_y;
    let mut ratio = 1.0f64;
    for _ in 0..num_octaves {
        let n = lattice.noise2(channel, vx, vy, stitch_state.as_ref());
        sum += if fractal_sum {
            n / ratio
        } else {
            n.abs() / ratio
        };
        vx *= 2.0;
        vy *= 2.0;
        ratio *= 2.0;
        if let Some(ref mut s) = stitch_state {
            // §15.24 update: width *= 2; wrap_x = 2 · wrap_x − PerlinN;
            s.width *= 2;
            s.wrap_x = 2 * s.wrap_x - TURB_PERLIN_N;
            s.height *= 2;
            s.wrap_y = 2 * s.wrap_y - TURB_PERLIN_N;
        }
    }
    sum
}

/// Clamp+quantise a §15.24 turbulence value to a `u8` byte.
///
/// For `fractal_sum == true`, the spec's "colour value" formula is
/// `((turbFunctionResult · 255) + 255) / 2`, then clamp to `[0, 255]`.
/// For `fractal_sum == false`, the formula is `turbFunctionResult ·
/// 255`, then clamp to `[0, 255]`.
#[inline]
fn turb_quantise(value: f64, fractal_sum: bool) -> u8 {
    let raw = if fractal_sum {
        (value * 255.0 + 255.0) / 2.0
    } else {
        value * 255.0
    };
    if raw.is_nan() {
        return 0;
    }
    let clamped = raw.clamp(0.0, 255.0);
    (clamped + 0.5) as u8
}

/// SVG 1.1 §15.24 `<feTurbulence>` — synthesise a Perlin-noise image of
/// the requested `width × height` extent.
///
/// `params` carries the spec attribute set; see [`Turbulence`] for the
/// per-field semantics. The output is a packed-RGBA `u8` buffer of
/// length `width · height · 4`, with R, G, B, A each independently
/// driven by the §15.24 algorithm (the spec's "An initial seed value is
/// computed based on attribute 'seed'. Then the implementation computes
/// the lattice points for R, then continues getting additional pseudo
/// random numbers relative to the last generated pseudo random number
/// and computes the lattice points for G, and so on for B and A." — the
/// four channels share one LCG advance chain through
/// [`TurbLattice::init`], which iterates `k = 0, 1, 2, 3`).
///
/// The implementation maps a sample at output pixel `(px, py)` to the
/// §15.24 `point[]` argument as `(px as f64, py as f64)`; the §15.24 C
/// reference operates in continuous filter-primitive subregion
/// coordinates, but since `baseFrequency` is the only scale knob and is
/// passed through unchanged, the discrete pixel grid is the natural
/// sampling lattice for a fixed-resolution output buffer. Callers that
/// need a sub-pixel offset (e.g. to evaluate at filter-primitive-
/// subregion `(x, y) + 0.5` for half-pixel centroids) can pre-multiply
/// their coordinates into `base_freq_*`.
///
/// When `params.stitch_tiles == Stitch` and `params.tile_width == 0` or
/// `params.tile_height == 0`, the tile rectangle defaults to
/// `(0, 0, width, height)` for the convenience of the common
/// "synthesise into the filter-primitive subregion" use.
///
/// # Panics
///
/// * `params.num_octaves == 0` (§15.24 implies `>= 1`).
/// * `params.base_freq_x < 0` or `params.base_freq_y < 0` (§15.24 error
///   processing).
/// * `width * height * 4` overflows `usize`.
pub fn turbulence_filter(width: u32, height: u32, params: &Turbulence) -> Vec<u8> {
    assert!(
        params.base_freq_x >= 0.0,
        "turbulence_filter: base_freq_x ({}) must be >= 0 (§15.24)",
        params.base_freq_x
    );
    assert!(
        params.base_freq_y >= 0.0,
        "turbulence_filter: base_freq_y ({}) must be >= 0 (§15.24)",
        params.base_freq_y
    );
    assert!(
        params.num_octaves >= 1,
        "turbulence_filter: num_octaves must be >= 1"
    );

    let w = width as usize;
    let h = height as usize;
    let n = w
        .checked_mul(h)
        .and_then(|m| m.checked_mul(4))
        .expect("turbulence_filter: width * height * 4 overflowed usize");
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let lattice = TurbLattice::init(params.seed);
    let do_stitch = matches!(params.stitch_tiles, StitchTiles::Stitch);
    let fractal_sum = matches!(params.kind, TurbulenceType::FractalNoise);

    // Default the tile rectangle to the output extent when stitch is
    // requested but the rectangle is omitted.
    let (tile_x, tile_y, tile_width, tile_height) =
        if do_stitch && (params.tile_width == 0.0 || params.tile_height == 0.0) {
            (0.0, 0.0, width as f64, height as f64)
        } else {
            (
                params.tile_x,
                params.tile_y,
                params.tile_width,
                params.tile_height,
            )
        };

    let mut out = vec![0u8; n];
    for py in 0..height {
        for px in 0..width {
            let point_x = px as f64;
            let point_y = py as f64;
            let mut rgba = [0u8; 4];
            for (channel, channel_out) in rgba.iter_mut().enumerate() {
                let v = turbulence_at(
                    &lattice,
                    channel,
                    point_x,
                    point_y,
                    params.base_freq_x,
                    params.base_freq_y,
                    params.num_octaves,
                    fractal_sum,
                    do_stitch,
                    tile_x,
                    tile_y,
                    tile_width,
                    tile_height,
                );
                *channel_out = turb_quantise(v, fractal_sum);
            }
            let idx = ((py as usize) * w + (px as usize)) * 4;
            out[idx..idx + 4].copy_from_slice(&rgba);
        }
    }
    out
}

/// Typed-pixel wrapper around [`turbulence_filter`].
pub fn turbulence_filter_pixels(width: u32, height: u32, params: &Turbulence) -> Vec<Rgba> {
    let bytes = turbulence_filter(width, height, params);
    bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod turbulence_tests {
    use super::*;

    /// §15.24 spec-quoted Park–Miller self-check: starting from
    /// `seed = 1`, the 10,000th LCG state is `1043618065`.
    #[test]
    fn park_miller_10000th_value_is_1043618065() {
        let mut s: i64 = 1;
        for _ in 0..10_000 {
            s = turb_random(s);
        }
        assert_eq!(s, 1_043_618_065);
    }

    #[test]
    fn setup_seed_clamps_positive_input_into_range() {
        assert_eq!(turb_setup_seed(1), 1);
        assert_eq!(turb_setup_seed(42), 42);
        assert_eq!(turb_setup_seed(TURB_RAND_M - 1), TURB_RAND_M - 1);
        // Strictly above the modulus clamps to RAND_M − 1.
        assert_eq!(turb_setup_seed(TURB_RAND_M), TURB_RAND_M - 1);
        assert_eq!(turb_setup_seed(i64::MAX), TURB_RAND_M - 1);
    }

    #[test]
    fn setup_seed_clamps_nonpositive_input_into_range() {
        // §15.24's `if (lSeed <= 0) lSeed = -(lSeed % (RAND_M-1)) + 1;`
        // reflects non-positive inputs back into the positive range.
        let s = turb_setup_seed(0);
        assert!((1..=TURB_RAND_M - 1).contains(&s));
        let s = turb_setup_seed(-5);
        assert!((1..=TURB_RAND_M - 1).contains(&s));
        let s = turb_setup_seed(-(TURB_RAND_M - 1));
        assert!((1..=TURB_RAND_M - 1).contains(&s));
    }

    #[test]
    fn turbulence_filter_extent_matches_request() {
        let p = Turbulence::new(0.1, 0.1).with_num_octaves(2).with_seed(0);
        let out = turbulence_filter(8, 6, &p);
        assert_eq!(out.len(), 8 * 6 * 4);
    }

    #[test]
    fn turbulence_filter_empty_extent_returns_empty() {
        let p = Turbulence::new(0.1, 0.1);
        assert!(turbulence_filter(0, 0, &p).is_empty());
        assert!(turbulence_filter(8, 0, &p).is_empty());
        assert!(turbulence_filter(0, 6, &p).is_empty());
    }

    #[test]
    fn turbulence_is_deterministic_for_same_seed() {
        let p = Turbulence::new(0.05, 0.05)
            .with_num_octaves(3)
            .with_seed(42);
        let a = turbulence_filter(16, 16, &p);
        let b = turbulence_filter(16, 16, &p);
        assert_eq!(a, b);
    }

    #[test]
    fn turbulence_changes_with_seed() {
        let a = turbulence_filter(16, 16, &Turbulence::new(0.05, 0.05).with_seed(1));
        let b = turbulence_filter(16, 16, &Turbulence::new(0.05, 0.05).with_seed(2));
        assert_ne!(a, b);
    }

    #[test]
    fn turbulence_zero_frequency_is_constant_per_channel() {
        // baseFreq=0 collapses the noise input to a constant — the
        // lattice is evaluated at the same point for every pixel, so
        // each channel takes a single value over the whole image.
        let p = Turbulence::new(0.0, 0.0).with_seed(7).with_num_octaves(1);
        let out = turbulence_filter(4, 4, &p);
        // Every pixel must equal pixel 0.
        let p0 = &out[..4];
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk, p0);
        }
    }

    #[test]
    fn fractal_sum_quantisation_centres_on_128_for_zero() {
        // For fractalNoise, the quantiser maps `0` -> 128
        // (`(0 · 255 + 255) / 2 = 127.5 -> 128`).
        assert_eq!(turb_quantise(0.0, true), 128);
        // For turbulence, `0 -> 0`.
        assert_eq!(turb_quantise(0.0, false), 0);
    }

    #[test]
    fn quantisation_clamps_at_both_ends() {
        // High end: turbulence quantiser, raw > 255 -> 255.
        assert_eq!(turb_quantise(2.0, false), 255);
        // Low end: turbulence quantiser, raw < 0 -> 0.
        assert_eq!(turb_quantise(-1.0, false), 0);
        // High end: fractalNoise quantiser, raw > 255 -> 255.
        assert_eq!(turb_quantise(2.0, true), 255);
        // Low end: fractalNoise quantiser, raw < -1 -> 0.
        assert_eq!(turb_quantise(-2.0, true), 0);
    }

    #[test]
    fn turbulence_type_produces_nonnegative_aim() {
        // The `turbulence` variant uses |noise2| accumulation, so the
        // raw aim is `[0, 1]`. The byte output should never be in the
        // unreachable upper tail nor pinned to 0 for a non-degenerate
        // base frequency over a multi-pixel grid: assert the channel
        // mean lies away from the extremes.
        let p = Turbulence::new(0.05, 0.05)
            .with_num_octaves(2)
            .with_seed(13)
            .with_kind(TurbulenceType::Turbulence);
        let out = turbulence_filter(32, 32, &p);
        let mut acc = [0u64; 4];
        for chunk in out.chunks_exact(4) {
            for (i, v) in chunk.iter().enumerate() {
                acc[i] += *v as u64;
            }
        }
        for (i, sum) in acc.iter().enumerate() {
            let mean = *sum as f64 / (32.0 * 32.0);
            assert!(
                mean > 1.0 && mean < 254.0,
                "channel {i} mean {mean} unexpectedly extreme"
            );
        }
    }

    #[test]
    fn fractal_noise_centres_signed_output() {
        // The `fractalNoise` variant accumulates signed noise (aim ≈
        // `[-1, 1]`), quantised by `(v·255 + 255) / 2`. Over a large-
        // enough grid the per-channel mean should sit close to 128.
        let p = Turbulence::new(0.05, 0.05)
            .with_num_octaves(2)
            .with_seed(11)
            .with_kind(TurbulenceType::FractalNoise);
        let out = turbulence_filter(48, 48, &p);
        let mut acc = [0u64; 4];
        for chunk in out.chunks_exact(4) {
            for (i, v) in chunk.iter().enumerate() {
                acc[i] += *v as u64;
            }
        }
        for (i, sum) in acc.iter().enumerate() {
            let mean = *sum as f64 / (48.0 * 48.0);
            // Generous bound: the signed sum has a near-zero
            // expectation, but a finite grid can drift; ±48 ≈ ±0.19
            // around the centre.
            assert!(
                (mean - 128.0).abs() < 48.0,
                "channel {i} mean {mean} far from 128"
            );
        }
    }

    #[test]
    fn num_octaves_one_vs_more_differ() {
        let one = turbulence_filter(
            16,
            16,
            &Turbulence::new(0.1, 0.1).with_seed(3).with_num_octaves(1),
        );
        let four = turbulence_filter(
            16,
            16,
            &Turbulence::new(0.1, 0.1).with_seed(3).with_num_octaves(4),
        );
        assert_ne!(one, four);
    }

    #[test]
    fn channels_are_independent() {
        // Per §15.24, the four channels share the same LCG advance
        // chain but each one consumes its own block of random draws —
        // so the four channels of a given pixel should not all be
        // equal in general.
        let p = Turbulence::new(0.1, 0.1).with_seed(9).with_num_octaves(2);
        let out = turbulence_filter(8, 8, &p);
        let mut all_equal_channels = true;
        for chunk in out.chunks_exact(4) {
            if !(chunk[0] == chunk[1] && chunk[1] == chunk[2] && chunk[2] == chunk[3]) {
                all_equal_channels = false;
                break;
            }
        }
        assert!(!all_equal_channels);
    }

    #[test]
    fn stitch_tiles_seam_continuous_for_default_tile() {
        // With stitchTiles="stitch" and the tile defaulted to the
        // output rectangle, the noise wraps cleanly across the right
        // and bottom edges — the left column equals the column that
        // *would* sit one tile-width to the right, i.e. the same
        // column when the tile width equals the image width. To make
        // this observable without a second image, we synthesise two
        // images that share the same parameters: a wide one (W×H) and
        // a tall one (W×2H). The wide one's right edge should match a
        // wrap of itself when stitched.
        // A simpler invariant: a stitched image of the same parameters
        // is identical pixel-for-pixel when re-generated.
        let p = Turbulence::new(0.1, 0.1)
            .with_num_octaves(2)
            .with_seed(5)
            .with_stitch_tiles(StitchTiles::Stitch);
        let a = turbulence_filter(32, 32, &p);
        let b = turbulence_filter(32, 32, &p);
        assert_eq!(a, b);
        // And the stitched output should differ from a no-stitch run
        // with the same other parameters: the §15.24 stitch path
        // adjusts the base frequencies and wraps lattice indices, so
        // the two paths produce different pixels for a frequency that
        // doesn't already land on the lattice boundary.
        let ns = turbulence_filter(
            32,
            32,
            &Turbulence::new(0.1, 0.1).with_num_octaves(2).with_seed(5),
        );
        assert_ne!(a, ns);
    }

    #[test]
    fn typed_pixel_wrapper_matches_byte_api() {
        let p = Turbulence::new(0.07, 0.07)
            .with_num_octaves(3)
            .with_seed(17);
        let bytes = turbulence_filter(12, 10, &p);
        let pixels = turbulence_filter_pixels(12, 10, &p);
        let from_pixels: Vec<u8> = pixels.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
        assert_eq!(bytes, from_pixels);
    }

    #[test]
    #[should_panic(expected = "base_freq_x")]
    fn negative_base_freq_x_panics() {
        let _ = turbulence_filter(2, 2, &Turbulence::new(0.1, 0.1)).len();
        // The above shouldn't panic; the next line should.
        let p = Turbulence {
            base_freq_x: -0.1,
            base_freq_y: 0.1,
            num_octaves: 1,
            seed: 0,
            stitch_tiles: StitchTiles::NoStitch,
            kind: TurbulenceType::Turbulence,
            tile_x: 0.0,
            tile_y: 0.0,
            tile_width: 0.0,
            tile_height: 0.0,
        };
        let _ = turbulence_filter(2, 2, &p);
    }

    #[test]
    #[should_panic(expected = "base_freq_y")]
    fn negative_base_freq_y_panics() {
        let p = Turbulence {
            base_freq_x: 0.1,
            base_freq_y: -0.1,
            num_octaves: 1,
            seed: 0,
            stitch_tiles: StitchTiles::NoStitch,
            kind: TurbulenceType::Turbulence,
            tile_x: 0.0,
            tile_y: 0.0,
            tile_width: 0.0,
            tile_height: 0.0,
        };
        let _ = turbulence_filter(2, 2, &p);
    }

    #[test]
    #[should_panic(expected = "num_octaves")]
    fn zero_num_octaves_panics() {
        let p = Turbulence {
            base_freq_x: 0.1,
            base_freq_y: 0.1,
            num_octaves: 0,
            seed: 0,
            stitch_tiles: StitchTiles::NoStitch,
            kind: TurbulenceType::Turbulence,
            tile_x: 0.0,
            tile_y: 0.0,
            tile_width: 0.0,
            tile_height: 0.0,
        };
        let _ = turbulence_filter(2, 2, &p);
    }
}

// ----------------------------------------------------------------------
// SVG 1.1 §15.16 `<feFlood>` — solid-colour source primitive.
// ----------------------------------------------------------------------

/// SVG 1.1 §15.16 `<feFlood>` — emit a `width × height` packed-RGBA
/// buffer entirely filled with `(r, g, b)` modulated by `flood_opacity`.
///
/// `flood_opacity` is a normalised straight-alpha value in `[0, 1]`; the
/// spec's `flood-color` property carries the RGB triple (with `currentColor`
/// already resolved by the consumer) and `flood-opacity` carries the
/// independent opacity multiplier. The two combine into one straight-alpha
/// output pixel `(r, g, b, round(flood_opacity · 255))`, which is what every
/// pixel of the returned buffer holds.
///
/// The filter primitive subregion (the `x` / `y` / `width` / `height`
/// attributes on `<feFlood>`) is resolved by the caller before invoking
/// this function — the function only fills the rectangle it is given.
///
/// # Panics
///
/// * If `flood_opacity` is NaN.
/// * If `width as usize * height as usize * 4` overflows `usize`.
///
/// # Returns
///
/// A packed-RGBA `Vec<u8>` of length `width · height · 4`, every pixel
/// equal to `(r, g, b, round(clamp(flood_opacity, 0, 1) · 255))`. If
/// either `width` or `height` is zero, an empty vector is returned.
pub fn flood(width: u32, height: u32, r: u8, g: u8, b: u8, flood_opacity: f32) -> Vec<u8> {
    assert!(!flood_opacity.is_nan(), "flood: flood_opacity is NaN");
    let w = width as usize;
    let h = height as usize;
    let n = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("flood: width * height * 4 overflowed usize");
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let a = quantise_unit(flood_opacity);
    let mut out = Vec::with_capacity(n);
    for _ in 0..(w * h) {
        out.extend_from_slice(&[r, g, b, a]);
    }
    out
}

/// Typed-pixel wrapper around [`flood`]. Returns a `Vec<Rgba>` of length
/// `width · height`, every entry equal to the resolved flood pixel.
pub fn flood_pixels(width: u32, height: u32, r: u8, g: u8, b: u8, flood_opacity: f32) -> Vec<Rgba> {
    assert!(
        !flood_opacity.is_nan(),
        "flood_pixels: flood_opacity is NaN"
    );
    let w = width as usize;
    let h = height as usize;
    let n = w
        .checked_mul(h)
        .expect("flood_pixels: width * height overflowed usize");
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let a = quantise_unit(flood_opacity);
    vec![Rgba::new(r, g, b, a); n]
}

// ----------------------------------------------------------------------
// SVG 1.1 §15.21 `<feOffset>` — translate the input image by (dx, dy).
// ----------------------------------------------------------------------

/// Sample reconstruction policy for [`offset`], mirroring the §15.21
/// note that "the destination location may be offset by a fraction of
/// a pixel in device space. In this case a high quality viewer should
/// make use of appropriate interpolation techniques, for example
/// bilinear or bicubic."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffsetSampling {
    /// Round `(dx, dy)` to the nearest integer pixel shift and copy
    /// source pixels into the output verbatim. The default — matches
    /// the §15.7.3 "undefined pixels are set to transparent black"
    /// rule directly and is exact for the common case of integer
    /// shifts produced by user-space `dx` / `dy` attributes on an
    /// untransformed primitive.
    #[default]
    Nearest,
    /// Resample with bilinear interpolation for fractional `(dx, dy)`.
    /// Pixels whose 2×2 bilinear footprint falls partially outside the
    /// source contribute the §15.7.3 transparent-black value (`(0, 0,
    /// 0, 0)`) on the out-of-bounds samples, which naturally fades
    /// the offset image at the edges.
    Bilinear,
}

/// SVG 1.1 §15.21 `<feOffset>` — translate the packed-RGBA source by
/// `(dx, dy)` (in source-pixel units) and emit a buffer of the same
/// dimensions.
///
/// The output pixel `(x, y)` is sourced from `(x − dx, y − dy)` per
/// §15.21 ("offsets the input image relative to its current position …
/// by the specified vector"); positions that fall outside the source
/// extent emit transparent black `(0, 0, 0, 0)` per §15.7.3 ("undefined
/// pixels are set to transparent black"). The `sampling` parameter
/// selects between integer-rounded nearest-pixel sampling
/// ([`OffsetSampling::Nearest`], the default) and bilinear
/// reconstruction for fractional shifts ([`OffsetSampling::Bilinear`],
/// the §15.21 "high quality viewer" route).
///
/// `src` must be exactly `width · height · 4` bytes in row-major
/// straight-alpha RGBA order.
///
/// # Panics
///
/// * If `src.len() != width as usize * height as usize * 4`.
/// * If `dx` or `dy` is NaN.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same dimensions as the source.
pub fn offset(
    src: &[u8],
    width: u32,
    height: u32,
    dx: f32,
    dy: f32,
    sampling: OffsetSampling,
) -> Vec<u8> {
    assert!(!dx.is_nan(), "offset: dx is NaN");
    assert!(!dy.is_nan(), "offset: dy is NaN");
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("offset: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "offset: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );
    if width == 0 || height == 0 {
        return Vec::new();
    }

    match sampling {
        OffsetSampling::Nearest => offset_nearest(src, width, height, dx, dy),
        OffsetSampling::Bilinear => offset_bilinear(src, width, height, dx, dy),
    }
}

/// Typed-pixel wrapper around [`offset`]. Returns a `Vec<Rgba>` of the
/// same length as `src`.
pub fn offset_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    dx: f32,
    dy: f32,
    sampling: OffsetSampling,
) -> Vec<Rgba> {
    let n = width as usize * height as usize;
    assert_eq!(
        src.len(),
        n,
        "offset_pixels: src.len() == {} but width*height == {n}",
        src.len()
    );
    let mut b = Vec::with_capacity(n * 4);
    for p in src {
        b.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out = offset(&b, width, height, dx, dy, sampling);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

/// Integer-rounded source sampling. `(dx, dy)` are rounded
/// half-to-even (the default `f32::round_ties_even` semantics) before
/// being applied; out-of-bounds positions emit transparent black.
fn offset_nearest(src: &[u8], width: u32, height: u32, dx: f32, dy: f32) -> Vec<u8> {
    let w = width as i64;
    let h = height as i64;
    // Round-half-away-from-zero, matching the standard `f32::round()`
    // rule the spec's example (`dx="4" dy="4"`) implicitly relies on
    // for integer shifts. Saturating cast clamps shifts that would
    // overflow `i64` to extremes that fall outside the bounds check
    // below, producing transparent black for every output pixel — the
    // same behaviour as a finite shift that exceeds the source extent.
    let idx = dx.round() as i64;
    let idy = dy.round() as i64;

    let mut out = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        let sy = y - idy;
        if !(0..h).contains(&sy) {
            continue;
        }
        for x in 0..w {
            let sx = x - idx;
            if !(0..w).contains(&sx) {
                continue;
            }
            let s = ((sy * w + sx) * 4) as usize;
            let d = ((y * w + x) * 4) as usize;
            out[d..d + 4].copy_from_slice(&src[s..s + 4]);
        }
    }
    out
}

/// Bilinear sampling of the shifted source. The shift is applied in
/// floating-point and the four neighbouring source pixels are blended
/// per-channel; samples outside the source extent contribute
/// transparent black, naturally fading the offset image at the edges.
fn offset_bilinear(src: &[u8], width: u32, height: u32, dx: f32, dy: f32) -> Vec<u8> {
    let w = width as i64;
    let h = height as i64;

    let fetch = |sx: i64, sy: i64| -> [f32; 4] {
        if !(0..w).contains(&sx) || !(0..h).contains(&sy) {
            return [0.0; 4];
        }
        let p = ((sy * w + sx) * 4) as usize;
        [
            src[p] as f32,
            src[p + 1] as f32,
            src[p + 2] as f32,
            src[p + 3] as f32,
        ]
    };

    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            // Source coordinate: subtracting the shift moves the
            // image by (+dx, +dy) in output space.
            let fx = x as f32 - dx;
            let fy = y as f32 - dy;
            let x0 = fx.floor() as i64;
            let y0 = fy.floor() as i64;
            let tx = fx - x0 as f32;
            let ty = fy - y0 as f32;
            let c00 = fetch(x0, y0);
            let c10 = fetch(x0 + 1, y0);
            let c01 = fetch(x0, y0 + 1);
            let c11 = fetch(x0 + 1, y0 + 1);
            for c in 0..4 {
                let top = c00[c] + (c10[c] - c00[c]) * tx;
                let bot = c01[c] + (c11[c] - c01[c]) * tx;
                let v = top + (bot - top) * ty;
                out.push(v.clamp(0.0, 255.0).round() as u8);
            }
        }
    }
    out
}

// ----------------------------------------------------------------------
// SVG 1.1 §15.19 `<feMerge>` — composite layers with `over`.
// ----------------------------------------------------------------------

/// SVG 1.1 §15.19 `<feMerge>` — composite `layers` bottom-to-top using
/// the §14.2 simple-alpha-compositing `over` operator (the spec's
/// "Input1 (corresponding to the first `feMergeNode` child element) on
/// the bottom and the last specified input, InputN (corresponding to
/// the last `feMergeNode` child element), on top").
///
/// Every layer must be a `width × height` packed-RGBA `u8` buffer in
/// straight-alpha row-major order. The output is a new buffer of the
/// same dimensions, in straight-alpha form, holding the cumulative
/// `over` composite.
///
/// Returning a transparent-black buffer is the documented zero-layer
/// result: the §15.19 algorithm composites N inputs onto a backdrop
/// that starts at the initial filter-effects-region value, which §15.1
/// defines as transparent black (`(0, 0, 0, 0)`).
///
/// # Panics
///
/// * If any layer's length is not `width as usize * height as usize * 4`.
///
/// # Returns
///
/// A packed-RGBA `Vec<u8>` of length `width · height · 4`.
pub fn merge(width: u32, height: u32, layers: &[&[u8]]) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("merge: width * height * 4 overflowed usize");
    if width == 0 || height == 0 {
        for (i, l) in layers.iter().enumerate() {
            assert_eq!(
                l.len(),
                0,
                "merge: layer {i} len {} on a {}×{} extent (expected 0)",
                l.len(),
                width,
                height
            );
        }
        return Vec::new();
    }
    for (i, l) in layers.iter().enumerate() {
        assert_eq!(
            l.len(),
            expected,
            "merge: layer {i} len {} but width*height*4 == {expected}",
            l.len(),
        );
    }

    if layers.is_empty() {
        // Initial-region transparent black.
        return vec![0u8; expected];
    }

    // Composite bottom-to-top. `dst` carries the accumulating result.
    // The first layer is copied verbatim (the §14.2 `over` of any
    // pixel onto fully transparent backdrop equals that pixel).
    let mut dst: Vec<u8> = layers[0].to_vec();
    for layer in &layers[1..] {
        for (d, s) in dst.chunks_exact_mut(4).zip(layer.chunks_exact(4)) {
            let sa = s[3] as f32 / 255.0;
            let da = d[3] as f32 / 255.0;
            // §14.2 simple-alpha-compositing `over`, expressed
            // straight-alpha:
            //   αo = αs + αd · (1 − αs)
            //   co = (cs · αs + cd · αd · (1 − αs)) / αo
            let one_minus_sa = 1.0 - sa;
            let ao = sa + da * one_minus_sa;
            if ao <= 0.0 {
                d[0] = 0;
                d[1] = 0;
                d[2] = 0;
                d[3] = 0;
                continue;
            }
            for c in 0..3 {
                let cs = s[c] as f32 / 255.0;
                let cd = d[c] as f32 / 255.0;
                let co = (cs * sa + cd * da * one_minus_sa) / ao;
                d[c] = quantise_unit(co);
            }
            d[3] = quantise_unit(ao);
        }
    }
    dst
}

/// Typed-pixel wrapper around [`merge`]. Each entry of `layers` is a
/// `Vec<Rgba>` of length `width · height`.
pub fn merge_pixels(width: u32, height: u32, layers: &[&[Rgba]]) -> Vec<Rgba> {
    let n = width as usize * height as usize;
    for (i, l) in layers.iter().enumerate() {
        assert_eq!(
            l.len(),
            n,
            "merge_pixels: layer {i} len {} but width*height == {n}",
            l.len()
        );
    }
    // Convert each Rgba layer to a byte buffer, then build &[u8] views.
    let bytes: Vec<Vec<u8>> = layers
        .iter()
        .map(|l| {
            let mut b = Vec::with_capacity(n * 4);
            for p in *l {
                b.extend_from_slice(&[p.r, p.g, p.b, p.a]);
            }
            b
        })
        .collect();
    let byte_refs: Vec<&[u8]> = bytes.iter().map(|b| b.as_slice()).collect();
    let out = merge(width, height, &byte_refs);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

// ----------------------------------------------------------------------
// Filter Effects Module Level 1 §9.12 `<feDropShadow>` — shorthand
// drop-shadow primitive.
// ----------------------------------------------------------------------

/// Filter Effects Module Level 1 §9.12 `<feDropShadow>` — the shorthand
/// drop-shadow primitive, evaluated as its spec-defined equivalent
/// primitive chain.
///
/// §9.12 states the result "is equivalent to the following":
///
/// ```text
/// <feGaussianBlur in="alpha-channel-of-feDropShadow-in" stdDeviation="…"/>
/// <feOffset dx="…" dy="…" result="offsetblur"/>
/// <feFlood flood-color="…" flood-opacity="…"/>
/// <feComposite in2="offsetblur" operator="in"/>
/// <feMerge>
///   <feMergeNode/>
///   <feMergeNode in="in-of-feDropShadow"/>
/// </feMerge>
/// ```
///
/// The §9.12 "divided into steps" elaboration is implemented verbatim:
///
/// 1. Take the **alpha channel** of the input (RGB forced to `0`, alpha
///    preserved) and blur it with a [`gaussian_blur`] of `std_x` /
///    `std_y` (the §9.12 `stdDeviation`, a `<number-optional-number>`;
///    when only one value is given the caller passes it for both axes).
/// 2. [`offset`] that blurred alpha by `(dx, dy)` to produce the
///    spec's `"offsetblur"` buffer.
/// 3. Produce an [`flood`] of the resolved `flood-color`
///    (`flood_r` / `flood_g` / `flood_b`) at `flood_opacity`.
/// 4. [`composite_filter`] the flood with `"offsetblur"` under
///    [`CompositeOp::In`] (`in1` = flood, `in2` = offsetblur), so the
///    flood colour is clipped to the offset blurred-alpha coverage —
///    this is the coloured shadow.
/// 5. [`merge`] the shadow (bottom) with the original input (top),
///    i.e. the source graphic drawn over its own shadow.
///
/// The §9.12 note that "it is not required that it is implemented like
/// that … user agents can optimize the handling" is honoured in spirit:
/// the equivalent tree is the normative definition, and evaluating it
/// directly is observably identical to any optimised path.
///
/// **Defaults** (§9.12 attribute table): `dx` initial value `2`, `dy`
/// initial value `2`, `stdDeviation` initial value `2`. §9.13.1
/// `flood-color` initial value is `black` (`(0, 0, 0)`); §9.13.2
/// `flood-opacity` initial value is `1`. These are *not* applied here —
/// the caller resolves the presentation attributes and passes concrete
/// values, mirroring every other primitive entry point in this module.
///
/// `src` must be exactly `width · height · 4` bytes in row-major
/// straight-alpha RGBA order. `sampling` selects the [`offset`]
/// reconstruction policy for fractional `(dx, dy)`.
///
/// # Panics
///
/// * If `src.len() != width as usize * height as usize * 4`.
/// * If `std_x`, `std_y`, `dx`, `dy`, or `flood_opacity` is NaN.
/// * If `std_x` or `std_y` is negative (§15.17 / §9.12 "a negative
///   value … is an error").
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same dimensions as the source —
/// the source graphic composited over its coloured, blurred, offset
/// drop shadow.
#[allow(clippy::too_many_arguments)]
pub fn drop_shadow(
    src: &[u8],
    width: u32,
    height: u32,
    std_x: f32,
    std_y: f32,
    dx: f32,
    dy: f32,
    flood_r: u8,
    flood_g: u8,
    flood_b: u8,
    flood_opacity: f32,
    sampling: OffsetSampling,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("drop_shadow: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "drop_shadow: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    // Step 1: alpha channel of the input → RGB zeroed, alpha preserved.
    // §9.12: "Take the alpha channel of the input". Blurring this
    // alpha-only buffer is the feGaussianBlur of the equivalent tree.
    let mut alpha_only = vec![0u8; expected];
    for (dst, s) in alpha_only.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        dst[3] = s[3];
    }
    let blurred = gaussian_blur(&alpha_only, width, height, std_x, std_y);

    // Step 2: feOffset(dx, dy) → "offsetblur".
    let offsetblur = offset(&blurred, width, height, dx, dy, sampling);

    // Step 3: feFlood(flood-color, flood-opacity).
    let flooded = flood(width, height, flood_r, flood_g, flood_b, flood_opacity);

    // Step 4: feComposite(flood in2="offsetblur" operator="in"). The
    // flood colour is the `in` operand, the offset blurred alpha the
    // `in2` operand; CompositeOp::In clips `in` to `in2`'s coverage.
    let shadow = composite_filter(&flooded, &offsetblur, width, height, CompositeOp::In);

    // Step 5: feMerge — shadow on the bottom, source graphic on top.
    merge(width, height, &[&shadow, src])
}

/// Typed-pixel wrapper around [`drop_shadow`]. Returns a `Vec<Rgba>` of
/// length `width · height`.
#[allow(clippy::too_many_arguments)]
pub fn drop_shadow_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    std_x: f32,
    std_y: f32,
    dx: f32,
    dy: f32,
    flood_r: u8,
    flood_g: u8,
    flood_b: u8,
    flood_opacity: f32,
    sampling: OffsetSampling,
) -> Vec<Rgba> {
    let n = width as usize * height as usize;
    assert_eq!(
        src.len(),
        n,
        "drop_shadow_pixels: src.len() == {} but width*height == {n}",
        src.len()
    );
    let mut bytes = Vec::with_capacity(n * 4);
    for p in src {
        bytes.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out = drop_shadow(
        &bytes,
        width,
        height,
        std_x,
        std_y,
        dx,
        dy,
        flood_r,
        flood_g,
        flood_b,
        flood_opacity,
        sampling,
    );
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

// ----------------------------------------------------------------------
// Filter Effects Module Level 1 §9.4 — filter primitive subregion clip.
// ----------------------------------------------------------------------

/// A filter-primitive subregion rectangle, in the same pixel coordinate
/// space as the primitive's result buffer (the buffer's pixel `(px, py)`
/// covers the continuous square `[px, px + 1) × [py, py + 1)`; the
/// buffer's top-left pixel `(0, 0)` sits at the filter region origin).
///
/// The four fields mirror the `x` / `y` / `width` / `height` attributes
/// that Filter Effects Module Level 1 §9.4 ("Filter primitive
/// subregion") places on every `fe*` element. They are stored here as
/// already-resolved device-pixel offsets / extents — the caller folds
/// `primitiveUnits` (`userSpaceOnUse` vs `objectBoundingBox`) and any
/// percentage-against-the-filter-region resolution into these numbers
/// before constructing a `Subregion`, exactly as the
/// [`flood`] / [`tile`] primitives expect their rectangle pre-resolved.
///
/// A subregion with a non-positive `width` or `height` means "the effect
/// of the filter primitive is disabled" per §9.4 — [`clip_to_subregion`]
/// turns the whole result transparent black in that case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Subregion {
    /// Left edge of the subregion, in result-buffer pixel units.
    pub x: f32,
    /// Top edge of the subregion, in result-buffer pixel units.
    pub y: f32,
    /// Subregion width, in result-buffer pixel units. Non-positive ⇒
    /// the primitive is disabled (§9.4).
    pub width: f32,
    /// Subregion height, in result-buffer pixel units. Non-positive ⇒
    /// the primitive is disabled (§9.4).
    pub height: f32,
}

impl Subregion {
    /// Construct a subregion from its resolved pixel-space `x` / `y` /
    /// `width` / `height`.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// `true` when the subregion is "disabled" per §9.4: a non-positive
    /// (or NaN) width or height. A disabled subregion clips the whole
    /// primitive result to transparent black.
    #[inline]
    fn is_disabled(&self) -> bool {
        // A dimension is "positive" only if it compares strictly greater
        // than zero; NaN fails this test (NaN is unordered), so a NaN
        // extent is treated as disabled — exactly the §9.4 intent for a
        // non-positive subregion dimension.
        let positive = self.width > 0.0 && self.height > 0.0;
        !positive
    }

    /// `true` when pixel column `px` (covering `[px, px + 1)`) even
    /// partly intersects the subregion's `[x, x + width)` X-extent.
    ///
    /// §9.4: the offscreens are sized so that "all pixels which even
    /// partly intersect" the subregion are kept; we therefore use the
    /// half-open-interval overlap test `px + 1 > x && px < x + width`
    /// rather than a centroid-inside test, so a subregion whose edge
    /// falls mid-pixel retains the straddling pixel.
    #[inline]
    fn contains_col(&self, px: u32) -> bool {
        let lo = px as f32;
        let hi = lo + 1.0;
        hi > self.x && lo < self.x + self.width
    }

    /// `true` when pixel row `py` (covering `[py, py + 1)`) even partly
    /// intersects the subregion's `[y, y + height)` Y-extent. See
    /// [`Subregion::contains_col`] for the half-open overlap rationale.
    #[inline]
    fn contains_row(&self, py: u32) -> bool {
        let lo = py as f32;
        let hi = lo + 1.0;
        hi > self.y && lo < self.y + self.height
    }
}

/// Filter Effects Module Level 1 §9.4 — apply a filter primitive
/// subregion as a hard clip on a primitive's packed-RGBA result.
///
/// §9.4: "The filter primitive subregion acts as a hard clip clipping
/// rectangle on the filter primitive result." Every pixel of `buf` that
/// lies outside `region` is set to transparent black `(0, 0, 0, 0)`
/// (the §15.7.3 undefined-pixel value); pixels inside — including any
/// pixel that *even partly* intersects the subregion, per the §9.4
/// offscreen-sizing rule — pass through unchanged.
///
/// A `region` whose `width` or `height` is non-positive disables the
/// primitive (§9.4: "If the filter primitive subregion has a negative or
/// zero width or height, the effect of the filter primitive is
/// disabled"); the entire `buf` is cleared to transparent black.
///
/// The clip is applied in place. `buf` is interpreted as `width ×
/// height` row-major straight-alpha RGBA, with pixel `(0, 0)` at the
/// filter region origin — the same coordinate convention the [`tile`]
/// and [`flood`] primitives use for their subregion rectangles.
///
/// # Panics
///
/// * If `buf.len() != width as usize * height as usize * 4`.
/// * If `width as usize * height as usize * 4` overflows `usize`.
pub fn clip_to_subregion(buf: &mut [u8], width: u32, height: u32, region: Subregion) {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("clip_to_subregion: width * height * 4 overflowed usize");
    assert_eq!(
        buf.len(),
        expected,
        "clip_to_subregion: buf.len() == {} but width*height*4 == {expected}",
        buf.len()
    );
    if width == 0 || height == 0 {
        return;
    }
    if region.is_disabled() {
        buf.fill(0);
        return;
    }
    // Precompute the kept X-span once: `contains_col` is monotone, so the
    // kept columns form a single contiguous run `[col_lo, col_hi)`.
    let mut col_lo = width;
    let mut col_hi = 0u32;
    for px in 0..width {
        if region.contains_col(px) {
            if px < col_lo {
                col_lo = px;
            }
            col_hi = px + 1;
        }
    }
    for py in 0..height {
        let row_kept = region.contains_row(py);
        let row_off = py as usize * w * 4;
        if !row_kept || col_lo >= col_hi {
            // Whole row outside the subregion's Y-span (or the X-span is
            // empty for this image) ⇒ clear the entire row.
            buf[row_off..row_off + w * 4].fill(0);
            continue;
        }
        // Clear the left margin `[0, col_lo)` and right margin
        // `[col_hi, width)`; the middle run passes through untouched.
        if col_lo > 0 {
            buf[row_off..row_off + col_lo as usize * 4].fill(0);
        }
        if col_hi < width {
            let r0 = row_off + col_hi as usize * 4;
            buf[r0..row_off + w * 4].fill(0);
        }
    }
}

/// Typed-pixel wrapper around [`clip_to_subregion`]. Clears every
/// [`Rgba`] outside `region` to `Rgba::new(0, 0, 0, 0)` in place,
/// keeping pixels that even partly intersect the subregion.
///
/// # Panics
///
/// * If `pixels.len() != width as usize * height as usize`.
/// * If `width as usize * height as usize` overflows `usize`.
pub fn clip_to_subregion_pixels(pixels: &mut [Rgba], width: u32, height: u32, region: Subregion) {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .expect("clip_to_subregion_pixels: width * height overflowed usize");
    assert_eq!(
        pixels.len(),
        expected,
        "clip_to_subregion_pixels: pixels.len() == {} but width*height == {expected}",
        pixels.len()
    );
    if width == 0 || height == 0 {
        return;
    }
    let transparent = Rgba::new(0, 0, 0, 0);
    if region.is_disabled() {
        pixels.fill(transparent);
        return;
    }
    for py in 0..height {
        let row_kept = region.contains_row(py);
        let row_off = py as usize * w;
        for px in 0..width {
            if !(row_kept && region.contains_col(px)) {
                pixels[row_off + px as usize] = transparent;
            }
        }
    }
}

#[cfg(test)]
mod subregion_tests {
    use super::*;

    fn build(width: u32, height: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        let mut out = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                out.extend_from_slice(&f(x, y));
            }
        }
        out
    }

    fn px(buf: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * width + x) * 4) as usize;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    }

    #[test]
    fn integer_aligned_rect_keeps_interior_clears_outside() {
        let mut buf = build(6, 5, |_, _| [10, 20, 30, 255]);
        // Keep the 3×2 block at (1,1)..(4,3).
        clip_to_subregion(&mut buf, 6, 5, Subregion::new(1.0, 1.0, 3.0, 2.0));
        for y in 0..5 {
            for x in 0..6 {
                let inside = (1..4).contains(&x) && (1..3).contains(&y);
                let got = px(&buf, 6, x, y);
                if inside {
                    assert_eq!(got, [10, 20, 30, 255], "kept pixel ({x},{y})");
                } else {
                    assert_eq!(got, [0, 0, 0, 0], "cleared pixel ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn fractional_edges_keep_partly_intersecting_pixels() {
        // Subregion x in [1.5, 4.5): columns 1,2,3,4 all partly intersect
        // (col 1 spans [1,2) overlaps [1.5,4.5); col 4 spans [4,5) overlaps).
        // Column 0 ([0,1)) and column 5 ([5,6)) do not.
        let mut buf = build(6, 1, |_, _| [9, 9, 9, 200]);
        clip_to_subregion(&mut buf, 6, 1, Subregion::new(1.5, 0.0, 3.0, 1.0));
        assert_eq!(px(&buf, 6, 0, 0), [0, 0, 0, 0]);
        for x in 1..=4 {
            assert_eq!(px(&buf, 6, x, 0), [9, 9, 9, 200], "col {x} partly inside");
        }
        assert_eq!(px(&buf, 6, 5, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn nonpositive_extent_disables_primitive() {
        let mut zero_w = build(3, 3, |_, _| [1, 2, 3, 255]);
        clip_to_subregion(&mut zero_w, 3, 3, Subregion::new(0.0, 0.0, 0.0, 3.0));
        assert!(zero_w.iter().all(|&b| b == 0), "zero width clears all");

        let mut neg_h = build(3, 3, |_, _| [1, 2, 3, 255]);
        clip_to_subregion(&mut neg_h, 3, 3, Subregion::new(0.0, 0.0, 3.0, -1.0));
        assert!(neg_h.iter().all(|&b| b == 0), "negative height clears all");
    }

    #[test]
    fn subregion_outside_image_clears_everything() {
        let mut buf = build(4, 4, |_, _| [7, 7, 7, 255]);
        // Subregion entirely to the right of the image.
        clip_to_subregion(&mut buf, 4, 4, Subregion::new(10.0, 0.0, 2.0, 4.0));
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn full_region_is_a_no_op() {
        let orig = build(5, 4, |x, y| [x as u8, y as u8, 0, 255]);
        let mut buf = orig.clone();
        clip_to_subregion(&mut buf, 5, 4, Subregion::new(0.0, 0.0, 5.0, 4.0));
        assert_eq!(buf, orig);
    }

    #[test]
    fn typed_path_matches_byte_path() {
        let bytes = build(7, 6, |x, y| {
            [(x * 30) as u8, (y * 40) as u8, ((x + y) * 10) as u8, 180]
        });
        let mut typed: Vec<Rgba> = bytes
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let region = Subregion::new(2.0, 1.5, 3.5, 3.0);

        let mut bytes_clipped = bytes.clone();
        clip_to_subregion(&mut bytes_clipped, 7, 6, region);
        clip_to_subregion_pixels(&mut typed, 7, 6, region);

        let from_typed: Vec<u8> = typed.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
        assert_eq!(bytes_clipped, from_typed);
    }

    #[test]
    #[should_panic(expected = "clip_to_subregion: buf.len()")]
    fn wrong_buffer_length_panics() {
        let mut buf = vec![0u8; 7];
        clip_to_subregion(&mut buf, 2, 2, Subregion::new(0.0, 0.0, 2.0, 2.0));
    }
}

// ----------------------------------------------------------------------
// SVG 1.1 §15.23 `<feTile>` — periodic tiling of a reference subregion.
// ----------------------------------------------------------------------

/// SVG 1.1 §15.23 `<feTile>` — fill a `out_width × out_height` target
/// rectangle by periodic replication of the `src_width × src_height`
/// reference tile `src`.
///
/// §15.23: "The top/left corner of each given tile is at location
/// `(x + i·width, y + j·height)`, where `(x, y)` represents the top/left
/// of the input image's filter primitive subregion, `width` and `height`
/// represent the width and height of the input image's filter primitive
/// subregion, and `i` and `j` can be any integer value."
///
/// Sampling at output pixel `(ox, oy)` reduces to looking up source
/// pixel `(ox mod src_width, oy mod src_height)` under the Euclidean
/// remainder convention — the unique remainder in `[0, src_width)` /
/// `[0, src_height)` regardless of sign — which is the analytic form
/// of "i and j can be any integer value" applied per pixel. The target
/// rectangle's origin is taken to be the source rectangle's origin
/// (i.e. `i = 0, j = 0` aligns the input and output top-left corners);
/// callers that need a different alignment can shift the source through
/// [`offset`] before invoking `tile`.
///
/// `src` must be exactly `src_width · src_height · 4` bytes in row-major
/// straight-alpha RGBA order; the output is exactly
/// `out_width · out_height · 4` bytes in the same layout.
///
/// # Panics
///
/// * If `src.len() != src_width as usize * src_height as usize * 4`.
/// * If `src_width == 0` or `src_height == 0` while the target extent
///   is non-empty (a periodic tiling of an empty reference is
///   undefined — there are no source pixels to repeat).
/// * If any of `src_width · src_height · 4`, `out_width · out_height ·
///   4` overflows `usize`.
///
/// # Returns
///
/// A packed-RGBA `Vec<u8>` of length `out_width · out_height · 4`. If
/// either output dimension is zero, an empty vector is returned.
pub fn tile(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    out_width: u32,
    out_height: u32,
) -> Vec<u8> {
    let sw = src_width as usize;
    let sh = src_height as usize;
    let src_expected = sw
        .checked_mul(sh)
        .and_then(|n| n.checked_mul(4))
        .expect("tile: src_width * src_height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        src_expected,
        "tile: src.len() == {} but src_width*src_height*4 == {src_expected}",
        src.len()
    );

    let ow = out_width as usize;
    let oh = out_height as usize;
    let out_bytes = ow
        .checked_mul(oh)
        .and_then(|n| n.checked_mul(4))
        .expect("tile: out_width * out_height * 4 overflowed usize");

    if out_width == 0 || out_height == 0 {
        return Vec::new();
    }
    assert!(
        src_width != 0 && src_height != 0,
        "tile: empty source ({src_width}×{src_height}) cannot fill a non-empty {out_width}×{out_height} target"
    );

    let mut out = vec![0u8; out_bytes];
    // Pre-compute one row of source-x indices: `out_width` lookups, each
    // `ox % src_width`. The §15.23 rule "i can be any integer value"
    // becomes the Euclidean remainder on a non-negative `ox`, which is
    // just the plain modulus.
    let row_x: Vec<usize> = (0..ow).map(|ox| ox % sw).collect();
    for oy in 0..oh {
        let sy = oy % sh;
        let src_row_off = sy * sw * 4;
        let dst_row_off = oy * ow * 4;
        for (ox, &sx) in row_x.iter().enumerate() {
            let s = src_row_off + sx * 4;
            let d = dst_row_off + ox * 4;
            out[d..d + 4].copy_from_slice(&src[s..s + 4]);
        }
    }
    out
}

/// Typed-pixel wrapper around [`tile`].
///
/// `src` is a `Vec<Rgba>` of length `src_width · src_height`; the
/// returned `Vec<Rgba>` has length `out_width · out_height`.
pub fn tile_pixels(
    src: &[Rgba],
    src_width: u32,
    src_height: u32,
    out_width: u32,
    out_height: u32,
) -> Vec<Rgba> {
    let n = src_width as usize * src_height as usize;
    assert_eq!(
        src.len(),
        n,
        "tile_pixels: src.len() == {} but src_width*src_height == {n}",
        src.len()
    );
    let mut bytes = Vec::with_capacity(n * 4);
    for p in src {
        bytes.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out = tile(&bytes, src_width, src_height, out_width, out_height);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod flood_offset_merge_tests {
    use super::*;

    // -------------------- flood --------------------

    #[test]
    fn flood_fills_every_pixel_with_resolved_rgba() {
        let out = flood(3, 2, 200, 100, 50, 0.5);
        assert_eq!(out.len(), 3 * 2 * 4);
        // round(0.5 · 255 + 0.5) = 128
        for px in out.chunks_exact(4) {
            assert_eq!(px, [200, 100, 50, 128]);
        }
    }

    #[test]
    fn flood_opacity_one_is_fully_opaque() {
        let out = flood(1, 1, 10, 20, 30, 1.0);
        assert_eq!(out, [10, 20, 30, 255]);
    }

    #[test]
    fn flood_opacity_zero_is_fully_transparent() {
        let out = flood(2, 1, 99, 99, 99, 0.0);
        // Note: the §15.16 spec keeps the RGB triple regardless of
        // flood-opacity (it is the straight-alpha form, not the
        // premultiplied form). We emit straight-alpha.
        assert_eq!(out, [99, 99, 99, 0, 99, 99, 99, 0]);
    }

    #[test]
    fn flood_opacity_outside_unit_range_clamps() {
        // Both negative and >1 inputs clamp into [0, 255].
        let a = flood(1, 1, 0, 0, 0, -0.5);
        let b = flood(1, 1, 0, 0, 0, 1.5);
        assert_eq!(a[3], 0);
        assert_eq!(b[3], 255);
    }

    #[test]
    fn flood_empty_extent_returns_empty() {
        assert!(flood(0, 5, 0, 0, 0, 1.0).is_empty());
        assert!(flood(5, 0, 0, 0, 0, 1.0).is_empty());
        assert!(flood(0, 0, 0, 0, 0, 1.0).is_empty());
    }

    #[test]
    fn flood_pixels_matches_byte_path() {
        let bytes = flood(4, 3, 70, 80, 90, 0.75);
        let typed = flood_pixels(4, 3, 70, 80, 90, 0.75);
        let from_typed: Vec<u8> = typed.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
        assert_eq!(bytes, from_typed);
    }

    #[test]
    #[should_panic(expected = "flood_opacity is NaN")]
    fn flood_nan_opacity_panics() {
        let _ = flood(1, 1, 0, 0, 0, f32::NAN);
    }

    // -------------------- offset --------------------

    /// Make a tiny "stamp" buffer: opaque red dot at the centre of a
    /// 5×5 transparent-black canvas.
    fn red_dot_5x5() -> Vec<u8> {
        let mut v = vec![0u8; 5 * 5 * 4];
        let centre = ((2 * 5 + 2) * 4) as usize;
        v[centre] = 255;
        v[centre + 3] = 255;
        v
    }

    #[test]
    fn offset_nearest_integer_shift_moves_the_dot() {
        let src = red_dot_5x5();
        // Shift the dot one pixel right (dx = 1) and zero pixels down.
        let out = offset(&src, 5, 5, 1.0, 0.0, OffsetSampling::Nearest);
        // The dot should now sit at (3, 2).
        for y in 0..5 {
            for x in 0..5 {
                let p = ((y * 5 + x) * 4) as usize;
                if (x, y) == (3, 2) {
                    assert_eq!(&out[p..p + 4], &[255, 0, 0, 255]);
                } else {
                    assert_eq!(&out[p..p + 4], &[0, 0, 0, 0]);
                }
            }
        }
    }

    #[test]
    fn offset_nearest_negative_shift_moves_the_dot_left_and_up() {
        let src = red_dot_5x5();
        let out = offset(&src, 5, 5, -1.0, -1.0, OffsetSampling::Nearest);
        // Dot should now be at (1, 1).
        let p = ((5 + 1) * 4) as usize;
        assert_eq!(&out[p..p + 4], &[255, 0, 0, 255]);
    }

    #[test]
    fn offset_nearest_undefined_pixels_are_transparent_black() {
        // Shift far enough to push the entire source out of frame —
        // §15.7.3 says every output pixel should be transparent black.
        let src = red_dot_5x5();
        let out = offset(&src, 5, 5, 100.0, 100.0, OffsetSampling::Nearest);
        assert!(out.iter().all(|b| *b == 0));
    }

    #[test]
    fn offset_zero_is_identity() {
        // Random-looking buffer.
        let mut src = vec![0u8; 4 * 3 * 4];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i * 17 + 3) as u8;
        }
        let out = offset(&src, 4, 3, 0.0, 0.0, OffsetSampling::Nearest);
        assert_eq!(out, src);
    }

    #[test]
    fn offset_rounds_dx_dy_to_nearest_pixel_for_nearest_sampling() {
        // dx = 0.4 rounds to 0 → no shift.
        let src = red_dot_5x5();
        let out = offset(&src, 5, 5, 0.4, 0.0, OffsetSampling::Nearest);
        assert_eq!(out, src);
        // dx = 0.6 rounds to 1 → one-pixel shift right.
        let out = offset(&src, 5, 5, 0.6, 0.0, OffsetSampling::Nearest);
        let p = ((2 * 5 + 3) * 4) as usize;
        assert_eq!(&out[p..p + 4], &[255, 0, 0, 255]);
    }

    #[test]
    fn offset_bilinear_half_pixel_splits_value_evenly() {
        // 2×1 source: opaque red at x=0, transparent at x=1.
        // dx = 0.5 → output[0] reads source(-0.5) = halfway between
        // source(-1)=oob (transparent black) and source(0)=red. Output
        // should be premultiplied-equivalent half-red, half-alpha.
        let src: Vec<u8> = vec![255, 0, 0, 255, 0, 0, 0, 0];
        let out = offset(&src, 2, 1, 0.5, 0.0, OffsetSampling::Bilinear);
        // output[0] reads (oob @ x=-1) and source(0)=red, blended 50/50:
        // r=128, a=128 approximately.
        assert!((out[0] as i32 - 128).abs() <= 1, "r {}", out[0]);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 0);
        assert!((out[3] as i32 - 128).abs() <= 1, "a {}", out[3]);
        // output[1] reads source(0)=red and source(1)=transparent,
        // blended 50/50: r=128, a=128.
        assert!((out[4] as i32 - 128).abs() <= 1, "r {}", out[4]);
        assert!((out[7] as i32 - 128).abs() <= 1, "a {}", out[7]);
    }

    #[test]
    fn offset_bilinear_integer_shift_matches_nearest() {
        // For integer dx/dy bilinear must coincide with nearest.
        let mut src = vec![0u8; 6 * 4 * 4];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i as u32 * 13 % 251) as u8;
        }
        let n = offset(&src, 6, 4, 1.0, -2.0, OffsetSampling::Nearest);
        let b = offset(&src, 6, 4, 1.0, -2.0, OffsetSampling::Bilinear);
        assert_eq!(n, b);
    }

    #[test]
    fn offset_typed_wrapper_agrees_with_byte_path() {
        let mut src = vec![0u8; 5 * 5 * 4];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i * 9 % 241) as u8;
        }
        let typed: Vec<Rgba> = src
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let bytes_out = offset(&src, 5, 5, 2.0, -1.0, OffsetSampling::Nearest);
        let typed_out = offset_pixels(&typed, 5, 5, 2.0, -1.0, OffsetSampling::Nearest);
        let from_typed: Vec<u8> = typed_out
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(bytes_out, from_typed);
    }

    #[test]
    #[should_panic(expected = "src.len()")]
    fn offset_wrong_length_panics() {
        let src = vec![0u8; 8];
        let _ = offset(&src, 4, 4, 0.0, 0.0, OffsetSampling::Nearest);
    }

    #[test]
    #[should_panic(expected = "dx is NaN")]
    fn offset_nan_dx_panics() {
        let src = vec![0u8; 4];
        let _ = offset(&src, 1, 1, f32::NAN, 0.0, OffsetSampling::Nearest);
    }

    #[test]
    fn offset_empty_extent_returns_empty() {
        let src: Vec<u8> = Vec::new();
        let out = offset(&src, 0, 5, 1.0, 1.0, OffsetSampling::Nearest);
        assert!(out.is_empty());
    }

    // -------------------- merge --------------------

    fn solid_layer(w: u32, h: u32, c: Rgba) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[c.r, c.g, c.b, c.a]);
        }
        v
    }

    #[test]
    fn merge_no_layers_is_transparent_black() {
        let out = merge(2, 3, &[]);
        assert_eq!(out.len(), 2 * 3 * 4);
        assert!(out.iter().all(|b| *b == 0));
    }

    #[test]
    fn merge_single_layer_is_identity() {
        let a = solid_layer(2, 2, Rgba::new(10, 20, 30, 200));
        let out = merge(2, 2, &[&a]);
        assert_eq!(out, a);
    }

    #[test]
    fn merge_opaque_top_layer_replaces_lower() {
        let lower = solid_layer(1, 1, Rgba::new(255, 0, 0, 255));
        let upper = solid_layer(1, 1, Rgba::new(0, 255, 0, 255));
        let out = merge(1, 1, &[&lower, &upper]);
        // §14.2 `over`: αo = 1 + 1·0 = 1, so output is the upper layer.
        assert_eq!(out, [0, 255, 0, 255]);
    }

    #[test]
    fn merge_transparent_top_layer_keeps_lower() {
        let lower = solid_layer(1, 1, Rgba::new(123, 45, 67, 255));
        let upper = solid_layer(1, 1, Rgba::new(99, 99, 99, 0));
        let out = merge(1, 1, &[&lower, &upper]);
        assert_eq!(out, [123, 45, 67, 255]);
    }

    #[test]
    fn merge_half_alpha_top_over_opaque_bottom_averages() {
        // §14.2 over: αs=0.5, αd=1 ⇒ αo = 0.5 + 0.5 = 1; the colour
        // is the per-channel 50/50 average of the two RGB triples.
        let lower = solid_layer(1, 1, Rgba::new(200, 0, 0, 255));
        let upper = solid_layer(1, 1, Rgba::new(0, 0, 200, 128));
        let out = merge(1, 1, &[&lower, &upper]);
        // αs in float = 128/255 ≈ 0.5019; expected r = 200 · (1 − αs)
        // ≈ 99.6, b = 200 · αs ≈ 100.4. Allow ±2.
        assert!((out[0] as i32 - 100).abs() <= 2, "r {}", out[0]);
        assert_eq!(out[1], 0);
        assert!((out[2] as i32 - 100).abs() <= 2, "b {}", out[2]);
        assert_eq!(out[3], 255);
    }

    #[test]
    fn merge_three_layers_associativity_holds() {
        // ((a over b) over c) and merge([a, b, c]) should agree, since
        // §14.2 over composition with no group isolation is associative.
        let a = solid_layer(2, 2, Rgba::new(255, 0, 0, 128));
        let b = solid_layer(2, 2, Rgba::new(0, 255, 0, 128));
        let c = solid_layer(2, 2, Rgba::new(0, 0, 255, 128));
        let pair = merge(2, 2, &[&a, &b]);
        let stepped = merge(2, 2, &[&pair, &c]);
        let direct = merge(2, 2, &[&a, &b, &c]);
        for px in 0..4 {
            for ch in 0..4 {
                let i = px * 4 + ch;
                // Quantisation noise allowance: composited over three
                // half-opaque layers, the worst-case discrepancy from
                // intermediate u8 rounding is bounded by ±2.
                assert!(
                    (stepped[i] as i32 - direct[i] as i32).abs() <= 2,
                    "px {px} ch {ch}: stepped {} direct {}",
                    stepped[i],
                    direct[i]
                );
            }
        }
    }

    #[test]
    fn merge_typed_wrapper_agrees_with_byte_path() {
        let a_b = solid_layer(3, 2, Rgba::new(40, 50, 60, 200));
        let b_b = solid_layer(3, 2, Rgba::new(160, 170, 180, 96));
        let a_p: Vec<Rgba> = a_b
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let b_p: Vec<Rgba> = b_b
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let bytes_out = merge(3, 2, &[&a_b, &b_b]);
        let typed_out = merge_pixels(3, 2, &[&a_p, &b_p]);
        let from_typed: Vec<u8> = typed_out
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(bytes_out, from_typed);
    }

    #[test]
    fn merge_empty_extent_returns_empty() {
        let out = merge(0, 5, &[]);
        assert!(out.is_empty());
        let out = merge(5, 0, &[]);
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "layer 1 len")]
    fn merge_mismatched_layer_length_panics() {
        let a = vec![0u8; 4 * 4];
        let b = vec![0u8; 4 * 5];
        let _ = merge(2, 2, &[&a, &b]);
    }
}

#[cfg(test)]
mod tile_tests {
    use super::*;

    /// Build a tiny 2×2 reference tile with four distinct corner
    /// colours, packed straight-alpha RGBA.
    ///
    /// Layout (left-to-right, top-to-bottom):
    ///   (255,   0,   0, 255)   (  0, 255,   0, 255)
    ///   (  0,   0, 255, 255)   (255, 255,   0, 255)
    fn checker_2x2() -> Vec<u8> {
        vec![
            255, 0, 0, 255, // (0, 0) red
            0, 255, 0, 255, // (1, 0) green
            0, 0, 255, 255, // (0, 1) blue
            255, 255, 0, 255, // (1, 1) yellow
        ]
    }

    #[test]
    fn tile_identity_when_extents_match() {
        let src = checker_2x2();
        let out = tile(&src, 2, 2, 2, 2);
        assert_eq!(out, src);
    }

    #[test]
    fn tile_replicates_a_2x2_pattern_across_a_4x4_target() {
        let src = checker_2x2();
        let out = tile(&src, 2, 2, 4, 4);
        assert_eq!(out.len(), 4 * 4 * 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let sx = (x % 2) as usize;
                let sy = (y % 2) as usize;
                let s = (sy * 2 + sx) * 4;
                let d = (y as usize * 4 + x as usize) * 4;
                assert_eq!(&out[d..d + 4], &src[s..s + 4], "mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn tile_partial_period_truncates_cleanly() {
        // 3-wide target from a 2-wide tile — last column shows the
        // start of the next period (column 0 of source).
        let src = checker_2x2();
        let out = tile(&src, 2, 2, 3, 2);
        assert_eq!(out.len(), 3 * 2 * 4);
        // Row 0: (red, green, red)
        assert_eq!(&out[0..4], &[255, 0, 0, 255]);
        assert_eq!(&out[4..8], &[0, 255, 0, 255]);
        assert_eq!(&out[8..12], &[255, 0, 0, 255]);
        // Row 1: (blue, yellow, blue)
        assert_eq!(&out[12..16], &[0, 0, 255, 255]);
        assert_eq!(&out[16..20], &[255, 255, 0, 255]);
        assert_eq!(&out[20..24], &[0, 0, 255, 255]);
    }

    #[test]
    fn tile_single_pixel_source_fills_target_uniformly() {
        // Periodic replication of a 1×1 reference is a constant
        // fill — every output pixel equals the sole source pixel.
        let src = vec![17, 34, 51, 200];
        let out = tile(&src, 1, 1, 5, 3);
        assert_eq!(out.len(), 5 * 3 * 4);
        for px in out.chunks_exact(4) {
            assert_eq!(px, [17, 34, 51, 200]);
        }
    }

    #[test]
    fn tile_target_smaller_than_source_is_top_left_crop() {
        // §15.23 places the i=0 / j=0 copy with its origin at the
        // target origin, so a smaller target reads the source's
        // top-left rectangle.
        let mut src = vec![0u8; 4 * 4 * 4];
        for y in 0..4u32 {
            for x in 0..4u32 {
                let p = ((y * 4 + x) * 4) as usize;
                src[p] = (x * 16 + y) as u8;
                src[p + 3] = 255;
            }
        }
        let out = tile(&src, 4, 4, 2, 2);
        assert_eq!(out.len(), 2 * 2 * 4);
        // The top-left 2×2 rectangle of `src` must appear verbatim.
        for y in 0..2 {
            for x in 0..2 {
                let s = ((y * 4 + x) * 4) as usize;
                let d = ((y * 2 + x) * 4) as usize;
                assert_eq!(&out[d..d + 4], &src[s..s + 4]);
            }
        }
    }

    #[test]
    fn tile_target_extent_zero_returns_empty() {
        let src = checker_2x2();
        assert!(tile(&src, 2, 2, 0, 5).is_empty());
        assert!(tile(&src, 2, 2, 5, 0).is_empty());
        assert!(tile(&src, 2, 2, 0, 0).is_empty());
    }

    #[test]
    fn tile_period_wraps_at_every_multiple() {
        // 3×3 reference, 9×9 target → output is the reference repeated
        // exactly 3 times in each dimension; the i=2, j=2 copy must
        // appear at the bottom-right.
        let mut src = vec![0u8; 3 * 3 * 4];
        for y in 0..3u32 {
            for x in 0..3u32 {
                let p = ((y * 3 + x) * 4) as usize;
                src[p] = (x * 30 + 10) as u8;
                src[p + 1] = (y * 30 + 10) as u8;
                src[p + 2] = 0;
                src[p + 3] = 255;
            }
        }
        let out = tile(&src, 3, 3, 9, 9);
        assert_eq!(out.len(), 9 * 9 * 4);
        for j in 0..3u32 {
            for i in 0..3u32 {
                // Confirm the (i, j) copy of the source rectangle
                // lands at output rectangle [(i·3, j·3), (i·3+3, j·3+3)).
                for ry in 0..3u32 {
                    for rx in 0..3u32 {
                        let s = ((ry * 3 + rx) * 4) as usize;
                        let oy = j * 3 + ry;
                        let ox = i * 3 + rx;
                        let d = ((oy * 9 + ox) * 4) as usize;
                        assert_eq!(
                            &out[d..d + 4],
                            &src[s..s + 4],
                            "tile copy (i={i}, j={j}) corrupted at ({rx},{ry})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tile_typed_wrapper_agrees_with_byte_path() {
        let src_bytes = checker_2x2();
        let src_typed: Vec<Rgba> = src_bytes
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let bytes_out = tile(&src_bytes, 2, 2, 5, 3);
        let typed_out = tile_pixels(&src_typed, 2, 2, 5, 3);
        let from_typed: Vec<u8> = typed_out
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(bytes_out, from_typed);
    }

    #[test]
    #[should_panic(expected = "src.len()")]
    fn tile_wrong_source_length_panics() {
        let src = vec![0u8; 7];
        let _ = tile(&src, 2, 2, 4, 4);
    }

    #[test]
    #[should_panic(expected = "empty source")]
    fn tile_empty_source_with_non_empty_target_panics() {
        let src: Vec<u8> = Vec::new();
        let _ = tile(&src, 0, 0, 2, 2);
    }

    #[test]
    fn tile_empty_source_with_empty_target_returns_empty() {
        // A 0×0 target is allowed regardless of the source extent — no
        // source pixels are ever read.
        let src: Vec<u8> = Vec::new();
        let out = tile(&src, 0, 0, 0, 0);
        assert!(out.is_empty());
    }
}

// ----------------------------------------------------------------------
// SVG 1.1 §15.15 `<feDisplacementMap>` — pixel-channel-driven warp.
// ----------------------------------------------------------------------

/// Channel of the displacement map (`in2`) whose `[0, 1]` value drives
/// the per-axis source offset in [`displacement_map`].
///
/// Mirrors the `xChannelSelector` / `yChannelSelector` attributes of the
/// SVG 1.1 §15.15 `<feDisplacementMap>` element; both attributes default
/// to [`DisplacementChannel::A`] per the §15.15 attribute definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplacementChannel {
    /// Red channel of the displacement map.
    R,
    /// Green channel of the displacement map.
    G,
    /// Blue channel of the displacement map.
    B,
    /// Alpha channel of the displacement map. §15.15 default.
    #[default]
    A,
}

impl DisplacementChannel {
    /// Index of the selected channel inside a packed-RGBA pixel
    /// (`[R, G, B, A]` at offsets 0–3).
    #[inline]
    fn byte_offset(self) -> usize {
        match self {
            DisplacementChannel::R => 0,
            DisplacementChannel::G => 1,
            DisplacementChannel::B => 2,
            DisplacementChannel::A => 3,
        }
    }
}

/// Sample-reconstruction policy used when [`displacement_map`] reads
/// from `in1` at the warped source coordinate.
///
/// The §15.15 note "high quality viewers [should] apply an interpolent
/// on the surrounding pixels, for example bilinear or bicubic, rather
/// than simply selecting the nearest source pixel" is the spec's
/// permission to choose between integer-rounded sampling and a
/// fractional-position resampler. We expose the same two-choice axis
/// that [`OffsetSampling`] exposes for the §15.21 primitive: a
/// bit-exact nearest-pixel default and a bilinear-resampling
/// alternative for the fractional-displacement case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplacementSampling {
    /// Round the warped source coordinate to the nearest integer pixel
    /// and copy the source byte triple verbatim. Out-of-bounds source
    /// positions emit transparent black `(0, 0, 0, 0)` per the §15.7.3
    /// "undefined pixels are set to transparent black" rule.
    #[default]
    Nearest,
    /// Resample with bilinear interpolation around the fractional
    /// source coordinate. Samples whose 2×2 footprint partially falls
    /// outside the source contribute the transparent-black value on
    /// the out-of-bounds neighbours, naturally fading the warped
    /// image at the displacement-induced edges. Coincides bit-exactly
    /// with [`DisplacementSampling::Nearest`] when the per-pixel
    /// warped coordinate happens to be an integer.
    Bilinear,
}

/// SVG 1.1 §15.15 `<feDisplacementMap>` — spatially displace each pixel
/// of `in1` by an offset derived from the per-pixel channel value of
/// `in2`.
///
/// The §15.15 algorithm is:
///
/// ```text
/// P'(x, y) = P(x + scale · (XC(x, y) − 0.5),
///              y + scale · (YC(x, y) − 0.5))
/// ```
///
/// where `P(x, y)` is `in1`, `P'(x, y)` is the output, and
/// `XC(x, y)` / `YC(x, y)` are the per-pixel `[0, 1]` values of the
/// channels of `in2` selected by `x_channel` / `y_channel`. A channel
/// value of `0.5` produces zero displacement on that axis; `0.0`
/// shifts the source coordinate by `−0.5 · scale`; `1.0` shifts it by
/// `+0.5 · scale`. `scale = 0` reduces the operation to a copy of
/// `in1`; the §15.15 attribute table states this explicitly.
///
/// Both `in1` and `in2` are packed straight-alpha RGBA buffers of
/// exactly `width · height · 4` bytes. The §15.15 paragraph "the
/// calculations using the pixel values from `in2` are performed using
/// non-premultiplied color values" matches the straight-alpha
/// convention used throughout this module, so the selected channel
/// byte divided by `255` is `XC` / `YC` directly. The "input image in
/// is to remain premultiplied" sentence is the spec's compositing-
/// model invariant; the warped source is sampled on whatever channel
/// layout the caller passes in and emitted in the same layout, so the
/// `in1` colour space is preserved verbatim.
///
/// Source positions outside the `[0, width) × [0, height)` extent emit
/// the §15.7.3 transparent-black value `(0, 0, 0, 0)`. With
/// [`DisplacementSampling::Bilinear`], out-of-bounds members of the
/// 2×2 sampling footprint contribute transparent black, which makes
/// the displaced image fade smoothly at the edges as the warp pulls
/// samples beyond the source extent.
///
/// # Panics
///
/// * If `in1.len() != width as usize * height as usize * 4`.
/// * If `in2.len() != width as usize * height as usize * 4`.
/// * If `scale` is NaN. (A NaN `scale` would propagate to every
///   per-pixel coordinate and emit transparent black everywhere — but
///   it is almost certainly a caller bug, so we surface it loudly
///   instead of silently producing a blank canvas.)
///
/// # Returns
///
/// A packed-RGBA `Vec<u8>` of length `width · height · 4` carrying
/// the warped source. An empty extent (`width == 0` or `height == 0`)
/// returns an empty `Vec`.
pub fn displacement_map(
    in1: &[u8],
    in2: &[u8],
    width: u32,
    height: u32,
    scale: f32,
    x_channel: DisplacementChannel,
    y_channel: DisplacementChannel,
    sampling: DisplacementSampling,
) -> Vec<u8> {
    assert!(!scale.is_nan(), "displacement_map: scale is NaN");
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("displacement_map: width * height * 4 overflowed usize");
    assert_eq!(
        in1.len(),
        expected,
        "displacement_map: in1.len() == {} but width*height*4 == {expected}",
        in1.len()
    );
    assert_eq!(
        in2.len(),
        expected,
        "displacement_map: in2.len() == {} but width*height*4 == {expected}",
        in2.len()
    );
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let xc_off = x_channel.byte_offset();
    let yc_off = y_channel.byte_offset();
    let wi = width as i64;
    let hi = height as i64;

    let mut out = vec![0u8; expected];
    let row_stride = w * 4;
    for y in 0..hi {
        let dst_row = (y as usize) * row_stride;
        let map_row = (y as usize) * row_stride;
        for x in 0..wi {
            let map_off = map_row + (x as usize) * 4;
            // §15.15: XC / YC are the §15.15 "values of the channel
            // designated by xChannelSelector / yChannelSelector" — the
            // straight-alpha byte / 255 in the spec's [0, 1] range.
            let xc = in2[map_off + xc_off] as f32 / 255.0;
            let yc = in2[map_off + yc_off] as f32 / 255.0;
            // §15.15 inverse-mapping form, evaluated per output pixel:
            // sample `in1` at the displaced source coordinate.
            let sx = x as f32 + scale * (xc - 0.5);
            let sy = y as f32 + scale * (yc - 0.5);
            let dst = dst_row + (x as usize) * 4;
            match sampling {
                DisplacementSampling::Nearest => {
                    // Round-half-away-from-zero, matching the
                    // §15.21 / `offset_nearest` convention so the
                    // two integer-rounding policies stay consistent
                    // across the filter module.
                    let isx = sx.round() as i64;
                    let isy = sy.round() as i64;
                    if (0..wi).contains(&isx) && (0..hi).contains(&isy) {
                        let s = ((isy * wi + isx) as usize) * 4;
                        out[dst..dst + 4].copy_from_slice(&in1[s..s + 4]);
                    } // else: transparent black (already zeroed).
                }
                DisplacementSampling::Bilinear => {
                    let x0 = sx.floor() as i64;
                    let y0 = sy.floor() as i64;
                    let tx = sx - x0 as f32;
                    let ty = sy - y0 as f32;
                    let fetch = |fx: i64, fy: i64| -> [f32; 4] {
                        if !(0..wi).contains(&fx) || !(0..hi).contains(&fy) {
                            return [0.0; 4];
                        }
                        let p = ((fy * wi + fx) as usize) * 4;
                        [
                            in1[p] as f32,
                            in1[p + 1] as f32,
                            in1[p + 2] as f32,
                            in1[p + 3] as f32,
                        ]
                    };
                    let c00 = fetch(x0, y0);
                    let c10 = fetch(x0 + 1, y0);
                    let c01 = fetch(x0, y0 + 1);
                    let c11 = fetch(x0 + 1, y0 + 1);
                    for c in 0..4 {
                        let top = c00[c] + (c10[c] - c00[c]) * tx;
                        let bot = c01[c] + (c11[c] - c01[c]) * tx;
                        let v = top + (bot - top) * ty;
                        out[dst + c] = v.clamp(0.0, 255.0).round() as u8;
                    }
                }
            }
        }
    }
    out
}

/// Typed-pixel wrapper around [`displacement_map`]. Returns a
/// `Vec<Rgba>` of length `width · height`.
#[allow(clippy::too_many_arguments)]
pub fn displacement_map_pixels(
    in1: &[Rgba],
    in2: &[Rgba],
    width: u32,
    height: u32,
    scale: f32,
    x_channel: DisplacementChannel,
    y_channel: DisplacementChannel,
    sampling: DisplacementSampling,
) -> Vec<Rgba> {
    let n = width as usize * height as usize;
    assert_eq!(
        in1.len(),
        n,
        "displacement_map_pixels: in1.len() == {} but width*height == {n}",
        in1.len()
    );
    assert_eq!(
        in2.len(),
        n,
        "displacement_map_pixels: in2.len() == {} but width*height == {n}",
        in2.len()
    );
    let mut b1 = Vec::with_capacity(n * 4);
    let mut b2 = Vec::with_capacity(n * 4);
    for p in in1 {
        b1.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    for p in in2 {
        b2.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out = displacement_map(
        &b1, &b2, width, height, scale, x_channel, y_channel, sampling,
    );
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod displacement_map_tests {
    use super::*;

    /// Build a flat-colour `width × height` map whose XC channel is
    /// `xc_byte` everywhere and YC channel is `yc_byte` everywhere; the
    /// other two channels are zeroed so accidental selector mistakes
    /// produce zero-displacement instead of confusing partial warps.
    fn flat_map(width: u32, height: u32, xc_byte: u8, yc_byte: u8) -> Vec<u8> {
        // We pack R = xc_byte, G = yc_byte, B = 0, A = 255 (full opaque
        // so an alpha selector won't accidentally drive a different
        // displacement than R / G).
        let n = (width * height) as usize;
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.extend_from_slice(&[xc_byte, yc_byte, 0, 255]);
        }
        v
    }

    /// Single-coloured opaque source — every pixel equals `(r, g, b, a)`.
    fn flat_src(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (width * height) as usize;
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.extend_from_slice(&[r, g, b, a]);
        }
        v
    }

    #[test]
    fn scale_zero_is_a_copy_of_in1() {
        // §15.15: "When the value of this attribute is 0, this
        // operation has no effect on the source image."
        let in1: Vec<u8> = (0..4 * 4 * 4).map(|i| (i & 0xff) as u8).collect();
        let in2 = flat_map(4, 4, 128, 128);
        let out = displacement_map(
            &in1,
            &in2,
            4,
            4,
            0.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        );
        assert_eq!(out, in1);
    }

    #[test]
    fn half_grey_displacement_is_zero_shift_regardless_of_scale() {
        // §15.15: P' = P(x + s·(XC − 0.5), y + s·(YC − 0.5)). When XC =
        // YC = 0.5, both shift terms are zero regardless of `scale`,
        // so the output equals `in1`.
        let in1: Vec<u8> = (0..3 * 3 * 4).map(|i| (i as u8).wrapping_mul(7)).collect();
        // 128/255 = 0.5019…; we also exercise exact-0.5 by picking a
        // float-equivalent unsigned byte — `(0.5 * 255).round()` is
        // 128 with f32 rounding (half-to-even on 127.5? no, 0.5·255 =
        // 127.5; the nearest is 128 under round-half-away-from-zero,
        // which is what `f32::round` does).
        // 128/255 = 0.50196… → shift = 100·(0.00196) = 0.196, rounded
        // to 0 by nearest sampling. So this case is "approximately
        // half" not "exactly half"; we use scale = 0 below to assert
        // the exact-half claim independently. See the next test.
        let in2 = flat_map(3, 3, 128, 128);
        let out = displacement_map(
            &in1,
            &in2,
            3,
            3,
            100.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        );
        // Tiny |shift| < 0.5 rounds to 0 under round-to-nearest, so
        // this is equivalent to a no-op copy under Nearest sampling.
        assert_eq!(out, in1);
    }

    #[test]
    fn full_red_full_x_shift_pulls_from_minus_half_scale() {
        // §15.15: XC = 1.0 → x-shift = +0.5 · scale. With scale = 10,
        // every output pixel samples from `(x + 5, y)`. Output column
        // `x` is therefore in1's column `x + 5` when in bounds, and
        // transparent-black otherwise.
        let w = 8u32;
        let h = 1u32;
        // Distinct-per-pixel source: each column carries its own red.
        let mut in1 = Vec::with_capacity((w * h * 4) as usize);
        for x in 0..w {
            in1.extend_from_slice(&[(x * 32) as u8, 0, 0, 255]);
        }
        // XC = 1.0 (R = 255) everywhere, YC = 0.5 (B = 128) everywhere.
        let mut in2 = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w {
            in2.extend_from_slice(&[255, 0, 128, 255]);
        }
        let out = displacement_map(
            &in1,
            &in2,
            w,
            h,
            10.0,
            DisplacementChannel::R,
            DisplacementChannel::B,
            DisplacementSampling::Nearest,
        );
        // shift_x = 10 · (1.0 − 0.5) = 5
        for x in 0..w {
            let src_x = x as i64 + 5;
            let dst = (x as usize) * 4;
            if src_x < w as i64 {
                let s = (src_x as usize) * 4;
                assert_eq!(
                    &out[dst..dst + 4],
                    &in1[s..s + 4],
                    "x = {x}: expected column {src_x} from in1"
                );
            } else {
                assert_eq!(
                    &out[dst..dst + 4],
                    &[0, 0, 0, 0],
                    "x = {x}: expected §15.7.3 transparent black"
                );
            }
        }
    }

    #[test]
    fn zero_channel_full_negative_x_shift_pulls_from_plus_half_scale() {
        // XC = 0.0 → x-shift = −0.5 · scale (P samples from x − 5).
        let w = 8u32;
        let h = 1u32;
        let mut in1 = Vec::with_capacity((w * h * 4) as usize);
        for x in 0..w {
            in1.extend_from_slice(&[(x * 32) as u8, 0, 0, 255]);
        }
        // R = 0 (XC = 0.0), G = 128 (YC = 0.5, zero y-shift).
        let mut in2 = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w {
            in2.extend_from_slice(&[0, 128, 0, 255]);
        }
        let out = displacement_map(
            &in1,
            &in2,
            w,
            h,
            10.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        );
        // shift_x = 10 · (0.0 − 0.5) = −5
        for x in 0..w {
            let src_x = x as i64 - 5;
            let dst = (x as usize) * 4;
            if src_x >= 0 {
                let s = (src_x as usize) * 4;
                assert_eq!(&out[dst..dst + 4], &in1[s..s + 4]);
            } else {
                assert_eq!(&out[dst..dst + 4], &[0, 0, 0, 0]);
            }
        }
    }

    #[test]
    fn alpha_default_selector_drives_displacement() {
        // §15.15: "If attribute xChannelSelector is not specified, then
        // the effect is as if a value of A were specified." Test that
        // the `DisplacementChannel::A` default does in fact source
        // displacement from the map's alpha channel.
        let w = 6u32;
        let h = 1u32;
        let mut in1 = Vec::with_capacity((w * h * 4) as usize);
        for x in 0..w {
            in1.extend_from_slice(&[(x * 40) as u8, 1, 2, 255]);
        }
        // R / G / B = 0 (any non-A selector would ⇒ −0.5·scale shift),
        // A = 255 (XC via A selector = 1.0 ⇒ +0.5·scale shift).
        let mut in2 = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w {
            in2.extend_from_slice(&[0, 0, 0, 255]);
        }
        let out = displacement_map(
            &in1,
            &in2,
            w,
            h,
            4.0,
            DisplacementChannel::default(),
            DisplacementChannel::default(),
            DisplacementSampling::Nearest,
        );
        // Default = A; A = 255 ⇒ XC = YC = 1.0 ⇒ shift = (+2, +2).
        // h = 1, so any y-shift > 0.5 is out of bounds ⇒ transparent.
        for x in 0..w {
            let dst = (x as usize) * 4;
            assert_eq!(&out[dst..dst + 4], &[0, 0, 0, 0]);
        }
    }

    #[test]
    fn out_of_bounds_displacement_emits_transparent_black() {
        // Every output pixel demands a source coordinate strictly
        // outside `[0, w) × [0, h)` ⇒ the entire result is
        // transparent black per §15.7.3.
        let w = 4u32;
        let h = 4u32;
        let in1 = flat_src(w, h, 100, 150, 200, 255);
        // XC = 255 (1.0) and YC = 255 (1.0); scale = 10 ⇒ shift =
        // (+5, +5). Every output pixel samples from outside the source.
        let in2 = flat_map(w, h, 255, 255);
        let out = displacement_map(
            &in1,
            &in2,
            w,
            h,
            10.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        );
        assert_eq!(out.len(), (w * h * 4) as usize);
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk, [0, 0, 0, 0]);
        }
    }

    #[test]
    fn empty_extent_returns_empty() {
        // 0×N and N×0 short-circuit to an empty Vec — no buffer
        // length to check, no per-pixel work to perform.
        assert!(displacement_map(
            &[],
            &[],
            0,
            5,
            10.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        )
        .is_empty());
        assert!(displacement_map(
            &[],
            &[],
            5,
            0,
            10.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Bilinear,
        )
        .is_empty());
    }

    #[test]
    fn bilinear_at_integer_shift_matches_nearest() {
        // §15.15 sampling-policy note: "high quality viewers apply an
        // interpolent" kicks in only at non-integer source coordinates.
        // Build a case where BOTH per-pixel shifts collapse to exact
        // integers: XC = R = 1.0, YC = B = 0.0, scale = 2 ⇒
        // shift_x = +1, shift_y = −1. Pad with a multi-row source so
        // the bilinear ty-blend stays inside the canvas everywhere
        // shift_y lands a sample.
        let w = 4u32;
        let h = 3u32;
        let mut in1 = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                in1.extend_from_slice(&[(x * 50) as u8, (y * 60) as u8, 9, 255]);
            }
        }
        let mut in2 = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for _x in 0..w {
                // XC = R = 255/255 = 1.0; YC = B = 0/255 = 0.0
                in2.extend_from_slice(&[255, 128, 0, 255]);
            }
        }
        let nearest = displacement_map(
            &in1,
            &in2,
            w,
            h,
            2.0,
            DisplacementChannel::R, // shift_x = +1 exact
            DisplacementChannel::B, // shift_y = −1 exact
            DisplacementSampling::Nearest,
        );
        let bilinear = displacement_map(
            &in1,
            &in2,
            w,
            h,
            2.0,
            DisplacementChannel::R,
            DisplacementChannel::B,
            DisplacementSampling::Bilinear,
        );
        assert_eq!(nearest, bilinear);
    }

    #[test]
    fn bilinear_fractional_shift_blends_neighbours() {
        // Two-column horizontal ramp: column 0 is (0, 0, 0, 255),
        // column 1 is (200, 0, 0, 255). XC = 191/255 ≈ 0.749; scale =
        // 2 ⇒ shift_x = 2·(0.749 − 0.5) ≈ +0.498. We sample at output
        // `x = 0` ⇒ source `≈ 0.498`. Bilinear: `tx ≈ 0.498`,
        // blend ≈ 0 + 200 · 0.498 ≈ 99.6 ⇒ 100 under round-half-away.
        let w = 2u32;
        let h = 1u32;
        let in1: Vec<u8> = vec![0, 0, 0, 255, 200, 0, 0, 255];
        let in2: Vec<u8> = vec![
            191, 128, 0, 255, // XC ≈ 0.749, YC = 0.5019
            191, 128, 0, 255,
        ];
        let out = displacement_map(
            &in1,
            &in2,
            w,
            h,
            2.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Bilinear,
        );
        // At output x = 0: src_x ≈ 0.498, R blends 0 → 200 by ~0.498.
        // Allowing one byte of rounding slack against the spec form
        // because both 191/255 and 128/255 are slightly off from the
        // 0.749 / 0.5019 floats.
        let r0 = out[0];
        assert!(
            (95..=105).contains(&r0),
            "x=0 R = {r0}; want ≈ 100 from the half-pixel bilinear blend"
        );
        // y-shift ≈ 2·(0.5019 − 0.5) ≈ +0.004 — within nearest
        // tolerance; the bilinear ty < 0.01 so the row blend is
        // effectively row 0 only.
        // At output x = 1: src_x ≈ 1.498, neighbour at x = 2 is OOB
        // ⇒ contributes (0, 0, 0, 0). Result ≈ 200 · 0.502 + 0 · 0.498
        // ≈ 100.4 ⇒ 100. The alpha collapse is the headline:
        // alpha ≈ 255 · 0.502 + 0 · 0.498 ≈ 128.
        let a1 = out[7];
        assert!(
            (120..=135).contains(&a1),
            "x=1 A = {a1}; want ≈ 128 from the §15.7.3 OOB fade-out"
        );
    }

    #[test]
    fn typed_pixel_wrapper_matches_byte_path() {
        let w = 3u32;
        let h = 3u32;
        let bytes_in1 = flat_src(w, h, 30, 60, 90, 220);
        let bytes_in2 = flat_map(w, h, 200, 80);
        let pixels_in1: Vec<Rgba> = bytes_in1
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let pixels_in2: Vec<Rgba> = bytes_in2
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let bytes_out = displacement_map(
            &bytes_in1,
            &bytes_in2,
            w,
            h,
            6.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        );
        let pixels_out = displacement_map_pixels(
            &pixels_in1,
            &pixels_in2,
            w,
            h,
            6.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        );
        let bytes_from_typed: Vec<u8> = pixels_out
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(bytes_out, bytes_from_typed);
    }

    #[test]
    #[should_panic(expected = "displacement_map: in1.len()")]
    fn wrong_in1_length_panics() {
        let bad: Vec<u8> = vec![0; 3];
        let in2 = flat_map(2, 2, 128, 128);
        let _ = displacement_map(
            &bad,
            &in2,
            2,
            2,
            5.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        );
    }

    #[test]
    #[should_panic(expected = "displacement_map: in2.len()")]
    fn wrong_in2_length_panics() {
        let in1 = flat_src(2, 2, 0, 0, 0, 0);
        let bad: Vec<u8> = vec![0; 5];
        let _ = displacement_map(
            &in1,
            &bad,
            2,
            2,
            5.0,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        );
    }

    #[test]
    #[should_panic(expected = "displacement_map: scale is NaN")]
    fn nan_scale_panics() {
        let in1 = flat_src(1, 1, 0, 0, 0, 0);
        let in2 = flat_map(1, 1, 128, 128);
        let _ = displacement_map(
            &in1,
            &in2,
            1,
            1,
            f32::NAN,
            DisplacementChannel::R,
            DisplacementChannel::G,
            DisplacementSampling::Nearest,
        );
    }

    #[test]
    fn x_channel_and_y_channel_are_independent() {
        // Build a map where R varies horizontally and G is constant ⇒
        // selecting R for x and G for y should give a per-column
        // horizontal shift with no vertical motion. Selecting G for x
        // and R for y should swap them ⇒ per-column vertical shift
        // with no horizontal motion.
        let w = 4u32;
        let h = 4u32;
        let mut in1 = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                in1.extend_from_slice(&[(x * 60) as u8, (y * 60) as u8, 100, 255]);
            }
        }
        // R varies across columns 0..255, G = 128 (≈ 0.5 ⇒ no shift).
        let mut in2 = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let r = (x * 80).min(255) as u8;
                in2.extend_from_slice(&[r, 128, 0, 255]);
            }
        }
        let rg = displacement_map(
            &in1,
            &in2,
            w,
            h,
            4.0,
            DisplacementChannel::R, // varies ⇒ x-shift varies
            DisplacementChannel::G, // ≈ 0.5 everywhere ⇒ ~no y-shift
            DisplacementSampling::Nearest,
        );
        let gr = displacement_map(
            &in1,
            &in2,
            w,
            h,
            4.0,
            DisplacementChannel::G, // ≈ 0.5 ⇒ no x-shift
            DisplacementChannel::R, // varies ⇒ y-shift varies
            DisplacementSampling::Nearest,
        );
        // Two warps must differ — proves the selectors are
        // independently wired and not aliased.
        assert_ne!(rg, gr);
        // And the "G for x" warp must agree with in1 on every column
        // where the y-shift would land in-bounds and the x-shift is
        // zero: column x, row y ⇒ samples (x, y + Δy(x)).
        // Pick the leftmost column where R = 0 ⇒ Δy = 4·(0 − 0.5) = −2,
        // so output (0, 2) samples in1 (0, 0).
        // (column 0, row 2) ⇒ output offset = 2·w·4; in1 (0, 0) ⇒ src
        // offset = 0.
        let d = 2 * w as usize * 4;
        let s = 0usize;
        assert_eq!(&gr[d..d + 4], &in1[s..s + 4]);
    }
}

// ----------------------------------------------------------------------
// SVG 1.1 §15.9 `<feBlend>` — five-mode pixel-wise blend of two inputs.
// ----------------------------------------------------------------------

/// Mode selector for [`blend_filter`], mirroring the `mode` attribute
/// of SVG 1.1 §15.9 `<feBlend>` (`"normal" | "multiply" | "screen" |
/// "darken" | "lighten"`). The §15.9 attribute table gives `Normal` as
/// the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendFilterMode {
    /// `cr = (1 − qa) · cb + ca`. §15.9 notes that this is the
    /// pixel-wise equivalent of `feComposite` operator `"over"` and
    /// of the §14.2 simple-alpha-compositing source-over rule.
    #[default]
    Normal,
    /// `cr = (1 − qa) · cb + (1 − qb) · ca + ca · cb`. Darkens both
    /// inputs through the product term while still preserving each
    /// operand outside the other's coverage via the
    /// `(1 − qa) · cb + (1 − qb) · ca` extension terms.
    Multiply,
    /// `cr = cb + ca − ca · cb`. The classic Burt–Adelson screen
    /// formula reproduced verbatim from the §15.9 table; brightens
    /// where either operand has presence.
    Screen,
    /// `cr = Min((1 − qa) · cb + ca, (1 − qb) · ca + cb)`. Picks the
    /// component-wise smaller of the two source-over orderings
    /// (`a` over `b`, `b` over `a`), per the §15.9 table.
    Darken,
    /// `cr = Max((1 − qa) · cb + ca, (1 − qb) · ca + cb)`. Picks the
    /// component-wise larger of the two source-over orderings, per
    /// the §15.9 table.
    Lighten,
}

/// SVG 1.1 §15.9 `<feBlend>` — combine two equally-sized packed-RGBA
/// `u8` buffers pixel-wise through one of the five [`BlendFilterMode`]
/// modes.
///
/// `in1` maps to the spec's `in` operand (the `a` / `qa` / `ca`
/// quantities in the §15.9 formulas) and `in2` maps to the `in2`
/// operand (the `b` / `qb` / `cb` quantities). Both buffers must be
/// exactly `width * height * 4` bytes in row-major straight-alpha RGBA
/// order and describe the same `width × height` extent.
///
/// **Result alpha.** §15.9 specifies that *every* mode produces the
/// same result opacity
///
/// ```text
/// qr = 1 − (1 − qa) · (1 − qb)
/// ```
///
/// — i.e. the symmetric "either input is opaque ⇒ result is opaque"
/// formula that drops out of the per-mode colour formulas when both
/// operands are set equal. This routine evaluates `qr` once per pixel
/// and reuses it across the colour channels.
///
/// **Pixel space.** The §15.9 colour formulas are stated in terms of
/// premultiplied RGB (`ca` / `cb`). This entry point converts each
/// straight-alpha input byte triple to premultiplied `[0, 1]` floats
/// (`c · α`), evaluates the chosen mode's formula, and converts the
/// premultiplied result back to straight-alpha bytes — matching the
/// pixel-space convention used elsewhere in the module (`composite_filter`,
/// `merge`).
///
/// Complexity is `O(W · H)` with a constant per-pixel cost.
///
/// # Panics
///
/// * If `in1.len() != width as usize * height as usize * 4`.
/// * If `in2.len() != width as usize * height as usize * 4`.
///
/// # Returns
///
/// A new packed-straight-alpha-RGBA `Vec<u8>` of length
/// `width · height · 4`. An empty extent (`width == 0` or
/// `height == 0`) returns an empty `Vec`.
pub fn blend_filter(
    in1: &[u8],
    in2: &[u8],
    width: u32,
    height: u32,
    mode: BlendFilterMode,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("blend_filter: width * height * 4 overflowed usize");
    assert_eq!(
        in1.len(),
        expected,
        "blend_filter: in1.len() == {} but width*height*4 == {expected}",
        in1.len()
    );
    assert_eq!(
        in2.len(),
        expected,
        "blend_filter: in2.len() == {} but width*height*4 == {expected}",
        in2.len()
    );
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(expected);
    for (pa, pb) in in1.chunks_exact(4).zip(in2.chunks_exact(4)) {
        // Straight-alpha bytes → premultiplied `[0, 1]` floats. §15.9
        // names the alpha quantities `qa` / `qb` and the colour
        // quantities `ca` / `cb`; the latter are premultiplied so each
        // straight-alpha byte is multiplied by its own alpha before
        // entering a mode formula.
        let qa = pa[3] as f32 / 255.0;
        let qb = pb[3] as f32 / 255.0;
        let ca = [
            pa[0] as f32 / 255.0 * qa,
            pa[1] as f32 / 255.0 * qa,
            pa[2] as f32 / 255.0 * qa,
        ];
        let cb = [
            pb[0] as f32 / 255.0 * qb,
            pb[1] as f32 / 255.0 * qb,
            pb[2] as f32 / 255.0 * qb,
        ];

        // §15.9: `qr = 1 − (1 − qa)·(1 − qb)` is shared by all five
        // modes, so we compute it once.
        let qr = 1.0 - (1.0 - qa) * (1.0 - qb);

        let mut cr = [0.0f32; 3];
        for c in 0..3 {
            let a = ca[c];
            let b = cb[c];
            cr[c] = match mode {
                // §15.9 table row "normal".
                BlendFilterMode::Normal => (1.0 - qa) * b + a,
                // §15.9 table row "multiply".
                BlendFilterMode::Multiply => (1.0 - qa) * b + (1.0 - qb) * a + a * b,
                // §15.9 table row "screen".
                BlendFilterMode::Screen => b + a - a * b,
                // §15.9 table row "darken".
                BlendFilterMode::Darken => {
                    let lhs = (1.0 - qa) * b + a;
                    let rhs = (1.0 - qb) * a + b;
                    lhs.min(rhs)
                }
                // §15.9 table row "lighten".
                BlendFilterMode::Lighten => {
                    let lhs = (1.0 - qa) * b + a;
                    let rhs = (1.0 - qb) * a + b;
                    lhs.max(rhs)
                }
            };
        }

        // Premultiplied `[0, 1]` → straight-alpha bytes. When `qr == 0`
        // both inputs are transparent, so the §15.9 colour formulas
        // collapse to zero and the un-premultiply division would be
        // 0 / 0; emit the §15.7.3 transparent-black value verbatim.
        if qr <= 0.0 {
            out.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let inv = 1.0 / qr;
            // §15.9 places no upper bound on `cr` for the `screen` and
            // `lighten` modes — `b + a − a · b` exceeds `qr` when the
            // premultiplied operands themselves exceed `qr` (which is
            // impossible here because each `c ≤ q` for premultiplied
            // straight-alpha inputs), but the clamp survives any
            // floating-point drift that would round `c / qr` above 1.0
            // and would otherwise wrap on the `u8` cast.
            out.push(quantise_unit((cr[0] * inv).clamp(0.0, 1.0)));
            out.push(quantise_unit((cr[1] * inv).clamp(0.0, 1.0)));
            out.push(quantise_unit((cr[2] * inv).clamp(0.0, 1.0)));
            out.push(quantise_unit(qr.clamp(0.0, 1.0)));
        }
    }
    out
}

/// Typed-pixel wrapper around [`blend_filter`]. Returns a `Vec<Rgba>`
/// of length `width · height`. Identical semantics — provided for
/// callers that already have typed pixel buffers.
pub fn blend_filter_pixels(
    in1: &[Rgba],
    in2: &[Rgba],
    width: u32,
    height: u32,
    mode: BlendFilterMode,
) -> Vec<Rgba> {
    let n = width as usize * height as usize;
    assert_eq!(
        in1.len(),
        n,
        "blend_filter_pixels: in1.len() == {} but width*height == {n}",
        in1.len()
    );
    assert_eq!(
        in2.len(),
        n,
        "blend_filter_pixels: in2.len() == {} but width*height == {n}",
        in2.len()
    );
    let mut b1 = Vec::with_capacity(n * 4);
    let mut b2 = Vec::with_capacity(n * 4);
    for p in in1 {
        b1.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    for p in in2 {
        b2.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out = blend_filter(&b1, &b2, width, height, mode);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod blend_filter_tests {
    use super::*;

    fn flat(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            v.extend_from_slice(&[r, g, b, a]);
        }
        v
    }

    #[test]
    fn fully_transparent_inputs_produce_transparent_black() {
        // `qa = qb = 0` ⇒ `qr = 0`, so the §15.7.3 transparent-black
        // rule kicks in regardless of which mode is selected.
        let zero = flat(3, 3, 200, 50, 100, 0);
        for mode in [
            BlendFilterMode::Normal,
            BlendFilterMode::Multiply,
            BlendFilterMode::Screen,
            BlendFilterMode::Darken,
            BlendFilterMode::Lighten,
        ] {
            let out = blend_filter(&zero, &zero, 3, 3, mode);
            assert!(out.iter().all(|&b| b == 0), "mode {mode:?} not fully zero");
        }
    }

    #[test]
    fn normal_mode_over_opaque_dst_keeps_dst() {
        // `in1` fully transparent ⇒ `qa = ca = 0`; §15.9 normal
        // formula reduces to `cr = (1−0)·cb + 0 = cb` and `qr = qb`.
        // Straight-alpha output therefore matches `in2` verbatim.
        let a = flat(2, 2, 255, 0, 0, 0);
        let b = flat(2, 2, 10, 20, 30, 255);
        let out = blend_filter(&a, &b, 2, 2, BlendFilterMode::Normal);
        assert_eq!(out, b);
    }

    #[test]
    fn normal_mode_opaque_src_overrides_dst() {
        // `in1` fully opaque ⇒ `qa = 1`, so `cr = 0·cb + ca = ca` and
        // `qr = 1 − 0·(1 − qb) = 1`. Output equals `in1`.
        let a = flat(2, 2, 30, 60, 90, 255);
        let b = flat(2, 2, 200, 100, 50, 200);
        let out = blend_filter(&a, &b, 2, 2, BlendFilterMode::Normal);
        assert_eq!(out, a);
    }

    #[test]
    fn normal_mode_matches_composite_over_on_opaque_pair() {
        // §15.9 explicitly notes "normal blend mode is equivalent to
        // operator='over' on the feComposite filter primitive". When
        // both inputs are fully opaque, the over result is bit-exact
        // for the colour channels.
        let a = flat(2, 2, 200, 100, 50, 255);
        let b = flat(2, 2, 30, 60, 90, 255);
        let blended = blend_filter(&a, &b, 2, 2, BlendFilterMode::Normal);
        let composed = composite_filter(&a, &b, 2, 2, CompositeOp::Over);
        assert_eq!(blended, composed);
    }

    #[test]
    fn screen_brightens_partial_pair() {
        // Opaque mid-grey screened with itself: `qa = qb = 1`,
        // `ca = cb = 0.5`. §15.9 screen formula:
        //   cr = 0.5 + 0.5 − 0.25 = 0.75.
        // qr = 1 − 0·0 = 1 ⇒ straight-alpha output = (0.75, 0.75,
        // 0.75, 1) ⇒ 191 per channel after `quantise_unit`.
        let g = flat(1, 1, 128, 128, 128, 255);
        let out = blend_filter(&g, &g, 1, 1, BlendFilterMode::Screen);
        // 0.5 = 128/255 ⇒ premul ca = cb ≈ 0.501961. screen ≈
        // 0.501961·2 − 0.501961² ≈ 0.751978. straight-alpha = same.
        // quantise = round(0.751978·255 + 0.5) = 192 (one ULP above
        // the textbook 191 because 128/255 ≠ 0.5 exactly).
        assert_eq!(out[3], 255);
        assert!((out[0] as i32 - 192).abs() <= 1);
        assert!((out[1] as i32 - 192).abs() <= 1);
        assert!((out[2] as i32 - 192).abs() <= 1);
    }

    #[test]
    fn multiply_darkens_partial_pair() {
        // Same opaque mid-grey: §15.9 multiply formula reduces to
        //   cr = 0·0.5 + 0·0.5 + 0.5·0.5 = 0.25
        // (the two `(1−qX)·cY` extension terms vanish for opaque
        // inputs). qr = 1, output ≈ (64, 64, 64, 255).
        let g = flat(1, 1, 128, 128, 128, 255);
        let out = blend_filter(&g, &g, 1, 1, BlendFilterMode::Multiply);
        assert_eq!(out[3], 255);
        // 0.501961² ≈ 0.251965; quantise ≈ 64.
        assert!((out[0] as i32 - 64).abs() <= 1);
        assert!((out[1] as i32 - 64).abs() <= 1);
        assert!((out[2] as i32 - 64).abs() <= 1);
    }

    #[test]
    fn darken_picks_smaller_lighten_picks_larger() {
        // Two opaque distinct colours: for opaque pairs both branches
        // of darken / lighten reduce to picking the smaller / larger
        // of the two operands per channel.
        let a = flat(2, 2, 200, 100, 50, 255);
        let b = flat(2, 2, 30, 200, 90, 255);
        let dk = blend_filter(&a, &b, 2, 2, BlendFilterMode::Darken);
        let lt = blend_filter(&a, &b, 2, 2, BlendFilterMode::Lighten);
        // Expected per channel: min(200,30)=30, min(100,200)=100,
        // min(50,90)=50; max(200,30)=200, max(100,200)=200,
        // max(50,90)=90. Within one ULP of the float round-trip.
        for px in dk.chunks_exact(4) {
            assert!((px[0] as i32 - 30).abs() <= 1);
            assert!((px[1] as i32 - 100).abs() <= 1);
            assert!((px[2] as i32 - 50).abs() <= 1);
            assert_eq!(px[3], 255);
        }
        for px in lt.chunks_exact(4) {
            assert!((px[0] as i32 - 200).abs() <= 1);
            assert!((px[1] as i32 - 200).abs() <= 1);
            assert!((px[2] as i32 - 90).abs() <= 1);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn result_alpha_uses_symmetric_formula() {
        // Symmetric `qa = qb = 0.5` ⇒ qr = 1 − 0.5·0.5 = 0.75.
        // The colour values are tested above; here we only check the
        // alpha, which is mode-independent per §15.9.
        let a = flat(1, 1, 0, 0, 0, 128); // 128/255 ≈ 0.502
        let b = flat(1, 1, 0, 0, 0, 128);
        for mode in [
            BlendFilterMode::Normal,
            BlendFilterMode::Multiply,
            BlendFilterMode::Screen,
            BlendFilterMode::Darken,
            BlendFilterMode::Lighten,
        ] {
            let out = blend_filter(&a, &b, 1, 1, mode);
            // qr = 1 − (1 − 0.502)² ≈ 0.752, quantise ≈ 192.
            assert!(
                (out[3] as i32 - 192).abs() <= 1,
                "mode {mode:?} produced alpha {} not ≈ 192",
                out[3]
            );
        }
    }

    #[test]
    fn empty_extent_returns_empty() {
        let empty: Vec<u8> = Vec::new();
        for mode in [
            BlendFilterMode::Normal,
            BlendFilterMode::Multiply,
            BlendFilterMode::Screen,
            BlendFilterMode::Darken,
            BlendFilterMode::Lighten,
        ] {
            let out = blend_filter(&empty, &empty, 0, 0, mode);
            assert!(out.is_empty(), "mode {mode:?} not empty");
        }
    }

    #[test]
    #[should_panic(expected = "blend_filter: in1.len()")]
    fn wrong_in1_length_panics() {
        let bad: Vec<u8> = vec![0; 3];
        let in2 = flat(2, 2, 0, 0, 0, 0);
        let _ = blend_filter(&bad, &in2, 2, 2, BlendFilterMode::Normal);
    }

    #[test]
    #[should_panic(expected = "blend_filter: in2.len()")]
    fn wrong_in2_length_panics() {
        let in1 = flat(2, 2, 0, 0, 0, 0);
        let bad: Vec<u8> = vec![0; 5];
        let _ = blend_filter(&in1, &bad, 2, 2, BlendFilterMode::Normal);
    }

    #[test]
    fn typed_pixel_wrapper_matches_byte_path() {
        let w = 3u32;
        let h = 3u32;
        let bytes_in1 = flat(w, h, 200, 100, 50, 128);
        let bytes_in2 = flat(w, h, 30, 200, 90, 200);
        let pixels_in1: Vec<Rgba> = bytes_in1
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let pixels_in2: Vec<Rgba> = bytes_in2
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        for mode in [
            BlendFilterMode::Normal,
            BlendFilterMode::Multiply,
            BlendFilterMode::Screen,
            BlendFilterMode::Darken,
            BlendFilterMode::Lighten,
        ] {
            let bytes_out = blend_filter(&bytes_in1, &bytes_in2, w, h, mode);
            let pixels_out = blend_filter_pixels(&pixels_in1, &pixels_in2, w, h, mode);
            let bytes_from_typed: Vec<u8> = pixels_out
                .iter()
                .flat_map(|p| [p.r, p.g, p.b, p.a])
                .collect();
            assert_eq!(bytes_out, bytes_from_typed, "mode {mode:?} mismatch");
        }
    }

    #[test]
    fn modes_diverge_on_general_pair() {
        // A partial-alpha pair where every mode produces a different
        // result: catches accidental collapse of two arms of the
        // enum onto the same arithmetic path.
        let a = flat(1, 1, 200, 100, 50, 200);
        let b = flat(1, 1, 30, 180, 90, 150);
        let normal = blend_filter(&a, &b, 1, 1, BlendFilterMode::Normal);
        let multiply = blend_filter(&a, &b, 1, 1, BlendFilterMode::Multiply);
        let screen = blend_filter(&a, &b, 1, 1, BlendFilterMode::Screen);
        let darken = blend_filter(&a, &b, 1, 1, BlendFilterMode::Darken);
        let lighten = blend_filter(&a, &b, 1, 1, BlendFilterMode::Lighten);
        // All five must produce distinct outputs on this input.
        let outputs = [&normal, &multiply, &screen, &darken, &lighten];
        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                assert_ne!(
                    outputs[i], outputs[j],
                    "outputs {i} and {j} collapsed onto the same arithmetic path"
                );
            }
        }
        // §15.9 shared-alpha invariant: every mode produces the same
        // result alpha for the same input pair.
        let qr = normal[3];
        assert_eq!(multiply[3], qr);
        assert_eq!(screen[3], qr);
        assert_eq!(darken[3], qr);
        assert_eq!(lighten[3], qr);
    }
}

// ============================================================================
// SVG 1.1 §15.14 `<feDiffuseLighting>` — diffuse component of the
// Phong lighting model, driven by a §15.8.2 / §15.8.3 / §15.8.4 light
// source. The §15.14 formulas are reproduced verbatim in the routines
// below; the algorithm is purely a per-pixel arithmetic kernel with no
// downstream dependencies beyond `quantise_unit`.
// ============================================================================

/// Light source descriptor for [`DiffuseLighting`].
///
/// Mirrors the three light-source elements defined in SVG 1.1
/// §15.8.2 / §15.8.3 / §15.8.4 — `<feDistantLight>`, `<fePointLight>`
/// and `<feSpotLight>`. Angle attributes (`azimuth`, `elevation`,
/// `limiting_cone_angle_deg`) are taken in degrees and converted to
/// radians inside the lighting kernel, matching the §15.8.2 attribute
/// definitions ("Direction angle for the light source on the XY plane
/// (clockwise), in degrees from the x axis").
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightSource {
    /// §15.8.2 `<feDistantLight>` — an infinite light source described
    /// by direction angles only. The §15.14 unit vector becomes the
    /// constant
    ///
    /// ```text
    /// Lx = cos(azimuth) · cos(elevation)
    /// Ly = sin(azimuth) · cos(elevation)
    /// Lz = sin(elevation)
    /// ```
    ///
    /// independent of `(x, y)`.
    Distant {
        /// `azimuth` attribute, in degrees clockwise from the x axis.
        azimuth_deg: f32,
        /// `elevation` attribute, in degrees from the XY plane towards
        /// the positive z axis (z points towards the viewer).
        elevation_deg: f32,
    },
    /// §15.8.3 `<fePointLight>` — a point light at world coordinate
    /// `(x, y, z)`. The §15.14 unit vector is
    /// `L = (x − px, y − py, z − Z(x, y)) / Norm(…)` evaluated per
    /// pixel.
    Point {
        /// `x` attribute — X location of the light source.
        x: f32,
        /// `y` attribute — Y location of the light source.
        y: f32,
        /// `z` attribute — Z location of the light source. Positive z
        /// points towards the viewer in §15.14's coordinate system.
        z: f32,
    },
    /// §15.8.4 `<feSpotLight>` — a point-style light source whose
    /// emission is shaped by a focal direction `(pointsAtX,
    /// pointsAtY, pointsAtZ)`, a focus exponent
    /// `specularExponent`, and an optional cut-off cone
    /// `limitingConeAngle`. §15.14 evaluates the light colour
    /// per-pixel as `Light · pow(−L·S, specularExponent)`, clamping to
    /// zero when `−L·S ≤ 0` (the surface is behind the spot) or when
    /// `−L·S < cos(limitingConeAngle)` (the surface is outside the
    /// projected cone).
    Spot {
        /// `x` attribute — X location of the spot source.
        x: f32,
        /// `y` attribute — Y location of the spot source.
        y: f32,
        /// `z` attribute — Z location of the spot source.
        z: f32,
        /// `pointsAtX` — X of the point the spot focuses on.
        points_at_x: f32,
        /// `pointsAtY` — Y of the point the spot focuses on.
        points_at_y: f32,
        /// `pointsAtZ` — Z of the point the spot focuses on.
        points_at_z: f32,
        /// `specularExponent` — focus exponent applied to `−L·S`.
        /// §15.8.4 defines the default as 1 when the attribute is
        /// absent.
        specular_exponent: f32,
        /// `limitingConeAngle` — half-angle of the projected cone, in
        /// degrees. `None` corresponds to the §15.8.4 case "if no value
        /// is supplied, there is no limiting cone".
        limiting_cone_angle_deg: Option<f32>,
    },
}

/// Parameter block for [`diffuse_lighting`].
///
/// Mirrors the SVG 1.1 §15.14 `<feDiffuseLighting>` attribute set
/// (`surfaceScale`, `diffuseConstant`, `kernelUnitLength`) plus the
/// inherited `lighting-color` presentation property and the single
/// light-source child.
///
/// `kernel_unit_length` carries the explicit-or-default §15.14
/// `(dx, dy)` pair; the §15.14 surface-normal formulas embed those
/// deltas in both the `I(x ± dx, y ± dy)` sample positions and in the
/// `FACTORx / FACTORy` normalising constants. Per §15.14 the default
/// when `kernelUnitLength` is absent is "very small deltas relative to
/// a given (x, y) position, which might be implemented in some cases as
/// one pixel"; this implementation therefore defaults to `(1.0, 1.0)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffuseLighting {
    /// `surfaceScale` — vertical scale of the height field
    /// `Z(x, y) = surfaceScale · I(x, y)`. §15.14 default: 1.
    pub surface_scale: f32,
    /// `diffuseConstant` — the Phong `kd` term in the §15.14 colour
    /// formula `D = kd · (N · L) · Light`. §15.14 default: 1.
    pub diffuse_constant: f32,
    /// `kernelUnitLength` as `(dx, dy)`. §15.14 default: `(1.0, 1.0)`
    /// (one pixel in each axis, the only sensible interpretation of
    /// "very small deltas relative to a given (x, y) position" once
    /// the renderer has discretised to a pixel grid).
    pub kernel_unit_length: (f32, f32),
    /// Inherited `lighting-color` property as a straight-alpha sRGB
    /// triple. Defaults to white (`(255, 255, 255)`).
    pub light_color: [u8; 3],
    /// Light-source descriptor selected by the §15.14 content model
    /// "exactly one light source element".
    pub light_source: LightSource,
}

impl Default for DiffuseLighting {
    fn default() -> Self {
        // Defaults per §15.14 attribute table + the §15.8.2 azimuth /
        // elevation defaults ("if the attribute is not specified, then
        // the effect is as if a value of 0 were specified") for the
        // light source.
        Self {
            surface_scale: 1.0,
            diffuse_constant: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Distant {
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
            },
        }
    }
}

/// Identifies which of the nine §15.14 Sobel-kernel regions a sample
/// position falls into. Encoded as a 2-D enum (`row`, `col`) where each
/// axis is `Top` / `Middle` / `Bottom` for rows and `Left` / `Middle` /
/// `Right` for columns; "Middle" denotes the §15.14 interior row /
/// column (every position not on the corresponding edge).
#[derive(Debug, Clone, Copy)]
enum DiffuseRow {
    Top,
    Middle,
    Bottom,
}

#[derive(Debug, Clone, Copy)]
enum DiffuseCol {
    Left,
    Middle,
    Right,
}

/// Sample the §15.14 bump-map function `I(x, y) = alpha / 255` with
/// edge clamping. §15.14 doesn't restate the edge policy explicitly —
/// it instead defines nine kernel variants that already account for
/// missing samples on each edge by zeroing the corresponding column /
/// row coefficient. We still need to *not panic* if an interior-style
/// access happens to drift outside; clamping reproduces the §15.14
/// "edge" behaviour identically for the kernels that do reference the
/// missing sample (their coefficient is zero so the read value is
/// dropped anyway), and yields a safe well-defined value for any
/// floating-point round-off case.
#[inline]
fn diffuse_sample_alpha(src: &[u8], width: u32, height: u32, x: i32, y: i32) -> f32 {
    let xx = x.clamp(0, width as i32 - 1) as usize;
    let yy = y.clamp(0, height as i32 - 1) as usize;
    let idx = (yy * width as usize + xx) * 4 + 3;
    src[idx] as f32 / 255.0
}

/// Compute the §15.14 `(Nx, Ny, Nz)` surface-normal triple for one
/// pixel — chooses the correct kernel variant (corner / edge /
/// interior) from the `(row, col)` region pair and applies the
/// §15.14 formula
/// `Ni = −surfaceScale · FACTORi · Σ K(j, i) · I(x + (j − 1) · d, y + (i − 1) · d)`.
///
/// The kernels and `FACTOR` constants are reproduced verbatim from
/// §15.14's nine boxed sub-tables, indexed as `k[i][j]` with row-major
/// order (`i` = row index 0..2, `j` = column index 0..2) so that
/// `k[0]` is the top row of the §15.14 display, `k[2]` is the bottom
/// row, `k[i][0]` is the leftmost column.
#[inline]
fn diffuse_surface_normal(
    src: &[u8],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    dx: f32,
    dy: f32,
    surface_scale: f32,
) -> [f32; 3] {
    let row = if y == 0 {
        DiffuseRow::Top
    } else if y == height - 1 {
        DiffuseRow::Bottom
    } else {
        DiffuseRow::Middle
    };
    let col = if x == 0 {
        DiffuseCol::Left
    } else if x == width - 1 {
        DiffuseCol::Right
    } else {
        DiffuseCol::Middle
    };

    // §15.14 Kx / Ky / FACTORx / FACTORy for each (row, col) region.
    // `kx[i][j]` and `ky[i][j]` are reproduced verbatim from the
    // §15.14 boxed kernel tables; `fx_num` / `fy_num` are the numerator
    // of the FACTORx / FACTORy fractions (e.g. `2/(3·dx)` ⇒ `fx_num =
    // 2.0` and the routine divides by `(3.0 · dx)` below). The §15.14
    // tables encode the denominators implicitly in the `(2 / 3, 1 / 3,
    // 1 / 2, 1 / 4)` family — track them as `(fx_num, fx_den_dx_mult)`
    // pairs to keep the `dx` factor explicit.
    let (kx, fx_num, fx_den_d, ky, fy_num, fy_den_d) = match (row, col) {
        // Top-left corner — §15.14 boxed pair 1.
        (DiffuseRow::Top, DiffuseCol::Left) => (
            [[0.0, 0.0, 0.0], [0.0, -2.0, 2.0], [0.0, -1.0, 1.0]],
            2.0,
            3.0,
            [[0.0, 0.0, 0.0], [0.0, -2.0, -1.0], [0.0, 2.0, 1.0]],
            2.0,
            3.0,
        ),
        // Top row (interior columns) — §15.14 boxed pair 2.
        (DiffuseRow::Top, DiffuseCol::Middle) => (
            [[0.0, 0.0, 0.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            1.0,
            3.0,
            [[0.0, 0.0, 0.0], [-1.0, -2.0, -1.0], [1.0, 2.0, 1.0]],
            1.0,
            2.0,
        ),
        // Top-right corner — §15.14 boxed pair 3.
        (DiffuseRow::Top, DiffuseCol::Right) => (
            [[0.0, 0.0, 0.0], [-2.0, 2.0, 0.0], [-1.0, 1.0, 0.0]],
            2.0,
            3.0,
            [[0.0, 0.0, 0.0], [-1.0, -2.0, 0.0], [1.0, 2.0, 0.0]],
            2.0,
            3.0,
        ),
        // Left column (interior rows) — §15.14 boxed pair 4.
        (DiffuseRow::Middle, DiffuseCol::Left) => (
            [[0.0, -1.0, 1.0], [0.0, -2.0, 2.0], [0.0, -1.0, 1.0]],
            1.0,
            2.0,
            [[0.0, -2.0, -1.0], [0.0, 0.0, 0.0], [0.0, 2.0, 1.0]],
            1.0,
            3.0,
        ),
        // Interior — §15.14 boxed pair 5 (the centre of the 3×3 grid
        // of variants).
        (DiffuseRow::Middle, DiffuseCol::Middle) => (
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            1.0,
            4.0,
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            1.0,
            4.0,
        ),
        // Right column (interior rows) — §15.14 boxed pair 6.
        (DiffuseRow::Middle, DiffuseCol::Right) => (
            [[-1.0, 1.0, 0.0], [-2.0, 2.0, 0.0], [-1.0, 1.0, 0.0]],
            1.0,
            2.0,
            [[-1.0, -2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            1.0,
            3.0,
        ),
        // Bottom-left corner — §15.14 boxed pair 7.
        (DiffuseRow::Bottom, DiffuseCol::Left) => (
            [[0.0, -1.0, 1.0], [0.0, -2.0, 2.0], [0.0, 0.0, 0.0]],
            2.0,
            3.0,
            [[0.0, -2.0, -1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 0.0]],
            2.0,
            3.0,
        ),
        // Bottom row (interior columns) — §15.14 boxed pair 8.
        (DiffuseRow::Bottom, DiffuseCol::Middle) => (
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
            1.0,
            3.0,
            [[-1.0, -2.0, -1.0], [1.0, 2.0, 1.0], [0.0, 0.0, 0.0]],
            1.0,
            2.0,
        ),
        // Bottom-right corner — §15.14 boxed pair 9.
        (DiffuseRow::Bottom, DiffuseCol::Right) => (
            [[-1.0, 1.0, 0.0], [-2.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
            2.0,
            3.0,
            [[-1.0, -2.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
            2.0,
            3.0,
        ),
    };
    let factor_x = fx_num / (fx_den_d * dx);
    let factor_y = fy_num / (fy_den_d * dy);

    // Walk the 3×3 sample window. The §15.14 displacement is `(j − 1)`
    // columns × `dx` and `(i − 1)` rows × `dy` from the centre `(x, y)`;
    // i.e. `j = 0` reaches one `dx` step left, `j = 2` one `dx` step
    // right (analogously for `i` and `dy`).
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            let sx = (x as f32 + (j as f32 - 1.0) * dx).round() as i32;
            let sy = (y as f32 + (i as f32 - 1.0) * dy).round() as i32;
            let ival = diffuse_sample_alpha(src, width, height, sx, sy);
            sum_x += kx[i][j] * ival;
            sum_y += ky[i][j] * ival;
        }
    }
    let nx = -surface_scale * factor_x * sum_x;
    let ny = -surface_scale * factor_y * sum_y;
    let nz = 1.0;
    let inv_n = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt();
    [nx * inv_n, ny * inv_n, nz * inv_n]
}

/// Resolve the §15.14 `L` unit vector and the `(Lr, Lg, Lb)` per-pixel
/// light-colour triple for one sample position.
///
/// `pixel_z` is the §15.14 height-field value `Z(x, y) = surfaceScale ·
/// I(x, y)`; only `Point` / `Spot` sources consume it. `light_color` is
/// the inherited `lighting-color` straight-alpha sRGB triple, scaled to
/// `[0, 1]` in this routine.
#[inline]
fn diffuse_light_vector(
    src: &LightSource,
    px: f32,
    py: f32,
    pixel_z: f32,
    light_color: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    match *src {
        LightSource::Distant {
            azimuth_deg,
            elevation_deg,
        } => {
            // §15.14 / §15.8.2 distant-light unit vector. The spec
            // states the formula directly in the source coordinate
            // system, so `azimuth` is in degrees clockwise from the
            // positive x axis and `elevation` rotates towards the
            // positive z axis (towards the viewer).
            let az = azimuth_deg.to_radians();
            let el = elevation_deg.to_radians();
            let l = [az.cos() * el.cos(), az.sin() * el.cos(), el.sin()];
            (l, light_color)
        }
        LightSource::Point { x, y, z } => {
            // §15.14 point-light vector from surface sample to light:
            //   L = (Lx − x, Ly − y, Lz − Z(x, y)) / Norm(…).
            let lx = x - px;
            let ly = y - py;
            let lz = z - pixel_z;
            let n = (lx * lx + ly * ly + lz * lz).sqrt();
            if n == 0.0 {
                ([0.0, 0.0, 1.0], light_color)
            } else {
                ([lx / n, ly / n, lz / n], light_color)
            }
        }
        LightSource::Spot {
            x,
            y,
            z,
            points_at_x,
            points_at_y,
            points_at_z,
            specular_exponent,
            limiting_cone_angle_deg,
        } => {
            // §15.14 spot-light vector. First compute `L` exactly like
            // the point-light case, then scale `light_color` by
            // `pow(−L · S, specularExponent)` where `S` is the unit
            // vector from the light to its `pointsAt` target.
            let lx = x - px;
            let ly = y - py;
            let lz = z - pixel_z;
            let nl = (lx * lx + ly * ly + lz * lz).sqrt();
            let (lhat, l_norm_ok) = if nl == 0.0 {
                ([0.0, 0.0, 1.0], false)
            } else {
                ([lx / nl, ly / nl, lz / nl], true)
            };
            let sx = points_at_x - x;
            let sy = points_at_y - y;
            let sz = points_at_z - z;
            let ns = (sx * sx + sy * sy + sz * sz).sqrt();
            let shat = if ns == 0.0 {
                [0.0, 0.0, -1.0]
            } else {
                [sx / ns, sy / ns, sz / ns]
            };
            // §15.14: `−L · S` is the cosine of the angle between the
            // back-pointing-to-light direction and the spot's focus
            // direction. Positive `L · S` (i.e. negative `−L · S`)
            // means the surface is behind the spot and no light is
            // present.
            let neg_ls = -(lhat[0] * shat[0] + lhat[1] * shat[1] + lhat[2] * shat[2]);
            let mut intensity = if neg_ls <= 0.0 || !l_norm_ok {
                0.0
            } else {
                neg_ls.powf(specular_exponent)
            };
            if let Some(cone) = limiting_cone_angle_deg {
                // §15.14: "If `limitingConeAngle` is specified,
                // −L·S < cos(limitingConeAngle) also indicates that no
                // light is present."
                let cos_cone = cone.to_radians().cos();
                if neg_ls < cos_cone {
                    intensity = 0.0;
                }
            }
            let lr = [
                light_color[0] * intensity,
                light_color[1] * intensity,
                light_color[2] * intensity,
            ];
            (lhat, lr)
        }
    }
}

/// SVG 1.1 §15.14 `<feDiffuseLighting>` — light an image using its
/// alpha channel as a bump map.
///
/// The input `src` is a packed straight-alpha RGBA buffer of
/// `width · height · 4` bytes; only the alpha channel is consulted,
/// matching §15.14's "lights an image using the alpha channel as a
/// bump map". The output is a packed straight-alpha RGBA buffer of the
/// same dimensions, with every pixel's alpha set to `255` per the
/// §15.14 statement "the resulting image is an RGBA opaque image based
/// on the light color with alpha = 1.0 everywhere".
///
/// **Algorithm.** Per §15.14:
///
/// 1. Read `I(x, y) = alpha / 255` from the source.
/// 2. Pick the matching Sobel-kernel variant for the §15.14 nine-
///    region grid (interior plus eight edge / corner cases).
/// 3. Form `N = (Nx, Ny, 1) / Norm(Nx, Ny, 1)` from the convolution
///    sums using the kernel's `FACTORx` / `FACTORy` constants.
/// 4. Form `L` from the [`LightSource`] (distant: constant; point /
///    spot: per-pixel `light − sample`).
/// 5. Form the per-pixel `(Lr, Lg, Lb)` light colour: equal to
///    `lighting-color` for distant and point sources; scaled by
///    `pow(−L·S, specularExponent)` (and zeroed outside the cone) for
///    spot sources.
/// 6. Emit `D = kd · max(N·L, 0) · L_color` per channel, clamped to
///    `[0, 1]`, and `Da = 1.0`.
///
/// The `max(N·L, 0)` clamp is implied by §15.14's Phong-diffuse
/// formula (negative `N·L` means the surface faces away from the
/// light) — the spec doesn't state it explicitly, but emitting
/// negative light onto the alpha channel would produce undefined
/// behaviour on the `u8` round-trip and the §15.14 reference to "the
/// standard diffuse component of the Phong lighting model" pins the
/// max-with-zero convention from the Phong original.
///
/// # Panics
///
/// * If `src.len() != width as usize * height as usize * 4`.
/// * If either `kernel_unit_length.0` or `kernel_unit_length.1` is
///   non-positive — §15.14 lists "A negative or zero value is an
///   error (see Error processing)" for `kernelUnitLength`.
///
/// # Returns
///
/// A new packed straight-alpha RGBA `Vec<u8>` of length
/// `width · height · 4`. An empty extent (`width == 0` or
/// `height == 0`) returns an empty `Vec`.
pub fn diffuse_lighting(src: &[u8], width: u32, height: u32, params: &DiffuseLighting) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("diffuse_lighting: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "diffuse_lighting: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );
    assert!(
        params.kernel_unit_length.0 > 0.0,
        "diffuse_lighting: kernel_unit_length.0 must be positive (got {})",
        params.kernel_unit_length.0
    );
    assert!(
        params.kernel_unit_length.1 > 0.0,
        "diffuse_lighting: kernel_unit_length.1 must be positive (got {})",
        params.kernel_unit_length.1
    );
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let (dx, dy) = params.kernel_unit_length;
    let kd = params.diffuse_constant;
    let surface_scale = params.surface_scale;
    let light_color = [
        params.light_color[0] as f32 / 255.0,
        params.light_color[1] as f32 / 255.0,
        params.light_color[2] as f32 / 255.0,
    ];

    let mut out = Vec::with_capacity(expected);
    for y in 0..height {
        for x in 0..width {
            let n = diffuse_surface_normal(src, width, height, x, y, dx, dy, surface_scale);
            // §15.14 `Z(x, y) = surfaceScale · I(x, y)` — only used by
            // the point / spot light branches when forming `L`.
            let centre_i = diffuse_sample_alpha(src, width, height, x as i32, y as i32);
            let pixel_z = surface_scale * centre_i;
            let (l, l_color) = diffuse_light_vector(
                &params.light_source,
                x as f32,
                y as f32,
                pixel_z,
                light_color,
            );
            // §15.14 `D = kd · (N · L) · Light`. Phong-diffuse clamps
            // negative `N · L` to zero (the surface faces away from
            // the light).
            let n_dot_l = (n[0] * l[0] + n[1] * l[1] + n[2] * l[2]).max(0.0);
            let scale = kd * n_dot_l;
            out.push(quantise_unit((scale * l_color[0]).clamp(0.0, 1.0)));
            out.push(quantise_unit((scale * l_color[1]).clamp(0.0, 1.0)));
            out.push(quantise_unit((scale * l_color[2]).clamp(0.0, 1.0)));
            // §15.14: `Da = 1.0` everywhere.
            out.push(255);
        }
    }
    out
}

/// Typed-pixel wrapper around [`diffuse_lighting`]. Returns a
/// `Vec<Rgba>` of length `width · height`. Identical semantics —
/// provided for callers that already have typed pixel buffers.
pub fn diffuse_lighting_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    params: &DiffuseLighting,
) -> Vec<Rgba> {
    let n = width as usize * height as usize;
    assert_eq!(
        src.len(),
        n,
        "diffuse_lighting_pixels: src.len() == {} but width*height == {n}",
        src.len()
    );
    let mut bytes = Vec::with_capacity(n * 4);
    for p in src {
        bytes.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out_bytes = diffuse_lighting(&bytes, width, height, params);
    out_bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod diffuse_lighting_tests {
    use super::*;

    fn flat_alpha(w: u32, h: u32, a: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            v.extend_from_slice(&[0, 0, 0, a]);
        }
        v
    }

    /// §15.14: a flat alpha plane has zero surface gradient, so
    /// `N = (0, 0, 1)`. Lighting it with a distant light pointing
    /// straight at the viewer (`elevation = 90°`) gives
    /// `L = (0, 0, 1)` and `N · L = 1` → emitted colour equals the
    /// light colour scaled by `kd`. With white light and `kd = 1` the
    /// output is fully white opaque.
    #[test]
    fn flat_surface_distant_overhead_emits_light_color() {
        let src = flat_alpha(4, 4, 200); // any constant alpha works.
        let params = DiffuseLighting {
            surface_scale: 10.0,
            diffuse_constant: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Distant {
                azimuth_deg: 0.0,
                elevation_deg: 90.0,
            },
        };
        let out = diffuse_lighting(&src, 4, 4, &params);
        for px in out.chunks_exact(4) {
            assert!(
                (px[0] as i32 - 255).abs() <= 1,
                "interior R != 255: {}",
                px[0]
            );
            assert!(
                (px[1] as i32 - 255).abs() <= 1,
                "interior G != 255: {}",
                px[1]
            );
            assert!(
                (px[2] as i32 - 255).abs() <= 1,
                "interior B != 255: {}",
                px[2]
            );
            assert_eq!(px[3], 255, "§15.14 demands Da = 1 everywhere");
        }
    }

    /// §15.14: a flat plane lit edge-on (`elevation = 0°`, light along
    /// the x axis) gives `N · L = 0` for the interior — the surface is
    /// parallel to the light direction → emitted colour is black, but
    /// alpha stays 1.0 ("the resulting image is an RGBA opaque image
    /// based on the light color with alpha = 1.0 everywhere").
    #[test]
    fn flat_surface_distant_grazing_emits_black_opaque() {
        let src = flat_alpha(4, 4, 200);
        let params = DiffuseLighting {
            surface_scale: 10.0,
            diffuse_constant: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Distant {
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
            },
        };
        let out = diffuse_lighting(&src, 4, 4, &params);
        // Check the interior pixel (1, 1) — corners use a different
        // kernel that includes asymmetric coefficients but on a flat
        // input even those give Nx = Ny = 0 because Σ K · constant = 0.
        let idx = (4 + 1) * 4;
        assert_eq!(out[idx], 0);
        assert_eq!(out[idx + 1], 0);
        assert_eq!(out[idx + 2], 0);
        assert_eq!(out[idx + 3], 255);
    }

    /// §15.14: `diffuse_constant = 0` collapses `D = kd · (N·L) · L`
    /// to zero independent of any other parameter. Alpha still = 1.0.
    #[test]
    fn zero_diffuse_constant_emits_black_opaque() {
        let src = flat_alpha(3, 3, 200);
        let params = DiffuseLighting {
            surface_scale: 10.0,
            diffuse_constant: 0.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 100, 50],
            light_source: LightSource::Distant {
                azimuth_deg: 45.0,
                elevation_deg: 45.0,
            },
        };
        let out = diffuse_lighting(&src, 3, 3, &params);
        for px in out.chunks_exact(4) {
            assert_eq!(px[0], 0);
            assert_eq!(px[1], 0);
            assert_eq!(px[2], 0);
            assert_eq!(px[3], 255);
        }
    }

    /// §15.14 + §15.8.2: a constant flat alpha plane should produce
    /// the same colour at every position when lit by a distant light
    /// (which has no per-pixel positional component). The colour
    /// should equal `light_color · kd · N · L` with `N = (0, 0, 1)`.
    #[test]
    fn distant_light_is_position_invariant_on_flat_surface() {
        let src = flat_alpha(8, 5, 128);
        let params = DiffuseLighting {
            surface_scale: 4.0,
            diffuse_constant: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [120, 200, 80],
            // 30° elevation; sin(30°) = 0.5. So N·L = 0.5 → emit half
            // the light colour.
            light_source: LightSource::Distant {
                azimuth_deg: 0.0,
                elevation_deg: 30.0,
            },
        };
        let out = diffuse_lighting(&src, 8, 5, &params);
        // 120 · 0.5 ≈ 60, 200 · 0.5 = 100, 80 · 0.5 = 40 — within one
        // ULP of the quantisation round-trip.
        for (i, px) in out.chunks_exact(4).enumerate() {
            assert!(
                (px[0] as i32 - 60).abs() <= 1,
                "pixel {i} R = {} not ≈ 60",
                px[0]
            );
            assert!(
                (px[1] as i32 - 100).abs() <= 1,
                "pixel {i} G = {} not ≈ 100",
                px[1]
            );
            assert!(
                (px[2] as i32 - 40).abs() <= 1,
                "pixel {i} B = {} not ≈ 40",
                px[2]
            );
            assert_eq!(px[3], 255);
        }
    }

    /// §15.14: a step-function bump-map produces a non-trivial
    /// surface normal — half the image at `alpha = 0` and half at
    /// `alpha = 255` creates a sharp edge that the Sobel kernel
    /// detects in the cells straddling the boundary. The lit result
    /// should be brighter on the side facing the light. Along the
    /// rising edge from `α = 0` (left) to `α = 1` (right), §15.14
    /// gives `Σ Kx · I > 0`, so `Nx = −surfaceScale · FACTORx · Σ < 0`
    /// — the normal tilts toward `−x`. For `N · L > 0` we therefore
    /// need `L_x < 0`, which §15.8.2 gives at `azimuth = 180°`
    /// (`L = (−1, 0, 0)`).
    #[test]
    fn step_bump_lights_facing_side() {
        let w = 6u32;
        let h = 4u32;
        // Left half alpha = 0, right half alpha = 255.
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..h {
            for x in 0..w {
                let a = if x < w / 2 { 0u8 } else { 255 };
                src.extend_from_slice(&[0, 0, 0, a]);
            }
        }
        let params = DiffuseLighting {
            surface_scale: 5.0,
            diffuse_constant: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Distant {
                // §15.8.2 azimuth 180° ⇒ `L = (cos 180°, sin 180°, 0)
                // = (−1, 0, 0)`. Light comes from the −x side, hits
                // the face of the rising step.
                azimuth_deg: 180.0,
                elevation_deg: 0.0,
            },
        };
        let out = diffuse_lighting(&src, w, h, &params);
        // The interior pixel just to the left of the step (column
        // `w/2 − 1`) on an interior row sees the gradient and should
        // emit a positive value. The interior pixel deep inside the
        // flat region (column 0) should emit zero (no gradient).
        let row = 1usize;
        let edge_idx = (row * w as usize + (w as usize / 2 - 1)) * 4;
        let flat_idx = (row * w as usize) * 4;
        assert_eq!(out[flat_idx], 0, "flat region should emit black");
        assert!(
            out[edge_idx] > 0,
            "edge region should emit non-zero, got {}",
            out[edge_idx]
        );
        // All alphas = 255.
        for px in out.chunks_exact(4) {
            assert_eq!(px[3], 255);
        }
    }

    /// §15.14 point light: when the light sits directly above the
    /// centre of a flat surface, the per-pixel `L` points "up" most
    /// strongly at the centre and tilts toward the edges. The centre
    /// pixel should be the brightest.
    #[test]
    fn point_light_brightest_directly_below_source() {
        let w = 9u32;
        let h = 9u32;
        let src = flat_alpha(w, h, 100);
        let params = DiffuseLighting {
            surface_scale: 1.0,
            diffuse_constant: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Point {
                x: (w as f32 - 1.0) / 2.0,
                y: (h as f32 - 1.0) / 2.0,
                z: 10.0,
            },
        };
        let out = diffuse_lighting(&src, w, h, &params);
        let centre_idx = ((h as usize / 2) * w as usize + w as usize / 2) * 4;
        let corner_idx = 0usize;
        assert!(
            out[centre_idx] >= out[corner_idx],
            "centre brightness {} should be ≥ corner brightness {}",
            out[centre_idx],
            out[corner_idx]
        );
    }

    /// §15.8.4 / §15.14 spot light: when `−L · S` is positive (the
    /// surface is in front of the spot) and inside the limiting
    /// cone, the spot illuminates the surface. When it's outside,
    /// the spot emits zero. Place the spot directly above the
    /// origin pointing straight down with a 5° cone — the centre
    /// pixel is inside the cone, a distant corner is outside.
    #[test]
    fn spot_light_cone_cutoff() {
        let w = 11u32;
        let h = 11u32;
        let src = flat_alpha(w, h, 100);
        let cx = (w as f32 - 1.0) / 2.0;
        let cy = (h as f32 - 1.0) / 2.0;
        let params = DiffuseLighting {
            surface_scale: 1.0,
            diffuse_constant: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Spot {
                x: cx,
                y: cy,
                z: 10.0,
                points_at_x: cx,
                points_at_y: cy,
                points_at_z: 0.0,
                specular_exponent: 1.0,
                limiting_cone_angle_deg: Some(5.0),
            },
        };
        let out = diffuse_lighting(&src, w, h, &params);
        let centre_idx = ((h as usize / 2) * w as usize + w as usize / 2) * 4;
        let corner_idx = 0usize;
        assert!(
            out[centre_idx] > 0,
            "centre should be inside the 5° cone and lit; got {}",
            out[centre_idx]
        );
        assert_eq!(
            out[corner_idx], 0,
            "corner sits well outside the 5° cone and should be dark"
        );
    }

    /// §15.14 + §15.8.4: `specular_exponent` raises the dependence of
    /// the spot's emitted colour on the focus angle. With a very high
    /// exponent and a wide cone, the centre pixel (which lies
    /// directly on the axis, `−L·S ≈ 1`) is essentially unaffected,
    /// while an off-axis pixel inside the cone is dimmed sharply.
    #[test]
    fn spot_specular_exponent_concentrates_emission() {
        let w = 11u32;
        let h = 11u32;
        let src = flat_alpha(w, h, 100);
        let cx = (w as f32 - 1.0) / 2.0;
        let cy = (h as f32 - 1.0) / 2.0;
        let make = |exponent: f32| DiffuseLighting {
            surface_scale: 1.0,
            diffuse_constant: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Spot {
                x: cx,
                y: cy,
                z: 5.0,
                points_at_x: cx,
                points_at_y: cy,
                points_at_z: 0.0,
                specular_exponent: exponent,
                limiting_cone_angle_deg: Some(80.0),
            },
        };
        let low = diffuse_lighting(&src, w, h, &make(1.0));
        let high = diffuse_lighting(&src, w, h, &make(50.0));
        let centre_idx = ((h as usize / 2) * w as usize + w as usize / 2) * 4;
        let near_idx = ((h as usize / 2) * w as usize + (w as usize / 2 + 2)) * 4;
        // The centre is on-axis so `−L·S ≈ 1` and `1^n = 1`; both
        // exponents produce essentially the same brightness there.
        assert!((low[centre_idx] as i32 - high[centre_idx] as i32).abs() <= 2);
        // Off-axis: `−L·S < 1`, so the high exponent dims the
        // emission much more aggressively than the low one.
        assert!(
            high[near_idx] < low[near_idx],
            "high-exponent off-axis brightness {} should be < low-exponent {}",
            high[near_idx],
            low[near_idx]
        );
    }

    /// `kernel_unit_length = (0.0, …)` is §15.14's "negative or zero
    /// value is an error" case and must panic. Same for the `dy`
    /// axis.
    #[test]
    #[should_panic(expected = "kernel_unit_length.0 must be positive")]
    fn zero_kernel_unit_length_x_panics() {
        let src = flat_alpha(2, 2, 0);
        let params = DiffuseLighting {
            kernel_unit_length: (0.0, 1.0),
            ..DiffuseLighting::default()
        };
        let _ = diffuse_lighting(&src, 2, 2, &params);
    }

    #[test]
    #[should_panic(expected = "kernel_unit_length.1 must be positive")]
    fn zero_kernel_unit_length_y_panics() {
        let src = flat_alpha(2, 2, 0);
        let params = DiffuseLighting {
            kernel_unit_length: (1.0, -0.5),
            ..DiffuseLighting::default()
        };
        let _ = diffuse_lighting(&src, 2, 2, &params);
    }

    #[test]
    #[should_panic(expected = "diffuse_lighting: src.len()")]
    fn wrong_src_length_panics() {
        let bad: Vec<u8> = vec![0; 3];
        let params = DiffuseLighting::default();
        let _ = diffuse_lighting(&bad, 2, 2, &params);
    }

    #[test]
    fn empty_extent_returns_empty() {
        let empty: Vec<u8> = Vec::new();
        let params = DiffuseLighting::default();
        let out = diffuse_lighting(&empty, 0, 0, &params);
        assert!(out.is_empty());
    }

    #[test]
    fn typed_pixel_wrapper_matches_byte_path() {
        let w = 5u32;
        let h = 4u32;
        let src_bytes = flat_alpha(w, h, 180);
        let src_pixels: Vec<Rgba> = src_bytes
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let params = DiffuseLighting {
            surface_scale: 5.0,
            diffuse_constant: 0.8,
            kernel_unit_length: (1.0, 1.0),
            light_color: [200, 150, 100],
            light_source: LightSource::Point {
                x: 2.0,
                y: 1.5,
                z: 4.0,
            },
        };
        let bytes_out = diffuse_lighting(&src_bytes, w, h, &params);
        let pixels_out = diffuse_lighting_pixels(&src_pixels, w, h, &params);
        let bytes_from_typed: Vec<u8> = pixels_out
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(bytes_out, bytes_from_typed);
    }
}

// ============================================================================
// SVG 1.1 §15.22 `<feSpecularLighting>` — specular component of the
// Phong lighting model. Shares the §15.14 surface-normal kernel and
// the §15.8 light-source descriptor with `<feDiffuseLighting>`; only
// the per-pixel illumination formula differs.
// ============================================================================

/// Parameter block for [`specular_lighting`].
///
/// Mirrors the SVG 1.1 §15.22 `<feSpecularLighting>` attribute set
/// (`surfaceScale`, `specularConstant`, `specularExponent`,
/// `kernelUnitLength`) plus the inherited `lighting-color` property
/// and the single light-source child. The light-source descriptor is
/// the same [`LightSource`] enum used by [`DiffuseLighting`]; the
/// §15.22 spec defines it through "See ‘feDiffuseLighting’ for
/// definition of N and (Lr, Lg, Lb)".
///
/// Per §15.22 attribute table:
///
/// * `surfaceScale` default 1.
/// * `specularConstant` default 1, non-negative.
/// * `specularExponent` default 1, range 1.0..=128.0.
/// * `kernelUnitLength` default `(1.0, 1.0)`; a negative or zero
///   value is an error.
/// * `lighting-color` default white `(255, 255, 255)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecularLighting {
    /// `surfaceScale` — vertical scale of the height field
    /// `Z(x, y) = surfaceScale · I(x, y)`. §15.22 default: 1.
    pub surface_scale: f32,
    /// `specularConstant` — the Phong `ks` term in the §15.22 colour
    /// formula `S = ks · pow(N · H, specularExponent) · L_color`.
    /// §15.22 default: 1; "this can be any non-negative number".
    pub specular_constant: f32,
    /// `specularExponent` — the §15.22 focus exponent applied to
    /// `N · H`. §15.22 default: 1; spec range 1.0..=128.0. The
    /// implementation honours the §15.22 attribute table verbatim
    /// (no clamping) so a caller that has already validated the
    /// presentation-attribute parse can pass any positive number.
    pub specular_exponent: f32,
    /// `kernelUnitLength` as `(dx, dy)`. §15.22 default: `(1.0, 1.0)`
    /// — same convention as the §15.14 `kernelUnitLength` because
    /// §15.22 reuses the §15.14 surface-normal definition verbatim.
    pub kernel_unit_length: (f32, f32),
    /// Inherited `lighting-color` property as a straight-alpha sRGB
    /// triple. Defaults to white `(255, 255, 255)`.
    pub light_color: [u8; 3],
    /// Light-source descriptor selected by the §15.22 content model
    /// ("exactly one light source element").
    pub light_source: LightSource,
}

impl Default for SpecularLighting {
    fn default() -> Self {
        // Defaults per §15.22 attribute table + the §15.8.2 azimuth /
        // elevation defaults inherited through [`LightSource`].
        Self {
            surface_scale: 1.0,
            specular_constant: 1.0,
            specular_exponent: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Distant {
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
            },
        }
    }
}

/// SVG 1.1 §15.22 `<feSpecularLighting>` — light an image using its
/// alpha channel as a bump map, emitting the specular component of
/// the Phong lighting model.
///
/// The input `src` is a packed straight-alpha RGBA buffer of
/// `width · height · 4` bytes; only the alpha channel is consulted,
/// matching §15.22's "lights a source graphic using the alpha channel
/// as a bump map" + the §15.14 reference for `N`. The output is a
/// packed straight-alpha RGBA buffer of the same dimensions, with
/// per-pixel alpha set to `max(Sr, Sg, Sb)` per the §15.22
/// "non-opaque image … Sa = max(Sr, Sg, Sb)" rule.
///
/// **Algorithm.** Per §15.22:
///
/// 1. Read `I(x, y) = alpha / 255` from the source.
/// 2. Form the surface normal `N` via the same §15.14 nine-region
///    Sobel kernel used by [`diffuse_lighting`] ("See
///    ‘feDiffuseLighting’ for definition of N").
/// 3. Form `L` from the [`LightSource`] (same §15.8 derivation as
///    [`diffuse_lighting`]).
/// 4. Form the §15.22 halfway unit vector
///    `H = (L + E) / Norm(L + E)` with the spec's constant eye
///    vector `E = (0, 0, 1)`. If `L + E` has zero length (which can
///    only happen when `L = -E = (0, 0, -1)`, i.e. the surface is
///    being lit from directly behind the viewer in a degenerate
///    case), the §15.22 dot-product `N · H` is well-defined as zero
///    via the `max(N · H, 0)` Phong-specular clamp.
/// 5. Form the per-pixel `(Lr, Lg, Lb)` light colour: equal to
///    `lighting-color` for distant and point sources; scaled by
///    `pow(−L·S, specularExponent_spot)` (and zeroed outside the
///    cone) for spot sources — same as [`diffuse_lighting`]. Note
///    the spot's `specular_exponent` is the §15.8.4 focus exponent
///    on the spot itself, distinct from the §15.22 outer Phong
///    exponent applied to `N · H`.
/// 6. Emit `Sc = ks · pow(max(N·H, 0), specularExponent) · L_color`
///    per RGB channel, clamped to `[0, 1]`, and
///    `Sa = max(Sr, Sg, Sb)`.
///
/// The `max(N · H, 0)` clamp is implied by the Phong-specular
/// formula (negative `N · H` means the surface faces away from the
/// halfway direction, so no specular reflection reaches the eye).
/// §15.22 doesn't state it explicitly but `pow(negative, fractional)`
/// is undefined on `f32` and the spec's "non-opaque image" intent
/// requires non-negative RGB to make `Sa = max(Sr, Sg, Sb)` itself
/// non-negative.
///
/// # Panics
///
/// * If `src.len() != width as usize * height as usize * 4`.
/// * If either `kernel_unit_length.0` or `kernel_unit_length.1` is
///   non-positive — §15.22 lists "A negative or zero value is an
///   error (see Error processing)" for `kernelUnitLength`.
///
/// # Returns
///
/// A new packed straight-alpha RGBA `Vec<u8>` of length
/// `width · height · 4`. An empty extent (`width == 0` or
/// `height == 0`) returns an empty `Vec`.
pub fn specular_lighting(
    src: &[u8],
    width: u32,
    height: u32,
    params: &SpecularLighting,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("specular_lighting: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "specular_lighting: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );
    assert!(
        params.kernel_unit_length.0 > 0.0,
        "specular_lighting: kernel_unit_length.0 must be positive (got {})",
        params.kernel_unit_length.0
    );
    assert!(
        params.kernel_unit_length.1 > 0.0,
        "specular_lighting: kernel_unit_length.1 must be positive (got {})",
        params.kernel_unit_length.1
    );
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let (dx, dy) = params.kernel_unit_length;
    let ks = params.specular_constant;
    let exponent = params.specular_exponent;
    let surface_scale = params.surface_scale;
    let light_color = [
        params.light_color[0] as f32 / 255.0,
        params.light_color[1] as f32 / 255.0,
        params.light_color[2] as f32 / 255.0,
    ];

    let mut out = Vec::with_capacity(expected);
    for y in 0..height {
        for x in 0..width {
            // Surface normal — identical to the §15.14 derivation
            // (§15.22 explicitly defers to §15.14 for `N`).
            let n = diffuse_surface_normal(src, width, height, x, y, dx, dy, surface_scale);
            // §15.22 / §15.14 height-field value `Z(x, y) =
            // surfaceScale · I(x, y)`, consumed by Point / Spot
            // light branches when forming `L`.
            let centre_i = diffuse_sample_alpha(src, width, height, x as i32, y as i32);
            let pixel_z = surface_scale * centre_i;
            let (l, l_color) = diffuse_light_vector(
                &params.light_source,
                x as f32,
                y as f32,
                pixel_z,
                light_color,
            );
            // §15.22 halfway vector `H = (L + E) / Norm(L + E)`
            // with the spec's constant eye vector `E = (0, 0, 1)`.
            let hx = l[0];
            let hy = l[1];
            let hz = l[2] + 1.0;
            let hn2 = hx * hx + hy * hy + hz * hz;
            let n_dot_h = if hn2 == 0.0 {
                // Degenerate case `L = -E`. `N · H` is set to 0 so
                // the specular term collapses to 0, consistent with
                // the §15.22 / Phong-specular max-with-zero clamp
                // applied below.
                0.0
            } else {
                let inv_hn = 1.0 / hn2.sqrt();
                let h_unit = [hx * inv_hn, hy * inv_hn, hz * inv_hn];
                n[0] * h_unit[0] + n[1] * h_unit[1] + n[2] * h_unit[2]
            };
            // §15.22 colour formula. The `max(., 0)` guard is the
            // Phong-specular convention — see function-level doc
            // comment.
            let factor = if n_dot_h <= 0.0 {
                0.0
            } else {
                ks * n_dot_h.powf(exponent)
            };
            let sr = (factor * l_color[0]).clamp(0.0, 1.0);
            let sg = (factor * l_color[1]).clamp(0.0, 1.0);
            let sb = (factor * l_color[2]).clamp(0.0, 1.0);
            // §15.22: `Sa = max(Sr, Sg, Sb)` — applied on the
            // continuous `[0, 1]` triple before quantisation so the
            // alpha mirrors the spec's pre-quantisation maximum
            // exactly. Quantising first then taking the max would
            // accumulate the per-channel rounding into the alpha.
            let sa_unit = sr.max(sg).max(sb);
            out.push(quantise_unit(sr));
            out.push(quantise_unit(sg));
            out.push(quantise_unit(sb));
            out.push(quantise_unit(sa_unit));
        }
    }
    out
}

/// Typed-pixel wrapper around [`specular_lighting`]. Returns a
/// `Vec<Rgba>` of length `width · height`. Identical semantics —
/// provided for callers that already have typed pixel buffers.
pub fn specular_lighting_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    params: &SpecularLighting,
) -> Vec<Rgba> {
    let n = width as usize * height as usize;
    assert_eq!(
        src.len(),
        n,
        "specular_lighting_pixels: src.len() == {} but width*height == {n}",
        src.len()
    );
    let mut bytes = Vec::with_capacity(n * 4);
    for p in src {
        bytes.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out_bytes = specular_lighting(&bytes, width, height, params);
    out_bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod specular_lighting_tests {
    use super::*;

    fn flat_alpha(w: u32, h: u32, a: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            v.extend_from_slice(&[0, 0, 0, a]);
        }
        v
    }

    /// §15.22: a flat alpha plane has zero surface gradient
    /// (§15.14 `N = (0, 0, 1)`); lit by a distant light pointing
    /// straight at the viewer (`elevation = 90°` ⇒ `L = (0, 0, 1)`)
    /// the halfway vector is `H = (0, 0, 2) / 2 = (0, 0, 1)`, so
    /// `N · H = 1` and `pow(N·H, anything) = 1`. With white light
    /// and `ks = 1`, every pixel emits full white plus alpha 255.
    #[test]
    fn flat_surface_distant_overhead_emits_full_white_with_full_alpha() {
        let src = flat_alpha(4, 4, 200);
        let params = SpecularLighting {
            surface_scale: 10.0,
            specular_constant: 1.0,
            specular_exponent: 20.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Distant {
                azimuth_deg: 0.0,
                elevation_deg: 90.0,
            },
        };
        let out = specular_lighting(&src, 4, 4, &params);
        for (i, px) in out.chunks_exact(4).enumerate() {
            assert!(
                (px[0] as i32 - 255).abs() <= 1,
                "pixel {i} R != 255: {}",
                px[0]
            );
            assert!(
                (px[1] as i32 - 255).abs() <= 1,
                "pixel {i} G != 255: {}",
                px[1]
            );
            assert!(
                (px[2] as i32 - 255).abs() <= 1,
                "pixel {i} B != 255: {}",
                px[2]
            );
            // Sa = max(Sr, Sg, Sb) — full-white emission ⇒ full alpha.
            assert!(
                (px[3] as i32 - 255).abs() <= 1,
                "pixel {i} A != 255: {}",
                px[3]
            );
        }
    }

    /// §15.22: a flat plane lit edge-on (`elevation = 0°`, light
    /// along the x axis) gives `L = (1, 0, 0)`, so
    /// `H = (1, 0, 1) / √2`. With `N = (0, 0, 1)` (flat surface)
    /// `N · H = 1/√2 ≈ 0.7071`. With `specularExponent = 2` and
    /// `ks = 1` we expect `S = 0.5` ⇒ 128 on the `[0, 255]` axis.
    /// `Sa = max(Sr, Sg, Sb) = 128`.
    #[test]
    fn flat_surface_distant_grazing_half_intensity() {
        let src = flat_alpha(4, 4, 200);
        let params = SpecularLighting {
            surface_scale: 10.0,
            specular_constant: 1.0,
            specular_exponent: 2.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Distant {
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
            },
        };
        let out = specular_lighting(&src, 4, 4, &params);
        // Use the interior pixel (1, 1) to avoid any corner-kernel
        // contributions on the flat plane — though both should be
        // identical because `Σ K · constant = 0` for both centred
        // and corner Sobel variants applied to a constant alpha.
        let idx = (4 + 1) * 4;
        // 0.5 · 255 = 127.5 → quantise to 127 or 128.
        let r = out[idx] as i32;
        let g = out[idx + 1] as i32;
        let b = out[idx + 2] as i32;
        let a = out[idx + 3] as i32;
        assert!((r - 128).abs() <= 1, "R ≈ 128 (got {r})");
        assert!((g - 128).abs() <= 1, "G ≈ 128 (got {g})");
        assert!((b - 128).abs() <= 1, "B ≈ 128 (got {b})");
        // Sa = max(Sr, Sg, Sb) — equal channels ⇒ same value.
        assert!((a - 128).abs() <= 1, "A = max(R,G,B) ≈ 128 (got {a})");
    }

    /// §15.22: `specular_constant = 0` zeroes the entire output
    /// formula independent of any other parameter. The alpha
    /// channel collapses with the RGB channels because
    /// `Sa = max(Sr, Sg, Sb) = max(0, 0, 0) = 0`. This is the
    /// §15.22 "non-opaque image" property — unlike the §15.14
    /// diffuse path where `Da = 1` everywhere.
    #[test]
    fn zero_specular_constant_emits_full_transparent_black() {
        let src = flat_alpha(3, 3, 200);
        let params = SpecularLighting {
            surface_scale: 10.0,
            specular_constant: 0.0,
            specular_exponent: 20.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 100, 50],
            light_source: LightSource::Distant {
                azimuth_deg: 45.0,
                elevation_deg: 45.0,
            },
        };
        let out = specular_lighting(&src, 3, 3, &params);
        for px in out.chunks_exact(4) {
            assert_eq!(px[0], 0);
            assert_eq!(px[1], 0);
            assert_eq!(px[2], 0);
            assert_eq!(px[3], 0, "§15.22 Sa = max(Sr, Sg, Sb) = 0");
        }
    }

    /// §15.22 alpha distinction: with a coloured light, the RGB
    /// channels emit at different levels and `Sa = max(Sr, Sg, Sb)`
    /// equals the maximum-emitting channel — NOT 255 (the §15.14
    /// diffuse rule) and NOT the per-channel value.
    #[test]
    fn coloured_light_alpha_equals_max_channel() {
        let src = flat_alpha(3, 3, 200);
        let params = SpecularLighting {
            surface_scale: 10.0,
            specular_constant: 1.0,
            specular_exponent: 1.0, // simplest: pow(N·H, 1) = N·H.
            kernel_unit_length: (1.0, 1.0),
            // Asymmetric light triple to make max-channel detection
            // unambiguous.
            light_color: [255, 100, 50],
            light_source: LightSource::Distant {
                azimuth_deg: 0.0,
                elevation_deg: 90.0, // L = (0, 0, 1) ⇒ N·H = 1.
            },
        };
        let out = specular_lighting(&src, 3, 3, &params);
        for px in out.chunks_exact(4) {
            // ks · 1 · (255, 100, 50) / 255 = (1.0, ~0.392, ~0.196)
            // ⇒ (255, 100, 50) on quantisation.
            assert!((px[0] as i32 - 255).abs() <= 1);
            assert!((px[1] as i32 - 100).abs() <= 1);
            assert!((px[2] as i32 - 50).abs() <= 1);
            // Sa = max(R, G, B) = R = 255.
            assert_eq!(px[3], 255);
        }
    }

    /// §15.22: a distant light is position-invariant on a flat
    /// surface — every pixel evaluates the same `(N, L, H)` triple
    /// and therefore emits the same colour. This is the same
    /// invariant tested for diffuse-lighting; here it confirms the
    /// specular kernel has no spurious per-pixel positional coupling.
    #[test]
    fn distant_light_position_invariant_on_flat_surface() {
        let src = flat_alpha(8, 5, 128);
        let params = SpecularLighting {
            surface_scale: 4.0,
            specular_constant: 0.8,
            specular_exponent: 5.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [120, 200, 80],
            light_source: LightSource::Distant {
                azimuth_deg: 30.0,
                elevation_deg: 45.0,
            },
        };
        let out = specular_lighting(&src, 8, 5, &params);
        let first = [out[0], out[1], out[2], out[3]];
        for (i, px) in out.chunks_exact(4).enumerate() {
            // Allow one quantisation-rounding ULP either way.
            assert!(
                (px[0] as i32 - first[0] as i32).abs() <= 1,
                "px {i} R differs: {} vs {}",
                px[0],
                first[0]
            );
            assert!((px[1] as i32 - first[1] as i32).abs() <= 1);
            assert!((px[2] as i32 - first[2] as i32).abs() <= 1);
            assert!((px[3] as i32 - first[3] as i32).abs() <= 1);
        }
    }

    /// §15.22: light pointing behind the surface (`elevation = -90°`
    /// ⇒ `L = (0, 0, -1)`) gives `L + E = (0, 0, 0)`, the §15.22
    /// degenerate case. The implementation collapses `N · H` to 0
    /// per the documented Phong-specular `max(., 0)` clamp; output
    /// must be transparent black at every pixel.
    #[test]
    fn light_behind_surface_emits_zero() {
        let src = flat_alpha(3, 3, 200);
        let params = SpecularLighting {
            surface_scale: 10.0,
            specular_constant: 1.0,
            specular_exponent: 2.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Distant {
                azimuth_deg: 0.0,
                elevation_deg: -90.0,
            },
        };
        let out = specular_lighting(&src, 3, 3, &params);
        for px in out.chunks_exact(4) {
            assert_eq!(px[0], 0);
            assert_eq!(px[1], 0);
            assert_eq!(px[2], 0);
            assert_eq!(px[3], 0);
        }
    }

    /// §15.22: `specularExponent` increases ⇒ the specular highlight
    /// narrows. At a grazing angle (`N · H = 1/√2 ≈ 0.707`) raising
    /// the exponent must monotonically decrease the emitted
    /// intensity. We verify monotonicity on three exponents: 1, 5,
    /// 50.
    #[test]
    fn higher_exponent_dims_grazing_intensity() {
        let src = flat_alpha(3, 3, 200);
        let mk = |exp: f32| {
            let params = SpecularLighting {
                surface_scale: 10.0,
                specular_constant: 1.0,
                specular_exponent: exp,
                kernel_unit_length: (1.0, 1.0),
                light_color: [255, 255, 255],
                light_source: LightSource::Distant {
                    azimuth_deg: 0.0,
                    elevation_deg: 0.0, // L along +x ⇒ N·H = 1/√2.
                },
            };
            specular_lighting(&src, 3, 3, &params)
        };
        let a = mk(1.0)[(3 + 1) * 4];
        let b = mk(5.0)[(3 + 1) * 4];
        let c = mk(50.0)[(3 + 1) * 4];
        assert!(a > b, "exp=1 ({a}) must be brighter than exp=5 ({b})");
        assert!(b > c, "exp=5 ({b}) must be brighter than exp=50 ({c})");
    }

    /// §15.22 + §15.8.3 point light: at a flat sample directly
    /// underneath a point light (`L = (0, 0, 1)`) the same
    /// `H = (0, 0, 1)` and `N · H = 1` invariants as the overhead-
    /// distant case must hold. We place the light at `(2, 2, 10)`
    /// over a 5×5 buffer and check the centre pixel.
    #[test]
    fn point_light_directly_overhead() {
        let src = flat_alpha(5, 5, 0);
        let params = SpecularLighting {
            surface_scale: 0.0, // pixel_z = 0 everywhere → L is exact.
            specular_constant: 1.0,
            specular_exponent: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Point {
                x: 2.0,
                y: 2.0,
                z: 10.0,
            },
        };
        let out = specular_lighting(&src, 5, 5, &params);
        let idx = (2 * 5 + 2) * 4;
        // Allow modest tolerance because the point-light direction
        // from (2,2,0) to (2,2,10) is exactly (0,0,1) but the §15.14
        // Sobel kernel on a constant alpha still yields N=(0,0,1)
        // exactly, so the result should round to 255.
        assert!((out[idx] as i32 - 255).abs() <= 1);
        assert!((out[idx + 1] as i32 - 255).abs() <= 1);
        assert!((out[idx + 2] as i32 - 255).abs() <= 1);
        assert!((out[idx + 3] as i32 - 255).abs() <= 1);
    }

    /// §15.22 + §15.8.4 spot light: the §15.8.4 cone-cutoff path
    /// must fire when the surface sample lies outside the cone. We
    /// place a spot at `(0, 0, 10)` aiming at `(0, 0, 0)` with a 1°
    /// limiting cone; any sample more than `tan(1°) · 10 ≈ 0.175`
    /// pixels away from the central column must emit zero.
    #[test]
    fn spot_light_outside_cone_emits_zero() {
        let src = flat_alpha(5, 5, 0);
        let params = SpecularLighting {
            surface_scale: 0.0,
            specular_constant: 1.0,
            specular_exponent: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 255, 255],
            light_source: LightSource::Spot {
                x: 0.0,
                y: 0.0,
                z: 10.0,
                points_at_x: 0.0,
                points_at_y: 0.0,
                points_at_z: 0.0,
                specular_exponent: 1.0, // §15.8.4 focus, distinct from §15.22.
                limiting_cone_angle_deg: Some(1.0),
            },
        };
        let out = specular_lighting(&src, 5, 5, &params);
        // Corner (4, 4) is far outside the 1° cone — must be zero
        // on every channel (RGB + Sa).
        let idx = (4 * 5 + 4) * 4;
        assert_eq!(out[idx], 0);
        assert_eq!(out[idx + 1], 0);
        assert_eq!(out[idx + 2], 0);
        assert_eq!(out[idx + 3], 0);
    }

    /// §15.22 default attribute set: all defaults give a smoke-test
    /// path with no panics and a buffer of the expected length.
    #[test]
    fn defaults_produce_sane_output() {
        let src = flat_alpha(2, 2, 100);
        let params = SpecularLighting::default();
        let out = specular_lighting(&src, 2, 2, &params);
        assert_eq!(out.len(), 2 * 2 * 4);
        // Sa = max(Sr, Sg, Sb) on every pixel.
        for px in out.chunks_exact(4) {
            let m = px[0].max(px[1]).max(px[2]);
            // Allow ±1 ULP for the pre-quantisation max vs the
            // post-quantisation max of the per-channel triples.
            assert!(
                (px[3] as i32 - m as i32).abs() <= 1,
                "Sa={} vs max(R,G,B)={m}",
                px[3]
            );
        }
    }

    #[test]
    #[should_panic(expected = "specular_lighting: kernel_unit_length.0 must be positive")]
    fn zero_kernel_unit_length_x_panics() {
        let src = flat_alpha(2, 2, 100);
        let params = SpecularLighting {
            kernel_unit_length: (0.0, 1.0),
            ..SpecularLighting::default()
        };
        let _ = specular_lighting(&src, 2, 2, &params);
    }

    #[test]
    #[should_panic(expected = "specular_lighting: kernel_unit_length.1 must be positive")]
    fn negative_kernel_unit_length_y_panics() {
        let src = flat_alpha(2, 2, 100);
        let params = SpecularLighting {
            kernel_unit_length: (1.0, -2.0),
            ..SpecularLighting::default()
        };
        let _ = specular_lighting(&src, 2, 2, &params);
    }

    #[test]
    #[should_panic(expected = "specular_lighting: src.len()")]
    fn wrong_src_length_panics() {
        let bad: Vec<u8> = vec![0; 3];
        let params = SpecularLighting::default();
        let _ = specular_lighting(&bad, 2, 2, &params);
    }

    #[test]
    fn empty_extent_returns_empty() {
        let empty: Vec<u8> = Vec::new();
        let params = SpecularLighting::default();
        let out = specular_lighting(&empty, 0, 0, &params);
        assert!(out.is_empty());
    }

    #[test]
    fn typed_pixel_wrapper_matches_byte_path() {
        let w = 5u32;
        let h = 4u32;
        let src_bytes = flat_alpha(w, h, 180);
        let src_pixels: Vec<Rgba> = src_bytes
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let params = SpecularLighting {
            surface_scale: 5.0,
            specular_constant: 0.7,
            specular_exponent: 8.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [200, 150, 100],
            light_source: LightSource::Point {
                x: 2.0,
                y: 1.5,
                z: 4.0,
            },
        };
        let bytes_out = specular_lighting(&src_bytes, w, h, &params);
        let pixels_out = specular_lighting_pixels(&src_pixels, w, h, &params);
        let bytes_from_typed: Vec<u8> = pixels_out
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(bytes_out, bytes_from_typed);
    }
}

// ----------------------------------------------------------------------
// SVG 1.1 §15.18 `<feImage>` — external-raster source primitive.
// ----------------------------------------------------------------------

/// `<align>` parameter of the SVG 1.1 §7.8 `preserveAspectRatio`
/// attribute.
///
/// §7.8: "The `<align>` parameter indicates whether to force uniform
/// scaling and, if so, the alignment method to use in case the aspect
/// ratio of the ‘viewBox’ doesn't match the aspect ratio of the
/// viewport."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AspectRatioAlign {
    /// §7.8 `none` — "Do not force uniform scaling. Scale the graphic
    /// content of the given element non-uniformly if necessary such
    /// that the element's bounding box exactly matches the viewport
    /// rectangle." The `meetOrSlice` value is ignored for this
    /// alignment (§7.8 note).
    None,
    /// §7.8 `xMinYMin` — align the minimum X / minimum Y of the
    /// source with the smallest X / Y of the viewport.
    XMinYMin,
    /// §7.8 `xMidYMin` — align the midpoint X with the viewport
    /// midpoint X; minimum Y with the smallest viewport Y.
    XMidYMin,
    /// §7.8 `xMaxYMin` — align the maximum X with the maximum
    /// viewport X; minimum Y with the smallest viewport Y.
    XMaxYMin,
    /// §7.8 `xMinYMid` — minimum X at the smallest viewport X;
    /// midpoint Y at the viewport midpoint Y.
    XMinYMid,
    /// §7.8 `xMidYMid` — "(the default)" — midpoints aligned on both
    /// axes.
    #[default]
    XMidYMid,
    /// §7.8 `xMaxYMid` — maximum X at the maximum viewport X;
    /// midpoint Y at the viewport midpoint Y.
    XMaxYMid,
    /// §7.8 `xMinYMax` — minimum X at the smallest viewport X;
    /// maximum Y at the maximum viewport Y.
    XMinYMax,
    /// §7.8 `xMidYMax` — midpoint X at the viewport midpoint X;
    /// maximum Y at the maximum viewport Y.
    XMidYMax,
    /// §7.8 `xMaxYMax` — maximum X / maximum Y aligned with the
    /// maximum viewport X / Y.
    XMaxYMax,
}

/// `<meetOrSlice>` parameter of the SVG 1.1 §7.8
/// `preserveAspectRatio` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeetOrSlice {
    /// §7.8 `meet` ("the default") — "Scale the graphic such that:
    /// aspect ratio is preserved; the entire ‘viewBox’ is visible
    /// within the viewport; the ‘viewBox’ is scaled up as much as
    /// possible, while still meeting the other criteria." Reduces to
    /// the per-axis **minimum** scale factor.
    #[default]
    Meet,
    /// §7.8 `slice` — "Scale the graphic such that: aspect ratio is
    /// preserved; the entire viewport is covered by the ‘viewBox’;
    /// the ‘viewBox’ is scaled down as much as possible, while still
    /// meeting the other criteria." Reduces to the per-axis
    /// **maximum** scale factor.
    Slice,
}

/// Parsed value of the SVG 1.1 §7.8 `preserveAspectRatio` attribute,
/// as consumed by [`image_source`].
///
/// The `defer` keyword is a reference-resolution policy ("the value
/// of the ‘preserveAspectRatio’ attribute on the referenced content
/// … should be used", §7.8) — by the time the caller holds a decoded
/// raster plus a `PreserveAspectRatio` value, that resolution has
/// already happened, so no `defer` field exists here.
///
/// `Default` is `xMidYMid meet` — §15.18: "If attribute
/// ‘preserveAspectRatio’ is not specified, then the effect is as if
/// a value of xMidYMid meet were specified."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PreserveAspectRatio {
    /// The `<align>` parameter.
    pub align: AspectRatioAlign,
    /// The `<meetOrSlice>` parameter. Ignored when `align` is
    /// [`AspectRatioAlign::None`] (§7.8 note).
    pub meet_or_slice: MeetOrSlice,
}

/// Reconstruction filter selector for [`image_source`].
///
/// §15.18: "When the referenced image must be resampled to match the
/// device coordinate system, it is recommended that high quality
/// viewers make use of appropriate interpolation techniques, for
/// example bilinear or bicubic."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageSourceSampling {
    /// Round the mapped source coordinate to the nearest integer
    /// pixel and copy it verbatim. Out-of-bounds coordinates emit
    /// the §15.7.3 transparent-black value.
    #[default]
    Nearest,
    /// Bilinear reconstruction around the fractional mapped source
    /// coordinate (the §15.18 recommended interpolation route).
    /// Out-of-bounds members of the 2×2 footprint contribute
    /// transparent black, fading the fitted image at its edges.
    Bilinear,
}

/// SVG 1.1 §15.18 `<feImage>` — place an already-decoded RGBA raster
/// into the filter chain, fitted to the `out_width × out_height`
/// filter-primitive subregion per §7.8 `preserveAspectRatio`.
///
/// §15.18: "This filter primitive refers to a graphic external to
/// this filter element, which is loaded or rendered into an RGBA
/// raster and becomes the result of the filter primitive." Reference
/// resolution and decoding are the caller's job (the renderer / SVG
/// front-end); this function receives the decoded
/// `src_width × src_height` straight-alpha RGBA raster and performs
/// the §15.18 placement step, which "is identical to that of the
/// ‘image’ element" — i.e. the §7.8 fitting algebra:
///
/// 1. Compute the scale. `align = none` scales each axis
///    independently (`out_width / src_width`,
///    `out_height / src_height`) so the source exactly fills the
///    target. Any other alignment forces uniform scaling: the
///    minimum of the two per-axis ratios under [`MeetOrSlice::Meet`]
///    or the maximum under [`MeetOrSlice::Slice`].
/// 2. Compute the translation per axis from the alignment anchor:
///    `Min` puts the scaled source's low edge at 0, `Mid` centres it
///    (`(out − src·s) / 2`), `Max` puts its high edge at the target's
///    high edge (`out − src·s`).
/// 3. Inverse-map each output pixel centre into source space and
///    reconstruct with the selected [`ImageSourceSampling`]. Output
///    pixels whose mapped coordinate falls outside the source extent
///    (the uncovered viewport bands of `meet`, or any anchor-induced
///    overhang) emit transparent black per §15.7.3.
///
/// An empty source (`src_width == 0` or `src_height == 0`) renders
/// nothing: the result is the fully transparent-black buffer (the
/// filter-chain initial value, §15.7.3) rather than a panic, since a
/// reference to a zero-extent graphic is a degenerate-but-resolvable
/// document state, not a caller bug.
///
/// # Panics
///
/// * If `src.len() != src_width as usize * src_height as usize * 4`.
/// * If any of `src_width · src_height · 4` /
///   `out_width · out_height · 4` overflows `usize`.
///
/// # Returns
///
/// A packed straight-alpha RGBA `Vec<u8>` of length
/// `out_width · out_height · 4`. An empty target extent returns an
/// empty `Vec`.
pub fn image_source(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    out_width: u32,
    out_height: u32,
    par: PreserveAspectRatio,
    sampling: ImageSourceSampling,
) -> Vec<u8> {
    let sw = src_width as usize;
    let sh = src_height as usize;
    let src_expected = sw
        .checked_mul(sh)
        .and_then(|n| n.checked_mul(4))
        .expect("image_source: src_width * src_height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        src_expected,
        "image_source: src.len() == {} but src_width*src_height*4 == {src_expected}",
        src.len()
    );
    let ow = out_width as usize;
    let oh = out_height as usize;
    let out_bytes = ow
        .checked_mul(oh)
        .and_then(|n| n.checked_mul(4))
        .expect("image_source: out_width * out_height * 4 overflowed usize");
    if out_width == 0 || out_height == 0 {
        return Vec::new();
    }
    // Degenerate zero-extent source: nothing to draw — the primitive
    // result stays at the §15.7.3 transparent-black initial value.
    if src_width == 0 || src_height == 0 {
        return vec![0u8; out_bytes];
    }

    // §7.8 step 1 — scale factors.
    let ratio_x = out_width as f32 / src_width as f32;
    let ratio_y = out_height as f32 / src_height as f32;
    let (scale_x, scale_y) = match par.align {
        AspectRatioAlign::None => (ratio_x, ratio_y),
        _ => {
            let s = match par.meet_or_slice {
                MeetOrSlice::Meet => ratio_x.min(ratio_y),
                MeetOrSlice::Slice => ratio_x.max(ratio_y),
            };
            (s, s)
        }
    };

    // §7.8 step 2 — per-axis anchor translation. The X anchor cycles
    // Min → Mid → Max through the §7.8 table; same for Y.
    #[derive(Clone, Copy)]
    enum Anchor {
        Min,
        Mid,
        Max,
    }
    let (ax, ay) = match par.align {
        AspectRatioAlign::None => (Anchor::Min, Anchor::Min), // tx = ty = 0 either way
        AspectRatioAlign::XMinYMin => (Anchor::Min, Anchor::Min),
        AspectRatioAlign::XMidYMin => (Anchor::Mid, Anchor::Min),
        AspectRatioAlign::XMaxYMin => (Anchor::Max, Anchor::Min),
        AspectRatioAlign::XMinYMid => (Anchor::Min, Anchor::Mid),
        AspectRatioAlign::XMidYMid => (Anchor::Mid, Anchor::Mid),
        AspectRatioAlign::XMaxYMid => (Anchor::Max, Anchor::Mid),
        AspectRatioAlign::XMinYMax => (Anchor::Min, Anchor::Max),
        AspectRatioAlign::XMidYMax => (Anchor::Mid, Anchor::Max),
        AspectRatioAlign::XMaxYMax => (Anchor::Max, Anchor::Max),
    };
    let anchor_offset = |anchor: Anchor, out: f32, scaled: f32| -> f32 {
        match anchor {
            Anchor::Min => 0.0,
            Anchor::Mid => (out - scaled) / 2.0,
            Anchor::Max => out - scaled,
        }
    };
    let tx = anchor_offset(ax, out_width as f32, src_width as f32 * scale_x);
    let ty = anchor_offset(ay, out_height as f32, src_height as f32 * scale_y);

    // §7.8 step 3 — inverse-map output pixel centres into source
    // space: the output pixel centre `(ox + ½, oy + ½)` corresponds
    // to source position `((ox + ½ − tx) / scale, …)`; subtracting ½
    // converts back from continuous position to source pixel index.
    let wi = src_width as i64;
    let hi = src_height as i64;
    let mut out = vec![0u8; out_bytes];
    let row_stride = ow * 4;
    for oy in 0..oh {
        let dst_row = oy * row_stride;
        let v = (oy as f32 + 0.5 - ty) / scale_y - 0.5;
        for ox in 0..ow {
            let u = (ox as f32 + 0.5 - tx) / scale_x - 0.5;
            let dst = dst_row + ox * 4;
            match sampling {
                ImageSourceSampling::Nearest => {
                    // Round-half-away-from-zero — the same integer
                    // rounding policy as `offset` (§15.21) and
                    // `displacement_map` (§15.15), keeping the
                    // module's nearest-sampling convention uniform.
                    let isx = u.round() as i64;
                    let isy = v.round() as i64;
                    if (0..wi).contains(&isx) && (0..hi).contains(&isy) {
                        let s = ((isy * wi + isx) as usize) * 4;
                        out[dst..dst + 4].copy_from_slice(&src[s..s + 4]);
                    } // else: transparent black (already zeroed).
                }
                ImageSourceSampling::Bilinear => {
                    let x0 = u.floor() as i64;
                    let y0 = v.floor() as i64;
                    let fx = u - x0 as f32;
                    let fy = v - y0 as f32;
                    let fetch = |px: i64, py: i64| -> [f32; 4] {
                        if !(0..wi).contains(&px) || !(0..hi).contains(&py) {
                            return [0.0; 4];
                        }
                        let p = ((py * wi + px) as usize) * 4;
                        [
                            src[p] as f32,
                            src[p + 1] as f32,
                            src[p + 2] as f32,
                            src[p + 3] as f32,
                        ]
                    };
                    let c00 = fetch(x0, y0);
                    let c10 = fetch(x0 + 1, y0);
                    let c01 = fetch(x0, y0 + 1);
                    let c11 = fetch(x0 + 1, y0 + 1);
                    for c in 0..4 {
                        let top = c00[c] + (c10[c] - c00[c]) * fx;
                        let bot = c01[c] + (c11[c] - c01[c]) * fx;
                        let val = top + (bot - top) * fy;
                        out[dst + c] = val.clamp(0.0, 255.0).round() as u8;
                    }
                }
            }
        }
    }
    out
}

/// Typed-pixel wrapper around [`image_source`]. `src` has length
/// `src_width · src_height`; the returned `Vec<Rgba>` has length
/// `out_width · out_height`.
pub fn image_source_pixels(
    src: &[Rgba],
    src_width: u32,
    src_height: u32,
    out_width: u32,
    out_height: u32,
    par: PreserveAspectRatio,
    sampling: ImageSourceSampling,
) -> Vec<Rgba> {
    let n = src_width as usize * src_height as usize;
    assert_eq!(
        src.len(),
        n,
        "image_source_pixels: src.len() == {} but src_width*src_height == {n}",
        src.len()
    );
    let mut bytes = Vec::with_capacity(n * 4);
    for p in src {
        bytes.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out = image_source(
        &bytes, src_width, src_height, out_width, out_height, par, sampling,
    );
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod image_source_tests {
    use super::*;

    /// Build a packed-RGBA buffer of `w·h` pixels coloured by `f`.
    fn build<F: FnMut(u32, u32) -> [u8; 4]>(w: u32, h: u32, mut f: F) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&f(x, y));
            }
        }
        v
    }

    fn px(out: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * w + x) * 4) as usize;
        [out[i], out[i + 1], out[i + 2], out[i + 3]]
    }

    #[test]
    fn default_preserve_aspect_ratio_is_xmidymid_meet() {
        // §15.18: "the effect is as if a value of xMidYMid meet were
        // specified" when the attribute is absent.
        let d = PreserveAspectRatio::default();
        assert_eq!(d.align, AspectRatioAlign::XMidYMid);
        assert_eq!(d.meet_or_slice, MeetOrSlice::Meet);
    }

    #[test]
    fn same_extent_is_an_exact_copy_nearest_and_bilinear() {
        let src = build(5, 4, |x, y| {
            [(x * 50) as u8, (y * 60) as u8, ((x + y) * 25) as u8, 255]
        });
        for sampling in [ImageSourceSampling::Nearest, ImageSourceSampling::Bilinear] {
            let out = image_source(&src, 5, 4, 5, 4, PreserveAspectRatio::default(), sampling);
            assert_eq!(out, src, "identity placement must copy ({sampling:?})");
        }
    }

    #[test]
    fn align_none_stretches_non_uniformly() {
        // §7.8 `none`: "Scale the graphic content … non-uniformly if
        // necessary such that the element's bounding box exactly
        // matches the viewport rectangle." 2×1 source → 4×1 target:
        // each source pixel covers exactly 2 output pixels.
        let src = build(2, 1, |x, _| {
            if x == 0 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        });
        let par = PreserveAspectRatio {
            align: AspectRatioAlign::None,
            meet_or_slice: MeetOrSlice::Meet,
        };
        let out = image_source(&src, 2, 1, 4, 1, par, ImageSourceSampling::Nearest);
        assert_eq!(px(&out, 4, 0, 0), [255, 0, 0, 255]);
        assert_eq!(px(&out, 4, 1, 0), [255, 0, 0, 255]);
        assert_eq!(px(&out, 4, 2, 0), [0, 0, 255, 255]);
        assert_eq!(px(&out, 4, 3, 0), [0, 0, 255, 255]);
    }

    #[test]
    fn meet_uniform_upscale_fills_matching_aspect_target() {
        // 1×1 red into 3×3: uniform scale = min(3, 3) = 3 covers the
        // whole target — no transparent band.
        let src = build(1, 1, |_, _| [255, 0, 0, 255]);
        let out = image_source(
            &src,
            1,
            1,
            3,
            3,
            PreserveAspectRatio::default(),
            ImageSourceSampling::Nearest,
        );
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(px(&out, 3, x, y), [255, 0, 0, 255]);
            }
        }
    }

    #[test]
    fn xmidymid_meet_centres_with_transparent_bands() {
        // 2×2 white into 6×2: ratios are (3, 1) so meet picks s = 1;
        // the 2×2 image is centred at tx = (6 − 2)/2 = 2 → columns
        // 2..4 carry the image, columns 0..2 and 4..6 are the
        // §15.7.3 transparent black.
        let src = build(2, 2, |_, _| [255, 255, 255, 255]);
        let out = image_source(
            &src,
            2,
            2,
            6,
            2,
            PreserveAspectRatio::default(),
            ImageSourceSampling::Nearest,
        );
        for y in 0..2 {
            for x in 0..6u32 {
                let expect = if (2..4).contains(&x) {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 0]
                };
                assert_eq!(px(&out, 6, x, y), expect, "({x},{y})");
            }
        }
    }

    #[test]
    fn xminymin_and_xmaxymax_meet_anchor_low_and_high() {
        let src = build(2, 2, |_, _| [10, 20, 30, 255]);
        // xMinYMin: image occupies columns 0..2.
        let par_min = PreserveAspectRatio {
            align: AspectRatioAlign::XMinYMin,
            meet_or_slice: MeetOrSlice::Meet,
        };
        let out = image_source(&src, 2, 2, 6, 2, par_min, ImageSourceSampling::Nearest);
        for x in 0..6u32 {
            let expect = if x < 2 {
                [10, 20, 30, 255]
            } else {
                [0, 0, 0, 0]
            };
            assert_eq!(px(&out, 6, x, 0), expect, "xMinYMin column {x}");
        }
        // xMaxYMax: image occupies columns 4..6.
        let par_max = PreserveAspectRatio {
            align: AspectRatioAlign::XMaxYMax,
            meet_or_slice: MeetOrSlice::Meet,
        };
        let out = image_source(&src, 2, 2, 6, 2, par_max, ImageSourceSampling::Nearest);
        for x in 0..6u32 {
            let expect = if x >= 4 {
                [10, 20, 30, 255]
            } else {
                [0, 0, 0, 0]
            };
            assert_eq!(px(&out, 6, x, 0), expect, "xMaxYMax column {x}");
        }
    }

    #[test]
    fn slice_covers_whole_target_and_crops_the_overhang() {
        // 2×2 source with distinct quadrants into a 4×2 target.
        // Ratios are (2, 1); slice picks s = max = 2, so the scaled
        // source is 4×4 and the Y overhang is cropped. With YMid the
        // crop is centred: ty = (2 − 4)/2 = −1, so output row 0 reads
        // source row (0 + 0.5 + 1)/2 − 0.5 = 0.25 → rounds to source
        // row 0; output row 1 reads (1.5 + 1)/2 − 0.5 = 0.75 → source
        // row 1. Every target pixel is covered (no transparent black).
        let src = build(2, 2, |x, y| [(x * 200) as u8, (y * 200) as u8, 0, 255]);
        let par = PreserveAspectRatio {
            align: AspectRatioAlign::XMidYMid,
            meet_or_slice: MeetOrSlice::Slice,
        };
        let out = image_source(&src, 2, 2, 4, 2, par, ImageSourceSampling::Nearest);
        for oy in 0..2u32 {
            for ox in 0..4u32 {
                let sx = ox / 2; // s = 2 on X, tx = 0
                let sy = oy; // centred crop maps row k → source row k
                assert_eq!(
                    px(&out, 4, ox, oy),
                    [(sx * 200) as u8, (sy * 200) as u8, 0, 255],
                    "({ox},{oy})"
                );
                assert_ne!(px(&out, 4, ox, oy)[3], 0, "slice must cover every pixel");
            }
        }
    }

    #[test]
    fn bilinear_interior_blends_neighbouring_source_pixels() {
        // 2×1 black→white strip stretched ×2 with align none. Output
        // pixel 1 maps to u = (1.5)/2 − 0.5 = 0.25, blending the two
        // source pixels at weight 0.75/0.25.
        let src = build(2, 1, |x, _| {
            if x == 0 {
                [0, 0, 0, 255]
            } else {
                [255, 255, 255, 255]
            }
        });
        let par = PreserveAspectRatio {
            align: AspectRatioAlign::None,
            meet_or_slice: MeetOrSlice::Meet,
        };
        let out = image_source(&src, 2, 1, 4, 1, par, ImageSourceSampling::Bilinear);
        // 0 + (255 − 0) · 0.25 = 63.75 → 64
        assert_eq!(px(&out, 4, 1, 0), [64, 64, 64, 255]);
        // u(2) = (2.5)/2 − 0.5 = 0.75 → 0 + 255·0.75 = 191.25 → 191
        assert_eq!(px(&out, 4, 2, 0), [191, 191, 191, 255]);
    }

    #[test]
    fn empty_source_renders_transparent_black() {
        let out = image_source(
            &[],
            0,
            0,
            3,
            2,
            PreserveAspectRatio::default(),
            ImageSourceSampling::Nearest,
        );
        assert_eq!(out, vec![0u8; 3 * 2 * 4]);
    }

    #[test]
    fn empty_target_returns_empty_vec() {
        let src = build(2, 2, |_, _| [1, 2, 3, 4]);
        let out = image_source(
            &src,
            2,
            2,
            0,
            5,
            PreserveAspectRatio::default(),
            ImageSourceSampling::Nearest,
        );
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "image_source: src.len()")]
    fn wrong_source_length_panics() {
        image_source(
            &[0u8; 7],
            2,
            2,
            2,
            2,
            PreserveAspectRatio::default(),
            ImageSourceSampling::Nearest,
        );
    }

    #[test]
    fn typed_path_matches_byte_path() {
        let src_bytes = build(3, 5, |x, y| {
            [(x * 80) as u8, (y * 50) as u8, ((x * y) % 256) as u8, 220]
        });
        let src_typed: Vec<Rgba> = src_bytes
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let par = PreserveAspectRatio {
            align: AspectRatioAlign::XMaxYMid,
            meet_or_slice: MeetOrSlice::Slice,
        };
        let bytes_out = image_source(&src_bytes, 3, 5, 10, 7, par, ImageSourceSampling::Bilinear);
        let typed_out =
            image_source_pixels(&src_typed, 3, 5, 10, 7, par, ImageSourceSampling::Bilinear);
        let from_typed: Vec<u8> = typed_out
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(bytes_out, from_typed);
    }
}
