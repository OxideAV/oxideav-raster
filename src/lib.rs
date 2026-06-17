//! Pure-Rust vector→raster rendering kernel for the oxideav framework.
//!
//! Consumes a [`oxideav_core::VectorFrame`] (resolution-independent scene
//! graph: paths, paints, strokes, groups, transforms, gradients, clip,
//! opacity, embedded raster images) and produces a packed `Rgba`
//! [`oxideav_core::VideoFrame`].
//!
//! # Pipeline
//!
//! 1. **Walk** the scene graph; flatten each [`Node`](oxideav_core::Node)
//!    against an accumulated [`Transform2D`](oxideav_core::Transform2D).
//! 2. **Flatten** every path command sequence
//!    ([`PathCommand`](oxideav_core::PathCommand)) into one polyline per
//!    contour using de Casteljau subdivision (quadratic + cubic) and the
//!    SVG 1.1 Appendix F.6.5 elliptic-arc → cubic-Bezier algorithm.
//! 3. **Rasterize** each contour set into an alpha mask via the standard
//!    active-edge-list scanline fill, with a configurable supersampling
//!    factor for anti-aliasing (1 / 2 / 4 / 8×).
//! 4. **Paint** the mask: solid colors, linear/radial gradients (sRGB
//!    interpolation, Pad / Reflect / Repeat spread).
//! 5. **Composite** premultiplied-alpha onto the output frame; group
//!    opacity multiplies the alpha during composite.
//!
//! # Provenance
//!
//! Algorithms are implemented from first principles against the
//! SVG 1.1 + PDF 1.4 specifications. The closest prior art in this
//! repository is `oxideav-scribe`'s glyph rasterizer (whose de Casteljau
//! and active-edge-list implementations are generalised here), but
//! every module is a fresh implementation that does not depend on
//! `oxideav-scribe`.
//!
//! # Implemented
//!
//! * paths (move / line / quad / cubic / arc / close),
//! * fill (even-odd + non-zero) with configurable supersampling,
//! * stroke geometry (Butt/Round/Square caps, Miter/Round/Bevel joins,
//!   dash patterns, miter limit),
//! * linear + radial gradients (Pad/Reflect/Repeat spread, off-centre
//!   focal, selectable sRGB or linear-light `color-interpolation`
//!   space — see [`InterpolationSpace`] and
//!   [`Renderer::color_interpolation`]),
//! * single-path clip,
//! * group opacity,
//! * embedded raster images (`Node::Image`) — nearest-neighbour,
//!   bilinear, Lanczos2 (4×4 windowed sinc), Mitchell–Netravali bicubic
//!   (`B = C = 1/3`), Catmull–Rom bicubic (`B = 0, C = 1/2`, the
//!   interpolating BC-family member), or cubic B-spline (`B = 1, C = 0`,
//!   the smoothest, ring-free approximating BC-family member) sampling,
//!   configurable via [`Renderer::image_filter`],
//! * bitmap cache for memoised group subtrees (consumes
//!   [`oxideav_core::Group::cache_key`]) — round 3 stores only the
//!   touched-pixel bbox crop per entry, not the full canvas,
//! * soft masks (luminance + alpha) on `Node::SoftMask`,
//! * 12 standard separable blend modes from PDF 32000-1:2008 §11.3.5.2
//!   / W3C Compositing-1 §10 (Normal / Multiply / Screen / Overlay /
//!   Darken / Lighten / ColorDodge / ColorBurn / HardLight / SoftLight /
//!   Difference / Exclusion) + 4 non-separable HSL modes from PDF
//!   §11.3.5.3 / W3C Compositing-1 §11 (Hue / Saturation / Color /
//!   Luminosity, using the spec's PDF-coefficient `Lum` /
//!   `SetLum` / `ClipColor` / `Sat` / `SetSat` helpers) — selectable
//!   via [`Renderer::blend_mode`] or the standalone [`blend_over`]
//!   helper. Normal stays on the fast premultiplied source-over path;
//!   non-normal modes evaluate the spec's basic-compositing formula
//!   per-pixel.
//!
//! * `<feMorphology>` (SVG 1.1 §15.20) erosion + dilation by an
//!   axis-aligned rectangular structuring element of half-extents
//!   `(rx, ry)`, exposed as the standalone [`morphology`] /
//!   [`morphology_pixels`] functions selectable via [`MorphologyOp`].
//!   Implemented via the separable-rectangle decomposition
//!   `f ⊖ B = (f ⊖ Bx) ⊖ By`, `f ⊕ B = (f ⊕ Bx) ⊕ By` (Serra 1982
//!   §I.4 Theorem 4.1 / Gonzalez & Woods 2008 §9.4.1) so the
//!   per-pixel work scales as `O(rx + ry)` instead of `O(rx · ry)`.
//! * `<feColorMatrix>` (SVG 1.1 §15.10) per-pixel 4×5 colour-matrix
//!   transform, exposed as [`color_matrix`] (with a typed-pixel
//!   wrapper [`color_matrix_pixels`]) and the [`color_matrix_op`]
//!   shortcut that dispatches a [`ColorMatrixOp`] (general matrix /
//!   `saturate(s)` / `hueRotate(θ)` / `luminanceToAlpha`). The three
//!   parameterised operators are pre-built from the spec's
//!   coefficient tables in [`ColorMatrix::saturate`] /
//!   [`ColorMatrix::hue_rotate`] / [`ColorMatrix::luminance_to_alpha`].
//! * `<feGaussianBlur>` (SVG 1.1 §15.17) separable Gaussian blur on
//!   packed-RGBA, exposed as [`gaussian_blur`] (with the typed-pixel
//!   wrapper [`gaussian_blur_pixels`]). For small `stdDeviation`
//!   (`s < `[`GAUSSIAN_BLUR_BOX_THRESHOLD`]`= 2.0`) a direct discrete
//!   convolution with a half-width `ceil(3·s)` kernel is used; for
//!   larger `s` the spec's three-box-blur approximation
//!   (`d = floor(s · 3·sqrt(2π)/4 + 0.5)`) is used.
//! * `<feComponentTransfer>` (SVG 1.1 §15.11) per-pixel per-channel
//!   transfer function — `identity`, `table` (piecewise-linear N+1),
//!   `discrete` (step N), `linear` (`slope · C + intercept`), and
//!   `gamma` (`amplitude · C^exponent + offset`). Exposed as
//!   [`component_transfer`] (with the typed-pixel wrapper
//!   [`component_transfer_pixels`]), driven by a [`ComponentTransfer`]
//!   struct of four independent [`TransferFunc`] values (one per RGBA
//!   channel).
//! * `<feComposite>` (SVG 1.1 §15.12) pixel-wise combination of two
//!   equal-sized RGBA buffers via a [`CompositeOp`] — the five
//!   Porter–Duff operators (`over` / `in` / `out` / `atop` / `xor`,
//!   expressed through premultiplied blend factors `(Fa, Fb)` consistent
//!   with the §14.2 simple-alpha-compositing algebra) plus the §15.12
//!   `arithmetic` operator `result = k1·i1·i2 + k2·i1 + k3·i2 + k4`
//!   (premultiplied, clamped to `[0, 1]`). Exposed as [`composite_filter`]
//!   (with the typed-pixel wrapper [`composite_filter_pixels`]).
//! * `<feConvolveMatrix>` (SVG 1.1 §15.13) general 2-D matrix
//!   convolution on packed-RGBA. The spec formula `COLOR_{X,Y} =
//!   (SUM SUM SOURCE · kernelMatrix[orderX-J-1, orderY-I-1]) /
//!   divisor + bias` is applied verbatim, including the §15.13
//!   180° kernel rotation. All three `edgeMode` policies (`duplicate`,
//!   `wrap`, `none`) and both `preserveAlpha` settings are supported.
//!   Exposed as [`convolve_matrix`] (with the typed-pixel wrapper
//!   [`convolve_matrix_pixels`]), driven by a [`ConvolveMatrix`]
//!   parameter block and [`ConvolveEdgeMode`] enum.
//! * `<feFlood>` (SVG 1.1 §15.16) constant-colour source primitive.
//!   Emits a `width × height` packed-RGBA buffer in which every pixel
//!   carries the resolved `flood-color` triple modulated by
//!   `flood-opacity` (straight-alpha form). Exposed as [`flood`] (with
//!   the typed-pixel wrapper [`flood_pixels`]).
//! * `<feOffset>` (SVG 1.1 §15.21) translates the input image by
//!   `(dx, dy)` and emits a buffer of the same dimensions. Output
//!   pixels whose source position falls outside the input extent are
//!   set to transparent black per §15.7.3 ("undefined pixels are set
//!   to transparent black"). Two sampling policies are wired in
//!   ([`OffsetSampling`]): nearest-pixel sampling with `(dx, dy)`
//!   rounded to the nearest integer (the default) and bilinear
//!   reconstruction for fractional shifts (the §15.21 "high quality
//!   viewer" route). Exposed as [`offset`] (with the typed-pixel
//!   wrapper [`offset_pixels`]).
//! * `<feMerge>` (SVG 1.1 §15.19) composites N input layers
//!   bottom-to-top using the §14.2 simple-alpha-compositing `over`
//!   operator. Exposed as [`merge`] (with the typed-pixel wrapper
//!   [`merge_pixels`]).
//! * `<feDropShadow>` (Filter Effects Module Level 1 §9.12) — the
//!   shorthand drop-shadow primitive, evaluated as its spec-defined
//!   equivalent chain `feGaussianBlur(alpha) → feOffset → feFlood →
//!   feComposite(in) → feMerge(source)`. Exposed as [`drop_shadow`]
//!   (with the typed-pixel wrapper [`drop_shadow_pixels`]).
//! * `<feDisplacementMap>` (SVG 1.1 §15.15) per-pixel channel-driven
//!   warp of `in1` by the §15.15 formula
//!   `P'(x, y) = P(x + scale · (XC − 0.5), y + scale · (YC − 0.5))`,
//!   where `XC` / `YC` are the `[0, 1]` channel values of `in2`
//!   selected by `xChannelSelector` / `yChannelSelector` (both default
//!   to `A`). Two sampling policies are wired in
//!   ([`DisplacementSampling`]): nearest-pixel sampling on the warped
//!   coordinate (the default) and bilinear reconstruction for
//!   fractional shifts (the §15.15 "high quality viewers apply an
//!   interpolent" route). Out-of-bounds source positions emit
//!   transparent black per §15.7.3. Exposed as [`displacement_map`]
//!   (with the typed-pixel wrapper [`displacement_map_pixels`]).
//! * `<feBlend>` (SVG 1.1 §15.9) per-pixel five-mode blend of two
//!   equally-sized inputs (`Normal` / `Multiply` / `Screen` / `Darken`
//!   / `Lighten`). Result alpha follows the §15.9 mode-independent
//!   formula `qr = 1 − (1 − qa) · (1 − qb)`. Exposed as
//!   [`blend_filter`] (with the typed-pixel wrapper
//!   [`blend_filter_pixels`]).
//! * `<feDiffuseLighting>` (SVG 1.1 §15.14) Phong-diffuse lighting of
//!   an image's alpha channel treated as a height field
//!   `Z(x, y) = surfaceScale · I(x, y)`. The §15.14 surface normal is
//!   formed from one of nine §15.14-tabulated Sobel-kernel variants
//!   (one per corner / edge / interior region) and combined with the
//!   light direction `L` via `D = kd · max(N · L, 0) · L_color`. Three
//!   §15.8 light-source kinds are supported via [`LightSource`]:
//!   `Distant` (§15.8.2: constant unit vector from `azimuth` /
//!   `elevation`), `Point` (§15.8.3: per-pixel
//!   `L = (light_pos − sample_pos) / Norm`), and `Spot` (§15.8.4:
//!   per-pixel point-style `L` plus the §15.14 spot-emission scaling
//!   `Light · pow(−L · S, specularExponent)` with optional
//!   `limitingConeAngle` cut-off). Result alpha is `1.0` everywhere
//!   per §15.14. Exposed as [`diffuse_lighting`] (with the typed-pixel
//!   wrapper [`diffuse_lighting_pixels`]) and parameterised through
//!   the [`DiffuseLighting`] struct.
//! * `<pattern>` paint servers (SVG 2 §14.3) — a [`Pattern`] carries a
//!   user-space tile rectangle, an optional `patternTransform`, an
//!   optional `viewBox` + `preserveAspectRatio` (content fitted onto
//!   the tile rectangle by the §8.2 equivalent-transform algorithm,
//!   [`view_box_fit_transform`] — all 10 alignments × meet / slice),
//!   and vector content (tile-origin-relative without a `viewBox`,
//!   viewBox-space with one); the tile is rasterised once offscreen at
//!   device resolution and sampled with periodic (wrap-around)
//!   addressing through the inverse pattern transform, so rotated /
//!   skewed pattern transforms resolve exactly and seams stay
//!   continuous under bilinear filtering. Exposed as
//!   [`Renderer::fill_path_with_pattern`] /
//!   [`Renderer::stroke_path_with_pattern`].
//! * filter primitive subregion clip (Filter Effects Module Level 1
//!   §9.4) — every `fe*` primitive's `x` / `y` / `width` / `height`
//!   "act[s] as a hard clip clipping rectangle on the filter primitive
//!   result"; [`clip_to_subregion`] (with the typed-pixel wrapper
//!   [`clip_to_subregion_pixels`]) clears every pixel outside the
//!   resolved [`Subregion`] to transparent black, keeping any pixel that
//!   even partly intersects it, and disables the whole primitive
//!   (transparent-black result) when the subregion width or height is
//!   non-positive.
//!
//! Deferred to a later round:
//!
//! * `feDropShadow` (the W3C Filter Effects Module Level 1 spec is not
//!   staged under `docs/`),
//! * a `Paint`-level pattern variant (pending an `oxideav-core` IR
//!   addition — patterns currently paint through the direct
//!   [`Renderer`] endpoints),
//! * full ICC-tagged color-managed pipeline (the `linearRGB`
//!   interpolation space is implemented; per-channel ICC profile
//!   transforms are still TODO),
//! * patterned dashes interacting with miter / round joins beyond
//!   single-segment dashes.

#![deny(rust_2018_idioms)]
#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

mod blend;
mod cache;
mod composite;
mod fill;
mod filter;
mod flatten;
mod gradient;
mod paint;
mod pattern;
mod renderer;
mod stroke;

pub use blend::{blend_channel, blend_over, BlendMode};
pub use cache::{CacheStats, RasterizedSubtree};
pub use composite::{composite_rgba_premultiplied, composite_rgba_premultiplied_blend};
pub use fill::{rasterize_fill, AlphaMask};
pub use filter::{
    blend_filter, blend_filter_pixels, clip_to_subregion, clip_to_subregion_pixels, color_matrix,
    color_matrix_op, color_matrix_pixels, component_transfer, component_transfer_pixels,
    composite_filter, composite_filter_pixels, convolve_matrix, convolve_matrix_pixels,
    diffuse_lighting, diffuse_lighting_pixels, displacement_map, displacement_map_pixels,
    drop_shadow, drop_shadow_pixels, flood, flood_pixels, gaussian_blur, gaussian_blur_pixels,
    image_source, image_source_pixels, in_filter_space, in_filter_space_pixels, linear_to_srgb_f32,
    merge, merge_pixels, morphology, morphology_pixels, offset, offset_pixels, specular_lighting,
    specular_lighting_pixels, srgb_to_linear_f32, tile, tile_pixels, to_linear_rgb,
    to_linear_rgb_pixels, to_srgb, to_srgb_pixels, turbulence_filter, turbulence_filter_pixels,
    AspectRatioAlign, BlendFilterMode, ColorMatrix, ColorMatrixOp, ComponentTransfer, CompositeOp,
    ConvolveEdgeMode, ConvolveMatrix, DiffuseLighting, DisplacementChannel, DisplacementSampling,
    FilterColorSpace, ImageSourceSampling, LightSource, MeetOrSlice, MorphologyOp, OffsetSampling,
    PreserveAspectRatio, SpecularLighting, StitchTiles, Subregion, TransferFunc, Turbulence,
    TurbulenceType, GAUSSIAN_BLUR_BOX_THRESHOLD,
};
pub use flatten::{flatten_arc_to_cubics, flatten_path, FlatContour};
pub use gradient::{
    eval_focal_radial_gradient, eval_focal_radial_gradient_in, eval_focal_radial_gradient_lut,
    eval_linear_gradient, eval_linear_gradient_in, eval_linear_gradient_lut, eval_radial_gradient,
    eval_radial_gradient_in, eval_radial_gradient_lut, FocalRadialGradient, InterpolationSpace,
    StopsLut,
};
pub use paint::{sample_paint, sample_paint_in};
pub use pattern::{view_box_fit_transform, Pattern};
pub use renderer::{rasterize, ImageFilter, Renderer, DEFAULT_CACHE_CAPACITY};
pub use stroke::stroke_to_fill_path;

#[cfg(test)]
use oxideav_core::Rgba;

/// Premultiply a straight-alpha sRGB color into 16-bit-per-channel
/// premultiplied form for blending. The promotion to `u16` keeps a
/// little extra headroom for the multiplications inside the composite
/// loop.
#[inline]
#[cfg(test)]
fn premultiply(c: Rgba) -> [u16; 4] {
    let a = c.a as u16;
    [
        (c.r as u16 * a + 127) / 255,
        (c.g as u16 * a + 127) / 255,
        (c.b as u16 * a + 127) / 255,
        a,
    ]
}

/// Reverse the premultiplication. `a == 0` maps to `(0, 0, 0, 0)`.
#[inline]
#[cfg(test)]
fn unpremultiply(c: [u16; 4]) -> Rgba {
    let a = c[3];
    if a == 0 {
        return Rgba::new(0, 0, 0, 0);
    }
    Rgba::new(
        ((c[0] as u32 * 255 + a as u32 / 2) / a as u32).min(255) as u8,
        ((c[1] as u32 * 255 + a as u32 / 2) / a as u32).min(255) as u8,
        ((c[2] as u32 * 255 + a as u32 / 2) / a as u32).min(255) as u8,
        a as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premultiply_roundtrip_opaque() {
        let c = Rgba::new(123, 200, 50, 255);
        let p = premultiply(c);
        assert_eq!(p[3], 255);
        let r = unpremultiply(p);
        assert_eq!(r, c);
    }

    #[test]
    fn premultiply_zero_alpha_is_transparent_black() {
        let c = Rgba::new(255, 255, 255, 0);
        let p = premultiply(c);
        assert_eq!(p, [0, 0, 0, 0]);
        assert_eq!(unpremultiply(p), Rgba::new(0, 0, 0, 0));
    }
}
