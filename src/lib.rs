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
//! # Workspace policy
//!
//! No `lyon` / `tiny-skia` / `resvg` / `kurbo` / `pathfinder` source
//! material is used here. Algorithms are implemented from first
//! principles against SVG 1.1 + PDF 1.4 specifications. The closest
//! prior art in this repository is `oxideav-scribe`'s glyph rasterizer
//! (which inspired the de Casteljau + active-edge-list implementations
//! that ship here in generalised form), but every module is a fresh
//! implementation that does not depend on `oxideav-scribe`.
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
//!   bilinear, Lanczos2 (4×4 windowed sinc), or Mitchell–Netravali
//!   bicubic (`B = C = 1/3`) sampling, configurable via
//!   [`Renderer::image_filter`],
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
//! Deferred to a later round:
//!
//! * filters (Gaussian blur, drop shadow, feColorMatrix, etc.),
//! * `<pattern>` paints,
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
mod flatten;
mod gradient;
mod paint;
mod renderer;
mod stroke;

pub use blend::{blend_channel, blend_over, BlendMode};
pub use cache::{CacheStats, RasterizedSubtree};
pub use composite::{composite_rgba_premultiplied, composite_rgba_premultiplied_blend};
pub use fill::{rasterize_fill, AlphaMask};
pub use flatten::{flatten_arc_to_cubics, flatten_path, FlatContour};
pub use gradient::{
    eval_linear_gradient, eval_linear_gradient_in, eval_linear_gradient_lut, eval_radial_gradient,
    eval_radial_gradient_in, eval_radial_gradient_lut, InterpolationSpace, StopsLut,
};
pub use paint::{sample_paint, sample_paint_in};
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
