//! `<pattern>` paint server — tiled fill / stroke paint (SVG 2 §14.3,
//! "Paint Servers: Gradients and Patterns"; SVG 1.1 §13.3 is the same
//! model).
//!
//! A pattern paints a region by replicating ("tiling") a reference
//! rectangle of vector content at fixed intervals in x and y. The
//! reference rectangle has its top/left at `(x, y)` and its
//! bottom/right at `(x + width, y + height)`; the tiling conceptually
//! extends such rectangles to infinity in both directions, with tiles
//! starting at `(x + m·width, y + n·height)` for every integer pair
//! `(m, n)` (§14.3).
//!
//! [`Pattern`] carries the tile rectangle in **user space**
//! (`patternUnits="userSpaceOnUse"` semantics — callers that resolve
//! `objectBoundingBox` units do so before constructing the value, which
//! is a pure affine remap of `x`/`y`/`width`/`height` against the
//! target's bounding box), an optional `patternTransform` (an
//! additional transform from the pattern coordinate system onto the
//! target coordinate system, §14.3.1 — post-multiplied onto the
//! user→device transform, i.e. inserted to the right), and the tile
//! content as a list of [`Node`]s whose coordinates are relative to the
//! tile origin (the §14.3.2 "new coordinate system … has its origin at
//! `(x, y)`" rule for patterns without a `viewBox`).
//!
//! Per §14.3.1, a `width` or `height` of zero disables rendering — no
//! paint is applied (negative values are an error in the source
//! document; this implementation treats them, and non-finite values,
//! the same as zero: no paint).
//!
//! # Rasterisation strategy
//!
//! The tile content is rendered **once** into an offscreen RGBA buffer
//! at device resolution (sized from the pattern→device scale factors,
//! and clipped to the tile rectangle — the user-agent
//! `overflow: hidden` default of §14.3.2). Each destination pixel
//! inside the fill / stroke coverage mask is then inverse-mapped
//! through the combined `user→device ∘ patternTransform` matrix into
//! pattern space, reduced modulo the tile extent, and the tile buffer
//! is sampled with **periodic (wrap-around) addressing**. Arbitrary
//! affine pattern transforms (rotation / skew / non-uniform scale)
//! therefore resolve exactly, and tile seams stay continuous under
//! bilinear filtering — the 2×2 footprint wraps to the opposite tile
//! edge instead of clamping.

use oxideav_core::{Node, Rgba, Transform2D, VideoFrame};

/// A tiled paint server (SVG `<pattern>`).
///
/// See the [module docs](self) for coordinate-system semantics. Use the
/// builder methods to attach content and a `patternTransform`, then
/// paint with [`Renderer::fill_path_with_pattern`] /
/// [`Renderer::stroke_path_with_pattern`].
///
/// [`Renderer::fill_path_with_pattern`]: crate::Renderer::fill_path_with_pattern
/// [`Renderer::stroke_path_with_pattern`]: crate::Renderer::stroke_path_with_pattern
#[derive(Clone, Debug)]
pub struct Pattern {
    /// Tile-rectangle left edge in user space.
    pub x: f32,
    /// Tile-rectangle top edge in user space.
    pub y: f32,
    /// Tile width in user space. Zero (or negative / non-finite)
    /// disables painting per §14.3.1.
    pub width: f32,
    /// Tile height in user space. Zero (or negative / non-finite)
    /// disables painting per §14.3.1.
    pub height: f32,
    /// `patternTransform` — additional transform from the pattern
    /// coordinate system onto the user coordinate system. Identity by
    /// default.
    pub transform: Transform2D,
    /// Tile content. Coordinates are relative to the tile origin: a
    /// point `(px, py)` of content renders at user-space
    /// `(x + px + m·width, y + py + n·height)` for every tile `(m, n)`.
    /// Content outside `[0, width) × [0, height)` is clipped to the
    /// tile (`overflow: hidden`, §14.3.2).
    pub content: Vec<Node>,
}

impl Pattern {
    /// Build an empty pattern with the given tile rectangle, identity
    /// `patternTransform`, and no content.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            transform: Transform2D::identity(),
            content: Vec::new(),
        }
    }

    /// Set the `patternTransform`.
    pub fn with_transform(mut self, transform: Transform2D) -> Self {
        self.transform = transform;
        self
    }

    /// Append one content node (tile-origin-relative coordinates).
    pub fn with_child(mut self, child: Node) -> Self {
        self.content.push(child);
        self
    }

    /// Replace the content list wholesale.
    pub fn with_children(mut self, content: Vec<Node>) -> Self {
        self.content = content;
        self
    }

    /// `true` when the tile rectangle disables painting (§14.3.1:
    /// zero `width` / `height` means no paint; negative and non-finite
    /// extents are treated the same way).
    pub fn is_degenerate(&self) -> bool {
        !(self.width > 0.0
            && self.height > 0.0
            && self.width.is_finite()
            && self.height.is_finite())
    }
}

/// Extract `(width, height, stride, data)` of a packed-RGBA tile
/// buffer. Returns `None` for an empty / degenerate frame.
fn tile_dims(frame: &VideoFrame) -> Option<(usize, usize, usize, &[u8])> {
    let p = frame.planes.first()?;
    let w = p.stride / 4;
    let h = p.data.len().checked_div(p.stride).unwrap_or(0);
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h, p.stride, &p.data))
}

/// Nearest-neighbour sample of a tile buffer with periodic (wrap)
/// addressing. `(u, v)` are normalised tile coordinates — `[0, 1)`
/// spans one tile; values outside wrap.
pub(crate) fn sample_tile_nearest_wrap(frame: &VideoFrame, u: f32, v: f32) -> Rgba {
    let Some((w, h, stride, data)) = tile_dims(frame) else {
        return Rgba::new(0, 0, 0, 0);
    };
    let px = ((u * w as f32).floor() as i64).rem_euclid(w as i64) as usize;
    let py = ((v * h as f32).floor() as i64).rem_euclid(h as i64) as usize;
    let i = py * stride + px * 4;
    Rgba::new(data[i], data[i + 1], data[i + 2], data[i + 3])
}

/// Bilinear sample of a tile buffer with periodic (wrap) addressing.
///
/// Taps whose 2×2 footprint crosses a tile edge wrap to the opposite
/// edge, so adjacent tiles filter into each other seamlessly (the tiling
/// is a true torus). Filtering happens in premultiplied-alpha space —
/// mirroring the renderer's bounded-image bilinear sampler — so
/// transparent texels don't bleed colour, and the straight-alpha result
/// is recovered at the end.
pub(crate) fn sample_tile_bilinear_wrap(frame: &VideoFrame, u: f32, v: f32) -> Rgba {
    let Some((w, h, stride, data)) = tile_dims(frame) else {
        return Rgba::new(0, 0, 0, 0);
    };
    // Continuous texel coordinate where integer values land on texel
    // centres.
    let tx = u * w as f32 - 0.5;
    let ty = v * h as f32 - 0.5;
    let fx = tx.floor();
    let fy = ty.floor();
    let wx = (tx - fx).clamp(0.0, 1.0);
    let wy = (ty - fy).clamp(0.0, 1.0);
    let x0 = (fx as i64).rem_euclid(w as i64) as usize;
    let y0 = (fy as i64).rem_euclid(h as i64) as usize;
    let x1 = (x0 + 1) % w;
    let y1 = (y0 + 1) % h;
    // Premultiplied fetch.
    let fetch = |x: usize, y: usize| -> (f32, f32, f32, f32) {
        let i = y * stride + x * 4;
        let a = data[i + 3] as f32;
        let s = a / 255.0;
        (
            data[i] as f32 * s,
            data[i + 1] as f32 * s,
            data[i + 2] as f32 * s,
            a,
        )
    };
    let p00 = fetch(x0, y0);
    let p10 = fetch(x1, y0);
    let p01 = fetch(x0, y1);
    let p11 = fetch(x1, y1);
    let w00 = (1.0 - wx) * (1.0 - wy);
    let w10 = wx * (1.0 - wy);
    let w01 = (1.0 - wx) * wy;
    let w11 = wx * wy;
    let lerp4 = |a: f32, b: f32, c: f32, d: f32| a * w00 + b * w10 + c * w01 + d * w11;
    let pr = lerp4(p00.0, p10.0, p01.0, p11.0);
    let pg = lerp4(p00.1, p10.1, p01.1, p11.1);
    let pb = lerp4(p00.2, p10.2, p01.2, p11.2);
    let pa = lerp4(p00.3, p10.3, p01.3, p11.3);
    if pa <= 0.5 {
        return Rgba::new(0, 0, 0, 0);
    }
    let inv = 255.0 / pa;
    Rgba::new(
        (pr * inv).round().clamp(0.0, 255.0) as u8,
        (pg * inv).round().clamp(0.0, 255.0) as u8,
        (pb * inv).round().clamp(0.0, 255.0) as u8,
        pa.round().clamp(0.0, 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::VideoPlane;

    fn two_texel_frame(left: Rgba, right: Rgba) -> VideoFrame {
        // 2×1 RGBA tile.
        let data = vec![
            left.r, left.g, left.b, left.a, right.r, right.g, right.b, right.a,
        ];
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride: 8, data }],
        }
    }

    #[test]
    fn builder_defaults() {
        let p = Pattern::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!((p.x, p.y, p.width, p.height), (1.0, 2.0, 3.0, 4.0));
        assert!(p.transform.is_identity());
        assert!(p.content.is_empty());
        assert!(!p.is_degenerate());
    }

    #[test]
    fn degenerate_extents() {
        assert!(Pattern::new(0.0, 0.0, 0.0, 4.0).is_degenerate());
        assert!(Pattern::new(0.0, 0.0, 3.0, 0.0).is_degenerate());
        assert!(Pattern::new(0.0, 0.0, -3.0, 4.0).is_degenerate());
        assert!(Pattern::new(0.0, 0.0, f32::NAN, 4.0).is_degenerate());
        assert!(Pattern::new(0.0, 0.0, f32::INFINITY, 4.0).is_degenerate());
        assert!(!Pattern::new(0.0, 0.0, 3.0, 4.0).is_degenerate());
    }

    #[test]
    fn nearest_wrap_addresses_periodically() {
        let red = Rgba::opaque(255, 0, 0);
        let blue = Rgba::opaque(0, 0, 255);
        let f = two_texel_frame(red, blue);
        // In-tile.
        assert_eq!(sample_tile_nearest_wrap(&f, 0.25, 0.5), red);
        assert_eq!(sample_tile_nearest_wrap(&f, 0.75, 0.5), blue);
        // One tile to the right / left — identical.
        assert_eq!(sample_tile_nearest_wrap(&f, 1.25, 0.5), red);
        assert_eq!(sample_tile_nearest_wrap(&f, -0.75, 0.5), red);
        assert_eq!(sample_tile_nearest_wrap(&f, -0.25, 0.5), blue);
    }

    #[test]
    fn bilinear_wrap_blends_across_tile_seam() {
        let red = Rgba::opaque(255, 0, 0);
        let blue = Rgba::opaque(0, 0, 255);
        let f = two_texel_frame(red, blue);
        // u = 0.0 → texel coordinate -0.5, exactly between the right
        // texel of the previous tile (blue, wraps) and the left texel
        // of this tile (red): a 50/50 mix, NOT clamp-to-edge red.
        let c = sample_tile_bilinear_wrap(&f, 0.0, 0.5);
        assert_eq!(c.a, 255);
        assert!((c.r as i32 - 128).abs() <= 1, "r = {}", c.r);
        assert!((c.b as i32 - 128).abs() <= 1, "b = {}", c.b);
        // Texel centres reproduce the texels exactly.
        assert_eq!(sample_tile_bilinear_wrap(&f, 0.25, 0.5), red);
        assert_eq!(sample_tile_bilinear_wrap(&f, 0.75, 0.5), blue);
    }

    #[test]
    fn empty_frame_samples_transparent() {
        let f = VideoFrame {
            pts: None,
            planes: vec![],
        };
        assert_eq!(
            sample_tile_nearest_wrap(&f, 0.5, 0.5),
            Rgba::new(0, 0, 0, 0)
        );
        assert_eq!(
            sample_tile_bilinear_wrap(&f, 0.5, 0.5),
            Rgba::new(0, 0, 0, 0)
        );
    }
}
