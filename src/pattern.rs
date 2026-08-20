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
//! # `viewBox` tile fitting
//!
//! When a `viewBox` is attached ([`Pattern::with_view_box`]), §14.3.2
//! applies instead: "If there is a ‘viewBox’ attribute, then the new
//! coordinate system is fitted into the region defined by the ‘x’,
//! ‘y’, ‘width’, ‘height’ … attributes on the ‘pattern’ element using
//! the standard rules for ‘viewBox’ and ‘preserveAspectRatio’." The
//! standard rules are the §8.2 "equivalent transform of an SVG
//! viewport" algorithm ([`view_box_fit_transform`]'s doc walks through
//! it); content coordinates then live in viewBox space and the fitted
//! transform maps them onto the tile rectangle. Per §14.3.1 a
//! `patternContentUnits` value "has no effect if attribute ‘viewBox’
//! is specified", which is why no content-units knob exists here — the
//! no-viewBox content space is tile-origin-relative user space by
//! construction, and the viewBox case overrides it entirely.
//!
//! Per §8.6 a `viewBox` width or height of zero "disables rendering of
//! the element" and a negative value is an error; both (and non-finite
//! values) make the pattern paint nothing, exactly like a degenerate
//! tile rectangle.
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

use crate::filter::{AspectRatioAlign, MeetOrSlice, PreserveAspectRatio};
use oxideav_core::{Node, Rgba, Transform2D, VideoFrame, ViewBox};

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
    /// Optional `viewBox` — when present, content coordinates live in
    /// viewBox space and are fitted onto the tile rectangle "using the
    /// standard rules for ‘viewBox’ and ‘preserveAspectRatio’"
    /// (§14.3.2); see [`view_box_fit_transform`]. A zero / negative /
    /// non-finite `width` or `height` disables painting (§8.6: zero
    /// "disables rendering of the element", negative "is an error").
    pub view_box: Option<ViewBox>,
    /// `preserveAspectRatio` governing the [`Self::view_box`] fitting.
    /// Ignored when `view_box` is `None`. Defaults to `xMidYMid meet`
    /// (§7.8 / §8.2 defaults).
    pub preserve_aspect_ratio: PreserveAspectRatio,
    /// Tile content. Without a `view_box`, coordinates are relative to
    /// the tile origin: a point `(px, py)` of content renders at
    /// user-space `(x + px + m·width, y + py + n·height)` for every
    /// tile `(m, n)`. With a `view_box`, coordinates are in viewBox
    /// space and reach the tile through the §8.2 fitted transform.
    /// Content outside the tile rectangle is clipped to the tile
    /// (`overflow: hidden`, §14.3.2).
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
            view_box: None,
            preserve_aspect_ratio: PreserveAspectRatio::default(),
            content: Vec::new(),
        }
    }

    /// Set the `patternTransform`.
    pub fn with_transform(mut self, transform: Transform2D) -> Self {
        self.transform = transform;
        self
    }

    /// Attach a `viewBox`: content coordinates become viewBox-space and
    /// are fitted onto the tile rectangle per §14.3.2 / §8.2 (see
    /// [`view_box_fit_transform`]). The fitting uses the pattern's
    /// [`Self::preserve_aspect_ratio`] (default `xMidYMid meet`).
    pub fn with_view_box(mut self, view_box: ViewBox) -> Self {
        self.view_box = Some(view_box);
        self
    }

    /// Set the `preserveAspectRatio` used for [`Self::view_box`]
    /// fitting. Has no effect without a `viewBox`.
    pub fn with_preserve_aspect_ratio(mut self, par: PreserveAspectRatio) -> Self {
        self.preserve_aspect_ratio = par;
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

    /// `true` when the tile rectangle or the `viewBox` disables
    /// painting (§14.3.1: zero tile `width` / `height` means no paint;
    /// §8.6: a zero viewBox `width` / `height` "disables rendering of
    /// the element" and a negative value is an error; non-finite
    /// extents are treated the same way).
    pub fn is_degenerate(&self) -> bool {
        let bad = |v: f32| !(v > 0.0 && v.is_finite());
        bad(self.width)
            || bad(self.height)
            || self
                .view_box
                .map(|vb| bad(vb.width) || bad(vb.height))
                .unwrap_or(false)
    }
}

/// SVG 2 §8.2 — "Computing the equivalent transform of an SVG
/// viewport": the translation + scale that fits the `vb` viewBox into
/// the viewport rectangle `(e_x, e_y, e_width, e_height)` under the
/// `preserveAspectRatio` value `par`.
///
/// The §8.2 steps, verbatim in structure:
///
/// 1. `scale-x = e-width / vb-width`, `scale-y = e-height / vb-height`.
/// 2. If `align` is not `none` and `meetOrSlice` is `meet`, "set the
///    larger of scale-x and scale-y to the smaller"; if `slice`, "set
///    the smaller … to the larger".
/// 3. `translate-x = e-x − (vb-x · scale-x)`,
///    `translate-y = e-y − (vb-y · scale-y)`.
/// 4. If `align` contains `xMid`, add `(e-width − vb-width·scale-x)/2`
///    to `translate-x`; `xMax` adds the whole difference. Same per-axis
///    rule for `yMid` / `yMax`.
///
/// "The transform applied to content contained by the element is given
/// by `translate(translate-x, translate-y) scale(scale-x, scale-y)`."
///
/// The caller guarantees positive finite `vb.width` / `vb.height`
/// (degenerate viewBoxes never reach the fitting step — they disable
/// painting per §8.6, see [`Pattern::is_degenerate`]).
pub fn view_box_fit_transform(
    vb: &ViewBox,
    e_x: f32,
    e_y: f32,
    e_width: f32,
    e_height: f32,
    par: PreserveAspectRatio,
) -> Transform2D {
    let mut scale_x = e_width / vb.width;
    let mut scale_y = e_height / vb.height;
    if par.align != AspectRatioAlign::None {
        let s = match par.meet_or_slice {
            MeetOrSlice::Meet => scale_x.min(scale_y),
            MeetOrSlice::Slice => scale_x.max(scale_y),
        };
        scale_x = s;
        scale_y = s;
    }
    let mut translate_x = e_x - vb.min_x * scale_x;
    let mut translate_y = e_y - vb.min_y * scale_y;
    use AspectRatioAlign as A;
    match par.align {
        A::XMidYMin | A::XMidYMid | A::XMidYMax => {
            translate_x += (e_width - vb.width * scale_x) / 2.0;
        }
        A::XMaxYMin | A::XMaxYMid | A::XMaxYMax => {
            translate_x += e_width - vb.width * scale_x;
        }
        _ => {}
    }
    match par.align {
        A::XMinYMid | A::XMidYMid | A::XMaxYMid => {
            translate_y += (e_height - vb.height * scale_y) / 2.0;
        }
        A::XMinYMax | A::XMidYMax | A::XMaxYMax => {
            translate_y += e_height - vb.height * scale_y;
        }
        _ => {}
    }
    // translate(tx, ty) · scale(sx, sy).
    Transform2D {
        a: scale_x,
        b: 0.0,
        c: 0.0,
        d: scale_y,
        e: translate_x,
        f: translate_y,
    }
}

/// Extract `(width, height, stride, data)` of a packed-RGBA tile
/// buffer. Returns `None` for an empty / degenerate frame.
pub(crate) fn tile_dims(frame: &VideoFrame) -> Option<(usize, usize, usize, &[u8])> {
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

/// One axis's wrapped nearest-neighbour texel index for the normalised
/// tile coordinate `u` — the exact index arithmetic of
/// [`sample_tile_nearest_wrap`], factored out so an axis-aligned
/// pattern transform can cache it per destination column / row.
#[inline]
pub(crate) fn tile_nearest_axis(u: f32, extent: usize) -> usize {
    ((u * extent as f32).floor() as i64).rem_euclid(extent as i64) as usize
}

/// One axis's wrapped bilinear taps `(i0, i1, w1)` for the normalised
/// tile coordinate `u` — the exact tap/weight arithmetic of
/// [`sample_tile_bilinear_wrap`], factored out so an axis-aligned
/// pattern transform can cache it per destination column / row.
#[inline]
pub(crate) fn tile_bilinear_axis(u: f32, extent: usize) -> (usize, usize, f32) {
    let t = u * extent as f32 - 0.5;
    let f = t.floor();
    let w1 = (t - f).clamp(0.0, 1.0);
    let i0 = (f as i64).rem_euclid(extent as i64) as usize;
    let i1 = (i0 + 1) % extent;
    (i0, i1, w1)
}

/// Bilinear tile sample from pre-resolved per-axis taps — the
/// fetch / weight-combination / un-premultiply tail of
/// [`sample_tile_bilinear_wrap`], byte-identical given taps produced
/// by [`tile_bilinear_axis`] from the same coordinates.
#[inline]
pub(crate) fn sample_tile_bilinear_taps(
    data: &[u8],
    stride: usize,
    col: (usize, usize, f32),
    row: (usize, usize, f32),
) -> Rgba {
    let (x0, x1, wx) = col;
    let (y0, y1, wy) = row;
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
        assert!(p.view_box.is_none());
        // §7.8 / §8.2 default: xMidYMid meet.
        assert_eq!(p.preserve_aspect_ratio, PreserveAspectRatio::default());
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
    fn degenerate_view_box_extents() {
        // §8.6: zero disables rendering; negative is an error; treat
        // non-finite the same. A healthy viewBox is not degenerate.
        for (w, h) in [
            (0.0f32, 10.0f32),
            (10.0, 0.0),
            (-10.0, 10.0),
            (f32::NAN, 10.0),
            (10.0, f32::INFINITY),
        ] {
            let p = Pattern::new(0.0, 0.0, 3.0, 4.0).with_view_box(ViewBox::new(0.0, 0.0, w, h));
            assert!(p.is_degenerate(), "viewBox {w}×{h} must be degenerate");
        }
        let p = Pattern::new(0.0, 0.0, 3.0, 4.0).with_view_box(ViewBox::new(0.0, 0.0, 10.0, 10.0));
        assert!(!p.is_degenerate());
    }

    // §8.2 fitting algebra. Helper: extract (sx, sy, tx, ty) from the
    // axis-aligned result.
    fn fit(vb: ViewBox, e: (f32, f32, f32, f32), par: PreserveAspectRatio) -> (f32, f32, f32, f32) {
        let t = view_box_fit_transform(&vb, e.0, e.1, e.2, e.3, par);
        assert_eq!((t.b, t.c), (0.0, 0.0), "fit must be translate·scale");
        (t.a, t.d, t.e, t.f)
    }

    #[test]
    fn fit_uniform_scale_no_mismatch() {
        // vb 10×10 into a 20×20 viewport: scale 2, no alignment slack.
        let par = PreserveAspectRatio::default(); // xMidYMid meet
        let vb = ViewBox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(fit(vb, (0.0, 0.0, 20.0, 20.0), par), (2.0, 2.0, 0.0, 0.0));
    }

    #[test]
    fn fit_min_xy_translates_origin() {
        // §8.2 step 3: translate-x = e-x − vb-x·scale-x.
        let par = PreserveAspectRatio::default();
        let vb = ViewBox::new(5.0, 5.0, 10.0, 10.0);
        assert_eq!(
            fit(vb, (0.0, 0.0, 20.0, 20.0), par),
            (2.0, 2.0, -10.0, -10.0)
        );
        // Element offset adds straight through.
        assert_eq!(fit(vb, (3.0, 7.0, 20.0, 20.0), par), (2.0, 2.0, -7.0, -3.0));
    }

    #[test]
    fn fit_meet_takes_smaller_scale_and_centres() {
        // vb 10×10 into 40×20: scale-x 4, scale-y 2 → meet picks 2;
        // xMid adds (40 − 10·2)/2 = 10 to translate-x.
        let par = PreserveAspectRatio::default();
        let vb = ViewBox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(fit(vb, (0.0, 0.0, 40.0, 20.0), par), (2.0, 2.0, 10.0, 0.0));
    }

    #[test]
    fn fit_slice_takes_larger_scale_and_centres_overflow() {
        // Same geometry under slice: scale 4; yMid adds
        // (20 − 10·4)/2 = −10 to translate-y.
        let par = PreserveAspectRatio {
            align: AspectRatioAlign::XMidYMid,
            meet_or_slice: MeetOrSlice::Slice,
        };
        let vb = ViewBox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(fit(vb, (0.0, 0.0, 40.0, 20.0), par), (4.0, 4.0, 0.0, -10.0));
    }

    #[test]
    fn fit_align_none_scales_each_axis() {
        // align=none: non-uniform fill, meetOrSlice ignored.
        for mos in [MeetOrSlice::Meet, MeetOrSlice::Slice] {
            let par = PreserveAspectRatio {
                align: AspectRatioAlign::None,
                meet_or_slice: mos,
            };
            let vb = ViewBox::new(0.0, 0.0, 10.0, 10.0);
            assert_eq!(fit(vb, (0.0, 0.0, 40.0, 20.0), par), (4.0, 2.0, 0.0, 0.0));
        }
    }

    #[test]
    fn fit_min_and_max_anchors() {
        let vb = ViewBox::new(0.0, 0.0, 10.0, 10.0);
        let e = (0.0, 0.0, 40.0, 20.0); // meet scale = 2, slack-x = 20
        let anchor = |align| PreserveAspectRatio {
            align,
            meet_or_slice: MeetOrSlice::Meet,
        };
        // xMin leaves translate-x at 0; xMax adds the full slack.
        assert_eq!(
            fit(vb, e, anchor(AspectRatioAlign::XMinYMin)),
            (2.0, 2.0, 0.0, 0.0)
        );
        assert_eq!(
            fit(vb, e, anchor(AspectRatioAlign::XMaxYMax)),
            (2.0, 2.0, 20.0, 0.0)
        );
        // Portrait viewport: slack moves to y.
        let e = (0.0, 0.0, 20.0, 40.0);
        assert_eq!(
            fit(vb, e, anchor(AspectRatioAlign::XMinYMax)),
            (2.0, 2.0, 0.0, 20.0)
        );
        assert_eq!(
            fit(vb, e, anchor(AspectRatioAlign::XMidYMid)),
            (2.0, 2.0, 0.0, 10.0)
        );
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
