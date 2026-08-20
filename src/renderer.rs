//! Top-level renderer: walks a [`VectorFrame`]'s scene graph and
//! produces a packed `Rgba` [`VideoFrame`].
//!
//! See the crate-level docs for the overall pipeline.

use oxideav_core::{
    FillRule, Frame, Group, ImageRef, MaskKind, Node, Paint, Path, PathNode, Rect, Rgba, Stroke,
    Transform2D, VectorFrame, VideoFrame, VideoPlane,
};

use crate::blend::BlendMode;
use crate::cache::{composite_key, CacheStats, RasterizedSubtree, SharedCache};
use crate::composite::composite_rgba_premultiplied_blend_rect;
use crate::fill::{rasterize_fill, rasterize_fill_with_bounds, AlphaMask};
use crate::filter::PreserveAspectRatio;
use crate::flatten::{flatten_path, FlatContour};
use crate::gradient::InterpolationSpace;
use crate::paint::{build_paint_lut, sample_paint_in, sample_paint_with_lut};
use crate::pattern::{
    sample_tile_bilinear_wrap, sample_tile_nearest_wrap, view_box_fit_transform, Pattern,
};
use crate::stroke::stroke_to_fill_path;

/// Top-level vector→raster renderer.
///
/// Mutates an internal `Vec<u8>` packed `Rgba` buffer through the
/// scene walk; returns the buffer wrapped in a
/// [`VideoFrame`](oxideav_core::VideoFrame) at the end of
/// [`Renderer::render`].
///
/// Holds a thread-safe bitmap cache shared across `render` calls — when
/// a [`Group`](oxideav_core::Group) carries a `cache_key` the rendered
/// children are memoised so a long-lived `Renderer` (subtitle compositor,
/// scene renderer) avoids re-rasterising the same glyph / decorative
/// subtree on every frame. See [`Renderer::with_cache_capacity`].
#[derive(Debug, Clone)]
pub struct Renderer {
    /// Output canvas width in pixels.
    pub width: u32,
    /// Output canvas height in pixels.
    pub height: u32,
    /// Per-axis-Y vertical supersampling factor for AA. 1, 2, 4, or 8.
    /// Defaults to 4. Other values are clamped to the closest valid
    /// value at fill time.
    pub supersampling: u8,
    /// Sub-pixel positioning toggle. Currently a no-op for the
    /// shape-rasterizer (subpixel positioning matters mostly for text
    /// glyphs); reserved for round-2 expansion.
    pub subpixel_positioning: bool,
    /// Initial canvas clear color. Defaults to fully transparent.
    pub background: Rgba,
    /// Texture sampling filter for `Node::Image` paints. Defaults to
    /// [`ImageFilter::Bilinear`]; nearest-neighbour is available for
    /// callers that explicitly want pixel-perfect block-replication.
    pub image_filter: ImageFilter,
    /// Color-interpolation space for gradient (and future per-pixel
    /// paint) stops. Defaults to [`InterpolationSpace::Srgb`] (the
    /// historical SVG 1.1 default — fast, matches naive renderers).
    /// Set to [`InterpolationSpace::LinearRgb`] for SVG 2 §13.9
    /// `color-interpolation: linearRGB` behaviour (avoids the dark
    /// midpoint where complementary primaries cross).
    pub color_interpolation: InterpolationSpace,
    /// Per-paint compositing blend mode (PDF 32000-1:2008 §11.3.5 /
    /// W3C Compositing-1 §10–§11 standard modes — 12 separable plus
    /// 4 non-separable HSL modes). Defaults to [`BlendMode::Normal`]
    /// — the historical source-over operator matching SVG 1.1 and the
    /// CSS `mix-blend-mode: normal` default; any other value routes
    /// the composite through the per-pixel
    /// [`blend_over`](crate::blend_over) evaluation. Applied uniformly
    /// to fill, stroke, and image paints in the current renderer; SVG 2
    /// `mix-blend-mode` per-element override is a future extension.
    pub blend_mode: BlendMode,
    /// How the frame's `viewBox` is fitted into the output canvas
    /// (SVG 1.1 §7.8 / §8.2 `preserveAspectRatio`). Defaults to
    /// `xMidYMid meet` — the SVG §7.8 / §15.18 default ("If attribute
    /// `preserveAspectRatio` is not specified, then the effect is as if
    /// a value of xMidYMid meet were specified"): the viewBox is scaled
    /// uniformly to fit entirely within the canvas (letterboxed) and
    /// centred. Set to [`AspectRatioAlign::None`] for the legacy
    /// non-uniform stretch (viewBox bounding box mapped exactly onto the
    /// canvas rectangle), or to a `slice` value to cover the whole
    /// canvas with overflow cropped at the canvas edge (§8.6
    /// `overflow: hidden`). Has no effect when the frame carries no
    /// `viewBox` (user-space is already in pixels).
    pub preserve_aspect_ratio: PreserveAspectRatio,
    /// Bitmap cache for memoised group subtrees. Shared via
    /// `Arc<Mutex<...>>` so the cache survives `Clone`, and so the
    /// cache lookup remains coherent across the scene walk's
    /// (otherwise immutable) `&self` borrow.
    cache: SharedCache,
}

/// Texture sampling filter for `Node::Image` paints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImageFilter {
    /// Nearest-neighbour. Pixel-perfect block-replication; aliasing
    /// for non-integer scales but useful for low-resolution sprite
    /// rendering.
    Nearest,
    /// Bilinear. 4-tap weighted by sub-pixel position; the default,
    /// matching CSS `image-rendering: auto` and the SVG spec.
    #[default]
    Bilinear,
    /// Lanczos2 (windowed sinc, a = 2). 4×4 separable kernel
    /// `lanczos2(x) = sinc(x) * sinc(x/2)` for `|x| < 2`. Sharper than
    /// bilinear with the windowed-sinc impulse response — matches the
    /// CSS `image-rendering: high-quality` hint. Slightly slower (16
    /// taps vs 4) and may overshoot, so the per-pixel result is
    /// clamped back into `[0, 255]` per channel.
    Lanczos2,
    /// Lanczos3 (windowed sinc, a = 3). 6×6 separable kernel
    /// `lanczos3(x) = sinc(π·x) * sinc(π·x/3)` for `|x| < 3`. The wider
    /// 6-tap window captures more of the sinc's main lobe (and a piece
    /// of the first negative side-lobe), which gives noticeably sharper
    /// reconstruction than Lanczos2 with a less abrupt impulse-response
    /// truncation — the standard "high-quality" choice in image-
    /// processing literature (e.g. Turkowski, "Filters for Common
    /// Resampling Tasks", 1990; Duchon, "Lanczos filtering in one and
    /// two dimensions", 1979). The price is 36 taps per output pixel
    /// vs Lanczos2's 16, and a stronger second negative lobe that can
    /// overshoot further — the per-pixel result is clamped back into
    /// `[0, 255]` per channel, identical to the Lanczos2 path. Best for
    /// high-quality downscaling of photographic content where Lanczos2
    /// still alias-shows on fine repeating texture.
    Lanczos3,
    /// Mitchell–Netravali bicubic with the classic `B = 1/3, C = 1/3`
    /// parameters (the "Mitchell" point in the Mitchell–Netravali
    /// 1988 reconstruction-filter family — minimises a balanced sum of
    /// blur + ringing). 4×4 separable kernel; smoother than Lanczos2
    /// (less ringing, no negative side-lobes large enough to overshoot
    /// in practice) and sharper than bilinear. Useful as the default
    /// for *downscaling* photographic content where Lanczos2's
    /// negative lobes can show as halo banding.
    Mitchell,
    /// Catmull–Rom bicubic — the `B = 0, C = 1/2` interpolating member
    /// of the same Mitchell–Netravali (1988) BC family. Unlike Mitchell
    /// it passes exactly through the source samples (`k(0) = 1`,
    /// `k(±1) = k(±2) = 0`), so it preserves crisp edges without the
    /// blur Mitchell's `B = 1/3` term introduces. The price is a
    /// stronger negative side-lobe (more ringing potential than
    /// Mitchell), absorbed by the sampler's per-channel `[0, 255]`
    /// clamp. A good default for *upscaling* line art / UI where edge
    /// sharpness matters more than maximal smoothness.
    CatmullRom,
    /// Cubic B-spline — the `B = 1, C = 0` *approximating* member of the
    /// same Mitchell–Netravali (1988) BC family. It is the smoothest of
    /// the three canonical points: the kernel is everywhere non-negative
    /// (no ringing side-lobes, so no overshoot and nothing to clamp), at
    /// the cost of the strongest blur. Unlike Catmull–Rom it does *not*
    /// pass through the source samples (`k(0) = 4/6 ≈ 0.667`, not 1), so
    /// it smooths rather than interpolates. The best choice for
    /// noise-suppressing *downscaling* or anywhere a halo-free,
    /// monotone-preserving reconstruction matters more than edge
    /// sharpness (e.g. heavy minification of photographic content where
    /// even Mitchell's small negative lobe would alias). 4×4 separable
    /// kernel; same premultiplied-alpha machinery as Mitchell /
    /// Catmull–Rom.
    BSpline,
}

/// Default cache capacity (entries) when none is specified.
pub const DEFAULT_CACHE_CAPACITY: usize = 256;

impl Renderer {
    /// Build a renderer for the given destination size with sane
    /// defaults (`supersampling = 4`, transparent background, no
    /// sub-pixel positioning, bilinear image filtering, 256-entry
    /// bitmap cache).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            supersampling: 4,
            subpixel_positioning: false,
            background: Rgba::new(0, 0, 0, 0),
            image_filter: ImageFilter::default(),
            color_interpolation: InterpolationSpace::default(),
            blend_mode: BlendMode::default(),
            preserve_aspect_ratio: PreserveAspectRatio::default(),
            cache: SharedCache::with_capacity(DEFAULT_CACHE_CAPACITY),
        }
    }

    /// Build a renderer with the bitmap cache sized to `capacity`
    /// entries. Useful for long-lived consumers (subtitle compositor,
    /// scene renderer) that want to widen the cache footprint, or for
    /// short-lived one-shot renders that want to disable it
    /// (`capacity = 1` keeps the smallest possible LRU; the cache
    /// itself never disables — `Group::cache_key = None` is the
    /// per-node opt-out).
    pub fn with_cache_capacity(width: u32, height: u32, capacity: usize) -> Self {
        Self {
            cache: SharedCache::with_capacity(capacity),
            ..Self::new(width, height)
        }
    }

    /// Snapshot the cache's hit / miss / occupancy statistics. Useful
    /// for tests and tuning.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Reset the cache's hit / miss counters. The cached entries are
    /// preserved; only the running tallies are zeroed.
    pub fn reset_cache_stats(&self) {
        self.cache.reset_stats();
    }

    /// Set the [`Self::preserve_aspect_ratio`] viewport-fitting policy
    /// (SVG 1.1 §7.8) and return `self`, for builder-style construction.
    pub fn with_preserve_aspect_ratio(mut self, par: PreserveAspectRatio) -> Self {
        self.preserve_aspect_ratio = par;
        self
    }

    /// Render a [`VectorFrame`] into a packed `Rgba` `VideoFrame`.
    ///
    /// The frame's `view_box` (when present) is used to derive the
    /// user-space → raster-space transform; without one the canvas is
    /// assumed to be `(0, 0, frame.width, frame.height)`.
    pub fn render(&self, frame: &VectorFrame) -> VideoFrame {
        let stride = (self.width as usize) * 4;
        let mut buf = vec![0u8; stride * (self.height as usize)];
        // Background fill.
        if self.background.a != 0 {
            for y in 0..self.height {
                let row = (y as usize) * stride;
                for x in 0..self.width {
                    let i = row + (x as usize) * 4;
                    buf[i] = self.background.r;
                    buf[i + 1] = self.background.g;
                    buf[i + 2] = self.background.b;
                    buf[i + 3] = self.background.a;
                }
            }
        }
        // Compute the user-space → raster transform from the view box
        // (or default to "user space already in pixels"). The viewBox is
        // fitted into the (0, 0, width, height) viewport rectangle under
        // the configured `preserveAspectRatio` policy (SVG 1.1 §7.8 /
        // §8.2). The default `xMidYMid meet` scales uniformly and centres
        // (letterboxing when the aspect ratios differ); `none` reproduces
        // the legacy non-uniform stretch; a `slice` value covers the
        // whole canvas with the overflow cropped at the canvas edge
        // (§8.6 `overflow: hidden`, which the fixed-size output buffer
        // enforces for free).
        let initial = if let Some(vb) = frame.view_box {
            // Degenerate viewBox extents disable rendering (§8.6); guard
            // the division so a zero/negative/non-finite extent doesn't
            // produce a NaN transform.
            if !(vb.width > 0.0 && vb.height > 0.0 && vb.width.is_finite() && vb.height.is_finite())
            {
                return VideoFrame {
                    pts: frame.pts,
                    planes: vec![VideoPlane { stride, data: buf }],
                };
            }
            view_box_fit_transform(
                &vb,
                0.0,
                0.0,
                self.width as f32,
                self.height as f32,
                self.preserve_aspect_ratio,
            )
        } else {
            Transform2D::identity()
        };
        self.draw_group(&frame.root, initial, 1.0, &mut buf, stride, None);
        VideoFrame {
            pts: frame.pts,
            planes: vec![VideoPlane { stride, data: buf }],
        }
    }

    /// Render a single node at the given transform. Returns a packed
    /// `Rgba` `VideoFrame` of the renderer's full canvas size, with
    /// only that node painted onto the `background` clear.
    pub fn render_node(&self, node: &Node, transform: Transform2D) -> VideoFrame {
        let stride = (self.width as usize) * 4;
        let mut buf = vec![0u8; stride * (self.height as usize)];
        if self.background.a != 0 {
            for y in 0..self.height {
                let row = (y as usize) * stride;
                for x in 0..self.width {
                    let i = row + (x as usize) * 4;
                    buf[i] = self.background.r;
                    buf[i + 1] = self.background.g;
                    buf[i + 2] = self.background.b;
                    buf[i + 3] = self.background.a;
                }
            }
        }
        self.draw_node(node, transform, 1.0, &mut buf, stride, None);
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data: buf }],
        }
    }

    fn draw_group(
        &self,
        g: &Group,
        parent_transform: Transform2D,
        parent_opacity: f32,
        buf: &mut [u8],
        stride: usize,
        clip_mask: Option<&AlphaMask>,
    ) {
        let local = parent_transform.compose(&g.transform);
        // Bitmap-cache fast path: when the group is tagged with a
        // `cache_key`, memoise its rendered children under
        // `composite_key(g.cache_key, local)` so a re-render at the
        // same effective transform reuses the prior bitmap. The cached
        // bitmap is the group at *full opacity*; neither `g.opacity` nor
        // the parent's opacity is baked in, so an opacity change never
        // invalidates the cache. `g.opacity` and the parent opacity are
        // both applied uniformly at blit time (along with the parent
        // clip mask), exactly matching the SVG group-opacity model.
        if let Some(key) = g.cache_key {
            let composite = composite_key(key, &local);
            let bitmap = match self.cache.get(composite) {
                Some(b) => b,
                None => self.render_group_offscreen(g, local, composite),
            };
            blit_rgba_over(
                buf,
                stride,
                self.width,
                self.height,
                &bitmap.rgba,
                bitmap.width,
                bitmap.height,
                bitmap.offset_x,
                bitmap.offset_y,
                clip_mask,
                parent_opacity * g.opacity,
            );
            return;
        }
        // Group-opacity isolation (SVG 2 §3.4 "group opacity" / Simple
        // Alpha Compositing): when the group is partially transparent we
        // must render its children into an offscreen image *at full
        // opacity*, then blend that image onto the canvas as a unit with
        // `g.opacity` applied uniformly. Pushing `g.opacity` onto each
        // child individually (the cheap direct path below) double-darkens
        // the overlap region of any two overlapping children, which the
        // spec's post-process model forbids. The group's own clip is
        // baked into the offscreen; the inherited parent clip is applied
        // at blit time so it intersects the composited unit.
        if g.opacity < 1.0 {
            let bitmap = self.rasterize_group_to_subtree(g, local);
            blit_rgba_over(
                buf,
                stride,
                self.width,
                self.height,
                &bitmap.rgba,
                bitmap.width,
                bitmap.height,
                bitmap.offset_x,
                bitmap.offset_y,
                clip_mask,
                parent_opacity * g.opacity,
            );
            return;
        }
        // Build the group's own clip mask if it has one. The clip
        // mask is filled with NonZero of the clip path under the
        // *current* (post-transform) coordinate system, intersected
        // with whatever clip mask is being inherited.
        let group_clip_storage: Option<AlphaMask> = g.clip.as_ref().map(|clip_path| {
            let cs = flatten_path(&clip_path.commands, &local);
            let m = rasterize_fill(
                &cs,
                self.width,
                self.height,
                FillRule::NonZero,
                self.supersampling,
            );
            match clip_mask {
                Some(parent) => intersect_masks(parent, &m),
                None => m,
            }
        });
        let effective_clip: Option<&AlphaMask> = group_clip_storage.as_ref().or(clip_mask);
        // Reached only when `g.opacity == 1.0` (the partially-transparent
        // case isolated offscreen above), so the group contributes no
        // extra alpha and children inherit the parent opacity directly.
        for child in &g.children {
            self.draw_node(child, local, parent_opacity, buf, stride, effective_clip);
        }
    }

    /// Render `g`'s children onto a fresh canvas-sized RGBA buffer at
    /// `local` transform — at **full opacity**, with only the group's
    /// own clip path applied (the inherited parent clip and the group's
    /// `opacity` are applied uniformly at blit time, per the SVG group-
    /// opacity rendering model). Then scan the offscreen buffer for the
    /// touched-pixel bounding box, allocate a tight crop, and return it
    /// as a [`RasterizedSubtree`].
    ///
    /// Rendering children at full opacity (rather than pre-multiplying
    /// each child by `g.opacity`) is what makes group opacity behave as
    /// a *post-process* on the composited group rather than a per-child
    /// alpha — overlapping children inside the group no longer
    /// double-darken in their overlap region (SVG 2 §3.4 / "Simple
    /// Alpha Compositing": "the group ... is rendered into an RGBA
    /// offscreen image[;] the offscreen image as [a] whole is then
    /// blended into the canvas with the specified opacity value used
    /// uniformly across the offscreen image").
    fn rasterize_group_to_subtree(&self, g: &Group, local: Transform2D) -> RasterizedSubtree {
        let stride = (self.width as usize) * 4;
        let mut off = vec![0u8; stride * (self.height as usize)];
        // Apply the group's own clip mask only — the parent clip is
        // handled at blit time.
        let group_clip_storage: Option<AlphaMask> = g.clip.as_ref().map(|clip_path| {
            let cs = flatten_path(&clip_path.commands, &local);
            rasterize_fill(
                &cs,
                self.width,
                self.height,
                FillRule::NonZero,
                self.supersampling,
            )
        });
        let effective_clip = group_clip_storage.as_ref();
        for child in &g.children {
            self.draw_node(child, local, 1.0, &mut off, stride, effective_clip);
        }
        crop_to_bbox(&off, stride, self.width, self.height, local)
    }

    /// Cache-aware wrapper around [`Self::rasterize_group_to_subtree`]:
    /// rasterises the group offscreen and stores the crop in the bitmap
    /// cache under `composite` (saves significant memory for tiny glyphs
    /// in large canvases — a 16 px glyph on a 4096 px canvas drops from
    /// 64 MB to ~1 KB per cache entry). The cached bitmap is the group
    /// at full opacity; callers apply `g.opacity` at blit time so an
    /// opacity change never invalidates the cache.
    fn render_group_offscreen(
        &self,
        g: &Group,
        local: Transform2D,
        composite: u64,
    ) -> RasterizedSubtree {
        let subtree = self.rasterize_group_to_subtree(g, local);
        self.cache.put(composite, subtree.clone());
        subtree
    }

    fn draw_node(
        &self,
        node: &Node,
        transform: Transform2D,
        group_opacity: f32,
        buf: &mut [u8],
        stride: usize,
        clip_mask: Option<&AlphaMask>,
    ) {
        match node {
            Node::Path(p) => self.draw_path(p, transform, group_opacity, buf, stride, clip_mask),
            Node::Group(g) => {
                self.draw_group(g, transform, group_opacity, buf, stride, clip_mask);
            }
            Node::Image(img) => {
                self.draw_image(img, transform, group_opacity, buf, stride, clip_mask)
            }
            Node::SoftMask {
                mask,
                mask_kind,
                content,
            } => {
                self.draw_soft_mask(
                    mask,
                    *mask_kind,
                    content,
                    transform,
                    group_opacity,
                    buf,
                    stride,
                    clip_mask,
                );
            }
            // `Node` is #[non_exhaustive]; future variants (text,
            // filters) silently no-op until handled.
            _ => {}
        }
    }

    /// Render a soft-mask composite: rasterise `mask` and `content`
    /// to separate offscreen RGBA buffers, convert the mask buffer to
    /// a per-pixel coverage byte (luminance-weighted Y or alpha
    /// channel, per `kind`), then blit the content buffer on top of
    /// `buf` with that coverage as a per-pixel modulator (multiplied
    /// in with `group_opacity` and the inherited `clip_mask`).
    fn draw_soft_mask(
        &self,
        mask: &Node,
        kind: MaskKind,
        content: &Node,
        transform: Transform2D,
        group_opacity: f32,
        buf: &mut [u8],
        stride: usize,
        clip_mask: Option<&AlphaMask>,
    ) {
        let off_stride = (self.width as usize) * 4;
        let cap = off_stride * (self.height as usize);
        // Rasterise the mask subtree.
        let mut mask_buf = vec![0u8; cap];
        self.draw_node(mask, transform, 1.0, &mut mask_buf, off_stride, None);
        // Rasterise the content subtree.
        let mut content_buf = vec![0u8; cap];
        self.draw_node(content, transform, 1.0, &mut content_buf, off_stride, None);
        // Restrict the coverage conversion + blit to the intersection
        // of the content's and the mask's touched-alpha bounding
        // boxes: outside the content box the blit skips every pixel
        // (source alpha 0) no matter what the coverage says, and
        // outside the mask box the coverage is provably 0 (both mask
        // kinds multiply by / read the mask pixel's alpha), so the
        // blit skips there too — the skipped pixels are exactly the
        // ones the full scan would have skipped, and the output bytes
        // are unchanged.
        let Some((cx0, cy0, cx1, cy1)) =
            alpha_bbox(&content_buf, off_stride, self.width, self.height)
        else {
            return; // fully-transparent content — the blit would no-op
        };
        let Some((mx0, my0, mx1, my1)) = alpha_bbox(&mask_buf, off_stride, self.width, self.height)
        else {
            return; // fully-transparent mask — zero coverage everywhere
        };
        let bbox = (cx0.max(mx0), cy0.max(my0), cx1.min(mx1), cy1.min(my1));
        if bbox.0 > bbox.2 || bbox.1 > bbox.3 {
            return; // disjoint content / mask — nothing is composited
        }
        // Build a coverage AlphaMask from the mask buffer (inside the
        // content bbox only; everywhere else it stays 0, unread).
        let cov =
            mask_buffer_to_alpha_rect(&mask_buf, self.width, self.height, off_stride, kind, bbox);
        // Intersect with any inherited clip (again only inside the
        // bbox — the per-pixel min is only ever read there).
        let cov = match clip_mask {
            Some(c) => {
                let mut out = AlphaMask::new(self.width, self.height);
                let (bx0, by0, bx1, by1) = bbox;
                for y in by0..=by1 {
                    let row = (y * self.width) as usize;
                    for x in bx0..=bx1 {
                        let i = row + x as usize;
                        out.data[i] = c.data[i].min(cov.data[i]);
                    }
                }
                out
            }
            None => cov,
        };
        // Blit content_buf onto buf, modulated by cov (soft-mask
        // coverage, treated as a per-pixel alpha multiplier exactly
        // like a clip mask) + group_opacity.
        blit_rgba_over_rect(
            buf,
            stride,
            self.width,
            self.height,
            &content_buf,
            self.width,
            self.height,
            0,
            0,
            Some(&cov),
            group_opacity,
            bbox,
        );
    }

    fn draw_path(
        &self,
        node: &PathNode,
        transform: Transform2D,
        group_opacity: f32,
        buf: &mut [u8],
        stride: usize,
        clip_mask: Option<&AlphaMask>,
    ) {
        // Fill pass.
        if let Some(fill) = &node.fill {
            let contours = flatten_path(&node.path.commands, &transform);
            let (mask, bounds) = rasterize_fill_with_bounds(
                &contours,
                self.width,
                self.height,
                node.fill_rule,
                self.supersampling,
            );
            let mask = match clip_mask {
                Some(c) => intersect_masks(c, &mask),
                None => mask,
            };
            self.composite_with_paint(buf, stride, &mask, bounds, fill, group_opacity);
        }
        // Stroke pass.
        if let Some(stroke) = &node.stroke {
            let stroke_geom = build_stroke_geometry(&node.path, &transform, stroke);
            let (mask, bounds) = rasterize_fill_with_bounds(
                &stroke_geom,
                self.width,
                self.height,
                FillRule::NonZero,
                self.supersampling,
            );
            let mask = match clip_mask {
                Some(c) => intersect_masks(c, &mask),
                None => mask,
            };
            self.composite_with_paint(buf, stride, &mask, bounds, &stroke.paint, group_opacity);
        }
    }

    fn composite_with_paint(
        &self,
        buf: &mut [u8],
        stride: usize,
        mask: &AlphaMask,
        bounds: Option<(u32, u32, u32, u32)>,
        paint: &Paint,
        group_opacity: f32,
    ) {
        if mask.is_empty() {
            return;
        }
        // No span was rasterised: the mask is all-zero, so the
        // composite scan would skip every pixel anyway.
        let Some((bx0, by0, bx1, by1)) = bounds else {
            return;
        };
        let rect = (bx0, by0, bx1, by1);
        // Clone gradient-bearing paints so the per-pixel sampler
        // doesn't have to re-resolve at each call.
        let blend = self.blend_mode;
        match paint {
            Paint::Solid(c) => {
                let c = *c;
                composite_rgba_premultiplied_blend_rect(
                    buf,
                    stride,
                    self.width,
                    self.height,
                    mask,
                    0,
                    0,
                    group_opacity,
                    blend,
                    rect,
                    move |_x, _y| c,
                );
            }
            // Gradient + future non-Solid variants. Cloned upfront so
            // the closure owns its data and survives the composite
            // call's lifetime. Gradients additionally pre-bake a
            // 256-entry stops LUT once per fill — this collapses the
            // per-pixel stops-window scan + (for `LinearRgb`) the
            // `srgb_to_linear` / `linear_to_srgb` `powf` evaluation
            // down to a single clamped table lookup. The build cost
            // is `O(stops × 256)` (256× per-stop arithmetic);
            // amortised against `O(canvas-area)` per-pixel calls this
            // pays for itself any time the gradient covers more than
            // ~10 destination pixels, which is essentially always in
            // practice.
            other => {
                let space = self.color_interpolation;
                let lut = build_paint_lut(other, space);
                let other = other.clone();
                match lut {
                    Some(lut) => composite_rgba_premultiplied_blend_rect(
                        buf,
                        stride,
                        self.width,
                        self.height,
                        mask,
                        0,
                        0,
                        group_opacity,
                        blend,
                        rect,
                        move |x, y| {
                            sample_paint_with_lut(&other, x as f32 + 0.5, y as f32 + 0.5, &lut)
                        },
                    ),
                    None => composite_rgba_premultiplied_blend_rect(
                        buf,
                        stride,
                        self.width,
                        self.height,
                        mask,
                        0,
                        0,
                        group_opacity,
                        blend,
                        rect,
                        move |x, y| sample_paint_in(&other, x as f32 + 0.5, y as f32 + 0.5, space),
                    ),
                }
            }
        }
    }

    fn draw_image(
        &self,
        img: &ImageRef,
        transform: Transform2D,
        group_opacity: f32,
        buf: &mut [u8],
        stride: usize,
        clip_mask: Option<&AlphaMask>,
    ) {
        // Build a rectangle for the image's user-space bounds, fill
        // it with NonZero, then sample the embedded frame as the
        // paint source. Sampling filter is configurable on the
        // [`Renderer`] — defaults to bilinear, with nearest-neighbour
        // available for callers that want pixel-perfect block
        // replication.
        let rect_path = rect_to_path(img.bounds);
        let local = transform.compose(&img.transform);
        let contours = flatten_path(&rect_path.commands, &local);
        let (mask, fill_bounds) = rasterize_fill_with_bounds(
            &contours,
            self.width,
            self.height,
            FillRule::NonZero,
            self.supersampling,
        );
        let mask = match clip_mask {
            Some(c) => intersect_masks(c, &mask),
            None => mask,
        };
        let Some((bx0, by0, bx1, by1)) = fill_bounds else {
            return;
        };
        // Inverse transform pixel → user-space → image local UV.
        let inv = match invert_2d(&local) {
            Some(t) => t,
            None => return,
        };
        let bounds = img.bounds;
        let frame: &VideoFrame = &img.frame;
        let filter = self.image_filter;
        let blend = self.blend_mode;
        // Axis-aligned fast path: when the image transform has no
        // rotation / shear component, the texture coordinate `u`
        // depends only on the destination column and `v` only on the
        // destination row, so the per-axis filter taps (positions +
        // weights) can be computed once per column / row instead of
        // once per pixel. The cached values are produced by the exact
        // arithmetic the per-pixel samplers use, and the 2-D
        // accumulation walks the taps in the same order, so the
        // output bytes are identical — only the redundant per-pixel
        // weight recomputation (including the `sin` evaluations of
        // the Lanczos kernels) is eliminated.
        if local.b == 0.0 && local.c == 0.0 {
            self.composite_image_axis_aligned(
                buf,
                stride,
                &mask,
                (bx0, by0, bx1, by1),
                group_opacity,
                frame,
                &bounds,
                &inv,
            );
            return;
        }
        composite_rgba_premultiplied_blend_rect(
            buf,
            stride,
            self.width,
            self.height,
            &mask,
            0,
            0,
            group_opacity,
            blend,
            (bx0, by0, bx1, by1),
            move |x, y| {
                let user = inv.apply(oxideav_core::Point::new(x as f32 + 0.5, y as f32 + 0.5));
                match filter {
                    ImageFilter::Nearest => sample_image_nearest(frame, &bounds, user.x, user.y),
                    ImageFilter::Bilinear => sample_image_bilinear(frame, &bounds, user.x, user.y),
                    ImageFilter::Lanczos2 => sample_image_lanczos2(frame, &bounds, user.x, user.y),
                    ImageFilter::Lanczos3 => sample_image_lanczos3(frame, &bounds, user.x, user.y),
                    ImageFilter::Mitchell => sample_image_mitchell(frame, &bounds, user.x, user.y),
                    ImageFilter::CatmullRom => {
                        sample_image_catmull_rom(frame, &bounds, user.x, user.y)
                    }
                    ImageFilter::BSpline => sample_image_b_spline(frame, &bounds, user.x, user.y),
                }
            },
        );
    }

    /// Axis-aligned `Node::Image` composite: per-axis tap caching for
    /// the separable sampling kernels (see the call site in
    /// [`Self::draw_image`] for the bit-exactness argument).
    #[allow(clippy::too_many_arguments)]
    fn composite_image_axis_aligned(
        &self,
        buf: &mut [u8],
        stride: usize,
        mask: &AlphaMask,
        rect: (u32, u32, u32, u32),
        group_opacity: f32,
        frame: &VideoFrame,
        bounds: &Rect,
        inv: &Transform2D,
    ) {
        // Hoisted frame validation — identical checks to
        // `prepare_image_sample`; an invalid frame means every sample
        // is transparent, i.e. the composite writes nothing.
        if bounds.width <= 0.0 || bounds.height <= 0.0 {
            return;
        }
        let Some(plane) = frame.planes.first() else {
            return;
        };
        let tex_stride = plane.stride;
        if tex_stride < 4 {
            return;
        }
        let tex_w = (tex_stride / 4) as u32;
        let tex_h = (plane.data.len() / tex_stride) as u32;
        if tex_w == 0 || tex_h == 0 {
            return;
        }
        let data: &[u8] = &plane.data;
        let blend = self.blend_mode;

        // Per-column texture coordinate `u` (independent of y because
        // `local.c == 0`, hence `inv.c == 0`) and per-row `v`
        // (independent of x). The fixed 0.5 stand-in for the other
        // coordinate multiplies a (signed) zero matrix entry exactly
        // like any other positive in-canvas coordinate would, so the
        // result is the bit-identical `u` / `v` the per-pixel path
        // computes.
        let (bx0, by0, bx1, by1) = rect;
        let col_t = |x: u32| -> Option<f32> {
            let user = inv.apply(oxideav_core::Point::new(x as f32 + 0.5, 0.5));
            let u = (user.x - bounds.x) / bounds.width;
            if !(0.0..=1.0).contains(&u) {
                return None;
            }
            Some(u * tex_w as f32 - 0.5)
        };
        let row_t = |y: u32| -> Option<f32> {
            let user = inv.apply(oxideav_core::Point::new(0.5, y as f32 + 0.5));
            let v = (user.y - bounds.y) / bounds.height;
            if !(0.0..=1.0).contains(&v) {
                return None;
            }
            Some(v * tex_h as f32 - 0.5)
        };

        match self.image_filter {
            ImageFilter::Nearest => {
                // Nearest keeps its own (simpler) pixel fetch: cache
                // the clamped texel index per column / row.
                let cols: Vec<Option<usize>> = (bx0..=bx1)
                    .map(|x| {
                        let user = inv.apply(oxideav_core::Point::new(x as f32 + 0.5, 0.5));
                        let u = (user.x - bounds.x) / bounds.width;
                        if !(0.0..=1.0).contains(&u) {
                            return None;
                        }
                        Some(
                            ((u * tex_w as f32).floor() as i64).clamp(0, tex_w as i64 - 1) as usize,
                        )
                    })
                    .collect();
                let rows: Vec<Option<usize>> = (by0..=by1)
                    .map(|y| {
                        let user = inv.apply(oxideav_core::Point::new(0.5, y as f32 + 0.5));
                        let v = (user.y - bounds.y) / bounds.height;
                        if !(0.0..=1.0).contains(&v) {
                            return None;
                        }
                        Some(
                            ((v * tex_h as f32).floor() as i64).clamp(0, tex_h as i64 - 1) as usize,
                        )
                    })
                    .collect();
                composite_rgba_premultiplied_blend_rect(
                    buf,
                    stride,
                    self.width,
                    self.height,
                    mask,
                    0,
                    0,
                    group_opacity,
                    blend,
                    rect,
                    move |x, y| {
                        let (Some(px), Some(py)) =
                            (cols[(x - bx0) as usize], rows[(y - by0) as usize])
                        else {
                            return Rgba::new(0, 0, 0, 0);
                        };
                        let i = py * tex_stride + px * 4;
                        Rgba::new(data[i], data[i + 1], data[i + 2], data[i + 3])
                    },
                );
            }
            ImageFilter::Bilinear => self.composite_image_separable::<2>(
                buf,
                stride,
                mask,
                rect,
                group_opacity,
                data,
                tex_stride,
                &col_t,
                &row_t,
                &|t, extent| {
                    let x0 = t.floor() as i64;
                    let x1 = x0 + 1;
                    let w1 = (t - x0 as f32).clamp(0.0, 1.0);
                    (
                        [
                            x0.clamp(0, extent as i64 - 1) as usize,
                            x1.clamp(0, extent as i64 - 1) as usize,
                        ],
                        [1.0 - w1, w1],
                    )
                },
                (tex_w, tex_h),
                false,
            ),
            ImageFilter::Lanczos2 => self.composite_image_separable::<4>(
                buf,
                stride,
                mask,
                rect,
                group_opacity,
                data,
                tex_stride,
                &col_t,
                &row_t,
                &|t, extent| kernel_taps::<4>(t, extent, 2, lanczos2),
                (tex_w, tex_h),
                true,
            ),
            ImageFilter::Lanczos3 => self.composite_image_separable::<6>(
                buf,
                stride,
                mask,
                rect,
                group_opacity,
                data,
                tex_stride,
                &col_t,
                &row_t,
                &|t, extent| kernel_taps::<6>(t, extent, 3, lanczos3),
                (tex_w, tex_h),
                true,
            ),
            ImageFilter::Mitchell => self.composite_image_separable::<4>(
                buf,
                stride,
                mask,
                rect,
                group_opacity,
                data,
                tex_stride,
                &col_t,
                &row_t,
                &|t, extent| kernel_taps::<4>(t, extent, 2, mitchell_netravali),
                (tex_w, tex_h),
                true,
            ),
            ImageFilter::CatmullRom => self.composite_image_separable::<4>(
                buf,
                stride,
                mask,
                rect,
                group_opacity,
                data,
                tex_stride,
                &col_t,
                &row_t,
                &|t, extent| kernel_taps::<4>(t, extent, 2, catmull_rom),
                (tex_w, tex_h),
                true,
            ),
            ImageFilter::BSpline => self.composite_image_separable::<4>(
                buf,
                stride,
                mask,
                rect,
                group_opacity,
                data,
                tex_stride,
                &col_t,
                &row_t,
                &|t, extent| kernel_taps::<4>(t, extent, 2, b_spline),
                (tex_w, tex_h),
                true,
            ),
        }
    }

    /// Shared axis-aligned separable sampler: builds per-column and
    /// per-row tap caches with `taps`, then accumulates the W×W
    /// premultiplied-alpha footprint per covered pixel in the same
    /// `j`-outer / `i`-inner order (and with the same `wx * wy`
    /// product order) as the per-pixel samplers.
    ///
    /// `clamp_alpha` selects the accumulator tail: the windowed-sinc /
    /// BC-cubic kernels clamp the accumulated alpha into `[0, 255]`
    /// before the un-premultiply (their negative side-lobes can
    /// overshoot), while bilinear uses the raw accumulated alpha —
    /// matching each original sampler exactly.
    #[allow(clippy::too_many_arguments)]
    fn composite_image_separable<const W: usize>(
        &self,
        buf: &mut [u8],
        stride: usize,
        mask: &AlphaMask,
        rect: (u32, u32, u32, u32),
        group_opacity: f32,
        data: &[u8],
        tex_stride: usize,
        col_t: &dyn Fn(u32) -> Option<f32>,
        row_t: &dyn Fn(u32) -> Option<f32>,
        taps: &dyn Fn(f32, u32) -> ([usize; W], [f32; W]),
        tex_extent: (u32, u32),
        clamp_alpha: bool,
    ) {
        let (bx0, by0, bx1, by1) = rect;
        let (tex_w, tex_h) = tex_extent;
        let cols: Vec<Option<([usize; W], [f32; W])>> =
            (bx0..=bx1).map(|x| Some(taps(col_t(x)?, tex_w))).collect();
        let rows: Vec<Option<([usize; W], [f32; W])>> =
            (by0..=by1).map(|y| Some(taps(row_t(y)?, tex_h))).collect();
        composite_rgba_premultiplied_blend_rect(
            buf,
            stride,
            self.width,
            self.height,
            mask,
            0,
            0,
            group_opacity,
            self.blend_mode,
            rect,
            move |x, y| {
                let (Some((cpos, cw)), Some((rpos, rw))) =
                    (&cols[(x - bx0) as usize], &rows[(y - by0) as usize])
                else {
                    return Rgba::new(0, 0, 0, 0);
                };
                let mut acc = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for j in 0..W {
                    let py = rpos[j];
                    let wy = rw[j];
                    for i in 0..W {
                        let w = cw[i] * wy;
                        let (r, g, b, a) = fetch_premul(data, tex_stride, cpos[i], py);
                        acc.0 += r * w;
                        acc.1 += g * w;
                        acc.2 += b * w;
                        acc.3 += a * w;
                    }
                }
                let pa = if clamp_alpha {
                    acc.3.clamp(0.0, 255.0)
                } else {
                    acc.3
                };
                if pa <= 0.5 {
                    return Rgba::new(0, 0, 0, 0);
                }
                let inv_a = 255.0 / pa;
                Rgba::new(
                    (acc.0 * inv_a).round().clamp(0.0, 255.0) as u8,
                    (acc.1 * inv_a).round().clamp(0.0, 255.0) as u8,
                    (acc.2 * inv_a).round().clamp(0.0, 255.0) as u8,
                    pa.round().clamp(0.0, 255.0) as u8,
                )
            },
        );
    }

    /// Fill `path` with a tiled [`Pattern`] paint server (SVG 2 §14.3),
    /// returning a full-canvas RGBA frame with only that fill painted
    /// onto the renderer's `background` clear.
    ///
    /// `transform` is the user→device transform in effect for the
    /// element referencing the pattern; the pattern's own
    /// `patternTransform` is post-multiplied onto it (§14.3.1). A
    /// degenerate tile rectangle ([`Pattern::is_degenerate`]) paints
    /// nothing.
    ///
    /// Tile sampling honours [`Renderer::image_filter`] in reduced
    /// form: `Nearest` samples nearest-neighbour, every other filter
    /// uses seam-free periodic bilinear (the higher-order kernels are
    /// not wrap-aware).
    pub fn fill_path_with_pattern(
        &self,
        path: &Path,
        fill_rule: FillRule,
        pattern: &Pattern,
        transform: Transform2D,
    ) -> VideoFrame {
        let stride = (self.width as usize) * 4;
        let mut buf = vec![0u8; stride * (self.height as usize)];
        let contours = flatten_path(&path.commands, &transform);
        let (mask, bounds) = rasterize_fill_with_bounds(
            &contours,
            self.width,
            self.height,
            fill_rule,
            self.supersampling,
        );
        self.composite_with_pattern(&mut buf, stride, &mask, bounds, pattern, transform, 1.0);
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data: buf }],
        }
    }

    /// Stroke `path` with a tiled [`Pattern`] paint server (SVG 2
    /// §14.3), returning a full-canvas RGBA frame with only that
    /// stroke painted onto the renderer's `background` clear.
    ///
    /// The stroke geometry (width, caps, joins, miter limit, dashes)
    /// comes from `stroke`; its `paint` field is ignored — the pattern
    /// is the paint. See [`Renderer::fill_path_with_pattern`] for the
    /// transform and filtering semantics.
    pub fn stroke_path_with_pattern(
        &self,
        path: &Path,
        stroke: &Stroke,
        pattern: &Pattern,
        transform: Transform2D,
    ) -> VideoFrame {
        let stride = (self.width as usize) * 4;
        let mut buf = vec![0u8; stride * (self.height as usize)];
        let geom = build_stroke_geometry(path, &transform, stroke);
        let (mask, bounds) = rasterize_fill_with_bounds(
            &geom,
            self.width,
            self.height,
            FillRule::NonZero,
            self.supersampling,
        );
        self.composite_with_pattern(&mut buf, stride, &mask, bounds, pattern, transform, 1.0);
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data: buf }],
        }
    }

    /// Composite a [`Pattern`] paint through `mask` onto `buf`.
    ///
    /// Pipeline (see the [`crate::pattern`] module docs): rasterise one
    /// tile offscreen at device resolution, then for every covered
    /// destination pixel inverse-map the pixel centre through
    /// `transform ∘ patternTransform` into pattern space, reduce modulo
    /// the tile extent (`(q − tile_origin) mod tile_size` — the §14.3
    /// "tiles at `(x + m·width, y + n·height)` for all integers m, n"
    /// rule), and sample the tile with periodic addressing.
    fn composite_with_pattern(
        &self,
        buf: &mut [u8],
        stride: usize,
        mask: &AlphaMask,
        bounds: Option<(u32, u32, u32, u32)>,
        pattern: &Pattern,
        transform: Transform2D,
        group_opacity: f32,
    ) {
        if mask.is_empty() || pattern.is_degenerate() {
            return;
        }
        let Some(rect) = bounds else {
            return; // all-zero mask — nothing to composite
        };
        // Pattern space → device space: patternTransform is
        // post-multiplied (inserted to the right, §14.3.1).
        let pat_to_dev = transform.compose(&pattern.transform);
        let inv = match invert_2d(&pat_to_dev) {
            Some(t) => t,
            None => return,
        };
        // Device-resolution tile size from the affine's column norms
        // (the device lengths of the pattern-space unit vectors, exact
        // under rotation).
        let sx = (pat_to_dev.a * pat_to_dev.a + pat_to_dev.b * pat_to_dev.b).sqrt();
        let sy = (pat_to_dev.c * pat_to_dev.c + pat_to_dev.d * pat_to_dev.d).sqrt();
        let tw = ((pattern.width * sx).ceil() as i64).clamp(1, MAX_PATTERN_TILE_DIM) as u32;
        let th = ((pattern.height * sy).ceil() as i64).clamp(1, MAX_PATTERN_TILE_DIM) as u32;
        let tile = self.rasterize_pattern_tile(pattern, tw, th);
        let (ox, oy, pw, ph) = (pattern.x, pattern.y, pattern.width, pattern.height);
        let nearest = matches!(self.image_filter, ImageFilter::Nearest);
        let blend = self.blend_mode;
        // Axis-aligned fast path (same argument as the image sampler:
        // with no rotation / shear the tile coordinate `u` depends only
        // on the destination column and `v` only on the row, and the
        // per-axis wrap/tap arithmetic is cached from the identical
        // expressions, so output bytes are unchanged).
        if pat_to_dev.b == 0.0 && pat_to_dev.c == 0.0 {
            let Some((tw_px, th_px, tile_stride, data)) = crate::pattern::tile_dims(&tile) else {
                return; // degenerate tile — every sample is transparent
            };
            let (rx0, ry0, rx1, ry1) = rect;
            let col_u = |x: u32| -> f32 {
                let q = inv.apply(oxideav_core::Point::new(x as f32 + 0.5, 0.5));
                (q.x - ox) / pw
            };
            let row_v = |y: u32| -> f32 {
                let q = inv.apply(oxideav_core::Point::new(0.5, y as f32 + 0.5));
                (q.y - oy) / ph
            };
            if nearest {
                let cols: Vec<usize> = (rx0..=rx1)
                    .map(|x| crate::pattern::tile_nearest_axis(col_u(x), tw_px))
                    .collect();
                let rows: Vec<usize> = (ry0..=ry1)
                    .map(|y| crate::pattern::tile_nearest_axis(row_v(y), th_px))
                    .collect();
                composite_rgba_premultiplied_blend_rect(
                    buf,
                    stride,
                    self.width,
                    self.height,
                    mask,
                    0,
                    0,
                    group_opacity,
                    blend,
                    rect,
                    move |x, y| {
                        let px = cols[(x - rx0) as usize];
                        let py = rows[(y - ry0) as usize];
                        let i = py * tile_stride + px * 4;
                        Rgba::new(data[i], data[i + 1], data[i + 2], data[i + 3])
                    },
                );
            } else {
                let cols: Vec<(usize, usize, f32)> = (rx0..=rx1)
                    .map(|x| crate::pattern::tile_bilinear_axis(col_u(x), tw_px))
                    .collect();
                let rows: Vec<(usize, usize, f32)> = (ry0..=ry1)
                    .map(|y| crate::pattern::tile_bilinear_axis(row_v(y), th_px))
                    .collect();
                composite_rgba_premultiplied_blend_rect(
                    buf,
                    stride,
                    self.width,
                    self.height,
                    mask,
                    0,
                    0,
                    group_opacity,
                    blend,
                    rect,
                    move |x, y| {
                        crate::pattern::sample_tile_bilinear_taps(
                            data,
                            tile_stride,
                            cols[(x - rx0) as usize],
                            rows[(y - ry0) as usize],
                        )
                    },
                );
            }
            return;
        }
        composite_rgba_premultiplied_blend_rect(
            buf,
            stride,
            self.width,
            self.height,
            mask,
            0,
            0,
            group_opacity,
            blend,
            rect,
            move |x, y| {
                let q = inv.apply(oxideav_core::Point::new(x as f32 + 0.5, y as f32 + 0.5));
                // Normalised tile coordinates; the samplers wrap, so no
                // explicit rem_euclid is needed here.
                let u = (q.x - ox) / pw;
                let v = (q.y - oy) / ph;
                if nearest {
                    sample_tile_nearest_wrap(&tile, u, v)
                } else {
                    sample_tile_bilinear_wrap(&tile, u, v)
                }
            },
        );
    }

    /// Render one pattern tile into an offscreen `tw × th` RGBA buffer.
    ///
    /// Without a `viewBox`, content coordinates are tile-origin-relative
    /// (§14.3.2), so the only transform needed is the pattern-space →
    /// tile-pixel scale. With a `viewBox`, content coordinates are first
    /// fitted onto the tile rectangle by the §8.2 equivalent-transform
    /// algorithm ([`crate::pattern::view_box_fit_transform`]) under the
    /// pattern's `preserveAspectRatio`, then scaled to tile pixels. The
    /// offscreen canvas is exactly the tile rectangle, which realises
    /// the user-agent `overflow: hidden` clip for free (including the
    /// overhang a `slice` fitting produces).
    fn rasterize_pattern_tile(&self, pattern: &Pattern, tw: u32, th: u32) -> VideoFrame {
        let mut tile_renderer = Renderer::with_cache_capacity(tw, th, 1);
        tile_renderer.supersampling = self.supersampling;
        tile_renderer.image_filter = self.image_filter;
        tile_renderer.color_interpolation = self.color_interpolation;
        let scale = Transform2D::scale(
            tw as f32 / pattern.width.max(1e-9),
            th as f32 / pattern.height.max(1e-9),
        );
        let content_tf = match &pattern.view_box {
            Some(vb) => scale.compose(&crate::pattern::view_box_fit_transform(
                vb,
                0.0,
                0.0,
                pattern.width,
                pattern.height,
                pattern.preserve_aspect_ratio,
            )),
            None => scale,
        };
        let group = Group::new().with_children(pattern.content.clone());
        tile_renderer.render_node(&Node::Group(group), content_tf)
    }
}

/// Largest device-pixel extent of a rasterised pattern tile. A huge
/// pattern→device scale would otherwise ask for an unbounded offscreen
/// allocation; past this size the tile is rendered at the cap and
/// magnified by the per-pixel sampler.
const MAX_PATTERN_TILE_DIM: i64 = 4096;

/// Build the stroke geometry for `path` under `transform`, producing
/// closed contours ready for a NonZero fill.
fn build_stroke_geometry(
    path: &Path,
    transform: &Transform2D,
    stroke: &Stroke,
) -> Vec<FlatContour> {
    let contours = flatten_path(&path.commands, transform);
    if contours.is_empty() {
        return Vec::new();
    }
    // Approximate the per-pixel stroke width as `stroke.width *
    // average_scale` where `average_scale = 0.5 * (|a| + |d|)` for the
    // diagonal portion of the affine matrix. Adequate for uniform /
    // near-uniform scales; non-uniform scales get a slight bias that
    // round 2 may refine via a true principal-axis decomposition.
    let avg_scale = 0.5 * (transform.a.abs() + transform.d.abs());
    let width_px = stroke.width * avg_scale.max(1e-6);
    let mut out = Vec::new();
    for c in &contours {
        out.extend(stroke_to_fill_path(c, stroke, width_px));
    }
    out
}

/// Blit a packed straight-alpha RGBA source buffer over an RGBA
/// destination using premultiplied source-over.
///
/// `(offset_x, offset_y)` places the source bitmap's top-left at that
/// destination pixel. `clip_mask` (when `Some`) further multiplies the
/// source alpha per-pixel and is sampled in **destination** coordinates
/// (so a cached-and-cropped subtree blitted at an offset reads its clip
/// at the matching destination pixel). `extra_alpha` (in `0.0..=1.0`)
/// modulates the source alpha globally (used for parent group opacity).
/// The blit touches only pixels where the source alpha (after clip +
/// extra) is non-zero.
fn blit_rgba_over(
    dst: &mut [u8],
    dst_stride: usize,
    dst_width: u32,
    dst_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
    offset_x: u32,
    offset_y: u32,
    clip_mask: Option<&AlphaMask>,
    extra_alpha: f32,
) {
    if src_width == 0 || src_height == 0 {
        return;
    }
    blit_rgba_over_rect(
        dst,
        dst_stride,
        dst_width,
        dst_height,
        src,
        src_width,
        src_height,
        offset_x,
        offset_y,
        clip_mask,
        extra_alpha,
        (0, 0, src_width - 1, src_height - 1),
    )
}

/// [`blit_rgba_over`] restricted to the inclusive **source**-coordinate
/// rectangle `src_rect = (x0, y0, x1, y1)`. Callers pass a rectangle
/// outside which every source pixel has zero alpha, so the skipped
/// pixels are exactly the ones the full scan would have skipped.
#[allow(clippy::too_many_arguments)]
fn blit_rgba_over_rect(
    dst: &mut [u8],
    dst_stride: usize,
    dst_width: u32,
    dst_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
    offset_x: u32,
    offset_y: u32,
    clip_mask: Option<&AlphaMask>,
    extra_alpha: f32,
    src_rect: (u32, u32, u32, u32),
) {
    if src_width == 0 || src_height == 0 {
        return;
    }
    if offset_x >= dst_width || offset_y >= dst_height {
        return;
    }
    let extra = extra_alpha.clamp(0.0, 1.0);
    let extra_q = (extra * 255.0).round() as u32;
    if extra_q == 0 {
        return;
    }
    let w = src_width.min(dst_width - offset_x) as usize;
    let h = src_height.min(dst_height - offset_y) as usize;
    let src_stride = (src_width as usize) * 4;
    let (rx0, ry0, rx1, ry1) = src_rect;
    let y_lo = (ry0 as usize).min(h);
    let y_hi = ((ry1 as usize) + 1).min(h);
    let x_lo = (rx0 as usize).min(w);
    let x_hi = ((rx1 as usize) + 1).min(w);
    for y in y_lo..y_hi {
        let dst_row = (offset_y as usize + y) * dst_stride;
        let src_row = y * src_stride;
        for x in x_lo..x_hi {
            let si = src_row + x * 4;
            let sa_raw = src[si + 3] as u32;
            if sa_raw == 0 {
                continue;
            }
            let dx = offset_x + x as u32;
            let dy = offset_y + y as u32;
            let cov = match clip_mask {
                Some(c) => c.get(dx, dy) as u32,
                None => 255,
            };
            if cov == 0 {
                continue;
            }
            // src_alpha = sa_raw * cov * extra / (255 * 255)
            let combined = (sa_raw * cov * extra_q + (255 * 255 / 2)) / (255 * 255);
            if combined == 0 {
                continue;
            }
            if combined == 255 {
                // Fully-opaque source pixel: the over operator reduces
                // to a plain overwrite (with `sa = 255` the premultiply
                // is exact, `inv = 0` zeroes every destination term,
                // and the final un-premultiply divides by 255 giving
                // the source bytes back verbatim).
                let pidx = dst_row + (offset_x as usize + x) * 4;
                dst[pidx] = src[si];
                dst[pidx + 1] = src[si + 1];
                dst[pidx + 2] = src[si + 2];
                dst[pidx + 3] = 255;
                continue;
            }
            let sa = combined;
            // Promote source RGB into premultiplied space (src is
            // straight-alpha at sa_raw — re-premultiply with the
            // *combined* alpha so the over operator behaves).
            let sr = (src[si] as u32 * sa + 127) / 255;
            let sg = (src[si + 1] as u32 * sa + 127) / 255;
            let sb = (src[si + 2] as u32 * sa + 127) / 255;

            let pidx = dst_row + (offset_x as usize + x) * 4;
            let dr0 = dst[pidx] as u32;
            let dg0 = dst[pidx + 1] as u32;
            let db0 = dst[pidx + 2] as u32;
            let da0 = dst[pidx + 3] as u32;
            let dr = (dr0 * da0 + 127) / 255;
            let dg = (dg0 * da0 + 127) / 255;
            let db = (db0 * da0 + 127) / 255;

            let inv = 255 - sa;
            let or = sr + (dr * inv + 127) / 255;
            let og = sg + (dg * inv + 127) / 255;
            let ob = sb + (db * inv + 127) / 255;
            let oa = sa + (da0 * inv + 127) / 255;
            let or_s = (or * 255 + oa / 2)
                .checked_div(oa)
                .map_or(0, |v| v.min(255));
            let og_s = (og * 255 + oa / 2)
                .checked_div(oa)
                .map_or(0, |v| v.min(255));
            let ob_s = (ob * 255 + oa / 2)
                .checked_div(oa)
                .map_or(0, |v| v.min(255));
            dst[pidx] = or_s as u8;
            dst[pidx + 1] = og_s as u8;
            dst[pidx + 2] = ob_s as u8;
            dst[pidx + 3] = oa.min(255) as u8;
        }
    }
}

/// Scan a canvas-sized RGBA buffer for the touched-pixel bounding box
/// (any pixel with non-zero alpha), allocate a tight crop, and return
/// it as a [`RasterizedSubtree`].
///
/// Empty subtrees (no touched pixels) return `width = height = 0` with
/// an empty `rgba` vec — the blitter treats that as a no-op.
fn crop_to_bbox(
    src: &[u8],
    src_stride: usize,
    src_width: u32,
    src_height: u32,
    transform_at_cache_time: Transform2D,
) -> RasterizedSubtree {
    if src_width == 0 || src_height == 0 {
        return RasterizedSubtree {
            width: 0,
            height: 0,
            offset_x: 0,
            offset_y: 0,
            rgba: Vec::new(),
            transform_at_cache_time,
        };
    }
    let mut min_x = src_width;
    let mut max_x = 0u32; // inclusive max
    let mut min_y = src_height;
    let mut max_y = 0u32;
    let mut any = false;
    for y in 0..src_height {
        let row = (y as usize) * src_stride;
        for x in 0..src_width {
            let i = row + (x as usize) * 4;
            if src[i + 3] != 0 {
                any = true;
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }
    if !any {
        return RasterizedSubtree {
            width: 0,
            height: 0,
            offset_x: 0,
            offset_y: 0,
            rgba: Vec::new(),
            transform_at_cache_time,
        };
    }
    let w = max_x - min_x + 1;
    let h = max_y - min_y + 1;
    let dst_stride = (w as usize) * 4;
    let mut rgba = vec![0u8; dst_stride * (h as usize)];
    for y in 0..h {
        let src_row = ((min_y + y) as usize) * src_stride + (min_x as usize) * 4;
        let dst_row = (y as usize) * dst_stride;
        rgba[dst_row..dst_row + dst_stride].copy_from_slice(&src[src_row..src_row + dst_stride]);
    }
    RasterizedSubtree {
        width: w,
        height: h,
        offset_x: min_x,
        offset_y: min_y,
        rgba,
        transform_at_cache_time,
    }
}

/// Convert a packed RGBA offscreen buffer (the rasterised mask
/// subtree) into a single-channel coverage [`AlphaMask`].
///
/// * [`MaskKind::Luminance`]: Y = 0.2126·R + 0.7152·G + 0.0722·B
///   (BT.709), then multiplied by the pixel's own alpha so a
///   transparent mask pixel always contributes zero coverage. Matches
///   SVG `<mask mask-type="luminance">` (default) and PDF `SMask`
///   `/Luminosity` semantics.
/// * [`MaskKind::Alpha`]: the pixel's own alpha channel verbatim.
///   Matches SVG `<mask mask-type="alpha">` and PDF `SMask` `/Alpha`.
///
/// Restricted to the inclusive pixel rectangle `rect`; everything
/// outside stays `0`. Callers pair the result with a blit restricted
/// to the same rectangle, so the unconverted pixels are never read.
fn mask_buffer_to_alpha_rect(
    buf: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    kind: MaskKind,
    rect: (u32, u32, u32, u32),
) -> AlphaMask {
    let mut out = AlphaMask::new(width, height);
    let (rx0, ry0, rx1, ry1) = rect;
    let rx1 = rx1.min(width.saturating_sub(1));
    let ry1 = ry1.min(height.saturating_sub(1));
    if width == 0 || height == 0 || rx0 > rx1 || ry0 > ry1 {
        return out;
    }
    for y in ry0..=ry1 {
        let row = (y as usize) * stride;
        for x in rx0..=rx1 {
            let i = row + (x as usize) * 4;
            let cov = match kind {
                MaskKind::Alpha => buf[i + 3],
                MaskKind::Luminance => {
                    let r = buf[i] as f32;
                    let g = buf[i + 1] as f32;
                    let b = buf[i + 2] as f32;
                    let a = buf[i + 3] as f32 / 255.0;
                    let y_val = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    (y_val * a).round().clamp(0.0, 255.0) as u8
                }
            };
            out.data[(y * width + x) as usize] = cov;
        }
    }
    out
}

/// Scan a packed-RGBA buffer for the inclusive bounding box of every
/// pixel with non-zero alpha. Returns `None` when no pixel is touched.
fn alpha_bbox(buf: &[u8], stride: usize, width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = width;
    let mut max_x = 0u32;
    let mut min_y = height;
    let mut max_y = 0u32;
    let mut any = false;
    let w = width as usize;
    for y in 0..height {
        let row = (y as usize) * stride;
        // Short-circuit per row: first touched column from the left,
        // last touched column from the right. Dense rows resolve in a
        // few probes; empty rows cost one full scan.
        let Some(first) = (0..w).find(|&x| buf[row + x * 4 + 3] != 0) else {
            continue;
        };
        let last = (first..w)
            .rev()
            .find(|&x| buf[row + x * 4 + 3] != 0)
            .unwrap_or(first);
        any = true;
        min_x = min_x.min(first as u32);
        max_x = max_x.max(last as u32);
        min_y = min_y.min(y);
        max_y = y;
    }
    any.then_some((min_x, min_y, max_x, max_y))
}

/// Intersect two alpha masks (per-pixel min). Both must have the same
/// dimensions; mismatched inputs return an empty mask.
fn intersect_masks(a: &AlphaMask, b: &AlphaMask) -> AlphaMask {
    if a.width != b.width || a.height != b.height {
        return AlphaMask::default();
    }
    let mut out = AlphaMask::new(a.width, a.height);
    for i in 0..a.data.len() {
        out.data[i] = a.data[i].min(b.data[i]);
    }
    out
}

/// Build a closed rectangular [`Path`] for `r`.
fn rect_to_path(r: Rect) -> Path {
    let mut p = Path::new();
    p.move_to(oxideav_core::Point::new(r.x, r.y))
        .line_to(oxideav_core::Point::new(r.x + r.width, r.y))
        .line_to(oxideav_core::Point::new(r.x + r.width, r.y + r.height))
        .line_to(oxideav_core::Point::new(r.x, r.y + r.height))
        .close();
    p
}

/// Inverse of an affine 2D transform. Returns `None` if singular.
fn invert_2d(t: &Transform2D) -> Option<Transform2D> {
    let det = t.a * t.d - t.b * t.c;
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let a = t.d * inv_det;
    let b = -t.b * inv_det;
    let c = -t.c * inv_det;
    let d = t.a * inv_det;
    let e = -(a * t.e + c * t.f);
    let f = -(b * t.e + d * t.f);
    Some(Transform2D { a, b, c, d, e, f })
}

/// Nearest-neighbour sample of an `Rgba` `VideoFrame` (the only
/// pixel format our composite path targets directly). UV lookup is
/// done in the image's user-space rectangle. Out-of-bounds returns
/// transparent.
fn sample_image_nearest(frame: &VideoFrame, bounds: &Rect, ux: f32, uy: f32) -> Rgba {
    let info = match prepare_image_sample(frame, bounds, ux, uy) {
        Some(i) => i,
        None => return Rgba::new(0, 0, 0, 0),
    };
    let px = ((info.u * info.width as f32).floor() as i64).clamp(0, info.width as i64 - 1) as usize;
    let py =
        ((info.v * info.height as f32).floor() as i64).clamp(0, info.height as i64 - 1) as usize;
    let i = py * info.stride + px * 4;
    Rgba::new(
        info.data[i],
        info.data[i + 1],
        info.data[i + 2],
        info.data[i + 3],
    )
}

/// Bilinear sample of an `Rgba` `VideoFrame`. Samples in
/// premultiplied-alpha space (so the colour stays meaningful when one
/// of the 4 taps is fully transparent), then un-premultiplies for the
/// straight-alpha return.
///
/// Uses *clamp-to-edge* for samples whose 2×2 footprint extends past
/// the texture boundary, matching CSS / SVG default sampling.
/// Out-of-bounds (the user point is outside `bounds`) returns
/// transparent — same as the nearest-neighbour path.
fn sample_image_bilinear(frame: &VideoFrame, bounds: &Rect, ux: f32, uy: f32) -> Rgba {
    let info = match prepare_image_sample(frame, bounds, ux, uy) {
        Some(i) => i,
        None => return Rgba::new(0, 0, 0, 0),
    };
    // Continuous texture coordinate where integer values land on pixel
    // centres. (u * width) - 0.5 gives the position relative to the
    // pixel-centre grid.
    let tx = info.u * info.width as f32 - 0.5;
    let ty = info.v * info.height as f32 - 0.5;
    let x0 = tx.floor() as i64;
    let y0 = ty.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let wx = (tx - x0 as f32).clamp(0.0, 1.0);
    let wy = (ty - y0 as f32).clamp(0.0, 1.0);
    let cx0 = x0.clamp(0, info.width as i64 - 1) as usize;
    let cx1 = x1.clamp(0, info.width as i64 - 1) as usize;
    let cy0 = y0.clamp(0, info.height as i64 - 1) as usize;
    let cy1 = y1.clamp(0, info.height as i64 - 1) as usize;
    let p00 = fetch_premul(info.data, info.stride, cx0, cy0);
    let p10 = fetch_premul(info.data, info.stride, cx1, cy0);
    let p01 = fetch_premul(info.data, info.stride, cx0, cy1);
    let p11 = fetch_premul(info.data, info.stride, cx1, cy1);
    let w00 = (1.0 - wx) * (1.0 - wy);
    let w10 = wx * (1.0 - wy);
    let w01 = (1.0 - wx) * wy;
    let w11 = wx * wy;
    let lerp4 = |a: f32, b: f32, c: f32, d: f32| -> f32 { a * w00 + b * w10 + c * w01 + d * w11 };
    let pr = lerp4(p00.0, p10.0, p01.0, p11.0);
    let pg = lerp4(p00.1, p10.1, p01.1, p11.1);
    let pb = lerp4(p00.2, p10.2, p01.2, p11.2);
    let pa = lerp4(p00.3, p10.3, p01.3, p11.3);
    if pa <= 0.5 {
        return Rgba::new(0, 0, 0, 0);
    }
    // Un-premultiply for straight-alpha return.
    let inv = 255.0 / pa;
    Rgba::new(
        (pr * inv).round().clamp(0.0, 255.0) as u8,
        (pg * inv).round().clamp(0.0, 255.0) as u8,
        (pb * inv).round().clamp(0.0, 255.0) as u8,
        pa.round().clamp(0.0, 255.0) as u8,
    )
}

/// Lanczos2 (windowed sinc, `a = 2`) sample of an `Rgba` `VideoFrame`.
///
/// `lanczos2(x) = sinc(π·x) * sinc(π·x/2)` for `|x| < 2`, zero
/// elsewhere. The 2-D filter is the separable product
/// `lanczos2(x) * lanczos2(y)` evaluated over a 4×4 footprint centred
/// at the sample point. Samples are taken in **premultiplied-alpha**
/// space so that alpha-zero source pixels don't bleed colour into the
/// blend; the un-premultiply at the end mirrors
/// [`sample_image_bilinear`]. The filter can ring (negative side-lobes
/// pull the channel outside `[0, 255]`); the result is clamped per
/// channel before un-premultiplication.
///
/// Uses *clamp-to-edge* for samples whose footprint extends past the
/// texture boundary. Out-of-bounds (the user point is outside `bounds`)
/// returns transparent — same as the other samplers.
fn sample_image_lanczos2(frame: &VideoFrame, bounds: &Rect, ux: f32, uy: f32) -> Rgba {
    sample_image_lanczos_a::<2, 4>(frame, bounds, ux, uy, lanczos2)
}

/// Lanczos3 (windowed sinc, `a = 3`) sample of an `Rgba` `VideoFrame`.
///
/// `lanczos3(x) = sinc(π·x) * sinc(π·x/3)` for `|x| < 3`, zero
/// elsewhere. The 2-D filter is the separable product
/// `lanczos3(x) * lanczos3(y)` evaluated over a 6×6 footprint centred
/// at the sample point. Wider main-lobe than Lanczos2 → sharper
/// reconstruction; the second negative side-lobe (around `x = 1.5..2.5`)
/// can push the per-channel premultiplied accumulator outside
/// `[0, 255]`, so the result is clamped per channel before
/// un-premultiplication — identical bookkeeping to the Lanczos2 path.
/// Reference: Duchon, "Lanczos filtering in one and two dimensions"
/// (1979); Turkowski, "Filters for Common Resampling Tasks" (1990).
///
/// Uses *clamp-to-edge* for samples whose footprint extends past the
/// texture boundary. Out-of-bounds (the user point is outside `bounds`)
/// returns transparent — same as the other samplers.
fn sample_image_lanczos3(frame: &VideoFrame, bounds: &Rect, ux: f32, uy: f32) -> Rgba {
    sample_image_lanczos_a::<3, 6>(frame, bounds, ux, uy, lanczos3)
}

/// Shared 2N×2N separable Lanczos sampler (`N = a`). The fixed-size
/// `[f32; W]` weight arrays (`W = 2 * N`) and the const-generic loop
/// bound let the compiler unroll the per-axis weight build, while the
/// shared boundary clamp + premultiplied-alpha accumulation + per-axis
/// weight re-normalisation keeps the Lanczos2 and Lanczos3 paths
/// byte-identical apart from `a` and the kernel function pointer.
///
/// `N` is the Lanczos half-window in source pixels and `W = 2 * N` the
/// total tap count per axis; the trailing const generic is required
/// because const arithmetic in array lengths isn't yet stable on
/// `[f32; 2 * N]`.
fn sample_image_lanczos_a<const N: i64, const W: usize>(
    frame: &VideoFrame,
    bounds: &Rect,
    ux: f32,
    uy: f32,
    kernel: fn(f32) -> f32,
) -> Rgba {
    debug_assert_eq!(W, 2 * N as usize);
    let info = match prepare_image_sample(frame, bounds, ux, uy) {
        Some(i) => i,
        None => return Rgba::new(0, 0, 0, 0),
    };
    let tx = info.u * info.width as f32 - 0.5;
    let ty = info.v * info.height as f32 - 0.5;
    let xc = tx.floor() as i64;
    let yc = ty.floor() as i64;
    // W taps at offsets (-(N-1), .., 0, +1, .., +N) from the floored centre.
    let mut wxs = [0.0f32; W];
    let mut wys = [0.0f32; W];
    let mut sumwx = 0.0f32;
    let mut sumwy = 0.0f32;
    for (k, (wx, wy)) in wxs.iter_mut().zip(wys.iter_mut()).enumerate() {
        let off = (k as i64) - (N - 1);
        *wx = kernel(tx - (xc + off) as f32);
        *wy = kernel(ty - (yc + off) as f32);
        sumwx += *wx;
        sumwy += *wy;
    }
    // Re-normalise the per-axis weight sums so the kernel is a
    // partition of unity (compensates for the truncated tail of the
    // sinc — without this, edge samples shift slightly in luminance).
    if sumwx.abs() > 1e-6 {
        for w in wxs.iter_mut() {
            *w /= sumwx;
        }
    }
    if sumwy.abs() > 1e-6 {
        for w in wys.iter_mut() {
            *w /= sumwy;
        }
    }
    let mut acc = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for (j, &wy) in wys.iter().enumerate() {
        let py = (yc + j as i64 - (N - 1)).clamp(0, info.height as i64 - 1) as usize;
        for (i, &wx) in wxs.iter().enumerate() {
            let px = (xc + i as i64 - (N - 1)).clamp(0, info.width as i64 - 1) as usize;
            let w = wx * wy;
            let (r, g, b, a) = fetch_premul(info.data, info.stride, px, py);
            acc.0 += r * w;
            acc.1 += g * w;
            acc.2 += b * w;
            acc.3 += a * w;
        }
    }
    let pa = acc.3.clamp(0.0, 255.0);
    if pa <= 0.5 {
        return Rgba::new(0, 0, 0, 0);
    }
    let inv = 255.0 / pa;
    Rgba::new(
        (acc.0 * inv).round().clamp(0.0, 255.0) as u8,
        (acc.1 * inv).round().clamp(0.0, 255.0) as u8,
        (acc.2 * inv).round().clamp(0.0, 255.0) as u8,
        pa.round().clamp(0.0, 255.0) as u8,
    )
}

/// Mitchell–Netravali bicubic sample of an `Rgba` `VideoFrame`.
///
/// `B = 1/3, C = 1/3` (the "Mitchell" filter), the parameter pair
/// recommended in *Mitchell & Netravali, "Reconstruction Filters in
/// Computer Graphics" (1988)* as the best subjective trade-off between
/// blur and ringing. The 4×4 separable kernel evaluates
/// [`mitchell_netravali`] per axis, samples in **premultiplied-alpha**
/// space, and clamps the result back to `[0, 255]` per channel. The
/// out-of-bounds handling and edge-clamp behaviour mirrors the bilinear
/// and Lanczos2 paths.
fn sample_image_mitchell(frame: &VideoFrame, bounds: &Rect, ux: f32, uy: f32) -> Rgba {
    sample_image_bc_cubic(frame, bounds, ux, uy, mitchell_netravali)
}

/// Catmull–Rom bicubic sample of an `Rgba` `VideoFrame` — the `B = 0,
/// C = 1/2` interpolating member of the BC family (sharper than
/// Mitchell, passes through source samples). Shares the 4×4 separable
/// premultiplied-alpha machinery via [`sample_image_bc_cubic`]; the only
/// difference from the Mitchell path is the per-axis kernel.
fn sample_image_catmull_rom(frame: &VideoFrame, bounds: &Rect, ux: f32, uy: f32) -> Rgba {
    sample_image_bc_cubic(frame, bounds, ux, uy, catmull_rom)
}

/// Cubic B-spline sample of an `Rgba` `VideoFrame` — the `B = 1, C = 0`
/// approximating member of the BC family (smoothest, ring-free, blurs
/// rather than interpolates). Shares the 4×4 separable premultiplied-
/// alpha machinery via [`sample_image_bc_cubic`]; the only difference
/// from the Mitchell / Catmull–Rom paths is the per-axis kernel.
fn sample_image_b_spline(frame: &VideoFrame, bounds: &Rect, ux: f32, uy: f32) -> Rgba {
    sample_image_bc_cubic(frame, bounds, ux, uy, b_spline)
}

/// Shared 4×4 separable BC-family-cubic sampler. `kernel` is one of the
/// Mitchell–Netravali BC kernels ([`mitchell_netravali`] /
/// [`catmull_rom`]); the geometry, premultiplied-alpha accumulation,
/// boundary clamp + per-axis weight re-normalisation, and final
/// un-premultiply are identical across the family.
fn sample_image_bc_cubic(
    frame: &VideoFrame,
    bounds: &Rect,
    ux: f32,
    uy: f32,
    kernel: fn(f32) -> f32,
) -> Rgba {
    let info = match prepare_image_sample(frame, bounds, ux, uy) {
        Some(i) => i,
        None => return Rgba::new(0, 0, 0, 0),
    };
    let tx = info.u * info.width as f32 - 0.5;
    let ty = info.v * info.height as f32 - 0.5;
    let xc = tx.floor() as i64;
    let yc = ty.floor() as i64;
    // 4×4 taps at offsets (-1, 0, +1, +2) from the floored centre.
    let mut wxs = [0.0f32; 4];
    let mut wys = [0.0f32; 4];
    let mut sumwx = 0.0f32;
    let mut sumwy = 0.0f32;
    for (k, (wx, wy)) in wxs.iter_mut().zip(wys.iter_mut()).enumerate() {
        let off = (k as i64) - 1;
        *wx = kernel(tx - (xc + off) as f32);
        *wy = kernel(ty - (yc + off) as f32);
        sumwx += *wx;
        sumwy += *wy;
    }
    // Every BC-family cubic sums to 1 across an integer-aligned 4-tap
    // window, but at the clamped boundary the contributing taps shift
    // and the partition-of-unity property is broken; re-normalise so
    // edge samples don't shift in luminance.
    if sumwx.abs() > 1e-6 {
        for w in wxs.iter_mut() {
            *w /= sumwx;
        }
    }
    if sumwy.abs() > 1e-6 {
        for w in wys.iter_mut() {
            *w /= sumwy;
        }
    }
    let mut acc = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for (j, &wy) in wys.iter().enumerate() {
        let py = (yc + j as i64 - 1).clamp(0, info.height as i64 - 1) as usize;
        for (i, &wx) in wxs.iter().enumerate() {
            let px = (xc + i as i64 - 1).clamp(0, info.width as i64 - 1) as usize;
            let w = wx * wy;
            let (r, g, b, a) = fetch_premul(info.data, info.stride, px, py);
            acc.0 += r * w;
            acc.1 += g * w;
            acc.2 += b * w;
            acc.3 += a * w;
        }
    }
    let pa = acc.3.clamp(0.0, 255.0);
    if pa <= 0.5 {
        return Rgba::new(0, 0, 0, 0);
    }
    let inv = 255.0 / pa;
    Rgba::new(
        (acc.0 * inv).round().clamp(0.0, 255.0) as u8,
        (acc.1 * inv).round().clamp(0.0, 255.0) as u8,
        (acc.2 * inv).round().clamp(0.0, 255.0) as u8,
        pa.round().clamp(0.0, 255.0) as u8,
    )
}

/// Generic Mitchell–Netravali BC-family cubic reconstruction kernel.
/// The piecewise polynomial (from *Mitchell & Netravali, "Reconstruction
/// Filters in Computer Graphics" (1988)*) is:
///
/// ```text
///                    1
/// k(x) =  ──────── × {
///                    6
///   (12 − 9B − 6C)·|x|³ + (−18 + 12B + 6C)·|x|²            + (6 − 2B)            if |x| < 1
///   (−B − 6C)·|x|³ + (6B + 30C)·|x|² + (−12B − 48C)·|x| + (8B + 24C)             if 1 ≤ |x| < 2
///   0                                                                            otherwise
/// }
/// ```
///
/// `(B, C)` selects the member of the family:
///
/// * `(1/3, 1/3)` — the "Mitchell" point (balanced blur + ringing); see
///   [`mitchell_netravali`].
/// * `(0, 1/2)` — Catmull–Rom: the interpolating cubic spline
///   (`k(0) = 1`, `k(n) = 0` for non-zero integer `n`), so it passes
///   exactly through the source samples and is noticeably sharper than
///   Mitchell at the cost of a slightly stronger negative side-lobe; see
///   [`catmull_rom`].
/// * `(1, 0)` — the cubic B-spline: the maximally-smooth, everywhere
///   non-negative approximating spline (no ringing, no overshoot,
///   `k(0) = 4/6`), the blurriest of the three canonical points; see
///   [`b_spline`].
///
/// The naive un-factored form below is the one we actually evaluate —
/// it keeps the source close to the textbook formula and is fast enough
/// for a 4-tap-per-axis hot loop.
#[inline]
fn bc_cubic(x: f32, b: f32, c: f32) -> f32 {
    let ax = x.abs();
    if ax < 1.0 {
        let x2 = ax * ax;
        let x3 = x2 * ax;
        ((12.0 - 9.0 * b - 6.0 * c) * x3 + (-18.0 + 12.0 * b + 6.0 * c) * x2 + (6.0 - 2.0 * b))
            / 6.0
    } else if ax < 2.0 {
        let x2 = ax * ax;
        let x3 = x2 * ax;
        ((-b - 6.0 * c) * x3
            + (6.0 * b + 30.0 * c) * x2
            + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

/// Mitchell–Netravali cubic reconstruction kernel with `B = 1/3, C = 1/3`
/// — the "Mitchell" point of the BC family, the subjective best blur +
/// ringing trade-off per the 1988 paper.
///
/// With `B = 1/3, C = 1/3` the [`bc_cubic`] polynomial reduces to:
///
/// ```text
///   |x| < 1   → (7·|x|³ − 12·|x|² + 16/3) / 6
///   1 ≤ |x| < 2 → (−7/3·|x|³ + 12·|x|² − 20·|x| + 32/3) / 6
///   else        → 0
/// ```
#[inline]
fn mitchell_netravali(x: f32) -> f32 {
    bc_cubic(x, 1.0 / 3.0, 1.0 / 3.0)
}

/// Catmull–Rom cubic reconstruction kernel — the `B = 0, C = 1/2` member
/// of the Mitchell–Netravali BC family. It is the unique cubic in the
/// family that *interpolates* the source samples (`k(0) = 1`, and
/// `k(±1) = k(±2) = 0`), so a sample landing exactly on a source-pixel
/// centre reproduces that pixel unchanged. Sharper than Mitchell (no
/// blur term, `B = 0`); the price is a stronger negative side-lobe in
/// `1 ≤ |x| < 2`, which the sampler's per-channel `[0, 255]` clamp
/// absorbs. A good choice for *upscaling* where edge crispness matters.
#[inline]
fn catmull_rom(x: f32) -> f32 {
    bc_cubic(x, 0.0, 0.5)
}

/// Cubic B-spline reconstruction kernel — the `B = 1, C = 0` member of
/// the Mitchell–Netravali BC family. It is the unique *approximating*
/// cubic in the family: everywhere non-negative (so it never rings or
/// overshoots — no negative side-lobe to clamp), but it does not
/// interpolate the source (`k(0) = 4/6 ≈ 0.667`, and the neighbours at
/// `±1` carry the rest, `k(±1) = 1/6`). It is the smoothest and
/// blurriest of the three canonical points, the best choice when
/// halo-free, monotone-preserving minification matters more than edge
/// sharpness.
///
/// With `B = 1, C = 0` the [`bc_cubic`] polynomial reduces to the
/// classic uniform cubic B-spline basis:
///
/// ```text
///   |x| < 1     → (3·|x|³ − 6·|x|² + 4) / 6
///   1 ≤ |x| < 2 → (2 − |x|)³ / 6
///   else        → 0
/// ```
#[inline]
fn b_spline(x: f32) -> f32 {
    bc_cubic(x, 1.0, 0.0)
}

/// Lanczos kernel for `a = 2`. `lanczos2(0) = 1`; vanishes at integer
/// non-zero `x`; zero outside `|x| < 2`. The naive `sin(πx) / (πx)`
/// form is fine for our 4-tap evaluation — the divide is cheap and
/// the special case at `x = 0` is the only branch.
#[inline]
fn lanczos2(x: f32) -> f32 {
    let ax = x.abs();
    if ax >= 2.0 {
        return 0.0;
    }
    if ax < 1e-7 {
        return 1.0;
    }
    let pi = std::f32::consts::PI;
    let pix = pi * x;
    let pix2 = pix * 0.5;
    let s = pix.sin() / pix;
    let s2 = pix2.sin() / pix2;
    s * s2
}

/// Lanczos kernel for `a = 3`. `lanczos3(0) = 1`; vanishes at integer
/// non-zero `x` (the `sinc(πx)` factor); zero outside `|x| < 3`.
/// Identical structure to [`lanczos2`] with the window stretched to
/// `sinc(πx/3)` — adds two more taps per axis, picks up the first
/// negative side-lobe of the source sinc, and gives a noticeably
/// sharper reconstruction at the cost of one extra `sin` per tap.
#[inline]
fn lanczos3(x: f32) -> f32 {
    let ax = x.abs();
    if ax >= 3.0 {
        return 0.0;
    }
    if ax < 1e-7 {
        return 1.0;
    }
    let pi = std::f32::consts::PI;
    let pix = pi * x;
    let pix3 = pix / 3.0;
    let s = pix.sin() / pix;
    let s3 = pix3.sin() / pix3;
    s * s3
}

/// Build one axis's `W` filter taps (clamped texel positions +
/// re-normalised weights) for the continuous texture coordinate `t`,
/// exactly as the per-pixel windowed-sinc / BC-cubic samplers do:
/// `W` kernel evaluations at offsets `-(half-1) ..= half` from
/// `floor(t)`, summed in tap order, re-normalised to a partition of
/// unity when the sum is non-negligible (`|sum| > 1e-6`), positions
/// clamped to the texture extent.
#[inline]
fn kernel_taps<const W: usize>(
    t: f32,
    extent: u32,
    half: i64,
    kernel: fn(f32) -> f32,
) -> ([usize; W], [f32; W]) {
    debug_assert_eq!(W as i64, 2 * half);
    let c = t.floor() as i64;
    let mut w = [0.0f32; W];
    let mut sum = 0.0f32;
    for (k, wk) in w.iter_mut().enumerate() {
        let off = (k as i64) - (half - 1);
        *wk = kernel(t - (c + off) as f32);
        sum += *wk;
    }
    if sum.abs() > 1e-6 {
        for wk in w.iter_mut() {
            *wk /= sum;
        }
    }
    let mut pos = [0usize; W];
    for (k, pk) in pos.iter_mut().enumerate() {
        *pk = (c + (k as i64) - (half - 1)).clamp(0, extent as i64 - 1) as usize;
    }
    (pos, w)
}

/// Fetch a single texel and convert to premultiplied-alpha floats.
#[inline]
fn fetch_premul(data: &[u8], stride: usize, x: usize, y: usize) -> (f32, f32, f32, f32) {
    let i = y * stride + x * 4;
    let r = data[i] as f32;
    let g = data[i + 1] as f32;
    let b = data[i + 2] as f32;
    let a = data[i + 3] as f32;
    let pm = a / 255.0;
    (r * pm, g * pm, b * pm, a)
}

struct ImageSampleInfo<'a> {
    data: &'a [u8],
    stride: usize,
    width: u32,
    height: u32,
    u: f32,
    v: f32,
}

/// Common bookkeeping for both samplers: validate the frame, project
/// the user-space point into the `[0, 1]` UV square, and reject
/// out-of-bounds reads.
fn prepare_image_sample<'a>(
    frame: &'a VideoFrame,
    bounds: &Rect,
    ux: f32,
    uy: f32,
) -> Option<ImageSampleInfo<'a>> {
    if bounds.width <= 0.0 || bounds.height <= 0.0 {
        return None;
    }
    let u = (ux - bounds.x) / bounds.width;
    let v = (uy - bounds.y) / bounds.height;
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return None;
    }
    let plane = frame.planes.first()?;
    let stride = plane.stride;
    if stride < 4 {
        return None;
    }
    let width = (stride / 4) as u32;
    let height = (plane.data.len() / stride) as u32;
    if width == 0 || height == 0 {
        return None;
    }
    Some(ImageSampleInfo {
        data: &plane.data,
        stride,
        width,
        height,
        u,
        v,
    })
}

/// Convenience: rasterize a [`VectorFrame`] using the renderer's
/// defaults (4× supersampling, transparent background) at the
/// frame's natural pixel size.
///
/// Wraps the result in a [`Frame`](oxideav_core::Frame::Video) ready
/// for the rest of the pipeline.
pub fn rasterize(frame: &VectorFrame) -> Frame {
    let w = frame.width.max(1.0).round() as u32;
    let h = frame.height.max(1.0).round() as u32;
    let r = Renderer::new(w, h);
    Frame::Video(r.render(frame))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{AspectRatioAlign, MeetOrSlice};
    use oxideav_core::{
        FillRule, Group, Node, Paint, Path, PathNode, Point, Rgba, Transform2D, VectorFrame,
    };

    fn frame(w: u32, h: u32, root: Group) -> VectorFrame {
        VectorFrame {
            width: w as f32,
            height: h as f32,
            view_box: None,
            root,
            pts: None,
            time_base: oxideav_core::time::TimeBase::new(1, 1),
        }
    }

    fn red_rect_node(x: f32, y: f32, w: f32, h: f32) -> Node {
        let mut p = Path::new();
        p.move_to(Point::new(x, y))
            .line_to(Point::new(x + w, y))
            .line_to(Point::new(x + w, y + h))
            .line_to(Point::new(x, y + h))
            .close();
        Node::Path(PathNode {
            path: p,
            fill: Some(Paint::Solid(Rgba::opaque(255, 0, 0))),
            stroke: None,
            fill_rule: FillRule::NonZero,
        })
    }

    #[test]
    fn render_empty_scene_yields_transparent_buffer() {
        let r = Renderer::new(4, 4);
        let v = frame(4, 4, Group::default());
        let out = r.render(&v);
        assert_eq!(out.planes.len(), 1);
        assert!(out.planes[0].data.iter().all(|&b| b == 0));
    }

    #[test]
    fn render_red_rect_paints_red() {
        let mut root = Group::default();
        root.children.push(red_rect_node(1.0, 1.0, 4.0, 4.0));
        let r = Renderer::new(8, 8);
        let v = frame(8, 8, root);
        let out = r.render(&v);
        let stride = out.planes[0].stride;
        // Centre pixel should be solid red.
        let i = 3 * stride + 3 * 4;
        let p = &out.planes[0].data[i..i + 4];
        assert_eq!(p[0], 255);
        assert_eq!(p[1], 0);
        assert_eq!(p[2], 0);
        assert_eq!(p[3], 255);
    }

    #[test]
    fn render_with_view_box_scales_into_canvas() {
        let mut root = Group::default();
        root.children.push(red_rect_node(0.0, 0.0, 10.0, 10.0));
        let mut v = frame(20, 20, root);
        v.view_box = Some(oxideav_core::ViewBox {
            min_x: 0.0,
            min_y: 0.0,
            width: 10.0,
            height: 10.0,
        });
        let r = Renderer::new(20, 20);
        let out = r.render(&v);
        let stride = out.planes[0].stride;
        // Centre of canvas (10, 10) maps to (5, 5) in user-space, well
        // inside the 10×10 rect → must be red.
        let i = 10 * stride + 10 * 4;
        assert_eq!(out.planes[0].data[i], 255);
    }

    /// Helper: read the RGBA at (x, y) of a rendered frame.
    fn px_at(out: &VideoFrame, x: u32, y: u32) -> [u8; 4] {
        let stride = out.planes[0].stride;
        let i = (y as usize) * stride + (x as usize) * 4;
        let s = &out.planes[0].data[i..i + 4];
        [s[0], s[1], s[2], s[3]]
    }

    /// `xMidYMid meet` is the SVG §7.8 default. A wide viewBox painted
    /// fully red, fitted into a square canvas, must letterbox: the
    /// content is scaled uniformly to the limiting (here horizontal)
    /// dimension and centred vertically, leaving transparent bands top
    /// and bottom.
    #[test]
    fn view_box_meet_letterboxes_wide_content() {
        // viewBox 20×10 (2:1) into a 20×20 canvas. meet → uniform
        // scale = min(20/20, 20/10) = min(1, 2) = 1. Content height
        // 10·1 = 10, centred → occupies canvas rows 5..15.
        let mut root = Group::default();
        root.children.push(red_rect_node(0.0, 0.0, 20.0, 10.0));
        let mut v = frame(20, 20, root);
        v.view_box = Some(oxideav_core::ViewBox::new(0.0, 0.0, 20.0, 10.0));
        let r = Renderer::new(20, 20); // default xMidYMid meet
        let out = r.render(&v);
        // Middle band (row 10) is red across the width.
        assert_eq!(px_at(&out, 10, 10), [255, 0, 0, 255]);
        // Top band (row 1) and bottom band (row 18) are transparent
        // (outside the centred content).
        assert_eq!(px_at(&out, 10, 1), [0, 0, 0, 0]);
        assert_eq!(px_at(&out, 10, 18), [0, 0, 0, 0]);
    }

    /// `xMinYMin meet`: a tall viewBox into a square canvas aligns the
    /// content's top-left to the viewport top-left rather than centring,
    /// so the empty band is on the right.
    #[test]
    fn view_box_meet_xminymin_aligns_top_left() {
        // viewBox 10×20 (1:2) into 20×20. meet uniform scale =
        // min(20/10, 20/20) = min(2, 1) = 1. Content width 10·1 = 10,
        // aligned left → occupies columns 0..10, columns 10..20 empty.
        let mut root = Group::default();
        root.children.push(red_rect_node(0.0, 0.0, 10.0, 20.0));
        let mut v = frame(20, 20, root);
        v.view_box = Some(oxideav_core::ViewBox::new(0.0, 0.0, 10.0, 20.0));
        let r = Renderer::new(20, 20).with_preserve_aspect_ratio(PreserveAspectRatio {
            align: AspectRatioAlign::XMinYMin,
            meet_or_slice: MeetOrSlice::Meet,
        });
        let out = r.render(&v);
        // Left half painted, right half empty.
        assert_eq!(px_at(&out, 2, 10), [255, 0, 0, 255]);
        assert_eq!(px_at(&out, 18, 10), [0, 0, 0, 0]);
    }

    /// `none` reproduces the legacy non-uniform stretch: a square
    /// viewBox painted red fills the whole rectangular canvas (no
    /// letterboxing), matching pre-`preserveAspectRatio` behaviour.
    #[test]
    fn view_box_none_stretches_to_fill() {
        let mut root = Group::default();
        root.children.push(red_rect_node(0.0, 0.0, 10.0, 10.0));
        let mut v = frame(40, 20, root);
        v.view_box = Some(oxideav_core::ViewBox::new(0.0, 0.0, 10.0, 10.0));
        let r = Renderer::new(40, 20).with_preserve_aspect_ratio(PreserveAspectRatio {
            align: AspectRatioAlign::None,
            meet_or_slice: MeetOrSlice::Meet,
        });
        let out = r.render(&v);
        // Every corner is red — the content stretched across the whole
        // 40×20 canvas.
        assert_eq!(px_at(&out, 1, 1), [255, 0, 0, 255]);
        assert_eq!(px_at(&out, 38, 1), [255, 0, 0, 255]);
        assert_eq!(px_at(&out, 1, 18), [255, 0, 0, 255]);
        assert_eq!(px_at(&out, 38, 18), [255, 0, 0, 255]);
    }

    /// `xMidYMid slice` covers the whole canvas (uniform scale to the
    /// larger dimension) with the overflow cropped at the canvas edge
    /// (§8.6 `overflow: hidden`). A wide red viewBox sliced into a
    /// square canvas fills every pixel.
    #[test]
    fn view_box_slice_covers_canvas() {
        // viewBox 20×10 into 20×20. slice uniform scale =
        // max(20/20, 20/10) = max(1, 2) = 2. Content 40×20 centred
        // horizontally (overflow cropped) → every canvas pixel covered.
        let mut root = Group::default();
        root.children.push(red_rect_node(0.0, 0.0, 20.0, 10.0));
        let mut v = frame(20, 20, root);
        v.view_box = Some(oxideav_core::ViewBox::new(0.0, 0.0, 20.0, 10.0));
        let r = Renderer::new(20, 20).with_preserve_aspect_ratio(PreserveAspectRatio {
            align: AspectRatioAlign::XMidYMid,
            meet_or_slice: MeetOrSlice::Slice,
        });
        let out = r.render(&v);
        // Corners and centre all red — full coverage.
        assert_eq!(px_at(&out, 1, 1), [255, 0, 0, 255]);
        assert_eq!(px_at(&out, 18, 18), [255, 0, 0, 255]);
        assert_eq!(px_at(&out, 10, 10), [255, 0, 0, 255]);
    }

    /// A degenerate (zero-extent) viewBox disables rendering (§8.6):
    /// the output is the cleared canvas, never a NaN-transform smear.
    #[test]
    fn view_box_zero_extent_disables_rendering() {
        let mut root = Group::default();
        root.children.push(red_rect_node(0.0, 0.0, 10.0, 10.0));
        let mut v = frame(8, 8, root);
        v.view_box = Some(oxideav_core::ViewBox::new(0.0, 0.0, 0.0, 10.0));
        let r = Renderer::new(8, 8);
        let out = r.render(&v);
        assert!(out.planes[0].data.iter().all(|&b| b == 0));
    }

    /// When the viewBox aspect ratio already matches the canvas, all
    /// three policies (`meet`, `slice`, `none`) produce the same fit —
    /// there is nothing to letterbox or crop.
    #[test]
    fn view_box_matching_aspect_is_policy_invariant() {
        let build = |par: PreserveAspectRatio| {
            let mut root = Group::default();
            root.children.push(red_rect_node(0.0, 0.0, 10.0, 10.0));
            let mut v = frame(20, 20, root);
            v.view_box = Some(oxideav_core::ViewBox::new(0.0, 0.0, 10.0, 10.0));
            Renderer::new(20, 20)
                .with_preserve_aspect_ratio(par)
                .render(&v)
        };
        let meet = build(PreserveAspectRatio::default());
        let slice = build(PreserveAspectRatio {
            align: AspectRatioAlign::XMidYMid,
            meet_or_slice: MeetOrSlice::Slice,
        });
        let none = build(PreserveAspectRatio {
            align: AspectRatioAlign::None,
            meet_or_slice: MeetOrSlice::Meet,
        });
        assert_eq!(meet.planes[0].data, slice.planes[0].data);
        assert_eq!(meet.planes[0].data, none.planes[0].data);
    }

    #[test]
    fn rasterize_convenience_returns_video_frame() {
        let mut root = Group::default();
        root.children.push(red_rect_node(0.0, 0.0, 2.0, 2.0));
        let v = frame(2, 2, root);
        match rasterize(&v) {
            Frame::Video(_) => {}
            _ => panic!("expected Frame::Video"),
        }
    }

    #[test]
    fn renderer_with_background_clears_first() {
        let mut r = Renderer::new(2, 2);
        r.background = Rgba::opaque(0, 255, 0);
        let v = frame(2, 2, Group::default());
        let out = r.render(&v);
        // Every pixel should be the background color.
        for px in out.planes[0].data.chunks_exact(4) {
            assert_eq!(px, &[0, 255, 0, 255]);
        }
    }

    #[test]
    fn invert_2d_round_trip() {
        let t = Transform2D::translate(3.0, -1.0).compose(&Transform2D::scale(2.0, 4.0));
        let inv = invert_2d(&t).unwrap();
        let p = oxideav_core::Point::new(7.0, 13.0);
        let q = inv.apply(t.apply(p));
        assert!((q.x - p.x).abs() < 1e-4);
        assert!((q.y - p.y).abs() < 1e-4);
    }

    #[test]
    fn crop_to_bbox_extracts_painted_pixels_only() {
        // 8×8 buffer with a single non-zero pixel at (3, 5).
        let w = 8u32;
        let h = 8u32;
        let stride = (w as usize) * 4;
        let mut buf = vec![0u8; stride * (h as usize)];
        let i = 5 * stride + 3 * 4;
        buf[i] = 11;
        buf[i + 1] = 22;
        buf[i + 2] = 33;
        buf[i + 3] = 44;
        let s = crop_to_bbox(&buf, stride, w, h, Transform2D::identity());
        assert_eq!(s.width, 1);
        assert_eq!(s.height, 1);
        assert_eq!(s.offset_x, 3);
        assert_eq!(s.offset_y, 5);
        assert_eq!(&s.rgba, &[11, 22, 33, 44]);
    }

    #[test]
    fn crop_to_bbox_empty_buffer_yields_empty_subtree() {
        let w = 4u32;
        let h = 4u32;
        let stride = (w as usize) * 4;
        let buf = vec![0u8; stride * (h as usize)];
        let s = crop_to_bbox(&buf, stride, w, h, Transform2D::identity());
        assert_eq!(s.width, 0);
        assert_eq!(s.height, 0);
        assert!(s.rgba.is_empty());
    }

    #[test]
    fn crop_to_bbox_inclusive_bounds_around_two_pixels() {
        // Two non-zero pixels at (1, 2) and (4, 6) → bbox is 4×5.
        let w = 8u32;
        let h = 8u32;
        let stride = (w as usize) * 4;
        let mut buf = vec![0u8; stride * (h as usize)];
        for &(x, y, a) in &[(1u32, 2u32, 100u8), (4u32, 6u32, 200u8)] {
            let i = (y as usize) * stride + (x as usize) * 4;
            buf[i + 3] = a;
        }
        let s = crop_to_bbox(&buf, stride, w, h, Transform2D::identity());
        assert_eq!(s.offset_x, 1);
        assert_eq!(s.offset_y, 2);
        assert_eq!(s.width, 4);
        assert_eq!(s.height, 5);
        assert_eq!(s.rgba.len(), (4 * 5 * 4) as usize);
    }

    #[test]
    fn blit_with_offset_paints_at_destination_position() {
        // 8×8 destination, 2×2 source of opaque red, blit at (3, 4).
        let mut dst = vec![0u8; 8 * 8 * 4];
        let src: Vec<u8> = (0..2 * 2).flat_map(|_| [255, 0, 0, 255]).collect();
        blit_rgba_over(&mut dst, 32, 8, 8, &src, 2, 2, 3, 4, None, 1.0);
        // Pixel (3, 4) should be red.
        let i = 4 * 32 + 3 * 4;
        assert_eq!(&dst[i..i + 4], &[255, 0, 0, 255]);
        // Pixel (4, 5) should be red.
        let i = 5 * 32 + 4 * 4;
        assert_eq!(&dst[i..i + 4], &[255, 0, 0, 255]);
        // Pixel (0, 0) untouched.
        assert_eq!(&dst[0..4], &[0, 0, 0, 0]);
        // Pixel (5, 4) just past the 2×2 src — untouched.
        let i = 4 * 32 + 5 * 4;
        assert_eq!(&dst[i..i + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn blit_with_offset_clips_oversized_source() {
        // 4×4 destination, 4×4 source, blit at (2, 2) → only the
        // top-left 2×2 of the source should land on dst.
        let mut dst = vec![0u8; 4 * 4 * 4];
        let src: Vec<u8> = (0..4 * 4).flat_map(|_| [10, 20, 30, 255]).collect();
        blit_rgba_over(&mut dst, 16, 4, 4, &src, 4, 4, 2, 2, None, 1.0);
        // Pixel (2, 2): painted.
        let i = 2 * 16 + 2 * 4;
        assert_eq!(&dst[i..i + 4], &[10, 20, 30, 255]);
        // Pixel (3, 3): painted.
        let i = 3 * 16 + 3 * 4;
        assert_eq!(&dst[i..i + 4], &[10, 20, 30, 255]);
        // Pixel (1, 1): untouched.
        let i = 16 + 4;
        assert_eq!(&dst[i..i + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn lanczos2_kernel_unit_at_zero_and_zero_at_integers() {
        assert!((lanczos2(0.0) - 1.0).abs() < 1e-6);
        assert!(lanczos2(1.0).abs() < 1e-3);
        assert!(lanczos2(-1.0).abs() < 1e-3);
        assert_eq!(lanczos2(2.0), 0.0);
        assert_eq!(lanczos2(-2.0), 0.0);
        assert_eq!(lanczos2(2.5), 0.0);
    }

    #[test]
    fn lanczos2_kernel_is_even_symmetric() {
        for x in [0.25f32, 0.5, 0.75, 1.25, 1.5, 1.75] {
            assert!((lanczos2(x) - lanczos2(-x)).abs() < 1e-6);
        }
    }

    #[test]
    fn lanczos3_kernel_unit_at_zero_and_zero_at_integers() {
        assert!((lanczos3(0.0) - 1.0).abs() < 1e-6);
        // sinc(π·k) is zero for non-zero integer k, so the kernel
        // vanishes at every integer inside its support except 0.
        for k in [1.0f32, 2.0, -1.0, -2.0] {
            assert!(
                lanczos3(k).abs() < 1e-3,
                "lanczos3({k}) should vanish at integer non-zero x"
            );
        }
        // Zero outside the 6-tap support.
        assert_eq!(lanczos3(3.0), 0.0);
        assert_eq!(lanczos3(-3.0), 0.0);
        assert_eq!(lanczos3(4.0), 0.0);
    }

    #[test]
    fn lanczos3_kernel_is_even_symmetric() {
        for x in [0.25f32, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75] {
            assert!(
                (lanczos3(x) - lanczos3(-x)).abs() < 1e-6,
                "lanczos3 should be even-symmetric at x={x}"
            );
        }
    }

    #[test]
    fn lanczos3_has_negative_side_lobes_unlike_b_spline() {
        // The second cycle of sinc inside |x| < 3 gives a negative
        // side-lobe; this is the textbook ringing signature of any
        // truncated-sinc reconstruction and what distinguishes Lanczos
        // from the everywhere-non-negative cubic B-spline.
        let mut saw_negative = false;
        let mut x = 1.05f32;
        while x < 1.95 {
            if lanczos3(x) < 0.0 {
                saw_negative = true;
                break;
            }
            x += 0.05;
        }
        assert!(
            saw_negative,
            "lanczos3 should dip below zero in |x| ∈ (1, 2)"
        );
    }

    #[test]
    fn lanczos3_kernel_window_is_wider_than_lanczos2() {
        // The defining contrast vs Lanczos2: lanczos3 still has
        // support past |x| = 2 (because sinc(πx/3) is non-zero there),
        // while lanczos2 is zero by construction.
        assert_eq!(lanczos2(2.0), 0.0);
        assert_eq!(lanczos2(2.5), 0.0);
        // Lanczos3 is non-zero (and indeed negative) in (2, 3).
        assert!(lanczos3(2.5).abs() > 1e-3);
    }

    #[test]
    fn mitchell_kernel_unit_at_zero_and_zero_outside_support() {
        // k(0) = (6 - 2B) / 6 = (6 - 2/3) / 6 = 16/18 = 8/9 for the
        // Mitchell choice (B = 1/3) — the kernel is *not* a partition of
        // unity in isolation; only the sum across the 4-tap window is.
        let v0 = mitchell_netravali(0.0);
        assert!(
            (v0 - 8.0 / 9.0).abs() < 1e-5,
            "k(0)={}, want 8/9 ≈ 0.8889",
            v0
        );
        // Vanishes outside the support.
        assert_eq!(mitchell_netravali(2.0), 0.0);
        assert_eq!(mitchell_netravali(-2.0), 0.0);
        assert_eq!(mitchell_netravali(2.5), 0.0);
        // At the piecewise boundary x = 1 the two branches meet at zero
        // for Mitchell (B = C = 1/3): substituting into the |x| < 1
        // branch gives (7 − 12 + 16/3) / 6 = (−5 + 16/3) / 6 = (1/3) / 6
        // = 1/18; the |x| ≥ 1 branch also evaluates to 1/18 at x = 1.
        // Confirm both pieces agree at the boundary.
        let left = mitchell_netravali(1.0 - 1e-5);
        let right = mitchell_netravali(1.0 + 1e-5);
        assert!(
            (left - right).abs() < 1e-3,
            "branch mismatch at x=1: left={}, right={}",
            left,
            right
        );
    }

    #[test]
    fn mitchell_kernel_is_even_symmetric() {
        for x in [0.25f32, 0.5, 0.75, 1.25, 1.5, 1.75] {
            assert!(
                (mitchell_netravali(x) - mitchell_netravali(-x)).abs() < 1e-6,
                "asymmetry at |x|={}",
                x
            );
        }
    }

    #[test]
    fn mitchell_kernel_partition_of_unity_on_offset_grid() {
        // Across a 4-tap window with the sample landing at any
        // fractional offset, the four kernel weights must sum to 1
        // (within float tolerance). This is the property that makes
        // re-normalisation a no-op for interior samples.
        for frac in [0.0f32, 0.1, 0.25, 0.5, 0.75, 0.9] {
            let sum = mitchell_netravali(-1.0 - frac)
                + mitchell_netravali(-frac)
                + mitchell_netravali(1.0 - frac)
                + mitchell_netravali(2.0 - frac);
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "partition broken at frac={}, sum={}",
                frac,
                sum
            );
        }
    }

    #[test]
    fn catmull_rom_interpolates_source_samples() {
        // The defining property of Catmull–Rom (the interpolating member
        // of the BC family): k(0) = 1 and k vanishes at every non-zero
        // integer offset, so a sample on a pixel centre reproduces it.
        assert!((catmull_rom(0.0) - 1.0).abs() < 1e-6, "k(0) must be 1");
        assert!(catmull_rom(1.0).abs() < 1e-6, "k(1) must be 0");
        assert!(catmull_rom(-1.0).abs() < 1e-6, "k(-1) must be 0");
        assert_eq!(catmull_rom(2.0), 0.0);
        assert_eq!(catmull_rom(-2.0), 0.0);
        assert_eq!(catmull_rom(2.5), 0.0);
    }

    #[test]
    fn catmull_rom_kernel_is_even_symmetric() {
        for x in [0.25f32, 0.5, 0.75, 1.25, 1.5, 1.75] {
            assert!(
                (catmull_rom(x) - catmull_rom(-x)).abs() < 1e-6,
                "asymmetry at |x|={}",
                x
            );
        }
    }

    #[test]
    fn catmull_rom_kernel_partition_of_unity_on_offset_grid() {
        // Like every BC-family cubic, the four weights across a 4-tap
        // window sum to 1 at any fractional offset.
        for frac in [0.0f32, 0.1, 0.25, 0.5, 0.75, 0.9] {
            let sum = catmull_rom(-1.0 - frac)
                + catmull_rom(-frac)
                + catmull_rom(1.0 - frac)
                + catmull_rom(2.0 - frac);
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "partition broken at frac={}, sum={}",
                frac,
                sum
            );
        }
    }

    #[test]
    fn catmull_rom_has_negative_side_lobe() {
        // The B = 0 family member rings: there is a region in
        // 1 < |x| < 2 where the kernel goes negative. (Mitchell's
        // B = 1/3 mostly suppresses this.) Confirm at the side-lobe
        // extreme x = 1.5.
        assert!(
            catmull_rom(1.5) < 0.0,
            "Catmull–Rom should have a negative side-lobe at x=1.5, got {}",
            catmull_rom(1.5)
        );
    }

    #[test]
    fn bc_cubic_matches_named_specialisations() {
        // The named wrappers must equal the generic kernel at their
        // respective (B, C) points across the whole support.
        for i in -25..=25 {
            let x = i as f32 / 10.0;
            assert!((bc_cubic(x, 1.0 / 3.0, 1.0 / 3.0) - mitchell_netravali(x)).abs() < 1e-7);
            assert!((bc_cubic(x, 0.0, 0.5) - catmull_rom(x)).abs() < 1e-7);
            assert!((bc_cubic(x, 1.0, 0.0) - b_spline(x)).abs() < 1e-7);
        }
    }

    #[test]
    fn b_spline_approximates_not_interpolates() {
        // The defining property of the B-spline (the approximating member
        // of the BC family): k(0) = 4/6, NOT 1 — so a sample landing on a
        // pixel centre is still blended with its neighbours rather than
        // reproduced. The ±1 neighbours carry 1/6 each.
        assert!(
            (b_spline(0.0) - 4.0 / 6.0).abs() < 1e-6,
            "k(0) must be 4/6, got {}",
            b_spline(0.0)
        );
        assert!(
            (b_spline(1.0) - 1.0 / 6.0).abs() < 1e-6,
            "k(1) must be 1/6, got {}",
            b_spline(1.0)
        );
        assert!(
            (b_spline(-1.0) - 1.0 / 6.0).abs() < 1e-6,
            "k(-1) must be 1/6, got {}",
            b_spline(-1.0)
        );
        // Vanishes at and beyond the support edge.
        assert_eq!(b_spline(2.0), 0.0);
        assert_eq!(b_spline(-2.0), 0.0);
        assert_eq!(b_spline(2.5), 0.0);
    }

    #[test]
    fn b_spline_kernel_is_even_symmetric() {
        for x in [0.25f32, 0.5, 0.75, 1.25, 1.5, 1.75] {
            assert!(
                (b_spline(x) - b_spline(-x)).abs() < 1e-6,
                "asymmetry at |x|={}",
                x
            );
        }
    }

    #[test]
    fn b_spline_kernel_is_everywhere_non_negative() {
        // Unlike Mitchell (tiny negative tail) and Catmull–Rom (real
        // side-lobe), the B = 1, C = 0 spline never goes negative — that
        // is exactly why it cannot ring or overshoot. The mathematical
        // value is non-negative on the whole support; the tolerance only
        // absorbs f32 cancellation noise near |x| → 2, where the
        // expanded-polynomial form recovers (2 − |x|)³ / 6 from
        // near-cancelling cubic terms (~1.6e-7 of round-off).
        for i in -200..=200 {
            let x = i as f32 / 100.0;
            assert!(
                b_spline(x) >= -1e-6,
                "B-spline must be non-negative everywhere, got k({})={}",
                x,
                b_spline(x)
            );
        }
    }

    #[test]
    fn b_spline_kernel_partition_of_unity_on_offset_grid() {
        // Like every BC-family cubic, the four weights across a 4-tap
        // window sum to 1 at any fractional offset.
        for frac in [0.0f32, 0.1, 0.25, 0.5, 0.75, 0.9] {
            let sum = b_spline(-1.0 - frac)
                + b_spline(-frac)
                + b_spline(1.0 - frac)
                + b_spline(2.0 - frac);
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "partition broken at frac={}, sum={}",
                frac,
                sum
            );
        }
    }
}
