//! Top-level renderer: walks a [`VectorFrame`]'s scene graph and
//! produces a packed `Rgba` [`VideoFrame`].
//!
//! See the crate-level docs for the overall pipeline.

use oxideav_core::{
    FillRule, Frame, Group, ImageRef, MaskKind, Node, Paint, Path, PathNode, Rect, Rgba, Stroke,
    Transform2D, VectorFrame, VideoFrame, VideoPlane,
};

use crate::cache::{composite_key, CacheStats, RasterizedSubtree, SharedCache};
use crate::composite::composite_rgba_premultiplied;
use crate::fill::{rasterize_fill, AlphaMask};
use crate::flatten::{flatten_path, FlatContour};
use crate::paint::sample_paint;
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
        // (or default to "user space already in pixels").
        let initial = if let Some(vb) = frame.view_box {
            let sx = self.width as f32 / vb.width.max(1e-9);
            let sy = self.height as f32 / vb.height.max(1e-9);
            // Translate so view-box top-left lands at (0, 0), then scale.
            Transform2D::scale(sx, sy).compose(&Transform2D::translate(-vb.min_x, -vb.min_y))
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
        // same effective transform reuses the prior bitmap. The cache
        // bakes in the group's *own* opacity (`g.opacity`) but never
        // the parent's, so parent-opacity changes don't invalidate the
        // cache; the parent's clip mask is also applied at blit time
        // (not baked in) for the same reason.
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
                clip_mask,
                parent_opacity,
            );
            return;
        }
        // Build the group's own clip mask if it has one. The clip
        // mask is filled with NonZero of the clip path under the
        // *current* (post-transform) coordinate system, intersected
        // with whatever clip mask is being inherited.
        let mut group_clip_storage: Option<AlphaMask> = None;
        let effective_clip: Option<&AlphaMask> = if let Some(clip_path) = &g.clip {
            let cs = flatten_path(&clip_path.commands, &local);
            let m = rasterize_fill(
                &cs,
                self.width,
                self.height,
                FillRule::NonZero,
                self.supersampling,
            );
            let intersected = match clip_mask {
                Some(parent) => intersect_masks(parent, &m),
                None => m,
            };
            group_clip_storage = Some(intersected);
            group_clip_storage.as_ref()
        } else {
            clip_mask
        };
        let combined = parent_opacity * g.opacity;
        for child in &g.children {
            self.draw_node(child, local, combined, buf, stride, effective_clip);
        }
    }

    /// Render `g`'s children onto a fresh canvas-sized RGBA buffer at
    /// `local` transform, with `g.opacity` as the modulator and the
    /// group's own clip path (but no inherited clip — that's applied
    /// at blit time). Stores the result in the bitmap cache and
    /// returns a clone for immediate reuse.
    fn render_group_offscreen(
        &self,
        g: &Group,
        local: Transform2D,
        composite: u64,
    ) -> RasterizedSubtree {
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
            self.draw_node(child, local, g.opacity, &mut off, stride, effective_clip);
        }
        let subtree = RasterizedSubtree {
            width: self.width,
            height: self.height,
            rgba: off,
            transform_at_cache_time: local,
        };
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
        // Build a coverage AlphaMask from the mask buffer.
        let cov = mask_buffer_to_alpha(&mask_buf, self.width, self.height, off_stride, kind);
        // Intersect with any inherited clip.
        let cov = match clip_mask {
            Some(c) => intersect_masks(c, &cov),
            None => cov,
        };
        // Blit content_buf onto buf, modulated by cov (soft-mask
        // coverage, treated as a per-pixel alpha multiplier exactly
        // like a clip mask) + group_opacity.
        blit_rgba_over(
            buf,
            stride,
            self.width,
            self.height,
            &content_buf,
            self.width,
            self.height,
            Some(&cov),
            group_opacity,
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
            let mask = rasterize_fill(
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
            self.composite_with_paint(buf, stride, &mask, fill, group_opacity);
        }
        // Stroke pass.
        if let Some(stroke) = &node.stroke {
            let stroke_geom = build_stroke_geometry(&node.path, &transform, stroke);
            let mask = rasterize_fill(
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
            self.composite_with_paint(buf, stride, &mask, &stroke.paint, group_opacity);
        }
    }

    fn composite_with_paint(
        &self,
        buf: &mut [u8],
        stride: usize,
        mask: &AlphaMask,
        paint: &Paint,
        group_opacity: f32,
    ) {
        if mask.is_empty() {
            return;
        }
        // Clone gradient-bearing paints so the per-pixel sampler
        // doesn't have to re-resolve at each call.
        match paint {
            Paint::Solid(c) => {
                let c = *c;
                composite_rgba_premultiplied(
                    buf,
                    stride,
                    self.width,
                    self.height,
                    mask,
                    0,
                    0,
                    group_opacity,
                    move |_x, _y| c,
                );
            }
            // Gradient + future non-Solid variants. Cloned upfront so
            // the closure owns its data and survives the composite
            // call's lifetime.
            other => {
                let other = other.clone();
                composite_rgba_premultiplied(
                    buf,
                    stride,
                    self.width,
                    self.height,
                    mask,
                    0,
                    0,
                    group_opacity,
                    move |x, y| sample_paint(&other, x as f32 + 0.5, y as f32 + 0.5),
                );
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
        let mask = rasterize_fill(
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
        // Inverse transform pixel → user-space → image local UV.
        let inv = match invert_2d(&local) {
            Some(t) => t,
            None => return,
        };
        let bounds = img.bounds;
        let frame = img.frame.clone();
        let filter = self.image_filter;
        composite_rgba_premultiplied(
            buf,
            stride,
            self.width,
            self.height,
            &mask,
            0,
            0,
            group_opacity,
            move |x, y| {
                let user = inv.apply(oxideav_core::Point::new(x as f32 + 0.5, y as f32 + 0.5));
                match filter {
                    ImageFilter::Nearest => sample_image_nearest(&frame, &bounds, user.x, user.y),
                    ImageFilter::Bilinear => sample_image_bilinear(&frame, &bounds, user.x, user.y),
                }
            },
        );
    }
}

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
/// `clip_mask` (when `Some`) further multiplies the source alpha
/// per-pixel; `extra_alpha` (in `0.0..=1.0`) modulates the source
/// alpha globally (used for parent group opacity). The source and
/// destination share the same canvas size and stride; the blit
/// touches only pixels where the source alpha (after clip + extra)
/// is non-zero.
fn blit_rgba_over(
    dst: &mut [u8],
    dst_stride: usize,
    dst_width: u32,
    dst_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
    clip_mask: Option<&AlphaMask>,
    extra_alpha: f32,
) {
    if src_width == 0 || src_height == 0 {
        return;
    }
    let extra = extra_alpha.clamp(0.0, 1.0);
    let extra_q = (extra * 255.0).round() as u32;
    if extra_q == 0 {
        return;
    }
    let w = src_width.min(dst_width) as usize;
    let h = src_height.min(dst_height) as usize;
    let src_stride = (src_width as usize) * 4;
    for y in 0..h {
        let dst_row = y * dst_stride;
        let src_row = y * src_stride;
        for x in 0..w {
            let si = src_row + x * 4;
            let sa_raw = src[si + 3] as u32;
            if sa_raw == 0 {
                continue;
            }
            let cov = match clip_mask {
                Some(c) => c.get(x as u32, y as u32) as u32,
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
            let sa = combined;
            // Promote source RGB into premultiplied space (src is
            // straight-alpha at sa_raw — re-premultiply with the
            // *combined* alpha so the over operator behaves).
            let sr = (src[si] as u32 * sa + 127) / 255;
            let sg = (src[si + 1] as u32 * sa + 127) / 255;
            let sb = (src[si + 2] as u32 * sa + 127) / 255;

            let pidx = dst_row + x * 4;
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
            let (or_s, og_s, ob_s) = if oa == 0 {
                (0u32, 0u32, 0u32)
            } else {
                (
                    ((or * 255 + oa / 2) / oa).min(255),
                    ((og * 255 + oa / 2) / oa).min(255),
                    ((ob * 255 + oa / 2) / oa).min(255),
                )
            };
            dst[pidx] = or_s as u8;
            dst[pidx + 1] = og_s as u8;
            dst[pidx + 2] = ob_s as u8;
            dst[pidx + 3] = oa.min(255) as u8;
        }
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
fn mask_buffer_to_alpha(
    buf: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    kind: MaskKind,
) -> AlphaMask {
    let mut out = AlphaMask::new(width, height);
    for y in 0..height {
        let row = (y as usize) * stride;
        for x in 0..width {
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
}
