//! Top-level renderer: walks a [`VectorFrame`]'s scene graph and
//! produces a packed `Rgba` [`VideoFrame`].
//!
//! See the crate-level docs for the overall pipeline.

use std::sync::Arc;

use oxideav_core::{
    ExecutionContext, FillRule, Frame, Group, ImageRef, MaskKind, Node, Paint, Path, PathNode,
    Point, Rect, Rgba, Stroke, Transform2D, VectorFrame, VideoFrame, VideoPlane,
};

use crate::blend::BlendMode;
use crate::bounds::{group_content_bounds, node_bounds, path_node_bounds};
use crate::cache::{BitmapKey, CacheStats, CachedBitmap, GroupCache, RenderSettings, snap_axis};
use crate::composite::{SolidSource, composite_coverage_rect, composite_rect_in, composite_row};
use crate::fill::{AlphaMask, Coverage, CoverageMode, EdgeTable, rasterize_fill_with_bounds};
use crate::filter::PreserveAspectRatio;
use crate::flatten::{FLATTEN_TOLERANCE_PX, FlatContour, FlattenOptions, flatten_path_with};
use crate::gradient::InterpolationSpace;
use crate::gradient_paint::{GradientPaint, GradientSampler, invert_2d, path_bounding_box};
use crate::gradient_row::{GradientDither, GradientRows};
use crate::paint::sample_paint_in;
use crate::pattern::{
    Pattern, sample_tile_bilinear_wrap, sample_tile_nearest_wrap, view_box_fit_transform,
};
use crate::stroke::stroke_to_fill_path_bounded;

/// Top-level vector→raster renderer.
///
/// Mutates an internal `Vec<u8>` packed `Rgba` buffer through the
/// scene walk; returns the buffer wrapped in a
/// [`VideoFrame`](oxideav_core::VideoFrame) at the end of
/// [`Renderer::render`].
///
/// Holds a [`GroupCache`] shared across `render` calls — when a
/// [`Group`](oxideav_core::Group) carries a `cache_key` its rasterised
/// children are memoised, so the same glyph or decorative subtree is
/// rasterised once and blitted wherever it appears again (at any
/// whole-pixel position). A long-lived `Renderer` keeps its cache by
/// itself; renderers built per frame share one through
/// [`Renderer::with_group_cache`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Renderer {
    /// Output canvas width in pixels.
    pub width: u32,
    /// Output canvas height in pixels.
    pub height: u32,
    /// Anti-aliasing coverage kernel for every fill, stroke, clip and
    /// image-rectangle mask. Defaults to [`CoverageMode::Analytic`]
    /// (exact pixel-area coverage, no sampling grid). Select
    /// [`CoverageMode::Supersampled`] to use the historical vertical
    /// supersampler driven by [`Renderer::supersampling`].
    pub coverage: CoverageMode,
    /// Per-axis-Y vertical supersampling factor for AA. 1, 2, 4, or 8.
    /// Defaults to 4. Other values are clamped to the closest valid
    /// value at fill time. Only consulted when [`Renderer::coverage`]
    /// is [`CoverageMode::Supersampled`].
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
    /// Quantisation of gradient paints to 8 bits. Defaults to
    /// [`GradientDither::None`] (round to nearest);
    /// [`GradientDither::Ordered`] trades a fixed ±½-level texture for
    /// banding-free wide, low-contrast ramps.
    pub gradient_dither: GradientDither,
    /// Whole-scene supersampling factor per axis for
    /// [`Renderer::render`] (`1`, the default, disables it; values are
    /// clamped to `1..=8`).
    ///
    /// Per-shape anti-aliasing composites each shape's edge coverage
    /// independently, so two shapes that share an edge let the
    /// background bleed through the seam (*conflation*: two 50 %
    /// coverages leave 25 % of the background). Rendering the scene at
    /// `k×` and averaging `k × k` blocks bounds that bleed by `1/(4k)`
    /// of the background (`k = 4`: ≤ 6 %, against up to 25 %), at
    /// `k²` the time and memory.
    pub scene_supersampling: u8,
    /// Colour space of source-over compositing for painted paths,
    /// strokes, images and patterns (with [`BlendMode::Normal`]).
    /// Defaults to [`InterpolationSpace::Srgb`] — the SVG / CSS model,
    /// where anti-aliased edges and translucent paints mix encoded sRGB
    /// values. [`InterpolationSpace::LinearRgb`] mixes linear light
    /// instead: edges keep their perceived weight (black text on white
    /// no longer looks heavier than white on black) and translucent
    /// colours blend like light. Group-opacity, soft-mask and cached
    /// bitmap blits and non-normal blend modes stay in sRGB.
    pub compositing_space: InterpolationSpace,
    /// Worker budget for internal parallelism (row bands of large
    /// shapes). `1` (the default) renders serially; set it through
    /// [`Renderer::with_execution_context`]. Output bytes never depend
    /// on it.
    pub threads: usize,
    /// Sub-pixel grid of the group cache, in steps per pixel
    /// (default [`DEFAULT_CACHE_SUBPIXEL_GRID`] = 4, i.e. ¼ px).
    ///
    /// A group with a `cache_key` is drawn at its translation snapped
    /// to this grid, so that every position sharing a sub-pixel phase
    /// reuses one bitmap: the group moves by at most `1 / (2 · grid)`
    /// px per axis from where the same group without a key would be
    /// drawn (⅛ px at the default), and not at all when its translation
    /// already lies on the grid. `1` snaps to whole pixels (one bitmap
    /// per glyph, the least anti-aliasing fidelity); larger values
    /// trade more cache entries for finer placement; `0` keeps the
    /// exact fractional position (bitmaps are then shared only between
    /// positions whose fractional parts are bit-identical).
    pub cache_subpixel_grid: u8,
    /// Bitmap cache for memoised group subtrees (a shared handle, so
    /// it survives `Clone` and can be handed to other renderers).
    cache: GroupCache,
    /// `Some((y0, y1))` on a scene-band worker: this renderer owns
    /// canvas rows `y0..y1` only, and the pixel buffer handed through
    /// the scene walk holds exactly those rows. Geometry is still
    /// processed in full canvas coordinates, so every row comes out as
    /// in a serial render.
    band: Option<(u32, u32)>,
}

/// Texture sampling filter for `Node::Image` paints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
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

/// Largest [`Renderer::scene_supersampling`] factor.
const MAX_SCENE_SUPERSAMPLING: u8 = 8;

/// Paints one coverage row: `(dst, y, x0, coverage, scratch)`, `dst`
/// being the RGBA bytes of device pixels `x0 .. x0 + coverage.len()` on
/// row `y`.
type RowPainter<'a> = dyn Fn(&mut [u8], u32, u32, &[u8], &mut Vec<Rgba>) + Sync + 'a;

/// A shape is scanned in parallel bands only when its window holds at
/// least this many pixels (below it, thread start-up dominates).
const PARALLEL_MIN_AREA: u64 = 1 << 18;

/// Minimum rows per parallel band.
const PARALLEL_MIN_ROWS: u32 = 64;

/// Minimum rows per scene band (see [`Renderer::draw_scene`]).
const SCENE_BAND_MIN_ROWS: u32 = 16;

/// Historical default entry count of the group cache.
///
/// The cache is now bounded by bytes
/// ([`DEFAULT_CACHE_BYTES`](crate::DEFAULT_CACHE_BYTES)) with no entry
/// limit unless [`Renderer::with_cache_capacity`] sets one; this
/// constant is kept for callers that pass it there.
pub const DEFAULT_CACHE_CAPACITY: usize = 256;

/// Default [`Renderer::cache_subpixel_grid`]: ¼ px.
pub const DEFAULT_CACHE_SUBPIXEL_GRID: u8 = 4;

/// A cached group may be rasterised at up to four times its visible
/// area plus this many pixels; a group that is mostly off-canvas beyond
/// that is drawn directly (clipped to the canvas) instead.
const CACHE_OFFCANVAS_SLACK_PX: u64 = 1 << 16;

impl Renderer {
    /// Build a renderer for the given destination size with sane
    /// defaults (exact-area anti-aliasing, `supersampling = 4` for the
    /// supersampled mode, transparent background, no sub-pixel
    /// positioning, bilinear image filtering, a 32 MiB group cache on a
    /// ¼ px sub-pixel grid).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            coverage: CoverageMode::Analytic,
            supersampling: 4,
            subpixel_positioning: false,
            background: Rgba::new(0, 0, 0, 0),
            image_filter: ImageFilter::default(),
            color_interpolation: InterpolationSpace::default(),
            blend_mode: BlendMode::default(),
            preserve_aspect_ratio: PreserveAspectRatio::default(),
            gradient_dither: GradientDither::default(),
            compositing_space: InterpolationSpace::default(),
            scene_supersampling: 1,
            threads: 1,
            cache_subpixel_grid: DEFAULT_CACHE_SUBPIXEL_GRID,
            cache: GroupCache::new(),
            band: None,
        }
    }

    /// Build a renderer whose group cache additionally holds at most
    /// `capacity` entries (the byte budget still applies). Prefer
    /// [`Renderer::with_cache_bytes`]: bitmaps differ in size by orders
    /// of magnitude, so a byte budget is the meaningful bound.
    pub fn with_cache_capacity(width: u32, height: u32, capacity: usize) -> Self {
        Self {
            cache: GroupCache::new().with_max_entries(capacity),
            ..Self::new(width, height)
        }
    }

    /// Give this renderer a fresh group cache holding at most
    /// `max_bytes` of pixel data (`0` disables caching: keyed groups
    /// are drawn directly).
    pub fn with_cache_bytes(mut self, max_bytes: usize) -> Self {
        self.cache = GroupCache::with_max_bytes(max_bytes);
        self
    }

    /// Use `cache` — typically one kept by the caller across frames, or
    /// shared with other renderers — instead of this renderer's own.
    /// Entries record every setting that affects their pixels, so
    /// renderers of different sizes or settings can share one cache.
    pub fn with_group_cache(mut self, cache: GroupCache) -> Self {
        self.cache = cache;
        self
    }

    /// A handle to this renderer's group cache (to keep it past the
    /// renderer's lifetime, share it, resize or clear it).
    pub fn group_cache(&self) -> GroupCache {
        self.cache.clone()
    }

    /// Set the group cache's sub-pixel grid (builder form of the
    /// [`Renderer::cache_subpixel_grid`] field).
    pub fn with_cache_subpixel_grid(mut self, steps_per_pixel: u8) -> Self {
        self.cache_subpixel_grid = steps_per_pixel;
        self
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
        let mut buf = self.cleared_canvas();
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
        let k = self.scene_supersampling.clamp(1, MAX_SCENE_SUPERSAMPLING) as u32;
        if k > 1 {
            // Render the whole scene at k× in both axes, then box-filter
            // it down in premultiplied space.
            let mut big = self.clone();
            big.width = self.width * k;
            big.height = self.height * k;
            big.scene_supersampling = 1;
            let big_stride = big.width as usize * 4;
            let mut hi = vec![0u8; big_stride * big.height as usize];
            if self.background.a != 0 {
                let bg = [
                    self.background.r,
                    self.background.g,
                    self.background.b,
                    self.background.a,
                ];
                for px in hi.as_chunks_mut::<4>().0 {
                    *px = bg;
                }
            }
            let up = Transform2D::scale(k as f32, k as f32).compose(&initial);
            big.draw_scene(&frame.root, up, &mut hi, big_stride);
            downsample_premultiplied(
                &hi,
                big_stride,
                k,
                &mut buf,
                stride,
                self.width,
                self.height,
            );
        } else {
            self.draw_scene(&frame.root, initial, &mut buf, stride);
        }
        VideoFrame {
            pts: frame.pts,
            planes: vec![VideoPlane { stride, data: buf }],
        }
    }

    /// Walk the scene into `buf` (the whole canvas).
    ///
    /// With a thread budget the canvas is cut into horizontal bands,
    /// one worker each. Every worker walks the whole scene with the
    /// same geometry in the same canvas coordinates — same flattened
    /// outlines, same edge tables, same offscreen windows — and only
    /// restricts which rows it scans, composites and blits, skipping
    /// nodes whose bounds miss its band. A row's pixels depend on
    /// nothing outside the row's own inputs, so the result is
    /// byte-identical to the serial walk at every budget.
    fn draw_scene(&self, root: &Group, transform: Transform2D, buf: &mut [u8], stride: usize) {
        let workers = self
            .threads
            .min((self.height / SCENE_BAND_MIN_ROWS) as usize)
            .max(1);
        if workers <= 1 || stride == 0 {
            self.draw_group(root, transform, 1.0, buf, stride, None);
            return;
        }
        let band_rows = self.height.div_ceil(workers as u32);
        std::thread::scope(|sc| {
            for (i, chunk) in buf.chunks_mut(band_rows as usize * stride).enumerate() {
                let y0 = i as u32 * band_rows;
                let mut worker = self.clone();
                worker.band = Some((y0, (y0 + band_rows).min(self.height)));
                worker.threads = 1;
                sc.spawn(move || worker.draw_group(root, transform, 1.0, chunk, stride, None));
            }
        });
    }

    /// The canvas rows `y0..y1` this renderer draws (all of them unless
    /// it is a scene-band worker).
    fn rows(&self) -> (u32, u32) {
        self.band.unwrap_or((0, self.height))
    }

    /// Whether canvas rows `y0..y1` overlap this renderer's rows.
    fn owns_rows(&self, y0: i64, y1: i64) -> bool {
        let (b0, b1) = self.rows();
        y0 < b1 as i64 && y1 > b0 as i64
    }

    /// [`blit_rgba_at`] onto this renderer's rows of the canvas.
    #[allow(clippy::too_many_arguments)]
    fn blit(
        &self,
        buf: &mut [u8],
        stride: usize,
        src: &[u8],
        src_width: u32,
        src_height: u32,
        offset_x: i64,
        offset_y: i64,
        mask: Option<(&AlphaMask, i64, i64)>,
        extra_alpha: f32,
        src_rect: Option<(u32, u32, u32, u32)>,
    ) {
        let (b0, b1) = self.rows();
        blit_rgba_at(
            buf,
            stride,
            self.width,
            b1 - b0,
            src,
            src_width,
            src_height,
            offset_x,
            offset_y - b0 as i64,
            mask.map(|(m, mx, my)| (m, mx, my - b0 as i64)),
            extra_alpha,
            src_rect,
        );
    }

    /// Composite `paint_at` (device coordinates) through `cov` over the
    /// inclusive device rectangle `rect`, onto this renderer's rows.
    fn composite_cov(
        &self,
        buf: &mut [u8],
        stride: usize,
        cov: &Coverage,
        extra_alpha: f32,
        rect: (u32, u32, u32, u32),
        mut paint_at: impl FnMut(u32, u32) -> Rgba,
    ) {
        let (rx0, ry0, rx1, ry1) = rect;
        if cov.mask.is_empty() || rx1 < cov.x0 || ry1 < cov.y0 {
            return;
        }
        let (b0, b1) = self.rows();
        composite_rect_in(
            buf,
            stride,
            self.width,
            b1 - b0,
            &cov.mask,
            cov.x0 as i32,
            cov.y0 as i32 - b0 as i32,
            extra_alpha,
            self.blend_mode,
            matches!(self.compositing_space, InterpolationSpace::LinearRgb),
            (
                rx0.saturating_sub(cov.x0),
                ry0.saturating_sub(cov.y0),
                rx1 - cov.x0,
                ry1 - cov.y0,
            ),
            // The buffer's row 0 is canvas row `b0`.
            |x, y| paint_at(x, y + b0),
        );
    }

    /// [`Renderer::render`] behind a scene check: a NaN or ±∞ anywhere
    /// in the scene's geometry (coordinates, transforms, stroke
    /// parameters, gradient geometry, opacities) is reported as a typed
    /// [`RenderError`](crate::RenderError) instead of being rendered
    /// with the offending segments dropped.
    pub fn try_render(&self, frame: &VectorFrame) -> Result<VideoFrame, crate::RenderError> {
        crate::validate_frame(frame)?;
        Ok(self.render(frame))
    }

    /// Render whole scenes at `k×` resolution per axis (clamped to
    /// `1..=8`) and box-filter down (builder form of the
    /// [`Renderer::scene_supersampling`] field).
    pub fn with_scene_supersampling(mut self, k: u8) -> Self {
        self.scene_supersampling = k.clamp(1, MAX_SCENE_SUPERSAMPLING);
        self
    }

    /// Adopt the caller's thread budget (`oxideav-core`'s threading
    /// contract: serial until told otherwise, fan-out bounded by
    /// [`ExecutionContext::effective_workers`]). Large fills and
    /// strokes are then scanned in concurrent row bands; the output is
    /// byte-identical to the serial render.
    pub fn with_execution_context(mut self, ctx: &ExecutionContext) -> Self {
        self.threads = ctx.effective_workers(usize::MAX).max(1);
        self
    }

    /// Select the compositing space (builder form of the
    /// [`Renderer::compositing_space`] field).
    pub fn with_compositing_space(mut self, space: InterpolationSpace) -> Self {
        self.compositing_space = space;
        self
    }

    /// Select the gradient quantisation (builder form of the
    /// [`Renderer::gradient_dither`] field).
    pub fn with_gradient_dither(mut self, dither: GradientDither) -> Self {
        self.gradient_dither = dither;
        self
    }

    /// Select the anti-aliasing coverage kernel (builder form of the
    /// [`Renderer::coverage`] field).
    pub fn with_coverage(mut self, coverage: CoverageMode) -> Self {
        self.coverage = coverage;
        self
    }

    /// A canvas-sized RGBA buffer cleared to [`Renderer::background`]
    /// (a transparent background costs only the zeroed allocation).
    fn cleared_canvas(&self) -> Vec<u8> {
        let len = (self.width as usize) * 4 * (self.height as usize);
        let bg = self.background;
        if bg.a == 0 {
            return vec![0u8; len];
        }
        let px = [bg.r, bg.g, bg.b, bg.a];
        if px[0] == px[1] && px[1] == px[2] && px[2] == px[3] {
            return vec![px[0]; len];
        }
        let mut buf = vec![0u8; len];
        for p in buf.as_chunks_mut::<4>().0 {
            *p = px;
        }
        buf
    }

    /// Rasterize `contours` into a canvas-sized alpha mask with the
    /// configured coverage kernel, reporting the touched bounds.
    fn rasterize_with_bounds(
        &self,
        contours: &[FlatContour],
        fill_rule: FillRule,
    ) -> (Coverage, Option<(u32, u32, u32, u32)>) {
        let cov = match self.coverage {
            CoverageMode::Analytic => {
                Coverage::analytic(contours, self.width, self.height, fill_rule)
            }
            _ => {
                let (m, b) = rasterize_fill_with_bounds(
                    contours,
                    self.width,
                    self.height,
                    fill_rule,
                    self.supersampling,
                );
                Coverage::from_canvas_mask(m, b)
            }
        };
        let b = cov.bounds;
        (cov, b)
    }

    /// Flatten a path for **filling** on this canvas: device-space
    /// tolerance flattening with off-canvas curves replaced by their
    /// chords (exact for a fill — see [`FlattenOptions::cull`]).
    fn flatten_fill(
        &self,
        commands: &[oxideav_core::PathCommand],
        t: &Transform2D,
    ) -> Vec<FlatContour> {
        let opts = FlattenOptions {
            cull: Some((
                -1.0,
                -1.0,
                self.width as f64 + 1.0,
                self.height as f64 + 1.0,
            )),
            ..FlattenOptions::default()
        };
        flatten_path_with(commands, t, &opts)
    }

    /// Rasterize a clip path's `contours` (non-zero rule) into a
    /// windowed coverage: the mask spans only the pixels the path can
    /// touch and reads as 0 everywhere else.
    fn rasterize_clip(&self, contours: &[FlatContour]) -> Coverage {
        self.rasterize_with_bounds(contours, FillRule::NonZero).0
    }

    /// Render a single node at the given transform. Returns a packed
    /// `Rgba` `VideoFrame` of the renderer's full canvas size, with
    /// only that node painted onto the `background` clear.
    pub fn render_node(&self, node: &Node, transform: Transform2D) -> VideoFrame {
        let stride = (self.width as usize) * 4;
        let mut buf = self.cleared_canvas();
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
        clip_mask: Option<&Coverage>,
    ) {
        let mut local = parent_transform.compose(&g.transform);
        // Bitmap-cache fast path: a group tagged with a `cache_key` is
        // rasterised once into a bitmap of its own bounds and blitted
        // from then on (see `draw_group_cached`). A keyed group that
        // cannot be cached is still composited the way a cached one is
        // — isolated, at the position the cache would have drawn it —
        // so the picture never depends on what the cache holds.
        let mut isolate = g.opacity < 1.0;
        if let Some(key) = g.cache_key {
            match self.draw_group_cached(g, key, &local, parent_opacity, buf, stride, clip_mask) {
                None => return,
                Some(uncached) => {
                    local = uncached;
                    isolate = true;
                }
            }
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
        //
        // The offscreen image covers only the part of the canvas the
        // group can touch (its analytic bounds, see `crate::bounds`).
        if isolate {
            let opacity = parent_opacity * g.opacity;
            if (opacity.clamp(0.0, 1.0) * 255.0).round() as u32 == 0 {
                return;
            }
            let Some(win) =
                group_content_bounds(g, &local).pixel_box_on_canvas(self.width, self.height)
            else {
                return;
            };
            let (ox, oy) = (window_origin(win.x0), window_origin(win.y0));
            let (w, h) = ((win.x1 - ox) as u32, (win.y1 - oy) as u32);
            // The window rows this renderer owns (all of them unless it
            // is a scene-band worker).
            let (b0, b1) = self.rows();
            let r0 = (b0 as i64 - oy).clamp(0, h as i64) as u32;
            let r1 = (b1 as i64 - oy).clamp(0, h as i64) as u32;
            if r0 >= r1 {
                return;
            }
            let off = self.render_group_window(g, &local, ox, oy, w, h, (r0, r1));
            self.blit(
                buf,
                stride,
                &off,
                w,
                r1 - r0,
                ox,
                oy + r0 as i64,
                clip_mask.map(Coverage::as_blit_mask),
                opacity,
                None,
            );
            return;
        }
        // Build the group's own clip mask if it has one. The clip
        // mask is filled with NonZero of the clip path under the
        // *current* (post-transform) coordinate system, intersected
        // with whatever clip mask is being inherited.
        let group_clip_storage: Option<Coverage> = g.clip.as_ref().map(|clip_path| {
            let cs = self.flatten_fill(&clip_path.commands, &local);
            let m = self.rasterize_clip(&cs);
            match clip_mask {
                Some(parent) => m.intersect(parent),
                None => m,
            }
        });
        let effective_clip: Option<&Coverage> = group_clip_storage.as_ref().or(clip_mask);
        // Reached only when `g.opacity == 1.0` (the partially-transparent
        // case isolated offscreen above), so the group contributes no
        // extra alpha and children inherit the parent opacity directly.
        for child in &g.children {
            self.draw_node(child, local, parent_opacity, buf, stride, effective_clip);
        }
    }

    /// Draw a keyed group through the bitmap cache. Returns `None`
    /// when the group has been drawn, or `Some(transform)` when it
    /// cannot be cached: the caller then renders it through a transient
    /// offscreen window under that transform — `local` with its
    /// translation snapped exactly as the cache would have — so a keyed
    /// group is always an isolated group at the same position, cached
    /// or not.
    ///
    /// The effective translation is split into a whole-pixel offset and
    /// a sub-pixel phase snapped to [`Renderer::cache_subpixel_grid`].
    /// The bitmap is rasterised under the transform with only the phase
    /// as its translation — so it depends on neither the pen position
    /// nor the canvas — into a buffer the size of the group's analytic
    /// bounds, and is blitted at the whole-pixel offset, clipped to the
    /// canvas. The bitmap holds the group at full opacity with its own
    /// clip path baked in; the group's opacity, the inherited opacity
    /// and the inherited clip are applied by the blit, so none of them
    /// invalidates an entry.
    ///
    /// Not cached: a non-finite or astronomically large transform
    /// (drawn unsnapped);
    /// bounds that are unbounded (non-finite geometry); a bitmap larger
    /// than a quarter of the cache budget; and a group whose bounds
    /// exceed four times their on-canvas part (plus a small slack) —
    /// rasterising all of it for the cache would cost more than
    /// rendering the visible part.
    #[allow(clippy::too_many_arguments)]
    fn draw_group_cached(
        &self,
        g: &Group,
        key: u64,
        local: &Transform2D,
        parent_opacity: f32,
        buf: &mut [u8],
        stride: usize,
        clip_mask: Option<&Coverage>,
    ) -> Option<Transform2D> {
        let grid = self.cache_subpixel_grid;
        let (Some((ix, fx)), Some((iy, fy))) = (snap_axis(local.e, grid), snap_axis(local.f, grid))
        else {
            return Some(*local);
        };
        if ![local.a, local.b, local.c, local.d]
            .iter()
            .all(|v| v.is_finite())
        {
            return Some(*local);
        }
        // Where the group is drawn when it is not blitted from the cache.
        let direct = Transform2D {
            e: (ix as f64 + fx as f64) as f32,
            f: (iy as f64 + fy as f64) as f32,
            ..*local
        };
        let snapped = Transform2D {
            e: fx,
            f: fy,
            ..*local
        };
        let bkey = BitmapKey::new(key, &snapped, self.render_settings());
        let bitmap = match self.cache.get(&bkey) {
            Some(b) => b,
            None => {
                let bounds = group_content_bounds(g, &snapped);
                let bitmap = if bounds.is_empty() {
                    CachedBitmap {
                        width: 0,
                        height: 0,
                        dx: 0,
                        dy: 0,
                        rgba: Vec::new(),
                    }
                } else {
                    let Some(pb) = bounds.pixel_box() else {
                        return Some(direct);
                    };
                    let visible = pb
                        .translated(ix, iy)
                        .area_on_canvas(self.width, self.height);
                    if visible == 0 {
                        // Entirely off-canvas: nothing to draw, and no
                        // reason to rasterise it now.
                        return None;
                    }
                    if !self.owns_rows(pb.y0 + iy, pb.y1 + iy) {
                        // Another scene band's rows: that band draws it.
                        return None;
                    }
                    let area = pb.area();
                    if area.saturating_mul(4) > self.cache.max_entry_bytes() as u64
                        || area > visible.saturating_mul(4) + CACHE_OFFCANVAS_SLACK_PX
                    {
                        return Some(direct);
                    }
                    let (w, h) = (pb.width() as u32, pb.height() as u32);
                    CachedBitmap {
                        width: w,
                        height: h,
                        dx: pb.x0,
                        dy: pb.y0,
                        rgba: self.render_group_window(g, &snapped, pb.x0, pb.y0, w, h, (0, h)),
                    }
                };
                let bitmap = Arc::new(bitmap);
                self.cache.put(bkey, Arc::clone(&bitmap));
                bitmap
            }
        };
        self.blit(
            buf,
            stride,
            &bitmap.rgba,
            bitmap.width,
            bitmap.height,
            ix + bitmap.dx,
            iy + bitmap.dy,
            clip_mask.map(Coverage::as_blit_mask),
            parent_opacity * g.opacity,
            None,
        );
        None
    }

    /// The settings that change a subtree's pixels (part of every
    /// cache entry's identity).
    fn render_settings(&self) -> RenderSettings {
        use std::mem::discriminant;
        RenderSettings {
            coverage: discriminant(&self.coverage),
            // Only the supersampled kernel reads the factor, and it
            // folds it onto 1 / 2 / 4 / 8.
            supersampling: match (self.coverage, self.supersampling) {
                (CoverageMode::Analytic, _) => 0,
                (_, 0 | 1) => 1,
                (_, 2) => 2,
                (_, 3 | 4) => 4,
                _ => 8,
            },
            image_filter: discriminant(&self.image_filter),
            color_interpolation: discriminant(&self.color_interpolation),
            blend_mode: discriminant(&self.blend_mode),
            gradient_dither: discriminant(&self.gradient_dither),
            compositing_space: discriminant(&self.compositing_space),
        }
    }

    /// Render `g`'s children — at **full opacity**, with only the
    /// group's own clip path applied — into a fresh RGBA buffer holding
    /// rows `rows.0 .. rows.1` of a `w × h` window whose top-left pixel
    /// is device pixel `(ox, oy)` of the coordinate system `local` maps
    /// into.
    ///
    /// Rendering children at full opacity (rather than pre-multiplying
    /// each child by `g.opacity`) is what makes group opacity behave as
    /// a *post-process* on the composited group rather than a per-child
    /// alpha — overlapping children inside the group do not
    /// double-darken in their overlap region (SVG 2 §3.4 / "Simple
    /// Alpha Compositing": "the group ... is rendered into an RGBA
    /// offscreen image[;] the offscreen image as [a] whole is then
    /// blended into the canvas with the specified opacity value used
    /// uniformly across the offscreen image").
    fn render_group_window(
        &self,
        g: &Group,
        local: &Transform2D,
        ox: i64,
        oy: i64,
        w: u32,
        h: u32,
        rows: (u32, u32),
    ) -> Vec<u8> {
        let mut sub = self.window_renderer(w, h);
        if rows != (0, h) {
            sub.band = Some(rows);
        }
        let t = shift_transform(local, ox, oy);
        let stride = (w as usize) * 4;
        let mut off = vec![0u8; stride * ((rows.1 - rows.0) as usize)];
        // The group's own clip only — the inherited clip is applied
        // when the window is blitted.
        let clip: Option<Coverage> = g.clip.as_ref().map(|clip_path| {
            let cs = sub.flatten_fill(&clip_path.commands, &t);
            sub.rasterize_clip(&cs)
        });
        for child in &g.children {
            sub.draw_node(child, t, 1.0, &mut off, stride, clip.as_ref());
        }
        off
    }

    /// This renderer with a `w × h` canvas (same settings, same cache).
    fn window_renderer(&self, w: u32, h: u32) -> Renderer {
        let mut sub = self.clone();
        sub.width = w;
        sub.height = h;
        sub.band = None;
        sub
    }

    fn draw_node(
        &self,
        node: &Node,
        transform: Transform2D,
        group_opacity: f32,
        buf: &mut [u8],
        stride: usize,
        clip_mask: Option<&Coverage>,
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
        clip_mask: Option<&Coverage>,
    ) {
        // Both subtrees are rasterised into a window covering only the
        // pixels where the content *and* the mask can be non-zero
        // (their analytic bounds, intersected with the canvas): outside
        // it the composite writes nothing.
        let bounds = node_bounds(content, &transform).intersect(&node_bounds(mask, &transform));
        let Some(win) = bounds.pixel_box_on_canvas(self.width, self.height) else {
            return;
        };
        let (ox, oy) = (window_origin(win.x0), window_origin(win.y0));
        let (w, h) = ((win.x1 - ox) as u32, (win.y1 - oy) as u32);
        if !self.owns_rows(oy, oy + h as i64) {
            return; // another scene band's rows
        }
        let sub = self.window_renderer(w, h);
        let t = shift_transform(&transform, ox, oy);
        let off_stride = (w as usize) * 4;
        let cap = off_stride * (h as usize);
        // Rasterise the mask subtree.
        let mut mask_buf = vec![0u8; cap];
        sub.draw_node(mask, t, 1.0, &mut mask_buf, off_stride, None);
        // Rasterise the content subtree.
        let mut content_buf = vec![0u8; cap];
        sub.draw_node(content, t, 1.0, &mut content_buf, off_stride, None);
        // Restrict the coverage conversion + blit to the intersection
        // of the content's and the mask's touched-alpha bounding
        // boxes: outside the content box the blit skips every pixel
        // (source alpha 0) no matter what the coverage says, and
        // outside the mask box the coverage is provably 0 (both mask
        // kinds multiply by / read the mask pixel's alpha), so the
        // blit skips there too.
        let Some((cx0, cy0, cx1, cy1)) = alpha_bbox(&content_buf, off_stride, w, h) else {
            return; // fully-transparent content — the blit would no-op
        };
        let Some((mx0, my0, mx1, my1)) = alpha_bbox(&mask_buf, off_stride, w, h) else {
            return; // fully-transparent mask — zero coverage everywhere
        };
        let bbox = (cx0.max(mx0), cy0.max(my0), cx1.min(mx1), cy1.min(my1));
        if bbox.0 > bbox.2 || bbox.1 > bbox.3 {
            return; // disjoint content / mask — nothing is composited
        }
        // Build a coverage AlphaMask (window coordinates) from the mask
        // buffer, inside the bbox only; everywhere else it stays 0,
        // unread.
        let mut cov = mask_buffer_to_alpha_rect(&mask_buf, w, h, off_stride, kind, bbox);
        // Intersect with any inherited clip (canvas coordinates).
        if let Some(c) = clip_mask {
            let (bx0, by0, bx1, by1) = bbox;
            for y in by0..=by1 {
                let row = (y * w) as usize;
                for x in bx0..=bx1 {
                    let i = row + x as usize;
                    cov.data[i] = cov.data[i].min(c.get(x + ox as u32, y + oy as u32));
                }
            }
        }
        // Blit content_buf onto buf, modulated by cov (soft-mask
        // coverage, treated as a per-pixel alpha multiplier exactly
        // like a clip mask) + group_opacity.
        self.blit(
            buf,
            stride,
            &content_buf,
            w,
            h,
            ox,
            oy,
            Some((&cov, ox, oy)),
            group_opacity,
            Some(bbox),
        );
    }

    fn draw_path(
        &self,
        node: &PathNode,
        transform: Transform2D,
        group_opacity: f32,
        buf: &mut [u8],
        stride: usize,
        clip_mask: Option<&Coverage>,
    ) {
        // A scene-band worker skips paths that cannot reach its rows
        // (the bounds are conservative, so nothing it would have drawn
        // is lost).
        if self.band.is_some() {
            match path_node_bounds(node, &transform).pixel_box_on_canvas(self.width, self.height) {
                Some(pb) if self.owns_rows(pb.y0, pb.y1) => {}
                _ => return,
            }
        }
        // Fill pass.
        if let Some(fill) = &node.fill {
            let contours = self.flatten_fill(&node.path.commands, &transform);
            self.paint_contours(
                buf,
                stride,
                &contours,
                node.fill_rule,
                clip_mask,
                fill,
                &transform,
                group_opacity,
            );
        }
        // Stroke pass.
        if let Some(stroke) = &node.stroke {
            let stroke_geom =
                build_stroke_geometry(&node.path, &transform, stroke, (self.width, self.height));
            self.paint_contours(
                buf,
                stride,
                &stroke_geom,
                FillRule::NonZero,
                clip_mask,
                &stroke.paint,
                &transform,
                group_opacity,
            );
        }
    }

    /// Stream the coverage of `contours`, intersected with `clip`
    /// (per-pixel minimum), into `paint_row(dst, y, x0, coverage,
    /// scratch)` calls — `dst` being the packed RGBA bytes of device
    /// pixels `x0 .. x0 + coverage.len()` on row `y`. The analytic
    /// kernel never materialises a mask: each row is composited while
    /// it is still in cache.
    ///
    /// With a thread budget ([`Renderer::with_execution_context`]) a
    /// large shape's window is cut into horizontal bands scanned and
    /// painted concurrently. Every row is computed by exactly the same
    /// code from the same edge table whichever band it falls in (the
    /// scan carries no state between rows beyond the monotone chain
    /// cursors, which a band re-derives), so the output is
    /// byte-identical to the serial render.
    #[allow(clippy::too_many_arguments)]
    fn paint_rows(
        &self,
        buf: &mut [u8],
        stride: usize,
        contours: &[FlatContour],
        fill_rule: FillRule,
        clip: Option<&Coverage>,
        paint_row: &RowPainter<'_>,
    ) {
        // Clip, then hand a row to the painter.
        let deliver = |dst_rows: &mut [u8],
                       row_base: u32,
                       y: u32,
                       x0: u32,
                       row: &[u8],
                       tmp: &mut Vec<u8>,
                       scratch: &mut Vec<Rgba>| {
            // Under a clip only the part of the row inside the clip's
            // window can be painted (the clip is 0 outside it).
            let (x0, cov): (u32, &[u8]) = match clip {
                None => (x0, row),
                Some(c) => {
                    let Some(crow) = c.row(y) else {
                        return;
                    };
                    let lo = x0.max(c.x0);
                    let hi = (x0 + row.len() as u32).min(c.x0 + c.mask.width);
                    if lo >= hi {
                        return;
                    }
                    tmp.clear();
                    tmp.extend(
                        row[(lo - x0) as usize..(hi - x0) as usize]
                            .iter()
                            .zip(&crow[(lo - c.x0) as usize..(hi - c.x0) as usize])
                            .map(|(a, b)| (*a).min(*b)),
                    );
                    (lo, tmp)
                }
            };
            let off = (y - row_base) as usize * stride + x0 as usize * 4;
            paint_row(&mut dst_rows[off..off + cov.len() * 4], y, x0, cov, scratch);
        };
        let (mut tmp, mut scratch) = (Vec::new(), Vec::new());
        match self.coverage {
            CoverageMode::Analytic => {
                let Some(t) = EdgeTable::build(contours, self.width, self.height, None) else {
                    return;
                };
                let win = t.window();
                // Window-local rows this renderer owns (the whole
                // window unless it is a scene-band worker); the table
                // itself is always the full one, so a row's coverage
                // does not depend on who scans it.
                let (b0, b1) = self.rows();
                let r0 = b0.max(win.y0) - win.y0;
                let r1 = b1.min(win.y1).saturating_sub(win.y0);
                if r0 >= r1 {
                    return;
                }
                let rows = r1 - r0;
                let area = rows as u64 * (win.x1 - win.x0) as u64;
                let workers = if area >= PARALLEL_MIN_AREA {
                    self.threads
                        .min((rows / PARALLEL_MIN_ROWS).max(1) as usize)
                        .max(1)
                } else {
                    1
                };
                if workers <= 1 {
                    t.scan_rows(fill_rule, r0..r1, |y, x0, row| {
                        deliver(buf, b0, y, x0, row, &mut tmp, &mut scratch)
                    });
                    return;
                }
                let band = rows.div_ceil(workers as u32);
                let first = (win.y0 + r0 - b0) as usize * stride;
                let region = &mut buf[first..first + rows as usize * stride];
                std::thread::scope(|sc| {
                    for (k, chunk) in region.chunks_mut(band as usize * stride).enumerate() {
                        let t = &t;
                        let deliver = &deliver;
                        let q0 = r0 + k as u32 * band;
                        let q1 = (q0 + band).min(r1);
                        sc.spawn(move || {
                            let (mut tmp, mut scratch) = (Vec::new(), Vec::new());
                            t.scan_rows(fill_rule, q0..q1, |y, x0, row| {
                                deliver(chunk, win.y0 + q0, y, x0, row, &mut tmp, &mut scratch)
                            });
                        });
                    }
                });
            }
            _ => {
                let (cov, bounds) = self.rasterize_with_bounds(contours, fill_rule);
                let Some((bx0, by0, bx1, by1)) = bounds else {
                    return;
                };
                let mw = cov.mask.width as usize;
                let (b0, b1) = self.rows();
                for y in by0.max(b0)..=by1.min(b1.saturating_sub(1)) {
                    let off = (y - cov.y0) as usize * mw + (bx0 - cov.x0) as usize;
                    let row = &cov.mask.data[off..off + (bx1 - bx0 + 1) as usize];
                    deliver(buf, b0, y, bx0, row, &mut tmp, &mut scratch);
                }
            }
        }
    }

    /// Paint `contours` (filled with `fill_rule`, clipped by `clip`)
    /// with `paint` onto `buf`.
    ///
    /// `transform` is the current transformation matrix (CTM) the
    /// painted geometry was flattened with — user space → device
    /// pixels. A [`Paint::Solid`] ignores it. Gradient paints are
    /// defined in the painted element's user space (SVG 2 §14.2.2.1 /
    /// §14.2.3.1 `gradientUnits="userSpaceOnUse"`: "the user coordinate
    /// system for the element referencing the gradient"), so every
    /// device pixel centre is mapped back through the inverse CTM
    /// before the gradient is evaluated. That is the same mapping the
    /// path itself went through — the gradient geometry therefore
    /// follows the `viewBox` → viewport scale, every ancestor group
    /// `transform` and the element's own `transform`, and a radial
    /// gradient under a non-uniform CTM renders elliptical rings
    /// exactly as the spec describes, without any special casing.
    #[allow(clippy::too_many_arguments)]
    fn paint_contours(
        &self,
        buf: &mut [u8],
        stride: usize,
        contours: &[FlatContour],
        fill_rule: FillRule,
        clip: Option<&Coverage>,
        paint: &Paint,
        transform: &Transform2D,
        group_opacity: f32,
    ) {
        let extra_q = (group_opacity.clamp(0.0, 1.0) * 255.0).round() as u32;
        if extra_q == 0 {
            return;
        }
        let blend = self.blend_mode;
        let linear = matches!(self.compositing_space, InterpolationSpace::LinearRgb);
        match paint {
            Paint::Solid(c) if blend.is_normal() => {
                let src = SolidSource::new_in(*c, extra_q, linear);
                self.paint_rows(
                    buf,
                    stride,
                    contours,
                    fill_rule,
                    clip,
                    &|dst, _, _, cov, _| src.over_row(dst, cov),
                );
            }
            Paint::Solid(c) => {
                let c = *c;
                self.paint_rows(
                    buf,
                    stride,
                    contours,
                    fill_rule,
                    clip,
                    &|dst, y, x0, cov, _| {
                        composite_row(dst, cov, x0, y, extra_q, blend, linear, &mut |_, _| c)
                    },
                );
            }
            // Gradients: whole rows at a time, the stops evaluated
            // exactly (see `gradient_row`).
            other => {
                // Device → user space. A singular CTM collapses the
                // geometry to zero area, so there is nothing a gradient
                // could be sampled over.
                let Some(inv) = invert_2d(transform) else {
                    return;
                };
                let space = self.color_interpolation;
                if let Some(rows) = GradientRows::new(other, &inv, space, self.gradient_dither) {
                    self.paint_rows(
                        buf,
                        stride,
                        contours,
                        fill_rule,
                        clip,
                        &|dst, y, x0, cov, colors| {
                            rows.row(y, x0, cov.len(), colors);
                            composite_row(dst, cov, x0, y, extra_q, blend, linear, &mut |x, _| {
                                colors[(x - x0) as usize]
                            });
                        },
                    );
                    return;
                }
                self.paint_rows(
                    buf,
                    stride,
                    contours,
                    fill_rule,
                    clip,
                    &|dst, y, x0, cov, _| {
                        composite_row(dst, cov, x0, y, extra_q, blend, linear, &mut |x, y| {
                            let p = inv.apply(Point::new(x as f32 + 0.5, y as f32 + 0.5));
                            sample_paint_in(other, p.x, p.y, space)
                        })
                    },
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
        clip_mask: Option<&Coverage>,
    ) {
        // Build a rectangle for the image's user-space bounds, fill
        // it with NonZero, then sample the embedded frame as the
        // paint source. Sampling filter is configurable on the
        // [`Renderer`] — defaults to bilinear, with nearest-neighbour
        // available for callers that want pixel-perfect block
        // replication.
        let rect_path = rect_to_path(img.bounds);
        let local = transform.compose(&img.transform);
        let contours = self.flatten_fill(&rect_path.commands, &local);
        let (mask, fill_bounds) = self.rasterize_with_bounds(&contours, FillRule::NonZero);
        let mask = match clip_mask {
            Some(c) => mask.intersect_clip(c),
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
        self.composite_cov(
            buf,
            stride,
            &mask,
            group_opacity,
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
        mask: &Coverage,
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
                self.composite_cov(buf, stride, mask, group_opacity, rect, move |x, y| {
                    let (Some(px), Some(py)) = (cols[(x - bx0) as usize], rows[(y - by0) as usize])
                    else {
                        return Rgba::new(0, 0, 0, 0);
                    };
                    let i = py * tex_stride + px * 4;
                    Rgba::new(data[i], data[i + 1], data[i + 2], data[i + 3])
                });
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
        mask: &Coverage,
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
        self.composite_cov(buf, stride, mask, group_opacity, rect, move |x, y| {
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
        });
    }

    /// Fill `path` with a [`GradientPaint`] — a gradient carrying its
    /// own `gradientUnits` / `gradientTransform` (SVG 2 §14.2) —
    /// returning a full-canvas RGBA frame with only that fill painted.
    ///
    /// `transform` is the user → device transform in effect for the
    /// element referencing the gradient (its CTM). The gradient is
    /// resolved against it and, under `ObjectBoundingBox` units, the
    /// path's §8.11 geometry bounding box ([`path_bounding_box`]); see
    /// [`GradientPaint::sampler`] for when nothing is painted.
    ///
    /// For gradients already expressed in user space, a
    /// [`Paint::LinearGradient`] / [`Paint::RadialGradient`] inside a
    /// scene rendered by [`Renderer::render`] goes through the very same
    /// inverse-CTM sampling.
    pub fn fill_path_with_gradient(
        &self,
        path: &Path,
        fill_rule: FillRule,
        gradient: &GradientPaint,
        transform: Transform2D,
    ) -> VideoFrame {
        let stride = (self.width as usize) * 4;
        let mut buf = vec![0u8; stride * (self.height as usize)];
        if let Some(sampler) = gradient.sampler(
            &transform,
            path_bounding_box(path),
            self.color_interpolation,
        ) {
            let contours = self.flatten_fill(&path.commands, &transform);
            let (mask, bounds) = self.rasterize_with_bounds(&contours, fill_rule);
            self.composite_with_gradient(&mut buf, stride, &mask, bounds, &sampler, 1.0);
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data: buf }],
        }
    }

    /// Stroke `path` with a [`GradientPaint`], returning a full-canvas
    /// RGBA frame with only that stroke painted. The object bounding
    /// box used for `ObjectBoundingBox` units is the path's fill
    /// geometry (SVG 2 §8.11 — the stroke width does not contribute).
    pub fn stroke_path_with_gradient(
        &self,
        path: &Path,
        stroke: &Stroke,
        gradient: &GradientPaint,
        transform: Transform2D,
    ) -> VideoFrame {
        let stride = (self.width as usize) * 4;
        let mut buf = vec![0u8; stride * (self.height as usize)];
        if let Some(sampler) = gradient.sampler(
            &transform,
            path_bounding_box(path),
            self.color_interpolation,
        ) {
            let geom = build_stroke_geometry(path, &transform, stroke, (self.width, self.height));
            let (mask, bounds) = self.rasterize_with_bounds(&geom, FillRule::NonZero);
            self.composite_with_gradient(&mut buf, stride, &mask, bounds, &sampler, 1.0);
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data: buf }],
        }
    }

    /// Composite a resolved gradient through `mask`, sampling at device
    /// pixel centres.
    fn composite_with_gradient(
        &self,
        buf: &mut [u8],
        stride: usize,
        mask: &Coverage,
        bounds: Option<(u32, u32, u32, u32)>,
        sampler: &GradientSampler,
        group_opacity: f32,
    ) {
        if mask.mask.is_empty() {
            return;
        }
        let Some(rect) = bounds else {
            return;
        };
        composite_coverage_rect(
            buf,
            stride,
            self.width,
            self.height,
            mask,
            group_opacity,
            self.blend_mode,
            matches!(self.compositing_space, InterpolationSpace::LinearRgb),
            rect,
            |x, y| sampler.sample(x as f32 + 0.5, y as f32 + 0.5),
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
        let contours = self.flatten_fill(&path.commands, &transform);
        let (mask, bounds) = self.rasterize_with_bounds(&contours, fill_rule);
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
        let geom = build_stroke_geometry(path, &transform, stroke, (self.width, self.height));
        let (mask, bounds) = self.rasterize_with_bounds(&geom, FillRule::NonZero);
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
        mask: &Coverage,
        bounds: Option<(u32, u32, u32, u32)>,
        pattern: &Pattern,
        transform: Transform2D,
        group_opacity: f32,
    ) {
        if mask.mask.is_empty() || pattern.is_degenerate() {
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
                self.composite_cov(buf, stride, mask, group_opacity, rect, move |x, y| {
                    let px = cols[(x - rx0) as usize];
                    let py = rows[(y - ry0) as usize];
                    let i = py * tile_stride + px * 4;
                    Rgba::new(data[i], data[i + 1], data[i + 2], data[i + 3])
                });
            } else {
                let cols: Vec<(usize, usize, f32)> = (rx0..=rx1)
                    .map(|x| crate::pattern::tile_bilinear_axis(col_u(x), tw_px))
                    .collect();
                let rows: Vec<(usize, usize, f32)> = (ry0..=ry1)
                    .map(|y| crate::pattern::tile_bilinear_axis(row_v(y), th_px))
                    .collect();
                composite_coverage_rect(
                    buf,
                    stride,
                    self.width,
                    self.height,
                    mask,
                    group_opacity,
                    blend,
                    matches!(self.compositing_space, InterpolationSpace::LinearRgb),
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
        composite_coverage_rect(
            buf,
            stride,
            self.width,
            self.height,
            mask,
            group_opacity,
            blend,
            matches!(self.compositing_space, InterpolationSpace::LinearRgb),
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
        let mut tile_renderer = Renderer::new(tw, th).with_group_cache(self.cache.clone());
        tile_renderer.supersampling = self.supersampling;
        tile_renderer.coverage = self.coverage;
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
/// closed device-space contours ready for a NonZero fill.
///
/// # Stroking space
///
/// SVG 2 §13.5 defines the stroke in the element's **user space**
/// (width, dashes, joins and caps are all user-space geometry), and
/// the result is then mapped by the CTM. Stroking the device-space
/// polyline instead is only right for a similarity transform, and even
/// then needs the right scale factor (the mean of `|a|` and `|d|`
/// vanishes under a 90° rotation).
///
/// So the outline is built in a *pre-space*: user space scaled
/// uniformly by `s`, the CTM's largest singular value, and then mapped
/// to device space by `F = CTM ∘ scale(1/s)`. `F` never stretches a
/// length (its singular values are ≤ 1), so every pixel-unit tolerance
/// the flattener and the round-join / cap tessellation use in
/// pre-space bounds the device error too — while a non-uniform CTM
/// still turns the round pen into the correct ellipse, and dash
/// lengths stay in user units.
fn build_stroke_geometry(
    path: &Path,
    transform: &Transform2D,
    stroke: &Stroke,
    canvas: (u32, u32),
) -> Vec<FlatContour> {
    let (a, b, c, d) = (
        transform.a as f64,
        transform.b as f64,
        transform.c as f64,
        transform.d as f64,
    );
    let det = a * d - b * c;
    let e2 = a * a + b * b + c * c + d * d;
    let s_max = ((e2 + (e2 * e2 - 4.0 * det * det).max(0.0).sqrt()) * 0.5).sqrt();
    if !(s_max.is_finite() && s_max > 1e-12) || det.abs() <= 1e-18 * e2.max(1e-300) {
        // A singular CTM collapses the stroke to zero area.
        return Vec::new();
    }
    // Pre-space → device: F = CTM · diag(1/s).
    let (fa, fb, fc, fd) = (a / s_max, b / s_max, c / s_max, d / s_max);
    let (fe, ff) = (transform.e as f64, transform.f as f64);
    let width_pre = stroke.width as f64 * s_max;
    if !(width_pre > 0.0 && width_pre.is_finite()) {
        return Vec::new();
    }
    let half = width_pre * 0.5;
    let mut st = stroke.clone();
    if let Some(dash) = &mut st.dash {
        for v in &mut dash.array {
            *v = (*v as f64 * s_max) as f32;
        }
        dash.offset = (dash.offset as f64 * s_max) as f32;
    }
    // Off-canvas culling (undashed strokes only: a dash pattern depends
    // on the arc length of everything before it). The device rectangle
    // grown by the stroke's reach — a miter apex lies at most
    // `miter_limit · half` from its vertex, a square cap `√2 · half` —
    // is pulled back to a pre-space bounding box.
    let cull = if st.dash.is_none() {
        let reach_dev = 2.0;
        let (w, h) = (canvas.0 as f64, canvas.1 as f64);
        let fdet = fa * fd - fb * fc;
        let inv = |x: f64, y: f64| -> (f64, f64) {
            let (x, y) = (x - fe, y - ff);
            ((fd * x - fc * y) / fdet, (-fb * x + fa * y) / fdet)
        };
        let corners = [
            inv(-reach_dev, -reach_dev),
            inv(w + reach_dev, -reach_dev),
            inv(-reach_dev, h + reach_dev),
            inv(w + reach_dev, h + reach_dev),
        ];
        let reach = half * (stroke.miter_limit as f64).max(1.5) + 1.0;
        let (mut x0, mut y0, mut x1, mut y1) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for (x, y) in corners {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        (x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()).then_some((
            x0 - reach,
            y0 - reach,
            x1 + reach,
            y1 + reach,
        ))
    } else {
        None
    };
    let opts = FlattenOptions {
        // Dashing walks the polyline's arc length, which a chord
        // shortens by ~δ²/24 per segment; a finer polyline keeps the
        // dash phase from drifting along long curved paths.
        tolerance: if st.dash.is_some() {
            FLATTEN_TOLERANCE_PX / 4.0
        } else {
            FLATTEN_TOLERANCE_PX
        },
        cull,
        stroke_half: half,
    };
    let pre = Transform2D::scale(s_max as f32, s_max as f32);
    let contours = flatten_path_with(&path.commands, &pre, &opts);
    if contours.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for c in &contours {
        // Dash periods are bounded by the *device* length (four per
        // device pixel on average): a pattern finer than that cannot be
        // seen, and pre-space lengths can be arbitrarily longer than
        // device ones under a strongly anisotropic CTM.
        let device_len: f64 = c
            .points
            .windows(2)
            .map(|w| {
                let (dx, dy) = ((w[1].0 - w[0].0) as f64, (w[1].1 - w[0].1) as f64);
                (fa * dx + fc * dy).hypot(fb * dx + fd * dy)
            })
            .sum();
        let max_periods = (4.0 * device_len + 16.0).min(f32::MAX as f64) as f32;
        out.extend(stroke_to_fill_path_bounded(
            c,
            &st,
            width_pre as f32,
            max_periods,
        ));
    }
    for c in &mut out {
        for p in &mut c.points {
            let (x, y) = (p.0 as f64, p.1 as f64);
            *p = ((fa * x + fc * y + fe) as f32, (fb * x + fd * y + ff) as f32);
        }
    }
    out
}

/// Average `k × k` blocks of the straight-alpha RGBA image `hi` into
/// `out` (`width × height`), weighting colours by alpha (a
/// premultiplied box filter), rounding to nearest.
fn downsample_premultiplied(
    hi: &[u8],
    hi_stride: usize,
    k: u32,
    out: &mut [u8],
    stride: usize,
    width: u32,
    height: u32,
) {
    let n = k * k;
    for y in 0..height {
        for x in 0..width {
            let (mut sr, mut sg, mut sb, mut sa) = (0u32, 0u32, 0u32, 0u32);
            for dy in 0..k {
                let row = (y * k + dy) as usize * hi_stride;
                for dx in 0..k {
                    let i = row + (x * k + dx) as usize * 4;
                    let a = hi[i + 3] as u32;
                    sr += hi[i] as u32 * a;
                    sg += hi[i + 1] as u32 * a;
                    sb += hi[i + 2] as u32 * a;
                    sa += a;
                }
            }
            let o = y as usize * stride + x as usize * 4;
            if sa == 0 {
                out[o..o + 4].fill(0);
                continue;
            }
            out[o] = ((sr + sa / 2) / sa) as u8;
            out[o + 1] = ((sg + sa / 2) / sa) as u8;
            out[o + 2] = ((sb + sa / 2) / sa) as u8;
            out[o + 3] = ((sa + n / 2) / n) as u8;
        }
    }
}

/// The origin of an offscreen window whose content starts at canvas
/// coordinate `v ≥ 0`: `v` rounded down to a multiple of 8, so that the
/// 8 × 8 ordered-dither lattice of gradient paints has the same phase
/// inside the window as on the canvas.
fn window_origin(v: i64) -> i64 {
    v & !7
}

/// `t` followed by a translation of `(-ox, -oy)`: the transform that
/// draws into a window whose top-left pixel is device pixel `(ox, oy)`.
fn shift_transform(t: &Transform2D, ox: i64, oy: i64) -> Transform2D {
    Transform2D {
        e: (t.e as f64 - ox as f64) as f32,
        f: (t.f as f64 - oy as f64) as f32,
        ..*t
    }
}

/// Blit a packed straight-alpha RGBA source buffer over an RGBA
/// destination using premultiplied source-over.
///
/// `(offset_x, offset_y)` places the source bitmap's top-left at that
/// destination pixel; it may be negative or beyond the destination, in
/// which case the blit is clipped to the destination. `mask` (when
/// `Some((mask, mx, my))`) further multiplies the source alpha
/// per-pixel: the mask's pixel `(0, 0)` sits at destination pixel
/// `(mx, my)` and it reads as 0 outside its extent — `(0, 0)` for a
/// canvas-sized clip mask, the blit offset for a mask in source
/// coordinates. `extra_alpha` (in `0.0..=1.0`) modulates the source
/// alpha globally (group opacity). `src_rect`, when given, is an
/// inclusive source-coordinate rectangle `(x0, y0, x1, y1)` outside
/// which every source pixel is known to have zero alpha. The blit
/// touches only pixels where the source alpha (after mask + extra) is
/// non-zero.
#[allow(clippy::too_many_arguments)]
fn blit_rgba_at(
    dst: &mut [u8],
    dst_stride: usize,
    dst_width: u32,
    dst_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
    offset_x: i64,
    offset_y: i64,
    mask: Option<(&AlphaMask, i64, i64)>,
    extra_alpha: f32,
    src_rect: Option<(u32, u32, u32, u32)>,
) {
    if src_width == 0 || src_height == 0 {
        return;
    }
    let extra = extra_alpha.clamp(0.0, 1.0);
    let extra_q = (extra * 255.0).round() as u32;
    if extra_q == 0 {
        return;
    }
    let src_stride = (src_width as usize) * 4;
    let (rx0, ry0, rx1, ry1) = src_rect.unwrap_or((0, 0, src_width - 1, src_height - 1));
    // Source range that lands on the destination (half-open).
    let x_lo = (rx0 as i64).max(-offset_x);
    let x_hi = (rx1 as i64 + 1)
        .min(src_width as i64)
        .min(dst_width as i64 - offset_x);
    let y_lo = (ry0 as i64).max(-offset_y);
    let y_hi = (ry1 as i64 + 1)
        .min(src_height as i64)
        .min(dst_height as i64 - offset_y);
    if x_lo >= x_hi || y_lo >= y_hi {
        return;
    }
    let (x_lo, x_hi, y_lo, y_hi) = (x_lo as usize, x_hi as usize, y_lo as usize, y_hi as usize);
    // Destination column of source column `x_lo` (≥ 0 by construction).
    let dx0 = (offset_x + x_lo as i64) as usize;
    for y in y_lo..y_hi {
        let dy = (offset_y + y as i64) as usize; // in 0..dst_height
        let src_row = &src[y * src_stride + x_lo * 4..y * src_stride + x_hi * 4];
        let dst_row = &mut dst[dy * dst_stride + dx0 * 4..][..src_row.len()];
        let Some((m, mx, my)) = mask else {
            for (s, d) in src_row
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst_row.as_chunks_mut::<4>().0)
            {
                over_pixel(s, d, 255, extra_q);
            }
            continue;
        };
        // The mask row facing this destination row, cut to the columns
        // of the blit it covers (it reads as 0 everywhere else, where
        // nothing is composited).
        let my = dy as i64 - my;
        if my < 0 || my >= m.height as i64 {
            continue;
        }
        let c0 = dx0 as i64 - mx; // mask column under blit column 0
        let n = (x_hi - x_lo) as i64;
        let k_lo = (-c0).clamp(0, n);
        let k_hi = (m.width as i64 - c0).clamp(k_lo, n);
        if k_lo >= k_hi {
            continue; // the mask's columns miss the blit entirely
        }
        let (k_lo, k_hi) = (k_lo as usize, k_hi as usize);
        let m0 = my as usize * m.width as usize + (c0 + k_lo as i64) as usize;
        let cov_row = &m.data[m0..m0 + (k_hi - k_lo)];
        for ((s, d), &cov) in src_row[k_lo * 4..k_hi * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .zip(dst_row[k_lo * 4..k_hi * 4].as_chunks_mut::<4>().0)
            .zip(cov_row)
        {
            over_pixel(s, d, cov as u32, extra_q);
        }
    }
}

/// Composite one straight-alpha source pixel over one straight-alpha
/// destination pixel with the source alpha scaled by `cov / 255` and
/// `extra_q / 255` (premultiplied source-over, rounded to nearest at
/// each step).
#[inline(always)]
fn over_pixel(src: &[u8; 4], dst: &mut [u8; 4], cov: u32, extra_q: u32) {
    let sa_raw = src[3] as u32;
    if sa_raw == 0 || cov == 0 {
        return;
    }
    // src_alpha = sa_raw * cov * extra / (255 * 255)
    let sa = (sa_raw * cov * extra_q + (255 * 255 / 2)) / (255 * 255);
    if sa == 0 {
        return;
    }
    if sa == 255 {
        // Fully-opaque source pixel: the over operator reduces to a
        // plain overwrite (with `sa = 255` the premultiply is exact,
        // `inv = 0` zeroes every destination term, and the final
        // un-premultiply divides by 255 giving the source bytes back
        // verbatim).
        *dst = [src[0], src[1], src[2], 255];
        return;
    }
    // Promote source RGB into premultiplied space (src is
    // straight-alpha at sa_raw — re-premultiply with the *combined*
    // alpha so the over operator behaves).
    let sr = (src[0] as u32 * sa + 127) / 255;
    let sg = (src[1] as u32 * sa + 127) / 255;
    let sb = (src[2] as u32 * sa + 127) / 255;

    let da0 = dst[3] as u32;
    let dr = (dst[0] as u32 * da0 + 127) / 255;
    let dg = (dst[1] as u32 * da0 + 127) / 255;
    let db = (dst[2] as u32 * da0 + 127) / 255;

    let inv = 255 - sa;
    let or = sr + (dr * inv + 127) / 255;
    let og = sg + (dg * inv + 127) / 255;
    let ob = sb + (db * inv + 127) / 255;
    // `sa ≥ 1`, so the composite alpha is never zero.
    let oa = sa + (da0 * inv + 127) / 255;
    *dst = [
        ((or * 255 + oa / 2) / oa).min(255) as u8,
        ((og * 255 + oa / 2) / oa).min(255) as u8,
        ((ob * 255 + oa / 2) / oa).min(255) as u8,
        oa.min(255) as u8,
    ];
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
        for px in out.planes[0].data.as_chunks::<4>().0 {
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
    fn blit_clips_negative_offsets_and_windowed_masks() {
        // 4×4 opaque source at (-2, -1) on a 6×6 destination: only its
        // lower-right 2×3 lands, at the top-left corner.
        let src = [9u8, 8, 7, 255].repeat(16);
        let mut dst = vec![0u8; 6 * 6 * 4];
        blit_rgba_at(&mut dst, 24, 6, 6, &src, 4, 4, -2, -1, None, 1.0, None);
        let painted: Vec<usize> = (0..36).filter(|i| dst[i * 4 + 3] != 0).collect();
        assert_eq!(painted, vec![0, 1, 6, 7, 12, 13]);
        // A 2×2 mask window at (3, 3) lets through only the pixels it
        // covers; a window that misses the blit lets through nothing.
        for (mx, my, want) in [
            (3i64, 3i64, vec![21usize, 22, 27, 28]),
            (0, 0, vec![]),
            (5, 2, vec![17, 23]),
            (-7, 3, vec![]),
            (40, 40, vec![]),
        ] {
            let mask = AlphaMask {
                width: 2,
                height: 2,
                data: vec![255; 4],
            };
            let mut dst = vec![0u8; 6 * 6 * 4];
            blit_rgba_at(
                &mut dst,
                24,
                6,
                6,
                &src,
                4,
                4,
                2,
                2,
                Some((&mask, mx, my)),
                1.0,
                None,
            );
            let painted: Vec<usize> = (0..36).filter(|i| dst[i * 4 + 3] != 0).collect();
            assert_eq!(painted, want, "mask at ({mx}, {my})");
        }
        // Entirely off the destination: nothing happens.
        let mut dst = vec![0u8; 6 * 6 * 4];
        for (ox, oy) in [(-4, 0), (6, 0), (0, -4), (0, 6), (i64::MIN / 4, 0)] {
            blit_rgba_at(&mut dst, 24, 6, 6, &src, 4, 4, ox, oy, None, 1.0, None);
        }
        assert!(dst.iter().all(|&b| b == 0));
    }

    #[test]
    fn blit_with_offset_paints_at_destination_position() {
        // 8×8 destination, 2×2 source of opaque red, blit at (3, 4).
        let mut dst = vec![0u8; 8 * 8 * 4];
        let src: Vec<u8> = (0..2 * 2).flat_map(|_| [255, 0, 0, 255]).collect();
        blit_rgba_at(&mut dst, 32, 8, 8, &src, 2, 2, 3, 4, None, 1.0, None);
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
        blit_rgba_at(&mut dst, 16, 4, 4, &src, 4, 4, 2, 2, None, 1.0, None);
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
