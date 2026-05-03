//! Top-level renderer: walks a [`VectorFrame`]'s scene graph and
//! produces a packed `Rgba` [`VideoFrame`].
//!
//! See the crate-level docs for the overall pipeline.

use oxideav_core::{
    FillRule, Frame, Group, ImageRef, Node, Paint, Path, PathNode, Rect, Rgba, Stroke, Transform2D,
    VectorFrame, VideoFrame, VideoPlane,
};

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
}

impl Renderer {
    /// Build a renderer for the given destination size with sane
    /// defaults (`supersampling = 4`, transparent background, no
    /// sub-pixel positioning).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            supersampling: 4,
            subpixel_positioning: false,
            background: Rgba::new(0, 0, 0, 0),
        }
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
        self.draw_group(&frame.root, initial, &mut buf, stride, None);
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
        buf: &mut [u8],
        stride: usize,
        clip_mask: Option<&AlphaMask>,
    ) {
        let local = parent_transform.compose(&g.transform);
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
        for child in &g.children {
            self.draw_node(child, local, g.opacity, buf, stride, effective_clip);
        }
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
                // Compose group opacity with our own. (Each call's
                // `group_opacity` is the *parent* group's opacity;
                // inside this group, children get the multiplied
                // value.)
                let combined = group_opacity * g.opacity;
                let mut child = g.clone();
                child.opacity = combined;
                self.draw_group(&child, transform, buf, stride, clip_mask);
            }
            Node::Image(img) => {
                self.draw_image(img, transform, group_opacity, buf, stride, clip_mask)
            }
            // `Node` is #[non_exhaustive]; future variants
            // (text, masks, filters) silently no-op until handled.
            _ => {}
        }
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
        // paint source. The actual texture sampling implementation
        // is nearest-neighbour; bilinear is round 2.
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
                sample_image_nearest(&frame, &bounds, user.x, user.y)
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
    if bounds.width <= 0.0 || bounds.height <= 0.0 {
        return Rgba::new(0, 0, 0, 0);
    }
    let u = (ux - bounds.x) / bounds.width;
    let v = (uy - bounds.y) / bounds.height;
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return Rgba::new(0, 0, 0, 0);
    }
    let plane = match frame.planes.first() {
        Some(p) => p,
        None => return Rgba::new(0, 0, 0, 0),
    };
    let stride = plane.stride;
    if stride < 4 {
        return Rgba::new(0, 0, 0, 0);
    }
    // Infer width / height from the buffer length and stride. The
    // frame doesn't carry width/height directly; compute height from
    // the plane data and assume the caller-supplied stride is
    // `width * 4` for packed Rgba.
    let width = (stride / 4) as u32;
    let height = (plane.data.len() / stride) as u32;
    if width == 0 || height == 0 {
        return Rgba::new(0, 0, 0, 0);
    }
    let px = ((u * width as f32).floor() as i64).clamp(0, width as i64 - 1) as usize;
    let py = ((v * height as f32).floor() as i64).clamp(0, height as i64 - 1) as usize;
    let i = py * stride + px * 4;
    Rgba::new(
        plane.data[i],
        plane.data[i + 1],
        plane.data[i + 2],
        plane.data[i + 3],
    )
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
