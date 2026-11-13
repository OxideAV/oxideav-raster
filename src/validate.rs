//! Scene validation: a typed rejection of non-finite geometry at the
//! API boundary.
//!
//! [`Renderer::render`](crate::Renderer::render) is total — it never
//! fails, and a NaN / ±∞ coordinate simply drops the segment it belongs
//! to before the scan converter sees it. Callers that would rather hear
//! about a malformed scene than get a partial picture use
//! [`Renderer::try_render`](crate::Renderer::try_render) (or
//! [`validate_frame`] directly), which walks the scene first and
//! reports the first non-finite value it finds.

use oxideav_core::{
    Group, Node, Paint, Path, PathCommand, Point, Stroke, Transform2D, VectorFrame,
};

/// Why a scene was rejected by [`validate_frame`] /
/// [`Renderer::try_render`](crate::Renderer::try_render).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RenderError {
    /// A coordinate, length, matrix entry or opacity is NaN or ±∞.
    NonFiniteGeometry(GeometryPart),
}

/// Which part of the scene carried the non-finite value reported by
/// [`RenderError::NonFiniteGeometry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeometryPart {
    /// The frame's `viewBox`.
    ViewBox,
    /// A group or image transform.
    Transform,
    /// A group opacity.
    Opacity,
    /// A path command of a painted path.
    PathPoint,
    /// A group's clip path.
    ClipPath,
    /// A stroke width, miter limit or dash length.
    Stroke,
    /// A gradient's geometry or stop offsets.
    Paint,
    /// An image's destination rectangle.
    ImageBounds,
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::NonFiniteGeometry(part) => {
                write!(f, "non-finite value in the scene ({part:?})")
            }
        }
    }
}

impl std::error::Error for RenderError {}

fn bad(part: GeometryPart) -> Result<(), RenderError> {
    Err(RenderError::NonFiniteGeometry(part))
}

fn point_ok(p: &Point) -> bool {
    p.x.is_finite() && p.y.is_finite()
}

fn transform_ok(t: &Transform2D) -> bool {
    [t.a, t.b, t.c, t.d, t.e, t.f].iter().all(|v| v.is_finite())
}

fn path_ok(path: &Path) -> bool {
    path.commands.iter().all(|c| match c {
        PathCommand::MoveTo(p) | PathCommand::LineTo(p) => point_ok(p),
        PathCommand::QuadCurveTo { control, end } => point_ok(control) && point_ok(end),
        PathCommand::CubicCurveTo { c1, c2, end } => point_ok(c1) && point_ok(c2) && point_ok(end),
        PathCommand::ArcTo {
            rx,
            ry,
            x_axis_rot,
            end,
            ..
        } => rx.is_finite() && ry.is_finite() && x_axis_rot.is_finite() && point_ok(end),
        _ => true,
    })
}

fn paint_ok(paint: &Paint) -> bool {
    match paint {
        Paint::LinearGradient(g) => {
            point_ok(&g.start) && point_ok(&g.end) && g.stops.iter().all(|s| s.offset.is_finite())
        }
        Paint::RadialGradient(g) => {
            point_ok(&g.center)
                && g.radius.is_finite()
                && g.focal.as_ref().is_none_or(point_ok)
                && g.stops.iter().all(|s| s.offset.is_finite())
        }
        _ => true,
    }
}

fn stroke_ok(s: &Stroke) -> bool {
    s.width.is_finite()
        && s.miter_limit.is_finite()
        && s.dash
            .as_ref()
            .is_none_or(|d| d.offset.is_finite() && d.array.iter().all(|v| v.is_finite()))
}

fn group(g: &Group) -> Result<(), RenderError> {
    if !transform_ok(&g.transform) {
        return bad(GeometryPart::Transform);
    }
    if !g.opacity.is_finite() {
        return bad(GeometryPart::Opacity);
    }
    if g.clip.as_ref().is_some_and(|c| !path_ok(c)) {
        return bad(GeometryPart::ClipPath);
    }
    g.children.iter().try_for_each(node)
}

fn node(n: &Node) -> Result<(), RenderError> {
    match n {
        Node::Path(p) => {
            if !path_ok(&p.path) {
                return bad(GeometryPart::PathPoint);
            }
            if p.fill.as_ref().is_some_and(|f| !paint_ok(f)) {
                return bad(GeometryPart::Paint);
            }
            if let Some(s) = &p.stroke {
                if !stroke_ok(s) {
                    return bad(GeometryPart::Stroke);
                }
                if !paint_ok(&s.paint) {
                    return bad(GeometryPart::Paint);
                }
            }
            Ok(())
        }
        Node::Group(g) => group(g),
        Node::Image(img) => {
            if !transform_ok(&img.transform) {
                return bad(GeometryPart::Transform);
            }
            let b = &img.bounds;
            if ![b.x, b.y, b.width, b.height].iter().all(|v| v.is_finite()) {
                return bad(GeometryPart::ImageBounds);
            }
            Ok(())
        }
        Node::SoftMask { mask, content, .. } => {
            node(mask)?;
            node(content)
        }
        _ => Ok(()),
    }
}

/// Check that every coordinate, length, matrix entry and opacity in
/// `frame` is finite. `Ok(())` means the whole scene reaches the
/// renderer as written; an error names the first offending part in
/// paint order.
pub fn validate_frame(frame: &VectorFrame) -> Result<(), RenderError> {
    if frame.view_box.as_ref().is_some_and(|vb| {
        ![vb.min_x, vb.min_y, vb.width, vb.height]
            .iter()
            .all(|v| v.is_finite())
    }) {
        return bad(GeometryPart::ViewBox);
    }
    group(&frame.root)
}
