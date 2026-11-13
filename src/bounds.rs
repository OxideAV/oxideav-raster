//! Conservative device-space bounds of a scene subtree, computed from
//! the geometry alone.
//!
//! Offscreen rendering (group opacity, soft masks, cached groups) needs
//! to know how large a buffer a subtree can touch *before* rasterising
//! it. The bounds here are derived analytically:
//!
//! * a filled path lies inside the bounding box of its transformed
//!   control points (a Bézier segment lies in the convex hull of its
//!   control polygon; an elliptic arc is first replaced by the cubics
//!   the flattener itself uses);
//! * a stroke reaches at most `half_width · max(miter_limit, √2)` user
//!   units from the path (a miter apex lies `miter_limit · half_width`
//!   from its vertex at most, a square cap corner `√2 · half_width`),
//!   scaled by the transform's largest singular value;
//! * an image covers the transformed corners of its rectangle;
//! * a group is the union of its children intersected with its clip
//!   path's bounds; a soft mask is its content intersected with its
//!   mask.
//!
//! One pixel of margin on every side absorbs anti-aliasing and the
//! flattening tolerance; a group with a cache key adds half a pixel for
//! the sub-pixel snapping of its position. Anything non-finite makes the affected bounds
//! [`Bounds::EVERYTHING`], which callers treat as "the whole canvas".

use oxideav_core::{Group, Node, Path, PathCommand, PathNode, Point, Stroke, Transform2D};

use crate::flatten::flatten_arc_to_cubics;

/// Margin, in device pixels, added around the analytic geometry bounds.
const PIXEL_MARGIN: f64 = 1.0;

/// Largest displacement, per axis, of a keyed group from its nominal
/// position (the coarsest sub-pixel grid snaps to whole pixels).
const KEYED_GROUP_SNAP: f64 = 0.5;

/// Pixel boxes are only reported for bounds within this many pixels of
/// the origin (far beyond any canvas, and exactly representable).
const PIXEL_BOX_LIMIT: f64 = (1u64 << 30) as f64;

/// An axis-aligned device-space rectangle `[x0, x1] × [y0, y1]`; empty
/// when `x0 > x1` or `y0 > y1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bounds {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Bounds {
    /// Contains no point.
    pub const EMPTY: Bounds = Bounds {
        x0: f64::INFINITY,
        y0: f64::INFINITY,
        x1: f64::NEG_INFINITY,
        y1: f64::NEG_INFINITY,
    };

    /// Contains every point (unknown / unbounded geometry).
    pub const EVERYTHING: Bounds = Bounds {
        x0: f64::NEG_INFINITY,
        y0: f64::NEG_INFINITY,
        x1: f64::INFINITY,
        y1: f64::INFINITY,
    };

    pub fn is_empty(&self) -> bool {
        // Written so a NaN field (never produced, but cheap to defend
        // against) reads as empty rather than as a valid rectangle.
        !(self.x0 <= self.x1 && self.y0 <= self.y1)
    }

    pub fn union(&self, o: &Bounds) -> Bounds {
        if self.is_empty() {
            return *o;
        }
        if o.is_empty() {
            return *self;
        }
        Bounds {
            x0: self.x0.min(o.x0),
            y0: self.y0.min(o.y0),
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
        }
    }

    pub fn intersect(&self, o: &Bounds) -> Bounds {
        if self.is_empty() || o.is_empty() {
            return Bounds::EMPTY;
        }
        let b = Bounds {
            x0: self.x0.max(o.x0),
            y0: self.y0.max(o.y0),
            x1: self.x1.min(o.x1),
            y1: self.y1.min(o.y1),
        };
        if b.is_empty() { Bounds::EMPTY } else { b }
    }

    /// Grow to include `(x, y)`; a non-finite coordinate makes the
    /// bounds [`Bounds::EVERYTHING`].
    fn add(&mut self, x: f64, y: f64) {
        if !(x.is_finite() && y.is_finite()) {
            *self = Bounds::EVERYTHING;
            return;
        }
        self.x0 = self.x0.min(x);
        self.y0 = self.y0.min(y);
        self.x1 = self.x1.max(x);
        self.y1 = self.y1.max(y);
    }

    fn grown(&self, m: f64) -> Bounds {
        if self.is_empty() {
            return *self;
        }
        if !m.is_finite() {
            return Bounds::EVERYTHING;
        }
        Bounds {
            x0: self.x0 - m,
            y0: self.y0 - m,
            x1: self.x1 + m,
            y1: self.y1 + m,
        }
    }

    /// The half-open pixel rectangle `[x0, x1) × [y0, y1)` covering the
    /// bounds plus the anti-aliasing margin. `None` when the bounds are
    /// empty, unbounded or absurdly far from the origin — use
    /// [`Bounds::is_empty`] to tell the first case apart.
    pub fn pixel_box(&self) -> Option<PixelBox> {
        if self.is_empty() {
            return None;
        }
        let b = self.grown(PIXEL_MARGIN);
        let ok = |v: f64| v.is_finite() && v.abs() < PIXEL_BOX_LIMIT;
        if !(ok(b.x0) && ok(b.y0) && ok(b.x1) && ok(b.y1)) {
            return None;
        }
        Some(PixelBox {
            x0: b.x0.floor() as i64,
            y0: b.y0.floor() as i64,
            x1: b.x1.ceil() as i64,
            y1: b.y1.ceil() as i64,
        })
    }

    /// [`Bounds::pixel_box`] clipped to a `width × height` canvas at the
    /// origin; unbounded bounds become the whole canvas. `None` when
    /// nothing of the bounds is on the canvas.
    pub fn pixel_box_on_canvas(&self, width: u32, height: u32) -> Option<PixelBox> {
        if self.is_empty() || width == 0 || height == 0 {
            return None;
        }
        let canvas = PixelBox {
            x0: 0,
            y0: 0,
            x1: width as i64,
            y1: height as i64,
        };
        // Clamp before the float → integer conversion so an unbounded
        // or far-away edge lands exactly on the canvas edge.
        let b = self.grown(PIXEL_MARGIN);
        let lo = |v: f64, max: i64| (v.max(0.0).min(max as f64)).floor() as i64;
        let hi = |v: f64, max: i64| (v.max(0.0).min(max as f64)).ceil() as i64;
        let pb = PixelBox {
            x0: lo(b.x0, canvas.x1),
            y0: lo(b.y0, canvas.y1),
            x1: hi(b.x1, canvas.x1),
            y1: hi(b.y1, canvas.y1),
        };
        (pb.x0 < pb.x1 && pb.y0 < pb.y1).then_some(pb)
    }
}

/// A half-open integer pixel rectangle `[x0, x1) × [y0, y1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PixelBox {
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
}

impl PixelBox {
    pub fn width(&self) -> u64 {
        (self.x1 - self.x0).max(0) as u64
    }

    pub fn height(&self) -> u64 {
        (self.y1 - self.y0).max(0) as u64
    }

    pub fn area(&self) -> u64 {
        self.width().saturating_mul(self.height())
    }

    pub fn translated(&self, dx: i64, dy: i64) -> PixelBox {
        PixelBox {
            x0: self.x0 + dx,
            y0: self.y0 + dy,
            x1: self.x1 + dx,
            y1: self.y1 + dy,
        }
    }

    /// Area of the part inside a `width × height` canvas at the origin.
    pub fn area_on_canvas(&self, width: u32, height: u32) -> u64 {
        let w = self.x1.min(width as i64) - self.x0.max(0);
        let h = self.y1.min(height as i64) - self.y0.max(0);
        if w <= 0 || h <= 0 {
            0
        } else {
            w as u64 * h as u64
        }
    }
}

/// Bounding box of the transformed control points of `path`.
fn path_points_bounds(path: &Path, t: &Transform2D) -> Bounds {
    let (ta, tb, tc, td, te, tf) = (
        t.a as f64, t.b as f64, t.c as f64, t.d as f64, t.e as f64, t.f as f64,
    );
    let mut b = Bounds::EMPTY;
    let add = |b: &mut Bounds, p: Point| {
        let (x, y) = (p.x as f64, p.y as f64);
        b.add(ta * x + tc * y + te, tb * x + td * y + tf);
    };
    let origin = Point::new(0.0, 0.0);
    let mut pen = origin;
    let mut start = origin;
    let mut open = false;
    // A drawing command with no current subpath starts one at the
    // origin, as the flattener does.
    let begin = |b: &mut Bounds, pen: &mut Point, start: &mut Point, open: &mut bool| {
        if !*open {
            *pen = origin;
            *start = origin;
            *open = true;
            add(b, origin);
        }
    };
    for cmd in &path.commands {
        match *cmd {
            PathCommand::MoveTo(p) => {
                add(&mut b, p);
                pen = p;
                start = p;
                open = true;
            }
            PathCommand::LineTo(p) => {
                begin(&mut b, &mut pen, &mut start, &mut open);
                add(&mut b, p);
                pen = p;
            }
            PathCommand::QuadCurveTo { control, end } => {
                begin(&mut b, &mut pen, &mut start, &mut open);
                add(&mut b, control);
                add(&mut b, end);
                pen = end;
            }
            PathCommand::CubicCurveTo { c1, c2, end } => {
                begin(&mut b, &mut pen, &mut start, &mut open);
                add(&mut b, c1);
                add(&mut b, c2);
                add(&mut b, end);
                pen = end;
            }
            PathCommand::ArcTo {
                rx,
                ry,
                x_axis_rot,
                large_arc,
                sweep,
                end,
            } => {
                begin(&mut b, &mut pen, &mut start, &mut open);
                for [c1, c2, e] in
                    flatten_arc_to_cubics(pen, rx, ry, x_axis_rot, large_arc, sweep, end)
                {
                    add(&mut b, c1);
                    add(&mut b, c2);
                    add(&mut b, e);
                }
                add(&mut b, end);
                pen = end;
            }
            PathCommand::Close if open => {
                pen = start;
                open = false;
            }
            // Commands the flattener does not draw contribute nothing.
            _ => {}
        }
    }
    b
}

/// Largest singular value of the linear part of `t` — the most a
/// user-space length can grow in device space.
fn max_stretch(t: &Transform2D) -> f64 {
    let (a, b, c, d) = (t.a as f64, t.b as f64, t.c as f64, t.d as f64);
    let det = a * d - b * c;
    let e2 = a * a + b * b + c * c + d * d;
    ((e2 + (e2 * e2 - 4.0 * det * det).max(0.0).sqrt()) * 0.5).sqrt()
}

/// Device-space distance a stroke can reach from its path.
fn stroke_reach(stroke: &Stroke, t: &Transform2D) -> Option<f64> {
    let w = stroke.width as f64;
    if !(w > 0.0 && w.is_finite()) {
        // Not painted (the stroker emits nothing for such a width).
        return None;
    }
    let limit = (stroke.miter_limit as f64).max(std::f64::consts::SQRT_2);
    // A NaN limit or an overflowing product becomes "unbounded".
    Some(0.5 * w * limit * max_stretch(t))
}

/// Bounds of a path node's fill and stroke under the device transform
/// `t`.
pub(crate) fn path_node_bounds(p: &PathNode, t: &Transform2D) -> Bounds {
    let reach = p.stroke.as_ref().and_then(|s| stroke_reach(s, t));
    if p.fill.is_none() && reach.is_none() {
        return Bounds::EMPTY;
    }
    let b = path_points_bounds(&p.path, t);
    match reach {
        Some(r) => b.grown(r),
        None => b,
    }
}

/// Bounds of everything `node` can paint under the device transform
/// `t`.
pub(crate) fn node_bounds(node: &Node, t: &Transform2D) -> Bounds {
    match node {
        Node::Path(p) => path_node_bounds(p, t),
        Node::Group(g) => {
            let b = group_content_bounds(g, &t.compose(&g.transform));
            // A keyed group is drawn at a translation snapped to the
            // cache's sub-pixel grid: up to half a pixel per axis away
            // (per level of keyed nesting, hence here and not in the
            // fixed margin).
            if g.cache_key.is_some() {
                b.grown(KEYED_GROUP_SNAP)
            } else {
                b
            }
        }
        Node::Image(img) => {
            let m = t.compose(&img.transform);
            let r = &img.bounds;
            let mut b = Bounds::EMPTY;
            for (x, y) in [
                (r.x, r.y),
                (r.x + r.width, r.y),
                (r.x, r.y + r.height),
                (r.x + r.width, r.y + r.height),
            ] {
                let (x, y) = (x as f64, y as f64);
                b.add(
                    m.a as f64 * x + m.c as f64 * y + m.e as f64,
                    m.b as f64 * x + m.d as f64 * y + m.f as f64,
                );
            }
            b
        }
        Node::SoftMask { mask, content, .. } => {
            node_bounds(content, t).intersect(&node_bounds(mask, t))
        }
        // Node kinds the renderer does not draw paint nothing.
        _ => Bounds::EMPTY,
    }
}

/// Bounds of a group's children under `local` (the device transform
/// *including* the group's own), intersected with its clip path.
pub(crate) fn group_content_bounds(g: &Group, local: &Transform2D) -> Bounds {
    let mut b = Bounds::EMPTY;
    for child in &g.children {
        b = b.union(&node_bounds(child, local));
    }
    match &g.clip {
        Some(clip) => b.intersect(&path_points_bounds(clip, local)),
        None => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{FillRule, Paint, Rgba};

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Path {
        let mut p = Path::new();
        p.move_to(Point::new(x, y))
            .line_to(Point::new(x + w, y))
            .line_to(Point::new(x + w, y + h))
            .line_to(Point::new(x, y + h))
            .close();
        p
    }

    fn filled(path: Path) -> Node {
        Node::Path(PathNode {
            path,
            fill: Some(Paint::Solid(Rgba::opaque(1, 2, 3))),
            stroke: None,
            fill_rule: FillRule::NonZero,
        })
    }

    #[test]
    fn filled_rect_bounds_follow_the_transform() {
        let n = filled(rect(2.0, 3.0, 4.0, 5.0));
        let b = node_bounds(&n, &Transform2D::identity());
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (2.0, 3.0, 6.0, 8.0));
        let t = Transform2D::translate(10.0, 20.0).compose(&Transform2D::scale(2.0, 3.0));
        let b = node_bounds(&n, &t);
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (14.0, 29.0, 22.0, 44.0));
        let pb = b.pixel_box().unwrap();
        assert_eq!((pb.x0, pb.y0, pb.x1, pb.y1), (13, 28, 23, 45));
    }

    #[test]
    fn stroke_grows_by_its_reach_and_unpainted_paths_are_empty() {
        let mut node = PathNode::new(rect(10.0, 10.0, 10.0, 10.0));
        assert!(node_bounds(&Node::Path(node.clone()), &Transform2D::identity()).is_empty());
        let mut s = Stroke::solid(4.0, Rgba::opaque(0, 0, 0));
        s.miter_limit = 4.0;
        node.stroke = Some(s);
        let b = node_bounds(&Node::Path(node.clone()), &Transform2D::scale(2.0, 2.0));
        // half width 2 · miter limit 4 · scale 2 = 16 px reach.
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (4.0, 4.0, 56.0, 56.0));
        // A zero-width stroke paints nothing.
        node.stroke = Some(Stroke::solid(0.0, Rgba::opaque(0, 0, 0)));
        assert!(node_bounds(&Node::Path(node), &Transform2D::identity()).is_empty());
    }

    #[test]
    fn curves_and_arcs_stay_inside_their_bounds() {
        let mut p = Path::new();
        p.move_to(Point::new(10.0, 40.0))
            .cubic_to(
                Point::new(10.0, -20.0),
                Point::new(70.0, 90.0),
                Point::new(70.0, 40.0),
            )
            .quad_to(Point::new(100.0, 0.0), Point::new(40.0, 10.0));
        p.commands.push(PathCommand::ArcTo {
            rx: 25.0,
            ry: 12.0,
            x_axis_rot: 0.6,
            large_arc: true,
            sweep: false,
            end: Point::new(20.0, 60.0),
        });
        let t = Transform2D::rotate(0.4).compose(&Transform2D::scale(1.5, 0.75));
        let b = node_bounds(&filled(p.clone()), &t);
        for c in crate::flatten::flatten_path(&p.commands, &t) {
            for (x, y) in c.points {
                assert!(x as f64 >= b.x0 - 1e-3 && x as f64 <= b.x1 + 1e-3);
                assert!(y as f64 >= b.y0 - 1e-3 && y as f64 <= b.y1 + 1e-3);
            }
        }
    }

    #[test]
    fn group_bounds_union_children_and_intersect_the_clip() {
        let g = Group {
            transform: Transform2D::translate(100.0, 0.0),
            clip: Some(rect(0.0, 0.0, 12.0, 50.0)),
            children: vec![
                filled(rect(0.0, 0.0, 10.0, 10.0)),
                filled(rect(5.0, 20.0, 30.0, 10.0)),
            ],
            ..Group::default()
        };
        let b = node_bounds(&Node::Group(g.clone()), &Transform2D::identity());
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (100.0, 0.0, 112.0, 30.0));
        let mask = Node::SoftMask {
            mask: Box::new(filled(rect(105.0, 5.0, 100.0, 100.0))),
            mask_kind: oxideav_core::MaskKind::Alpha,
            content: Box::new(Node::Group(g)),
        };
        let b = node_bounds(&mask, &Transform2D::identity());
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (105.0, 5.0, 112.0, 30.0));
        assert!(group_content_bounds(&Group::default(), &Transform2D::identity()).is_empty());
    }

    #[test]
    fn keyed_groups_add_their_snapping_range_per_nesting_level() {
        let inner = Group {
            cache_key: Some(1),
            children: vec![filled(rect(10.0, 10.0, 4.0, 4.0))],
            ..Group::default()
        };
        let outer = Group {
            cache_key: Some(2),
            children: vec![Node::Group(inner.clone())],
            ..Group::default()
        };
        let id = Transform2D::identity();
        let b = node_bounds(&Node::Group(inner), &id);
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (9.5, 9.5, 14.5, 14.5));
        let b = node_bounds(&Node::Group(outer), &id);
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (9.0, 9.0, 15.0, 15.0));
    }

    #[test]
    fn non_finite_geometry_is_unbounded_not_wrong() {
        let n = filled(rect(f32::NAN, 0.0, 4.0, 4.0));
        let b = node_bounds(&n, &Transform2D::identity());
        assert_eq!(b, Bounds::EVERYTHING);
        assert!(b.pixel_box().is_none());
        let pb = b.pixel_box_on_canvas(64, 32).unwrap();
        assert_eq!((pb.x0, pb.y0, pb.x1, pb.y1), (0, 0, 64, 32));
        let far = filled(rect(3.0e30, 0.0, 4.0, 4.0));
        let b = node_bounds(&far, &Transform2D::identity());
        assert!(b.pixel_box().is_none());
        assert!(b.pixel_box_on_canvas(64, 32).is_none());
    }

    #[test]
    fn pixel_box_on_canvas_clips() {
        let b = Bounds {
            x0: -5.5,
            y0: 3.2,
            x1: 7.9,
            y1: 100.0,
        };
        let pb = b.pixel_box_on_canvas(20, 10).unwrap();
        assert_eq!((pb.x0, pb.y0, pb.x1, pb.y1), (0, 2, 9, 10));
        assert_eq!(pb.area(), 72);
        let off = Bounds {
            x0: 30.0,
            y0: 0.0,
            x1: 40.0,
            y1: 5.0,
        };
        assert!(off.pixel_box_on_canvas(20, 10).is_none());
        let full = b.pixel_box().unwrap();
        assert_eq!((full.x0, full.y0, full.x1, full.y1), (-7, 2, 9, 101));
        assert_eq!(full.area_on_canvas(20, 10), 72);
        assert_eq!(full.translated(7, -2).area_on_canvas(20, 10), 160);
    }
}
