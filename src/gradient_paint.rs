//! Gradient paint servers with their own coordinate system — the
//! SVG 2 §14.2 `gradientUnits` / `gradientTransform` model.
//!
//! `oxideav-core`'s [`Paint`] gradients are expressed directly in the
//! painted element's user space. SVG (and PDF shading patterns with a
//! `Matrix`) define gradients in a coordinate system of their own that
//! is mapped onto user space by
//!
//! ```text
//!   user = B · gradientTransform · gradient
//! ```
//!
//! where `B` is the identity under `gradientUnits="userSpaceOnUse"` and
//! the object-bounding-box matrix `[bw 0 0 bh bx by]` under
//! `objectBoundingBox` (SVG 2 §8.11: "(0,0) is at the top/left of the
//! object bounding box and (1,1) is at the bottom/right"). The
//! `gradientTransform` "is post-multiplied to (i.e., inserted to the
//! right of) any previously defined transformations, including the
//! implicit transformation necessary to convert from object bounding
//! box units to user coordinate system" (§14.2.2.1 / §14.2.3.1).
//!
//! A [`GradientPaint`] carries the gradient geometry together with its
//! units and transform; [`GradientPaint::sampler`] resolves it against
//! a node's current transformation matrix (and bounding box) into a
//! [`GradientSampler`] that maps device pixels back into gradient
//! space and evaluates the colour there. Because the mapping is a
//! single affine inverse, every spec consequence falls out without
//! special cases: a radial gradient on a non-square bounding box, or
//! under a skewing `gradientTransform`, renders elliptical rings
//! (§14.2.3.1 "the rings … will render as elliptical"), and a linear
//! gradient's normal turns non-perpendicular to its vector exactly as
//! §14.2.2.1 describes.
//!
//! Decoders that already resolve gradients into user space keep working
//! unchanged through [`Paint`]; `GradientPaint::from_paint` lifts such a
//! paint into this record so a decoder can attach the transform it
//! could not express (the typical case is a radial gradient under a
//! non-conformal mapping, where `RadialGradient` alone cannot carry the
//! ellipse).

use oxideav_core::{LinearGradient, Paint, Path, Point, RadialGradient, Rect, Rgba, Transform2D};

use crate::flatten::flatten_path;
use crate::gradient::{
    FocalRadialGradient, InterpolationSpace, StopsLut, eval_focal_radial_gradient_lut,
    eval_linear_gradient_lut, eval_radial_gradient_lut,
};

/// Coordinate system the gradient geometry is expressed in (SVG 2
/// §14.2.2.1 / §14.2.3.1 `gradientUnits`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum GradientUnits {
    /// Geometry is in the referencing element's user space. The default
    /// here (it is what [`Paint`] gradients already mean); note SVG's
    /// own attribute default is `objectBoundingBox`.
    #[default]
    UserSpaceOnUse,
    /// Geometry is in object-bounding-box units: `(0, 0)` is the
    /// top-left and `(1, 1)` the bottom-right of the painted element's
    /// geometry bounding box (SVG 2 §8.11). An element whose box has no
    /// width or no height is not painted with this gradient at all
    /// (§8.11 "the given effect … will be ignored").
    ObjectBoundingBox,
}

/// The gradient geometry a [`GradientPaint`] carries.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum GradientGeometry {
    /// A linear gradient (SVG 2 §14.2.2).
    Linear(LinearGradient),
    /// A radial gradient with a point focus (SVG 1.1 §13.2.3).
    Radial(RadialGradient),
    /// A two-circle radial gradient with a focal radius (SVG 2
    /// §14.2.3 `fr`).
    FocalRadial(FocalRadialGradient),
}

/// A gradient paint server with its own coordinate system.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GradientPaint {
    /// The gradient, in gradient space.
    pub geometry: GradientGeometry,
    /// Which space `geometry` is expressed in.
    pub units: GradientUnits,
    /// `gradientTransform`: gradient space → the space named by
    /// `units`. Identity when absent.
    pub transform: Transform2D,
}

impl GradientPaint {
    /// A gradient in user space with no extra transform.
    pub fn new(geometry: GradientGeometry) -> Self {
        Self {
            geometry,
            units: GradientUnits::UserSpaceOnUse,
            transform: Transform2D::identity(),
        }
    }

    /// Shorthand for `new(GradientGeometry::Linear(g))`.
    pub fn linear(g: LinearGradient) -> Self {
        Self::new(GradientGeometry::Linear(g))
    }

    /// Shorthand for `new(GradientGeometry::Radial(g))`.
    pub fn radial(g: RadialGradient) -> Self {
        Self::new(GradientGeometry::Radial(g))
    }

    /// Shorthand for `new(GradientGeometry::FocalRadial(g))`.
    pub fn focal_radial(g: FocalRadialGradient) -> Self {
        Self::new(GradientGeometry::FocalRadial(g))
    }

    /// Lift a gradient [`Paint`] (user-space geometry) into this
    /// record. Returns `None` for solid and any non-gradient paint.
    pub fn from_paint(paint: &Paint) -> Option<Self> {
        match paint {
            Paint::LinearGradient(g) => Some(Self::linear(g.clone())),
            Paint::RadialGradient(g) => Some(Self::radial(g.clone())),
            _ => None,
        }
    }

    /// Set the coordinate system of the geometry.
    pub fn with_units(mut self, units: GradientUnits) -> Self {
        self.units = units;
        self
    }

    /// Set `gradientTransform`.
    pub fn with_transform(mut self, transform: Transform2D) -> Self {
        self.transform = transform;
        self
    }

    /// The affine map from gradient space to the referencing element's
    /// user space: `B · gradientTransform`, where `B` is the
    /// object-bounding-box matrix for `ObjectBoundingBox` units and the
    /// identity otherwise.
    ///
    /// Returns `None` when the paint must not be applied: `bbox` is
    /// `None` or has a non-positive width or height under
    /// `ObjectBoundingBox` (SVG 2 §8.11), or the resulting matrix is
    /// singular (the gradient collapses to a line — nothing can be
    /// sampled).
    pub fn gradient_to_user(&self, bbox: Option<Rect>) -> Option<Transform2D> {
        let b = match self.units {
            GradientUnits::UserSpaceOnUse => Transform2D::identity(),
            GradientUnits::ObjectBoundingBox => {
                let r = bbox?;
                if !(r.width > 0.0 && r.height > 0.0) {
                    return None;
                }
                Transform2D {
                    a: r.width,
                    b: 0.0,
                    c: 0.0,
                    d: r.height,
                    e: r.x,
                    f: r.y,
                }
            }
            #[allow(unreachable_patterns)]
            _ => Transform2D::identity(),
        };
        let m = b.compose(&self.transform);
        invert_2d(&m)?;
        Some(m)
    }

    /// Resolve the paint against a node: `ctm` is the node's current
    /// transformation matrix (user → device pixels) and `bbox` its
    /// geometry bounding box in user space (only consulted under
    /// `ObjectBoundingBox`; see [`path_bounding_box`]). The returned
    /// sampler evaluates the gradient at device pixel positions.
    ///
    /// `None` means nothing is painted — see
    /// [`GradientPaint::gradient_to_user`], plus a singular `ctm`.
    pub fn sampler(
        &self,
        ctm: &Transform2D,
        bbox: Option<Rect>,
        space: InterpolationSpace,
    ) -> Option<GradientSampler> {
        let to_user = self.gradient_to_user(bbox)?;
        let to_device = ctm.compose(&to_user);
        let inv = invert_2d(&to_device)?;
        let stops = match &self.geometry {
            GradientGeometry::Linear(g) => &g.stops,
            GradientGeometry::Radial(g) => &g.stops,
            GradientGeometry::FocalRadial(g) => &g.stops,
            #[allow(unreachable_patterns)]
            _ => return None,
        };
        Some(GradientSampler {
            inv,
            geometry: self.geometry.clone(),
            lut: StopsLut::build(stops, space),
        })
    }
}

/// A [`GradientPaint`] resolved against a node's transform: maps device
/// positions back into gradient space and evaluates the colour there
/// through a pre-built stops LUT.
#[derive(Debug, Clone)]
pub struct GradientSampler {
    /// Device → gradient space.
    inv: Transform2D,
    geometry: GradientGeometry,
    lut: StopsLut,
}

impl GradientSampler {
    /// Straight-alpha colour at device position `(x, y)` — pass pixel
    /// centres (`px + 0.5`, `py + 0.5`) for a raster.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> Rgba {
        let p = self.inv.apply(Point::new(x, y));
        match &self.geometry {
            GradientGeometry::Linear(g) => eval_linear_gradient_lut(g, p.x, p.y, &self.lut),
            GradientGeometry::Radial(g) => eval_radial_gradient_lut(g, p.x, p.y, &self.lut),
            GradientGeometry::FocalRadial(g) => {
                eval_focal_radial_gradient_lut(g, p.x, p.y, &self.lut)
            }
            #[allow(unreachable_patterns)]
            _ => Rgba::new(0, 0, 0, 0),
        }
    }

    /// The device → gradient-space transform this sampler applies.
    pub fn device_to_gradient(&self) -> Transform2D {
        self.inv
    }
}

/// Geometry bounding box of `path` in its own user space — the SVG 2
/// §8.11 object bounding box (fill geometry only; stroke width, markers
/// and filters do not contribute). Curves are measured on their
/// flattened polylines, which are within the flattening tolerance of
/// the exact extrema. `None` for a path with no points.
pub fn path_bounding_box(path: &Path) -> Option<Rect> {
    let contours = flatten_path(&path.commands, &Transform2D::identity());
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any = false;
    for c in &contours {
        for &(x, y) in &c.points {
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            any = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    any.then(|| Rect::new(min_x, min_y, max_x - min_x, max_y - min_y))
}

/// Inverse of an affine 2D transform. `None` if singular.
pub(crate) fn invert_2d(t: &Transform2D) -> Option<Transform2D> {
    let det = t.a * t.d - t.b * t.c;
    // Also rejects a NaN determinant.
    if det.is_nan() || det.abs() <= 1e-12 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{GradientStop, SpreadMethod};

    fn bw() -> Vec<GradientStop> {
        vec![
            GradientStop::new(0.0, Rgba::opaque(0, 0, 0)),
            GradientStop::new(1.0, Rgba::opaque(255, 255, 255)),
        ]
    }

    fn unit_linear() -> LinearGradient {
        LinearGradient::new(Point::new(0.0, 0.0), Point::new(1.0, 0.0)).with_stops(bw())
    }

    #[test]
    fn bbox_units_map_unit_square_onto_the_box() {
        let gp = GradientPaint::linear(unit_linear()).with_units(GradientUnits::ObjectBoundingBox);
        let m = gp
            .gradient_to_user(Some(Rect::new(10.0, 20.0, 40.0, 8.0)))
            .unwrap();
        let p = m.apply(Point::new(0.0, 0.0));
        assert!((p.x - 10.0).abs() < 1e-5 && (p.y - 20.0).abs() < 1e-5);
        let q = m.apply(Point::new(1.0, 1.0));
        assert!((q.x - 50.0).abs() < 1e-5 && (q.y - 28.0).abs() < 1e-5);
    }

    #[test]
    fn bbox_units_without_a_usable_box_paint_nothing() {
        let gp = GradientPaint::linear(unit_linear()).with_units(GradientUnits::ObjectBoundingBox);
        assert!(gp.gradient_to_user(None).is_none());
        assert!(
            gp.gradient_to_user(Some(Rect::new(0.0, 0.0, 0.0, 10.0)))
                .is_none()
        );
        assert!(
            gp.gradient_to_user(Some(Rect::new(0.0, 0.0, 10.0, 0.0)))
                .is_none()
        );
        // User-space units ignore the box entirely.
        let us = GradientPaint::linear(unit_linear());
        assert!(us.gradient_to_user(None).is_some());
    }

    #[test]
    fn gradient_transform_is_post_multiplied() {
        // B = [40 0 0 8 10 20], T = translate(0.5, 0): user = B·T·g, so
        // gradient (0, 0) → bbox-space (0.5, 0) → user (30, 20).
        let gp = GradientPaint::linear(unit_linear())
            .with_units(GradientUnits::ObjectBoundingBox)
            .with_transform(Transform2D::translate(0.5, 0.0));
        let m = gp
            .gradient_to_user(Some(Rect::new(10.0, 20.0, 40.0, 8.0)))
            .unwrap();
        let p = m.apply(Point::new(0.0, 0.0));
        assert!(
            (p.x - 30.0).abs() < 1e-4 && (p.y - 20.0).abs() < 1e-4,
            "{p:?}"
        );
    }

    #[test]
    fn singular_transform_paints_nothing() {
        let gp = GradientPaint::linear(unit_linear()).with_transform(Transform2D::scale(1.0, 0.0));
        assert!(gp.gradient_to_user(None).is_none());
        let ok = GradientPaint::linear(unit_linear());
        assert!(
            ok.sampler(
                &Transform2D::scale(0.0, 1.0),
                None,
                InterpolationSpace::Srgb
            )
            .is_none()
        );
    }

    #[test]
    fn from_paint_lifts_gradients_only() {
        assert!(GradientPaint::from_paint(&Paint::Solid(Rgba::opaque(1, 2, 3))).is_none());
        let lifted = GradientPaint::from_paint(&Paint::LinearGradient(unit_linear())).unwrap();
        assert!(matches!(lifted.geometry, GradientGeometry::Linear(_)));
        assert_eq!(lifted.units, GradientUnits::UserSpaceOnUse);
        assert!(lifted.transform.is_identity());
    }

    #[test]
    fn sampler_maps_device_pixels_through_ctm_bbox_and_transform() {
        // Gradient along bbox x; bbox 10..50 × 20..28 in user space; CTM
        // scales user space by 2. Device x = 60 ↔ user 30 ↔ bbox 0.5 → mid.
        let gp = GradientPaint::linear(unit_linear()).with_units(GradientUnits::ObjectBoundingBox);
        let s = gp
            .sampler(
                &Transform2D::scale(2.0, 2.0),
                Some(Rect::new(10.0, 20.0, 40.0, 8.0)),
                InterpolationSpace::Srgb,
            )
            .unwrap();
        assert_eq!(s.sample(20.0, 44.0).r, 0);
        assert!((s.sample(60.0, 44.0).r as i32 - 128).abs() <= 1);
        assert_eq!(s.sample(100.0, 44.0).r, 255);
        let back = s.device_to_gradient().apply(Point::new(60.0, 44.0));
        assert!((back.x - 0.5).abs() < 1e-5 && (back.y - 0.25).abs() < 1e-5);
    }

    #[test]
    fn radial_on_non_square_box_is_elliptical() {
        // Unit circle in bbox space on a 40 × 8 box: the t = 0.5
        // iso-contour is 10 px from the centre horizontally but 2 px
        // vertically.
        let rg = RadialGradient::new(Point::new(0.5, 0.5), 0.5).with_stops(bw());
        let gp = GradientPaint::radial(rg).with_units(GradientUnits::ObjectBoundingBox);
        let s = gp
            .sampler(
                &Transform2D::identity(),
                Some(Rect::new(0.0, 0.0, 40.0, 8.0)),
                InterpolationSpace::Srgb,
            )
            .unwrap();
        let east = s.sample(30.0, 4.0).r;
        let south = s.sample(20.0, 6.0).r;
        assert!((east as i32 - 128).abs() <= 1, "east {east}");
        assert!((south as i32 - 128).abs() <= 1, "south {south}");
        assert_eq!(s.sample(40.0, 4.0).r, 255);
    }

    #[test]
    fn focal_radial_geometry_samples() {
        let fr = FocalRadialGradient::new(Point::new(0.0, 0.0), 10.0, Point::new(0.0, 0.0), 2.0)
            .with_stops(bw());
        let s = GradientPaint::focal_radial(fr)
            .sampler(&Transform2D::identity(), None, InterpolationSpace::Srgb)
            .unwrap();
        assert_eq!(s.sample(1.0, 0.0).r, 0);
        assert!((s.sample(6.0, 0.0).r as i32 - 128).abs() <= 2);
    }

    #[test]
    fn path_bounding_box_is_the_geometry_box() {
        let mut p = Path::new();
        p.move_to(Point::new(3.0, 4.0))
            .line_to(Point::new(13.0, 4.0))
            .line_to(Point::new(13.0, 9.0))
            .close();
        let r = path_bounding_box(&p).unwrap();
        assert_eq!((r.x, r.y, r.width, r.height), (3.0, 4.0, 10.0, 5.0));
        assert!(path_bounding_box(&Path::new()).is_none());
    }

    #[test]
    fn sampler_has_no_spread_surprises_under_pad() {
        let g = LinearGradient::new(Point::new(0.0, 0.0), Point::new(10.0, 0.0))
            .with_stops(bw())
            .with_spread(SpreadMethod::Pad);
        let s = GradientPaint::linear(g)
            .sampler(&Transform2D::identity(), None, InterpolationSpace::Srgb)
            .unwrap();
        assert_eq!(s.sample(-5.0, 0.0).r, 0);
        assert_eq!(s.sample(50.0, 0.0).r, 255);
    }
}
