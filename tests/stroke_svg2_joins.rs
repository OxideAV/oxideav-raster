//! Integration tests for the SVG 2 §13.5.5 `stroke-linejoin` values
//! `miter-clip` and `arcs`, exercised end-to-end through the public
//! `stroke_to_fill_path_ext` entry point.

use oxideav_core::{LineCap, LineJoin, Paint, Path, Point, Rgba, Stroke, Transform2D};
use oxideav_raster::{
    flatten_path, stroke_to_fill_path, stroke_to_fill_path_ext, ExtendedLineJoin, ExtendedStroke,
};

fn sharp_corner_stroke(width: f32, miter_limit: f32) -> Stroke {
    Stroke {
        width,
        paint: Paint::Solid(Rgba::opaque(0, 0, 0)),
        cap: LineCap::Butt,
        join: LineJoin::Miter,
        miter_limit,
        dash: None,
    }
}

/// A V-shape with a very sharp interior angle: the un-clamped miter
/// would spike far past the join. Incoming +x, vertex at (20, 0),
/// outgoing back toward the origin one unit up.
fn sharp_v() -> Vec<oxideav_raster::FlatContour> {
    let mut p = Path::new();
    p.move_to(Point::new(0.0, 0.0))
        .line_to(Point::new(20.0, 0.0))
        .line_to(Point::new(0.0, 1.0));
    flatten_path(&p.commands, &Transform2D::identity())
}

fn max_x(contours: &[oxideav_raster::FlatContour]) -> f32 {
    contours
        .iter()
        .flat_map(|c| c.points.iter())
        .map(|p| p.0)
        .fold(f32::NEG_INFINITY, f32::max)
}

#[test]
fn miter_clip_overshoots_less_than_full_miter_but_more_than_bevel() {
    let cs = sharp_v();
    let width = 4.0;
    // A high miter limit keeps the *full* miter spike on a plain miter.
    let miter = sharp_corner_stroke(width, 100.0);
    let g_miter = stroke_to_fill_path(&cs[0], &miter, width);

    // The same geometry with a *low* limit: plain miter collapses to a
    // bevel, miter-clip trims the spike to the limit line.
    let limit = 3.0_f32;
    let bevel_base = sharp_corner_stroke(width, limit);
    let bevel = ExtendedStroke::with_join(bevel_base.clone(), ExtendedLineJoin::Bevel);
    let g_bevel = stroke_to_fill_path_ext(&cs[0], &bevel, width);

    let clip = ExtendedStroke::with_join(bevel_base, ExtendedLineJoin::MiterClip);
    let g_clip = stroke_to_fill_path_ext(&cs[0], &clip, width);

    let x_miter = max_x(&g_miter);
    let x_clip = max_x(&g_clip);
    let x_bevel = max_x(&g_bevel);

    // Full miter spikes the farthest; bevel is the most conservative;
    // miter-clip sits strictly between the two.
    assert!(
        x_clip < x_miter,
        "miter-clip ({x_clip}) must not reach the full-miter apex ({x_miter})"
    );
    assert!(
        x_clip > x_bevel,
        "miter-clip ({x_clip}) keeps a flat top past the bevel chord ({x_bevel})"
    );
}

#[test]
fn miter_clip_extent_is_bounded_by_the_limit_line() {
    // For a near-symmetric sharp corner the clip line caps the outward
    // reach at roughly miter_limit/2 * stroke-width measured from the
    // join along the bisector; the rendered geometry must never exceed
    // the join apex distance of the full miter.
    let cs = sharp_v();
    let width = 4.0;
    let limit = 2.0_f32;
    let base = sharp_corner_stroke(width, limit);
    let clip = ExtendedStroke::with_join(base, ExtendedLineJoin::MiterClip);
    let g = stroke_to_fill_path_ext(&cs[0], &clip, width);

    // The join sits at x = 20; the clip distance along the (≈ +x)
    // bisector is limit/2 * width = 2.0 * 4 ... but the corner is not
    // perfectly axis-aligned, so bound generously: the outline must not
    // reach the unbounded miter apex, which for this acute angle lands
    // tens of pixels beyond the join.
    let x = max_x(&g);
    assert!(x.is_finite());
    assert!(
        x < 20.0 + limit * 0.5 * width + width,
        "miter-clip extent {x} exceeded the limit-line bound"
    );
}

#[test]
fn arcs_renders_identically_to_miter_clip_on_flattened_paths() {
    // Per the spec, with zero edge curvature `arcs` falls through to
    // `miter-clip`. The renderer always flattens to polylines, so the
    // two produce byte-identical geometry across limits.
    let cs = sharp_v();
    let width = 3.0;
    for limit in [1.0_f32, 2.5, 8.0] {
        let base = sharp_corner_stroke(width, limit);
        let clip = ExtendedStroke::with_join(base.clone(), ExtendedLineJoin::MiterClip);
        let arcs = ExtendedStroke::with_join(base, ExtendedLineJoin::Arcs);
        let g_clip = stroke_to_fill_path_ext(&cs[0], &clip, width);
        let g_arcs = stroke_to_fill_path_ext(&cs[0], &arcs, width);
        assert_eq!(g_clip.len(), g_arcs.len());
        for (a, b) in g_clip.iter().zip(g_arcs.iter()) {
            assert_eq!(a.points, b.points, "arcs == miter-clip at limit {limit}");
        }
    }
}

#[test]
fn ext_path_reproduces_core_path_for_the_three_svg11_joins() {
    // stroke_to_fill_path_ext with a lifted core join must reproduce the
    // legacy stroke_to_fill_path output exactly (no regression from the
    // shared-geometry refactor).
    let mut p = Path::new();
    p.move_to(Point::new(2.0, 2.0))
        .line_to(Point::new(18.0, 2.0))
        .line_to(Point::new(18.0, 18.0))
        .line_to(Point::new(2.0, 18.0))
        .close();
    let cs = flatten_path(&p.commands, &Transform2D::identity());
    let width = 3.0;
    for (core, ext) in [
        (LineJoin::Miter, ExtendedLineJoin::Miter),
        (LineJoin::Round, ExtendedLineJoin::Round),
        (LineJoin::Bevel, ExtendedLineJoin::Bevel),
    ] {
        let mut s = sharp_corner_stroke(width, 4.0);
        s.join = core;
        let legacy = stroke_to_fill_path(&cs[0], &s, width);
        let es = ExtendedStroke::with_join(s, ext);
        let extended = stroke_to_fill_path_ext(&cs[0], &es, width);
        assert_eq!(legacy.len(), extended.len());
        for (a, b) in legacy.iter().zip(extended.iter()) {
            assert_eq!(a.points, b.points, "{ext:?} reproduces {core:?}");
        }
    }
}

#[test]
fn miter_clip_keeps_dashed_strokes_well_formed() {
    // A dashed open path with sharp interior corners and miter-clip
    // joins must still yield closed, finite outlines.
    let mut p = Path::new();
    p.move_to(Point::new(0.0, 0.0))
        .line_to(Point::new(30.0, 0.0))
        .line_to(Point::new(0.0, 3.0));
    let cs = flatten_path(&p.commands, &Transform2D::identity());
    let mut base = sharp_corner_stroke(4.0, 2.0);
    base.dash = Some(oxideav_core::DashPattern {
        array: vec![5.0, 3.0],
        offset: 0.0,
    });
    let es = ExtendedStroke::with_join(base, ExtendedLineJoin::MiterClip);
    let g = stroke_to_fill_path_ext(&cs[0], &es, 4.0);
    assert!(!g.is_empty());
    for c in &g {
        assert!(c.closed);
        for pt in &c.points {
            assert!(pt.0.is_finite() && pt.1.is_finite());
        }
    }
}
