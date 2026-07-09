//! Filter primitive tree integration tests — Filter Effects 1 §9.2 /
//! §9.3, driven through [`oxideav_raster::FilterGraph`].
//!
//! The two worked examples in §9.3 are reproduced verbatim as golden
//! tests: the three-primitive filter chain (`feColorMatrix` →
//! `feOffset` → `feGaussianBlur`) and the two-tree filter whose primary
//! tree is `feFlood` + `feComposite(operator=in, in=SourceAlpha,
//! in2="flood")`.

use oxideav_raster::{
    color_matrix_op, composite_filter, flood, gaussian_blur, in_filter_space, merge, offset,
    ColorMatrix, ColorMatrixOp, CompositeOp, ConvolveEdgeMode, FilterColorSpace, FilterGraph,
    FilterInput, FilterPrimitive, FilterStep, OffsetSampling, Subregion,
};

/// A deterministic non-uniform test image with varied alpha.
fn test_image(width: u32, height: u32) -> Vec<u8> {
    let mut buf = vec![0u8; width as usize * height as usize * 4];
    for y in 0..height {
        for x in 0..width {
            let i = (y as usize * width as usize + x as usize) * 4;
            buf[i] = (x * 37 + 11) as u8;
            buf[i + 1] = (y * 53 + 5) as u8;
            buf[i + 2] = ((x + y) * 29) as u8;
            buf[i + 3] = if (x + y) % 3 == 0 { 255 } else { 160 };
        }
    }
    buf
}

fn source_alpha_of(src: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; src.len()];
    for (dst, s) in buf.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        dst[3] = s[3];
    }
    buf
}

/// §9.3's first example, quoted: "`feColorMatrix`, `feOffset` and
/// `feGaussianBlur` create a filter chain. `feColorMatrix` takes
/// `SourceGraphic` as input. The result is the input of `feOffset` with
/// its result being the input of `feGaussianBlur`." Every `in` is left
/// unspecified, so the whole chain runs on the §9.2 Auto rule.
#[test]
fn spec_9_3_filter_chain_example_matches_manual_composition() {
    let (w, h) = (12u32, 10u32);
    let src = test_image(w, h);

    let mut g = FilterGraph::with_color_space(FilterColorSpace::Srgb);
    g.push(FilterStep::new(FilterPrimitive::ColorMatrix {
        op: ColorMatrixOp::HueRotate(45.0),
        matrix: ColorMatrix::identity(),
    }));
    g.push(FilterStep::new(FilterPrimitive::Offset {
        dx: 10.0,
        dy: 10.0,
        sampling: OffsetSampling::default(),
    }));
    g.push(FilterStep::new(FilterPrimitive::GaussianBlur {
        std_x: 3.0,
        std_y: 3.0,
        edge: ConvolveEdgeMode::default(),
    }));

    let hue = color_matrix_op(
        &src,
        w,
        h,
        ColorMatrixOp::HueRotate(45.0),
        &ColorMatrix::identity(),
    );
    let off = offset(&hue, w, h, 10.0, 10.0, OffsetSampling::default());
    let blur = gaussian_blur(&off, w, h, 3.0, 3.0);

    assert_eq!(g.evaluate(&src, w, h), blur);
}

/// §9.3's second example, quoted: a filter with "2 filter primitive
/// trees" — `feColorMatrix` / `feOffset` / `feGaussianBlur` (ignored)
/// and `feFlood` + `feComposite` (the primary tree). "Only the 2nd, the
/// primary filter primitive tree contributes to the filter process."
#[test]
fn spec_9_3_multiple_trees_example_only_primary_tree_contributes() {
    let (w, h) = (9u32, 7u32);
    let src = test_image(w, h);

    let mut g = FilterGraph::with_color_space(FilterColorSpace::Srgb);
    // First (ignored) tree.
    g.push(FilterStep::new(FilterPrimitive::ColorMatrix {
        op: ColorMatrixOp::HueRotate(45.0),
        matrix: ColorMatrix::identity(),
    }));
    g.push(FilterStep::new(FilterPrimitive::Offset {
        dx: 10.0,
        dy: 10.0,
        sampling: OffsetSampling::default(),
    }));
    g.push(FilterStep::new(FilterPrimitive::GaussianBlur {
        std_x: 3.0,
        std_y: 3.0,
        edge: ConvolveEdgeMode::default(),
    }));
    // Primary tree: <feFlood flood-color="green" result="flood"/>
    // <feComposite operator="in" in="SourceAlpha" in2="flood"/>
    g.push(
        FilterStep::new(FilterPrimitive::Flood {
            color: (0, 128, 0),
            opacity: 1.0,
        })
        .with_result("flood"),
    );
    g.push(
        FilterStep::new(FilterPrimitive::Composite {
            op: CompositeOp::In,
        })
        .with_input(FilterInput::SourceAlpha)
        .with_input2(FilterInput::Reference("flood".into())),
    );

    let alpha = source_alpha_of(&src);
    let green = flood(w, h, 0, 128, 0, 1.0);
    let expected = composite_filter(&alpha, &green, w, h, CompositeOp::In);

    assert_eq!(g.evaluate(&src, w, h), expected);
}

/// A linear-space graph must equal running the same primitive through
/// the §10 `in_filter_space` wrapper: linearise → operate → re-encode.
#[test]
fn linear_space_graph_matches_in_filter_space_wrapper() {
    let (w, h) = (8u32, 8u32);
    let src = test_image(w, h);

    let mut g = FilterGraph::with_color_space(FilterColorSpace::LinearRgb);
    g.push(FilterStep::new(FilterPrimitive::GaussianBlur {
        std_x: 1.5,
        std_y: 1.5,
        edge: ConvolveEdgeMode::default(),
    }));

    let expected = in_filter_space(FilterColorSpace::LinearRgb, &src, |buf| {
        gaussian_blur(buf, w, h, 1.5, 1.5)
    });

    assert_eq!(g.evaluate(&src, w, h), expected);
}

/// `color-interpolation-filters: auto` resolves to `linearRGB` (§10:
/// the property's initial value), so an `Auto` graph and a `LinearRgb`
/// graph agree byte for byte.
#[test]
fn auto_color_space_resolves_to_linear() {
    let (w, h) = (6u32, 6u32);
    let src = test_image(w, h);
    let step = FilterStep::new(FilterPrimitive::GaussianBlur {
        std_x: 1.0,
        std_y: 1.0,
        edge: ConvolveEdgeMode::default(),
    });

    let mut ga = FilterGraph::with_color_space(FilterColorSpace::Auto);
    ga.push(step.clone());
    let mut gl = FilterGraph::with_color_space(FilterColorSpace::LinearRgb);
    gl.push(step);

    assert_eq!(ga.evaluate(&src, w, h), gl.evaluate(&src, w, h));
}

/// A flood colour specified in sRGB must come back out in sRGB even
/// when the graph works in linear space: the generator linearises the
/// specified colour on entry and the graph re-encodes on exit. The
/// extremes are exactly invertible; mid-range channels may pick up a
/// couple of quantisation steps from the double byte-rounding. (Very
/// dark non-zero channels are excluded here: an 8-bit linear
/// intermediate is coarsest in the darks — sRGB 17 maps to linear
/// ≈ 1.4/255, so one linear quantisation step spans several sRGB
/// steps. That is a property of the crate-wide byte-buffer pipeline,
/// not of the graph evaluator.)
#[test]
fn flood_color_survives_linear_working_space() {
    let (w, h) = (4u32, 4u32);
    let src = test_image(w, h);

    for color in [(255u8, 0u8, 0u8), (0, 255, 0), (200, 60, 0), (80, 130, 250)] {
        let mut g = FilterGraph::with_color_space(FilterColorSpace::LinearRgb);
        g.push(FilterStep::new(FilterPrimitive::Flood {
            color,
            opacity: 1.0,
        }));
        let out = g.evaluate(&src, w, h);
        let px = &out[..4];
        assert!(
            (px[0] as i32 - color.0 as i32).abs() <= 2
                && (px[1] as i32 - color.1 as i32).abs() <= 2
                && (px[2] as i32 - color.2 as i32).abs() <= 2,
            "flood {color:?} came back as {px:?}"
        );
        assert_eq!(px[3], 255);
    }
}

/// `feMerge` (§9.16) inside a graph: a merge of `SourceGraphic` and a
/// named flood equals the standalone [`merge`] of the same layers.
#[test]
fn merge_composites_named_layers_bottom_to_top() {
    let (w, h) = (5u32, 5u32);
    let src = test_image(w, h);

    let mut g = FilterGraph::with_color_space(FilterColorSpace::Srgb);
    g.push(
        FilterStep::new(FilterPrimitive::Flood {
            color: (10, 200, 30),
            opacity: 0.5,
        })
        .with_result("wash"),
    );
    g.push(FilterStep::new(FilterPrimitive::Merge {
        inputs: vec![
            FilterInput::SourceGraphic,
            FilterInput::Reference("wash".into()),
        ],
    }));

    let wash = flood(w, h, 10, 200, 30, 0.5);
    let expected = merge(w, h, &[&src, &wash]);

    assert_eq!(g.evaluate(&src, w, h), expected);
}

/// The §9.4 subregion on a step hard-clips that step's result before
/// downstream steps consume it.
#[test]
fn step_subregion_clips_before_downstream_consumption() {
    let (w, h) = (6u32, 6u32);
    let src = test_image(w, h);

    let mut g = FilterGraph::with_color_space(FilterColorSpace::Srgb);
    g.push(
        FilterStep::new(FilterPrimitive::Flood {
            color: (255, 255, 255),
            opacity: 1.0,
        })
        .with_subregion(Subregion::new(0.0, 0.0, 3.0, 6.0))
        .with_result("half"),
    );
    g.push(FilterStep::new(FilterPrimitive::Offset {
        dx: 0.0,
        dy: 0.0,
        sampling: OffsetSampling::default(),
    }));

    let out = g.evaluate(&src, w, h);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            if x < 3 {
                assert_eq!(&out[i..i + 4], &[255, 255, 255, 255], "inside at {x},{y}");
            } else {
                assert_eq!(&out[i..i + 4], &[0, 0, 0, 0], "clipped at {x},{y}");
            }
        }
    }
}

/// `feTile` (§9.20) anchored at its input-subregion origin: a 2×2 tile
/// cut at (2, 2) repeats across the whole canvas with tile corners at
/// `(2 + 2i, 2 + 2j)` — quoting SVG 1.1 §15.23, "the top/left corner
/// of each given tile is at location (x+i*width, y+j*height)".
#[test]
fn tile_repeats_input_subregion_across_canvas() {
    let (w, h) = (6u32, 6u32);
    let mut src = vec![0u8; (w * h * 4) as usize];
    // Distinct 2×2 block at (2, 2).
    let colors = [
        [1u8, 2, 3, 255],
        [4, 5, 6, 255],
        [7, 8, 9, 255],
        [10, 11, 12, 255],
    ];
    for (k, c) in colors.iter().enumerate() {
        let (x, y) = (2 + (k as u32 % 2), 2 + (k as u32 / 2));
        let i = ((y * w + x) * 4) as usize;
        src[i..i + 4].copy_from_slice(c);
    }

    let mut g = FilterGraph::with_color_space(FilterColorSpace::Srgb);
    g.push(FilterStep::new(FilterPrimitive::Tile {
        tile: Subregion::new(2.0, 2.0, 2.0, 2.0),
    }));
    let out = g.evaluate(&src, w, h);

    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let k = ((y as i64 - 2).rem_euclid(2) * 2 + (x as i64 - 2).rem_euclid(2)) as usize;
            assert_eq!(&out[i..i + 4], &colors[k], "tile phase at {x},{y}");
        }
    }
}

/// A dangling reference on the *first* step degrades to Auto, which on
/// the first step reads `SourceGraphic` (§9.2).
#[test]
fn dangling_reference_on_first_step_reads_source_graphic() {
    let (w, h) = (4u32, 4u32);
    let src = test_image(w, h);

    let mut g = FilterGraph::with_color_space(FilterColorSpace::Srgb);
    g.push(
        FilterStep::new(FilterPrimitive::Offset {
            dx: 0.0,
            dy: 0.0,
            sampling: OffsetSampling::default(),
        })
        .with_input(FilterInput::Reference("nowhere".into())),
    );

    assert_eq!(g.evaluate(&src, w, h), src);
}
