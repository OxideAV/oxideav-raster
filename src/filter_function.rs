//! CSS filter shorthand functions — Filter Effects Module Level 1
//! §6 (Filter Functions) + §13.1 (Filter primitive representation).
//!
//! The `filter` property (§5) accepts a `<filter-value-list>` of ten
//! shorthand `<filter-function>`s. §13.1 defines each one as an
//! equivalent `filter` element; this module transcribes those
//! equivalents onto the crate's [`FilterGraph`](crate::FilterGraph)
//! steps, so a shorthand list evaluates through exactly the same
//! primitive code paths a hand-built filter would.
//!
//! Per §10, "The color-interpolation-filters property has no affect on
//! Filter Functions, which operate in the sRGB color space" —
//! [`apply_filter_functions`] therefore always evaluates in
//! [`FilterColorSpace::Srgb`].
//!
//! §6.1 value handling implemented here:
//!
//! * `grayscale` / `sepia` / `invert` / `opacity` — "Values of amount
//!   over 100% are allowed but UAs must clamp the values to 1";
//! * `brightness` / `contrast` / `saturate` — over-100% values pass
//!   through ("providing brighter results" / "more contrast" /
//!   "super-saturated results");
//! * `hue-rotate` — "Implementations must not normalize this value in
//!   order to allow animations beyond 360deg";
//! * negative amounts are "not allowed" by the grammar; the
//!   constructors here defensively clamp them to `0` so a hostile
//!   value cannot reach the primitives (the §5 parser rejects them
//!   outright before construction);
//! * `blur` — §13.1.9: "edge mode computes to `none` for the `filter`
//!   property", so the equivalent `feGaussianBlur` runs with
//!   [`ConvolveEdgeMode::None`].

use crate::filter::{
    ColorMatrix, ColorMatrixOp, ComponentTransfer, ConvolveEdgeMode, FilterColorSpace,
    OffsetSampling, TransferFunc,
};
use crate::filter_graph::{FilterGraph, FilterPrimitive, FilterStep};

/// One CSS `<filter-function>` (Filter Effects 1 §6.1).
#[derive(Debug, Clone, PartialEq)]
pub enum FilterFunction {
    /// `grayscale(<number-percentage>?)` — §13.1.1.
    Grayscale(f32),
    /// `sepia(<number-percentage>?)` — §13.1.2.
    Sepia(f32),
    /// `saturate(<number-percentage>?)` — §13.1.3.
    Saturate(f32),
    /// `hue-rotate([<angle> | <zero>]?)` in degrees — §13.1.4.
    HueRotate(f32),
    /// `invert(<number-percentage>?)` — §13.1.5.
    Invert(f32),
    /// `opacity(<number-percentage>?)` — §13.1.6.
    Opacity(f32),
    /// `brightness(<number-percentage>?)` — §13.1.7.
    Brightness(f32),
    /// `contrast(<number-percentage>?)` — §13.1.8.
    Contrast(f32),
    /// `blur(<length>?)` — the Gaussian standard deviation in pixels
    /// (§6.1: "The passed parameter defines the value of the standard
    /// deviation to the Gaussian function") — §13.1.9.
    Blur(f32),
    /// `drop-shadow(<color>? && <length>{2,3})` — §13.1.10.
    DropShadow {
        /// Shadow offset, X axis (first `<length>`).
        dx: f32,
        /// Shadow offset, Y axis (second `<length>`).
        dy: f32,
        /// Gaussian standard deviation (optional third `<length>`;
        /// §6.1: "the standard deviation instead of blur radius").
        std_dev: f32,
        /// Shadow colour, sRGB.
        color: (u8, u8, u8),
        /// Shadow colour alpha, `[0, 1]`.
        opacity: f32,
    },
}

/// Clamp a `<number-percentage>` amount to `[0, 1]` (§6.1: negative
/// values are not allowed; over-100% values "must clamp … to 1").
/// NaN collapses to `0` (no effect for the four functions this guards).
#[inline]
fn clamp01(amount: f32) -> f32 {
    if amount.is_nan() {
        0.0
    } else {
        amount.clamp(0.0, 1.0)
    }
}

/// Clamp an unbounded-above amount to `[0, ∞)` (negative values are
/// not allowed; NaN collapses to the identity value `1`).
#[inline]
fn clamp_pos(amount: f32) -> f32 {
    if amount.is_nan() {
        1.0
    } else {
        amount.max(0.0)
    }
}

impl FilterFunction {
    /// The §13.1 filter-element equivalent of this function, as
    /// [`FilterGraph`](crate::FilterGraph) steps (most functions are a
    /// single primitive; `drop-shadow` maps onto the crate's composite
    /// `feDropShadow`, itself the §9.12 five-primitive shorthand).
    pub fn to_steps(&self) -> Vec<FilterStep> {
        match *self {
            // §13.1.1: feColorMatrix type="matrix" with rows built
            // from x = 1 - amount:
            //   (0.2126 + 0.7874x) (0.7152 − 0.7152x) (0.0722 − 0.0722x) 0 0
            //   (0.2126 − 0.2126x) (0.7152 + 0.2848x) (0.0722 − 0.0722x) 0 0
            //   (0.2126 − 0.2126x) (0.7152 − 0.7152x) (0.0722 + 0.9278x) 0 0
            //   0 0 0 1 0
            FilterFunction::Grayscale(amount) => {
                let x = 1.0 - clamp01(amount);
                let m = ColorMatrix([
                    [
                        0.2126 + 0.7874 * x,
                        0.7152 - 0.7152 * x,
                        0.0722 - 0.0722 * x,
                        0.0,
                        0.0,
                    ],
                    [
                        0.2126 - 0.2126 * x,
                        0.7152 + 0.2848 * x,
                        0.0722 - 0.0722 * x,
                        0.0,
                        0.0,
                    ],
                    [
                        0.2126 - 0.2126 * x,
                        0.7152 - 0.7152 * x,
                        0.0722 + 0.9278 * x,
                        0.0,
                        0.0,
                    ],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]);
                vec![FilterStep::new(FilterPrimitive::ColorMatrix {
                    op: ColorMatrixOp::Matrix,
                    matrix: m,
                })]
            }
            // §13.1.2: feColorMatrix type="matrix" with x = 1 - amount:
            //   (0.393 + 0.607x) (0.769 − 0.769x) (0.189 − 0.189x) 0 0
            //   (0.349 − 0.349x) (0.686 + 0.314x) (0.168 − 0.168x) 0 0
            //   (0.272 − 0.272x) (0.534 − 0.534x) (0.131 + 0.869x) 0 0
            //   0 0 0 1 0
            FilterFunction::Sepia(amount) => {
                let x = 1.0 - clamp01(amount);
                let m = ColorMatrix([
                    [
                        0.393 + 0.607 * x,
                        0.769 - 0.769 * x,
                        0.189 - 0.189 * x,
                        0.0,
                        0.0,
                    ],
                    [
                        0.349 - 0.349 * x,
                        0.686 + 0.314 * x,
                        0.168 - 0.168 * x,
                        0.0,
                        0.0,
                    ],
                    [
                        0.272 - 0.272 * x,
                        0.534 - 0.534 * x,
                        0.131 + 0.869 * x,
                        0.0,
                        0.0,
                    ],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]);
                vec![FilterStep::new(FilterPrimitive::ColorMatrix {
                    op: ColorMatrixOp::Matrix,
                    matrix: m,
                })]
            }
            // §13.1.3: feColorMatrix type="saturate" values="[amount]".
            FilterFunction::Saturate(amount) => {
                vec![FilterStep::new(FilterPrimitive::ColorMatrix {
                    op: ColorMatrixOp::Saturate(clamp_pos(amount)),
                    matrix: ColorMatrix::identity(),
                })]
            }
            // §13.1.4: feColorMatrix type="hueRotate" values="[angle]".
            FilterFunction::HueRotate(degrees) => {
                let d = if degrees.is_nan() { 0.0 } else { degrees };
                vec![FilterStep::new(FilterPrimitive::ColorMatrix {
                    op: ColorMatrixOp::HueRotate(d),
                    matrix: ColorMatrix::identity(),
                })]
            }
            // §13.1.5: feComponentTransfer with feFuncR/G/B
            // type="table" tableValues="[amount] (1 - [amount])".
            FilterFunction::Invert(amount) => {
                let a = clamp01(amount);
                let table = TransferFunc::Table(vec![a, 1.0 - a]);
                let ct = ComponentTransfer::identity()
                    .with_r(table.clone())
                    .with_g(table.clone())
                    .with_b(table);
                vec![FilterStep::new(FilterPrimitive::ComponentTransfer(ct))]
            }
            // §13.1.6: feComponentTransfer with feFuncA type="table"
            // tableValues="0 [amount]".
            FilterFunction::Opacity(amount) => {
                let ct = ComponentTransfer::identity()
                    .with_a(TransferFunc::Table(vec![0.0, clamp01(amount)]));
                vec![FilterStep::new(FilterPrimitive::ComponentTransfer(ct))]
            }
            // §13.1.7: feComponentTransfer with feFuncR/G/B
            // type="linear" slope="[amount]".
            FilterFunction::Brightness(amount) => {
                let f = TransferFunc::Linear {
                    slope: clamp_pos(amount),
                    intercept: 0.0,
                };
                let ct = ComponentTransfer::identity()
                    .with_r(f.clone())
                    .with_g(f.clone())
                    .with_b(f);
                vec![FilterStep::new(FilterPrimitive::ComponentTransfer(ct))]
            }
            // §13.1.8: feComponentTransfer with feFuncR/G/B
            // type="linear" slope="[amount]"
            // intercept="-(0.5 * [amount]) + 0.5".
            FilterFunction::Contrast(amount) => {
                let a = clamp_pos(amount);
                let f = TransferFunc::Linear {
                    slope: a,
                    intercept: -(0.5 * a) + 0.5,
                };
                let ct = ComponentTransfer::identity()
                    .with_r(f.clone())
                    .with_g(f.clone())
                    .with_b(f);
                vec![FilterStep::new(FilterPrimitive::ComponentTransfer(ct))]
            }
            // §13.1.9: feGaussianBlur stdDeviation="[radius radius]"
            // with edgeMode `none` for the filter property.
            FilterFunction::Blur(std_dev) => {
                // Non-finite standard deviations (NaN / ±inf) collapse
                // to 0 (no blur): the §6.1 grammar only produces finite
                // non-negative lengths, so anything else is hostile.
                let s = if std_dev.is_finite() {
                    std_dev.max(0.0)
                } else {
                    0.0
                };
                vec![FilterStep::new(FilterPrimitive::GaussianBlur {
                    std_x: s,
                    std_y: s,
                    edge: ConvolveEdgeMode::None,
                })]
            }
            // §13.1.10: the five-primitive drop-shadow chain — the
            // crate's `feDropShadow` primitive is that §9.12 shorthand.
            FilterFunction::DropShadow {
                dx,
                dy,
                std_dev,
                color,
                opacity,
            } => {
                let s = if std_dev.is_finite() {
                    std_dev.max(0.0)
                } else {
                    0.0
                };
                vec![FilterStep::new(FilterPrimitive::DropShadow {
                    std_x: s,
                    std_y: s,
                    dx,
                    dy,
                    color,
                    opacity: clamp01(opacity),
                    sampling: OffsetSampling::default(),
                })]
            }
        }
    }
}

/// Evaluate a `<filter-value-list>` (§5) against a packed
/// straight-alpha RGBA image.
///
/// §5: "The list of functions are applied in the order provided. The
/// first filter function … in the list takes the element
/// (`SourceGraphic`) as the input image" — each function's steps chain
/// through the §9.2 unspecified-`in` rule. Per §10 the evaluation is
/// pinned to the sRGB working space ("The color-interpolation-filters
/// property has no affect on Filter Functions, which operate in the
/// sRGB color space").
///
/// An empty list returns the source unchanged (the `none` value: "No
/// filter effect gets applied").
///
/// # Panics
///
/// * If `source.len() != width as usize * height as usize * 4`.
/// * If `width as usize * height as usize * 4` overflows `usize`.
pub fn apply_filter_functions(
    functions: &[FilterFunction],
    source: &[u8],
    width: u32,
    height: u32,
) -> Vec<u8> {
    if functions.is_empty() {
        let n = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .expect("apply_filter_functions: width * height * 4 overflowed usize");
        assert_eq!(
            source.len(),
            n,
            "apply_filter_functions: source.len() == {} but width*height*4 == {n}",
            source.len()
        );
        return source.to_vec();
    }
    let mut graph = FilterGraph::with_color_space(FilterColorSpace::Srgb);
    for f in functions {
        for step in f.to_steps() {
            graph.push(step);
        }
    }
    graph.evaluate(source, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32) -> Vec<u8> {
        let mut buf = vec![0u8; width as usize * height as usize * 4];
        for (i, px) in buf.chunks_exact_mut(4).enumerate() {
            px[0] = (i * 41 + 13) as u8;
            px[1] = (i * 89 + 7) as u8;
            px[2] = (i * 23 + 191) as u8;
            px[3] = 255;
        }
        buf
    }

    #[test]
    fn empty_list_is_identity() {
        let src = image(4, 4);
        assert_eq!(apply_filter_functions(&[], &src, 4, 4), src);
    }

    #[test]
    fn zero_amount_functions_are_identity() {
        // §6.1: 0% "leaves the input unchanged" for grayscale / sepia /
        // invert / hue-rotate; blur(0) likewise.
        let src = image(4, 4);
        for f in [
            FilterFunction::Grayscale(0.0),
            FilterFunction::Sepia(0.0),
            FilterFunction::HueRotate(0.0),
            FilterFunction::Invert(0.0),
            FilterFunction::Blur(0.0),
        ] {
            let out = apply_filter_functions(std::slice::from_ref(&f), &src, 4, 4);
            for (o, s) in out.iter().zip(src.iter()) {
                assert!(
                    (*o as i32 - *s as i32).abs() <= 1,
                    "{f:?} moved a byte: {s} -> {o}"
                );
            }
        }
        // saturate / brightness / contrast / opacity are identity at 1.
        for f in [
            FilterFunction::Saturate(1.0),
            FilterFunction::Brightness(1.0),
            FilterFunction::Contrast(1.0),
            FilterFunction::Opacity(1.0),
        ] {
            let out = apply_filter_functions(std::slice::from_ref(&f), &src, 4, 4);
            for (o, s) in out.iter().zip(src.iter()) {
                assert!(
                    (*o as i32 - *s as i32).abs() <= 1,
                    "{f:?} moved a byte: {s} -> {o}"
                );
            }
        }
    }

    #[test]
    fn full_grayscale_matches_bt709_luminance() {
        // At amount = 1 the §13.1.1 matrix rows all collapse to
        // (0.2126 0.7152 0.0722 0 0) — every output channel is the
        // BT.709 luminance of the input.
        let src = vec![200u8, 40, 90, 255];
        let out = apply_filter_functions(&[FilterFunction::Grayscale(1.0)], &src, 1, 1);
        let y: f32 = 0.2126 * 200.0 + 0.7152 * 40.0 + 0.0722 * 90.0;
        let y = (y / 255.0).clamp(0.0, 1.0);
        let y8 = (y * 255.0).round() as i32;
        assert!((out[0] as i32 - y8).abs() <= 1, "{} vs {}", out[0], y8);
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
        assert_eq!(out[3], 255);
    }

    #[test]
    fn full_sepia_matches_spec_matrix_on_white() {
        // §13.1.2 at amount = 1 maps white (1,1,1) to the row sums
        // (0.393+0.769+0.189, 0.349+0.686+0.168, 0.272+0.534+0.131)
        // = (1.351, 1.203, 0.937), clamped to (1, 1, 0.937).
        let src = vec![255u8, 255, 255, 255];
        let out = apply_filter_functions(&[FilterFunction::Sepia(1.0)], &src, 1, 1);
        assert_eq!(out[0], 255);
        assert_eq!(out[1], 255);
        let b = (0.937f32 * 255.0).round() as i32;
        assert!((out[2] as i32 - b).abs() <= 1, "{} vs {}", out[2], b);
    }

    #[test]
    fn invert_full_flips_channels_and_preserves_alpha() {
        let src = vec![200u8, 40, 90, 128];
        let out = apply_filter_functions(&[FilterFunction::Invert(1.0)], &src, 1, 1);
        assert_eq!(&out[..3], &[55, 215, 165]);
        assert_eq!(out[3], 128, "invert touches RGB only (§13.1.5)");
    }

    #[test]
    fn opacity_scales_alpha_only() {
        let src = vec![200u8, 40, 90, 200];
        let out = apply_filter_functions(&[FilterFunction::Opacity(0.5)], &src, 1, 1);
        assert_eq!(&out[..3], &[200, 40, 90], "opacity touches alpha only");
        assert_eq!(out[3], 100);
    }

    #[test]
    fn brightness_is_linear_multiplier_with_clamp() {
        let src = vec![100u8, 200, 0, 255];
        let out = apply_filter_functions(&[FilterFunction::Brightness(2.0)], &src, 1, 1);
        assert_eq!(&out[..4], &[200, 255, 0, 255]);
        // brightness(0): "completely black" (§6.1).
        let out = apply_filter_functions(&[FilterFunction::Brightness(0.0)], &src, 1, 1);
        assert_eq!(&out[..4], &[0, 0, 0, 255]);
    }

    #[test]
    fn contrast_zero_is_mid_gray() {
        // §6.1: "A value of 0% will create an image that is completely
        // gray" — slope 0, intercept 0.5.
        let src = image(3, 3);
        let out = apply_filter_functions(&[FilterFunction::Contrast(0.0)], &src, 3, 3);
        for px in out.chunks_exact(4) {
            assert_eq!(&px[..3], &[128, 128, 128]);
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn over_one_amounts_clamp_for_the_four_clamped_functions() {
        // §6.1: grayscale / sepia / invert / opacity clamp to 1.
        let src = image(4, 4);
        for (over, exact) in [
            (
                FilterFunction::Grayscale(7.5),
                FilterFunction::Grayscale(1.0),
            ),
            (FilterFunction::Sepia(2.0), FilterFunction::Sepia(1.0)),
            (FilterFunction::Invert(1e30), FilterFunction::Invert(1.0)),
            (FilterFunction::Opacity(3.0), FilterFunction::Opacity(1.0)),
        ] {
            assert_eq!(
                apply_filter_functions(&[over], &src, 4, 4),
                apply_filter_functions(&[exact], &src, 4, 4)
            );
        }
    }

    #[test]
    fn hostile_amounts_do_not_panic() {
        let src = image(4, 4);
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 1e38] {
            for f in [
                FilterFunction::Grayscale(v),
                FilterFunction::Sepia(v),
                FilterFunction::Saturate(v),
                FilterFunction::HueRotate(v),
                FilterFunction::Invert(v),
                FilterFunction::Opacity(v),
                FilterFunction::Brightness(v),
                FilterFunction::Contrast(v),
                FilterFunction::Blur(if v.is_finite() { v.min(50.0) } else { v }),
            ] {
                let out = apply_filter_functions(&[f], &src, 4, 4);
                assert_eq!(out.len(), src.len());
            }
        }
    }
}
