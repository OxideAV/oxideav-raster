#![no_main]

//! Filter primitive tree fuzz target (Filter Effects 1 §9.2 / §9.3).
//!
//! Fuzzer bytes become a `FilterGraph` on a small canvas: each step's
//! primitive, `in` / `in2` wiring (including dangling references,
//! forward references, and duplicate `result` names — the §9.2
//! degradation paths), optional subregion rectangle built from raw f32
//! bits (NaN / ±inf / negative extents reach the §9.4 disabled-region
//! logic), and the graph-level working colour space are all
//! fuzzer-chosen. The contract is panic-free evaluation returning an
//! exactly `width × height × 4` buffer.
//!
//! Parameters whose cost is inherently linear in their magnitude
//! (turbulence octaves, convolve-matrix order, morphology radii) are
//! range-clamped in the harness — the wire format documents each
//! clamp — so the iteration budget lands on the graph/edge-case logic
//! rather than on legitimately-expensive kernels.
//!
//! ## Wire framing
//!
//! * byte 0 — canvas width, clamped to 1..=24
//! * byte 1 — canvas height, clamped to 1..=24
//! * byte 2 — colour space selector (% 3)
//! * bytes 3.. — per-step records, 12 bytes each:
//!   `[primitive, in_sel, in2_sel, result_sel, sub_flag, p0..p6]`
//!   (up to 12 steps consumed)

use libfuzzer_sys::fuzz_target;
use oxideav_raster::{
    BlendFilterMode, ColorMatrix, ColorMatrixOp, ComponentTransfer, CompositeOp,
    ConvolveEdgeMode, ConvolveMatrix, DisplacementChannel, DisplacementSampling,
    FilterColorSpace, FilterGraph, FilterInput, FilterPrimitive, FilterStep, MorphologyOp,
    OffsetSampling, Subregion, TransferFunc, Turbulence,
};

fn input_sel(b: u8) -> FilterInput {
    match b % 6 {
        0 => FilterInput::Auto,
        1 => FilterInput::SourceGraphic,
        2 => FilterInput::SourceAlpha,
        3 => FilterInput::Reference("a".into()),
        4 => FilterInput::Reference("b".into()),
        // Dangling on purpose: no step ever names "ghost".
        _ => FilterInput::Reference("ghost".into()),
    }
}

fn f32_from(b0: u8, b1: u8, b2: u8, b3: u8) -> f32 {
    f32::from_le_bytes([b0, b1, b2, b3])
}

/// Map raw f32 bits into a finite clamped range (NaN / ±inf → 0.0) for
/// parameters whose functions document a panics-on-NaN contract.
fn finite_clamped(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        0.0
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let w = (data[0] % 24 + 1) as u32;
    let h = (data[1] % 24 + 1) as u32;
    let space = match data[2] % 3 {
        0 => FilterColorSpace::LinearRgb,
        1 => FilterColorSpace::Srgb,
        _ => FilterColorSpace::Auto,
    };

    let mut graph = FilterGraph::with_color_space(space);
    for rec in data[3..].chunks_exact(12).take(12) {
        let p = &rec[5..12];
        let primitive = match rec[0] % 12 {
            0 => FilterPrimitive::Flood {
                color: (p[0], p[1], p[2]),
                opacity: p[3] as f32 / 255.0,
            },
            1 => FilterPrimitive::GaussianBlur {
                // Raw f32 bits: NaN / inf / negative stdDeviation
                // would violate the documented panic contract of
                // gaussian_blur_edge, so magnitude-map instead; the
                // prefix-sum pass makes any finite width cheap.
                std_x: p[0] as f32 * 2.0,
                std_y: p[1] as f32 * 2.0,
                edge: match p[2] % 3 {
                    0 => ConvolveEdgeMode::Duplicate,
                    1 => ConvolveEdgeMode::Wrap,
                    _ => ConvolveEdgeMode::None,
                },
            },
            2 => FilterPrimitive::Offset {
                dx: finite_clamped(f32_from(p[0], p[1], p[2], p[3]), -1e6, 1e6),
                dy: finite_clamped(f32_from(p[3], p[2], p[1], p[0]), -1e6, 1e6),
                sampling: OffsetSampling::default(),
            },
            3 => FilterPrimitive::ColorMatrix {
                op: match p[0] % 4 {
                    0 => ColorMatrixOp::Matrix,
                    1 => ColorMatrixOp::Saturate(p[1] as f32 / 64.0),
                    2 => ColorMatrixOp::HueRotate(p[1] as f32 * 47.0),
                    _ => ColorMatrixOp::LuminanceToAlpha,
                },
                matrix: ColorMatrix::identity(),
            },
            4 => FilterPrimitive::Composite {
                op: match p[0] % 6 {
                    0 => CompositeOp::Over,
                    1 => CompositeOp::In,
                    2 => CompositeOp::Out,
                    3 => CompositeOp::Atop,
                    4 => CompositeOp::Xor,
                    _ => CompositeOp::Arithmetic {
                        k1: p[1] as f32 / 32.0 - 4.0,
                        k2: p[2] as f32 / 32.0 - 4.0,
                        k3: p[3] as f32 / 32.0 - 4.0,
                        k4: p[4] as f32 / 32.0 - 4.0,
                    },
                },
            },
            5 => FilterPrimitive::Blend {
                mode: match p[0] % 16 {
                    0 => BlendFilterMode::Normal,
                    1 => BlendFilterMode::Multiply,
                    2 => BlendFilterMode::Screen,
                    3 => BlendFilterMode::Darken,
                    4 => BlendFilterMode::Lighten,
                    5 => BlendFilterMode::Overlay,
                    6 => BlendFilterMode::ColorDodge,
                    7 => BlendFilterMode::ColorBurn,
                    8 => BlendFilterMode::HardLight,
                    9 => BlendFilterMode::SoftLight,
                    10 => BlendFilterMode::Difference,
                    11 => BlendFilterMode::Exclusion,
                    12 => BlendFilterMode::Hue,
                    13 => BlendFilterMode::Saturation,
                    14 => BlendFilterMode::Color,
                    _ => BlendFilterMode::Luminosity,
                },
            },
            6 => FilterPrimitive::Morphology {
                // Cost is O(radius) per pixel; clamp to keep the
                // budget on graph logic.
                radius_x: (p[0] % 32) as u32,
                radius_y: (p[1] % 32) as u32,
                op: if p[2] % 2 == 0 {
                    MorphologyOp::Erode
                } else {
                    MorphologyOp::Dilate
                },
            },
            7 => FilterPrimitive::ComponentTransfer(
                ComponentTransfer::identity()
                    .with_r(TransferFunc::Linear {
                        slope: p[0] as f32 / 16.0 - 8.0,
                        intercept: p[1] as f32 / 128.0 - 1.0,
                    })
                    .with_a(TransferFunc::Table(vec![
                        p[2] as f32 / 255.0,
                        p[3] as f32 / 255.0,
                        p[4] as f32 / 255.0,
                    ])),
            ),
            8 => FilterPrimitive::DisplacementMap {
                scale: finite_clamped(f32_from(p[0], p[1], p[2], p[3]), -1e4, 1e4),
                x_channel: match p[4] % 4 {
                    0 => DisplacementChannel::R,
                    1 => DisplacementChannel::G,
                    2 => DisplacementChannel::B,
                    _ => DisplacementChannel::A,
                },
                y_channel: match p[5] % 4 {
                    0 => DisplacementChannel::R,
                    1 => DisplacementChannel::G,
                    2 => DisplacementChannel::B,
                    _ => DisplacementChannel::A,
                },
                sampling: DisplacementSampling::default(),
            },
            9 => FilterPrimitive::Turbulence(
                // Octave count is O(n) per pixel: clamp to 6.
                Turbulence::new(p[0] as f64 / 64.0, p[1] as f64 / 64.0)
                    .with_num_octaves((p[2] % 6 + 1) as u32)
                    .with_seed(p[3] as i64 * 131),
            ),
            10 => FilterPrimitive::Tile {
                tile: Subregion::new(
                    f32_from(p[0], 0, 0, p[1]),
                    f32_from(p[2], 0, 0, p[3]),
                    f32_from(p[4], 0, 0, p[5]),
                    f32_from(p[5], 0, 0, p[4]),
                ),
            },
            _ => {
                // Kernel is O(orderX·orderY) per pixel: clamp to 5×5.
                let ox = (p[0] % 5 + 1) as u32;
                let oy = (p[1] % 5 + 1) as u32;
                let kernel = (0..ox * oy)
                    .map(|i| (p[2] as f32 - i as f32) / 8.0)
                    .collect();
                FilterPrimitive::ConvolveMatrix(ConvolveMatrix::new(ox, oy, kernel))
            }
        };

        let mut step = FilterStep::new(primitive)
            .with_input(input_sel(rec[1]))
            .with_input2(input_sel(rec[2]));
        match rec[3] % 4 {
            0 => {}
            1 => step = step.with_result("a"),
            2 => step = step.with_result("b"),
            // Duplicate "a" on purpose — closest-preceding rule.
            _ => step = step.with_result("a"),
        }
        if rec[4] % 2 == 1 {
            // Raw f32 bit patterns: NaN / ±inf / negative extents all
            // reach the §9.4 disabled-region handling.
            step = step.with_subregion(Subregion::new(
                f32_from(p[0], p[1], 0, p[2]),
                f32_from(p[1], p[2], 0, p[3]),
                f32_from(p[2], p[3], 0, p[4]),
                f32_from(p[3], p[4], 0, p[5]),
            ));
        }
        graph.push(step);
    }

    let mut src = vec![0u8; (w * h * 4) as usize];
    for (i, b) in src.iter_mut().enumerate() {
        *b = (i * 37 + 11) as u8;
    }
    let out = graph.evaluate(&src, w, h);
    assert_eq!(out.len(), src.len());
});
