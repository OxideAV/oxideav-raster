//! Criterion benchmarks for the filter-primitive hot loops.
//!
//! Round 401 depth work: the crate has full Filter Effects 1 coverage
//! (all 17 primitives + the §9.2/§9.3 graph evaluator + the §5/§6.1
//! CSS parser), so per the workspace "saturated → fuzz/bench/profile"
//! memo this file wires up A/B-able baselines for the kernels the
//! evaluator dispatches into. Every scenario is self-contained (LCG
//! pixel soup, no fixtures).
//!
//!   - **gaussian_small_kernel_256**: `stdDeviation = 1.0` — the
//!     direct discrete-kernel branch (`s < 2`).
//!   - **gaussian_box_256**: `stdDeviation = 4.0` — the three-box
//!     §15.17 approximation.
//!   - **gaussian_box_huge_std_256**: `stdDeviation = 1e6` — pins the
//!     prefix-sum property that cost is independent of the derived
//!     window width (this configuration hung before the r401
//!     hardening).
//!   - **morphology_dilate_r15_256**: separable-rectangle §15.20
//!     dilation with a wide element.
//!   - **color_matrix_saturate_256** / **component_transfer_256** /
//!     **composite_arithmetic_256** / **blend_soft_light_256**: the
//!     per-pixel arithmetic kernels.
//!   - **turbulence_fractal_128**: §15.24 fractalNoise, 4 octaves.
//!   - **graph_chain_128**: the §9.3 example chain (feColorMatrix →
//!     feOffset → feGaussianBlur) through `FilterGraph::evaluate` in
//!     the linearRGB working space — measures evaluator overhead plus
//!     the one-pass linearise/re-encode envelope.
//!   - **css_parse_apply_64**: `parse_filter_value_list` +
//!     `apply_filter_functions` on a representative five-function
//!     `filter` value.
//!
//! Round 449 additions (previously unbenched primitive kernels):
//!
//!   - **convolve_matrix_3x3_256**: general §15.13 convolution with an
//!     edge-detect 3×3 kernel, `duplicate` edges.
//!   - **displacement_map_bilinear_256**: §15.15 channel-driven warp,
//!     bilinear reconstruction.
//!   - **diffuse_lighting_distant_256 / specular_lighting_point_256**:
//!     the §15.14 / §15.22 Phong lighting kernels (Sobel surface
//!     normals + per-pixel light vector).
//!   - **drop_shadow_256**: the §9.12 equivalent chain (alpha blur →
//!     offset → flood → composite-in → merge) end-to-end.
//!
//! Run with: `cargo bench -p oxideav-raster --bench filter`

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use oxideav_raster::{
    apply_filter_functions, blend_filter, color_matrix_op, component_transfer, composite_filter,
    convolve_matrix, diffuse_lighting, displacement_map, drop_shadow, gaussian_blur_edge,
    morphology, parse_filter_value_list, specular_lighting, turbulence_filter, BlendFilterMode,
    ColorMatrix, ColorMatrixOp, ComponentTransfer, CompositeOp, ConvolveEdgeMode, ConvolveMatrix,
    DiffuseLighting, DisplacementChannel, DisplacementSampling, FilterColorSpace, FilterGraph,
    FilterPrimitive, FilterStep, LightSource, MorphologyOp, OffsetSampling, SpecularLighting,
    TransferFunc, Turbulence,
};

fn lcg_image(width: u32, height: u32, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..width as usize * height as usize * 4)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        })
        .collect()
}

fn bench_filters(c: &mut Criterion) {
    let img = lcg_image(256, 256, 0x1234_5678);
    let img_b = lcg_image(256, 256, 0x9abc_def0);
    let small = lcg_image(128, 128, 0x0dd_ba11);

    c.bench_function("gaussian_small_kernel_256", |b| {
        b.iter(|| {
            gaussian_blur_edge(
                black_box(&img),
                256,
                256,
                1.0,
                1.0,
                ConvolveEdgeMode::Duplicate,
            )
        })
    });
    c.bench_function("gaussian_box_256", |b| {
        b.iter(|| {
            gaussian_blur_edge(
                black_box(&img),
                256,
                256,
                4.0,
                4.0,
                ConvolveEdgeMode::Duplicate,
            )
        })
    });
    c.bench_function("gaussian_box_huge_std_256", |b| {
        b.iter(|| {
            gaussian_blur_edge(
                black_box(&img),
                256,
                256,
                1e6,
                1e6,
                ConvolveEdgeMode::Duplicate,
            )
        })
    });
    c.bench_function("morphology_dilate_r15_256", |b| {
        b.iter(|| morphology(black_box(&img), 256, 256, 15, 15, MorphologyOp::Dilate))
    });
    c.bench_function("color_matrix_saturate_256", |b| {
        b.iter(|| {
            color_matrix_op(
                black_box(&img),
                256,
                256,
                ColorMatrixOp::Saturate(0.4),
                &ColorMatrix::identity(),
            )
        })
    });
    let ct = ComponentTransfer::identity()
        .with_r(TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 2.2,
            offset: 0.0,
        })
        .with_g(TransferFunc::Linear {
            slope: 1.2,
            intercept: -0.1,
        })
        .with_b(TransferFunc::Table(vec![0.0, 0.6, 0.4, 1.0]));
    c.bench_function("component_transfer_256", |b| {
        b.iter(|| component_transfer(black_box(&img), 256, 256, &ct))
    });
    c.bench_function("composite_arithmetic_256", |b| {
        b.iter(|| {
            composite_filter(
                black_box(&img),
                black_box(&img_b),
                256,
                256,
                CompositeOp::Arithmetic {
                    k1: 0.5,
                    k2: 0.3,
                    k3: 0.2,
                    k4: 0.0,
                },
            )
        })
    });
    c.bench_function("blend_soft_light_256", |b| {
        b.iter(|| {
            blend_filter(
                black_box(&img),
                black_box(&img_b),
                256,
                256,
                BlendFilterMode::SoftLight,
            )
        })
    });
    c.bench_function("turbulence_fractal_128", |b| {
        let t = Turbulence::new(0.05, 0.05).with_num_octaves(4).with_seed(7);
        b.iter(|| turbulence_filter(128, 128, black_box(&t)))
    });
    c.bench_function("graph_chain_128", |b| {
        let mut g = FilterGraph::with_color_space(FilterColorSpace::LinearRgb);
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
        b.iter(|| g.evaluate(black_box(&small), 128, 128))
    });
    #[rustfmt::skip]
    let edge_detect = ConvolveMatrix::new(3, 3, vec![
        -1.0, -1.0, -1.0,
        -1.0,  8.0, -1.0,
        -1.0, -1.0, -1.0,
    ]);
    c.bench_function("convolve_matrix_3x3_256", |b| {
        b.iter(|| convolve_matrix(black_box(&img), 256, 256, &edge_detect))
    });
    c.bench_function("displacement_map_bilinear_256", |b| {
        b.iter(|| {
            displacement_map(
                black_box(&img),
                black_box(&img_b),
                256,
                256,
                12.5,
                DisplacementChannel::R,
                DisplacementChannel::G,
                DisplacementSampling::Bilinear,
            )
        })
    });
    let diffuse = DiffuseLighting {
        surface_scale: 3.0,
        diffuse_constant: 1.0,
        kernel_unit_length: (1.0, 1.0),
        light_color: [255, 200, 150],
        light_source: LightSource::Distant {
            azimuth_deg: 45.0,
            elevation_deg: 30.0,
        },
    };
    c.bench_function("diffuse_lighting_distant_256", |b| {
        b.iter(|| diffuse_lighting(black_box(&img), 256, 256, &diffuse))
    });
    let specular = SpecularLighting {
        surface_scale: 3.0,
        specular_constant: 1.0,
        specular_exponent: 20.0,
        kernel_unit_length: (1.0, 1.0),
        light_color: [255, 255, 255],
        light_source: LightSource::Point {
            x: 64.0,
            y: 64.0,
            z: 40.0,
        },
    };
    c.bench_function("specular_lighting_point_256", |b| {
        b.iter(|| specular_lighting(black_box(&img), 256, 256, &specular))
    });
    c.bench_function("drop_shadow_256", |b| {
        b.iter(|| {
            drop_shadow(
                black_box(&img),
                256,
                256,
                3.0,
                3.0,
                4.0,
                4.0,
                0,
                0,
                0,
                0.8,
                OffsetSampling::Nearest,
            )
        })
    });
    c.bench_function("css_parse_apply_64", |b| {
        let src = lcg_image(64, 64, 0xf00d);
        let value = "grayscale(30%) blur(2px) hue-rotate(90deg) contrast(150%) \
                     drop-shadow(rgb(10, 20, 30) 2px 3px 1px)";
        b.iter(|| {
            let fns = parse_filter_value_list(black_box(value)).unwrap();
            apply_filter_functions(&fns, black_box(&src), 64, 64)
        })
    });
}

criterion_group!(benches, bench_filters);
criterion_main!(benches);
