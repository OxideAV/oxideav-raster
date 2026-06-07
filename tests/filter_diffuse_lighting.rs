//! Integration coverage for [`oxideav_raster::diffuse_lighting`] —
//! the SVG 1.1 §15.14 `<feDiffuseLighting>` filter primitive.
//!
//! The unit tests inside `src/filter.rs::diffuse_lighting_tests`
//! cover the per-pixel arithmetic (Sobel kernels, distant / point /
//! spot light sources, cone cut-off, specular focus exponent,
//! parameter panics). This file is the consumer-facing API exercise
//! — driving the public re-exports through the cases a real
//! `<feDiffuseLighting>` element would target inside a `<filter>`.

use oxideav_core::Rgba;
use oxideav_raster::{diffuse_lighting, diffuse_lighting_pixels, DiffuseLighting, LightSource};

/// Build a packed-RGBA buffer of `w·h` pixels coloured by `f`.
fn build<F: FnMut(u32, u32) -> Rgba>(w: u32, h: u32, mut f: F) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let c = f(x, y);
            v.extend_from_slice(&[c.r, c.g, c.b, c.a]);
        }
    }
    v
}

#[test]
fn default_params_match_spec_defaults() {
    // §15.14 attribute table defaults: surfaceScale = 1,
    // diffuseConstant = 1. §15.8.2 azimuth = 0, elevation = 0.
    // `kernelUnitLength` default per §15.14 is "very small deltas
    // … one pixel in the intermediate image" — we encode that as
    // (1, 1). `lighting-color` initial value is `white` per the
    // §15.7.2 presentation-property table; encoded as (255, 255,
    // 255). The `Default` impl needs to surface all five.
    let p = DiffuseLighting::default();
    assert_eq!(p.surface_scale, 1.0);
    assert_eq!(p.diffuse_constant, 1.0);
    assert_eq!(p.kernel_unit_length, (1.0, 1.0));
    assert_eq!(p.light_color, [255, 255, 255]);
    matches!(
        p.light_source,
        LightSource::Distant {
            azimuth_deg: 0.0,
            elevation_deg: 0.0,
        }
    );
}

#[test]
fn flat_alpha_gives_constant_output_for_distant_light() {
    // §15.14: a flat input alpha has zero surface gradient
    // everywhere, so `N = (0, 0, 1)` at every pixel. A distant light
    // produces a positional-invariant `L`, so the lit output is
    // identical at every pixel. Pick `elevation = 90°` so `L·N = 1`
    // → emit the full light colour.
    let src = build(10, 7, |_, _| Rgba::new(0, 0, 0, 180));
    let params = DiffuseLighting {
        surface_scale: 3.0,
        diffuse_constant: 1.0,
        kernel_unit_length: (1.0, 1.0),
        light_color: [128, 64, 32],
        light_source: LightSource::Distant {
            azimuth_deg: 17.0, // any azimuth works at 90° elevation.
            elevation_deg: 90.0,
        },
    };
    let out = diffuse_lighting(&src, 10, 7, &params);
    let first = &out[..4];
    for chunk in out.chunks_exact(4) {
        // All pixels equal up to one ULP from the per-pixel quantise
        // round-trip.
        assert!((chunk[0] as i32 - first[0] as i32).abs() <= 1);
        assert!((chunk[1] as i32 - first[1] as i32).abs() <= 1);
        assert!((chunk[2] as i32 - first[2] as i32).abs() <= 1);
        assert_eq!(chunk[3], 255);
    }
    // First pixel ≈ light_color.
    assert!((first[0] as i32 - 128).abs() <= 1);
    assert!((first[1] as i32 - 64).abs() <= 1);
    assert!((first[2] as i32 - 32).abs() <= 1);
}

#[test]
fn light_color_scales_output_linearly() {
    // §15.14 `D = kd · (N·L) · Light`: doubling `light_color` (when
    // possible without saturation) should double the lit output, up
    // to `u8` quantisation.
    let src = build(4, 4, |_, _| Rgba::new(0, 0, 0, 100));
    let make = |color: [u8; 3]| DiffuseLighting {
        surface_scale: 1.0,
        diffuse_constant: 1.0,
        kernel_unit_length: (1.0, 1.0),
        light_color: color,
        light_source: LightSource::Distant {
            azimuth_deg: 0.0,
            elevation_deg: 90.0, // L = (0, 0, 1) → N·L = 1.
        },
    };
    let dim = diffuse_lighting(&src, 4, 4, &make([40, 60, 80]));
    let bright = diffuse_lighting(&src, 4, 4, &make([80, 120, 160]));
    // 2× light colour ⇒ 2× output (within rounding).
    for (d, b) in dim.chunks_exact(4).zip(bright.chunks_exact(4)) {
        assert!((b[0] as i32 - 2 * d[0] as i32).abs() <= 2);
        assert!((b[1] as i32 - 2 * d[1] as i32).abs() <= 2);
        assert!((b[2] as i32 - 2 * d[2] as i32).abs() <= 2);
        assert_eq!(d[3], 255);
        assert_eq!(b[3], 255);
    }
}

#[test]
fn diffuse_constant_scales_output_linearly() {
    // §15.14 `D = kd · (N·L) · Light`: `kd = 0.5` halves the lit
    // output; `kd = 1.0` is the reference.
    let src = build(4, 4, |_, _| Rgba::new(0, 0, 0, 100));
    let make = |kd: f32| DiffuseLighting {
        surface_scale: 1.0,
        diffuse_constant: kd,
        kernel_unit_length: (1.0, 1.0),
        light_color: [200, 200, 200],
        light_source: LightSource::Distant {
            azimuth_deg: 0.0,
            elevation_deg: 90.0,
        },
    };
    let half = diffuse_lighting(&src, 4, 4, &make(0.5));
    let full = diffuse_lighting(&src, 4, 4, &make(1.0));
    for (h, f) in half.chunks_exact(4).zip(full.chunks_exact(4)) {
        assert!((2 * h[0] as i32 - f[0] as i32).abs() <= 2);
        assert!((2 * h[1] as i32 - f[1] as i32).abs() <= 2);
        assert!((2 * h[2] as i32 - f[2] as i32).abs() <= 2);
    }
}

#[test]
fn distant_light_results_independent_of_pixel_position() {
    // §15.8.2 distant-light unit vector has no positional component
    // — every pixel's `L` is the same. On a flat input, then, the
    // output at `(0, 0)` matches the output at `(w-1, h-1)`.
    let src = build(6, 6, |_, _| Rgba::new(0, 0, 0, 200));
    let params = DiffuseLighting {
        surface_scale: 2.0,
        diffuse_constant: 1.0,
        kernel_unit_length: (1.0, 1.0),
        light_color: [255, 255, 255],
        light_source: LightSource::Distant {
            azimuth_deg: 30.0,
            elevation_deg: 60.0,
        },
    };
    let out = diffuse_lighting(&src, 6, 6, &params);
    let first = &out[0..4];
    let last_idx = (5 * 6 + 5) * 4;
    let last = &out[last_idx..last_idx + 4];
    // §15.14 explicitly notes that corner kernels differ from
    // interior ones, but on a *flat* input the convolution sum is
    // zero for every variant — they all collapse to `N = (0, 0, 1)`.
    // The corner output therefore matches the interior output.
    for c in 0..3 {
        assert!((first[c] as i32 - last[c] as i32).abs() <= 1);
    }
}

#[test]
fn typed_pixel_wrapper_round_trips_against_byte_path() {
    // Cross-API equivalence: `diffuse_lighting_pixels` is the typed
    // wrapper around `diffuse_lighting` and must emit bit-exact the
    // same bytes when its input/output buffers are repackaged.
    let w = 7u32;
    let h = 5u32;
    let bytes = build(w, h, |x, y| {
        Rgba::new(0, 0, 0, ((x + y) * 20).min(255) as u8)
    });
    let pixels: Vec<Rgba> = bytes
        .chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect();
    let params = DiffuseLighting {
        surface_scale: 4.0,
        diffuse_constant: 0.75,
        kernel_unit_length: (1.0, 1.0),
        light_color: [200, 100, 50],
        light_source: LightSource::Spot {
            x: 3.0,
            y: 2.0,
            z: 6.0,
            points_at_x: 3.0,
            points_at_y: 2.0,
            points_at_z: 0.0,
            specular_exponent: 2.5,
            limiting_cone_angle_deg: Some(70.0),
        },
    };
    let bytes_out = diffuse_lighting(&bytes, w, h, &params);
    let pixels_out = diffuse_lighting_pixels(&pixels, w, h, &params);
    let bytes_from_typed: Vec<u8> = pixels_out
        .iter()
        .flat_map(|p| [p.r, p.g, p.b, p.a])
        .collect();
    assert_eq!(bytes_out, bytes_from_typed);
}

#[test]
fn output_alpha_is_one_everywhere() {
    // §15.14: "the resulting image is an RGBA opaque image based on
    // the light color with alpha = 1.0 everywhere". This must hold
    // regardless of the input alpha (used only as a height field),
    // the light source type, or the lighting parameters.
    let src = build(5, 5, |x, y| {
        Rgba::new((x * 50) as u8, (y * 50) as u8, 100, (x * y) as u8)
    });
    for src_kind in [
        LightSource::Distant {
            azimuth_deg: 45.0,
            elevation_deg: 30.0,
        },
        LightSource::Point {
            x: 2.0,
            y: 2.0,
            z: 5.0,
        },
        LightSource::Spot {
            x: 2.0,
            y: 2.0,
            z: 5.0,
            points_at_x: 2.0,
            points_at_y: 2.0,
            points_at_z: 0.0,
            specular_exponent: 1.0,
            limiting_cone_angle_deg: None,
        },
    ] {
        let params = DiffuseLighting {
            surface_scale: 1.0,
            diffuse_constant: 1.0,
            kernel_unit_length: (1.0, 1.0),
            light_color: [255, 200, 100],
            light_source: src_kind,
        };
        let out = diffuse_lighting(&src, 5, 5, &params);
        for px in out.chunks_exact(4) {
            assert_eq!(
                px[3], 255,
                "§15.14 demands alpha = 1.0 for every pixel and every light source"
            );
        }
    }
}

#[test]
fn spot_light_with_no_cone_lights_full_hemisphere() {
    // §15.8.4 "if no value is supplied, there is no limiting cone".
    // A spot light pointing straight down with no cone cap and a
    // specular_exponent of 1 should illuminate every pixel under the
    // hemisphere with positive intensity (modulated by the
    // `(−L·S)^1` factor).
    let w = 7u32;
    let h = 7u32;
    let src = build(w, h, |_, _| Rgba::new(0, 0, 0, 100));
    let params = DiffuseLighting {
        surface_scale: 1.0,
        diffuse_constant: 1.0,
        kernel_unit_length: (1.0, 1.0),
        light_color: [255, 255, 255],
        light_source: LightSource::Spot {
            x: 3.0,
            y: 3.0,
            z: 5.0,
            points_at_x: 3.0,
            points_at_y: 3.0,
            points_at_z: 0.0,
            specular_exponent: 1.0,
            limiting_cone_angle_deg: None,
        },
    };
    let out = diffuse_lighting(&src, w, h, &params);
    // All in-hemisphere pixels (which is all of them on a flat
    // surface with z = 5) should be lit. With `z = 5` and the surface
    // at `Z = 0`, every pixel has `−L · S > 0` so the cone cap
    // doesn't matter and intensity is positive.
    for px in out.chunks_exact(4) {
        assert!(
            px[0] > 0,
            "in-cone spot should produce R > 0, got {}",
            px[0]
        );
        assert_eq!(px[3], 255);
    }
}

#[test]
fn point_light_intensity_falls_off_with_distance_via_angle() {
    // §15.14: the point-light `L` vector is normalised, so there's
    // no `1/r²` falloff in the formula — the only thing that changes
    // with position is the angle between `N` and `L`. Place a point
    // light directly above the centre of a flat surface; the centre
    // pixel has `L = (0, 0, 1)` ⇒ `N·L = 1`, edge pixels have
    // `L = (±dx, ±dy, dz) / Norm` so `N·L = dz/Norm < 1`.
    let w = 9u32;
    let h = 9u32;
    let src = build(w, h, |_, _| Rgba::new(0, 0, 0, 100));
    let params = DiffuseLighting {
        surface_scale: 0.01, // ≈ flat normal regardless of alpha.
        diffuse_constant: 1.0,
        kernel_unit_length: (1.0, 1.0),
        light_color: [255, 255, 255],
        light_source: LightSource::Point {
            x: 4.0,
            y: 4.0,
            z: 5.0,
        },
    };
    let out = diffuse_lighting(&src, w, h, &params);
    let centre = (4 * w as usize + 4) * 4;
    let corner = 0usize;
    // Centre is `N·L ≈ 1` ⇒ output ≈ 255. Corner is closer to grazing.
    assert!(out[centre] > out[corner]);
    assert!(out[centre] >= 240);
}
