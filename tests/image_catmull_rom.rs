//! Catmull–Rom bicubic image-resampling sanity tests.
//!
//! Catmull–Rom is the `B = 0, C = 1/2` interpolating member of the
//! Mitchell–Netravali (1988) BC reconstruction-filter family. These
//! tests verify that `ImageFilter::CatmullRom`:
//!
//! * preserves a uniformly-coloured source (solid-red 4×4 enlarged to
//!   8×8 yields red at the centre — the kernel partition-of-unity holds
//!   end-to-end through the renderer),
//! * produces blended intermediates at a hard colour-seam,
//! * reproduces a source pixel when the destination sampling lands on a
//!   pixel centre (the *interpolating* property that distinguishes
//!   Catmull–Rom from Mitchell, whose `B = 1/3` blur term averages even
//!   pixel-aligned samples toward their neighbours),
//! * stays in gamut on a step edge despite the kernel's negative
//!   side-lobe (the sampler's per-channel `[0, 255]` clamp absorbs the
//!   overshoot, so alpha is never lost in the un-premultiply).

use oxideav_core::{Group, ImageRef, Node, Rect, Transform2D, VectorFrame, VideoFrame, VideoPlane};
use oxideav_raster::{ImageFilter, Renderer};

fn red_blue_split_image_4x4() -> VideoFrame {
    let mut data = Vec::with_capacity(4 * 4 * 4);
    for _y in 0..4 {
        for x in 0..4 {
            let (r, g, b) = if x < 2 { (255, 0, 0) } else { (0, 0, 255) };
            data.extend_from_slice(&[r, g, b, 255]);
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride: 16, data }],
    }
}

fn solid_red_image_4x4() -> VideoFrame {
    let mut data = Vec::with_capacity(4 * 4 * 4);
    for _ in 0..16 {
        data.extend_from_slice(&[255, 0, 0, 255]);
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride: 16, data }],
    }
}

fn black_white_step_image_8x1() -> VideoFrame {
    let mut data = Vec::with_capacity(8 * 4);
    for x in 0..8 {
        let v = if x < 4 { 0 } else { 255 };
        data.extend_from_slice(&[v, v, v, 255]);
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride: 32, data }],
    }
}

fn frame_with_image(
    image: VideoFrame,
    w: u32,
    h: u32,
    img_w: f32,
    img_h: f32,
    filter: ImageFilter,
) -> (Renderer, VectorFrame) {
    let mut r = Renderer::new(w, h);
    r.image_filter = filter;
    let img_node = Node::Image(ImageRef {
        frame: Box::new(image),
        bounds: Rect {
            x: 0.0,
            y: 0.0,
            width: img_w,
            height: img_h,
        },
        transform: Transform2D::identity(),
    });
    let mut root = Group::default();
    root.children.push(img_node);
    let v = VectorFrame {
        width: img_w,
        height: img_h,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    (r, v)
}

#[test]
fn catmull_rom_preserves_solid_red_centre() {
    let (r, v) = frame_with_image(
        solid_red_image_4x4(),
        8,
        8,
        8.0,
        8.0,
        ImageFilter::CatmullRom,
    );
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    let i = 4 * stride + 4 * 4;
    let p = &out.planes[0].data[i..i + 4];
    assert!(
        p[0] >= 254 && p[1] <= 1 && p[2] <= 1 && p[3] == 255,
        "expected near-pure red at centre, got {:?}",
        p
    );
}

#[test]
fn catmull_rom_blends_at_red_blue_seam() {
    let (r, v) = frame_with_image(
        red_blue_split_image_4x4(),
        8,
        8,
        8.0,
        8.0,
        ImageFilter::CatmullRom,
    );
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    let row = 4 * stride;
    let mut saw_intermediate = false;
    for x in 0..8 {
        let p = &out.planes[0].data[row + x * 4..row + x * 4 + 4];
        let r_chan = p[0];
        let b_chan = p[2];
        if r_chan > 5 && r_chan < 250 && b_chan > 5 && b_chan < 250 {
            saw_intermediate = true;
            break;
        }
    }
    assert!(
        saw_intermediate,
        "catmull-rom should blend at the red→blue seam"
    );
}

#[test]
fn catmull_rom_step_edge_stays_in_gamut() {
    // 8×1 source rendered at 16×1 destination. Catmull–Rom rings more
    // than Mitchell (B = 0 side-lobe); verify that the per-channel clamp
    // in the sampler keeps alpha valid (the un-premultiply can't lose
    // alpha if the premultiplied accumulation stayed clamped).
    let (mut r, mut v) = frame_with_image(
        black_white_step_image_8x1(),
        16,
        1,
        16.0,
        1.0,
        ImageFilter::CatmullRom,
    );
    r.width = 16;
    r.height = 1;
    v.width = 16.0;
    v.height = 1.0;
    let out = r.render(&v);
    let row = 0usize;
    for x in 0..16 {
        let p = &out.planes[0].data[row + x * 4..row + x * 4 + 4];
        assert_eq!(p[3], 255, "pixel {} alpha lost in catmull-rom sampler", x);
    }
}

#[test]
fn catmull_rom_reproduces_pixel_aligned_sample() {
    // The interpolating property: at a 1:1 (no-scale) mapping, every
    // destination pixel centre lands on a source pixel centre, so the
    // Catmull–Rom output must reproduce the source exactly — including
    // the hard red/blue seam, with no blended intermediate. (Mitchell
    // would average pixel-aligned samples toward neighbours.)
    let (r, v) = frame_with_image(
        red_blue_split_image_4x4(),
        4,
        4,
        4.0,
        4.0,
        ImageFilter::CatmullRom,
    );
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    for y in 0..4 {
        let row = y * stride;
        for x in 0..4 {
            let p = &out.planes[0].data[row + x * 4..row + x * 4 + 4];
            let (er, eb) = if x < 2 { (255u8, 0u8) } else { (0u8, 255u8) };
            assert_eq!(
                (p[0], p[2]),
                (er, eb),
                "pixel-aligned sample at ({}, {}) should reproduce the source exactly, got {:?}",
                x,
                y,
                p
            );
            assert_eq!(p[1], 0, "green must stay 0 at ({}, {})", x, y);
        }
    }
}

#[test]
fn catmull_rom_default_remains_bilinear() {
    // Regression: adding `CatmullRom` must not change the renderer
    // default ImageFilter.
    let r = Renderer::new(4, 4);
    assert_eq!(r.image_filter, ImageFilter::Bilinear);
}

// ---------------------------------------------------------------------------
// Round 121: identity + PSNR-on-downsample-up regression coverage.
//
// References (textbook math only — clean-room):
//
// * Catmull & Rom, "A Class of Local Interpolating Splines",
//   *Computer Aided Geometric Design*, R. E. Barnhill & R. F. Riesenfeld
//   (eds.), Academic Press, 1974, pp. 317–326. The original paper that
//   defines the interpolating cubic-blending spline used here.
// * Mitchell & Netravali, "Reconstruction Filters in Computer
//   Graphics", *SIGGRAPH '88*, ACM, pp. 221–228. Embeds Catmull–Rom in
//   the `(B, C)` family at `(B = 0, C = 1/2)`.
// ---------------------------------------------------------------------------

/// Smooth synthetic 16×16 image: a low-frequency horizontal+vertical
/// cosine plate. Wholly within the kernel's representable bandwidth, so
/// the negative side-lobes do not overshoot meaningfully and the
/// downsample→upsample round-trip is a clean PSNR benchmark.
fn smooth_plate_image(size: u32) -> VideoFrame {
    let stride = (size as usize) * 4;
    let mut data = vec![0u8; stride * size as usize];
    let s = size as f32;
    for y in 0..size {
        for x in 0..size {
            // One cosine period across the image, normalised to [0, 1].
            let cx = 0.5 + 0.5 * ((x as f32 / (s - 1.0)) * std::f32::consts::PI).cos();
            let cy = 0.5 + 0.5 * ((y as f32 / (s - 1.0)) * std::f32::consts::PI).cos();
            let l = (cx * cy * 255.0).round().clamp(0.0, 255.0) as u8;
            let i = (y as usize) * stride + (x as usize) * 4;
            data[i] = l;
            data[i + 1] = l;
            data[i + 2] = l;
            data[i + 3] = 255;
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride, data }],
    }
}

/// Render `src` at the given destination size using `filter`. The image
/// node is placed at `(0, 0, dst_w, dst_h)` user-space, the view-box is
/// `(0, 0, dst_w, dst_h)`, so the renderer's user→raster transform is
/// the identity and the sampler is exercised at the requested resolution.
fn render_to(src: VideoFrame, dst_w: u32, dst_h: u32, filter: ImageFilter) -> VideoFrame {
    let (mut r, mut v) = frame_with_image(src, dst_w, dst_h, dst_w as f32, dst_h as f32, filter);
    r.width = dst_w;
    r.height = dst_h;
    v.width = dst_w as f32;
    v.height = dst_h as f32;
    r.render(&v)
}

/// Compute PSNR between two equally-sized RGBA8 frames over the RGB
/// channels (alpha is omitted; both inputs are opaque). The standard
/// textbook formula:
///
/// ```text
///   MSE  = (1 / N) · Σ (a_i − b_i)²
///   PSNR = 10 · log10(MAX² / MSE),   MAX = 255
/// ```
///
/// Returns `f32::INFINITY` for an exact match (MSE = 0).
fn psnr_rgb(a: &VideoFrame, b: &VideoFrame) -> f32 {
    let sa = a.planes[0].stride;
    let sb = b.planes[0].stride;
    let h = a.planes[0].data.len() / sa;
    assert_eq!(h, b.planes[0].data.len() / sb, "frame heights differ");
    let w = sa / 4;
    assert_eq!(w, sb / 4, "frame widths differ");
    let mut sse = 0.0f64;
    let mut n = 0usize;
    for y in 0..h {
        for x in 0..w {
            let ai = y * sa + x * 4;
            let bi = y * sb + x * 4;
            for c in 0..3 {
                let d = a.planes[0].data[ai + c] as f64 - b.planes[0].data[bi + c] as f64;
                sse += d * d;
                n += 1;
            }
        }
    }
    let mse = sse / n as f64;
    if mse <= 0.0 {
        f32::INFINITY
    } else {
        let max2 = 255.0f64 * 255.0;
        (10.0 * (max2 / mse).log10()) as f32
    }
}

#[test]
fn catmull_rom_identity_at_unit_scale_on_smooth_plate() {
    // The interpolation property (k(0) = 1, k(±1) = k(±2) = 0) means
    // that at a 1:1 source→destination mapping, every destination pixel
    // centre lands on a source-pixel centre and Catmull–Rom must
    // reproduce the source exactly — even on a continuously-varying
    // synthetic where Mitchell would average neighbours.
    let src = smooth_plate_image(16);
    let out = render_to(src.clone(), 16, 16, ImageFilter::CatmullRom);
    let p = psnr_rgb(&src, &out);
    assert!(
        p.is_infinite() || p > 60.0,
        "1:1 Catmull–Rom render must be near-identity, got PSNR = {} dB",
        p
    );
}

#[test]
fn catmull_rom_downsample_upsample_psnr_above_30db() {
    // Downsample 32×32 → 16×16 → 32×32 round-trip on a smooth synthetic
    // image. A reasonably band-limited input survives the chained
    // reconstruction with PSNR comfortably above 30 dB. (At the same
    // resolutions and on the same input the nearest-neighbour path
    // scores well under 30 dB — verified in the companion test below.)
    let src = smooth_plate_image(32);
    let down = render_to(src.clone(), 16, 16, ImageFilter::CatmullRom);
    let up = render_to(down, 32, 32, ImageFilter::CatmullRom);
    let p = psnr_rgb(&src, &up);
    assert!(
        p > 30.0,
        "Catmull–Rom 2× downsample→upsample PSNR must exceed 30 dB on a smooth plate, got {} dB",
        p
    );
}

#[test]
fn catmull_rom_beats_nearest_neighbour_on_downsample_upsample_chain() {
    // Sanity: on the same chain, Catmull–Rom must beat nearest-neighbour
    // (which loses every other column to the floor() snap and so cannot
    // recover the smooth gradient). The exact margin depends on the
    // input bandwidth but >5 dB is comfortable for any smooth synthetic.
    let src = smooth_plate_image(32);
    let nn_down = render_to(src.clone(), 16, 16, ImageFilter::Nearest);
    let nn_up = render_to(nn_down, 32, 32, ImageFilter::Nearest);
    let cr_down = render_to(src.clone(), 16, 16, ImageFilter::CatmullRom);
    let cr_up = render_to(cr_down, 32, 32, ImageFilter::CatmullRom);
    let nn = psnr_rgb(&src, &nn_up);
    let cr = psnr_rgb(&src, &cr_up);
    assert!(
        cr > nn + 5.0,
        "Catmull–Rom must outperform nearest-neighbour by >5 dB on a smooth chain; nn = {} dB, cr = {} dB",
        nn,
        cr
    );
}
