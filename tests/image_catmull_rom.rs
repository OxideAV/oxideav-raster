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
