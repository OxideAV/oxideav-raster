//! Cubic B-spline image-resampling sanity tests.
//!
//! The cubic B-spline is the `B = 1, C = 0` *approximating* member of the
//! Mitchell–Netravali (1988) BC reconstruction-filter family. These tests
//! verify that `ImageFilter::BSpline`:
//!
//! * preserves a uniformly-coloured source (solid-red 4×4 enlarged to 8×8
//!   yields red at the centre — the kernel partition-of-unity holds
//!   end-to-end through the renderer, even though the kernel does not
//!   interpolate),
//! * produces blended intermediates at a hard colour-seam (it blurs, like
//!   every BC cubic with a wide support),
//! * *smooths* rather than reproduces a pixel-aligned hard seam — the
//!   property that distinguishes the B-spline from Catmull–Rom (which is
//!   interpolating). At a 1:1 mapping the B-spline still pulls each pixel
//!   toward its neighbours, so the red/blue seam shows a blended column,
//! * stays trivially in gamut on a step edge — the B-spline has no
//!   negative side-lobe, so there is no overshoot to clamp and alpha is
//!   never lost.

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
fn b_spline_preserves_solid_red_centre() {
    let (r, v) = frame_with_image(solid_red_image_4x4(), 8, 8, 8.0, 8.0, ImageFilter::BSpline);
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
fn b_spline_blends_at_red_blue_seam() {
    let (r, v) = frame_with_image(
        red_blue_split_image_4x4(),
        8,
        8,
        8.0,
        8.0,
        ImageFilter::BSpline,
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
        "B-spline should blend at the red→blue seam"
    );
}

#[test]
fn b_spline_step_edge_stays_in_gamut() {
    // 8×1 source rendered at 16×1 destination. The B-spline has no
    // negative side-lobe, so there is no overshoot at all; alpha must
    // stay 255 across the whole row (a fortiori, since nothing rings).
    let (mut r, mut v) = frame_with_image(
        black_white_step_image_8x1(),
        16,
        1,
        16.0,
        1.0,
        ImageFilter::BSpline,
    );
    r.width = 16;
    r.height = 1;
    v.width = 16.0;
    v.height = 1.0;
    let out = r.render(&v);
    let row = 0usize;
    for x in 0..16 {
        let p = &out.planes[0].data[row + x * 4..row + x * 4 + 4];
        assert_eq!(p[3], 255, "pixel {} alpha lost in b-spline sampler", x);
    }
}

#[test]
fn b_spline_smooths_pixel_aligned_seam() {
    // The approximating property: even at a 1:1 (no-scale) mapping, where
    // every destination pixel centre lands on a source pixel centre, the
    // B-spline does NOT reproduce the source — it blends each pixel with
    // its neighbours (k(0) = 4/6, k(±1) = 1/6). So the hard red/blue seam
    // must show at least one column with a blended (purple) pixel.
    // (Catmull–Rom, the interpolating member, would reproduce the seam
    // exactly with no blended column.)
    let (r, v) = frame_with_image(
        red_blue_split_image_4x4(),
        4,
        4,
        4.0,
        4.0,
        ImageFilter::BSpline,
    );
    let out = r.render(&v);
    let row = 0usize;
    let mut saw_blend = false;
    for x in 0..4 {
        let p = &out.planes[0].data[row + x * 4..row + x * 4 + 4];
        if p[0] > 5 && p[0] < 250 && p[2] > 5 && p[2] < 250 {
            saw_blend = true;
            break;
        }
    }
    assert!(
        saw_blend,
        "B-spline must smooth (not reproduce) the pixel-aligned seam"
    );
}

#[test]
fn b_spline_default_remains_bilinear() {
    // Regression: adding `BSpline` must not change the renderer default
    // ImageFilter.
    let r = Renderer::new(4, 4);
    assert_eq!(r.image_filter, ImageFilter::Bilinear);
}
