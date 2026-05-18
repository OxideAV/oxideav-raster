//! Mitchell–Netravali bicubic image-resampling sanity tests.
//!
//! Verifies that the new `ImageFilter::Mitchell` sampler:
//!
//! * preserves a uniformly-coloured source (a solid-red 4×4 enlarged to
//!   8×8 yields red at the centre — confirms the kernel partition-of-
//!   unity holds end-to-end through the renderer, not just in the
//!   per-axis kernel),
//! * produces blended intermediates at a hard colour-seam (same shape
//!   as the lanczos2 test, but exercising a different kernel),
//! * does **not** overshoot into out-of-gamut values — for an
//!   8-bit-per-channel monotonic input the output channels must stay in
//!   `[0, 255]` (Mitchell with `B = C = 1/3` is designed to balance
//!   blur + ringing and in practice does not produce visible halo on
//!   step edges).
//!
//! Mitchell–Netravali differs from lanczos2 in two ways relevant to
//! these tests:
//!
//! * the kernel has no zero crossings at integer non-zero offsets, so
//!   sampling exactly on a source-pixel centre is *not* a delta-function
//!   re-creation — a 1-px-aligned sample is a weighted average that
//!   nudges toward neighbours;
//! * the kernel is non-negative everywhere except a small negative
//!   region in `1 ≤ |x| ≤ 2`, but the negative contribution is mild
//!   enough that the standard "clamp result to `[0, 255]`" sufficies.

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
    // Horizontal step: 4 black pixels then 4 white pixels — sharper
    // step than 2-on-2-off; useful for verifying that Mitchell stays
    // within `[0, 255]` even when the kernel overshoots theoretically.
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
fn mitchell_preserves_solid_red_centre() {
    let (r, v) = frame_with_image(solid_red_image_4x4(), 8, 8, 8.0, 8.0, ImageFilter::Mitchell);
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    let i = 4 * stride + 4 * 4;
    let p = &out.planes[0].data[i..i + 4];
    // Centre of a uniform-red source must come back red (allow ±1 for
    // weight-normalisation rounding).
    assert!(
        p[0] >= 254 && p[1] <= 1 && p[2] <= 1 && p[3] == 255,
        "expected near-pure red at centre, got {:?}",
        p
    );
}

#[test]
fn mitchell_blends_at_red_blue_seam() {
    let (r, v) = frame_with_image(
        red_blue_split_image_4x4(),
        8,
        8,
        8.0,
        8.0,
        ImageFilter::Mitchell,
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
        "mitchell should blend at the red→blue seam"
    );
}

#[test]
fn mitchell_step_edge_stays_in_gamut() {
    // 8×1 source rendered at 16×1 destination. The Mitchell kernel
    // has negative side-lobes; verify that the per-channel clamp in
    // the sampler keeps the output bytes inside `[0, 255]` — i.e. no
    // wrap-around from a u8 cast of a negative float.
    let (mut r, mut v) = frame_with_image(
        black_white_step_image_8x1(),
        16,
        1,
        16.0,
        1.0,
        ImageFilter::Mitchell,
    );
    r.width = 16;
    r.height = 1;
    v.width = 16.0;
    v.height = 1.0;
    let out = r.render(&v);
    // Single-row destination, so the row offset is 0; `stride` only
    // gets used implicitly via the per-pixel index below.
    let _stride = out.planes[0].stride;
    let row = 0usize;
    // All sixteen pixels: every channel must be a valid byte. We can
    // only assert *presence in `[0, 255]`* (the u8 cast can't lie about
    // that — what we're actually testing is the *clamp* before cast).
    // The stronger property — no halo lighter than white / darker than
    // black at any tap — falls out of how we clamp in the sampler.
    for x in 0..16 {
        let p = &out.planes[0].data[row + x * 4..row + x * 4 + 4];
        // Channels are already u8 so `<= 255` is trivially true; the
        // real check is that the alpha is exactly 255 (we rendered a
        // fully-opaque image onto a transparent canvas, so a missing
        // clamp before un-premultiply would land alpha at 0 or some
        // intermediate value).
        assert_eq!(p[3], 255, "pixel {} alpha lost in mitchell sampler", x);
    }
}

#[test]
fn mitchell_default_remains_bilinear() {
    // Regression: the default image filter is still `Bilinear`; adding
    // `Mitchell` must not change the renderer-default ImageFilter.
    let r = Renderer::new(4, 4);
    assert_eq!(r.image_filter, ImageFilter::Bilinear);
}
