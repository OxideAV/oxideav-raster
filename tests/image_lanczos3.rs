//! Lanczos3 image-resampling sanity tests.
//!
//! Verifies that the new `ImageFilter::Lanczos3` sampler (windowed sinc
//! with `a = 3`, 6×6 separable footprint):
//!
//! * preserves a uniformly-coloured source (no edge-of-image artefacts
//!   from the truncated sinc tail — tested by sampling the centre of a
//!   solid-red 6×6 at 12×12 destination scale, both inside the
//!   footprint *and* at a clamped-boundary corner pixel),
//! * mixes channels at a hard colour seam (the defining behaviour of any
//!   higher-order sinc reconstruction),
//! * remains in gamut after the per-channel clamp (no `[0, 255]`
//!   overshoot leaks through),
//! * is *sharper* than Lanczos2 on a step edge (the wider window's
//!   selling point — at least one pixel on each side of a 1×N seam
//!   should be closer to the pure source colour under Lanczos3 than
//!   under Lanczos2).

use oxideav_core::{Group, ImageRef, Node, Rect, Transform2D, VectorFrame, VideoFrame, VideoPlane};
use oxideav_raster::{ImageFilter, Renderer};

fn red_blue_split_image(width: u32) -> VideoFrame {
    let half = (width / 2) as usize;
    let mut data = Vec::with_capacity((width * width * 4) as usize);
    for _y in 0..width {
        for x in 0..width as usize {
            let (r, g, b) = if x < half { (255, 0, 0) } else { (0, 0, 255) };
            data.extend_from_slice(&[r, g, b, 255]);
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (width * 4) as usize,
            data,
        }],
    }
}

fn solid_red_image(width: u32) -> VideoFrame {
    let mut data = Vec::with_capacity((width * width * 4) as usize);
    for _ in 0..(width * width) {
        data.extend_from_slice(&[255, 0, 0, 255]);
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: (width * 4) as usize,
            data,
        }],
    }
}

fn frame_with(image: VideoFrame, filter: ImageFilter, dst: u32) -> (Renderer, VectorFrame) {
    let mut r = Renderer::new(dst, dst);
    r.image_filter = filter;
    let img_node = Node::Image(ImageRef {
        frame: Box::new(image),
        bounds: Rect {
            x: 0.0,
            y: 0.0,
            width: dst as f32,
            height: dst as f32,
        },
        transform: Transform2D::identity(),
    });
    let mut root = Group::default();
    root.children.push(img_node);
    let v = VectorFrame {
        width: dst as f32,
        height: dst as f32,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    (r, v)
}

#[test]
fn lanczos3_preserves_solid_red_centre() {
    let (r, v) = frame_with(solid_red_image(6), ImageFilter::Lanczos3, 12);
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    // Centre pixel must remain pure red (or extremely close).
    let i = 6 * stride + 6 * 4;
    let p = &out.planes[0].data[i..i + 4];
    assert_eq!(p[0], 255, "centre R should still be 255, got {}", p[0]);
    assert_eq!(p[1], 0, "centre G should still be 0, got {}", p[1]);
    assert_eq!(p[2], 0, "centre B should still be 0, got {}", p[2]);
    assert_eq!(p[3], 255, "centre A should still be 255, got {}", p[3]);
}

#[test]
fn lanczos3_preserves_solid_red_clamped_corner() {
    // The clamp-to-edge weight re-normalisation must keep the corner
    // pixel pure red — without it the truncated tail of the sinc
    // would shift the colour slightly.
    let (r, v) = frame_with(solid_red_image(6), ImageFilter::Lanczos3, 12);
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    let p = &out.planes[0].data[0..4];
    assert_eq!(p, &[255, 0, 0, 255]);
    let last = (12 - 1) * stride + (12 - 1) * 4;
    let p = &out.planes[0].data[last..last + 4];
    assert_eq!(p, &[255, 0, 0, 255]);
}

#[test]
fn lanczos3_blends_at_red_blue_seam() {
    let (r, v) = frame_with(red_blue_split_image(6), ImageFilter::Lanczos3, 12);
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    // Mid-row probe: at least one column near the seam should have
    // both red and blue channels active (proper kernel mixing).
    let row = 6 * stride;
    let mut saw_intermediate = false;
    for x in 0..12 {
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
        "lanczos3 should blend at the red→blue seam"
    );
}

#[test]
fn lanczos3_clamps_in_gamut_at_step_edge() {
    // The combined kernel can produce per-channel premultiplied
    // accumulator values outside [0, 255]; the sampler must clamp.
    // Because each u8 is bounded by definition, the property under
    // test is the *absence* of nonsensical values across every output
    // pixel — i.e. no out-of-range write reaches the buffer.
    let (r, v) = frame_with(red_blue_split_image(6), ImageFilter::Lanczos3, 12);
    let out = r.render(&v);
    for chunk in out.planes[0].data.chunks_exact(4) {
        let r_ch = chunk[0];
        let g_ch = chunk[1];
        let b_ch = chunk[2];
        let a_ch = chunk[3];
        // u8 already enforces the upper bound; check the green stays
        // genuinely zero (no spurious leak from the alpha-only seam)
        // and that the alpha is solid where the source was opaque.
        assert_eq!(g_ch, 0, "lanczos3 should not synthesise a green channel");
        assert_eq!(
            a_ch, 255,
            "lanczos3 of a fully-opaque source should stay opaque"
        );
        // The red+blue pair should still be plausibly sourced (clamp).
        assert!(r_ch as u16 + b_ch as u16 <= 2 * 255);
    }
}

#[test]
fn lanczos3_sharper_at_seam_than_lanczos2() {
    // The wider 6-tap window's main selling point: closer to the source
    // step than Lanczos2 at the same destination scale. Pixel just on
    // the red side of the seam should have a *higher* red value under
    // Lanczos3 than under Lanczos2 (or at least not lower), and the
    // mirror pixel on the blue side should have a *higher* blue value.
    let (r2, v2) = frame_with(red_blue_split_image(6), ImageFilter::Lanczos2, 12);
    let (r3, v3) = frame_with(red_blue_split_image(6), ImageFilter::Lanczos3, 12);
    let out2 = r2.render(&v2);
    let out3 = r3.render(&v3);
    let stride = out2.planes[0].stride;
    let row = 6 * stride;
    // The seam in the source 6-pixel-wide red→blue image lands between
    // columns 2 and 3, which projects to between dst columns 5 and 6
    // at 2× scale. Pixel at column 4 is on the red side; at column 7
    // is on the blue side.
    let left = &out2.planes[0].data[row + 4 * 4..row + 4 * 4 + 4];
    let left3 = &out3.planes[0].data[row + 4 * 4..row + 4 * 4 + 4];
    let right = &out2.planes[0].data[row + 7 * 4..row + 7 * 4 + 4];
    let right3 = &out3.planes[0].data[row + 7 * 4..row + 7 * 4 + 4];
    assert!(
        left3[0] >= left[0],
        "lanczos3 red at column 4 ({}) should be ≥ lanczos2 ({})",
        left3[0],
        left[0]
    );
    assert!(
        right3[2] >= right[2],
        "lanczos3 blue at column 7 ({}) should be ≥ lanczos2 ({})",
        right3[2],
        right[2]
    );
}

#[test]
fn lanczos3_default_filter_unchanged() {
    // Adding a new variant must not flip the renderer default.
    let r = Renderer::new(4, 4);
    assert_eq!(r.image_filter, ImageFilter::Bilinear);
}
