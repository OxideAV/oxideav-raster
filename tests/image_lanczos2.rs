//! Lanczos2 image-resampling sanity tests.
//!
//! Verifies that the new `ImageFilter::Lanczos2` sampler:
//!
//! * preserves a uniformly-coloured source (no edge-of-image artefacts
//!   from the truncated sinc tail — tested by sampling the centre of a
//!   solid-red 4×4 at 8×8 destination scale),
//! * produces interpolation between source pixels along a hard
//!   colour-seam (just like bilinear, but typically sharper).

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

fn frame_with(image: VideoFrame, filter: ImageFilter) -> (Renderer, VectorFrame) {
    let mut r = Renderer::new(8, 8);
    r.image_filter = filter;
    let img_node = Node::Image(ImageRef {
        frame: Box::new(image),
        bounds: Rect {
            x: 0.0,
            y: 0.0,
            width: 8.0,
            height: 8.0,
        },
        transform: Transform2D::identity(),
    });
    let mut root = Group::default();
    root.children.push(img_node);
    let v = VectorFrame {
        width: 8.0,
        height: 8.0,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    (r, v)
}

#[test]
fn lanczos2_preserves_solid_red_centre() {
    let (r, v) = frame_with(solid_red_image_4x4(), ImageFilter::Lanczos2);
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    // Centre pixel must remain pure red.
    let i = 4 * stride + 4 * 4;
    let p = &out.planes[0].data[i..i + 4];
    assert_eq!(p, &[255, 0, 0, 255]);
}

#[test]
fn lanczos2_blends_at_red_blue_seam() {
    let (r, v) = frame_with(red_blue_split_image_4x4(), ImageFilter::Lanczos2);
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    // Mid-row probe: at least one column near the seam should have
    // both red and blue channels active (proper kernel mixing).
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
        "lanczos2 should blend at the red→blue seam"
    );
}
