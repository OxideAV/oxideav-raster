//! Bilinear image resampling test.
//!
//! Renders a 4×4 source image scaled to 8×8 and checks that the
//! bilinear filter produces intermediate values between source pixels
//! (i.e. proper interpolation), while the nearest-neighbour filter
//! block-replicates source pixels verbatim.

use oxideav_core::{Group, ImageRef, Node, Rect, Transform2D, VectorFrame, VideoFrame, VideoPlane};
use oxideav_raster::{ImageFilter, Renderer};

/// Build a 4×4 RGBA image whose left half is opaque red and right
/// half is opaque blue. The hard vertical seam between x=1 and x=2 is
/// the test's interpolation target.
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

fn frame_with_image(filter: ImageFilter) -> (Renderer, VectorFrame) {
    let mut r = Renderer::new(8, 8);
    r.image_filter = filter;
    let img_node = Node::Image(ImageRef {
        frame: Box::new(red_blue_split_image_4x4()),
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
fn nearest_neighbour_block_replicates() {
    let (r, v) = frame_with_image(ImageFilter::Nearest);
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    // 4x4 source mapped to 8x8 dst → each src pixel covers 2×2 dst
    // pixels. Pixels 0..4 (x) should be red, pixels 4..8 should be
    // blue, with a hard transition right at x=4.
    let row = 4 * stride;
    let p3 = &out.planes[0].data[row + 3 * 4..row + 3 * 4 + 4];
    let p4 = &out.planes[0].data[row + 4 * 4..row + 4 * 4 + 4];
    assert_eq!(p3, &[255, 0, 0, 255], "nearest at x=3 must be solid red");
    assert_eq!(p4, &[0, 0, 255, 255], "nearest at x=4 must be solid blue");
}

#[test]
fn bilinear_interpolates_between_source_pixels() {
    let (r, v) = frame_with_image(ImageFilter::Bilinear);
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    // Pick a row well inside the image (y=4), and probe across the
    // x-axis. With the 4×4 → 8×8 map and pixel-centre alignment,
    // there must be at least one column (somewhere around x=3..5)
    // where neither channel is fully 0 or fully 255 — i.e. genuine
    // interpolation between red and blue source pixels.
    let row = 4 * stride;
    let mut saw_intermediate = false;
    for x in 0..8 {
        let p = &out.planes[0].data[row + x * 4..row + x * 4 + 4];
        let r_chan = p[0];
        let b_chan = p[2];
        // "Intermediate" = the seam blended both colours. Both
        // channels must be > 0 (so it's not the purely-red or
        // purely-blue limit).
        if r_chan > 10 && r_chan < 245 && b_chan > 10 && b_chan < 245 {
            saw_intermediate = true;
            break;
        }
    }
    assert!(
        saw_intermediate,
        "bilinear should produce at least one purple pixel along the red→blue seam, got {:?}",
        (0..8)
            .map(|x| {
                let p = &out.planes[0].data[row + x * 4..row + x * 4 + 4];
                (p[0], p[1], p[2], p[3])
            })
            .collect::<Vec<_>>()
    );
}

#[test]
fn bilinear_default_filter_is_active() {
    // The default `Renderer::new` should already select bilinear
    // (this round's default), independent of explicit configuration.
    let r = Renderer::new(8, 8);
    assert_eq!(r.image_filter, ImageFilter::Bilinear);
}

#[test]
fn bilinear_centre_of_solid_block_matches_source_colour() {
    // A 4×4 image of pure red, scaled to 8×8 with bilinear: every
    // interior dst pixel must remain pure red (clamp-to-edge means no
    // bleed of "transparent" sneaks in along the edges).
    let mut data = Vec::with_capacity(4 * 4 * 4);
    for _ in 0..16 {
        data.extend_from_slice(&[255, 0, 0, 255]);
    }
    let img = VideoFrame {
        pts: None,
        planes: vec![VideoPlane { stride: 16, data }],
    };
    let mut r = Renderer::new(8, 8);
    r.image_filter = ImageFilter::Bilinear;
    let mut root = Group::default();
    root.children.push(Node::Image(ImageRef {
        frame: Box::new(img),
        bounds: Rect {
            x: 0.0,
            y: 0.0,
            width: 8.0,
            height: 8.0,
        },
        transform: Transform2D::identity(),
    }));
    let v = VectorFrame {
        width: 8.0,
        height: 8.0,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    };
    let out = r.render(&v);
    let stride = out.planes[0].stride;
    // Centre pixel.
    let i = 4 * stride + 4 * 4;
    let p = &out.planes[0].data[i..i + 4];
    assert_eq!(p, &[255, 0, 0, 255]);
}
