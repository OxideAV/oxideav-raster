//! SoftMask render-path tests (luminance + alpha mask kinds).

use oxideav_core::{
    FillRule, Group, MaskKind, Node, Paint, Path, PathNode, Point, Rgba, VectorFrame,
};
use oxideav_raster::Renderer;

fn rect_path_node(x: f32, y: f32, w: f32, h: f32, fill: Rgba) -> Node {
    let mut p = Path::new();
    p.move_to(Point::new(x, y))
        .line_to(Point::new(x + w, y))
        .line_to(Point::new(x + w, y + h))
        .line_to(Point::new(x, y + h))
        .close();
    Node::Path(PathNode {
        path: p,
        fill: Some(Paint::Solid(fill)),
        stroke: None,
        fill_rule: FillRule::NonZero,
    })
}

fn vector_frame(w: u32, h: u32, root: Group) -> VectorFrame {
    VectorFrame {
        width: w as f32,
        height: h as f32,
        view_box: None,
        root,
        pts: None,
        time_base: oxideav_core::time::TimeBase::new(1, 1),
    }
}

#[test]
fn luminance_mask_clips_content_to_white_region() {
    // Content: a solid red 8×8 rectangle filling the whole canvas.
    // Mask: a white 4×4 square in the top-left half (luminance 255 →
    // full coverage there); everything else stays the buffer's
    // black-zero clear (luminance 0 → fully masked out).
    let content = rect_path_node(0.0, 0.0, 8.0, 8.0, Rgba::opaque(255, 0, 0));
    let mask = rect_path_node(0.0, 0.0, 4.0, 4.0, Rgba::opaque(255, 255, 255));
    let masked = Node::SoftMask {
        mask: Box::new(mask),
        mask_kind: MaskKind::Luminance,
        content: Box::new(content),
    };
    let mut root = Group::default();
    root.children.push(masked);
    let r = Renderer::new(8, 8);
    let out = r.render(&vector_frame(8, 8, root));
    let stride = out.planes[0].stride;
    // Pixel (1, 1): inside both content and mask → red.
    let i = stride + 4;
    assert_eq!(&out.planes[0].data[i..i + 4], &[255, 0, 0, 255]);
    // Pixel (5, 5): inside content, outside mask → transparent.
    let i = 5 * stride + 5 * 4;
    assert_eq!(&out.planes[0].data[i..i + 4], &[0, 0, 0, 0]);
}

#[test]
fn alpha_mask_uses_mask_alpha_directly() {
    // Same shape as above but mask_kind = Alpha. The mask is now an
    // opaque red rectangle (alpha 255 in the 4×4 region; alpha 0
    // outside). Behaviour should be identical to the luminance test.
    let content = rect_path_node(0.0, 0.0, 8.0, 8.0, Rgba::opaque(0, 255, 0));
    // Note: red mask — under Luminance this has Y ≈ 54, partial
    // coverage. Under Alpha it's full coverage where the rect is
    // drawn. The test assertion picks the Alpha path.
    let mask = rect_path_node(0.0, 0.0, 4.0, 4.0, Rgba::opaque(255, 0, 0));
    let masked = Node::SoftMask {
        mask: Box::new(mask),
        mask_kind: MaskKind::Alpha,
        content: Box::new(content),
    };
    let mut root = Group::default();
    root.children.push(masked);
    let r = Renderer::new(8, 8);
    let out = r.render(&vector_frame(8, 8, root));
    let stride = out.planes[0].stride;
    // Pixel (1, 1): full coverage from mask alpha → solid green.
    let i = stride + 4;
    assert_eq!(&out.planes[0].data[i..i + 4], &[0, 255, 0, 255]);
    // Pixel (5, 5): outside mask → transparent.
    let i = 5 * stride + 5 * 4;
    assert_eq!(&out.planes[0].data[i..i + 4], &[0, 0, 0, 0]);
}

#[test]
fn luminance_grey_mask_produces_partial_coverage() {
    // 50%-grey mask should give roughly half coverage on the
    // content. Grey RGB (128,128,128) has BT.709 Y ≈ 128.
    let content = rect_path_node(0.0, 0.0, 4.0, 4.0, Rgba::opaque(255, 0, 0));
    let mask = rect_path_node(0.0, 0.0, 4.0, 4.0, Rgba::opaque(128, 128, 128));
    let masked = Node::SoftMask {
        mask: Box::new(mask),
        mask_kind: MaskKind::Luminance,
        content: Box::new(content),
    };
    let mut root = Group::default();
    root.children.push(masked);
    let r = Renderer::new(4, 4);
    let out = r.render(&vector_frame(4, 4, root));
    let stride = out.planes[0].stride;
    // Pixel (1, 1): coverage ≈ 128/255 → half-opaque red.
    let i = stride + 4;
    let p = &out.planes[0].data[i..i + 4];
    // Final destination (transparent → red over transparent at
    // 50% alpha) un-premultiplied: r=255, g=0, b=0, a≈128.
    assert_eq!(p[0], 255, "red ch should still be 255 after un-premul");
    assert_eq!(p[1], 0);
    assert_eq!(p[2], 0);
    assert!(
        (p[3] as i32 - 128).abs() <= 4,
        "alpha should be ~128 (50% mask coverage), got {}",
        p[3]
    );
}

#[test]
fn empty_mask_renders_nothing() {
    // No mask geometry → coverage is all zero → nothing of the
    // content reaches the destination buffer.
    let content = rect_path_node(0.0, 0.0, 4.0, 4.0, Rgba::opaque(255, 0, 0));
    let empty_mask = Node::Group(Group::default()); // no children → no pixels
    let masked = Node::SoftMask {
        mask: Box::new(empty_mask),
        mask_kind: MaskKind::Luminance,
        content: Box::new(content),
    };
    let mut root = Group::default();
    root.children.push(masked);
    let r = Renderer::new(4, 4);
    let out = r.render(&vector_frame(4, 4, root));
    // Whole buffer should still be the renderer's transparent clear.
    assert!(out.planes[0].data.iter().all(|&b| b == 0));
}
