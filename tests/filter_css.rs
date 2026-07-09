//! End-to-end CSS `filter` property tests: parse a `<filter-value-list>`
//! (Filter Effects 1 §5 / §6.1) and evaluate it through
//! [`oxideav_raster::apply_filter_functions`].

use oxideav_raster::{apply_filter_functions, parse_filter_value_list, FilterFunction};

fn image(width: u32, height: u32) -> Vec<u8> {
    let mut buf = vec![0u8; width as usize * height as usize * 4];
    for (i, px) in buf.chunks_exact_mut(4).enumerate() {
        px[0] = (i * 31 + 40) as u8;
        px[1] = (i * 67 + 90) as u8;
        px[2] = (i * 13 + 200) as u8;
        px[3] = 255;
    }
    buf
}

#[test]
fn parsed_grayscale_equals_constructed_grayscale() {
    let src = image(6, 6);
    let parsed = parse_filter_value_list("grayscale(100%)").unwrap();
    let constructed = [FilterFunction::Grayscale(1.0)];
    assert_eq!(
        apply_filter_functions(&parsed, &src, 6, 6),
        apply_filter_functions(&constructed, &src, 6, 6)
    );
}

#[test]
fn parsed_none_is_identity() {
    let src = image(4, 4);
    let parsed = parse_filter_value_list("none").unwrap();
    assert_eq!(apply_filter_functions(&parsed, &src, 4, 4), src);
}

#[test]
fn parsed_multi_function_list_applies_in_document_order() {
    // §5: "The list of functions are applied in the order provided."
    // invert ∘ invert is the identity, but invert ∘ brightness(0) is
    // white — order must matter.
    let src = image(4, 4);
    let double_invert = parse_filter_value_list("invert(1) invert(1)").unwrap();
    let out = apply_filter_functions(&double_invert, &src, 4, 4);
    assert_eq!(out, src);

    let black_then_invert = parse_filter_value_list("brightness(0) invert(1)").unwrap();
    let out = apply_filter_functions(&black_then_invert, &src, 4, 4);
    for px in out.chunks_exact(4) {
        assert_eq!(&px[..3], &[255, 255, 255]);
    }
    let invert_then_black = parse_filter_value_list("invert(1) brightness(0)").unwrap();
    let out = apply_filter_functions(&invert_then_black, &src, 4, 4);
    for px in out.chunks_exact(4) {
        assert_eq!(&px[..3], &[0, 0, 0]);
    }
}

#[test]
fn parsed_drop_shadow_offsets_shadow_pixels() {
    // A single opaque pixel at (1, 1); drop-shadow(2px 1px red) must
    // paint a red shadow at (3, 2) beneath the source.
    let (w, h) = (6u32, 6u32);
    let mut src = vec![0u8; (w * h * 4) as usize];
    let i = ((w + 1) * 4) as usize;
    src[i..i + 4].copy_from_slice(&[0, 0, 255, 255]);

    let parsed = parse_filter_value_list("drop-shadow(2px 1px red)").unwrap();
    let out = apply_filter_functions(&parsed, &src, w, h);

    // Source pixel is preserved on top.
    assert_eq!(&out[i..i + 4], &[0, 0, 255, 255]);
    // Shadow lands at (1 + 2, 1 + 1) = (3, 2), in the flood colour.
    let s = ((2 * w + 3) * 4) as usize;
    assert_eq!(&out[s..s + 4], &[255, 0, 0, 255]);
}
