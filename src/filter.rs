//! Image-domain filter primitives.
//!
//! This module hosts post-rasterisation pixel-domain filters — the SVG
//! 1.1 `<feXxx>` family / PDF transparency-group filter family. The
//! pipeline elsewhere in the crate produces a packed-`Rgba`
//! [`oxideav_core::VideoFrame`]; the operators here take that buffer
//! (or a slice of it) and produce a new buffer of the same dimensions.
//!
//! # Currently implemented
//!
//! * **Mathematical morphology** — erosion and dilation by an axis-
//!   aligned rectangular structuring element (`feMorphology` in SVG
//!   1.1 §15.20). Implemented per-channel on premultiplied RGBA in
//!   `u8` space. The rectangular kernel decomposes into a horizontal
//!   1-D pass followed by a vertical 1-D pass, a standard textbook
//!   property of morphology with a flat separable structuring element
//!   (Serra, *Image Analysis and Mathematical Morphology*, 1982 §I.4;
//!   Gonzalez & Woods, *Digital Image Processing*, 3rd ed., 2008
//!   §9.4.1) — so the per-pixel work scales linearly with the radius
//!   instead of quadratically.
//!
//! * **Colour-matrix transform** — `feColorMatrix` from SVG 1.1
//!   §15.10. A 4×5 matrix `M` is applied to the un-premultiplied
//!   colour-tuple `(R, G, B, A, 1)ᵀ` per pixel to produce the new
//!   `(R', G', B', A')ᵀ`. The spec defines four sub-operations
//!   ([`ColorMatrixOp`]): a general user matrix, a one-parameter
//!   saturation matrix, a one-parameter hue rotation, and a fixed
//!   "luminance to alpha" matrix. The latter three are pre-computed
//!   matrices given verbatim by §15.10; we expose them through
//!   [`ColorMatrix::saturate`], [`ColorMatrix::hue_rotate`] and
//!   [`ColorMatrix::luminance_to_alpha`] so callers do not have to
//!   re-derive them.
//!
//! # Deferred
//!
//! Gaussian blur (`feGaussianBlur`), drop shadow (`feDropShadow`),
//! `feComponentTransfer`, `feConvolveMatrix`, `feTurbulence` (Perlin),
//! `feDisplacementMap`, `feSpecularLighting`, `feDiffuseLighting`.
//!
//! # Wall provenance
//!
//! Math transcribed from `docs/image/svg/svg11-second-edition.pdf`
//! §15.20 ("This filter primitive performs 'fattening' or 'thinning'
//! of artwork. … In dilation, the output pixel is the individual
//! component-wise maximum of the corresponding R,G,B,A values in the
//! input image's kernel rectangle. In erosion, the output pixel is the
//! individual component-wise minimum…") and Serra (1982) §I.4 + Gonzalez
//! & Woods (2008) §9.4.1 for the separability decomposition; §15.10
//! for the `feColorMatrix` matrix forms (general 4×5 matrix; the
//! saturate / hueRotate / luminanceToAlpha pre-built matrices are
//! reproduced verbatim from the spec table). No `image` / `imageproc`
//! / `opencv` / `cairo` / `skia` source consulted.

use oxideav_core::Rgba;

/// Operator selector for [`morphology`].
///
/// Mirrors the `operator` attribute of SVG 1.1 §15.20 `<feMorphology>`
/// (`"erode"` | `"dilate"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphologyOp {
    /// Component-wise *minimum* over the structuring element ("thin"
    /// the artwork — shrinks bright regions, grows dark / transparent
    /// regions).
    Erode,
    /// Component-wise *maximum* over the structuring element ("fatten"
    /// the artwork — grows bright regions, shrinks dark / transparent
    /// regions).
    Dilate,
}

/// SVG 1.1 §15.20 `<feMorphology>`: morphological erosion or dilation by
/// an axis-aligned rectangular structuring element.
///
/// `width × height` is the input/output image extent in pixels. `src`
/// is a packed-RGBA byte buffer of exactly `width * height * 4` bytes
/// in row-major order (the same packing produced by
/// [`crate::Renderer::render`]).
///
/// `radius_x` and `radius_y` are the half-extents of the structuring
/// element along the X and Y axes respectively, in source pixels. The
/// rectangle is `(2·rx + 1) × (2·ry + 1)` (i.e. inclusive of the
/// centre — the standard "ball of radius r" definition in discrete
/// morphology). A radius of zero on **either** axis disables the
/// effect on that axis (the corresponding 1-D pass is a no-op); a
/// radius of zero on **both** axes returns the input unchanged.
///
/// Per the SVG 1.1 spec the operation runs component-wise on the
/// premultiplied colour values, which makes
/// `Cr,Cg,Cb ≤ Ca`an invariant the operator preserves: erosion of a
/// premultiplied tuple gives `(min Ci) ≤ (min Ca)`, dilation gives
/// `(max Ci) ≤ (max Ca)`. (The caller is responsible for handing in
/// premultiplied data; the crate's renderer already produces straight-
/// alpha output, so a caller wanting strict spec semantics should
/// premultiply first.)
///
/// # Algorithm
///
/// A rectangular structuring element `B = Bx ⊕ By` (Minkowski sum of a
/// horizontal `(2·rx + 1)`-line and a vertical `(2·ry + 1)`-line)
/// is *separable*: for any flat SE that decomposes as `Bx ⊕ By`,
///
/// ```text
/// f ⊖ B  =  (f ⊖ Bx) ⊖ By
/// f ⊕ B  =  (f ⊕ Bx) ⊕ By
/// ```
///
/// (Serra 1982 §I.4 Theorem 4.1; Gonzalez & Woods 2008 §9.4.1
/// eq. 9.4-1). The implementation therefore runs a horizontal 1-D pass
/// across each row, then a vertical 1-D pass down each column. Each
/// pass is a naive sliding-window min/max with **clamp-to-edge**
/// boundary handling (extending the image by its border pixels so the
/// kernel is always full at the corners — the same SVG default the
/// rest of the filter pipeline uses for missing samples).
///
/// Complexity: `O(W · H · (rx + ry))` instead of the
/// `O(W · H · rx · ry)` naive 2-D scan.
///
/// # Panics
///
/// Panics if `src.len() != width as usize * height as usize * 4`.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same dimensions, written
/// channel-independently. `radius_x == 0 && radius_y == 0` returns a
/// straight copy of `src`.
pub fn morphology(
    src: &[u8],
    width: u32,
    height: u32,
    radius_x: u32,
    radius_y: u32,
    op: MorphologyOp,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("morphology: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "morphology: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    // Fast path: both radii zero ⇒ identity per the SVG spec note that
    // a "value of zero disables the effect of the given filter
    // primitive". (Strictly the spec then says "result is a transparent
    // black image" — but only for radius=0 on the *filter primitive
    // level*. We treat the function-level same-pixel pass-through
    // here; the SVG-element-level transparent-black behaviour is the
    // caller's job at the scene-graph layer, on top of `radius == 0`
    // being an *error* per the spec error-processing rules anyway.)
    if radius_x == 0 && radius_y == 0 {
        return src.to_vec();
    }

    let mut buf = src.to_vec();

    if radius_x > 0 {
        let rx = radius_x as usize;
        let mut row = vec![0u8; w * 4];
        for y in 0..h {
            let off = y * w * 4;
            row.copy_from_slice(&buf[off..off + w * 4]);
            morphology_1d_horizontal(&row, &mut buf[off..off + w * 4], w, rx, op);
        }
    }

    if radius_y > 0 {
        let ry = radius_y as usize;
        // Column scratch: gather one column at a time as a
        // contiguous `4·H`-byte slice (so the inner loop is sequential
        // memory access), filter it, write it back.
        let mut col_in = vec![0u8; h * 4];
        let mut col_out = vec![0u8; h * 4];
        for x in 0..w {
            for y in 0..h {
                let src_off = (y * w + x) * 4;
                let dst_off = y * 4;
                col_in[dst_off..dst_off + 4].copy_from_slice(&buf[src_off..src_off + 4]);
            }
            morphology_1d_horizontal(&col_in, &mut col_out, h, ry, op);
            for y in 0..h {
                let dst_off = (y * w + x) * 4;
                let src_off = y * 4;
                buf[dst_off..dst_off + 4].copy_from_slice(&col_out[src_off..src_off + 4]);
            }
        }
    }

    buf
}

/// Single-pass 1-D morphology along a packed-RGBA row of `len` pixels.
///
/// `src` and `dst` must be `len * 4` bytes. Boundary handling is
/// clamp-to-edge: window samples outside `[0, len)` reuse the nearest
/// in-bounds pixel. Naive `O(len · radius)` scan — chosen for
/// clarity; a van Herk / Gil-Werman `O(len)` formulation is a future
/// optimisation.
fn morphology_1d_horizontal(
    src: &[u8],
    dst: &mut [u8],
    len: usize,
    radius: usize,
    op: MorphologyOp,
) {
    debug_assert_eq!(src.len(), len * 4);
    debug_assert_eq!(dst.len(), len * 4);
    let imax = len as isize - 1;
    for i in 0..len {
        let lo = (i as isize - radius as isize).max(0) as usize;
        let hi = (i as isize + radius as isize).min(imax) as usize;
        let mut acc = [
            src[lo * 4],
            src[lo * 4 + 1],
            src[lo * 4 + 2],
            src[lo * 4 + 3],
        ];
        for j in (lo + 1)..=hi {
            let p = &src[j * 4..j * 4 + 4];
            match op {
                MorphologyOp::Erode => {
                    if p[0] < acc[0] {
                        acc[0] = p[0];
                    }
                    if p[1] < acc[1] {
                        acc[1] = p[1];
                    }
                    if p[2] < acc[2] {
                        acc[2] = p[2];
                    }
                    if p[3] < acc[3] {
                        acc[3] = p[3];
                    }
                }
                MorphologyOp::Dilate => {
                    if p[0] > acc[0] {
                        acc[0] = p[0];
                    }
                    if p[1] > acc[1] {
                        acc[1] = p[1];
                    }
                    if p[2] > acc[2] {
                        acc[2] = p[2];
                    }
                    if p[3] > acc[3] {
                        acc[3] = p[3];
                    }
                }
            }
        }
        dst[i * 4..i * 4 + 4].copy_from_slice(&acc);
    }
}

/// Convenience wrapper that runs [`morphology`] on a slice of [`Rgba`]
/// pixels and returns a `Vec<Rgba>` of the same length. Identical
/// semantics — provided for callers that already have a typed
/// pixel buffer (the crate's tests are the immediate consumer).
pub fn morphology_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    radius_x: u32,
    radius_y: u32,
    op: MorphologyOp,
) -> Vec<Rgba> {
    assert_eq!(
        src.len(),
        width as usize * height as usize,
        "morphology_pixels: src.len() == {} but width*height == {}",
        src.len(),
        width as usize * height as usize
    );
    let mut bytes = Vec::with_capacity(src.len() * 4);
    for p in src {
        bytes.push(p.r);
        bytes.push(p.g);
        bytes.push(p.b);
        bytes.push(p.a);
    }
    let out = morphology(&bytes, width, height, radius_x, radius_y, op);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, c: Rgba) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.push(c.r);
            v.push(c.g);
            v.push(c.b);
            v.push(c.a);
        }
        v
    }

    #[test]
    fn zero_radius_is_identity() {
        let img = solid_rgba(4, 3, Rgba::new(10, 20, 30, 40));
        // Perturb one pixel so we'd notice if it got smeared.
        let mut input = img.clone();
        input[5 * 4] = 200;
        let out = morphology(&input, 4, 3, 0, 0, MorphologyOp::Dilate);
        assert_eq!(out, input);
        let out = morphology(&input, 4, 3, 0, 0, MorphologyOp::Erode);
        assert_eq!(out, input);
    }

    #[test]
    fn solid_image_is_invariant_under_either_operator() {
        // Morphology is idempotent on constant images: min(const set) ==
        // max(const set) == const.
        let img = solid_rgba(8, 5, Rgba::new(80, 120, 200, 255));
        for op in [MorphologyOp::Erode, MorphologyOp::Dilate] {
            for (rx, ry) in [(0, 0), (1, 0), (0, 1), (1, 1), (3, 2), (4, 4)] {
                let out = morphology(&img, 8, 5, rx, ry, op);
                assert_eq!(out, img, "op={op:?} rx={rx} ry={ry}");
            }
        }
    }

    #[test]
    fn dilate_grows_isolated_bright_pixel_to_full_kernel() {
        // 7×7 black canvas with a single fully-bright pixel at the
        // centre. Dilate with rx = ry = 2 ⇒ a 5×5 bright square
        // centred on the original pixel.
        let w = 7u32;
        let h = 7u32;
        let mut img = solid_rgba(w, h, Rgba::new(0, 0, 0, 255));
        let cx = 3usize;
        let cy = 3usize;
        let off = (cy * w as usize + cx) * 4;
        img[off..off + 4].copy_from_slice(&[200, 150, 100, 255]);

        let out = morphology(&img, w, h, 2, 2, MorphologyOp::Dilate);

        // Count the pixels that ended up with the bright colour. The
        // kernel is (2·2+1)·(2·2+1) = 25 pixels and they all live
        // fully inside the canvas → exactly 25 bright pixels.
        let mut bright = 0;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let p = &out[(y * w as usize + x) * 4..(y * w as usize + x) * 4 + 4];
                if p == [200, 150, 100, 255] {
                    bright += 1;
                    // Cell must be inside the centred 5×5 box.
                    let dx = x as isize - cx as isize;
                    let dy = y as isize - cy as isize;
                    assert!(dx.abs() <= 2 && dy.abs() <= 2, "stray bright at ({x},{y})");
                }
            }
        }
        assert_eq!(bright, 25, "dilate 5×5 footprint pixel count");
    }

    #[test]
    fn erode_shrinks_solid_block_by_radius_on_each_side() {
        // 9×9 canvas: 7×7 fully-opaque red block centred on a
        // transparent background. Erode rx = ry = 1 ⇒ a 5×5 red
        // block (1-pixel shaving off each side).
        let w = 9u32;
        let h = 9u32;
        let mut img = solid_rgba(w, h, Rgba::new(0, 0, 0, 0));
        for y in 1..8usize {
            for x in 1..8usize {
                let off = (y * w as usize + x) * 4;
                img[off..off + 4].copy_from_slice(&[255, 0, 0, 255]);
            }
        }

        let out = morphology(&img, w, h, 1, 1, MorphologyOp::Erode);

        let mut red = 0;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let p = &out[(y * w as usize + x) * 4..(y * w as usize + x) * 4 + 4];
                if p == [255, 0, 0, 255] {
                    red += 1;
                    assert!(
                        (2..=6).contains(&x) && (2..=6).contains(&y),
                        "stray red at ({x},{y}) after erode"
                    );
                }
            }
        }
        assert_eq!(red, 25, "erode shrinks 7×7 to 5×5 with rx=ry=1");
    }

    #[test]
    fn erode_and_dilate_are_duals_via_complement() {
        // Discrete duality: erode(f) == complement(dilate(complement(f)))
        // for any radius. This is the textbook morphology relation
        // (Serra 1982 §I.4 Theorem 4.2; G&W §9.2.2 eq. 9.2-6). We
        // verify it bit-exact on a small noisy pattern.
        let w = 5u32;
        let h = 4u32;
        let pattern: Vec<u8> = (0..(w * h * 4) as u8)
            .map(|i| (i.wrapping_mul(37)).wrapping_add(11))
            .collect();

        for (rx, ry) in [(1u32, 0u32), (0, 1), (1, 1), (2, 1), (2, 2)] {
            let eroded = morphology(&pattern, w, h, rx, ry, MorphologyOp::Erode);
            // complement → dilate → complement
            let neg: Vec<u8> = pattern.iter().map(|b| 255 - b).collect();
            let dil = morphology(&neg, w, h, rx, ry, MorphologyOp::Dilate);
            let dual: Vec<u8> = dil.iter().map(|b| 255 - b).collect();
            assert_eq!(eroded, dual, "duality fails at rx={rx} ry={ry}");
        }
    }

    #[test]
    fn separability_matches_naive_2d() {
        // The 2-D sliding-window min/max over a rectangular SE must
        // match the separable H-then-V evaluation (this is the
        // implementation detail the production path relies on for
        // its `O(W·H·(rx+ry))` complexity, so test it explicitly).
        let w = 11u32;
        let h = 9u32;
        let mut pat: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                pat.push(((x * 17 + y * 3) & 0xff) as u8);
                pat.push(((x * 5 + y * 23) & 0xff) as u8);
                pat.push(((x + y) & 0xff) as u8);
                pat.push((x.wrapping_mul(y).wrapping_add(7) & 0xff) as u8);
            }
        }

        for (rx, ry) in [(1u32, 1u32), (2, 1), (1, 2), (3, 2), (2, 3)] {
            for op in [MorphologyOp::Erode, MorphologyOp::Dilate] {
                let sep = morphology(&pat, w, h, rx, ry, op);
                let naive = naive_2d_morphology(&pat, w, h, rx, ry, op);
                assert_eq!(sep, naive, "rx={rx} ry={ry} op={op:?}");
            }
        }
    }

    /// Reference implementation used only inside the test module:
    /// straight 2-D scan over every kernel cell with clamp-to-edge.
    /// Quadratic in radius — kept ONLY for verifying the production
    /// separable path. NOT used at runtime.
    fn naive_2d_morphology(
        src: &[u8],
        w: u32,
        h: u32,
        rx: u32,
        ry: u32,
        op: MorphologyOp,
    ) -> Vec<u8> {
        let mut out = vec![0u8; src.len()];
        let wi = w as isize;
        let hi = h as isize;
        for y in 0..h as isize {
            for x in 0..w as isize {
                let mut acc = [0u8; 4];
                let mut first = true;
                for ky in -(ry as isize)..=(ry as isize) {
                    for kx in -(rx as isize)..=(rx as isize) {
                        let sx = (x + kx).clamp(0, wi - 1);
                        let sy = (y + ky).clamp(0, hi - 1);
                        let off = (sy * wi + sx) as usize * 4;
                        let p = [src[off], src[off + 1], src[off + 2], src[off + 3]];
                        if first {
                            acc = p;
                            first = false;
                        } else {
                            for c in 0..4 {
                                match op {
                                    MorphologyOp::Erode => {
                                        if p[c] < acc[c] {
                                            acc[c] = p[c];
                                        }
                                    }
                                    MorphologyOp::Dilate => {
                                        if p[c] > acc[c] {
                                            acc[c] = p[c];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let off = (y * wi + x) as usize * 4;
                out[off..off + 4].copy_from_slice(&acc);
            }
        }
        out
    }

    #[test]
    fn dilate_dominates_input_dominates_erode_per_pixel() {
        // For any pixel and any channel: erode(f)[i] ≤ f[i] ≤ dilate(f)[i].
        // This is the extensivity / anti-extensivity property of the
        // operators (Serra 1982 §I.4).
        let w = 7u32;
        let h = 6u32;
        let mut input: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                input.push(((x * 19 + y * 41 + 13) & 0xff) as u8);
                input.push(((x * 7 + y * 11 + 3) & 0xff) as u8);
                input.push(((x * 53 + y * 29 + 99) & 0xff) as u8);
                input.push(((x.wrapping_mul(y) + 17) & 0xff) as u8);
            }
        }

        for (rx, ry) in [(1u32, 0u32), (0, 1), (1, 1), (2, 2), (3, 1)] {
            let eroded = morphology(&input, w, h, rx, ry, MorphologyOp::Erode);
            let dilated = morphology(&input, w, h, rx, ry, MorphologyOp::Dilate);
            for i in 0..input.len() {
                assert!(
                    eroded[i] <= input[i],
                    "erode at byte {i} value {} > input {}",
                    eroded[i],
                    input[i]
                );
                assert!(
                    dilated[i] >= input[i],
                    "dilate at byte {i} value {} < input {}",
                    dilated[i],
                    input[i]
                );
            }
        }
    }

    #[test]
    fn morphology_pixels_wrapper_matches_byte_api() {
        let w = 4u32;
        let h = 3u32;
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for i in 0..(w * h) as u8 {
            let r = i.wrapping_mul(7);
            let g = i.wrapping_mul(17).wrapping_add(3);
            let b = i.wrapping_mul(31).wrapping_add(5);
            let a = 255u8;
            bytes.extend_from_slice(&[r, g, b, a]);
            pixels.push(Rgba::new(r, g, b, a));
        }
        for op in [MorphologyOp::Erode, MorphologyOp::Dilate] {
            for (rx, ry) in [(0u32, 1u32), (1, 0), (1, 1), (2, 1)] {
                let bytes_out = morphology(&bytes, w, h, rx, ry, op);
                let pix_out = morphology_pixels(&pixels, w, h, rx, ry, op);
                let recombined: Vec<u8> =
                    pix_out.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
                assert_eq!(bytes_out, recombined, "op={op:?} rx={rx} ry={ry}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "morphology: src.len() ==")]
    fn morphology_panics_on_wrong_length() {
        let bad = vec![0u8; 7]; // not w·h·4 for w=2, h=2
        let _ = morphology(&bad, 2, 2, 1, 1, MorphologyOp::Erode);
    }
}

// ---------------------------------------------------------------------------
//  feColorMatrix (SVG 1.1 §15.10)
// ---------------------------------------------------------------------------

/// Selector mirroring the `type` attribute of SVG 1.1 §15.10
/// `<feColorMatrix>` (`"matrix"` | `"saturate"` | `"hueRotate"` |
/// `"luminanceToAlpha"`).
///
/// The variants carry the spec's user-supplied parameter (none for
/// `Matrix` — the matrix itself is supplied as a [`ColorMatrix`]; a
/// single scalar for `Saturate` and `HueRotate`). The fully-fixed
/// `LuminanceToAlpha` operator takes no parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorMatrixOp {
    /// User-supplied 4×5 matrix applied verbatim. The matrix itself
    /// lives in the [`ColorMatrix`] passed to [`color_matrix`].
    Matrix,
    /// SVG `type="saturate"`: pre-built saturation matrix
    /// parameterised by `s ∈ [0, ∞)`. `s = 1` is identity, `s = 0`
    /// collapses RGB to the BT.709 luminance scalar, `s > 1` boosts
    /// saturation past the original.
    Saturate(f32),
    /// SVG `type="hueRotate"`: rotate the colour vector around the
    /// achromatic axis by `degrees` degrees (positive = R→G→B in the
    /// spec's convention). Constructed from the spec's static
    /// constant / cos-term / sin-term matrix triple.
    HueRotate(f32),
    /// SVG `type="luminanceToAlpha"`: fixed matrix that produces a
    /// transparent black image whose alpha is the BT.709 luminance of
    /// the input. Used as the back-end for SVG `mask`'s
    /// `mask-type="luminance"` and for chained PDF `SMask`
    /// `Luminosity` subtypes.
    LuminanceToAlpha,
}

/// A 4×5 colour-transform matrix in row-major layout (the same layout
/// SVG 1.1 §15.10 uses for the `values` attribute).
///
/// The four rows produce the new R, G, B, A in turn; the five columns
/// are the coefficients of `(R, G, B, A, 1)` — the trailing `1` is the
/// homogeneous bias term that lets the matrix add a constant offset to
/// each output channel.
///
/// ```text
/// | R' |   | m00 m01 m02 m03 m04 |   | R |
/// | G' |   | m10 m11 m12 m13 m14 |   | G |
/// | B' | = | m20 m21 m22 m23 m24 | × | B |
/// | A' |   | m30 m31 m32 m33 m34 |   | A |
///                                    | 1 |
/// ```
///
/// All values are floats in the same `[0, 1]` normalised scale the
/// spec uses internally (the byte-API entry point [`color_matrix`]
/// promotes/demotes around the matrix multiply).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorMatrix(pub [[f32; 5]; 4]);

impl ColorMatrix {
    /// 4×5 identity matrix: `(R', G', B', A') = (R, G, B, A)`. Useful
    /// as a base for callers that want to perturb a small number of
    /// entries.
    pub const fn identity() -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])
    }

    /// SVG 1.1 §15.10 `type="saturate"` matrix for saturation factor
    /// `s` (clamped to `[0, ∞)` — the spec leaves the upper bound
    /// open; `s = 0` is the explicit "fully desaturated" point).
    ///
    /// The matrix is reproduced verbatim from §15.10:
    ///
    /// ```text
    /// | 0.213 + 0.787·s   0.715 − 0.715·s   0.072 − 0.072·s   0   0 |
    /// | 0.213 − 0.213·s   0.715 + 0.285·s   0.072 − 0.072·s   0   0 |
    /// | 0.213 − 0.213·s   0.715 − 0.715·s   0.072 + 0.928·s   0   0 |
    /// |        0                  0                  0        1   0 |
    /// ```
    ///
    /// The `(0.213, 0.715, 0.072)` row is the spec-mandated BT.709
    /// luminance triple (§15.10 — slightly different from the PDF
    /// blend-mode `Lum` coefficients used elsewhere in this crate,
    /// which come from PDF 32000-1:2008 §11.3.5.3).
    pub fn saturate(s: f32) -> Self {
        let s = if s.is_nan() { 1.0 } else { s.max(0.0) };
        Self([
            [
                0.213 + 0.787 * s,
                0.715 - 0.715 * s,
                0.072 - 0.072 * s,
                0.0,
                0.0,
            ],
            [
                0.213 - 0.213 * s,
                0.715 + 0.285 * s,
                0.072 - 0.072 * s,
                0.0,
                0.0,
            ],
            [
                0.213 - 0.213 * s,
                0.715 - 0.715 * s,
                0.072 + 0.928 * s,
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])
    }

    /// SVG 1.1 §15.10 `type="hueRotate"` matrix for rotation angle
    /// `degrees` (positive direction is the spec's R → G → B
    /// convention).
    ///
    /// The spec gives the result as the sum of three constant 3×3
    /// matrices weighted by `1`, `cos(θ)` and `sin(θ)` respectively;
    /// the alpha row is identity. Coefficients are verbatim from
    /// §15.10:
    ///
    /// ```text
    /// const   = |  0.213  0.715  0.072 |
    ///           |  0.213  0.715  0.072 |
    ///           |  0.213  0.715  0.072 |
    ///
    /// cos·    = |  0.787 −0.715 −0.072 |
    ///           | −0.213  0.285 −0.072 |
    ///           | −0.213 −0.715  0.928 |
    ///
    /// sin·    = | −0.213 −0.715  0.928 |
    ///           |  0.143  0.140 −0.283 |
    ///           | −0.787  0.715  0.072 |
    /// ```
    ///
    /// `degrees.is_nan()` is treated as zero rotation (identity row);
    /// the rotation is otherwise unbounded — the trig functions are
    /// inherently periodic.
    pub fn hue_rotate(degrees: f32) -> Self {
        let theta = if degrees.is_nan() {
            0.0
        } else {
            degrees.to_radians()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Spec-verbatim 3×3 contributions.
        let const_m = [
            [0.213, 0.715, 0.072],
            [0.213, 0.715, 0.072],
            [0.213, 0.715, 0.072],
        ];
        let cos_m = [
            [0.787, -0.715, -0.072],
            [-0.213, 0.285, -0.072],
            [-0.213, -0.715, 0.928],
        ];
        let sin_m = [
            [-0.213, -0.715, 0.928],
            [0.143, 0.140, -0.283],
            [-0.787, 0.715, 0.072],
        ];

        let mut rows = [[0.0f32; 5]; 4];
        for i in 0..3 {
            for j in 0..3 {
                rows[i][j] = const_m[i][j] + c * cos_m[i][j] + s * sin_m[i][j];
            }
        }
        rows[3][3] = 1.0; // alpha untouched
        Self(rows)
    }

    /// SVG 1.1 §15.10 `type="luminanceToAlpha"` matrix.
    ///
    /// ```text
    /// |   0      0      0     0  0 |
    /// |   0      0      0     0  0 |
    /// |   0      0      0     0  0 |
    /// | 0.2125 0.7154 0.0721 0  0 |
    /// ```
    ///
    /// Note the §15.10 luminance row uses the slightly different
    /// `(0.2125, 0.7154, 0.0721)` triple — this is the BT.709 set of
    /// coefficients SVG 1.1 cites for "luminance to alpha"
    /// specifically. (The saturation / hueRotate matrices above use
    /// the rounded `(0.213, 0.715, 0.072)` set the same section
    /// hands out for those operators; we follow the spec literally
    /// for each.)
    pub const fn luminance_to_alpha() -> Self {
        Self([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2125, 0.7154, 0.0721, 0.0, 0.0],
        ])
    }

    /// Resolve a [`ColorMatrixOp`] to the concrete `ColorMatrix` it
    /// names. Used internally by [`color_matrix`] / [`color_matrix_op`]
    /// and re-exported for callers that want to inspect / cache /
    /// compose the matrix before applying it.
    pub fn from_op(op: ColorMatrixOp, user: &ColorMatrix) -> ColorMatrix {
        match op {
            ColorMatrixOp::Matrix => *user,
            ColorMatrixOp::Saturate(s) => ColorMatrix::saturate(s),
            ColorMatrixOp::HueRotate(d) => ColorMatrix::hue_rotate(d),
            ColorMatrixOp::LuminanceToAlpha => ColorMatrix::luminance_to_alpha(),
        }
    }
}

/// SVG 1.1 §15.10 `<feColorMatrix>` with an explicit user-supplied
/// matrix — apply `m` per pixel and clamp the result to `[0, 255]`.
///
/// `width × height` is the input/output image extent in pixels. `src`
/// is a packed-RGBA byte buffer of exactly `width * height * 4` bytes
/// in row-major order.
///
/// **Pixel space.** The §15.10 multiply is defined on *un-premultiplied*
/// channel values normalised to `[0, 1]`. This entry point therefore
/// treats the input bytes as straight-alpha sRGB samples (no
/// linearisation — §15.10 is in the device gamut, like the rest of the
/// SVG colour-matrix algebra). The output bytes are also straight
/// alpha; the caller is responsible for any subsequent premultiplication
/// before compositing.
///
/// # Algorithm
///
/// 1. For each pixel: normalise `(R, G, B, A) → (r, g, b, a) ∈ [0, 1]⁴`.
/// 2. Compute `(r', g', b', a') = M · (r, g, b, a, 1)ᵀ` per the §15.10
///    matrix form.
/// 3. Clamp each output channel to `[0, 1]` and quantise back to
///    `round(x · 255)`.
///
/// Complexity is `O(W · H)` with a constant 20-term inner product per
/// channel — independent of any radius or kernel size.
///
/// # Panics
///
/// Panics if `src.len() != width as usize * height as usize * 4`.
pub fn color_matrix(src: &[u8], width: u32, height: u32, m: &ColorMatrix) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("color_matrix: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "color_matrix: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    let rows = &m.0;
    let mut out = Vec::with_capacity(src.len());
    for chunk in src.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let a = chunk[3] as f32 / 255.0;
        for row in rows {
            let v = row[0] * r + row[1] * g + row[2] * b + row[3] * a + row[4];
            out.push(quantise_unit(v));
        }
    }
    out
}

/// SVG 1.1 §15.10 `<feColorMatrix>` with the `type=` attribute
/// expanded into a [`ColorMatrixOp`]. `user_matrix` is consulted only
/// when `op == ColorMatrixOp::Matrix` — the other three operators
/// build their matrix from the operator parameters and the spec's
/// hard-coded coefficient tables.
///
/// Equivalent to:
///
/// ```text
/// let m = ColorMatrix::from_op(op, user_matrix);
/// color_matrix(src, w, h, &m)
/// ```
///
/// — exposed as a separate entry point so callers that hold a parsed
/// SVG `feColorMatrix` element can dispatch in one line.
///
/// # Panics
///
/// Same as [`color_matrix`].
pub fn color_matrix_op(
    src: &[u8],
    width: u32,
    height: u32,
    op: ColorMatrixOp,
    user_matrix: &ColorMatrix,
) -> Vec<u8> {
    let m = ColorMatrix::from_op(op, user_matrix);
    color_matrix(src, width, height, &m)
}

/// Convenience wrapper that runs [`color_matrix`] on a slice of [`Rgba`]
/// pixels and returns a `Vec<Rgba>` of the same length. Identical
/// semantics — provided for callers that already have a typed pixel
/// buffer.
pub fn color_matrix_pixels(src: &[Rgba], width: u32, height: u32, m: &ColorMatrix) -> Vec<Rgba> {
    assert_eq!(
        src.len(),
        width as usize * height as usize,
        "color_matrix_pixels: src.len() == {} but width*height == {}",
        src.len(),
        width as usize * height as usize
    );
    let mut bytes = Vec::with_capacity(src.len() * 4);
    for p in src {
        bytes.push(p.r);
        bytes.push(p.g);
        bytes.push(p.b);
        bytes.push(p.a);
    }
    let out = color_matrix(&bytes, width, height, m);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

/// Clamp a normalised-`[0, 1]` colour-channel value to that range and
/// quantise it back to a `u8` via the standard `round(x · 255)`
/// rule. NaNs map to zero (a defensive choice — the §15.10 multiply is
/// a finite linear combination of finite inputs, so a NaN would have
/// to come from a NaN entry in the user matrix).
#[inline]
fn quantise_unit(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let clamped = v.clamp(0.0, 1.0);
    (clamped * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod color_matrix_tests {
    use super::*;

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
    fn identity_matrix_is_bytewise_identity() {
        // The 4×5 identity must round-trip every (R, G, B, A) tuple
        // bit-exactly — the only loss in the pipeline is the f32
        // round-trip, and `v / 255 * 255` followed by the standard
        // half-up rounding is provably idempotent for `v ∈ [0, 255]`.
        let img = build(7, 5, |x, y| {
            Rgba::new(
                (x.wrapping_mul(37) ^ y.wrapping_mul(13)) as u8,
                (x.wrapping_mul(7).wrapping_add(y)) as u8,
                (x.wrapping_add(y.wrapping_mul(53))) as u8,
                (x.wrapping_mul(y).wrapping_add(31)) as u8,
            )
        });
        let out = color_matrix(&img, 7, 5, &ColorMatrix::identity());
        assert_eq!(out, img);
    }

    #[test]
    fn saturate_one_is_identity_on_arbitrary_colours() {
        // s = 1 ⇒ the spec saturate matrix collapses to the 4×4
        // identity (every off-diagonal RGB term cancels: 0.213·1 −
        // 0.213·1 = 0, etc.). Therefore saturate(1) applied to any
        // colour must reproduce it within the 1-LSB rounding window
        // (one round-trip through normalise→multiply→quantise).
        let img = build(4, 4, |x, y| {
            Rgba::new((x * 60) as u8, (y * 80) as u8, 17, 200)
        });
        let m = ColorMatrix::saturate(1.0);
        let out = color_matrix(&img, 4, 4, &m);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "saturate(1) differs by {d} at byte {i}");
        }
    }

    #[test]
    fn saturate_zero_collapses_rgb_to_luminance() {
        // s = 0 ⇒ every RGB output row is the same luminance combination
        // (0.213·R + 0.715·G + 0.072·B). Apply to a fixed (200, 100, 50)
        // pixel and check the three colour channels emerge equal.
        let img = build(1, 1, |_, _| Rgba::new(200, 100, 50, 255));
        let m = ColorMatrix::saturate(0.0);
        let out = color_matrix(&img, 1, 1, &m);
        assert_eq!(out[0], out[1], "R != G after full desaturation");
        assert_eq!(out[1], out[2], "G != B after full desaturation");
        assert_eq!(out[3], 255, "alpha must pass through");
        // Spot-check the analytic value: 0.213·200 + 0.715·100 + 0.072·50
        // = 42.6 + 71.5 + 3.6 = 117.7 / 255 ⇒ quantises to 118.
        let expected = ((0.213_f32 * 200.0 + 0.715 * 100.0 + 0.072 * 50.0)
            .clamp(0.0, 255.0)
            .round()) as u8;
        let d = (out[0] as i32 - expected as i32).abs();
        assert!(
            d <= 1,
            "luminance scalar off by {d} (got {} vs {expected})",
            out[0]
        );
    }

    #[test]
    fn hue_rotate_zero_is_identity() {
        // θ = 0 ⇒ cos = 1, sin = 0. const + 1·cos_m = identity 3×3
        // (verified by hand: row 0 → (0.213+0.787, 0.715−0.715,
        // 0.072−0.072) = (1, 0, 0); row 1 → (0, 1, 0); row 2 → (0, 0,
        // 1)). Therefore hue_rotate(0) is the identity modulo the
        // quantise round-off (≤ 1 LSB).
        let img = build(5, 3, |x, y| {
            Rgba::new((x * 41) as u8, (y * 71) as u8, 100, 200)
        });
        let m = ColorMatrix::hue_rotate(0.0);
        let out = color_matrix(&img, 5, 3, &m);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "hue_rotate(0) byte {i} differs by {d}");
        }
    }

    #[test]
    fn hue_rotate_preserves_grey_axis() {
        // A pure grey pixel (R == G == B) lies on the achromatic axis,
        // which is the rotation axis of `hueRotate`. Therefore any
        // rotation must leave a grey pixel grey (and very nearly the
        // same brightness — modulo the inevitable round-off in the
        // float multiply, the coefficients have ≤ 3 dp so the rounding
        // can be up to ~1 LSB).
        let img = build(1, 1, |_, _| Rgba::new(140, 140, 140, 255));
        for theta in [30.0_f32, 90.0, 120.0, 180.0, 270.0, -45.0] {
            let m = ColorMatrix::hue_rotate(theta);
            let out = color_matrix(&img, 1, 1, &m);
            assert!(
                (out[0] as i32 - out[1] as i32).abs() <= 1,
                "θ={theta}: R != G after rotation"
            );
            assert!(
                (out[1] as i32 - out[2] as i32).abs() <= 1,
                "θ={theta}: G != B after rotation"
            );
            assert!(
                (out[0] as i32 - 140).abs() <= 2,
                "θ={theta}: grey brightness drift {} → {}",
                140,
                out[0]
            );
            assert_eq!(out[3], 255, "alpha must pass through hue rotation");
        }
    }

    #[test]
    fn hue_rotate_full_turn_is_identity() {
        // 360° rotation ⇒ cos = 1, sin = 0 again ⇒ same as θ = 0.
        // Round-off ≤ 1 LSB on each channel.
        let img = build(4, 3, |x, y| {
            Rgba::new((x * 60) as u8, (y * 80) as u8, ((x + y) * 30) as u8, 255)
        });
        let m = ColorMatrix::hue_rotate(360.0);
        let out = color_matrix(&img, 4, 3, &m);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "θ=360 byte {i} differs by {d}");
        }
    }

    #[test]
    fn luminance_to_alpha_zeros_rgb_and_writes_luminance_to_alpha() {
        // The luminanceToAlpha matrix is the §15.10-mandated fixed
        // matrix that produces transparent black RGB and alpha equal
        // to the BT.709 luminance of the input. Verify both halves.
        let img = build(1, 1, |_, _| Rgba::new(255, 128, 64, 200));
        let m = ColorMatrix::luminance_to_alpha();
        let out = color_matrix(&img, 1, 1, &m);
        assert_eq!(out[0], 0, "R must be cleared");
        assert_eq!(out[1], 0, "G must be cleared");
        assert_eq!(out[2], 0, "B must be cleared");
        // Expected alpha: 0.2125·255 + 0.7154·128 + 0.0721·64
        // = 54.1875 + 91.5712 + 4.6144 = 150.3731 ⇒ quantises to 150.
        let expected = ((0.2125_f32 * 255.0 + 0.7154 * 128.0 + 0.0721 * 64.0).round()) as u8;
        let d = (out[3] as i32 - expected as i32).abs();
        assert!(
            d <= 1,
            "luminance-to-alpha α off by {d} (got {} vs {expected})",
            out[3]
        );
    }

    #[test]
    fn op_dispatch_matches_direct_construction() {
        // The `color_matrix_op` thin wrapper must produce byte-exact
        // output relative to `color_matrix` on the resolved matrix —
        // this is the documented invariant `ColorMatrix::from_op`.
        let img = build(3, 2, |x, y| {
            Rgba::new((x * 90) as u8, (y * 110) as u8, 60, 240)
        });
        let user = ColorMatrix::identity(); // unused for non-Matrix ops
        for op in [
            ColorMatrixOp::Matrix,
            ColorMatrixOp::Saturate(0.5),
            ColorMatrixOp::HueRotate(123.0),
            ColorMatrixOp::LuminanceToAlpha,
        ] {
            let direct = color_matrix(&img, 3, 2, &ColorMatrix::from_op(op, &user));
            let via_op = color_matrix_op(&img, 3, 2, op, &user);
            assert_eq!(direct, via_op, "dispatch mismatch for {op:?}");
        }
    }

    #[test]
    fn channel_clamping_is_applied() {
        // A matrix that would produce R' = 2·R (out of gamut for any
        // R > 127) must clamp to 255, not wrap.
        let m = ColorMatrix([
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]);
        let img = build(1, 1, |_, _| Rgba::new(200, 0, 0, 255));
        let out = color_matrix(&img, 1, 1, &m);
        assert_eq!(out[0], 255, "R must clamp at upper bound");
        // A negative-doubling matrix must clamp to 0.
        let m = ColorMatrix([
            [-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]);
        let out = color_matrix(&img, 1, 1, &m);
        assert_eq!(out[0], 0, "R must clamp at lower bound");
    }

    #[test]
    fn bias_column_adds_constant_offset() {
        // The 5th column (the homogeneous-1 multiplier) lets a matrix
        // add a constant offset. A row of (0, 0, 0, 0, 0.5) should
        // produce a constant 0.5 → 128 in that channel regardless of
        // the input pixel.
        let m = ColorMatrix([
            [0.0, 0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.75],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]);
        let img = build(2, 2, |x, y| {
            Rgba::new((x * 100) as u8, (y * 50) as u8, 33, 200)
        });
        let out = color_matrix(&img, 2, 2, &m);
        for chunk in out.chunks_exact(4) {
            assert!(
                (chunk[0] as i32 - 128).abs() <= 1,
                "R bias drift: {}",
                chunk[0]
            );
            assert!(
                (chunk[1] as i32 - 64).abs() <= 1,
                "G bias drift: {}",
                chunk[1]
            );
            assert!(
                (chunk[2] as i32 - 191).abs() <= 1,
                "B bias drift: {}",
                chunk[2]
            );
            assert_eq!(chunk[3], 200, "alpha pass-through");
        }
    }

    #[test]
    fn color_matrix_pixels_wrapper_matches_byte_api() {
        let w = 4u32;
        let h = 3u32;
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for i in 0..(w * h) as u8 {
            let r = i.wrapping_mul(7);
            let g = i.wrapping_mul(17).wrapping_add(3);
            let b = i.wrapping_mul(31).wrapping_add(5);
            let a = 255u8;
            bytes.extend_from_slice(&[r, g, b, a]);
            pixels.push(Rgba::new(r, g, b, a));
        }
        for m in [
            ColorMatrix::identity(),
            ColorMatrix::saturate(0.3),
            ColorMatrix::hue_rotate(72.0),
            ColorMatrix::luminance_to_alpha(),
        ] {
            let bytes_out = color_matrix(&bytes, w, h, &m);
            let pix_out = color_matrix_pixels(&pixels, w, h, &m);
            let recombined: Vec<u8> = pix_out.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
            assert_eq!(bytes_out, recombined, "wrapper / byte API divergence");
        }
    }

    #[test]
    fn empty_image_returns_empty_buffer() {
        let img: Vec<u8> = Vec::new();
        let out = color_matrix(&img, 0, 0, &ColorMatrix::saturate(0.5));
        assert!(
            out.is_empty(),
            "zero-area image must produce zero-byte output"
        );
    }

    #[test]
    #[should_panic(expected = "color_matrix: src.len() ==")]
    fn color_matrix_panics_on_wrong_length() {
        let bad = vec![0u8; 7]; // not w·h·4 for w=2, h=2
        let _ = color_matrix(&bad, 2, 2, &ColorMatrix::identity());
    }
}
