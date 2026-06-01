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
//! * **Gaussian blur** — `feGaussianBlur` from SVG 1.1 §15.17.
//!   Convolves a packed-RGBA buffer with the separable normalised
//!   Gaussian `G(x, y) = H(x)·I(y)` where
//!   `H(x) = exp(-x²/(2s²)) / sqrt(2π·s²)` and
//!   `I(y) = exp(-y²/(2t²)) / sqrt(2π·t²)`. For small standard
//!   deviations (`s < 2.0`) the spec's "separable convolution" route is
//!   taken directly with a discrete kernel of half-width `ceil(3·s)`
//!   (≈99.7% of the analytical mass). For `s ≥ 2.0` the spec's
//!   three-box-blur approximation is used:
//!   `d = floor(s · 3·sqrt(2π)/4 + 0.5)`; if `d` is odd three centred
//!   box-blurs of size `d` are composed, if `d` is even two box-blurs
//!   of size `d` (centred on the left and right pixel boundaries
//!   respectively) are composed with one box-blur of size `d + 1`.
//!   Boundary handling is clamp-to-edge for both modes.
//!
//! * **Component transfer** — `feComponentTransfer` from SVG 1.1
//!   §15.11. Per-pixel, per-channel transfer function — each of R, G,
//!   B, A carries one of the five §15.11 modes (`identity`, `table`
//!   piecewise-linear over `N+1` values, `discrete` step over `N`
//!   values, `linear` affine `slope · C + intercept`, `gamma`
//!   `amplitude · C^exponent + offset`). The operation is performed
//!   on un-premultiplied channels normalised to `[0, 1]`, then
//!   clamped to that range, then re-quantised back to `u8`.
//!
//! * **Composite** — `feComposite` from SVG 1.1 §15.12. Combines two
//!   equal-sized RGBA buffers pixel-wise. The five Porter–Duff
//!   operators (`over` / `in` / `out` / `atop` / `xor`) are expressed
//!   through the premultiplied-alpha blend-factor pair `(Fa, Fb)` —
//!   `co = ca·Fa + cb·Fb`, `αo = αa·Fa + αb·Fb` — consistent with the
//!   §14.2 simple-alpha-compositing formula (`Cr' = (1 − Ea)·Cr + Er`,
//!   i.e. `over`) already used by [`crate::composite_rgba_premultiplied`].
//!   The `arithmetic` operator evaluates the §15.12 per-channel formula
//!   `result = k1·i1·i2 + k2·i1 + k3·i2 + k4` on premultiplied channels,
//!   clamped to `[0, 1]`. Selectable via [`CompositeOp`].
//!
//! # Deferred
//!
//! Drop shadow (`feDropShadow`), `feConvolveMatrix`, `feTurbulence`
//! (Perlin), `feDisplacementMap`, `feSpecularLighting`,
//! `feDiffuseLighting`.
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
//! reproduced verbatim from the spec table); §15.17 for the Gaussian
//! blur kernel definition and the three-box-blur approximation
//! formula `d = floor(s · 3·sqrt(2π)/4 + 0.5)` for `s ≥ 2.0`; §15.11
//! for the five `feComponentTransfer` modes (identity / table /
//! discrete / linear / gamma); §15.12 for the `feComposite` arithmetic
//! formula `result = k1·i1·i2 + k2·i1 + k3·i2 + k4` (reproduced
//! verbatim) and the §14.2 premultiplied simple-alpha-compositing
//! algebra (`Cr' = (1 − Ea)·Cr + Er`) from which the Porter–Duff
//! `over` / `in` / `out` / `atop` / `xor` blend-factor pairs are
//! derived. No `image` / `imageproc` / `opencv` /
//! `cairo` / `skia` / `resvg` / `librsvg` source consulted.

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

// ---------------------------------------------------------------------------
// feGaussianBlur — SVG 1.1 §15.17
// ---------------------------------------------------------------------------

/// Threshold above which the SVG 1.1 §15.17 three-box-blur approximation
/// is used in place of the direct discrete Gaussian convolution. The
/// spec says "For larger values of 's' (s >= 2.0), an approximation can
/// be used"; the same `2.0` value is exposed here so that callers and
/// tests can reason about which mode a given `stdDeviation` will take.
pub const GAUSSIAN_BLUR_BOX_THRESHOLD: f32 = 2.0;

/// SVG 1.1 §15.17 `<feGaussianBlur>`: separable Gaussian blur on a
/// packed-RGBA byte buffer.
///
/// `width × height` is the image extent in pixels. `src` is a
/// packed-RGBA byte buffer of exactly `width * height * 4` bytes in
/// row-major order — the same packing produced by
/// [`crate::Renderer::render`].
///
/// `std_x` and `std_y` are the X / Y standard deviations of the
/// Gaussian, in pixels. Per the spec, a negative value is an error and
/// will panic; a value of zero disables the effect along that axis (a
/// zero on both axes returns the input unchanged).
///
/// # Algorithm
///
/// Per the spec the kernel is the normalised separable Gaussian
/// `G(x, y) = H(x)·I(y)` with
/// `H(x) = exp(-x²/(2s²)) / sqrt(2π·s²)` and `I(y) =
/// exp(-y²/(2t²)) / sqrt(2π·t²)`. Two implementation modes are used:
///
/// * **Direct mode (`s < 2.0`)** — separable convolution with a
///   discrete kernel of half-width `r = ceil(3·s)` (capturing
///   ≈99.7% of the analytical Gaussian mass). The kernel is
///   re-normalised after discretisation so that the row sum equals
///   exactly 1.0 in `f32`, eliminating the residual DC error from
///   truncating the tails.
/// * **Box mode (`s ≥ 2.0`)** — the §15.17 three-box-blur
///   approximation. Let `d = floor(s · 3·sqrt(2π)/4 + 0.5)`. If `d`
///   is odd, three centred box-blurs of size `d` are composed; if
///   `d` is even, two box-blurs of size `d` (centred on the pixel
///   boundary on the left and right respectively) and one
///   box-blur of size `d + 1` centred on the pixel are composed.
///   This is the formula given verbatim by the spec and produces
///   a piece-wise quadratic kernel that matches the Gaussian to
///   within ≈3%.
///
/// Both modes are fully separable: an X-pass over rows followed by a
/// Y-pass over columns. Boundary handling is **clamp-to-edge** (the
/// same convention used by [`morphology`]).
///
/// # Panics
///
/// * If `src.len() != width as usize * height as usize * 4`.
/// * If `std_x < 0.0` or `std_y < 0.0` (spec error).
/// * If either standard deviation is NaN.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same dimensions. With
/// `std_x == 0.0` and `std_y == 0.0` the input is returned unchanged.
pub fn gaussian_blur(src: &[u8], width: u32, height: u32, std_x: f32, std_y: f32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("gaussian_blur: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "gaussian_blur: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );
    assert!(
        !std_x.is_nan() && !std_y.is_nan(),
        "gaussian_blur: stdDeviation must not be NaN"
    );
    assert!(
        std_x >= 0.0 && std_y >= 0.0,
        "gaussian_blur: stdDeviation must be non-negative (spec error processing): \
         got std_x={std_x}, std_y={std_y}"
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }
    if std_x == 0.0 && std_y == 0.0 {
        return src.to_vec();
    }

    let mut buf = src.to_vec();

    if std_x > 0.0 {
        if std_x < GAUSSIAN_BLUR_BOX_THRESHOLD {
            let kernel = build_gaussian_kernel(std_x);
            gaussian_separable_pass(&mut buf, w, h, &kernel, Axis::X);
        } else {
            box_blur_three_pass(&mut buf, w, h, box_sizes_for_std(std_x), Axis::X);
        }
    }
    if std_y > 0.0 {
        if std_y < GAUSSIAN_BLUR_BOX_THRESHOLD {
            let kernel = build_gaussian_kernel(std_y);
            gaussian_separable_pass(&mut buf, w, h, &kernel, Axis::Y);
        } else {
            box_blur_three_pass(&mut buf, w, h, box_sizes_for_std(std_y), Axis::Y);
        }
    }
    buf
}

/// Convenience wrapper that runs [`gaussian_blur`] on a slice of [`Rgba`]
/// pixels and returns a `Vec<Rgba>` of the same length.
pub fn gaussian_blur_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    std_x: f32,
    std_y: f32,
) -> Vec<Rgba> {
    assert_eq!(
        src.len(),
        width as usize * height as usize,
        "gaussian_blur_pixels: src.len() == {} but width*height == {}",
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
    let out = gaussian_blur(&bytes, width, height, std_x, std_y);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

/// Build a discrete 1-D normalised Gaussian kernel for the given
/// standard deviation. Half-width is `ceil(3·s)`, capturing ≈99.7% of
/// the analytical mass; the kernel is renormalised so that the sum
/// equals exactly `1.0` in `f32`, which removes the DC bias that would
/// otherwise creep in from truncating the tails.
fn build_gaussian_kernel(std: f32) -> Vec<f32> {
    debug_assert!(std > 0.0);
    let half = (std * 3.0).ceil() as usize;
    let len = 2 * half + 1;
    let mut k = Vec::with_capacity(len);
    let two_sigma_sq = 2.0 * std * std;
    // Per the §15.17 closed form: H(x) = exp(-x²/(2s²)) / sqrt(2π·s²).
    // The leading 1/sqrt(2π·s²) factor is a constant per-kernel — it
    // drops out cleanly when we renormalise below, so we omit it from
    // the discrete samples.
    for i in 0..len {
        let x = i as f32 - half as f32;
        k.push((-x * x / two_sigma_sq).exp());
    }
    let sum: f32 = k.iter().sum();
    debug_assert!(sum > 0.0);
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Sliding-window separable Gaussian pass along the requested axis,
/// in-place on a packed-RGBA byte buffer. Channels are processed
/// independently; the accumulator is `f32` per-channel; boundary
/// samples reuse the nearest edge pixel (clamp-to-edge).
fn gaussian_separable_pass(buf: &mut [u8], w: usize, h: usize, kernel: &[f32], axis: Axis) {
    debug_assert_eq!(buf.len(), w * h * 4);
    let half = (kernel.len() - 1) / 2;
    let (stride, primary, secondary) = match axis {
        Axis::X => (4usize, w, h),
        Axis::Y => (w * 4, h, w),
    };
    let mut line_in = vec![0u8; primary * 4];
    let mut line_out = vec![0u8; primary * 4];
    let imax = primary as isize - 1;
    for s in 0..secondary {
        // Gather one "line" (row for X-pass, column for Y-pass) into a
        // contiguous scratch buffer so the inner loop is sequential.
        let start = match axis {
            Axis::X => s * w * 4,
            Axis::Y => s * 4,
        };
        for i in 0..primary {
            let off = start + i * stride;
            line_in[i * 4..i * 4 + 4].copy_from_slice(&buf[off..off + 4]);
        }

        for i in 0..primary {
            let mut acc = [0f32; 4];
            for (k_idx, w_coef) in kernel.iter().enumerate() {
                let pi = (i as isize + k_idx as isize - half as isize).clamp(0, imax) as usize;
                let p = &line_in[pi * 4..pi * 4 + 4];
                acc[0] += p[0] as f32 * w_coef;
                acc[1] += p[1] as f32 * w_coef;
                acc[2] += p[2] as f32 * w_coef;
                acc[3] += p[3] as f32 * w_coef;
            }
            let off = i * 4;
            line_out[off] = quantise_u8(acc[0]);
            line_out[off + 1] = quantise_u8(acc[1]);
            line_out[off + 2] = quantise_u8(acc[2]);
            line_out[off + 3] = quantise_u8(acc[3]);
        }

        for i in 0..primary {
            let off = start + i * stride;
            buf[off..off + 4].copy_from_slice(&line_out[i * 4..i * 4 + 4]);
        }
    }
}

/// Round a non-negative `f32` channel accumulator to `u8` with
/// clamp-to-[0, 255]. Negative inputs (which Gaussian blur should not
/// produce but the helper is conservative anyway) clamp to `0`.
#[inline]
fn quantise_u8(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let clamped = v.clamp(0.0, 255.0);
    (clamped + 0.5) as u8
}

/// Compute the three box-blur sizes for the §15.17 approximation.
///
/// Per the spec:
///
/// > let `d = floor(s · 3·sqrt(2π)/4 + 0.5)`
/// > * if `d` is odd, use three box-blurs of size `d`, centered on the
/// >   output pixel.
/// > * if `d` is even, two box-blurs of size `d` (the first one
/// >   centered on the pixel boundary between the output pixel and the
/// >   one to the left, the second one centered on the pixel boundary
/// >   between the output pixel and the one to the right) and one box
/// >   blur of size `d + 1` centered on the output pixel.
///
/// Returns `[BoxKind; 3]` describing the three passes in order. The
/// "kind" carries both the box size and, for even-`d` passes, the
/// side the boundary is offset toward.
fn box_sizes_for_std(std: f32) -> [BoxKind; 3] {
    debug_assert!(std >= GAUSSIAN_BLUR_BOX_THRESHOLD);
    // 3·sqrt(2π)/4 ≈ 1.8800316...
    let d_f = std * (3.0 * (2.0 * std::f32::consts::PI).sqrt() / 4.0) + 0.5;
    let d = d_f.floor() as u32;
    let d = d.max(1);
    if d % 2 == 1 {
        [
            BoxKind::Centered(d),
            BoxKind::Centered(d),
            BoxKind::Centered(d),
        ]
    } else {
        [
            BoxKind::OffsetLeft(d),
            BoxKind::OffsetRight(d),
            BoxKind::Centered(d + 1),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoxKind {
    /// Box-blur of (odd) size `n`, kernel centred on the output pixel.
    Centered(u32),
    /// Box-blur of (even) size `n`, kernel centred on the pixel
    /// boundary between the output pixel and the one to the **left**.
    /// In integer indices, the window covers `[i - n/2 .. i + n/2 - 1]`
    /// inclusive (i.e. one more sample to the left than to the right).
    OffsetLeft(u32),
    /// Box-blur of (even) size `n`, kernel centred on the pixel
    /// boundary between the output pixel and the one to the **right**.
    /// In integer indices the window covers
    /// `[i - n/2 + 1 .. i + n/2]` inclusive.
    OffsetRight(u32),
}

/// Apply the three box-blur passes that approximate one Gaussian pass,
/// in-place along the requested axis.
fn box_blur_three_pass(buf: &mut [u8], w: usize, h: usize, kinds: [BoxKind; 3], axis: Axis) {
    for k in kinds {
        box_blur_pass(buf, w, h, k, axis);
    }
}

/// Run one box-blur pass along `axis` on the packed-RGBA buffer using
/// a rolling-sum O(W·H) per pass (per channel). Boundary handling is
/// clamp-to-edge.
fn box_blur_pass(buf: &mut [u8], w: usize, h: usize, kind: BoxKind, axis: Axis) {
    debug_assert_eq!(buf.len(), w * h * 4);
    let (n, lo_off, hi_off) = match kind {
        BoxKind::Centered(n) => {
            let half = (n / 2) as isize;
            (n as usize, -half, half)
        }
        BoxKind::OffsetLeft(n) => {
            let half = (n / 2) as isize;
            // Window inclusive of [i - half .. i + half - 1], total `n`
            // samples; the window centroid sits half a pixel to the
            // left of `i`.
            (n as usize, -half, half - 1)
        }
        BoxKind::OffsetRight(n) => {
            let half = (n / 2) as isize;
            (n as usize, -half + 1, half)
        }
    };
    debug_assert_eq!((hi_off - lo_off + 1) as usize, n);

    let (stride, primary, secondary) = match axis {
        Axis::X => (4usize, w, h),
        Axis::Y => (w * 4, h, w),
    };
    let mut line_in = vec![0u8; primary * 4];
    let mut line_out = vec![0u8; primary * 4];
    let imax = primary as isize - 1;
    let inv_n = 1.0f32 / n as f32;
    for s in 0..secondary {
        let start = match axis {
            Axis::X => s * w * 4,
            Axis::Y => s * 4,
        };
        for i in 0..primary {
            let off = start + i * stride;
            line_in[i * 4..i * 4 + 4].copy_from_slice(&buf[off..off + 4]);
        }

        // Rolling sum: prime the accumulator with the window at i=0,
        // then for each subsequent i add the new right edge and
        // subtract the old left edge. Edge handling uses clamp-to-edge
        // (samples outside [0, primary) reuse index 0 / imax).
        let mut acc = [0u32; 4];
        for k_off in lo_off..=hi_off {
            let pi = k_off.clamp(0, imax) as usize;
            let p = &line_in[pi * 4..pi * 4 + 4];
            acc[0] += p[0] as u32;
            acc[1] += p[1] as u32;
            acc[2] += p[2] as u32;
            acc[3] += p[3] as u32;
        }

        for i in 0..primary {
            // Write current sample.
            let off = i * 4;
            line_out[off] = (acc[0] as f32 * inv_n + 0.5) as u8;
            line_out[off + 1] = (acc[1] as f32 * inv_n + 0.5) as u8;
            line_out[off + 2] = (acc[2] as f32 * inv_n + 0.5) as u8;
            line_out[off + 3] = (acc[3] as f32 * inv_n + 0.5) as u8;

            // Advance: incoming sample at i + 1 + hi_off, outgoing at
            // i + lo_off.
            if i + 1 < primary {
                let next_i = (i + 1) as isize;
                let in_idx = (next_i + hi_off).clamp(0, imax) as usize;
                let out_idx = (next_i + lo_off - 1).clamp(0, imax) as usize;
                let p_in = &line_in[in_idx * 4..in_idx * 4 + 4];
                let p_out = &line_in[out_idx * 4..out_idx * 4 + 4];
                acc[0] = acc[0] + p_in[0] as u32 - p_out[0] as u32;
                acc[1] = acc[1] + p_in[1] as u32 - p_out[1] as u32;
                acc[2] = acc[2] + p_in[2] as u32 - p_out[2] as u32;
                acc[3] = acc[3] + p_in[3] as u32 - p_out[3] as u32;
            }
        }

        for i in 0..primary {
            let off = start + i * stride;
            buf[off..off + 4].copy_from_slice(&line_out[i * 4..i * 4 + 4]);
        }
    }
}

#[cfg(test)]
mod gaussian_blur_tests {
    use super::*;

    fn solid(w: u32, h: u32, c: Rgba) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&[c.r, c.g, c.b, c.a]);
        }
        v
    }

    #[test]
    fn zero_stddev_is_identity() {
        let mut img = solid(5, 4, Rgba::new(10, 20, 30, 40));
        // Perturb a couple of pixels so we'd notice smearing.
        img[5 * 4] = 200;
        img[7 * 4 + 2] = 250;
        let out = gaussian_blur(&img, 5, 4, 0.0, 0.0);
        assert_eq!(out, img);
    }

    #[test]
    fn solid_image_is_invariant_for_any_stddev() {
        // Gaussian convolution of a constant image gives the same
        // constant; only the boundary handling could perturb it. With
        // clamp-to-edge a constant image is perfectly preserved.
        let img = solid(11, 7, Rgba::new(80, 120, 200, 255));
        for &(sx, sy) in &[
            (0.3, 0.0),
            (0.0, 0.7),
            (0.5, 0.5),
            (1.0, 1.0),
            (1.7, 1.9),
            (2.0, 2.0),
            (3.5, 0.0),
            (4.0, 6.0),
            (8.0, 8.0),
        ] {
            let out = gaussian_blur(&img, 11, 7, sx, sy);
            assert_eq!(out, img, "(sx, sy) = ({sx}, {sy})");
        }
    }

    #[test]
    fn small_stddev_uses_direct_kernel_branch() {
        // For s < 2.0 we use the direct discrete convolution. Sanity:
        // a centred bright pixel diffuses into a symmetric pattern.
        let w = 9u32;
        let h = 9u32;
        let mut img = solid(w, h, Rgba::new(0, 0, 0, 0));
        let cx = 4usize;
        let cy = 4usize;
        let off = (cy * w as usize + cx) * 4;
        img[off..off + 4].copy_from_slice(&[255, 255, 255, 255]);

        let out = gaussian_blur(&img, w, h, 1.0, 1.0);

        // Centre pixel must still be the brightest.
        let centre = out[off];
        assert!(centre > 0);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let o = (y * w as usize + x) * 4;
                if (x, y) != (cx, cy) {
                    assert!(
                        out[o] <= centre,
                        "centre {centre} must dominate ({x},{y}) = {}",
                        out[o]
                    );
                }
            }
        }

        // Four-fold symmetry: the diffusion is identical along ±x, ±y.
        for d in 1..=3usize {
            let n = out[((cy - d) * w as usize + cx) * 4];
            let s = out[((cy + d) * w as usize + cx) * 4];
            let e = out[(cy * w as usize + (cx + d)) * 4];
            let west = out[(cy * w as usize + (cx - d)) * 4];
            assert_eq!(n, s, "vertical symmetry at d={d}");
            assert_eq!(e, west, "horizontal symmetry at d={d}");
            // Separable kernel ⇒ horizontal and vertical neighbours at
            // the same distance get the same coefficient.
            assert_eq!(n, e, "x/y symmetry at d={d}");
        }
    }

    #[test]
    fn large_stddev_uses_box_approximation_branch() {
        // For s >= 2.0 we use the three-box-blur approximation. A
        // bright impulse should spread into a roughly bell-shaped
        // pattern that's centred on the original pixel.
        let w = 21u32;
        let h = 21u32;
        let mut img = solid(w, h, Rgba::new(0, 0, 0, 0));
        let cx = 10usize;
        let cy = 10usize;
        let off = (cy * w as usize + cx) * 4;
        img[off..off + 4].copy_from_slice(&[255, 255, 255, 255]);

        let out = gaussian_blur(&img, w, h, 3.0, 3.0);

        // Diffusion is monotone non-increasing radially (we only check
        // axis-aligned because box-blur cascades aren't exactly
        // rotation-invariant — but along the axes monotonicity holds).
        for d in 1..=4usize {
            let a = out[(cy * w as usize + cx + d - 1) * 4];
            let b = out[(cy * w as usize + cx + d) * 4];
            assert!(b <= a, "horizontal monotonicity broken at d={d}: {a}→{b}");
            let a = out[((cy + d - 1) * w as usize + cx) * 4];
            let b = out[((cy + d) * w as usize + cx) * 4];
            assert!(b <= a, "vertical monotonicity broken at d={d}: {a}→{b}");
        }

        // Centre got dimmer than the original impulse.
        let centre = out[off];
        assert!(
            centre < 255,
            "centre should diffuse below the impulse but got {centre}"
        );
        assert!(centre > 0);
    }

    #[test]
    fn separable_decomposition_matches_two_separate_passes() {
        // gaussian_blur(s_x, s_y) ≡ gaussian_blur(s_x, 0) followed by
        // gaussian_blur(0, s_y). (Both branches of the algorithm
        // satisfy this because separability is the defining property of
        // the Gaussian kernel; the spec literally says "this can be
        // implemented as a separable convolution".)
        let w = 11u32;
        let h = 9u32;
        let mut img = vec![0u8; (w * h * 4) as usize];
        // Pseudo-random deterministic content.
        let mut acc: u32 = 0x1234_5678;
        for byte in &mut img {
            acc = acc.wrapping_mul(1_103_515_245).wrapping_add(12345);
            *byte = (acc >> 16) as u8;
        }
        for &(sx, sy) in &[(0.7, 0.5), (1.5, 1.0), (2.0, 2.0), (3.0, 4.0)] {
            let combined = gaussian_blur(&img, w, h, sx, sy);
            let stepwise_x = gaussian_blur(&img, w, h, sx, 0.0);
            let stepwise = gaussian_blur(&stepwise_x, w, h, 0.0, sy);
            assert_eq!(
                combined, stepwise,
                "separability broke at (sx={sx}, sy={sy})"
            );
        }
    }

    #[test]
    fn axis_only_blur_preserves_orthogonal_axis() {
        // A horizontal-only Gaussian blur must leave the vertical
        // signal alone. We set up a buffer with one bright row at
        // y = 0 and verify that no other row gains any energy from
        // the X-only pass.
        let w = 9u32;
        let h = 5u32;
        let mut img = solid(w, h, Rgba::new(0, 0, 0, 0));
        for x in 0..w as usize {
            let off = x * 4;
            img[off..off + 4].copy_from_slice(&[200, 100, 50, 255]);
        }
        let out = gaussian_blur(&img, w, h, 1.5, 0.0);
        // Row 0 sits inside the kernel of itself only; rows 1..h must
        // remain transparent black (they had no source energy).
        for y in 1..h as usize {
            for x in 0..w as usize {
                let off = (y * w as usize + x) * 4;
                assert_eq!(
                    &out[off..off + 4],
                    &[0, 0, 0, 0],
                    "y={y} x={x} should still be transparent"
                );
            }
        }
        // And the bright row must still be the bright colour exactly
        // (clamp-to-edge horizontally + a row that's constant in x).
        for x in 0..w as usize {
            let off = x * 4;
            assert_eq!(&out[off..off + 4], &[200, 100, 50, 255]);
        }
    }

    #[test]
    fn pixels_wrapper_matches_byte_api() {
        let w = 7u32;
        let h = 5u32;
        let pixels: Vec<Rgba> = (0..(w * h))
            .map(|i| Rgba::new((i * 7) as u8, (i * 11) as u8, (i * 13) as u8, 255))
            .collect();
        let mut bytes = Vec::with_capacity(pixels.len() * 4);
        for p in &pixels {
            bytes.extend_from_slice(&[p.r, p.g, p.b, p.a]);
        }

        for &(sx, sy) in &[(0.0, 0.0), (0.6, 0.6), (1.5, 0.0), (2.5, 2.5)] {
            let from_bytes = gaussian_blur(&bytes, w, h, sx, sy);
            let from_pixels = gaussian_blur_pixels(&pixels, w, h, sx, sy);
            let repacked: Vec<u8> = from_pixels
                .iter()
                .flat_map(|p| [p.r, p.g, p.b, p.a])
                .collect();
            assert_eq!(repacked, from_bytes, "(sx, sy) = ({sx}, {sy})");
        }
    }

    #[test]
    fn empty_image_returns_empty_buffer() {
        let out = gaussian_blur(&[], 0, 5, 1.0, 1.0);
        assert!(out.is_empty());
        let out = gaussian_blur(&[], 5, 0, 1.0, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn box_sizes_table_for_known_stddevs() {
        // s = 2.0 ⇒ d = floor(2.0 · 1.8800316 + 0.5) = floor(4.26..)
        // = 4. d is even ⇒ [OffsetLeft(4), OffsetRight(4), Centered(5)].
        let k = box_sizes_for_std(2.0);
        assert_eq!(k[0], BoxKind::OffsetLeft(4));
        assert_eq!(k[1], BoxKind::OffsetRight(4));
        assert_eq!(k[2], BoxKind::Centered(5));

        // s = 3.0 ⇒ d = floor(3.0 · 1.8800316 + 0.5) = floor(6.14..)
        // = 6. d is even ⇒ [OffsetLeft(6), OffsetRight(6), Centered(7)].
        let k = box_sizes_for_std(3.0);
        assert_eq!(k[0], BoxKind::OffsetLeft(6));
        assert_eq!(k[1], BoxKind::OffsetRight(6));
        assert_eq!(k[2], BoxKind::Centered(7));

        // s = 4.0 ⇒ d = floor(4.0 · 1.8800316 + 0.5) = floor(8.02..)
        // = 8. d is even ⇒ [OffsetLeft(8), OffsetRight(8), Centered(9)].
        let k = box_sizes_for_std(4.0);
        assert_eq!(k[0], BoxKind::OffsetLeft(8));
        assert_eq!(k[1], BoxKind::OffsetRight(8));
        assert_eq!(k[2], BoxKind::Centered(9));

        // s = 5.0 ⇒ d = floor(5.0 · 1.8800316 + 0.5) = floor(9.90..)
        // = 9. d is odd ⇒ three Centered(9).
        let k = box_sizes_for_std(5.0);
        assert_eq!(
            k,
            [
                BoxKind::Centered(9),
                BoxKind::Centered(9),
                BoxKind::Centered(9)
            ]
        );
    }

    #[test]
    fn kernel_normalisation_preserves_constant_image_exactly() {
        // The renormalisation step in build_gaussian_kernel ensures
        // that on an unsaturated constant image the result is byte-
        // exactly the input — i.e. no DC drift from truncating the
        // Gaussian tails. We already check the surface property in
        // `solid_image_is_invariant_for_any_stddev`; here we exercise
        // the kernel-construction invariant directly.
        for &s in &[0.3f32, 0.7, 1.0, 1.5, 1.99] {
            let k = build_gaussian_kernel(s);
            let sum: f32 = k.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "kernel for s={s} has sum {sum}");
            // The largest coefficient is the centre tap.
            let half = (k.len() - 1) / 2;
            assert!(k[half] > 0.0);
            for (i, &v) in k.iter().enumerate() {
                if i != half {
                    assert!(v <= k[half], "tap {i} {} > centre {}", v, k[half]);
                }
            }
            // Symmetry: k[half - d] == k[half + d].
            for d in 1..=half {
                let l = k[half - d];
                let r = k[half + d];
                assert!((l - r).abs() < 1e-12, "asymmetry at d={d}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "stdDeviation must be non-negative")]
    fn negative_stddev_panics() {
        let img = solid(2, 2, Rgba::new(0, 0, 0, 0));
        let _ = gaussian_blur(&img, 2, 2, -0.5, 1.0);
    }

    #[test]
    #[should_panic(expected = "gaussian_blur: src.len() ==")]
    fn wrong_length_panics() {
        let bad = vec![0u8; 7];
        let _ = gaussian_blur(&bad, 2, 2, 1.0, 1.0);
    }
}

// =====================================================================
// feComponentTransfer — SVG 1.1 §15.11
// =====================================================================
//
// §15.11 defines a per-pixel, per-channel transfer function applied
// independently to R, G, B, A. Each channel selects one of five
// `type=` modes; channel inputs and outputs are normalised to the
// `[0, 1]` interval, the operation is performed on un-premultiplied
// channels, and the result is clamped to `[0, 1]` before re-quantising
// back to `u8`. The five modes, in spec wording:
//
//   identity   C' = C
//   table      a piecewise-linear function over a list of N+1 values
//              v₀ … v_N; for a channel value C ∈ [0, 1], let
//              k = floor(C · N) clamped to [0, N-1], let
//              C_step = (C · N) − k; then C' = v_k + C_step · (v_{k+1} − v_k).
//              (At C = 1.0 the spec's clamp on k pins the interpolation
//              to the final segment v_{N-1}…v_N, so C' = v_N.)
//   discrete   a step function over a list of N values
//              v₀ … v_{N-1}; for C ∈ [0, 1], let k = floor(C · N)
//              clamped to [0, N-1]; then C' = v_k.
//   linear     C' = slope · C + intercept.
//   gamma      C' = amplitude · pow(C, exponent) + offset.
//
// `feFuncR` / `feFuncG` / `feFuncB` / `feFuncA` are independent: each
// channel carries its own [`TransferFunc`]. The API here lifts that
// directly into a [`ComponentTransfer`] struct of four
// [`TransferFunc`] values plus per-channel setters.

/// Per-channel transfer function for [`component_transfer`], mirroring
/// SVG 1.1 §15.11's `type=` attribute on the `<feFuncX>` child elements.
///
/// All inputs and outputs are normalised to `[0, 1]`; the implementation
/// performs the operation, clamps the result to that range, and only
/// then re-quantises back to `u8` for the byte API.
#[derive(Debug, Clone, PartialEq)]
pub enum TransferFunc {
    /// `type="identity"` — output equals input.
    Identity,
    /// `type="table"` — piecewise-linear interpolation across `N+1`
    /// `tableValues`. With `N = 0` (`tableValues` empty or a single
    /// value) the channel collapses to the identity (the spec's
    /// fall-back when the value list is too short to define a segment).
    Table(Vec<f32>),
    /// `type="discrete"` — step function across `N` `tableValues`. An
    /// empty list also collapses to identity (no step boundaries to
    /// place).
    Discrete(Vec<f32>),
    /// `type="linear"` — affine transform.
    Linear {
        /// Spec attribute `slope`.
        slope: f32,
        /// Spec attribute `intercept`.
        intercept: f32,
    },
    /// `type="gamma"` — `amplitude · C^exponent + offset`. `exponent`
    /// must be strictly positive (§15.11 error processing); a
    /// non-positive or NaN exponent panics.
    Gamma {
        /// Spec attribute `amplitude`.
        amplitude: f32,
        /// Spec attribute `exponent`. Must be strictly positive.
        exponent: f32,
        /// Spec attribute `offset`.
        offset: f32,
    },
}

impl TransferFunc {
    /// Evaluate the transfer function on a single normalised channel
    /// value `c ∈ [0, 1]`. The result is **not** pre-clamped — the
    /// caller is responsible for the final clamp / quantisation. Out-of-
    /// range inputs are passed through the spec formula unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `self` is [`TransferFunc::Gamma`] with a non-positive or
    /// NaN exponent. The §15.11 error-processing rule forbids both, and
    /// they make the `pow` undefined in general.
    #[inline]
    pub fn apply(&self, c: f32) -> f32 {
        match self {
            TransferFunc::Identity => c,
            TransferFunc::Table(v) => table_interpolate(v, c),
            TransferFunc::Discrete(v) => discrete_step(v, c),
            TransferFunc::Linear { slope, intercept } => slope * c + intercept,
            TransferFunc::Gamma {
                amplitude,
                exponent,
                offset,
            } => {
                assert!(
                    exponent.is_finite() && *exponent > 0.0,
                    "feComponentTransfer: gamma exponent must be strictly positive, got {exponent}"
                );
                // The spec leaves pow on negative inputs unspecified;
                // §15.11 normalises through clamp-to-[0,1] only at the
                // output, so a negative `c` from the caller (e.g. via
                // a feColorMatrix bias that left the channel slightly
                // negative) would produce NaN under `f32::powf`. Treat
                // negative inputs as zero before the pow — the
                // subsequent output clamp already handles the positive
                // side, and zero is the natural floor of the §15.11
                // gamma curve at the channel boundary.
                let base = c.max(0.0);
                amplitude * base.powf(*exponent) + offset
            }
        }
    }
}

/// Piecewise-linear table lookup for [`TransferFunc::Table`]. The spec
/// places `N + 1` tableValues at the breakpoints `k / N` for `k ∈ 0..=N`
/// and linearly interpolates between adjacent breakpoints.
#[inline]
fn table_interpolate(v: &[f32], c: f32) -> f32 {
    let n_plus_one = v.len();
    if n_plus_one < 2 {
        // Empty list or a single tableValue defines no segment ⇒
        // identity per the §15.11 fall-back.
        return c;
    }
    let n = (n_plus_one - 1) as f32;
    let scaled = c * n;
    if scaled <= 0.0 {
        return v[0];
    }
    if scaled >= n {
        return v[n_plus_one - 1];
    }
    let k = scaled.floor() as usize;
    let frac = scaled - k as f32;
    v[k] + frac * (v[k + 1] - v[k])
}

/// Step lookup for [`TransferFunc::Discrete`]. The spec partitions
/// `[0, 1]` into `N` half-open buckets `[k / N, (k+1) / N)` (with the
/// final bucket closed at `1`) and emits `v[k]`.
#[inline]
fn discrete_step(v: &[f32], c: f32) -> f32 {
    let n = v.len();
    if n == 0 {
        return c;
    }
    let scaled = c * n as f32;
    if scaled <= 0.0 {
        return v[0];
    }
    if scaled >= n as f32 {
        return v[n - 1];
    }
    let k = scaled.floor() as usize;
    v[k.min(n - 1)]
}

/// SVG 1.1 §15.11 `<feComponentTransfer>` configuration — one
/// [`TransferFunc`] per RGBA channel, applied independently per pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentTransfer {
    /// Red-channel function (`<feFuncR>`).
    pub r: TransferFunc,
    /// Green-channel function (`<feFuncG>`).
    pub g: TransferFunc,
    /// Blue-channel function (`<feFuncB>`).
    pub b: TransferFunc,
    /// Alpha-channel function (`<feFuncA>`).
    pub a: TransferFunc,
}

impl ComponentTransfer {
    /// Identity on every channel — equivalent to `<feComponentTransfer>`
    /// with no `<feFuncX>` children, which the spec defines as a
    /// no-op.
    pub fn identity() -> Self {
        Self {
            r: TransferFunc::Identity,
            g: TransferFunc::Identity,
            b: TransferFunc::Identity,
            a: TransferFunc::Identity,
        }
    }

    /// Replace the red-channel transfer function and return `self`
    /// (builder-style chain).
    pub fn with_r(mut self, f: TransferFunc) -> Self {
        self.r = f;
        self
    }

    /// Replace the green-channel transfer function and return `self`.
    pub fn with_g(mut self, f: TransferFunc) -> Self {
        self.g = f;
        self
    }

    /// Replace the blue-channel transfer function and return `self`.
    pub fn with_b(mut self, f: TransferFunc) -> Self {
        self.b = f;
        self
    }

    /// Replace the alpha-channel transfer function and return `self`.
    pub fn with_a(mut self, f: TransferFunc) -> Self {
        self.a = f;
        self
    }
}

impl Default for ComponentTransfer {
    fn default() -> Self {
        Self::identity()
    }
}

/// SVG 1.1 §15.11 `<feComponentTransfer>` — apply one [`TransferFunc`]
/// per channel to each pixel of a packed-RGBA `u8` buffer.
///
/// `width × height` is the input/output image extent in pixels. `src`
/// is a packed-RGBA byte buffer of exactly `width * height * 4` bytes
/// in row-major order.
///
/// **Pixel space.** §15.11 explicitly says the operation is performed
/// on the un-premultiplied colour channels, with channel values in
/// `[0, 1]`. This entry point therefore treats input bytes as
/// straight-alpha samples, normalises by `/ 255.0`, applies the
/// per-channel function, clamps to `[0, 1]`, and re-quantises back via
/// the same `round(x · 255)` rule the rest of the filter module uses
/// (`quantise_unit`).
///
/// Complexity is `O(W · H)` with a constant per-pixel cost — independent
/// of the size of any [`TransferFunc::Table`] / [`TransferFunc::Discrete`]
/// list (each lookup is `O(1)` against `floor`).
///
/// # Panics
///
/// * If `src.len() != width as usize * height as usize * 4`.
/// * If any channel uses [`TransferFunc::Gamma`] with a non-positive or
///   NaN exponent.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same dimensions. With every
/// channel set to [`TransferFunc::Identity`] the input is returned
/// bytewise unchanged (no f32 round-trip).
pub fn component_transfer(src: &[u8], width: u32, height: u32, ct: &ComponentTransfer) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("component_transfer: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "component_transfer: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    // Fast path: all-identity is a pure byte copy (avoids the f32
    // round-trip which, while idempotent under the half-up quantiser
    // for in-range integers, would unnecessarily reallocate).
    if matches!(ct.r, TransferFunc::Identity)
        && matches!(ct.g, TransferFunc::Identity)
        && matches!(ct.b, TransferFunc::Identity)
        && matches!(ct.a, TransferFunc::Identity)
    {
        return src.to_vec();
    }

    let mut out = Vec::with_capacity(src.len());
    for chunk in src.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let a = chunk[3] as f32 / 255.0;
        out.push(quantise_unit(ct.r.apply(r)));
        out.push(quantise_unit(ct.g.apply(g)));
        out.push(quantise_unit(ct.b.apply(b)));
        out.push(quantise_unit(ct.a.apply(a)));
    }
    out
}

/// Convenience wrapper that runs [`component_transfer`] on a slice of
/// [`Rgba`] pixels and returns a `Vec<Rgba>` of the same length.
/// Identical semantics — provided for callers that already have a typed
/// pixel buffer.
pub fn component_transfer_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    ct: &ComponentTransfer,
) -> Vec<Rgba> {
    assert_eq!(
        src.len(),
        width as usize * height as usize,
        "component_transfer_pixels: src.len() == {} but width*height == {}",
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
    let out = component_transfer(&bytes, width, height, ct);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

#[cfg(test)]
mod component_transfer_tests {
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
    fn identity_on_every_channel_is_bytewise_identity() {
        // §15.11 collapses to a no-op when every <feFuncX> is
        // `type="identity"`. The fast path avoids the f32 round-trip
        // and returns the input bytes verbatim.
        let img = build(5, 3, |x, y| {
            Rgba::new(
                (x * 47 + y * 11) as u8,
                (x * 17 + y * 5) as u8,
                (y.wrapping_mul(31).wrapping_add(x)) as u8,
                ((x + y) * 7) as u8,
            )
        });
        let ct = ComponentTransfer::identity();
        let out = component_transfer(&img, 5, 3, &ct);
        assert_eq!(out, img);
    }

    #[test]
    fn linear_slope_minus_one_intercept_one_inverts_red_only() {
        // Linear { slope: -1, intercept: 1 } on R encodes the §15.11
        // standard channel inversion: `R' = 1 - R`, equivalent to
        // `255 - R` after quantisation. The other channels carry the
        // identity, so they must round-trip byte-exact.
        let img = build(4, 4, |x, y| {
            Rgba::new((x * 60) as u8, (y * 80) as u8, 17, 200)
        });
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Linear {
            slope: -1.0,
            intercept: 1.0,
        });
        let out = component_transfer(&img, 4, 4, &ct);
        for i in (0..img.len()).step_by(4) {
            // R inverted (±1 LSB for the f32 round-trip).
            let want_r = 255 - img[i] as i32;
            let got_r = out[i] as i32;
            assert!(
                (got_r - want_r).abs() <= 1,
                "R at pixel {} got {got_r} want {want_r}",
                i / 4
            );
            // G, B, A untouched (the byte path is the identity).
            assert_eq!(out[i + 1], img[i + 1]);
            assert_eq!(out[i + 2], img[i + 2]);
            assert_eq!(out[i + 3], img[i + 3]);
        }
    }

    #[test]
    fn gamma_2_2_then_inverse_round_trips_in_mid_to_high_range() {
        // gamma(amp=1, exp=2.2, off=0) followed by its inverse
        // gamma(amp=1, exp=1/2.2, off=0) reproduces the analytical
        // input C exactly: `(C^2.2)^(1/2.2) = C`. The lossy step is
        // the pair of u8 quantisations on either side. The forward
        // curve compresses hard at the low end (e.g. 16/255 → 1/255
        // → 21/255 after inverse) because 8-bit precision is too
        // coarse to represent the bent samples below the curve's
        // knee. Restrict the round-trip assertion to mid-to-high
        // inputs (C ≥ 64/255) where the forward derivative is
        // moderate enough that the two ½-LSB quantisation errors
        // compose into ≤ 2 LSBs at the output. Low-end recovery is
        // a property of the bit depth, not the implementation; the
        // analytical correctness of the §15.11 formula is exercised
        // by the linear and table tests in this same suite.
        let img = build(8, 8, |x, y| {
            Rgba::new(
                64 + (x * 24) as u8,
                64 + (y * 24) as u8,
                100 + (((x + y) * 11) as u8),
                255,
            )
        });
        let fwd = ComponentTransfer {
            r: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 2.2,
                offset: 0.0,
            },
            g: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 2.2,
                offset: 0.0,
            },
            b: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 2.2,
                offset: 0.0,
            },
            a: TransferFunc::Identity,
        };
        let inv = ComponentTransfer {
            r: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 1.0 / 2.2,
                offset: 0.0,
            },
            g: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 1.0 / 2.2,
                offset: 0.0,
            },
            b: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 1.0 / 2.2,
                offset: 0.0,
            },
            a: TransferFunc::Identity,
        };
        let bent = component_transfer(&img, 8, 8, &fwd);
        let back = component_transfer(&bent, 8, 8, &inv);
        for i in 0..img.len() {
            let d = (back[i] as i32 - img[i] as i32).abs();
            assert!(
                d <= 2,
                "gamma round-trip differs by {d} at byte {i} ({} → {} → {})",
                img[i],
                bent[i],
                back[i]
            );
        }
    }

    #[test]
    fn discrete_three_entry_thresholds_correctly() {
        // Discrete(vec![0.0, 0.5, 1.0]) with N=3 partitions [0,1] into
        // three buckets: [0, 1/3) → 0.0 → 0u8, [1/3, 2/3) → 0.5 → 128u8,
        // [2/3, 1] → 1.0 → 255u8. The boundary samples land at
        // 85 (1/3·255) and 170 (2/3·255); test exact-integer inputs
        // that bracket those breakpoints.
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Discrete(vec![0.0, 0.5, 1.0]));
        let pixels: Vec<u8> = vec![0, 84, 86, 169, 171, 255];
        // 6 R values, each in its own 1×1 pixel with G/B/A = 0.
        let mut img = Vec::new();
        for r in &pixels {
            img.extend_from_slice(&[*r, 0, 0, 0]);
        }
        let out = component_transfer(&img, 6, 1, &ct);
        let want: [u8; 6] = [0, 0, 128, 128, 255, 255];
        for (idx, &w) in want.iter().enumerate() {
            assert_eq!(
                out[idx * 4],
                w,
                "discrete[{idx}] for input {} got {}, want {w}",
                pixels[idx],
                out[idx * 4]
            );
        }
    }

    #[test]
    fn table_two_entry_is_identity() {
        // Table(vec![0.0, 1.0]) with N=1 places the breakpoints at
        // C=0 → 0 and C=1 → 1, so the piecewise-linear interpolation
        // is C' = C — the identity. Must be byte-exact (±1 LSB for
        // the f32 round-trip; ½ LSB on each side and a ≤1 LSB
        // worst-case after rounding).
        let img = build(6, 2, |x, y| {
            Rgba::new((x * 51) as u8, (y * 127) as u8, 13, 200)
        });
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Table(vec![0.0, 1.0]));
        let out = component_transfer(&img, 6, 2, &ct);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "table-identity differs by {d} at byte {i}");
        }
    }

    #[test]
    fn table_two_entry_zero_half_halves_the_value() {
        // Table(vec![0.0, 0.5]) with N=1 places C=0 → 0, C=1 → 0.5,
        // so C' = 0.5 · C, equivalent to `R / 2` after quantisation
        // (±1 LSB). Apply to R, leave G/B/A alone.
        let img = build(4, 4, |x, _| Rgba::new((x * 60) as u8, 100, 50, 200));
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Table(vec![0.0, 0.5]));
        let out = component_transfer(&img, 4, 4, &ct);
        for i in (0..img.len()).step_by(4) {
            // Expected: (R / 255) · 0.5 · 255 = R · 0.5, rounded half-up.
            let want = ((img[i] as f32 / 255.0) * 0.5 * 255.0 + 0.5) as i32;
            let got = out[i] as i32;
            assert!(
                (got - want).abs() <= 1,
                "table-half R got {got} want {want} for input {}",
                img[i]
            );
            assert_eq!(out[i + 1], img[i + 1]);
            assert_eq!(out[i + 2], img[i + 2]);
            assert_eq!(out[i + 3], img[i + 3]);
        }
    }

    #[test]
    fn empty_image_returns_empty_buffer() {
        // §15.11 doesn't pin down behaviour on a zero-extent input
        // (the filter region degenerates to empty); we choose the
        // same convention as gaussian_blur and color_matrix and
        // return an empty buffer.
        let ct = ComponentTransfer::identity();
        assert_eq!(component_transfer(&[], 0, 5, &ct), Vec::<u8>::new());
        assert_eq!(component_transfer(&[], 5, 0, &ct), Vec::<u8>::new());
    }

    #[test]
    fn solid_image_under_identity_is_invariant() {
        // A uniform image under all-identity must come back byte-exact;
        // this exercises the fast path explicitly on a non-trivial
        // pixel value.
        let img = build(4, 4, |_, _| Rgba::new(123, 45, 67, 200));
        let ct = ComponentTransfer::identity();
        let out = component_transfer(&img, 4, 4, &ct);
        assert_eq!(out, img);
    }

    #[test]
    fn linear_on_alpha_keeps_rgb_byte_exact() {
        // Linear { slope: 0.5, intercept: 0 } on A only must scale
        // alpha by half and leave RGB byte-exact. Exercises that the
        // per-channel dispatch really is per-channel.
        let img = build(3, 3, |x, y| {
            Rgba::new((x * 80) as u8, (y * 80) as u8, 7, 200)
        });
        let ct = ComponentTransfer::identity().with_a(TransferFunc::Linear {
            slope: 0.5,
            intercept: 0.0,
        });
        let out = component_transfer(&img, 3, 3, &ct);
        for i in (0..img.len()).step_by(4) {
            assert_eq!(out[i], img[i]);
            assert_eq!(out[i + 1], img[i + 1]);
            assert_eq!(out[i + 2], img[i + 2]);
            let want = ((img[i + 3] as f32 / 255.0) * 0.5 * 255.0 + 0.5) as i32;
            let got = out[i + 3] as i32;
            assert!((got - want).abs() <= 1, "alpha half got {got} want {want}");
        }
    }

    #[test]
    fn linear_intercept_one_saturates_to_255() {
        // Linear { slope: 0, intercept: 1 } sets every pixel of the
        // affected channel to 1.0 → 255u8, regardless of input. Tests
        // the §15.11 output clamp implicitly via the saturating cap.
        let img = build(2, 2, |_, _| Rgba::new(0, 0, 0, 0));
        let ct = ComponentTransfer::identity().with_g(TransferFunc::Linear {
            slope: 0.0,
            intercept: 1.0,
        });
        let out = component_transfer(&img, 2, 2, &ct);
        for i in (0..img.len()).step_by(4) {
            assert_eq!(out[i], 0);
            assert_eq!(out[i + 1], 255);
            assert_eq!(out[i + 2], 0);
            assert_eq!(out[i + 3], 0);
        }
    }

    #[test]
    fn typed_pixel_wrapper_agrees_with_byte_api() {
        // The Vec<Rgba> wrapper must produce the byte-equivalent result
        // for any ComponentTransfer; pick a non-trivial mix of modes
        // across the four channels.
        let img: Vec<Rgba> = (0..16)
            .map(|i| Rgba::new(i as u8 * 16, (16 - i) as u8 * 15, 50, 200))
            .collect();
        let bytes: Vec<u8> = img.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
        let ct = ComponentTransfer {
            r: TransferFunc::Gamma {
                amplitude: 1.0,
                exponent: 2.0,
                offset: 0.0,
            },
            g: TransferFunc::Linear {
                slope: 0.75,
                intercept: 0.25,
            },
            b: TransferFunc::Discrete(vec![0.0, 1.0]),
            a: TransferFunc::Identity,
        };
        let via_bytes = component_transfer(&bytes, 4, 4, &ct);
        let via_pixels = component_transfer_pixels(&img, 4, 4, &ct);
        let want_pixels: Vec<Rgba> = via_bytes
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        assert_eq!(via_pixels, want_pixels);
    }

    #[test]
    fn empty_table_collapses_to_identity() {
        // §15.11's "type=table with insufficient values" fall-back is
        // an identity. An empty Vec defines no segment, so we return
        // C' = C. Exercises that the fall-back is taken before any
        // f32 lookup that would dereference out of range.
        let img = build(2, 2, |x, y| {
            Rgba::new((x * 100) as u8, (y * 100) as u8, 7, 200)
        });
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Table(Vec::new()));
        let out = component_transfer(&img, 2, 2, &ct);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "table-empty differs by {d} at byte {i}");
        }
    }

    #[test]
    fn empty_discrete_collapses_to_identity() {
        // Same as table-empty: zero-entry discrete defines no buckets,
        // collapse to identity rather than panic on the floor lookup.
        let img = build(2, 2, |x, y| {
            Rgba::new((x * 100) as u8, (y * 100) as u8, 7, 200)
        });
        let ct = ComponentTransfer::identity().with_g(TransferFunc::Discrete(Vec::new()));
        let out = component_transfer(&img, 2, 2, &ct);
        for i in 0..img.len() {
            let d = (out[i] as i32 - img[i] as i32).abs();
            assert!(d <= 1, "discrete-empty differs by {d} at byte {i}");
        }
    }

    #[test]
    #[should_panic(expected = "component_transfer: src.len() ==")]
    fn wrong_length_panics() {
        let bad = vec![0u8; 7];
        let ct = ComponentTransfer::identity();
        let _ = component_transfer(&bad, 2, 2, &ct);
    }

    #[test]
    #[should_panic(expected = "gamma exponent must be strictly positive")]
    fn gamma_zero_exponent_panics() {
        let img = vec![0u8; 16];
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: 0.0,
            offset: 0.0,
        });
        let _ = component_transfer(&img, 2, 2, &ct);
    }

    #[test]
    #[should_panic(expected = "gamma exponent must be strictly positive")]
    fn gamma_negative_exponent_panics() {
        let img = vec![0u8; 16];
        let ct = ComponentTransfer::identity().with_r(TransferFunc::Gamma {
            amplitude: 1.0,
            exponent: -0.5,
            offset: 0.0,
        });
        let _ = component_transfer(&img, 2, 2, &ct);
    }
}

// ---------------------------------------------------------------------------
// feComposite — SVG 1.1 §15.12
// ---------------------------------------------------------------------------

/// Operator selector for [`composite_filter`], mirroring the `operator`
/// attribute of SVG 1.1 §15.12 `<feComposite>`
/// (`"over" | "in" | "out" | "atop" | "xor" | "arithmetic"`).
///
/// The five non-arithmetic operators are the Porter–Duff compositing
/// operations referenced by §15.12. They are expressed here through the
/// standard premultiplied-alpha blend-factor pair `(Fa, Fb)` so that the
/// per-channel result is
///
/// ```text
/// co = ca · Fa + cb · Fb        (premultiplied colour)
/// αo = αa · Fa + αb · Fb        (alpha)
/// ```
///
/// where the *first* input (`in` / `i1`) is the `a` operand and the
/// *second* input (`in2` / `i2`) is the `b` operand, consistent with the
/// premultiplied source-over algebra of SVG 1.1 §14.2 already used by
/// the crate's [`composite_rgba_premultiplied`](crate::composite_rgba_premultiplied)
/// path (`Cr' = (1 − Ea)·Cr + Er`, i.e. `over` with `Fa = 1`,
/// `Fb = 1 − αa`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositeOp {
    /// `in` source-over `in2`. `Fa = 1`, `Fb = 1 − αa`.
    Over,
    /// `in` clipped to `in2`'s coverage. `Fa = αb`, `Fb = 0`.
    In,
    /// `in` outside `in2`'s coverage. `Fa = 1 − αb`, `Fb = 0`.
    Out,
    /// `in` atop `in2`. `Fa = αb`, `Fb = 1 − αa`.
    Atop,
    /// Symmetric difference. `Fa = 1 − αb`, `Fb = 1 − αa`.
    Xor,
    /// Component-wise arithmetic combination
    /// `result = k1·i1·i2 + k2·i1 + k3·i2 + k4` (§15.12), evaluated on
    /// premultiplied channels in `[0, 1]` and clamped to `[0, 1]`.
    Arithmetic {
        /// Coefficient on the `i1 · i2` product term.
        k1: f32,
        /// Coefficient on the `i1` term.
        k2: f32,
        /// Coefficient on the `i2` term.
        k3: f32,
        /// Constant offset term.
        k4: f32,
    },
}

impl CompositeOp {
    /// The Porter–Duff `(Fa, Fb)` blend factors for the five
    /// non-arithmetic operators, given the premultiplied alphas
    /// `αa` (= `in`) and `αb` (= `in2`), both in `[0, 1]`.
    ///
    /// Returns `None` for [`CompositeOp::Arithmetic`], which is not a
    /// blend-factor operator and is handled separately.
    #[inline]
    fn blend_factors(self, alpha_a: f32, alpha_b: f32) -> Option<(f32, f32)> {
        Some(match self {
            CompositeOp::Over => (1.0, 1.0 - alpha_a),
            CompositeOp::In => (alpha_b, 0.0),
            CompositeOp::Out => (1.0 - alpha_b, 0.0),
            CompositeOp::Atop => (alpha_b, 1.0 - alpha_a),
            CompositeOp::Xor => (1.0 - alpha_b, 1.0 - alpha_a),
            CompositeOp::Arithmetic { .. } => return None,
        })
    }
}

/// SVG 1.1 §15.12 `<feComposite>` — combine two equally-sized
/// packed-RGBA `u8` buffers pixel-wise with a [`CompositeOp`].
///
/// `in1` maps to the spec's `in` / `i1` operand and `in2` to `in2` /
/// `i2`. Both buffers must be exactly `width * height * 4` bytes in
/// row-major RGBA order and describe the same `width × height` extent.
///
/// **Pixel space.** §14.2 ("all color values use premultiplied alpha")
/// fixes the compositing algebra in the premultiplied domain. This
/// entry point converts each straight-alpha input byte triple to
/// premultiplied `[0, 1]` floats (`c · α`), evaluates the operator, and
/// converts the premultiplied result back to straight-alpha bytes. The
/// arithmetic operator is likewise evaluated on the premultiplied
/// channels, matching the spec's example use (combining lighting output
/// with texture) where the operands are already premultiplied filter
/// results.
///
/// Complexity is `O(W · H)` with a constant per-pixel cost.
///
/// # Panics
///
/// * If `in1.len() != width as usize * height as usize * 4`.
/// * If `in2.len() != width as usize * height as usize * 4`.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same dimensions.
pub fn composite_filter(
    in1: &[u8],
    in2: &[u8],
    width: u32,
    height: u32,
    op: CompositeOp,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("composite_filter: width * height * 4 overflowed usize");
    assert_eq!(
        in1.len(),
        expected,
        "composite_filter: in1.len() == {} but width*height*4 == {expected}",
        in1.len()
    );
    assert_eq!(
        in2.len(),
        expected,
        "composite_filter: in2.len() == {} but width*height*4 == {expected}",
        in2.len()
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(expected);
    for (pa, pb) in in1.chunks_exact(4).zip(in2.chunks_exact(4)) {
        // Straight-alpha bytes → premultiplied [0, 1].
        let aa = pa[3] as f32 / 255.0;
        let ab = pb[3] as f32 / 255.0;
        let a = [
            pa[0] as f32 / 255.0 * aa,
            pa[1] as f32 / 255.0 * aa,
            pa[2] as f32 / 255.0 * aa,
            aa,
        ];
        let b = [
            pb[0] as f32 / 255.0 * ab,
            pb[1] as f32 / 255.0 * ab,
            pb[2] as f32 / 255.0 * ab,
            ab,
        ];

        let (pr, pg, pb_, palpha) = match op {
            CompositeOp::Arithmetic { k1, k2, k3, k4 } => {
                let arith =
                    |i1: f32, i2: f32| (k1 * i1 * i2 + k2 * i1 + k3 * i2 + k4).clamp(0.0, 1.0);
                (
                    arith(a[0], b[0]),
                    arith(a[1], b[1]),
                    arith(a[2], b[2]),
                    arith(a[3], b[3]),
                )
            }
            _ => {
                // SAFETY of unwrap: every variant except Arithmetic
                // returns Some, and Arithmetic is handled above.
                let (fa, fb) = op.blend_factors(a[3], b[3]).unwrap();
                (
                    (a[0] * fa + b[0] * fb).clamp(0.0, 1.0),
                    (a[1] * fa + b[1] * fb).clamp(0.0, 1.0),
                    (a[2] * fa + b[2] * fb).clamp(0.0, 1.0),
                    (a[3] * fa + b[3] * fb).clamp(0.0, 1.0),
                )
            }
        };

        // Premultiplied [0, 1] → straight-alpha bytes. When the result
        // alpha is zero the colour is fully transparent; emit (0,0,0,0)
        // so the un-premultiply division is well-defined.
        if palpha <= 0.0 {
            out.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let inv = 1.0 / palpha;
            out.push(quantise_unit((pr * inv).min(1.0)));
            out.push(quantise_unit((pg * inv).min(1.0)));
            out.push(quantise_unit((pb_ * inv).min(1.0)));
            out.push(quantise_unit(palpha));
        }
    }
    out
}

/// Convenience wrapper that runs [`composite_filter`] on two slices of
/// [`Rgba`] pixels and returns a `Vec<Rgba>` of the same length.
/// Identical semantics — provided for callers that already have typed
/// pixel buffers.
pub fn composite_filter_pixels(
    in1: &[Rgba],
    in2: &[Rgba],
    width: u32,
    height: u32,
    op: CompositeOp,
) -> Vec<Rgba> {
    let n = width as usize * height as usize;
    assert_eq!(
        in1.len(),
        n,
        "composite_filter_pixels: in1.len() == {} but width*height == {n}",
        in1.len()
    );
    assert_eq!(
        in2.len(),
        n,
        "composite_filter_pixels: in2.len() == {} but width*height == {n}",
        in2.len()
    );
    let mut b1 = Vec::with_capacity(n * 4);
    let mut b2 = Vec::with_capacity(n * 4);
    for p in in1 {
        b1.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    for p in in2 {
        b2.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out = composite_filter(&b1, &b2, width, height, op);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

/// Edge-extension policy for [`convolve_matrix`], mirroring the
/// `edgeMode` attribute of SVG 1.1 §15.13 `<feConvolveMatrix>`.
///
/// Selects how the kernel reads beyond the borders of the source
/// image. The default per the spec is `Duplicate` (clamp-to-edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvolveEdgeMode {
    /// Extend by replicating the border row / column — `"duplicate"`,
    /// the spec default. The same clamp-to-edge policy used by
    /// [`gaussian_blur`] and [`morphology`].
    #[default]
    Duplicate,
    /// Extend by wrapping around — sample column `-1` reads column
    /// `width - 1`, sample row `height` reads row `0`, and so on
    /// (toroidal addressing). Spec value `"wrap"`.
    Wrap,
    /// Extend with `(R, G, B, A) = (0, 0, 0, 0)` — sample positions
    /// outside the source contribute zero to the sum. Spec value
    /// `"none"`.
    None,
}

/// Parameter block for [`convolve_matrix`] / [`convolve_matrix_pixels`].
///
/// Mirrors the attribute set of SVG 1.1 §15.13 `<feConvolveMatrix>`:
/// `order`, `kernelMatrix`, `divisor`, `bias`, `targetX`, `targetY`,
/// `edgeMode`, and `preserveAlpha`. Construct directly with the public
/// fields or through [`ConvolveMatrix::new`] which applies the spec
/// defaults for `divisor` (= sum of kernel, falling back to `1.0`),
/// `target_x` / `target_y` (= floor of order / 2), and `bias` (= 0).
///
/// Field interpretation matches §15.13 verbatim:
///
/// * `order_x` × `order_y` cells in `kernel`, stored row-major: index
///   `row * order_x + col` reads `kernelMatrix[col, row]`. Spec
///   §15.13 mandates `1 ≤ order_{x,y}`.
/// * `divisor` divides the convolved sum before `bias` is added.
///   Must be non-zero (§15.13 error processing).
/// * `bias` is added to each output channel after the divisor step.
/// * `target_x` / `target_y` reposition the kernel relative to the
///   target pixel; the default centres a square kernel
///   (`floor(order / 2)`). Spec mandates `0 ≤ target_{x,y} < order_{x,y}`.
/// * `edge_mode` chooses how out-of-bounds samples are extended.
/// * `preserve_alpha = false` (default per §15.13) convolves the alpha
///   channel alongside RGB; `preserve_alpha = true` un-premultiplies the
///   source, convolves only RGB, leaves alpha as the original
///   `SOURCE_{X,Y}`, then re-premultiplies the result before
///   re-quantising to `u8`.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvolveMatrix {
    /// Number of columns in `kernel`. Must be `≥ 1`.
    pub order_x: u32,
    /// Number of rows in `kernel`. Must be `≥ 1`.
    pub order_y: u32,
    /// Row-major `order_y × order_x` kernel weights, length
    /// `order_x * order_y`.
    pub kernel: Vec<f32>,
    /// Divisor applied to the convolved sum (§15.13 — default is
    /// `sum(kernel)`, falling back to `1.0` when the sum is zero).
    /// Must be non-zero.
    pub divisor: f32,
    /// Additive bias applied after the divisor step.
    pub bias: f32,
    /// X-position of the kernel's anchor (0 ≤ `target_x` < `order_x`).
    pub target_x: u32,
    /// Y-position of the kernel's anchor (0 ≤ `target_y` < `order_y`).
    pub target_y: u32,
    /// Out-of-bounds sample extension policy.
    pub edge_mode: ConvolveEdgeMode,
    /// When `true`, convolve only the RGB channels of the
    /// un-premultiplied source and pass alpha through unchanged
    /// (§15.13 `preserveAlpha="true"`). When `false`, convolve all
    /// four channels of the straight-alpha source uniformly.
    pub preserve_alpha: bool,
}

impl ConvolveMatrix {
    /// Construct a [`ConvolveMatrix`] applying the §15.13 defaults for
    /// every attribute the spec leaves unspecified:
    ///
    /// * `divisor` = sum of `kernel`, falling back to `1.0` when the
    ///   sum is zero (§15.13).
    /// * `bias` = `0`.
    /// * `target_x` = `floor(order_x / 2)`, `target_y` = `floor(order_y / 2)`.
    /// * `edge_mode` = [`ConvolveEdgeMode::Duplicate`].
    /// * `preserve_alpha` = `false`.
    ///
    /// # Panics
    ///
    /// * If `order_x == 0` or `order_y == 0` (§15.13 error
    ///   processing — both must be ≥ 1).
    /// * If `kernel.len() != order_x * order_y` (the spec requires the
    ///   kernel list length to equal `orderX * orderY`).
    pub fn new(order_x: u32, order_y: u32, kernel: Vec<f32>) -> Self {
        assert!(
            order_x >= 1 && order_y >= 1,
            "ConvolveMatrix::new: order_x ({order_x}) and order_y ({order_y}) must each be >= 1"
        );
        let n = (order_x as usize)
            .checked_mul(order_y as usize)
            .expect("ConvolveMatrix::new: order_x * order_y overflowed usize");
        assert_eq!(
            kernel.len(),
            n,
            "ConvolveMatrix::new: kernel.len() ({}) must equal order_x * order_y ({n})",
            kernel.len()
        );

        let sum: f32 = kernel.iter().sum();
        let divisor = if sum == 0.0 { 1.0 } else { sum };
        let target_x = order_x / 2;
        let target_y = order_y / 2;

        ConvolveMatrix {
            order_x,
            order_y,
            kernel,
            divisor,
            bias: 0.0,
            target_x,
            target_y,
            edge_mode: ConvolveEdgeMode::Duplicate,
            preserve_alpha: false,
        }
    }

    /// Builder: set [`Self::bias`].
    pub fn with_bias(mut self, bias: f32) -> Self {
        self.bias = bias;
        self
    }

    /// Builder: set [`Self::divisor`]. Must be non-zero (panic
    /// otherwise — §15.13 error processing).
    pub fn with_divisor(mut self, divisor: f32) -> Self {
        assert!(
            divisor != 0.0,
            "ConvolveMatrix::with_divisor: divisor must be non-zero (§15.13)"
        );
        self.divisor = divisor;
        self
    }

    /// Builder: set [`Self::target_x`] / [`Self::target_y`]. Each must
    /// satisfy `0 ≤ target < order` (§15.13 error processing).
    pub fn with_target(mut self, target_x: u32, target_y: u32) -> Self {
        assert!(
            target_x < self.order_x,
            "ConvolveMatrix::with_target: target_x ({target_x}) >= order_x ({})",
            self.order_x
        );
        assert!(
            target_y < self.order_y,
            "ConvolveMatrix::with_target: target_y ({target_y}) >= order_y ({})",
            self.order_y
        );
        self.target_x = target_x;
        self.target_y = target_y;
        self
    }

    /// Builder: set [`Self::edge_mode`].
    pub fn with_edge_mode(mut self, edge_mode: ConvolveEdgeMode) -> Self {
        self.edge_mode = edge_mode;
        self
    }

    /// Builder: set [`Self::preserve_alpha`].
    pub fn with_preserve_alpha(mut self, preserve_alpha: bool) -> Self {
        self.preserve_alpha = preserve_alpha;
        self
    }

    /// Read the kernel weight at logical position `(col, row)` —
    /// equivalent to `kernel[row * order_x + col]`. Spec-named
    /// `kernelMatrix[col, row]`.
    #[inline]
    fn kernel_at(&self, col: u32, row: u32) -> f32 {
        debug_assert!(col < self.order_x && row < self.order_y);
        self.kernel[(row * self.order_x + col) as usize]
    }
}

/// Read a sample from a packed-RGBA `u8` buffer with edge handling.
///
/// `x`, `y` may be negative or `>= width / height`. The four channels
/// are returned in straight-alpha `u8` order (R, G, B, A). For
/// [`ConvolveEdgeMode::None`] out-of-bounds reads return `(0, 0, 0, 0)`;
/// for `Duplicate` they clamp to the nearest in-range coordinate; for
/// `Wrap` they take `x.rem_euclid(width)` / `y.rem_euclid(height)`.
#[inline]
fn sample_with_edge(
    src: &[u8],
    width: u32,
    height: u32,
    x: i64,
    y: i64,
    mode: ConvolveEdgeMode,
) -> [u8; 4] {
    let w = width as i64;
    let h = height as i64;
    let (sx, sy) = match mode {
        ConvolveEdgeMode::Duplicate => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
        ConvolveEdgeMode::Wrap => (x.rem_euclid(w), y.rem_euclid(h)),
        ConvolveEdgeMode::None => {
            if x < 0 || x >= w || y < 0 || y >= h {
                return [0, 0, 0, 0];
            }
            (x, y)
        }
    };
    let idx = ((sy as usize) * (width as usize) + (sx as usize)) * 4;
    [src[idx], src[idx + 1], src[idx + 2], src[idx + 3]]
}

/// SVG 1.1 §15.13 `<feConvolveMatrix>` — apply a general matrix
/// convolution to a packed-RGBA `u8` buffer.
///
/// Implements the spec formula verbatim:
///
/// ```text
/// COLOR_{X,Y} = (
///     SUM_{I=0..orderY-1} {
///         SUM_{J=0..orderX-1} {
///             SOURCE_{X − targetX + J, Y − targetY + I}
///                 · kernelMatrix[orderX − J − 1, orderY − I − 1]
///         }
///     }
/// ) / divisor + bias
/// ```
///
/// The kernel is rotated 180° when sampling — that is, the
/// top-left-most coefficient `kernelMatrix[0, 0]` lands on the
/// bottom-right-most contributing source pixel — matching the
/// mathematical convention noted in §15.13 ("the values in the
/// kernel matrix are applied such that the kernel matrix is rotated
/// 180 degrees relative to the source and destination images").
///
/// `divisor`, `bias`, `target_x`, `target_y`, `edge_mode`, and
/// `preserve_alpha` are taken from `cm`; see [`ConvolveMatrix`] for the
/// per-field semantics.
///
/// When `preserve_alpha == false` (§15.13 default), the convolution is
/// applied uniformly across all four straight-alpha channels of the
/// source — the same formula above governs alpha as governs colour.
/// When `preserve_alpha == true`, the source's straight-alpha colour
/// channels are first un-premultiplied (the alpha-multiplication
/// inverse), the RGB channels are then convolved, alpha is left as
/// `SOURCE_{X,Y}` (the original pixel's alpha), and the RGB result is
/// re-premultiplied before the final clamp-and-quantise step. (The
/// source is treated as straight-alpha throughout; the
/// "un-premultiply / re-premultiply" wording in §15.13 refers to the
/// internal premultiplication used while applying the kernel, not to
/// the encoding of the input or output buffers.)
///
/// Output channels are clamped to `[0, 255]` before quantisation.
///
/// Complexity: `O(W · H · order_x · order_y)`. The kernel is **not**
/// separable in general — `feConvolveMatrix` is the catch-all
/// arbitrary-2D-kernel primitive; spec-named separable kernels
/// (Gaussian) live in dedicated filter primitives.
///
/// # Panics
///
/// * If `src.len() != width as usize * height as usize * 4`.
/// * If `cm.divisor == 0.0` (§15.13 error processing).
/// * If `cm.order_x == 0` or `cm.order_y == 0`.
/// * If `cm.kernel.len() != cm.order_x * cm.order_y`.
/// * If `cm.target_x >= cm.order_x` or `cm.target_y >= cm.order_y`.
///
/// # Returns
///
/// A new packed-RGBA `Vec<u8>` of the same `width × height` extent.
pub fn convolve_matrix(src: &[u8], width: u32, height: u32, cm: &ConvolveMatrix) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(4))
        .expect("convolve_matrix: width * height * 4 overflowed usize");
    assert_eq!(
        src.len(),
        expected,
        "convolve_matrix: src.len() == {} but width*height*4 == {expected}",
        src.len()
    );
    assert!(
        cm.divisor != 0.0,
        "convolve_matrix: divisor must be non-zero (§15.13)"
    );
    assert!(
        cm.order_x >= 1 && cm.order_y >= 1,
        "convolve_matrix: order_x ({}) and order_y ({}) must each be >= 1",
        cm.order_x,
        cm.order_y
    );
    let n_cells = (cm.order_x as usize) * (cm.order_y as usize);
    assert_eq!(
        cm.kernel.len(),
        n_cells,
        "convolve_matrix: kernel.len() ({}) must equal order_x * order_y ({n_cells})",
        cm.kernel.len()
    );
    assert!(
        cm.target_x < cm.order_x,
        "convolve_matrix: target_x ({}) >= order_x ({})",
        cm.target_x,
        cm.order_x
    );
    assert!(
        cm.target_y < cm.order_y,
        "convolve_matrix: target_y ({}) >= order_y ({})",
        cm.target_y,
        cm.order_y
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }

    let inv_div = 1.0 / cm.divisor;
    let bias = cm.bias;
    let order_x = cm.order_x as i64;
    let order_y = cm.order_y as i64;
    let tx = cm.target_x as i64;
    let ty = cm.target_y as i64;

    let mut out = vec![0u8; expected];

    for y in 0..height as i64 {
        for x in 0..width as i64 {
            let mut sum = [0f32; 4];
            // Original alpha at the target pixel — needed for the
            // preserve_alpha pass-through and for the un-premultiply
            // step before the kernel is applied to RGB.
            let target = sample_with_edge(src, width, height, x, y, cm.edge_mode);
            let target_alpha = target[3] as f32 / 255.0;

            for i in 0..order_y {
                for j in 0..order_x {
                    // The 180°-rotated kernel index per §15.13:
                    // kernelMatrix[orderX − J − 1, orderY − I − 1].
                    let k = cm.kernel_at((order_x - j - 1) as u32, (order_y - i - 1) as u32);
                    if k == 0.0 {
                        continue;
                    }
                    let sx = x - tx + j;
                    let sy = y - ty + i;
                    let s = sample_with_edge(src, width, height, sx, sy, cm.edge_mode);
                    if cm.preserve_alpha {
                        // RGB only; alpha is taken from the target pixel
                        // and not summed. The spec wording is "the filter
                        // will temporarily unpremultiply the color
                        // component values, apply the kernel, and then
                        // re-premultiply at the end". Source straight-
                        // alpha bytes already store un-premultiplied RGB,
                        // so the un-premultiply step is the identity.
                        sum[0] += k * (s[0] as f32);
                        sum[1] += k * (s[1] as f32);
                        sum[2] += k * (s[2] as f32);
                    } else {
                        sum[0] += k * (s[0] as f32);
                        sum[1] += k * (s[1] as f32);
                        sum[2] += k * (s[2] as f32);
                        sum[3] += k * (s[3] as f32);
                    }
                }
            }

            // Apply divisor + bias, clamp, quantise.
            let out_idx = ((y as usize) * w + (x as usize)) * 4;
            let r = sum[0] * inv_div + bias * 255.0;
            let g = sum[1] * inv_div + bias * 255.0;
            let b = sum[2] * inv_div + bias * 255.0;
            out[out_idx] = quantise_byte(r);
            out[out_idx + 1] = quantise_byte(g);
            out[out_idx + 2] = quantise_byte(b);
            if cm.preserve_alpha {
                // Alpha pass-through: ALPHAX,Y = SOURCEX,Y per §15.13.
                out[out_idx + 3] = target[3];
            } else {
                let a = sum[3] * inv_div + bias * 255.0;
                out[out_idx + 3] = quantise_byte(a);
            }
            // Silence unused-variable warning when preserve_alpha=false;
            // target_alpha is still computed once per pixel for symmetry
            // with the preserveAlpha branch (no measurable overhead).
            let _ = target_alpha;
        }
    }

    out
}

/// Convenience wrapper that runs [`convolve_matrix`] on a slice of
/// [`Rgba`] pixels and returns a `Vec<Rgba>` of the same length.
/// Identical semantics — provided for callers that already have typed
/// pixel buffers.
pub fn convolve_matrix_pixels(
    src: &[Rgba],
    width: u32,
    height: u32,
    cm: &ConvolveMatrix,
) -> Vec<Rgba> {
    let n = (width as usize) * (height as usize);
    assert_eq!(
        src.len(),
        n,
        "convolve_matrix_pixels: src.len() == {} but width*height == {n}",
        src.len()
    );
    let mut bytes = Vec::with_capacity(n * 4);
    for p in src {
        bytes.extend_from_slice(&[p.r, p.g, p.b, p.a]);
    }
    let out = convolve_matrix(&bytes, width, height, cm);
    out.chunks_exact(4)
        .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
        .collect()
}

/// Round-half-up clamp from an arbitrary `f32` to `[0, 255]`, NaN to 0.
/// Distinct from [`quantise_unit`] (which expects a `[0, 1]` unit value)
/// and [`quantise_u8`] (whose contract is different); this one takes a
/// already-byte-scale `f32` straight off the convolution sum.
#[inline]
fn quantise_byte(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let clamped = v.clamp(0.0, 255.0);
    (clamped + 0.5) as u8
}

#[cfg(test)]
mod convolve_matrix_tests {
    use super::*;

    fn solid(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            v.extend_from_slice(&rgba);
        }
        v
    }

    /// 3×3 identity kernel — the centre cell is 1.0, everything else
    /// is zero. Divisor defaults to 1.0 (sum is 1.0). Spec-conformant
    /// per §15.13 default behaviour: byte-for-byte identity on the
    /// interior; with `Duplicate` edge mode the borders are also
    /// identity because the kernel only reads the centre pixel.
    #[test]
    fn identity_kernel_3x3_is_identity() {
        let w = 5;
        let h = 4;
        // Pseudo-random but deterministic pixel pattern.
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for i in 0..(w * h) {
            src.push(((i * 37 + 3) % 256) as u8);
            src.push(((i * 53 + 11) % 256) as u8);
            src.push(((i * 71 + 19) % 256) as u8);
            src.push(((i * 17 + 41) % 256) as u8);
        }

        #[rustfmt::skip]
        let cm = ConvolveMatrix::new(3, 3, vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ]);
        let out = convolve_matrix(&src, w, h, &cm);
        assert_eq!(out, src, "identity kernel should round-trip every pixel");
    }

    /// Default constructor picks `divisor = sum(kernel)` (§15.13). A
    /// solid colour through a box-blur kernel must be byte-identical
    /// regardless of edge mode (Duplicate / Wrap both return the
    /// same colour at every position; None also works because the
    /// kernel-sum / divisor = 1 cancels the in-bounds-only sum on the
    /// interior, and on the border the missing samples are zero,
    /// shrinking the sum proportionally — we expect identity on the
    /// interior pixels for None and possibly different for borders).
    #[test]
    fn box_blur_3x3_solid_is_identity_interior() {
        let w = 7;
        let h = 5;
        let colour = [120, 200, 60, 255];
        let src = solid(w, h, colour);
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]); // divisor = 9
        for mode in [
            ConvolveEdgeMode::Duplicate,
            ConvolveEdgeMode::Wrap,
            ConvolveEdgeMode::None,
        ] {
            let cm = cm.clone().with_edge_mode(mode);
            let out = convolve_matrix(&src, w, h, &cm);
            // Interior pixels must be identity regardless of edge
            // mode (the kernel only reads in-bounds samples for them).
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let i = ((y * w + x) * 4) as usize;
                    assert_eq!(
                        &out[i..i + 4],
                        &colour,
                        "interior pixel ({x},{y}) under {mode:?} should be identity"
                    );
                }
            }
            // Under Duplicate + Wrap edge modes the borders also see
            // a uniform colour everywhere the kernel reaches, so the
            // whole image must be identity.
            if matches!(mode, ConvolveEdgeMode::Duplicate | ConvolveEdgeMode::Wrap) {
                assert_eq!(out, src, "{mode:?} should be identity for solid image");
            }
        }
    }

    /// Hand-evaluate one pixel against the §15.13 worked example.
    /// With the spec's 5×5 input and the 3×3 kernel
    /// `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` (divisor = 45) the source
    /// pixel value at the second row and second column (the spec's
    /// example pixel) is 120, and the spec writes the result as
    /// `(9·0 + 8·20 + 7·40 + 6·100 + 5·120 + 4·140 + 3·200 + 2·220 +
    /// 1·240) / 45 = 3480 / 45 = 77.33…` — that is, the kernel is
    /// 180°-rotated before being applied, so coefficient 9 lands on
    /// the source pixel at the *top-left* of the 3×3 window
    /// (the 0 in the corner), coefficient 1 lands on the *bottom-right*
    /// (the 240). Our `convolve_matrix` must reproduce that 77 to a
    /// byte after the round-half-up quantisation.
    #[test]
    fn spec_15_13_worked_example_pixel() {
        // 5x5 single-channel-style image populated only in R. We use
        // grey (R == G == B) and full opaque so all four channels
        // share the same convolved value and the assertion stays
        // clean.
        let w = 5u32;
        let h = 5u32;
        #[rustfmt::skip]
        let grey: Vec<u8> = vec![
              0,  20,  40, 235, 235,
            100, 120, 140, 235, 235,
            200, 220, 240, 235, 235,
            225, 225, 255, 255, 255,
            225, 225, 255, 255, 255,
        ];
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for v in &grey {
            src.extend_from_slice(&[*v, *v, *v, 255]);
        }
        #[rustfmt::skip]
        let cm = ConvolveMatrix::new(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        // Default divisor is the kernel sum (45). The spec example
        // evaluates pixel (1, 1) to 3480 / 45 = 77.333… → 77 after
        // round-half-up. (The spec text lists the individual products
        // 9·0 + 8·20 + 7·40 + 6·100 + 5·120 + 4·140 + 3·200 + 2·220 +
        // 1·240 = 0 + 160 + 280 + 600 + 600 + 560 + 600 + 440 + 240
        // = 3480.)
        let out = convolve_matrix(&src, w, h, &cm);
        let idx = ((w + 1) * 4) as usize;
        let expected = 77u8;
        assert_eq!(out[idx], expected, "spec example pixel R should be 77");
        assert_eq!(out[idx + 1], expected, "spec example pixel G should be 77");
        assert_eq!(out[idx + 2], expected, "spec example pixel B should be 77");
    }

    /// `preserve_alpha = true` leaves the alpha channel byte-identical
    /// to the source while still convolving the RGB channels. We use a
    /// 3×3 box-blur on a sharp alpha mask — the RGB output blurs but
    /// the alpha must match the input pixel-by-pixel.
    #[test]
    fn preserve_alpha_passes_alpha_through() {
        let w = 5;
        let h = 5;
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let alpha = if (x + y) % 2 == 0 { 255 } else { 64 };
                src.extend_from_slice(&[100, 200, 50, alpha]);
            }
        }
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_preserve_alpha(true);
        let out = convolve_matrix(&src, w, h, &cm);
        for i in 0..(w * h) as usize {
            assert_eq!(
                out[i * 4 + 3],
                src[i * 4 + 3],
                "alpha[{i}] should pass through under preserve_alpha"
            );
        }
    }

    /// `edgeMode = "none"` zeros the kernel reads outside the source.
    /// On the top-left corner pixel of a 1×1-of-non-zero image (every
    /// other pixel is zero) we know exactly how many cells contribute.
    #[test]
    fn edge_mode_none_zeros_oob() {
        let w = 3;
        let h = 3;
        let mut src = vec![0u8; (w * h * 4) as usize];
        // Put 100 in R at the centre pixel only; everything else 0.
        let centre = ((w + 1) * 4) as usize;
        src[centre] = 100;
        src[centre + 3] = 255; // give it some alpha so it's well-formed
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9])
            .with_divisor(1.0)
            .with_edge_mode(ConvolveEdgeMode::None);
        let out = convolve_matrix(&src, w, h, &cm);
        // Every output pixel that contains the centre in its 3×3
        // neighbourhood (i.e. every pixel — at this size the kernel
        // touches every cell) sees R = 100 added once. Output R must
        // equal 100 everywhere in this configuration.
        for i in 0..(w * h) as usize {
            assert_eq!(
                out[i * 4],
                100,
                "every neighbour pixel should see the centre R contribution exactly once"
            );
        }
    }

    /// `edgeMode = "duplicate"` extends by clamping — sample at column
    /// -1 reads column 0, etc. Easy sanity: a 1×1 image under any kernel
    /// summing to `divisor` is byte-identical to the source.
    #[test]
    fn duplicate_1x1_is_identity_under_unit_divisor() {
        let src = [80u8, 160, 240, 255];
        let cm =
            ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_edge_mode(ConvolveEdgeMode::Duplicate);
        // divisor defaults to 9 → average of 9 reads of the only pixel
        // = the pixel itself.
        let out = convolve_matrix(&src, 1, 1, &cm);
        assert_eq!(out, &src);
    }

    /// `edgeMode = "wrap"` makes a `delta` impulse centred on a 1-pixel
    /// image symmetric. We don't test wrapping in detail here (covered
    /// in integration tests) — just that the type-checker / runtime
    /// don't panic on a small canvas.
    #[test]
    fn wrap_smoke_test() {
        let src = solid(2, 2, [10, 20, 30, 255]);
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_edge_mode(ConvolveEdgeMode::Wrap);
        let out = convolve_matrix(&src, 2, 2, &cm);
        assert_eq!(out, src, "wrap+box-blur on solid 2x2 is identity");
    }

    /// Bias shifts every output channel by a constant.
    ///
    /// Identity kernel + `bias = 0.25` must add ~64 (= 0.25 · 255 =
    /// 63.75 → 64) to every channel of a 100-grey input.
    #[test]
    fn bias_shifts_all_channels() {
        let src = solid(2, 2, [100, 100, 100, 100]);
        let cm = ConvolveMatrix::new(1, 1, vec![1.0]).with_bias(0.25);
        let out = convolve_matrix(&src, 2, 2, &cm);
        // 100 + 0.25*255 = 100 + 63.75 → round-half-up → 164.
        for (i, channel) in out.iter().take(4).enumerate() {
            assert_eq!(*channel, 164, "channel {i} should be 100 + 64 = 164");
        }
    }

    /// Negative bias also clamps at 0 — bias of -1.0 must zero every
    /// channel of any input.
    #[test]
    fn negative_bias_clamps_to_zero() {
        let src = solid(2, 2, [100, 200, 50, 255]);
        let cm = ConvolveMatrix::new(1, 1, vec![1.0]).with_bias(-1.0);
        let out = convolve_matrix(&src, 2, 2, &cm);
        for v in out {
            assert_eq!(v, 0);
        }
    }

    /// Standard 3×3 Sobel-X kernel produces zero on a constant image
    /// (high-pass / edge-detection property: it sums to zero).
    #[test]
    fn sobel_x_on_constant_is_zero() {
        let w = 5;
        let h = 5;
        let src = solid(w, h, [128, 128, 128, 255]);
        #[rustfmt::skip]
        let kernel = vec![
            -1.0, 0.0, 1.0,
            -2.0, 0.0, 2.0,
            -1.0, 0.0, 1.0,
        ];
        let cm = ConvolveMatrix::new(3, 3, kernel)
            // The kernel sums to zero so the default-divisor path would
            // fall back to 1.0. Set explicitly to silence ambiguity.
            .with_divisor(1.0)
            .with_edge_mode(ConvolveEdgeMode::Duplicate)
            .with_preserve_alpha(true);
        let out = convolve_matrix(&src, w, h, &cm);
        for i in 0..(w * h) as usize {
            assert_eq!(out[i * 4], 0, "R should be zero on constant");
            assert_eq!(out[i * 4 + 1], 0, "G should be zero on constant");
            assert_eq!(out[i * 4 + 2], 0, "B should be zero on constant");
            assert_eq!(out[i * 4 + 3], 255, "alpha pass-through");
        }
    }

    /// Asymmetric kernel — verify the 180° rotation specified by
    /// §15.13. Take a delta-impulse image (single non-zero pixel) and
    /// an asymmetric kernel and check the output landing pattern.
    #[test]
    fn kernel_is_rotated_180_per_spec() {
        let w = 5;
        let h = 5;
        let mut src = vec![0u8; (w * h * 4) as usize];
        // Impulse at (2, 2).
        let centre = ((2 * w + 2) * 4) as usize;
        src[centre] = 100;
        src[centre + 3] = 255;

        // Kernel with the only non-zero entry at the bottom-right
        // corner: kernelMatrix[2, 2] = 1. With the 180° rotation per
        // §15.13, this coefficient lands on
        // sample(X - targetX + 0, Y - targetY + 0) → top-left
        // neighbour of the target. So the impulse at (2,2) → output
        // contribution lands at (target offset relative to source).
        //
        // With targetX=targetY=1 (default for 3×3), the kernel
        // coefficient at index (orderX-J-1, orderY-I-1) = (2, 2)
        // corresponds to (J, I) = (0, 0), reading from source
        // (X-1, Y-1). So the impulse at (2, 2) shows up in the output
        // at (3, 3) (one step down-right).
        #[rustfmt::skip]
        let kernel = vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let cm = ConvolveMatrix::new(3, 3, kernel)
            .with_divisor(1.0)
            .with_edge_mode(ConvolveEdgeMode::None)
            .with_preserve_alpha(false);
        let out = convolve_matrix(&src, w, h, &cm);
        // Where the impulse lands per the 180° rotation formula:
        // output (3, 3).
        let landed = ((3 * w + 3) * 4) as usize;
        assert_eq!(
            out[landed], 100,
            "impulse should land at (3,3) per §15.13 rotation"
        );
        // Sanity: should NOT land at (1, 1) (the naive-no-rotation
        // location).
        let naive = ((w + 1) * 4) as usize;
        assert_eq!(
            out[naive], 0,
            "no contribution at (1,1) per §15.13 rotation"
        );
    }

    /// Typed-pixel wrapper agrees with the byte API across an
    /// asymmetric kernel + bias + non-default edge mode.
    #[test]
    fn typed_pixel_wrapper_matches_byte_api() {
        let w = 4;
        let h = 3;
        let mut src_b = Vec::with_capacity((w * h * 4) as usize);
        let mut src_p = Vec::with_capacity((w * h) as usize);
        for i in 0..(w * h) {
            let p = Rgba::new(
                ((i * 41) % 256) as u8,
                ((i * 59) % 256) as u8,
                ((i * 67) % 256) as u8,
                ((i * 23 + 70) % 256) as u8,
            );
            src_b.extend_from_slice(&[p.r, p.g, p.b, p.a]);
            src_p.push(p);
        }
        #[rustfmt::skip]
        let kernel = vec![
            1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let cm = ConvolveMatrix::new(3, 3, kernel)
            .with_bias(-0.05)
            .with_edge_mode(ConvolveEdgeMode::Wrap);
        let via_bytes = convolve_matrix(&src_b, w, h, &cm);
        let via_typed = convolve_matrix_pixels(&src_p, w, h, &cm);
        let typed_bytes: Vec<u8> = via_typed
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(via_bytes, typed_bytes);
    }

    /// Empty image returns an empty buffer.
    #[test]
    fn empty_extent_is_empty_vec() {
        let cm = ConvolveMatrix::new(1, 1, vec![1.0]);
        assert!(convolve_matrix(&[], 0, 0, &cm).is_empty());
    }

    /// Non-square kernel — order 5×1 horizontal moving-average
    /// produces blur along X only, leaves Y untouched. The orthogonal
    /// edge of a vertical step is unaffected by row.
    #[test]
    fn horizontal_kernel_only_blurs_x() {
        let w = 9;
        let h = 3;
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                // Step edge along x = 4: 0 on the left, 200 on the
                // right. Same colour on every row.
                let v = if x < 4 { 0 } else { 200 };
                src.extend_from_slice(&[v, v, v, 255]);
                let _ = y;
            }
        }
        // 1×5 horizontal averaging kernel.
        let cm = ConvolveMatrix::new(5, 1, vec![1.0; 5]);
        let out = convolve_matrix(&src, w, h, &cm);
        // All rows must be byte-identical to the first row.
        let row0: Vec<u8> = out[..(w * 4) as usize].to_vec();
        for y in 1..h {
            let start = (y * w * 4) as usize;
            let row = &out[start..start + (w * 4) as usize];
            assert_eq!(row, row0.as_slice(), "row {y} should match row 0");
        }
    }

    /// Custom `target` shifts the kernel anchor. With a 3×3 identity
    /// kernel (centre = 1.0) and `target = (0, 0)`, the impulse moves
    /// one pixel up-and-to-the-left because the anchor is now at the
    /// top-left of the kernel — but after the §15.13 180° rotation,
    /// the centre of the kernel ends up reading from
    /// (X - 0 + 1, Y - 0 + 1), i.e. one pixel down-and-right of the
    /// target. So a centred impulse at (2, 2) shows up at (1, 1).
    #[test]
    fn target_shifts_anchor() {
        let w = 5;
        let h = 5;
        let mut src = vec![0u8; (w * h * 4) as usize];
        let centre = ((2 * w + 2) * 4) as usize;
        src[centre] = 100;
        src[centre + 3] = 255;
        #[rustfmt::skip]
        let kernel = vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let cm = ConvolveMatrix::new(3, 3, kernel)
            .with_divisor(1.0)
            .with_target(0, 0)
            .with_edge_mode(ConvolveEdgeMode::None);
        let out = convolve_matrix(&src, w, h, &cm);
        // With target=(0,0), kernel index (1,1) corresponds to (J,I) =
        // (orderX-1-1, orderY-1-1) = (1, 1); the kernel reads source
        // at (X - 0 + 1, Y - 0 + 1) = (X+1, Y+1). The impulse at
        // (2,2) therefore lands at output (1,1) (the pixel whose
        // kernel reaches (2,2) via the (+1, +1) shift).
        let landed = ((w + 1) * 4) as usize;
        assert_eq!(
            out[landed], 100,
            "impulse should land at (1,1) under target=(0,0)"
        );
    }

    #[test]
    #[should_panic(expected = "src.len()")]
    fn wrong_input_length_panics() {
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]);
        let _ = convolve_matrix(&[0u8; 8], 3, 3, &cm);
    }

    #[test]
    #[should_panic(expected = "divisor must be non-zero")]
    fn zero_divisor_panics() {
        let cm = ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_divisor(1.0);
        // Manually break the invariant — exercise the convolve_matrix
        // guard, not the builder's.
        let mut bad = cm.clone();
        bad.divisor = 0.0;
        let _ = convolve_matrix(&[0u8; 4], 1, 1, &bad);
    }

    #[test]
    #[should_panic(expected = "kernel.len()")]
    fn wrong_kernel_length_panics() {
        ConvolveMatrix::new(3, 3, vec![1.0; 5]);
    }

    #[test]
    #[should_panic(expected = "order_x")]
    fn zero_order_panics() {
        ConvolveMatrix::new(0, 3, vec![]);
    }

    #[test]
    #[should_panic(expected = "target_x")]
    fn out_of_range_target_panics() {
        ConvolveMatrix::new(3, 3, vec![1.0; 9]).with_target(3, 1);
    }
}

#[cfg(test)]
mod composite_tests {
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
    fn over_matches_simple_alpha_compositing() {
        // §14.2 premultiplied "over": Cr' = (1 - Ea)*Cr + Er with
        // E = in (a), C = in2 (b). Opaque blue over opaque red is just
        // opaque blue. Half-alpha blue over opaque red blends.
        let a = build(2, 1, |x, _| {
            if x == 0 {
                Rgba::new(0, 0, 255, 255) // opaque blue
            } else {
                Rgba::new(0, 0, 255, 128) // ~half blue
            }
        });
        let b = build(2, 1, |_, _| Rgba::new(255, 0, 0, 255)); // opaque red
        let out = composite_filter(&a, &b, 2, 1, CompositeOp::Over);
        // Pixel 0: opaque blue wins outright.
        assert_eq!(&out[0..4], &[0, 0, 255, 255]);
        // Pixel 1: alpha stays opaque (1 - (1-0.5)*(1-1) = 1).
        assert_eq!(out[7], 255);
        // Red premult ≈ (1 - 0.502)*1.0 = 0.498 → round-trip ~127.
        let r = out[4] as i32;
        assert!((r - 127).abs() <= 2, "expected ~127 red got {r}");
        // Blue premult ≈ 0.502 → straight ~128.
        let bl = out[6] as i32;
        assert!((bl - 128).abs() <= 2, "expected ~128 blue got {bl}");
    }

    #[test]
    fn in_keeps_a_color_masked_by_b_alpha() {
        // operator="in": Fa = αb, Fb = 0. Result = in clipped to in2's
        // coverage. Opaque green "in" half-alpha anything → green at
        // 50% alpha, original green RGB preserved (straight-alpha).
        let a = build(1, 1, |_, _| Rgba::new(0, 200, 0, 255));
        let b = build(1, 1, |_, _| Rgba::new(9, 9, 9, 128));
        let out = composite_filter(&a, &b, 1, 1, CompositeOp::In);
        // Alpha = 1.0 * (128/255) → ~128.
        assert!((out[3] as i32 - 128).abs() <= 1, "alpha {}", out[3]);
        // Straight-alpha green preserved exactly (premult then /α).
        assert_eq!(out[1], 200);
        // No contribution from in2's colour.
        assert_eq!(out[0], 0);
        assert_eq!(out[2], 0);
    }

    #[test]
    fn out_removes_a_where_b_covers() {
        // operator="out": Fa = 1 - αb, Fb = 0. Opaque a "out" opaque b
        // → fully transparent everywhere b covers.
        let a = build(1, 1, |_, _| Rgba::new(10, 20, 30, 255));
        let b = build(1, 1, |_, _| Rgba::new(0, 0, 0, 255));
        let out = composite_filter(&a, &b, 1, 1, CompositeOp::Out);
        assert_eq!(&out[0..4], &[0, 0, 0, 0]);
        // Where b is empty, a passes through untouched.
        let b0 = build(1, 1, |_, _| Rgba::new(0, 0, 0, 0));
        let out2 = composite_filter(&a, &b0, 1, 1, CompositeOp::Out);
        assert_eq!(&out2[0..4], &[10, 20, 30, 255]);
    }

    #[test]
    fn xor_of_disjoint_opaque_is_union() {
        // operator="xor": Fa = 1 - αb, Fb = 1 - αa. Two opaque inputs
        // that don't overlap (one empty per pixel) yield the non-empty
        // one; full overlap of two opaque inputs cancels to transparent.
        // `a` is opaque red at both pixels; only `b`'s coverage varies.
        let a = build(2, 1, |_, _| Rgba::new(255, 0, 0, 255));
        let b = build(2, 1, |x, _| {
            if x == 0 {
                Rgba::new(0, 0, 255, 0) // empty at pixel 0
            } else {
                Rgba::new(0, 0, 255, 255) // opaque overlap at pixel 1
            }
        });
        let out = composite_filter(&a, &b, 2, 1, CompositeOp::Xor);
        // Pixel 0: a opaque, b empty → a passes (Fa = 1 - 0 = 1).
        assert_eq!(&out[0..4], &[255, 0, 0, 255]);
        // Pixel 1: both opaque → αo = 1*0 + 1*0 = 0 → transparent.
        assert_eq!(out[7], 0);
    }

    #[test]
    fn atop_preserves_b_alpha() {
        // operator="atop": Fa = αb, Fb = 1 - αa. The result alpha
        // equals αb (a "atop" b is confined to b's silhouette):
        // αo = αa*αb + αb*(1-αa) = αb.
        let a = build(1, 1, |_, _| Rgba::new(255, 255, 255, 200));
        let b = build(1, 1, |_, _| Rgba::new(0, 0, 0, 128));
        let out = composite_filter(&a, &b, 1, 1, CompositeOp::Atop);
        assert!((out[3] as i32 - 128).abs() <= 1, "alpha {}", out[3]);
    }

    #[test]
    fn arithmetic_k2_one_passes_in1_through() {
        // result = 0*i1*i2 + 1*i1 + 0*i2 + 0 = i1. With both inputs
        // opaque the premultiplied channels equal the straight ones, so
        // in1 round-trips byte-exact.
        let a = build(3, 2, |x, y| {
            Rgba::new((x * 50) as u8, (y * 90) as u8, 33, 255)
        });
        let b = build(3, 2, |_, _| Rgba::new(200, 200, 200, 255));
        let out = composite_filter(
            &a,
            &b,
            3,
            2,
            CompositeOp::Arithmetic {
                k1: 0.0,
                k2: 1.0,
                k3: 0.0,
                k4: 0.0,
            },
        );
        assert_eq!(out, a);
    }

    #[test]
    fn arithmetic_k4_floods_constant() {
        // result = k4 on every channel (k1=k2=k3=0). k4=1 → opaque
        // white everywhere regardless of inputs.
        let a = build(2, 2, |_, _| Rgba::new(10, 20, 30, 40));
        let b = build(2, 2, |_, _| Rgba::new(50, 60, 70, 80));
        let out = composite_filter(
            &a,
            &b,
            2,
            2,
            CompositeOp::Arithmetic {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                k4: 1.0,
            },
        );
        // Premult result = (1,1,1,1) → straight (255,255,255,255).
        assert!(out.iter().all(|&v| v == 255));
    }

    #[test]
    fn arithmetic_clamps_to_unit_range() {
        // k2=k3=2 drives premultiplied sums well past 1.0; the spec
        // clamps each channel to [0,1]. Opaque inputs → channels clamp
        // to opaque white.
        let a = build(1, 1, |_, _| Rgba::new(200, 0, 0, 255));
        let b = build(1, 1, |_, _| Rgba::new(0, 200, 0, 255));
        let out = composite_filter(
            &a,
            &b,
            1,
            1,
            CompositeOp::Arithmetic {
                k1: 0.0,
                k2: 2.0,
                k3: 2.0,
                k4: 0.0,
            },
        );
        // Alpha: 2*1 + 2*1 = 4 → clamp 1 → 255.
        assert_eq!(out[3], 255);
    }

    #[test]
    fn typed_wrapper_matches_byte_path() {
        let a_b = build(4, 3, |x, y| {
            Rgba::new((x * 30) as u8, (y * 40) as u8, 11, ((x + y) * 25) as u8)
        });
        let b_b = build(4, 3, |x, y| {
            Rgba::new(7, (x * 20) as u8, (y * 50) as u8, 200)
        });
        let a_p: Vec<Rgba> = a_b
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        let b_p: Vec<Rgba> = b_b
            .chunks_exact(4)
            .map(|c| Rgba::new(c[0], c[1], c[2], c[3]))
            .collect();
        for op in [
            CompositeOp::Over,
            CompositeOp::In,
            CompositeOp::Out,
            CompositeOp::Atop,
            CompositeOp::Xor,
            CompositeOp::Arithmetic {
                k1: 0.25,
                k2: 0.5,
                k3: 0.5,
                k4: 0.1,
            },
        ] {
            let via_bytes = composite_filter(&a_b, &b_b, 4, 3, op);
            let via_typed = composite_filter_pixels(&a_p, &b_p, 4, 3, op);
            let typed_bytes: Vec<u8> = via_typed
                .iter()
                .flat_map(|p| [p.r, p.g, p.b, p.a])
                .collect();
            assert_eq!(via_bytes, typed_bytes, "mismatch for {op:?}");
        }
    }

    #[test]
    fn empty_extent_is_empty_vec() {
        assert!(composite_filter(&[], &[], 0, 0, CompositeOp::Over).is_empty());
    }

    #[test]
    #[should_panic(expected = "in1.len()")]
    fn wrong_in1_length_panics() {
        let _ = composite_filter(&[0u8; 8], &[0u8; 16], 2, 2, CompositeOp::Over);
    }

    #[test]
    #[should_panic(expected = "in2.len()")]
    fn wrong_in2_length_panics() {
        let _ = composite_filter(&[0u8; 16], &[0u8; 8], 2, 2, CompositeOp::Over);
    }
}
