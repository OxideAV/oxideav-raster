//! Filter primitive tree evaluation — Filter Effects Module Level 1
//! §9.2 (common attributes `in` / `result`) + §9.3 (filter primitive
//! tree).
//!
//! The [`crate::filter`] module implements every Filter Effects 1
//! primitive as a standalone buffer→buffer function. This module adds
//! the piece that connects them: a [`FilterGraph`] holds an ordered
//! list of [`FilterStep`]s (mirroring the filter primitive children of
//! a `filter` element, in document order) and evaluates them against a
//! source image with the §9.2 input-resolution rules:
//!
//! * **`in` unspecified** ([`FilterInput::Auto`]) — "If no value is
//!   provided and this is the first filter primitive, then this filter
//!   primitive will use `SourceGraphic` as its input. If no value is
//!   provided and this is a subsequent filter primitive, then this
//!   filter primitive will use the result from the previous filter
//!   primitive as its input." (§9.2)
//! * **`SourceGraphic`** — the original rasterised input image.
//! * **`SourceAlpha`** — "an RGBA image consisting of implicitly black
//!   color values for the RGB channels, but whose alpha channel is the
//!   same as `SourceGraphic`" (§9.2).
//! * **`result` back-references** — "If the value for `result` appears
//!   multiple times within a given `filter` element, then a reference
//!   to that result will use the closest preceding filter primitive
//!   with the given value for attribute `result`." Forward references
//!   and references to non-existent results "will be treated as if no
//!   result was specified" (§9.2) — i.e. they fall back to the `Auto`
//!   rule above.
//!
//! Per §9.3 only the **primary filter primitive tree** — the tree
//! rooted at the *last* filter primitive child — contributes to the
//! filter output, so [`FilterGraph::evaluate`] returns the final
//! step's result; earlier disconnected trees are naturally computed
//! and discarded. A graph with no steps renders nothing (§9.3: "If a
//! `filter` element has no filter primitive tree then the element the
//! filter applies to does not get rendered") — the evaluator returns
//! transparent black.
//!
//! The §9.2 keywords `BackgroundImage` / `BackgroundAlpha` are not
//! offered: Filter Effects 1 Appendix A drops the `enable-background`
//! property that defined the backdrop they read, so no conforming
//! content can produce one. `FillPaint` / `StrokePaint` require the
//! filtered element's paint context, which a buffer-level evaluator
//! does not have; both are out of scope for this module.
//!
//! # Working colour space
//!
//! A graph carries one [`FilterColorSpace`]
//! (`color-interpolation-filters`, Filter Effects 1 §10; initial value
//! `linearRGB`). Under a linear space the evaluator linearises
//! `SourceGraphic` **once** on entry, keeps every intermediate result
//! linear (chained primitives therefore pay no per-step conversions),
//! and re-encodes the final result to sRGB on exit. Colours that enter
//! the graph mid-chain as *specified values* — the `feFlood` /
//! `feDropShadow` flood colour and `feImage` pixels, which are
//! sRGB-encoded by definition — are linearised at the step that
//! introduces them. `feTurbulence` is generated directly in the
//! working space: its §9.21 noise function defines channel values with
//! no reference to an encoding, so the generated numbers *are* the
//! working-space values.

use crate::filter::{
    blend_filter, clip_to_subregion, color_matrix_op, component_transfer, composite_filter,
    convolve_matrix, diffuse_lighting, displacement_map, drop_shadow, flood, gaussian_blur_edge,
    image_source, merge, morphology, offset, specular_lighting, srgb_to_linear_f32, to_linear_rgb,
    to_srgb, turbulence_filter, BlendFilterMode, ColorMatrix, ColorMatrixOp, ComponentTransfer,
    CompositeOp, ConvolveEdgeMode, ConvolveMatrix, DiffuseLighting, DisplacementChannel,
    DisplacementSampling, FilterColorSpace, ImageSourceSampling, MorphologyOp, OffsetSampling,
    PreserveAspectRatio, SpecularLighting, Subregion, Turbulence,
};

/// Input selector for a filter primitive — the §9.2 `in` / `in2`
/// attribute value.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FilterInput {
    /// No `in` value provided. §9.2: the first filter primitive reads
    /// `SourceGraphic`; a subsequent one reads the previous
    /// primitive's result.
    #[default]
    Auto,
    /// The `SourceGraphic` keyword: the original input image.
    SourceGraphic,
    /// The `SourceAlpha` keyword: black RGB with `SourceGraphic`'s
    /// alpha channel.
    SourceAlpha,
    /// A `<filter-primitive-reference>` naming a preceding step's
    /// `result`. Resolution uses the **closest preceding** step with
    /// that name; a forward or dangling reference degrades to
    /// [`Auto`](Self::Auto) per §9.2.
    Reference(String),
}

/// One filter primitive plus its primitive-specific parameters.
///
/// Each variant wraps the parameter set of the corresponding
/// standalone function in [`crate::filter`]; the *inputs* it consumes
/// live on the enclosing [`FilterStep`] (`input` for `in`, `input2`
/// for `in2`), except [`Merge`](Self::Merge), whose per-`feMergeNode`
/// input list is part of the primitive itself.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterPrimitive {
    /// `feBlend` (§9.5) — blends `in` over `in2` under a
    /// Compositing-1 blend mode.
    Blend {
        /// The `mode` attribute.
        mode: BlendFilterMode,
    },
    /// `feColorMatrix` (§9.6).
    ColorMatrix {
        /// The `type` attribute (matrix / saturate / hueRotate /
        /// luminanceToAlpha).
        op: ColorMatrixOp,
        /// The user matrix consumed when `op` is
        /// [`ColorMatrixOp::Matrix`]; ignored otherwise.
        matrix: ColorMatrix,
    },
    /// `feComponentTransfer` (§9.7).
    ComponentTransfer(ComponentTransfer),
    /// `feComposite` (§9.8) — combines `in` with `in2`.
    Composite {
        /// The `operator` attribute (Porter-Duff or arithmetic).
        op: CompositeOp,
    },
    /// `feConvolveMatrix` (§9.9).
    ConvolveMatrix(ConvolveMatrix),
    /// `feDiffuseLighting` (§9.10).
    DiffuseLighting(DiffuseLighting),
    /// `feDisplacementMap` (§9.11) — warps `in` by the channel values
    /// of `in2`.
    DisplacementMap {
        /// The `scale` attribute.
        scale: f32,
        /// The `xChannelSelector` attribute.
        x_channel: DisplacementChannel,
        /// The `yChannelSelector` attribute.
        y_channel: DisplacementChannel,
        /// Reconstruction filter for the displaced sample.
        sampling: DisplacementSampling,
    },
    /// `feDropShadow` (§9.12).
    DropShadow {
        /// Gaussian standard deviation, X axis.
        std_x: f32,
        /// Gaussian standard deviation, Y axis.
        std_y: f32,
        /// Shadow offset, X axis.
        dx: f32,
        /// Shadow offset, Y axis.
        dy: f32,
        /// `flood-color` RGB (sRGB-encoded, as specified).
        color: (u8, u8, u8),
        /// `flood-opacity`.
        opacity: f32,
        /// Reconstruction filter for the offset stage.
        sampling: OffsetSampling,
    },
    /// `feFlood` (§9.13) — generator; ignores `input`.
    Flood {
        /// `flood-color` RGB (sRGB-encoded, as specified).
        color: (u8, u8, u8),
        /// `flood-opacity`.
        opacity: f32,
    },
    /// `feGaussianBlur` (§9.14).
    GaussianBlur {
        /// `stdDeviation`, X axis.
        std_x: f32,
        /// `stdDeviation`, Y axis.
        std_y: f32,
        /// The §9.14 `edgeMode` attribute.
        edge: ConvolveEdgeMode,
    },
    /// `feImage` (§9.15) — generator; ignores `input`. The referenced
    /// raster is fitted to the canvas under `preserveAspectRatio`.
    Image {
        /// Source raster, packed straight-alpha RGBA (sRGB-encoded).
        pixels: Vec<u8>,
        /// Source raster width in pixels.
        width: u32,
        /// Source raster height in pixels.
        height: u32,
        /// The `preserveAspectRatio` fitting attribute.
        par: PreserveAspectRatio,
        /// Reconstruction filter for the fit.
        sampling: ImageSourceSampling,
    },
    /// `feMerge` (§9.16) — composites the listed inputs bottom-to-top
    /// with `over`. Each entry mirrors one `feMergeNode`'s `in`
    /// attribute and resolves under the same §9.2 rules as a step
    /// input (an [`FilterInput::Auto`] node reads the previous
    /// *step*'s result, or `SourceGraphic` if this is the first step).
    /// The step-level `input` / `input2` are ignored.
    Merge {
        /// One input per `feMergeNode`, bottom layer first.
        inputs: Vec<FilterInput>,
    },
    /// `feMorphology` (§9.17).
    Morphology {
        /// Structuring-element half-extent, X axis (`radius` x).
        radius_x: u32,
        /// Structuring-element half-extent, Y axis (`radius` y).
        radius_y: u32,
        /// The `operator` attribute (erode / dilate).
        op: MorphologyOp,
    },
    /// `feOffset` (§9.18).
    Offset {
        /// The `dx` attribute.
        dx: f32,
        /// The `dy` attribute.
        dy: f32,
        /// Reconstruction filter for fractional offsets.
        sampling: OffsetSampling,
    },
    /// `feSpecularLighting` (§9.19).
    SpecularLighting(SpecularLighting),
    /// `feTile` (§9.20) — fills the canvas with a repeated tile cut
    /// from the input. `tile` plays the role of the *input*
    /// primitive's subregion: the §9.20 tiles sit at
    /// `(tile.x + i·tile.width, tile.y + j·tile.height)` for all
    /// integers `i`, `j`. A non-positive tile extent disables the
    /// primitive (transparent black), mirroring §9.4.
    Tile {
        /// The input tile rectangle, in canvas pixel units.
        tile: Subregion,
    },
    /// `feTurbulence` (§9.21) — generator; ignores `input`.
    Turbulence(Turbulence),
}

/// One filter primitive child of a `filter` element: the primitive
/// itself plus its §9.2 common attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterStep {
    /// The primitive and its parameters.
    pub primitive: FilterPrimitive,
    /// The `in` attribute.
    pub input: FilterInput,
    /// The `in2` attribute (consumed by `feBlend` / `feComposite` /
    /// `feDisplacementMap`; ignored by single-input primitives).
    pub input2: FilterInput,
    /// The `result` attribute — an assigned name later steps may
    /// reference via [`FilterInput::Reference`].
    pub result: Option<String>,
    /// The §9.4 filter primitive subregion, applied as a hard clip on
    /// this step's result.
    pub subregion: Option<Subregion>,
}

impl FilterStep {
    /// A step around `primitive` with both inputs [`FilterInput::Auto`],
    /// no `result` name, and no subregion.
    pub fn new(primitive: FilterPrimitive) -> Self {
        Self {
            primitive,
            input: FilterInput::Auto,
            input2: FilterInput::Auto,
            result: None,
            subregion: None,
        }
    }

    /// Set the `in` attribute.
    pub fn with_input(mut self, input: FilterInput) -> Self {
        self.input = input;
        self
    }

    /// Set the `in2` attribute.
    pub fn with_input2(mut self, input2: FilterInput) -> Self {
        self.input2 = input2;
        self
    }

    /// Assign a `result` name.
    pub fn with_result(mut self, name: impl Into<String>) -> Self {
        self.result = Some(name.into());
        self
    }

    /// Attach a §9.4 filter primitive subregion.
    pub fn with_subregion(mut self, subregion: Subregion) -> Self {
        self.subregion = Some(subregion);
        self
    }
}

/// An ordered filter primitive list (§9.3 filter primitive tree) plus
/// its working colour space, evaluated against a source image with
/// [`FilterGraph::evaluate`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FilterGraph {
    steps: Vec<FilterStep>,
    color_space: FilterColorSpace,
}

impl FilterGraph {
    /// An empty graph in the initial working space
    /// (`color-interpolation-filters: linearRGB`, §10).
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty graph in an explicit working space.
    pub fn with_color_space(color_space: FilterColorSpace) -> Self {
        Self {
            steps: Vec::new(),
            color_space,
        }
    }

    /// Append a step (one filter primitive child, in document order).
    pub fn push(&mut self, step: FilterStep) -> &mut Self {
        self.steps.push(step);
        self
    }

    /// The steps appended so far, in document order.
    pub fn steps(&self) -> &[FilterStep] {
        &self.steps
    }

    /// The graph's working colour space.
    pub fn color_space(&self) -> FilterColorSpace {
        self.color_space
    }

    /// Evaluate the graph against `source` (packed straight-alpha
    /// RGBA, `width × height`, sRGB-encoded) and return the filtered
    /// image in the same format.
    ///
    /// An empty graph returns transparent black (§9.3: an element
    /// whose filter has no filter primitive tree is not rendered).
    ///
    /// # Panics
    ///
    /// * If `source.len() != width as usize * height as usize * 4`.
    /// * If `width as usize * height as usize * 4` overflows `usize`.
    pub fn evaluate(&self, source: &[u8], width: u32, height: u32) -> Vec<u8> {
        let n = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .expect("FilterGraph::evaluate: width * height * 4 overflowed usize");
        assert_eq!(
            source.len(),
            n,
            "FilterGraph::evaluate: source.len() == {} but width*height*4 == {n}",
            source.len()
        );
        if self.steps.is_empty() {
            return vec![0u8; n];
        }

        let linear = self.color_space.needs_linearisation();

        // SourceGraphic, converted once into the working space.
        let source_graphic: Vec<u8> = if linear {
            let mut buf = source.to_vec();
            to_linear_rgb(&mut buf);
            buf
        } else {
            source.to_vec()
        };
        // SourceAlpha (§9.2): black RGB + SourceGraphic's alpha. Alpha
        // is never gamma-encoded, so this is space-independent; built
        // lazily since many graphs never read it.
        let mut source_alpha: Option<Vec<u8>> = None;

        // Computed step results, indexed by step position.
        let mut computed: Vec<Vec<u8>> = Vec::with_capacity(self.steps.len());

        for (idx, step) in self.steps.iter().enumerate() {
            let out = match &step.primitive {
                FilterPrimitive::Blend { mode } => {
                    let a = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    )
                    .to_vec();
                    let b = resolve_input(
                        &step.input2,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    blend_filter(&a, b, width, height, *mode)
                }
                FilterPrimitive::ColorMatrix { op, matrix } => {
                    let src = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    color_matrix_op(src, width, height, *op, matrix)
                }
                FilterPrimitive::ComponentTransfer(ct) => {
                    let src = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    component_transfer(src, width, height, ct)
                }
                FilterPrimitive::Composite { op } => {
                    let a = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    )
                    .to_vec();
                    let b = resolve_input(
                        &step.input2,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    composite_filter(&a, b, width, height, *op)
                }
                FilterPrimitive::ConvolveMatrix(cm) => {
                    let src = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    convolve_matrix(src, width, height, cm)
                }
                FilterPrimitive::DiffuseLighting(params) => {
                    let src = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    diffuse_lighting(src, width, height, params)
                }
                FilterPrimitive::DisplacementMap {
                    scale,
                    x_channel,
                    y_channel,
                    sampling,
                } => {
                    let a = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    )
                    .to_vec();
                    let b = resolve_input(
                        &step.input2,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    displacement_map(
                        &a, b, width, height, *scale, *x_channel, *y_channel, *sampling,
                    )
                }
                FilterPrimitive::DropShadow {
                    std_x,
                    std_y,
                    dx,
                    dy,
                    color,
                    opacity,
                    sampling,
                } => {
                    let src = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    // flood-color is a specified (sRGB) colour; move it
                    // into the working space at the step that
                    // introduces it (see the module-level colour-space
                    // notes).
                    let (r, g, b) = if linear {
                        (
                            linearise_u8(color.0),
                            linearise_u8(color.1),
                            linearise_u8(color.2),
                        )
                    } else {
                        *color
                    };
                    drop_shadow(
                        src, width, height, *std_x, *std_y, *dx, *dy, r, g, b, *opacity, *sampling,
                    )
                }
                FilterPrimitive::Flood { color, opacity } => {
                    let (r, g, b) = if linear {
                        (
                            linearise_u8(color.0),
                            linearise_u8(color.1),
                            linearise_u8(color.2),
                        )
                    } else {
                        *color
                    };
                    flood(width, height, r, g, b, *opacity)
                }
                FilterPrimitive::GaussianBlur { std_x, std_y, edge } => {
                    let src = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    gaussian_blur_edge(src, width, height, *std_x, *std_y, *edge)
                }
                FilterPrimitive::Image {
                    pixels,
                    width: img_w,
                    height: img_h,
                    par,
                    sampling,
                } => {
                    let mut out =
                        image_source(pixels, *img_w, *img_h, width, height, *par, *sampling);
                    if linear {
                        // feImage pixels are sRGB-encoded; convert on
                        // entry into the working space.
                        to_linear_rgb(&mut out);
                    }
                    out
                }
                FilterPrimitive::Merge { inputs } => {
                    let layers: Vec<Vec<u8>> = inputs
                        .iter()
                        .map(|input| {
                            resolve_input(
                                input,
                                idx,
                                &self.steps,
                                &computed,
                                &source_graphic,
                                &mut source_alpha,
                            )
                            .to_vec()
                        })
                        .collect();
                    let refs: Vec<&[u8]> = layers.iter().map(|l| l.as_slice()).collect();
                    merge(width, height, &refs)
                }
                FilterPrimitive::Morphology {
                    radius_x,
                    radius_y,
                    op,
                } => {
                    let src = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    morphology(src, width, height, *radius_x, *radius_y, *op)
                }
                FilterPrimitive::Offset { dx, dy, sampling } => {
                    let src = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    offset(src, width, height, *dx, *dy, *sampling)
                }
                FilterPrimitive::SpecularLighting(params) => {
                    let src = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    specular_lighting(src, width, height, params)
                }
                FilterPrimitive::Tile { tile } => {
                    let src = resolve_input(
                        &step.input,
                        idx,
                        &self.steps,
                        &computed,
                        &source_graphic,
                        &mut source_alpha,
                    );
                    anchored_tile(src, width, height, *tile)
                }
                FilterPrimitive::Turbulence(params) => {
                    // Generated directly in the working space (see the
                    // module-level colour-space notes).
                    turbulence_filter(width, height, params)
                }
            };

            let mut out = out;
            if let Some(sub) = step.subregion {
                clip_to_subregion(&mut out, width, height, sub);
            }
            computed.push(out);
        }

        let mut result = computed
            .pop()
            .expect("FilterGraph::evaluate: non-empty step list produced no result");
        if linear {
            to_srgb(&mut result);
        }
        result
    }
}

/// Move one sRGB-encoded byte into the linear working space.
#[inline]
fn linearise_u8(c: u8) -> u8 {
    (srgb_to_linear_f32(c as f32 / 255.0) * 255.0).round() as u8
}

/// Resolve a [`FilterInput`] for the step at `idx` per the §9.2 rules.
///
/// `computed` holds the results of steps `0..idx`. A dangling or
/// forward [`FilterInput::Reference`] degrades to [`FilterInput::Auto`]
/// ("treated as if no result was specified", §9.2), which in turn reads
/// the previous step's result — or `SourceGraphic` when `idx == 0`.
fn resolve_input<'a>(
    input: &FilterInput,
    idx: usize,
    steps: &[FilterStep],
    computed: &'a [Vec<u8>],
    source_graphic: &'a [u8],
    source_alpha: &'a mut Option<Vec<u8>>,
) -> &'a [u8] {
    match input {
        FilterInput::SourceGraphic => source_graphic,
        FilterInput::SourceAlpha => source_alpha.get_or_insert_with(|| {
            let mut buf = vec![0u8; source_graphic.len()];
            for (dst, src) in buf.chunks_exact_mut(4).zip(source_graphic.chunks_exact(4)) {
                dst[3] = src[3];
            }
            buf
        }),
        FilterInput::Reference(name) => {
            // "a reference to that result will use the closest
            // preceding filter primitive with the given value for
            // attribute result" (§9.2) — scan steps 0..idx backwards.
            for prev in (0..idx).rev() {
                if steps[prev].result.as_deref() == Some(name.as_str()) {
                    return &computed[prev];
                }
            }
            // Forward / non-existent reference → Auto (§9.2).
            resolve_auto(idx, computed, source_graphic)
        }
        FilterInput::Auto => resolve_auto(idx, computed, source_graphic),
    }
}

/// The §9.2 unspecified-`in` rule: previous step's result, or
/// `SourceGraphic` for the first step.
fn resolve_auto<'a>(idx: usize, computed: &'a [Vec<u8>], source_graphic: &'a [u8]) -> &'a [u8] {
    if idx == 0 {
        source_graphic
    } else {
        &computed[idx - 1]
    }
}

/// `feTile` (§9.20) with an explicit tile anchor: output pixel
/// `(px, py)` samples the input at the tile-relative coordinate
/// obtained from the Euclidean remainder of `(px - x, py - y)` against
/// the tile extent, so tiles sit at `(x + i·w, y + j·h)` for all
/// integers `i`, `j` — "the top/left corner of each given tile is at
/// location `(x + i*width, y + j*height)` … i and j can be any integer
/// value" (SVG 1.1 §15.23 / Filter Effects 1 §9.20).
///
/// The tile rectangle is snapped outward to whole pixels with the same
/// partly-intersecting-pixels-kept convention [`clip_to_subregion`]
/// uses. A non-positive (or NaN) tile extent disables the primitive:
/// the result is transparent black, mirroring the §9.4 disabled-
/// subregion rule.
fn anchored_tile(src: &[u8], width: u32, height: u32, tile: Subregion) -> Vec<u8> {
    let n = width as usize * height as usize * 4;
    let positive = tile.width > 0.0 && tile.height > 0.0;
    if !positive {
        return vec![0u8; n];
    }
    // Snap outward to the pixel grid, clamped to the canvas.
    let x0 = (tile.x.floor().max(0.0) as u32).min(width);
    let y0 = (tile.y.floor().max(0.0) as u32).min(height);
    let x1 = ((tile.x + tile.width).ceil().max(0.0) as u32).min(width);
    let y1 = ((tile.y + tile.height).ceil().max(0.0) as u32).min(height);
    let tw = x1.saturating_sub(x0);
    let th = y1.saturating_sub(y0);
    if tw == 0 || th == 0 {
        return vec![0u8; n];
    }
    let mut out = vec![0u8; n];
    let w = width as usize;
    for py in 0..height {
        // Euclidean remainder keeps the anchor correct for output
        // pixels above / left of the tile origin.
        let ty = (py as i64 - y0 as i64).rem_euclid(th as i64) as u32 + y0;
        for px in 0..width {
            let tx = (px as i64 - x0 as i64).rem_euclid(tw as i64) as u32 + x0;
            let src_i = (ty as usize * w + tx as usize) * 4;
            let dst_i = (py as usize * w + px as usize) * 4;
            out[dst_i..dst_i + 4].copy_from_slice(&src[src_i..src_i + 4]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::gaussian_blur;

    fn checker(width: u32, height: u32) -> Vec<u8> {
        let mut buf = vec![0u8; width as usize * height as usize * 4];
        for y in 0..height {
            for x in 0..width {
                let i = (y as usize * width as usize + x as usize) * 4;
                if (x + y) % 2 == 0 {
                    buf[i..i + 4].copy_from_slice(&[200, 40, 90, 255]);
                } else {
                    buf[i..i + 4].copy_from_slice(&[10, 220, 60, 128]);
                }
            }
        }
        buf
    }

    #[test]
    fn empty_graph_is_transparent_black() {
        let g = FilterGraph::new();
        let out = g.evaluate(&checker(4, 4), 4, 4);
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn auto_input_on_first_step_reads_source_graphic() {
        let src = checker(6, 6);
        let mut g = FilterGraph::with_color_space(FilterColorSpace::Srgb);
        g.push(FilterStep::new(FilterPrimitive::GaussianBlur {
            std_x: 1.0,
            std_y: 1.0,
            edge: ConvolveEdgeMode::Duplicate,
        }));
        let direct = gaussian_blur(&src, 6, 6, 1.0, 1.0);
        assert_eq!(g.evaluate(&src, 6, 6), direct);
    }

    #[test]
    fn source_alpha_is_black_rgb_with_source_alpha_channel() {
        let src = checker(4, 4);
        let mut g = FilterGraph::with_color_space(FilterColorSpace::Srgb);
        g.push(
            FilterStep::new(FilterPrimitive::Offset {
                dx: 0.0,
                dy: 0.0,
                sampling: OffsetSampling::default(),
            })
            .with_input(FilterInput::SourceAlpha),
        );
        let out = g.evaluate(&src, 4, 4);
        for (o, s) in out.chunks_exact(4).zip(src.chunks_exact(4)) {
            assert_eq!(&o[..3], &[0, 0, 0], "RGB must be implicitly black");
            assert_eq!(o[3], s[3], "alpha channel must match SourceGraphic");
        }
    }

    #[test]
    fn forward_reference_degrades_to_auto() {
        // Step 1 flood(red, result later shadowed), step 2 references a
        // name only defined by step 3 → §9.2 says the reference is
        // treated as unspecified, i.e. step 2 reads step 1's result.
        let src = checker(4, 4);
        let mut g = FilterGraph::with_color_space(FilterColorSpace::Srgb);
        g.push(FilterStep::new(FilterPrimitive::Flood {
            color: (255, 0, 0),
            opacity: 1.0,
        }));
        g.push(
            FilterStep::new(FilterPrimitive::Offset {
                dx: 0.0,
                dy: 0.0,
                sampling: OffsetSampling::default(),
            })
            .with_input(FilterInput::Reference("late".into())),
        );
        g.push(
            FilterStep::new(FilterPrimitive::Flood {
                color: (0, 255, 0),
                opacity: 1.0,
            })
            .with_result("late"),
        );
        // Evaluate steps 1+2 only by dropping the last step: emulate by
        // reading the full graph's step-2 semantics through a 2-step
        // graph instead.
        let mut g2 = FilterGraph::with_color_space(FilterColorSpace::Srgb);
        g2.push(FilterStep::new(FilterPrimitive::Flood {
            color: (255, 0, 0),
            opacity: 1.0,
        }));
        g2.push(
            FilterStep::new(FilterPrimitive::Offset {
                dx: 0.0,
                dy: 0.0,
                sampling: OffsetSampling::default(),
            })
            .with_input(FilterInput::Reference("late".into())),
        );
        let out = g2.evaluate(&src, 4, 4);
        assert_eq!(&out[..4], &[255, 0, 0, 255]);
        // And the 3-step graph's final output is the green flood.
        let out3 = g.evaluate(&src, 4, 4);
        assert_eq!(&out3[..4], &[0, 255, 0, 255]);
    }

    #[test]
    fn duplicate_result_names_use_closest_preceding() {
        let src = checker(4, 4);
        let mut g = FilterGraph::with_color_space(FilterColorSpace::Srgb);
        g.push(
            FilterStep::new(FilterPrimitive::Flood {
                color: (255, 0, 0),
                opacity: 1.0,
            })
            .with_result("f"),
        );
        g.push(
            FilterStep::new(FilterPrimitive::Flood {
                color: (0, 0, 255),
                opacity: 1.0,
            })
            .with_result("f"),
        );
        g.push(
            FilterStep::new(FilterPrimitive::Offset {
                dx: 0.0,
                dy: 0.0,
                sampling: OffsetSampling::default(),
            })
            .with_input(FilterInput::Reference("f".into())),
        );
        let out = g.evaluate(&src, 4, 4);
        assert_eq!(&out[..4], &[0, 0, 255, 255], "closest preceding wins");
    }

    #[test]
    fn anchored_tile_repeats_with_offset_anchor() {
        // 1×1 tile at (1, 1) cut from a buffer whose (1, 1) pixel is
        // unique → whole output becomes that pixel.
        let mut src = vec![0u8; 3 * 3 * 4];
        let i = (3 + 1) * 4;
        src[i..i + 4].copy_from_slice(&[9, 8, 7, 255]);
        let out = anchored_tile(&src, 3, 3, Subregion::new(1.0, 1.0, 1.0, 1.0));
        for px in out.chunks_exact(4) {
            assert_eq!(px, &[9, 8, 7, 255]);
        }
    }

    #[test]
    fn anchored_tile_disabled_on_non_positive_extent() {
        let src = checker(3, 3);
        for (w, h) in [(0.0, 1.0), (1.0, -2.0), (f32::NAN, 1.0)] {
            let out = anchored_tile(&src, 3, 3, Subregion::new(0.0, 0.0, w, h));
            assert!(out.iter().all(|&b| b == 0));
        }
    }
}
