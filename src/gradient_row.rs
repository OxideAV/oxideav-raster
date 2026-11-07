//! Row-wise gradient painting for the scene renderer.
//!
//! The per-pixel gradient path maps every device pixel centre back
//! through the inverse CTM, solves for the gradient parameter `t` and
//! looks the colour up in a 256-entry table. This module evaluates a
//! whole coverage row at once instead:
//!
//! * **Geometry.** For a linear gradient `t` is an affine function of
//!   the device position, `t = A·x + B·y + C`, so a row is one
//!   multiply-add per pixel; a radial gradient's user-space position is
//!   affine in `x` as well. The set-up is done in `f64`, so gradients
//!   spanning huge user-space ranges keep their precision.
//! * **Colour.** The stops are evaluated exactly per pixel from a
//!   precomputed segment list (no table quantisation): a hard stop
//!   (two stops at one offset) stays a hard edge at any gradient
//!   length, where a 256-entry table smears it across `1/255` of the
//!   gradient vector (15 px on a 4K-wide gradient). For
//!   `color-interpolation: linearRGB` the re-encoding to sRGB uses a
//!   4096-interval table of the IEC 61966-2-1 transfer function with
//!   linear interpolation (error < 0.01 of an output level).
//! * **Dither (optional).** With [`GradientDither::Ordered`] the
//!   quantisation to 8 bits adds an 8×8 Bayer ordered-dither threshold
//!   (index matrix from the dispersed-dot recurrence) instead of
//!   rounding, so the local average of the output equals the exact
//!   gradient value and wide, low-contrast ramps show no 1-level
//!   banding. Off by default: it adds deterministic ±½-level texture,
//!   which downstream video encoders spend bits on.

use oxideav_core::{GradientStop, Paint, Rgba, SpreadMethod, Transform2D};

use crate::gradient::{
    InterpolationSpace, apply_spread, clamp_focal_inside_circle, srgb_to_linear,
};

/// Quantisation of gradient colours to 8 bits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum GradientDither {
    /// Round to nearest (the default).
    #[default]
    None,
    /// 8×8 Bayer ordered dither: removes 1-level banding from wide,
    /// low-contrast gradients at the cost of a fixed ±½-level texture.
    Ordered,
}

/// The 8×8 Bayer index matrix, from the recurrence
/// `M₂ₙ = [[4Mₙ, 4Mₙ+2], [4Mₙ+3, 4Mₙ+1]]` with `M₂ = [[0, 2], [3, 1]]`.
const BAYER8: [[u8; 8]; 8] = {
    let m2 = [[0u8, 2], [3, 1]];
    let mut m4 = [[0u8; 4]; 4];
    let mut y = 0;
    while y < 4 {
        let mut x = 0;
        while x < 4 {
            m4[y][x] = 4 * m2[y % 2][x % 2] + m2[y / 2][x / 2];
            x += 1;
        }
        y += 1;
    }
    let mut m8 = [[0u8; 8]; 8];
    let mut y = 0;
    while y < 8 {
        let mut x = 0;
        while x < 8 {
            m8[y][x] = 4 * m4[y % 4][x % 4] + m2[y / 4][x / 4];
            x += 1;
        }
        y += 1;
    }
    m8
};

/// Rounding threshold for device pixel `(x, y)`: `0.5` (round to
/// nearest) or the Bayer threshold `(M[y][x] + ½) / 64`.
#[inline(always)]
fn threshold(dither: GradientDither, x: u32, y: u32) -> f32 {
    match dither {
        GradientDither::Ordered => (BAYER8[(y & 7) as usize][(x & 7) as usize] as f32 + 0.5) / 64.0,
        _ => 0.5,
    }
}

pub(crate) const ENCODE_INTERVALS: usize = 4096;

/// sRGB encoding (scaled to 0..255) of linear light at
/// `i / ENCODE_INTERVALS`.
pub(crate) fn encode_table() -> &'static [f32; ENCODE_INTERVALS + 1] {
    static T: std::sync::OnceLock<Box<[f32; ENCODE_INTERVALS + 1]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = Box::new([0f32; ENCODE_INTERVALS + 1]);
        for (i, v) in t.iter_mut().enumerate() {
            let c = i as f64 / ENCODE_INTERVALS as f64;
            let cs = if c <= 0.003_130_8 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            *v = (cs * 255.0) as f32;
        }
        t
    })
}

#[inline(always)]
pub(crate) fn encode_linear(table: &[f32; ENCODE_INTERVALS + 1], c: f32) -> f32 {
    let x = c.clamp(0.0, 1.0) * ENCODE_INTERVALS as f32;
    let i = (x as usize).min(ENCODE_INTERVALS - 1);
    let f = x - i as f32;
    table[i] + (table[i + 1] - table[i]) * f
}

/// One stop pair, in the interpolation space: sRGB channels as 0..255,
/// or linear-light RGB as 0..1 (alpha always 0..255).
#[derive(Debug, Clone, Copy)]
struct Seg {
    t0: f32,
    t1: f32,
    c0: [f32; 4],
    /// `(c1 − c0) / (t1 − t0)`.
    slope: [f32; 4],
}

/// The colour ramp of a gradient's stops, evaluated exactly.
#[derive(Debug, Clone)]
struct Ramp {
    first_t: f32,
    last_t: f32,
    first: [f32; 4],
    last: [f32; 4],
    segs: Vec<Seg>,
    linear: bool,
}

impl Ramp {
    fn new(stops: &[GradientStop], space: InterpolationSpace) -> Self {
        let linear = matches!(space, InterpolationSpace::LinearRgb);
        let conv = |c: Rgba| -> [f32; 4] {
            if linear {
                [
                    srgb_to_linear(c.r),
                    srgb_to_linear(c.g),
                    srgb_to_linear(c.b),
                    c.a as f32,
                ]
            } else {
                [c.r as f32, c.g as f32, c.b as f32, c.a as f32]
            }
        };
        let segs = stops
            .windows(2)
            .map(|w| {
                let span = (w[1].offset - w[0].offset).max(1e-9);
                let (c0, c1) = (conv(w[0].color), conv(w[1].color));
                Seg {
                    t0: w[0].offset,
                    t1: w[1].offset,
                    c0,
                    slope: std::array::from_fn(|k| (c1[k] - c0[k]) / span),
                }
            })
            .collect();
        let (f, l) = (stops[0], stops[stops.len() - 1]);
        Ramp {
            first_t: f.offset,
            last_t: l.offset,
            first: conv(f.color),
            last: conv(l.color),
            segs,
            linear,
        }
    }

    /// The colour at `t` (already mapped through the spread method),
    /// in the interpolation space — the same stop-selection rule as the
    /// per-pixel evaluator.
    #[inline]
    fn eval(&self, t: f32) -> [f32; 4] {
        if t <= self.first_t {
            return self.first;
        }
        if t >= self.last_t {
            return self.last;
        }
        for s in &self.segs {
            if t >= s.t0 && t <= s.t1 {
                let dt = t - s.t0;
                return std::array::from_fn(|k| s.c0[k] + s.slope[k] * dt);
            }
        }
        self.last
    }

    #[inline(always)]
    fn quantise(&self, v: [f32; 4], thr: f32, table: &[f32; ENCODE_INTERVALS + 1]) -> Rgba {
        // `as u8` saturates (and maps NaN to 0), so this is
        // `clamp(floor(x + thr), 0, 255)`.
        let q = |x: f32| (x + thr) as u8;
        if self.linear {
            Rgba::new(
                q(encode_linear(table, v[0])),
                q(encode_linear(table, v[1])),
                q(encode_linear(table, v[2])),
                q(v[3]),
            )
        } else {
            Rgba::new(q(v[0]), q(v[1]), q(v[2]), q(v[3]))
        }
    }
}

#[derive(Debug, Clone)]
enum Geometry {
    /// Every pixel paints this colour.
    Constant(Rgba),
    /// `t = a·x + b·y + c` at device pixel centre `(x + ½, y + ½)`.
    Linear { a: f64, b: f64, c: f64 },
    /// Device → user affine plus the radial constants.
    Radial {
        inv: [f64; 6],
        fx: f64,
        fy: f64,
        cdx: f64,
        cdy: f64,
        r: f64,
    },
}

/// A linear / radial gradient paint prepared for row evaluation.
#[derive(Debug, Clone)]
pub(crate) struct GradientRows {
    geom: Geometry,
    ramp: Option<Ramp>,
    spread: SpreadMethod,
    dither: GradientDither,
}

impl GradientRows {
    /// Prepare `paint` (a gradient in the user space that `inv` maps
    /// device pixels into). `None` for non-gradient paints.
    pub(crate) fn new(
        paint: &Paint,
        inv: &Transform2D,
        space: InterpolationSpace,
        dither: GradientDither,
    ) -> Option<Self> {
        let m = [
            inv.a as f64,
            inv.b as f64,
            inv.c as f64,
            inv.d as f64,
            inv.e as f64,
            inv.f as f64,
        ];
        let (geom, stops, spread) = match paint {
            Paint::LinearGradient(g) => {
                let geom = if g.stops.is_empty() {
                    Geometry::Constant(Rgba::new(0, 0, 0, 0))
                } else if g.stops.len() == 1 {
                    Geometry::Constant(g.stops[0].color)
                } else {
                    let dx = g.end.x - g.start.x;
                    let dy = g.end.y - g.start.y;
                    let denom = dx * dx + dy * dy;
                    if denom <= 1e-12 {
                        Geometry::Constant(g.stops[g.stops.len() - 1].color)
                    } else {
                        let (dx, dy, denom) = (dx as f64, dy as f64, denom as f64);
                        let (sx, sy) = (g.start.x as f64, g.start.y as f64);
                        // user(x, y) = (m0·X + m2·Y + m4, m1·X + m3·Y + m5)
                        // at X = x + ½, Y = y + ½.
                        let a = (m[0] * dx + m[1] * dy) / denom;
                        let b = (m[2] * dx + m[3] * dy) / denom;
                        let ux = m[4] + 0.5 * (m[0] + m[2]) - sx;
                        let uy = m[5] + 0.5 * (m[1] + m[3]) - sy;
                        Geometry::Linear {
                            a,
                            b,
                            c: (ux * dx + uy * dy) / denom,
                        }
                    }
                };
                (geom, &g.stops, g.spread)
            }
            Paint::RadialGradient(g) => {
                let geom = if g.stops.is_empty() {
                    Geometry::Constant(Rgba::new(0, 0, 0, 0))
                } else if g.stops.len() == 1 || g.radius <= 1e-12 {
                    Geometry::Constant(g.stops[0].color)
                } else {
                    let r = g.radius;
                    let (fx, fy) =
                        clamp_focal_inside_circle(g.focal.unwrap_or(g.center), g.center, r);
                    Geometry::Radial {
                        inv: m,
                        fx: fx as f64,
                        fy: fy as f64,
                        cdx: (g.center.x - fx) as f64,
                        cdy: (g.center.y - fy) as f64,
                        r: r as f64,
                    }
                };
                (geom, &g.stops, g.spread)
            }
            _ => return None,
        };
        let ramp = (!stops.is_empty()).then(|| Ramp::new(stops, space));
        Some(GradientRows {
            geom,
            ramp,
            spread,
            dither,
        })
    }

    /// Fill `out` with the paint colour of device pixels
    /// `x0 .. x0 + n` on row `y`.
    pub(crate) fn row(&self, y: u32, x0: u32, n: usize, out: &mut Vec<Rgba>) {
        out.clear();
        let ramp = match (&self.geom, &self.ramp) {
            (Geometry::Constant(_), _) | (_, None) => {
                let c = match &self.geom {
                    Geometry::Constant(c) => *c,
                    _ => Rgba::new(0, 0, 0, 0),
                };
                out.resize(n, c);
                return;
            }
            (_, Some(r)) => r,
        };
        let table = encode_table();
        let yf = y as f64;
        match self.geom {
            Geometry::Linear { a, b, c } => {
                let row = b * yf + c;
                for k in 0..n {
                    let x = x0 + k as u32;
                    let t = (row + a * x as f64) as f32;
                    let v = ramp.eval(apply_spread(t, self.spread));
                    out.push(ramp.quantise(v, threshold(self.dither, x, y), table));
                }
            }
            Geometry::Radial {
                inv: m,
                fx,
                fy,
                cdx,
                cdy,
                r,
            } => {
                let cc = cdx * cdx + cdy * cdy;
                let aa = cc - r * r;
                let inv_aa = 1.0 / aa;
                let inv_r = 1.0 / r;
                let ry = yf + 0.5;
                // Row start and per-pixel step of the focal-relative
                // user-space position.
                let dx0 = m[0] * (x0 as f64 + 0.5) + m[2] * ry + m[4] - fx;
                let dy0 = m[1] * (x0 as f64 + 0.5) + m[3] * ry + m[5] - fy;
                for k in 0..n {
                    let x = x0 + k as u32;
                    let dx = dx0 + m[0] * k as f64;
                    let dy = dy0 + m[1] * k as f64;
                    let cd = cdx * dx + cdy * dy;
                    let dd = dx * dx + dy * dy;
                    // Same root selection as the per-pixel evaluator.
                    let t = if aa.abs() < 1e-12 {
                        if cd.abs() < 1e-12 {
                            dd.sqrt() * inv_r
                        } else {
                            dd / (2.0 * cd)
                        }
                    } else {
                        let disc = cd * cd - aa * dd;
                        if disc < 0.0 {
                            dd.sqrt() * inv_r
                        } else {
                            (cd - disc.sqrt()) * inv_aa
                        }
                    };
                    let v = ramp.eval(apply_spread(t as f32, self.spread));
                    out.push(ramp.quantise(v, threshold(self.dither, x, y), table));
                }
            }
            Geometry::Constant(_) => unreachable!("handled above"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer8_is_a_permutation_of_0_to_63() {
        let mut seen = [false; 64];
        for row in BAYER8 {
            for v in row {
                assert!(!seen[v as usize]);
                seen[v as usize] = true;
            }
        }
        // The 2×2 base pattern survives in the top-left corner (scaled).
        assert_eq!(BAYER8[0][0], 0);
        assert_eq!(BAYER8[0][1], 32);
        assert_eq!(BAYER8[1][0], 48);
        assert_eq!(BAYER8[1][1], 16);
    }

    #[test]
    fn encode_table_matches_the_transfer_function() {
        let t = encode_table();
        for i in 0..=1000 {
            let c = i as f32 / 1000.0;
            let exact = if c <= 0.003_130_8 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            } * 255.0;
            assert!((encode_linear(t, c) - exact).abs() < 0.01, "c={c}");
        }
    }
}
