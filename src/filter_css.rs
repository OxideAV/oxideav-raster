//! CSS `filter` property value parser — Filter Effects Module Level 1
//! §5 (`<filter-value-list>`) + §6.1 (`<filter-function>` grammar).
//!
//! Parses the textual value of the `filter` property into
//! [`FilterFunction`]s ready for
//! [`apply_filter_functions`](crate::apply_filter_functions):
//!
//! ```text
//! <filter-value-list> = [ <filter-function> | <url> ]+
//! <filter-function>   = <blur()> | <brightness()> | <contrast()> |
//!                       <drop-shadow()> | <grayscale()> | <hue-rotate()> |
//!                       <invert()> | <opacity()> | <sepia()> | <saturate()>
//! ```
//!
//! The parser is hardened against hostile input: it never panics, its
//! work is linear in the input length, every numeric token is checked
//! finite (the CSS number grammar cannot produce `NaN` / `inf`
//! spellings, and an overflow like `1e999` is rejected rather than
//! saturated), and the §6.1 "Negative values are not allowed" rules
//! are enforced per function.
//!
//! Grammar-level decisions, each anchored in the staged spec text:
//!
//! * `none` → an empty function list (§5: "No filter effect gets
//!   applied").
//! * `<url>` items are rejected with [`FilterParseError::UrlReference`]:
//!   resolving a `url(…)` filter reference requires a document context
//!   a buffer-level library does not have. (Per §5 an unresolvable
//!   reference disables the whole chain anyway.)
//! * Omitted arguments take the §6.1 defaults — `1` for the
//!   `<number-percentage>` functions except the note's convenience
//!   defaults (`grayscale()` / `sepia()` / `invert()` are also `1`),
//!   `0px` for `blur()`, `0deg` for `hue-rotate()`.
//! * `<length>` values require a `px` unit (or a unitless zero, which
//!   the CSS length grammar permits). Font- or viewport-relative CSS
//!   units cannot be resolved against a pixel buffer and are rejected
//!   with [`FilterParseError::UnsupportedUnit`].
//! * `<angle>` accepts `deg` / `grad` / `rad` / `turn` plus the §6.1
//!   `<zero>` production. Negative angles are legal (hue rotation is
//!   directional and never normalised).
//! * `drop-shadow()` follows `<color>? && <length>{2,3}` — the colour
//!   may precede or follow the lengths; the optional third length (the
//!   standard deviation) must be non-negative; a missing colour falls
//!   back to the caller-supplied `currentColor` (§6.1: "the missing
//!   used color is taken from the color property").

use crate::filter_function::FilterFunction;

/// Errors produced by [`parse_filter_value_list`]. Every variant
/// carries the byte offset (into the original input) where the
/// offending token starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterParseError {
    /// The input was empty (or all whitespace). The `filter` property
    /// grammar requires `none` or at least one item.
    Empty,
    /// A function name that is not one of the ten §6.1 functions.
    UnknownFunction {
        /// Byte offset of the name.
        offset: usize,
        /// The unrecognised name (truncated to 32 bytes).
        name: String,
    },
    /// A `url(…)` filter reference, which this buffer-level parser
    /// cannot resolve.
    UrlReference {
        /// Byte offset of the item.
        offset: usize,
    },
    /// A function's opening parenthesis was never closed.
    UnbalancedParenthesis {
        /// Byte offset of the opening parenthesis.
        offset: usize,
    },
    /// An argument failed its grammar (malformed number, wrong token
    /// count, non-finite value, …).
    InvalidArgument {
        /// Byte offset of the argument list.
        offset: usize,
        /// Human-readable description of the failure.
        detail: String,
    },
    /// A negative value where §6.1 says "Negative values are not
    /// allowed".
    NegativeValue {
        /// Byte offset of the argument list.
        offset: usize,
    },
    /// A dimension with a unit this pixel-buffer parser cannot resolve
    /// (anything but `px` for lengths; anything but `deg` / `grad` /
    /// `rad` / `turn` for angles).
    UnsupportedUnit {
        /// Byte offset of the argument list.
        offset: usize,
        /// The rejected unit (truncated to 32 bytes).
        unit: String,
    },
    /// Non-whitespace bytes after a syntactically complete list that
    /// could not start a new item.
    UnexpectedToken {
        /// Byte offset of the token.
        offset: usize,
    },
}

impl std::fmt::Display for FilterParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterParseError::Empty => write!(f, "empty filter value"),
            FilterParseError::UnknownFunction { offset, name } => {
                write!(f, "unknown filter function `{name}` at byte {offset}")
            }
            FilterParseError::UrlReference { offset } => {
                write!(
                    f,
                    "url() filter reference at byte {offset} cannot be resolved by a buffer-level parser"
                )
            }
            FilterParseError::UnbalancedParenthesis { offset } => {
                write!(f, "unbalanced parenthesis opened at byte {offset}")
            }
            FilterParseError::InvalidArgument { offset, detail } => {
                write!(f, "invalid argument at byte {offset}: {detail}")
            }
            FilterParseError::NegativeValue { offset } => {
                write!(f, "negative value at byte {offset} is not allowed (§6.1)")
            }
            FilterParseError::UnsupportedUnit { offset, unit } => {
                write!(f, "unsupported unit `{unit}` at byte {offset}")
            }
            FilterParseError::UnexpectedToken { offset } => {
                write!(f, "unexpected token at byte {offset}")
            }
        }
    }
}

impl std::error::Error for FilterParseError {}

/// Parse a `filter` property value into filter functions, defaulting a
/// colour-less `drop-shadow()` to opaque black (the initial value of
/// the CSS `color` property). Callers that track a live `color` value
/// should use [`parse_filter_value_list_with_color`].
pub fn parse_filter_value_list(input: &str) -> Result<Vec<FilterFunction>, FilterParseError> {
    parse_filter_value_list_with_color(input, (0, 0, 0, 1.0))
}

/// Parse a `filter` property value into filter functions, substituting
/// `current_color` (RGB + alpha) for a `drop-shadow()` whose colour is
/// omitted (§6.1: "the missing used color is taken from the color
/// property").
pub fn parse_filter_value_list_with_color(
    input: &str,
    current_color: (u8, u8, u8, f32),
) -> Result<Vec<FilterFunction>, FilterParseError> {
    let bytes = input.as_bytes();
    let mut pos = skip_ws(bytes, 0);
    if pos >= bytes.len() {
        return Err(FilterParseError::Empty);
    }

    // `none` — exactly the keyword, alone.
    if let Some(end) = match_keyword(bytes, pos, "none") {
        let rest = skip_ws(bytes, end);
        if rest >= bytes.len() {
            return Ok(Vec::new());
        }
        return Err(FilterParseError::UnexpectedToken { offset: rest });
    }

    let mut out = Vec::new();
    while pos < bytes.len() {
        let name_start = pos;
        let name_end = scan_ident(bytes, pos);
        if name_end == pos {
            return Err(FilterParseError::UnexpectedToken { offset: pos });
        }
        let name = input[name_start..name_end].to_ascii_lowercase();
        let paren = skip_ws(bytes, name_end);
        if paren >= bytes.len() || bytes[paren] != b'(' {
            return Err(FilterParseError::UnexpectedToken {
                offset: paren.min(bytes.len()),
            });
        }
        let close = find_matching_paren(bytes, paren)
            .ok_or(FilterParseError::UnbalancedParenthesis { offset: paren })?;
        let args_start = paren + 1;
        let args = input[args_start..close].trim();
        let arg_off = args_start;

        match name.as_str() {
            "url" => return Err(FilterParseError::UrlReference { offset: name_start }),
            "blur" => {
                let std_dev = if args.is_empty() {
                    0.0 // §6.1: "Default value when omitted is 0px."
                } else {
                    parse_length(args, arg_off, false)?
                };
                out.push(FilterFunction::Blur(std_dev));
            }
            "brightness" | "contrast" | "saturate" | "grayscale" | "sepia" | "invert"
            | "opacity" => {
                // §6.1 omitted-value defaults: 1 for all seven (the
                // note's convenience defaults make grayscale / sepia /
                // invert 1 as well — "apply the effect to 100%").
                let amount = if args.is_empty() {
                    1.0
                } else {
                    parse_number_percentage(args, arg_off)?
                };
                out.push(match name.as_str() {
                    "brightness" => FilterFunction::Brightness(amount),
                    "contrast" => FilterFunction::Contrast(amount),
                    "saturate" => FilterFunction::Saturate(amount),
                    "grayscale" => FilterFunction::Grayscale(amount),
                    "sepia" => FilterFunction::Sepia(amount),
                    "invert" => FilterFunction::Invert(amount),
                    _ => FilterFunction::Opacity(amount),
                });
            }
            "hue-rotate" => {
                let degrees = if args.is_empty() {
                    0.0 // §6.1: "Default value when omitted is 0deg."
                } else {
                    parse_angle(args, arg_off)?
                };
                out.push(FilterFunction::HueRotate(degrees));
            }
            "drop-shadow" => {
                out.push(parse_drop_shadow(args, arg_off, current_color)?);
            }
            _ => {
                let mut trunc = name;
                trunc.truncate(32);
                return Err(FilterParseError::UnknownFunction {
                    offset: name_start,
                    name: trunc,
                });
            }
        }
        pos = skip_ws(bytes, close + 1);
    }
    Ok(out)
}

/// Advance past ASCII whitespace.
fn skip_ws(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }
    pos
}

/// If `keyword` (ASCII case-insensitive) starts at `pos` and is not
/// followed by an ident character, return the end offset.
fn match_keyword(bytes: &[u8], pos: usize, keyword: &str) -> Option<usize> {
    let end = pos + keyword.len();
    if end > bytes.len() {
        return None;
    }
    if !bytes[pos..end].eq_ignore_ascii_case(keyword.as_bytes()) {
        return None;
    }
    if end < bytes.len() && is_ident_byte(bytes[end]) {
        return None;
    }
    Some(end)
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_'
}

/// Scan a CSS identifier (letters / digits / `-` / `_`, must start
/// with a letter or `-`).
fn scan_ident(bytes: &[u8], pos: usize) -> usize {
    let mut end = pos;
    while end < bytes.len() && is_ident_byte(bytes[end]) {
        end += 1;
    }
    if end > pos && bytes[pos].is_ascii_digit() {
        return pos; // idents cannot start with a digit
    }
    end
}

/// Find the `)` matching the `(` at `open`, honouring nesting (for
/// `rgb(…)` inside `drop-shadow(…)`).
fn find_matching_paren(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse a finite `f32` with the CSS number grammar (optional sign,
/// digits, optional fraction, optional `e`-exponent). Rust's
/// `f32::from_str` also accepts `NaN` / `inf` / `infinity` spellings,
/// which the CSS grammar does not — the leading-character check below
/// excludes them before the conversion, and the `is_finite` check
/// rejects overflow like `1e999`.
fn parse_css_number(token: &str, offset: usize) -> Result<f32, FilterParseError> {
    let t = token.trim();
    let ok_start = t
        .bytes()
        .next()
        .is_some_and(|b| b.is_ascii_digit() || b == b'+' || b == b'-' || b == b'.');
    if !ok_start {
        return Err(FilterParseError::InvalidArgument {
            offset,
            detail: format!("expected a number, got `{}`", truncate(t)),
        });
    }
    match t.parse::<f32>() {
        Ok(v) if v.is_finite() => Ok(v),
        Ok(_) => Err(FilterParseError::InvalidArgument {
            offset,
            detail: format!("number `{}` overflows the finite range", truncate(t)),
        }),
        Err(_) => Err(FilterParseError::InvalidArgument {
            offset,
            detail: format!("malformed number `{}`", truncate(t)),
        }),
    }
}

fn truncate(s: &str) -> String {
    let mut out: String = s.chars().take(24).collect();
    if out.len() < s.len() {
        out.push('…');
    }
    out
}

/// `<number-percentage>` (§6.1): a plain number, or a percentage
/// converted "to real numbers" (§13.1). Negative values are not
/// allowed.
fn parse_number_percentage(token: &str, offset: usize) -> Result<f32, FilterParseError> {
    let t = token.trim();
    // Dividing (rather than multiplying by 0.01) keeps `80%` exactly
    // equal to the `0.8` literal after f32 rounding.
    let v = match t.strip_suffix('%') {
        Some(body) => parse_css_number(body, offset)? / 100.0,
        None => parse_css_number(t, offset)?,
    };
    if v < 0.0 {
        return Err(FilterParseError::NegativeValue { offset });
    }
    Ok(v)
}

/// Split a dimension token into (number, unit) by scanning the longest
/// CSS-number prefix (optional sign, digits/dot, optional
/// `e[+-]?digits` exponent); the remainder is the unit. Scanning the
/// exponent explicitly keeps `1e400deg` classified as (number
/// `1e400`, unit `deg`) — an overflow, not an unknown unit.
fn split_unit(token: &str) -> (&str, &str) {
    let t = token.trim();
    let b = t.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let digits_start = i;
    while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
        i += 1;
    }
    if i > digits_start && i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        if j < b.len() && b[j].is_ascii_digit() {
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            i = j;
        }
    }
    (&t[..i], &t[i..])
}

/// `<length>` in pixel units: `<number>px`, or unitless zero.
/// Percentages are not accepted (§6.1 for `blur()`: the parameter "does
/// not accept percentage values"). `allow_negative` is true for
/// drop-shadow offsets, false for blur radii / standard deviations.
fn parse_length(token: &str, offset: usize, allow_negative: bool) -> Result<f32, FilterParseError> {
    let (num, unit) = split_unit(token);
    let v = parse_css_number(num, offset)?;
    match unit.to_ascii_lowercase().as_str() {
        "px" => {}
        "" => {
            // The CSS length grammar only permits a unitless length
            // when it is zero.
            if v != 0.0 {
                return Err(FilterParseError::InvalidArgument {
                    offset,
                    detail: format!("length `{}` is missing its unit", truncate(token.trim())),
                });
            }
        }
        other => {
            return Err(FilterParseError::UnsupportedUnit {
                offset,
                unit: truncate(other),
            });
        }
    }
    if !allow_negative && v < 0.0 {
        return Err(FilterParseError::NegativeValue { offset });
    }
    Ok(v)
}

/// `[<angle> | <zero>]` (§6.1) in degrees. "The unit identifier may be
/// omitted if the `<angle>` is zero."
fn parse_angle(token: &str, offset: usize) -> Result<f32, FilterParseError> {
    let (num, unit) = split_unit(token);
    let v = parse_css_number(num, offset)?;
    let degrees = match unit.to_ascii_lowercase().as_str() {
        "deg" => v,
        "grad" => v * 0.9,
        "rad" => v.to_degrees(),
        "turn" => v * 360.0,
        "" => {
            if v != 0.0 {
                return Err(FilterParseError::InvalidArgument {
                    offset,
                    detail: format!("angle `{}` is missing its unit", truncate(token.trim())),
                });
            }
            0.0
        }
        other => {
            return Err(FilterParseError::UnsupportedUnit {
                offset,
                unit: truncate(other),
            });
        }
    };
    if !degrees.is_finite() {
        return Err(FilterParseError::InvalidArgument {
            offset,
            detail: "angle overflows the finite range".into(),
        });
    }
    Ok(degrees)
}

/// Split a `drop-shadow()` argument string into top-level
/// whitespace-separated tokens (whitespace inside nested parentheses,
/// e.g. `rgb(1, 2, 3)`, does not split).
fn split_top_level(args: &str) -> Vec<(usize, &str)> {
    let bytes = args.as_bytes();
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => {
                depth += 1;
                if start.is_none() {
                    start = Some(i);
                }
            }
            b')' => depth = depth.saturating_sub(1),
            _ if b.is_ascii_whitespace() && depth == 0 => {
                if let Some(s) = start.take() {
                    out.push((s, &args[s..i]));
                }
            }
            _ => {
                if start.is_none() {
                    start = Some(i);
                }
            }
        }
    }
    if let Some(s) = start {
        out.push((s, &args[s..]));
    }
    out
}

/// `drop-shadow( <color>? && <length>{2,3} )` (§6.1). The `&&`
/// combinator means the colour may appear before or after the length
/// run, but not between lengths.
fn parse_drop_shadow(
    args: &str,
    offset: usize,
    current_color: (u8, u8, u8, f32),
) -> Result<FilterFunction, FilterParseError> {
    let tokens = split_top_level(args);
    if tokens.is_empty() {
        return Err(FilterParseError::InvalidArgument {
            offset,
            detail: "drop-shadow() requires at least two lengths".into(),
        });
    }

    let mut color: Option<(u8, u8, u8, f32)> = None;
    let mut lengths: Vec<(usize, &str)> = Vec::new();
    let mut seen_length = false;
    let mut length_run_ended = false;

    for (tok_off, tok) in tokens {
        let abs = offset + tok_off;
        let starts_numeric = tok
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_digit() || b == b'+' || b == b'-' || b == b'.');
        if starts_numeric {
            if length_run_ended {
                // colour appeared *between* lengths — `&&` forbids it.
                return Err(FilterParseError::InvalidArgument {
                    offset: abs,
                    detail: "lengths must be consecutive in drop-shadow()".into(),
                });
            }
            lengths.push((abs, tok));
            seen_length = true;
        } else {
            if seen_length {
                length_run_ended = true;
            }
            if color.is_some() {
                return Err(FilterParseError::InvalidArgument {
                    offset: abs,
                    detail: "drop-shadow() accepts a single color".into(),
                });
            }
            color = Some(parse_color(tok, abs)?);
        }
    }

    if lengths.len() < 2 || lengths.len() > 3 {
        return Err(FilterParseError::InvalidArgument {
            offset,
            detail: format!("drop-shadow() takes 2 or 3 lengths, got {}", lengths.len()),
        });
    }
    let dx = parse_length(lengths[0].1, lengths[0].0, true)?;
    let dy = parse_length(lengths[1].1, lengths[1].0, true)?;
    // §6.1: the optional third length is "the standard deviation";
    // box-shadow's blur value must be non-negative.
    let std_dev = if lengths.len() == 3 {
        parse_length(lengths[2].1, lengths[2].0, false)?
    } else {
        0.0
    };
    let (r, g, b, a) = color.unwrap_or(current_color);
    Ok(FilterFunction::DropShadow {
        dx,
        dy,
        std_dev,
        color: (r, g, b),
        opacity: a,
    })
}

/// Parse a CSS `<color>`: `#rgb` / `#rgba` / `#rrggbb` / `#rrggbbaa`
/// hex forms, `rgb()` / `rgba()` (legacy comma or modern space syntax,
/// `<number>` or `<percentage>` channels, optional alpha), the
/// `transparent` keyword, and the 16 basic CSS color keywords.
fn parse_color(token: &str, offset: usize) -> Result<(u8, u8, u8, f32), FilterParseError> {
    let t = token.trim();
    if let Some(hex) = t.strip_prefix('#') {
        return parse_hex_color(hex, offset);
    }
    let lower = t.to_ascii_lowercase();
    if let Some(rest) = lower
        .strip_prefix("rgba(")
        .or_else(|| lower.strip_prefix("rgb("))
    {
        let body = rest
            .strip_suffix(')')
            .ok_or(FilterParseError::UnbalancedParenthesis { offset })?;
        return parse_rgb_body(body, offset);
    }
    // Keyword colours: `transparent` + the 16 basic CSS keywords.
    let named: Option<(u8, u8, u8, f32)> = match lower.as_str() {
        "transparent" => Some((0, 0, 0, 0.0)),
        "black" => Some((0, 0, 0, 1.0)),
        "silver" => Some((192, 192, 192, 1.0)),
        "gray" | "grey" => Some((128, 128, 128, 1.0)),
        "white" => Some((255, 255, 255, 1.0)),
        "maroon" => Some((128, 0, 0, 1.0)),
        "red" => Some((255, 0, 0, 1.0)),
        "purple" => Some((128, 0, 128, 1.0)),
        "fuchsia" | "magenta" => Some((255, 0, 255, 1.0)),
        "green" => Some((0, 128, 0, 1.0)),
        "lime" => Some((0, 255, 0, 1.0)),
        "olive" => Some((128, 128, 0, 1.0)),
        "yellow" => Some((255, 255, 0, 1.0)),
        "navy" => Some((0, 0, 128, 1.0)),
        "blue" => Some((0, 0, 255, 1.0)),
        "teal" => Some((0, 128, 128, 1.0)),
        "aqua" | "cyan" => Some((0, 255, 255, 1.0)),
        "currentcolor" => None, // handled by the caller's fallback
        _ => None,
    };
    named.ok_or_else(|| FilterParseError::InvalidArgument {
        offset,
        detail: format!("unrecognised color `{}`", truncate(t)),
    })
}

fn parse_hex_color(hex: &str, offset: usize) -> Result<(u8, u8, u8, f32), FilterParseError> {
    let err = || FilterParseError::InvalidArgument {
        offset,
        detail: format!("malformed hex color `#{}`", truncate(hex)),
    };
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(err());
    }
    let nib = |i: usize| u8::from_str_radix(&hex[i..i + 1], 16).unwrap_or(0);
    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap_or(0);
    match hex.len() {
        3 => Ok((nib(0) * 17, nib(1) * 17, nib(2) * 17, 1.0)),
        4 => Ok((
            nib(0) * 17,
            nib(1) * 17,
            nib(2) * 17,
            (nib(3) * 17) as f32 / 255.0,
        )),
        6 => Ok((byte(0), byte(2), byte(4), 1.0)),
        8 => Ok((byte(0), byte(2), byte(4), byte(6) as f32 / 255.0)),
        _ => Err(err()),
    }
}

fn parse_rgb_body(body: &str, offset: usize) -> Result<(u8, u8, u8, f32), FilterParseError> {
    // Accept both the legacy comma syntax and the modern
    // space-separated syntax with `/` before alpha.
    let normalised = body.replace([',', '/'], " ");
    let parts: Vec<&str> = normalised.split_ascii_whitespace().collect();
    if parts.len() != 3 && parts.len() != 4 {
        return Err(FilterParseError::InvalidArgument {
            offset,
            detail: format!("rgb() takes 3 or 4 components, got {}", parts.len()),
        });
    }
    let channel = |tok: &str| -> Result<u8, FilterParseError> {
        let v = match tok.strip_suffix('%') {
            Some(b) => parse_css_number(b, offset)? / 100.0 * 255.0,
            None => parse_css_number(tok, offset)?,
        };
        Ok(v.clamp(0.0, 255.0).round() as u8)
    };
    let r = channel(parts[0])?;
    let g = channel(parts[1])?;
    let b = channel(parts[2])?;
    let a = if parts.len() == 4 {
        let tok = parts[3];
        let v = match tok.strip_suffix('%') {
            Some(body) => parse_css_number(body, offset)? / 100.0,
            None => parse_css_number(tok, offset)?,
        };
        v.clamp(0.0, 1.0)
    } else {
        1.0
    };
    Ok((r, g, b, a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_none_to_empty_list() {
        assert_eq!(parse_filter_value_list("none").unwrap(), vec![]);
        assert_eq!(parse_filter_value_list("  NONE  ").unwrap(), vec![]);
    }

    #[test]
    fn parses_the_ten_functions() {
        let list = parse_filter_value_list(
            "blur(2px) brightness(1.5) contrast(200%) drop-shadow(2px 4px 1px red) \
             grayscale(50%) hue-rotate(90deg) invert(0.2) opacity(80%) sepia(1) saturate(300%)",
        )
        .unwrap();
        assert_eq!(list.len(), 10);
        assert_eq!(list[0], FilterFunction::Blur(2.0));
        assert_eq!(list[1], FilterFunction::Brightness(1.5));
        assert_eq!(list[2], FilterFunction::Contrast(2.0));
        assert_eq!(
            list[3],
            FilterFunction::DropShadow {
                dx: 2.0,
                dy: 4.0,
                std_dev: 1.0,
                color: (255, 0, 0),
                opacity: 1.0,
            }
        );
        assert_eq!(list[4], FilterFunction::Grayscale(0.5));
        assert_eq!(list[5], FilterFunction::HueRotate(90.0));
        assert_eq!(list[6], FilterFunction::Invert(0.2));
        assert_eq!(list[7], FilterFunction::Opacity(0.8));
        assert_eq!(list[8], FilterFunction::Sepia(1.0));
        assert_eq!(list[9], FilterFunction::Saturate(3.0));
    }

    #[test]
    fn omitted_arguments_take_spec_defaults() {
        // §6.1 note: grayscale() / sepia() / invert() default to 1.
        assert_eq!(
            parse_filter_value_list("grayscale() sepia() invert()").unwrap(),
            vec![
                FilterFunction::Grayscale(1.0),
                FilterFunction::Sepia(1.0),
                FilterFunction::Invert(1.0),
            ]
        );
        assert_eq!(
            parse_filter_value_list("blur() hue-rotate() brightness()").unwrap(),
            vec![
                FilterFunction::Blur(0.0),
                FilterFunction::HueRotate(0.0),
                FilterFunction::Brightness(1.0),
            ]
        );
    }

    #[test]
    fn angle_units_convert_to_degrees() {
        let list = parse_filter_value_list("hue-rotate(100grad) hue-rotate(0.5turn) hue-rotate(0)")
            .unwrap();
        assert_eq!(list[0], FilterFunction::HueRotate(90.0));
        assert_eq!(list[1], FilterFunction::HueRotate(180.0));
        assert_eq!(list[2], FilterFunction::HueRotate(0.0));
        // Negative angles are legal and unnormalised.
        let neg = parse_filter_value_list("hue-rotate(-450deg)").unwrap();
        assert_eq!(neg[0], FilterFunction::HueRotate(-450.0));
    }

    #[test]
    fn drop_shadow_color_position_is_flexible() {
        let before = parse_filter_value_list("drop-shadow(#0000ff 1px 2px)").unwrap();
        let after = parse_filter_value_list("drop-shadow(1px 2px #0000ff)").unwrap();
        assert_eq!(before, after);
        assert_eq!(
            before[0],
            FilterFunction::DropShadow {
                dx: 1.0,
                dy: 2.0,
                std_dev: 0.0,
                color: (0, 0, 255),
                opacity: 1.0,
            }
        );
        // Negative offsets are fine; rgb() nesting survives the split.
        let neg = parse_filter_value_list("drop-shadow(rgb(10, 20, 30) -3px -4px 2px)").unwrap();
        assert_eq!(
            neg[0],
            FilterFunction::DropShadow {
                dx: -3.0,
                dy: -4.0,
                std_dev: 2.0,
                color: (10, 20, 30),
                opacity: 1.0,
            }
        );
    }

    #[test]
    fn drop_shadow_missing_color_uses_current_color() {
        let list =
            parse_filter_value_list_with_color("drop-shadow(1px 2px)", (7, 8, 9, 0.5)).unwrap();
        assert_eq!(
            list[0],
            FilterFunction::DropShadow {
                dx: 1.0,
                dy: 2.0,
                std_dev: 0.0,
                color: (7, 8, 9),
                opacity: 0.5,
            }
        );
    }

    #[test]
    fn negative_amounts_are_rejected() {
        for s in [
            "grayscale(-0.1)",
            "sepia(-5%)",
            "brightness(-1)",
            "contrast(-200%)",
            "saturate(-0.5)",
            "invert(-1)",
            "opacity(-0.01)",
            "blur(-1px)",
            "drop-shadow(1px 2px -3px)", // negative std dev
        ] {
            assert!(
                matches!(
                    parse_filter_value_list(s),
                    Err(FilterParseError::NegativeValue { .. })
                ),
                "{s} must be rejected as negative"
            );
        }
    }

    #[test]
    fn non_finite_and_malformed_numbers_are_rejected() {
        for s in [
            "brightness(1e999)",
            "blur(NaNpx)",
            "blur(infpx)",
            "grayscale(nan)",
            "opacity(--1)",
            "contrast(1..5)",
            "hue-rotate(1e400deg)",
        ] {
            let r = parse_filter_value_list(s);
            assert!(
                matches!(r, Err(FilterParseError::InvalidArgument { .. })),
                "{s} must be an InvalidArgument, got {r:?}"
            );
        }
    }

    #[test]
    fn unresolvable_units_are_rejected() {
        assert!(matches!(
            parse_filter_value_list("blur(2em)"),
            Err(FilterParseError::UnsupportedUnit { .. })
        ));
        assert!(matches!(
            parse_filter_value_list("blur(50%)"),
            Err(FilterParseError::UnsupportedUnit { .. })
        ));
        assert!(matches!(
            parse_filter_value_list("hue-rotate(1rem)"),
            Err(FilterParseError::UnsupportedUnit { .. })
        ));
        // Unitless non-zero lengths violate the CSS length grammar.
        assert!(matches!(
            parse_filter_value_list("blur(5)"),
            Err(FilterParseError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn url_references_are_reported_as_unsupported() {
        assert!(matches!(
            parse_filter_value_list("url(#f)"),
            Err(FilterParseError::UrlReference { .. })
        ));
        assert!(matches!(
            parse_filter_value_list("blur(1px) url(commonfilters.svg#filter)"),
            Err(FilterParseError::UrlReference { .. })
        ));
    }

    #[test]
    fn structural_errors_are_reported() {
        assert!(matches!(
            parse_filter_value_list(""),
            Err(FilterParseError::Empty)
        ));
        assert!(matches!(
            parse_filter_value_list("   "),
            Err(FilterParseError::Empty)
        ));
        assert!(matches!(
            parse_filter_value_list("blur(5px"),
            Err(FilterParseError::UnbalancedParenthesis { .. })
        ));
        assert!(matches!(
            parse_filter_value_list("shine(1)"),
            Err(FilterParseError::UnknownFunction { .. })
        ));
        assert!(matches!(
            parse_filter_value_list("none blur(1px)"),
            Err(FilterParseError::UnexpectedToken { .. })
        ));
        assert!(matches!(
            parse_filter_value_list("blur(1px) )"),
            Err(FilterParseError::UnexpectedToken { .. })
        ));
    }

    #[test]
    fn hostile_inputs_never_panic() {
        // Deterministic LCG byte soup — the parser must return without
        // panicking on arbitrary input.
        let mut state = 0x2545F491_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for _ in 0..20_000 {
            let len = (next() % 64) as usize;
            let s: String = (0..len)
                .map(|_| {
                    let c = next() % 96 + 32; // printable ASCII
                    char::from_u32(c).unwrap_or(' ')
                })
                .collect();
            let _ = parse_filter_value_list(&s);
        }
        // Structured hostile shapes.
        let deep = format!("blur{}5px{}", "(".repeat(4096), ")".repeat(4096));
        let _ = parse_filter_value_list(&deep);
        let long = "blur(1px) ".repeat(10_000);
        assert_eq!(parse_filter_value_list(&long).unwrap().len(), 10_000);
        let _ = parse_filter_value_list(&"#".repeat(65_536));
        let _ = parse_filter_value_list("drop-shadow(rgb(999999999999, -1, 3.5e38) 0 0)");
        let _ = parse_filter_value_list("ｇrayscale(1)"); // full-width g
    }
}
