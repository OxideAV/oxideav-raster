#![no_main]

//! CSS `filter` property grammar fuzz target (Filter Effects 1 §5 /
//! §6.1) — parse arbitrary bytes, then evaluate whatever parses.
//!
//! Contract:
//!
//! 1. **No-panic parse.** `parse_filter_value_list` must return a
//!    `Result` on any input — malformed numbers, unterminated
//!    parentheses, unit soup, exponent overflows, multi-byte UTF-8
//!    confusables, kilobyte-deep paren nesting — never panic, abort,
//!    or hang (the scan is linear in the input length).
//! 2. **No-panic apply.** When the input parses, the resulting
//!    `<filter-value-list>` is applied to an 8×8 canvas through the
//!    §13.1 shorthand expansion. Every §6.1-legal numeric extreme the
//!    parser admits (huge-but-finite blur standard deviations,
//!    million-degree hue rotations, 3e38 brightness slopes,
//!    drop-shadow offsets far outside the canvas) must come back as a
//!    correctly-sized buffer. The box-blur prefix-sum hardening makes
//!    the blur cost O(1) in the derived window width, so no length cap
//!    is applied here — a hang in this target is a real finding.
//!
//! The `currentColor`-substituting variant is driven with a fixed
//! non-black colour so both entry points stay covered.

use libfuzzer_sys::fuzz_target;
use oxideav_raster::{
    apply_filter_functions, parse_filter_value_list, parse_filter_value_list_with_color,
};

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    let parsed = parse_filter_value_list(&text);
    let _ = parse_filter_value_list_with_color(&text, (10, 200, 30, 0.5));

    if let Ok(functions) = parsed {
        // Keep the per-iteration cost bounded by list length, not by
        // repeated giant lists: 64 functions exercise every chaining
        // path there is.
        if functions.len() <= 64 {
            let src = vec![127u8; 8 * 8 * 4];
            let out = apply_filter_functions(&functions, &src, 8, 8);
            assert_eq!(out.len(), src.len());
        }
    }
});
