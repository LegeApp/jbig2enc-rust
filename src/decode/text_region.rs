//! Arithmetic text-region decoding (jbig2decplan.md §17, T.88 §6.4/§7.4.3).
//!
//! Inverts the encoder's `encode_text_region_mapped`
//! ([`crate::encode::document::symbol::text_region`]): the encoder emits, per
//! strip, `IADT` (strip T delta), then for each instance `IAFS`/`IADS` (S
//! coordinate), optional `IAIT` (T offset when `SBSTRIPS > 1`), and `IAID`
//! (symbol id), ending each strip with an `OOB` on `IADS`. Symbols are combined
//! into the region bitmap as they are decoded (no placement list), then the
//! region is composed onto the page by the caller.
//!
//! The encoder emits `REFCORNER = TOPLEFT` and `TRANSPOSED = 0`; this decoder
//! supports all four (non-transposed) reference corners and rejects transposed,
//! Huffman, and refined text regions with typed `Unsupported` errors.

use std::sync::Arc;

use crate::decode::arith::ArithmeticDecoder;
use crate::decode::error::{DecodeError, LimitError, UnsupportedFeature};
use crate::decode::iaid::IaidContexts;
use crate::decode::integer::{DecodedInteger, IntegerContexts};
use crate::decode::refinement::decode_refinement_region;
use crate::shared::bitmap::{CombinationOperator, MonoBitmap};
use crate::shared::int_proc::IntProc;
use crate::shared::limits::DecodeLimits;
use crate::shared::mq_table::MqContext;
use crate::shared::reader::Reader;

/// A decoded text region plus where it composes onto the page.
pub struct TextRegionResult {
    pub bitmap: MonoBitmap,
    pub x: u32,
    pub y: u32,
    /// External (region) combination operator (region-info flags low 3 bits).
    pub comb_operator: u8,
}

/// `ceil(log2(v))`, matching the encoder's `log2up` (no floor of 1). Returns 0
/// for `v <= 1` — a single-symbol dictionary reads zero symbol-id bits.
#[inline]
fn log2_ceil(v: u32) -> u32 {
    if v <= 1 {
        return 0;
    }
    let pow2 = v.is_power_of_two();
    let floor = 31 - v.leading_zeros();
    if pow2 { floor } else { floor + 1 }
}

#[inline]
fn comb_op(code: u8) -> CombinationOperator {
    match code {
        0 => CombinationOperator::Or,
        1 => CombinationOperator::And,
        2 => CombinationOperator::Xor,
        3 => CombinationOperator::Xnor,
        _ => CombinationOperator::Replace,
    }
}

#[inline]
fn clamp_i32(v: i64) -> i32 {
    if v > i32::MAX as i64 {
        i32::MAX
    } else if v < i32::MIN as i64 {
        i32::MIN
    } else {
        v as i32
    }
}

/// Decode an arithmetic text-region segment payload (T.88 §7.4.3).
///
/// `symbols` is the combined exported-symbol list of every referred dictionary,
/// in reference order. `int_ctx` and `iaid_ctx` are reused, worker-local scratch
/// reset here (a fresh text region starts every context at zero).
#[allow(clippy::too_many_arguments)]
pub fn decode_text_region(
    payload: &[u8],
    symbols: &[Arc<MonoBitmap>],
    limits: &DecodeLimits,
    int_ctx: &mut IntegerContexts,
    iaid_ctx: &mut IaidContexts,
    refine_ctx: &mut [MqContext],
) -> Result<TextRegionResult, DecodeError> {
    let mut r = Reader::new(payload);

    // §7.4.1 region segment information.
    let width = r.read_u32_be()?;
    let height = r.read_u32_be()?;
    let x = r.read_u32_be()?;
    let y = r.read_u32_be()?;
    let region_flags = r.read_u8()?;
    let ext_comb = region_flags & 0x07;

    // §7.4.3.1.1 text region segment flags (16-bit, big-endian).
    let flags = r.read_u16_be()?;
    let sbhuff = flags & 0x0001 != 0;
    let sbrefine = flags & 0x0002 != 0;
    let log_strips = ((flags >> 2) & 0x0003) as u8;
    let ref_corner = ((flags >> 4) & 0x0003) as u8;
    let transposed = (flags >> 6) & 0x0001 != 0;
    let sb_comb_op = ((flags >> 7) & 0x0003) as u8;
    let sb_def_pixel = (flags >> 9) & 0x0001 != 0;
    let ds_offset_raw = ((flags >> 10) & 0x001F) as u32;
    let sb_rtemplate = ((flags >> 15) & 0x0001) as u8;

    if sbhuff {
        return Err(DecodeError::Unsupported(UnsupportedFeature::HuffmanCoding));
    }
    if transposed {
        return Err(DecodeError::Unsupported(
            UnsupportedFeature::TransposedTextRegion,
        ));
    }

    // §7.4.3.1.3 SBRAT: refinement adaptive-template pixels, present only when
    // SBREFINE=1 and SBRTEMPLATE=0 (GRTEMPLATE-0 uses two AT pairs = 4 bytes).
    // Only the first (target) pair GRAT1 is used by the template-0 context.
    let mut grat: (i8, i8) = (-1, -1);
    if sbrefine && sb_rtemplate == 0 {
        let g1x = r.read_i8()?;
        let g1y = r.read_i8()?;
        let _g2x = r.read_i8()?;
        let _g2y = r.read_i8()?;
        grat = (g1x, g1y);
    }
    if sbrefine && sb_rtemplate != 0 {
        // GRTEMPLATE-1 refinement is Phase 3.
        return Err(DecodeError::Unsupported(UnsupportedFeature::RefinementRegion));
    }

    // §7.4.3.1.7 SBNUMINSTANCES.
    let num_instances = r.read_u32_be()?;
    if num_instances as usize > limits.max_symbols {
        return Err(DecodeError::limit(LimitError::Count {
            what: "text instances",
            value: num_instances as u64,
            limit: limits.max_symbols as u64,
        }));
    }

    // Sign-extend the 5-bit SBDSOFFSET.
    let sb_ds_offset: i32 = if ds_offset_raw >= 16 {
        ds_offset_raw as i32 - 32
    } else {
        ds_offset_raw as i32
    };

    let sb_num_syms = symbols.len() as u32;
    let code_len = log2_ceil(sb_num_syms) as u8;
    iaid_ctx.reset_for_bits(code_len)?;
    int_ctx.reset();
    if sbrefine {
        for c in refine_ctx.iter_mut() {
            *c = MqContext(0);
        }
    }

    let sb_strips: i64 = 1i64 << (log_strips & 0x03);
    let region_op = comb_op(ext_comb);
    let symbol_op = comb_op(sb_comb_op);

    let data = &payload[r.position()..];
    let mut bitmap = MonoBitmap::new(width, height, sb_def_pixel, limits)?;
    let mut dec = ArithmeticDecoder::new(data);

    // §6.4.5 1) initial STRIPT.
    let dt0 = decode_value(int_ctx.decode(&mut dec, IntProc::Iadt))?;
    let mut strip_t: i64 = -(dt0 as i64 * sb_strips);
    let mut first_s: i64 = 0;
    let mut n_inst: u32 = 0;

    while n_inst < num_instances {
        // §6.4.5 3b) strip T delta.
        let dt = decode_value(int_ctx.decode(&mut dec, IntProc::Iadt))?;
        strip_t = strip_t
            .checked_add(dt as i64 * sb_strips)
            .ok_or(DecodeError::Overflow { operation: "strip T" })?;

        // §6.4.5 3c) first symbol S coordinate.
        let dfs = decode_value(int_ctx.decode(&mut dec, IntProc::Iafs))?;
        first_s = first_s
            .checked_add(dfs as i64)
            .ok_or(DecodeError::Overflow { operation: "first S" })?;
        let mut cur_s = first_s;
        let mut first_in_strip = true;

        loop {
            if !first_in_strip {
                // §6.4.5 3c) subsequent S: IADS, OOB ends the strip.
                match int_ctx.decode(&mut dec, IntProc::Iads) {
                    DecodedInteger::OutOfBand => break,
                    DecodedInteger::Value(ids) => {
                        cur_s = cur_s
                            .checked_add(ids as i64 + sb_ds_offset as i64)
                            .ok_or(DecodeError::Overflow { operation: "S coordinate" })?;
                    }
                }
            }
            first_in_strip = false;

            if n_inst >= num_instances {
                return Err(DecodeError::Malformed {
                    reason: "more text instances than SBNUMINSTANCES",
                });
            }

            // §6.4.5 3c vi) current T within the strip.
            let cur_t = if sb_strips == 1 {
                0
            } else {
                decode_value(int_ctx.decode(&mut dec, IntProc::Iait))? as i64
            };
            let t_i = strip_t
                .checked_add(cur_t)
                .ok_or(DecodeError::Overflow { operation: "T coordinate" })?;

            // §6.4.5 3c x) symbol id.
            let id = iaid_ctx.decode(&mut dec, code_len);
            let symbol = symbols.get(id as usize).ok_or(DecodeError::Malformed {
                reason: "symbol id out of range",
            })?;
            let hi = symbol.height() as i64;

            // §6.4.11 refinement indicator (only present when SBREFINE=1).
            let ri = if sbrefine {
                decode_value(int_ctx.decode(&mut dec, IntProc::Iari))?
            } else {
                0
            };

            // The glyph actually placed: the reference symbol, or a decoded
            // refinement of it. `placed_height` anchors bottom reference corners.
            let (placed, placed_height): (PlacedSymbol<'_>, i64) = if ri != 0 {
                let rdw = decode_value(int_ctx.decode(&mut dec, IntProc::Iardw))? as i64;
                let rdh = decode_value(int_ctx.decode(&mut dec, IntProc::Iardh))? as i64;
                let rdx = decode_value(int_ctx.decode(&mut dec, IntProc::Iardx))?;
                let rdy = decode_value(int_ctx.decode(&mut dec, IntProc::Iardy))?;
                let grw = symbol.width() as i64 + rdw;
                let grh = symbol.height() as i64 + rdh;
                if grw <= 0
                    || grh <= 0
                    || grw > limits.max_width as i64
                    || grh > limits.max_height as i64
                {
                    return Err(DecodeError::Malformed {
                        reason: "refined glyph dimensions out of range",
                    });
                }
                // §6.3.5.3: GRREFERENCEDX/DY = floor(RDW/2)+RDX, floor(RDH/2)+RDY.
                let grdx = (rdw as i32).div_euclid(2) + rdx;
                let grdy = (rdh as i32).div_euclid(2) + rdy;
                let refined = decode_refinement_region(
                    &mut dec,
                    symbol,
                    grw as u32,
                    grh as u32,
                    grdx,
                    grdy,
                    grat,
                    refine_ctx,
                    limits,
                )?;
                (PlacedSymbol::Owned(refined), grh)
            } else {
                (PlacedSymbol::Borrowed(symbol), hi)
            };

            // Non-transposed placement: top-left corner at (cur_s, t_i); the
            // reference corner only shifts the vertical anchor. CURS advances by
            // the reference WI-1.
            let s_left = cur_s;
            let t_top = match ref_corner {
                // BOTTOMLEFT / BOTTOMRIGHT: anchor is the glyph's bottom row.
                0 | 2 => t_i - (placed_height - 1),
                // TOPLEFT / TOPRIGHT.
                _ => t_i,
            };
            let placed_width = placed.bitmap().width() as i64;
            bitmap.combine(
                placed.bitmap(),
                clamp_i32(s_left),
                clamp_i32(t_top),
                symbol_op,
            );

            // §6.4.5 3c x): advance by the *placed* glyph width (the refined
            // bitmap when RI=1), matching jbig2dec exactly.
            cur_s = cur_s
                .checked_add(placed_width - 1)
                .ok_or(DecodeError::Overflow { operation: "S advance" })?;
            n_inst += 1;
            if n_inst == num_instances {
                return Ok(TextRegionResult {
                    bitmap,
                    x,
                    y,
                    comb_operator: ext_comb,
                });
            }
        }
    }

    let _ = region_op; // region composition operator is applied by the caller
    Ok(TextRegionResult {
        bitmap,
        x,
        y,
        comb_operator: ext_comb,
    })
}

/// The glyph placed for one instance: a borrowed dictionary symbol (RI=0) or a
/// freshly decoded refinement bitmap (RI=1).
enum PlacedSymbol<'a> {
    Borrowed(&'a Arc<MonoBitmap>),
    Owned(MonoBitmap),
}

impl PlacedSymbol<'_> {
    #[inline]
    fn bitmap(&self) -> &MonoBitmap {
        match self {
            PlacedSymbol::Borrowed(s) => s.as_ref(),
            PlacedSymbol::Owned(b) => b,
        }
    }
}

/// Require a finite value (OOB where a value is mandatory is malformed).
#[inline]
fn decode_value(d: DecodedInteger) -> Result<i32, DecodeError> {
    match d {
        DecodedInteger::Value(v) => Ok(v),
        DecodedInteger::OutOfBand => Err(DecodeError::Malformed {
            reason: "unexpected OOB in text region",
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn log2_ceil_matches_encoder() {
        // Mirror encoder::log2up for a spread of counts.
        fn log2up(v: u32) -> u32 {
            if v == 0 {
                return 0;
            }
            let is_pow = (v & (v - 1)) == 0;
            let mut r = 0;
            let mut val = v;
            while val > 1 {
                val >>= 1;
                r += 1;
            }
            r + if is_pow { 0 } else { 1 }
        }
        for v in [0u32, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 255, 256, 257, 1000] {
            assert_eq!(log2_ceil(v.max(1)), log2up(v.max(1)), "v={v}");
        }
    }
}
