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
use crate::decode::huffman::{standard_table, BitReader, HuffmanTable, HuffmanValue};
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
    huffman_tables: &[Arc<HuffmanTable>],
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

    // Sign-extend the 5-bit SBDSOFFSET (shared by both coding paths).
    let sb_ds_offset: i32 = if ds_offset_raw >= 16 {
        ds_offset_raw as i32 - 32
    } else {
        ds_offset_raw as i32
    };

    if sbhuff {
        return decode_text_region_huffman(
            &mut r,
            payload,
            TextGeometry { width, height, x, y, ext_comb },
            TextFlags {
                sbrefine,
                log_strips,
                ref_corner,
                transposed,
                sb_comb_op,
                sb_def_pixel,
                sb_ds_offset,
            },
            symbols,
            huffman_tables,
            limits,
        );
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
                // RDW/RDH are bounded by the grw/grh range check above, but RDX/RDY
                // are unbounded decoded integers, so a malformed stream can drive
                // the sum past i32. Saturate: an out-of-range offset just makes the
                // reference window read as all-zero (a bounded, if wrong, bitmap).
                let grdx = (rdw.div_euclid(2) as i32).saturating_add(rdx);
                let grdy = (rdh.div_euclid(2) as i32).saturating_add(rdy);
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

/// Geometry from the region-segment information field.
struct TextGeometry {
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    ext_comb: u8,
}

/// The text-region flag fields shared by both coding paths.
struct TextFlags {
    sbrefine: bool,
    log_strips: u8,
    ref_corner: u8,
    transposed: bool,
    sb_comb_op: u8,
    sb_def_pixel: bool,
    sb_ds_offset: i32,
}

/// Select an FS/DS/DT/... Huffman table from a two-bit selection, consuming a
/// custom table from `custom` (in field order) for selection value 3.
fn tr_select<'a, I>(
    selection: u32,
    a: u8,
    b: u8,
    c: Option<u8>,
    custom: &mut I,
) -> Result<TrTable, DecodeError>
where
    I: Iterator<Item = &'a Arc<HuffmanTable>>,
{
    match selection {
        0 => Ok(TrTable::Owned(standard_table(a)?)),
        1 => Ok(TrTable::Owned(standard_table(b)?)),
        2 if c.is_some() => {
            let idx = c.ok_or(DecodeError::Malformed {
                reason: "reserved text-region Huffman table selection",
            })?;
            Ok(TrTable::Owned(standard_table(idx)?))
        }
        3 => custom
            .next()
            .map(|t| TrTable::Shared(t.clone()))
            .ok_or(DecodeError::Malformed {
                reason: "custom Huffman table referenced but not supplied",
            }),
        _ => Err(DecodeError::Malformed {
            reason: "reserved text-region Huffman table selection",
        }),
    }
}

enum TrTable {
    Owned(HuffmanTable),
    Shared(Arc<HuffmanTable>),
}
impl TrTable {
    #[inline]
    fn get(&self) -> &HuffmanTable {
        match self {
            TrTable::Owned(t) => t,
            TrTable::Shared(t) => t,
        }
    }
}

/// Decode the symbol-ID Huffman table (T.88 §7.4.3.1.7): 35 four-bit RUNCODE
/// lengths, run-length-coded symbol-ID code lengths, then B.3 code assignment.
fn decode_symbol_id_table(
    r: &mut BitReader<'_>,
    num_syms: usize,
    limits: &DecodeLimits,
) -> Result<HuffmanTable, DecodeError> {
    // 1) 35 RUNCODE code lengths (4 bits each).
    let mut runcode_lengths = [0u32; 35];
    for len in runcode_lengths.iter_mut() {
        *len = r.read_bits(4);
    }
    let runcode_table = HuffmanTable::from_code_lengths(&runcode_lengths)?;

    // 3–5) decode `num_syms` symbol-ID code lengths.
    let mut lengths: Vec<u32> = Vec::with_capacity(num_syms.min(1 << 16));
    let mut prev_len = 0u32;
    let mut guard = 0usize;
    let guard_max = num_syms.saturating_mul(2).saturating_add(64);
    while lengths.len() < num_syms {
        guard += 1;
        if guard > guard_max {
            return Err(DecodeError::Malformed {
                reason: "symbol-ID code-length run overrun",
            });
        }
        let code = match runcode_table.decode(r)? {
            HuffmanValue::Value(v) => v as u32,
            HuffmanValue::Oob => {
                return Err(DecodeError::Malformed {
                    reason: "OOB in symbol-ID runcode stream",
                });
            }
        };
        match code {
            0..=31 => {
                lengths.push(code);
                prev_len = code;
            }
            32 => {
                // Copy the previous length 3–6 times (2 extra bits + 3).
                let repeat = r.read_bits(2) as usize + 3;
                for _ in 0..repeat {
                    if lengths.len() >= num_syms {
                        break;
                    }
                    lengths.push(prev_len);
                }
            }
            33 => {
                // Repeat length 0 for 3–10 times (3 extra bits + 3).
                let repeat = r.read_bits(3) as usize + 3;
                for _ in 0..repeat {
                    if lengths.len() >= num_syms {
                        break;
                    }
                    lengths.push(0);
                }
                prev_len = 0;
            }
            34 => {
                // Repeat length 0 for 11–138 times (7 extra bits + 11).
                let repeat = r.read_bits(7) as usize + 11;
                for _ in 0..repeat {
                    if lengths.len() >= num_syms {
                        break;
                    }
                    lengths.push(0);
                }
                prev_len = 0;
            }
            _ => {
                return Err(DecodeError::Malformed {
                    reason: "invalid symbol-ID runcode",
                });
            }
        }
        if lengths.len() > limits.max_symbols {
            return Err(DecodeError::limit(LimitError::Count {
                what: "symbol-ID code lengths",
                value: lengths.len() as u64,
                limit: limits.max_symbols as u64,
            }));
        }
    }
    // 6) byte-align, then 7) build SBSYMCODES.
    r.align_to_byte();
    HuffmanTable::from_code_lengths(&lengths)
}

/// Decode a Huffman-coded text region (SBHUFF=1). Non-transposed, non-refined
/// (SBREFINE=0) — transposed and refined Huffman text regions are Phase 5e.
#[allow(clippy::too_many_arguments)]
fn decode_text_region_huffman(
    r: &mut Reader<'_>,
    payload: &[u8],
    geom: TextGeometry,
    tf: TextFlags,
    symbols: &[Arc<MonoBitmap>],
    huffman_tables: &[Arc<HuffmanTable>],
    limits: &DecodeLimits,
) -> Result<TextRegionResult, DecodeError> {
    if tf.transposed {
        return Err(DecodeError::Unsupported(
            UnsupportedFeature::TransposedTextRegion,
        ));
    }
    if tf.sbrefine {
        // Huffman + refinement is Phase 5e.
        return Err(DecodeError::Unsupported(UnsupportedFeature::RefinementRegion));
    }

    // §7.4.3.1.2 text region Huffman flags (16-bit).
    let hflags = r.read_u16_be()?;
    let fs_sel = (hflags & 0x0003) as u32;
    let ds_sel = ((hflags >> 2) & 0x0003) as u32;
    let dt_sel = ((hflags >> 4) & 0x0003) as u32;
    // RDW/RDH/RDX/RDY/RSIZE selections are only meaningful when SBREFINE=1.

    // §7.4.3.1.4 SBNUMINSTANCES.
    let num_instances = r.read_u32_be()?;
    if num_instances as usize > limits.max_symbols {
        return Err(DecodeError::limit(LimitError::Count {
            what: "text instances",
            value: num_instances as u64,
            limit: limits.max_symbols as u64,
        }));
    }

    // Select FS/DS/DT tables (custom tables consumed in field order).
    let mut custom = huffman_tables.iter();
    let fs_table = tr_select(fs_sel, 6, 7, None, &mut custom)?;
    let ds_table = tr_select(ds_sel, 8, 9, Some(10), &mut custom)?;
    let dt_table = tr_select(dt_sel, 11, 12, Some(13), &mut custom)?;

    // The symbol-ID table and the strip data share one bit stream starting at
    // the current byte position (§7.4.3.1.5 then the strip data).
    let bit_start = r.position();
    let mut br = BitReader::new(&payload[bit_start..]);
    let sb_num_syms = symbols.len();
    let sym_table = decode_symbol_id_table(&mut br, sb_num_syms, limits)?;

    let sb_strips: i64 = 1i64 << (tf.log_strips & 0x03);
    let log_strips = (tf.log_strips & 0x03) as u32;
    let symbol_op = comb_op(tf.sb_comb_op);
    let mut bitmap = MonoBitmap::new(geom.width, geom.height, tf.sb_def_pixel, limits)?;

    // §6.4.5 2) initial STRIPT = -(DT0 * SBSTRIPS); FIRSTS = 0.
    let dt0 = huff_value(dt_table.get().decode(&mut br)?)? as i64;
    let mut strip_t: i64 = -(dt0 * sb_strips);
    let mut first_s: i64 = 0;
    let mut n_inst: u32 = 0;

    while n_inst < num_instances {
        // §6.4.5 4b) strip delta T.
        let dt = huff_value(dt_table.get().decode(&mut br)?)? as i64;
        strip_t = strip_t
            .checked_add(dt * sb_strips)
            .ok_or(DecodeError::Overflow { operation: "strip T" })?;

        // §6.4.5 4c i) first S coordinate.
        let dfs = huff_value(fs_table.get().decode(&mut br)?)? as i64;
        first_s = first_s
            .checked_add(dfs)
            .ok_or(DecodeError::Overflow { operation: "first S" })?;
        let mut cur_s = first_s;
        let mut first_in_strip = true;

        loop {
            if !first_in_strip {
                // §6.4.5 4c ii) subsequent S; OOB ends the strip.
                match ds_table.get().decode(&mut br)? {
                    HuffmanValue::Oob => break,
                    HuffmanValue::Value(ids) => {
                        cur_s = cur_s
                            .checked_add(ids as i64 + tf.sb_ds_offset as i64)
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

            // §6.4.5 4c iii) T within the strip.
            let cur_t = if sb_strips == 1 {
                0
            } else {
                br.read_bits(log_strips) as i64
            };
            let t_i = strip_t
                .checked_add(cur_t)
                .ok_or(DecodeError::Overflow { operation: "T coordinate" })?;

            // §6.4.5 4c iv) symbol ID via SBSYMCODES.
            let id = match sym_table.decode(&mut br)? {
                HuffmanValue::Value(v) => v as usize,
                HuffmanValue::Oob => {
                    return Err(DecodeError::Malformed {
                        reason: "OOB decoding symbol ID",
                    });
                }
            };
            let symbol = symbols.get(id).ok_or(DecodeError::Malformed {
                reason: "symbol id out of range",
            })?;
            let wi = symbol.width() as i64;
            let hi = symbol.height() as i64;

            place_symbol(
                &mut bitmap,
                symbol,
                tf.ref_corner,
                &mut cur_s,
                t_i,
                wi,
                hi,
                symbol_op,
            )?;

            n_inst += 1;
        }
    }

    Ok(TextRegionResult {
        bitmap,
        x: geom.x,
        y: geom.y,
        comb_operator: geom.ext_comb,
    })
}

/// Non-transposed symbol placement (T.88 §6.4.5 4c vi–xi) for all four
/// reference corners, advancing `cur_s` per the spec's before/after rules.
#[allow(clippy::too_many_arguments)]
fn place_symbol(
    bitmap: &mut MonoBitmap,
    symbol: &MonoBitmap,
    ref_corner: u8,
    cur_s: &mut i64,
    t_i: i64,
    wi: i64,
    hi: i64,
    op: CombinationOperator,
) -> Result<(), DecodeError> {
    // REFCORNER: 0 BOTTOMLEFT, 1 TOPLEFT, 2 BOTTOMRIGHT, 3 TOPRIGHT.
    let right = ref_corner == 2 || ref_corner == 3;
    let bottom = ref_corner == 0 || ref_corner == 2;

    // vi) right corners advance CURS by WI-1 *before* placement.
    if right {
        *cur_s = cur_s
            .checked_add(wi - 1)
            .ok_or(DecodeError::Overflow { operation: "S advance (pre)" })?;
    }
    let si = *cur_s;
    // viii) placement corner → top-left of the bitmap.
    let s_left = if right { si - (wi - 1) } else { si };
    let t_top = if bottom { t_i - (hi - 1) } else { t_i };
    bitmap.combine(symbol, clamp_i32(s_left), clamp_i32(t_top), op);

    // xi) left corners advance CURS by WI-1 *after* placement.
    if !right {
        *cur_s = cur_s
            .checked_add(wi - 1)
            .ok_or(DecodeError::Overflow { operation: "S advance (post)" })?;
    }
    Ok(())
}

/// Require a finite Huffman value (OOB where a value is mandatory is malformed).
#[inline]
fn huff_value(v: HuffmanValue) -> Result<i32, DecodeError> {
    match v {
        HuffmanValue::Value(v) => Ok(v),
        HuffmanValue::Oob => Err(DecodeError::Malformed {
            reason: "unexpected OOB in Huffman text region",
        }),
    }
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
