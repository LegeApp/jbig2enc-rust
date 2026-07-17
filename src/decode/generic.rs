//! Arithmetic generic-region decoding, template 0 (jbig2decplan.md §14, T.88
//! §6.2).
//!
//! This mirrors the encoder's rolling `c1`/`c2`/`c3` context organisation
//! (`encode::arith::Jbig2ArithCoder::encode_generic_region_inner`) so a
//! round-trip is pixel-identical. Two paths share one 16-bit context layout:
//!
//! * a fast rolling path for the nominal template-0 AT positions (the only
//!   thing the encoder emits), reading packed rows directly with no per-pixel
//!   `get()`; and
//! * a general per-pixel path for arbitrary valid AT positions, which reduces
//!   to exactly the same context values for nominal AT (asserted by a test).
//!
//! MMR generic, TPGDON/typical prediction, and templates 1–3 are parsed and
//! returned as typed `Unsupported` errors (later phases).

use crate::decode::arith::ArithmeticDecoder;
use crate::decode::error::{DecodeError, ParseError, UnsupportedFeature};
use crate::shared::bitmap::MonoBitmap;
use crate::shared::limits::DecodeLimits;
use crate::shared::mq_table::MqContext;
use crate::shared::reader::Reader;

/// Nominal template-0 adaptive-template offsets (T.88 §6.2.5.3): AT1 (3,-1),
/// AT2 (-3,-1), AT3 (2,-2), AT4 (-2,-2). Also this crate's encoder default.
pub const NOMINAL_AT0: [(i8, i8); 4] = [(3, -1), (-3, -1), (2, -2), (-2, -2)];

/// A parsed generic-region segment: geometry, coding flags, and the coded data.
#[derive(Clone, Debug)]
pub struct GenericRegion<'a> {
    pub width: u32,
    pub height: u32,
    pub x: u32,
    pub y: u32,
    /// External combination operator (region flags low 3 bits).
    pub comb_operator: u8,
    pub mmr: bool,
    pub template: u8,
    pub tpgdon: bool,
    pub at: [(i8, i8); 4],
    /// Arithmetic (or MMR) coded data.
    pub data: &'a [u8],
}

/// Parse the generic-region segment payload (T.88 §7.4.6): region segment info
/// (§7.4.1) + generic flags + AT pixels + coded data.
pub fn parse_generic_region<'a>(payload: &'a [u8]) -> Result<GenericRegion<'a>, ParseError> {
    let mut r = Reader::new(payload);
    // §7.4.1 region segment information field.
    let width = r.read_u32_be()?;
    let height = r.read_u32_be()?;
    let x = r.read_u32_be()?;
    let y = r.read_u32_be()?;
    let region_flags = r.read_u8()?;
    let comb_operator = region_flags & 0x07;

    // §7.4.6.2 generic region segment flags.
    let flags = r.read_u8()?;
    let mmr = flags & 0x01 != 0;
    let template = (flags >> 1) & 0x03;
    let tpgdon = flags & 0x08 != 0;

    // AT pixels: template 0 has 4, template 1 has 1, templates 2/3 have 0.
    // (Only present for arithmetic coding, i.e. when MMR is off.)
    let mut at = [(0i8, 0i8); 4];
    if !mmr {
        let at_count = match template {
            0 => 4,
            1 => 1,
            _ => 0,
        };
        for slot in at.iter_mut().take(at_count) {
            let ax = r.read_i8()?;
            let ay = r.read_i8()?;
            *slot = (ax, ay);
        }
    }

    let data = &payload[r.position()..];
    Ok(GenericRegion {
        width,
        height,
        x,
        y,
        comb_operator,
        mmr,
        template,
        tpgdon,
        at,
        data,
    })
}

/// Decode a parsed generic region into a fresh region-local [`MonoBitmap`].
///
/// `contexts` must be at least `1 << 16` entries; it is not reset here (the
/// caller resets before each independent region — a fresh segment starts with
/// all-zero states).
pub fn decode_generic_region(
    region: &GenericRegion<'_>,
    limits: &DecodeLimits,
    contexts: &mut [MqContext],
) -> Result<MonoBitmap, DecodeError> {
    if region.mmr {
        // MMR (Group 4) generic region: no arithmetic coder, no AT pixels; the
        // payload is byte-aligned T.6 data for the full region (T.88 §6.2.6).
        return crate::decode::mmr::decode_mmr_bitmap(
            region.data,
            region.width,
            region.height,
            limits,
        );
    }
    if region.tpgdon {
        return Err(DecodeError::Unsupported(UnsupportedFeature::TypicalPrediction));
    }
    if region.template != 0 {
        return Err(DecodeError::Unsupported(UnsupportedFeature::GenericTemplate(
            region.template,
        )));
    }
    if (contexts.len() as u64) < (1u64 << 16) {
        return Err(DecodeError::Overflow {
            operation: "generic region context array too small",
        });
    }

    let mut bitmap = MonoBitmap::new(region.width, region.height, false, limits)?;
    if region.width == 0 || region.height == 0 {
        return Ok(bitmap);
    }

    let mut decoder = ArithmeticDecoder::new(region.data);
    if region.at == NOMINAL_AT0 {
        decode_template0_nominal(&mut bitmap, &mut decoder, contexts);
    } else {
        decode_template0_general(&mut bitmap, &mut decoder, contexts, region.at);
    }
    Ok(bitmap)
}

/// Decode a single template-0 generic bitmap from an *existing* arithmetic
/// decoder, without resetting `contexts` (jbig2decplan.md §16).
///
/// Symbol-dictionary bitmaps share one arithmetic stream and one generic-context
/// bank across every symbol in the dictionary, so this variant neither creates
/// its own decoder nor zeroes the contexts — the dictionary decoder owns both.
/// `contexts` must be at least `1 << 16` entries.
pub fn decode_generic_bitmap(
    decoder: &mut ArithmeticDecoder<'_>,
    width: u32,
    height: u32,
    at: [(i8, i8); 4],
    contexts: &mut [MqContext],
    limits: &DecodeLimits,
) -> Result<MonoBitmap, DecodeError> {
    if (contexts.len() as u64) < (1u64 << 16) {
        return Err(DecodeError::Overflow {
            operation: "generic bitmap context array too small",
        });
    }
    let mut bitmap = MonoBitmap::new(width, height, false, limits)?;
    if width == 0 || height == 0 {
        return Ok(bitmap);
    }
    if at == NOMINAL_AT0 {
        decode_template0_nominal(&mut bitmap, decoder, contexts);
    } else {
        decode_template0_general(&mut bitmap, decoder, contexts, at);
    }
    Ok(bitmap)
}

#[inline(always)]
fn sample(row: &[u32], width: u32, x: i64) -> u32 {
    if x < 0 || x >= width as i64 {
        return 0;
    }
    let x = x as usize;
    // row length is stride = ceil(width/32); x < width so x>>5 is in range.
    (row[x >> 5] >> (31 - (x & 31) as u32)) & 1
}

/// Fast rolling path for nominal AT (mirrors the encoder exactly).
fn decode_template0_nominal(
    bitmap: &mut MonoBitmap,
    decoder: &mut ArithmeticDecoder<'_>,
    contexts: &mut [MqContext],
) {
    let width = bitmap.width();
    let height = bitmap.height();
    let stride = bitmap.stride_words() as usize;
    let zero_row = vec![0u32; stride];
    let mut cur = vec![0u32; stride];

    for y in 0..height {
        for w in cur.iter_mut() {
            *w = 0;
        }
        // Immutable borrows of previously decoded rows.
        let prev1: &[u32] = if y >= 1 { bitmap.row(y - 1) } else { &zero_row };
        let prev2: &[u32] = if y >= 2 { bitmap.row(y - 2) } else { &zero_row };

        let mut c1 = (sample(prev2, width, 0) << 2)
            | (sample(prev2, width, 1) << 1)
            | sample(prev2, width, 2);
        let mut c2 = (sample(prev1, width, 0) << 3)
            | (sample(prev1, width, 1) << 2)
            | (sample(prev1, width, 2) << 1)
            | sample(prev1, width, 3);
        let mut c3 = 0u32;

        for x in 0..width {
            let tval = ((c1 << 11) | (c2 << 4) | c3) as usize;
            // tval < 2^16 <= contexts.len(); indexing is in range.
            let bit = decoder.decode_bit(&mut contexts[tval]);
            if bit {
                let xi = x as usize;
                cur[xi >> 5] |= 1u32 << (31 - (xi & 31));
            }
            let xi = x as i64;
            c1 = ((c1 << 1) | sample(prev2, width, xi + 3)) & 31;
            c2 = ((c2 << 1) | sample(prev1, width, xi + 4)) & 127;
            c3 = ((c3 << 1) | bit as u32) & 15;
        }

        bitmap.row_mut(y).copy_from_slice(&cur);
    }
}

/// General per-pixel path for arbitrary AT positions. Produces identical
/// context values to the rolling path when `at == NOMINAL_AT0`.
fn decode_template0_general(
    bitmap: &mut MonoBitmap,
    decoder: &mut ArithmeticDecoder<'_>,
    contexts: &mut [MqContext],
    at: [(i8, i8); 4],
) {
    let width = bitmap.width();
    let height = bitmap.height();
    let stride = bitmap.stride_words() as usize;
    let zero_row = vec![0u32; stride];
    let mut cur = vec![0u32; stride];

    // Context bit layout (MSB..LSB), matching the encoder's tval arrangement.
    // Fixed pixels and the four AT slots (nominal offsets in comments):
    //   bit15 AT4 (-2,-2)   bit14 (-1,-2)  bit13 (0,-2)  bit12 (1,-2)
    //   bit11 AT3 (2,-2)    bit10 AT2 (-3,-1) bit9 (-2,-1) bit8 (-1,-1)
    //   bit7  (0,-1)        bit6  (1,-1)   bit5 (2,-1)   bit4 AT1 (3,-1)
    //   bit3  (-4,0)        bit2  (-3,0)   bit1 (-2,0)   bit0 (-1,0)
    let (a1x, a1y) = (at[0].0 as i64, at[0].1 as i64);
    let (a2x, a2y) = (at[1].0 as i64, at[1].1 as i64);
    let (a3x, a3y) = (at[2].0 as i64, at[2].1 as i64);
    let (a4x, a4y) = (at[3].0 as i64, at[3].1 as i64);

    for y in 0..height {
        for w in cur.iter_mut() {
            *w = 0;
        }
        let prev1: &[u32] = if y >= 1 { bitmap.row(y - 1) } else { &zero_row };
        let prev2: &[u32] = if y >= 2 { bitmap.row(y - 2) } else { &zero_row };

        // Read a pixel at (x+dx, y+dy) from already-decoded data.
        let px = |cur: &[u32], x: i64, dx: i64, dy: i64| -> u32 {
            let xx = x + dx;
            match dy {
                0 => {
                    // Current row: only x' < x is decoded.
                    if xx < 0 || xx >= x || xx >= width as i64 {
                        0
                    } else {
                        sample(cur, width, xx)
                    }
                }
                -1 => sample(prev1, width, xx),
                -2 => sample(prev2, width, xx),
                _ => 0,
            }
        };

        for x in 0..width {
            let x = x as i64;
            let mut t = 0u32;
            t |= px(&cur, x, a4x, a4y) << 15;
            t |= px(&cur, x, -1, -2) << 14;
            t |= px(&cur, x, 0, -2) << 13;
            t |= px(&cur, x, 1, -2) << 12;
            t |= px(&cur, x, a3x, a3y) << 11;
            t |= px(&cur, x, a2x, a2y) << 10;
            t |= px(&cur, x, -2, -1) << 9;
            t |= px(&cur, x, -1, -1) << 8;
            t |= px(&cur, x, 0, -1) << 7;
            t |= px(&cur, x, 1, -1) << 6;
            t |= px(&cur, x, 2, -1) << 5;
            t |= px(&cur, x, a1x, a1y) << 4;
            t |= px(&cur, x, -4, 0) << 3;
            t |= px(&cur, x, -3, 0) << 2;
            t |= px(&cur, x, -2, 0) << 1;
            t |= px(&cur, x, -1, 0);

            let bit = decoder.decode_bit(&mut contexts[t as usize]);
            if bit {
                let xi = x as usize;
                cur[xi >> 5] |= 1u32 << (31 - (xi & 31));
            }
        }
        bitmap.row_mut(y).copy_from_slice(&cur);
    }
}

#[cfg(all(test, feature = "encode"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::decode::context::GENERIC_CONTEXT_COUNT;
    use crate::encode::arith::Jbig2ArithCoder;
    use crate::encode::sym::BitImage;

    fn contexts() -> Vec<MqContext> {
        vec![MqContext::default(); GENERIC_CONTEXT_COUNT]
    }

    /// Encode `img` as a raw generic payload and decode it back, asserting a
    /// pixel-exact round-trip.
    fn roundtrip_bitimage(img: &BitImage) {
        let data = Jbig2ArithCoder::encode_generic_payload(img, 0, &nominal_at0_i8()).unwrap();
        let (w, h) = (img.width as u32, img.height as u32);
        let region = GenericRegion {
            width: w,
            height: h,
            x: 0,
            y: 0,
            comb_operator: 0,
            mmr: false,
            template: 0,
            tpgdon: false,
            at: NOMINAL_AT0,
            data: &data,
        };
        let limits = DecodeLimits::default();
        let mut ctx = contexts();
        let bm = decode_generic_region(&region, &limits, &mut ctx).unwrap();
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    bm.get(x, y),
                    img.get(x, y),
                    "pixel ({x},{y}) w={w} h={h}"
                );
            }
        }
    }

    fn nominal_at0_i8() -> Vec<(i8, i8)> {
        NOMINAL_AT0.to_vec()
    }

    fn make_image(w: u32, h: u32, seed: u32) -> BitImage {
        let mut img = BitImage::new(w, h).unwrap();
        let mut s = seed.wrapping_add(1);
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                if (s >> 16) & 1 == 1 {
                    img.set(x, y, true);
                }
            }
        }
        img
    }

    #[test]
    fn roundtrip_odd_widths() {
        for &w in &[1u32, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
            for &h in &[1u32, 3, 8, 17] {
                roundtrip_bitimage(&make_image(w, h, w * 31 + h));
            }
        }
    }

    #[test]
    fn nominal_and_general_paths_agree() {
        // Decode the same stream with both paths and compare.
        let img = make_image(37, 20, 7);
        let data = Jbig2ArithCoder::encode_generic_payload(&img, 0, &nominal_at0_i8()).unwrap();
        let limits = DecodeLimits::default();

        let mut bm_fast = MonoBitmap::new(37, 20, false, &limits).unwrap();
        let mut d1 = ArithmeticDecoder::new(&data);
        let mut c1 = contexts();
        decode_template0_nominal(&mut bm_fast, &mut d1, &mut c1);

        let mut bm_gen = MonoBitmap::new(37, 20, false, &limits).unwrap();
        let mut d2 = ArithmeticDecoder::new(&data);
        let mut c2 = contexts();
        decode_template0_general(&mut bm_gen, &mut d2, &mut c2, NOMINAL_AT0);

        assert_eq!(bm_fast, bm_gen);
    }

    #[test]
    fn unsupported_surfaces() {
        let limits = DecodeLimits::default();
        let mut ctx = contexts();
        let base = GenericRegion {
            width: 8,
            height: 8,
            x: 0,
            y: 0,
            comb_operator: 0,
            mmr: false,
            template: 0,
            tpgdon: false,
            at: NOMINAL_AT0,
            data: &[0u8; 4],
        };
        let tp = GenericRegion { tpgdon: true, ..base.clone() };
        assert!(matches!(
            decode_generic_region(&tp, &limits, &mut ctx),
            Err(DecodeError::Unsupported(UnsupportedFeature::TypicalPrediction))
        ));
        let t2 = GenericRegion { template: 2, ..base.clone() };
        assert!(matches!(
            decode_generic_region(&t2, &limits, &mut ctx),
            Err(DecodeError::Unsupported(UnsupportedFeature::GenericTemplate(2)))
        ));
    }
}
