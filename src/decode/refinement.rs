//! Generic refinement-region decoding, GRTEMPLATE 0 (jbig2decplan.md §17, T.88
//! §6.3).
//!
//! Inverts the encoder's
//! [`crate::encode::arith::Jbig2ArithCoder::encode_refinement_region`]: a 13-bit
//! context is formed from a 3×3 reference neighbourhood, three already-decoded
//! target pixels, and one target adaptive-template pixel (GRAT1, nominally
//! `(-1, -1)`). The reference is offset from the target by `(GRDX, GRDY)`.
//!
//! The context bank is shared across every refined instance of one text region
//! (T.88 §6.4.11 carries the GR statistics across instances) and is *not* reset
//! per instance — the caller resets it once per text-region segment.

use crate::decode::arith::ArithmeticDecoder;
use crate::decode::error::DecodeError;
use crate::shared::bitmap::MonoBitmap;
use crate::shared::limits::DecodeLimits;
use crate::shared::mq_table::MqContext;

/// The number of GRTEMPLATE-0 refinement contexts (13-bit context).
pub const REFINEMENT_CONTEXT_COUNT: usize = 1 << 13;

/// Read a reference/target pixel at signed coordinates, zero outside the bitmap.
#[inline]
fn pget(bm: &MonoBitmap, x: i64, y: i64) -> u32 {
    if x < 0 || y < 0 || x >= bm.width() as i64 || y >= bm.height() as i64 {
        0
    } else {
        bm.get(x as u32, y as u32) as u32
    }
}

/// Decode a GRTEMPLATE-0 generic refinement region into a fresh `width × height`
/// bitmap, using `reference` (offset by `(grdx, grdy)`) as the predictor
/// (T.88 §6.3). `contexts` must be at least [`REFINEMENT_CONTEXT_COUNT`] entries
/// and is *not* reset here.
#[allow(clippy::too_many_arguments)]
pub fn decode_refinement_region(
    dec: &mut ArithmeticDecoder<'_>,
    reference: &MonoBitmap,
    width: u32,
    height: u32,
    grdx: i32,
    grdy: i32,
    grat: (i8, i8),
    contexts: &mut [MqContext],
    limits: &DecodeLimits,
) -> Result<MonoBitmap, DecodeError> {
    if contexts.len() < REFINEMENT_CONTEXT_COUNT {
        return Err(DecodeError::Overflow {
            operation: "refinement context array too small",
        });
    }
    let mut target = MonoBitmap::new(width, height, false, limits)?;
    if width == 0 || height == 0 {
        return Ok(target);
    }

    let grat_x = grat.0 as i64;
    let grat_y = grat.1 as i64;

    for y in 0..height as i64 {
        for x in 0..width as i64 {
            let rx = x - grdx as i64;
            let ry = y - grdy as i64;

            // Context layout matches the encoder exactly (T.88 Table 12,
            // GRTEMPLATE=0).
            let mut cx: usize = 0;
            cx |= pget(reference, rx - 1, ry - 1) as usize;
            cx |= (pget(reference, rx, ry - 1) as usize) << 1;
            cx |= (pget(reference, rx + 1, ry - 1) as usize) << 2;
            cx |= (pget(reference, rx - 1, ry) as usize) << 3;
            cx |= (pget(reference, rx, ry) as usize) << 4;
            cx |= (pget(reference, rx + 1, ry) as usize) << 5;
            cx |= (pget(&target, x - 1, y) as usize) << 6;
            cx |= (pget(reference, rx - 1, ry + 1) as usize) << 7;
            cx |= (pget(reference, rx, ry + 1) as usize) << 8;
            cx |= (pget(reference, rx + 1, ry + 1) as usize) << 9;
            cx |= (pget(&target, x + 1, y - 1) as usize) << 10;
            cx |= (pget(&target, x, y - 1) as usize) << 11;
            cx |= (pget(&target, x + grat_x, y + grat_y) as usize) << 12;

            let bit = dec.decode_bit(&mut contexts[cx]);
            if bit {
                target.set(x as u32, y as u32, true);
            }
        }
    }

    Ok(target)
}

#[cfg(all(test, feature = "encode"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::encode::arith::Jbig2ArithCoder;
    use crate::encode::sym::BitImage;

    /// Round-trip: encode the difference between a target and a reference glyph
    /// with the encoder's refinement coder, then decode it back exactly.
    #[test]
    fn refinement_roundtrip() {
        let limits = DecodeLimits::default();
        // Reference glyph.
        let mut refimg = BitImage::new(12, 16).unwrap();
        for y in 0..16 {
            for x in 0..12 {
                if x == 0 || x == 11 || y == 0 || y == 15 {
                    refimg.set(x, y, true);
                }
            }
        }
        // Target: reference with a few pixels toggled (same size, GRDX=GRDY=0).
        let mut target = refimg.clone();
        target.set(5, 5, true);
        target.set(6, 8, true);
        target.set(2, 10, false);

        let mut enc = Jbig2ArithCoder::new();
        enc.encode_refinement_region(&target, &refimg, 0, 0, 0, &[(-1, -1)])
            .unwrap();
        let data = enc.into_vec();

        let ref_mono = MonoBitmap::from_bit_image(&refimg, &limits).unwrap();
        let mut dec = ArithmeticDecoder::new(&data);
        let mut ctx = vec![MqContext::default(); REFINEMENT_CONTEXT_COUNT];
        let out = decode_refinement_region(
            &mut dec, &ref_mono, 12, 16, 0, 0, (-1, -1), &mut ctx, &limits,
        )
        .unwrap();

        for y in 0..16u32 {
            for x in 0..12u32 {
                assert_eq!(out.get(x, y), target.get(x, y), "pixel ({x},{y})");
            }
        }
    }
}
