//! Arithmetic symbol-dictionary decoding (jbig2decplan.md §16, T.88 §6.5/§7.4.2).
//!
//! This inverts the encoder's default dictionary path
//! ([`crate::encode::document::symbol::dictionary::encode_symbol_dict_with_order`]):
//! arithmetic coding, template 0, `SDHUFF = 0`, `SDREFAGG = 0`, symbols sorted
//! by height then width, direct-coded bitmaps, run-length export flags. Every
//! new symbol bitmap is generic-decoded (template 0) from the *same* arithmetic
//! stream and generic-context bank, so contexts carry across symbols exactly as
//! the encoder's single coder does.
//!
//! Huffman, refinement/aggregate, and non-zero templates surface as typed
//! `Unsupported` errors (Phases 3/5). Limits are checked before every symbol
//! allocation and against the running dictionary pixel total.

use std::sync::Arc;

use crate::decode::arith::ArithmeticDecoder;
use crate::decode::error::{DecodeError, LimitError, UnsupportedFeature, usize_from_u32};
use crate::decode::generic::decode_generic_bitmap;
use crate::decode::integer::{DecodedInteger, IntegerContexts};
use crate::shared::bitmap::MonoBitmap;
use crate::shared::int_proc::IntProc;
use crate::shared::limits::DecodeLimits;
use crate::shared::mq_table::MqContext;
use crate::shared::reader::Reader;

/// A decoded symbol dictionary: the exported symbols, shareable across pages.
pub struct SymbolDictionary {
    /// Exported symbols in export order (jbig2decplan.md §16).
    pub exported_symbols: Box<[Arc<MonoBitmap>]>,
}

impl SymbolDictionary {
    /// The exported symbols.
    #[inline]
    pub fn symbols(&self) -> &[Arc<MonoBitmap>] {
        &self.exported_symbols
    }
}

/// Decode an arithmetic symbol-dictionary segment payload (T.88 §7.4.2).
///
/// `imported` holds the exported symbols of every referred dictionary, in
/// reference order; they precede this dictionary's new symbols in the combined
/// symbol space that the export flags select from.
///
/// `int_ctx` and `generic_ctx` are reused, worker-local scratch: both are reset
/// here (a fresh dictionary starts every context at zero, matching the
/// encoder's per-segment coder). `generic_ctx` must be at least `1 << 16`
/// entries.
pub fn decode_symbol_dictionary(
    payload: &[u8],
    imported: &[Arc<MonoBitmap>],
    limits: &DecodeLimits,
    int_ctx: &mut IntegerContexts,
    generic_ctx: &mut [MqContext],
) -> Result<SymbolDictionary, DecodeError> {
    let mut r = Reader::new(payload);

    // §7.4.2.1.1 symbol dictionary flags (16-bit).
    let flags = r.read_u16_be()?;
    let sdhuff = flags & 0x0001 != 0;
    let sdrefagg = flags & 0x0002 != 0;
    let sdtemplate = ((flags >> 10) & 0x0003) as u8;
    if sdhuff {
        return Err(DecodeError::Unsupported(UnsupportedFeature::HuffmanCoding));
    }
    if sdrefagg {
        return Err(DecodeError::Unsupported(UnsupportedFeature::SymbolRefinement));
    }
    if sdtemplate != 0 {
        return Err(DecodeError::Unsupported(UnsupportedFeature::GenericTemplate(
            sdtemplate,
        )));
    }

    // §7.4.2.1.2 SDAT: template 0 with SDHUFF=0 carries 4 adaptive pixels.
    let mut at = [(0i8, 0i8); 4];
    for slot in at.iter_mut() {
        let ax = r.read_i8()?;
        let ay = r.read_i8()?;
        *slot = (ax, ay);
    }
    // SDRAT (refinement AT) is only present when SDREFAGG=1, which we rejected.

    // §7.4.2.1.5/.6 exported and new symbol counts.
    let num_ex = r.read_u32_be()?;
    let num_new = r.read_u32_be()?;
    let num_new = usize_from_u32(num_new, "SDNUMNEWSYMS")?;
    let num_ex_usize = usize_from_u32(num_ex, "SDNUMEXSYMS")?;

    if num_new > limits.max_symbols {
        return Err(DecodeError::limit(LimitError::Count {
            what: "new symbols",
            value: num_new as u64,
            limit: limits.max_symbols as u64,
        }));
    }
    let total = imported
        .len()
        .checked_add(num_new)
        .ok_or(DecodeError::Overflow {
            operation: "dictionary symbol total",
        })?;
    if total > limits.max_symbols {
        return Err(DecodeError::limit(LimitError::Count {
            what: "dictionary symbols",
            value: total as u64,
            limit: limits.max_symbols as u64,
        }));
    }

    let data = &payload[r.position()..];

    // Fresh coder state: zero the reused context banks.
    int_ctx.reset();
    for c in generic_ctx.iter_mut() {
        *c = MqContext(0);
    }

    let mut dec = ArithmeticDecoder::new(data);

    let mut new_symbols: Vec<Arc<MonoBitmap>> = Vec::with_capacity(num_new.min(4096));
    let mut hc_height: i64 = 0;
    let mut total_pixels: u64 = 0;

    // §6.5.5 height-class loop.
    while new_symbols.len() < num_new {
        let dh = match int_ctx.decode(&mut dec, IntProc::Iadh) {
            DecodedInteger::Value(v) => v as i64,
            DecodedInteger::OutOfBand => {
                return Err(DecodeError::Malformed {
                    reason: "OOB where height-class delta expected",
                });
            }
        };
        hc_height = hc_height.checked_add(dh).ok_or(DecodeError::Overflow {
            operation: "height class height",
        })?;
        if hc_height <= 0 || hc_height > limits.max_height as i64 {
            return Err(DecodeError::Malformed {
                reason: "non-positive or oversized height class",
            });
        }

        let mut sym_width: i64 = 0;
        // §6.5.5 4) width loop within the class, terminated by OOB(IADW).
        loop {
            match int_ctx.decode(&mut dec, IntProc::Iadw) {
                DecodedInteger::OutOfBand => break,
                DecodedInteger::Value(v) => {
                    sym_width =
                        sym_width.checked_add(v as i64).ok_or(DecodeError::Overflow {
                            operation: "symbol width",
                        })?;
                }
            }
            if sym_width <= 0 || sym_width > limits.max_width as i64 {
                return Err(DecodeError::Malformed {
                    reason: "non-positive or oversized symbol width",
                });
            }
            if new_symbols.len() >= num_new {
                return Err(DecodeError::Malformed {
                    reason: "more symbols coded than SDNUMNEWSYMS",
                });
            }

            let px = (sym_width as u64)
                .checked_mul(hc_height as u64)
                .ok_or(DecodeError::Overflow {
                    operation: "symbol pixel count",
                })?;
            if px > limits.max_symbol_pixels {
                return Err(DecodeError::limit(LimitError::Pixels {
                    what: "symbol",
                    value: px,
                    limit: limits.max_symbol_pixels,
                }));
            }
            total_pixels = total_pixels.checked_add(px).ok_or(DecodeError::Overflow {
                operation: "dictionary pixel total",
            })?;
            if total_pixels > limits.max_total_dictionary_pixels {
                return Err(DecodeError::limit(LimitError::Pixels {
                    what: "dictionary",
                    value: total_pixels,
                    limit: limits.max_total_dictionary_pixels,
                }));
            }

            let bitmap = decode_generic_bitmap(
                &mut dec,
                sym_width as u32,
                hc_height as u32,
                at,
                generic_ctx,
                limits,
            )?;
            new_symbols.push(Arc::new(bitmap));
        }
    }

    // §6.5.10 export flags: alternating run lengths over [imported .. new].
    let exported =
        decode_export_flags(&mut dec, int_ctx, imported, &new_symbols, total, num_ex_usize)?;

    Ok(SymbolDictionary {
        exported_symbols: exported.into_boxed_slice(),
    })
}

/// Decode the run-length export flags and materialise the exported symbol list.
///
/// The combined symbol space is `imported` followed by `new_symbols`; runs
/// alternate not-exported/exported starting from not-exported, and must cover
/// the space exactly and export exactly `expected_exports` symbols.
fn decode_export_flags(
    dec: &mut ArithmeticDecoder<'_>,
    int_ctx: &mut IntegerContexts,
    imported: &[Arc<MonoBitmap>],
    new_symbols: &[Arc<MonoBitmap>],
    total: usize,
    expected_exports: usize,
) -> Result<Vec<Arc<MonoBitmap>>, DecodeError> {
    let mut exported: Vec<Arc<MonoBitmap>> = Vec::with_capacity(expected_exports.min(total));
    let mut index = 0usize;
    let mut ex_flag = false;
    // Each run either advances `index` or flips `ex_flag`; bound the iterations
    // so a malicious all-zero-run stream cannot spin forever.
    let max_runs = total.saturating_mul(2).saturating_add(4);
    let mut runs = 0usize;

    while index < total {
        runs += 1;
        if runs > max_runs {
            return Err(DecodeError::Malformed {
                reason: "export-flag run count exceeded",
            });
        }
        let run = match int_ctx.decode(dec, IntProc::Iaex) {
            DecodedInteger::Value(v) if v >= 0 => v as usize,
            _ => {
                return Err(DecodeError::Malformed {
                    reason: "invalid export run length",
                });
            }
        };
        if run > total - index {
            return Err(DecodeError::Malformed {
                reason: "export run overruns symbol space",
            });
        }
        if ex_flag {
            for j in index..index + run {
                let sym = if j < imported.len() {
                    imported[j].clone()
                } else {
                    new_symbols[j - imported.len()].clone()
                };
                exported.push(sym);
            }
        }
        index += run;
        ex_flag = !ex_flag;
    }

    if index != total {
        return Err(DecodeError::Malformed {
            reason: "export runs do not cover symbol space exactly",
        });
    }
    if exported.len() != expected_exports {
        return Err(DecodeError::Malformed {
            reason: "exported symbol count does not match SDNUMEXSYMS",
        });
    }
    Ok(exported)
}
