//! Page information parsing and region composition (jbig2decplan.md §15).
//!
//! Segments are processed in stream order. A page-information segment allocates
//! the page buffer; each immediate generic region is decoded and composited
//! onto its associated page. When a region exactly covers a blank page at the
//! origin with a compatible operator, the decoded region bitmap is moved into
//! the page with no copy (the dominant single-region scanned-page case).

use std::sync::Arc;

use crate::decode::context::DecoderContext;
use crate::decode::error::{DecodeError, LimitError, ParseError, UnsupportedFeature};
use crate::decode::file::{ParsedDocument, ParsedSegment};
use crate::decode::generic::{decode_generic_region, parse_generic_region};
use crate::decode::globals::DecodedGlobals;
use crate::decode::halftone_region::{decode_halftone_region, parse_halftone_region};
use crate::decode::pattern_dictionary::{decode_pattern_dictionary, PatternDictionary};
use crate::decode::refinement::{
    decode_refinement_region_templated, page_reference_window, parse_refinement_region,
    REFINEMENT_CONTEXT_COUNT,
};
use crate::decode::arith::ArithmeticDecoder;
use crate::decode::store::{DecodedSegment, SegmentStore};
use crate::decode::symbol_dictionary::{decode_symbol_dictionary, SymbolDictionary};
use crate::decode::text_region::decode_text_region;
use crate::shared::bitmap::{CombinationOperator, MonoBitmap};
use crate::shared::limits::DecodeLimits;
use crate::shared::reader::Reader;
use crate::shared::segment::SegmentType;

/// The unknown-height sentinel for a striped page (T.88 §7.4.8.5).
const UNKNOWN_HEIGHT: u32 = 0xFFFF_FFFF;

/// Parsed page-information segment (T.88 §7.4.8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageInformation {
    pub width: u32,
    pub height: u32,
    pub x_resolution: u32,
    pub y_resolution: u32,
    pub is_lossless: bool,
    pub contains_refinements: bool,
    /// Default pixel value: `true` => the page starts all black.
    pub default_pixel: bool,
    /// Default combination operator (bits 3–4).
    pub default_operator: u8,
    pub requires_aux_buffers: bool,
    pub operator_may_be_overridden: bool,
    pub striping_info: u16,
}

impl PageInformation {
    #[inline]
    pub fn has_unknown_height(&self) -> bool {
        self.height == UNKNOWN_HEIGHT
    }
}

/// Parse a page-information segment payload (T.88 §7.4.8).
pub fn parse_page_info(payload: &[u8]) -> Result<PageInformation, ParseError> {
    let mut r = Reader::new(payload);
    let width = r.read_u32_be()?;
    let height = r.read_u32_be()?;
    let x_resolution = r.read_u32_be()?;
    let y_resolution = r.read_u32_be()?;
    let flags = r.read_u8()?;
    let striping_info = r.read_u16_be()?;
    Ok(PageInformation {
        width,
        height,
        x_resolution,
        y_resolution,
        is_lossless: flags & 0x01 != 0,
        contains_refinements: flags & 0x02 != 0,
        default_pixel: flags & 0x04 != 0,
        default_operator: (flags >> 3) & 0x03,
        requires_aux_buffers: flags & 0x20 != 0,
        operator_may_be_overridden: flags & 0x40 != 0,
        striping_info,
    })
}

/// A decoded page: its association number and pixel bitmap.
#[derive(Clone, Debug)]
pub struct DecodedPage {
    pub page_number: u32,
    pub bitmap: MonoBitmap,
}

#[inline]
fn combination_operator(code: u8) -> CombinationOperator {
    match code {
        0 => CombinationOperator::Or,
        1 => CombinationOperator::And,
        2 => CombinationOperator::Xor,
        3 => CombinationOperator::Xnor,
        _ => CombinationOperator::Replace,
    }
}

/// Process every segment of a parsed document into decoded pages
/// (jbig2decplan.md §15). Unsupported segment types (symbols, halftone,
/// refinement, MMR) surface as typed `Unsupported` errors.
pub fn process_document(
    doc: &ParsedDocument<'_>,
    limits: &DecodeLimits,
    ctx: &mut DecoderContext,
) -> Result<Vec<DecodedPage>, DecodeError> {
    process_document_with_globals(doc, None, limits, ctx)
}

/// Decode a symbol-dictionary segment and register it in `local`, returning the
/// shared handle (jbig2decplan.md §16, §19). Imported symbols are gathered from
/// the segment's referred dictionaries (in `local`, then `globals`).
pub(crate) fn decode_symbol_dict_into(
    seg: &ParsedSegment<'_>,
    local: &mut SegmentStore,
    globals: Option<&SegmentStore>,
    limits: &DecodeLimits,
    ctx: &mut DecoderContext,
) -> Result<Arc<SymbolDictionary>, DecodeError> {
    let imported =
        local.gather_symbols(seg.header.number, &seg.header.referred_to, globals)?;
    let tables = local.gather_huffman_tables(&seg.header.referred_to, globals);

    ctx.ensure_generic();
    // Disjoint field borrows: generic + integer context banks at once.
    let generic = &mut ctx.generic_contexts[..crate::decode::context::GENERIC_CONTEXT_COUNT];
    let int_ctx = &mut ctx.integer_contexts;
    let dict = decode_symbol_dictionary(seg.data, &imported, &tables, limits, int_ctx, generic)
        .map_err(|source| annotate(seg.header.number, source))?;

    let arc = Arc::new(dict);
    local.insert(
        seg.header.number,
        DecodedSegment::SymbolDictionary(arc.clone()),
    )?;
    Ok(arc)
}

/// Decode a pattern-dictionary segment and register it in `local`, returning the
/// shared handle (jbig2decplan.md §18, §19).
pub(crate) fn decode_pattern_dict_into(
    seg: &ParsedSegment<'_>,
    local: &mut SegmentStore,
    limits: &DecodeLimits,
    ctx: &mut DecoderContext,
) -> Result<Arc<PatternDictionary>, DecodeError> {
    let generic = ctx.generic_contexts();
    let dict = decode_pattern_dictionary(seg.data, limits, generic)
        .map_err(|source| annotate(seg.header.number, source))?;
    let arc = Arc::new(dict);
    local.insert(
        seg.header.number,
        DecodedSegment::PatternDictionary(arc.clone()),
    )?;
    Ok(arc)
}

/// Parse a custom Huffman-table segment (type 53, T.88 §7.4.13) and register it
/// in `local` for later symbol dictionaries / text regions to reference.
pub(crate) fn decode_tables_into(
    seg: &ParsedSegment<'_>,
    local: &mut SegmentStore,
    limits: &DecodeLimits,
) -> Result<(), DecodeError> {
    let table = crate::decode::huffman::parse_custom_table(seg.data, limits)
        .map_err(|source| annotate(seg.header.number, source))?;
    local.insert(
        seg.header.number,
        DecodedSegment::HuffmanTable(Arc::new(table)),
    )
}

/// Attach the segment number to a bare parse error; other errors pass through.
fn annotate(segment: u32, err: DecodeError) -> DecodeError {
    match err {
        DecodeError::Parse(source) => DecodeError::Segment { segment, source },
        other => other,
    }
}

/// Process a parsed document, resolving referred symbol dictionaries against an
/// optional set of shared globals (jbig2decplan.md §6, §16, §17).
pub fn process_document_with_globals(
    doc: &ParsedDocument<'_>,
    globals: Option<&DecodedGlobals>,
    limits: &DecodeLimits,
    ctx: &mut DecoderContext,
) -> Result<Vec<DecodedPage>, DecodeError> {
    let mut pages: Vec<DecodedPage> = Vec::new();
    // Parallel to `pages`: whether each page's height was originally unknown
    // (0xFFFFFFFF) and so grows stripe-by-stripe (T.88 §7.4.8.5).
    let mut unknown_height: Vec<bool> = Vec::new();
    let mut current: Option<usize> = None;
    let mut store = SegmentStore::new();
    let globals_store = globals.map(|g| g.store());

    for seg in &doc.segments {
        let ty = seg.header.segment_type();
        match ty {
            Some(SegmentType::SymbolDictionary) => {
                decode_symbol_dict_into(seg, &mut store, globals_store, limits, ctx)?;
                continue;
            }
            Some(
                SegmentType::ImmediateTextRegion
                | SegmentType::ImmediateLosslessTextRegion
                | SegmentType::IntermediateTextRegion,
            ) => {
                let symbols = store.gather_symbols(
                    seg.header.number,
                    &seg.header.referred_to,
                    globals_store,
                )?;
                let tables = store.gather_huffman_tables(&seg.header.referred_to, globals_store);
                // Ensure the refinement bank is sized, then split disjoint field
                // borrows for the integer, IAID, and refinement context banks.
                if ctx.refinement_contexts.len() < REFINEMENT_CONTEXT_COUNT {
                    ctx.refinement_contexts
                        .resize(REFINEMENT_CONTEXT_COUNT, Default::default());
                }
                let int_ctx = &mut ctx.integer_contexts;
                let iaid_ctx = &mut ctx.iaid_contexts;
                let refine_ctx = &mut ctx.refinement_contexts[..REFINEMENT_CONTEXT_COUNT];
                let result = decode_text_region(
                    seg.data, &symbols, &tables, limits, int_ctx, iaid_ctx, refine_ctx,
                )
                .map_err(|source| annotate(seg.header.number, source))?;
                let target =
                    resolve_page(&mut pages, current, seg.header.page_association)?;
                grow_if_unknown(
                    &mut pages[target].bitmap,
                    unknown_height[target],
                    result.y,
                    result.bitmap.height(),
                    limits,
                )?;
                compose_region(
                    &mut pages[target].bitmap,
                    result.bitmap,
                    result.x,
                    result.y,
                    result.comb_operator,
                );
                continue;
            }
            _ => {}
        }
        match ty {
            Some(SegmentType::PageInformation) => {
                let info = parse_page_info(seg.data).map_err(|source| DecodeError::Segment {
                    segment: seg.header.number,
                    source,
                })?;
                let is_unknown = info.has_unknown_height();
                // A page of unknown height starts empty and grows as stripes /
                // regions arrive; a known-height page is allocated up front.
                let init_height = if is_unknown { 0 } else { info.height };
                if !is_unknown {
                    check_page_pixels(&info, limits)?;
                }
                let bitmap =
                    MonoBitmap::new(info.width, init_height, info.default_pixel, limits)?;
                pages.push(DecodedPage {
                    page_number: seg.header.page_association,
                    bitmap,
                });
                unknown_height.push(is_unknown);
                current = Some(pages.len() - 1);
            }
            Some(
                SegmentType::ImmediateGenericRegion
                | SegmentType::ImmediateLosslessGenericRegion
                | SegmentType::IntermediateGenericRegion,
            ) => {
                let mut region = parse_generic_region(seg.data).map_err(|source| {
                    DecodeError::Segment {
                        segment: seg.header.number,
                        source,
                    }
                })?;
                if seg.header.is_unknown_length() {
                    // §7.2.7 / §6.2.6: the data part ends with a four-byte row
                    // count; the region's true height is that count (<= the
                    // region-info height). Trim it from the coded data.
                    let dlen = region.data.len();
                    if dlen < 4 {
                        return Err(DecodeError::Malformed {
                            reason: "unknown-length generic region missing row count",
                        });
                    }
                    let rc = &region.data[dlen - 4..];
                    let row_count =
                        u32::from_be_bytes([rc[0], rc[1], rc[2], rc[3]]);
                    if row_count > region.height {
                        return Err(DecodeError::Malformed {
                            reason: "unknown-length row count exceeds region height",
                        });
                    }
                    region.height = row_count;
                    region.data = &region.data[..dlen - 4];
                }
                let region_bm =
                    decode_generic_region(&region, limits, ctx.generic_contexts())?;
                let target = resolve_page(&mut pages, current, seg.header.page_association)?;
                grow_if_unknown(
                    &mut pages[target].bitmap,
                    unknown_height[target],
                    region.y,
                    region_bm.height(),
                    limits,
                )?;
                compose_region(
                    &mut pages[target].bitmap,
                    region_bm,
                    region.x,
                    region.y,
                    region.comb_operator,
                );
            }
            // §7.4.10 end of stripe: a 4-byte end row. For an unknown-height
            // page this communicates the page size, so grow to include it.
            Some(SegmentType::EndOfStripe) => {
                if seg.data.len() >= 4 {
                    let end_row =
                        u32::from_be_bytes([seg.data[0], seg.data[1], seg.data[2], seg.data[3]]);
                    if let Some(idx) = current.filter(|&i| unknown_height[i]) {
                        let target_h = end_row.saturating_add(1);
                        pages[idx].bitmap.grow_to_height(
                            target_h,
                            limits.max_page_pixels,
                            limits,
                        )?;
                    }
                }
            }
            // Segments the self-decoder ignores structurally.
            Some(
                SegmentType::EndOfPage
                | SegmentType::EndOfFile
                | SegmentType::Profiles
                | SegmentType::Extension,
            ) => {}
            // Not handled in Phase 1.
            Some(
                SegmentType::SymbolDictionary
                | SegmentType::IntermediateTextRegion
                | SegmentType::ImmediateTextRegion
                | SegmentType::ImmediateLosslessTextRegion,
            ) => {
                return Err(DecodeError::Unsupported(UnsupportedFeature::SymbolCoding));
            }
            Some(SegmentType::PatternDictionary) => {
                decode_pattern_dict_into(seg, &mut store, limits, ctx)?;
            }
            Some(
                SegmentType::IntermediateHalftoneRegion
                | SegmentType::ImmediateHalftoneRegion
                | SegmentType::ImmediateLosslessHalftoneRegion,
            ) => {
                let region = parse_halftone_region(seg.data).map_err(|source| {
                    DecodeError::Segment {
                        segment: seg.header.number,
                        source,
                    }
                })?;
                let patterns = store
                    .pattern_dictionary(seg.header.number, &seg.header.referred_to, globals_store)?
                    .clone();
                let region_bm = decode_halftone_region(
                    &region,
                    &patterns,
                    limits,
                    ctx.generic_contexts(),
                )
                .map_err(|source| annotate(seg.header.number, source))?;
                let target = resolve_page(&mut pages, current, seg.header.page_association)?;
                grow_if_unknown(
                    &mut pages[target].bitmap,
                    unknown_height[target],
                    region.y,
                    region_bm.height(),
                    limits,
                )?;
                compose_region(
                    &mut pages[target].bitmap,
                    region_bm,
                    region.x,
                    region.y,
                    region.comb_operator,
                );
            }
            Some(
                SegmentType::ImmediateGenericRefinementRegion
                | SegmentType::ImmediateLosslessGenericRefinementRegion,
            ) => {
                // T.88 §7.4.7: an *immediate* generic refinement region with no
                // referred region segment refines the page buffer in place; the
                // reference is the page-buffer window under the region box
                // (§7.4.7.4) and the external combination operator is REPLACE.
                // Refinement regions that refer to another region segment reuse a
                // retained intermediate/auxiliary buffer, which the Phase 3
                // decoder does not keep — those stay Unsupported (Phase 5).
                if !seg.header.referred_to.is_empty() {
                    return Err(DecodeError::Unsupported(
                        UnsupportedFeature::RefinementRegion,
                    ));
                }
                let region = parse_refinement_region(seg.data)
                    .map_err(|source| annotate(seg.header.number, source))?;
                let target =
                    resolve_page(&mut pages, current, seg.header.page_association)?;
                grow_if_unknown(
                    &mut pages[target].bitmap,
                    unknown_height[target],
                    region.y,
                    region.height,
                    limits,
                )?;
                let reference = page_reference_window(
                    &pages[target].bitmap,
                    region.x,
                    region.y,
                    region.width,
                    region.height,
                    limits,
                )?;
                // §7.4.7.5 step 2: reset all arithmetic statistics to zero. Each
                // refinement region segment starts with a fresh context bank.
                let refine_ctx = ctx.refinement_contexts();
                let mut dec = ArithmeticDecoder::new(region.data);
                let refined = decode_refinement_region_templated(
                    &mut dec,
                    &reference,
                    region.width,
                    region.height,
                    0,
                    0,
                    region.grtemplate,
                    region.tpgron,
                    region.grat,
                    refine_ctx,
                    limits,
                )
                .map_err(|source| annotate(seg.header.number, source))?;
                compose_region(
                    &mut pages[target].bitmap,
                    refined,
                    region.x,
                    region.y,
                    // §7.4.7.5 step 1: no referred region => REPLACE.
                    4,
                );
            }
            Some(SegmentType::IntermediateGenericRefinementRegion) => {
                return Err(DecodeError::Unsupported(
                    UnsupportedFeature::RefinementRegion,
                ));
            }
            Some(SegmentType::Tables) => {
                decode_tables_into(seg, &mut store, limits)?;
            }
            Some(SegmentType::ColorPalette) | Some(SegmentType::FileHeader) => {}
            None => {
                return Err(DecodeError::Unsupported(UnsupportedFeature::SegmentType(
                    seg.header.type_code,
                )));
            }
        }
    }

    Ok(pages)
}

fn check_page_pixels(info: &PageInformation, limits: &DecodeLimits) -> Result<(), DecodeError> {
    let pixels = (info.width as u64)
        .checked_mul(info.height as u64)
        .ok_or(DecodeError::Overflow {
            operation: "page pixel count",
        })?;
    if pixels > limits.max_page_pixels {
        return Err(DecodeError::limit(LimitError::Pixels {
            what: "page",
            value: pixels,
            limit: limits.max_page_pixels,
        }));
    }
    Ok(())
}

/// Find the page a region should draw onto: prefer a page whose association
/// matches, else the most recently created page.
fn resolve_page(
    pages: &mut [DecodedPage],
    current: Option<usize>,
    association: u32,
) -> Result<usize, DecodeError> {
    if let Some(idx) = pages.iter().position(|p| p.page_number == association) {
        return Ok(idx);
    }
    current.ok_or(DecodeError::Unsupported(UnsupportedFeature::SegmentType(
        // A region before any page-information segment.
        38,
    )))
}

/// Composite a decoded region onto a page. When the region exactly covers a
/// blank page at the origin with an OR/REPLACE operator, move it in with no
/// per-word combine.
fn compose_region(
    page: &mut MonoBitmap,
    region: MonoBitmap,
    x: u32,
    y: u32,
    comb_operator: u8,
) {
    let op = combination_operator(comb_operator);
    let full_cover = x == 0
        && y == 0
        && region.width() == page.width()
        && region.height() == page.height();
    let movable = matches!(op, CombinationOperator::Or | CombinationOperator::Replace)
        && page_is_blank(page);
    if full_cover && movable {
        *page = region;
        return;
    }
    page.combine(&region, x as i32, y as i32, op);
}

/// Grow an unknown-height page so a region of height `h` placed at row `y`
/// fits (T.88 §7.4.8.5). A no-op for known-height pages.
fn grow_if_unknown(
    page: &mut MonoBitmap,
    unknown: bool,
    y: u32,
    h: u32,
    limits: &DecodeLimits,
) -> Result<(), DecodeError> {
    if unknown {
        let needed = (y as u64)
            .checked_add(h as u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or(DecodeError::Overflow {
                operation: "striped page height",
            })?;
        page.grow_to_height(needed, limits.max_page_pixels, limits)?;
    }
    Ok(())
}

/// Whether every word of the page is zero (blank / all-white default).
fn page_is_blank(page: &MonoBitmap) -> bool {
    for y in 0..page.height() {
        if page.row(y).iter().any(|&w| w != 0) {
            return false;
        }
    }
    true
}

#[cfg(all(test, feature = "encode"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::encode::structs::PageInfo;

    #[test]
    fn parse_page_info_matches_encoder() {
        let payload = PageInfo {
            width: 640,
            height: 480,
            xres: 300,
            yres: 300,
            is_lossless: true,
            default_pixel: false,
            default_operator: 2,
            ..Default::default()
        }
        .to_bytes();
        let info = parse_page_info(&payload).unwrap();
        assert_eq!(info.width, 640);
        assert_eq!(info.height, 480);
        assert_eq!(info.x_resolution, 300);
        assert!(info.is_lossless);
        assert!(!info.default_pixel);
        assert_eq!(info.default_operator, 2);
    }

    #[test]
    fn default_pixel_bit_sets_black_page() {
        let payload = PageInfo {
            width: 8,
            height: 2,
            default_pixel: true,
            ..Default::default()
        }
        .to_bytes();
        let info = parse_page_info(&payload).unwrap();
        assert!(info.default_pixel);
    }
}
