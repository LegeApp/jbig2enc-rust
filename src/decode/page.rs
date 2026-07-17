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
use crate::decode::refinement::REFINEMENT_CONTEXT_COUNT;
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

    ctx.ensure_generic();
    // Disjoint field borrows: generic + integer context banks at once.
    let generic = &mut ctx.generic_contexts[..crate::decode::context::GENERIC_CONTEXT_COUNT];
    let int_ctx = &mut ctx.integer_contexts;
    let dict = decode_symbol_dictionary(seg.data, &imported, limits, int_ctx, generic)
        .map_err(|source| annotate(seg.header.number, source))?;

    let arc = Arc::new(dict);
    local.insert(
        seg.header.number,
        DecodedSegment::SymbolDictionary(arc.clone()),
    )?;
    Ok(arc)
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
                // Ensure the refinement bank is sized, then split disjoint field
                // borrows for the integer, IAID, and refinement context banks.
                if ctx.refinement_contexts.len() < REFINEMENT_CONTEXT_COUNT {
                    ctx.refinement_contexts
                        .resize(REFINEMENT_CONTEXT_COUNT, Default::default());
                }
                let int_ctx = &mut ctx.integer_contexts;
                let iaid_ctx = &mut ctx.iaid_contexts;
                let refine_ctx = &mut ctx.refinement_contexts[..REFINEMENT_CONTEXT_COUNT];
                let result =
                    decode_text_region(seg.data, &symbols, limits, int_ctx, iaid_ctx, refine_ctx)
                        .map_err(|source| annotate(seg.header.number, source))?;
                let target =
                    resolve_page(&mut pages, current, seg.header.page_association)?;
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
                if info.has_unknown_height() {
                    return Err(DecodeError::Unsupported(UnsupportedFeature::StripedPage));
                }
                check_page_pixels(&info, limits)?;
                let bitmap =
                    MonoBitmap::new(info.width, info.height, info.default_pixel, limits)?;
                pages.push(DecodedPage {
                    page_number: seg.header.page_association,
                    bitmap,
                });
                current = Some(pages.len() - 1);
            }
            Some(
                SegmentType::ImmediateGenericRegion
                | SegmentType::ImmediateLosslessGenericRegion
                | SegmentType::IntermediateGenericRegion,
            ) => {
                let region = parse_generic_region(seg.data).map_err(|source| {
                    DecodeError::Segment {
                        segment: seg.header.number,
                        source,
                    }
                })?;
                let region_bm =
                    decode_generic_region(&region, limits, ctx.generic_contexts())?;
                let target = resolve_page(&mut pages, current, seg.header.page_association)?;
                compose_region(
                    &mut pages[target].bitmap,
                    region_bm,
                    region.x,
                    region.y,
                    region.comb_operator,
                );
            }
            // Segments the self-decoder ignores structurally.
            Some(
                SegmentType::EndOfPage
                | SegmentType::EndOfStripe
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
            Some(
                SegmentType::PatternDictionary
                | SegmentType::IntermediateHalftoneRegion
                | SegmentType::ImmediateHalftoneRegion
                | SegmentType::ImmediateLosslessHalftoneRegion,
            ) => {
                return Err(DecodeError::Unsupported(UnsupportedFeature::HalftoneCoding));
            }
            Some(
                SegmentType::IntermediateGenericRefinementRegion
                | SegmentType::ImmediateGenericRefinementRegion
                | SegmentType::ImmediateLosslessGenericRefinementRegion,
            ) => {
                return Err(DecodeError::Unsupported(
                    UnsupportedFeature::RefinementRegion,
                ));
            }
            Some(SegmentType::Tables) => {
                return Err(DecodeError::Unsupported(UnsupportedFeature::HuffmanCoding));
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
