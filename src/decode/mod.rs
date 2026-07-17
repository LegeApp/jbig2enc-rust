//! The decoder half of the crate.
//!
//! Implemented in phases per `jbig2decplan.md` §23. Decoder code operates on
//! untrusted PDF input: no input-reachable `unwrap`/`expect`/`assert!`, all
//! allocations checked against `DecodeLimits`.
//!
//! Phase 1 (this milestone) decodes every generic-region stream the encoder
//! emits in its default and lossless configurations — standalone sequential
//! files and bare embedded/PDF segment sequences — exactly. Symbol, halftone,
//! refinement, MMR, and Product-B compatibility features surface as typed
//! `Unsupported` errors.
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod arith;
pub mod context;
pub mod error;
pub mod file;
pub mod generic;
pub mod page;
pub mod segment;

pub use context::{DecodeOptions, DecodeStrictness, DecoderContext};
pub use error::{DecodeError, LimitError, ParseError, UnsupportedFeature};
pub use file::{FileOrganization, ParsedDocument, ParsedSegment};
pub use page::{DecodedPage, PageInformation};

pub use crate::shared::limits::DecodeLimits;

use crate::shared::bitmap::MonoBitmap;

/// A fully decoded document: one bitmap per page, in stream order.
#[derive(Clone, Debug)]
pub struct DecodedDocument {
    pub organization: FileOrganization,
    pub pages: Vec<DecodedPage>,
}

impl DecodedDocument {
    /// The first page's bitmap, if any.
    pub fn first_page(&self) -> Option<&MonoBitmap> {
        self.pages.first().map(|p| &p.bitmap)
    }
}

/// Decode a standalone JBIG2 file (or a bare embedded stream — the organisation
/// is auto-detected from the file magic).
pub fn decode_file(data: &[u8], options: &DecodeOptions) -> Result<DecodedDocument, DecodeError> {
    let mut ctx = DecoderContext::new();
    decode_file_with_context(data, options, &mut ctx)
}

/// Decode a standalone/auto-detected document reusing a worker-local context.
pub fn decode_file_with_context(
    data: &[u8],
    options: &DecodeOptions,
    ctx: &mut DecoderContext,
) -> Result<DecodedDocument, DecodeError> {
    let doc = file::parse_auto(data, &options.limits)?;
    let organization = doc.organization;
    let pages = page::process_document(&doc, &options.limits, ctx)?;
    Ok(DecodedDocument { organization, pages })
}

/// Decode a single PDF-embedded page stream, returning its page bitmap
/// (jbig2decplan.md §5).
///
/// `globals` is accepted for API stability. Any globals content is symbol-
/// dictionary material (Phase 2); a non-empty globals stream that contributes
/// segments other than metadata therefore currently errors `Unsupported`.
pub fn decode_embedded(
    globals: Option<&[u8]>,
    page_data: &[u8],
    options: &DecodeOptions,
) -> Result<MonoBitmap, DecodeError> {
    let mut ctx = DecoderContext::new();
    decode_embedded_with_context(globals, page_data, options, &mut ctx)
}

/// Decode a single embedded page stream reusing a worker-local context.
pub fn decode_embedded_with_context(
    globals: Option<&[u8]>,
    page_data: &[u8],
    options: &DecodeOptions,
    ctx: &mut DecoderContext,
) -> Result<MonoBitmap, DecodeError> {
    // Phase 1: globals only ever carry symbol dictionaries, which are not yet
    // supported. Reject a globals stream that contains any decodable segment
    // beyond metadata so callers get a typed error rather than a wrong image.
    if let Some(g) = globals {
        let gdoc = file::parse_auto(g, &options.limits)?;
        for seg in &gdoc.segments {
            match seg.header.segment_type() {
                Some(
                    crate::shared::segment::SegmentType::EndOfFile
                    | crate::shared::segment::SegmentType::EndOfPage
                    | crate::shared::segment::SegmentType::EndOfStripe
                    | crate::shared::segment::SegmentType::Extension
                    | crate::shared::segment::SegmentType::Profiles,
                ) => {}
                _ => {
                    return Err(DecodeError::Unsupported(UnsupportedFeature::SymbolCoding));
                }
            }
        }
    }

    let doc = file::parse_auto(page_data, &options.limits)?;
    let mut pages = page::process_document(&doc, &options.limits, ctx)?;
    if pages.is_empty() {
        return Err(DecodeError::InvalidFileHeader);
    }
    Ok(pages.remove(0).bitmap)
}
