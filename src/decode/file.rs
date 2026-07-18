//! File- and stream-level parsing (jbig2decplan.md §9): the standalone
//! sequential file organisation (T.88 Annex D) and the bare embedded/PDF
//! segment sequence. Parsing is separated from execution — this module only
//! locates segment boundaries and slices payloads; nothing is decoded here.

use crate::decode::error::{DecodeError, ParseError, UnsupportedFeature};
use crate::decode::segment::{SegmentHeader, parse_segment_header};
use crate::shared::limits::DecodeLimits;
use crate::shared::reader::Reader;
use crate::shared::segment::SegmentType;

/// The 8-byte JBIG2 file identification string (T.88 §D.4.1).
pub const FILE_MAGIC: [u8; 8] = [0x97, 0x4A, 0x42, 0x32, 0x0D, 0x0A, 0x1A, 0x0A];

/// File organisation (T.88 §D.4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileOrganization {
    /// Sequential: each segment header is immediately followed by its data.
    Sequential,
    /// Embedded/PDF: a bare segment sequence with no file header.
    Embedded,
}

/// One parsed segment: its header plus a borrow of its data bytes.
#[derive(Clone, Debug)]
pub struct ParsedSegment<'a> {
    pub header: SegmentHeader,
    pub data: &'a [u8],
}

/// A parsed document: the organisation plus every segment in stream order.
#[derive(Clone, Debug)]
pub struct ParsedDocument<'a> {
    pub organization: FileOrganization,
    pub segments: Vec<ParsedSegment<'a>>,
}

/// Parse a standalone JBIG2 file (T.88 Annex D). Only sequential organisation
/// is supported; random access returns a typed `Unsupported` error.
pub fn parse_file<'a>(
    data: &'a [u8],
    limits: &DecodeLimits,
) -> Result<ParsedDocument<'a>, DecodeError> {
    let mut reader = Reader::new(data);

    let magic = reader.take(FILE_MAGIC.len())?;
    if magic != FILE_MAGIC {
        return Err(DecodeError::InvalidFileHeader);
    }

    // §D.4.2 file-header flags. Bit 0 set => sequential organisation (matches
    // this crate's encoder and jbig2dec 0.20); clear => random access. Bit 1 =>
    // number of pages unknown (the 4-byte page count is then omitted).
    let flags = reader.read_u8()?;
    let sequential = flags & 0x01 != 0;
    let unknown_pages = flags & 0x02 != 0;
    if !sequential {
        return Err(DecodeError::Unsupported(
            UnsupportedFeature::RandomAccessOrganisation,
        ));
    }
    if !unknown_pages {
        // 4-byte number of pages; parsed but unused here.
        let _n_pages = reader.read_u32_be()?;
    }

    let segments = parse_segment_sequence(&mut reader, limits)?;
    Ok(ParsedDocument {
        organization: FileOrganization::Sequential,
        segments,
    })
}

/// Parse a bare embedded/PDF segment sequence (no file header).
pub fn parse_embedded<'a>(
    data: &'a [u8],
    limits: &DecodeLimits,
) -> Result<ParsedDocument<'a>, DecodeError> {
    let mut reader = Reader::new(data);
    let segments = parse_segment_sequence(&mut reader, limits)?;
    Ok(ParsedDocument {
        organization: FileOrganization::Embedded,
        segments,
    })
}

/// Parse consecutive `[header][data]` segments until the reader is exhausted.
fn parse_segment_sequence<'a>(
    reader: &mut Reader<'a>,
    limits: &DecodeLimits,
) -> Result<Vec<ParsedSegment<'a>>, DecodeError> {
    let mut segments = Vec::new();
    while !reader.is_empty() {
        if segments.len() >= limits.max_segments {
            return Err(DecodeError::limit(crate::decode::error::LimitError::Count {
                what: "segments",
                value: segments.len() as u64,
                limit: limits.max_segments as u64,
            }));
        }
        let header = parse_segment_header(reader).map_err(DecodeError::Parse)?;
        if header.referred_to.len() > limits.max_referred_segments {
            return Err(DecodeError::limit(crate::decode::error::LimitError::Count {
                what: "referred-to segments",
                value: header.referred_to.len() as u64,
                limit: limits.max_referred_segments as u64,
            }));
        }
        // §7.2.7 unknown data length: legal only for immediate generic regions,
        // whose length is recovered by scanning for the terminator sequence.
        let data = if header.is_unknown_length() {
            let ty = header.segment_type();
            if !matches!(ty, Some(SegmentType::ImmediateGenericRegion)) {
                return Err(DecodeError::Unsupported(
                    UnsupportedFeature::UnknownSegmentLength,
                ));
            }
            let remaining = reader.peek(reader.remaining()).unwrap_or(&[]);
            let len = unknown_generic_length(remaining)?;
            reader.take(len).map_err(|source| DecodeError::Segment {
                segment: header.number,
                source,
            })?
        } else {
            let len = usize::try_from(header.data_length).map_err(|_| DecodeError::Overflow {
                operation: "segment data length to usize",
            })?;
            reader.take(len).map_err(|source| DecodeError::Segment {
                segment: header.number,
                source,
            })?
        };
        segments.push(ParsedSegment { header, data });
    }
    Ok(segments)
}

/// Determine the data-part length of an unknown-length immediate generic region
/// (T.88 §7.2.7). The data begins with the 17-byte region-segment information
/// field and a 1-byte generic-flags field (the "eighteenth byte"), whose bit 0
/// selects MMR. The data ends with a terminator — `0xFF 0xAC` (arithmetic) or
/// `0x00 0x00` (MMR) — that can occur anywhere after the eighteenth byte,
/// followed by a four-byte row count. Returns the total data length in bytes.
fn unknown_generic_length(data: &[u8]) -> Result<usize, DecodeError> {
    // Need at least the 18-byte prefix plus a 2-byte terminator and 4-byte count.
    const PREFIX: usize = 18;
    if data.len() < PREFIX + 6 {
        return Err(DecodeError::Malformed {
            reason: "unknown-length generic region too short",
        });
    }
    let mmr = data[17] & 0x01 != 0;
    let (a, b) = if mmr { (0x00, 0x00) } else { (0xFF, 0xAC) };
    // Search for the terminator starting at the eighteenth byte (index 17),
    // scanning forward. The row count is the four bytes immediately after it.
    let mut i = PREFIX;
    while i + 6 <= data.len() {
        if data[i] == a && data[i + 1] == b {
            return Ok(i + 2 + 4);
        }
        i += 1;
    }
    Err(DecodeError::Malformed {
        reason: "unknown-length generic region terminator not found",
    })
}

/// Detect whether `data` begins with the JBIG2 file magic.
#[inline]
pub fn has_file_magic(data: &[u8]) -> bool {
    data.len() >= FILE_MAGIC.len() && data[..FILE_MAGIC.len()] == FILE_MAGIC
}

/// Parse either a standalone file (if the magic is present) or an embedded
/// stream (otherwise).
pub fn parse_auto<'a>(
    data: &'a [u8],
    limits: &DecodeLimits,
) -> Result<ParsedDocument<'a>, DecodeError> {
    if has_file_magic(data) {
        parse_file(data, limits)
    } else {
        parse_embedded(data, limits)
    }
}

/// Guard: a stray `ParseError` used only internally.
#[allow(dead_code)]
fn _assert_parse_error_is_used(_: ParseError) {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::shared::segment::SegmentType;

    fn limits() -> DecodeLimits {
        DecodeLimits::default()
    }

    /// A minimal sequential file: magic + flags + n_pages + one page-info
    /// segment (empty data) + one EOF segment.
    fn minimal_file() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&FILE_MAGIC);
        v.push(0x01); // sequential
        v.extend_from_slice(&1u32.to_be_bytes()); // n_pages
        // Segment 0: page info, 4 bytes data.
        v.extend_from_slice(&0u32.to_be_bytes());
        v.push(48); // type page info
        v.push(0); // no refs
        v.push(1); // page
        v.extend_from_slice(&4u32.to_be_bytes()); // data length
        v.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        // Segment 1: EOF, 0 bytes.
        v.extend_from_slice(&1u32.to_be_bytes());
        v.push(51);
        v.push(0);
        v.push(0);
        v.extend_from_slice(&0u32.to_be_bytes());
        v
    }

    #[test]
    fn parses_sequential_file() {
        let bytes = minimal_file();
        let doc = parse_file(&bytes, &limits()).unwrap();
        assert_eq!(doc.organization, FileOrganization::Sequential);
        assert_eq!(doc.segments.len(), 2);
        assert_eq!(
            doc.segments[0].header.segment_type(),
            Some(SegmentType::PageInformation)
        );
        assert_eq!(doc.segments[0].data, &[0xAA, 0xBB, 0xCC, 0xDD]);
        assert_eq!(
            doc.segments[1].header.segment_type(),
            Some(SegmentType::EndOfFile)
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = minimal_file();
        bytes[0] = 0x00;
        assert!(matches!(
            parse_file(&bytes, &limits()),
            Err(DecodeError::InvalidFileHeader)
        ));
    }

    #[test]
    fn random_access_is_unsupported() {
        let mut bytes = minimal_file();
        bytes[8] = 0x00; // clear sequential bit
        assert!(matches!(
            parse_file(&bytes, &limits()),
            Err(DecodeError::Unsupported(
                UnsupportedFeature::RandomAccessOrganisation
            ))
        ));
    }

    #[test]
    fn embedded_sequence_without_header() {
        // Just the two segments, no file magic.
        let full = minimal_file();
        let embedded = &full[13..]; // skip 8 magic + 1 flags + 4 n_pages
        let doc = parse_embedded(embedded, &limits()).unwrap();
        assert_eq!(doc.organization, FileOrganization::Embedded);
        assert_eq!(doc.segments.len(), 2);
    }

    #[test]
    fn auto_detects_organisation() {
        let full = minimal_file();
        assert_eq!(
            parse_auto(&full, &limits()).unwrap().organization,
            FileOrganization::Sequential
        );
        let embedded = &full[13..];
        assert_eq!(
            parse_auto(embedded, &limits()).unwrap().organization,
            FileOrganization::Embedded
        );
    }

    #[test]
    fn truncated_payload_is_error() {
        let mut bytes = minimal_file();
        bytes.truncate(bytes.len() - 6); // cut into the first segment payload/EOF
        assert!(parse_file(&bytes, &limits()).is_err());
    }
}
