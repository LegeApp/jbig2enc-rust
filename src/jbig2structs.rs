/// Pruned Rust equivalents of JBIG2 structs and segment headers
use byteorder::{BigEndian, WriteBytesExt};
use std::io::{self, Write};

#[cfg(feature = "trace_encoder")]
use log::debug;

#[cfg(not(feature = "trace_encoder"))]
use crate::debug;

/// JBIG2 file format magic number
pub const JB2_MAGIC: &[u8; 10] = b"\x97JBIG2\r\n\x1A\n";

/// JBIG2 file format version
pub const JB2_VERSION: u8 = 0x02;

/// JBIG2 segment types as defined in the specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SegmentType {
    #[default]
    SymbolDictionary = 0,
    IntermediateTextRegion = 4,
    ImmediateTextRegion = 6,
    ImmediateLosslessTextRegion = 7,
    PatternDictionary = 16,
    IntermediateHalftoneRegion = 20,
    ImmediateHalftoneRegion = 22,
    ImmediateLosslessHalftoneRegion = 23,
    IntermediateGenericRegion = 36,
    ImmediateGenericRegion = 38,
    ImmediateLosslessGenericRegion = 39,
    IntermediateGenericRefinementRegion = 40,
    ImmediateGenericRefinementRegion = 42,
    ImmediateLosslessGenericRefinementRegion = 43,
    PageInformation = 48,
    EndOfPage = 49,
    EndOfStripe = 50,
    EndOfFile = 51,
    Profiles = 52,
    Tables = 53,
    ColorPalette = 54,
    FileHeader = 56,
    Extension = 62,
}

impl TryFrom<u8> for SegmentType {
    type Error = io::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SegmentType::SymbolDictionary),
            4 => Ok(SegmentType::IntermediateTextRegion),
            6 => Ok(SegmentType::ImmediateTextRegion),
            7 => Ok(SegmentType::ImmediateLosslessTextRegion),
            16 => Ok(SegmentType::PatternDictionary),
            20 => Ok(SegmentType::IntermediateHalftoneRegion),
            22 => Ok(SegmentType::ImmediateHalftoneRegion),
            23 => Ok(SegmentType::ImmediateLosslessHalftoneRegion),
            36 => Ok(SegmentType::IntermediateGenericRegion),
            38 => Ok(SegmentType::ImmediateGenericRegion),
            39 => Ok(SegmentType::ImmediateLosslessGenericRegion),
            40 => Ok(SegmentType::IntermediateGenericRefinementRegion),
            42 => Ok(SegmentType::ImmediateGenericRefinementRegion),
            43 => Ok(SegmentType::ImmediateLosslessGenericRefinementRegion),
            48 => Ok(SegmentType::PageInformation),
            49 => Ok(SegmentType::EndOfPage),
            50 => Ok(SegmentType::EndOfStripe),
            51 => Ok(SegmentType::EndOfFile),
            52 => Ok(SegmentType::Profiles),
            53 => Ok(SegmentType::Tables),
            54 => Ok(SegmentType::ColorPalette),
            56 => Ok(SegmentType::FileHeader),
            62 => Ok(SegmentType::Extension),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid segment type: {}", value),
            )),
        }
    }
}

// -----------------------------------------------------------------------------
// File header (magic + flags + number of pages)
// -----------------------------------------------------------------------------

/// Represents the JBIG2 file header as per the specification (§D.4.1)
#[derive(Debug)]
pub struct FileHeader {
    pub organisation_type: bool, // 1 bit: 0 = sequential, 1 = random-access
    pub unknown_n_pages: bool,   // 1 bit: 1 = number of pages unknown
    pub n_pages: u32,            // Number of pages (big-endian), omitted if unknown_n_pages is true
}

impl FileHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        const MAGIC: &[u8] = b"\x97JB2\r\n\x1A\n";
        let mut buf = Vec::with_capacity(8 + 1 + if self.unknown_n_pages { 0 } else { 4 });
        buf.extend_from_slice(MAGIC);

        let mut flags = 0u8;
        if self.organisation_type {
            flags |= 0x01;
        }
        if self.unknown_n_pages {
            flags |= 0x02;
        }
        buf.push(flags);

        if !self.unknown_n_pages {
            buf.write_u32::<BigEndian>(self.n_pages).unwrap();
        }
        buf
    }
}

// -----------------------------------------------------------------------------
// Page information segment payload (§7.4.8)
// -----------------------------------------------------------------------------

/// Represents the page information segment payload
#[derive(Debug, Default)]
pub struct PageInfo {
    pub width: u32,                 // Page width in pixels
    pub height: u32,                // Page height in pixels
    pub xres: u32,                  // X resolution in pixels per inch
    pub yres: u32,                  // Y resolution in pixels per inch
    pub is_lossless: bool,          // Bit 0: 1 if lossless
    pub contains_refinements: bool, // Bit 1: 1 if contains refinement regions
    pub default_pixel: bool,        // Bit 2: Default pixel value (0 = black, 1 = white)
    pub default_operator: u8,       // Bits 3-4: Default combination operator (0-3)
    pub aux_buffers: bool,          // Bit 5: 1 if auxiliary buffers are used
    pub operator_override: bool,    // Bit 6: 1 if combination operator can be overridden
    pub reserved: bool,             // Bit 7: Must be 0
    pub segment_flags: u16,         // Additional segment flags
}

impl PageInfo {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 * 4 + 1 + 2);
        buf.write_u32::<BigEndian>(self.width).unwrap();
        buf.write_u32::<BigEndian>(self.height).unwrap();
        buf.write_u32::<BigEndian>(self.xres).unwrap();
        buf.write_u32::<BigEndian>(self.yres).unwrap();

        let mut b = 0u8;
        if self.is_lossless {
            b |= 0x01;
        }
        if self.contains_refinements {
            b |= 0x02;
        }
        if self.default_pixel {
            b |= 0x04;
        }
        b |= (self.default_operator & 0x03) << 3;
        if self.aux_buffers {
            b |= 0x20;
        }
        if self.operator_override {
            b |= 0x40;
        }
        // Bit 7 (reserved) remains 0
        buf.push(b);
        buf.write_u16::<BigEndian>(self.segment_flags).unwrap();
        buf
    }
}

// -----------------------------------------------------------------------------
// Generic region parameters (§7.4.6)
// -----------------------------------------------------------------------------

/// Represents the parameters for a generic region segment as per the JBIG2 specification
#[derive(Debug)]
pub struct GenericRegionParams {
    pub width: u32,         // Region width in pixels
    pub height: u32,        // Region height in pixels
    pub x: u32,             // X-coordinate of the top-left corner
    pub y: u32,             // Y-coordinate of the top-left corner
    pub comb_operator: u8,  // Combination operator (0-4: OR, AND, XOR, XNOR, REPLACE)
    pub mmr: bool,          // 1 = MMR coding, 0 = arithmetic coding
    pub template: u8,       // Generic region template (0-3)
    pub tpgdon: bool,       // Typical prediction generic decoding on/off
    pub at: [(i8, i8); 4], // Adaptive template coordinates (a1x, a1y, ..., a4x, a4y)
}

impl GenericRegionParams {
    pub fn to_bytes(&self) -> Vec<u8> {
        use byteorder::{BigEndian, WriteBytesExt};
        // 18 bytes (width, height, x, y, comb_op, flags) + AT bytes (8 if template == 0, 4 otherwise)
        let mut buf = Vec::with_capacity(18 + if self.template == 0 { 8 } else { 4 });
        
        buf.write_u32::<BigEndian>(self.width).unwrap();
        buf.write_u32::<BigEndian>(self.height).unwrap();
        buf.write_u32::<BigEndian>(self.x).unwrap();
        buf.write_u32::<BigEndian>(self.y).unwrap();
        buf.push(self.comb_operator);
        
        let mut flags = 0u8;
        if self.mmr {
            flags |= 0x01; // Bit 0: MMR (only for MMR coding)
        }
        flags |= (self.template & 0x03) << 1; // Bits 1-2: GBTEMPLATE
        if self.tpgdon {
            flags |= 0x08; // Bit 3: TPGDON
        }
        // Bits 4-7 are reserved and set to 0
        buf.push(flags);

        // Write AT coordinates: all 4 pairs if template == 0, first 2 pairs otherwise
        let at_count = match self.template {
            0 => 4,   // Header requires 4 but context uses 0
            1 => 1,   // Template 1 uses 1 AT pixel
            _ => 0,   // Templates 2-3 use 0
        };
        }
        buf
    }
}

/// High-level configuration for generic region segments
#[derive(Clone, Debug)]
pub struct GenericRegionConfig {
    // Segment header parameters
    pub width: u32,
    pub height: u32,
    pub x: u32,
    pub y: u32,
    pub comb_operator: u8,  // Combination operator (0 = OR, 1 = AND, etc.)

    // Arithmetic encoding parameters
    pub template: u8,        // Template ID (0–3)
    pub tpgdon: bool,        // Typical prediction generic decoding
    pub mmr: bool,           // MMR coding (true) or arithmetic (false)
    pub at_pixels: Vec<(i8, i8)>, // Adaptive template pixels (dx, dy)

    // Metadata (optional, for page info alignment)
    pub dpi: u32,            // Resolution in DPI
}

impl GenericRegionConfig {
    /// Creates a new generic region config with defaults
    pub fn new(width: u32, height: u32, dpi: u32) -> Self {
        Self {
            width,
            height,
            x: 0,
            y: 0,
            comb_operator: 0, // Default to OR
            template: 0,      // Default to template 0
            tpgdon: true,     // Enable typical prediction
            at_pixels: vec![(3, -1), (-3, -1), (2, -2), (-2, -2)], // Standard AT pixels
            mmr: false,         // Default to arithmetic coding
            dpi,
        }
    }

    /// Validation to ensure compliance with JBIG2 spec
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.template > 3 {
            return Err("Template ID must be 0–3");
        }
        if self.at_pixels.len() > 4 {
            return Err("Maximum 4 AT pixels allowed");
        }
        if self.comb_operator > 4 {
            return Err("Invalid combination operator");
        }
        Ok(())
    }
}

impl From<GenericRegionConfig> for GenericRegionParams {
    fn from(cfg: GenericRegionConfig) -> Self {
        let mut at = [(0i8, 0i8); 4];
        for (i, &(dx, dy)) in cfg.at_pixels.iter().enumerate().take(4) {
            at[i] = (dx, dy);
        }
        GenericRegionParams {
            width: cfg.width,
            height: cfg.height,
            x: cfg.x,
            y: cfg.y,
            comb_operator: cfg.comb_operator,
            mmr: cfg.mmr,      // MMR coding flag from config
            template: cfg.template,
            tpgdon: cfg.tpgdon,
            at,
        }
    }
}

// -----------------------------------------------------------------------------
// Symbol dictionary parameters (§7.4.2)
// -----------------------------------------------------------------------------

/// Represents the parameters for a symbol dictionary segment
#[derive(Debug)]
pub struct SymbolDictParams {
    pub sd_template: u8, // Symbol dictionary template (0-3)
    pub at: [(i8, i8); 4], // Adaptive template coordinates (a1x, a1y, ..., a4x, a4y)
    pub exsyms: u32,     // Number of exported symbols
    pub newsyms: u32,    // Number of new symbols
}

impl SymbolDictParams {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(2 + 8 + 4 + 4);
        let b = self.sd_template & 0x03; // SDTEMPLATE in low 2 bits
        buf.push(b);
        buf.push(0); // Reserved flags
        for &(x, y) in &self.at {
            buf.push(x as u8);
            buf.push(y as u8);
        }
        buf.write_u32::<BigEndian>(self.exsyms).unwrap();
        buf.write_u32::<BigEndian>(self.newsyms).unwrap();
        buf
    }
}

// -----------------------------------------------------------------------------
// Text region parameters (§7.4.3)
// -----------------------------------------------------------------------------

/// Represents the parameters for a text region segment
#[derive(Debug)]
pub struct TextRegionParams {
    pub width: u32,          // Region width in pixels
    pub height: u32,         // Region height in pixels
    pub x: u32,              // X-coordinate of the top-left corner
    pub y: u32,              // Y-coordinate of the top-left corner
    pub ds_offset: u8,       // Signed 5-bit offset (SBDSOFFSET)
    pub refine: bool,        // SBREFINE flag
    pub log_strips: u8,      // LOGSBSTRIPS (0-3)
    pub ref_corner: u8,      // REFCORNER (0-3)
    pub transposed: bool,    // TRANSPOSED flag
    pub comb_op: u8,         // SBCOMBOP (0-4)
    pub refine_template: u8, // SBRTEMPLATE (0 or 1)
}

impl TextRegionParams {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + 2 + if self.refine { 1 } else { 0 });
        buf.write_u32::<BigEndian>(self.width).unwrap();
        buf.write_u32::<BigEndian>(self.height).unwrap();
        buf.write_u32::<BigEndian>(self.x).unwrap();
        buf.write_u32::<BigEndian>(self.y).unwrap();

        let mut sbrflags: u16 = 0;
        // SBHUFF is 0 for arithmetic coding
        if self.refine {
            sbrflags |= 1 << 1; // SBREFINE
        }
        sbrflags |= ((self.log_strips as u16) & 0x03) << 2; // LOGSBSTRIPS
        sbrflags |= ((self.ref_corner as u16) & 0x03) << 4; // REFCORNER
        if self.transposed {
            sbrflags |= 1 << 6; // TRANSPOSED
        }
        sbrflags |= ((self.comb_op as u16) & 0x03) << 7; // SBCOMBOP
        // SBDEFPIXEL is 0
        sbrflags |= ((self.ds_offset as u16) & 0x1F) << 10; // SBDSOFFSET
        if self.refine && self.refine_template == 1 {
            sbrflags |= 1 << 15; // SBRTEMPLATE
        }
        buf.write_u16::<BigEndian>(sbrflags).unwrap();

        if self.refine && self.refine_template == 1 {
            buf.write_u8(self.refine_template).unwrap();
        }
        buf
    }
}

// -----------------------------------------------------------------------------
// Segment header + payload writer (§7.2)
// -----------------------------------------------------------------------------

/// Represents a JBIG2 segment, including header and payload
#[derive(Default)]
pub struct Segment {
    pub number: u32,               // Segment number
    pub seg_type: SegmentType,     // Segment type
    pub deferred_non_retain: bool, // Bit 7 of Flags1: 0 = retain, 1 = non-retain
    pub retain_flags: u8,          // Up to 5 bits for retention flags
    pub page_association_type: u8, // Bits 0-1 of Flags2: 0=explicit, 1=deferred, 2=all pages
    pub referred_to: Vec<u32>,     // List of referred-to segment numbers
    pub page: Option<u32>,         // Page number if applicable
    pub payload: Vec<u8>,          // Segment data
}

fn encode_varint(mut v: u32, buf: &mut Vec<u8>) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

impl Segment {
    pub fn write_into<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<BigEndian>(self.number)?;

        let page_num_val = self.page.unwrap_or(0);
        let page_size_is_4_bytes = self.page_association_type <= 1 && page_num_val > 0xFF;
        let flags1 = (self.seg_type as u8 & 0x3F)
            | ((page_size_is_4_bytes as u8) << 6)
            | ((self.deferred_non_retain as u8) << 7);
        w.write_u8(flags1)?;

        let referred_to_count = self.referred_to.len();
        let mut referred_to_count_is_extended = false;
        let segment_count = if referred_to_count <= 7 {
            referred_to_count as u8
        } else {
            referred_to_count_is_extended = true;
            7
        };
        let flags2 = (self.page_association_type & 0x03)
            | ((self.retain_flags & 0x1F) << 2) // 5 bits for retain_flags
            | (segment_count << 5);
        w.write_u8(flags2)?;

        if referred_to_count_is_extended {
            let mut varint_buf = Vec::new();
            encode_varint(referred_to_count as u32, &mut varint_buf);
            w.write_all(&varint_buf)?;
        }

        let ref_num_size = if self.number <= 0xFF {
            1
        } else if self.number <= 0xFFFF {
            2
        } else {
            4
        };
        for &r_num in &self.referred_to {
            match ref_num_size {
                1 => w.write_u8(r_num as u8)?,
                2 => w.write_u16::<BigEndian>(r_num as u16)?,
                _ => w.write_u32::<BigEndian>(r_num)?,
            }
        }

        if self.page_association_type <= 1 {
            if let Some(p_num) = self.page {
                if page_size_is_4_bytes {
                    w.write_u32::<BigEndian>(p_num)?;
                } else {
                    w.write_u8(p_num as u8)?;
                }
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Page number required for explicit or deferred association",
                ));
            }
        }

        let payload_len = self.payload.len() as u32;
        debug!("Segment {} payload length: {}", self.number, payload_len);
        w.write_u32::<BigEndian>(payload_len)?;
        w.write_all(&self.payload)?;

        debug!(
            "Segment::write_into: Wrote segment {}: Type={:?}, Page={:?}, PA Type={}, Data Length={}",
            self.number, self.seg_type, self.page, self.page_association_type, payload_len
        );
        Ok(())
    }
}