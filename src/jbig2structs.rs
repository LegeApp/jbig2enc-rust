/// Pruned Rust equivalents of JBIG2 structs and segment headers
use byteorder::{BigEndian, WriteBytesExt};
use std::io::{self, Write};

#[cfg(feature = "trace_encoder")]
use tracing::debug;

#[cfg(not(feature = "trace_encoder"))]
#[macro_use]
mod trace_stubs {
    macro_rules! debug {
        ($($arg:tt)*) => { std::convert::identity(format_args!($($arg)*)) };
    }
}

#[cfg(not(feature = "trace_encoder"))]
use trace_stubs::*;

/// JBIG2 file format magic number
pub const JB2_MAGIC: &[u8; 10] = b"\x97JBIG2\r\n\x1A\n";

/// JBIG2 file format version
pub const JB2_VERSION: u8 = 0x02;

/// JBIG2 segment types
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
    Extension = 62,
}

/// JBIG2 compression type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None = 0,
    MMR = 1,
    Arithmetic = 2,
    JB2 = 3,
}

/// JBIG2 refinement template
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementTemplate {
    TPL1 = 0,
    TPL2 = 1,
}

/// JBIG2 symbol dictionary flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolDictFlags {
    pub sd_huff: bool,
    pub sd_ref_agg: bool,
    pub sd_huff_dh: u8,
    pub sd_huff_dw: u8,
    pub sd_huff_bm_size: u8,
    pub sd_huff_agg_inst: u8,
    pub sd_ex_sym: bool,
    pub sd_agg_inst_only: bool,
}

impl Default for SymbolDictFlags {
    fn default() -> Self {
        Self {
            sd_huff: false,
            sd_ref_agg: false,
            sd_huff_dh: 0,
            sd_huff_dw: 0,
            sd_huff_bm_size: 0,
            sd_huff_agg_inst: 0,
            sd_ex_sym: false,
            sd_agg_inst_only: false,
        }
    }
}

/// JBIG2 text region flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextRegionFlags {
    pub sb_huff: bool,
    pub sb_refine: bool,
    pub log_sb_stripes: u8,
    pub sb_refine_template: bool,
    pub sb_def_pixel: bool,
    pub sb_ds_offset: i32,
    pub sb_refine_at: bool,
    pub sb_transposed: bool,
    pub sb_combo_op: u8,
    pub sb_huff_dw: u8,
    pub sb_huff_dh: u8,
    pub sb_huff_bm_size: u8,
    pub sb_huff_inst_width: u8,
    pub sb_huff_inst: u8,
}

impl Default for TextRegionFlags {
    fn default() -> Self {
        Self {
            sb_huff: false,
            sb_refine: false,
            log_sb_stripes: 0,
            sb_refine_template: false,
            sb_def_pixel: false,
            sb_ds_offset: 0,
            sb_refine_at: false,
            sb_transposed: false,
            sb_combo_op: 0,
            sb_huff_dw: 0,
            sb_huff_dh: 0,
            sb_huff_bm_size: 0,
            sb_huff_inst_width: 0,
            sb_huff_inst: 0,
        }
    }
}

/// JBIG2 generic region flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenericRegionFlags {
    pub gr_template: u8,
    pub gr_refine: bool,
    pub gr_at_x: [i8; 4],
    pub gr_at_y: [i8; 4],
}

impl Default for GenericRegionFlags {
    fn default() -> Self {
        Self {
            gr_template: 0,
            gr_refine: false,
            gr_at_x: [0, 0, 0, 0],
            gr_at_y: [0, 0, 0, 0],
        }
    }
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
pub struct FileHeader {
    pub organisation_type: bool, // 1 bit
    pub unknown_n_pages: bool,   // 1 bit
    pub n_pages: u32,            // big-endian
}

impl FileHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        // 8-byte JBIG2 identifier (ITU-T T.88, §D.4.1)
        // 0x97 0x4A 0x42 0x32 0x0D 0x0A 0x1A 0x0A
        const MAGIC: &[u8] = b"\x97JB2\r\n\x1A\n";

        let mut buf = Vec::with_capacity(8 + 1 + 4);
        buf.extend_from_slice(MAGIC);

        let mut flags = 0u8;
        if self.organisation_type {
            flags |= 0x01;
        }
        if self.unknown_n_pages {
            flags |= 0x02;
        }
        buf.push(flags);
        buf.write_u32::<BigEndian>(self.n_pages).unwrap();
        buf
    }
}

// -----------------------------------------------------------------------------
// Page information segment payload
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct PageInfo {
    pub width: u32,                 // big-endian
    pub height: u32,                // big-endian
    pub xres: u32,                  // big-endian
    pub yres: u32,                  // big-endian
    pub is_lossless: bool,          // bit0
    pub contains_refinements: bool, // bit1
    pub default_pixel: bool,        // bit2
    pub default_operator: u8,       // bits3-4
    pub aux_buffers: bool,          // bit5
    pub operator_override: bool,    // bit6
    pub reserved: bool,             // bit7 (must be zero)
    pub segment_flags: u16,         // big-endian (often 0)
}

impl Default for PageInfo {
    fn default() -> Self {
        PageInfo {
            width: 0,
            height: 0,
            xres: 300, // Default 300 DPI
            yres: 300, // Default 300 DPI
            is_lossless: false,
            contains_refinements: false,
            default_pixel: false,
            default_operator: 0,
            aux_buffers: false,
            operator_override: false,
            reserved: false,
            segment_flags: 0,
        }
    }
}

impl PageInfo {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 * 4 + 1 + 2);
        buf.write_u32::<BigEndian>(self.width).unwrap();
        buf.write_u32::<BigEndian>(self.height).unwrap();
        buf.write_u32::<BigEndian>(self.xres).unwrap();
        buf.write_u32::<BigEndian>(self.yres).unwrap();
        // pack 8 flags into one byte
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
        // bit7 reserved = 0
        buf.push(b);
        buf.write_u16::<BigEndian>(self.segment_flags).unwrap();
        buf
    }
}

// -----------------------------------------------------------------------------
// Symbol dictionary parameters (pruned)
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct SymbolDictParams {
    // Only core flags for text-only; all other refinement flags omitted
    pub sd_template: u8, // 0..3
    // Adaptive template coords (usually zero for default)
    pub a1x: i8,
    pub a1y: i8,
    pub a2x: i8,
    pub a2y: i8,
    pub a3x: i8,
    pub a3y: i8,
    pub a4x: i8,
    pub a4y: i8,
    pub exsyms: u32,  // big-endian
    pub newsyms: u32, // big-endian
}

impl SymbolDictParams {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(2 + 8 + 4 + 4);
        // pack sd_template (2 bits) into low bits of a byte; rest reserved
        let b = self.sd_template & 0x03;
        buf.push(b);
        buf.push(0); // reserved flags
        buf.push(self.a1x as u8);
        buf.push(self.a1y as u8);
        buf.push(self.a2x as u8);
        buf.push(self.a2y as u8);
        buf.push(self.a3x as u8);
        buf.push(self.a3y as u8);
        buf.push(self.a4x as u8);
        buf.push(self.a4y as u8);
        buf.write_u32::<BigEndian>(self.exsyms).unwrap();
        buf.write_u32::<BigEndian>(self.newsyms).unwrap();
        buf
    }
}

// -----------------------------------------------------------------------------
// Text region parameters (pruned for immediate text regions)
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct TextRegionParams {
    pub width: u32,          // big-endian
    pub height: u32,         // big-endian
    pub x: u32,              // big-endian
    pub y: u32,              // big-endian
    pub ds_offset: u8,       // SBDSOFFSET (signed 5 bits, effectively 0-31 for positive offset)
    pub refine: bool,        // SBREFINE flag (bit 1)
    pub log_strips: u8,      // LOGSBSTRIPS (bits 2-3)
    pub ref_corner: u8,      // REFCORNER (bits 4-5)
    pub transposed: bool,    // TRANSPOSED flag (bit 6)
    pub comb_op: u8,         // SBCOMBOP (bits 7-8)
    pub refine_template: u8, // SBRTEMPLATE (bit 15) and GRTEMPLATE value (0 or 1)
}

impl TextRegionParams {
    pub fn to_bytes(&self) -> Vec<u8> {
        // Capacity: 16 (W,H,X,Y) + 2 (sbrflags) + 1 (optional GRTEMPLATE byte)
        let mut buf = Vec::with_capacity(16 + 2 + if self.refine { 1 } else { 0 });
        buf.write_u32::<BigEndian>(self.width).unwrap();
        buf.write_u32::<BigEndian>(self.height).unwrap();
        buf.write_u32::<BigEndian>(self.x).unwrap();
        buf.write_u32::<BigEndian>(self.y).unwrap();

        // Assemble the 16-bit SBRFLAGS field (Figure 36 from ISO/IEC 14492:2001)
        let _sbhuff_flag: u16 = 0; // Bit 0: SBHUFF (0 for arithmetic coding)
        let _sbdefpixel_flag: u16 = 0; // Bit 9: SBDEFPIXEL (0 for default pixel value)

        let mut sbrflags: u16 = 0;
        // sbhuff_flag is 0 (Bit 0: SBHUFF = 0 for arithmetic coding)
        // sbdefpixel_flag is 0 (Bit 9: SBDEFPIXEL = 0 for default pixel value)

        // Bits 2-3: LOGSBSTRIPS
        sbrflags |= (self.log_strips as u16 & 0x03) << 2;
        // Bits 4-5: REFCORNER
        sbrflags |= (self.ref_corner as u16 & 0x03) << 4;
        // Bit 6: TRANSPOSED
        sbrflags |= (self.transposed as u16) << 6;
        // Bits 7-8: SBCOMBOP
        sbrflags |= (self.comb_op as u16 & 0x03) << 7;
        // Bits 10-14: SBDSOFFSET
        sbrflags |= (self.ds_offset as u16 & 0x1F) << 10;

        let mut sbrtemplate_bit_value: u16 = 0; // Default: SBRTEMPLATE=0 (GRTEMPLATE byte not present, template 0 implied)

        if self.refine {
            sbrflags |= 1u16 << 1; // Bit 1: SBREFINE = 1

            // SBRTEMPLATE (Bit 15) is 1 if GRTEMPLATE byte is present (i.e., self.refine_template == 1).
            // Otherwise, SBRTEMPLATE is 0 (GRTEMPLATE byte not present, template 0 used implicitly).
            if self.refine_template == 1 {
                sbrtemplate_bit_value = 1;
            }
            sbrflags |= sbrtemplate_bit_value << 15; // Set SBRTEMPLATE bit
        }
        // If not self.refine, SBREFINE (bit 1) is 0, and SBRTEMPLATE (bit 15) is 0 via sbrtemplate_bit_value default.

        buf.write_u16::<BigEndian>(sbrflags).unwrap();

        // Conditionally write the GRTEMPLATE byte.
        // This byte is present if SBREFINE is 1 (self.refine == true)
        // AND SBRTEMPLATE bit in SBRFLAGS is 1 (sbrtemplate_bit_value == 1).
        // (SBHUFF is 0 for arithmetic coding, which is a precondition for GRTEMPLATE byte).
        if self.refine && sbrtemplate_bit_value == 1 {
            // The value of the GRTEMPLATE byte is self.refine_template (which must be 1 here).
            buf.write_u8(self.refine_template).unwrap();
        }
        buf
    }
}

// -----------------------------------------------------------------------------
// Segment header + payload writer (pruned)
// -----------------------------------------------------------------------------
#[derive(Default)]
pub struct Segment {
    pub number: u32,
    pub seg_type: SegmentType,
    pub deferred_non_retain: bool, // Bit 7 of Flags1: 0 = retain, 1 = non-retain (if deferred)
    pub retain_flags: u8, // Bits 2-4 of Flags2: Referred-to segment retention flags (3 bits)
    pub page_association_type: u8, // Bits 0-1 of Flags2: 00=explicit, 01=deferred, 10=all pages
    pub referred_to: Vec<u32>, // List of referred-to segment numbers
    pub page: Option<u32>, // Page number if page_association_type is explicit (00) or deferred (01)
    pub payload: Vec<u8>, // Segment data
}

fn encode_varint(mut v: u32, buf: &mut Vec<u8>) {
    // 7-bit little-endian continuation format
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

impl Segment {
    pub fn write_into<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        // Segment number (DWORD - 4 bytes, big-endian)
        w.write_u32::<BigEndian>(self.number)?;

        // --- Segment header flags (2 bytes) ---
        // Flags Byte 1 (Flags1)
        let page_num_val = self.page.unwrap_or(0); // Use 0 if page is None, though it might not be written
        let page_size_is_4_bytes_bit = if self.page_association_type <= 1 && page_num_val > 0xFF {
            1
        } else {
            0
        }; // 1 if 4-byte page #, 0 if 1-byte. Only relevant if PA type is explicit or deferred.

        let flags1 = (self.seg_type as u8 & 0x3F) // Bits 0-5: Segment type
                   | (page_size_is_4_bytes_bit << 6)    // Bit 6: Page association size (0 for 1-byte, 1 for 4-byte page #)
                   | ((self.deferred_non_retain as u8) << 7); // Bit 7: Deferred non-retain
        w.write_u8(flags1)?;

        // Flags Byte 2 (Flags2)
        let referred_to_count = self.referred_to.len();
        let mut referred_to_count_is_extended = false;
        let flags2 = if referred_to_count > 4 {
            referred_to_count_is_extended = true;
            (self.page_association_type & 0x03) | ((self.retain_flags & 0x07) << 2) | (0b111 << 5)
        } else {
            (self.page_association_type & 0x03)
                | ((self.retain_flags & 0x07) << 2)
                | ((referred_to_count as u8) << 5)
        };
        w.write_u8(flags2)?;

        // Referred-to segment count (if extended)
        if referred_to_count_is_extended {
            let mut varint_buf = Vec::new();
            encode_varint(referred_to_count as u32, &mut varint_buf); // encode_varint should match spec 7.3.4
            w.write_all(&varint_buf)?;
        }

        // Referred-to segment numbers
        // Size depends on current segment's number (self.number) - Spec 7.3.5
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

        // Page number (if page_association_type is explicit or deferred)
        if self.page_association_type <= 1 {
            // 00 (explicit) or 01 (deferred)
            if let Some(p_num) = self.page {
                if page_size_is_4_bytes_bit == 1 {
                    // 4-byte page number
                    w.write_u32::<BigEndian>(p_num)?;
                } else {
                    // 1-byte page number
                    w.write_u8(p_num as u8)?;
                }
            }
            // Else: page is None but type is explicit/deferred. This is an invalid state.
            // Consider adding error handling or assertion if self.page is None here.
        }

        // Segment data length (DWORD - 4 bytes, big-endian)
        let payload_len = self.payload.len() as u32;
        debug!("Segment {} payload length: {}", self.number, payload_len);
        w.write_u32::<BigEndian>(payload_len)?;

        // Segment data
        w.write_all(&self.payload)?;

        debug!(
            "Segment::write_into: Wrote segment {}: Type={:?}, Page={:?}, PA Type={}, Data Length={}", 
            self.number, self.seg_type, self.page, self.page_association_type, payload_len
        );
        Ok(())
    }
}
