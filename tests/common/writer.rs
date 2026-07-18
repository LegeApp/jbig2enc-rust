//! Test-only JBIG2 stream writer (Phase 5, jbig2dec-phase5-plan.md §5a/§5b).
//!
//! This crate's *encoder* only ever emits generic-region template 0 with no
//! TPGDON, so a self-round-trip cannot cover templates 1–3, TPGDON, or other
//! "wild" forms found in real PDFs. This helper builds minimal valid page
//! streams for those forms by driving the encoder's MQ arithmetic coder
//! (`Jbig2ArithCoder::encode_bit`) directly, with context layouts that mirror
//! `src/decode/generic.rs`.
//!
//! Because the MQ context *numbering* is a pure relabelling of the same pixel
//! partition, a stream produced with the decoder's layout is byte-identical to
//! one produced with jbig2dec's canonical layout — so the same bytes decode
//! correctly under both. The jbig2dec oracle is what actually validates spec
//! conformance; self-round-trip only checks internal consistency.

#![allow(dead_code)]

use jbig2enc_rust::Jbig2ArithCoder;

/// A simple mutable test bitmap (row-major bool grid, `false` = white).
#[derive(Clone)]
pub struct TestBitmap {
    pub width: u32,
    pub height: u32,
    px: Vec<bool>,
}

impl TestBitmap {
    pub fn new(width: u32, height: u32) -> Self {
        TestBitmap {
            width,
            height,
            px: vec![false; (width as usize) * (height as usize)],
        }
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, v: bool) {
        if x < self.width && y < self.height {
            self.px[(y as usize) * (self.width as usize) + x as usize] = v;
        }
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> bool {
        if x < self.width && y < self.height {
            self.px[(y as usize) * (self.width as usize) + x as usize]
        } else {
            false
        }
    }

    /// Causal sample at `(x+dx, y+dy)` as a `u32` bit, matching the decoder's
    /// convention: previous rows are read directly; the current row is only
    /// visible for `x' < x`; anything else (future / far rows) reads as 0.
    #[inline]
    fn causal(&self, x: i64, y: i64, dx: i64, dy: i64) -> u32 {
        let xx = x + dx;
        let yy = y + dy;
        match dy {
            0 => {
                if xx < 0 || xx >= x {
                    0
                } else {
                    self.get(xx as u32, yy as u32) as u32
                }
            }
            -1 | -2 => {
                if xx < 0 || yy < 0 {
                    0
                } else {
                    self.get(xx as u32, yy as u32) as u32
                }
            }
            _ => 0,
        }
    }

    fn row_equals_prev(&self, y: u32) -> bool {
        // Compare row y to row y-1 (all-zero above row 0), matching the TPGDON
        // duplicate-row test the decoder applies.
        for x in 0..self.width {
            let above = if y >= 1 { self.get(x, y - 1) } else { false };
            if self.get(x, y) != above {
                return false;
            }
        }
        true
    }
}

/// SLTP contexts, mirroring `src/decode/generic.rs`.
const SLTP_CTX_T0: usize = 0xB325;
const SLTP_CTX_T1: usize = 0x0795;
const SLTP_CTX_T2: usize = 0x00E5;
const SLTP_CTX_T3: usize = 0x0195;

/// Compute the generic-region context index for the pixel at `(x, y)` under the
/// given template, mirroring the decoder's per-template bit layout exactly.
fn context(template: u8, bm: &TestBitmap, x: i64, y: i64, at: &[(i8, i8); 4]) -> usize {
    let (a1x, a1y) = (at[0].0 as i64, at[0].1 as i64);
    let (a2x, a2y) = (at[1].0 as i64, at[1].1 as i64);
    let (a3x, a3y) = (at[2].0 as i64, at[2].1 as i64);
    let (a4x, a4y) = (at[3].0 as i64, at[3].1 as i64);
    let p = |dx: i64, dy: i64| bm.causal(x, y, dx, dy);
    let t = match template {
        0 => {
            (p(a4x, a4y) << 15)
                | (p(-1, -2) << 14)
                | (p(0, -2) << 13)
                | (p(1, -2) << 12)
                | (p(a3x, a3y) << 11)
                | (p(a2x, a2y) << 10)
                | (p(-2, -1) << 9)
                | (p(-1, -1) << 8)
                | (p(0, -1) << 7)
                | (p(1, -1) << 6)
                | (p(2, -1) << 5)
                | (p(a1x, a1y) << 4)
                | (p(-4, 0) << 3)
                | (p(-3, 0) << 2)
                | (p(-2, 0) << 1)
                | p(-1, 0)
        }
        1 => {
            p(-1, 0)
                | (p(-2, 0) << 1)
                | (p(-3, 0) << 2)
                | (p(a1x, a1y) << 3)
                | (p(2, -1) << 4)
                | (p(1, -1) << 5)
                | (p(0, -1) << 6)
                | (p(-1, -1) << 7)
                | (p(-2, -1) << 8)
                | (p(2, -2) << 9)
                | (p(1, -2) << 10)
                | (p(0, -2) << 11)
                | (p(-1, -2) << 12)
        }
        2 => {
            p(-1, 0)
                | (p(-2, 0) << 1)
                | (p(a1x, a1y) << 2)
                | (p(1, -1) << 3)
                | (p(0, -1) << 4)
                | (p(-1, -1) << 5)
                | (p(-2, -1) << 6)
                | (p(1, -2) << 7)
                | (p(0, -2) << 8)
                | (p(-1, -2) << 9)
        }
        _ => {
            p(-1, 0)
                | (p(-2, 0) << 1)
                | (p(-3, 0) << 2)
                | (p(-4, 0) << 3)
                | (p(a1x, a1y) << 4)
                | (p(1, -1) << 5)
                | (p(0, -1) << 6)
                | (p(-1, -1) << 7)
                | (p(-2, -1) << 8)
                | (p(-3, -1) << 9)
        }
    };
    t as usize
}

fn sltp_ctx(template: u8) -> usize {
    match template {
        0 => SLTP_CTX_T0,
        1 => SLTP_CTX_T1,
        2 => SLTP_CTX_T2,
        _ => SLTP_CTX_T3,
    }
}

/// Nominal AT for a template, expanded to 4 slots (only the leading slots are
/// used by templates 1–3).
pub fn nominal_at(template: u8) -> [(i8, i8); 4] {
    match template {
        0 => [(3, -1), (-3, -1), (2, -2), (-2, -2)],
        1 => [(3, -1), (0, 0), (0, 0), (0, 0)],
        _ => [(2, -1), (0, 0), (0, 0), (0, 0)],
    }
}

/// Encode the arithmetic data for a generic region: the pixels of `bm` under
/// `template`/`at`, optionally with TPGDON typical prediction. Returns the raw
/// MQ byte stream (terminated with the standard marker).
pub fn generic_arith_data(
    bm: &TestBitmap,
    template: u8,
    at: &[(i8, i8); 4],
    tpgdon: bool,
) -> Vec<u8> {
    let mut coder = Jbig2ArithCoder::new();
    let mut ltp = false;
    for y in 0..bm.height {
        if tpgdon {
            let dup = bm.row_equals_prev(y);
            let sltp = dup ^ ltp;
            coder.encode_bit(sltp_ctx(template), sltp);
            ltp = dup;
            if dup {
                continue;
            }
        }
        for x in 0..bm.width {
            let ctx = context(template, bm, x as i64, y as i64, at);
            coder.encode_bit(ctx, bm.get(x, y));
        }
    }
    coder.flush(true);
    coder.into_vec()
}

/// Build a generic-region segment payload (T.88 §7.4.6): region info + generic
/// flags + AT pixels + arithmetic data.
pub fn generic_region_payload(
    bm: &TestBitmap,
    template: u8,
    at: &[(i8, i8); 4],
    tpgdon: bool,
    comb_operator: u8,
) -> Vec<u8> {
    let mut v = Vec::new();
    // §7.4.1 region segment information field.
    v.extend_from_slice(&bm.width.to_be_bytes());
    v.extend_from_slice(&bm.height.to_be_bytes());
    v.extend_from_slice(&0u32.to_be_bytes()); // x
    v.extend_from_slice(&0u32.to_be_bytes()); // y
    v.push(comb_operator & 0x07);
    // §7.4.6.2 generic region flags: bit0 MMR, bits1-2 template, bit3 TPGDON.
    let flags = ((template & 0x03) << 1) | ((tpgdon as u8) << 3);
    v.push(flags);
    // AT pixels: template 0 has 4, templates 1–3 have 1.
    let at_count = if template == 0 { 4 } else { 1 };
    for &(ax, ay) in at.iter().take(at_count) {
        v.push(ax as u8);
        v.push(ay as u8);
    }
    v.extend_from_slice(&generic_arith_data(bm, template, at, tpgdon));
    v
}

/// Build a page-information segment payload (T.88 §7.4.8), 19 bytes.
pub fn page_info_payload(width: u32, height: u32) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&width.to_be_bytes());
    v.extend_from_slice(&height.to_be_bytes());
    v.extend_from_slice(&0u32.to_be_bytes()); // x resolution
    v.extend_from_slice(&0u32.to_be_bytes()); // y resolution
    v.push(0x00); // page flags: lossy, default pixel 0, OR combination
    v.extend_from_slice(&0u16.to_be_bytes()); // striping information (not striped)
    v
}

/// Emit a short-form segment header (T.88 §7.2) followed by nothing — the
/// caller appends the data. `referred` are 1-byte referred numbers (segment
/// numbers must be <= 256 for that to be valid, which all our test streams are).
fn segment_header(number: u32, type_code: u8, referred: &[u32], page: u8, data_len: u32) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&number.to_be_bytes());
    v.push(type_code & 0x3F);
    v.push((referred.len() as u8) << 5);
    for &r in referred {
        v.push(r as u8);
    }
    v.push(page);
    v.extend_from_slice(&data_len.to_be_bytes());
    v
}

/// A minimal embedded page stream (no file header) that jbig2dec `-e` and the
/// native `decode_embedded` both accept: page-info segment, one immediate
/// generic region, end-of-page.
pub fn single_generic_page(bm: &TestBitmap, template: u8, at: &[(i8, i8); 4], tpgdon: bool) -> Vec<u8> {
    let mut stream = Vec::new();

    // Segment 0: page information (type 48).
    let page_data = page_info_payload(bm.width, bm.height);
    stream.extend_from_slice(&segment_header(0, 48, &[], 1, page_data.len() as u32));
    stream.extend_from_slice(&page_data);

    // Segment 1: immediate generic region (type 38).
    let region = generic_region_payload(bm, template, at, tpgdon, 0);
    stream.extend_from_slice(&segment_header(1, 38, &[], 1, region.len() as u32));
    stream.extend_from_slice(&region);

    // Segment 2: end of page (type 49), no data.
    stream.extend_from_slice(&segment_header(2, 49, &[], 1, 0));

    stream
}

/// Wrap an embedded page stream in a standalone sequential file (T.88 Annex D):
/// file magic + flags (sequential, unknown page count) + segments.
pub fn standalone_file(embedded: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&[0x97, 0x4A, 0x42, 0x32, 0x0D, 0x0A, 0x1A, 0x0A]);
    v.push(0x03); // bit0 sequential, bit1 unknown page count
    v.extend_from_slice(embedded);
    v
}
