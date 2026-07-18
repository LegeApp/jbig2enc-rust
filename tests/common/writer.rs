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

use jbig2enc_rust::decode::huffman::HuffmanTable;
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

// ------------------------------------------------------------------------
// Huffman writer (Phase 5c oracle support)
// ------------------------------------------------------------------------

use jbig2enc_rust::decode::huffman::standard_table;

/// MSB-first bit writer, the mirror of `decode::huffman::BitReader`.
#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write_bit(&mut self, b: u32) {
        self.cur = (self.cur << 1) | ((b & 1) as u8);
        self.nbits += 1;
        if self.nbits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    pub fn write_bits(&mut self, value: u64, n: u8) {
        for i in (0..n).rev() {
            self.write_bit(((value >> i) & 1) as u32);
        }
    }

    pub fn align(&mut self) {
        while self.nbits != 0 {
            self.write_bit(0);
        }
    }

    pub fn into_bytes(mut self) -> Vec<u8> {
        self.align();
        self.bytes
    }
}

fn emit_value(w: &mut BitWriter, table: &HuffmanTable, value: i32) {
    let (code, len, offset, range_len) = table
        .encode_value(value)
        .unwrap_or_else(|| panic!("no Huffman line covers value {value}"));
    w.write_bits(code as u64, len);
    if range_len > 0 {
        w.write_bits(offset, range_len);
    }
}

fn emit_oob(w: &mut BitWriter, table: &HuffmanTable) {
    let (code, len) = table.encode_oob().expect("table has no OOB line");
    w.write_bits(code as u64, len);
}

fn log2_ceil(v: usize) -> u8 {
    if v <= 1 {
        return 0;
    }
    let mut l = 0u8;
    let mut n = 1usize;
    while n < v {
        n <<= 1;
        l += 1;
    }
    l
}

/// Build a Huffman symbol-dictionary segment payload (SDHUFF=1, SDREFAGG=0).
/// All symbols must share one height (one height class); the collective bitmap
/// is emitted uncompressed (BMSIZE=0). Every symbol is exported.
pub fn huffman_symbol_dict_payload(symbols: &[TestBitmap]) -> Vec<u8> {
    assert!(!symbols.is_empty());
    let height = symbols[0].height;
    assert!(symbols.iter().all(|s| s.height == height), "one height class");

    let dh = standard_table(4).unwrap(); // SDHUFFDH
    let dw = standard_table(2).unwrap(); // SDHUFFDW (has OOB)
    let bmsize = standard_table(1).unwrap(); // SDHUFFBMSIZE
    let export = standard_table(1).unwrap(); // §6.5.10 export runs

    let mut w = BitWriter::new();
    // Height class delta height (from HCHEIGHT=0).
    emit_value(&mut w, &dh, height as i32);
    // Delta widths, then OOB.
    let mut prev = 0i32;
    for s in symbols {
        emit_value(&mut w, &dw, s.width as i32 - prev);
        prev = s.width as i32;
    }
    emit_oob(&mut w, &dw);
    // BMSIZE = 0 (uncompressed), then byte-align.
    emit_value(&mut w, &bmsize, 0);
    w.align();
    // Uncompressed collective bitmap: `height` rows of TOTWIDTH pixels, each row
    // byte-padded. Symbols are concatenated left to right.
    for y in 0..height {
        for s in symbols {
            for x in 0..s.width {
                w.write_bit(s.get(x, y) as u32);
            }
        }
        w.align();
    }
    // Export flags: run 0 (not exported, length 0), run N (exported).
    emit_value(&mut w, &export, 0);
    emit_value(&mut w, &export, symbols.len() as i32);
    let data = w.into_bytes();

    let mut payload = Vec::new();
    payload.extend_from_slice(&0x0001u16.to_be_bytes()); // flags: SDHUFF=1, all selections standard
    payload.extend_from_slice(&(symbols.len() as u32).to_be_bytes()); // SDNUMEXSYMS
    payload.extend_from_slice(&(symbols.len() as u32).to_be_bytes()); // SDNUMNEWSYMS
    payload.extend_from_slice(&data);
    payload
}

/// Build a Huffman text-region segment payload (SBHUFF=1, SBREFINE=0,
/// TRANSPOSED=0, TOPLEFT, SBSTRIPS=1). `placements` are `(symbol_index, s, t)`
/// with all instances in a single strip at `t = 0`, ordered by increasing S.
pub fn huffman_text_region_payload(
    width: u32,
    height: u32,
    num_symbols: usize,
    symbol_widths: &[u32],
    placements: &[(usize, i32)],
) -> Vec<u8> {
    let fs = standard_table(6).unwrap();
    let ds = standard_table(8).unwrap(); // has OOB
    let dt = standard_table(11).unwrap();

    let l = log2_ceil(num_symbols).max(1);
    // Symbol-ID Huffman table: all symbols get code length `l`.
    let mut runcode_lengths = [0u32; 35];
    runcode_lengths[0] = 1;
    runcode_lengths[l as usize] = 1;
    let runcode_table = HuffmanTable::from_code_lengths(&runcode_lengths).unwrap();
    let sym_lengths = vec![l as u32; num_symbols];
    let sym_table = HuffmanTable::from_code_lengths(&sym_lengths).unwrap();

    let mut w = BitWriter::new();
    // §7.4.3.1.7: 35 four-bit runcode lengths.
    for len in runcode_lengths.iter() {
        w.write_bits(*len as u64, 4);
    }
    // One RUNCODE per symbol, each emitting code length `l`.
    let (rc_code, rc_len, _, _) = runcode_table.encode_value(l as i32).unwrap();
    for _ in 0..num_symbols {
        w.write_bits(rc_code as u64, rc_len);
    }
    w.align();

    // Strip data (SBSTRIPS=1 → no T bits; single strip at T=0). Table B.11
    // (SBHUFFDT) codes values >= 1, so use DT0 = 1 and strip DT = 1: initial
    // STRIPT = -(1) then += 1, giving strip T = 0.
    emit_value(&mut w, &dt, 1); // initial STRIPT: DT0 = 1
    emit_value(&mut w, &dt, 1); // strip delta T = 1 -> strip T becomes 0
    // First S.
    let (first_sym, first_s) = placements[0];
    emit_value(&mut w, &fs, first_s);
    emit_symbol_id(&mut w, &sym_table, first_sym);
    let mut cur_s = first_s + symbol_widths[first_sym] as i32 - 1;
    // Subsequent instances via IDS.
    for &(sym, s) in &placements[1..] {
        let ids = s - cur_s; // SBDSOFFSET = 0
        emit_value(&mut w, &ds, ids);
        emit_symbol_id(&mut w, &sym_table, sym);
        cur_s = s + symbol_widths[sym] as i32 - 1;
    }
    emit_oob(&mut w, &ds); // end of strip
    let data = w.into_bytes();

    let mut payload = Vec::new();
    // §7.4.1 region info.
    payload.extend_from_slice(&width.to_be_bytes());
    payload.extend_from_slice(&height.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes()); // x
    payload.extend_from_slice(&0u32.to_be_bytes()); // y
    payload.push(0); // region flags: OR
    // §7.4.3.1.1 text flags: SBHUFF=1 (bit0), REFCORNER=TOPLEFT=1 (bits4-5=01).
    let text_flags: u16 = 0x0001 | (1 << 4);
    payload.extend_from_slice(&text_flags.to_be_bytes());
    // §7.4.3.1.2 Huffman flags: all standard (0).
    payload.extend_from_slice(&0u16.to_be_bytes());
    // §7.4.3.1.4 SBNUMINSTANCES.
    payload.extend_from_slice(&(placements.len() as u32).to_be_bytes());
    payload.extend_from_slice(&data);
    payload
}

fn emit_symbol_id(w: &mut BitWriter, sym_table: &HuffmanTable, id: usize) {
    let (code, len, _, _) = sym_table
        .encode_value(id as i32)
        .unwrap_or_else(|| panic!("no symbol-ID code for {id}"));
    w.write_bits(code as u64, len);
}

/// Assemble a page with a Huffman symbol dictionary (segment 1) and a Huffman
/// text region (segment 2) referring to it, plus page info and end-of-page.
pub fn huffman_symbol_text_page(
    page_w: u32,
    page_h: u32,
    symbols: &[TestBitmap],
    placements: &[(usize, i32)],
) -> Vec<u8> {
    let mut stream = Vec::new();
    // Segment 0: page info.
    let page_data = page_info_payload(page_w, page_h);
    stream.extend_from_slice(&segment_header(0, 48, &[], 1, page_data.len() as u32));
    stream.extend_from_slice(&page_data);
    // Segment 1: Huffman symbol dictionary (type 0).
    let dict = huffman_symbol_dict_payload(symbols);
    stream.extend_from_slice(&segment_header(1, 0, &[], 1, dict.len() as u32));
    stream.extend_from_slice(&dict);
    // Segment 2: Huffman text region (type 6, immediate) referring to segment 1.
    let widths: Vec<u32> = symbols.iter().map(|s| s.width).collect();
    let region = huffman_text_region_payload(page_w, page_h, symbols.len(), &widths, placements);
    stream.extend_from_slice(&segment_header(2, 6, &[1], 1, region.len() as u32));
    stream.extend_from_slice(&region);
    // Segment 3: end of page.
    stream.extend_from_slice(&segment_header(3, 49, &[], 1, 0));
    stream
}
