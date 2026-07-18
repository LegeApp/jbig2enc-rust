//! Phase 5c: Huffman-coded symbol dictionaries and text regions.
//!
//! The encoder emits only arithmetic streams, so the Huffman paths are
//! exercised with the test-only writer (`tests/common/writer.rs`): a page made
//! of a Huffman symbol dictionary (SDHUFF=1, uncompressed collective bitmap)
//! plus a Huffman text region (SBHUFF=1) placing its symbols. Each page is
//! checked against jbig2dec (spec conformance) and, since we know exactly where
//! each glyph is drawn, against an expected bitmap built directly.

mod common;

use common::writer::{huffman_symbol_text_page, standalone_file, TestBitmap};
use jbig2enc_rust::decode::{decode_embedded, DecodeOptions};

/// A small distinct glyph of the given size.
fn glyph(w: u32, h: u32, seed: u32) -> TestBitmap {
    let mut bm = TestBitmap::new(w, h);
    let mut s = seed.wrapping_add(1);
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            // Ensure at least the border is patterned so glyphs are non-empty.
            if (s >> 16) & 1 == 1 || x == 0 || y == 0 {
                bm.set(x, y, true);
            }
        }
    }
    bm
}

/// Build the bitmap we expect: each placed glyph OR-composited at (s, 0),
/// TOPLEFT, on a blank page.
fn expected_page(
    w: u32,
    h: u32,
    symbols: &[TestBitmap],
    placements: &[(usize, i32)],
) -> TestBitmap {
    let mut page = TestBitmap::new(w, h);
    for &(sym, s) in placements {
        let g = &symbols[sym];
        for gy in 0..g.height {
            for gx in 0..g.width {
                if g.get(gx, gy) {
                    let px = s + gx as i32;
                    if px >= 0 && (px as u32) < w && gy < h {
                        page.set(px as u32, gy, true);
                    }
                }
            }
        }
    }
    page
}

fn opts() -> DecodeOptions {
    DecodeOptions::default()
}

fn check(w: u32, h: u32, symbols: &[TestBitmap], placements: &[(usize, i32)], label: &str) {
    let stream = huffman_symbol_text_page(w, h, symbols, placements);
    let expected = expected_page(w, h, symbols, placements);

    // Native decode must match the expected placement exactly.
    let got = decode_embedded(None, &stream, &opts())
        .unwrap_or_else(|e| panic!("{label}: native decode failed: {e}"));
    assert_eq!(got.width(), w, "{label}: width");
    assert_eq!(got.height(), h, "{label}: height");
    for y in 0..h {
        for x in 0..w {
            assert_eq!(
                got.get(x, y),
                expected.get(x, y),
                "{label}: native pixel ({x},{y})"
            );
        }
    }

    // jbig2dec must agree byte-for-byte with native (spec conformance).
    let file = standalone_file(&stream);
    if let Some(res) = common::oracle::decode_standalone(&file) {
        let pbm = res.unwrap_or_else(|e| panic!("{label}: jbig2dec failed: {e}"));
        assert_eq!(pbm.width, w, "{label}: oracle width");
        assert_eq!(pbm.height, h, "{label}: oracle height");
        for y in 0..h {
            for x in 0..w {
                assert_eq!(
                    pbm.get(x, y) != 0,
                    expected.get(x, y),
                    "{label}: oracle pixel ({x},{y})"
                );
            }
        }
    }
}

#[test]
fn two_glyphs_side_by_side() {
    // Widths must be non-decreasing (standard SDHUFFDW table B.2 codes DW >= 0).
    let symbols = vec![glyph(5, 8, 1), glyph(6, 8, 2)];
    let placements = [(0usize, 1i32), (1, 10)];
    check(24, 12, &symbols, &placements, "two-glyphs");
}

#[test]
fn several_glyphs_varied_widths() {
    // Symbols ordered by non-decreasing width within the single height class.
    let symbols = vec![glyph(3, 10, 3), glyph(4, 10, 4), glyph(7, 10, 5), glyph(9, 10, 6)];
    let placements = [(0usize, 0i32), (2, 6), (1, 16), (3, 22), (0, 34)];
    check(60, 14, &symbols, &placements, "varied-widths");
}

#[test]
fn repeated_symbol_ids() {
    // Same symbol placed several times exercises the symbol-ID Huffman table.
    let symbols = vec![glyph(5, 9, 7), glyph(5, 9, 8), glyph(5, 9, 9)];
    let placements = [(1usize, 2i32), (1, 9), (0, 16), (2, 23), (1, 30)];
    check(48, 12, &symbols, &placements, "repeated-ids");
}

#[test]
fn many_symbols_wide_id_codes() {
    // Enough symbols to force multi-bit symbol-ID codes (L = ceil(log2 N)).
    let symbols: Vec<TestBitmap> = (0..10).map(|i| glyph(4, 8, 100 + i)).collect();
    let placements: Vec<(usize, i32)> =
        (0..10).map(|i| (i as usize, (i as i32) * 6)).collect();
    check(80, 12, &symbols, &placements, "many-symbols");
}
