//! Structured-chaos robustness test (jbig2dec-gaps-plan.md Gap E, Phase 1).
//!
//! Takes valid encoder streams and mutates them deterministically, byte by
//! byte, asserting the decoder never panics — it must always return a bounded
//! result or a typed error. This is the stable-toolchain phase gate that stands
//! in for `cargo fuzz` (which needs nightly).

use jbig2enc_rust::decode::{DecodeOptions, decode_embedded, decode_file};
use jbig2enc_rust::{encode_single_image, encode_single_image_lossless};

fn seed_streams() -> Vec<Vec<u8>> {
    let w = 24u32;
    let h = 17u32;
    let mut px = vec![0u8; (w * h) as usize];
    let mut s = 0xABCDu32;
    for p in px.iter_mut() {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        *p = ((s >> 16) & 1) as u8;
    }
    vec![
        encode_single_image(&px, w, h, false).unwrap().page_data,
        encode_single_image(&px, w, h, true).unwrap().page_data,
        encode_single_image_lossless(&px, w, h, false).unwrap().page_data,
    ]
}

#[test]
fn single_byte_mutations_never_panic() {
    let opts = DecodeOptions::default();
    for stream in seed_streams() {
        // Every byte position × a spread of replacement values.
        for pos in 0..stream.len() {
            for &val in &[0x00u8, 0x01, 0x7F, 0x80, 0xFF, 0x55, 0xAA] {
                let mut m = stream.clone();
                m[pos] = val;
                // Both entry points; ignore Ok/Err, only care about no panic.
                let _ = decode_file(&m, &opts);
                let _ = decode_embedded(None, &m, &opts);
            }
        }
    }
}

#[test]
fn truncations_never_panic() {
    let opts = DecodeOptions::default();
    for stream in seed_streams() {
        for cut in 0..=stream.len() {
            let _ = decode_file(&stream[..cut], &opts);
            let _ = decode_embedded(None, &stream[..cut], &opts);
        }
    }
}

#[test]
fn xorshift_multibyte_mutations_never_panic() {
    let opts = DecodeOptions::default();
    let mut s: u32 = 0x1234_5678;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    for stream in seed_streams() {
        if stream.is_empty() {
            continue;
        }
        for _ in 0..2000 {
            let mut m = stream.clone();
            let flips = 1 + (next() % 5) as usize;
            for _ in 0..flips {
                let pos = (next() as usize) % m.len();
                m[pos] ^= (next() & 0xFF) as u8;
            }
            let _ = decode_file(&m, &opts);
            let _ = decode_embedded(None, &m, &opts);
        }
    }
}
