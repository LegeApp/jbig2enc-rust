//! Decoder benchmarks (jbig2dec-gaps-plan.md Gap D).
//!
//! `decode_generic_a4_300dpi` synthesises a text-like A4-at-300dpi bitmap
//! (2480×3508), encodes it once as a generic region in setup, and benchmarks
//! decoding it. This is the dominant real-world case for the PDF renderer.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use jbig2enc_rust::decode::{DecodeOptions, DecoderContext, decode_file_with_context};
use jbig2enc_rust::{Jbig2Context, encode_single_image_with_config};

/// Deterministically synthesise a "text-like" bitmap: rows of short black runs
/// (glyph strokes) with generous white gaps, at the given size.
fn synth_textlike(w: u32, h: u32) -> Vec<u8> {
    let mut px = vec![0u8; (w as usize) * (h as usize)];
    let mut s: u32 = 0x9E37_79B9;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    // Lay down ~40 text lines; within each line, scatter short ink runs.
    let line_h = 60u32;
    for line in 0..(h / line_h) {
        let base_y = line * line_h + 10;
        let mut x = 8u32;
        while x < w - 20 {
            let run = 2 + (next() % 6);
            let gap = 2 + (next() % 10);
            let glyph_h = 8 + (next() % 24);
            if next() % 3 != 0 {
                for yy in base_y..(base_y + glyph_h).min(h) {
                    for xx in x..(x + run).min(w) {
                        px[(yy * w + xx) as usize] = 1;
                    }
                }
            }
            x += run + gap;
        }
    }
    px
}

fn bench_decode_generic_a4(c: &mut Criterion) {
    let (w, h) = (2480u32, 3508u32);
    let px = synth_textlike(w, h);

    // Encode once in a generic (lossless, symbol-off) configuration.
    let ctx = Jbig2Context::with_lossless_config(false);
    let encoded = encode_single_image_with_config(&px, w, h, ctx)
        .expect("encode A4 generic")
        .page_data;
    eprintln!("encoded A4 generic stream = {} bytes", encoded.len());

    let opts = DecodeOptions::default();
    let mut dctx = DecoderContext::new();

    let mut group = c.benchmark_group("decode_generic_a4_300dpi");
    group.sample_size(20);
    group.bench_function("decode", |b| {
        b.iter(|| {
            let doc = decode_file_with_context(black_box(&encoded), &opts, &mut dctx)
                .expect("decode");
            black_box(doc.pages.len());
        });
    });
    group.finish();
}

criterion_group!(benches, bench_decode_generic_a4);
criterion_main!(benches);
