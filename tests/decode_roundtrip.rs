//! Phase 1 decoder round-trip matrix (jbig2decplan.md §21.2/§21.3, Gap C/E).
//!
//! For both PBM fixtures × {generic, lossless} × {standalone, embedded}, assert
//! that `native_decode(encode(img))` is pixel-exact to the original AND, when
//! `jbig2dec` is available, that native output matches `jbig2dec` on the same
//! stream. Plus the odd-width property corpus.

mod common;

use common::oracle;
use common::pbm::{Pbm, assert_pixels_eq};

use jbig2enc_rust::decode::{DecodeOptions, decode_embedded, decode_file};
use jbig2enc_rust::shared::bitmap::MonoBitmap;
use jbig2enc_rust::{
    Jbig2Context, encode_single_image, encode_single_image_lossless,
    encode_single_image_with_config,
};

/// Convert a decoded `MonoBitmap` to a one-byte-per-pixel `Pbm`.
fn mono_to_pbm(bm: &MonoBitmap) -> Pbm {
    let w = bm.width();
    let h = bm.height();
    let mut pixels = vec![0u8; (w as usize) * (h as usize)];
    for y in 0..h {
        for x in 0..w {
            pixels[(y * w + x) as usize] = bm.get(x, y) as u8;
        }
    }
    Pbm::new(w, h, pixels)
}

/// Load a fixture PBM as a `Pbm` (1 byte/pixel).
fn load_fixture(path: &str) -> Pbm {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    Pbm::from_p4(&bytes).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

#[derive(Clone, Copy)]
enum Mode {
    Generic,
    Lossless,
}

fn encode_standalone(mode: Mode, img: &Pbm) -> Vec<u8> {
    let (w, h) = (img.width, img.height);
    match mode {
        Mode::Generic => encode_single_image(&img.pixels, w, h, false)
            .expect("encode generic standalone")
            .page_data,
        Mode::Lossless => encode_single_image_lossless(&img.pixels, w, h, false)
            .expect("encode lossless standalone")
            .page_data,
    }
}

fn encode_embedded(mode: Mode, img: &Pbm) -> Vec<u8> {
    let (w, h) = (img.width, img.height);
    match mode {
        Mode::Generic => encode_single_image(&img.pixels, w, h, true)
            .expect("encode generic embedded")
            .page_data,
        Mode::Lossless => {
            let ctx = Jbig2Context::with_lossless_config(true);
            encode_single_image_with_config(&img.pixels, w, h, ctx)
                .expect("encode lossless embedded")
                .page_data
        }
    }
}

fn check_standalone(mode: Mode, img: &Pbm, label: &str) {
    let stream = encode_standalone(mode, img);
    let opts = DecodeOptions::default();
    let doc = decode_file(&stream, &opts).unwrap_or_else(|e| panic!("{label} native: {e}"));
    let native = mono_to_pbm(doc.first_page().expect("a page"));
    assert_pixels_eq(img, &native, &format!("{label} native vs original"));

    if let Some(res) = oracle::decode_standalone(&stream) {
        let jd = res.unwrap_or_else(|e| panic!("{label} jbig2dec: {e}"));
        assert_pixels_eq(&native, &jd, &format!("{label} native vs jbig2dec"));
    }
}

fn check_embedded(mode: Mode, img: &Pbm, label: &str) {
    let stream = encode_embedded(mode, img);
    let opts = DecodeOptions::default();
    let native_bm = decode_embedded(None, &stream, &opts)
        .unwrap_or_else(|e| panic!("{label} native: {e}"));
    let native = mono_to_pbm(&native_bm);
    assert_pixels_eq(img, &native, &format!("{label} native vs original"));

    if let Some(res) = oracle::decode_embedded(None, &stream) {
        let jd = res.unwrap_or_else(|e| panic!("{label} jbig2dec: {e}"));
        assert_pixels_eq(&native, &jd, &format!("{label} native vs jbig2dec"));
    }
}

fn fixtures() -> Vec<(&'static str, Pbm)> {
    vec![
        ("test_image", load_fixture("tests/fixtures/test_image.pbm")),
        ("test_image1", load_fixture("tests/fixtures/test_image1.pbm")),
    ]
}

#[test]
fn fixtures_generic_standalone() {
    for (name, img) in fixtures() {
        check_standalone(Mode::Generic, &img, &format!("{name} generic standalone"));
    }
}

#[test]
fn fixtures_generic_embedded() {
    for (name, img) in fixtures() {
        check_embedded(Mode::Generic, &img, &format!("{name} generic embedded"));
    }
}

#[test]
fn fixtures_lossless_standalone() {
    for (name, img) in fixtures() {
        check_standalone(Mode::Lossless, &img, &format!("{name} lossless standalone"));
    }
}

#[test]
fn fixtures_lossless_embedded() {
    for (name, img) in fixtures() {
        check_embedded(Mode::Lossless, &img, &format!("{name} lossless embedded"));
    }
}

/// Deterministic random bitmap of the given size.
fn random_bitmap(w: u32, h: u32, seed: u64) -> Pbm {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut pixels = vec![0u8; (w as usize) * (h as usize)];
    for p in pixels.iter_mut() {
        // splitmix64 step.
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        *p = (z & 1) as u8;
    }
    Pbm::new(w, h, pixels)
}

const ODD_WIDTHS: [u32; 14] = [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65];

#[test]
fn property_odd_widths_native_roundtrip() {
    let opts = DecodeOptions::default();
    for &w in &ODD_WIDTHS {
        for &h in &[1u32, 5, 16, 40] {
            let img = random_bitmap(w, h, (w as u64) * 131 + h as u64);
            // Standalone generic.
            let stream = encode_single_image(&img.pixels, w, h, false)
                .expect("encode")
                .page_data;
            let doc = decode_file(&stream, &opts).expect("native decode");
            let native = mono_to_pbm(doc.first_page().expect("page"));
            assert_pixels_eq(&img, &native, &format!("odd w={w} h={h} native"));
        }
    }
}

#[test]
fn property_odd_widths_jbig2dec_agreement() {
    // A subset also cross-checked against jbig2dec (embedded mode).
    if oracle::jbig2dec_path().is_none() {
        eprintln!("jbig2dec not found; skipping oracle agreement");
        return;
    }
    let opts = DecodeOptions::default();
    for &w in &[7u32, 8, 17, 33, 64, 65] {
        let h = 24u32;
        let img = random_bitmap(w, h, 777 + w as u64);
        let stream = encode_single_image(&img.pixels, w, h, true)
            .expect("encode")
            .page_data;
        let native = mono_to_pbm(&decode_embedded(None, &stream, &opts).expect("native"));
        assert_pixels_eq(&img, &native, &format!("odd w={w} native"));
        let jd = oracle::decode_embedded(None, &stream)
            .expect("jbig2dec present")
            .expect("jbig2dec decode");
        assert_pixels_eq(&native, &jd, &format!("odd w={w} native vs jbig2dec"));
    }
}
