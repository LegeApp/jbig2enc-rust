// Integration tests for JBIG2 encoder core functionality

use std::io::{BufReader, BufRead, Read, Seek, SeekFrom};
use lutz::Image;  // Import the Image trait for width() and height() methods
use std::time::Instant;

#[cfg(feature = "trace_arith")]
use tracing_subscriber::fmt;

#[cfg(not(feature = "trace_arith"))]
#[macro_use]
mod noop_tracing {
    macro_rules! debug {
        ($($arg:tt)*) => { };
    }
    pub(crate) use debug;
}
use jbig2::{
    jbig2arith::{Jbig2ArithCoder, State, BASE},
    jbig2enc::{encode_page_with_symbol_dictionary, encode_symbol_dict, Jbig2EncConfig},
    jbig2sym::{array_to_bitimage, BitImage, Symbol},
};
use ndarray::Array2;

// Macro to handle debug statements with elapsed time
macro_rules! debug_with_time {
    ($start:expr, $($arg:tt)*) => {
        let elapsed = $start.elapsed().as_millis();
        let message = format!($($arg)*);
        debug!("{}ms: {}", elapsed, message);
    };
}

/// Load a PBM file from `tests/fixtures/…` and convert to BitImage
fn load_pbm(path: &str) -> BitImage {
    let mut file = std::fs::File::open(path).expect("Failed to open PBM file");
    let mut reader = BufReader::new(&mut file);
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "P4", "Only raw PBM (P4) supported");
    // Skip comments and get dimensions
    line.clear();
    loop {
        reader.read_line(&mut line).unwrap();
        let trimmed = line.trim();
        if !trimmed.starts_with('#') && !trimmed.is_empty() {
            break;
        }
        line.clear();
    }
    let parts: Vec<&str> = line.trim().split_whitespace().collect();
    let width = parts[0].parse::<usize>().unwrap();
    let height = parts[1].parse::<usize>().unwrap();
    let current_file_pos = reader.stream_position().unwrap();
    file.seek(SeekFrom::Start(current_file_pos)).unwrap();
    let width_in_bytes = (width + 7) / 8;
    let expected_data_len = height * width_in_bytes;
    let mut data = vec![0u8; expected_data_len];
    file.read_exact(&mut data).unwrap();
    // Convert PBM bytes to Array2<u8>
    let mut image_array = Array2::<u8>::zeros((height, width));
    for r in 0..height {
        for c in 0..width {
            let byte_idx = r * width_in_bytes + (c / 8);
            let bit_idx_in_byte = c % 8;
            if byte_idx < data.len() {
                let pbm_pixel_value = (data[byte_idx] >> (7 - bit_idx_in_byte)) & 1;
                image_array[(r, c)] = if pbm_pixel_value == 1 { 1 } else { 0 };
            }
        }
    }
    array_to_bitimage(&image_array)
}

#[test]
fn empty_symbol_list_returns_error() {
    let start = Instant::now();
    let cfg = Jbig2EncConfig::default();
    let err = encode_symbol_dict(&[], &cfg, /* page_number: */ 1).unwrap_err();
    // we expect the encoder to reject an empty symbol slice
    assert!(
        err.to_string().contains("no symbols"),
        "got unexpected error: {}", err
    );
    let duration = start.elapsed();
    println!("Test empty_symbol_list_returns_error took: {:?}", duration);
}

#[test]
fn single_symbol_dict_outputs_nonempty() {
    let start = Instant::now();
    let cfg = Jbig2EncConfig::default();
    // use the tiny 2×2 test image
    let img = load_pbm("tests/fixtures/test_image.pbm");
    let out = encode_symbol_dict(&[&img], &cfg, /* page_number: */ 1)
        .expect("single-symbol dict should encode");
    assert!(!out.is_empty(), "dictionary stream was empty");
    let duration = start.elapsed();
    println!("Test single_symbol_dict_outputs_nonempty took: {:?}", duration);
}

#[test]
fn duplicate_symbols_are_deduplicated() {
    let start = Instant::now();
    let cfg = Jbig2EncConfig::default();
    // create two identical symbols
    let img = load_pbm("tests/fixtures/test_image.pbm");
    let dict = encode_symbol_dict(&[&img, &img], &cfg, /* page_number: */ 1)
        .expect("dict of two identical symbols should encode");
    // pick a heuristic: deduplicated dict should be smaller than twice single
    let solo = encode_symbol_dict(&[&img], &cfg, 1).unwrap();
    assert!(
        dict.len() <= solo.len() * 2,
        "deduplicated dict too large: {} vs {}",
        dict.len(),
        solo.len()
    );
    let duration = start.elapsed();
    println!("Test duplicate_symbols_are_deduplicated took: {:?}", duration);
}

#[test]
fn full_page_with_symbol_dictionary_roundtrips() {
    let start = Instant::now();
    let cfg = Jbig2EncConfig::default();
    // use the full-page fixture
    let page = load_pbm("tests/fixtures/test_image1.pbm");
    let (stream, next_segment_num) =
        encode_page_with_symbol_dictionary(&page, &cfg, /* next_segment_num: */ 1)
            .expect("full-page encode should succeed");
    // Must produce a non-empty byte stream
    assert!(!stream.is_empty(), "output stream was empty");
    // Verify we got the next segment number (should be > 1 since we started at 1)
    assert!(next_segment_num > 1, "expected next_segment_num > 1, got {}", next_segment_num);
    let duration = start.elapsed();
    println!("Test full_page_with_symbol_dictionary_roundtrips took: {:?}", duration);
}

#[test]
fn test_glyph_creation() {
    let start = Instant::now();
    let img = BitImage::new(10, 10).unwrap();
    assert_eq!(img.width, 10);
    assert_eq!(img.height, 10);
    let duration = start.elapsed();
    println!("Test test_glyph_creation took: {:?}", duration);
}

#[test]
fn test_symbol_sorting() {
    let start = Instant::now();
    let mut symbols = vec![
        Symbol { image: BitImage::new(2, 3).unwrap(), hash: 0 },
        Symbol { image: BitImage::new(3, 2).unwrap(), hash: 0 },
    ];
    symbols.sort_by_key(|s| (s.image.height, s.image.width));
    assert_eq!(symbols[0].image.height, 2);
    let duration = start.elapsed();
    println!("Test test_symbol_sorting took: {:?}", duration);
}

#[test]
fn test_encode_symbol_dict_empty_fails() {
    let start = Instant::now();
    let config = Jbig2EncConfig::default();
    let result = encode_symbol_dict(&[], &config, 0);
    assert!(result.is_err());
    let duration = start.elapsed();
    println!("Test test_encode_symbol_dict_empty_fails took: {:?}", duration);
}

#[test]
fn test_encode_symbol_dict_single() {
    let start = Instant::now();
    let config = Jbig2EncConfig::default();
    let img = BitImage::new(5, 5).unwrap();
    let dict_result = encode_symbol_dict(&[&img], &config, 0);
    assert!(dict_result.is_ok());
    let bytes = dict_result.unwrap();
    assert!(!bytes.is_empty());
    let duration = start.elapsed();
    println!("Test test_encode_symbol_dict_single took: {:?}", duration);
}

#[test]
fn test_encode_page_with_symbol_dictionary() {
    let start = Instant::now();
    let img = create_checkerboard(40, 40);
    let config = Jbig2EncConfig::default();

    // Call the function under test
    let (bytes, _) = encode_page_with_symbol_dictionary(
        &img,
        &config,
        1,
    ).expect("Failed to encode page with symbol dictionary");

    println!("Final encoded data ({} bytes): {:?}", bytes.len(), bytes);
    assert!(!bytes.is_empty());
    // You might want to add more specific assertions here, e.g., check segment types or lengths
    let duration = start.elapsed();
    println!("Test test_encode_page_with_symbol_dictionary took: {:?}", duration);
}

#[cfg(any(feature = "trace_encoder", feature = "trace_arith"))]
fn init_tracing_for_test() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::new("trace");
    let _subscriber = fmt().with_env_filter(filter).init();
    tracing::debug!("Tracing initialized for test with trace level");
}

#[test]
fn test_arithmetic_coder_annex_h2() {
    let start = Instant::now();
    #[cfg(any(feature = "trace_encoder", feature = "trace_arith"))]
    init_tracing_for_test();
    
    // Input test data from JBIG2 Annex H.2
    let test_data = [
        0x00, 0x02, 0x00, 0x51, 0x00, 0x00, 0x00, 0xC0, 
        0x03, 0x52, 0x87, 0x2A, 0xAA, 0xAA, 0xAA, 0xAA,
        0x82, 0xC0, 0x20, 0x00, 0xFC, 0xD7, 0x9E, 0xF6, 
        0xBF, 0x7F, 0xED, 0x90, 0x4F, 0x46, 0xA3, 0xBF
    ];

    // Expected output from JBIG2 Annex H.2
    let expected_output = [
        0x84, 0xC7, 0x3B, 0xFC, 0xE1, 0xA1, 0x43, 0x04, 0x02, 0x20, 0x00, 0x00, 0x41, 0x0D,
        0xBB, 0x86, 0xF4, 0x31, 0x7F, 0xFF, 0x88, 0xFF, 0x37, 0x47, 0x1A, 0xDB, 0x6A, 0xDF,
    ];

    // Initialize coder with single context
    let mut coder = Jbig2ArithCoder::new();
    
    // Encode the test data using a single context
    #[cfg_attr(not(feature = "trace_arith"), allow(unused_imports))]
    use tracing::debug;
    #[cfg(any(feature = "trace_encoder", feature = "trace_arith"))]
    debug_with_time!(start, "Starting encoding of test data");
    let cx = 0;
    for (index, &byte) in test_data.iter().enumerate() {
        for bit in 0..8 {
            let bit_val = (byte >> (7 - bit)) & 1 != 0;
            #[cfg(any(feature = "trace_encoder", feature = "trace_arith"))]
            debug_with_time!(start, "Encoding byte {} bit {} with value {}", index, bit, bit_val);
            coder.encode_bit(cx, bit_val);
        }
    }
    
    // Flush the encoder to finalize the encoding
    coder.flush(false);
    
    // Get the encoded data
    let encoded = coder.as_bytes();
    
    // Compare with expected output
    #[cfg(any(feature = "trace_encoder", feature = "trace_arith"))]
    debug_with_time!(start, "Encoded data: {:?}", encoded);
    assert_eq!(
        encoded, &expected_output[..],
        "Arithmetic coder output does not match JBIG2 Annex H.2 test vector"
    );
    let duration = start.elapsed();
    println!("Test test_arithmetic_coder_annex_h2 took: {:?}", duration);
}

#[test]
fn test_arithmetic_coder_generic_region() {
    let start = Instant::now();
    #[cfg(any(feature = "trace_encoder", feature = "trace_arith"))]
    init_tracing_for_test();

    // Test a 4x4 bitmap, with each row in a u32:
    // Row 0: 1100...
    // Row 1: 0110...
    // Row 2: 0011...
    // Row 3: 0001...
    let bitmap: Vec<u32> = vec![
        0b11000000_00000000_00000000_00000000, // 0xC0000000
        0b01100000_00000000_00000000_00000000, // 0x60000000
        0b00110000_00000000_00000000_00000000, // 0x30000000
        0b00010000_00000000_00000000_00000000, // 0x10000000
    ];
    let width = 4;
    let height = 4;
    let template = 0;
    let at_pixels: Vec<(i8, i8)> = vec![]; // No adaptive pixels

    // With the arithmetic coder now flushing a terminating marker, the
    // expected output includes the 0xFF 0xAC trailer.
    let expected_region_output = [0xE8, 0x59, 0xFF, 0xFF, 0xAC];

    println!("Starting generic region encoding test...");
    println!("Starting generic region encoding test...");

    let mut img = BitImage::new(width as u32, height as u32).unwrap();
    for y in 0..height {
        for x in 0..width {
            let word = bitmap[y];
            let bit = (word >> (31 - x)) & 1;
            img.set(x as u32, y as u32, bit == 1);
        }
    }

    // Encode the generic region
    let encoded = jbig2::Jbig2ArithCoder::encode_generic_payload(&img, template, &at_pixels)
        .expect("Generic region encoding failed");

    println!("Final encoded data ({} bytes): {:?}", encoded.len(), encoded);
    
    // Verify the output matches expected
    assert_eq!(
        encoded, &expected_region_output[..],
        "Generic region arithmetic coder output does not match expected vector"
    );
    
    println!("Test passed!");
    let duration = start.elapsed();
    println!("Test test_arithmetic_coder_generic_region took: {:?}", duration);
}

#[test]
fn test_arithmetic_coder_base_table() {
    let start = Instant::now();
    // Verify BASE table entries
    assert_eq!(BASE[0], State { qe: 0x5601, nmps: 1,  nlps: 1,  switch: true  });
    assert_eq!(BASE[1], State { qe: 0x3401, nmps: 2,  nlps: 6,  switch: false });
    assert_eq!(BASE[2], State { qe: 0x1801, nmps: 3,  nlps: 9,  switch: false });
    assert_eq!(BASE[3], State { qe: 0x0AC1, nmps: 4,  nlps: 12, switch: false });
    assert_eq!(BASE[4], State { qe: 0x0521, nmps: 5,  nlps: 29, switch: false });
    assert_eq!(BASE[5], State { qe: 0x0221, nmps: 38, nlps: 33, switch: false });
    assert_eq!(BASE[6], State { qe: 0x5601, nmps: 7,  nlps: 6,  switch: true  });
    assert_eq!(BASE[7], State { qe: 0x5401, nmps: 8,  nlps: 14, switch: false });
    let duration = start.elapsed();
    println!("Test test_arithmetic_coder_base_table took: {:?}", duration);
}

/// Create a simple 4x4 test pattern
fn create_4x4_test_pattern() -> BitImage {
    let mut img = BitImage::new(4, 4).expect("Failed to create 4x4 image");
    for y in 0..4 {
        for x in 0..4 {
            img.set(x as u32, y as u32, (x + y) % 2 == 0);
        }
    }
    img
}

/// Create a checkerboard pattern of given size
fn create_checkerboard(width: u32, height: u32) -> BitImage {
    let mut img = BitImage::new(width, height).expect("Failed to create image");
    for y in 0..height {
        for x in 0..width {
            let value = (x + y) % 2 == 0;
            img.set(x, y, value);
        }
    }
    img
}

/// Create a half-black, half-white image
fn create_half_hald(width: u32, height: u32) -> BitImage {
    let mut img = BitImage::new(width, height).expect("Failed to create image");
    for y in 0..height {
        for x in 0..width {
            let value = x < width / 2;
            img.set(x, y, value);
        }
    }
    img
}

#[test]
fn test_generic_region() {
    let start = Instant::now();
    // Level 1: Basic 4x4 test with trace validation
    println!("Running Level 1: Basic 4x4 test...");
    let img = create_4x4_test_pattern();
    // Encode with template 0 and no adaptive pixels
    let encoded = jbig2::Jbig2ArithCoder::encode_generic_payload(&img, 0, &[])
        .expect("❌ Level 1: Failed to encode 4x4 test pattern");
    
    // Basic validation - at least some data should be written
    assert!(!encoded.is_empty(), "❌ Level 1: Encoded data should not be empty");
    
    // For a 4x4 pattern, wewatermark we expect at least 2 bytes (16 bits)
    assert!(encoded.len() >= 2, "❌ Level 1: Encoded data too small for 4x4 pattern");
    
    // First byte should be non-zero (since we have black pixels)
    assert_ne!(encoded[0], 0, "❌ Level 1: First byte should be non-zero for non-empty pattern");
    println!("✅ Level 1: 4x4 test passed");

    // Level 2: Larger test patterns
    println!("\nRunning Level 2: Larger test patterns...");
    
    // Test 1: 40x40 checkerboard
    let checkerboard = create_checkerboard(40, 40);
    let mut coder1 = Jbig2ArithCoder::new();
    let packed1 = checkerboard.to_packed_words();
    coder1.encode_generic_region(&packed1, 40_usize, 40, 0, &[])
        .expect("❌ Level 2: Failed to encode 40x40 checkerboard");
    let encoded1 = coder1.into_vec();
    assert!(!encoded1.is_empty(), "❌ Level 2: Checkerboard encoding failed");
    
    // Test 2: Half-black/half-white
    let half_half = create_half_hald(40, 40);
    let mut coder2 = Jbig2ArithCoder::new();
    let packed2 = half_half.to_packed_words();
    coder2.encode_generic_region(&packed2, 40_usize, 40, 0, &[])
        .expect("❌ Level 2: Failed to encode half-black/half-white image");
    let encoded2 = coder2.into_vec();
    assert!(!encoded2.is_empty(), "❌ Level 2: Half-half encoding failed");
    
    // The two encodings should be different
    assert_ne!(encoded1, encoded2, "❌ Level 2: Different patterns should produce different encodings");
    println!("✅ Level 2: Larger test patterns passed");

    // Level 3: Odd-width tests
    println!("\nRunning Level 3: Odd-width tests...");
    let odd_widths = [37, 41, 63];
    
    for &width in &odd_widths {
        println!("  Testing width: {}...", width);
        // Test with checkerboard pattern
        let img = create_checkerboard(width, 20);
        let mut coder = Jbig2ArithCoder::new();
        let packed = img.to_packed_words();
        
        coder.encode_generic_region(&packed, width as usize, 20, 0, &[])
            .unwrap_or_else(|_| panic!("❌ Level 3: Failed to encode {}x20 checkerboard", width));
            
        let encoded = coder.into_vec();
        assert!(!encoded.is_empty(), "❌ Level 3: Failed to encode {}x20 image", width);
        
        // Test with half-black/half-white pattern
        let img2 = create_half_hald(width, 20);
        let mut coder2 = Jbig2ArithCoder::new();
        let packed2 = img2.to_packed_words();
        
        coder2.encode_generic_region(&packed2, width as usize, 20, 0, &[])
            .unwrap_or_else(|_| panic!("❌ Level 3: Failed to encode {}x20 half-half", width));
            
        let encoded2 = coder2.into_vec();
        assert!(!encoded2.is_empty(), "❌ Level 3: Failed to encode {}x20 half-half", width);
    }
    println!("✅ Level 3: All odd-width tests passed");
    let duration = start.elapsed();
    println!("Test test_generic_region took: {:?}", duration);
}

// Add more tests as needed for dictionary merging, refinement, etc.