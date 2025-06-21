// Integration tests using PBM images
const TEST_IMAGE_PBM: &str = "tests/fixtures/test_image.pbm";
const TEST_IMAGE1_PBM: &str = "tests/fixtures/test_image1.pbm";

use std::io::{BufReader, BufRead, Read, Seek, SeekFrom};

use jbig2::jbig2sym::{BitImage, array_to_bitimage};
use jbig2::jbig2enc::{encode_page_with_symbol_dictionary, Jbig2EncConfig, encode_symbol_dict};
use ndarray::Array2;

/// Load a PBM file and convert to BitImage
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
fn test_encode_test_image_pbm() {
    let img = load_pbm(TEST_IMAGE_PBM);
    let config = Jbig2EncConfig::default();
    let dict_result = encode_symbol_dict(&[&img], &config, 0);
    assert!(dict_result.is_ok());
    let bytes = dict_result.unwrap();
    assert!(!bytes.is_empty());
}

#[test]
fn test_encode_test_image1_pbm_full_page() {
    let img = load_pbm(TEST_IMAGE1_PBM);
    let config = Jbig2EncConfig::default();
    let result = encode_page_with_symbol_dictionary(&img, &config, 0);
    assert!(result.is_ok());
    let (bytes, _seg) = result.unwrap();
    assert!(!bytes.is_empty());
}
