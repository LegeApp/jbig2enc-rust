//! Generate test PDFs with embedded JBIG2 for Okular testing
//!
//! This binary loads PBM images, encodes them with JBIG2 symbol mode,
//! and embeds them into a PDF for testing with Okular.
//!
//! Usage:
//!   cargo run --release --bin generate_test_pdf [--pages N] [--output FILE.pdf]

use jbig2enc_rust::jbig2enc::Jbig2Encoder;
use jbig2enc_rust::jbig2structs::Jbig2Config;
use lopdf::{Dictionary, Document, Object, Stream};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

/// Load a PBM file into a 2D array
fn load_pbm_to_array(path: &Path) -> Result<(Vec<u8>, u32, u32), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim() != "P4" {
        return Err(format!("Not a P4 PBM file: {}", path.display()).into());
    }

    let (width, height) = loop {
        line.clear();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let dims: Vec<usize> = trimmed
            .split_whitespace()
            .map(|s| s.parse())
            .collect::<Result<Vec<_>, _>>()?;
        if dims.len() == 2 {
            break (dims[0] as u32, dims[1] as u32);
        }
    };

    let bytes_per_row = (width + 7) / 8;
    let mut packed = vec![0u8; bytes_per_row as usize * height as usize];
    reader.read_exact(&mut packed)?;

    // Convert packed bits to 0/255 binary format
    let mut binary = vec![0u8; width as usize * height as usize];
    for (y, row) in packed.chunks(bytes_per_row as usize).enumerate() {
        for x in 0..width as usize {
            let byte = row[x / 8];
            let bit = (byte >> (7 - (x % 8))) & 1;
            binary[y * width as usize + x] = if bit == 1 { 255 } else { 0 };
        }
    }

    Ok((binary, width, height))
}

/// Encode a binary image to JBIG2 with symbol mode
fn encode_jbig2_symbol_mode(
    binary: &[u8],
    width: u32,
    height: u32,
) -> Result<(Vec<u8>, Option<Vec<u8>>), Box<dyn std::error::Error>> {
    // Configure for symbol mode with PDF embedding
    let mut cfg = Jbig2Config::text();
    cfg.want_full_headers = false; // PDF mode - no file headers

    // Convert to Array2
    let array = ndarray::Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
        binary[y * width as usize + x]
    });

    let mut encoder = Jbig2Encoder::new(&cfg);
    encoder.add_page(&array)?;

    let split = encoder.flush_pdf_split()?;

    let page_data = if split.page_streams.is_empty() {
        Vec::new()
    } else {
        split.page_streams[0].clone()
    };

    Ok((page_data, split.global_segments))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    let mut num_pages = 5;
    let mut output_path = "test_output_pdfs/jbig2_symbol_test.pdf".to_string();
    let mut source_dir = "confed".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pages" => {
                i += 1;
                if i < args.len() {
                    num_pages = args[i].parse().unwrap_or(num_pages);
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output_path = args[i].clone();
                }
            }
            "--source" => {
                i += 1;
                if i < args.len() {
                    source_dir = args[i].clone();
                }
            }
            "--help" | "-h" => {
                println!("Generate test PDFs with embedded JBIG2 for Okular testing");
                println!();
                println!("Usage: cargo run --release --bin generate_test_pdf [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --pages N      Number of pages to include (default: 5)");
                println!("  --output FILE  Output PDF path (default: test_output_pdfs/jbig2_symbol_test.pdf)");
                println!("  --source DIR   Source PBM directory (default: confed)");
                println!("  --help, -h     Show this help");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     JBIG2 Symbol Mode PDF Generator (Fixed Version)      ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Find PBM files
    let source_path = PathBuf::from(&source_dir);
    if !source_path.exists() {
        eprintln!("Error: Source directory '{}' not found", source_dir);
        eprintln!("Available sources: confed/");
        return Ok(());
    }

    let mut pbm_files: Vec<PathBuf> = std::fs::read_dir(&source_path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "pbm"))
        .collect();
    pbm_files.sort();
    pbm_files.truncate(num_pages);

    if pbm_files.is_empty() {
        eprintln!("No PBM files found in {}", source_dir);
        return Ok(());
    }

    println!("Source: {}/", source_dir);
    println!("Pages: {}", pbm_files.len());
    println!("Output: {}", output_path);
    println!();

    // Create output directory
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Load and encode pages
    println!("Loading and encoding pages...");
    
    struct EncodedPage {
        page_data: Vec<u8>,
        global_data: Option<Vec<u8>>,
        width: u32,
        height: u32,
    }
    
    let mut encoded_pages: Vec<EncodedPage> = Vec::new();
    let mut total_global_bytes = 0;
    let mut total_page_bytes = 0;

    for (_idx, pbm_path) in pbm_files.iter().enumerate() {
        print!("  Processing {}... ", pbm_path.display());
        
        let (binary, width, height) = load_pbm_to_array(pbm_path)?;
        let (page_data, global_data) = encode_jbig2_symbol_mode(&binary, width, height)?;

        if page_data.is_empty() {
            println!("skipped (no data)");
            continue;
        }

        if let Some(ref globals) = global_data {
            total_global_bytes += globals.len();
        }
        total_page_bytes += page_data.len();

        let data_len = page_data.len();
        encoded_pages.push(EncodedPage {
            page_data,
            global_data,
            width,
            height,
        });
        println!("{}x{} ({} bytes)", width, height, data_len);
    }

    if encoded_pages.is_empty() {
        eprintln!("No pages were successfully processed");
        return Ok(());
    }

    println!();
    println!("Encoding summary:");
    println!("  Total pages: {}", encoded_pages.len());
    println!("  Global dict bytes: {}", total_global_bytes);
    println!("  Page stream bytes: {}", total_page_bytes);
    println!();

    // Assemble PDF using lopdf
    println!("Assembling PDF...");
    
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut page_ids = Vec::new();
    
    // Create global dictionary object (shared across all pages)
    let globals_id = if total_global_bytes > 0 {
        // Collect all unique global dictionaries
        let mut all_globals = Vec::new();
        for page in &encoded_pages {
            if let Some(ref g) = page.global_data {
                all_globals.push(g.clone());
            }
        }
        
        if !all_globals.is_empty() {
            // Use first global dict (they should all be the same for same document)
            let global_stream = Stream::new(Dictionary::new(), all_globals[0].clone());
            Some(doc.add_object(global_stream))
        } else {
            None
        }
    } else {
        None
    };

    for (_page_idx, encoded) in encoded_pages.iter().enumerate() {
        let page_id = doc.new_object_id();
        let mut page_dict = Dictionary::new();
        page_dict.set("Type", Object::Name(b"Page".to_vec()));
        page_dict.set("Parent", Object::Reference(pages_id));
        page_dict.set(
            "MediaBox",
            vec![
                Object::Real(0.0),
                Object::Real(0.0),
                Object::Real(encoded.width as f32),
                Object::Real(encoded.height as f32),
            ],
        );

        // Create image XObject
        let mut img_dict = Dictionary::new();
        img_dict.set("Type", Object::Name(b"XObject".to_vec()));
        img_dict.set("Subtype", Object::Name(b"Image".to_vec()));
        img_dict.set("Width", Object::Integer(encoded.width as i64));
        img_dict.set("Height", Object::Integer(encoded.height as i64));
        img_dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        img_dict.set("BitsPerComponent", Object::Integer(1));
        img_dict.set("Filter", Object::Name(b"JBIG2Decode".to_vec()));
        img_dict.set("Decode", vec![0.into(), 1.into()]);

        // Add global dictionary reference if present
        if let Some(gid) = globals_id {
            let mut decode_parms = Dictionary::new();
            decode_parms.set("JBIG2Globals", Object::Reference(gid));
            img_dict.set("DecodeParms", Object::Dictionary(decode_parms));
        }

        let img_id = doc.add_object(Stream::new(img_dict, encoded.page_data.clone()));
        
        // Create content stream
        let mut content_ops = Vec::new();
        content_ops.push(lopdf::content::Operation::new("q", vec![]));
        content_ops.push(lopdf::content::Operation::new(
            "cm",
            vec![
                Object::Real(encoded.width as f32),
                Object::Real(0.0),
                Object::Real(0.0),
                Object::Real(encoded.height as f32),
                Object::Real(0.0),
                Object::Real(0.0),
            ],
        ));
        content_ops.push(lopdf::content::Operation::new(
            "Do",
            vec![Object::Name(b"Im1".to_vec())],
        ));
        content_ops.push(lopdf::content::Operation::new("Q", vec![]));
        
        let content = lopdf::content::Content { operations: content_ops };
        let content_stream = Stream::new(Dictionary::new(), content.encode()?);
        let content_id = doc.add_object(content_stream);

        // Create resources
        let mut xobject_dict = Dictionary::new();
        xobject_dict.set("Im1", Object::Reference(img_id));
        let mut resources = Dictionary::new();
        resources.set("XObject", Object::Dictionary(xobject_dict));
        page_dict.set("Resources", Object::Dictionary(resources));
        page_dict.set("Contents", Object::Reference(content_id));

        doc.objects.insert(page_id, Object::Dictionary(page_dict));
        page_ids.push(Object::Reference(page_id));
    }

    // Create pages tree
    let pages_dict = Dictionary::from_iter([
        (b"Type".to_vec(), Object::Name(b"Pages".to_vec())),
        (b"Kids".to_vec(), Object::Array(page_ids)),
        (b"Count".to_vec(), Object::Integer(encoded_pages.len() as i64)),
    ]);
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    // Create catalog
    let catalog_id = doc.new_object_id();
    let catalog_dict = Dictionary::from_iter([
        (b"Type".to_vec(), Object::Name(b"Catalog".to_vec())),
        (b"Pages".to_vec(), Object::Reference(pages_id)),
    ]);
    doc.objects.insert(catalog_id, Object::Dictionary(catalog_dict));
    doc.trailer.set("Root", Object::Reference(catalog_id));

    // Save the PDF
    doc.save(&output_path)?;

    println!();
    println!("✓ PDF written to: {}", output_path);
    println!();
    println!("To test in Okular:");
    println!("  okular {}", output_path);
    println!();
    println!("To inspect:");
    println!("  pdfinfo {}", output_path);
    println!("  qpdf --show-encryption {}", output_path);

    Ok(())
}
