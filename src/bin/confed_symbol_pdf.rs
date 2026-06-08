//! Generate a complete PDF from all confed pages using JBIG2 symbol mode
//!
//! Processes every PBM file in the confed/ directory and creates
//! a single multi-page PDF with symbol mode encoding.

use jbig2enc_rust::jbig2enc::Jbig2Encoder;
use jbig2enc_rust::jbig2structs::Jbig2Config;
use lopdf::{Dictionary, Document, Object, Stream};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    let mut source_dir = "confed".to_string();
    let mut output_path = "test_output_pdfs/confed_complete_symbol_mode.pdf".to_string();
    let mut max_pages: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--source" => {
                i += 1;
                if i < args.len() {
                    source_dir = args[i].clone();
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output_path = args[i].clone();
                }
            }
            "--max-pages" => {
                i += 1;
                if i < args.len() {
                    max_pages = args[i].parse().ok();
                }
            }
            "--help" | "-h" => {
                println!("Generate complete PDF from confed pages with JBIG2 symbol mode");
                println!();
                println!("Usage: cargo run --release --bin confed_symbol_pdf [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --source DIR     Source directory (default: confed)");
                println!("  --output FILE    Output PDF path");
                println!("  --max-pages N    Limit to N pages (default: all)");
                println!("  --help, -h       Show this help");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     Confed Complete Symbol Mode PDF Generator            ║");
    println!("║     (All Pages → Single PDF with Global Dictionary)      ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Find all PBM files
    let source_path = PathBuf::from(&source_dir);
    if !source_path.exists() {
        eprintln!("Error: Source directory '{}' not found", source_dir);
        return Ok(());
    }

    let mut pbm_files: Vec<PathBuf> = std::fs::read_dir(&source_path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "pbm"))
        .collect();
    
    pbm_files.sort();
    
    let total_files = pbm_files.len();
    if let Some(max) = max_pages {
        pbm_files.truncate(max);
    }

    if pbm_files.is_empty() {
        eprintln!("No PBM files found in {}", source_dir);
        return Ok(());
    }

    println!("Source: {}/", source_dir);
    println!("Total PBM files found: {}", total_files);
    println!("Processing: {} pages", pbm_files.len());
    println!("Output: {}", output_path);
    println!();

    // Create output directory
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Configure for symbol mode with PDF embedding
    let mut cfg = Jbig2Config::text();
    cfg.want_full_headers = false; // PDF mode - no file headers
    cfg.auto_thresh = true; // Enable automatic thresholding
    cfg.symbol_mode = true; // Explicitly enable symbol mode
    cfg.duplicate_line_removal = true;

    println!("Encoding with JBIG2 symbol mode...");
    println!("  - Auto threshold: {}", cfg.auto_thresh);
    println!("  - Duplicate line removal: {}", cfg.duplicate_line_removal);
    println!();

    let mut encoder = Jbig2Encoder::new(&cfg);
    
    let mut page_dims: Vec<(u32, u32)> = Vec::new();
    let mut total_pbm_bytes = 0u64;

    for (idx, pbm_path) in pbm_files.iter().enumerate() {
        if (idx + 1) % 10 == 0 || idx == 0 {
            print!("  Processing page {:4}/{}...", idx + 1, pbm_files.len());
            std::io::stdout().flush().ok();
        }

        let (binary, width, height) = load_pbm_to_array(pbm_path)?;
        total_pbm_bytes += binary.len() as u64;
        page_dims.push((width, height));

        let array = ndarray::Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
            binary[y * width as usize + x]
        });

        encoder.add_page(&array)?;

        if (idx + 1) % 10 == 0 || idx == pbm_files.len() - 1 {
            println!(" done ({}x{})", width, height);
        }
    }

    println!();
    println!("Flushing encoder...");
    let split = encoder.flush_pdf_split()?;
    let metrics = encoder.metrics_snapshot();

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Encoding Complete                                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Symbol Mode Statistics:");
    println!("  Symbols discovered:     {}", metrics.symbol_stats.symbols_discovered);
    println!("  Symbols exported:       {}", metrics.symbol_stats.symbols_exported);
    println!("  Global symbols:         {}", metrics.symbol_stats.global_symbol_count);
    println!("  Local symbols:          {}", metrics.symbol_stats.local_symbol_count);
    println!("  Avg symbol reuse:       {:.2}x", metrics.symbol_stats.avg_symbol_reuse);
    println!();
    
    let globals_bytes = split.global_segments.as_ref().map_or(0, |g| g.len());
    let total_jbig2_bytes: usize = split.page_streams.iter().map(|p| p.len()).sum();
    let compression_ratio = if total_jbig2_bytes > 0 {
        total_pbm_bytes as f64 / total_jbig2_bytes as f64
    } else {
        0.0
    };

    println!("Size Summary:");
    println!("  Original PBM data:      {:.2} MB", total_pbm_bytes as f64 / (1024.0 * 1024.0));
    println!("  JBIG2 encoded:          {:.2} MB", total_jbig2_bytes as f64 / (1024.0 * 1024.0));
    println!("  Global dictionary:      {} bytes", globals_bytes);
    println!("  Compression ratio:      {:.1}x", compression_ratio);
    println!();

    if globals_bytes > 0 {
        println!("  ✓ Global dictionary created ({} bytes)", globals_bytes);
    } else {
        println!("  ⚠ No global dictionary (using generic region encoding)");
    }
    println!();

    // Assemble PDF
    println!("Assembling PDF with {} pages...", split.page_streams.len());
    
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut page_ids = Vec::new();
    
    // Create global dictionary object
    let globals_id = if let Some(ref globals) = split.global_segments {
        if !globals.is_empty() {
            let global_stream = Stream::new(Dictionary::new(), globals.clone());
            Some(doc.add_object(global_stream))
        } else {
            None
        }
    } else {
        None
    };

    for (page_idx, (page_data, dims)) in split.page_streams.iter().zip(page_dims.iter()).enumerate() {
        let page_id = doc.new_object_id();
        let mut page_dict = Dictionary::new();
        page_dict.set("Type", Object::Name(b"Page".to_vec()));
        page_dict.set("Parent", Object::Reference(pages_id));
        page_dict.set(
            "MediaBox",
            vec![
                Object::Real(0.0),
                Object::Real(0.0),
                Object::Real(dims.0 as f32),
                Object::Real(dims.1 as f32),
            ],
        );

        // Create JBIG2 image XObject
        let mut img_dict = Dictionary::new();
        img_dict.set("Type", Object::Name(b"XObject".to_vec()));
        img_dict.set("Subtype", Object::Name(b"Image".to_vec()));
        img_dict.set("Width", Object::Integer(dims.0 as i64));
        img_dict.set("Height", Object::Integer(dims.1 as i64));
        img_dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        img_dict.set("BitsPerComponent", Object::Integer(1));
        img_dict.set("Filter", Object::Name(b"JBIG2Decode".to_vec()));
        img_dict.set("Decode", vec![0.into(), 1.into()]);

        // Reference global dictionary if available
        if let Some(gid) = globals_id {
            let mut decode_parms = Dictionary::new();
            decode_parms.set("JBIG2Globals", Object::Reference(gid));
            img_dict.set("DecodeParms", Object::Dictionary(decode_parms));
        }

        let img_id = doc.add_object(Stream::new(img_dict, page_data.clone()));
        
        // Create content stream
        let mut content_ops = Vec::new();
        content_ops.push(lopdf::content::Operation::new("q", vec![]));
        content_ops.push(lopdf::content::Operation::new(
            "cm",
            vec![
                Object::Real(dims.0 as f32),
                Object::Real(0.0),
                Object::Real(0.0),
                Object::Real(dims.1 as f32),
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
        (b"Count".to_vec(), Object::Integer(split.page_streams.len() as i64)),
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

    doc.save(&output_path)?;

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  ✓ PDF Successfully Generated                            ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Output file: {}", output_path);
    let file_size = std::fs::metadata(&output_path)?.len();
    println!("File size:     {:.2} MB", file_size as f64 / (1024.0 * 1024.0));
    println!("Pages:         {}", split.page_streams.len());
    println!();
    println!("To test in Okular:");
    println!("  okular {}", output_path);
    println!();
    println!("To verify:");
    println!("  pdfinfo {}", output_path);

    Ok(())
}
