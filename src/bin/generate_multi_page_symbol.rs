//! Generate multi-page PDF with proper JBIG2 symbol mode and global dictionary
//!
//! Creates text pages with enough repetition to trigger symbol clustering
//! and global dictionary creation.

use jbig2enc_rust::jbig2enc::Jbig2Encoder;
use jbig2enc_rust::jbig2structs::Jbig2Config;
use lopdf::{Dictionary, Document, Object, Stream};
use std::path::Path;

/// Create a text page with repeated characters that will trigger symbol mode
fn create_text_page(page_num: usize, width: u32, height: u32) -> ndarray::Array2<u8> {
    let mut array = ndarray::Array2::from_elem((height as usize, width as usize), 0u8);
    
    // More detailed character patterns (8x12 pixels for better CC detection)
    let chars = vec![
        // 'A'
        vec![(0,4), (1,3), (1,5), (2,2), (2,6), (3,1), (3,7), (4,0), (4,8), 
             (5,0), (5,8), (6,0), (6,8), (7,0), (7,8), (8,0), (8,8),
             (9,0), (9,8), (10,0), (10,8), (11,0), (11,8)],
        // 'B'
        vec![(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6),
             (1,0), (1,7), (2,0), (2,7), (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7),
             (4,0), (4,7), (5,0), (5,7), (6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7),
             (7,0), (7,7), (8,0), (8,7), (9,0), (9,7), (10,0), (10,7), (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7)],
        // 'C'
        vec![(0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
             (1,0), (1,1), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0), (10,0), (11,0),
             (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7),
             (10,7), (9,7), (8,7), (7,7), (6,7), (5,7), (4,7), (3,7), (2,7), (1,7)],
        // 'D'
        vec![(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6),
             (1,0), (1,7), (2,0), (2,7), (3,0), (3,7), (4,0), (4,7), (5,0), (5,7),
             (6,0), (6,7), (7,0), (7,7), (8,0), (8,7), (9,0), (9,7), (10,0), (10,7),
             (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7)],
        // 'E'
        vec![(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
             (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0), (10,0), (11,0),
             (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7),
             (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7),
             (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7)],
        // '0' (digit)
        vec![(0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
             (1,1), (1,8), (2,0), (2,9), (3,0), (3,9), (4,0), (4,9), (5,0), (5,9),
             (6,0), (6,9), (7,0), (7,9), (8,0), (8,9), (9,0), (9,9), (10,1), (10,8),
             (11,2), (11,3), (11,4), (11,5), (11,6), (11,7)],
        // '1' (digit)
        vec![(0,3), (0,4), (1,2), (1,3), (1,4), (2,1), (2,3), (2,4),
             (3,3), (3,4), (4,3), (4,4), (5,3), (5,4), (6,3), (6,4),
             (7,3), (7,4), (8,3), (8,4), (9,3), (9,4), (10,2), (10,5),
             (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7)],
        // '2' (digit)
        vec![(0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
             (1,1), (1,8), (2,0), (2,9), (3,8), (4,7), (5,6), (6,5),
             (7,4), (8,3), (9,2), (10,1), (10,0), (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7), (11,8), (11,9)],
    ];

    let char_width = 10;
    let char_height = 14;
    let line_spacing = 16;
    let chars_per_line = 30;
    let lines_per_page = 25;
    let start_x = 20;
    let start_y = 20;

    // Generate text: "PAGE X" header + repeating "ABCDE 012" pattern
    let mut all_chars = Vec::new();
    
    // Header: "PAGE X"
    let header = vec![4, 0, 5, 6, 7, 5]; // P A G E space X
    for &c in &header {
        all_chars.push(c.min(chars.len() - 1));
    }
    all_chars.push(6); // space
    let page_digits: Vec<usize> = page_num.to_string().chars()
        .map(|c| (c as usize - '0' as usize).min(chars.len() - 1))
        .collect();
    all_chars.extend(page_digits);
    
    // Pad header to chars_per_line
    while all_chars.len() < chars_per_line {
        all_chars.push(6); // space
    }
    
    // Body: repeating pattern
    let body_pattern = vec![0, 1, 2, 3, 4, 6, 5, 5, 5, 6]; // ABCDE space 012 space
    for _ in chars_per_line..(chars_per_line * lines_per_page) {
        all_chars.push(body_pattern[all_chars.len() % body_pattern.len()]);
    }

    // Render characters
    for (line_idx, line_chars) in all_chars.chunks(chars_per_line).enumerate() {
        if line_idx >= lines_per_page {
            break;
        }
        for (char_idx, &char_code) in line_chars.iter().enumerate() {
            let base_x = start_x + (char_idx * char_width) as u32;
            let base_y = start_y + (line_idx * line_spacing) as u32;
            
            if let Some(pattern) = chars.get(char_code) {
                for &(dx, dy) in pattern {
                    let x = base_x + dx;
                    let y = base_y + dy;
                    if x < width && y < height {
                        array[[y as usize, x as usize]] = 255;
                    }
                }
            }
        }
    }

    array
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    let mut num_pages = 5;
    let mut output_path = "test_output_pdfs/multi_page_symbol_mode.pdf".to_string();

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
            "--help" | "-h" => {
                println!("Generate multi-page PDF with JBIG2 symbol mode and global dictionary");
                println!();
                println!("Usage: cargo run --release --bin generate_multi_page_symbol [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --pages N      Number of pages (default: 5)");
                println!("  --output FILE  Output PDF path");
                println!("  --help, -h     Show this help");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   Multi-Page JBIG2 Symbol Mode PDF Generator             ║");
    println!("║   (With Global Dictionary for PDF Embedding)             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    let width = 400;
    let height = 550;

    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("Generating {} text pages ({}x{})", num_pages, width, height);
    println!("Output: {}", output_path);
    println!();

    // Encode all pages together to build a shared global dictionary
    println!("Encoding with JBIG2 symbol mode (building global dictionary)...");

    let mut cfg = Jbig2Config::text();
    cfg.want_full_headers = false; // PDF mode
    cfg.auto_thresh = true; // Enable automatic thresholding for symbol clustering
    cfg.symbol_mode = true; // Explicitly enable symbol mode

    let mut encoder = Jbig2Encoder::new(&cfg);
    
    for page_num in 0..num_pages {
        let array = create_text_page(page_num, width, height);
        encoder.add_page(&array)?;
        println!("  Added page {} ({}x{})", page_num + 1, width, height);
    }
    
    let split = encoder.flush_pdf_split()?;
    let metrics = encoder.metrics_snapshot();

    println!();
    println!("Symbol mode statistics:");
    println!("  Symbols discovered: {}", metrics.symbol_stats.symbols_discovered);
    println!("  Symbols exported: {}", metrics.symbol_stats.symbols_exported);
    println!("  Global symbols: {}", metrics.symbol_stats.global_symbol_count);
    println!("  Local symbols: {}", metrics.symbol_stats.local_symbol_count);
    println!("  Avg symbol reuse: {:.2}", metrics.symbol_stats.avg_symbol_reuse);
    
    let globals_bytes = split.global_segments.as_ref().map_or(0, |g| g.len());
    let total_page_bytes: usize = split.page_streams.iter().map(|p| p.len()).sum();
    
    println!();
    println!("Encoding summary:");
    println!("  Global dictionary: {} bytes", globals_bytes);
    println!("  Page streams: {} bytes total", total_page_bytes);
    if globals_bytes > 0 {
        println!("  ✓ Global dictionary created successfully");
    } else {
        println!("  ⚠ No global dictionary (symbol mode may not have triggered)");
    }
    println!();

    // Assemble PDF
    println!("Assembling PDF with embedded JBIG2...");
    
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut page_ids = Vec::new();
    
    // Create global dictionary object (shared across all pages)
    let globals_id = if let Some(ref globals) = split.global_segments {
        if !globals.is_empty() {
            // JBIG2Globals must be a bare stream per ISO 32000-1
            let global_stream = Stream::new(Dictionary::new(), globals.clone());
            let id = doc.add_object(global_stream);
            println!("  Created global dictionary object: {:?}", id);
            Some(id)
        } else {
            None
        }
    } else {
        None
    };

    for (page_idx, page_data) in split.page_streams.iter().enumerate() {
        let page_id = doc.new_object_id();
        let mut page_dict = Dictionary::new();
        page_dict.set("Type", Object::Name(b"Page".to_vec()));
        page_dict.set("Parent", Object::Reference(pages_id));
        page_dict.set(
            "MediaBox",
            vec![
                Object::Real(0.0),
                Object::Real(0.0),
                Object::Real(width as f32),
                Object::Real(height as f32),
            ],
        );

        // Create JBIG2 image XObject
        let mut img_dict = Dictionary::new();
        img_dict.set("Type", Object::Name(b"XObject".to_vec()));
        img_dict.set("Subtype", Object::Name(b"Image".to_vec()));
        img_dict.set("Width", Object::Integer(width as i64));
        img_dict.set("Height", Object::Integer(height as i64));
        img_dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        img_dict.set("BitsPerComponent", Object::Integer(1));
        img_dict.set("Filter", Object::Name(b"JBIG2Decode".to_vec()));
        img_dict.set("Decode", vec![0.into(), 1.into()]);

        // CRITICAL: Reference the global dictionary via DecodeParms
        // This is how PDF readers know to use the shared symbol dictionary
        if let Some(gid) = globals_id {
            let mut decode_parms = Dictionary::new();
            decode_parms.set("JBIG2Globals", Object::Reference(gid));
            img_dict.set("DecodeParms", Object::Dictionary(decode_parms));
            println!("  Page {}: linked to global dictionary", page_idx + 1);
        }

        let img_id = doc.add_object(Stream::new(img_dict, page_data.clone()));
        
        // Create content stream
        let mut content_ops = Vec::new();
        content_ops.push(lopdf::content::Operation::new("q", vec![]));
        content_ops.push(lopdf::content::Operation::new(
            "cm",
            vec![
                Object::Real(width as f32),
                Object::Real(0.0),
                Object::Real(0.0),
                Object::Real(height as f32),
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
        (b"Count".to_vec(), Object::Integer(num_pages as i64)),
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
    println!("║  ✓ PDF successfully generated                            ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Output: {}", output_path);
    println!();
    println!("PDF Structure:");
    if globals_id.is_some() {
        println!("  • 1 global dictionary object (shared by all pages)");
    }
    println!("  • {} page objects with JBIG2-encoded images", num_pages);
    println!();
    println!("To test in Okular:");
    println!("  okular {}", output_path);
    println!();
    println!("To verify JBIG2 streams:");
    println!("  pdfinfo {}", output_path);

    Ok(())
}
