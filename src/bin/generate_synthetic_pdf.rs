//! Generate test PDFs with synthetic text for JBIG2 symbol mode testing
//!
//! Creates synthetic text pages with repeated characters to exercise
//! symbol mode encoding with a global dictionary.

use jbig2enc_rust::jbig2enc::Jbig2Encoder;
use jbig2enc_rust::jbig2structs::Jbig2Config;
use lopdf::{Dictionary, Document, Object, Stream};
use std::path::Path;

/// Create a synthetic text-like image with repeated characters
fn create_synthetic_text_page(_page_num: usize, width: u32, height: u32) -> ndarray::Array2<u8> {
    let mut array = ndarray::Array2::from_elem((height as usize, width as usize), 0u8);
    
    // Define character patterns (5x7 pixel glyphs)
    let char_patterns = vec![
        // 'H'
        vec![(0,0), (0,4), (1,0), (1,4), (2,0), (2,1), (2,2), (2,3), (2,4), (3,0), (3,4), (4,0), (4,4)],
        // 'E'
        vec![(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (2,0), (2,1), (2,2), (2,3), (2,4), (3,0), (4,0), (4,1), (4,2), (4,3), (4,4)],
        // 'L'
        vec![(0,0), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3), (4,4)],
        // 'O'
        vec![(0,1), (0,2), (0,3), (1,0), (1,4), (2,0), (2,4), (3,0), (3,4), (4,1), (4,2), (4,3)],
        // 'W'
        vec![(0,0), (0,4), (1,0), (1,4), (2,0), (2,2), (2,4), (3,0), (3,1), (3,3), (3,4), (4,0), (4,4)],
        // 'R'
        vec![(0,0), (0,1), (0,2), (0,3), (1,0), (1,4), (2,0), (2,2), (2,3), (3,0), (3,4), (4,0), (4,4)],
    ];
    
    let chars_per_line = 25;
    let lines_per_page = 35;
    let char_width = 6;
    let char_height = 8;
    let start_x = 30;
    let start_y = 30;
    
    // Create text "HELLO WORLD" repeated with variation per page
    let text: Vec<usize> = (0..chars_per_line * lines_per_page)
        .map(|i| {
            let word_pos = i % 11;
            match word_pos {
                0..=4 => word_pos, // HELLO
                5 => 4, // space (use O as placeholder)
                6..=10 => word_pos - 5, // WORLD
                _ => 0,
            }
        })
        .collect();
    
    for (line_idx, line_start) in text.chunks(chars_per_line).enumerate() {
        if line_idx >= lines_per_page {
            break;
        }
        for (char_idx, &pattern_idx) in line_start.iter().enumerate() {
            let base_x = start_x + (char_idx * char_width) as u32;
            let base_y = start_y + (line_idx * char_height) as u32;
            
            for &(dx, dy) in &char_patterns[pattern_idx] {
                let x = base_x + dx;
                let y = base_y + dy;
                if x < width && y < height {
                    array[[y as usize, x as usize]] = 255;
                }
            }
        }
    }
    
    array
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    let mut num_pages = 3;
    let mut output_path = "test_output_pdfs/synthetic_text_symbol_mode.pdf".to_string();

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
                println!("Generate synthetic text PDFs with JBIG2 symbol mode");
                println!();
                println!("Usage: cargo run --release --bin generate_synthetic_pdf [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --pages N      Number of pages (default: 3)");
                println!("  --output FILE  Output PDF path");
                println!("  --help, -h     Show this help");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║   Synthetic Text JBIG2 Symbol Mode PDF Generator         ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    let width = 300;
    let height = 500;

    // Create output directory
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("Generating {} synthetic text pages ({}x{})", num_pages, width, height);
    println!("Output: {}", output_path);
    println!();

    // Generate and encode pages
    println!("Encoding with JBIG2 symbol mode...");
    
    struct EncodedPage {
        page_data: Vec<u8>,
        global_data: Option<Vec<u8>>,
    }
    
    let mut encoded_pages: Vec<EncodedPage> = Vec::new();
    let mut total_global_bytes = 0;
    let mut total_page_bytes = 0;

    for page_num in 0..num_pages {
        let array = create_synthetic_text_page(page_num, width, height);
        
        // Configure for symbol mode with PDF embedding
        let mut cfg = Jbig2Config::text();
        cfg.want_full_headers = false;

        let mut encoder = Jbig2Encoder::new(&cfg);
        encoder.add_page(&array)?;
        let split = encoder.flush_pdf_split()?;

        let page_data = split.page_streams.first().cloned().unwrap_or_default();
        let global_data = split.global_segments.clone();
        let data_len = page_data.len();

        if let Some(ref g) = global_data {
            total_global_bytes += g.len();
        }
        total_page_bytes += page_data.len();

        encoded_pages.push(EncodedPage { page_data, global_data });
        println!("  Page {}: {} bytes", page_num + 1, data_len);
    }

    let metrics = encoded_pages.first().map(|_| {
        // Re-encode one page to get metrics
        let array = create_synthetic_text_page(0, width, height);
        let mut cfg = Jbig2Config::text();
        cfg.want_full_headers = false;
        let mut encoder = Jbig2Encoder::new(&cfg);
        let _ = encoder.add_page(&array);
        let _ = encoder.flush_pdf_split();
        encoder.metrics_snapshot()
    });

    if let Some(m) = metrics {
        println!();
        println!("Symbol mode stats:");
        println!("  Symbols discovered: {}", m.symbol_stats.symbols_discovered);
        println!("  Symbols exported: {}", m.symbol_stats.symbols_exported);
        println!("  Global symbols: {}", m.symbol_stats.global_symbol_count);
        println!("  Avg reuse: {:.2}", m.symbol_stats.avg_symbol_reuse);
    }

    println!();
    println!("Encoding summary:");
    println!("  Global dict bytes: {}", total_global_bytes);
    println!("  Page stream bytes: {}", total_page_bytes);
    println!();

    // Assemble PDF
    println!("Assembling PDF...");
    
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut page_ids = Vec::new();
    
    // Create global dictionary object
    let globals_id = if total_global_bytes > 0 {
        let mut all_globals = Vec::new();
        for page in &encoded_pages {
            if let Some(ref g) = page.global_data {
                all_globals.push(g.clone());
            }
        }
        
        if !all_globals.is_empty() {
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
                Object::Real(width as f32),
                Object::Real(height as f32),
            ],
        );

        // Create image XObject
        let mut img_dict = Dictionary::new();
        img_dict.set("Type", Object::Name(b"XObject".to_vec()));
        img_dict.set("Subtype", Object::Name(b"Image".to_vec()));
        img_dict.set("Width", Object::Integer(width as i64));
        img_dict.set("Height", Object::Integer(height as i64));
        img_dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        img_dict.set("BitsPerComponent", Object::Integer(1));
        img_dict.set("Filter", Object::Name(b"JBIG2Decode".to_vec()));
        img_dict.set("Decode", vec![0.into(), 1.into()]);

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
    println!("✓ PDF written to: {}", output_path);
    println!();
    if total_global_bytes > 0 {
        println!("This PDF uses JBIG2 symbol mode with a global dictionary.");
    } else {
        println!("Note: This PDF uses generic region encoding (no symbol dictionary).");
        println!("      The synthetic text may be too simple for symbol extraction.");
    }
    println!();
    println!("To test in Okular:");
    println!("  okular {}", output_path);

    Ok(())
}
