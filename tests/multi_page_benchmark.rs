//! Multi-page JBIG2 compression benchmark
//!
//! Measures encode performance and compression for generic vs symbol modes.
//! Separates encode time from PDF packaging and disk I/O.
//!
//! Run with:
//!   cargo test --test multi_page_benchmark --features symboldict -- --nocapture
//!
//! Environment variables:
//!   BENCH_PAGES   — comma-separated page counts (default: "10,20,50")
//!                   example: BENCH_PAGES=1,5,10,20,50,100,all
//!   BENCH_WRITE   — set to "1" to write PDFs to disk (default: memory only)

use jbig2enc_rust::jbig2enc::{EncoderMetrics, Jbig2Encoder, PdfSplitOutput};
use jbig2enc_rust::jbig2structs::Jbig2Config;
use lopdf::dictionary;
use lopdf::{Document, Object, Stream};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

// ── Data types ───────────────────────────────────────────────────────

struct PageInfo {
    array: ndarray::Array2<u8>,
    width: u32,
    height: u32,
    pbm_bytes: u64,
}

struct EncodeResult {
    split: PdfSplitOutput,
    encode_secs: f64,
    metrics: EncoderMetrics,
}

struct BenchRow {
    pages: usize,
    mode: String,
    encode_secs: f64,
    ms_per_page: f64,
    mpix_per_s: f64,
    raw_jbig2_bytes: usize,
    globals_bytes: usize,
    pdf_bytes: u64,
    savings_vs_generic: f64,
    savings_vs_pbm: f64,
    avg_page_kb: f64,
    min_page_kb: f64,
    max_page_kb: f64,
    cc_secs: f64,
    match_secs: f64,
    cluster_secs: f64,
    planning_secs: f64,
    dict_secs: f64,
    text_secs: f64,
    generic_secs: f64,
    symbols_discovered: usize,
    symbols_exported: usize,
    avg_symbol_reuse: f64,
    global_symbol_count: usize,
    local_symbol_count: usize,
}

// ── PBM loader (straight to Array2, no BitImage intermediate) ────────

fn load_pbm_to_array(path: &Path) -> (ndarray::Array2<u8>, u32, u32) {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open PBM {}: {}", path.display(), e));
    let mut reader = BufReader::new(file);

    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "P4", "Only raw PBM (P4) supported");

    let (width, height) = loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let dims: Vec<usize> = trimmed
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        if dims.len() == 2 {
            break (dims[0], dims[1]);
        }
    };

    let bytes_per_row = (width + 7) / 8;
    let mut packed = vec![0u8; bytes_per_row * height];
    reader.read_exact(&mut packed).unwrap();

    let array = ndarray::Array2::from_shape_fn((height, width), |(y, x)| {
        let byte = packed[y * bytes_per_row + x / 8];
        let bit = 7 - (x % 8);
        ((byte >> bit) & 1) as u8
    });

    (array, width as u32, height as u32)
}

// ── Encode helper (measures only encode, not PDF) ────────────────────

fn run_encode(cfg: &Jbig2Config, pages: &[PageInfo]) -> EncodeResult {
    let t0 = Instant::now();
    let mut enc = Jbig2Encoder::new(cfg);
    for p in pages {
        enc.add_page(&p.array).expect("add_page failed");
    }
    let split = enc.flush_pdf_split().expect("flush failed");
    EncodeResult {
        split,
        encode_secs: t0.elapsed().as_secs_f64(),
        metrics: enc.metrics_snapshot(),
    }
}

// ── Raw JBIG2 stream stats ───────────────────────────────────────────

fn raw_jbig2_stats(split: &PdfSplitOutput) -> (usize, usize, Vec<f64>) {
    let globals_bytes = split.global_segments.as_ref().map_or(0, |g| g.len());
    let page_sizes_kb: Vec<f64> = split
        .page_streams
        .iter()
        .map(|s| s.len() as f64 / 1024.0)
        .collect();
    let raw_pages: usize = split.page_streams.iter().map(|s| s.len()).sum();
    (raw_pages + globals_bytes, globals_bytes, page_sizes_kb)
}

// ── PDF builder ──────────────────────────────────────────────────────

fn build_multi_page_pdf(pages: &[(&[u8], u32, u32)], global_data: Option<&[u8]>) -> Document {
    let mut doc = Document::with_version("1.7");

    let globals_id = global_data.map(|gd| {
        let stream = Stream::new(lopdf::Dictionary::new(), gd.to_vec());
        doc.add_object(stream)
    });

    let mut page_ids = Vec::new();

    for &(page_data, width, height) in pages {
        let pt_w = ((width as f64 / 300.0) * 72.0) as f32;
        let pt_h = ((height as f64 / 300.0) * 72.0) as f32;

        let mut img_dict = lopdf::Dictionary::new();
        img_dict.set("Type", Object::Name(b"XObject".to_vec()));
        img_dict.set("Subtype", Object::Name(b"Image".to_vec()));
        img_dict.set("Width", Object::Integer(width as i64));
        img_dict.set("Height", Object::Integer(height as i64));
        img_dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        img_dict.set("BitsPerComponent", Object::Integer(1));
        img_dict.set("Filter", Object::Name(b"JBIG2Decode".to_vec()));
        img_dict.set("Decode", vec![Object::Integer(0), Object::Integer(1)]);

        if let Some(gid) = globals_id {
            let mut dp = lopdf::Dictionary::new();
            dp.set("JBIG2Globals", Object::Reference(gid));
            img_dict.set("DecodeParms", Object::Dictionary(dp));
        }

        let img_stream = Stream::new(img_dict, page_data.to_vec());
        let img_id = doc.add_object(img_stream);

        let xobject_dict = dictionary! { "Im0" => Object::Reference(img_id) };
        let resources = dictionary! { "XObject" => Object::Dictionary(xobject_dict) };

        let content_str = format!("q {:.2} 0 0 {:.2} 0 0 cm /Im0 Do Q", pt_w, pt_h);
        let content_id = doc.add_object(Stream::new(
            lopdf::Dictionary::new(),
            content_str.into_bytes(),
        ));

        let page_dict = dictionary! {
            "Type" => "Page",
            "MediaBox" => vec![0.into(), 0.into(), Object::Real(pt_w), Object::Real(pt_h)],
            "Contents" => Object::Reference(content_id),
            "Resources" => Object::Dictionary(resources),
        };
        let page_id = doc.add_object(page_dict);
        page_ids.push(page_id);
    }

    let kids: Vec<Object> = page_ids.iter().map(|id| Object::Reference(*id)).collect();
    let pages_dict = dictionary! {
        "Type" => "Pages",
        "Kids" => kids,
        "Count" => Object::Integer(page_ids.len() as i64),
    };
    let pages_id = doc.add_object(pages_dict);

    for &pid in &page_ids {
        if let Ok(Object::Dictionary(d)) = doc.get_object_mut(pid) {
            d.set("Parent", Object::Reference(pages_id));
        }
    }

    let catalog = dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id),
    };
    let catalog_id = doc.add_object(catalog);
    doc.trailer.set("Root", Object::Reference(catalog_id));
    doc
}

/// Build PDF in memory or save to disk. Returns byte size.
fn build_pdf_size(split: &PdfSplitOutput, pages: &[PageInfo], out_path: Option<&Path>) -> u64 {
    let page_tuples: Vec<(&[u8], u32, u32)> = split
        .page_streams
        .iter()
        .zip(pages.iter())
        .map(|(data, pi)| (data.as_slice(), pi.width, pi.height))
        .collect();

    let mut pdf = build_multi_page_pdf(&page_tuples, split.global_segments.as_deref());

    match out_path {
        Some(path) => {
            pdf.save(path).unwrap();
            std::fs::metadata(path).unwrap().len()
        }
        None => {
            let mut buf = Vec::new();
            pdf.save_to(&mut buf).unwrap();
            buf.len() as u64
        }
    }
}

// ── Config helpers ───────────────────────────────────────────────────

fn parse_page_counts(total_pages: usize) -> Vec<usize> {
    if let Ok(val) = std::env::var("BENCH_PAGES") {
        let mut counts: Vec<usize> = val
            .split(',')
            .map(|s| s.trim())
            .map(|s| {
                if s.eq_ignore_ascii_case("all") {
                    total_pages
                } else {
                    s.parse()
                        .unwrap_or_else(|_| panic!("Invalid BENCH_PAGES value: {s}"))
                }
            })
            .collect();
        counts.sort_unstable();
        counts.dedup();
        counts.retain(|&c| c >= 1 && c <= total_pages);
        counts
    } else {
        let mut counts = vec![10, 20, 50];
        counts.retain(|&c| c <= total_pages);
        counts
    }
}

fn should_write_pdfs() -> bool {
    std::env::var("BENCH_WRITE").map_or(false, |v| v == "1")
}

fn configs_for_count(count: usize, total_pixels: u64) -> Vec<(&'static str, Jbig2Config)> {
    let mut cfgs = vec![];

    let mut cfg_no_at = Jbig2Config::text();
    cfg_no_at.auto_thresh = false;
    cfg_no_at.want_full_headers = false;
    cfgs.push(("sym_no_at", cfg_no_at));

    let _ = (count, total_pixels);
    let mut cfg_at = Jbig2Config::text();
    cfg_at.want_full_headers = false;
    cfgs.push(("sym_at", cfg_at));

    cfgs
}

// ── Main benchmark ───────────────────────────────────────────────────

#[test]
fn multi_page_compression_benchmark() {
    let confed_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("confed");
    assert!(confed_dir.exists(), "confed/ directory not found");

    let mut pbm_files: Vec<PathBuf> = std::fs::read_dir(&confed_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "pbm"))
        .collect();
    pbm_files.sort();

    let total_pages = pbm_files.len();
    assert!(
        total_pages >= 10,
        "Need at least 10 PBM pages, found {}",
        total_pages
    );

    // ── Load dataset ─────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║      Multi-Page JBIG2 Compression Benchmark                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Loading {} PBM files...", total_pages);
    let load_start = Instant::now();
    let all_pages: Vec<PageInfo> = pbm_files
        .iter()
        .map(|p| {
            let pbm_bytes = std::fs::metadata(p).unwrap().len();
            let (array, w, h) = load_pbm_to_array(p);
            PageInfo {
                array,
                width: w,
                height: h,
                pbm_bytes,
            }
        })
        .collect();
    println!("Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    let page_counts = parse_page_counts(total_pages);
    let write_pdfs = should_write_pdfs();

    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_output_pdfs")
        .join(format!("benchmark_{ts}"));
    if write_pdfs {
        std::fs::create_dir_all(&out_dir).unwrap();
    }

    println!("Source: confed/ ({total_pages} pages)  |  Write PDFs: {write_pdfs}");
    println!("Page counts: {:?}\n", page_counts);

    // ── Warmup ───────────────────────────────────────────────────
    {
        let n = std::cmp::min(3, all_pages.len());
        let mut warm_cfg = Jbig2Config::lossless();
        warm_cfg.want_full_headers = false;
        let mut warm = Jbig2Encoder::new(&warm_cfg);
        for p in &all_pages[..n] {
            let _ = warm.add_page(&p.array);
        }
        let _ = warm.flush_pdf_split();
    }

    // ── Table header ─────────────────────────────────────────────
    println!(
        "{:<6} {:<10} {:>10} {:>10} {:>10} {:>8} {:>10} {:>8} {:>8} {:>8}",
        "Pages",
        "Mode",
        "Raw KB",
        "Globals",
        "PDF KB",
        "vs Gen",
        "Enc (s)",
        "ms/pg",
        "MPix/s",
        "vs PBM"
    );
    println!("{}", "─".repeat(102));

    let mut all_rows: Vec<BenchRow> = Vec::new();

    for &count in &page_counts {
        let pages = &all_pages[..count];
        let total_pixels: u64 = pages.iter().map(|p| p.width as u64 * p.height as u64).sum();
        let total_pbm_bytes: u64 = pages.iter().map(|p| p.pbm_bytes).sum();

        // ── Generic baseline ─────────────────────────────────────
        let mut cfg_generic = Jbig2Config::lossless();
        cfg_generic.want_full_headers = false;
        let gen_result = run_encode(&cfg_generic, pages);

        assert_eq!(
            gen_result.split.page_streams.len(),
            pages.len(),
            "generic: wrong page stream count"
        );

        let (gen_raw, gen_globals, gen_page_kb) = raw_jbig2_stats(&gen_result.split);
        let gen_pdf_path = out_dir.join(format!("generic_{count}p.pdf"));
        let gen_pdf_bytes = build_pdf_size(
            &gen_result.split,
            pages,
            if write_pdfs {
                Some(&gen_pdf_path)
            } else {
                None
            },
        );
        assert!(gen_pdf_bytes > 0, "generic PDF is empty");

        let gen_avg_page = gen_page_kb.iter().sum::<f64>() / gen_page_kb.len() as f64;
        let gen_min_page = gen_page_kb.iter().cloned().fold(f64::MAX, f64::min);
        let gen_max_page = gen_page_kb.iter().cloned().fold(0.0_f64, f64::max);
        let gen_ms = (gen_result.encode_secs * 1000.0) / count as f64;
        let gen_mpix = (total_pixels as f64 / 1_000_000.0) / gen_result.encode_secs;
        let gen_vs_pbm = (1.0 - gen_raw as f64 / total_pbm_bytes as f64) * 100.0;

        let gen_row = BenchRow {
            pages: count,
            mode: "generic".into(),
            encode_secs: gen_result.encode_secs,
            ms_per_page: gen_ms,
            mpix_per_s: gen_mpix,
            raw_jbig2_bytes: gen_raw,
            globals_bytes: gen_globals,
            pdf_bytes: gen_pdf_bytes,
            savings_vs_generic: 0.0,
            savings_vs_pbm: gen_vs_pbm,
            avg_page_kb: gen_avg_page,
            min_page_kb: gen_min_page,
            max_page_kb: gen_max_page,
            cc_secs: gen_result.metrics.symbol_mode.cc_extraction.as_secs_f64(),
            match_secs: gen_result.metrics.symbol_mode.matching_dedup.as_secs_f64(),
            cluster_secs: gen_result.metrics.symbol_mode.clustering.as_secs_f64(),
            planning_secs: gen_result.metrics.symbol_mode.planning.as_secs_f64(),
            dict_secs: gen_result
                .metrics
                .symbol_mode
                .symbol_dict_encoding
                .as_secs_f64(),
            text_secs: gen_result
                .metrics
                .symbol_mode
                .text_region_encoding
                .as_secs_f64(),
            generic_secs: gen_result
                .metrics
                .symbol_mode
                .generic_region_encoding
                .as_secs_f64(),
            symbols_discovered: gen_result.metrics.symbol_stats.symbols_discovered,
            symbols_exported: gen_result.metrics.symbol_stats.symbols_exported,
            avg_symbol_reuse: gen_result.metrics.symbol_stats.avg_symbol_reuse,
            global_symbol_count: gen_result.metrics.symbol_stats.global_symbol_count,
            local_symbol_count: gen_result.metrics.symbol_stats.local_symbol_count,
        };

        println!(
            "{:<6} {:<10} {:>9.1} {:>9.1} {:>9.1} {:>7}  {:>9.2} {:>7.1} {:>7.1} {:>7.1}%",
            count,
            "generic",
            gen_raw as f64 / 1024.0,
            gen_globals as f64 / 1024.0,
            gen_pdf_bytes as f64 / 1024.0,
            "—",
            gen_result.encode_secs,
            gen_ms,
            gen_mpix,
            gen_vs_pbm,
        );
        all_rows.push(gen_row);

        // ── Symbol modes ─────────────────────────────────────────
        let configs = configs_for_count(count, total_pixels);
        for (label, cfg) in &configs {
            let result = run_encode(cfg, pages);

            assert_eq!(
                result.split.page_streams.len(),
                pages.len(),
                "{label}: wrong page stream count"
            );

            let (raw_total, globals_bytes, page_kb) = raw_jbig2_stats(&result.split);
            let sym_pdf_path = out_dir.join(format!("{label}_{count}p.pdf"));
            let pdf_bytes = build_pdf_size(
                &result.split,
                pages,
                if write_pdfs {
                    Some(&sym_pdf_path)
                } else {
                    None
                },
            );

            let savings_gen = (1.0 - raw_total as f64 / gen_raw as f64) * 100.0;
            let savings_pbm = (1.0 - raw_total as f64 / total_pbm_bytes as f64) * 100.0;
            let ms = (result.encode_secs * 1000.0) / count as f64;
            let mpix = (total_pixels as f64 / 1_000_000.0) / result.encode_secs;
            let avg_page = page_kb.iter().sum::<f64>() / page_kb.len() as f64;
            let min_page = page_kb.iter().cloned().fold(f64::MAX, f64::min);
            let max_page = page_kb.iter().cloned().fold(0.0_f64, f64::max);

            println!(
                "{:<6} {:<10} {:>9.1} {:>9.1} {:>9.1} {:>6.1}%  {:>9.2} {:>7.1} {:>7.1} {:>7.1}%",
                count,
                label,
                raw_total as f64 / 1024.0,
                globals_bytes as f64 / 1024.0,
                pdf_bytes as f64 / 1024.0,
                savings_gen,
                result.encode_secs,
                ms,
                mpix,
                savings_pbm,
            );

            all_rows.push(BenchRow {
                pages: count,
                mode: label.to_string(),
                encode_secs: result.encode_secs,
                ms_per_page: ms,
                mpix_per_s: mpix,
                raw_jbig2_bytes: raw_total,
                globals_bytes,
                pdf_bytes,
                savings_vs_generic: savings_gen,
                savings_vs_pbm: savings_pbm,
                avg_page_kb: avg_page,
                min_page_kb: min_page,
                max_page_kb: max_page,
                cc_secs: result.metrics.symbol_mode.cc_extraction.as_secs_f64(),
                match_secs: result.metrics.symbol_mode.matching_dedup.as_secs_f64(),
                cluster_secs: result.metrics.symbol_mode.clustering.as_secs_f64(),
                planning_secs: result.metrics.symbol_mode.planning.as_secs_f64(),
                dict_secs: result
                    .metrics
                    .symbol_mode
                    .symbol_dict_encoding
                    .as_secs_f64(),
                text_secs: result
                    .metrics
                    .symbol_mode
                    .text_region_encoding
                    .as_secs_f64(),
                generic_secs: result
                    .metrics
                    .symbol_mode
                    .generic_region_encoding
                    .as_secs_f64(),
                symbols_discovered: result.metrics.symbol_stats.symbols_discovered,
                symbols_exported: result.metrics.symbol_stats.symbols_exported,
                avg_symbol_reuse: result.metrics.symbol_stats.avg_symbol_reuse,
                global_symbol_count: result.metrics.symbol_stats.global_symbol_count,
                local_symbol_count: result.metrics.symbol_stats.local_symbol_count,
            });
        }
        println!();
    }

    // ── Summary detail ───────────────────────────────────────────
    println!("─── Page stream detail ───");
    for row in &all_rows {
        if row.mode != "generic" {
            println!(
                "  {} {}p: globals {:.1}KB, pages avg {:.1}KB min {:.1}KB max {:.1}KB",
                row.mode,
                row.pages,
                row.globals_bytes as f64 / 1024.0,
                row.avg_page_kb,
                row.min_page_kb,
                row.max_page_kb
            );
            println!(
                "    stages cc {:.2}s match {:.2}s cluster {:.2}s plan {:.2}s dict {:.2}s text {:.2}s generic {:.2}s | symbols discovered {} exported {} reuse {:.2} global {} local {}",
                row.cc_secs,
                row.match_secs,
                row.cluster_secs,
                row.planning_secs,
                row.dict_secs,
                row.text_secs,
                row.generic_secs,
                row.symbols_discovered,
                row.symbols_exported,
                row.avg_symbol_reuse,
                row.global_symbol_count,
                row.local_symbol_count,
            );
        }
    }
    println!();

    // ── CSV output ───────────────────────────────────────────────
    if write_pdfs {
        let csv_path = out_dir.join("results.csv");
        let mut csv = String::from(
            "pages,mode,encode_secs,ms_per_page,mpix_per_s,raw_jbig2_bytes,globals_bytes,pdf_bytes,savings_vs_generic,savings_vs_pbm,cc_secs,match_secs,cluster_secs,planning_secs,dict_secs,text_secs,generic_secs,symbols_discovered,symbols_exported,avg_symbol_reuse,global_symbol_count,local_symbol_count\n",
        );
        for row in &all_rows {
            csv += &format!(
                "{},{},{:.4},{:.2},{:.2},{},{},{},{:.2},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{:.4},{},{}\n",
                row.pages,
                row.mode,
                row.encode_secs,
                row.ms_per_page,
                row.mpix_per_s,
                row.raw_jbig2_bytes,
                row.globals_bytes,
                row.pdf_bytes,
                row.savings_vs_generic,
                row.savings_vs_pbm,
                row.cc_secs,
                row.match_secs,
                row.cluster_secs,
                row.planning_secs,
                row.dict_secs,
                row.text_secs,
                row.generic_secs,
                row.symbols_discovered,
                row.symbols_exported,
                row.avg_symbol_reuse,
                row.global_symbol_count,
                row.local_symbol_count
            );
        }
        std::fs::write(&csv_path, csv).unwrap();
        println!("PDFs + CSV written to: {}", out_dir.display());
    } else {
        println!("(set BENCH_WRITE=1 to write PDFs and CSV to disk)");
    }
}
