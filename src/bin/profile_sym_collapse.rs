use clap::Parser;
use jbig2enc_rust::jbig2enc::Jbig2Encoder;
use jbig2enc_rust::jbig2structs::{Jbig2Config, LossyCollapsePrototypeMode};
use ndarray::Array2;
use pprof::ProfilerGuardBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "sahib")]
    source: String,
    #[arg(long, default_value_t = 50)]
    pages: usize,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    #[arg(long, default_value_t = 499)]
    frequency: i32,
    #[arg(long, default_value = "sym_collapse")]
    mode: String,
    #[arg(long, default_value = "cleanup")]
    prototype: String,
}

struct PageInfo {
    array: Array2<u8>,
}

fn load_pbm_to_array(path: &Path) -> (Array2<u8>, u32, u32) {
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

    let array = Array2::from_shape_fn((height, width), |(y, x)| {
        let byte = packed[y * bytes_per_row + x / 8];
        let bit = 7 - (x % 8);
        ((byte >> bit) & 1) as u8
    });

    (array, width as u32, height as u32)
}

fn collapse_proto_mode(name: &str) -> LossyCollapsePrototypeMode {
    match name.to_ascii_lowercase().as_str() {
        "medoid" => LossyCollapsePrototypeMode::Medoid,
        "majority" => LossyCollapsePrototypeMode::MajorityVote,
        "adaptive-majority" => LossyCollapsePrototypeMode::AdaptiveMajorityVote,
        "adaptive-cleanup" => LossyCollapsePrototypeMode::MedoidWithAdaptiveCleanup,
        _ => LossyCollapsePrototypeMode::MedoidThenCleanup,
    }
}

fn top_counts(report: &pprof::Report) -> (Vec<(String, isize)>, Vec<(String, isize)>) {
    let mut self_counts: HashMap<String, isize> = HashMap::new();
    let mut inclusive_counts: HashMap<String, isize> = HashMap::new();

    for (frames, count) in &report.data {
        if *count <= 0 {
            continue;
        }
        if let Some(leaf_syms) = frames.frames.first() {
            if let Some(sym) = leaf_syms.first() {
                *self_counts.entry(sym.name().to_owned()).or_default() += *count;
            }
        }
        for frame_syms in &frames.frames {
            if let Some(sym) = frame_syms.first() {
                *inclusive_counts.entry(sym.name().to_owned()).or_default() += *count;
            }
        }
    }

    let mut self_top: Vec<_> = self_counts.into_iter().collect();
    self_top.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    self_top.truncate(20);

    let mut inclusive_top: Vec<_> = inclusive_counts.into_iter().collect();
    inclusive_top.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    inclusive_top.truncate(20);

    (self_top, inclusive_top)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&args.source);
    anyhow::ensure!(source_dir.exists(), "source directory not found: {}", source_dir.display());

    let mut pbm_files: Vec<PathBuf> = std::fs::read_dir(&source_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "pbm"))
        .collect();
    pbm_files.sort();
    anyhow::ensure!(
        args.pages > 0 && args.pages <= pbm_files.len(),
        "requested {} pages, available {}",
        args.pages,
        pbm_files.len()
    );

    let pages: Vec<PageInfo> = pbm_files[..args.pages]
        .iter()
        .map(|p| {
            let (array, _, _) = load_pbm_to_array(p);
            PageInfo { array }
        })
        .collect();

    let mut cfg = match args.mode.as_str() {
        "symbol" => {
            let mut cfg = Jbig2Config::text();
            cfg.text_refine = false;
            cfg.refine = false;
            cfg
        }
        "sym_refine" => {
            let mut cfg = Jbig2Config::text();
            cfg.text_refine = true;
            cfg.refine = true;
            cfg
        }
        _ => {
            let mut cfg = Jbig2Config::text_lossy_collapse();
            cfg.lossy_collapse_prototype_mode = collapse_proto_mode(&args.prototype);
            cfg
        }
    };
    cfg.auto_thresh = false;
    cfg.want_full_headers = false;

    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_output_fragments")
        .join(format!("profile_{}_{}", args.mode, ts));
    std::fs::create_dir_all(&out_dir)?;

    let guard = ProfilerGuardBuilder::default()
        .frequency(args.frequency)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()?;

    let started = Instant::now();
    let mut final_metrics = None;
    let mut final_log = String::new();
    let mut final_raw_bytes = 0usize;

    for iter in 0..args.repeat {
        let mut enc = Jbig2Encoder::new(&cfg);
        for page in &pages {
            enc.add_page(&page.array)?;
        }
        let split = enc.flush_pdf_split()?;
        final_raw_bytes =
            split.global_segments.as_ref().map_or(0, |g| g.len()) + split.page_streams.iter().map(Vec::len).sum::<usize>();
        final_metrics = Some(enc.metrics_snapshot());
        final_log = enc.decision_debug_log();
        eprintln!("completed profiling iteration {}", iter + 1);
    }
    let elapsed = started.elapsed();

    let report = guard.report().build()?;
    let flamegraph_path = out_dir.join(format!("{}_flamegraph.svg", args.mode));
    let report_path = out_dir.join(format!("{}_profile.txt", args.mode));
    let decision_log_path = out_dir.join(format!("{}_decision_debug.log", args.mode));

    {
        let flamegraph = File::create(&flamegraph_path)?;
        report.flamegraph(flamegraph)?;
    }

    let (self_top, inclusive_top) = top_counts(&report);
    let metrics = final_metrics.expect("profiling run did not produce metrics");

    let mut text = String::new();
    text.push_str(&format!("{} profile\n", args.mode));
    text.push_str(&format!(
        "source={} pages={} repeat={} frequency={} mode={} prototype={:?}\n",
        args.source,
        args.pages,
        args.repeat,
        args.frequency,
        args.mode,
        cfg.lossy_collapse_prototype_mode
    ));
    text.push_str(&format!(
        "elapsed_secs={:.3} raw_jbig2_kb={:.1}\n",
        elapsed.as_secs_f64(),
        final_raw_bytes as f64 / 1024.0
    ));
    text.push_str(&format!(
        "stages: cc={:.3}s match={:.3}s cluster={:.3}s plan={:.3}s dict={:.3}s text={:.3}s generic={:.3}s\n",
        metrics.symbol_mode.cc_extraction.as_secs_f64(),
        metrics.symbol_mode.matching_dedup.as_secs_f64(),
        metrics.symbol_mode.clustering.as_secs_f64(),
        metrics.symbol_mode.planning.as_secs_f64(),
        metrics.symbol_mode.symbol_dict_encoding.as_secs_f64(),
        metrics.symbol_mode.text_region_encoding.as_secs_f64(),
        metrics.symbol_mode.generic_region_encoding.as_secs_f64(),
    ));
    text.push_str(&format!(
        "symbols: discovered={} exported={} reuse={:.2} global={} local={}\n",
        metrics.symbol_stats.symbols_discovered,
        metrics.symbol_stats.symbols_exported,
        metrics.symbol_stats.avg_symbol_reuse,
        metrics.symbol_stats.global_symbol_count,
        metrics.symbol_stats.local_symbol_count,
    ));
    text.push_str("\nTop self samples\n");
    for (name, count) in &self_top {
        text.push_str(&format!("{count:>8} {name}\n"));
    }
    text.push_str("\nTop inclusive samples\n");
    for (name, count) in &inclusive_top {
        text.push_str(&format!("{count:>8} {name}\n"));
    }
    text.push_str("\nRaw report\n");
    text.push_str(&format!("{report:?}\n"));

    std::fs::write(&report_path, text)?;
    std::fs::write(&decision_log_path, final_log)?;

    println!("profile written to {}", out_dir.display());
    println!("flamegraph: {}", flamegraph_path.display());
    println!("text report: {}", report_path.display());
    println!("decision log: {}", decision_log_path.display());

    Ok(())
}
