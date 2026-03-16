//! This module contains the main JBIG2 encoder logic.
use crate::jbig2arith::{IntProc, Jbig2ArithCoder};
use crate::jbig2comparator::{Comparator, MAX_DIMENSION_DELTA};
// Symbol extraction using CC analysis
#[cfg(feature = "cc-analysis")]
use crate::jbig2cc::analyze_page;
use crate::jbig2structs::{
    FileHeader, GenericRegionConfig, GenericRegionParams, Jbig2Config, PageInfo, Segment,
    SegmentType, SymbolDictParams, TextRegionParams,
};

use crate::jbig2sym::{BitImage, Rect};
use anyhow::{Result, anyhow};

// Define debug and trace macros at the crate root
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        #[cfg(feature = "trace_encoder")]
        log::debug!($($arg)*);

        #[cfg(not(feature = "trace_encoder"))]
        let _ = format_args!($($arg)*);
    };
}

#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        #[cfg(feature = "trace_encoder")]
        log::trace!($($arg)*);

        #[cfg(not(feature = "trace_encoder"))]
        let _ = format_args!($($arg)*);
    };
}

// Import the macros for use in this module
#[allow(unused_imports)]
use crate::{debug, trace};

use ndarray::Array2;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A key type for hashing bitmaps efficiently
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HashKey(u64);

#[derive(Debug, Clone, Copy, Default)]
struct SymbolSignature {
    black: u16,
    left_col: u16,
    right_col: u16,
    top_row: u16,
    bottom_row: u16,
    cx_times_256: u16,
    cy_times_256: u16,
}

#[derive(Debug)]
struct RecentSymbolCache {
    recent: VecDeque<usize>,
    cap: usize,
}

impl RecentSymbolCache {
    fn new(cap: usize) -> Self {
        Self {
            recent: VecDeque::with_capacity(cap),
            cap,
        }
    }

    fn clear(&mut self) {
        self.recent.clear();
    }

    fn touch(&mut self, idx: usize) {
        if let Some(pos) = self.recent.iter().position(|&entry| entry == idx) {
            self.recent.remove(pos);
        }
        self.recent.push_front(idx);
        while self.recent.len() > self.cap {
            self.recent.pop_back();
        }
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.recent.iter().copied()
    }
}

impl Hash for HashKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl std::fmt::Display for HashKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HashKey({:x})", self.0)
    }
}

/// A candidate symbol extracted from a document image.
#[derive(Debug, Clone)]
pub struct SymbolCandidate {
    /// The bitmap image of the symbol.
    pub bitmap: BitImage,
    /// The bounding box of the symbol in the original image.
    pub bbox: Rect,
}

/// Segment a document image into symbol candidates.
///
/// This function finds connected components in the input image and returns
/// them as symbol candidates. Each candidate has a bitmap and a bounding box.
///
/// # Arguments
/// * `image` - The input binary image to segment
/// * `dpi` - Resolution in dots per inch (typically 300 for scanned documents)
/// * `losslevel` - 0 for lossless, >0 to enable noise removal
pub fn segment_symbols(image: &BitImage, dpi: i32, losslevel: i32) -> Result<Vec<SymbolCandidate>> {
    #[cfg(feature = "cc-analysis")]
    {
        // Use the new CC analysis pipeline from jbig2cc
        let cc_image = analyze_page(image, dpi, losslevel);
        let shapes = cc_image.extract_shapes();

        let mut candidates = Vec::with_capacity(shapes.len());
        for (bitmap, bbox) in shapes {
            let rect = Rect {
                x: bbox.xmin as u32,
                y: bbox.ymin as u32,
                width: bbox.width() as u32,
                height: bbox.height() as u32,
            };
            candidates.push(SymbolCandidate { bitmap, bbox: rect });
        }
        Ok(candidates)
    }
    #[cfg(not(feature = "cc-analysis"))]
    {
        Err(anyhow!("Symbol segmentation requires cc-analysis feature"))
    }
}

// Jbig2EncConfig has been removed. Use jbig2structs::Jbig2Config directly.

#[derive(Clone)]
pub struct SymbolInstance {
    pub symbol_index: usize,
    pub position: Rect,
    pub instance_bitmap: BitImage,
    /// Whether this instance needs refinement coding (bitmap differs from prototype)
    pub needs_refinement: bool,
    /// Horizontal alignment offset for refinement (from Comparator)
    pub refinement_dx: i32,
    /// Vertical alignment offset for refinement (from Comparator)
    pub refinement_dy: i32,
}

impl SymbolInstance {
    pub fn symbol_index(&self) -> usize {
        self.symbol_index
    }

    pub fn position(&self) -> Rect {
        self.position
    }

    pub fn instance_bitmap(&self) -> &BitImage {
        &self.instance_bitmap
    }
}

#[derive(Clone)]
pub struct PageData {
    pub image: BitImage,
    pub symbol_instances: Vec<SymbolInstance>,
}

#[derive(Debug, Clone, Default)]
pub struct SymbolModeStageMetrics {
    pub cc_extraction: Duration,
    pub matching_dedup: Duration,
    pub clustering: Duration,
    pub planning: Duration,
    pub symbol_dict_encoding: Duration,
    pub text_region_encoding: Duration,
    pub generic_region_encoding: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct SymbolModeStats {
    pub symbols_discovered: usize,
    pub symbols_exported: usize,
    pub avg_symbol_reuse: f64,
    pub global_symbol_count: usize,
    pub local_symbol_count: usize,
    pub comparator_calls: usize,
    pub comparator_hits: usize,
    pub exact_hits: usize,
    pub refined_hits: usize,
    pub signature_rejects: usize,
}

#[derive(Debug, Clone, Default)]
pub struct EncoderMetrics {
    pub symbol_mode: SymbolModeStageMetrics,
    pub symbol_stats: SymbolModeStats,
}

#[derive(Debug, Clone)]
pub struct PdfSplitOutput {
    pub global_segments: Option<Vec<u8>>,
    pub page_streams: Vec<Vec<u8>>,
}

#[derive(Debug)]
struct PlannedPage {
    page_number: u32,
    segments: Vec<Segment>,
}

#[derive(Debug)]
struct PlannedDocument {
    file_header: Option<FileHeader>,
    global_segments: Vec<Segment>,
    pages: Vec<PlannedPage>,
    eof_segment: Option<Segment>,
    next_segment_number: u32,
}

#[derive(Debug, Clone)]
struct PlannedPageLayout {
    page_index: usize,
    page_number: u32,
    page_info_segment_number: u32,
    local_dict_segment_number: Option<u32>,
    region_segment_number: u32,
    end_of_page_segment_number: u32,
    local_symbols: Vec<usize>,
    use_generic_region: bool,
}

#[derive(Debug)]
struct BuiltPage {
    page: PlannedPage,
    symbol_dict_time: Duration,
    text_region_time: Duration,
    generic_region_time: Duration,
}

/// Mutable state for the encoder that can change during encoding.
#[derive(Debug, Default)]
struct EncoderState {
    pdf_mode: bool,
    full_headers_remaining: bool,
    segment: bool,
    use_refinement: bool,
    use_delta_encoding: bool,
}

/// Main JBIG2 encoder that handles document encoding
///
/// This struct manages the encoding state and configuration for JBIG2 documents.
/// It supports both symbol-based and generic region encoding strategies.
pub struct Jbig2Encoder<'a> {
    /// Configuration for the encoder
    config: &'a Jbig2Config,

    /// Internal encoder state
    state: EncoderState,

    /// Global symbols (shared across pages)
    global_symbols: Vec<BitImage>,

    /// Usage count for each global symbol
    symbol_usage: Vec<usize>,

    /// Black pixel count cache for each global symbol (for fast pre-filtering)
    symbol_pixel_counts: Vec<usize>,

    /// Cheap structural signatures used to reject bad matches before full comparison
    symbol_signatures: Vec<SymbolSignature>,

    /// Number of distinct pages where each symbol appears
    symbol_page_count: Vec<usize>,

    /// Last page where the symbol was seen, used to deduplicate per-page membership updates
    symbol_last_page_seen: Vec<Option<usize>>,

    /// Hash map for quick symbol lookup
    hash_map: HashMap<HashKey, Vec<usize>>,

    /// Page data for each page in the document
    pages: Vec<PageData>,

    /// Per-page unique symbol indices, built incrementally during extraction
    page_symbol_indices: Vec<Vec<usize>>,

    /// Next available segment number
    next_segment_number: u32,

    /// Segment number of the global dictionary (if any)
    global_dict_segment_number: Option<u32>,

    /// Encoder metrics used by the benchmark harness
    metrics: EncoderMetrics,
}

impl<'a> Jbig2Encoder<'a> {
    /// Creates a new JBIG2 encoder with the specified configuration
    ///
    /// # Arguments
    /// * `config` - Configuration for the encoder
    pub fn new(config: &'a Jbig2Config) -> Self {
        if config.refine && !config.symbol_mode {
            panic!("Refinement requires symbol mode to be enabled.");
        }

        Self {
            config,
            state: EncoderState {
                pdf_mode: false, // start in raw mode
                full_headers_remaining: config.want_full_headers,
                segment: true,                 // Default to using segments
                use_refinement: config.refine, // Enable refinement based on config
                use_delta_encoding: true,      // Default to using delta encoding
            },
            global_symbols: Vec::new(),
            symbol_usage: Vec::new(),
            symbol_pixel_counts: Vec::new(),
            symbol_signatures: Vec::new(),
            symbol_page_count: Vec::new(),
            symbol_last_page_seen: Vec::new(),
            hash_map: HashMap::new(),
            pages: Vec::new(),
            page_symbol_indices: Vec::new(),
            next_segment_number: 0,
            global_dict_segment_number: None,
            metrics: EncoderMetrics::default(),
        }
    }

    pub fn dict_only(mut self) -> Self {
        self.state.full_headers_remaining = false;
        self.state.pdf_mode = true;
        self
    }

    /// Returns the number of pages currently added to the encoder
    pub fn get_page_count(&self) -> usize {
        self.pages.len()
    }

    pub fn metrics_snapshot(&self) -> EncoderMetrics {
        self.metrics.clone()
    }

    /// Returns debug information about symbol usage
    pub fn get_symbol_stats(&self) -> String {
        let total_symbols = self.global_symbols.len();
        let avg_usage = if total_symbols > 0 {
            self.symbol_usage.iter().sum::<usize>() as f32 / total_symbols as f32
        } else {
            0.0
        };
        let low_usage_count = self.symbol_usage.iter().filter(|&&u| u < 2).count();

        format!(
            "Total symbols: {}, Average usage: {:.1}, Low usage (<2): {}",
            total_symbols, avg_usage, low_usage_count
        )
    }

    fn compute_symbol_signature(img: &BitImage) -> SymbolSignature {
        let mut black = 0usize;
        let mut left_col = img.width;
        let mut right_col = 0usize;
        let mut top_row = img.height;
        let mut bottom_row = 0usize;
        let mut sum_x = 0usize;
        let mut sum_y = 0usize;

        for y in 0..img.height {
            for x in 0..img.width {
                if img.get_usize(x, y) {
                    black += 1;
                    left_col = left_col.min(x);
                    right_col = right_col.max(x);
                    top_row = top_row.min(y);
                    bottom_row = bottom_row.max(y);
                    sum_x += x;
                    sum_y += y;
                }
            }
        }

        let (cx, cy) = if black == 0 {
            (0, 0)
        } else {
            (
                ((sum_x * 256) / black).min(u16::MAX as usize) as u16,
                ((sum_y * 256) / black).min(u16::MAX as usize) as u16,
            )
        };

        SymbolSignature {
            black: black.min(u16::MAX as usize) as u16,
            left_col: left_col.min(u16::MAX as usize) as u16,
            right_col: right_col.min(u16::MAX as usize) as u16,
            top_row: top_row.min(u16::MAX as usize) as u16,
            bottom_row: bottom_row.min(u16::MAX as usize) as u16,
            cx_times_256: cx,
            cy_times_256: cy,
        }
    }

    fn signatures_are_compatible(
        &self,
        candidate: SymbolSignature,
        symbol_index: usize,
        refine: bool,
    ) -> bool {
        let stored = self.symbol_signatures[symbol_index];
        let black_tol = if refine { 12 } else { 8 };
        let pos_tol = if refine { 2 } else { 2 };
        let centroid_tol = if refine { 96 } else { 64 };

        candidate.black.abs_diff(stored.black) <= black_tol
            && candidate.left_col.abs_diff(stored.left_col) <= pos_tol
            && candidate.right_col.abs_diff(stored.right_col) <= pos_tol
            && candidate.top_row.abs_diff(stored.top_row) <= pos_tol
            && candidate.bottom_row.abs_diff(stored.bottom_row) <= pos_tol
            && candidate.cx_times_256.abs_diff(stored.cx_times_256) <= centroid_tol
            && candidate.cy_times_256.abs_diff(stored.cy_times_256) <= centroid_tol
    }

    fn should_skip_symbol_candidate(width: usize, height: usize, black_pixels: usize) -> bool {
        if width == 0 || height == 0 || black_pixels <= 1 {
            return true;
        }
        if (width >= 64 && height <= 2) || (height >= 64 && width <= 2) {
            return true;
        }
        if width > 256 || height > 256 {
            return true;
        }

        let area = width.saturating_mul(height).max(1);
        let density = black_pixels as f32 / area as f32;
        !(0.01..=0.90).contains(&density)
    }

    fn should_accept_match(
        &self,
        err: u32,
        dx: i32,
        dy: i32,
        exact_dims: bool,
        max_err: u32,
    ) -> (bool, bool) {
        if err == 0 && dx == 0 && dy == 0 && exact_dims {
            return (true, false);
        }

        if self.config.text_refine {
            if dx.abs() <= 1 && dy.abs() <= 1 && err <= (max_err / 2).max(2) {
                return (true, true);
            }
            return (false, false);
        }

        if dx.abs() <= 1 && dy == 0 {
            return (true, false);
        }

        (false, false)
    }

    fn evaluate_symbol_match(
        &mut self,
        candidate: &BitImage,
        candidate_sig: SymbolSignature,
        candidate_pixels: usize,
        symbol_index: usize,
        comparator: &mut Comparator,
        max_err: u32,
    ) -> Option<(u32, i32, i32, bool)> {
        let proto = &self.global_symbols[symbol_index];
        let dim_limit = if self.config.text_refine { 2 } else { 0 };
        if (candidate.width as i32 - proto.width as i32).unsigned_abs() > dim_limit
            || (candidate.height as i32 - proto.height as i32).unsigned_abs() > dim_limit
        {
            return None;
        }
        if self.symbol_pixel_counts[symbol_index].abs_diff(candidate_pixels)
            > max_err as usize + if self.config.text_refine { 8 } else { 6 }
        {
            return None;
        }
        if !self.signatures_are_compatible(candidate_sig, symbol_index, self.config.text_refine) {
            self.metrics.symbol_stats.signature_rejects += 1;
            return None;
        }

        self.metrics.symbol_stats.comparator_calls += 1;
        let (err, dx, dy) = comparator.distance(candidate, proto, max_err)?;
        self.metrics.symbol_stats.comparator_hits += 1;

        let exact_dims = candidate.width == proto.width && candidate.height == proto.height;
        let (accept, needs_refinement) = self.should_accept_match(err, dx, dy, exact_dims, max_err);
        if !accept {
            return None;
        }

        if needs_refinement {
            self.metrics.symbol_stats.refined_hits += 1;
        } else if err == 0 && dx == 0 && dy == 0 && exact_dims {
            self.metrics.symbol_stats.exact_hits += 1;
        }

        Some((err, dx, dy, needs_refinement))
    }

    fn estimate_local_symbol_gain(&self, page: &PageData, symbol_index: usize) -> i64 {
        let uses = page
            .symbol_instances
            .iter()
            .filter(|instance| instance.symbol_index == symbol_index)
            .count() as i64;
        let symbol = &self.global_symbols[symbol_index];
        let area = (symbol.width * symbol.height) as i64;
        let dict_cost = 24 + (area / 8);
        let saved_per_use = (area / 10).max(2);
        (uses * saved_per_use) - dict_cost
    }

    fn choose_cluster_prototype(&self, members: &[usize]) -> usize {
        if members.len() <= 1 || !self.config.text_refine {
            return *members
                .iter()
                .max_by(|&&lhs, &&rhs| {
                    self.symbol_usage[lhs]
                        .cmp(&self.symbol_usage[rhs])
                        .then_with(|| {
                            self.symbol_pixel_counts[lhs].cmp(&self.symbol_pixel_counts[rhs])
                        })
                        .then_with(|| rhs.cmp(&lhs))
                })
                .unwrap();
        }

        let mut comparator = Comparator::default();
        let mut best_idx = members[0];
        let mut best_cost = u64::MAX;

        for &candidate in members {
            let candidate_symbol = &self.global_symbols[candidate];
            let mut total_cost = 0u64;
            for &other in members {
                if candidate == other {
                    continue;
                }
                let other_symbol = &self.global_symbols[other];
                let area = candidate_symbol.width.max(other_symbol.width)
                    * candidate_symbol.height.max(other_symbol.height);
                let max_err = ((self.symbol_pixel_counts[candidate]
                    .max(self.symbol_pixel_counts[other]) as f32
                    * 0.10) as u32)
                    .max((area / self.config.match_tolerance.max(1) as usize) as u32)
                    .clamp(3, 20);

                match comparator.distance(other_symbol, candidate_symbol, max_err) {
                    Some((err, dx, dy)) => {
                        let refinement_penalty = err as u64 + ((dx.abs() + dy.abs()) as u64 * 2);
                        total_cost += refinement_penalty * self.symbol_usage[other] as u64;
                    }
                    None => total_cost += 1_000_000,
                }
            }

            if total_cost < best_cost
                || (total_cost == best_cost
                    && (
                        self.symbol_usage[candidate],
                        self.symbol_pixel_counts[candidate],
                    ) > (
                        self.symbol_usage[best_idx],
                        self.symbol_pixel_counts[best_idx],
                    ))
            {
                best_cost = total_cost;
                best_idx = candidate;
            }
        }

        best_idx
    }

    fn note_symbol_page(&mut self, symbol_index: usize, page_num: usize) {
        if self.symbol_last_page_seen[symbol_index] != Some(page_num) {
            self.symbol_last_page_seen[symbol_index] = Some(page_num);
            self.symbol_page_count[symbol_index] += 1;
            self.page_symbol_indices[page_num].push(symbol_index);
        }
    }

    fn push_symbol(&mut self, symbol: BitImage, pixel_count: usize, page_num: usize) -> usize {
        let idx = self.global_symbols.len();
        self.symbol_signatures
            .push(Self::compute_symbol_signature(&symbol));
        self.symbol_pixel_counts.push(pixel_count);
        self.global_symbols.push(symbol);
        self.symbol_usage.push(1);
        self.symbol_page_count.push(0);
        self.symbol_last_page_seen.push(None);
        self.note_symbol_page(idx, page_num);
        idx
    }

    fn rebuild_symbol_metadata(&mut self) {
        self.symbol_usage = vec![0; self.global_symbols.len()];
        self.symbol_page_count = vec![0; self.global_symbols.len()];
        self.symbol_last_page_seen = vec![None; self.global_symbols.len()];
        self.page_symbol_indices = vec![Vec::new(); self.pages.len()];
        self.symbol_pixel_counts = self
            .global_symbols
            .iter()
            .map(BitImage::count_ones)
            .collect();
        self.symbol_signatures = self
            .global_symbols
            .iter()
            .map(Self::compute_symbol_signature)
            .collect();

        for page_num in 0..self.pages.len() {
            let instance_indices: Vec<usize> = self.pages[page_num]
                .symbol_instances
                .iter()
                .map(|inst| inst.symbol_index)
                .collect();
            for symbol_index in instance_indices {
                self.symbol_usage[symbol_index] += 1;
                self.note_symbol_page(symbol_index, page_num);
            }
        }
    }

    fn rebuild_hash_map(&mut self) {
        self.hash_map.clear();
        for (idx, symbol) in self.global_symbols.iter().enumerate() {
            let key = hash_key(symbol);
            self.hash_map.entry(key).or_default().push(idx);
        }
    }

    pub fn add_page(&mut self, image: &Array2<u8>) -> Result<()> {
        let bitimage = crate::jbig2sym::array_to_bitimage(image);
        self.add_page_bitimage(bitimage)
    }

    pub fn add_page_bitimage(&mut self, bitimage: BitImage) -> Result<()> {
        let page_num = self.pages.len();
        self.page_symbol_indices.push(Vec::new());
        let mut symbol_instances = Vec::new();
        let mut comparator = Comparator::default();
        let debug_matching =
            page_num == 0 && std::env::var("JBIG2_DEBUG").map_or(false, |v| v == "1");
        let no_reuse = std::env::var("JBIG2_NO_REUSE").map_or(false, |v| v == "1");

        let mut debug_lines: Vec<String> = Vec::new();
        if debug_matching {
            debug_lines.push("=== PAGE 0 MATCHING LOG ===".to_string());
            debug_lines.push(format!("Image: {}x{}", bitimage.width, bitimage.height));
        }
        let mut cc_index = 0usize;

        // Extract symbols if symbol mode is enabled
        if self.config.symbol_mode && self.state.segment {
            #[cfg(feature = "cc-analysis")]
            {
                let dpi = 300; // Default DPI
                let losslevel = if self.config.is_lossless { 0 } else { 1 };
                let cc_start = Instant::now();
                let cc_image = analyze_page(&bitimage, dpi, losslevel);
                let extracted = cc_image.extract_shape_refs();
                self.metrics.symbol_mode.cc_extraction += cc_start.elapsed();

                // Check if symbol extraction makes sense for this image
                // If we only get one symbol that covers the entire image,
                // it's better to use generic region encoding
                let should_use_symbols = if extracted.len() == 1 {
                    let bbox = extracted[0].bbox;
                    !(bbox.xmin == 0
                        && bbox.ymin == 0
                        && bbox.width() as usize >= bitimage.width.saturating_sub(2)
                        && bbox.height() as usize >= bitimage.height.saturating_sub(2))
                } else {
                    !extracted.is_empty()
                };

                if should_use_symbols {
                    let matching_start = Instant::now();
                    let mut recent_cache = RecentSymbolCache::new(64);
                    let mut last_y = 0u32;

                    for shape in extracted {
                        if Self::should_skip_symbol_candidate(
                            shape.bbox.width().max(0) as usize,
                            shape.bbox.height().max(0) as usize,
                            shape.black_pixels,
                        ) || shape.run_count == 0
                        {
                            continue;
                        }
                        let Some(symbol) = cc_image.get_bitmap_for_cc(shape.ccid) else {
                            continue;
                        };
                        let (trim_offset, trimmed) = symbol.trim();
                        let pixel_count = trimmed.count_ones();
                        if Self::should_skip_symbol_candidate(
                            trimmed.width,
                            trimmed.height,
                            pixel_count,
                        ) {
                            continue;
                        }

                        // The CC bbox is the bounding box from CC analysis.
                        // trim() may remove whitespace rows/cols from the symbol.
                        // Adjust position by trim offset so the dictionary bitmap
                        // renders at the correct location on the page.
                        let rect = Rect {
                            x: shape.bbox.xmin as u32 + trim_offset.x,
                            y: shape.bbox.ymin as u32 + trim_offset.y,
                            width: trimmed.width as u32,
                            height: trimmed.height as u32,
                        };
                        if rect.y > last_y.saturating_add(24) {
                            recent_cache.clear();
                        }
                        last_y = rect.y;

                        let key = hash_key(&trimmed);
                        let trimmed_sig = Self::compute_symbol_signature(&trimmed);
                        let mut matched = false;
                        let mut instance_bitmap = Some(symbol);

                        // Error tolerance for matching.
                        let area = (trimmed.width * trimmed.height) as u32;
                        let max_err = if self.config.text_refine {
                            (area / self.config.match_tolerance).max(3)
                        } else {
                            ((area as f32 * 0.05) as u32).max(2)
                        };

                        if !no_reuse {
                            let recent_candidates: Vec<usize> = recent_cache.iter().collect();
                            'recent_search: for idx in recent_candidates {
                                if let Some((err, dx, dy, needs_refinement)) = self
                                    .evaluate_symbol_match(
                                        &trimmed,
                                        trimmed_sig,
                                        pixel_count,
                                        idx,
                                        &mut comparator,
                                        max_err,
                                    )
                                {
                                    if debug_matching {
                                        let mode = if needs_refinement {
                                            "REFINE"
                                        } else if err == 0 && dx == 0 && dy == 0 {
                                            "EXACT "
                                        } else {
                                            "LOSSY "
                                        };
                                        let proto = &self.global_symbols[idx];
                                        debug_lines.push(format!(
                                            "CC#{:04} {} pos=({},{}) {}x{} → proto#{} {}x{} err={} dx={} dy={} [recent]",
                                            cc_index,
                                            mode,
                                            rect.x,
                                            rect.y,
                                            rect.width,
                                            rect.height,
                                            idx,
                                            proto.width,
                                            proto.height,
                                            err,
                                            dx,
                                            dy
                                        ));
                                    }

                                    self.symbol_usage[idx] += 1;
                                    self.note_symbol_page(idx, page_num);
                                    symbol_instances.push(SymbolInstance {
                                        symbol_index: idx,
                                        position: rect,
                                        instance_bitmap: instance_bitmap.take().unwrap(),
                                        needs_refinement,
                                        refinement_dx: if needs_refinement { dx } else { 0 },
                                        refinement_dy: if needs_refinement { dy } else { 0 },
                                    });
                                    recent_cache.touch(idx);
                                    matched = true;
                                    break 'recent_search;
                                }
                            }
                        }

                        if !matched && !no_reuse {
                            let h = trimmed.height as u64;
                            let w = trimmed.width as u64;
                            let dim_range: u64 = if self.config.text_refine { 2 } else { 0 };

                            'bucket_search: for dh_off in 0..=(dim_range * 2) {
                                let dh = h.wrapping_add(dh_off).wrapping_sub(dim_range);
                                if dh >= 10_000 {
                                    continue;
                                }
                                for dw_off in 0..=(dim_range * 2) {
                                    let dw = w.wrapping_add(dw_off).wrapping_sub(dim_range);
                                    if dw >= 10_000 {
                                        continue;
                                    }

                                    let nk = HashKey(dh * 10_000 + dw);
                                    if let Some(bucket) = self.hash_map.get(&nk).cloned() {
                                        for idx in bucket {
                                            let Some((err, dx, dy, needs_refinement)) = self
                                                .evaluate_symbol_match(
                                                    &trimmed,
                                                    trimmed_sig,
                                                    pixel_count,
                                                    idx,
                                                    &mut comparator,
                                                    max_err,
                                                )
                                            else {
                                                continue;
                                            };

                                            if debug_matching {
                                                let mode = if needs_refinement {
                                                    "REFINE"
                                                } else if err == 0 && dx == 0 && dy == 0 {
                                                    "EXACT "
                                                } else {
                                                    "LOSSY "
                                                };
                                                let proto = &self.global_symbols[idx];
                                                debug_lines.push(format!(
                                                    "CC#{:04} {} pos=({},{}) {}x{} → proto#{} {}x{} err={} dx={} dy={}",
                                                    cc_index,
                                                    mode,
                                                    rect.x,
                                                    rect.y,
                                                    rect.width,
                                                    rect.height,
                                                    idx,
                                                    proto.width,
                                                    proto.height,
                                                    err,
                                                    dx,
                                                    dy
                                                ));
                                            }

                                            self.symbol_usage[idx] += 1;
                                            self.note_symbol_page(idx, page_num);
                                            symbol_instances.push(SymbolInstance {
                                                symbol_index: idx,
                                                position: rect,
                                                instance_bitmap: instance_bitmap.take().unwrap(),
                                                needs_refinement,
                                                refinement_dx: if needs_refinement {
                                                    dx
                                                } else {
                                                    0
                                                },
                                                refinement_dy: if needs_refinement {
                                                    dy
                                                } else {
                                                    0
                                                },
                                            });
                                            recent_cache.touch(idx);
                                            matched = true;
                                            break 'bucket_search;
                                        }
                                    }
                                }
                            }
                        }

                        if !matched {
                            let idx = self.push_symbol(trimmed, pixel_count, page_num);
                            self.metrics.symbol_stats.symbols_discovered += 1;
                            if debug_matching {
                                debug_lines.push(format!(
                                    "CC#{:04} NEW    pos=({},{}) {}x{} trim_off=({},{}) → new proto#{} {}x{}",
                                    cc_index, rect.x, rect.y, rect.width, rect.height,
                                    trim_offset.x, trim_offset.y,
                                    idx, self.global_symbols[idx].width, self.global_symbols[idx].height
                                ));
                            }
                            self.hash_map.entry(key).or_default().push(idx);
                            symbol_instances.push(SymbolInstance {
                                symbol_index: idx,
                                position: rect,
                                instance_bitmap: instance_bitmap.take().unwrap(),
                                needs_refinement: false,
                                refinement_dx: 0,
                                refinement_dy: 0,
                            });
                            recent_cache.touch(idx);
                        }
                        cc_index += 1;
                    }
                    self.metrics.symbol_mode.matching_dedup += matching_start.elapsed();
                }
            }
        }

        // Write page 0 matching debug log
        if debug_matching && !debug_lines.is_empty() {
            debug_lines.push(format!(
                "\nTotal CCs: {}, Instances: {}",
                cc_index,
                symbol_instances.len()
            ));
            let log_path = std::path::Path::new("jbig2_debug_page0.log");
            if let Ok(mut f) = std::fs::File::create(log_path) {
                use std::io::Write;
                for line in &debug_lines {
                    let _ = writeln!(f, "{}", line);
                }
            }
        }

        self.pages.push(PageData {
            image: bitimage,
            symbol_instances,
        });
        Ok(())
    }

    pub fn collect_symbols(&mut self, roi: &Array2<u8>) -> Result<()> {
        let bitimage = crate::jbig2sym::array_to_bitimage(roi);
        let (_, trimmed) = bitimage.trim();
        let key = hash_key(&trimmed);
        let page_num = self.pages.len();
        if self.page_symbol_indices.len() <= page_num {
            self.page_symbol_indices.resize_with(page_num + 1, Vec::new);
        }

        if !self.hash_map.contains_key(&key) {
            let pixel_count = trimmed.count_ones();
            let idx = self.push_symbol(trimmed, pixel_count, page_num);
            self.metrics.symbol_stats.symbols_discovered += 1;
            self.hash_map.insert(key, vec![idx]);
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<Vec<u8>> {
        let include_header = self.state.full_headers_remaining;
        let plan = self.plan_document(include_header)?;
        self.validate_plan(&plan)?;
        let output = self.serialize_full_document(&plan)?;
        self.state.full_headers_remaining = false;
        self.next_segment_number = plan.next_segment_number;
        Ok(output)
    }

    pub fn flush_pdf_split(&mut self) -> Result<PdfSplitOutput> {
        self.state.pdf_mode = true;
        let plan = self.plan_document(false)?;
        self.validate_plan(&plan)?;
        let (global_segments, page_streams) = self.serialize_pdf_split(&plan)?;
        self.next_segment_number = plan.next_segment_number;
        Ok(PdfSplitOutput {
            global_segments,
            page_streams,
        })
    }

    fn plan_document(&mut self, include_header: bool) -> Result<PlannedDocument> {
        debug!("Symbol stats before encoding: {}", self.get_symbol_stats());
        let planning_start = Instant::now();

        if self.config.auto_thresh {
            let clustering_start = Instant::now();
            self.cluster_symbols()?;
            self.metrics.symbol_mode.clustering += clustering_start.elapsed();
        }

        self.prune_symbols_if_needed();
        self.validate_symbol_instance_indices()?;

        let global_symbol_indices: Vec<usize> = self
            .global_symbols
            .iter()
            .enumerate()
            .filter(|(i, _)| self.symbol_page_count[*i] > 1 || self.pages.len() == 1)
            .map(|(i, _)| i)
            .collect();

        let mut page_local_symbols: Vec<Vec<usize>> = self
            .page_symbol_indices
            .iter()
            .map(|symbols| {
                symbols
                    .iter()
                    .copied()
                    .filter(|&i| self.symbol_page_count[i] <= 1)
                    .collect()
            })
            .collect();

        let global_set: HashSet<usize> = global_symbol_indices.iter().copied().collect();
        let mut page_uses_generic_region = vec![false; self.pages.len()];
        for (page_num, page) in self.pages.iter().enumerate() {
            let local_symbols = &page_local_symbols[page_num];
            let page_local_gain: i64 = local_symbols
                .iter()
                .map(|&symbol_index| self.estimate_local_symbol_gain(page, symbol_index))
                .sum();
            let uses_only_locals = page
                .symbol_instances
                .iter()
                .all(|inst| !global_set.contains(&inst.symbol_index));
            if uses_only_locals
                && local_symbols.len() <= 2
                && page.symbol_instances.len() <= 2
                && page_local_gain <= 0
            {
                page_local_symbols[page_num].clear();
                page_uses_generic_region[page_num] = true;
            }
        }

        self.validate_symbol_partition(
            &global_symbol_indices,
            &page_local_symbols,
            &page_uses_generic_region,
        )?;
        self.metrics.symbol_stats.global_symbol_count = global_symbol_indices.len();
        self.metrics.symbol_stats.local_symbol_count =
            page_local_symbols.iter().map(Vec::len).sum::<usize>();
        self.metrics.symbol_stats.symbols_exported = self.metrics.symbol_stats.global_symbol_count
            + self.metrics.symbol_stats.local_symbol_count;
        self.metrics.symbol_stats.avg_symbol_reuse =
            if self.metrics.symbol_stats.symbols_exported > 0 {
                self.symbol_usage.iter().sum::<usize>() as f64
                    / self.metrics.symbol_stats.symbols_exported as f64
            } else {
                0.0
            };

        let mut current_segment_number = self.next_segment_number;
        let mut global_segments = Vec::new();

        self.global_dict_segment_number = None;
        let mut canonical_global_order = Vec::new();
        if !global_symbol_indices.is_empty() {
            let refs: Vec<&BitImage> = global_symbol_indices
                .iter()
                .map(|&i| &self.global_symbols[i])
                .collect();
            let dict_start = Instant::now();
            let (global_dict_payload, order) =
                encode_symbol_dict_with_order(&refs, &self.config, 0)?;
            self.metrics.symbol_mode.symbol_dict_encoding += dict_start.elapsed();
            canonical_global_order = order;
            let global_dict_segment = Segment {
                number: current_segment_number,
                seg_type: SegmentType::SymbolDictionary,
                deferred_non_retain: false,
                retain_flags: 0,
                page_association_type: 2,
                referred_to: Vec::new(),
                page: None,
                payload: global_dict_payload,
            };
            self.global_dict_segment_number = Some(global_dict_segment.number);
            global_segments.push(global_dict_segment);
            current_segment_number += 1;
        }

        let mut global_sym_to_dict_pos = vec![u32::MAX; self.global_symbols.len()];
        for (dict_pos, &refs_idx) in canonical_global_order.iter().enumerate() {
            let gs_idx = global_symbol_indices[refs_idx];
            global_sym_to_dict_pos[gs_idx] = dict_pos as u32;
        }
        let num_global_dict_symbols = canonical_global_order.len() as u32;

        let page_segment_start = current_segment_number;
        let mut page_layouts = Vec::with_capacity(self.pages.len());
        for (page_num, page) in self.pages.iter().enumerate() {
            let page_number = if self.state.pdf_mode {
                1u32
            } else {
                page_num as u32 + 1
            };
            if self.state.pdf_mode {
                current_segment_number = page_segment_start;
            }
            let page_info_segment_number = current_segment_number;
            current_segment_number += 1;
            let local_dict_segment_number = if self.config.symbol_mode
                && !page.symbol_instances.is_empty()
                && !page_local_symbols[page_num].is_empty()
            {
                let seg = current_segment_number;
                current_segment_number += 1;
                Some(seg)
            } else {
                None
            };
            let region_segment_number = current_segment_number;
            current_segment_number += 1;
            let end_of_page_segment_number = current_segment_number;
            current_segment_number += 1;
            let use_generic_region = page_uses_generic_region[page_num];

            page_layouts.push(PlannedPageLayout {
                page_index: page_num,
                page_number,
                page_info_segment_number,
                local_dict_segment_number,
                region_segment_number,
                end_of_page_segment_number,
                local_symbols: page_local_symbols[page_num].clone(),
                use_generic_region,
            });
        }

        self.metrics.symbol_mode.planning += planning_start.elapsed();

        #[cfg(feature = "parallel")]
        let built_pages = if self.state.pdf_mode || self.pages.len() > 1 {
            page_layouts
                .par_iter()
                .map(|layout| {
                    self.build_planned_page(
                        layout,
                        &global_sym_to_dict_pos,
                        num_global_dict_symbols,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            page_layouts
                .iter()
                .map(|layout| {
                    self.build_planned_page(
                        layout,
                        &global_sym_to_dict_pos,
                        num_global_dict_symbols,
                    )
                })
                .collect::<Vec<_>>()
        };

        #[cfg(not(feature = "parallel"))]
        let built_pages = page_layouts
            .iter()
            .map(|layout| {
                self.build_planned_page(layout, &global_sym_to_dict_pos, num_global_dict_symbols)
            })
            .collect::<Vec<_>>();

        let mut pages = Vec::with_capacity(built_pages.len());
        for built_page in built_pages {
            let built_page = built_page?;
            self.metrics.symbol_mode.symbol_dict_encoding += built_page.symbol_dict_time;
            self.metrics.symbol_mode.text_region_encoding += built_page.text_region_time;
            self.metrics.symbol_mode.generic_region_encoding += built_page.generic_region_time;
            pages.push(built_page.page);
        }

        let eof_segment = Some(Segment {
            number: current_segment_number,
            seg_type: SegmentType::EndOfFile,
            deferred_non_retain: false,
            retain_flags: 0,
            page_association_type: 2,
            referred_to: vec![],
            page: None,
            payload: vec![],
        });
        current_segment_number += 1;

        Ok(PlannedDocument {
            file_header: if include_header {
                Some(FileHeader {
                    organisation_type: true,
                    unknown_n_pages: false,
                    n_pages: self.pages.len() as u32,
                })
            } else {
                None
            },
            global_segments,
            pages,
            eof_segment,
            next_segment_number: current_segment_number,
        })
    }

    fn build_planned_page(
        &self,
        layout: &PlannedPageLayout,
        global_sym_to_dict_pos: &[u32],
        num_global_dict_symbols: u32,
    ) -> Result<BuiltPage> {
        let page = &self.pages[layout.page_index];
        let mut page_segments = Vec::new();
        let mut symbol_dict_time = Duration::default();
        let mut text_region_time = Duration::default();
        let mut generic_region_time = Duration::default();

        let page_info_payload = PageInfo {
            width: page.image.width as u32,
            height: page.image.height as u32,
            default_pixel: false,
            xres: self.config.generic.dpi,
            yres: self.config.generic.dpi,
            ..Default::default()
        }
        .to_bytes();

        page_segments.push(Segment {
            number: layout.page_info_segment_number,
            seg_type: SegmentType::PageInformation,
            deferred_non_retain: false,
            retain_flags: 0,
            page_association_type: 0,
            referred_to: vec![],
            page: Some(layout.page_number),
            payload: page_info_payload,
        });

        if self.config.symbol_mode
            && !page.symbol_instances.is_empty()
            && !layout.use_generic_region
        {
            let mut referred_to_for_text_region = Vec::new();
            if let Some(global_dict_seg_num) = self.global_dict_segment_number {
                referred_to_for_text_region.push(global_dict_seg_num);
            }

            let mut local_sym_to_dict_pos = vec![u32::MAX; self.global_symbols.len()];
            let num_local_dict_symbols =
                if let Some(local_dict_segment_number) = layout.local_dict_segment_number {
                    let refs: Vec<&BitImage> = layout
                        .local_symbols
                        .iter()
                        .map(|&i| &self.global_symbols[i])
                        .collect();
                    let dict_start = Instant::now();
                    let (local_dict_payload, local_order) =
                        encode_symbol_dict_with_order(&refs, self.config, 0)?;
                    symbol_dict_time += dict_start.elapsed();

                    for (dict_pos, &refs_idx) in local_order.iter().enumerate() {
                        let gs_idx = layout.local_symbols[refs_idx];
                        local_sym_to_dict_pos[gs_idx] = dict_pos as u32;
                    }

                    page_segments.push(Segment {
                        number: local_dict_segment_number,
                        seg_type: SegmentType::SymbolDictionary,
                        deferred_non_retain: false,
                        retain_flags: 0,
                        page_association_type: 0,
                        referred_to: Vec::new(),
                        page: Some(layout.page_number),
                        payload: local_dict_payload,
                    });
                    referred_to_for_text_region.push(local_dict_segment_number);
                    local_order.len() as u32
                } else {
                    0
                };

            let text_start = Instant::now();
            let region_payload = if self.config.text_refine {
                encode_text_region_with_refinement(
                    &page.symbol_instances,
                    self.config,
                    &self.global_symbols,
                    global_sym_to_dict_pos,
                    num_global_dict_symbols,
                    &local_sym_to_dict_pos,
                    num_local_dict_symbols,
                )?
            } else {
                encode_text_region_mapped(
                    &page.symbol_instances,
                    self.config,
                    &self.global_symbols,
                    global_sym_to_dict_pos,
                    num_global_dict_symbols,
                    &local_sym_to_dict_pos,
                    layout.page_index,
                    num_local_dict_symbols,
                )?
            };
            text_region_time += text_start.elapsed();

            page_segments.push(Segment {
                number: layout.region_segment_number,
                seg_type: SegmentType::ImmediateTextRegion,
                deferred_non_retain: false,
                retain_flags: 0,
                page_association_type: 0,
                referred_to: referred_to_for_text_region,
                page: Some(layout.page_number),
                payload: region_payload,
            });
        } else {
            let mut gr_cfg = GenericRegionConfig::new(
                page.image.width as u32,
                page.image.height as u32,
                self.config.generic.dpi,
            );
            gr_cfg.comb_operator = self.config.generic.comb_operator;
            gr_cfg.mmr = self.config.generic.mmr;
            gr_cfg.tpgdon = self.config.generic.tpgdon;
            gr_cfg.validate().map_err(|e: &'static str| anyhow!(e))?;

            let generic_start = Instant::now();
            let coder_data = Jbig2ArithCoder::encode_generic_payload_cfg(&page.image, &gr_cfg)?;
            let params: GenericRegionParams = gr_cfg.clone().into();
            let mut generic_region_payload = params.to_bytes();
            generic_region_payload.extend_from_slice(&coder_data);
            generic_region_time += generic_start.elapsed();

            page_segments.push(Segment {
                number: layout.region_segment_number,
                seg_type: SegmentType::ImmediateGenericRegion,
                deferred_non_retain: false,
                retain_flags: 0,
                page_association_type: 0,
                referred_to: Vec::new(),
                page: Some(layout.page_number),
                payload: generic_region_payload,
            });
        }

        page_segments.push(Segment {
            number: layout.end_of_page_segment_number,
            seg_type: SegmentType::EndOfPage,
            deferred_non_retain: false,
            retain_flags: 0,
            page_association_type: 0,
            referred_to: Vec::new(),
            page: Some(layout.page_number),
            payload: Vec::new(),
        });

        Ok(BuiltPage {
            page: PlannedPage {
                page_number: layout.page_number,
                segments: page_segments,
            },
            symbol_dict_time,
            text_region_time,
            generic_region_time,
        })
    }

    fn validate_plan(&self, plan: &PlannedDocument) -> Result<()> {
        let mut global_numbers = HashSet::new();

        for seg in &plan.global_segments {
            if !global_numbers.insert(seg.number) {
                anyhow::bail!("Duplicate segment number in globals: {}", seg.number);
            }
        }

        for (page_idx, page) in plan.pages.iter().enumerate() {
            // In PDF mode, each page is an independent stream, so segment
            // numbers only need to be unique within globals + that page.
            let mut page_numbers = global_numbers.clone();
            for seg in &page.segments {
                if !page_numbers.insert(seg.number) {
                    anyhow::bail!(
                        "Duplicate segment number {} on page {}",
                        seg.number,
                        page_idx
                    );
                }
            }

            for seg in &page.segments {
                for referred in &seg.referred_to {
                    if !page_numbers.contains(referred) {
                        anyhow::bail!(
                            "Page {} segment {} refers to missing segment {}",
                            page.page_number,
                            seg.number,
                            referred
                        );
                    }
                    if global_numbers.contains(referred) && plan.global_segments.is_empty() {
                        anyhow::bail!(
                            "Page {} segment {} refers to global {} but no globals stream exists",
                            page.page_number,
                            seg.number,
                            referred
                        );
                    }
                }
            }
        }

        if let Some(eof) = &plan.eof_segment {
            if global_numbers.contains(&eof.number) {
                anyhow::bail!("EOF segment number {} conflicts with globals", eof.number);
            }
        }

        for seg in &plan.global_segments {
            for referred in &seg.referred_to {
                if !global_numbers.contains(referred) {
                    anyhow::bail!(
                        "Global segment {} refers to missing segment {}",
                        seg.number,
                        referred
                    );
                }
            }
        }

        Ok(())
    }

    fn serialize_full_document(&self, plan: &PlannedDocument) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        if let Some(header) = &plan.file_header {
            output.extend(header.to_bytes());
        }
        for seg in &plan.global_segments {
            seg.write_into(&mut output)?;
        }
        for page in &plan.pages {
            for seg in &page.segments {
                seg.write_into(&mut output)?;
            }
        }
        if let Some(eof) = &plan.eof_segment {
            eof.write_into(&mut output)?;
        }
        Ok(output)
    }

    fn serialize_pdf_split(
        &self,
        plan: &PlannedDocument,
    ) -> Result<(Option<Vec<u8>>, Vec<Vec<u8>>)> {
        let global_segments = if plan.global_segments.is_empty() {
            None
        } else {
            let mut out = Vec::new();
            for seg in &plan.global_segments {
                seg.write_into(&mut out)?;
            }
            Some(out)
        };

        #[cfg(feature = "parallel")]
        let page_streams = plan
            .pages
            .par_iter()
            .map(|page| {
                let mut page_out = Vec::new();
                for seg in &page.segments {
                    seg.write_into(&mut page_out)?;
                }
                Ok(page_out)
            })
            .collect::<Vec<Result<Vec<u8>>>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        #[cfg(not(feature = "parallel"))]
        let page_streams = {
            let mut page_streams = Vec::with_capacity(plan.pages.len());
            for page in &plan.pages {
                let mut page_out = Vec::new();
                for seg in &page.segments {
                    seg.write_into(&mut page_out)?;
                }
                page_streams.push(page_out);
            }
            page_streams
        };

        Ok((global_segments, page_streams))
    }

    fn prune_symbols_if_needed(&mut self) {
        // No pruning — JBIG2 supports large dictionaries and pruning drops
        // symbol instances, leaving holes in the rendered output.
    }

    /// Cluster similar symbols into groups and select prototypes.
    ///
    /// This is the key optimization for symbol-mode compression. After all pages
    /// have been extracted, we group symbols that look similar (e.g., different
    /// renderings of the letter "e") into clusters. Only one prototype per cluster
    /// is stored in the dictionary. Instances that don't exactly match their
    /// prototype are marked for refinement coding (SPM).
    ///
    /// This replaces the naive O(n²) auto_threshold with a dimension-bucketed
    /// approach that's much faster for large symbol sets.
    fn cluster_symbols(&mut self) -> Result<()> {
        let n = self.global_symbols.len();
        if n < 2 {
            return Ok(());
        }

        // Union-find with path compression and union by rank
        let mut parent: Vec<usize> = (0..n).collect();
        let mut uf_rank: Vec<u32> = vec![0; n];
        let mut comparator = Comparator::default();

        // Group by exact dimensions and compare only neighboring sizes.
        let mut buckets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (i, sym) in self.global_symbols.iter().enumerate() {
            buckets.entry((sym.height, sym.width)).or_default().push(i);
        }

        // Compare within each bucket and adjacent buckets
        let mut bucket_keys: Vec<(usize, usize)> = buckets.keys().copied().collect();
        bucket_keys.sort_unstable();

        let mut compare_pair = |a_idx: usize, b_idx: usize| {
            if uf_find(&mut parent, a_idx) == uf_find(&mut parent, b_idx) {
                return;
            }

            let a_sym = &self.global_symbols[a_idx];
            let b_sym = &self.global_symbols[b_idx];
            let dim_limit = if self.config.text_refine { 2 } else { 1 };
            if (a_sym.width as i32 - b_sym.width as i32).abs() > dim_limit
                || (a_sym.height as i32 - b_sym.height as i32).abs() > dim_limit
            {
                return;
            }

            let area = a_sym.width.max(b_sym.width) * a_sym.height.max(b_sym.height);
            let max_err = if self.config.text_refine {
                ((self.symbol_pixel_counts[a_idx].max(self.symbol_pixel_counts[b_idx]) as f32
                    * 0.10) as u32)
                    .max(((area as f32) * 0.05) as u32)
                    .clamp(3, 20)
            } else {
                ((area as f32 * 0.04) as u32).clamp(2, 12)
            };
            if self.symbol_pixel_counts[a_idx].abs_diff(self.symbol_pixel_counts[b_idx])
                > max_err as usize
            {
                return;
            }

            if let Some((_, dx, dy)) = comparator.distance(a_sym, b_sym, max_err) {
                let dy_limit = if self.config.text_refine { 1 } else { 0 };
                if dx.abs() <= dim_limit && dy.abs() <= dy_limit {
                    uf_union(&mut parent, &mut uf_rank, a_idx, b_idx);
                }
            }
        };

        for &(bh, bw) in &bucket_keys {
            let current_bucket = &buckets[&(bh, bw)];
            for ci in 0..current_bucket.len() {
                for cj in (ci + 1)..current_bucket.len() {
                    compare_pair(current_bucket[ci], current_bucket[cj]);
                }
            }

            for dh in -1i32..=1 {
                for dw in -1i32..=1 {
                    let nh = bh as i32 + dh;
                    let nw = bw as i32 + dw;
                    if nh < 0 || nw < 0 {
                        continue;
                    }
                    let neighbor_key = (nh as usize, nw as usize);
                    if neighbor_key <= (bh, bw) {
                        continue;
                    }
                    if let Some(neighbor_bucket) = buckets.get(&neighbor_key) {
                        for &a_idx in current_bucket {
                            for &b_idx in neighbor_bucket {
                                compare_pair(a_idx, b_idx);
                            }
                        }
                    }
                }
            }
        }

        // Build cluster groups
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = uf_find(&mut parent, i);
            clusters.entry(root).or_default().push(i);
        }

        // Select prototype deterministically by usage, then black pixels, then original index.
        let mut old_to_prototype: Vec<usize> = (0..n).collect();
        for (_, members) in &clusters {
            if members.len() <= 1 {
                continue;
            }
            let prototype = self.choose_cluster_prototype(members);
            for &m in members {
                old_to_prototype[m] = prototype;
            }
        }

        // Build new compact symbol list (prototypes only) and index mapping
        let mut seen_prototypes: HashMap<usize, usize> = HashMap::new();
        let mut new_symbols: Vec<BitImage> = Vec::new();
        let mut old_to_new: Vec<usize> = vec![0; n];

        // Process in order so prototype positions are deterministic
        for i in 0..n {
            let proto = old_to_prototype[i];
            if let Some(&new_idx) = seen_prototypes.get(&proto) {
                old_to_new[i] = new_idx;
            } else {
                let new_idx = new_symbols.len();
                new_symbols.push(self.global_symbols[proto].clone());
                seen_prototypes.insert(proto, new_idx);
                old_to_new[i] = new_idx;
            }
        }

        let old_count = n;
        let new_count = new_symbols.len();

        // Remap all instances and mark which ones need refinement
        for page in &mut self.pages {
            for inst in &mut page.symbol_instances {
                let old_idx = inst.symbol_index;
                let new_idx = old_to_new[old_idx];
                let proto = old_to_prototype[old_idx];

                // If this instance's original symbol was NOT the prototype,
                // it needs refinement encoding to preserve quality
                if old_idx != proto {
                    inst.needs_refinement = true;
                    // Compute alignment offset between instance and prototype.
                    // Use a generous error limit (not u32::MAX which overflows in Comparator).
                    let (_, trimmed_inst) = inst.instance_bitmap.trim();
                    let max_ref_err = (trimmed_inst.width * trimmed_inst.height) as u32;
                    if let Some((_, dx, dy)) =
                        comparator.distance(&trimmed_inst, &new_symbols[new_idx], max_ref_err)
                    {
                        inst.refinement_dx = dx;
                        inst.refinement_dy = dy;
                    }
                }

                inst.symbol_index = new_idx;
            }
        }

        // Replace internal state
        self.global_symbols = new_symbols;
        self.symbol_pixel_counts = self
            .global_symbols
            .iter()
            .map(BitImage::count_ones)
            .collect();
        self.rebuild_symbol_metadata();
        self.rebuild_hash_map();

        eprintln!(
            "Clustering: {} → {} prototype symbols ({:.1}% reduction)",
            old_count,
            new_count,
            (1.0 - new_count as f64 / old_count.max(1) as f64) * 100.0
        );

        Ok(())
    }

    fn validate_symbol_instance_indices(&self) -> Result<()> {
        for (page_num, page) in self.pages.iter().enumerate() {
            for instance in &page.symbol_instances {
                if instance.symbol_index >= self.global_symbols.len() {
                    anyhow::bail!(
                        "Page {} has symbol instance {} out of range after pruning (max {})",
                        page_num + 1,
                        instance.symbol_index,
                        self.global_symbols.len().saturating_sub(1)
                    );
                }
            }
        }
        Ok(())
    }

    fn validate_symbol_partition(
        &self,
        global_symbol_indices: &[usize],
        page_local_symbols: &[Vec<usize>],
        page_uses_generic_region: &[bool],
    ) -> Result<()> {
        let global_set: HashSet<usize> = global_symbol_indices.iter().copied().collect();
        for (page_num, page) in self.pages.iter().enumerate() {
            if page_uses_generic_region[page_num] {
                continue;
            }
            let local_set: HashSet<usize> = page_local_symbols[page_num].iter().copied().collect();
            for inst in &page.symbol_instances {
                let idx = inst.symbol_index;
                if !global_set.contains(&idx) && !local_set.contains(&idx) {
                    anyhow::bail!(
                        "Page {} symbol {} was not resolved to global or local dictionary",
                        page_num + 1,
                        idx
                    );
                }
            }
        }
        Ok(())
    }

    fn auto_threshold(&mut self) -> Result<()> {
        let mut i = 0;
        let mut comparator = Comparator::default();
        while i < self.global_symbols.len() {
            let mut j = i + 1;
            while j < self.global_symbols.len() {
                if comparator
                    .distance(&self.global_symbols[i], &self.global_symbols[j], 0)
                    .is_some()
                {
                    self.unite_templates(i, j)?;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        Ok(())
    }

    fn auto_threshold_using_hash(&mut self) -> Result<()> {
        // Repeatedly scan for exact-match duplicates until no more merges occur.
        // Each call to unite_templates invalidates indices, so we rebuild the
        // hash buckets from scratch after every merge.
        loop {
            let mut hashed_templates: HashMap<u32, Vec<usize>> = HashMap::new();
            for (i, symbol) in self.global_symbols.iter().enumerate() {
                let hash = compute_symbol_hash(symbol);
                hashed_templates.entry(hash).or_default().push(i);
            }

            let mut comparator = Comparator::default();
            let mut merged = false;

            for (_, bucket) in &hashed_templates {
                if bucket.len() < 2 {
                    continue;
                }
                // Find first mergeable pair in this bucket
                'outer: for bi in 0..bucket.len() {
                    for bj in (bi + 1)..bucket.len() {
                        if comparator
                            .distance(
                                &self.global_symbols[bucket[bi]],
                                &self.global_symbols[bucket[bj]],
                                0,
                            )
                            .is_some()
                        {
                            self.unite_templates(bucket[bi], bucket[bj])?;
                            merged = true;
                            break 'outer;
                        }
                    }
                }
                if merged {
                    break; // Indices are stale, restart the scan
                }
            }

            if !merged {
                break;
            }
        }
        Ok(())
    }

    fn unite_templates(&mut self, target_idx: usize, source_idx: usize) -> Result<()> {
        if source_idx >= self.global_symbols.len() {
            anyhow::bail!("Source index out of range");
        }

        for page in &mut self.pages {
            for instance in &mut page.symbol_instances {
                if instance.symbol_index == source_idx {
                    instance.symbol_index = target_idx;
                } else if instance.symbol_index > source_idx {
                    instance.symbol_index -= 1;
                }
            }
        }

        self.global_symbols.remove(source_idx);
        self.symbol_pixel_counts.remove(source_idx);
        self.rebuild_symbol_metadata();
        self.rebuild_hash_map();

        Ok(())
    }

    pub fn next_segment_number(&mut self) -> u32 {
        let num = self.next_segment_number;
        self.next_segment_number += 1;
        num
    }

    pub fn flush_dict(&mut self) -> Result<Vec<u8>> {
        if self.global_symbols.is_empty() {
            return Ok(Vec::new());
        }

        let symbol_refs: Vec<&BitImage> = self.global_symbols.iter().collect();
        let dict_data = encode_symbol_dict(&symbol_refs, &self.config, 0)?;

        let dict_segment = Segment {
            number: self.next_segment_number,
            seg_type: SegmentType::SymbolDictionary,
            deferred_non_retain: false,
            retain_flags: 0,
            page_association_type: if self.state.pdf_mode { 2 } else { 0 },
            referred_to: Vec::new(),
            page: if self.state.pdf_mode { None } else { Some(1) },
            payload: dict_data,
        };
        self.next_segment_number += 1;

        let mut output = Vec::new();
        if self.state.pdf_mode {
            dict_segment.write_into(&mut output)?;
            return Ok(output);
        }

        let header = FileHeader {
            organisation_type: true,
            unknown_n_pages: false,
            n_pages: 1,
        };
        output.extend(header.to_bytes());
        dict_segment.write_into(&mut output)?;

        Ok(output)
    }
}

/// Encodes a generic region, optionally wrapping it in a complete JBIG2 file.
/// This function is intended to be the top-level entry point for encoding a single generic region.
pub fn encode_generic_region(img: &BitImage, cfg: &Jbig2Config) -> Result<Vec<u8>> {
    // Build generic region config from high-level parameters
    let mut gr_cfg = GenericRegionParams::new(img.width as u32, img.height as u32, cfg.generic.dpi);
    gr_cfg.comb_operator = cfg.generic.comb_operator;
    gr_cfg.mmr = cfg.generic.mmr;
    gr_cfg.tpgdon = cfg.generic.tpgdon;
    gr_cfg.validate().map_err(|e: &'static str| anyhow!(e))?;

    let coder_data =
        Jbig2ArithCoder::encode_generic_payload(img, gr_cfg.template, &gr_cfg.at_pixels)?;

    let params: GenericRegionParams = gr_cfg.clone();

    let mut generic_region_payload = params.to_bytes();
    generic_region_payload.extend_from_slice(&coder_data);

    // Create the generic region segment (segment number 1)
    let generic_region_segment = Segment {
        number: 1, // Segment number 1
        seg_type: SegmentType::ImmediateGenericRegion,
        deferred_non_retain: false,
        retain_flags: 0,
        page_association_type: 0, // Explicit page association
        referred_to: Vec::new(),
        page: Some(1),                           // Page 1
        payload: generic_region_payload.clone(), // Clone to avoid move
    };

    // If caller wants only the segment, we're done
    if !cfg.want_full_headers {
        let mut seg_bytes = Vec::new();
        generic_region_segment.write_into(&mut seg_bytes)?;
        return Ok(seg_bytes);
    }

    // Otherwise wrap it in a complete one-page JBIG2 file
    let mut out = Vec::with_capacity(generic_region_payload.len() + 64);

    // File header
    out.extend_from_slice(
        &FileHeader {
            organisation_type: true,
            unknown_n_pages: false,
            n_pages: 1,
        }
        .to_bytes(),
    );

    // Page Information segment (segment number 0)
    Segment {
        number: 0,
        seg_type: SegmentType::PageInformation,
        deferred_non_retain: false,
        retain_flags: 0,
        page_association_type: 0,
        referred_to: vec![],
        page: Some(1),
        payload: PageInfo {
            width: img.width as u32,
            height: img.height as u32,
            xres: cfg.generic.dpi,
            yres: cfg.generic.dpi,
            is_lossless: cfg.is_lossless,
            default_pixel: cfg.default_pixel,
            default_operator: cfg.generic.comb_operator,
            ..Default::default()
        }
        .to_bytes(),
    }
    .write_into(&mut out)?;

    // Generic region segment (segment number 1)
    generic_region_segment.write_into(&mut out)?;

    // EOF segment (segment number 2)
    Segment {
        number: 2,
        seg_type: SegmentType::EndOfFile,
        deferred_non_retain: false,
        retain_flags: 0,
        page_association_type: 2,
        referred_to: vec![],
        page: None,
        payload: vec![],
    }
    .write_into(&mut out)?;

    Ok(out)
}

pub fn encode_symbol_dict(
    symbols: &[&BitImage],
    _config: &Jbig2Config,
    num_imported_symbols: u32,
) -> Result<Vec<u8>> {
    let (payload, _order) = encode_symbol_dict_with_order(symbols, _config, num_imported_symbols)?;
    Ok(payload)
}

/// Computes the canonical encoding order for a list of symbols.
///
/// Returns a `Vec<usize>` where each element is an index into the input `symbols` slice,
/// giving the order symbols will appear in the encoded dictionary (after filtering out
/// zero-size symbols, deduplication, and sorting by height class then width).
///
/// This order must be used when mapping symbol instance IDs in text regions.
pub fn canonicalize_dict_symbols(symbols: &[&BitImage]) -> Vec<usize> {
    // Step 1: Filter zero-size, tracking original indices
    let mut valid: Vec<(usize, &BitImage)> = symbols
        .iter()
        .enumerate()
        .filter(|(_, sym)| sym.width > 0 && sym.height > 0)
        .map(|(i, sym)| (i, *sym))
        .collect();

    // Step 2: Sort by (height ASC, width ASC) — same order as sort_symbols_for_dictionary
    // Use stable sort to preserve input order for identical dimensions.
    // No dedup here: the encoder already deduplicates during extraction + auto_threshold.
    // Removing a symbol would leave text region instances without a valid dictionary mapping.
    valid.sort_by(|a, b| (a.1.height, a.1.width).cmp(&(b.1.height, b.1.width)));

    // Return original indices in canonical order
    valid.into_iter().map(|(orig_idx, _)| orig_idx).collect()
}

/// Encodes a symbol dictionary, returning both the payload and the mapping from
/// encoded dictionary position → input index.
pub fn encode_symbol_dict_with_order(
    symbols: &[&BitImage],
    _config: &Jbig2Config,
    num_imported_symbols: u32,
) -> Result<(Vec<u8>, Vec<usize>)> {
    // Compute canonical order (filter + dedup + sort)
    let canonical_order = canonicalize_dict_symbols(symbols);

    if canonical_order.is_empty() {
        return Err(anyhow!(
            "encode_symbol_dict: no valid symbols supplied (all symbols had zero width or height)"
        ));
    }

    // Build the ordered symbol list
    let ordered_symbols: Vec<&BitImage> = canonical_order.iter().map(|&i| symbols[i]).collect();

    // Verify symbol dimensions are within JBIG2 limits
    for (i, sym) in ordered_symbols.iter().enumerate() {
        if sym.width > (1 << 24) || sym.height > (1 << 24) {
            return Err(anyhow!(
                "Symbol at index {} exceeds maximum dimensions ({}x{})",
                i,
                sym.width,
                sym.height
            ));
        }
    }

    let mut payload = Vec::new();
    let mut coder = Jbig2ArithCoder::new();

    let num_export_syms = ordered_symbols.len() as u32;

    // Create symbol dictionary parameters
    let params = SymbolDictParams {
        sd_template: 0, // Use standard template 0
        // Match jbig2enc's template-0 adaptive pixels for symbol dictionaries.
        at: [(3, -1), (-3, -1), (2, -2), (-2, -2)],
        exsyms: num_export_syms,
        newsyms: ordered_symbols.len() as u32,
    };

    if cfg!(debug_assertions) {
        debug!("encode_symbol_dict: Exporting {} symbols", num_export_syms);
        trace!("encode_symbol_dict: SymbolDictParams details: {:?}", params);
    }

    // Write the symbol dictionary parameters
    payload.extend(params.to_bytes());

    // Symbols are already in canonical (height, width) order from canonicalize_dict_symbols.
    // We need to encode them in this exact order, grouped by height class for delta encoding.
    // Build height classes from the already-sorted ordered_symbols to preserve the canonical order.
    let mut height_classes: Vec<Vec<&BitImage>> = Vec::new();
    let mut current_height: Option<usize> = None;
    let mut current_class: Vec<&BitImage> = Vec::new();

    for &sym in &ordered_symbols {
        match current_height {
            None => {
                // First symbol
                current_height = Some(sym.height);
                current_class.push(sym);
            }
            Some(h) if sym.height == h => {
                // Same height class
                current_class.push(sym);
            }
            Some(_) => {
                // New height class - push previous and start new
                height_classes.push(current_class);
                current_height = Some(sym.height);
                current_class = vec![sym];
            }
        }
    }
    if !current_class.is_empty() {
        height_classes.push(current_class);
    }

    // Debug: log the encoding order and first few pixels of each symbol for verification
    #[cfg(debug_assertions)]
    {
        debug!(
            "Symbol dictionary encoding order ({} symbols):",
            ordered_symbols.len()
        );
        let mut dict_pos = 0u32;
        for (hc_idx, symbols_in_class) in height_classes.iter().enumerate() {
            debug!(
                "  Height class {}: {} symbols",
                hc_idx,
                symbols_in_class.len()
            );
            for (sym_idx, sym) in symbols_in_class.iter().enumerate() {
                // Log first pixel position for each symbol
                let first_pixel = first_black_pixel(sym);
                if sym_idx < 5 || sym_idx >= symbols_in_class.len() - 2 {
                    debug!(
                        "    dict_pos={} -> {}x{} first_pixel={:?}",
                        dict_pos, sym.width, sym.height, first_pixel
                    );
                } else if sym_idx == 5 {
                    debug!(
                        "    ... ({} symbols omitted) ...",
                        symbols_in_class.len() - 7
                    );
                }
                dict_pos += 1;
            }
        }
    }

    let mut last_height = 0;

    // 4. Encode the height classes
    for symbols_in_class in &height_classes {
        let h = symbols_in_class[0].height; // All symbols in class have same height
        // A. Encode Delta Height
        let delta_h = h as i32 - last_height as i32;
        let _ = coder.encode_integer(crate::jbig2arith::IntProc::Iadh, delta_h);
        last_height = h;

        let mut last_width = 0;
        #[cfg(debug_assertions)]
        let mut dict_pos = 0u32;

        // Debug: check symbols in this height class (disabled in release)
        #[cfg(debug_assertions)]
        {
            debug!("Height class {} has {} symbols:", h, symbols_in_class.len());
            for (i, symbol) in symbols_in_class.iter().enumerate() {
                debug!("  Symbol {}: {}x{}", i, symbol.width, symbol.height);
            }
        }

        // B. Encode symbols within this height class
        // Symbols within each height class are already sorted by width from canonicalize_dict_symbols.
        for (i, symbol) in symbols_in_class.iter().enumerate() {
            // I. Encode Delta Width
            let delta_w = symbol.width as i32 - last_width;

            // Debug output to help diagnose the issue (disabled in release)
            #[cfg(debug_assertions)]
            debug!(
                "Height class {}, Symbol {}: width={}, last_width={}, delta_w={}",
                h, i, symbol.width, last_width, delta_w
            );

            let _ = coder.encode_integer(crate::jbig2arith::IntProc::Iadw, delta_w);
            last_width = symbol.width as i32; // last_width becomes current width

            // II. Encode Symbol Bitmap using Generic Region Procedure
            let packed = symbol.packed_words();

            // Debug: dump first few symbols' bitmap data for verification
            #[cfg(debug_assertions)]
            {
                debug!(
                    "  dict_pos={} {}x{} first_word={:08x}",
                    dict_pos,
                    symbol.width,
                    symbol.height,
                    packed.get(0).unwrap_or(&0)
                );
            }

            // Verify bit-order correctness: first black pixel should match between symbol and packed data
            if let Some(expected_first_pixel) = first_black_pixel(symbol) {
                let actual_first_pixel = crate::jbig2sym::first_black_pixel_in_packed(
                    packed,
                    symbol.width,
                    symbol.height,
                );
                assert_eq!(
                    actual_first_pixel,
                    Some(expected_first_pixel),
                    "bit-order / row-order mismatch in symbol dict packer! Expected first black pixel at {:?}, got {:?}",
                    expected_first_pixel,
                    actual_first_pixel
                );
            }

            coder.encode_generic_region(
                packed,
                symbol.width,
                symbol.height,
                params.sd_template,
                &[(3, -1), (-3, -1), (2, -2), (-2, -2)],
            )?;

            #[cfg(debug_assertions)]
            {
                dict_pos += 1;
            }
        }

        // OOB marks the end of this height class.
        let _ = coder.encode_oob(IntProc::Iadw);
    }

    // Export flags come after the symbol bitmap data (run-length form).
    let _ = coder.encode_integer(IntProc::Iaex, 0);
    let _ = coder.encode_integer(IntProc::Iaex, num_export_syms as i32);

    // 5. flush the coder ONCE
    coder.flush(true);

    // 6. Append the single, complete arithmetic payload
    payload.extend(coder.as_bytes());

    Ok((payload, canonical_order))
}

/// Computes the bounding box that contains all symbol instances.
///
/// # Arguments
/// * `instances` - Slice of symbol instances to compute bounds for
/// * `all_known_symbols` - All available symbol bitmaps
///
/// # Returns
/// A tuple of (min_x, min_y, width, height) representing the bounding box
fn compute_region_bounds(
    instances: &[TextRegionSymbolInstance],
    all_known_symbols: &[&BitImage],
) -> (u32, u32, u32, u32) {
    if instances.is_empty() {
        return (0, 0, 0, 0);
    }
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x_coord = 0u32;
    let mut max_y_coord = 0u32;

    for instance in instances {
        let sym_idx = instance.symbol_id as usize;
        if sym_idx >= all_known_symbols.len() {
            continue; // Skip invalid symbol indices
        }

        let pos = Rect {
            x: instance.x as u32, // Convert i32 to u32
            y: instance.y as u32, // Convert i32 to u32
            width: crate::jbig2shared::usize_to_u32(all_known_symbols[sym_idx].width),
            height: crate::jbig2shared::usize_to_u32(all_known_symbols[sym_idx].height),
        };

        min_x = min_x.min(pos.x);
        min_y = min_y.min(pos.y);
        max_x_coord = max_x_coord.max(pos.x + pos.width);
        max_y_coord = max_y_coord.max(pos.y + pos.height);
    }

    // Handle potential underflow if max < min (shouldn't happen with valid coordinates)
    let region_width = if max_x_coord > min_x {
        max_x_coord - min_x
    } else {
        0
    };

    let region_height = if max_y_coord > min_y {
        max_y_coord - min_y
    } else {
        0
    };

    (min_x, min_y, region_width, region_height)
}

pub fn encode_refine(
    instances: &[TextRegionSymbolInstance],
    all_known_symbols: &[&BitImage],
    data: &mut Vec<u8>,
    coder: &mut Jbig2ArithCoder,
) -> Result<()> {
    // 1. Compute region bounds
    let (min_x, min_y, region_w, region_h) = compute_region_bounds(instances, all_known_symbols);
    let width = region_w.max(1);
    let height = region_h.max(1);

    // 2. Write TextRegion header (flags + params)
    // flags: TRREF=1, others zero (arithmetic coding)
    let mut flags: u8 = 0;
    flags |= 0x40; // TRREF bit
    data.push(flags);

    let params = TextRegionParams {
        width,
        height,
        x: min_x,
        y: min_y,
        ds_offset: 0,
        refine: true,
        log_strips: 0,
        ref_corner: 0,
        transposed: false,
        comb_op: 0,
        refine_template: 0,
    };
    data.extend(params.to_bytes());

    // 3. Encode number of instances
    let num_inst = instances.len() as u32;
    let _ = coder.encode_int_with_ctx(num_inst as i32, 16, IntProc::Iaai);

    // 4. Initialize an empty region buffer to track already emitted pixels
    let mut region_buf = BitImage::new(width, height).expect("region bitmap too large");

    // 5. Emit each instance
    for inst in instances {
        // IAID symbol ID
        let sym_id = inst.symbol_id;
        let _ = coder.encode_iaid(sym_id, 16);

        // Refinement deltas
        let _ = coder.encode_integer(IntProc::Iardx, inst.dx);
        let _ = coder.encode_integer(IntProc::Iardy, inst.dy);

        // If this is a refinement instance, encode pixel-by-pixel
        if inst.is_refinement {
            // locate the symbol bitmap
            if let Some(&sym) = all_known_symbols.get(sym_id as usize) {
                // offset of this instance in region coords
                let ox = inst.x as u32 - min_x;
                let oy = inst.y as u32 - min_y;

                // for each pixel in the symbol region
                for y in 0..sym.height as u32 {
                    for x in 0..sym.width as u32 {
                        // compute region coord
                        let rx = ox + x;
                        let ry = oy + y;

                        // skip out-of-bounds
                        if rx >= width || ry >= height {
                            continue;
                        }

                        // Bounds already verified above (rx < width, ry < height);
                        // use direct indexing to bypass redundant bounds checks.
                        let ref_bit = sym.get_pixel_unchecked(x as usize, y as usize) as u8;
                        let pred_bit =
                            region_buf.get_pixel_unchecked(rx as usize, ry as usize) as u8;

                        // Context = combine ref_bit, pred_bit, template (here simple sum)
                        let ctx = ((ref_bit << 1) | pred_bit) as usize;

                        // Encode the actual pixel: 1 if sym has pixel, 0 otherwise
                        let bit = ref_bit;
                        coder.encode_bit(ctx, bit != 0);

                        // Update region buffer so subsequent instances see it
                        if bit != 0 {
                            region_buf.set(rx, ry, true);
                        }
                    }
                }
            }
        }
    }

    // 6. flush and append coder payload
    coder.flush(true);
    data.extend(coder.as_bytes());

    Ok(())
}

/// Encodes a text region segment using pre-computed dictionary position maps.
///
/// Unlike `encode_text_region` which maps by list position, this function uses
/// explicit global_symbols_index → dictionary_position maps that account for the
/// canonical (filter/dedup/sort) order produced by `encode_symbol_dict_with_order`.
///
/// The decoder concatenates dictionary exports: global_dict[0..N] then local_dict[0..M].
/// Symbol IDs 0..N-1 map to the global dict, N..N+M-1 map to the local dict.
#[inline]
fn symbol_id_from_dense_maps(
    symbol_index: usize,
    global_sym_to_dict_pos: &[u32],
    num_global_dict_symbols: u32,
    local_sym_to_dict_pos: &[u32],
) -> Option<u32> {
    let global = global_sym_to_dict_pos
        .get(symbol_index)
        .copied()
        .unwrap_or(u32::MAX);
    if global != u32::MAX {
        return Some(global);
    }
    let local = local_sym_to_dict_pos
        .get(symbol_index)
        .copied()
        .unwrap_or(u32::MAX);
    if local != u32::MAX {
        Some(num_global_dict_symbols + local)
    } else {
        None
    }
}

pub fn encode_text_region_mapped(
    instances: &[SymbolInstance],
    config: &Jbig2Config,
    all_symbols: &[BitImage],
    global_sym_to_dict_pos: &[u32],
    num_global_dict_symbols: u32,
    local_sym_to_dict_pos: &[u32],
    page_num: usize,
    num_local_dict_symbols: u32,
) -> Result<Vec<u8>> {
    if instances.is_empty() {
        return Err(anyhow!("No symbol instances provided for text region"));
    }

    let debug_encoding = page_num == 0 && std::env::var("JBIG2_DEBUG").map_or(false, |v| v == "1");
    let mut enc_debug_lines: Vec<String> = Vec::new();

    let num_total_dict_symbols = num_global_dict_symbols + num_local_dict_symbols;

    let mut payload = Vec::new();
    let mut coder = Jbig2ArithCoder::new();

    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x_coord = 0u32;
    let mut max_y_coord = 0u32;

    for instance in instances {
        let sym = &all_symbols[instance.symbol_index];
        min_x = min_x.min(instance.position.x);
        min_y = min_y.min(instance.position.y);
        max_x_coord = max_x_coord.max(instance.position.x + sym.width as u32);
        max_y_coord = max_y_coord.max(instance.position.y + sym.height as u32);
    }

    let region_width = max_x_coord.saturating_sub(min_x);
    let region_height = max_y_coord.saturating_sub(min_y);

    let params = TextRegionParams {
        width: region_width,
        height: region_height,
        x: min_x,
        y: min_y,
        ds_offset: config.text_ds_offset,
        refine: config.text_refine,
        log_strips: config.text_log_strips,
        ref_corner: config.text_ref_corner,
        transposed: config.text_transposed,
        comb_op: config.text_comb_op,
        refine_template: config.text_refine_template,
    };

    payload.extend(params.to_bytes());
    payload.extend_from_slice(&(instances.len() as u32).to_be_bytes());

    let symbol_id_bits = log2up(num_total_dict_symbols.max(1)).max(1);

    #[derive(Clone, Copy)]
    struct EncodedInstance {
        strip_base: i32,
        x: i32,
        t_offset: i32,
        symbol_id: u32,
        symbol_width: i32,
    }

    let strip_width = 1i32 << params.log_strips.min(3);
    let mut encoded_instances = Vec::with_capacity(instances.len());

    for instance in instances {
        let gs_idx = instance.symbol_index;
        let sym = &all_symbols[gs_idx];

        // Map to dictionary position using the canonical order maps.
        // First check global dict, then local dict (offset by num_global_dict_symbols).
        let symbol_id = if let Some(symbol_id) = symbol_id_from_dense_maps(
            gs_idx,
            global_sym_to_dict_pos,
            num_global_dict_symbols,
            local_sym_to_dict_pos,
        ) {
            symbol_id
        } else {
            anyhow::bail!(
                "Symbol instance (global_symbols index {}) not found in any dictionary!",
                gs_idx
            );
        };

        let abs = instance.position;
        let rel_x = abs.x as i32 - min_x as i32;
        // REFCORNER=TOPLEFT (value 1): T is the top of the original bounding box.
        let rel_y = abs.y as i32 - min_y as i32;
        let strip_base = (rel_y / strip_width) * strip_width;
        let t_offset = rel_y - strip_base;

        encoded_instances.push(EncodedInstance {
            strip_base,
            x: rel_x,
            t_offset,
            symbol_id,
            symbol_width: sym.width as i32,
        });
    }

    encoded_instances.sort_by_key(|e| (e.strip_base, e.x));

    if debug_encoding {
        enc_debug_lines.push(format!("=== PAGE 0 ENCODING LOG ==="));
        enc_debug_lines.push(format!(
            "Region: {}x{} at ({},{})",
            params.width, params.height, params.x, params.y
        ));
        enc_debug_lines.push(format!(
            "min_x={} min_y={} strip_width={}",
            min_x, min_y, strip_width
        ));
        enc_debug_lines.push(format!(
            "Total instances: {}, dict symbols: {}",
            encoded_instances.len(),
            num_total_dict_symbols
        ));
        enc_debug_lines.push(String::new());

        // Show mapping from symbol_id to dimensions for reference
        enc_debug_lines.push("Symbol ID -> dimensions lookup (first 30):".to_string());
        for (dict_id, sym) in all_symbols.iter().enumerate().take(30) {
            let dict_pos = symbol_id_from_dense_maps(
                dict_id,
                global_sym_to_dict_pos,
                num_global_dict_symbols,
                local_sym_to_dict_pos,
            )
            .unwrap_or(u32::MAX);
            enc_debug_lines.push(format!(
                "  gs_idx={} -> dict_pos={} ({}x{})",
                dict_id, dict_pos, sym.width, sym.height
            ));
        }
        enc_debug_lines.push(String::new());

        enc_debug_lines.push(format!(
            "{:<6} {:<8} {:<8} {:<10} {:<8} {:<10} {:<10} {:<10}",
            "Idx", "SymID", "SymW", "StripBase", "TOffset", "RelX", "DeltaT", "DeltaS"
        ));
    }

    let mut strip_t = 0i32;
    let mut first_s = 0i32;
    let mut idx = 0usize;

    // §6.4.5 step 1: initial STRIPT value (decoder reads one IADT before the loop)
    let _ = coder.encode_integer(IntProc::Iadt, 0);

    while idx < encoded_instances.len() {
        let current_strip = encoded_instances[idx].strip_base;
        let delta_t = current_strip - strip_t;
        let _ = coder.encode_integer(IntProc::Iadt, delta_t / strip_width);

        if debug_encoding && delta_t != 0 {
            enc_debug_lines.push(format!(
                "--- strip break: IADT delta_t={} (strip_t {} → {})",
                delta_t, strip_t, current_strip
            ));
        }
        strip_t = current_strip;

        let mut first_symbol_in_strip = true;
        let mut current_s = 0i32;
        while idx < encoded_instances.len() && encoded_instances[idx].strip_base == current_strip {
            let item = encoded_instances[idx];
            let delta_s;
            if first_symbol_in_strip {
                delta_s = item.x - first_s;
                let _ = coder.encode_integer(IntProc::Iafs, delta_s);
                first_s += delta_s;
                current_s = first_s;
                first_symbol_in_strip = false;
            } else {
                delta_s = item.x - current_s;
                let _ = coder.encode_integer(IntProc::Iads, delta_s);
                current_s += delta_s;
            }

            if debug_encoding {
                enc_debug_lines.push(format!(
                    "{:<6} {:<8} {:<8} {:<10} {:<8} {:<10} {:<10} {:<10}",
                    idx,
                    item.symbol_id,
                    item.symbol_width,
                    item.strip_base,
                    item.t_offset,
                    item.x,
                    delta_t,
                    delta_s
                ));
            }

            if strip_width > 1 {
                let _ = coder.encode_integer(IntProc::Iait, item.t_offset);
            }
            let _ = coder.encode_iaid(item.symbol_id, symbol_id_bits as u8);
            current_s += item.symbol_width - 1;
            idx += 1;
        }
        let _ = coder.encode_oob(IntProc::Iads);
    }

    // Decode simulation: replay §6.4.5 from the encoder's perspective
    // to verify positions match what the decoder will compute.
    if debug_encoding {
        enc_debug_lines.push(String::new());
        enc_debug_lines.push(format!("=== DECODE SIMULATION ==="));
        enc_debug_lines.push(format!(
            "{:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}",
            "Idx", "ExpX", "ExpY", "DecS", "DecT", "AbsX", "AbsY", "Match?"
        ));

        // Collect the IADT, IAFS, IADS, IAIT values and symbol_ids in encoding order for replay
        let sbstrips = strip_width;
        let sbdsoffset = params.ds_offset as i32;
        let mut dec_stript = 0i32;
        let mut dec_firsts = 0i32;
        let mut sim_idx = 0usize;
        let mut strip_start = 0usize;

        // Group instances by strip_base (they're already sorted)
        while sim_idx < encoded_instances.len() {
            let current_strip = encoded_instances[sim_idx].strip_base;
            // Compute delta_t the same way the encoder did
            let delta_t = if sim_idx == 0 && current_strip == 0 {
                0 // first IADT is always 0 (the initial STRIPT)
            } else if sim_idx == strip_start {
                // Changed strip: the encoder emits IADT = (current_strip - prev_strip_t) / strip_width
                // But we need to replay the exact values. Let's just recompute.
                current_strip - dec_stript
            } else {
                0 // same strip, no IADT
            };

            // §6.4.5 step 2: STRIPT = STRIPT + IADT × SBSTRIPS
            if sim_idx == strip_start || sim_idx == 0 {
                let iadt_value = (current_strip - dec_stript) / sbstrips;
                dec_stript += iadt_value * sbstrips;
            }

            let mut first_in_strip = true;
            let mut dec_curs = 0i32;
            let strip_base = current_strip;

            while sim_idx < encoded_instances.len()
                && encoded_instances[sim_idx].strip_base == strip_base
            {
                let item = encoded_instances[sim_idx];

                if first_in_strip {
                    // §6.4.5: FIRSTS = FIRSTS + IAFS; CURS = FIRSTS
                    let iafs = item.x - dec_firsts;
                    dec_firsts += iafs;
                    dec_curs = dec_firsts;
                    first_in_strip = false;
                } else {
                    // §6.4.5: CURS = CURS + IADS + SBDSOFFSET
                    let iads = item.x - dec_curs;
                    dec_curs += iads + sbdsoffset;
                }

                // §6.4.5: TI = STRIPT * SBSTRIPS + IAIT (IAIT=0 when SBSTRIPS=1)
                let dec_ti = dec_stript;
                let dec_si = dec_curs;

                // Absolute page coords the decoder would compute
                let abs_x = dec_si + min_x as i32;
                let abs_y = dec_ti + min_y as i32;

                // Expected absolute coords (what we intended)
                let exp_x = item.x + min_x as i32;
                let exp_y = item.strip_base + min_y as i32;

                let ok = abs_x == exp_x && abs_y == exp_y;
                let tag = if ok { "OK" } else { "MISMATCH!" };

                if !ok || sim_idx < 60 {
                    enc_debug_lines.push(format!(
                        "{:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<8}",
                        sim_idx, exp_x, exp_y, dec_si, dec_ti, abs_x, abs_y, tag
                    ));
                }

                // §6.4.5 step 4g: CURS = CURS + WI - 1
                dec_curs += item.symbol_width - 1;
                sim_idx += 1;
            }
            strip_start = sim_idx;
        }
    }

    // Write encoding debug log for page 0
    if debug_encoding && !enc_debug_lines.is_empty() {
        let log_path = std::path::Path::new("jbig2_debug_page0.log");
        // Append to same file as matching log
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
        {
            use std::io::Write;
            let _ = writeln!(f, "");
            for line in &enc_debug_lines {
                let _ = writeln!(f, "{}", line);
            }
        }
    }

    coder.flush(true);
    payload.extend(coder.as_bytes());

    Ok(payload)
}

/// Encodes a text region with Soft Pattern Matching (SPM / refinement coding).
///
/// This is the SBREFINE=1 variant of text region encoding. For each symbol instance:
/// - Encode the symbol ID and position (same as non-refinement)
/// - Encode RI (refinement indicator) via IARI
///   - RI=0: direct substitution of the dictionary symbol (no refinement)
///   - RI=1: encode size deltas (IARDW, IARDH), position offsets (IARDX, IARDY),
///     then a pixel-by-pixel refinement region using the dictionary symbol as reference
///
/// This allows lossy symbol clustering (small dictionary) while preserving
/// per-instance fidelity through the refinement residual.
pub fn encode_text_region_with_refinement(
    instances: &[SymbolInstance],
    config: &Jbig2Config,
    all_symbols: &[BitImage],
    global_sym_to_dict_pos: &[u32],
    num_global_dict_symbols: u32,
    local_sym_to_dict_pos: &[u32],
    num_local_dict_symbols: u32,
) -> Result<Vec<u8>> {
    if instances.is_empty() {
        return Err(anyhow!("No symbol instances provided for text region"));
    }

    let num_total_dict_symbols = num_global_dict_symbols + num_local_dict_symbols;

    let mut payload = Vec::new();
    let mut coder = Jbig2ArithCoder::new();

    // Compute region bounds. For refined instances, use the actual instance
    // bitmap size (which may be larger than the prototype) so the region is
    // large enough to hold the refined glyphs.
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x_coord = 0u32;
    let mut max_y_coord = 0u32;

    for instance in instances {
        let (w, h) = if instance.needs_refinement {
            let (_, trimmed) = instance.instance_bitmap.trim();
            (trimmed.width as u32, trimmed.height as u32)
        } else {
            let sym = &all_symbols[instance.symbol_index];
            (sym.width as u32, sym.height as u32)
        };

        min_x = min_x.min(instance.position.x);
        min_y = min_y.min(instance.position.y);
        max_x_coord = max_x_coord.max(instance.position.x + w);
        max_y_coord = max_y_coord.max(instance.position.y + h);
    }

    let region_width = max_x_coord.saturating_sub(min_x);
    let region_height = max_y_coord.saturating_sub(min_y);

    // SBREFINE=1 in the text region params
    let params = TextRegionParams {
        width: region_width,
        height: region_height,
        x: min_x,
        y: min_y,
        ds_offset: config.text_ds_offset,
        refine: true, // SBREFINE = 1
        log_strips: config.text_log_strips,
        ref_corner: config.text_ref_corner,
        transposed: config.text_transposed,
        comb_op: config.text_comb_op,
        refine_template: config.text_refine_template,
    };

    payload.extend(params.to_bytes());
    payload.extend_from_slice(&(instances.len() as u32).to_be_bytes());

    let symbol_id_bits = log2up(num_total_dict_symbols.max(1)).max(1);

    // Prepare instances with dictionary mapping (same structure as non-refinement)
    #[derive(Clone)]
    struct RefinedInstance {
        strip_base: i32,
        x: i32,
        t_offset: i32,
        symbol_id: u32,
        symbol_width: i32,
        // Refinement data
        needs_refinement: bool,
        /// Index into original instances array (for accessing instance_bitmap)
        orig_idx: usize,
    }

    let strip_width = 1i32 << params.log_strips.min(3);
    let mut encoded_instances = Vec::with_capacity(instances.len());

    for (orig_idx, instance) in instances.iter().enumerate() {
        let gs_idx = instance.symbol_index;
        let sym = &all_symbols[gs_idx];

        let symbol_id = if let Some(symbol_id) = symbol_id_from_dense_maps(
            gs_idx,
            global_sym_to_dict_pos,
            num_global_dict_symbols,
            local_sym_to_dict_pos,
        ) {
            symbol_id
        } else {
            anyhow::bail!(
                "Symbol instance (global_symbols index {}) not found in any dictionary!",
                gs_idx
            );
        };

        let abs = instance.position;
        let rel_x = abs.x as i32 - min_x as i32;
        // REFCORNER=TOPLEFT (value 1): T is the top of the original bounding box.
        let rel_y = abs.y as i32 - min_y as i32;
        let strip_base = (rel_y / strip_width) * strip_width;
        let t_offset = rel_y - strip_base;

        encoded_instances.push(RefinedInstance {
            strip_base,
            x: rel_x,
            t_offset,
            symbol_id,
            symbol_width: sym.width as i32,
            needs_refinement: instance.needs_refinement,
            orig_idx,
        });
    }

    encoded_instances.sort_by_key(|e| (e.strip_base, e.x));

    // Encode strip-by-strip, symbol-by-symbol (same loop structure as non-refinement)
    let mut strip_t = 0i32;
    let mut first_s = 0i32;
    let mut idx = 0usize;

    // Default refinement AT pixel: (-1, -1), matching jbig2enc convention
    let grat: [(i8, i8); 1] = [(-1, -1)];

    // §6.4.5 step 1: initial STRIPT value (decoder reads one IADT before the loop)
    let _ = coder.encode_integer(IntProc::Iadt, 0);

    while idx < encoded_instances.len() {
        let current_strip = encoded_instances[idx].strip_base;
        let delta_t = current_strip - strip_t;
        let _ = coder.encode_integer(IntProc::Iadt, delta_t / strip_width);
        strip_t = current_strip;

        let mut first_symbol_in_strip = true;
        let mut current_s = 0i32;

        while idx < encoded_instances.len() && encoded_instances[idx].strip_base == current_strip {
            let item = &encoded_instances[idx];
            if first_symbol_in_strip {
                let delta_fs = item.x - first_s;
                let _ = coder.encode_integer(IntProc::Iafs, delta_fs);
                first_s += delta_fs;
                current_s = first_s;
                first_symbol_in_strip = false;
            } else {
                let delta_s = item.x - current_s;
                let _ = coder.encode_integer(IntProc::Iads, delta_s);
                current_s += delta_s;
            }

            if strip_width > 1 {
                let _ = coder.encode_integer(IntProc::Iait, item.t_offset);
            }

            // Symbol ID
            let _ = coder.encode_iaid(item.symbol_id, symbol_id_bits as u8);

            // ── SPM: Refinement indicator (RI) ──
            let ri = if item.needs_refinement { 1i32 } else { 0i32 };
            let _ = coder.encode_integer(IntProc::Iari, ri);

            if item.needs_refinement {
                // Get the original instance data and the prototype
                let orig_instance = &instances[item.orig_idx];
                let prototype = &all_symbols[orig_instance.symbol_index];

                // Trim the instance bitmap to get the actual glyph
                let (_, trimmed_instance) = orig_instance.instance_bitmap.trim();

                // Size deltas: how much wider/taller is the instance vs prototype
                let rdwi = trimmed_instance.width as i32 - prototype.width as i32;
                let rdhi = trimmed_instance.height as i32 - prototype.height as i32;

                let _ = coder.encode_integer(IntProc::Iardw, rdwi);
                let _ = coder.encode_integer(IntProc::Iardh, rdhi);

                // Position offsets for aligning the reference within the target.
                // Per §6.4.11.3.2: GRDX = (RDWI/2) + RDXI, GRDY = (RDHI/2) + RDYI
                // Use the pre-computed alignment offsets from clustering
                let rdxi = orig_instance.refinement_dx;
                let rdyi = orig_instance.refinement_dy;

                let _ = coder.encode_integer(IntProc::Iardx, rdxi);
                let _ = coder.encode_integer(IntProc::Iardy, rdyi);

                // Compute GRDX/GRDY for the refinement region
                let grdx = (rdwi / 2) + rdxi;
                let grdy = (rdhi / 2) + rdyi;

                // Encode the refinement region: pixel-by-pixel difference
                // between the trimmed instance and the prototype
                coder.encode_refinement_region(
                    &trimmed_instance,
                    prototype,
                    grdx,
                    grdy,
                    config.text_refine_template,
                    &grat,
                )?;

                // Reset refinement contexts between instances (per JBIG2 spec)
                coder.reset_refinement_contexts();
            }

            current_s += item.symbol_width - 1;
            idx += 1;
        }
        let _ = coder.encode_oob(IntProc::Iads);
    }

    coder.flush(true);
    payload.extend(coder.as_bytes());

    Ok(payload)
}

/// Encodes a text region segment to the output.
///
/// This function takes a list of symbols and their instances in the text region,
/// and encodes them according to JBIG2 spec §6.4.10. It supports both absolute coordinates
/// and IADW/IADH delta encoding for more efficient compression.
pub fn encode_text_region(
    instances: &[SymbolInstance],
    config: &Jbig2Config,
    all_known_symbols: &[&BitImage],
    global_dict_indices: &[usize],
    local_dict_indices: &[usize],
) -> Result<Vec<u8>> {
    // Validate instances
    if instances.is_empty() {
        return Err(anyhow!("No symbol instances provided for text region"));
    }

    // Validate global dictionary indices
    if global_dict_indices
        .iter()
        .any(|&idx| idx >= all_known_symbols.len())
    {
        return Err(anyhow!("Invalid global dictionary index in text region"));
    }

    // Validate local dictionary indices if provided
    if !local_dict_indices.is_empty() {
        if local_dict_indices
            .iter()
            .any(|&idx| idx >= all_known_symbols.len())
        {
            return Err(anyhow!("Invalid local dictionary index in text region"));
        }
    }

    // Validate each instance
    for (i, instance) in instances.iter().enumerate() {
        if instance.symbol_index >= all_known_symbols.len() {
            return Err(anyhow!(
                "Symbol instance {} references invalid symbol index {} (max {})",
                i,
                instance.symbol_index,
                all_known_symbols.len() - 1
            ));
        }

        let symbol = &all_known_symbols[instance.symbol_index];
        if instance.position.x as u64 + symbol.width as u64 > u32::MAX as u64
            || instance.position.y as u64 + symbol.height as u64 > u32::MAX as u64
        {
            return Err(anyhow!(
                "Symbol instance {} at position ({}, {}) would overflow 32-bit coordinates",
                i,
                instance.position.x,
                instance.position.y
            ));
        }
    }
    let mut payload = Vec::new();
    let mut coder = Jbig2ArithCoder::new();

    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x_coord = 0;
    let mut max_y_coord = 0;

    if instances.is_empty() {
        min_x = 0;
        min_y = 0;
    } else {
        for instance in instances {
            let pos = instance.position();
            let sym_idx_in_all_known_list = instance.symbol_index();
            let symbol_width = all_known_symbols[sym_idx_in_all_known_list].width as i32;
            let symbol_height = all_known_symbols[sym_idx_in_all_known_list].height as i32;

            min_x = min_x.min(pos.x as u32);
            min_y = min_y.min(pos.y as u32);
            max_x_coord = max_x_coord.max((pos.x as i32 + symbol_width) as u32);
            max_y_coord = max_y_coord.max((pos.y as i32 + symbol_height) as u32);
        }
    }

    let region_width = if max_x_coord > min_x {
        max_x_coord - min_x
    } else {
        0
    };
    let region_height = if max_y_coord > min_y {
        max_y_coord - min_y
    } else {
        0
    };

    let params = TextRegionParams {
        width: region_width,
        height: region_height,
        x: min_x,
        y: min_y,
        ds_offset: config.text_ds_offset,
        refine: config.text_refine,
        log_strips: config.text_log_strips,
        ref_corner: config.text_ref_corner,
        transposed: config.text_transposed,
        comb_op: config.text_comb_op,
        refine_template: config.text_refine_template,
    };
    if cfg!(debug_assertions) {
        trace!("encode_text_region: TextRegionParams details: {:?}", params);
    }
    // Write text-region header and number of instances (SBNUMINSTANCES).
    payload.extend(params.to_bytes());
    payload.extend_from_slice(&(instances.len() as u32).to_be_bytes());

    // Number of bits used by IAID symbol coding.
    let num_total_dict_symbols = (global_dict_indices.len() + local_dict_indices.len()) as u32;
    let symbol_id_bits = log2up(num_total_dict_symbols.max(1)).max(1);

    #[derive(Clone, Copy)]
    struct EncodedInstance {
        strip_base: i32,
        x: i32,
        t_offset: i32,
        symbol_id: u32,
        symbol_width: i32,
    }

    let strip_width = 1i32 << params.log_strips.min(3);
    let mut encoded_instances = Vec::with_capacity(instances.len());

    for instance in instances {
        let sym_idx_in_all_known_list = instance.symbol_index();
        let symbol_props = &all_known_symbols[sym_idx_in_all_known_list];
        let symbol_id_to_encode = if let Some(pos_global) = global_dict_indices
            .iter()
            .position(|&idx| idx == sym_idx_in_all_known_list)
        {
            pos_global as u32
        } else if let Some(pos_local) = local_dict_indices
            .iter()
            .position(|&idx| idx == sym_idx_in_all_known_list)
        {
            (global_dict_indices.len() + pos_local) as u32
        } else {
            anyhow::bail!(
                "Symbol instance (index {}) not found in referred dictionaries!",
                sym_idx_in_all_known_list
            );
        };

        // REFCORNER=TOPLEFT (value 1): T is the top of the original bounding box.
        let abs = instance.position();
        let rel_x = abs.x as i32 - min_x as i32;
        let rel_y = abs.y as i32 - min_y as i32;
        let strip_base = (rel_y / strip_width) * strip_width;
        let t_offset = rel_y - strip_base;

        encoded_instances.push(EncodedInstance {
            strip_base,
            x: rel_x,
            t_offset,
            symbol_id: symbol_id_to_encode,
            symbol_width: symbol_props.width as i32,
        });
    }

    // Sort strip-wise (top to bottom), then left to right inside each strip.
    encoded_instances.sort_by_key(|e| (e.strip_base, e.x));

    let mut strip_t = 0i32;
    let mut first_s = 0i32;
    let mut idx = 0usize;

    // §6.4.5 step 1: initial STRIPT value (decoder reads one IADT before the loop)
    let _ = coder.encode_integer(IntProc::Iadt, 0);

    while idx < encoded_instances.len() {
        let current_strip = encoded_instances[idx].strip_base;
        let delta_t = current_strip - strip_t;
        let _ = coder.encode_integer(IntProc::Iadt, delta_t / strip_width);
        strip_t = current_strip;

        let mut first_symbol_in_strip = true;
        let mut current_s = 0i32;
        while idx < encoded_instances.len() && encoded_instances[idx].strip_base == current_strip {
            let item = encoded_instances[idx];
            if first_symbol_in_strip {
                let delta_fs = item.x - first_s;
                let _ = coder.encode_integer(IntProc::Iafs, delta_fs);
                first_s += delta_fs;
                current_s = first_s;
                first_symbol_in_strip = false;
            } else {
                let delta_s = item.x - current_s;
                let _ = coder.encode_integer(IntProc::Iads, delta_s);
                current_s += delta_s;
            }

            if strip_width > 1 {
                let _ = coder.encode_integer(IntProc::Iait, item.t_offset);
            }
            let _ = coder.encode_iaid(item.symbol_id, symbol_id_bits as u8);
            current_s += item.symbol_width - 1;
            idx += 1;
        }
        let _ = coder.encode_oob(IntProc::Iads);
    }

    coder.flush(true);
    payload.extend(coder.as_bytes());

    Ok(payload)
}

// ── Union-Find helpers for symbol clustering ──────────────────────────

fn uf_find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]]; // path halving
        i = parent[i];
    }
    i
}

fn uf_union(parent: &mut [usize], rank: &mut [u32], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra == rb {
        return;
    }
    if rank[ra] < rank[rb] {
        parent[ra] = rb;
    } else if rank[ra] > rank[rb] {
        parent[rb] = ra;
    } else {
        parent[rb] = ra;
        rank[ra] += 1;
    }
}

fn compute_symbol_hash(symbol: &BitImage) -> u32 {
    let w = symbol.width as u32;
    let h = symbol.height as u32;
    (10 * h + 10000 * w) % 10000000
}

fn log2up(v: u32) -> u32 {
    if v == 0 {
        return 0;
    }
    let is_pow_of_2 = (v & (v - 1)) == 0;
    let mut r = 0;
    let mut val = v;
    while val > 1 {
        val >>= 1;
        r += 1;
    }
    r + if is_pow_of_2 { 0 } else { 1 }
}

/// Encodes a sequence of images as a JBIG2 document.
///
/// # Arguments
/// * `images` - A slice of 2D arrays containing the input images
/// * `config` - Configuration for the encoder
///
/// # Returns
/// A `Result` containing the encoded JBIG2 document as a byte vector if successful,
/// or an error if encoding fails.
pub fn encode_document(images: &[Array2<u8>], config: &Jbig2Config) -> Result<Vec<u8>> {
    let mut encoder = Jbig2Encoder::new(config);
    for image in images {
        encoder.add_page(image)?;
    }
    encoder.flush()
}

/// Represents a single symbol instance in a text region, with refinement info.
#[derive(Debug, Clone)]
pub struct TextRegionSymbolInstance {
    /// The ID of the symbol in the dictionary.
    pub symbol_id: u32,
    /// The x-coordinate of the instance's top-left corner.
    pub x: i32,
    /// The y-coordinate of the instance's top-left corner.
    pub y: i32,
    /// The horizontal refinement offset.
    pub dx: i32,
    /// The vertical refinement offset.
    pub dy: i32,
    /// Whether this instance is a refinement of a dictionary symbol.
    pub is_refinement: bool,
}

impl TextRegionSymbolInstance {
    /// Returns the position of this symbol instance as a Rect.
    pub fn position(&self) -> crate::jbig2sym::Rect {
        crate::jbig2sym::Rect {
            x: self.x as u32,
            y: self.y as u32,
            width: 0,  // These will be set by the caller
            height: 0, // These will be set by the caller
        }
    }

    /// Returns the symbol index for this instance.
    pub fn symbol_index(&self) -> usize {
        self.symbol_id as usize
    }

    /// Converts to a SymbolInstance
    pub fn to_symbol_instance(&self, symbol_bitmap: &BitImage) -> SymbolInstance {
        SymbolInstance {
            symbol_index: self.symbol_id as usize,
            position: self.position(),
            instance_bitmap: symbol_bitmap.clone(),
            needs_refinement: self.is_refinement,
            refinement_dx: self.dx,
            refinement_dy: self.dy,
        }
    }
}

pub fn build_dictionary_and_get_instances(
    symbols: &[(Rect, BitImage)],
    comparator: &mut Comparator,
) -> (Vec<BitImage>, Vec<TextRegionSymbolInstance>) {
    let mut dictionary_symbols: Vec<BitImage> = Vec::with_capacity(symbols.len());
    let mut dictionary_black_pixels = Vec::with_capacity(symbols.len());
    let mut instances = Vec::with_capacity(symbols.len());

    for (rect, symbol_image) in symbols.iter() {
        let mut found_match = false;
        // Use a 10% error threshold for matching, as recommended.
        let max_err = ((symbol_image.width * symbol_image.height) / 10).max(3) as u32;
        let symbol_black_pixels = symbol_image.count_ones();

        for (dict_idx, dict_symbol) in dictionary_symbols.iter().enumerate() {
            if symbol_image.width.abs_diff(dict_symbol.width) > MAX_DIMENSION_DELTA
                || symbol_image.height.abs_diff(dict_symbol.height) > MAX_DIMENSION_DELTA
            {
                continue;
            }

            if symbol_black_pixels.abs_diff(dictionary_black_pixels[dict_idx]) > max_err as usize {
                continue;
            }

            // Use a low max_err for finding near-duplicates
            if let Some((err, dx, dy)) = comparator.distance(symbol_image, dict_symbol, max_err) {
                instances.push(TextRegionSymbolInstance {
                    symbol_id: dict_idx as u32,
                    x: rect.x as i32,
                    y: rect.y as i32,
                    dx,
                    dy,
                    is_refinement: err > 0,
                });
                found_match = true;
                break;
            }
        }

        if !found_match {
            let new_idx = dictionary_symbols.len();
            dictionary_symbols.push(symbol_image.clone());
            dictionary_black_pixels.push(symbol_black_pixels);
            instances.push(TextRegionSymbolInstance {
                symbol_id: new_idx as u32,
                x: rect.x as i32,
                y: rect.y as i32,
                dx: 0,
                dy: 0,
                is_refinement: false,
            });
        }
    }

    (dictionary_symbols, instances)
}

/// Encodes a single page image using a symbol dictionary.
/// This is a high-level function that demonstrates the new encoding pipeline.
pub fn encode_page_with_symbol_dictionary(
    image: &BitImage,
    config: &Jbig2Config,
    next_segment_num: u32,
) -> Result<(Vec<u8>, u32)> {
    // 1. Extract symbols from the page image using CC analysis
    #[cfg(feature = "cc-analysis")]
    let extracted_symbols = {
        let dpi = 300; // Default DPI
        let losslevel = if config.is_lossless { 0 } else { 1 };
        let cc_image = analyze_page(image, dpi, losslevel);
        let shapes = cc_image.extract_shapes();
        // Convert to (Rect, BitImage) format
        shapes
            .into_iter()
            .map(|(bitmap, bbox)| {
                let rect = Rect {
                    x: bbox.xmin as u32,
                    y: bbox.ymin as u32,
                    width: bbox.width() as u32,
                    height: bbox.height() as u32,
                };
                (rect, bitmap)
            })
            .collect::<Vec<_>>()
    };
    #[cfg(not(feature = "cc-analysis"))]
    let extracted_symbols: Vec<(Rect, BitImage)> = Vec::new();

    if extracted_symbols.is_empty() {
        return Ok((Vec::new(), next_segment_num));
    }

    // 2. Build the symbol dictionary and get symbol instances
    let mut comparator = Comparator::default();
    let (dictionary_symbols, text_region_instances) =
        build_dictionary_and_get_instances(&extracted_symbols, &mut comparator);
    debug!(
        "Built dictionary with {} symbols and {} instances",
        dictionary_symbols.len(),
        text_region_instances.len()
    );

    let mut output = Vec::new();
    let mut current_segment_number = next_segment_num;

    // 3. Encode the symbol dictionary segment, getting the canonical order mapping.
    let dict_refs: Vec<&BitImage> = dictionary_symbols.iter().collect();
    let (dict_payload, canonical_order) = encode_symbol_dict_with_order(&dict_refs, config, 0)?;
    let dict_segment = Segment {
        number: current_segment_number,
        seg_type: SegmentType::SymbolDictionary,
        referred_to: Vec::new(),
        page: Some(1), // Assuming page 1 for now
        payload: dict_payload,
        ..Default::default()
    };

    dict_segment.write_into(&mut output)?;
    let dictionary_segment_number = current_segment_number;
    current_segment_number += 1;

    // Build mapping: original dictionary_symbols index → encoded dict position.
    // canonical_order[dict_pos] = original index into dict_refs (= dictionary_symbols).
    // We need the inverse for remapping text_region_instances' symbol_ids.
    let mut orig_to_dict_pos: HashMap<usize, u32> = HashMap::new();
    for (dict_pos, &orig_idx) in canonical_order.iter().enumerate() {
        orig_to_dict_pos.insert(orig_idx, dict_pos as u32);
    }

    // 4. Encode the text region segment using canonical symbol IDs.
    // Convert TextRegionSymbolInstance to SymbolInstance with corrected symbol_index.
    let symbol_instances: Vec<SymbolInstance> = text_region_instances
        .iter()
        .map(|instance| {
            let orig_id = instance.symbol_id as usize;
            let symbol_bitmap = if orig_id < dictionary_symbols.len() {
                &dictionary_symbols[orig_id]
            } else {
                &dictionary_symbols[0]
            };
            // Remap to canonical dict position
            let canonical_idx = orig_to_dict_pos.get(&orig_id).copied().unwrap_or(0) as usize;
            SymbolInstance {
                symbol_index: canonical_idx,
                position: instance.position(),
                instance_bitmap: symbol_bitmap.clone(),
                needs_refinement: instance.is_refinement,
                refinement_dx: instance.dx,
                refinement_dy: instance.dy,
            }
        })
        .collect();

    // Build ordered symbol list matching the encoded dictionary
    let ordered_dict_syms: Vec<&BitImage> = canonical_order
        .iter()
        .map(|&i| &dictionary_symbols[i])
        .collect();
    let ordered_indices: Vec<usize> = (0..canonical_order.len()).collect();
    let region_payload = encode_text_region(
        &symbol_instances,
        config,
        &ordered_dict_syms,
        &ordered_indices,
        &[],
    )?;

    let region_segment = Segment {
        number: current_segment_number,
        seg_type: SegmentType::ImmediateTextRegion,
        retain_flags: 0,
        referred_to: vec![dictionary_segment_number], // Refers to the dictionary
        page: Some(1),                                // Assuming page 1
        payload: region_payload,
        ..Default::default()
    };

    // You might want to log text_region_params here too if they are accessible
    region_segment.write_into(&mut output)?;
    current_segment_number += 1;

    Ok((output, current_segment_number))
}

pub fn get_version() -> &'static str {
    "0.2.0"
}

#[inline]
pub fn hash_key(img: &BitImage) -> HashKey {
    // Dimension-based bucketing: symbols with similar dimensions land in the same
    // bucket, enabling fuzzy matching via the Comparator during extraction.
    // The Comparator handles size differences up to MAX_DIMENSION_DELTA (10px),
    // so we bucket by (height, width) to keep buckets tight.
    let h = img.height as u64;
    let w = img.width as u64;
    HashKey(h * 10_000 + w)
}

/// Helper function to find the first black pixel in the BitImage
/// Returns (x, y) coordinates of the first black pixel, or None if no black pixels
pub fn first_black_pixel(image: &BitImage) -> Option<(usize, usize)> {
    for y in 0..image.height {
        for x in 0..image.width {
            if image.get_usize(x, y) {
                return Some((x, y));
            }
        }
    }
    None
}
