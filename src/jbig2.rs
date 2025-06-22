//! JBIG2 Encoder in Rust
//!
//! This crate provides functionality to encode binary images into the JBIG2 format.

#![allow(missing_docs)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

// Re-export commonly used types
pub use ndarray::Array2;

// Module declarations
pub mod jbig2arith;
pub mod jbig2comparator;
pub mod jbig2enc;
pub mod jbig2lutz;
pub mod jbig2pdf;
pub mod jbig2shared;
pub mod jbig2structs;
pub mod jbig2sym;

// Re-export the main encode functions
pub use crate::jbig2arith::Jbig2ArithCoder;
pub use jbig2enc::{encode_document, encode_page_with_symbol_dictionary, Jbig2EncConfig};

use jbig2enc::Jbig2Encoder;
use log::{debug, info}; // For logging
use std::{env, error::Error};

// Constants for default thresholds (symbol classification only)
const JBIG2_THRESHOLD_DEF: f32 = 0.92;
const JBIG2_WEIGHT_DEF: f32 = 0.5;

/// Context for JBIG2 encoding operations
#[derive(Debug, Clone, Default)]
pub struct Jbig2Context {
    threshold: f32,
    weight: f32,
    symbol_mode: bool,
    refine: bool,
    duplicate_line_removal: bool,
    auto_thresh: bool,
    hash: bool,
    dpi: u32,
    pdf_mode: bool,
}

impl Jbig2Context {
    /// Create a new context with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the threshold value
    pub fn get_threshold(&self) -> f32 {
        self.threshold
    }

    /// Get the weight value
    pub fn get_weight(&self) -> f32 {
        self.weight
    }

    /// Get the symbol mode setting
    pub fn get_symbol_mode(&self) -> bool {
        self.symbol_mode
    }

    /// Get the refine setting
    pub fn get_refine(&self) -> bool {
        self.refine
    }

    /// Get the duplicate line removal setting
    pub fn get_duplicate_line_removal(&self) -> bool {
        self.duplicate_line_removal
    }

    /// Get the auto threshold setting
    pub fn get_auto_thresh(&self) -> bool {
        self.auto_thresh
    }

    /// Get the hash setting
    pub fn get_use_hash(&self) -> bool {
        self.hash
    }

    /// Get the DPI setting
    pub fn get_dpi(&self) -> u32 {
        if self.dpi == 0 {
            300
        } else {
            self.dpi
        }
    }

    /// Get the PDF mode setting
    pub fn get_pdf_mode(&self) -> bool {
        self.pdf_mode
    }
}

/// Encodes a list of text-only binary PBM ROIs into JBIG2 streams.
///
/// # Arguments
/// * `rois` - A slice of 2D arrays where each array represents a binary image (0/255 or 0/1 values)
pub fn encode_rois(
    rois: &[Array2<u8>],
    ctx: Jbig2Context,
) -> Result<(Option<Vec<u8>>, Vec<Vec<u8>>), Box<dyn Error>> {
    if rois.is_empty() {
        return Ok((None, Vec::new()));
    }

    info!("Processing {} text ROIs...", rois.len());

    // Initialize encoder configuration with immutable fields only
    // Note: pdf_mode and other runtime flags are now handled by EncoderState
    let enc_config = Jbig2EncConfig {
        symbol_mode: ctx.get_symbol_mode(),
        refine: ctx.get_refine(),
        refine_template: 0, // Default value for refine_template
        duplicate_line_removal: ctx.get_duplicate_line_removal(),
        auto_thresh: ctx.get_auto_thresh(),
        hash: ctx.get_use_hash(),
        dpi: ctx.get_dpi(),
        want_full_headers: !ctx.get_pdf_mode(), // PDF mode uses fragments
    };

    // If we have a global dictionary, encode it and all ROIs
    let global_dict = if ctx.get_symbol_mode() && ctx.get_pdf_mode() {
        let dict_data = build_page_dict(rois, &enc_config, &ctx)?;
        Some(dict_data)
    } else {
        None
    };

    let mut roi_streams = Vec::with_capacity(rois.len());

    // Encode each ROI
    for (idx, roi) in rois.iter().enumerate() {
        debug!("Encoding ROI {}...", idx);
        let bitimage = crate::jbig2sym::array_to_bitimage(roi);
        // Use template 1 with AT-pixels for ROI encoding (better compression for similar regions)
        let at_pixels = [(-1, -1), (0, -1), (1, -1), (-1, 0)];
        let roi_stream = Jbig2ArithCoder::encode_generic_payload(&bitimage, 1, &at_pixels)?;
        roi_streams.push(roi_stream);
    }

    Ok((global_dict, roi_streams))
}

/// Helper struct to manage symbol dictionary across ROIs (for future per-ROI splitting).
struct RoiEncoder<'a> {
    encoder: Jbig2Encoder<'a>,
    roi_indices: Vec<usize>, // Maps ROIs to their symbol instances
    dict: Vec<u8>,
}

/// Encodes a dictionary covering every ROI on a page.
pub fn build_page_dict(
    rois: &[Array2<u8>],
    cfg: &Jbig2EncConfig,
    ctx: &Jbig2Context,
) -> Result<Vec<u8>, anyhow::Error> {
    // Create encoder with the configuration
    let mut encoder = Jbig2Encoder::new(cfg);

    // Set PDF mode if needed
    if ctx.get_pdf_mode() {
        encoder = encoder.dict_only();
    }

    // Add all ROIs to the encoder
    for roi in rois {
        encoder.add_page(roi)?;
    }

    encoder.flush_dict()
}

impl<'a> RoiEncoder<'a> {
    fn new(dict: Vec<u8>, config: &'a Jbig2EncConfig) -> Self {
        let mut encoder = Jbig2Encoder::new(config);
        if config.want_full_headers {
            encoder = encoder.dict_only();
        }

        Self {
            encoder,
            roi_indices: Vec::new(),
            dict,
        }
    }

    fn add_roi(&mut self, roi: &Array2<u8>) -> Result<(), Box<dyn Error>> {
        self.encoder.add_page(roi)?;
        let page_count = self.encoder.get_page_count();
        if page_count > 0 {
            self.roi_indices.push(page_count - 1);
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
        let output = self.encoder.flush()?;
        // The global dictionary is already handled by build_page_dict.
        // This RoiEncoder is for individual ROI streams, which should be
        // encoded using the shared dictionary.
        // The `flush()` method of Jbig2Encoder should produce the ROI stream.
        Ok(vec![output])
    }
}

/// Get the version string for the crate
pub fn get_version() -> String {
    let enc_version = option_env!("JBIG2ENC_VERSION").unwrap_or("unknown");
    format!(
        "jbig2-rs {}, jbig2enc {}",
        env!("CARGO_PKG_VERSION"),
        enc_version
    )
}

/// Get the build information string
pub fn get_build_info() -> String {
    let build_ts = option_env!("VERGEN_BUILD_TIMESTAMP").unwrap_or("unknown");
    format!(
        "{} (built with {})",
        build_ts,

        if cfg!(debug_assertions) { "debug" } else { "release" }

        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }

    )
}
