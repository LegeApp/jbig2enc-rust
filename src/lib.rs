//! JBIG2 Encoder in Rust
//!
//! This crate provides functionality to encode binary images into the JBIG2 format.
//! It supports both standalone JBIG2 files and PDF-embedded fragments with proper
//! global dictionary handling.

#![warn(missing_docs)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

// Re-export commonly used types
pub use ndarray::Array2;

use thiserror::Error;

/// Errors that can occur during JBIG2 encoding
#[derive(Error, Debug)]
pub enum Jbig2Error {
    /// Input buffer size mismatch
    #[error("Input buffer size mismatch: expected {expected}, got {actual} for {width}x{height} image (ratio: {ratio:.3})")]
    BufferSizeMismatch {
        expected: usize,
        actual: usize,
        width: u32,
        height: u32,
        ratio: f64,
    },

    /// Buffer too small for operation
    #[error("Buffer too small: expected {expected}, got {actual}")]
    BufferTooSmall { expected: usize, actual: usize },

    /// Array shape error during conversion
    #[error("Array shape error")]
    ArrayShapeError {
        #[from]
        source: ndarray::ShapeError,
    },

    /// Encoding failed
    #[error("Encoding failed: {message}")]
    EncodingFailed { message: String },

    /// Dictionary creation failed  
    #[error("Dictionary creation failed: {message}")]
    DictionaryFailed { message: String },

    /// Packed binary data detected when unpacked expected
    #[error("Input appears to be packed binary data (1 bit per pixel), but JBIG2 encoder expects unpacked data (1 byte per pixel)")]
    PackedDataDetected,

    /// Stream count mismatch
    #[error("Expected {expected} stream(s) for single image encoding, got {actual}")]
    StreamCountMismatch { expected: usize, actual: usize },

    /// Segment writing failed
    #[error("Failed to write {segment_type} segment: {message}")]
    SegmentWriteFailed {
        segment_type: String,
        message: String,
    },
}

// Module declarations
pub mod jbig2arith;
pub mod jbig2comparator;
pub mod jbig2enc;
pub mod jbig2lutz;
pub mod jbig2pdf;
pub mod jbig2shared;
pub mod jbig2structs;
pub mod jbig2sym;

// Re-export the main encode functions and config
pub use crate::jbig2arith::Jbig2ArithCoder;
pub use jbig2enc::encode_document;
pub use jbig2structs::Jbig2Config;

use jbig2enc::Jbig2Encoder;
use log::info;
use std::{env, io::Write};

// Constants for default thresholds (symbol classification only)
const JBIG2_THRESHOLD_DEF: f32 = 0.92;
const JBIG2_WEIGHT_DEF: f32 = 0.5;

/// Result of JBIG2 encoding with separate global and page data for PDF embedding
#[derive(Debug, Clone)]
pub struct Jbig2EncodeResult {
    /// Global dictionary data (if any) - should be stored as separate PDF object
    pub global_data: Option<Vec<u8>>,
    /// Page-specific data - the actual image stream
    pub page_data: Vec<u8>,
}

/// Context for JBIG2 encoding operations
#[derive(Debug, Clone)]
pub struct Jbig2Context {
    /// The underlying configuration
    config: Jbig2Config,

    // Legacy fields for backward compatibility
    threshold: f32,
    weight: f32,
    pdf_mode: bool,
}

impl Default for Jbig2Context {
    fn default() -> Self {
        Self {
            config: Jbig2Config::default(),
            threshold: JBIG2_THRESHOLD_DEF,
            weight: JBIG2_WEIGHT_DEF,
            pdf_mode: false,
        }
    }
}

impl Jbig2Context {
    /// Create a new context with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new context with specified PDF mode
    pub fn with_pdf_mode(pdf_mode: bool) -> Self {
        Self {
            config: Jbig2Config::default(),
            threshold: JBIG2_THRESHOLD_DEF,
            weight: JBIG2_WEIGHT_DEF,
            pdf_mode,
        }
    }

    /// Create a new context with lossless configuration (no symbol dictionaries)
    /// This is useful for PDF embedding when symbol dictionaries cause display issues
    pub fn with_lossless_config(pdf_mode: bool) -> Self {
        Self {
            config: Jbig2Config::lossless(),
            threshold: JBIG2_THRESHOLD_DEF,
            weight: JBIG2_WEIGHT_DEF,
            pdf_mode,
        }
    }

    /// Create a new context with custom config
    pub fn with_config(config: Jbig2Config, pdf_mode: bool) -> Self {
        Self {
            config,
            threshold: JBIG2_THRESHOLD_DEF,
            weight: JBIG2_WEIGHT_DEF,
            pdf_mode,
        }
    }

    /// Get the PDF mode setting
    pub fn get_pdf_mode(&self) -> bool {
        self.pdf_mode
    }

    /// Get the symbol mode setting
    pub fn get_symbol_mode(&self) -> bool {
        self.config.symbol_mode
    }


    /// Get the DPI setting
    pub fn get_dpi(&self) -> u32 {
        if self.config.generic.dpi == 0 {
            300
        } else {
            self.config.generic.dpi
        }
    }
}

/// Main encoding function that handles both standalone and PDF fragment modes
///
/// This function encodes a single binary image into JBIG2 format. When PDF mode is enabled,
/// it returns separate global dictionary and page data that can be properly embedded in PDF.
///
/// # Arguments
/// * `input` - Binary image data (0/1 values)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `pdf_mode` - Whether to create PDF fragments (true) or standalone file (false)
///
/// # Returns
/// A `Jbig2EncodeResult` containing separate global and page data for PDF embedding,
/// or combined data for standalone files.
pub fn encode_single_image(
    input: &[u8],
    width: u32,
    height: u32,
    pdf_mode: bool,
) -> Result<Jbig2EncodeResult, Jbig2Error> {
    let expected_len = width as usize * height as usize;
    if input.len() < expected_len {
        // Check if this might be packed binary data (1 bit per pixel)
        let packed_size = (width as usize * height as usize + 7) / 8;
        if input.len() == packed_size {
            return Err(Jbig2Error::PackedDataDetected);
        }

        return Err(Jbig2Error::BufferSizeMismatch {
            expected: expected_len,
            actual: input.len(),
            width,
            height,
            ratio: input.len() as f64 / expected_len as f64,
        });
    }

    // Convert to ndarray format
    let array = Array2::from_shape_vec((height as usize, width as usize), input.to_vec())?;

    // Create context with appropriate PDF mode setting
    let ctx = Jbig2Context::with_pdf_mode(pdf_mode);

    // Use the existing encode_rois function
    let (global_dict, streams) =
        encode_rois(&[array], ctx).map_err(|e| Jbig2Error::EncodingFailed {
            message: e.to_string(),
        })?;

    // Handle the result based on whether we have a global dictionary
    if let Some(global) = global_dict {
        // Global dictionary mode (symbol_mode = true)
        let page_data = streams.into_iter().next().unwrap_or_default();
        Ok(Jbig2EncodeResult {
            page_data,
            global_data: Some(global),
        })
    } else {
        // Standalone mode (symbol_mode = false or no symbols found)
        let page_data = streams.into_iter().next().unwrap_or_default();
        Ok(Jbig2EncodeResult {
            page_data,
            global_data: None,
        })
    }
}

/// Encodes a single binary image into JBIG2 format using lossless configuration.
///
/// This function forces symbol_mode = false to create standalone JBIG2 streams
/// without global dictionaries, which can resolve PDF display issues.
///
/// # Arguments
/// * `input` - Binary image data (0/1 values)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `pdf_mode` - Whether to create PDF fragments (true) or standalone file (false)
///
/// # Returns
/// A `Jbig2EncodeResult` containing page data without global dictionary
pub fn encode_single_image_lossless(
    input: &[u8],
    width: u32,
    height: u32,
    pdf_mode: bool,
) -> Result<Jbig2EncodeResult, Jbig2Error> {
    let expected_len = width as usize * height as usize;
    if input.len() < expected_len {
        // Check if this might be packed binary data (1 bit per pixel)
        let packed_size = (width as usize * height as usize + 7) / 8;
        if input.len() == packed_size {
            return Err(Jbig2Error::PackedDataDetected);
        }

        return Err(Jbig2Error::BufferSizeMismatch {
            expected: expected_len,
            actual: input.len(),
            width,
            height,
            ratio: input.len() as f64 / expected_len as f64,
        });
    }

    // Convert to ndarray format
    let array = Array2::from_shape_vec((height as usize, width as usize), input.to_vec())?;

    // Create context with lossless configuration (symbol_mode = false)
    let ctx = Jbig2Context::with_lossless_config(pdf_mode);

    // Use the existing encode_rois function
    let (global_dict, streams) =
        encode_rois(&[array], ctx).map_err(|e| Jbig2Error::EncodingFailed {
            message: e.to_string(),
        })?;

    // Lossless mode should not create global dictionaries
    let page_data = streams.into_iter().next().unwrap_or_default();
    Ok(Jbig2EncodeResult {
        page_data,
        global_data: global_dict, // Should be None in lossless mode
    })
}

/// Encodes a list of text-only binary PBM ROIs into JBIG2 streams.
///
/// # Arguments
/// * `rois` - A slice of 2D arrays where each array represents a binary image (0/255 or 0/1 values)
/// * `ctx` - JBIG2 encoding context with configuration
pub fn encode_rois(
    rois: &[Array2<u8>],
    ctx: Jbig2Context,
) -> Result<(Option<Vec<u8>>, Vec<Vec<u8>>), Box<dyn std::error::Error>> {
    if rois.is_empty() {
        return Ok((None, Vec::new()));
    }

    info!(
        "Processing {} ROIs in PDF mode: {}",
        rois.len(),
        ctx.get_pdf_mode()
    );

    // Initialize encoder configuration
    let mut enc_config = Jbig2Config::default();
    enc_config.symbol_mode = ctx.get_symbol_mode();
    enc_config.dpi = ctx.get_dpi();
    enc_config.want_full_headers = !ctx.get_pdf_mode(); // PDF mode shouldn't have file headers
    enc_config.auto_thresh = false; // Disable auto-threshold to avoid index errors

    // For PDF mode with symbol encoding, create global dictionary
    let global_dict = if ctx.get_symbol_mode() && ctx.get_pdf_mode() {
        let dict_data =
            build_page_dict(rois, &enc_config, &ctx).map_err(|e| Jbig2Error::DictionaryFailed {
                message: e.to_string(),
            })?;
        Some(dict_data)
    } else {
        None
    };

    let mut roi_streams = Vec::with_capacity(rois.len());

    for roi in rois {
        let mut encoder = Jbig2Encoder::new(&enc_config);

        // Add the image data to the encoder.
        encoder.add_page(roi).map_err(|e| e.to_string())?;

        // Encode the document to get the final stream.
        // This will produce a stream with or without headers based on `enc_config`.
        let stream = encoder.flush().map_err(|e| e.to_string())?;
        roi_streams.push(stream);
    }

    Ok((global_dict, roi_streams))
}

/// Encodes a dictionary covering every ROI on a page.
fn build_page_dict(
    rois: &[Array2<u8>],
    cfg: &Jbig2Config,
    ctx: &Jbig2Context,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Create encoder with the configuration
    let mut encoder = Jbig2Encoder::new(cfg);

    // Set PDF mode if needed
    if ctx.get_pdf_mode() {
        encoder = encoder.dict_only();
    }

    // Add all ROIs to the encoder
    for roi in rois {
        encoder
            .add_page(roi)
            .map_err(|e| format!("Failed to add page: {}", e))?;
    }

    encoder
        .flush_dict()
        .map_err(|e| format!("Failed to flush dictionary: {}", e).into())
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
    let build_type = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    format!("{} (built with {})", build_ts, build_type)
}
