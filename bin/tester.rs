use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use clap::Parser;
use jbig2::jbig2sym::array_to_bitimage;
use jbig2::jbig2enc::{Jbig2EncConfig, Jbig2Encoder};
use jbig2::jbig2structs::{FileHeader, Segment, SegmentType};
use anyhow::{anyhow, Result};
#[cfg(any(feature = "trace_encoder", feature = "trace_arith"))]
use tracing_subscriber::{fmt, EnvFilter, prelude::*};
use ndarray::Array2;
use log::info;

// Define no-op macros when tracing is not available
#[cfg(not(any(feature = "trace_encoder", feature = "trace_arith")))]
macro_rules! info {
    ($($arg:tt)*) => { println!($($arg)*) };
}

#[cfg(not(any(feature = "trace_encoder", feature = "trace_arith")))]
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {};
}

#[cfg(not(any(feature = "trace_encoder", feature = "trace_arith")))]
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}

/// Initialize logging with appropriate verbosity based on features
#[cfg(any(feature = "trace_encoder", feature = "trace_arith"))]
fn init_logging() -> Result<()> {
    let filter = if std::env::var_os("RUST_LOG").is_some() {
        EnvFilter::from_default_env()
    } else {
        let mut filter = EnvFilter::new("info");
        if cfg!(feature = "trace_encoder") {
            filter = filter.add_directive("jbig2::jbig2enc=debug".parse()?);
        }
        if cfg!(feature = "trace_arith") {
            filter = filter.add_directive("jbig2_arith=debug".parse()?);
        }
        filter
    };
    let subscriber = fmt()
        .with_ansi(false)
        .with_writer(std::io::stdout)
        .with_env_filter(filter)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global default subscriber");
    info!("Tracing initialized. Running with debug output.");
    Ok(())
}

#[cfg(not(any(feature = "trace_encoder", feature = "trace_arith")))]
fn init_logging() -> Result<()> {
    println!("Running with basic output (enable 'trace_encoder' or 'trace_arith' features for detailed logging)");
    Ok(())
}

use jbig2::jbig2pdf::{Jbig2Input, Jbig2Roi};
use jbig2::jbig2pdf;

/// JBIG2 Encoder Tester Application
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input PBM file path
    #[clap(short, long, value_parser, default_value = "./test_image.pbm")]
    input: String,

    /// Output file path (JBIG2 for standalone, PDF for --pdf mode)
    #[clap(short, long, value_parser, default_value = "./output.jb2")]
    output: String,

    /// Enable PDF mode (generates fragments, tester wraps for jbig2dec)
    #[clap(short, long)]
    pdf_mode: bool,

    /// Enable symbol mode for encoding (default is generic region)
    #[clap(short, long)]
    symbol_mode: bool,
    
    /// Enable console output in addition to file logging
    #[clap(long)]
    console: bool,
    
    /// Override the log directory [default: ./logs]
    #[clap(long, value_parser)]
    log_dir: Option<String>,
    
    /// Override the log file name [default: jbig2_trace.log]
    #[clap(long, value_parser)]
    log_file: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Set up environment for logging
    if args.console {
        std::env::set_var("RUST_LOG_CONSOLE", "1");
    }
    if let Some(dir) = &args.log_dir {
        std::env::set_var("JBIG2_LOG_DIR", dir);
    }
    if let Some(file) = &args.log_file {
        std::env::set_var("JBIG2_LOG_FILE", file);
    }
    
    // Initialize logging
    init_logging()?;
    
    info!("JBIG2 Encoder Tester");
    info!("Command line: {}", std::env::args().collect::<Vec<_>>().join(" "));
    
    if args.pdf_mode {
        info!("PDF Mode: ENABLED (fragments will be wrapped for jbig2dec)");
    }
    
    if args.symbol_mode {
        info!("Encoding Mode: Symbol Dictionary");
    } else {
        info!("Encoding Mode: Generic Region");
    }

    // 1. Read and parse the input PBM image
    info!("Reading PBM file: {}", args.input);
    let (width, height, data) = read_pbm(&args.input)?;
    info!("PBM dimensions: {}x{} ({} bytes)", width, height, data.len());

    // 2. Convert raw PBM data to the library's internal BitImage format
    let mut image_array = Array2::<u8>::from_shape_fn((height, width), |(r, c)| {
        let byte_idx = r * ((width + 7) / 8) + (c / 8);
        let bit_idx_in_byte = c % 8;
        // PBM black is 1, but our library expects 0 for black. Invert the pixel.
        if (data[byte_idx] >> (7 - bit_idx_in_byte)) & 1 == 0 { 1 } else { 0 }
    });
    info!("Converted PBM to ndarray::Array2<u8>.");

    // 3. Configure the encoder
    let config = Jbig2EncConfig {
        symbol_mode: args.symbol_mode,
        ..Jbig2EncConfig::default()
    };
    info!("Using encoder config: {:?}", config);

    // 4. Initialize and use the Jbig2Encoder
    let mut encoder = Jbig2Encoder::new(&config);
    encoder.add_page(&image_array)?;

    println!("Finalizing JBIG2 encoding...");
    let encoded_data = encoder.flush()?;
    println!("JBIG2 encoding complete.");

    // 5. Write the output file
    if args.pdf_mode {
        let output_path = args.output.replace(".jb2", ".pdf");
        println!("PDF Mode: Creating PDF file at {}", output_path);

        let jbig2_roi = Jbig2Roi {
            stream: encoded_data,
            width: width as u32,
            height: height as u32,
            xres: 300, 
            yres: 300, 
            pdf_x: 0.0,
            pdf_y: 0.0,
            pdf_width: width as f32,
            pdf_height: height as f32,
            page_index: 0,
        };

        let jbig2_input = Jbig2Input::from_memory(None, vec![jbig2_roi]);
        let page_dims = vec![(width as f32, height as f32)];

        jbig2pdf::create_jbig2_pdf(jbig2_input, &output_path, &page_dims)?;
        println!("Successfully created PDF with JBIG2 fragment.");
    } else {
        let mut file = File::create(&args.output)?;
        println!("Standalone mode: Wrapping content with FileHeader and EndOfFile segments for jbig2dec compatibility");
        
        let file_header = FileHeader {
            organisation_type: true,  
            unknown_n_pages: false,
            n_pages: 1,  
        };
        file.write_all(&file_header.to_bytes())?;
        
        file.write_all(&encoded_data)?;
        
        let eof_segment = Segment {
            number: 4, 
            seg_type: SegmentType::EndOfFile,
            deferred_non_retain: false,
            retain_flags: 0,
            page_association_type: 2,  
            referred_to: Vec::new(),
            page: None,
            payload: Vec::new(),
        };
        eof_segment.write_into(&mut file)?;
        
        let file_metadata = file.metadata()?;
        println!("Writing encoded data to: {} ({} bytes, Standalone mode with wrapper)", 
                 args.output, file_metadata.len());
    }

    println!("Encoded data written successfully.");

    Ok(())
}

fn read_pbm(path: &str) -> Result<(usize, usize, Vec<u8>)> {
    let mut file = File::open(path)?;
    let mut reader = BufReader::new(&mut file);

    let mut line = String::new();
    reader.read_line(&mut line)?; 
    if line.trim() != "P4" {
        return Err(anyhow!("Unsupported PBM magic number: {}", line.trim()));
    }
    println!("PBM magic number: {}", line.trim());

    line.clear();
    loop {
        reader.read_line(&mut line)?;
        let trimmed_line = line.trim();
        if !trimmed_line.starts_with('#') && !trimmed_line.is_empty() {
            break;
        }
        line.clear();
    }
    let parts: Vec<&str> = line.trim().split_whitespace().collect();
    if parts.len() != 2 {
        return Err(anyhow!("Invalid PBM dimensions line: {}", line.trim()));
    }
    let width = parts[0].parse::<usize>()?;
    let height = parts[1].parse::<usize>()?;
    println!("PBM dimensions: {}x{}", width, height);

    let current_file_pos = reader.stream_position()?;
    file.seek(SeekFrom::Start(current_file_pos))?;

    let width_in_bytes = (width + 7) / 8;
    let expected_data_len = height * width_in_bytes;
    println!("Calculated PBM data length ( H * ((W+7)/8) ): {} bytes", expected_data_len);
    let mut data = vec![0u8; expected_data_len];
    file.read_exact(&mut data)?;

    Ok((width, height, data))
}
