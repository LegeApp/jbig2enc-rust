//! The encoder half of the crate.
//!
//! Module names retain their historical `jbig2*` prefixes for now; they are
//! renamed as part of the Phase 0 shared-module refactor (see
//! `jbig2dec-gaps-plan.md`, Gap A).

pub mod jbig2;
pub mod jbig2arith;
#[cfg(feature = "symboldict")]
pub mod jbig2cc;
pub mod jbig2classify;
pub mod jbig2comparator;
pub mod jbig2context;
pub mod jbig2cost;
pub mod jbig2enc;
pub mod jbig2halftone;
pub mod jbig2shared;
pub(crate) mod jbig2simd;
pub mod jbig2structs;
pub mod jbig2sym;
pub mod jbig2unify;
