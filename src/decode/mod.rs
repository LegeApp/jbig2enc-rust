//! The decoder half of the crate.
//!
//! Implemented in phases per `jbig2decplan.md` §23. Decoder code operates on
//! untrusted PDF input: no input-reachable `unwrap`/`expect`/`assert!`, all
//! allocations checked against `DecodeLimits`.
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod error;
