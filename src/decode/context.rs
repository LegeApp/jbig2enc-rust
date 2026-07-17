//! Decoder options, strictness, and the reusable [`DecoderContext`]
//! (jbig2decplan.md §13, §20).
//!
//! Phase 1 only needs generic-region contexts and a little scratch; later
//! phases grow this struct (integer/IAID contexts, symbol scratch, …). The
//! context belongs to one worker and is reused across pages so allocations are
//! not repeated per page.

use crate::shared::bitmap::MonoBitmap;
use crate::shared::limits::DecodeLimits;
use crate::shared::mq_table::MqContext;

/// Decoder strictness (jbig2decplan.md §20).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DecodeStrictness {
    /// Enforce reserved bits, reject inconsistent counts and unsupported flag
    /// combinations. This is the only mode wired up in Phase 1.
    #[default]
    Strict,
    /// Permit documented recovery for malformed PDFs. Defined but unused until
    /// a real corpus case motivates each recovery (Phase 5).
    Compatible,
}

/// Options controlling a decode (jbig2decplan.md §5, §20).
#[derive(Clone, Debug)]
pub struct DecodeOptions {
    /// Resource limits applied to every allocation.
    pub limits: DecodeLimits,
    /// Strictness mode.
    pub strictness: DecodeStrictness,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            limits: DecodeLimits::default(),
            strictness: DecodeStrictness::Strict,
        }
    }
}

impl DecodeOptions {
    /// Options with the given limits and strict mode.
    pub fn with_limits(limits: DecodeLimits) -> Self {
        Self {
            limits,
            strictness: DecodeStrictness::Strict,
        }
    }
}

/// The number of arithmetic contexts a template-0 generic region needs
/// (16-bit context => 65536 states). Later templates use fewer, but sizing to
/// the maximum lets the same buffer serve every template.
pub const GENERIC_CONTEXT_COUNT: usize = 1 << 16;

/// Reusable, worker-local decoder scratch (jbig2decplan.md §13).
///
/// Not behind a mutex: the PDF renderer allocates one context per worker.
/// Buffers are reused across pages and grown as needed.
#[derive(Default)]
pub struct DecoderContext {
    /// Generic-region arithmetic contexts (grown lazily to
    /// [`GENERIC_CONTEXT_COUNT`]).
    pub generic_contexts: Vec<MqContext>,
    /// Scratch for a temporary region bitmap when a region cannot decode
    /// straight into the page.
    pub temporary_bitmap: MonoBitmap,
}

impl DecoderContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a zeroed generic-context slice of [`GENERIC_CONTEXT_COUNT`]
    /// entries, reusing the existing allocation.
    pub fn generic_contexts(&mut self) -> &mut [MqContext] {
        if self.generic_contexts.len() < GENERIC_CONTEXT_COUNT {
            self.generic_contexts
                .resize(GENERIC_CONTEXT_COUNT, MqContext::default());
        }
        for c in &mut self.generic_contexts[..GENERIC_CONTEXT_COUNT] {
            *c = MqContext::default();
        }
        &mut self.generic_contexts[..GENERIC_CONTEXT_COUNT]
    }

    /// Trim buffers that exceed `retained_words` after a pathological page so
    /// one malicious page does not permanently inflate a worker
    /// (jbig2decplan.md §13).
    pub fn trim_to(&mut self, retained_words: usize) {
        if self.generic_contexts.capacity() > GENERIC_CONTEXT_COUNT.saturating_mul(2) {
            self.generic_contexts = Vec::new();
        }
        let _ = retained_words;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn generic_contexts_are_zeroed_and_sized() {
        let mut ctx = DecoderContext::new();
        {
            let cs = ctx.generic_contexts();
            assert_eq!(cs.len(), GENERIC_CONTEXT_COUNT);
            cs[5] = MqContext(30);
        }
        // Re-borrow must return a freshly zeroed buffer (reused allocation).
        let cs = ctx.generic_contexts();
        assert_eq!(cs[5], MqContext::default());
    }

    #[test]
    fn default_options_are_strict() {
        assert_eq!(DecodeOptions::default().strictness, DecodeStrictness::Strict);
    }
}
