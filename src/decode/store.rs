//! Decoded-segment storage and dependency resolution (jbig2decplan.md §19).
//!
//! A typed [`DecodedSegment`] enum (never `Box<dyn Any>`) keyed by segment
//! number in an [`FxHashMap`]. Text regions resolve their referred symbol
//! dictionaries by number; a missing referred segment or a duplicate segment
//! number is a typed error.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::decode::error::DecodeError;
use crate::decode::pattern_dictionary::PatternDictionary;
use crate::decode::symbol_dictionary::SymbolDictionary;
use crate::shared::bitmap::MonoBitmap;

/// A decoded segment resource, resolvable by later segments.
#[derive(Clone)]
pub enum DecodedSegment {
    /// A symbol dictionary's exported symbols.
    SymbolDictionary(Arc<SymbolDictionary>),
    /// A pattern dictionary's patterns (referred by halftone regions).
    PatternDictionary(Arc<PatternDictionary>),
    /// A retained region bitmap (intermediate regions; not produced by the
    /// current encoder but modelled for completeness).
    Region(Arc<MonoBitmap>),
    /// A structural/metadata segment with no decoded resource.
    Metadata,
}

/// Segment-number keyed store of decoded resources (jbig2decplan.md §19).
#[derive(Default, Clone)]
pub struct SegmentStore {
    values: FxHashMap<u32, DecodedSegment>,
}

impl SegmentStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a decoded segment, rejecting a duplicate segment number.
    pub fn insert(&mut self, number: u32, seg: DecodedSegment) -> Result<(), DecodeError> {
        if self.values.contains_key(&number) {
            return Err(DecodeError::DuplicateSegment { number });
        }
        self.values.insert(number, seg);
        Ok(())
    }

    /// Look up a decoded segment by number.
    #[inline]
    pub fn get(&self, number: u32) -> Option<&DecodedSegment> {
        self.values.get(&number)
    }

    /// Resolve a symbol dictionary by number, erroring if it is missing or of
    /// the wrong type. `segment` is the referring segment (for the error).
    pub fn symbol_dictionary(
        &self,
        segment: u32,
        referred: u32,
    ) -> Result<&Arc<SymbolDictionary>, DecodeError> {
        match self.values.get(&referred) {
            Some(DecodedSegment::SymbolDictionary(d)) => Ok(d),
            Some(_) => Err(DecodeError::WrongReferredSegmentType { segment, referred }),
            None => Err(DecodeError::MissingReferredSegment { segment, referred }),
        }
    }

    /// Resolve a pattern dictionary by number, checking `self` then `globals`,
    /// erroring if it is missing or of the wrong type. `segment` is the
    /// referring (halftone) segment (for the error).
    pub fn pattern_dictionary<'a>(
        &'a self,
        segment: u32,
        referred: &[u32],
        globals: Option<&'a SegmentStore>,
    ) -> Result<&'a Arc<PatternDictionary>, DecodeError> {
        for &rn in referred {
            match self.values.get(&rn).or_else(|| globals.and_then(|g| g.get(rn))) {
                Some(DecodedSegment::PatternDictionary(p)) => return Ok(p),
                _ => continue,
            }
        }
        // No referred pattern dictionary found among the resolvable segments.
        let referred_num = referred.first().copied().unwrap_or(0);
        Err(DecodeError::MissingReferredSegment {
            segment,
            referred: referred_num,
        })
    }

    /// Gather the exported symbols of every referred *symbol dictionary*, in
    /// reference order, checking `self` first then `globals`
    /// (jbig2decplan.md §16, §17). A referred number present in neither store is
    /// a missing-referred-segment error; a referred non-dictionary (e.g. a
    /// Huffman table) contributes no symbols.
    pub fn gather_symbols(
        &self,
        segment: u32,
        referred: &[u32],
        globals: Option<&SegmentStore>,
    ) -> Result<Vec<Arc<MonoBitmap>>, DecodeError> {
        let mut out: Vec<Arc<MonoBitmap>> = Vec::new();
        for &rn in referred {
            let dict = match self.values.get(&rn) {
                Some(DecodedSegment::SymbolDictionary(d)) => Some(d),
                Some(_) => None,
                None => match globals.and_then(|g| g.get(rn)) {
                    Some(DecodedSegment::SymbolDictionary(d)) => Some(d),
                    Some(_) => None,
                    None => {
                        return Err(DecodeError::MissingReferredSegment {
                            segment,
                            referred: rn,
                        });
                    }
                },
            };
            if let Some(d) = dict {
                out.extend(d.exported_symbols.iter().cloned());
            }
        }
        Ok(out)
    }
}
