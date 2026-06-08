# Refactor plan: split `jbig2enc.rs` (~7100 lines) into a `symbol/` module tree

## Status: implemented, with two deviations from the file map below

The refactor landed as planned with two adjustments forced by constraints
that weren't visible until the move was attempted:

1. **`add_page`, `add_page_bitimage`, `collect_symbols`, `flush`, and
   `flush_pdf_split` stay in `mod.rs`, not `symbol/extraction.rs`.** They are
   `pub fn` methods that external integration-test crates call directly on
   `Jbig2Encoder` (`tests/encoder.rs`, `tests/headsup_c_vs_rust.rs`,
   `tests/refine_pdf_lossless.rs`, `tests/roundtrip_after_fix.rs`,
   `tests/multi_page_benchmark.rs`). `mod symbol` is — and must stay —
   **private** (see "nest, don't sibling" below): a `pub fn` defined inside a
   private module is not externally reachable through `value.method()`
   syntax, regardless of the method's own visibility. Moving them would
   either break those test crates or force `mod symbol` to become `pub`,
   which re-opens the encapsulation problem this plan exists to avoid. "Public
   paths must not change" wins over the file map's specific assignment.

2. **`add_page_bitimage` was not decomposed** — it is the plan's second
   documented monster-function exception (alongside `plan_document`), with
   the rationale recorded as a comment directly above it in `mod.rs`. Its
   ~546-line body threads a dozen mutable locals (`comparator`,
   `debug_lines`, `cc_index`, `symbol_instances`, `instance_bitmap`,
   `recent_cache`, `sym_unify_anchor_map`, the `sym_unify_*` counters)
   through three nested match-searches (recent-cache, anchor, hash-bucket)
   sharing early-exit control flow. Splitting it would require passing all of
   that state across function boundaries via a dozen `&mut` params or a new
   carrier struct — for a bit-exact encoder where a slip would only surface
   as silently different output bytes. The risk wasn't worth the line-count
   savings, especially with `mod.rs` already at ~1050 lines (close to the
   ~1000 target) without it.

`symbol/extraction.rs` ended up holding only `segment_symbols` (41 lines) —
smaller than planned, but correctly so given (1) above.

## Goal

`src/jbig2enc.rs` has grown to ~7100 lines (the rest of the crate's files are
300–1200 lines). Split it into a `jbig2enc/` directory with a `symbol/`
subfolder holding the symbol-dictionary/text-region machinery, targeting
~1000 lines per file. Two functions (`plan_document`, `add_page_bitimage`)
are large enough that hitting that target for the files containing them
requires decomposing them first — see "The monster-function exception" below.

## The key structural decision: nest, don't sibling

`src/symbol/` declared from `lib.rs` would force every private field of
`Jbig2Encoder` and every private helper type (`PlannedDocument`,
`EncoderState`, `SymbolDictLayout`, `RefinementPlan`, …) to become
`pub(crate)` — a large, encapsulation-weakening diff before a single line of
logic moves.

Instead, convert the file to a directory module:

```
src/jbig2enc.rs        →  src/jbig2enc/mod.rs
                          src/jbig2enc/symbol/mod.rs
                          src/jbig2enc/symbol/<files>.rs
                          src/jbig2enc/<non-symbol files>.rs   (if any remain)
```

Child modules can see their ancestors' private items, so `impl<'a>
Jbig2Encoder<'a> { ... }` blocks moved into `jbig2enc::symbol::*` keep
touching `self.symbols`, `self.hash_map`, etc. with **zero visibility
churn**. Multiple `impl<'a> Jbig2Encoder<'a>` blocks across files is normal
Rust — just repeat the header/lifetime in each file.

Public paths must not change: external code references
`crate::jbig2enc::{Jbig2Encoder, PageData, PdfSplitOutput, encode_document,
encode_generic_region, encode_symbol_dict, encode_text_region_mapped,
encode_page_with_symbol_dictionary, first_black_pixel, EncoderMetrics,
SymbolInstance, ...}` (confirmed via grep across `jbig2.rs`, `jbig2arith.rs`,
`jbig2context.rs`, `bin/tester.rs`, and the `tests/` integration suite).
`jbig2enc/mod.rs` must re-export everything that is `pub` today via `pub use
symbol::...::{...}` so none of those paths break.

## The monster-function exception

Two functions dwarf the 1000-line target on their own:

| Function | Lines | Notes |
|---|---|---|
| `plan_document` | ~1565 (2666–4229) | single function body |
| `add_page_bitimage` | ~550 (2060–2607) | single function body |
| `build_planned_page` | ~260 (4229–4487) | |

You cannot relocate `plan_document` into a ~1000-line file and also fit
anything else there — the function alone busts the budget. **Decompose it
in place first** (extract cohesive chunks — e.g. the "PDF split: everything
goes in the global dictionary" branch, the per-page layout loop, the
out-of-range-mapping debug pass at line ~3007 — into private helper methods
on `Jbig2Encoder`), run the full test suite to confirm no behavior change,
*then* move the now-smaller pieces. Treat `add_page_bitimage` the same way
(the symbol-extraction loop, the recent-cache/anchor matching block, and the
debug-log tail at line ~2554 are natural seams).

State this explicitly in the PR description: the file holding
`plan_document` (post-decomposition) may still land at ~1100–1300 lines, and
that is the deliberate exception to the "~1000 lines" target, not an
oversight.

## Proposed file map

All paths below are under `src/jbig2enc/`.

### `mod.rs` (~400–600 lines) — stays at the root, owns the public surface
- `Jbig2Encoder` struct definition + lifecycle methods: `new`, `dict_only`,
  `get_page_count`, `metrics_snapshot`, `decision_debug_log`,
  `get_symbol_stats`
- `EncoderState`, `EncoderMetrics`, `PageData`, `SymbolModeStats`,
  `SymbolModeStageMetrics`
- `mod symbol;` declaration + `pub use symbol::...` re-exports
- `get_version`, `hash_key`, `first_black_pixel`, `HashKey`
- `encode_document`, `encode_generic_region` (these are not symbol-specific —
  generic-region/document-level entry points belong at this level, not in
  `symbol/`)

### `symbol/mod.rs` (~50 lines)
- Submodule declarations and any cross-cutting `pub(super) use` plumbing.
  No logic.

### `symbol/types.rs` (~450 lines) — lines 61–518, 549–716 today
Pure data: `SymUnifyAnchorDecision`, `SymUnifyAnchorCandidate`,
`ResidualSymbolTrace`, `ResidualReasonCode`, `ResidualShapeKind`,
`ResidualReasonStats`, `CounterfactualProbeStats`,
`DetailedCompareProbeStats`, `RecentSymbolCache`, `SymbolCandidate`,
`SymbolInstance`, `PlannedPage`, `PlannedDocument`, `PlannedPageLayout`,
`BuiltPage`, `SymbolDictLayout` (+ `segment_count`), `SymbolDictDiagnostics`,
`RefinementPlan`, `EncodedSymbolDictionary`, plus the small free functions
`encoder_diagnostics_enabled`, `indexed_symbol_dictionary_bytes`,
`anchor_map_dictionary_bytes`, `bitmap_proxy_bytes`,
`classify_residual_shape`, the `record_*_probe` trace helpers.
No `impl Jbig2Encoder` here — just structs/enums and their inherent `impl`s.

### `symbol/extraction.rs` (~750 lines) — lines 519–548, 2055–2625
- `segment_symbols` (free fn)
- `impl Jbig2Encoder`: `add_page`, `add_page_bitimage` (post-decomposition),
  `collect_symbols`, `flush`, `flush_pdf_split`
- This is the "turn a page bitmap into symbol candidates" surface.

### `symbol/unify_match.rs` (~950 lines) — lines 844–1530
`impl Jbig2Encoder` matching/scoring methods: `compute_symbol_signature`,
`signatures_are_compatible`, `should_skip_symbol_candidate`,
`should_accept_match`, `symbol_unify_assignment_score`,
`sym_unify_context_rerank_cost`, `sym_unify_anchor_candidate_is_better`,
`maybe_update_best_sym_unify_anchor_candidate`, `sym_unify_anchor_ready`,
`build_sym_unify_anchor_map`, `maybe_add_sym_unify_anchor`,
`residual_symbol_matches_anchor`, `residual_symbol_anchor_decision`,
`residual_symbol_accept_with_dim_limit`,
`residual_symbol_accept_without_overlap_prescreen`,
`residual_symbol_anchor_decision_without_overlap_prescreen`,
`evaluate_symbol_match`, `evaluate_symbol_unify_anchor_match`.
This is the single largest cohesive cluster — the anchor/residual decision
logic that justifies the original "things have gotten unwieldy" comment.

### `symbol/unify_apply.rs` (~530 lines) — lines 1531–2054
`impl Jbig2Encoder`: `estimate_local_symbol_gain`,
`estimate_global_symbol_gain`, `should_keep_text_local_symbol`,
`choose_cluster_prototype`, `note_symbol_page`, `push_symbol`,
`rebuild_symbol_metadata`, `rebuild_hash_map`, `build_symbol_unify_classes`,
`compact_symbol_table_after_remap`, `alias_local_symbols_to_globals`,
`apply_symbol_unify`. This is "commit the unification decisions to the
symbol table" — naturally downstream of `unify_match.rs`.

### `symbol/planning.rs` (~1300–1500 lines even after decomposition) — lines 2666–4487
`impl Jbig2Encoder`: `plan_document` (decomposed per above),
`build_planned_page`, `validate_plan`. **This is the named exception to the
1000-line target** — call it out in the PR.  If it's still too large after
decomposition, a further split into `planning.rs` (the per-document/global
dictionary planning) and `planning_page.rs` (`build_planned_page` +
per-page layout helpers extracted from `plan_document`) is the fallback.

### `symbol/serialize.rs` (~250 lines) — lines 4553–4686, plus `flush_dict`/`next_segment_number`
`impl Jbig2Encoder`: `serialize_full_document`, `serialize_pdf_split`,
`prune_symbols_if_needed`, `next_segment_number`, `flush_dict`,
`build_instance_residual_bitmap`, `encode_generic_region_payload_at`.

### `symbol/clustering.rs` (~250 lines) — lines 4687–4928
`impl Jbig2Encoder`: `cluster_symbols`, `validate_symbol_instance_indices`,
`validate_symbol_partition`, `auto_threshold`, `auto_threshold_using_hash`,
`unite_templates`.

### `symbol/dictionary.rs` (~750 lines) — lines 5219–5915, 6825–6884
Free functions (with their private callees moved alongside them, not left
behind): `encode_symbol_dict`, `canonicalize_dict_symbols`,
`plan_symbol_dictionary_layout`, `build_refinement_family_layout`,
`family_refinement_gain`, `family_should_refine`, `choose_family_prototype`,
`encode_symbol_dictionary_segments`, `encode_symbol_dict_subset_with_order`,
`encode_symbol_dict_with_order`, `build_dictionary_and_get_instances`,
`TextRegionSymbolInstance` (+ impl).

### `symbol/text_region.rs` (~900 lines) — lines 5830–6770, 6885–7050
Free functions + their private callees: `compute_region_bounds`,
`symbol_id_from_dense_maps`, `encode_text_region_mapped`, `uf_find`,
`uf_union`, `compute_symbol_hash`, `log2up`,
`encode_page_with_symbol_dictionary`.

### `symbol/text_region_refine.rs` (~480 lines) — lines 6262–6770
`encode_text_region_with_refinement`, `encode_text_region` (the
refinement-aware variants share enough machinery to warrant their own file
separate from the base `encode_text_region_mapped`).

### `symbol/tests.rs` (~50 lines) — lines 7052–7094
`mod refine_tests` (move with the code it exercises so it keeps visibility
into private helpers; rename module file to avoid confusion with the crate's
top-level `tests/` integration directory).

## Move order (lowest risk first)

1. **Mechanical scaffold**: `git mv jbig2enc.rs jbig2enc/mod.rs`, create
   empty `jbig2enc/symbol/mod.rs` with `pub(crate) mod ...;` stubs. Build —
   should still compile (everything still lives in `mod.rs`).
2. **Free-function clusters with no `Jbig2Encoder` access**:
   `dictionary.rs` and `text_region*.rs`. These only need their private
   callees (`compute_region_bounds`, `uf_find`/`uf_union`, `log2up`, etc.)
   moved alongside them — grep each function's body for crate-private calls
   before cutting, so nothing is left orphaned in `mod.rs`. Lowest privacy
   risk because they don't touch `Jbig2Encoder` fields.
3. **Pure data**: `types.rs`. Nothing references `self`, so this is a pure
   cut-paste plus `use super::*` in dependents.
4. **Cohesive `impl` clusters**, in dependency order so each compiles before
   the next is cut: `unify_match.rs` → `unify_apply.rs` → `clustering.rs` →
   `serialize.rs` → `extraction.rs`.
5. **The monsters last**: decompose `add_page_bitimage` and `plan_document`
   in place (still in `mod.rs` or already-moved `extraction.rs`/`planning.rs`
   — whichever is more convenient), run tests, *then* relocate to
   `planning.rs`.
6. **Re-exports + path cleanup**: add `pub use symbol::{...}` to `mod.rs`,
   confirm `cargo build -p jbig2enc-rust` (and the workspace) still resolves
   every external `jbig2enc::Foo` path with no edits required at call sites.

## Validation gates (run after *every* step above, not just at the end)

```sh
cargo build -p jbig2enc-rust
cargo test -p jbig2enc-rust
cargo test -p Legencode          # downstream crate that depends on jbig2enc-rust
```

A step that fails to build or breaks a test gets fixed or reverted before
the next step starts — don't accumulate multiple half-migrated states.

## What NOT to do

- Don't promote private fields/types to `pub(crate)` to make a sibling
  `src/symbol/` work — the nested layout avoids that entirely.
- Don't split a function's body across files (e.g. half of `plan_document`
  in `planning.rs`, half in `planning_page.rs` via a `pub(crate)` helper that
  only exists to cross the boundary) — extract named, independently
  meaningful helper methods instead, each fully owned by one file.
- Don't change any public path under `crate::jbig2enc::*` — re-export from
  `mod.rs` so `jbig2.rs`, `lib.rs`, `bin/tester.rs`, and the `tests/`
  integration suite need no changes.
