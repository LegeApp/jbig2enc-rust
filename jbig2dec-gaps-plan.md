# Supplementary Plan: Gaps Around `jbig2decplan.md`

`jbig2decplan.md` is the architectural authority for the decoder itself
(module design, MQ decoder, limits, phases 0–5, testing strategy). This
document fills in what that plan does **not** cover: the crate-level
restructure into encode/decode halves, repairing the pre-existing broken
baseline, the concrete interoperability harness, performance verification,
and the acceptance checklist used to verify each implementation phase.

Read `jbig2decplan.md` first. Where the two documents disagree on decoder
internals, `jbig2decplan.md` wins. Where they disagree on crate layout,
this document wins (it supersedes the layout sketch in its §4).

---

## Gap A: Crate restructure into `encode` / `decode` / `shared`

The crate becomes two equal halves plus a shared protocol core:

```text
src/
├── lib.rs              # thin: feature gates, re-exports, top-level docs
├── shared/             # protocol definitions used by BOTH halves
│   ├── mod.rs
│   ├── segment.rs      # SegmentType, header field enums (Phase 0 extracts these)
│   ├── mq_table.rs     # 94-entry MQ probability state table (Phase 0)
│   ├── int_proc.rs     # IntProc identifiers, integer ranges (Phase 0)
│   ├── bitmap.rs       # MonoBitmap packed u32 bitmap + CombinationOperator (Phase 0)
│   ├── reader.rs       # checked slice Reader (Phase 0, decode-leaning but shared-testable)
│   └── limits.rs       # DecodeLimits (Phase 0)
├── encode/             # the existing encoder, renamed modules
│   ├── mod.rs          # was src/jbig2enc/mod.rs
│   ├── arith.rs        # was jbig2arith.rs (encoder registers stay here;
│   │                   #   MQ table moves to shared/mq_table.rs)
│   ├── cc.rs           # was jbig2cc.rs
│   ├── classify.rs     # was jbig2classify.rs
│   ├── comparator.rs   # was jbig2comparator.rs
│   ├── context.rs      # was jbig2context.rs
│   ├── cost.rs         # was jbig2cost.rs
│   ├── halftone.rs     # was jbig2halftone.rs
│   ├── shared.rs       # was jbig2shared.rs (encoder-internal "shared"; rename
│   │                   #   later if contents move to src/shared/)
│   ├── simd.rs         # was jbig2simd.rs
│   ├── structs.rs      # was jbig2structs.rs (segment writers; SegmentType enum
│   │                   #   re-exported from shared/segment.rs after Phase 0)
│   ├── sym.rs          # was jbig2sym.rs
│   ├── unify.rs        # was jbig2unify.rs
│   ├── api.rs          # was jbig2.rs (top-level encode API)
│   └── symbol/         # was src/jbig2enc/symbol/ (unchanged internally)
├── decode/             # new, per jbig2decplan.md §4 (its decoder/ + arithmetic
│   ├── mod.rs          #   decoder side lives here)
│   ├── arith.rs        # ArithmeticDecoder + MqContext ops
│   ├── integer.rs      # IntegerContexts decode side
│   ├── iaid.rs
│   ├── segment.rs      # full T.88 segment-header parser
│   ├── file.rs         # file organization, embedded streams, ParsedDocument
│   ├── context.rs      # DecoderContext
│   ├── page.rs
│   ├── store.rs
│   ├── generic.rs
│   ├── refinement.rs
│   ├── symbol_dictionary.rs
│   ├── text_region.rs
│   ├── pattern_dictionary.rs
│   ├── halftone_region.rs
│   └── mmr.rs
└── bin/                # unchanged
```

Rules:

1. The restructure commit is **moves and path fixes only** — `git mv`,
   module renames in `lib.rs`, `use` path updates. No behavior change.
   `cargo build` must succeed and the encoder integration tests that
   passed before must still pass. This keeps `git log --follow` working.
2. Backward-compatible re-exports stay in `lib.rs` for one release
   (`pub use encode::api as jbig2;` style aliases) because Lege depends
   on this crate; remove them when Lege migrates.
3. Phase 0 (from `jbig2decplan.md` §23) happens **after** the move:
   extracting the MQ table, `SegmentType`, `IntProc` into `src/shared/`
   is a semantic refactor and must be its own commit(s) with the
   byte-identical-output check.

### Feature gating

```toml
[features]
default = ["encode", "decode", "symboldict"]
encode = []
decode = []
```

The PDF renderer will depend on `default-features = false, features =
["decode"]`. Every module in `src/encode/` is gated on `encode`, every
module in `src/decode/` on `decode`, `src/shared/` is unconditional.
Gating may be introduced with the Phase 1 commit (when `decode` first
has content) rather than during the pure-move commit.

### Naming

The crate keeps the name `jbig2enc-rust` for now to avoid breaking the
Lege dependency; renaming (e.g. to `jbig2-rs`) is a user decision to
revisit before any crates.io publish. Bump version to 0.6.0 when the
decoder lands.

---

## Gap B: Baseline repair (pre-existing breakage)

Discovered while extracting the crate; all predate the extraction:

1. `tests/encoder.rs`, `tests/encoder_pbm.rs`, `tests/mode_comparison.rs`
   fail to compile: they import a crate alias `jbig2` that only exists
   inside `tests/common/mod.rs`, and `mode_comparison.rs` references
   removed fields `global_segments` / `page_streams` on
   `Jbig2EncodeResult` (now `global_data` / `page_data`).
2. `src/jbig2cc.rs` lib test `test_tiny_cc_removal` fails
   (asserts 1 CC, gets 2).
3. `tests/headsup_c_vs_rust.rs` uses undeclared feature
   `c-encoder-bench` (add it to `[features]` or gate differently).
4. `tests/roundtrip_after_fix.rs::standalone_roundtrip_default_config`
   fails: jbig2dec 0.20 rejects the encoder's standalone-organization
   output with "page has no image, cannot be completed" (verified
   failing before the extraction too). Since the native decoder's
   Phase 1 exit criterion includes standalone streams, root-cause this
   during Phase 0/1 — it is either an encoder bug in standalone
   file-organization output or a test-harness invocation problem.

These must be fixed **first** (part of Phase 0's entry criteria): the
decoder's differential tests build on the encoder test harness, so the
harness has to compile and pass before decoder work starts. Fixing the
`test_tiny_cc_removal` failure means understanding whether the encoder
regressed or the test's expectation is stale — check
`git log -p -- '*jbig2cc*'` before "fixing" the assertion.

---

## Gap C: Interoperability matrix and harness

`jbig2decplan.md` §21 lists what to test; this section pins down how.

### Required directions

| # | Producer | Consumer | Status |
|---|----------|----------|--------|
| 1 | jbig2enc-rust (all modes) | system `jbig2dec` (0.20 installed) | partially exists in tests/ |
| 2 | jbig2enc-rust (all modes) | **native decoder** (self-roundtrip) | the core deliverable |
| 3 | C `jbig2enc` corpus | native decoder | needs corpus (see below) |
| 4 | native decoder output vs `jbig2dec` output on same stream | pixel-exact compare | differential oracle |

Direction 3: the C `jbig2enc` binary is **not installed**. Options, in
order of preference: (a) build it once from source
(github.com/agl/jbig2enc, needs leptonica) into `tests/tools/` and check
in only the generated `.jb2`/globals fixtures, not the binary; (b) check
in a small fixture corpus generated elsewhere. Either way the corpus is
committed so CI never needs the C encoder. Keep fixtures small
(< 200 KB total): 2–3 scanned-text pages through symbol mode,
1 generic, 1 refinement, 1 halftone.

### Harness pieces to build (Phase 1, alongside first decoder code)

1. `tests/common/pbm.rs` — PBM read/write + first-differing-pixel
   report `(x, y, expected, got)` (per §21.5).
2. `tests/common/oracle.rs` — run `jbig2dec` via `std::process::Command`
   on a temp file, parse its PBM output. Skip (not fail) when the
   binary is absent, but CI must have it.
3. `tests/decode_roundtrip.rs` — the encoder-mode matrix of §21.3:
   every mode × {standalone, embedded, embedded+globals, multipage},
   asserting `native_decode(encode(img)) == img` pixel-exact AND
   `native_decode(stream) == jbig2dec(stream)`.
4. Grow the matrix with each phase (generic-only in Phase 1, symbols in
   Phase 2, …) so every phase's exit criterion is executable as
   `cargo test --test decode_roundtrip`.

### Fuzzing (Phase 2+, per §21.6)

`cargo fuzz` targets under `fuzz/` seeded from the roundtrip corpus.
Not a phase gate until Phase 5, but the `embedded_document` target
should exist from Phase 2 and run for 10 CPU-minutes before each phase
is accepted.

---

## Gap D: Performance verification (PDF-renderer requirement)

The decoder plan gives architecture rules but no measurable targets.

### Benchmarks

Add `benches/decode.rs` (criterion, dev-dependency, added in Phase 1):

1. `decode_generic_a4_300dpi` — one full-page generic region
   (2480×3508). This is the dominant real-world case.
2. `decode_symbol_page` — a text-heavy page through symbol mode.
3. `decode_page_reuse` — same page 10× through one `DecoderContext`,
   asserting no allocation growth after the first iteration
   (measure with a counting allocator in a test, not criterion).
4. `decode_multipage_parallel` — 8 pages sharing globals across 4
   threads (`parallel` feature).

### Targets (gate for "ready for the PDF renderer", not per-phase)

* Generic A4 300 dpi page: **≤ 25 ms** on this machine in release
  (jbig2dec 0.20 CLI does the comparable file in roughly that range —
  measure it in Phase 1 to set the real baseline, then require native
  ≤ 1.25× jbig2dec, stretch ≤ 1.0×).
* Symbol-mode page: ≤ 1.25× jbig2dec on the same stream.
* Zero heap allocations in steady-state page decode with a reused
  `DecoderContext` (after the first page of a given size class).
* Output is packed `MonoBitmap` handed to the renderer without a
  repack (the `decode_embedded_into` API from `jbig2decplan.md` §5).

### Method

* Measure only after correctness (never optimize before a phase's
  roundtrip tests pass).
* `cargo bench` before/after each phase ≥ 2; a phase may not regress
  earlier benchmarks by > 5 %.
* The existing `profiling` feature (pprof flamegraphs) is available for
  investigation; hot-loop rules of `jbig2decplan.md` §22 apply.

---

## Gap E: Phase acceptance checklist (run after every phase)

The checking agent/reviewer runs, in order:

```bash
cargo build --all-targets                 # everything compiles
cargo test                                # unit + integration
cargo test --no-default-features --features decode   # decode half stands alone (Phase 1+)
cargo test --no-default-features --features encode   # encode half stands alone
cargo clippy --all-targets -- -D warnings            # decoder code is warning-clean
cargo bench                               # Phase 2+: compare against saved baseline
```

Plus per-phase manual checks:

* **Phase 0**: encoder output byte-identical on the full test corpus —
  hash every stream the tests emit before and after the refactor and
  diff the hashes. Baseline breakage from Gap B fixed.
* **Phase 1**: generic + lossless roundtrip exact; jbig2dec agreement;
  fuzz target `parse_segment_header` exists and survives 10 min.
* **Phase 2**: symbol + sym_unify + globals + multipage exact;
  `DecodedGlobals` is `Send + Sync` (compile-time assertion test);
  deterministic across thread counts.
* **Phase 3**: refinement PDF tests pass natively.
* **Phase 4**: all halftone modes exact vs jbig2dec.
* **Every phase**: no `unwrap`/`expect`/`assert!` reachable from input
  in `src/decode/` — enforce with
  `#![deny(clippy::unwrap_used, clippy::expect_used)]` at the top of
  `decode/mod.rs` rather than by grep.

Each phase ends in its own commit(s) on `main`; the checker reviews the
diff, runs the checklist, and only then does the next phase start.

---

## Phase 3 findings (refinement completion + hardening)

* **Encoder S-advance bug — CONFIRMED and FIXED.** T.88 §6.4.5 step 3c v)/xi):
  `CURS` advances by `WI - 1` where `WI` is the width of the bitmap *actually
  placed* (`IBI`), i.e. the refined width `GRW = WOI + RDWI` for `RI=1`
  instances (§6.4.11, Table 12). The decoder (and jbig2dec 0.20) advance by the
  refined placed width. The encoder's `encode_text_region_with_refinement`
  advanced by the *prototype* width, so whenever `RDW != 0` the encoded `IADS`
  deltas desynced S for every later instance in the strip. Fixed by advancing
  `CURS` by the refined trimmed width. Evidence (test_image1, native == jbig2dec
  throughout): native-decode-vs-source pixel diffs dropped symbol 2348→175,
  sym_unify 2984→1215, refine 19897→175; a synthetic RDW=+1 page went 3912→0
  (pixel-exact vs source). The residual (e.g. 175) is lossy `RI=0` substitution,
  a separate matching-policy question, not S-advance. The encoder byte change is
  isolated in a dedicated `encode_hashes.txt` regeneration commit (only
  test_image1 page streams changed; globals unchanged — the fix is text-region
  S-coordinate only). Note: `RDW != 0` only arises when `config.text_refine` is
  set (dim tolerance 2); default `text()` matching requires exact dims
  (`evaluate_symbol_match` `dim_limit = 0`), so `RDW` is always 0 there.

* **Standalone immediate generic refinement regions (types 42/43) — decoded.**
  T.88 §7.4.7: an immediate refinement region with no referred region segment
  refines the page buffer in place; the reference is the page-buffer window under
  the region box (§7.4.7.4), `GRREFERENCEDX/DY = 0`, external combop REPLACE.
  Contexts reset per region segment (§7.4.7.5 step 2). Intermediate (type 40) and
  any refinement region that *refers to another region segment* (retained
  auxiliary buffer reuse) stay typed-`Unsupported` (Phase 5). GRTEMPLATE-1 and
  TPGRON also stay typed-`Unsupported` (the encoder emits neither).

* **jbig2dec 0.20 quirk (documented, not a native bug).** For a standalone
  refinement region whose reference is the page buffer, jbig2dec extracts the
  reference window from the page origin regardless of the region's `(x, y)`, so
  it desyncs for non-zero region offsets. The native decoder follows §7.4.7.4
  (window offset by `(x, y)`) and is validated vs the intended target;
  `standalone_refinement_region_vs_jbig2dec` compares vs jbig2dec only at offset
  `(0, 0)` where both agree.

* **Robustness fix.** `decode_text_region` computed
  `GRREFERENCEDX = floor(RDW/2) + RDX` with an unchecked `i32` add; `RDX`/`RDY`
  are unbounded decoded integers, so a malformed stream overflowed. Now
  saturating (an out-of-range offset just reads the reference window as zero).
  Found by extending the chaos gate to a refine-mode stream.

## Gap F: Out of scope for now (explicitly deferred)

* Product B / Phase 5 (Huffman, striped pages, random access, templates
  1–3, TPGDON) — planned in `jbig2decplan.md`, not scheduled here.
* Hayro differential oracle — `jbig2decplan.md` recommends it; jbig2dec
  is sufficient as the single oracle for Phases 1–4. Add Hayro as a
  dev-dependency comparison only if jbig2dec disagreements need a
  tiebreaker.
* Crate rename and crates.io publish.
* Migrating the encoder's `BitImage` internals onto `MonoBitmap`
  (post-decoder cleanup, `jbig2decplan.md` §7).
