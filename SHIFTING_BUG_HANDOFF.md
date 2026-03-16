# JBIG2 Symbol Mode: "Every ~5th Glyph Shifted Higher" Bug — Handoff Document

## Update: Root Cause Found

- **Root cause**: The code and this handoff both assumed `REFCORNER=0` meant `TOPLEFT`. Per T.88 §7.4.3.1.1, `REFCORNER=0` is `BOTTOMLEFT` and `REFCORNER=1` is `TOPLEFT`.
- **Impact**: The encoder computed text-region `T` coordinates as top-edge positions while the header told decoders to anchor symbols by their bottom-left corner. Taller glyphs therefore rendered too high.
- **Fix applied**: `Jbig2Config::default().text_ref_corner` now defaults to `1` (`TOPLEFT`), and the text-region comments were updated to match the spec.
- **Verification**:
  - Before fix: page `confed/page_0061.pbm` vs standalone symbol decode differed by `211709` pixels.
  - After fix: the same comparison dropped to `8018` pixels.
  - With `JBIG2_NO_REUSE=1`, the comparison dropped from `211672` pixels to `7506` pixels.
- **Status**: The high-positioning bug is effectively resolved. A smaller residual symbol-mode mismatch remains and should be investigated separately if full pixel identity is still required.

## Bug Description

In JBIG2 symbol mode output (both `sym_no_at` and `sym_at`), approximately every 4th–5th glyph is rendered slightly higher than its correct position. The generic encoding output is pixel-perfect. The artifact is consistent across all tested PDF viewers: SumatraPDF, Adobe Acrobat, Firefox, and Edge.

## Workspace Location

- **Crate**: `c:\Users\dk\Desktop\Testers\Legencode\jbig2enc-rust\`
- **Key source files**:
  - `src/jbig2enc.rs` — main encoder: matching, dictionary, text region encoding
  - `src/jbig2comparator.rs` — bitmap distance comparator
  - `src/jbig2arith.rs` — arithmetic coder + integer encoding
  - `src/jbig2structs.rs` — segment header serialization
  - `src/jbig2sym.rs` — BitImage type, trim(), packed_words()
  - `tests/multi_page_benchmark.rs` — benchmark test (run with `cargo test --release --features symboldict --test multi_page_benchmark -- --nocapture`)

## Test Data

- PBM files in `confed/` directory (141 pages of text-only scanned documents; image-heavy pages were already removed)
- Default benchmark: 10 and 20 pages

## Environment Variables for Testing

| Variable | Values | Purpose |
|----------|--------|---------|
| `BENCH_WRITE` | `1` | Write PDFs to `test_output_pdfs/benchmark_<timestamp>/` |
| `BENCH_PAGES` | `10` or `10,20` | Which page counts to benchmark |
| `JBIG2_DEBUG` | `1` | Write `jbig2_debug_page0.log` with matching + encoding + decode simulation |
| `JBIG2_NO_REUSE` | `1` | Skip all symbol matching — every CC becomes a unique prototype (isolation test) |

## What Has Been Investigated and Ruled Out

### 1. Position mutation in lossy matching (RULED OUT)
- **Hypothesis**: The `dy` offset from the comparator was being baked into `instance.position.y`, causing vertical drift.
- **Fix applied**: Reverted to using original `rect` position. Additionally, lossy PM&S now requires `dy == 0` (only `dx.abs() <= 1` allowed).
- **Result**: Bug persists. All lossy matches in the debug log now show `dx=0 dy=0`, so this is no longer a factor.

### 2. Dimension mismatch in lossy substitution (RULED OUT)
- **Hypothesis**: Lossy substitution was accepting prototypes with different dimensions (±2px), causing "scaling" artifacts.
- **Fix applied**: `dim_range = 0` for lossy mode — only exact-dimension prototypes are accepted.
- **Result**: Bug persists.

### 3. Trim offset not accounted for (RULED OUT)
- **Hypothesis**: `symbol.trim()` returns a trim offset that wasn't being added to the CC bbox position.
- **Fix applied**: Position is now `bbox.xmin + trim_offset.x`, `bbox.ymin + trim_offset.y`.
- **Result**: All trim offsets in the debug log are `(0,0)` — the CC analysis already returns tight bounding boxes. Not a factor.

### 4. Text region encoding position deltas (PARTIALLY RULED OUT)
- **Verified**: A decode simulation replaying JBIG2 §6.4.5 was added to `encode_text_region_mapped()`. It checks every instance position against the expected absolute coordinates.
- **Result at the time**: **Zero mismatches** across all 2234 instances on page 0. The integer delta stream matched the encoder's own reconstruction.
- **What was missed**: The reconstruction assumed `REFCORNER=0` meant `TOPLEFT`, which is incorrect. The stream math was internally consistent, but the header semantics were wrong.

### 5. SBDSOFFSET type bug (FIXED but not the cause)
- **Issue**: `ds_offset` was `u8` but SBDSOFFSET is signed 5-bit per the JBIG2 spec.
- **Fix applied**: Changed to `i8` with proper sign-preserving 5-bit packing.
- **Impact**: Dormant (value is always 0), so not the current bug.

### 6. Integer encoding / arithmetic coder (ANALYZED)
- **INT_ENC_RANGE table**: Manually traced prefix codes for several value ranges against JBIG2 spec Table A.1. LSB-first emission of prefix bits matches the context-tree traversal order.
- **Result**: The arithmetic coder and integer encoding appear correct. Since all 4 PDF viewers render identically, the encoded bitstream is being decoded consistently — the issue is what's being encoded, not how.

## Isolation Test: JBIG2_NO_REUSE=1

A `JBIG2_NO_REUSE=1` env var was added that skips all symbol matching. Every connected component becomes a unique prototype in the dictionary, requiring no reuse. This isolates text region encoding + dictionary encoding from the matching logic.

- **Test ran successfully** with NO_REUSE: `sym_no_at 10p: 0.3% savings, 0.15s`
- **Output folder**: `test_output_pdfs/benchmark_1773634938/` (sym_no_at_10p.pdf = 200.4KB)
- **CRITICAL**: The user has NOT yet visually verified whether the NO_REUSE output still shows the shifting. This is the most important next diagnostic step.

### What the NO_REUSE result tells us:
- **If shifting is GONE with NO_REUSE**: Bug is in the matching logic or symbol reuse pathway. Focus on how symbol IDs map between the dictionary and text region instances.
- **If shifting PERSISTS with NO_REUSE**: Bug is in either (a) the symbol dictionary generic bitmap encoding, (b) the text region encoding/header, or (c) the `packed_words()` / bitmap serialization.

## Current Code State

### Matching Logic (`add_page_bitimage`, ~line 305)
- CC extraction → `symbol.trim()` → position adjusted by trim offset
- `JBIG2_NO_REUSE=1` skips all matching (every CC → new prototype)
- Lossy PM&S (Case 3): requires `dy == 0`, `dx.abs() <= 1`, exact same dimensions
- Debug logging (JBIG2_DEBUG=1): logs every CC with match type, position, dimensions, trim offset

### Text Region Encoding (`encode_text_region_mapped`, ~line 1737)
- Takes `page_num` parameter for debug gating
- Instances sorted by `(strip_base, x)`
- Cursor advancement: `current_s += item.symbol_width - 1`
- `strip_width = 1` (LOGSBSTRIPS=0, SBSTRIPS=1)
- Initial IADT(0) before encoding loop
- Decode simulation appended to debug log (page 0 only)

### Text Region Header (`TextRegionParams.to_bytes()`, ~line 528)
- SBHUFF=0, SBREFINE=config, LOGSBSTRIPS=0, REFCORNER=1 (TOPLEFT), TRANSPOSED=0
- SBCOMBOP=0 (OR), SBDEFPIXEL=0, SBDSOFFSET=0 (now i8), SBRTEMPLATE=0/1
- 17-byte region info + 2-byte flags (big-endian)

### Symbol Dictionary Encoding (`encode_symbol_dict_with_order`, line 1446)
- `canonicalize_dict_symbols`: sorts by (height, width)
- Delta-height / delta-width encoding per height class
- Generic region encoding for each symbol bitmap
- Export flags: run-length form (0, N)
- SD_TEMPLATE=0, AT pixels: [(3,-1), (-3,-1), (2,-2), (-2,-2)]

### Config Defaults (`Jbig2Config::text()`, ~line 130)
- `text_refine: false` (lossy PM&S mode)
- `text_log_strips: 0` (SBSTRIPS=1)
- `text_ref_corner: 1` (TOPLEFT)
- `text_ds_offset: 0` (now i8)

## Key Debug Log Location

When `JBIG2_DEBUG=1`, a file `jbig2_debug_page0.log` is written to the project root. It contains:

1. **Matching section**: Every CC with type (NEW/EXACT/LOSSY), page position, dimensions, trim offset, prototype index
2. **Encoding section**: Every encoded instance with SymID, symbol width, strip_base, t_offset, relative X, IADT delta, IADS/IAFS delta
3. **Decode simulation**: Replays §6.4.5 decoding and compares each instance's decoded position to the expected position. Reports MISMATCH for any discrepancy (currently: zero mismatches).

## Performance Numbers (Latest)

| Pages | Mode | Size | Savings | Time |
|-------|------|------|---------|------|
| 10 | generic | 201.0 KB | — | 0.16s |
| 10 | sym_no_at | 171.8 KB | 15.1% | 1.4s |
| 10 | sym_at | 170.1 KB | 15.9% | 33-36s |
| 20 | sym_no_at | 326.7 KB | 18.7% | 4.4s |

## Recommended Next Steps (Priority Order)

### Step 1: Visual check of NO_REUSE output
Open `test_output_pdfs/benchmark_1773634938/sym_no_at_10p.pdf` (the NO_REUSE version, 200.4KB) and compare against `generic_10p.pdf`. This is the single most informative diagnostic.

### Step 2: If shifting persists in NO_REUSE → focus on dictionary/bitmap encoding
- The text region positions are verified correct by decode simulation.
- Check `packed_words()` in `jbig2sym.rs` — verify word packing produces correct bit-level output.
- Check `encode_generic_region()` in the arithmetic coder — this encodes each symbol bitmap. A context model error here would corrupt all symbols systematically.
- Try extracting individual symbol bitmaps from the dictionary and rendering them standalone to verify they're correct.

### Step 3: If shifting is GONE with NO_REUSE → focus on dict mapping
- The symbol ID mapping between `global_symbols` and the canonical dictionary order may be wrong.
- Check `global_sym_to_dict_pos` and `local_sym_to_dict_pos` construction.
- Verify that `encode_iaid(symbol_id, symbol_id_bits)` produces correct IDs.
- A wrong symbol_id means the decoder fetches a different-sized prototype from the dictionary, which would shift its rendering position since REFCORNER=TOPLEFT places the symbol at (S, T) and the symbol extends downward/rightward by its (width, height).

### Step 4: After fixing the shifting
- Remove debug instrumentation (or gate it properly behind a feature flag)
- Focus on performance: sym_at is extremely slow (33s for 10p)
- Consider adding CC size filtering to avoid text-region encoding large non-text blobs

## Summary of All Changes Made in This Session

1. **Lossy matching: dy rejection** — Lossy PM&S now requires `dy == 0`, `dx.abs() <= 1`
2. **Lossy matching: dim_range=0** — Only exact-dimension prototypes for lossy substitution
3. **Position: reverted to original rect** — No more `rect.y - dy` adjustment
4. **Position: trim offset adjustment** — Position now accounts for `symbol.trim()` offset (all zeros currently)
5. **SBDSOFFSET: u8 → i8** — Proper signed 5-bit type
6. **encode_text_region_mapped: page_num param** — For debug log gating
7. **Debug logging** — Matching decisions, encoding deltas, decode simulation (gated by JBIG2_DEBUG=1)
8. **JBIG2_NO_REUSE=1** — Isolation test that skips all symbol matching
9. **Benchmark: 10/20 page default** — Changed from 10/50
