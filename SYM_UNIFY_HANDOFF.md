# `sym_unify` Handoff Summary

This note summarizes the recent `sym_unify` work so development can resume without relying on prior chat context.

## Current State

- `sym_unify` replaced legacy `sym_collapse` as the active quality-first lossy symbol path.
- Legacy collapse code has been removed from the active codebase:
  - `src/jbig2collapse.rs`
  - `src/jbig2collapse_context.rs`
  - `src/bin/profile_sym_collapse.rs`
- Shared collapse-era logic was split into:
  - `src/jbig2classify.rs`
  - `src/jbig2context.rs`
- `sym_refine` is no longer an active mode.
- A lessons-learned document was added:
  - `REFINEMENT_LESSONS.md`

## Core Milestones Reached

### 1. `sym_unify` quality baseline

`sym_unify` was rebuilt as:
- non-destructive
- post-harvest
- representative-based
- quality-first

Key ideas:
- keep CC extraction as a firehose
- do not denoise away small marks
- split fragile marks out of unification
- build conservative classes
- choose dense representatives
- unify only safe members

This solved the earlier substitution problems that made `sym_collapse` unusable.

### 2. CC firehose change

Connected-component analysis was changed so the active path no longer drops or grid-splits tiny CCs during extraction.

Impact:
- periods and `i/j` dots reappeared
- this became the stable foundation for later `sym_unify` tuning

### 3. Adaptive live-anchor dictionary

The biggest compression improvement came from adding adaptive live representative reuse during ingestion in `src/jbig2enc.rs`.

Behavior:
- stable non-fragile symbols become live anchors
- new glyphs try live anchors before the broader symbol lookup
- final `apply_symbol_unify()` still runs afterward

This fixed the bad scaling where symbol count kept exploding as page count increased.

## Recent Architecture Work

### Narrow refinement architecture

A paper-style narrow refinement extension was added on top of `sym_unify`:
- accepted classes can now produce `refinement_subclusters`
- recurring stable subcluster members can be marked `needs_refinement=true`
- text-region refinement is allowed under `sym_unify` if instances actually require it

Important:
- this does **not** revive the old broad family-refine behavior
- refinement is driven only by `sym_unify`-identified subclusters

Files touched:
- `src/jbig2unify.rs`
- `src/jbig2enc.rs`
- `src/jbig2structs.rs`

Result:
- architecture is present
- economics were poor on the current `sahib` corpus
- conservative defaults remain in place, effectively keeping refinement dormant unless it truly pays

## Benchmark Story

### Strong recent clean baseline

Best important recent clean baseline:
- `benchmark_1773768370`

Key result on current 20-page text-only-ish `sahib` set:
- `symbol`: `244.2 KB`
- `sym_unify`: `193.7 KB`
- `sym_unify` is about `20.7%` smaller

### Why this matters

At this stage:
- no obvious errant substitutions in reviewed outputs
- no missing punctuation/dot regressions from the earlier collapse path
- local dictionaries are already tiny compared with vanilla `symbol`

### Dictionary/generic breakdown

For the current `sym_unify` baseline on 20 pages:
- globals: about `15.8 KB`
- local dicts: about `2.2 KB`
- text region: about `115.8 KB`
- generic region: about `59.1 KB`

Interpretation:
- dictionary size is no longer the dominant cost
- remaining size pressure is largely the residual generic tail

## What Was Tried Recently

### 1. Narrow refinement pass

Tried again in a controlled way:
- only stable recurring subclusters
- only under `sym_unify`
- strict score/gain gates

Observed outcomes:
- very conservative settings found almost nothing useful
- looser settings admitted some subclusters but slightly worsened size
- “heaviest recurring only” settings ended up admitting nothing on this corpus

Conclusion:
- the refinement architecture is good to keep
- refinement is not the next optimization lever on this corpus

### 2. Planner-side rescue of page-local one-offs

Goal:
- rescue text-like one-offs back into local symbol dictionaries instead of residual generic

Observed outcomes:
- generic bytes dropped sharply
- local dictionaries exploded
- total file size worsened

Example runs:
- `benchmark_1773768762`
- `benchmark_1773768813`

Conclusion:
- the direction is informative
- but broad local rescue is not economically good enough
- that experimental rescue logic was reverted back to baseline behavior

## Diagnostics and Profiling

### Diagnostics

Heavy diagnostics are gated behind:
- `JBIG2_DIAGNOSTICS=1`

Without the flag:
- benchmark speed returns to the clean path

Useful current diagnostics include:
- class build summary
- weak-core vs low-gain rejects
- anchor remap counts
- residual anchor scans
- local/global dictionary byte splits

### Flamegraph

Profiling was successfully used after enabling `perf`.

Main findings:
- `compute_symbol_signature()` and hole counting were hot
- some symbol matching paths were hot

One speed optimization that stayed:
- skip expensive hole counting for tiny/sparse symbols that cannot plausibly contain useful counters

This improved time without changing file size meaningfully.

## Current Design Interpretation

The current code now has:
- class formation
- dense core representatives
- border members
- adaptive live anchors
- subcluster candidate/refinement architecture

It still does **not** have a full paper-style representative hierarchy or economic planner for refinement dictionaries. That remains future work.

## Best Current Understanding Of The Next File-Size Frontier

Because production symbol-mode input will be text-only:
- residual generic regions should be treated as missed text-symbol opportunities
- not as mixed-content/image safety buckets

That means the next likely fruitful areas are:

1. Better live-anchor timing and readiness
- more text-like leftovers should attach to proven anchors earlier

2. Better second-pass attachment of text-like residuals
- but without loosening visual safety rules enough to risk substitutions

3. Better residual diagnostics for text
- identify exactly why a text-like symbol missed:
  - overlap
  - signature
  - compare
  - outside-ink
  - score

4. Better text-region efficiency
- especially by reducing false negatives in symbol attachment

## Areas That Are Probably Not The Next Lever

- reviving standalone `sym_refine`
- broadening refinement aggressively
- broad planner-side rescue of local one-offs
- more work on mixed-content/image residual handling for this corpus

## Current Important Commits Before This Uncommitted State

- `786314d` checkpoint: solid sym_unify baseline
- `d32817f` checkpoint: adaptive sym_unify dictionary
- `68a282c` checkpoint: tune adaptive sym_unify anchors
- `d66ad67` tune: gate encoder diagnostics behind env flag

## Recommendation For Next Session

Start from the current clean baseline and focus on:
- residual text-symbol false negatives
- live-anchor matching completeness
- text-region/generic tradeoff for text-only inputs

Do **not** start by loosening substitution thresholds or reviving broad refinement.
