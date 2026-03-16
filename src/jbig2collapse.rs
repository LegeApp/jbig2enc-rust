use crate::jbig2comparator::{CollapseCompareLimits, Comparator, CompareResult};
use crate::jbig2structs::{
    Jbig2Config, LossyCollapsePrototypeMode, LossyCollapsePrototypeSelectorMode,
};
use crate::jbig2sym::BitImage;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, Clone, Copy, Default)]
pub struct SymbolSignature {
    pub black: u32,
    pub area: u32,
    pub left_col: u16,
    pub right_col: u16,
    pub top_row: u16,
    pub bottom_row: u16,
    pub cx_times_256: u16,
    pub cy_times_256: u16,
    pub left_mass: u16,
    pub right_mass: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FamilyBucketKey {
    pub w_bin: u16,
    pub h_bin: u16,
    pub density_bin: u8,
    pub aspect_bin: u8,
    pub centroid_y_bin: u8,
    pub lr_balance_bin: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct LossyFamilyMatch {
    pub member_index: usize,
    pub dx: i32,
    pub dy: i32,
    pub err: u32,
}

#[derive(Debug, Clone)]
pub struct LossyFamily {
    pub prototype_index: usize,
    pub members: Vec<LossyFamilyMatch>,
    pub total_usage: usize,
    pub page_span: usize,
    pub prototype_score: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum LossyFamilyProbe {
    Accept(CompareResult),
    Reject {
        reason: &'static str,
        result: Option<CompareResult>,
        limits: Option<CollapseCompareLimits>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct PrototypeBuildStats {
    pub mode: String,
    pub medoid_black_pixels: usize,
    pub output_black_pixels: usize,
    pub pixels_kept: usize,
    pub pixels_removed: usize,
    pub pixels_added: usize,
    pub avg_member_score_before: f64,
    pub avg_member_score_after: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CollapseBuildDiagnostics {
    pub lines: Vec<String>,
}

pub struct CollapseFamilyBuildInputs<'a> {
    pub config: &'a Jbig2Config,
    pub global_symbols: &'a [BitImage],
    pub symbol_usage: &'a [usize],
    pub symbol_page_count: &'a [usize],
    pub symbol_signatures: &'a [SymbolSignature],
    pub symbol_pixel_counts: &'a [usize],
    pub page_symbol_indices: &'a [Vec<usize>],
}

pub struct PrototypeBuildInputs<'a> {
    pub config: &'a Jbig2Config,
    pub family: &'a LossyFamily,
    pub global_symbols: &'a [BitImage],
    pub symbol_usage: &'a [usize],
    pub symbol_page_count: &'a [usize],
    pub symbol_signatures: &'a [SymbolSignature],
    pub symbol_pixel_counts: &'a [usize],
    pub collect_stats: bool,
}

pub fn compute_symbol_signature(img: &BitImage) -> SymbolSignature {
    let mut black = 0u32;
    let mut left_col = img.width;
    let mut right_col = 0usize;
    let mut top_row = img.height;
    let mut bottom_row = 0usize;
    let mut sum_x = 0u32;
    let mut sum_y = 0u32;
    let mut left_mass = 0u32;
    let mut right_mass = 0u32;
    let mid_x = img.width / 2;

    for y in 0..img.height {
        for x in 0..img.width {
            if img.get_usize(x, y) {
                black += 1;
                left_col = left_col.min(x);
                right_col = right_col.max(x);
                top_row = top_row.min(y);
                bottom_row = bottom_row.max(y);
                sum_x += x as u32;
                sum_y += y as u32;
                if x < mid_x {
                    left_mass += 1;
                } else {
                    right_mass += 1;
                }
            }
        }
    }

    let (cx, cy) = if black == 0 {
        (0, 0)
    } else {
        (
            ((sum_x * 256) / black).min(u16::MAX as u32) as u16,
            ((sum_y * 256) / black).min(u16::MAX as u32) as u16,
        )
    };

    SymbolSignature {
        black,
        area: img.width.saturating_mul(img.height).min(u32::MAX as usize) as u32,
        left_col: left_col.min(u16::MAX as usize) as u16,
        right_col: right_col.min(u16::MAX as usize) as u16,
        top_row: top_row.min(u16::MAX as usize) as u16,
        bottom_row: bottom_row.min(u16::MAX as usize) as u16,
        cx_times_256: cx,
        cy_times_256: cy,
        left_mass: left_mass.min(u16::MAX as u32) as u16,
        right_mass: right_mass.min(u16::MAX as u32) as u16,
    }
}

#[inline]
fn quantize_ratio_u8(num: u32, den: u32, bins: u32) -> u8 {
    if den == 0 || bins == 0 {
        return 0;
    }
    ((num.saturating_mul(bins)) / den).min(bins.saturating_sub(1)) as u8
}

pub fn family_bucket_key_for_symbol(symbol: &BitImage, sig: &SymbolSignature) -> FamilyBucketKey {
    let w = symbol.width as u32;
    let h = symbol.height as u32;
    let area = sig.area.max(1);

    FamilyBucketKey {
        w_bin: ((w + 1) / 2).min(u16::MAX as u32) as u16,
        h_bin: ((h + 1) / 2).min(u16::MAX as u32) as u16,
        density_bin: quantize_ratio_u8(sig.black, area, 16),
        aspect_bin: quantize_ratio_u8(w, h.max(1), 16),
        centroid_y_bin: ((sig.cy_times_256 as u32 * 16) / (h.max(1) * 256)).min(15) as u8,
        lr_balance_bin: quantize_ratio_u8(sig.left_mass as u32, sig.black.max(1), 8),
    }
}

pub fn family_bucket_neighbors(key: FamilyBucketKey) -> Vec<FamilyBucketKey> {
    let mut out = Vec::with_capacity(27);
    for dh in -1i32..=1 {
        for dw in -1i32..=1 {
            for dd in -1i32..=1 {
                let candidate = FamilyBucketKey {
                    w_bin: (key.w_bin as i32 + dw).max(0) as u16,
                    h_bin: (key.h_bin as i32 + dh).max(0) as u16,
                    density_bin: (key.density_bin as i32 + dd).clamp(0, 15) as u8,
                    aspect_bin: key.aspect_bin,
                    centroid_y_bin: key.centroid_y_bin,
                    lr_balance_bin: key.lr_balance_bin,
                };
                if !out.contains(&candidate) {
                    out.push(candidate);
                }
            }
        }
    }
    out
}

#[inline]
pub fn for_each_family_bucket_neighbor(
    key: FamilyBucketKey,
    mut f: impl FnMut(FamilyBucketKey),
) {
    for dh in -1i32..=1 {
        for dw in -1i32..=1 {
            for dd in -1i32..=1 {
                f(FamilyBucketKey {
                    w_bin: (key.w_bin as i32 + dw).max(0) as u16,
                    h_bin: (key.h_bin as i32 + dh).max(0) as u16,
                    density_bin: (key.density_bin as i32 + dd).clamp(0, 15) as u8,
                    aspect_bin: key.aspect_bin,
                    centroid_y_bin: key.centroid_y_bin,
                    lr_balance_bin: key.lr_balance_bin,
                });
            }
        }
    }
}

#[inline]
fn pair_cache_key(lhs: usize, rhs: usize) -> u64 {
    ((lhs as u64) << 32) | rhs as u64
}

#[inline]
fn reverse_compare_result(mut result: CompareResult) -> CompareResult {
    result.dx = -result.dx;
    result.dy = -result.dy;
    result
}

#[inline]
fn reverse_lossy_family_probe(probe: LossyFamilyProbe) -> LossyFamilyProbe {
    match probe {
        LossyFamilyProbe::Accept(result) => LossyFamilyProbe::Accept(reverse_compare_result(result)),
        LossyFamilyProbe::Reject {
            reason,
            result,
            limits,
        } => LossyFamilyProbe::Reject {
            reason,
            result: result.map(reverse_compare_result),
            limits,
        },
    }
}

fn lossy_family_probe_cached(
    probe_cache: &mut FxHashMap<u64, LossyFamilyProbe>,
    comparator: &mut Comparator,
    target: &BitImage,
    target_index: usize,
    reference: &BitImage,
    reference_index: usize,
    signatures: &[SymbolSignature],
    black_counts: &[usize],
    max_err: u32,
    max_dx: i32,
    max_dy: i32,
) -> LossyFamilyProbe {
    use std::collections::hash_map::Entry;

    let (lhs_index, lhs_image, rhs_index, rhs_image, reverse) = if target_index <= reference_index
    {
        (target_index, target, reference_index, reference, false)
    } else {
        (reference_index, reference, target_index, target, true)
    };
    let key = pair_cache_key(lhs_index, rhs_index);
    let probe = match probe_cache.entry(key) {
        Entry::Occupied(entry) => *entry.get(),
        Entry::Vacant(entry) => *entry.insert(lossy_family_probe(
            comparator,
            lhs_image,
            lhs_index,
            rhs_image,
            rhs_index,
            signatures,
            black_counts,
            max_err,
            max_dx,
            max_dy,
        )),
    };
    if reverse {
        reverse_lossy_family_probe(probe)
    } else {
        probe
    }
}

pub fn family_signatures_are_compatible(
    lhs: SymbolSignature,
    rhs: SymbolSignature,
    lhs_black: usize,
    rhs_black: usize,
) -> bool {
    let area_scale = lhs_black.max(rhs_black).max(4);
    let black_tol = (area_scale / 10).clamp(4, 10) as u32;
    let mass_tol = (area_scale / 8).clamp(4, 12) as u16;

    lhs.black.abs_diff(rhs.black) <= black_tol
        && lhs.left_col.abs_diff(rhs.left_col) <= 2
        && lhs.right_col.abs_diff(rhs.right_col) <= 2
        && lhs.top_row.abs_diff(rhs.top_row) <= 1
        && lhs.bottom_row.abs_diff(rhs.bottom_row) <= 1
        && lhs.cx_times_256.abs_diff(rhs.cx_times_256) <= 96
        && lhs.cy_times_256.abs_diff(rhs.cy_times_256) <= 96
        && lhs.left_mass.abs_diff(rhs.left_mass) <= mass_tol
        && lhs.right_mass.abs_diff(rhs.right_mass) <= mass_tol
}

#[inline]
pub fn collapse_compare_score(result: &CompareResult) -> u32 {
    result
        .overlap_err
        .saturating_add(result.outside_ink_err.saturating_mul(2))
        .saturating_add(result.black_delta)
}

#[inline]
pub fn refine_compare_score(err: u32, dx: i32, dy: i32) -> u32 {
    err.saturating_add(((dx.abs() + dy.abs()) as u32).saturating_mul(2))
}

pub fn family_match_details(
    comparator: &mut Comparator,
    target: &BitImage,
    target_index: usize,
    reference: &BitImage,
    reference_index: usize,
    signatures: &[SymbolSignature],
    black_counts: &[usize],
) -> Option<(u32, i32, i32)> {
    if target.width.abs_diff(reference.width) > 2 || target.height.abs_diff(reference.height) > 2 {
        return None;
    }
    if !family_signatures_are_compatible(
        signatures[target_index],
        signatures[reference_index],
        black_counts[target_index],
        black_counts[reference_index],
    ) {
        return None;
    }

    let area = target
        .width
        .max(reference.width)
        .saturating_mul(target.height.max(reference.height));
    let max_err = ((area as f32 * 0.05).ceil() as u32).clamp(2, 16);
    let result = comparator.compare_for_refine_family(target, reference, max_err, 2, 1)?;
    Some((result.total_err, result.dx, result.dy))
}

pub fn lossy_family_match_details(
    comparator: &mut Comparator,
    target: &BitImage,
    target_index: usize,
    reference: &BitImage,
    reference_index: usize,
    signatures: &[SymbolSignature],
    black_counts: &[usize],
    max_err: u32,
    max_dx: i32,
    max_dy: i32,
) -> Option<CompareResult> {
    match lossy_family_probe(
        comparator,
        target,
        target_index,
        reference,
        reference_index,
        signatures,
        black_counts,
        max_err,
        max_dx,
        max_dy,
    ) {
        LossyFamilyProbe::Accept(result) => Some(result),
        LossyFamilyProbe::Reject { .. } => None,
    }
}

pub fn lossy_family_probe(
    comparator: &mut Comparator,
    target: &BitImage,
    target_index: usize,
    reference: &BitImage,
    reference_index: usize,
    signatures: &[SymbolSignature],
    black_counts: &[usize],
    max_err: u32,
    max_dx: i32,
    max_dy: i32,
) -> LossyFamilyProbe {
    if target.width.abs_diff(reference.width) > 1 || target.height.abs_diff(reference.height) > 1 {
        return LossyFamilyProbe::Reject {
            reason: "dim",
            result: None,
            limits: None,
        };
    }
    if !family_signatures_are_compatible(
        signatures[target_index],
        signatures[reference_index],
        black_counts[target_index],
        black_counts[reference_index],
    ) {
        return LossyFamilyProbe::Reject {
            reason: "signature",
            result: None,
            limits: None,
        };
    }

    let Some(result) = comparator.compare_detailed_with_limits(
        target,
        reference,
        max_err,
        max_dx,
        max_dy,
    ) else {
        return LossyFamilyProbe::Reject {
            reason: "overlap",
            result: None,
            limits: None,
        };
    };

    let limits = Comparator::collapse_compare_limits(&result);
    if result.outside_ink_err > limits.outside_limit {
        return LossyFamilyProbe::Reject {
            reason: "outside",
            result: Some(result),
            limits: Some(limits),
        };
    }
    if result.row_profile_err > limits.row_limit {
        return LossyFamilyProbe::Reject {
            reason: "row_profile",
            result: Some(result),
            limits: Some(limits),
        };
    }
    if result.col_profile_err > limits.col_limit {
        return LossyFamilyProbe::Reject {
            reason: "col_profile",
            result: Some(result),
            limits: Some(limits),
        };
    }

    LossyFamilyProbe::Accept(result)
}

pub fn format_collapse_probe_reject(
    symbol_index: usize,
    other_index: usize,
    reason: &'static str,
    result: Option<CompareResult>,
    limits: Option<CollapseCompareLimits>,
) -> String {
    match (result, limits) {
        (Some(result), Some(limits)) => format!(
            "collapse pair reject[{reason}]: lhs={} rhs={} dx={} dy={} overlap={} outside={}/{} row={}/{} col={}/{} black_delta={} total={}",
            symbol_index,
            other_index,
            result.dx,
            result.dy,
            result.overlap_err,
            result.outside_ink_err,
            limits.outside_limit,
            result.row_profile_err,
            limits.row_limit,
            result.col_profile_err,
            limits.col_limit,
            result.black_delta,
            result.total_err
        ),
        (Some(result), None) => format!(
            "collapse pair reject[{reason}]: lhs={} rhs={} dx={} dy={} overlap={} outside={} row={} col={} black_delta={} total={}",
            symbol_index,
            other_index,
            result.dx,
            result.dy,
            result.overlap_err,
            result.outside_ink_err,
            result.row_profile_err,
            result.col_profile_err,
            result.black_delta,
            result.total_err
        ),
        (None, _) => format!(
            "collapse pair reject[{reason}]: lhs={} rhs={}",
            symbol_index, other_index
        ),
    }
}

pub fn choose_lossy_family_prototype(
    config: &Jbig2Config,
    members: &[usize],
    symbols: &[BitImage],
    usage: &[usize],
    page_counts: &[usize],
    signatures: &[SymbolSignature],
    black_counts: &[usize],
    comparator: &mut Comparator,
    probe_cache: &mut FxHashMap<u64, LossyFamilyProbe>,
) -> (usize, u64) {
    if members.len() == 1 {
        return (members[0], 0);
    }

    let mut candidate_order = members.to_vec();
    candidate_order.sort_unstable_by(|&lhs, &rhs| {
        let lhs_support = (page_counts[lhs] as u64 * 12) + usage[lhs] as u64 * 2;
        let rhs_support = (page_counts[rhs] as u64 * 12) + usage[rhs] as u64 * 2;
        rhs_support
            .cmp(&lhs_support)
            .then_with(|| usage[rhs].cmp(&usage[lhs]))
            .then_with(|| black_counts[rhs].cmp(&black_counts[lhs]))
            .then_with(|| lhs.cmp(&rhs))
    });

    let mut best_idx = members[0];
    let mut best_score = u64::MAX;
    let mut best_support = 0u64;

    for &candidate in &candidate_order {
        let mut score = 0u64;
        let score_slack = match config.lossy_collapse_prototype_selector_mode {
            LossyCollapsePrototypeSelectorMode::Baseline => best_score / 50,
            LossyCollapsePrototypeSelectorMode::SupportBiased => best_score / 8,
        };
        for &other in members {
            if candidate == other {
                continue;
            }

            let weight = usage[other].max(1) as u64;
            match lossy_family_probe_cached(
                probe_cache,
                comparator,
                &symbols[other],
                other,
                &symbols[candidate],
                candidate,
                signatures,
                black_counts,
                config.lossy_collapse_max_err,
                config.lossy_collapse_max_dx,
                config.lossy_collapse_max_dy,
            ) {
                LossyFamilyProbe::Accept(result) => {
                    score += collapse_compare_score(&result) as u64 * weight;
                }
                LossyFamilyProbe::Reject { .. } => score += 1_000_000 * weight,
            }

            if best_score != u64::MAX && score > best_score.saturating_add(score_slack) {
                break;
            }
        }

        let candidate_support = (page_counts[candidate] as u64 * 12) + usage[candidate] as u64 * 2;
        let should_replace = match config.lossy_collapse_prototype_selector_mode {
            LossyCollapsePrototypeSelectorMode::Baseline => {
                let score_close = if best_score == u64::MAX {
                    false
                } else {
                    score <= best_score + best_score / 50
                };
                score < best_score || (score_close && candidate_support > best_support)
            }
            LossyCollapsePrototypeSelectorMode::SupportBiased => {
                let score_close = if best_score == u64::MAX {
                    false
                } else {
                    score <= best_score + best_score / 8
                };
                score < best_score || (score_close && candidate_support > best_support)
            }
        };

        if should_replace {
            best_score = score;
            best_idx = candidate;
            best_support = candidate_support;
        }
    }

    (best_idx, best_score)
}

pub fn build_lossy_symbol_families(
    inputs: CollapseFamilyBuildInputs<'_>,
) -> (Vec<LossyFamily>, CollapseBuildDiagnostics) {
    if !inputs.config.lossy_symbol_collapse || inputs.global_symbols.len() <= 1 {
        return (Vec::new(), CollapseBuildDiagnostics::default());
    }

    let symbol_count = inputs.global_symbols.len();
    let all_indices: Vec<usize> = (0..symbol_count).collect();
    let bucket_keys: Vec<FamilyBucketKey> = inputs
        .global_symbols
        .iter()
        .zip(inputs.symbol_signatures.iter())
        .map(|(symbol, signature)| family_bucket_key_for_symbol(symbol, signature))
        .collect();
    let mut bucket_map: FxHashMap<FamilyBucketKey, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(symbol_count, Default::default());
    for (symbol_index, &key) in bucket_keys.iter().enumerate() {
        bucket_map.entry(key).or_default().push(symbol_index);
    }

    let mut comparator = Comparator::default();
    let mut probe_cache: FxHashMap<u64, LossyFamilyProbe> =
        FxHashMap::with_capacity_and_hasher(symbol_count.saturating_mul(64), Default::default());
    let mut parent: Vec<usize> = (0..symbol_count).collect();
    let mut rank = vec![0u32; symbol_count];
    let mut accepted_pair_count = 0usize;
    let mut rejected_pair_count = 0usize;
    let mut accepted_samples = Vec::new();
    let mut rejected_samples = Vec::new();
    let mut reject_reason_counts: FxHashMap<&'static str, usize> = FxHashMap::default();
    let mut reject_reason_sample_counts: FxHashMap<&'static str, usize> = FxHashMap::default();

    for &symbol_index in &all_indices {
        let key = bucket_keys[symbol_index];
        for_each_family_bucket_neighbor(key, |neighbor| {
            let Some(bucket) = bucket_map.get(&neighbor) else {
                return;
            };
            let eligible_prefix = bucket.partition_point(|&other_index| other_index < symbol_index);
            for &other_index in &bucket[..eligible_prefix] {
                match lossy_family_probe_cached(
                    &mut probe_cache,
                    &mut comparator,
                    &inputs.global_symbols[symbol_index],
                    symbol_index,
                    &inputs.global_symbols[other_index],
                    other_index,
                    inputs.symbol_signatures,
                    inputs.symbol_pixel_counts,
                    inputs.config.lossy_collapse_max_err,
                    inputs.config.lossy_collapse_max_dx,
                    inputs.config.lossy_collapse_max_dy,
                ) {
                    LossyFamilyProbe::Accept(result) => {
                        accepted_pair_count += 1;
                        if accepted_samples.len() < 48 {
                            accepted_samples.push(format!(
                                "collapse pair accept: lhs={} rhs={} dx={} dy={} overlap={} outside={} row={} col={} black_delta={} total={}",
                                symbol_index,
                                other_index,
                                result.dx,
                                result.dy,
                                result.overlap_err,
                                result.outside_ink_err,
                                result.row_profile_err,
                                result.col_profile_err,
                                result.black_delta,
                                result.total_err
                            ));
                        }
                        uf_union(&mut parent, &mut rank, symbol_index, other_index);
                    }
                    LossyFamilyProbe::Reject { reason, result, limits } => {
                        rejected_pair_count += 1;
                        *reject_reason_counts.entry(reason).or_insert(0) += 1;
                        let sample_count = reject_reason_sample_counts.entry(reason).or_insert(0);
                        if *sample_count < 12 {
                            *sample_count += 1;
                            rejected_samples.push(format_collapse_probe_reject(
                                symbol_index,
                                other_index,
                                reason,
                                result,
                                limits,
                            ));
                        }
                    }
                }
            }
        });
    }

    let mut groups: FxHashMap<usize, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(all_indices.len(), Default::default());
    for &symbol_index in &all_indices {
        let root = uf_find(&mut parent, symbol_index);
        groups.entry(root).or_default().push(symbol_index);
    }

    let mut families = Vec::new();
    let mut family_size_buckets = [0usize; 4];
    let mut skipped_low_value = 0usize;
    let mut skipped_samples = Vec::new();
    let mut retained_members = 0usize;
    let mut discarded_members = 0usize;

    for mut members in groups.into_values() {
        members.sort_unstable();
        let eligible: Vec<usize> = members
            .into_iter()
            .filter(|&index| inputs.symbol_usage[index] >= inputs.config.lossy_collapse_min_usage)
            .collect();
        if eligible.len() < inputs.config.lossy_collapse_min_family_size {
            continue;
        }

        let (prototype_index, prototype_score) = choose_lossy_family_prototype(
            inputs.config,
            &eligible,
            inputs.global_symbols,
            inputs.symbol_usage,
            inputs.symbol_page_count,
            inputs.symbol_signatures,
            inputs.symbol_pixel_counts,
            &mut comparator,
            &mut probe_cache,
        );
        let family_total_usage = eligible.iter().map(|&index| inputs.symbol_usage[index]).sum();
        let prototype_usage = inputs.symbol_usage[prototype_index];
        let needs_exact_page_span = inputs.config.lossy_collapse_min_page_span > 0
            || inputs.config.lossy_collapse_min_total_usage > 0
            || inputs.config.lossy_collapse_min_prototype_usage > 0;
        let family_page_span = if needs_exact_page_span {
            let mut eligible_set: FxHashSet<usize> =
                FxHashSet::with_capacity_and_hasher(eligible.len(), Default::default());
            eligible_set.extend(eligible.iter().copied());
            inputs
                .page_symbol_indices
                .iter()
                .filter(|page_symbols| page_symbols.iter().any(|idx| eligible_set.contains(idx)))
                .count()
        } else {
            eligible
                .iter()
                .map(|&index| inputs.symbol_page_count[index])
                .max()
                .unwrap_or(0)
        };
        let is_economically_useful = family_total_usage
            >= inputs.config.lossy_collapse_min_total_usage
            && (prototype_usage >= inputs.config.lossy_collapse_min_prototype_usage
                || family_page_span >= inputs.config.lossy_collapse_min_page_span);
        if !is_economically_useful {
            skipped_low_value += 1;
            if skipped_samples.len() < 64 {
                skipped_samples.push(format!(
                    "collapse skip low-value: prototype={} members={} total_usage={} prototype_usage={} page_span={}",
                    prototype_index,
                    eligible.len(),
                    family_total_usage,
                    prototype_usage,
                    family_page_span
                ));
            }
            continue;
        }

        let mut family_members = Vec::new();
        for &member_index in &eligible {
            if member_index == prototype_index {
                continue;
            }
            if let LossyFamilyProbe::Accept(result) = lossy_family_probe_cached(
                &mut probe_cache,
                &mut comparator,
                &inputs.global_symbols[member_index],
                member_index,
                &inputs.global_symbols[prototype_index],
                prototype_index,
                inputs.symbol_signatures,
                inputs.symbol_pixel_counts,
                inputs.config.lossy_collapse_max_err,
                inputs.config.lossy_collapse_max_dx,
                inputs.config.lossy_collapse_max_dy,
            ) {
                family_members.push(LossyFamilyMatch {
                    member_index,
                    dx: result.dx,
                    dy: result.dy,
                    err: result.total_err,
                });
            }
        }

        if family_members.len() + 1 >= inputs.config.lossy_collapse_min_family_size {
            let family_size = family_members.len() + 1;
            retained_members += 1;
            discarded_members += family_members.len();
            match family_size {
                0..=3 => family_size_buckets[0] += 1,
                4..=7 => family_size_buckets[1] += 1,
                8..=15 => family_size_buckets[2] += 1,
                _ => family_size_buckets[3] += 1,
            }
            families.push(LossyFamily {
                prototype_index,
                members: family_members,
                total_usage: family_total_usage,
                page_span: family_page_span,
                prototype_score,
            });
        }
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "collapse pair probes: accepted={} rejected={}",
        accepted_pair_count, rejected_pair_count
    ));
    let mut reject_summary: Vec<_> = reject_reason_counts.into_iter().collect();
    reject_summary.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1).then_with(|| lhs.0.cmp(rhs.0)));
    for (reason, count) in reject_summary {
        lines.push(format!("collapse pair rejects[{reason}]={count}"));
    }
    lines.extend(accepted_samples);
    lines.extend(rejected_samples);
    lines.push(format!(
        "collapse family buckets: size2_3={} size4_7={} size8_15={} size16p={}",
        family_size_buckets[0], family_size_buckets[1], family_size_buckets[2], family_size_buckets[3]
    ));
    lines.push(format!(
        "collapse family retention: prototypes={} discarded_members={} skipped_low_value={}",
        retained_members, discarded_members, skipped_low_value
    ));
    lines.extend(skipped_samples);

    (families, CollapseBuildDiagnostics { lines })
}

pub fn build_lossy_prototype(inputs: PrototypeBuildInputs<'_>) -> (BitImage, PrototypeBuildStats) {
    let prototype_index = inputs.family.prototype_index;
    let medoid = inputs.global_symbols[prototype_index].clone();
    let medoid_black = medoid.count_ones();
    let prototype = match inputs.config.lossy_collapse_prototype_mode {
        LossyCollapsePrototypeMode::Medoid => medoid.clone(),
        LossyCollapsePrototypeMode::MajorityVote => {
            majority_vote_prototype(inputs.global_symbols, prototype_index, &inputs.family.members, 0.60)
        }
        LossyCollapsePrototypeMode::AdaptiveMajorityVote => {
            let family_size = inputs.family.members.len() + 1;
            let avg_support = inputs.family.total_usage as f32 / family_size.max(1) as f32;
            let threshold = if avg_support >= 8.0 {
                0.54
            } else if avg_support >= 4.0 {
                0.57
            } else {
                0.60
            };
            majority_vote_prototype(
                inputs.global_symbols,
                prototype_index,
                &inputs.family.members,
                threshold,
            )
        }
        LossyCollapsePrototypeMode::MedoidThenCleanup => cleanup_symbol_bitmap(&medoid, 1),
        LossyCollapsePrototypeMode::MedoidWithAdaptiveCleanup => {
            let iterations = if inputs.family.total_usage >= 24 || inputs.family.members.len() >= 7 {
                2
            } else {
                1
            };
            cleanup_symbol_bitmap(&medoid, iterations)
        }
    };

    let (kept_pixels, removed_pixels, added_pixels) = if inputs.collect_stats {
        bitmap_transition_stats(&medoid, &prototype)
    } else {
        (0, 0, 0)
    };
    let mut before_score = 0u64;
    let mut after_score = 0u64;
    let mut before_count = 0usize;
    let mut after_count = 0usize;
    if inputs.collect_stats {
        let mut comparator = Comparator::default();
        let max_diag_err = inputs.config.lossy_collapse_max_err.saturating_add(16);
        for member in &inputs.family.members {
            let member_bitmap = &inputs.global_symbols[member.member_index];
            if let Some(result) = comparator.compare_detailed(member_bitmap, &medoid, max_diag_err) {
                before_score += collapse_compare_score(&result) as u64;
                before_count += 1;
            }
            if let Some(result) = comparator.compare_detailed(member_bitmap, &prototype, max_diag_err) {
                after_score += collapse_compare_score(&result) as u64;
                after_count += 1;
            }
        }
    }

    let stats = PrototypeBuildStats {
        mode: format!("{:?}", inputs.config.lossy_collapse_prototype_mode),
        medoid_black_pixels: medoid_black,
        output_black_pixels: prototype.count_ones(),
        pixels_kept: kept_pixels,
        pixels_removed: removed_pixels,
        pixels_added: added_pixels,
        avg_member_score_before: if before_count > 0 {
            before_score as f64 / before_count as f64
        } else {
            0.0
        },
        avg_member_score_after: if after_count > 0 {
            after_score as f64 / after_count as f64
        } else {
            0.0
        },
    };

    (prototype, stats)
}

fn bitmap_transition_stats(before: &BitImage, after: &BitImage) -> (usize, usize, usize) {
    let width = before.width.max(after.width);
    let height = before.height.max(after.height);
    let mut kept = 0usize;
    let mut removed = 0usize;
    let mut added = 0usize;
    for y in 0..height {
        for x in 0..width {
            let before_on = x < before.width && y < before.height && before.get_usize(x, y);
            let after_on = x < after.width && y < after.height && after.get_usize(x, y);
            match (before_on, after_on) {
                (true, true) => kept += 1,
                (true, false) => removed += 1,
                (false, true) => added += 1,
                (false, false) => {}
            }
        }
    }
    (kept, removed, added)
}

fn cleanup_symbol_bitmap(symbol: &BitImage, iterations: usize) -> BitImage {
    let mut out = symbol.clone();
    for _ in 0..iterations.max(1) {
        out = cleanup_symbol_bitmap_once(&out);
    }
    out
}

fn cleanup_symbol_bitmap_once(symbol: &BitImage) -> BitImage {
    let mut out = symbol.clone();
    if symbol.width < 3 || symbol.height < 3 {
        return out;
    }

    for y in 1..(symbol.height - 1) {
        for x in 1..(symbol.width - 1) {
            let on = symbol.get_usize(x, y);
            let mut neighbors = 0usize;
            for yy in (y - 1)..=(y + 1) {
                for xx in (x - 1)..=(x + 1) {
                    if xx == x && yy == y {
                        continue;
                    }
                    if symbol.get_usize(xx, yy) {
                        neighbors += 1;
                    }
                }
            }

            if on && neighbors <= 1 {
                out.set_usize(x, y, false);
            } else if !on && neighbors >= 7 {
                out.set_usize(x, y, true);
            }
        }
    }
    out
}

fn majority_vote_prototype(
    symbols: &[BitImage],
    prototype_index: usize,
    members: &[LossyFamilyMatch],
    threshold_ratio: f32,
) -> BitImage {
    let prototype = &symbols[prototype_index];
    let mut votes = vec![0u16; prototype.width * prototype.height];
    let mut samples = 1u16;

    for y in 0..prototype.height {
        for x in 0..prototype.width {
            if prototype.get_usize(x, y) {
                votes[y * prototype.width + x] += 1;
            }
        }
    }

    for member in members {
        let image = &symbols[member.member_index];
        for y in 0..prototype.height {
            for x in 0..prototype.width {
                let sx = x as i32 - member.dx;
                let sy = y as i32 - member.dy;
                if sx < 0 || sy < 0 {
                    continue;
                }
                let sx = sx as usize;
                let sy = sy as usize;
                if sx >= image.width || sy >= image.height {
                    continue;
                }
                if image.get_usize(sx, sy) {
                    votes[y * prototype.width + x] += 1;
                }
            }
        }
        samples += 1;
    }

    let mut out = prototype.clone();
    let threshold = ((samples as f32) * threshold_ratio).ceil() as u16;
    for y in 0..prototype.height {
        for x in 0..prototype.width {
            out.set_usize(x, y, votes[y * prototype.width + x] >= threshold);
        }
    }
    out
}

fn uf_find(parent: &mut [usize], index: usize) -> usize {
    if parent[index] != index {
        let root = uf_find(parent, parent[index]);
        parent[index] = root;
    }
    parent[index]
}

fn uf_union(parent: &mut [usize], rank: &mut [u32], lhs: usize, rhs: usize) {
    let lhs_root = uf_find(parent, lhs);
    let rhs_root = uf_find(parent, rhs);
    if lhs_root == rhs_root {
        return;
    }

    match rank[lhs_root].cmp(&rank[rhs_root]) {
        std::cmp::Ordering::Less => parent[lhs_root] = rhs_root,
        std::cmp::Ordering::Greater => parent[rhs_root] = lhs_root,
        std::cmp::Ordering::Equal => {
            parent[rhs_root] = lhs_root;
            rank[lhs_root] += 1;
        }
    }
}
