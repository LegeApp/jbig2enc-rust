use crate::jbig2comparator::{CollapseCompareLimits, Comparator, CompareResult};
use crate::jbig2collapse_context::{CollapseContextModel, ContextDecision};
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
    pub top_mass: u16,
    pub bottom_mass: u16,
    pub hole_count: u8,
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
    pub context_model: Option<&'a CollapseContextModel>,
}

pub struct PrototypeBuildInputs<'a> {
    pub config: &'a Jbig2Config,
    pub family: &'a LossyFamily,
    pub scale_profile: CollapseScaleProfile,
    pub global_symbols: &'a [BitImage],
    pub symbol_usage: &'a [usize],
    pub symbol_page_count: &'a [usize],
    pub symbol_signatures: &'a [SymbolSignature],
    pub symbol_pixel_counts: &'a [usize],
    pub collect_stats: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CollapseScaleProfile {
    pub ref_width: usize,
    pub ref_height: usize,
    pub ref_black: u32,
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
    let mut top_mass = 0u32;
    let mut bottom_mass = 0u32;
    let mid_x = img.width / 2;
    let mid_y = img.height / 2;
    let packed = img.packed_words();
    let words_per_row = (img.width + 31) >> 5;
    let tail_bits = img.width & 31;
    let tail_mask = if tail_bits == 0 {
        u32::MAX
    } else {
        u32::MAX << (32 - tail_bits)
    };

    for y in 0..img.height {
        let row = &packed[y * words_per_row..(y + 1) * words_per_row];
        for (word_idx, &row_word) in row.iter().enumerate() {
            let mut word = row_word;
            if tail_bits != 0 && word_idx + 1 == words_per_row {
                word &= tail_mask;
            }
            while word != 0 {
                let bit = word.leading_zeros() as usize;
                let x = word_idx * 32 + bit;
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
                if y < mid_y {
                    top_mass += 1;
                } else {
                    bottom_mass += 1;
                }
                word &= !(1u32 << (31 - bit));
            }
        }
    }

    let hole_count = count_enclosed_white_components(img).min(u8::MAX as usize) as u8;

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
        top_mass: top_mass.min(u16::MAX as u32) as u16,
        bottom_mass: bottom_mass.min(u16::MAX as u32) as u16,
        hole_count,
    }
}

fn count_enclosed_white_components(img: &BitImage) -> usize {
    if img.width < 3 || img.height < 3 {
        return 0;
    }

    let mut visited = vec![false; img.width * img.height];
    let mut stack = Vec::new();
    let mut holes = 0usize;

    for y in 0..img.height {
        for x in 0..img.width {
            let idx = y * img.width + x;
            if visited[idx] || img.get_usize(x, y) {
                continue;
            }

            visited[idx] = true;
            stack.push((x, y));
            let mut touches_border = x == 0 || y == 0 || x + 1 == img.width || y + 1 == img.height;

            while let Some((cx, cy)) = stack.pop() {
                let x0 = cx.saturating_sub(1);
                let x1 = (cx + 1).min(img.width - 1);
                let y0 = cy.saturating_sub(1);
                let y1 = (cy + 1).min(img.height - 1);

                for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        let nidx = ny * img.width + nx;
                        if visited[nidx] || img.get_usize(nx, ny) {
                            continue;
                        }
                        visited[nidx] = true;
                        if nx == 0 || ny == 0 || nx + 1 == img.width || ny + 1 == img.height {
                            touches_border = true;
                        }
                        stack.push((nx, ny));
                    }
                }
            }

            if !touches_border {
                holes += 1;
            }
        }
    }

    holes
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

fn weighted_median_usize(values: &mut [(usize, usize)]) -> usize {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable_by_key(|&(value, _)| value);
    let total_weight: usize = values.iter().map(|&(_, weight)| weight.max(1)).sum();
    let target = total_weight.div_ceil(2);
    let mut running = 0usize;
    for &(value, weight) in values.iter() {
        running += weight.max(1);
        if running >= target {
            return value;
        }
    }
    values.last().map(|&(value, _)| value).unwrap_or(0)
}

fn weighted_median_u32(values: &mut [(u32, usize)]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable_by_key(|&(value, _)| value);
    let total_weight: usize = values.iter().map(|&(_, weight)| weight.max(1)).sum();
    let target = total_weight.div_ceil(2);
    let mut running = 0usize;
    for &(value, weight) in values.iter() {
        running += weight.max(1);
        if running >= target {
            return value;
        }
    }
    values.last().map(|&(value, _)| value).unwrap_or(0)
}

pub fn estimate_collapse_scale_profile(
    symbols: &[BitImage],
    signatures: &[SymbolSignature],
    usage: &[usize],
) -> CollapseScaleProfile {
    let mut widths = Vec::with_capacity(symbols.len());
    let mut heights = Vec::with_capacity(symbols.len());
    let mut blacks = Vec::with_capacity(symbols.len());

    for ((symbol, signature), &weight) in symbols.iter().zip(signatures.iter()).zip(usage.iter()) {
        if signature.black < 4 || symbol.width < 2 || symbol.height < 2 {
            continue;
        }
        let weight = weight.max(1);
        widths.push((symbol.width, weight));
        heights.push((symbol.height, weight));
        blacks.push((signature.black, weight));
    }

    CollapseScaleProfile {
        ref_width: weighted_median_usize(&mut widths).max(1),
        ref_height: weighted_median_usize(&mut heights).max(1),
        ref_black: weighted_median_u32(&mut blacks).max(1),
    }
}

#[inline]
fn collapse_shape_is_fragile(
    config: &Jbig2Config,
    scale_profile: CollapseScaleProfile,
    symbol: &BitImage,
    sig: &SymbolSignature,
) -> bool {
    let ref_width = scale_profile.ref_width.max(1);
    let ref_height = scale_profile.ref_height.max(1);
    let ref_black = scale_profile.ref_black.max(1);

    symbol.width.saturating_mul(1000)
        < ref_width.saturating_mul(config.lossy_collapse_min_width_ratio_permille as usize)
        || symbol.height.saturating_mul(1000)
            < ref_height.saturating_mul(config.lossy_collapse_min_height_ratio_permille as usize)
        || sig.black.saturating_mul(1000)
            < ref_black.saturating_mul(config.lossy_collapse_min_black_ratio_permille as u32)
}

#[inline]
fn should_cleanup_prototype(
    config: &Jbig2Config,
    scale_profile: CollapseScaleProfile,
    symbol: &BitImage,
    sig: &SymbolSignature,
    family_total_usage: usize,
) -> bool {
    !collapse_shape_is_fragile(config, scale_profile, symbol, sig)
        && family_total_usage >= 6
        && symbol.width >= 5
        && symbol.height >= 8
        && sig.black >= 20
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
        && lhs.top_mass.abs_diff(rhs.top_mass) <= mass_tol
        && lhs.bottom_mass.abs_diff(rhs.bottom_mass) <= mass_tol
        && lhs.hole_count == rhs.hole_count
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

    let Some(result) = comparator.compare_detailed(target, reference, max_err) else {
        return LossyFamilyProbe::Reject {
            reason: "overlap",
            result: None,
            limits: None,
        };
    };

    if result.dx.abs() > max_dx || result.dy.abs() > max_dy {
        return LossyFamilyProbe::Reject {
            reason: "shift",
            result: Some(result),
            limits: None,
        };
    }

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

#[inline]
fn context_decision(
    config: &Jbig2Config,
    context_model: Option<&CollapseContextModel>,
    lhs: usize,
    rhs: usize,
) -> ContextDecision {
    context_model
        .map(|model| model.merge_decision(lhs, rhs, config.lossy_collapse_context_mode))
        .unwrap_or(ContextDecision::Unknown)
}

#[inline]
fn collapse_score_allowed_with_unknown_context(config: &Jbig2Config, result: &CompareResult) -> bool {
    let score = collapse_compare_score(result);
    let strict_limit = config.lossy_collapse_max_err.saturating_div(2).saturating_add(2);
    score <= strict_limit && result.outside_ink_err == 0 && result.dx.abs() + result.dy.abs() <= 1
}

pub fn choose_lossy_family_prototype(
    config: &Jbig2Config,
    members: &[usize],
    symbols: &[BitImage],
    usage: &[usize],
    page_counts: &[usize],
    signatures: &[SymbolSignature],
    black_counts: &[usize],
    context_model: Option<&CollapseContextModel>,
    comparator: &mut Comparator,
    probe_cache: &mut FxHashMap<u64, LossyFamilyProbe>,
) -> (usize, u64) {
    if members.len() == 1 {
        return (members[0], 0);
    }

    #[derive(Clone, Copy)]
    struct PrototypeCandidateStats {
        index: usize,
        total_score: u64,
        close_weight: u64,
        close_score_sum: u64,
        support: u64,
    }

    impl PrototypeCandidateStats {
        fn avg_close_score(self) -> u64 {
            if self.close_weight == 0 {
                u64::MAX
            } else {
                self.close_score_sum / self.close_weight
            }
        }
    }

    let mut best_idx = members[0];
    let mut best_score = u64::MAX;
    let mut best_support = 0u64;
    let close_threshold = config.lossy_collapse_max_err.saturating_div(2).saturating_add(4) as u64;
    let mut candidate_stats = Vec::with_capacity(members.len());

    for &candidate in members {
        let mut score = 0u64;
        let mut close_weight = 0u64;
        let mut close_score_sum = 0u64;
        for &other in members {
            if candidate == other {
                continue;
            }

            let weight = usage[other].max(1) as u64;
            match context_decision(config, context_model, other, candidate) {
                ContextDecision::Reject => {
                    score += 1_000_000 * weight;
                    continue;
                }
                ContextDecision::Unknown => {
                    score += 2_000 * weight;
                }
                ContextDecision::Allow => {}
            }
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
                    let compare_score = collapse_compare_score(&result) as u64;
                    if compare_score <= close_threshold {
                        close_weight += weight;
                        close_score_sum += compare_score.saturating_mul(weight);
                    }
                    match context_decision(config, context_model, other, candidate) {
                        ContextDecision::Allow => score += compare_score * weight,
                        ContextDecision::Unknown => {
                            score += compare_score.saturating_mul(4).saturating_mul(weight)
                        }
                        ContextDecision::Reject => unreachable!(),
                    }
                }
                LossyFamilyProbe::Reject { .. } => score += 1_000_000 * weight,
            }
        }

        let candidate_support = (page_counts[candidate] as u64 * 8) + usage[candidate] as u64;
        candidate_stats.push(PrototypeCandidateStats {
            index: candidate,
            total_score: score,
            close_weight,
            close_score_sum,
            support: candidate_support,
        });
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
            LossyCollapsePrototypeSelectorMode::DenseCenter
            | LossyCollapsePrototypeSelectorMode::DenseTrimmed => false,
        };

        if should_replace {
            best_score = score;
            best_idx = candidate;
            best_support = candidate_support;
        }
    }

    match config.lossy_collapse_prototype_selector_mode {
        LossyCollapsePrototypeSelectorMode::DenseCenter
        | LossyCollapsePrototypeSelectorMode::DenseTrimmed => {
            let max_close = candidate_stats
                .iter()
                .map(|stats| stats.close_weight)
                .max()
                .unwrap_or(0);
            let dense_pool: Vec<PrototypeCandidateStats> = if config
                .lossy_collapse_prototype_selector_mode
                == LossyCollapsePrototypeSelectorMode::DenseTrimmed
            {
                let trimmed: Vec<_> = candidate_stats
                    .iter()
                    .copied()
                    .filter(|stats| stats.close_weight >= 2 && stats.close_weight.saturating_mul(2) >= max_close)
                    .collect();
                if trimmed.is_empty() {
                    candidate_stats.clone()
                } else {
                    trimmed
                }
            } else {
                candidate_stats.clone()
            };

            if let Some(best_dense) = dense_pool.into_iter().max_by(|lhs, rhs| {
                lhs.close_weight
                    .cmp(&rhs.close_weight)
                    .then_with(|| rhs.avg_close_score().cmp(&lhs.avg_close_score()))
                    .then_with(|| rhs.support.cmp(&lhs.support))
                    .then_with(|| rhs.total_score.cmp(&lhs.total_score))
            }) {
                return (best_dense.index, best_dense.total_score);
            }
        }
        _ => {}
    }

    (best_idx, best_score)
}

pub fn build_lossy_symbol_families(
    inputs: CollapseFamilyBuildInputs<'_>,
) -> (Vec<LossyFamily>, CollapseBuildDiagnostics) {
    if !inputs.config.uses_legacy_collapse() || inputs.global_symbols.len() <= 1 {
        return (Vec::new(), CollapseBuildDiagnostics::default());
    }

    let symbol_count = inputs.global_symbols.len();
    let scale_profile = estimate_collapse_scale_profile(
        inputs.global_symbols,
        inputs.symbol_signatures,
        inputs.symbol_usage,
    );
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
                let pair_context = context_decision(
                    inputs.config,
                    inputs.context_model,
                    symbol_index,
                    other_index,
                );
                if pair_context == ContextDecision::Reject {
                    rejected_pair_count += 1;
                    *reject_reason_counts.entry("context").or_insert(0) += 1;
                    let sample_count = reject_reason_sample_counts.entry("context").or_insert(0);
                    if *sample_count < 12 {
                        *sample_count += 1;
                        rejected_samples.push(format!(
                            "collapse pair reject[context]: lhs={} rhs={}",
                            symbol_index, other_index
                        ));
                    }
                    continue;
                }
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
                        if pair_context == ContextDecision::Unknown
                            && !collapse_score_allowed_with_unknown_context(inputs.config, &result)
                        {
                            rejected_pair_count += 1;
                            *reject_reason_counts.entry("context_unknown").or_insert(0) += 1;
                            let sample_count =
                                reject_reason_sample_counts.entry("context_unknown").or_insert(0);
                            if *sample_count < 12 {
                                *sample_count += 1;
                                rejected_samples.push(format!(
                                    "collapse pair reject[context_unknown]: lhs={} rhs={} score={} total={} dx={} dy={}",
                                    symbol_index,
                                    other_index,
                                    collapse_compare_score(&result),
                                    result.total_err,
                                    result.dx,
                                    result.dy
                                ));
                            }
                            continue;
                        }
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
    let mut skipped_fragile_shape = 0usize;
    let mut skipped_samples = Vec::new();
    let mut retained_members = 0usize;
    let mut discarded_members = 0usize;

    for mut members in groups.into_values() {
        members.sort_unstable();
        let eligible: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&index| {
                inputs.symbol_usage[index] >= inputs.config.lossy_collapse_min_usage
                    && !collapse_shape_is_fragile(
                        inputs.config,
                        scale_profile,
                        &inputs.global_symbols[index],
                        &inputs.symbol_signatures[index],
                    )
            })
            .collect();
        let skipped_in_group = members.len().saturating_sub(eligible.len());
        if skipped_in_group > 0 {
            skipped_fragile_shape += skipped_in_group;
            if skipped_samples.len() < 64 {
                for &index in members.iter().filter(|&&index| {
                    inputs.symbol_usage[index] >= inputs.config.lossy_collapse_min_usage
                        && collapse_shape_is_fragile(
                            inputs.config,
                            scale_profile,
                            &inputs.global_symbols[index],
                            &inputs.symbol_signatures[index],
                        )
                }) {
                    skipped_samples.push(format!(
                        "collapse skip fragile-shape: symbol={} w={} h={} black={} usage={}",
                        index,
                        inputs.global_symbols[index].width,
                        inputs.global_symbols[index].height,
                        inputs.symbol_signatures[index].black,
                        inputs.symbol_usage[index]
                    ));
                    if skipped_samples.len() >= 64 {
                        break;
                    }
                }
            }
        }
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
            inputs.context_model,
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
            let member_context = context_decision(
                inputs.config,
                inputs.context_model,
                member_index,
                prototype_index,
            );
            if member_context == ContextDecision::Reject {
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
                if member_context == ContextDecision::Unknown
                    && !collapse_score_allowed_with_unknown_context(inputs.config, &result)
                {
                    continue;
                }
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
        "collapse scale profile: ref_width={} ref_height={} ref_black={} width_ratio={} height_ratio={} black_ratio={}",
        scale_profile.ref_width,
        scale_profile.ref_height,
        scale_profile.ref_black,
        inputs.config.lossy_collapse_min_width_ratio_permille,
        inputs.config.lossy_collapse_min_height_ratio_permille,
        inputs.config.lossy_collapse_min_black_ratio_permille
    ));
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
        "collapse family retention: prototypes={} discarded_members={} skipped_low_value={} skipped_fragile_shape={}",
        retained_members, discarded_members, skipped_low_value, skipped_fragile_shape
    ));
    lines.extend(skipped_samples);

    (families, CollapseBuildDiagnostics { lines })
}

pub fn build_lossy_prototype(inputs: PrototypeBuildInputs<'_>) -> (BitImage, PrototypeBuildStats) {
    let prototype_index = inputs.family.prototype_index;
    let medoid = inputs.global_symbols[prototype_index].clone();
    let medoid_black = medoid.count_ones();
    let medoid_signature = inputs.symbol_signatures[prototype_index];
    let allow_cleanup = should_cleanup_prototype(
        inputs.config,
        inputs.scale_profile,
        &medoid,
        &medoid_signature,
        inputs.family.total_usage,
    );
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
        LossyCollapsePrototypeMode::MedoidThenCleanup => {
            if allow_cleanup {
                cleanup_symbol_bitmap(&medoid, 1)
            } else {
                medoid.clone()
            }
        }
        LossyCollapsePrototypeMode::MedoidWithAdaptiveCleanup => {
            let iterations = if allow_cleanup
                && (inputs.family.total_usage >= 24 || inputs.family.members.len() >= 7)
            {
                2
            } else if allow_cleanup {
                1
            } else {
                0
            };
            if iterations == 0 {
                medoid.clone()
            } else {
                cleanup_symbol_bitmap(&medoid, iterations)
            }
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
