use crate::jbig2collapse::{
    CollapseScaleProfile, FamilyBucketKey, SymbolSignature, estimate_collapse_scale_profile,
    family_bucket_key_for_symbol, family_signatures_are_compatible, for_each_family_bucket_neighbor,
    symbol_looks_like_fragile_mark,
};
use crate::jbig2collapse_context::{CollapseContextModel, ContextDecision};
use crate::jbig2comparator::{Comparator, CompareResult};
use crate::jbig2structs::Jbig2Config;
use crate::jbig2sym::BitImage;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, Clone, Copy)]
pub struct UnifiedClassMember {
    pub member_index: usize,
    pub dx: i32,
    pub dy: i32,
    pub score: u32,
}

#[derive(Debug, Clone)]
pub struct UnifiedClass {
    pub representative_index: usize,
    pub core_members: Vec<UnifiedClassMember>,
    pub border_members: Vec<UnifiedClassMember>,
    pub class_size: usize,
    pub dense_core_size: usize,
    pub total_usage: usize,
    pub page_span: usize,
    pub representative_score: u64,
    pub retained_border_members: usize,
    pub retained_outlier_members: usize,
    pub candidate_subclusters: usize,
    pub split_fragile_members: usize,
    pub estimated_gain: i32,
}

#[derive(Debug, Clone, Default)]
pub struct UnifyBuildDiagnostics {
    pub lines: Vec<String>,
}

pub struct SymbolUnifyInputs<'a> {
    pub config: &'a Jbig2Config,
    pub global_symbols: &'a [BitImage],
    pub symbol_usage: &'a [usize],
    pub symbol_page_count: &'a [usize],
    pub symbol_signatures: &'a [SymbolSignature],
    pub symbol_pixel_counts: &'a [usize],
    pub context_model: Option<&'a CollapseContextModel>,
    pub collect_diagnostics: bool,
}

#[derive(Debug, Clone, Copy)]
struct PairObservation {
    result: CompareResult,
    class_score: u32,
    assignment_score: u32,
}

#[derive(Debug, Clone, Default)]
struct ClassTriage {
    candidate_subclusters: usize,
    border_members: Vec<UnifiedClassMember>,
    retained_border_members: usize,
    retained_outlier_members: usize,
}

#[derive(Debug, Clone, Copy)]
struct GainBreakdown {
    bitmap_savings: i32,
    id_savings: i32,
    representative_penalty: i32,
    retained_penalty: i32,
    net_gain: i32,
}

#[derive(Debug, Clone, Copy)]
struct CandidateStats {
    index: usize,
    close_support: u64,
    close_score_sum: u64,
    total_score: u64,
    support: u64,
}

impl CandidateStats {
    fn avg_close_score(self) -> u64 {
        if self.close_support == 0 {
            u64::MAX
        } else {
            self.close_score_sum / self.close_support
        }
    }
}

#[inline]
fn pair_key(lhs: usize, rhs: usize) -> u64 {
    let (lo, hi) = if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) };
    ((lo as u64) << 32) | hi as u64
}

#[inline]
fn reverse_pair(mut obs: PairObservation) -> PairObservation {
    obs.result.dx = -obs.result.dx;
    obs.result.dy = -obs.result.dy;
    obs
}

#[inline]
fn class_pair_score(result: &CompareResult) -> u32 {
    result
        .total_err
        .saturating_add(result.black_delta)
        .saturating_add(((result.dx.abs() + result.dy.abs()) as u32).saturating_mul(2))
        .saturating_add((result.row_profile_err + result.col_profile_err) / 32)
}

#[inline]
fn assignment_pair_score(result: &CompareResult) -> u32 {
    result
        .total_err
        .saturating_add(result.black_delta.saturating_mul(2))
        .saturating_add(result.outside_ink_err.saturating_mul(3))
        .saturating_add(((result.dx.abs() + result.dy.abs()) as u32).saturating_mul(3))
        .saturating_add((result.row_profile_err + result.col_profile_err) / 24)
}

#[inline]
fn prescreen_pair(
    comparator: &mut Comparator,
    lhs: &BitImage,
    rhs: &BitImage,
    lhs_black: usize,
    rhs_black: usize,
    max_err: u32,
    max_dx: i32,
    max_dy: i32,
) -> bool {
    if lhs.width.abs_diff(rhs.width) > 1 || lhs.height.abs_diff(rhs.height) > 1 {
        return false;
    }

    let area = lhs.width.max(rhs.width).saturating_mul(lhs.height.max(rhs.height));
    let pixel_delta_limit = (area / 10).clamp(4, 16);
    if lhs_black.abs_diff(rhs_black) > pixel_delta_limit {
        return false;
    }

    let overlap_limit = max_err.saturating_add(2).min(14);
    let Some(result) = comparator.compare_overlap_only(lhs, rhs, overlap_limit) else {
        return false;
    };
    result.dx.abs() <= max_dx
        && result.dy.abs() <= max_dy
        && result.overlap_err <= overlap_limit
        && result.black_delta <= pixel_delta_limit as u32
}

#[inline]
fn fragile_mark_indices(
    config: &Jbig2Config,
    scale_profile: CollapseScaleProfile,
    symbols: &[BitImage],
    signatures: &[SymbolSignature],
) -> Vec<bool> {
    symbols
        .iter()
        .zip(signatures.iter())
        .map(|(symbol, sig)| symbol_looks_like_fragile_mark(config, scale_profile, symbol, sig))
        .collect()
}

fn find_root(parent: &mut [usize], index: usize) -> usize {
    if parent[index] != index {
        let root = find_root(parent, parent[index]);
        parent[index] = root;
    }
    parent[index]
}

fn union(parent: &mut [usize], rank: &mut [u8], lhs: usize, rhs: usize) {
    let lhs_root = find_root(parent, lhs);
    let rhs_root = find_root(parent, rhs);
    if lhs_root == rhs_root {
        return;
    }
    if rank[lhs_root] < rank[rhs_root] {
        parent[lhs_root] = rhs_root;
    } else if rank[lhs_root] > rank[rhs_root] {
        parent[rhs_root] = lhs_root;
    } else {
        parent[rhs_root] = lhs_root;
        rank[lhs_root] = rank[lhs_root].saturating_add(1);
    }
}

fn get_or_compute_pair(
    pair_cache: &mut FxHashMap<u64, Option<PairObservation>>,
    comparator: &mut Comparator,
    symbols: &[BitImage],
    signatures: &[SymbolSignature],
    black_counts: &[usize],
    lhs: usize,
    rhs: usize,
    max_err: u32,
    max_dx: i32,
    max_dy: i32,
) -> Option<PairObservation> {
    let reverse = lhs > rhs;
    let (lo, hi) = if reverse { (rhs, lhs) } else { (lhs, rhs) };
    let key = pair_key(lo, hi);
    let cached = pair_cache.get(&key).copied().flatten();
    let pair = if let Some(obs) = cached {
        obs
    } else {
        if !family_signatures_are_compatible(
            signatures[lo],
            signatures[hi],
            black_counts[lo],
            black_counts[hi],
        ) {
            pair_cache.insert(key, None);
            return None;
        }
        if !prescreen_pair(
            comparator,
            &symbols[lo],
            &symbols[hi],
            black_counts[lo],
            black_counts[hi],
            max_err,
            max_dx,
            max_dy,
        ) {
            pair_cache.insert(key, None);
            return None;
        }

        let result = comparator.compare_for_symbol_unify(
            &symbols[lo],
            &symbols[hi],
            max_err,
            max_dx,
            max_dy,
        )?;
        let obs = PairObservation {
            class_score: class_pair_score(&result),
            assignment_score: assignment_pair_score(&result),
            result,
        };
        pair_cache.insert(key, Some(obs));
        obs
    };

    if reverse {
        Some(reverse_pair(pair))
    } else {
        Some(pair)
    }
}

fn connected_components_for_members(
    members: &[usize],
    close_edges: &FxHashMap<usize, Vec<usize>>,
) -> Vec<Vec<usize>> {
    if members.is_empty() {
        return Vec::new();
    }

    let member_set: FxHashSet<usize> = members.iter().copied().collect();
    let mut seen = FxHashSet::default();
    let mut components = Vec::new();

    for start in members {
        if !seen.insert(*start) {
            continue;
        }
        let mut stack = vec![*start];
        let mut component = vec![*start];
        while let Some(node) = stack.pop() {
            if let Some(neighbors) = close_edges.get(&node) {
                for &neighbor in neighbors {
                    if !member_set.contains(&neighbor) || !seen.insert(neighbor) {
                        continue;
                    }
                    component.push(neighbor);
                    stack.push(neighbor);
                }
            }
        }
        components.push(component);
    }

    components.sort_by(|lhs, rhs| rhs.len().cmp(&lhs.len()).then_with(|| lhs[0].cmp(&rhs[0])));
    components
}

fn dense_core_component(
    members: &[usize],
    close_edges: &FxHashMap<usize, Vec<usize>>,
) -> (Vec<usize>, usize) {
    if members.len() < 2 {
        return (Vec::new(), 0);
    }

    let min_pts = members.len().div_ceil(4).max(2);
    let core_nodes: Vec<usize> = members
        .iter()
        .copied()
        .filter(|member| close_edges.get(member).map_or(0, Vec::len) + 1 >= min_pts)
        .collect();
    if core_nodes.len() < 2 {
        return (Vec::new(), 0);
    }

    let components = connected_components_for_members(&core_nodes, close_edges);
    let candidate_subclusters = components.len().saturating_sub(1);
    (components.into_iter().next().unwrap_or_default(), candidate_subclusters)
}

fn triage_non_core_members(
    non_core: &[usize],
    close_edges: &FxHashMap<usize, Vec<usize>>,
    representative_index: usize,
    representative_usage: usize,
    representative_page_span: usize,
    pair_cache: &mut FxHashMap<u64, Option<PairObservation>>,
    comparator: &mut Comparator,
    symbols: &[BitImage],
    signatures: &[SymbolSignature],
    black_counts: &[usize],
    usage: &[usize],
    page_counts: &[usize],
    context_model: Option<&CollapseContextModel>,
    context_mode: crate::jbig2structs::LossyCollapseContextMode,
    max_err: u32,
    max_dx: i32,
    max_dy: i32,
    border_accept_limit: u32,
    max_border_outside_ink: u32,
    min_class_usage: usize,
    min_page_span: usize,
) -> ClassTriage {
    if non_core.is_empty() {
        return ClassTriage::default();
    }

    let mut triage = ClassTriage::default();
    for component in connected_components_for_members(non_core, close_edges) {
        let mut retained_component = Vec::new();

        for &member in &component {
            let context_decision = context_model
                .map(|model| model.merge_decision(member, representative_index, context_mode))
                .unwrap_or(ContextDecision::Unknown);
            let Some(obs) = get_or_compute_pair(
                pair_cache,
                comparator,
                symbols,
                signatures,
                black_counts,
                member,
                representative_index,
                max_err,
                max_dx,
                max_dy,
            ) else {
                retained_component.push(member);
                continue;
            };

            let strong_representative = representative_usage >= min_class_usage
                && representative_page_span >= min_page_span;
            let border_accept = match context_decision {
                ContextDecision::Reject => false,
                ContextDecision::Allow => {
                    strong_representative
                        && obs.assignment_score <= border_accept_limit
                        && obs.result.outside_ink_err <= max_border_outside_ink
                }
                ContextDecision::Unknown => {
                    strong_representative
                        && obs.assignment_score <= border_accept_limit.saturating_sub(1)
                        && obs.result.outside_ink_err <= max_border_outside_ink.min(1)
                }
            };
            if border_accept
                && obs.result.dx.abs() <= max_dx
                && obs.result.dy.abs() <= max_dy
            {
                triage.border_members.push(UnifiedClassMember {
                    member_index: member,
                    dx: obs.result.dx,
                    dy: obs.result.dy,
                    score: obs.assignment_score,
                });
            } else {
                retained_component.push(member);
            }
        }

        if retained_component.is_empty() {
            continue;
        }

        let component_usage: usize = retained_component.iter().map(|&idx| usage[idx]).sum();
        let component_page_span = retained_component
            .iter()
            .map(|&idx| page_counts[idx])
            .max()
            .unwrap_or(0);

        if retained_component.len() >= 2
            && component_usage >= min_class_usage
            && component_page_span >= min_page_span
        {
            triage.candidate_subclusters += 1;
            triage.retained_border_members += retained_component.len();
        } else {
            triage.retained_outlier_members += retained_component.len();
        }
    }

    triage
}

fn estimate_class_gain(
    symbols: &[BitImage],
    usage: &[usize],
    page_counts: &[usize],
    representative_index: usize,
    unified_members: &[UnifiedClassMember],
    border_members: &[UnifiedClassMember],
    retained_border_members: usize,
    retained_outlier_members: usize,
    representative_score: u64,
) -> GainBreakdown {
    if unified_members.is_empty() && border_members.is_empty() {
        return GainBreakdown {
            bitmap_savings: 0,
            id_savings: 0,
            representative_penalty: 0,
            retained_penalty: 0,
            net_gain: i32::MIN / 4,
        };
    }

    let bitmap_savings: i32 = unified_members
        .iter()
        .chain(border_members.iter())
        .map(|member| {
            let symbol = &symbols[member.member_index];
            ((symbol.width.saturating_mul(symbol.height) + 7) / 8) as i32 + 10
        })
        .sum();
    let id_savings: i32 = unified_members
        .iter()
        .chain(border_members.iter())
        .map(|member| {
            (usage[member.member_index].min(12) as i32) * 3
                + (page_counts[member.member_index].min(6) as i32) * 4
                + 6
        })
        .sum();

    let representative_penalty = {
        let symbol = &symbols[representative_index];
        (((symbol.width.saturating_mul(symbol.height) + 7) / 8) as i32 / 2)
            + (representative_score.min(1024) as i32 / 12)
    };
    let retained_penalty = retained_border_members as i32 * 8
        + retained_outlier_members as i32 * 5
        + border_members.len() as i32 * 2;

    GainBreakdown {
        bitmap_savings,
        id_savings,
        representative_penalty,
        retained_penalty,
        net_gain: bitmap_savings + id_savings - representative_penalty - retained_penalty,
    }
}

fn select_dense_representative(
    core: &[usize],
    pair_cache: &mut FxHashMap<u64, Option<PairObservation>>,
    comparator: &mut Comparator,
    symbols: &[BitImage],
    signatures: &[SymbolSignature],
    black_counts: &[usize],
    usage: &[usize],
    page_counts: &[usize],
    max_err: u32,
    max_dx: i32,
    max_dy: i32,
    close_threshold: u32,
) -> Option<(usize, u64)> {
    let mut best: Option<CandidateStats> = None;

    for &candidate in core {
        let mut stats = CandidateStats {
            index: candidate,
            close_support: 0,
            close_score_sum: 0,
            total_score: 0,
            support: (page_counts[candidate] as u64).saturating_mul(8) + usage[candidate] as u64,
        };

        for &other in core {
            if candidate == other {
                continue;
            }
            let weight = ((page_counts[other].max(1) * 4) + usage[other].max(1)) as u64;
            let Some(obs) = get_or_compute_pair(
                pair_cache,
                comparator,
                symbols,
                signatures,
                black_counts,
                candidate,
                other,
                max_err,
                max_dx,
                max_dy,
            ) else {
                stats.total_score = stats.total_score.saturating_add(1_000_000 * weight);
                continue;
            };
            stats.total_score = stats
                .total_score
                .saturating_add(obs.class_score as u64 * weight);
            if obs.assignment_score <= close_threshold {
                stats.close_support = stats.close_support.saturating_add(weight);
                stats.close_score_sum = stats
                    .close_score_sum
                    .saturating_add(obs.assignment_score as u64 * weight);
            }
        }

        let replace = best.is_none_or(|current| {
            stats.close_support > current.close_support
                || (stats.close_support == current.close_support
                    && stats.avg_close_score() < current.avg_close_score())
                || (stats.close_support == current.close_support
                    && stats.avg_close_score() == current.avg_close_score()
                    && stats.total_score < current.total_score)
                || (stats.close_support == current.close_support
                    && stats.avg_close_score() == current.avg_close_score()
                    && stats.total_score == current.total_score
                    && stats.support > current.support)
        });

        if replace {
            best = Some(stats);
        }
    }

    best.map(|stats| (stats.index, stats.total_score))
}

pub fn build_symbol_unify_classes(
    inputs: SymbolUnifyInputs<'_>,
) -> (Vec<UnifiedClass>, UnifyBuildDiagnostics) {
    if inputs.global_symbols.len() <= 1 {
        return (Vec::new(), UnifyBuildDiagnostics::default());
    }

    let class_max_err = inputs.config.sym_unify_max_err.max(4);
    let class_max_dx = inputs.config.sym_unify_max_dx.max(0);
    let class_max_dy = inputs.config.sym_unify_max_dy.max(0);
    let class_accept_limit = inputs
        .config
        .sym_unify_class_accept_limit
        .max(class_max_err);
    let close_threshold = inputs
        .config
        .sym_unify_core_close_threshold
        .min(class_accept_limit);
    let border_accept_limit = class_accept_limit
        .saturating_add(inputs.config.sym_unify_border_score_slack);

    let scale_profile = estimate_collapse_scale_profile(
        inputs.global_symbols,
        inputs.symbol_signatures,
        inputs.symbol_usage,
    );
    let fragile_marks = fragile_mark_indices(
        inputs.config,
        scale_profile,
        inputs.global_symbols,
        inputs.symbol_signatures,
    );
    let bucket_keys: Vec<FamilyBucketKey> = inputs
        .global_symbols
        .iter()
        .zip(inputs.symbol_signatures.iter())
        .map(|(symbol, signature)| family_bucket_key_for_symbol(symbol, signature))
        .collect();

    let mut bucket_map: FxHashMap<FamilyBucketKey, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(inputs.global_symbols.len(), Default::default());
    for (index, &key) in bucket_keys.iter().enumerate() {
        bucket_map.entry(key).or_default().push(index);
    }

    let mut comparator = Comparator::default();
    let mut pair_cache: FxHashMap<u64, Option<PairObservation>> =
        FxHashMap::with_capacity_and_hasher(inputs.global_symbols.len().saturating_mul(16), Default::default());
    let mut parent: Vec<usize> = (0..inputs.global_symbols.len()).collect();
    let mut rank = vec![0u8; inputs.global_symbols.len()];
    let mut accepted_edges: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut reject_reason_counts: FxHashMap<&'static str, usize> = FxHashMap::default();
    let mut accepted_edge_count = 0usize;
    let mut skipped_fragile_pairs = 0usize;

    for symbol_index in 0..inputs.global_symbols.len() {
        let key = bucket_keys[symbol_index];
        for_each_family_bucket_neighbor(key, |neighbor| {
            let Some(bucket) = bucket_map.get(&neighbor) else {
                return;
            };
            for &other_index in bucket {
                if other_index <= symbol_index {
                    continue;
                }
                if fragile_marks[symbol_index] || fragile_marks[other_index] {
                    skipped_fragile_pairs += 1;
                    continue;
                }

                let context_decision = inputs
                    .context_model
                    .map(|model| {
                        model.merge_decision(
                            symbol_index,
                            other_index,
                            inputs.config.sym_unify_context_mode,
                        )
                    })
                    .unwrap_or(ContextDecision::Unknown);
                if context_decision == ContextDecision::Reject {
                    *reject_reason_counts.entry("context").or_insert(0) += 1;
                    continue;
                }

                let Some(obs) = get_or_compute_pair(
                    &mut pair_cache,
                    &mut comparator,
                    inputs.global_symbols,
                    inputs.symbol_signatures,
                    inputs.symbol_pixel_counts,
                    symbol_index,
                    other_index,
                    class_max_err,
                    class_max_dx,
                    class_max_dy,
                ) else {
                    *reject_reason_counts.entry("compare").or_insert(0) += 1;
                    continue;
                };

                let accept = if context_decision == ContextDecision::Unknown {
                    obs.class_score <= class_accept_limit.saturating_sub(2)
                        && obs.result.outside_ink_err == 0
                } else {
                    obs.class_score <= class_accept_limit
                };
                if !accept {
                    *reject_reason_counts.entry("score").or_insert(0) += 1;
                    continue;
                }

                accepted_edge_count += 1;
                union(&mut parent, &mut rank, symbol_index, other_index);
                accepted_edges.entry(symbol_index).or_default().push(other_index);
                accepted_edges.entry(other_index).or_default().push(symbol_index);
            }
        });
    }

    let mut class_map: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for index in 0..inputs.global_symbols.len() {
        let root = find_root(&mut parent, index);
        class_map.entry(root).or_default().push(index);
    }

    let mut classes = Vec::new();
    let mut diagnostics = UnifyBuildDiagnostics::default();
    if inputs.collect_diagnostics {
        diagnostics.lines.push(format!(
            "sym_unify class build: symbols={} accepted_edges={} compare_rejects={} score_rejects={} context_rejects={} fragile_pair_splits={}",
            inputs.global_symbols.len(),
            accepted_edge_count,
            reject_reason_counts.get("compare").copied().unwrap_or(0),
            reject_reason_counts.get("score").copied().unwrap_or(0),
            reject_reason_counts.get("context").copied().unwrap_or(0),
            skipped_fragile_pairs
        ));
        diagnostics.lines.push(format!(
            "sym_unify fragile marks: count={}",
            fragile_marks.iter().filter(|&&mark| mark).count()
        ));
    }

    let mut grouped: Vec<Vec<usize>> = class_map.into_values().collect();
    grouped.sort_by(|lhs, rhs| rhs.len().cmp(&lhs.len()).then_with(|| lhs[0].cmp(&rhs[0])));

    let mut rejected_weak_core = 0usize;
    let mut rejected_low_gain = 0usize;
    let mut classes_with_fragile_members = 0usize;

    for members in grouped {
        let split_fragile_members = members.iter().filter(|&&index| fragile_marks[index]).count();
        if split_fragile_members > 0 {
            classes_with_fragile_members += 1;
        }
        let non_fragile_members: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&index| !fragile_marks[index])
            .collect();

        if non_fragile_members.len() < inputs.config.sym_unify_min_class_size.max(2) {
            continue;
        }
        let class_usage: usize = non_fragile_members
            .iter()
            .map(|&index| inputs.symbol_usage[index])
            .sum();
        let class_page_span = non_fragile_members
            .iter()
            .map(|&index| inputs.symbol_page_count[index])
            .max()
            .unwrap_or(0);
        if class_usage < inputs.config.sym_unify_min_class_usage
            || class_page_span < inputs.config.sym_unify_min_page_span
        {
            continue;
        }

        let member_set: FxHashSet<usize> = non_fragile_members.iter().copied().collect();
        let close_edges: FxHashMap<usize, Vec<usize>> = non_fragile_members
            .iter()
            .copied()
            .map(|member| {
                let neighbors = accepted_edges
                    .get(&member)
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|neighbor| member_set.contains(neighbor))
                    .filter(|neighbor| {
                        get_or_compute_pair(
                            &mut pair_cache,
                            &mut comparator,
                            inputs.global_symbols,
                            inputs.symbol_signatures,
                            inputs.symbol_pixel_counts,
                            member,
                            *neighbor,
                            class_max_err,
                            class_max_dx,
                            class_max_dy,
                        )
                        .is_some_and(|obs| obs.assignment_score <= close_threshold)
                    })
                    .collect();
                (member, neighbors)
            })
            .collect();

        let (dense_core, core_subcluster_count) =
            dense_core_component(&non_fragile_members, &close_edges);
        let core_ratio_permille =
            ((dense_core.len() * 1000) / non_fragile_members.len().max(1)) as u16;
        if dense_core.len() < 2
            || core_ratio_permille < inputs.config.sym_unify_min_core_ratio_permille
        {
            rejected_weak_core += 1;
            if inputs.collect_diagnostics {
                diagnostics.lines.push(format!(
                    "sym_unify skip weak-core: class_size={} non_fragile={} core_size={} core_ratio_permille={} sample={:?}",
                    members.len(),
                    non_fragile_members.len(),
                    dense_core.len(),
                    core_ratio_permille,
                    &non_fragile_members[..non_fragile_members.len().min(8)]
                ));
            }
            continue;
        }

        let Some((representative_index, representative_score)) = select_dense_representative(
            &dense_core,
            &mut pair_cache,
            &mut comparator,
            inputs.global_symbols,
            inputs.symbol_signatures,
            inputs.symbol_pixel_counts,
            inputs.symbol_usage,
            inputs.symbol_page_count,
            class_max_err,
            class_max_dx,
            class_max_dy,
            close_threshold,
        ) else {
            continue;
        };

        let mut core_members = Vec::new();
        for &member in &dense_core {
            if member == representative_index {
                continue;
            }
            let Some(obs) = get_or_compute_pair(
                &mut pair_cache,
                &mut comparator,
                inputs.global_symbols,
                inputs.symbol_signatures,
                inputs.symbol_pixel_counts,
                member,
                representative_index,
                class_max_err,
                class_max_dx,
                class_max_dy,
            ) else {
                continue;
            };
            if obs.assignment_score > class_accept_limit
                || obs.result.outside_ink_err > 0
                || obs.result.dx.abs() > class_max_dx
                || obs.result.dy.abs() > class_max_dy
            {
                continue;
            }
            core_members.push(UnifiedClassMember {
                member_index: member,
                dx: obs.result.dx,
                dy: obs.result.dy,
                score: obs.assignment_score,
            });
        }

        if core_members.is_empty() {
            rejected_weak_core += 1;
            if inputs.collect_diagnostics {
                diagnostics.lines.push(format!(
                    "sym_unify skip empty-remap: class_size={} core_size={} representative={}",
                    non_fragile_members.len(),
                    dense_core.len(),
                    representative_index
                ));
            }
            continue;
        }

        let core_member_set: FxHashSet<usize> = dense_core.iter().copied().collect();
        let non_core_members: Vec<usize> = non_fragile_members
            .iter()
            .copied()
            .filter(|index| !core_member_set.contains(index) && *index != representative_index)
            .collect();
        let triage = triage_non_core_members(
            &non_core_members,
            &close_edges,
            representative_index,
            inputs.symbol_usage[representative_index],
            inputs.symbol_page_count[representative_index],
            &mut pair_cache,
            &mut comparator,
            inputs.global_symbols,
            inputs.symbol_signatures,
            inputs.symbol_pixel_counts,
            inputs.symbol_usage,
            inputs.symbol_page_count,
            inputs.context_model,
            inputs.config.sym_unify_context_mode,
            class_max_err,
            class_max_dx,
            class_max_dy,
            border_accept_limit,
            inputs.config.sym_unify_max_border_outside_ink,
            inputs.config.sym_unify_min_class_usage,
            inputs.config.sym_unify_min_page_span,
        );
        let candidate_subclusters = core_subcluster_count + triage.candidate_subclusters;
        let total_usage: usize = non_fragile_members
            .iter()
            .map(|&index| inputs.symbol_usage[index])
            .sum();
        let page_span = non_fragile_members
            .iter()
            .map(|&index| inputs.symbol_page_count[index])
            .max()
            .unwrap_or(1);
        let gain = estimate_class_gain(
            inputs.global_symbols,
            inputs.symbol_usage,
            inputs.symbol_page_count,
            representative_index,
            &core_members,
            &triage.border_members,
            triage.retained_border_members,
            triage.retained_outlier_members,
            representative_score,
        );
        if gain.net_gain < inputs.config.sym_unify_min_estimated_gain {
            rejected_low_gain += 1;
            if inputs.collect_diagnostics {
                diagnostics.lines.push(format!(
                    "sym_unify skip low-gain: representative={} class_size={} core_size={} gain={} bitmap={} ids={} rep_penalty={} retained_penalty={} border_unified={} retained_border={} retained_outliers={} split_fragile={}",
                    representative_index,
                    non_fragile_members.len(),
                    dense_core.len(),
                    gain.net_gain,
                    gain.bitmap_savings,
                    gain.id_savings,
                    gain.representative_penalty,
                    gain.retained_penalty,
                    triage.border_members.len(),
                    triage.retained_border_members,
                    triage.retained_outlier_members,
                    split_fragile_members
                ));
            }
            continue;
        }

        if inputs.collect_diagnostics {
            diagnostics.lines.push(format!(
                "sym_unify class: representative={} class_size={} core_size={} unified={} border_unified={} retained_border={} retained_outliers={} split_fragile={} total_usage={} page_span={} rep_usage={} rep_pages={} rep_score={} gain={} bitmap={} ids={} rep_penalty={} retained_penalty={} subclusters={}",
                representative_index,
                non_fragile_members.len(),
                dense_core.len(),
                core_members.len(),
                triage.border_members.len(),
                triage.retained_border_members,
                triage.retained_outlier_members,
                split_fragile_members,
                total_usage,
                page_span,
                inputs.symbol_usage[representative_index],
                inputs.symbol_page_count[representative_index],
                representative_score,
                gain.net_gain,
                gain.bitmap_savings,
                gain.id_savings,
                gain.representative_penalty,
                gain.retained_penalty,
                candidate_subclusters
            ));
        }

        classes.push(UnifiedClass {
            representative_index,
            core_members,
            border_members: triage.border_members,
            class_size: non_fragile_members.len(),
            dense_core_size: dense_core.len(),
            total_usage,
            page_span,
            representative_score,
            retained_border_members: triage.retained_border_members,
            retained_outlier_members: triage.retained_outlier_members,
            candidate_subclusters,
            split_fragile_members,
            estimated_gain: gain.net_gain,
        });
    }

    let unified_members: usize = classes.iter().map(|class| class.core_members.len()).sum();
    let border_unified_members: usize = classes.iter().map(|class| class.border_members.len()).sum();
    if inputs.collect_diagnostics {
        diagnostics.lines.push(format!(
            "sym_unify summary: classes={} unified_members={} border_unified_members={} retained_border_members={} retained_outlier_members={} rejected_weak_core={} rejected_low_gain={} classes_with_fragile_members={}",
            classes.len(),
            unified_members,
            border_unified_members,
            classes
                .iter()
                .map(|class| class.retained_border_members)
                .sum::<usize>(),
            classes
                .iter()
                .map(|class| class.retained_outlier_members)
                .sum::<usize>(),
            rejected_weak_core,
            rejected_low_gain,
            classes_with_fragile_members
        ));
    }

    (classes, diagnostics)
}
