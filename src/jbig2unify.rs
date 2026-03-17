use crate::jbig2collapse::{
    FamilyBucketKey, SymbolSignature, family_bucket_key_for_symbol, family_signatures_are_compatible,
    for_each_family_bucket_neighbor,
};
use crate::jbig2collapse_context::{CollapseContextModel, ContextDecision};
use crate::jbig2comparator::{Comparator, CompareResult};
use crate::jbig2structs::Jbig2Config;
use crate::jbig2sym::BitImage;
use rustc_hash::FxHashMap;

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
    pub class_size: usize,
    pub dense_core_size: usize,
    pub total_usage: usize,
    pub page_span: usize,
    pub representative_score: u64,
    pub retained_border_members: usize,
    pub candidate_subclusters: usize,
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
}

#[derive(Debug, Clone, Copy)]
struct PairObservation {
    result: CompareResult,
    class_score: u32,
    assignment_score: u32,
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

fn denoise_symbol(symbol: &BitImage) -> BitImage {
    if symbol.width < 3 || symbol.height < 3 {
        return symbol.clone();
    }

    let mut cleaned = symbol.clone();
    let mut updates = Vec::new();
    for y in 0..symbol.height {
        for x in 0..symbol.width {
            if !symbol.get_usize(x, y) {
                continue;
            }
            let mut neighbors = 0u8;
            let x0 = x.saturating_sub(1);
            let y0 = y.saturating_sub(1);
            let x1 = (x + 1).min(symbol.width - 1);
            let y1 = (y + 1).min(symbol.height - 1);
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    if nx == x && ny == y {
                        continue;
                    }
                    if symbol.get_usize(nx, ny) {
                        neighbors += 1;
                    }
                }
            }
            if neighbors <= 1 {
                updates.push((x, y, false));
            }
        }
    }

    for (x, y, value) in updates {
        cleaned.set_usize(x, y, value);
    }
    let (_, trimmed) = cleaned.trim();
    trimmed
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
    normalized_symbols: &[BitImage],
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
        if normalized_symbols[lo].width.abs_diff(normalized_symbols[hi].width) > 1
            || normalized_symbols[lo]
                .height
                .abs_diff(normalized_symbols[hi].height)
                > 1
        {
            pair_cache.insert(key, None);
            return None;
        }
        if !family_signatures_are_compatible(
            signatures[lo],
            signatures[hi],
            black_counts[lo],
            black_counts[hi],
        ) {
            pair_cache.insert(key, None);
            return None;
        }

        let result = comparator.compare_for_symbol_unify(
            &normalized_symbols[lo],
            &normalized_symbols[hi],
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

fn dense_core_component(
    members: &[usize],
    close_edges: &FxHashMap<usize, Vec<usize>>,
) -> (Vec<usize>, usize) {
    if members.len() < 2 {
        return (Vec::new(), 0);
    }

    let min_pts = members.len().div_ceil(3).max(2);
    let core_nodes: Vec<usize> = members
        .iter()
        .copied()
        .filter(|member| close_edges.get(member).map_or(0, Vec::len) + 1 >= min_pts)
        .collect();
    if core_nodes.len() < 2 {
        return (Vec::new(), 0);
    }

    let core_set: std::collections::HashSet<usize> = core_nodes.iter().copied().collect();
    let mut seen = std::collections::HashSet::new();
    let mut components = Vec::new();

    for start in &core_nodes {
        if !seen.insert(*start) {
            continue;
        }
        let mut stack = vec![*start];
        let mut component = vec![*start];
        while let Some(node) = stack.pop() {
            if let Some(neighbors) = close_edges.get(&node) {
                for &neighbor in neighbors {
                    if !core_set.contains(&neighbor) || !seen.insert(neighbor) {
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
    let candidate_subclusters = components.len().saturating_sub(1);
    (components.into_iter().next().unwrap_or_default(), candidate_subclusters)
}

fn select_dense_representative(
    core: &[usize],
    pair_cache: &mut FxHashMap<u64, Option<PairObservation>>,
    comparator: &mut Comparator,
    normalized_symbols: &[BitImage],
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
                normalized_symbols,
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

    let class_max_err = inputs.config.lossy_collapse_max_err.min(12).max(4);
    let class_max_dx = inputs.config.lossy_collapse_max_dx.min(1);
    let class_max_dy = inputs.config.lossy_collapse_max_dy.min(1);
    let class_accept_limit = class_max_err.saturating_add(4);
    let close_threshold = class_max_err.saturating_div(2).saturating_add(3);

    let normalized_symbols: Vec<BitImage> = inputs
        .global_symbols
        .iter()
        .map(denoise_symbol)
        .collect();
    let normalized_signatures: Vec<SymbolSignature> = normalized_symbols
        .iter()
        .map(crate::jbig2collapse::compute_symbol_signature)
        .collect();
    let bucket_keys: Vec<FamilyBucketKey> = normalized_symbols
        .iter()
        .zip(normalized_signatures.iter())
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

                let context_decision = inputs
                    .context_model
                    .map(|model| {
                        model.merge_decision(
                            symbol_index,
                            other_index,
                            inputs.config.lossy_collapse_context_mode,
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
                    &normalized_symbols,
                    &normalized_signatures,
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
    diagnostics.lines.push(format!(
        "sym_unify class build: symbols={} accepted_edges={} compare_rejects={} score_rejects={} context_rejects={}",
        inputs.global_symbols.len(),
        accepted_edge_count,
        reject_reason_counts.get("compare").copied().unwrap_or(0),
        reject_reason_counts.get("score").copied().unwrap_or(0),
        reject_reason_counts.get("context").copied().unwrap_or(0)
    ));

    let mut grouped: Vec<Vec<usize>> = class_map.into_values().collect();
    grouped.sort_by(|lhs, rhs| rhs.len().cmp(&lhs.len()).then_with(|| lhs[0].cmp(&rhs[0])));

    for members in grouped {
        if members.len() < inputs.config.lossy_collapse_min_family_size.max(2) {
            continue;
        }

        let close_edges: FxHashMap<usize, Vec<usize>> = members
            .iter()
            .copied()
            .map(|member| {
                let neighbors = accepted_edges
                    .get(&member)
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|neighbor| members.contains(neighbor))
                    .filter(|neighbor| {
                        get_or_compute_pair(
                            &mut pair_cache,
                            &mut comparator,
                            &normalized_symbols,
                            &normalized_signatures,
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

        let (dense_core, candidate_subclusters) = dense_core_component(&members, &close_edges);
        if dense_core.len() < 2 {
            diagnostics.lines.push(format!(
                "sym_unify skip no-dense-core: class_size={} sample={:?}",
                members.len(),
                &members[..members.len().min(8)]
            ));
            continue;
        }

        let Some((representative_index, representative_score)) = select_dense_representative(
            &dense_core,
            &mut pair_cache,
            &mut comparator,
            &normalized_symbols,
            &normalized_signatures,
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
                &normalized_symbols,
                &normalized_signatures,
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
            diagnostics.lines.push(format!(
                "sym_unify skip empty-remap: class_size={} core_size={} representative={}",
                members.len(),
                dense_core.len(),
                representative_index
            ));
            continue;
        }

        let total_usage: usize = members.iter().map(|&index| inputs.symbol_usage[index]).sum();
        let page_span = members
            .iter()
            .map(|&index| inputs.symbol_page_count[index])
            .max()
            .unwrap_or(1);
        let retained_border_members = members
            .len()
            .saturating_sub(core_members.len())
            .saturating_sub(1);

        diagnostics.lines.push(format!(
            "sym_unify class: representative={} class_size={} core_size={} unified={} retained_border={} total_usage={} page_span={} rep_usage={} rep_pages={} rep_score={} subclusters={}",
            representative_index,
            members.len(),
            dense_core.len(),
            core_members.len(),
            retained_border_members,
            total_usage,
            page_span,
            inputs.symbol_usage[representative_index],
            inputs.symbol_page_count[representative_index],
            representative_score,
            candidate_subclusters
        ));

        classes.push(UnifiedClass {
            representative_index,
            core_members,
            class_size: members.len(),
            dense_core_size: dense_core.len(),
            total_usage,
            page_span,
            representative_score,
            retained_border_members,
            candidate_subclusters,
        });
    }

    let unified_members: usize = classes.iter().map(|class| class.core_members.len()).sum();
    diagnostics.lines.push(format!(
        "sym_unify summary: classes={} unified_members={} retained_border_members={}",
        classes.len(),
        unified_members,
        classes
            .iter()
            .map(|class| class.retained_border_members)
            .sum::<usize>()
    ));

    (classes, diagnostics)
}
