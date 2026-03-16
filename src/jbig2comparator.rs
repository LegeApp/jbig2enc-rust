// "The comparator's performance is critical; rewriting it to pixel-by-pixel is unacceptable.
// Fix bugs without changing the high-level algorithm."

//! jbig2comparator.rs
//! ==========================================================
//! Stage 1 keeps the existing word-wise shifted XOR/popcount
//! search to find the best alignment quickly.
//!
//! Stage 2 runs only once for that best shift and extracts
//! richer glyph metrics so family matching can distinguish
//! structural mismatch from harmless edge noise.
//! ==========================================================

use crate::jbig2sym::BitImage;

/// Maximum absolute shift (in pixels) that we search in x/y.
const SEARCH_RADIUS: i32 = 5;
/// Maximum width/height delta that can still produce a match.
pub const MAX_DIMENSION_DELTA: usize = (SEARCH_RADIUS as usize) * 2;

#[derive(Debug, Clone, Copy, Default)]
pub struct CompareResult {
    pub total_err: u32,
    pub overlap_err: u32,
    pub outside_ink_err: u32,
    pub row_profile_err: u32,
    pub col_profile_err: u32,
    pub black_delta: u32,
    pub dx: i32,
    pub dy: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct CompareWeights {
    pub overlap_err: u32,
    pub outside_ink_err: u32,
    pub row_profile_err: u32,
    pub col_profile_err: u32,
    pub black_delta: u32,
}

impl CompareWeights {
    pub const LIVE_MATCH: Self = Self {
        overlap_err: 8,
        outside_ink_err: 5,
        row_profile_err: 2,
        col_profile_err: 2,
        black_delta: 1,
    };

    pub const COLLAPSE: Self = Self {
        overlap_err: 5,
        outside_ink_err: 8,
        row_profile_err: 5,
        col_profile_err: 5,
        black_delta: 2,
    };

    pub const REFINE: Self = Self {
        overlap_err: 6,
        outside_ink_err: 4,
        row_profile_err: 2,
        col_profile_err: 2,
        black_delta: 1,
    };

    #[inline]
    pub fn score(&self, r: &CompareResult) -> u32 {
        self.overlap_err.saturating_mul(r.overlap_err)
            .saturating_add(self.outside_ink_err.saturating_mul(r.outside_ink_err))
            .saturating_add(self.row_profile_err.saturating_mul(r.row_profile_err))
            .saturating_add(self.col_profile_err.saturating_mul(r.col_profile_err))
            .saturating_add(self.black_delta.saturating_mul(r.black_delta))
    }
}

#[derive(Default)]
pub struct Comparator {
    tmp: Vec<u32>,
    row_a: Vec<u32>,
    row_b: Vec<u32>,
    col_a: Vec<u32>,
    col_b: Vec<u32>,
}

impl Comparator {
    pub fn distance(
        &mut self,
        a: &BitImage,
        b: &BitImage,
        max_err: u32,
    ) -> Option<(u32, i32, i32)> {
        let result = self.compare_detailed(a, b, max_err)?;
        Some((result.total_err, result.dx, result.dy))
    }

    pub fn compare_detailed(
        &mut self,
        a: &BitImage,
        b: &BitImage,
        max_err: u32,
    ) -> Option<CompareResult> {
        if a == b {
            return Some(CompareResult {
                dx: 0,
                dy: 0,
                ..Default::default()
            });
        }

        if a.width.abs_diff(b.width) > MAX_DIMENSION_DELTA
            || a.height.abs_diff(b.height) > MAX_DIMENSION_DELTA
        {
            return None;
        }

        let (overlap_err, dx, dy) = self.best_alignment_by_xor(a, b, max_err)?;
        let mut result = self.metrics_for_alignment(a, b, dx, dy);
        result.overlap_err = overlap_err.max(result.overlap_err);
        result.total_err = result.overlap_err.saturating_add(result.outside_ink_err);

        if result.total_err <= max_err {
            Some(result)
        } else {
            None
        }
    }

    fn best_alignment_by_xor(
        &mut self,
        a: &BitImage,
        b: &BitImage,
        max_err: u32,
    ) -> Option<(u32, i32, i32)> {
        let awpr = (a.width + 31) >> 5;
        let bwpr = (b.width + 31) >> 5;
        let wpr_overlap = (a.width.max(b.width) + 31) >> 5;
        if self.tmp.len() < wpr_overlap {
            self.tmp.resize(wpr_overlap, 0);
        }

        let a_words = a.packed_words();
        let b_words = b.packed_words();

        let mut best_err = max_err + 1;
        let mut best_dx = 0;
        let mut best_dy = 0;

        for dy in -SEARCH_RADIUS..=SEARCH_RADIUS {
            let top_i = dy.max(0);
            let bottom_i = (dy + b.height as i32).min(a.height as i32);
            if top_i >= bottom_i {
                continue;
            }
            let top = top_i as usize;
            let bottom = bottom_i as usize;
            let overlap_height = bottom - top;
            let b_row_start = (top as i32 - dy) as usize;

            for dx in -SEARCH_RADIUS..=SEARCH_RADIUS {
                let left_i = dx.max(0);
                let right_i = (dx + b.width as i32).min(a.width as i32);
                if left_i >= right_i {
                    continue;
                }

                let overlap_width = (right_i - left_i) as usize;
                let cols_words = (overlap_width + 31) >> 5;
                let bit_shift = (dx & 31) as usize;
                let word_shift = dx >> 5;
                let mut err = 0u32;
                let mut early_break = false;

                for row in 0..overlap_height {
                    let a_slice = &a_words[(top + row) * awpr..(top + row + 1) * awpr];
                    let b_slice =
                        &b_words[(b_row_start + row) * bwpr..(b_row_start + row + 1) * bwpr];

                    if bit_shift == 0 {
                        for w in 0..cols_words {
                            let a_idx = w as i32 + word_shift;
                            let aw = if a_idx < 0 {
                                0
                            } else {
                                a_slice.get(a_idx as usize).copied().unwrap_or(0)
                            };
                            err += (aw ^ b_slice[w]).count_ones();
                            if err >= best_err || err > max_err {
                                early_break = true;
                                break;
                            }
                        }
                    } else {
                        let bit_shift = bit_shift as u32;
                        for w in 0..cols_words {
                            let a_idx = w as i32 + word_shift;
                            let aw = if a_idx < 0 {
                                0
                            } else {
                                a_slice.get(a_idx as usize).copied().unwrap_or(0)
                            };
                            let aw_next = if a_idx + 1 < 0 {
                                0
                            } else {
                                a_slice.get((a_idx + 1) as usize).copied().unwrap_or(0)
                            };
                            let aligned_a = (aw << bit_shift) | (aw_next >> (32 - bit_shift));
                            err += (aligned_a ^ b_slice[w]).count_ones();
                            if err >= best_err || err > max_err {
                                early_break = true;
                                break;
                            }
                        }
                    }

                    if early_break {
                        break;
                    }
                }

                if err < best_err {
                    best_err = err;
                    best_dx = dx;
                    best_dy = dy;
                } else if err == best_err && !early_break {
                    let curr_dist = dx.abs() + dy.abs();
                    let best_dist = best_dx.abs() + best_dy.abs();
                    if curr_dist < best_dist {
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }
        }

        if best_err <= max_err {
            Some((best_err, best_dx, best_dy))
        } else {
            None
        }
    }

    fn ensure_profile_buffers(&mut self, rows: usize, cols: usize) {
        if self.row_a.len() < rows {
            self.row_a.resize(rows, 0);
            self.row_b.resize(rows, 0);
        }
        if self.col_a.len() < cols {
            self.col_a.resize(cols, 0);
            self.col_b.resize(cols, 0);
        }
    }

    fn metrics_for_alignment(
        &mut self,
        a: &BitImage,
        b: &BitImage,
        dx: i32,
        dy: i32,
    ) -> CompareResult {
        let min_x = 0.min(dx);
        let min_y = 0.min(dy);
        let max_x = (a.width as i32).max(dx + b.width as i32);
        let max_y = (a.height as i32).max(dy + b.height as i32);
        let used_rows = (max_y - min_y).max(0) as usize;
        let used_cols = (max_x - min_x).max(0) as usize;

        self.ensure_profile_buffers(used_rows, used_cols);
        self.row_a[..used_rows].fill(0);
        self.row_b[..used_rows].fill(0);
        self.col_a[..used_cols].fill(0);
        self.col_b[..used_cols].fill(0);

        let mut overlap_err = 0u32;
        let mut outside_ink_err = 0u32;

        for gy in min_y..max_y {
            for gx in min_x..max_x {
                let ax = gx;
                let ay = gy;
                let bx = gx - dx;
                let by = gy - dy;

                let in_a = ax >= 0
                    && ay >= 0
                    && (ax as usize) < a.width
                    && (ay as usize) < a.height;
                let in_b = bx >= 0
                    && by >= 0
                    && (bx as usize) < b.width
                    && (by as usize) < b.height;
                let a_on = in_a && a.get_usize(ax as usize, ay as usize);
                let b_on = in_b && b.get_usize(bx as usize, by as usize);

                if !(a_on || b_on) {
                    continue;
                }

                let ry = (gy - min_y) as usize;
                let rx = (gx - min_x) as usize;
                if a_on {
                    self.row_a[ry] += 1;
                    self.col_a[rx] += 1;
                }
                if b_on {
                    self.row_b[ry] += 1;
                    self.col_b[rx] += 1;
                }

                if a_on != b_on {
                    if in_a && in_b {
                        overlap_err += 1;
                    } else {
                        outside_ink_err += 1;
                    }
                }
            }
        }

        let mut row_profile_err = 0u32;
        let mut col_profile_err = 0u32;
        for i in 0..used_rows {
            row_profile_err += self.row_a[i].abs_diff(self.row_b[i]);
        }
        for i in 0..used_cols {
            col_profile_err += self.col_a[i].abs_diff(self.col_b[i]);
        }

        CompareResult {
            total_err: overlap_err.saturating_add(outside_ink_err),
            overlap_err,
            outside_ink_err,
            row_profile_err,
            col_profile_err,
            black_delta: a.count_ones().abs_diff(b.count_ones()) as u32,
            dx,
            dy,
        }
    }
}
