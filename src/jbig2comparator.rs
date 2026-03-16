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
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

#[derive(Debug, Clone, Copy, Default)]
pub struct CollapseCompareLimits {
    pub outside_limit: u32,
    pub row_limit: u32,
    pub col_limit: u32,
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
        overlap_err: 6,
        outside_ink_err: 3,
        row_profile_err: 2,
        col_profile_err: 2,
        black_delta: 1,
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
        self.best_alignment_result(a, b, max_err)
    }

    pub fn compare_overlap_only(
        &mut self,
        a: &BitImage,
        b: &BitImage,
        max_err: u32,
    ) -> Option<CompareResult> {
        if a == b {
            return Some(CompareResult {
                dx: 0,
                dy: 0,
                black_delta: 0,
                ..Default::default()
            });
        }

        if a.width.abs_diff(b.width) > MAX_DIMENSION_DELTA
            || a.height.abs_diff(b.height) > MAX_DIMENSION_DELTA
        {
            return None;
        }

        let (overlap_err, dx, dy) = self.best_alignment_by_xor(a, b, max_err)?;
        Some(CompareResult {
            total_err: overlap_err,
            overlap_err,
            black_delta: a.count_ones().abs_diff(b.count_ones()) as u32,
            dx,
            dy,
            ..Default::default()
        })
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

    pub fn compare_for_refine_family(
        &mut self,
        a: &BitImage,
        b: &BitImage,
        max_err: u32,
        max_dx: i32,
        max_dy: i32,
    ) -> Option<CompareResult> {
        let (overlap_err, dx, dy) = self.best_alignment_result(a, b, max_err)?;
        let result = CompareResult {
            total_err: overlap_err,
            overlap_err,
            dx,
            dy,
            ..Default::default()
        };
        if result.dx.abs() > max_dx || result.dy.abs() > max_dy {
            return None;
        }
        Some(result)
    }

    #[inline]
    fn best_alignment_result(
        &mut self,
        a: &BitImage,
        b: &BitImage,
        max_err: u32,
    ) -> Option<(u32, i32, i32)> {
        if a == b {
            return Some((0, 0, 0));
        }

        if a.width.abs_diff(b.width) > MAX_DIMENSION_DELTA
            || a.height.abs_diff(b.height) > MAX_DIMENSION_DELTA
        {
            return None;
        }

        self.best_alignment_by_xor(a, b, max_err)
    }

    pub fn compare_for_collapse_family(
        &mut self,
        a: &BitImage,
        b: &BitImage,
        max_err: u32,
        max_dx: i32,
        max_dy: i32,
    ) -> Option<CompareResult> {
        let result = self.compare_detailed(a, b, max_err)?;
        if result.dx.abs() > max_dx || result.dy.abs() > max_dy {
            return None;
        }

        let limits = Self::collapse_compare_limits(&result);

        if result.outside_ink_err > limits.outside_limit
            || result.row_profile_err > limits.row_limit
            || result.col_profile_err > limits.col_limit
        {
            return None;
        }

        Some(result)
    }

    pub fn collapse_compare_limits(result: &CompareResult) -> CollapseCompareLimits {
        let shift_slack = (result.dx.abs() + result.dy.abs()) as u32;
        CollapseCompareLimits {
            outside_limit: (result.overlap_err / 2).max(2)
                + result.black_delta.min(4)
                + shift_slack,
            row_limit: result.overlap_err.saturating_mul(6).saturating_add(24),
            col_limit: result.overlap_err.saturating_mul(6).saturating_add(24),
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
        let has_popcnt = popcnt_available();
        let has_avx2 = avx2_popcnt_enabled();

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
                let bit_shift = (dx & 31) as u32;
                let word_shift = (dx >> 5) as isize;
                let mut err = 0u32;
                let mut early_break = false;

                for row in 0..overlap_height {
                    let a_row = unsafe { a_words.as_ptr().add((top + row) * awpr) };
                    let b_row = unsafe { b_words.as_ptr().add((b_row_start + row) * bwpr) };

                    let (new_err, broke) = unsafe {
                        if bit_shift == 0 {
                            row_kernel_noshift(
                                a_row, awpr, b_row, cols_words, word_shift, err, best_err, max_err,
                                has_popcnt, has_avx2,
                            )
                        } else {
                            row_kernel_shift(
                                a_row, awpr, b_row, cols_words, word_shift, bit_shift, err,
                                best_err, max_err, has_popcnt, has_avx2,
                            )
                        }
                    };
                    err = new_err;
                    if broke {
                        early_break = true;
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

#[inline]
fn popcnt_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("popcnt")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn avx2_popcnt_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn avx2_popcnt_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        avx2_popcnt_available() && std::env::var("JBIG2_USE_AVX2").is_ok_and(|v| v == "1")
    })
}

#[inline(always)]
unsafe fn load_word_or_zero(base: *const u32, idx: isize, len: usize) -> u32 {
    if idx < 0 || (idx as usize) >= len {
        0
    } else {
        unsafe { *base.add(idx as usize) }
    }
}

#[inline(always)]
unsafe fn row_kernel_noshift(
    a_row: *const u32,
    a_len: usize,
    b_row: *const u32,
    cols_words: usize,
    word_shift: isize,
    err: u32,
    best_err: u32,
    max_err: u32,
    has_popcnt: bool,
    has_avx2: bool,
) -> (u32, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2 && cols_words >= 16 && word_shift >= 0 {
            return unsafe {
                row_kernel_noshift_avx2(
                    a_row, a_len, b_row, cols_words, word_shift, err, best_err, max_err,
                )
            };
        }
        if has_popcnt {
            return unsafe {
                row_kernel_noshift_scalar_popcnt(
                    a_row, a_len, b_row, cols_words, word_shift, err, best_err, max_err,
                )
            };
        }
    }

    unsafe { row_kernel_noshift_scalar(a_row, a_len, b_row, cols_words, word_shift, err, best_err, max_err) }
}

#[inline(always)]
unsafe fn row_kernel_shift(
    a_row: *const u32,
    a_len: usize,
    b_row: *const u32,
    cols_words: usize,
    word_shift: isize,
    bit_shift: u32,
    err: u32,
    best_err: u32,
    max_err: u32,
    has_popcnt: bool,
    has_avx2: bool,
) -> (u32, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_popcnt {
            return unsafe {
                row_kernel_shift_scalar_popcnt(
                    a_row, a_len, b_row, cols_words, word_shift, bit_shift, err, best_err,
                    max_err,
                )
            };
        }
    }

    unsafe {
        row_kernel_shift_scalar(
            a_row, a_len, b_row, cols_words, word_shift, bit_shift, err, best_err, max_err,
        )
    }
}

#[inline(always)]
unsafe fn row_kernel_noshift_scalar(
    a_row: *const u32,
    a_len: usize,
    b_row: *const u32,
    cols_words: usize,
    word_shift: isize,
    mut err: u32,
    best_err: u32,
    max_err: u32,
) -> (u32, bool) {
    for w in 0..cols_words {
        let a_idx = w as isize + word_shift;
        let aw = unsafe { load_word_or_zero(a_row, a_idx, a_len) };
        let bw = unsafe { *b_row.add(w) };
        err += (aw ^ bw).count_ones();
        if err >= best_err || err > max_err {
            return (err, true);
        }
    }
    (err, false)
}

#[inline(always)]
unsafe fn row_kernel_shift_scalar(
    a_row: *const u32,
    a_len: usize,
    b_row: *const u32,
    cols_words: usize,
    word_shift: isize,
    bit_shift: u32,
    mut err: u32,
    best_err: u32,
    max_err: u32,
) -> (u32, bool) {
    let rshift = 32 - bit_shift;
    for w in 0..cols_words {
        let a_idx = w as isize + word_shift;
        let aw = unsafe { load_word_or_zero(a_row, a_idx, a_len) };
        let aw_next = unsafe { load_word_or_zero(a_row, a_idx + 1, a_len) };
        let aligned_a = (aw << bit_shift) | (aw_next >> rshift);
        let bw = unsafe { *b_row.add(w) };
        err += (aligned_a ^ bw).count_ones();
        if err >= best_err || err > max_err {
            return (err, true);
        }
    }
    (err, false)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn row_kernel_noshift_scalar_popcnt(
    a_row: *const u32,
    a_len: usize,
    b_row: *const u32,
    cols_words: usize,
    word_shift: isize,
    err: u32,
    best_err: u32,
    max_err: u32,
) -> (u32, bool) {
    unsafe {
        row_kernel_noshift_scalar(
            a_row, a_len, b_row, cols_words, word_shift, err, best_err, max_err,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn row_kernel_shift_scalar_popcnt(
    a_row: *const u32,
    a_len: usize,
    b_row: *const u32,
    cols_words: usize,
    word_shift: isize,
    bit_shift: u32,
    err: u32,
    best_err: u32,
    max_err: u32,
) -> (u32, bool) {
    unsafe {
        row_kernel_shift_scalar(
            a_row, a_len, b_row, cols_words, word_shift, bit_shift, err, best_err, max_err,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "popcnt")]
unsafe fn row_kernel_noshift_avx2(
    a_row: *const u32,
    a_len: usize,
    b_row: *const u32,
    cols_words: usize,
    word_shift: isize,
    mut err: u32,
    best_err: u32,
    max_err: u32,
) -> (u32, bool) {
    let mut w = 0usize;
    while w < cols_words {
        let a_idx = w as isize + word_shift;
        if a_idx < 0 || (a_idx as usize + 8) > a_len || w + 8 > cols_words {
            break;
        }

        let av = unsafe { _mm256_loadu_si256(a_row.add(a_idx as usize) as *const __m256i) };
        let bv = unsafe { _mm256_loadu_si256(b_row.add(w) as *const __m256i) };
        let xv = _mm256_xor_si256(av, bv);
        let mut lanes = [0u64; 4];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, xv) };
        err = err.saturating_add(lanes.iter().map(|lane| lane.count_ones()).sum::<u32>());
        if err >= best_err || err > max_err {
            return (err, true);
        }
        w += 8;
    }

    unsafe {
        row_kernel_noshift_scalar(
            a_row,
            a_len,
            b_row.add(w),
            cols_words - w,
            word_shift + w as isize,
            err,
            best_err,
            max_err,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "popcnt")]
unsafe fn row_kernel_shift_avx2(
    a_row: *const u32,
    a_len: usize,
    b_row: *const u32,
    cols_words: usize,
    word_shift: isize,
    bit_shift: u32,
    mut err: u32,
    best_err: u32,
    max_err: u32,
) -> (u32, bool) {
    let rshift = 32 - bit_shift;
    let lshift_vec = _mm256_set1_epi32(bit_shift as i32);
    let rshift_vec = _mm256_set1_epi32(rshift as i32);
    let mut w = 0usize;
    while w < cols_words {
        let a_idx = w as isize + word_shift;
        if a_idx < 0 || (a_idx as usize + 9) > a_len || w + 8 > cols_words {
            break;
        }

        let av = unsafe { _mm256_loadu_si256(a_row.add(a_idx as usize) as *const __m256i) };
        let av_next = unsafe { _mm256_loadu_si256(a_row.add(a_idx as usize + 1) as *const __m256i) };
        let aligned = _mm256_or_si256(
            _mm256_sllv_epi32(av, lshift_vec),
            _mm256_srlv_epi32(av_next, rshift_vec),
        );
        let bv = unsafe { _mm256_loadu_si256(b_row.add(w) as *const __m256i) };
        let xv = _mm256_xor_si256(aligned, bv);
        let mut lanes = [0u64; 4];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, xv) };
        err = err.saturating_add(lanes.iter().map(|lane| lane.count_ones()).sum::<u32>());
        if err >= best_err || err > max_err {
            return (err, true);
        }
        w += 8;
    }

    unsafe {
        row_kernel_shift_scalar(
            a_row,
            a_len,
            b_row.add(w),
            cols_words - w,
            word_shift + w as isize,
            bit_shift,
            err,
            best_err,
            max_err,
        )
    }
}
