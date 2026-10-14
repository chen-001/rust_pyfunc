// sandbox_xss070：numpy x86-simd-sort argsort(float64) 的标量复刻，先在沙箱里和 numpy 对拍。
//
// 结构逐行搬自 src/okay_database/hm039.rs（那份已对 int64 验证过），只把
//   · 键从 i64 换成 f64（比较、最大值哨兵都要换）
//   · 深度耗尽时的兜底从「按值稳定排序」换成 libstdc++ 的 std::sort（introsort）
// 两处改完后已对 170 组随机数组（含全相等、无并列、各种尺寸）与 numpy 2.3.5 完全一致。

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ============================================================================
// numpy x86-simd-sort argsort(float64) 的逐行复刻
// ============================================================================

const LANES: usize = 8;
const KEY_MAX: f64 = f64::MAX;
const IDX_MAX: i64 = i64::MAX;

#[inline]
fn swap_perm(scale: usize) -> [usize; LANES] {
    match scale {
        2 => [1, 0, 3, 2, 5, 4, 7, 6],
        4 => [2, 3, 0, 1, 6, 7, 4, 5],
        _ => [4, 5, 6, 7, 0, 1, 2, 3],
    }
}

#[inline]
fn reverse_perm(scale: usize) -> [usize; LANES] {
    match scale {
        2 => [1, 0, 3, 2, 5, 4, 7, 6],
        4 => [3, 2, 1, 0, 7, 6, 5, 4],
        _ => [7, 6, 5, 4, 3, 2, 1, 0],
    }
}

#[inline]
fn perm_k(k: &[f64; LANES], p: &[usize; LANES]) -> [f64; LANES] {
    let mut o = [0.0f64; LANES];
    for l in 0..LANES {
        o[l] = k[p[l]];
    }
    o
}

#[inline]
fn perm_i(i: &[i64; LANES], p: &[usize; LANES]) -> [i64; LANES] {
    let mut o = [0i64; LANES];
    for l in 0..LANES {
        o[l] = i[p[l]];
    }
    o
}

fn cmp_merge_k(k1: &[f64; LANES], k2: &[f64; LANES], mask: u8) -> [f64; LANES] {
    let mut o = [0.0f64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        o[l] = if (mask >> l) & 1 == 1 { hi } else { lo };
    }
    o
}

/// 带下标的 cmp_merge（xss-network-keyvaluesort.hpp）
fn cmp_merge_ki(
    k1: &[f64; LANES],
    i1: &[i64; LANES],
    k2: &[f64; LANES],
    i2: &[i64; LANES],
    mask: u8,
) -> ([f64; LANES], [i64; LANES]) {
    let mut ko = [0.0f64; LANES];
    let mut io = [0i64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        ko[l] = if (mask >> l) & 1 == 1 { hi } else { lo };
        // 结果来自 in1 就保留下标 1，否则用下标 2
        io[l] = if ko[l] == a { i1[l] } else { i2[l] };
    }
    (ko, io)
}

/// COEX：稳定的比较交换
fn coex(
    k1: &[f64; LANES],
    i1: &[i64; LANES],
    k2: &[f64; LANES],
    i2: &[i64; LANES],
) -> ([f64; LANES], [i64; LANES], [f64; LANES], [i64; LANES]) {
    let mut ko1 = [0.0f64; LANES];
    let mut ko2 = [0.0f64; LANES];
    let mut io1 = [0i64; LANES];
    let mut io2 = [0i64; LANES];
    for l in 0..LANES {
        let (a, b) = (k1[l], k2[l]);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        ko1[l] = lo;
        ko2[l] = hi;
        let eq = lo == a;
        io1[l] = if eq { i1[l] } else { i2[l] };
        io2[l] = if eq { i2[l] } else { i1[l] };
    }
    (ko1, io1, ko2, io2)
}

fn sort_reg_8lanes(k: &[f64; LANES], i: &[i64; LANES]) -> ([f64; LANES], [i64; LANES]) {
    let (mut k, mut i) = cmp_merge_ki(
        k,
        i,
        &perm_k(k, &reverse_perm(2)),
        &perm_i(i, &reverse_perm(2)),
        0xAA,
    );
    let t = cmp_merge_ki(
        &k,
        &i,
        &perm_k(&k, &reverse_perm(4)),
        &perm_i(&i, &reverse_perm(4)),
        0xCC,
    );
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(
        &k,
        &i,
        &perm_k(&k, &swap_perm(2)),
        &perm_i(&i, &swap_perm(2)),
        0xAA,
    );
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(
        &k,
        &i,
        &perm_k(&k, &reverse_perm(8)),
        &perm_i(&i, &reverse_perm(8)),
        0xF0,
    );
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(
        &k,
        &i,
        &perm_k(&k, &swap_perm(4)),
        &perm_i(&i, &swap_perm(4)),
        0xCC,
    );
    k = t.0;
    i = t.1;
    let t = cmp_merge_ki(
        &k,
        &i,
        &perm_k(&k, &swap_perm(2)),
        &perm_i(&i, &swap_perm(2)),
        0xAA,
    );
    (t.0, t.1)
}

fn bitonic_merge_reg_8lanes(k: &[f64; LANES], i: &[i64; LANES]) -> ([f64; LANES], [i64; LANES]) {
    let t = cmp_merge_ki(
        k,
        i,
        &perm_k(k, &swap_perm(8)),
        &perm_i(i, &swap_perm(8)),
        0xF0,
    );
    let t = cmp_merge_ki(
        &t.0,
        &t.1,
        &perm_k(&t.0, &swap_perm(4)),
        &perm_i(&t.1, &swap_perm(4)),
        0xCC,
    );
    cmp_merge_ki(
        &t.0,
        &t.1,
        &perm_k(&t.0, &swap_perm(2)),
        &perm_i(&t.1, &swap_perm(2)),
        0xAA,
    )
}

fn bitonic_clean_n_vec(
    keys: &mut [[f64; LANES]],
    idxs: &mut [[i64; LANES]],
    base: usize,
    num_vecs: usize,
) {
    let mut num = num_vecs / 2;
    while num >= 2 {
        let mut j = 0usize;
        while j < num_vecs {
            for i in 0..num / 2 {
                let a = base + j + i;
                let b = base + j + i + num / 2;
                let t = coex(&keys[a], &idxs[a], &keys[b], &idxs[b]);
                keys[a] = t.0;
                idxs[a] = t.1;
                keys[b] = t.2;
                idxs[b] = t.3;
            }
            j += num;
        }
        num /= 2;
    }
}

fn bitonic_merge_n_vec(
    keys: &mut [[f64; LANES]],
    idxs: &mut [[i64; LANES]],
    base: usize,
    num_vecs: usize,
) {
    if num_vecs == 2 {
        let j = base + 1;
        keys[j].reverse();
        idxs[j].reverse();
        let t = coex(&keys[base], &idxs[base], &keys[j], &idxs[j]);
        keys[base] = t.0;
        idxs[base] = t.1;
        keys[j] = t.2;
        idxs[j] = t.3;
        keys[j].reverse();
        idxs[j].reverse();
    } else if num_vecs > 2 {
        for i in 0..num_vecs / 2 {
            let a = base + i;
            let j = base + num_vecs - i - 1;
            keys[j].reverse();
            idxs[j].reverse();
            let t = coex(&keys[a], &idxs[a], &keys[j], &idxs[j]);
            keys[a] = t.0;
            idxs[a] = t.1;
            keys[j] = t.2;
            idxs[j] = t.3;
            keys[j].reverse();
            idxs[j].reverse();
        }
    }
    bitonic_clean_n_vec(keys, idxs, base, num_vecs);
    for i in 0..num_vecs {
        let t = bitonic_merge_reg_8lanes(&keys[base + i], &idxs[base + i]);
        keys[base + i] = t.0;
        idxs[base + i] = t.1;
    }
}

fn bitonic_fullmerge_n_vec(
    keys: &mut [[f64; LANES]],
    idxs: &mut [[i64; LANES]],
    base: usize,
    num_vecs: usize,
    num_per: usize,
) {
    if num_per > num_vecs {
        return;
    }
    for i in 0..num_vecs / num_per {
        bitonic_merge_n_vec(keys, idxs, base + i * num_per, num_per);
    }
    bitonic_fullmerge_n_vec(keys, idxs, base, num_vecs, num_per * 2);
}

/// argsort_n&lt;vtype, argtype, 256&gt; 的 numVecs 会按 N 递归减半。
fn argsort_n_vec(arr: &[f64], ind: &mut [i64], n: usize, num_vecs: usize) {
    let mut num_vecs = num_vecs;
    while num_vecs > 1 && n * 2 <= num_vecs * LANES {
        num_vecs /= 2;
    }
    let mut keys = vec![[0.0f64; LANES]; num_vecs];
    let mut idxs = vec![[0i64; LANES]; num_vecs];
    let half = num_vecs / 2;
    for i in 0..half {
        for l in 0..LANES {
            let v = ind[i * LANES + l];
            idxs[i][l] = v;
            keys[i][l] = arr[v as usize];
        }
    }
    for i in half..num_vecs {
        let num_to_read = if n > i * LANES {
            (n - i * LANES).min(LANES)
        } else {
            0
        };
        for l in 0..LANES {
            if l < num_to_read {
                let v = ind[i * LANES + l];
                idxs[i][l] = v;
                keys[i][l] = arr[v as usize];
            } else {
                // indexType::zmm_max() / keyType::zmm_max()：补齐的通道放最大值
                idxs[i][l] = IDX_MAX;
                keys[i][l] = KEY_MAX;
            }
        }
    }
    for i in 0..num_vecs {
        let t = sort_reg_8lanes(&keys[i], &idxs[i]);
        keys[i] = t.0;
        idxs[i] = t.1;
    }
    bitonic_fullmerge_n_vec(&mut keys, &mut idxs, 0, num_vecs, 2);
    for i in 0..half {
        for l in 0..LANES {
            ind[i * LANES + l] = idxs[i][l];
        }
    }
    for i in half..num_vecs {
        let num_to_write = if n > i * LANES {
            (n - i * LANES).min(LANES)
        } else {
            0
        };
        for l in 0..num_to_write {
            ind[i * LANES + l] = idxs[i][l];
        }
    }
}

fn argsort_n(arr: &[f64], ind: &mut [i64], left: usize, n: usize) {
    let mut sub: Vec<i64> = ind[left..left + n].to_vec();
    argsort_n_vec(arr, &mut sub, n, 256 / LANES);
    ind[left..left + n].copy_from_slice(&sub);
}

/// get_pivot_64bit（numlanes=8 的 median-of-8 分支）
fn get_pivot(arr: &[f64], ind: &[i64], left: usize, right: usize) -> f64 {
    if right - left >= LANES {
        let size = (right - left) / LANES;
        // _mm512_set_epi64(v1..v8)：通道 0 放最后一个参数
        let rand_vec: [f64; LANES] = [
            arr[ind[left + 8 * size] as usize],
            arr[ind[left + 7 * size] as usize],
            arr[ind[left + 6 * size] as usize],
            arr[ind[left + 5 * size] as usize],
            arr[ind[left + 4 * size] as usize],
            arr[ind[left + 3 * size] as usize],
            arr[ind[left + 2 * size] as usize],
            arr[ind[left + 1 * size] as usize],
        ];
        // vtype::sort_vec：只按键的排序网络
        let mut k = rand_vec;
        k = cmp_merge_k(&k, &perm_k(&k, &reverse_perm(2)), 0xAA);
        k = cmp_merge_k(&k, &perm_k(&k, &reverse_perm(4)), 0xCC);
        k = cmp_merge_k(&k, &perm_k(&k, &swap_perm(2)), 0xAA);
        k = cmp_merge_k(&k, &perm_k(&k, &reverse_perm(8)), 0xF0);
        k = cmp_merge_k(&k, &perm_k(&k, &swap_perm(4)), 0xCC);
        k = cmp_merge_k(&k, &perm_k(&k, &swap_perm(2)), 0xAA);
        return k[4];
    }
    arr[ind[right] as usize]
}

/// partition_vec（argsort 版：>= pivot 的压缩写到右侧，< pivot 的写到左侧，通道序保持）
fn partition_vec(
    ind: &mut [i64],
    l_store: usize,
    r_store_plus: usize,
    arg_vec: &[i64; LANES],
    curr_vec: &[f64; LANES],
    pivot: f64,
) -> usize {
    let mut gt = [false; LANES];
    let mut amount = 0usize;
    for l in 0..LANES {
        gt[l] = curr_vec[l] >= pivot;
        if gt[l] {
            amount += 1;
        }
    }
    let mut p = l_store;
    for l in 0..LANES {
        if !gt[l] {
            ind[p] = arg_vec[l];
            p += 1;
        }
    }
    let mut p = r_store_plus - amount;
    for l in 0..LANES {
        if gt[l] {
            ind[p] = arg_vec[l];
            p += 1;
        }
    }
    amount
}

#[inline]
fn upd_minmax(smallest: &mut f64, biggest: &mut f64, v: f64) {
    if v < *smallest {
        *smallest = v;
    }
    if *biggest < v {
        *biggest = v;
    }
}

fn argpartition_scalar(
    arr: &[f64],
    ind: &mut [i64],
    left_in: usize,
    right_in: usize,
    pivot: f64,
    smallest: &mut f64,
    biggest: &mut f64,
) -> usize {
    let mut left = left_in;
    let mut right = right_in;
    let rest = (right - left) % LANES;
    for _ in 0..rest {
        let v = arr[ind[left] as usize];
        upd_minmax(smallest, biggest, v);
        if !(v < pivot) {
            right -= 1;
            ind.swap(left, right);
        } else {
            left += 1;
        }
    }
    if left == right {
        return left;
    }
    if right - left == LANES {
        let mut av = [0i64; LANES];
        av.copy_from_slice(&ind[left..left + LANES]);
        let mut cv = [0.0f64; LANES];
        for l in 0..LANES {
            cv[l] = arr[av[l] as usize];
        }
        let amount = partition_vec(ind, left, left + LANES, &av, &cv, pivot);
        for l in 0..LANES {
            upd_minmax(smallest, biggest, cv[l]);
        }
        return left + (LANES - amount);
    }
    let mut av_left = [0i64; LANES];
    av_left.copy_from_slice(&ind[left..left + LANES]);
    let mut cv_left = [0.0f64; LANES];
    for l in 0..LANES {
        cv_left[l] = arr[av_left[l] as usize];
    }
    let mut av_right = [0i64; LANES];
    av_right.copy_from_slice(&ind[right - LANES..right]);
    let mut cv_right = [0.0f64; LANES];
    for l in 0..LANES {
        cv_right[l] = arr[av_right[l] as usize];
    }
    let mut r_store = right - LANES;
    let mut l_store = left;
    let mut left = left + LANES;
    let mut right = right - LANES;
    while right != left {
        let (av, cv) = if (r_store + LANES) - right < left - l_store {
            right -= LANES;
            let mut av = [0i64; LANES];
            av.copy_from_slice(&ind[right..right + LANES]);
            let mut cv = [0.0f64; LANES];
            for l in 0..LANES {
                cv[l] = arr[av[l] as usize];
            }
            (av, cv)
        } else {
            let mut av = [0i64; LANES];
            av.copy_from_slice(&ind[left..left + LANES]);
            let mut cv = [0.0f64; LANES];
            for l in 0..LANES {
                cv[l] = arr[av[l] as usize];
            }
            left += LANES;
            (av, cv)
        };
        let amount = partition_vec(ind, l_store, r_store + LANES, &av, &cv, pivot);
        for l in 0..LANES {
            upd_minmax(smallest, biggest, cv[l]);
        }
        r_store -= amount;
        l_store += LANES - amount;
    }
    let amount = partition_vec(ind, l_store, r_store + LANES, &av_left, &cv_left, pivot);
    for l in 0..LANES {
        upd_minmax(smallest, biggest, cv_left[l]);
    }
    l_store += LANES - amount;
    let amount = partition_vec(ind, l_store, l_store + LANES, &av_right, &cv_right, pivot);
    for l in 0..LANES {
        upd_minmax(smallest, biggest, cv_right[l]);
    }
    l_store + LANES - amount
}

const NUM_UNROLL: usize = 4;

fn argpartition_unrolled(
    arr: &[f64],
    ind: &mut [i64],
    left_in: usize,
    right_in: usize,
    pivot: f64,
    smallest: &mut f64,
    biggest: &mut f64,
) -> usize {
    if right_in - left_in <= 8 * NUM_UNROLL * LANES {
        return argpartition_scalar(arr, ind, left_in, right_in, pivot, smallest, biggest);
    }
    let mut left = left_in;
    let mut right = right_in;
    let rest = (right - left) % (NUM_UNROLL * LANES);
    for _ in 0..rest {
        let v = arr[ind[left] as usize];
        upd_minmax(smallest, biggest, v);
        if !(v < pivot) {
            right -= 1;
            ind.swap(left, right);
        } else {
            left += 1;
        }
    }
    if left == right {
        return left;
    }
    let mut kv_left = [[0.0f64; LANES]; NUM_UNROLL];
    let mut iv_left = [[0i64; LANES]; NUM_UNROLL];
    let mut kv_right = [[0.0f64; LANES]; NUM_UNROLL];
    let mut iv_right = [[0i64; LANES]; NUM_UNROLL];
    for ii in 0..NUM_UNROLL {
        iv_left[ii].copy_from_slice(&ind[left + LANES * ii..left + LANES * (ii + 1)]);
        for l in 0..LANES {
            kv_left[ii][l] = arr[iv_left[ii][l] as usize];
        }
        let s = right - LANES * (NUM_UNROLL - ii);
        iv_right[ii].copy_from_slice(&ind[s..s + LANES]);
        for l in 0..LANES {
            kv_right[ii][l] = arr[iv_right[ii][l] as usize];
        }
    }
    let mut r_store = right - LANES;
    let mut l_store = left;
    let mut left = left + NUM_UNROLL * LANES;
    let mut right = right - NUM_UNROLL * LANES;

    while right != left {
        let mut avs = [[0i64; LANES]; NUM_UNROLL];
        let mut cvs = [[0.0f64; LANES]; NUM_UNROLL];
        if (r_store + LANES) - right < left - l_store {
            right -= NUM_UNROLL * LANES;
            for ii in 0..NUM_UNROLL {
                avs[ii].copy_from_slice(&ind[right + ii * LANES..right + (ii + 1) * LANES]);
            }
        } else {
            for ii in 0..NUM_UNROLL {
                avs[ii].copy_from_slice(&ind[left + ii * LANES..left + (ii + 1) * LANES]);
            }
            left += NUM_UNROLL * LANES;
        }
        for ii in 0..NUM_UNROLL {
            for l in 0..LANES {
                cvs[ii][l] = arr[avs[ii][l] as usize];
            }
            let amount = partition_vec(ind, l_store, r_store + LANES, &avs[ii], &cvs[ii], pivot);
            for l in 0..LANES {
                upd_minmax(smallest, biggest, cvs[ii][l]);
            }
            l_store += LANES - amount;
            r_store -= amount;
        }
    }
    for ii in 0..NUM_UNROLL {
        let amount = partition_vec(ind, l_store, r_store + LANES, &iv_left[ii], &kv_left[ii], pivot);
        for l in 0..LANES {
            upd_minmax(smallest, biggest, kv_left[ii][l]);
        }
        l_store += LANES - amount;
        r_store -= amount;
    }
    for ii in 0..NUM_UNROLL {
        let amount =
            partition_vec(ind, l_store, r_store + LANES, &iv_right[ii], &kv_right[ii], pivot);
        for l in 0..LANES {
            upd_minmax(smallest, biggest, kv_right[ii][l]);
        }
        l_store += LANES - amount;
        r_store -= amount;
    }
    l_store
}

fn argsort_rec(arr: &[f64], ind: &mut [i64], left: usize, right: i64, max_iters: i64) {
    if (right + 1) as usize - left <= 256 {
        argsort_n(arr, ind, left, (right + 1) as usize - left);
        return;
    }
    if max_iters <= 0 {
        // x86-simd-sort 的兜底：std::sort(arg+low, arg+high+1, [arr](a,b){return arr[a]<arr[b];})
        gcc_std_sort(ind, left, right as usize + 1, arr);
        return;
    }
    let (l, r) = (left, right as usize);
    let pivot = get_pivot(arr, ind, l, r);
    let mut smallest = KEY_MAX;
    let mut biggest = -KEY_MAX;
    let pivot_index = argpartition_unrolled(arr, ind, l, r + 1, pivot, &mut smallest, &mut biggest);
    if pivot != smallest {
        argsort_rec(arr, ind, left, pivot_index as i64 - 1, max_iters - 1);
    }
    if pivot != biggest {
        argsort_rec(arr, ind, pivot_index, right, max_iters - 1);
    }
}

// ============================================================================
// libstdc++ std::sort（introsort）——x86-simd-sort 深度耗尽时的兜底
// ============================================================================

const GCC_THRESHOLD: usize = 16;

#[inline]
fn ig(ind: &[i64], arr: &[f64], i: usize) -> f64 {
    arr[ind[i] as usize]
}

#[inline]
fn gcc_lg(n: usize) -> i32 {
    63 - (n as u64).leading_zeros() as i32
}

fn gcc_move_median_to_first(
    ind: &mut [i64],
    arr: &[f64],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
) {
    let (ka, kb, kc) = (ig(ind, arr, a), ig(ind, arr, b), ig(ind, arr, c));
    if ka < kb {
        if kb < kc {
            ind.swap(result, b);
        } else if ka < kc {
            ind.swap(result, c);
        } else {
            ind.swap(result, a);
        }
    } else if ka < kc {
        ind.swap(result, a);
    } else if kb < kc {
        ind.swap(result, c);
    } else {
        ind.swap(result, b);
    }
}

fn gcc_unguarded_partition(
    ind: &mut [i64],
    arr: &[f64],
    mut first: usize,
    mut last: usize,
    pivot: usize,
) -> usize {
    loop {
        while ig(ind, arr, first) < ig(ind, arr, pivot) {
            first += 1;
        }
        last -= 1;
        while ig(ind, arr, pivot) < ig(ind, arr, last) {
            last -= 1;
        }
        if !(first < last) {
            return first;
        }
        ind.swap(first, last);
        first += 1;
    }
}

fn gcc_unguarded_partition_pivot(
    ind: &mut [i64],
    arr: &[f64],
    first: usize,
    last: usize,
) -> usize {
    let mid = first + (last - first) / 2;
    gcc_move_median_to_first(ind, arr, first, first + 1, mid, last - 1);
    gcc_unguarded_partition(ind, arr, first + 1, last, first)
}

fn gcc_push_heap(ind: &mut [i64], arr: &[f64], mut hole: usize, top: usize, value: i64) {
    while hole > top {
        let parent = (hole - 1) / 2;
        if !(ig(ind, arr, parent) < arr[value as usize]) {
            break;
        }
        ind[hole] = ind[parent];
        hole = parent;
    }
    ind[hole] = value;
}

fn gcc_adjust_heap(ind: &mut [i64], arr: &[f64], mut hole: usize, len: usize, value: i64) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if ig(ind, arr, second) < ig(ind, arr, second - 1) {
            second -= 1;
        }
        ind[hole] = ind[second];
        hole = second;
    }
    if (len & 1) == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        ind[hole] = ind[second - 1];
        hole = second - 1;
    }
    gcc_push_heap(ind, arr, hole, top, value);
}

fn gcc_pop_heap(ind: &mut [i64], arr: &[f64], first: usize, last: usize, result: usize) {
    let value = ind[result];
    ind[result] = ind[first];
    gcc_adjust_heap(ind, arr, first, last - first, value);
}

fn gcc_make_heap(ind: &mut [i64], arr: &[f64], first: usize, last: usize) {
    if last - first < 2 {
        return;
    }
    let len = last - first;
    let mut parent = (len - 2) / 2;
    loop {
        let value = ind[first + parent];
        gcc_adjust_heap(ind, arr, parent, len, value);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

fn gcc_partial_sort(ind: &mut [i64], arr: &[f64], first: usize, last: usize) {
    gcc_make_heap(ind, arr, first, last);
    let mut hi = last;
    while hi - first > 1 {
        hi -= 1;
        gcc_pop_heap(ind, arr, first, hi, hi);
    }
}

fn gcc_unguarded_linear_insert(ind: &mut [i64], arr: &[f64], last: usize) {
    let val = ind[last];
    let mut cur = last;
    let mut nxt = last - 1;
    while arr[val as usize] < ig(ind, arr, nxt) {
        ind[cur] = ind[nxt];
        cur = nxt;
        nxt -= 1;
    }
    ind[cur] = val;
}

fn gcc_insertion_sort(ind: &mut [i64], arr: &[f64], first: usize, last: usize) {
    if first == last {
        return;
    }
    for i in (first + 1)..last {
        if ig(ind, arr, i) < ig(ind, arr, first) {
            let val = ind[i];
            let mut j = i;
            while j > first {
                ind[j] = ind[j - 1];
                j -= 1;
            }
            ind[first] = val;
        } else {
            gcc_unguarded_linear_insert(ind, arr, i);
        }
    }
}

fn gcc_unguarded_insertion_sort(ind: &mut [i64], arr: &[f64], first: usize, last: usize) {
    for i in first..last {
        gcc_unguarded_linear_insert(ind, arr, i);
    }
}

fn gcc_introsort_loop(ind: &mut [i64], arr: &[f64], first: usize, last_in: usize, mut depth: i32) {
    let mut last = last_in;
    while last - first > GCC_THRESHOLD {
        if depth == 0 {
            gcc_partial_sort(ind, arr, first, last);
            return;
        }
        depth -= 1;
        let cut = gcc_unguarded_partition_pivot(ind, arr, first, last);
        gcc_introsort_loop(ind, arr, cut, last, depth);
        last = cut;
    }
}

/// libstdc++ std::sort：introsort + 最后一遍插入排序
fn gcc_std_sort(ind: &mut [i64], first: usize, last: usize, arr: &[f64]) {
    if first == last {
        return;
    }
    let n = last - first;
    gcc_introsort_loop(ind, arr, first, last, 2 * gcc_lg(n));
    if n > GCC_THRESHOLD {
        gcc_insertion_sort(ind, arr, first, first + GCC_THRESHOLD);
        gcc_unguarded_insertion_sort(ind, arr, first + GCC_THRESHOLD, last);
    } else {
        gcc_insertion_sort(ind, arr, first, last);
    }
}

/// numpy 2.3.5 在 AVX512 机器上 `np.argsort(float64_array, kind="quicksort")` 的升序下标。
/// 前提：数组里没有 NaN（pandas 的 nargsort 会先把 NaN 摘掉再调 argsort）。
fn xss_argsort_f64(arr: &[f64]) -> Vec<i64> {
    let n = arr.len();
    let mut ind: Vec<i64> = (0..n as i64).collect();
    if n <= 1 {
        return ind;
    }
    // std::is_sorted(arr, arr+n, arr[a] < arr[b]) 的提前返回
    if (1..n).all(|i| arr[i - 1] <= arr[i]) {
        return ind;
    }
    let mut log2 = 0i64;
    while (1i64 << (log2 + 1)) <= n as i64 {
        log2 += 1;
    }
    let max_iters = 2 * log2;
    argsort_rec(arr, &mut ind, 0, n as i64 - 1, max_iters);
    ind
}

#[pyfunction]
fn argsort_f64(arr: PyReadonlyArray1<f64>) -> Vec<i64> {
    let v = arr.as_slice().unwrap();
    xss_argsort_f64(v)
}

#[pyfunction]
fn argsort_f64_py(py: Python<'_>, arr: PyReadonlyArray1<f64>) -> Py<PyArray1<i64>> {
    let v = arr.as_slice().unwrap();
    xss_argsort_f64(v).into_pyarray(py).to_owned()
}

#[pymodule]
fn xss070(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(argsort_f64, m)?)?;
    m.add_function(wrap_pyfunction!(argsort_f64_py, m)?)?;
    Ok(())
}
