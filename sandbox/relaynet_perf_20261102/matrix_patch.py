"""Repeatable exact matrix optimizations; --compat retains the old cache-key bug."""
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("--compat", action="store_true")
p.add_argument("--source", type=Path, default=Path(__file__).parent / "src/relaynet_fast.rs")
args = p.parse_args()
text = args.source.read_text()


def replace_region(start, end, replacement):
    global text
    lo = text.index(start)
    hi = text.index(end, lo)
    text = text[:lo] + replacement.rstrip() + "\n\n" + text[hi:]


replace_region("/// 用预先中心化的列算皮尔逊", "/// 列标准化皮尔逊", r'''
/// 用预先中心化的列算皮尔逊。SIMD 的不同通道对应不同股票对，
/// 每个通道仍按原来的时间顺序逐项相加，不改变浮点归约顺序、不使用 FMA。
fn pearson_from(cols: &[Vec<f64>], sd: &[f64]) -> Array2<f64> {
    let n = cols.len();
    let t = cols.first().map_or(0, Vec::len);
    const LANES: usize = 8;
    let blocks = (n + LANES - 1) / LANES;
    let mut packed = vec![0.0; blocks * t * LANES];
    for j in 0..n {
        let base = (j / LANES) * t * LANES + j % LANES;
        for k in 0..t { packed[base + k * LANES] = cols[j][k]; }
    }
    let rows: Vec<Vec<f64>> = (0..n).into_par_iter().map(|i| {
        let mut row = vec![f64::NAN; n];
        if sd[i] > 0.0 {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                unsafe { pearson_row_avx2(&cols[i], &packed, sd, i, &mut row); }
                return row;
            }
            for j in i + 1..n {
                if sd[j] <= 0.0 { continue; }
                let mut num = 0.0;
                for k in 0..t { num += cols[i][k] * cols[j][k]; }
                row[j] = num / (sd[i] * sd[j]);
            }
        }
        row
    }).collect();
    let mut out = Array2::<f64>::from_elem((n, n), f64::NAN);
    for (i, row) in rows.into_iter().enumerate() {
        for j in i + 1..n { out[[i, j]] = row[j]; out[[j, i]] = row[j]; }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pearson_row_avx2(ci: &[f64], packed: &[f64], sd: &[f64], i: usize, row: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = sd.len();
    let t = ci.len();
    for block in (i + 1) / 8..(n + 7) / 8 {
        let p = packed.as_ptr().add(block * t * 8);
        let mut a = _mm256_setzero_pd();
        let mut b = _mm256_setzero_pd();
        for k in 0..t {
            let x = _mm256_set1_pd(*ci.get_unchecked(k));
            let pa = _mm256_mul_pd(x, _mm256_loadu_pd(p.add(k * 8)));
            let pb = _mm256_mul_pd(x, _mm256_loadu_pd(p.add(k * 8 + 4)));
            a = _mm256_add_pd(a, pa);
            b = _mm256_add_pd(b, pb);
        }
        let mut nums = [0.0f64; 8];
        _mm256_storeu_pd(nums.as_mut_ptr(), a);
        _mm256_storeu_pd(nums.as_mut_ptr().add(4), b);
        for lane in 0..8 {
            let j = block * 8 + lane;
            if j > i && j < n && !(sd[j] <= 0.0) { row[j] = nums[lane] / (sd[i] * sd[j]); }
        }
    }
}
''')

replace_region("/// 符号一致率", "/// 上行尾部相关", r'''
/// 符号一致率：两个位图分别保留非零和正数判定，连 NaN 的旧判定也一致。
fn sign_mat(rets: &Array2<f64>, min_pair: usize) -> Array2<f64> {
    let (t, n) = (rets.nrows(), rets.ncols());
    let nw = (t + 63) / 64;
    let mut nonzero = vec![0u64; n * nw];
    let mut positive = vec![0u64; n * nw];
    for k in 0..t {
        let mask = 1u64 << (k % 64);
        for j in 0..n {
            let v = rets[[k, j]];
            if v != 0.0 { nonzero[j * nw + k / 64] |= mask; }
            if v > 0.0 { positive[j * nw + k / 64] |= mask; }
        }
    }
    let rows: Vec<Vec<f64>> = (0..n).into_par_iter().map(|i| {
        let mut row = vec![f64::NAN; n];
        for j in i + 1..n {
            let (mut tot, mut different) = (0usize, 0usize);
            for w in 0..nw {
                let common = nonzero[i * nw + w] & nonzero[j * nw + w];
                tot += common.count_ones() as usize;
                different += (common & (positive[i * nw + w] ^ positive[j * nw + w])).count_ones() as usize;
            }
            if tot >= min_pair { row[j] = (tot - different) as f64 / tot as f64; }
        }
        row
    }).collect();
    let mut out = Array2::<f64>::from_elem((n, n), f64::NAN);
    for (i, row) in rows.into_iter().enumerate() {
        for j in i + 1..n { out[[i, j]] = row[j]; out[[j, i]] = row[j]; }
    }
    out
}
''')

replace_region("/// 领先-滞后", "/// 0/1 矩阵的 Jaccard", r'''
/// 领先-滞后反对称矩阵。仅算上三角：交换股票后前向和后向严格互换。
/// 精确相消保留 +0；非零分子除法下溢到零时仍保留正确的正负号。
fn leadlag_mat(rets: &Array2<f64>) -> Array2<f64> {
    let (t, n) = (rets.nrows(), rets.ncols());
    let (cols, sd) = centered_cols(rets);
    let rows: Vec<(Vec<f64>, Vec<usize>)> = (0..n).into_par_iter().map(|i| {
        let mut row = vec![f64::NAN; n];
        let mut exact_zeros = Vec::new();
        if sd[i] <= 0.0 { return (row, exact_zeros); }
        let ci = &cols[i];
        for j in i + 1..n {
            if sd[j] <= 0.0 { continue; }
            let cj = &cols[j];
            let (mut fwd, mut bwd) = (0.0, 0.0);
            for k in 0..t.saturating_sub(1) { fwd += ci[k] * cj[k + 1]; }
            for k in 1..t { bwd += ci[k] * cj[k - 1]; }
            let numerator = fwd - bwd;
            row[j] = numerator / (sd[i] * sd[j]);
            if numerator == 0.0 && row[j] == 0.0 { exact_zeros.push(j); }
        }
        (row, exact_zeros)
    }).collect();
    let mut out = Array2::<f64>::from_elem((n, n), f64::NAN);
    for (i, (row, exact_zeros)) in rows.into_iter().enumerate() {
        for j in i + 1..n {
            out[[i, j]] = row[j];
            out[[j, i]] = -row[j];
        }
        for j in exact_zeros { out[[j, i]] = 0.0; }
    }
    out
}
''')

# A series has several distinct similarity definitions (rd/ra/rs/rl).
# Include the definition in the final-matrix cache, keeping the shared centered
# statistics keyed by series/session. --compat allows strict old-output timing.
text = text.replace("#[derive(Clone, Copy)]\nenum SimKind", "#[derive(Clone, Copy, PartialEq, Eq, Hash)]\nenum SimKind")
old_type = "sims: RefCell<std::collections::HashMap<(u8, u8), Rc<Array2<f64>>>>"
new_type = "sims: RefCell<std::collections::HashMap<(u8, u8, SimKind), Rc<Array2<f64>>>>"
old_key = "let key = (fs.series, sess_u8(sess));"
new_key = "let key = (fs.series, sess_u8(sess), fs.kind);"
if args.compat:
    text = text.replace(new_type, old_type).replace(new_key, old_key)
else:
    text = text.replace(old_type, new_type).replace(old_key, new_key)

args.source.write_text(text)
print(f"matrix patch applied: {args.source}; corrected cache={not args.compat}")
