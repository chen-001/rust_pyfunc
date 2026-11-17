//! tlq_gate: base_resid 的分段口径 TLQ（时序分层质量），供 tail_pipeline_engine 的
//! 「按 TLQ 淘汰」环节使用；口径与 TLQ_report.pdf、同事的 factor_gate 流程一致。
//!
//! 整条链：rank pct → 行业回归填充（基准 = size 的 rank pct，三级）→ 行业中位填充（三级）
//!        → 限制股置空 → rank pct → Barra 线性中性化取残差 → rank pct
//!        → 十分组（自然周再平衡、组内等权、日频对数超额收益）→ 累计曲线
//!        → 水位口径 TLQ 与分段口径 TLQ（相邻采样点增量，159 段）
//!
//! 验证：与 sandbox 的实现逐位一致；sandbox 版已用 hm94 全部 35 个因子与上线因子报告对账
//! （水位口径秩相关 0.992，分段口径见 tlq_gate.json）。
use ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const EPS: f64 = 1e-12;

/// 行内 `pandas.rank(axis=1, pct=True)` 语义：非有限值不参与，并列取平均名次，pct = 名次 / 有效个数。
fn rank_pct_row_inplace(vals: &mut [f64]) {
    let n = vals.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| vals[i].is_finite()).collect();
    let nv = idx.len();
    if nv == 0 {
        return;
    }
    idx.sort_unstable_by_key(|&a| mono_key(vals[a]));
    let mut i = 0;
    while i < nv {
        let mut j = i;
        while j + 1 < nv && vals[idx[j + 1]] == vals[idx[i]] {
            j += 1;
        }
        let pct = (((i + 1) + (j + 1)) as f64 / 2.0) / nv as f64;
        for k in i..=j {
            vals[idx[k]] = pct;
        }
        i = j + 1;
    }
}

fn rank_pct_all(m: &mut Array2<f64>) {
    let n = m.ncols();
    m.as_slice_mut()
        .expect("矩阵必须行主序连续")
        .par_chunks_mut(n)
        .for_each(rank_pct_row_inplace);
}

/// f64 → 单调 u64 键（NaN 最大），整数比较比 f64 total_cmp 快得多。
#[inline]
fn mono_key(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits >> 63 == 0 {
        bits ^ 0x8000_0000_0000_0000
    } else {
        !bits
    }
}

#[inline]
fn code_l2(industry: f64) -> f64 {
    (industry / 100.0).floor()
}

#[inline]
fn code_l1(industry: f64) -> f64 {
    (industry / 10000.0).floor()
}

/// 回归填充用的三级编码：0 = 二级，1 = 一级，2 = 一级是否已知（0/1 两段）。
#[inline]
fn code_reg(level: usize, industry: f64) -> f64 {
    match level {
        0 => code_l2(industry),
        1 => code_l1(industry),
        _ => {
            if code_l1(industry).is_nan() {
                0.0
            } else {
                1.0
            }
        }
    }
}

/// 一行里按 code 分段：返回每个 code 段的下标列表（只含 code 有限的格子）。
fn group_indices(codes: &[f64]) -> Vec<(f64, usize)> {
    let n = codes.len();
    let mut pairs: Vec<(f64, usize)> = (0..n)
        .filter(|&j| codes[j].is_finite())
        .map(|j| (codes[j], j))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    pairs
}

/// 局部选择求中位数（偶数取中间两个的平均，O(n) 替代全排序，结果逐位一致）。
fn median_sorted(buf: &mut Vec<f64>) -> Option<f64> {
    let n = buf.len();
    if n == 0 {
        return None;
    }
    let mid = n / 2;
    let (_, &mut upper, _) = buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        Some(upper)
    } else {
        let lower = buf[..mid]
            .iter()
            .copied()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(upper);
        Some((lower + upper) / 2.0)
    }
}

/// 最小二乘（正规方程 + 部分主元高斯消元）。design 行主序 (m × p)，p 很小。
fn lstsq_solve(design: &[f64], m: usize, p: usize, y: &[f64]) -> Option<Vec<f64>> {
    if m < p {
        return None;
    }
    let mut gram = vec![0.0f64; p * p];
    let mut rhs = vec![0.0f64; p];
    for i in 0..m {
        let row = &design[i * p..(i + 1) * p];
        for a in 0..p {
            rhs[a] += row[a] * y[i];
            for b in a..p {
                gram[a * p + b] += row[a] * row[b];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            gram[a * p + b] = gram[b * p + a];
        }
    }
    let mut aug = vec![0.0f64; p * (p + 1)];
    for a in 0..p {
        aug[a * (p + 1)..a * (p + 1) + p].copy_from_slice(&gram[a * p..(a + 1) * p]);
        aug[a * (p + 1) + p] = rhs[a];
    }
    for col in 0..p {
        let mut pivot = col;
        for r in col + 1..p {
            if aug[r * (p + 1) + col].abs() > aug[pivot * (p + 1) + col].abs() {
                pivot = r;
            }
        }
        if aug[pivot * (p + 1) + col].abs() <= EPS {
            return None;
        }
        if pivot != col {
            for c in col..=p {
                aug.swap(col * (p + 1) + c, pivot * (p + 1) + c);
            }
        }
        let diag = aug[col * (p + 1) + col];
        for c in col..=p {
            aug[col * (p + 1) + c] /= diag;
        }
        for r in 0..p {
            if r != col {
                let factor = aug[r * (p + 1) + col];
                if factor != 0.0 {
                    for c in col..=p {
                        aug[r * (p + 1) + c] -= factor * aug[col * (p + 1) + c];
                    }
                }
            }
        }
    }
    Some((0..p).map(|a| aug[a * (p + 1) + p]).collect())
}

/// 行业回归填充：口径同 Python `_fill_industry_regression`。
fn fill_industry_regression(
    values: &mut Array2<f64>,
    base_rank: &Array2<f64>,
    industry: &ArrayView2<f64>,
) {
    let (t, n) = values.dim();
    let v = values.as_slice_mut().unwrap();
    let b = base_rank.as_slice().unwrap();
    let ind = industry.as_slice().unwrap();
    v.par_chunks_mut(n).enumerate().for_each(|(row, vrow)| {
        let brow = &b[row * n..(row + 1) * n];
        let irow = &ind[row * n..(row + 1) * n];
        // 本行有限值去重个数 < 10 就整行跳过（只数到 10 即短路）
        let mut seen: Vec<f64> = Vec::with_capacity(10);
        for &x in vrow.iter() {
            if x.is_finite() && !seen.iter().any(|&s| s == x) {
                seen.push(x);
                if seen.len() >= 10 {
                    break;
                }
            }
        }
        if seen.len() < 10 {
            return;
        }
        // 行业码 //100 与 //10000 都是行业码的单调函数 → 按二级码计数排序一次即可服务三级
        let order = counting_order_l2(irow, n);
        let mut design: Vec<f64> = Vec::with_capacity(1024);
        let mut y: Vec<f64> = Vec::with_capacity(512);
        let mut missing: Vec<usize> = Vec::with_capacity(512);
        for level in 0..3 {
            let code = |j: usize| -> f64 { code_reg(level, irow[j]) };
            let mut start = 0usize;
            // level 2 的两段：行业未知(0.0) 与 行业已知(1.0)，未知那批不在 order 里
            if level == 2 {
                let mut groups: Vec<Vec<usize>> = vec![Vec::new(), Vec::new()];
                for j in 0..n {
                    groups[if irow[j].is_finite() { 1 } else { 0 }].push(j);
                }
                for group in groups.iter() {
                    fill_one_group(group, vrow, brow, &mut design, &mut y, &mut missing);
                }
                continue;
            }
            while start < order.len() {
                let first_code = code(order[start]);
                let mut end = start;
                while end + 1 < order.len() && code(order[end + 1]) == first_code {
                    end += 1;
                }
                fill_one_group(&order[start..=end], vrow, brow, &mut design, &mut y, &mut missing);
                start = end + 1;
            }
        }
    });
}

/// 按二级行业码（industry//100 取整）对「行业已知」的股票做计数排序，码值升序。
fn counting_order_l2(irow: &[f64], n: usize) -> Vec<usize> {
    let mut codes: Vec<u32> = Vec::with_capacity(n);
    let mut idxs: Vec<usize> = Vec::with_capacity(n);
    let mut max_code = 0u32;
    for j in 0..n {
        let ind = irow[j];
        if !ind.is_finite() {
            continue;
        }
        let c = (ind / 100.0).floor() as u32;
        max_code = max_code.max(c);
        codes.push(c);
        idxs.push(j);
    }
    let mut count = vec![0u32; max_code as usize + 2];
    for &c in codes.iter() {
        count[c as usize + 1] += 1;
    }
    for i in 1..count.len() {
        count[i] += count[i - 1];
    }
    let mut order = vec![0usize; idxs.len()];
    for (k, &j) in idxs.iter().enumerate() {
        let c = codes[k] as usize;
        order[count[c] as usize] = j;
        count[c] += 1;
    }
    order
}

/// 组内做一次「用 (1, base) 拟合值，把缺失位置填成预测值」。
fn fill_one_group(
    group: &[usize],
    vrow: &mut [f64],
    brow: &[f64],
    design: &mut Vec<f64>,
    y: &mut Vec<f64>,
    missing: &mut Vec<usize>,
) {
    design.clear();
    y.clear();
    missing.clear();
    for &j in group {
        if !brow[j].is_finite() {
            continue;
        }
        if vrow[j].is_finite() {
            design.push(1.0);
            design.push(brow[j]);
            y.push(vrow[j]);
        } else {
            missing.push(j);
        }
    }
    let m = y.len();
    if m >= 10 && !missing.is_empty() {
        if let Some(coef) = lstsq_solve(design, m, 2, y) {
            for &j in missing.iter() {
                vrow[j] = coef[0] + coef[1] * brow[j];
            }
        }
    }
}

/// 行业中位填充：口径同 Python `_fill_industry_median`（中位数取自填充前快照）。
fn fill_industry_median(values: &mut Array2<f64>, industry: &ArrayView2<f64>) {
    let (t, n) = values.dim();
    let source = values.clone();
    let v = values.as_slice_mut().unwrap();
    let src = source.as_slice().unwrap();
    let ind = industry.as_slice().unwrap();
    v.par_chunks_mut(n).enumerate().for_each(|(row, vrow)| {
        let irow = &ind[row * n..(row + 1) * n];
        let srow = &src[row * n..(row + 1) * n];
        for j in 0..n {
            if code_l1(irow[j]).is_nan() {
                vrow[j] = f64::NAN;
            }
        }
        let order = counting_order_l2(irow, n);
        let mut buf: Vec<f64> = Vec::with_capacity(n);
        for level in 0..2 {
            let code = |j: usize| -> f64 {
                if level == 0 {
                    code_l2(irow[j])
                } else {
                    code_l1(irow[j])
                }
            };
            let mut start = 0usize;
            while start < order.len() {
                let first_code = code(order[start]);
                let mut end = start;
                while end + 1 < order.len() && code(order[end + 1]) == first_code {
                    end += 1;
                }
                fill_median_group(&order[start..=end], vrow, srow, &mut buf);
                start = end + 1;
            }
        }
        // 第三级：全市场一组
        let all: Vec<usize> = (0..n).collect();
        fill_median_group(&all, vrow, srow, &mut buf);
    });
}

fn fill_median_group(group: &[usize], vrow: &mut [f64], srow: &[f64], buf: &mut Vec<f64>) {
    buf.clear();
    for &j in group {
        if srow[j].is_finite() {
            buf.push(srow[j]);
        }
    }
    if let Some(med) = median_sorted(buf) {
        for &j in group {
            if !vrow[j].is_finite() {
                vrow[j] = med;
            }
        }
    }
}

/// Barra 线性中性化：对 [1, barra 各列] 做 OLS 取残差，再整体 rank pct。
fn residual_barra_ranked(values: &mut Array2<f64>, barra: &Array2<f64>, k: usize) {
    let (t, n) = values.dim();
    assert_eq!(barra.ncols(), n * k, "barra 列数必须等于 股票数 × 风格数");
    let p = k + 1;
    let v = values.as_slice_mut().unwrap();
    let bx = barra.as_slice().unwrap();
    v.par_chunks_mut(n).enumerate().for_each(|(row, vrow)| {
        let bslice = &bx[row * n * k..(row + 1) * n * k];
        let mut flag = vec![false; n];
        let mut idx: Vec<usize> = Vec::with_capacity(n);
        for j in 0..n {
            let mut ok = vrow[j].is_finite();
            for c in 0..k {
                ok &= bslice[j * k + c].is_finite();
            }
            if ok {
                flag[j] = true;
                idx.push(j);
            }
        }
        let m = idx.len();
        if m <= 10 {
            vrow.fill(f64::NAN);
            return;
        }
        let first = vrow[idx[0]];
        if idx.iter().all(|&j| vrow[j] == first) {
            for j in 0..n {
                vrow[j] = if flag[j] { 0.5 } else { f64::NAN };
            }
            return;
        }
        let mut design = vec![0.0f64; m * p];
        let mut y = vec![0.0f64; m];
        for (i, &j) in idx.iter().enumerate() {
            design[i * p] = 1.0;
            for c in 0..k {
                design[i * p + c + 1] = bslice[j * k + c];
            }
            y[i] = vrow[j];
        }
        let coef = lstsq_solve(&design, m, p, &y);
        let mut resid = vec![f64::NAN; n];
        if let Some(coef) = coef {
            for (i, &j) in idx.iter().enumerate() {
                let mut pred = coef[0];
                for c in 0..k {
                    pred += coef[c + 1] * design[i * p + c + 1];
                }
                resid[j] = y[i] - pred;
            }
        }
        vrow.copy_from_slice(&resid);
    });
    rank_pct_all(values);
}

/// 十分组日收益（周频换组、组内等权）。
fn group_returns(
    resid: &mut Array2<f64>,
    log_exret: &Array2<f64>,
    restrict: &ArrayView2<f64>,
    week_first: &Array1<i32>,
    group_from: usize,
    group_to: usize,
) -> Array2<f64> {
    let (t, n) = resid.dim();
    let v = resid.as_slice_mut().unwrap();
    let rs = restrict.as_slice().unwrap();
    // 1) 行标准差为 0 整行置空；2) 限制股置空
    v.par_chunks_mut(n).enumerate().for_each(|(row, vrow)| {
        let mut cnt = 0usize;
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        for &x in vrow.iter() {
            if x.is_finite() {
                cnt += 1;
                sum += x;
                sum2 += x * x;
            }
        }
        if cnt > 1 {
            let var = (sum2 - sum * sum / cnt as f64) / (cnt - 1) as f64;
            if var == 0.0 {
                vrow.fill(f64::NAN);
                return;
            }
        }
        for j in 0..n {
            let r = rs[row * n + j];
            if !(r.is_finite() && r == 0.0) {
                vrow[j] = f64::NAN;
            }
        }
    });
    // 3) percentile = 残差再 rank pct；有效股票 <= 10 只的行整行置空
    rank_pct_all(resid);
    let v = resid.as_slice_mut().unwrap();
    v.par_chunks_mut(n).for_each(|vrow| {
        let cnt = vrow.iter().filter(|x| x.is_finite()).count();
        if cnt <= 10 {
            vrow.fill(f64::NAN);
        }
    });
    // 4) 每只股票的逐日分组号 gnum = 10 - floor((1 - pct) * 10)；无效日记 -9999
    //    pandas 的周频是「按列（按股票）」取该周第一个交易日的上一交易日分组，所以这里要逐股算。
    let gnum_all: Vec<i32> = resid
        .as_slice()
        .unwrap()
        .par_iter()
        .map(|&p| {
            if p.is_finite() {
                10 - ((1.0 - p) * 10.0).floor() as i32
            } else {
                -9999
            }
        })
        .collect();
    let v = resid.as_slice().unwrap();
    let lr = log_exret.as_slice().unwrap();
    let rows: Vec<[f64; 10]> = (group_from..group_to)
        .into_par_iter()
        .map(|row| {
            let mut acc = [0.0f64; 10];
            let mut cnt = [0usize; 10];
            let wf = week_first[row];
            if wf >= 1 {
                let prev = (wf as usize - 1) * n;
                for j in 0..n {
                    let p = v[row * n + j];
                    if !p.is_finite() {
                        continue;
                    }
                    let g = gnum_all[prev + j];
                    if !(1..=10).contains(&g) {
                        continue;
                    }
                    let gi = (g - 1) as usize;
                    cnt[gi] += 1;
                    let r = lr[row * n + j];
                    if r.is_finite() {
                        acc[gi] += r;
                    }
                }
            }
            let mut out = [0.0f64; 10];
            for i in 0..10 {
                out[i] = if cnt[i] == 0 {
                    0.0
                } else {
                    acc[i] / cnt[i] as f64
                };
            }
            out
        })
        .collect();
    let mut rets = Array2::<f64>::zeros((rows.len(), 10));
    for (i, vals) in rows.iter().enumerate() {
        for g in 0..10 {
            rets[[i, g]] = vals[g];
        }
    }
    rets
}

/// np.rint：四舍六入五取偶。
fn rint(x: f64) -> f64 {
    let f = x.floor();
    let frac = x - f;
    if frac > 0.5 {
        f + 1.0
    } else if frac < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

fn sample_positions(len: usize, curve_points: usize) -> Vec<usize> {
    (0..curve_points)
        .map(|k| rint((len - 1) as f64 * k as f64 / (curve_points - 1) as f64) as usize)
        .collect()
}

/// TLQ：十组日收益 → 累计曲线 → 160 等距点 → 端点分离 × 方向保留比例（双向取最大）。
fn tlq_from_returns(returns: &Array2<f64>, curve_points: usize) -> f64 {
    let (t, _) = returns.dim();
    if t < 2 {
        return f64::NAN;
    }
    let cp = curve_points;
    let mut cum = vec![0.0f64; t * 10];
    for g in 0..10 {
        let mut acc = 0.0;
        for i in 0..t {
            let r = returns[[i, g]];
            acc += if r.is_finite() { r } else { 0.0 };
            cum[i * 10 + g] = acc;
        }
    }
    let pos = sample_positions(t, cp);
    let mut sampled = vec![0.0f64; 10 * cp];
    for (k, &p) in pos.iter().enumerate() {
        for g in 0..10 {
            sampled[g * cp + k] = cum[p * 10 + g];
        }
    }
    let mut q_up = 0.0f64;
    let mut q_down = 0.0f64;
    for k in 0..cp {
        let mut gmax = f64::NEG_INFINITY;
        let mut gmin = f64::INFINITY;
        for g in 0..10 {
            let x = sampled[g * cp + k];
            gmax = gmax.max(x);
            gmin = gmin.min(x);
        }
        let spread = gmax - gmin;
        let d_t = if spread > 0.0 {
            (sampled[9 * cp + k] - sampled[k]) / spread
        } else {
            0.0
        };
        let mut movement = 0.0f64;
        let mut up_viol = 0.0f64;
        let mut down_viol = 0.0f64;
        for i in 0..9 {
            let gap = sampled[(i + 1) * cp + k] - sampled[i * cp + k];
            movement += gap.abs();
            if -gap > 0.0 {
                up_viol += -gap;
            }
            if gap > 0.0 {
                down_viol += gap;
            }
        }
        let (v_up, v_down) = if movement > 0.0 {
            (up_viol / movement, down_viol / movement)
        } else {
            (0.0, 0.0)
        };
        q_up += d_t * (1.0 - v_up);
        q_down += -d_t * (1.0 - v_down);
    }
    (q_up / cp as f64).max(q_down / cp as f64).max(0.0)
}

/// 对一张 (T × N) 矩阵逐行做 rank pct（factor 无关的量只算一次，别每个因子重算）。
#[pyfunction]
pub fn tlq_gate_rank_pct_rows<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let mut m = matrix.as_array().to_owned();
    py.allow_threads(|| rank_pct_all(&mut m));
    Ok(m.into_pyarray(py))
}

/// 把 k 张 (T × N) 矩阵交错成 (T × N*k) 的行主序块，供回测逐行取用。
#[pyfunction]
pub fn tlq_gate_interleave_styles<'py>(
    py: Python<'py>,
    styles: Vec<PyReadonlyArray2<'py, f64>>,
    n_stocks: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let k = styles.len();
    let t = styles[0].as_array().nrows();
    let mut out = Array2::<f64>::zeros((t, n_stocks * k));
    for (c, item) in styles.iter().enumerate() {
        let arr = item.as_array();
        for row in 0..t {
            for j in 0..n_stocks {
                out[[row, j * k + c]] = arr[[row, j]];
            }
        }
    }
    Ok(out.into_pyarray(py))
}

/// 分段口径 TLQ：把「水位」换成「相邻采样点之间那一段的增量」，段与段不重叠。
///
/// 与水位口径用同一批采样点（默认 160 个），于是窗口长度不用人工拍：
/// 每段的视界 = 总天数 / (采样点数 - 1)，十年 160 点约 16 个交易日。
/// 返回 (TLQ, 每段的 [D_k, 递增方向保留比例, 递减方向保留比例, 选中方向的质量])。
fn tlq_segment_from_returns(returns: &Array2<f64>, curve_points: usize) -> (f64, Array2<f64>) {
    let (t, _) = returns.dim();
    if t < 2 || curve_points < 2 {
        return (f64::NAN, Array2::zeros((0, 4)));
    }
    // 累计曲线（含第 0 行 = 0），只用来做段内差分
    let mut cum = vec![0.0f64; (t + 1) * 10];
    for g in 0..10 {
        let mut acc = 0.0;
        for i in 0..t {
            let r = returns[[i, g]];
            acc += if r.is_finite() { r } else { 0.0 };
            cum[(i + 1) * 10 + g] = acc;
        }
    }
    let pos = sample_positions(t, curve_points);
    let nseg = pos.len() - 1;
    let mut out = Array2::<f64>::zeros((nseg, 4));
    let mut q_up = 0.0f64;
    let mut q_down = 0.0f64;
    for k in 0..nseg {
        let (a, b) = (pos[k], pos[k + 1]);
        let mut d = [0.0f64; 10];
        for g in 0..10 {
            d[g] = cum[(b + 1) * 10 + g] - cum[(a + 1) * 10 + g];
        }
        let mut gmax = f64::NEG_INFINITY;
        let mut gmin = f64::INFINITY;
        for g in 0..10 {
            gmax = gmax.max(d[g]);
            gmin = gmin.min(d[g]);
        }
        let spread = gmax - gmin;
        let dk = if spread > 0.0 { (d[9] - d[0]) / spread } else { 0.0 };
        let mut movement = 0.0f64;
        let mut up_viol = 0.0f64;
        let mut down_viol = 0.0f64;
        for i in 0..9 {
            let gap = d[i + 1] - d[i];
            movement += gap.abs();
            if -gap > 0.0 {
                up_viol += -gap;
            }
            if gap > 0.0 {
                down_viol += gap;
            }
        }
        let (v_up, v_down) = if movement > 0.0 {
            (up_viol / movement, down_viol / movement)
        } else {
            (0.0, 0.0)
        };
        let qu = dk * (1.0 - v_up);
        let qd = -dk * (1.0 - v_down);
        out[[k, 0]] = dk;
        out[[k, 1]] = 1.0 - v_up;
        out[[k, 2]] = 1.0 - v_down;
        out[[k, 3]] = if qu >= qd { qu } else { qd };
        q_up += qu;
        q_down += qd;
    }
    let tlq = (q_up / nseg as f64).max(q_down / nseg as f64).max(0.0);
    (tlq, out)
}

/// 一次算完一个因子的 base_resid 与两套 TLQ（水位口径 + 分段口径）。
///
/// 入参都是行主序连续的 f64 矩阵（日期 × 股票）：
///   factor        原始因子值
///   log_exret     对数超额收益（Python 侧算好）
///   restrict      0 = 可交易
///   industry      申万行业码
///   barra         k 个风格因子原始值（函数内各自 rank pct）
///   size          fill 基准（函数内 rank pct）
///   week_first    每个交易日所属自然周的第一个交易日的下标（-1 表示该日不参与）
///   group_from / group_to  参与 TLQ 的日期区间（Python 侧按 start/end 截）
///
/// 返回 (水位口径 TLQ, 分段口径 TLQ, 十组日收益, 160 点采样曲线, 每段明细)。
#[pyfunction]
#[pyo3(signature = (factor, log_exret, restrict, industry, barra_ranked, n_styles, size_ranked, week_first, group_from, group_to, curve_points=160))]
pub fn tlq_gate_base_resid<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f64>,
    log_exret: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f64>,
    industry: PyReadonlyArray2<'py, f64>,
    barra_ranked: PyReadonlyArray2<'py, f64>,
    n_styles: usize,
    size_ranked: PyReadonlyArray2<'py, f64>,
    week_first: PyReadonlyArray1<'py, i32>,
    group_from: usize,
    group_to: usize,
    curve_points: usize,
) -> PyResult<(f64, f64, &'py PyArray2<f64>, &'py PyArray2<f64>, &'py PyArray2<f64>)> {
    let factor = factor.as_array().to_owned();
    let log_exret = log_exret.as_array().to_owned();
    let restrict = restrict.as_array().to_owned();
    let industry = industry.as_array().to_owned();
    let size_ranked = size_ranked.as_array().to_owned();
    let barra_flat = barra_ranked.as_array().to_owned();
    let week_first_vec: Vec<i32> = week_first.as_array().to_vec();
    let (t, n) = factor.dim();
    let k = n_styles;

    let (tlq, tlq_seg, rets, curves, seg_stats) = py.allow_threads(move || {
        let mut value = factor;
        rank_pct_all(&mut value);
        fill_industry_regression(&mut value, &size_ranked, &industry.view());
        fill_industry_median(&mut value, &industry.view());
        {
            let rs = restrict.as_slice().unwrap();
            let vv = value.as_slice_mut().unwrap();
            for i in 0..t * n {
                if !(rs[i].is_finite() && rs[i] == 0.0) {
                    vv[i] = f64::NAN;
                }
            }
        }
        rank_pct_all(&mut value);
        residual_barra_ranked(&mut value, &barra_flat, k);
        let rets = group_returns(
            &mut value,
            &log_exret,
            &restrict.view(),
            &Array1::from(week_first_vec),
            group_from,
            group_to,
        );
        let tlq = tlq_from_returns(&rets, curve_points);
        let (tlq_seg, seg_stats) = tlq_segment_from_returns(&rets, curve_points);
        let (rows, _) = rets.dim();
        let mut cum = Array2::<f64>::zeros((rows, 10));
        for g in 0..10 {
            let mut acc = 0.0;
            for i in 0..rows {
                let r = rets[[i, g]];
                acc += if r.is_finite() { r } else { 0.0 };
                cum[[i, g]] = acc;
            }
        }
        let pos = sample_positions(rows, curve_points);
        let mut curves = Array2::<f64>::zeros((curve_points, 10));
        for (kk, &p) in pos.iter().enumerate() {
            for g in 0..10 {
                curves[[kk, g]] = cum[[p, g]];
            }
        }
        (tlq, tlq_seg, rets, curves, seg_stats)
    });
    Ok((
        tlq,
        tlq_seg,
        rets.into_pyarray(py),
        curves.into_pyarray(py),
        seg_stats.into_pyarray(py),
    ))
}

/// 每日与因子无关的预计算结构（行业分段 + 标准有效集合下的 Barra 正规方程逆矩阵）。
struct TlqDayMeta {
    order_l2: Vec<usize>,
    l0_ranges: Vec<(usize, usize)>,
    l1_ranges: Vec<(usize, usize)>,
    ind_unknown: Vec<usize>,
    ind_known: Vec<usize>,
    l1_nan_cols: Vec<usize>,
    /// 当天 restrict==0 且全部 k 列风格有限的股票列下标
    base_idx: Vec<usize>,
    /// 对应 base_idx 的紧凑设计矩阵 (m × p)，p = k + 1，第 0 列恒为 1.0
    base_design: Vec<f64>,
    /// 若当天 base_idx 样本数 > 10 且正规方程满秩，预存 (X^T X)^{-1} (p × p)
    base_gram_inv: Option<Vec<f64>>,
}

fn invert_spd_or_gauss(gram: &[f64], p: usize) -> Option<Vec<f64>> {
    let stride = 2 * p;
    let mut aug = vec![0.0f64; p * stride];
    for r in 0..p {
        aug[r * stride..r * stride + p].copy_from_slice(&gram[r * p..(r + 1) * p]);
        aug[r * stride + p + r] = 1.0;
    }
    for col in 0..p {
        let mut pivot = col;
        for r in (col + 1)..p {
            if aug[r * stride + col].abs() > aug[pivot * stride + col].abs() {
                pivot = r;
            }
        }
        if aug[pivot * stride + col].abs() <= EPS {
            return None;
        }
        if pivot != col {
            for c in col..stride {
                aug.swap(col * stride + c, pivot * stride + c);
            }
        }
        let diag = aug[col * stride + col];
        for c in col..stride {
            aug[col * stride + c] /= diag;
        }
        for r in 0..p {
            if r != col {
                let factor = aug[r * stride + col];
                if factor != 0.0 {
                    for c in col..stride {
                        aug[r * stride + c] -= factor * aug[col * stride + c];
                    }
                }
            }
        }
    }
    let mut inv = vec![0.0f64; p * p];
    for r in 0..p {
        inv[r * p..(r + 1) * p].copy_from_slice(&aug[r * stride + p..r * stride + 2 * p]);
    }
    Some(inv)
}

fn build_tlq_day_meta(
    t: usize,
    n: usize,
    k: usize,
    restrict: &[f64],
    industry: &[f64],
    barra_flat: &[f64],
) -> Vec<TlqDayMeta> {
    let p = k + 1;
    (0..t)
        .into_par_iter()
        .map(|row| {
            let irow = &industry[row * n..(row + 1) * n];
            let rrow = &restrict[row * n..(row + 1) * n];
            let bslice = &barra_flat[row * n * k..(row + 1) * n * k];

            let order_l2 = counting_order_l2(irow, n);
            let mut l0_ranges = Vec::new();
            let mut start = 0usize;
            while start < order_l2.len() {
                let first_code = code_l2(irow[order_l2[start]]);
                let mut end = start;
                while end + 1 < order_l2.len() && code_l2(irow[order_l2[end + 1]]) == first_code {
                    end += 1;
                }
                l0_ranges.push((start, end + 1));
                start = end + 1;
            }
            let mut l1_ranges = Vec::new();
            start = 0;
            while start < order_l2.len() {
                let first_code = code_l1(irow[order_l2[start]]);
                let mut end = start;
                while end + 1 < order_l2.len() && code_l1(irow[order_l2[end + 1]]) == first_code {
                    end += 1;
                }
                l1_ranges.push((start, end + 1));
                start = end + 1;
            }

            let mut ind_unknown = Vec::new();
            let mut ind_known = Vec::new();
            let mut l1_nan_cols = Vec::new();
            for j in 0..n {
                if irow[j].is_finite() {
                    ind_known.push(j);
                } else {
                    ind_unknown.push(j);
                }
                if code_l1(irow[j]).is_nan() {
                    l1_nan_cols.push(j);
                }
            }

            let mut base_idx = Vec::with_capacity(n);
            for j in 0..n {
                if !(rrow[j].is_finite() && rrow[j] == 0.0) {
                    continue;
                }
                let mut ok = true;
                for c in 0..k {
                    if !bslice[j * k + c].is_finite() {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    base_idx.push(j);
                }
            }
            let m = base_idx.len();
            let mut base_design = vec![0.0f64; m * p];
            let base_gram_inv = if m > 10 && m >= p {
                let mut gram = vec![0.0f64; p * p];
                for (i, &j) in base_idx.iter().enumerate() {
                    let drow = &mut base_design[i * p..(i + 1) * p];
                    drow[0] = 1.0;
                    drow[1..p].copy_from_slice(&bslice[j * k..j * k + k]);
                    for a in 0..p {
                        let va = drow[a];
                        for b in a..p {
                            gram[a * p + b] += va * drow[b];
                        }
                    }
                }
                for a in 0..p {
                    for b in 0..a {
                        gram[a * p + b] = gram[b * p + a];
                    }
                }
                invert_spd_or_gauss(&gram, p)
            } else {
                None
            };

            TlqDayMeta {
                order_l2,
                l0_ranges,
                l1_ranges,
                ind_unknown,
                ind_known,
                l1_nan_cols,
                base_idx,
                base_design,
                base_gram_inv,
            }
        })
        .collect()
}

/// 单线程处理单个因子的完整 TLQ（供外层跨因子批量并行调用，消除嵌套线程池争抢与重复预分解）。
#[allow(clippy::too_many_arguments)]
fn compute_one_factor_tlq_serial(
    mut value: Array2<f64>,
    day_meta: &[TlqDayMeta],
    all_cols: &[usize],
    size_ranked: &[f64],
    barra_flat: &[f64],
    restrict: &[f64],
    log_exret: &[f64],
    week_first: &[i32],
    k: usize,
    group_from: usize,
    group_to: usize,
    curve_points: usize,
) -> (f64, f64, Array2<f64>) {
    let (t, n) = value.dim();
    let p = k + 1;
    let mut rank_idx: Vec<usize> = Vec::with_capacity(n);
    let mut seen: Vec<f64> = Vec::with_capacity(10);
    let mut design: Vec<f64> = Vec::with_capacity(n * p);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    let mut missing: Vec<usize> = Vec::with_capacity(n);
    let mut srow_buf: Vec<f64> = vec![0.0f64; n];
    let mut med_buf: Vec<f64> = Vec::with_capacity(n);
    let mut resid_row: Vec<f64> = vec![f64::NAN; n];
    let mut idx_fallback: Vec<usize> = Vec::with_capacity(n);
    let mut flag_fallback: Vec<bool> = vec![false; n];
    let mut rhs: Vec<f64> = vec![0.0f64; p];
    let mut coef: Vec<f64> = vec![0.0f64; p];

    let rank_row_with_buf = |vals: &mut [f64], idx: &mut Vec<usize>| {
        idx.clear();
        for (i, &v) in vals.iter().enumerate() {
            if v.is_finite() {
                idx.push(i);
            }
        }
        let nv = idx.len();
        if nv == 0 {
            return;
        }
        idx.sort_unstable_by_key(|&a| mono_key(vals[a]));
        let mut i = 0;
        while i < nv {
            let mut j = i;
            while j + 1 < nv && vals[idx[j + 1]] == vals[idx[i]] {
                j += 1;
            }
            let pct = (((i + 1) + (j + 1)) as f64 / 2.0) / nv as f64;
            for pos in i..=j {
                vals[idx[pos]] = pct;
            }
            i = j + 1;
        }
    };

    let v_all = value.as_slice_mut().expect("矩阵连续");
    for row in 0..t {
        let vrow = &mut v_all[row * n..(row + 1) * n];
        let dm = &day_meta[row];
        let brow = &size_ranked[row * n..(row + 1) * n];
        let rrow = &restrict[row * n..(row + 1) * n];

        // 1) 初始 rank pct
        rank_row_with_buf(vrow, &mut rank_idx);

        // 2) 三级行业回归填充
        seen.clear();
        for &x in vrow.iter() {
            if x.is_finite() && !seen.iter().any(|&s| s == x) {
                seen.push(x);
                if seen.len() >= 10 {
                    break;
                }
            }
        }
        if seen.len() >= 10 {
            for &(s, e) in &dm.l0_ranges {
                fill_one_group(&dm.order_l2[s..e], vrow, brow, &mut design, &mut y, &mut missing);
            }
            for &(s, e) in &dm.l1_ranges {
                fill_one_group(&dm.order_l2[s..e], vrow, brow, &mut design, &mut y, &mut missing);
            }
            fill_one_group(&dm.ind_unknown, vrow, brow, &mut design, &mut y, &mut missing);
            fill_one_group(&dm.ind_known, vrow, brow, &mut design, &mut y, &mut missing);
        }

        // 3) 三级行业中位数填充（基于填充前快照 srow_buf）
        srow_buf.copy_from_slice(vrow);
        for &j in &dm.l1_nan_cols {
            vrow[j] = f64::NAN;
        }
        for &(s, e) in &dm.l0_ranges {
            fill_median_group(&dm.order_l2[s..e], vrow, &srow_buf, &mut med_buf);
        }
        for &(s, e) in &dm.l1_ranges {
            fill_median_group(&dm.order_l2[s..e], vrow, &srow_buf, &mut med_buf);
        }
        fill_median_group(all_cols, vrow, &srow_buf, &mut med_buf);

        // 4) 限制股置空 + 二次 rank pct
        for j in 0..n {
            if !(rrow[j].is_finite() && rrow[j] == 0.0) {
                vrow[j] = f64::NAN;
            }
        }
        rank_row_with_buf(vrow, &mut rank_idx);

        // 5) Barra 线性中性化（优先走每日预求逆快路径，遇缺值或退化无缝回退精确消元）
        let fast_ok = dm.base_gram_inv.is_some()
            && dm.base_idx.len() > 10
            && dm.base_idx.iter().all(|&j| vrow[j].is_finite());

        if fast_ok {
            let m = dm.base_idx.len();
            let first = vrow[dm.base_idx[0]];
            if dm.base_idx.iter().all(|&j| vrow[j] == first) {
                vrow.fill(f64::NAN);
                for &j in &dm.base_idx {
                    vrow[j] = 0.5;
                }
            } else {
                rhs.fill(0.0);
                for i in 0..m {
                    let y_val = vrow[dm.base_idx[i]];
                    let drow = &dm.base_design[i * p..(i + 1) * p];
                    for a in 0..p {
                        rhs[a] += drow[a] * y_val;
                    }
                }
                let inv = dm.base_gram_inv.as_ref().unwrap();
                for a in 0..p {
                    let mut acc = 0.0f64;
                    for b in 0..p {
                        acc += inv[a * p + b] * rhs[b];
                    }
                    coef[a] = acc;
                }
                resid_row.fill(f64::NAN);
                for i in 0..m {
                    let j = dm.base_idx[i];
                    let drow = &dm.base_design[i * p..(i + 1) * p];
                    let mut pred = coef[0];
                    for c in 1..p {
                        pred += coef[c] * drow[c];
                    }
                    resid_row[j] = vrow[j] - pred;
                }
                vrow.copy_from_slice(&resid_row);
            }
        } else {
            let bslice = &barra_flat[row * n * k..(row + 1) * n * k];
            flag_fallback.fill(false);
            idx_fallback.clear();
            for j in 0..n {
                let mut ok = vrow[j].is_finite();
                for c in 0..k {
                    ok &= bslice[j * k + c].is_finite();
                }
                if ok {
                    flag_fallback[j] = true;
                    idx_fallback.push(j);
                }
            }
            let m = idx_fallback.len();
            if m <= 10 {
                vrow.fill(f64::NAN);
            } else {
                let first = vrow[idx_fallback[0]];
                if idx_fallback.iter().all(|&j| vrow[j] == first) {
                    for j in 0..n {
                        vrow[j] = if flag_fallback[j] { 0.5 } else { f64::NAN };
                    }
                } else {
                    design.resize(m * p, 0.0);
                    y.resize(m, 0.0);
                    for (i, &j) in idx_fallback.iter().enumerate() {
                        design[i * p] = 1.0;
                        design[i * p + 1..i * p + p].copy_from_slice(&bslice[j * k..j * k + k]);
                        y[i] = vrow[j];
                    }
                    resid_row.fill(f64::NAN);
                    if let Some(sol) = lstsq_solve(&design, m, p, &y) {
                        for (i, &j) in idx_fallback.iter().enumerate() {
                            let mut pred = sol[0];
                            for c in 0..k {
                                pred += sol[c + 1] * design[i * p + c + 1];
                            }
                            resid_row[j] = y[i] - pred;
                        }
                    }
                    vrow.copy_from_slice(&resid_row);
                }
            }
        }

        // 残差 rank pct + group_returns 前置过滤合并在单趟完成
        rank_row_with_buf(vrow, &mut rank_idx);
        let mut cnt = 0usize;
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        for &x in vrow.iter() {
            if x.is_finite() {
                cnt += 1;
                sum += x;
                sum2 += x * x;
            }
        }
        let zero_var = cnt > 1 && ((sum2 - sum * sum / cnt as f64) / (cnt - 1) as f64) == 0.0;
        if zero_var {
            vrow.fill(f64::NAN);
        } else {
            for j in 0..n {
                if !(rrow[j].is_finite() && rrow[j] == 0.0) {
                    vrow[j] = f64::NAN;
                }
            }
            rank_row_with_buf(vrow, &mut rank_idx);
            let cnt2 = vrow.iter().filter(|x| x.is_finite()).count();
            if cnt2 <= 10 {
                vrow.fill(f64::NAN);
            }
        }
    }

    // 6) 周频十分组日收益 + TLQ
    let v_slice = value.as_slice().expect("矩阵连续");
    let gnum_all: Vec<i32> = v_slice
        .iter()
        .map(|&pct| {
            if pct.is_finite() {
                10 - ((1.0 - pct) * 10.0).floor() as i32
            } else {
                -9999
            }
        })
        .collect();

    let n_ret_rows = group_to.saturating_sub(group_from);
    let mut rets = Array2::<f64>::zeros((n_ret_rows, 10));
    for (out_r, row) in (group_from..group_to).enumerate() {
        let mut acc = [0.0f64; 10];
        let mut cnt = [0usize; 10];
        let wf = week_first[row];
        if wf >= 1 {
            let prev = (wf as usize - 1) * n;
            let row_off = row * n;
            for j in 0..n {
                if !v_slice[row_off + j].is_finite() {
                    continue;
                }
                let g = gnum_all[prev + j];
                if !(1..=10).contains(&g) {
                    continue;
                }
                let gi = (g - 1) as usize;
                cnt[gi] += 1;
                let r = log_exret[row_off + j];
                if r.is_finite() {
                    acc[gi] += r;
                }
            }
        }
        for g in 0..10 {
            rets[[out_r, g]] = if cnt[g] == 0 {
                0.0
            } else {
                acc[g] / cnt[g] as f64
            };
        }
    }

    let tlq = tlq_from_returns(&rets, curve_points);
    let (tlq_seg, _) = tlq_segment_from_returns(&rets, curve_points);
    let (rows, _) = rets.dim();
    let mut curves = Array2::<f64>::zeros((curve_points, 10));
    if rows > 0 && curve_points > 0 {
        let mut cum = vec![0.0f64; rows * 10];
        for g in 0..10 {
            let mut acc = 0.0;
            for i in 0..rows {
                let r = rets[[i, g]];
                acc += if r.is_finite() { r } else { 0.0 };
                cum[i * 10 + g] = acc;
            }
        }
        let pos = sample_positions(rows, curve_points);
        for (kk, &p_idx) in pos.iter().enumerate() {
            for g in 0..10 {
                curves[[kk, g]] = cum[p_idx * 10 + g];
            }
        }
    }
    (tlq, tlq_seg, curves)
}

/// 批量计算多个候选因子的 TLQ（一次预分解每日固定结构 + 单层线程池跨因子并行，支持 f32 输入零额外转换）。
#[pyfunction]
#[pyo3(signature = (factors_f32, log_exret, restrict, industry, barra_ranked, n_styles, size_ranked, week_first, group_from, group_to, curve_points=160, n_threads=32))]
#[allow(clippy::too_many_arguments)]
pub fn tlq_gate_base_resid_batch<'py>(
    py: Python<'py>,
    factors_f32: Vec<PyReadonlyArray2<'py, f32>>,
    log_exret: PyReadonlyArray2<'py, f64>,
    restrict: PyReadonlyArray2<'py, f64>,
    industry: PyReadonlyArray2<'py, f64>,
    barra_ranked: PyReadonlyArray2<'py, f64>,
    n_styles: usize,
    size_ranked: PyReadonlyArray2<'py, f64>,
    week_first: PyReadonlyArray1<'py, i32>,
    group_from: usize,
    group_to: usize,
    curve_points: usize,
    n_threads: usize,
) -> PyResult<Vec<(f64, f64, &'py PyArray2<f64>)>> {
    if factors_f32.is_empty() {
        return Ok(Vec::new());
    }
    let log_exret_arr = log_exret.as_array().as_standard_layout().into_owned();
    let restrict_arr = restrict.as_array().as_standard_layout().into_owned();
    let industry_arr = industry.as_array().as_standard_layout().into_owned();
    let size_arr = size_ranked.as_array().as_standard_layout().into_owned();
    let barra_arr = barra_ranked.as_array().as_standard_layout().into_owned();
    let week_first_vec: Vec<i32> = week_first.as_array().to_vec();

    let (t, n) = restrict_arr.dim();
    if barra_arr.ncols() != n * n_styles {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "barra_ranked 列数必须等于 股票数 × 风格数",
        ));
    }

    let mut owned_factors: Vec<Array2<f64>> = Vec::with_capacity(factors_f32.len());
    for f in &factors_f32 {
        let view = f.as_array();
        if view.dim() != (t, n) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "因子形状 {:?} 与模板 {:?} 不一致",
                view.dim(),
                (t, n)
            )));
        }
        let mut mat = Array2::<f64>::zeros((t, n));
        let dst = mat.as_slice_mut().unwrap();
        if let Some(src) = view.as_slice() {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d = if s.is_finite() { s as f64 } else { f64::NAN };
            }
        } else {
            for (d, &s) in dst.iter_mut().zip(view.iter()) {
                *d = if s.is_finite() { s as f64 } else { f64::NAN };
            }
        }
        owned_factors.push(mat);
    }

    let threads = n_threads.clamp(1, 64).min(owned_factors.len().max(1));
    let results = py
        .allow_threads(move || -> Result<Vec<(f64, f64, Array2<f64>)>, String> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| format!("建 TLQ 线程池失败: {e}"))?;
            let r_slice = restrict_arr.as_slice().unwrap();
            let i_slice = industry_arr.as_slice().unwrap();
            let b_slice = barra_arr.as_slice().unwrap();
            let s_slice = size_arr.as_slice().unwrap();
            let l_slice = log_exret_arr.as_slice().unwrap();
            let all_cols: Vec<usize> = (0..n).collect();

            pool.install(|| {
                let day_meta = build_tlq_day_meta(t, n, n_styles, r_slice, i_slice, b_slice);
                Ok(owned_factors
                    .into_par_iter()
                    .map(|fmat| {
                        compute_one_factor_tlq_serial(
                            fmat,
                            &day_meta,
                            &all_cols,
                            s_slice,
                            b_slice,
                            r_slice,
                            l_slice,
                            &week_first_vec,
                            n_styles,
                            group_from,
                            group_to,
                            curve_points,
                        )
                    })
                    .collect())
            })
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok(results
        .into_iter()
        .map(|(tlq, seg, curves)| (tlq, seg, curves.into_pyarray(py)))
        .collect())
}
