//! TLQ（时序分层质量）复刻 + 它需要的 base_resid 流水线（sandbox 版）。
//!
//! 对齐同事的 factor_gate_pipeline_v20260825.py 与 TLQ_report.pdf：
//!   fill_and_rank: rank pct → 行业回归填充(基准 = size 的 rank pct，3 级) → 行业中位填充(3 级)
//!                  → 限制股置空 → rank pct
//!   base_resid   : 对 10 个 Barra（各自先 rank pct）加截距做 OLS，取残差 → rank pct
//!   分组         : 残差再 rank pct（当天有效股票 <= 10 只的整行置空）→ 十分组
//!                  → 自然周再平衡（每周沿用上一交易日的分组）→ 组内等权
//!   组日收益     : Σ(有限收益) / 组内股票数；收益口径由 Python 侧给（对数超额）
//!   TLQ          : 十组累计曲线取 160 个等距点，端点分离 × 相邻组方向保留比例，双向取最大
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

/// 排序求中位数（偶数取中间两个的平均）。
fn median_sorted(buf: &mut Vec<f64>) -> Option<f64> {
    if buf.is_empty() {
        return None;
    }
    buf.sort_by(|a, b| a.total_cmp(b));
    let n = buf.len();
    Some(if n % 2 == 1 {
        buf[n / 2]
    } else {
        (buf[n / 2 - 1] + buf[n / 2]) / 2.0
    })
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
pub fn sandbox_rank_pct_rows<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let mut m = matrix.as_array().to_owned();
    py.allow_threads(|| rank_pct_all(&mut m));
    Ok(m.into_pyarray(py))
}

/// 把 k 张 (T × N) 矩阵交错成 (T × N*k) 的行主序块，供回测逐行取用。
#[pyfunction]
pub fn sandbox_interleave_styles<'py>(
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
pub fn sandbox_tlq_base_resid<'py>(
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
