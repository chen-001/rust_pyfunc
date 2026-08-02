//! 一呼百应（yhyb）第 4 层：网络级因子（纯 Rust，p0 全天）。
//!
//! 在 (A,B) **对级响应标量**矩阵上直接计算（矩阵在阶段 A 的对级累加中构建，
//! 不依赖 1380 聚合因子的输出）：
//! - lead：引领度 = mean_B(bwd 对级均值) − mean_B(fwd 对级均值)，正 = 别人响应它更快（龙头）
//! - hub：辐射广度 = fwd 命中（B 在 60s 内响应过 A）的 B 占比
//! - spoke：跟随广度 = bwd 命中（A 在 60s 内响应过 B）的 B 占比
//! - pr_fwd / pr_bwd：响应网络的 PageRank（top-50 邻接稀疏化，边权 = 1/(1+距离)）
//! - heter：响应异质性 = std_B(fwd 对级均值)（A 的传导速度在 B 间的离散度）
//! - lead_all：23 事件引领度平均；pca1：46 列（23 事件 × fwd/bwd 对级均值）
//!   横截面 PCA 第一主成分（综合协同指数）
//!
//! 输出由 yhyb_metrics 合并进 1520 因子（1380 + 140）统一返回，不再单独暴露 py 入口。

use crate::yhyb_metrics::{day_base, period_slice, N_EVENTS};
use rayon::prelude::*;

/// 第 4 层因子数：23 事件 × 6 + 2 全局 = 140
pub const N_L4: usize = N_EVENTS * 6 + 2;

/// 命中窗口（秒）：与 1380 因子的默认 hit_t_s=60 一致
const HIT_T_US: u64 = 60_000_000;
/// PageRank top-K 邻接（每节点最强传导对象数）
const PR_TOP_K: usize = 50;
const PR_DAMPING: f64 = 0.85;
const PR_ITERS: usize = 30;

pub fn l4_names() -> Vec<String> {
    let mut names = Vec::with_capacity(N_L4);
    for e in 0..N_EVENTS {
        for m in ["lead", "hub", "spoke", "pr_fwd", "pr_bwd", "heter"] {
            names.push(format!("yhyb4_e{e:02}_{m}"));
        }
    }
    names.push("yhyb4_lead_all".into());
    names.push("yhyb4_pca1".into());
    names
}

/// 裸指针的 Send/Sync 包装（rayon 闭包要求 Send+Sync；行/列独占写入保证无数据竞争）。
/// 注意：必须通过方法访问字段——Rust 2021 精确捕获会直接捕获 `ptr` 字段（*mut f32 非
/// Send），方法调用则强制捕获整个 SendPtr（Send+Sync）。
#[derive(Clone, Copy)]
pub struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    pub fn new(ptr: *mut f32) -> SendPtr {
        SendPtr(ptr)
    }
    #[inline(always)]
    pub fn w(&self, off: usize, v: f32) {
        unsafe {
            *self.0.add(off) = v;
        }
    }
    #[inline(always)]
    pub fn r(&self, off: usize) -> f32 {
        unsafe { *self.0.add(off) }
    }
    #[inline(always)]
    pub fn s(&self, off: usize, len: usize) -> &'static mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

/// 第 4 层计算核心（纯内存，供 yhyb_metrics 合并输出 1520 因子时调用）。
/// 返回 (vals140, valid 全量索引)。
pub fn compute_l4(
    codes: &[String],
    streams: &[Option<[crate::yhyb_metrics::EvStream; N_EVENTS]>],
) -> (Vec<f32>, Vec<usize>) {
    let n_all = codes.len();
    // 有效股票（有事件流）：与 1380 因子输出一致（过滤停牌/无逐笔数据）
    let valid: Vec<usize> = streams
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_some())
        .map(|(i, _)| i)
        .collect();
    let n = valid.len();
    let mut valid_pos = vec![usize::MAX; n_all];
    for (pos, &i) in valid.iter().enumerate() {
        valid_pos[i] = pos;
    }
    let ne = N_EVENTS;
    if std::env::var("YHYB4_DEBUG").is_ok() {
        eprintln!("YHYB4_DEBUG n={n} ne={ne} m_fwd_len={}", ne * n * n);
    }
    // 对级均值矩阵：M[e][A][B]（f32；NaN = 该对无响应距离）
    // 6.5GB（23 × 2 × 5914² × 4B）——512 核共享机内存充足，一次性分配
    let mut m_fwd = vec![f32::NAN; ne * n * n];
    let mut m_bwd = vec![f32::NAN; ne * n * n];
    let mut hub = vec![f32::NAN; ne * n];
    let mut spoke = vec![f32::NAN; ne * n];
    // 阶段 A：并行 (A, e) 任务；每个任务独占矩阵第 (e,A) 行（SendPtr 行写入，无竞争）
    let mf = SendPtr(m_fwd.as_mut_ptr());
    let mb = SendPtr(m_bwd.as_mut_ptr());
    let hp = SendPtr(hub.as_mut_ptr());
    let sp = SendPtr(spoke.as_mut_ptr());
    let valid_out = valid.clone(); // 阶段 A 闭包 move 捕获 valid，返回用副本
    (0..n * ne).into_par_iter().for_each(move |idx| {
        let ai = idx / ne;
        let e = idx % ne;
        let Some(sa) = streams[valid[ai]].as_ref() else { return };
        let ta = &sa[e].t;
        if ta.is_empty() {
            return;
        }
        let base = day_base(ta[0]);
        let (alo, ahi) = period_slice(ta, base, 0);
        if ahi == alo {
            return;
        }
        let mut f_hit_b = 0u64; // 命中率 > 0 的 B 数（hub 用）
        let mut b_hit_b = 0u64; // bwd 命中率 > 0 的 B 数（spoke 用）
        let mut n_b = 0u64;
        for (bi, sb) in streams.iter().enumerate() {
            if bi == valid[ai] {
                continue;
            }
            let Some(sb) = sb else { continue };
            let bpos = valid_pos[bi];
            if bpos == usize::MAX {
                continue;
            }
            let tb = &sb[e].t;
            let (blo, bhi) = period_slice(tb, base, 0);
            if blo == bhi {
                continue;
            }
            let mut j = blo;
            let mut fs = 0.0f64;
            let mut bs = 0.0f64;
            let mut fn_ = 0u64;
            let mut bn_ = 0u64;
            let mut fh = 0u64;
            let mut bh = 0u64;
            for i in alo..ahi {
                let a = ta[i];
                while j < bhi && tb[j] <= a {
                    j += 1;
                }
                if j > blo {
                    let d = (a - tb[j - 1]) as u64;
                    bs += d as f64;
                    bn_ += 1;
                    if d <= HIT_T_US {
                        bh += 1;
                    }
                }
                if j < bhi {
                    let d = (tb[j] - a) as u64;
                    fs += d as f64;
                    fn_ += 1;
                    if d <= HIT_T_US {
                        fh += 1;
                    }
                }
            }
            // 本任务独占第 (e,ai) 行：写入 B 列（bpos = 有效索引）
            let vf = if fn_ > 0 { (fs / fn_ as f64) as f32 } else { f32::NAN };
            let vb = if bn_ > 0 { (bs / bn_ as f64) as f32 } else { f32::NAN };
            mf.w((e * n + ai) * n + bpos, vf);
            mb.w((e * n + ai) * n + bpos, vb);
            if fh > 0 {
                f_hit_b += 1;
            }
            if bh > 0 {
                b_hit_b += 1;
            }
            n_b += 1;
        }
        hp.w(e * n + ai, if n_b > 0 { f_hit_b as f32 / n_b as f32 } else { f32::NAN });
        sp.w(e * n + ai, if n_b > 0 { b_hit_b as f32 / n_b as f32 } else { f32::NAN });
    });
    // 阶段 B：从对级矩阵计算因子（v1 生产路径由 yhyb_metrics 的 p0 融合任务
    // 直接构建矩阵后调用本函数；v2 路径由 compute_l4 内部调用）
    l4_factors(&m_fwd, &m_bwd, &hub, &spoke, n, valid_out)
}

/// 第 4 层阶段 B：从对级矩阵（M[e][A][B] f32，NaN = 无响应）计算全部 140 因子。
/// - 每事件 6：lead（对级均值方向差）、hub/spoke（命中 B 占比）、pr_fwd/pr_bwd（PageRank）、
///   heter（对级均值距离的 B 间离散度）
/// - 全局 2：lead_all（跨事件平均引领度）、pca1（46 列横截面 PCA 第一主成分）
/// 返回 vals（n × 140）。
pub fn l4_factors(
    m_fwd: &[f32],
    m_bwd: &[f32],
    hub: &[f32],
    spoke: &[f32],
    n: usize,
    valid: Vec<usize>,
) -> (Vec<f32>, Vec<usize>) {
    let ne = N_EVENTS;
    let mut vals = vec![f32::NAN; n * N_L4];
    let mut pca_f = vec![f32::NAN; ne * n]; // mean_B(fwd 对级均值)：PCA 的 46 列输入
    let mut pca_b = vec![f32::NAN; ne * n];
    let pf = SendPtr(pca_f.as_mut_ptr());
    let pb = SendPtr(pca_b.as_mut_ptr());
    let vp = SendPtr(vals.as_mut_ptr());
    let hp = SendPtr(hub.as_ptr() as *mut f32);
    let sp = SendPtr(spoke.as_ptr() as *mut f32);
    (0..ne).into_par_iter().for_each(move |e| {
        let row_f = &m_fwd[e * n * n..(e + 1) * n * n];
        let row_b = &m_bwd[e * n * n..(e + 1) * n * n];
        // 网络中心性：每事件全图一次 PageRank（fwd/bwd 两个响应网络）
        let pr_f = pagerank(row_f, n);
        let pr_b = pagerank(row_b, n);
        for ai in 0..n {
            let rf = &row_f[ai * n..ai * n + n];
            let rb = &row_b[ai * n..ai * n + n];
            let (mf, _nf) = mean_nonnan(rf);
            let (mb, _nb) = mean_nonnan(rb);
            pf.w(e * n + ai, mf);
            pb.w(e * n + ai, mb);
            // 本任务独占 (e, *) 列：直接写 vals 对应位置（距离单位 µs → 秒）
            let out = vp.s(ai * N_L4 + e * 6, 6);
            // lead：正 = 别人响应它更快（龙头）；hub/spoke 直接取阶段 A
            out[0] = (mb - mf) / 1e6;
            out[1] = hp.r(e * n + ai);
            out[2] = sp.r(e * n + ai);
            out[3] = pr_f[ai];
            out[4] = pr_b[ai];
            out[5] = std_nonnan(rf) / 1e6;
        }
    });
    // 全局因子
    let lead_all: f32 = {
        let mut s = 0.0f64;
        let mut cnt = 0usize;
        for e in 0..ne {
            for ai in 0..n {
                let v = vals[ai * N_L4 + e * 6] as f64;
                if !v.is_nan() {
                    s += v;
                    cnt += 1;
                }
            }
        }
        if cnt > 0 {
            (s / cnt as f64) as f32
        } else {
            f32::NAN
        }
    };
    for ai in 0..n {
        vals[ai * N_L4 + ne * 6 + 0] = lead_all;
    }
    let pca1 = pca_score(&pca_f, &pca_b, n, ne);
    for ai in 0..n {
        vals[ai * N_L4 + ne * 6 + 1] = pca1[ai];
    }
    (vals, valid)
}

/// 非 NaN 均值（返回 (均值, 计数)）。
fn mean_nonnan(x: &[f32]) -> (f32, usize) {
    let mut s = 0.0f64;
    let mut c = 0usize;
    for &v in x {
        if !v.is_nan() {
            s += v as f64;
            c += 1;
        }
    }
    if c > 0 {
        ((s / c as f64) as f32, c)
    } else {
        (f32::NAN, 0)
    }
}

/// 非 NaN 标准差（B 间离散度）。
fn std_nonnan(x: &[f32]) -> f32 {
    let (m, c) = mean_nonnan(x);
    if c < 2 || m.is_nan() {
        return f32::NAN;
    }
    let mut s = 0.0f64;
    for &v in x {
        if !v.is_nan() {
            let d = v as f64 - m as f64;
            s += d * d;
        }
    }
    (s / (c - 1) as f64).sqrt() as f32
}

/// PageRank：节点 = 股票；节点 A 的出边 = 距离最小（传导最强）的 top-K 个 B，
/// 边权 w = 1/(1+d) 归一化。返回全体节点的 PR（f32）。
fn pagerank(row_block: &[f32], n: usize) -> Vec<f32> {
    // 每节点 top-K 邻接（距离升序选 K 个非 NaN）
    let mut adj: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n);
    for a in 0..n {
        let row = &row_block[a * n..a * n + n];
        let mut cand: Vec<(f32, u32)> = row
            .iter()
            .enumerate()
            .filter(|(_, &v)| !v.is_nan())
            .map(|(i, &v)| (v, i as u32))
            .collect();
        if cand.is_empty() {
            adj.push(Vec::new());
            continue;
        }
        cand.sort_unstable_by(|x, y| x.0.total_cmp(&y.0));
        cand.truncate(PR_TOP_K);
        let wsum: f64 = cand.iter().map(|(d, _)| 1.0 / (1.0 + *d as f64)).sum();
        let edges: Vec<(u32, f32)> = cand
            .into_iter()
            .map(|(d, b)| (b, (1.0 / (1.0 + d as f64) / wsum) as f32))
            .collect();
        adj.push(edges);
    }
    let mut pr = vec![1.0f64 / n as f64; n];
    for _ in 0..PR_ITERS {
        let mut npr = vec![(1.0 - PR_DAMPING) / n as f64; n];
        for (a, edges) in adj.iter().enumerate() {
            if edges.is_empty() {
                continue;
            }
            let pa = pr[a] * PR_DAMPING;
            for &(b, w) in edges {
                npr[b as usize] += pa * w as f64;
            }
        }
        pr = npr;
    }
    pr.iter().map(|&v| v as f32).collect()
}

/// 横截面 PCA：46 列（23 事件 × fwd/bwd 对级均值 mean_B）标准化后
/// 求协方差矩阵第一主成分（幂迭代），得分 = 标准化 X × 第一特征向量。
/// NaN 列用全市场列均值填充（保证全体股票有值）。
fn pca_score(pca_f: &[f32], pca_b: &[f32], n: usize, ne: usize) -> Vec<f32> {
    let nc = 2 * ne; // 46 列
    // 列均值/标准差（跳过 NaN）
    let mut col_mean = vec![0.0f64; nc];
    let mut col_cnt = vec![0usize; nc];
    for c in 0..nc {
        let src = if c < ne { &pca_f[c * n..(c + 1) * n] } else { &pca_b[(c - ne) * n..(c - ne + 1) * n] };
        let (m, cnt) = mean_nonnan(src);
        col_mean[c] = m as f64;
        col_cnt[c] = cnt;
    }
    let mut col_std = vec![0.0f64; nc];
    for c in 0..nc {
        let src = if c < ne { &pca_f[c * n..(c + 1) * n] } else { &pca_b[(c - ne) * n..(c - ne + 1) * n] };
        let mut s = 0.0f64;
        let m = col_mean[c];
        for &v in src {
            if !v.is_nan() {
                let d = v as f64 - m;
                s += d * d;
            }
        }
        col_std[c] = if col_cnt[c] > 1 { (s / (col_cnt[c] - 1) as f64).sqrt() } else { 1.0 };
    }
    // 标准化矩阵 X[n][46]（NaN → 0，即列均值填充后的中心化值）
    let mut x = vec![0.0f64; n * nc];
    for c in 0..nc {
        let src = if c < ne { &pca_f[c * n..(c + 1) * n] } else { &pca_b[(c - ne) * n..(c - ne + 1) * n] };
        for a in 0..n {
            let v = src[a];
            x[a * nc + c] = if v.is_nan() {
                0.0
            } else {
                (v as f64 - col_mean[c]) / col_std[c].max(1e-12)
            };
        }
    }
    // 协方差矩阵 C[46][46]（对称，只算上三角）
    let mut c = vec![0.0f64; nc * nc];
    for i in 0..nc {
        for j in i..nc {
            let mut s = 0.0f64;
            for a in 0..n {
                s += x[a * nc + i] * x[a * nc + j];
            }
            s /= n as f64;
            c[i * nc + j] = s;
            c[j * nc + i] = s;
        }
    }
    // 幂迭代第一特征向量（30 次）
    let mut v = vec![1.0f64 / (nc as f64).sqrt(); nc];
    for _ in 0..30 {
        let mut nv = vec![0.0f64; nc];
        for i in 0..nc {
            for j in 0..nc {
                nv[i] += c[i * nc + j] * v[j];
            }
        }
        let norm = nv.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-12);
        for x in nv.iter_mut() {
            *x /= norm;
        }
        v = nv;
    }
    // 得分 = X · v（符号对齐：使大市值侧……简单起见取绝对值方向固定：v[0] > 0）
    if v[0] < 0.0 {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
    (0..n)
        .map(|a| {
            let mut s = 0.0f64;
            for c_ in 0..nc {
                s += x[a * nc + c_] * v[c_];
            }
            s as f32
        })
        .collect()
}
