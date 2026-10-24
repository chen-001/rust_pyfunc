//! percolation: 边权渗流阈值扫描（统计物理扩展，20251031 原型验证版）。
//! 对对称矩阵 S_sym = S_dir + S_dirᵀ（对角 0）取上三角正权边，按权值降序做增量并查集:
//!   theta* = 最大连通分量首次 ≥ N/2 时的边权 / w_max（归一，无量纲），逐股同值;
//!   birth_i = i 所在分量并入 anchor(全市场 strength 最大股, 并列取小索引) 分量时的边权/w_max;
//!   stranded_i = rmax_i − birth_i（未归一原始权差; rmax=行最大非对角元）。
//!   到 theta* 仍未并入主干的股 birth=NaN → stranded=NaN; 孤立股 rmax=NaN → stranded=NaN。
//!   width  = θ(maxC≥25%N) − θ(maxC≥75%N) 归一区间宽（转变陡缓，标量广播）;
//!   comp_i = θ* 之上（w>θ*）无权图中 i 的连通块大小/N（局部碎片度）。
//! 物理含义: theta* 刻画骨架成形位置（靠少数强边还是均摊）; width 刻画转变陡缓;
//! birth 与行最大强相关（原型 |rho|≈0.93），增量在 stranded——"有强边却接不进主干";
//! comp 把每股贴回破碎/凝聚侧（同在巨块=同值, 离群者各异）。
//! 确定性: 边按 (w 降序, i, j 升序) 全序排序, 并查集按序处理, 路径压缩+按大小合并;
//! birth 只在 θ* 首次跨越前的合并中记录（与第一版语义一致）。

use crate::yupei_dist::indicator_ctx::{IndicatorCtx, IndicatorResult};
use rayon::prelude::*;

pub fn name() -> &'static str {
    "percolation"
}

pub fn desc() -> &'static str {
    "渗流扫描: theta*+stranded(rmax−birth) (28矩阵)"
}

struct Uf {
    par: Vec<u32>,
    sz: Vec<u32>,
    head: Vec<u32>,       // 分量成员链表头
    nxt: Vec<i32>,        // 链表 next, -1 结尾
    tail: Vec<u32>,
    has_anchor: Vec<u8>,
    birth: Vec<f32>,      // NaN 初始化
    wmax_inv: f32,
}

impl Uf {
    fn new(n: usize, anchor: u32, wmax_inv: f32) -> Self {
        let mut has_anchor = vec![0u8; n];
        if (anchor as usize) < n {
            has_anchor[anchor as usize] = 1;
        }
        Uf {
            par: (0..n as u32).collect(),
            sz: vec![1; n],
            head: (0..n as u32).collect(),
            nxt: vec![-1; n],
            tail: (0..n as u32).collect(),
            has_anchor,
            birth: vec![f32::NAN; n],
            wmax_inv,
        }
    }

    #[inline]
    fn find(&mut self, x: u32) -> u32 {
        let mut r = x;
        while self.par[r as usize] != r {
            r = self.par[r as usize];
        }
        // 路径压缩
        let mut x = x;
        while self.par[x as usize] != r {
            let p = self.par[x as usize];
            self.par[x as usize] = r;
            x = p;
        }
        r
    }

    /// 合并含 a,b 的两分量（调用方保证不同根）; 返回新根。
    /// record_birth 为真时, 无 anchor 侧成员 birth ← w*wmax_inv。
    fn union(&mut self, a: u32, b: u32, w: f32, record_birth: bool) -> u32 {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return ra;
        }
        if self.sz[ra as usize] < self.sz[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        let ha = self.has_anchor[ra as usize];
        let hb = self.has_anchor[rb as usize];
        if record_birth && ha != hb {
            let other = if ha == 1 { rb } else { ra } as usize;
            let mut t = self.head[other] as i32;
            while t != -1 {
                self.birth[t as usize] = w * self.wmax_inv;
                t = self.nxt[t as usize];
            }
        }
        self.par[rb as usize] = ra;
        self.sz[ra as usize] += self.sz[rb as usize];
        let tb = self.tail[rb as usize];
        let hb_head = self.head[rb as usize];
        self.nxt[self.tail[ra as usize] as usize] = hb_head as i32;
        self.tail[ra as usize] = tb;
        self.has_anchor[ra as usize] = ha | hb;
        ra
    }

    #[inline]
    fn comp_size(&mut self, x: u32) -> u32 {
        let r = self.find(x);
        self.sz[r as usize]
    }
}

/// 单张对称矩阵的渗流扫描，返回 (theta_star_norm, birth, stranded, width, comp)。
/// width = θ(maxC≥25%N) − θ(maxC≥75%N) 的归一区间宽; comp = θ* 之上连通块大小/N。
/// need_comp=false 时跳过 comp 的第二遍并查集（comp 全 NaN）。
fn scan(
    sym: &[f32],
    n: usize,
    row_sums: &[f64],
    need_comp: bool,
) -> (f32, Vec<f32>, Vec<f32>, f32, Vec<f32>) {
    // anchor = strength 最大股（并列取小索引）
    let mut anchor = 0usize;
    for (i, &s) in row_sums.iter().enumerate() {
        if s > row_sums[anchor] {
            anchor = i;
        }
    }
    // rmax: 行最大非对角元
    let rmax: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &sym[i * n..(i + 1) * n];
            let mut m = f32::NEG_INFINITY;
            for (j, &v) in row.iter().enumerate() {
                if j != i && v > m {
                    m = v;
                }
            }
            if m == f32::NEG_INFINITY {
                f32::NAN
            } else {
                m
            }
        })
        .collect();
    // 边表: (w, i, j), i<j, w>0; 全序排序 w 降序 + (i,j) 升序（partial_cmp NaN→Equal）
    let mut edges: Vec<(f32, u32, u32)> = Vec::with_capacity(n * n / 4);
    for i in 0..n {
        let row = &sym[i * n..(i + 1) * n];
        for j in (i + 1)..n {
            let w = row[j];
            if w > 0.0 {
                edges.push((w, i as u32, j as u32));
            }
        }
    }
    edges.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    if edges.is_empty() {
        return (
            f32::NAN,
            vec![f32::NAN; n],
            vec![f32::NAN; n],
            f32::NAN,
            vec![f32::NAN; n]);
    }
    let wmax = edges[0].0;
    let wmax_inv = 1.0 / wmax;
    let mut uf = Uf::new(n, anchor as u32, wmax_inv);
    let mut maxc = 1u32;
    let mut theta_star = f32::NAN;
    let mut t25 = f32::NAN;
    let mut t75 = f32::NAN;
    let n4 = n as f64 * 0.25;
    let n34 = n as f64 * 0.75;
    for &(w, i, j) in &edges {
        // birth 只记 θ* 之前的并入（保持第一版语义: 到 θ* 未并入记 NaN）
        let record = theta_star.is_nan();
        let r = uf.union(i, j, w, record);
        let s = uf.comp_size(r);
        if s > maxc {
            maxc = s;
            if t25.is_nan() && (maxc as f64) >= n4 {
                t25 = w * wmax_inv;
            }
            if t75.is_nan() && (maxc as f64) >= n34 {
                t75 = w * wmax_inv;
            }
            if theta_star.is_nan() && (maxc as usize) * 2 >= n {
                theta_star = w * wmax_inv;
            }
            if t75.is_finite() {
                break; // θ* 与宽度都已定, 后续合并不再影响任何输出
            }
        }
    }
    // anchor 记归一化最大边权 1.0（对齐原型约定 birth[anchor]=ws[0]）
    uf.birth[anchor as usize] = 1.0;
    let stranded: Vec<f32> = (0..n)
        .map(|i| {
            if rmax[i].is_nan() || uf.birth[i].is_nan() {
                f32::NAN
            } else {
                // 原始权差: rmax 与 birth·wmax 同量纲
                rmax[i] - uf.birth[i] * wmax
            }
        })
        .collect();
    // 转变宽度: maxC 从 25%N 到 75%N 跨过的归一 θ 区间（未达 75% 记 NaN）
    let width = if t25.is_finite() && t75.is_finite() {
        t25 - t75
    } else {
        f32::NAN
    };
    // comp: θ* 之上（w > θ* 原始权）无权图的每股连通块大小 / N，第二遍并查集
    let mut comp = vec![f32::NAN; n];
    if need_comp && theta_star.is_finite() {
        let tw = theta_star * wmax;
        let mut uf2 = Uf::new(n, u32::MAX, wmax_inv); // anchor 无意义, 仅复用结构
        for &(w, i, j) in &edges {
            if w <= tw {
                break; // 边降序, 小于等于阈值即停
            }
            uf2.union(i, j, w, false);
        }
        for i in 0..n {
            comp[i] = uf2.comp_size(i as u32) as f32 / n as f32;
        }
    }
    (theta_star, uf.birth, stranded, width, comp)
}

pub fn compute(ctx: &IndicatorCtx) -> Vec<IndicatorResult> {
    let n = ctx.n();
    if n == 0 {
        return Vec::new();
    }
    // 矩阵循环并行: 每张矩阵的边排序/并查集是模块内最大开销, 28 张互相独立。
    // par_iter().map().collect() 保序（MATRIX_LIST 序），输出顺序与串行一致。
    let per_mat: Vec<Vec<IndicatorResult>> = crate::yupei_dist::indicator_ctx::MATRIX_LIST
        .par_iter()
        .filter(|mat| ctx.wants("perc_", mat))
        .map(|mat| {
            let sym = match ctx.symmetric(mat) {
                Some(s) => s,
                None => return Vec::new(),
            };
            let row_sums = match ctx.row_sum_sym(mat) {
                Some(s) => s,
                None => return Vec::new(),
            };
            // comp 列未幸存时跳过其第二遍并查集
            let need_comp = match &ctx.keep {
                None => true,
                Some(k) => k.contains(format!("perc_{mat}_comp").as_str()),
            };
            let (theta_star, _birth, stranded, width, comp) =
                scan(&sym, n, &row_sums, need_comp);
            let mut out = Vec::new();
            emit(ctx, &mut out, format!("perc_{mat}_theta"), vec![theta_star; n]);
            emit(ctx, &mut out, format!("perc_{mat}_stranded"), stranded);
            emit(ctx, &mut out, format!("perc_{mat}_width"), vec![width; n]);
            emit(ctx, &mut out, format!("perc_{mat}_comp"), comp);
            out
        })
        .collect();
    // 注: filter 后 collect 只含被算矩阵, 但每张矩阵内部列序不变;
    // run_indicators 按名字对账/注册序重排，矩阵间相对顺序不进入下游。
    per_mat.into_iter().flatten().collect()
}

/// 列级过滤: keep=None 全发; keep=Some 只发名单内列。
fn emit(ctx: &IndicatorCtx, out: &mut Vec<IndicatorResult>, name: String, values: Vec<f32>) {
    if let Some(k) = &ctx.keep {
        if !k.contains(name.as_str()) {
            return;
        }
    }
    out.push(IndicatorResult::new(name, values));
}
