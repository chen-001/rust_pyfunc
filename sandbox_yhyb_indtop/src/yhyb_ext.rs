//! 一呼百应补充因子（COR-39 行业头部配对池拓展）：池过滤版配对统计。
//!
//! 设计报告 design_report_yhyb_indtop.md：
//! - G1a `yhyb_ext_indtop10_*`（1740）：A 的同伴池 = 同申万一级行业内总市值前 10（排除 A）
//! - G1b `yhyb_ext_mkttop310_*`（1740）：A 的同伴池 = 全市场每行业市值前 10 并集 ≈310（排除 A）
//! - G3  `yhyb_ext_mkttop310_spec_*`（696）：G1b 池内统计 − 初版全市场统计
//!   （hit/rhit/rmed × fwd/bwd × 29 事件 × 4 时段）
//!
//! 统计口径与初版 agg_fused_blocked 逐位一致（桶=1 秒、geo 零模型、时段掩码、回退链条），
//! 仅 B 遍历范围从全市场缩到池内。行业从 /ssd_data/data/vars/SzBa/industry.h5 直读
//! （唯一源头）；市值从 /home/chenzongwei/database/daily_data/total_caps.parquet 直读。

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::yhyb_metrics::{
    build_slices_and_null, c_of_m, ln_c_of_m, load_streams, period_slice, EvStream, SliceTable,
    N_EVENTS, N_PERIODS, YhybParams,
};
use rayon::prelude::*;

// 与初版相同的常量
const PERIOD_HI_S: [i64; N_PERIODS] = [19620, 17820, 19620, 19800];
const PERIOD_LO_S: [i64; N_PERIODS] = [5400, 6000, 17820, 12600];
const NB_BUCKET: usize = 14220; // 与 PackGather 相同

// ---------------- 行业 h5（唯一源头） ----------------

static IND_MAT: OnceLock<(Vec<i64>, Vec<String>, Vec<f64>, usize)> = OnceLock::new();

/// 读 industry.h5 全矩阵一次（日历、symbol_map、行主序数据、列数）。行业编号 1~31，NaN=未知。
fn ind_mat() -> &'static (Vec<i64>, Vec<String>, Vec<f64>, usize) {
    IND_MAT.get_or_init(|| {
        let cal = std::fs::read_to_string("/ssd_data/data/vars/SzBa/calendar_map.csv").unwrap();
        let dates: Vec<i64> = cal
            .lines()
            .skip(1)
            .filter_map(|l| l.trim().parse().ok())
            .collect();
        let sym = std::fs::read_to_string("/ssd_data/data/basic_info/symbol_map.csv").unwrap();
        let sym_codes: Vec<String> = sym
            .lines()
            .skip(1)
            .map(|l| l.split(',').next().unwrap_or("").trim().to_string())
            .collect();
        let f = hdf5_metno::File::open("/ssd_data/data/vars/SzBa/industry.h5").unwrap();
        let ds = f.dataset("data").unwrap();
        let arr = ds.read_2d().unwrap();
        let ncols = arr.ncols();
        let raw: Vec<f64> = arr.as_slice().unwrap().to_vec();
        (dates, sym_codes, raw, ncols)
    })
}

/// 行业编号序列（index 与 symbol_map 全量列序对齐，0=未知）。
fn industry_of_day(date: i64) -> Vec<u16> {
    let (dates, syms, raw, ncols) = ind_mat();
    let row = dates.iter().rposition(|&d| d <= date).unwrap_or(0);
    let n = syms.len();
    let mut out = vec![0u16; n];
    for c in 0..n {
        let v = raw[row * ncols + c];
        out[c] = if v.is_nan() { 0 } else { v as u16 };
    }
    out
}

// ---------------- 市值 parquet（total_caps.parquet 直读） ----------------

static CAP_TABLE: OnceLock<(Vec<i64>, HashMap<String, Vec<f64>>)> = OnceLock::new();

/// 读 total_caps.parquet 一次：日期轴 + code→市值列（带 .SZ/.SH 后缀）。
fn cap_table() -> &'static (Vec<i64>, HashMap<String, Vec<f64>>) {
    CAP_TABLE.get_or_init(|| {
        use arrow::array::{Array, Float64Array, StringArray};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let f = std::fs::File::open("/home/chenzongwei/database/daily_data/total_caps.parquet")
            .unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(f).unwrap();
        let schema = builder.schema().clone();
        // index 是日期（列名可能为空），其余全是 f64 列
        let mut date_idx: Option<usize> = None;
        for (i, fld) in schema.fields().iter().enumerate() {
            if fld.data_type() == &arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None)
                || fld.name() == "date"
            {
                date_idx = Some(i);
                break;
            }
        }
        let reader = builder.build().unwrap();
        let mut dates: Vec<i64> = Vec::new();
        let mut cols: HashMap<String, Vec<f64>> = HashMap::new();
        let field_names: Vec<String> = schema.fields().iter().map(|x| x.name().clone()).collect();
        for batch in reader {
            let batch = batch.unwrap();
            if let Some(di) = date_idx {
                if di < batch.num_columns() {
                    // Timestamp 列转成 20150105 式 int
                    if let Some(ts) = batch.column(di).as_any().downcast_ref::<arrow::array::TimestampMillisecondArray>() {
                        use arrow::temporal_conversions::timestamp_ms_to_datetime;
                        for v in ts.iter() {
                            if let Some(ms) = v {
                                if let Some(dt) = timestamp_ms_to_datetime(ms) {
                                    use chrono::Datelike;
                                    dates.push(dt.year() as i64 * 10000 + dt.month() as i64 * 100 + dt.day() as i64);
                                } else {
                                    dates.push(0);
                                }
                            } else {
                                dates.push(0);
                            }
                        }
                    }
                }
            }
            for ci in 0..batch.num_columns() {
                if Some(ci) == date_idx {
                    continue;
                }
                let name = field_names[ci].clone();
                if let Some(a) = batch.column(ci).as_any().downcast_ref::<Float64Array>() {
                    let vals: Vec<f64> = a.iter().map(|x| x.unwrap_or(f64::NAN)).collect();
                    cols.entry(name).or_insert_with(Vec::new).extend(vals);
                }
            }
        }
        (dates, cols)
    })
}

/// 当日总市值：6 位代码 → 市值（NaN 忽略）。
fn cap_of_day(date: i64) -> HashMap<String, f64> {
    let (dates, cols) = cap_table();
    let r = dates.iter().rposition(|&d| d <= date).unwrap_or(0);
    let mut out = HashMap::new();
    for (k, v) in cols.iter() {
        let bare = k.trim_end_matches(".SZ").trim_end_matches(".SH").trim_end_matches(".BJ").to_string();
        if let Some(&x) = v.get(r) {
            if x.is_finite() {
                out.insert(bare, x);
            }
        }
    }
    out
}

// ---------------- 池构建 ----------------

/// A 的同伴池（streams 全量索引）：
/// - indtop10[i]：A_i 同行业市值前 10（排除自身；行业未知/池空 → 空）
/// - mkttop310[i]：全市场每行业市值前 10 并集（排除自身）
pub fn build_pools(
    date: i64,
    codes: &[String],
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let inds = industry_of_day(date); // 与 symbol_map 全量列序对齐
    let caps = cap_of_day(date);
    let sym = &ind_mat().1;
    let sym_pos: HashMap<&str, usize> = sym.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let code_pos: HashMap<&str, usize> = codes.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();
    // code → 行业
    let mut ind_of: HashMap<usize, u16> = HashMap::new();
    for (i, code) in codes.iter().enumerate() {
        if let Some(&p) = sym_pos.get(code.as_str()) {
            let v = inds[p];
            if v != 0 {
                ind_of.insert(i, v);
            }
        }
    }
    // 每行业股票按市值降序
    let mut by_ind: HashMap<u16, Vec<(f64, usize)>> = HashMap::new();
    for (i, code) in codes.iter().enumerate() {
        if let Some(&ind) = ind_of.get(&i) {
            if let Some(&cap) = caps.get(code.as_str()) {
                by_ind.entry(ind).or_default().push((cap, i));
            }
        }
    }
    let mut top10: Vec<usize> = Vec::new(); // 310 宇宙
    for v in by_ind.values_mut() {
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        v.truncate(10);
        for &(_, idx) in v.iter() {
            top10.push(idx);
        }
    }
    // indtop10：每 A 同行前 10（排除自身）
    let mut indtop10: Vec<Vec<usize>> = Vec::with_capacity(codes.len());
    for i in 0..codes.len() {
        let mut pool = Vec::new();
        if let Some(&ind) = ind_of.get(&i) {
            if let Some(v) = by_ind.get(&ind) {
                for &(_, idx) in v.iter() {
                    if idx != i {
                        pool.push(idx);
                    }
                }
            }
        }
        indtop10.push(pool);
    }
    // mkttop310：每 A 的池 = 310 宇宙（排除自身）
    let mut mkttop310: Vec<Vec<usize>> = Vec::with_capacity(codes.len());
    for i in 0..codes.len() {
        let pool: Vec<usize> = top10.iter().copied().filter(|&x| x != i).collect();
        mkttop310.push(pool);
    }
    (indtop10, mkttop310)
}

// ---------------- 池版聚合（口径逐位对齐 agg_fused_blocked） ----------------

struct PoolDir {
    cnt: [[u64; NB_BUCKET]; N_PERIODS],
    w: [[f32; NB_BUCKET]; N_PERIODS],
    hit: [u64; N_PERIODS],
    wsum: [f64; N_PERIODS],
    max_b: [usize; N_PERIODS],
    touched: Vec<usize>,
}

impl Default for PoolDir {
    fn default() -> Self {
        PoolDir {
            cnt: [[0; NB_BUCKET]; N_PERIODS],
            w: [[0.0f32; NB_BUCKET]; N_PERIODS],
            hit: [0; N_PERIODS],
            wsum: [0.0; N_PERIODS],
            max_b: [0; N_PERIODS],
            touched: Vec::with_capacity(64),
        }
    }
}

impl PoolDir {
    fn reset(&mut self) {
        for &t in self.touched.iter() {
            for p in 0..N_PERIODS {
                self.cnt[p][t] = 0;
                self.w[p][t] = 0.0;
            }
        }
        self.touched.clear();
        self.hit = [0; N_PERIODS];
        self.wsum = [0.0; N_PERIODS];
        self.max_b = [0; N_PERIODS];
    }

    #[inline(always)]
    fn push(&mut self, mask: u32, bb: usize, ww: f64, dd: u64, t_us: u64) {
        if self.cnt[0][bb] == 0 && self.cnt[1][bb] == 0 && self.cnt[2][bb] == 0 && self.cnt[3][bb] == 0 {
            self.touched.push(bb);
        }
        let wf = ww as f32;
        if mask & 1 != 0 {
            self.cnt[0][bb] += 1;
            self.w[0][bb] += wf;
        }
        if mask & 2 != 0 {
            self.cnt[1][bb] += 1;
            self.w[1][bb] += wf;
        }
        if mask & 4 != 0 {
            self.cnt[2][bb] += 1;
            self.w[2][bb] += wf;
        }
        if mask & 8 != 0 {
            self.cnt[3][bb] += 1;
            self.w[3][bb] += wf;
        }
        for p in 0..N_PERIODS {
            if mask & (1 << p) != 0 {
                self.hit[p] += (dd <= t_us) as u64;
                self.wsum[p] += ww;
                self.max_b[p] = self.max_b[p].max(bb);
            }
        }
    }

    /// 定位中位/加权中位/前 k 桶（同 PackGather::locate_p）。
    fn locate_p(&mut self, p: usize, half_n: u64, half_w: f64, k: u64) -> (usize, usize, usize) {
        let mut acc = 0u64;
        let mut f_acc = 0u64;
        let mut wacc = 0.0f64;
        let mut med_b = 0usize;
        let mut f5_b = 0usize;
        let mut wmed_b = 0usize;
        for i in 0..=self.max_b[p] {
            let c = self.cnt[p][i];
            if c == 0 {
                continue;
            }
            if acc <= half_n && half_n < acc + c as u64 {
                med_b = i;
            }
            if f_acc < k {
                f_acc += c as u64;
                if f_acc >= k {
                    f5_b = i;
                }
            }
            let wc = self.w[p][i] as f64;
            if wacc <= half_w && half_w < wacc + wc {
                wmed_b = i;
            }
            acc += c as u64;
            wacc += wc;
        }
        (med_b, f5_b, wmed_b)
    }

    /// 统计（同 PackGather::stats_p）：[med, mean, hit, fast5, wmed]。
    fn stats_p(&mut self, p: usize, n: u64, k: u64, med_b: usize, wmed_b: usize) -> [f64; 5] {
        if n == 0 {
            return [f64::NAN, f64::NAN, 0.0, f64::NAN, f64::NAN];
        }
        let med = med_b as f64 + 0.5;
        let mut mean_sum = 0.0f64;
        let mut f_acc = 0u64;
        let mut f_sum = 0.0f64;
        let mut fast5 = f64::NAN;
        for i in 0..=self.max_b[p] {
            let c = self.cnt[p][i] as u64;
            if c == 0 {
                continue;
            }
            let mid = i as f64 + 0.5;
            mean_sum += c as f64 * mid;
            if f_acc < k {
                let take = c.min(k - f_acc);
                f_sum += take as f64 * mid;
                f_acc += take;
                if f_acc >= k {
                    fast5 = f_sum / k as f64;
                }
            }
        }
        [
            med,
            mean_sum / n as f64,
            self.hit[p] as f64 / n as f64,
            fast5,
            wmed_b as f64 + 0.5,
        ]
    }
}

/// 池版单 (A, e) 聚合：60 值（4 时段 × [rate, fwd7, bwd7]），口径逐位对齐 fused。
pub fn agg_pool(
    streams: &[Option<[EvStream; N_EVENTS]>],
    slices: &SliceTable,
    ai: usize,
    e: usize,
    pool: &[usize],
    prm: &YhybParams,
) -> Option<Vec<f64>> {
    let sa = streams[ai].as_ref()?;
    let ta = &sa[e].t;
    let wa = &sa[e].w;
    let base = slices.base;
    let t_us = (prm.hit_t_s * 1e6) as u64;
    let us_hi = [
        base + PERIOD_HI_S[0] * 1_000_000,
        base + PERIOD_HI_S[1] * 1_000_000,
        base + PERIOD_HI_S[2] * 1_000_000,
        base + PERIOD_HI_S[3] * 1_000_000,
    ];
    let lo_s = [
        PERIOD_LO_S[0] * 1_000_000,
        PERIOD_LO_S[1] * 1_000_000,
        PERIOD_LO_S[2] * 1_000_000,
        PERIOD_LO_S[3] * 1_000_000,
    ];
    let hi_s = [
        PERIOD_HI_S[0] * 1_000_000,
        PERIOD_HI_S[1] * 1_000_000,
        PERIOD_HI_S[2] * 1_000_000,
        PERIOD_HI_S[3] * 1_000_000,
    ];
    // A 侧时段切片与 rate
    let mut alo = [0usize; N_PERIODS];
    let mut ahi = [0usize; N_PERIODS];
    let mut rate = [0.0f64; N_PERIODS];
    let mut any = false;
    for p in 0..N_PERIODS {
        let (lo, hi) = period_slice(ta, base, p);
        alo[p] = lo;
        ahi[p] = hi;
        rate[p] = (hi - lo) as f64;
        if hi > lo {
            any = true;
        }
    }
    if !any {
        let mut v = Vec::with_capacity(60);
        for _ in 0..N_PERIODS {
            v.push(0.0);
            for _ in 0..14 {
                v.push(f64::NAN);
            }
        }
        return Some(v);
    }
    let ualo = alo.iter().copied().min().unwrap();
    let uahi = ahi.iter().copied().max().unwrap();
    // 5 段
    let seg = [
        (ualo, ta.partition_point(|&x| x < base + 6_000_000_000)),
        (ta.partition_point(|&x| x < base + 6_000_000_000), ta.partition_point(|&x| x < base + 12_600_000_000)),
        (ta.partition_point(|&x| x < base + 12_600_000_000), ta.partition_point(|&x| x < base + 17_820_000_000)),
        (ta.partition_point(|&x| x < base + 17_820_000_000), ta.partition_point(|&x| x < base + 19_620_000_000)),
        (ta.partition_point(|&x| x < base + 19_620_000_000), uahi),
    ];
    // A 侧零模型（geo）
    let mut sum_x = [0.0f64; N_PERIODS];
    let mut sum_ln_x = [0.0f64; N_PERIODS];
    for p in 0..N_PERIODS {
        for &a in &ta[alo[p]..ahi[p]] {
            let x = (us_hi[p] - a).max(0) as f64;
            sum_x[p] += x;
            sum_ln_x[p] += (x.max(1.0)).ln();
        }
    }
    let k_a = rate;
    let x_bar = {
        let mut xb = [0.0f64; N_PERIODS];
        for p in 0..N_PERIODS {
            xb[p] = if k_a[p] > 0.0 { sum_x[p] / k_a[p] } else { 0.0 };
        }
        xb
    };
    // 桶与 B 侧零模型累积
    let mut fwd = PoolDir::default();
    let mut bwd = PoolDir::default();
    let mut sum_ln_c = [0.0f64; N_PERIODS];
    let mut n_b = [0u64; N_PERIODS];
    let mut null_hit_acc = [0.0f64; N_PERIODS];
    for &bi in pool {
        let Some(sb) = streams[bi].as_ref() else { continue };
        let tb = &sb[e].t;
        let tb_len = tb.len();
        if tb_len == 0 {
            continue;
        }
        // B 时段切片（零模型贡献条件）
        let mut b_blo = [0usize; N_PERIODS];
        let mut b_bhi = [0usize; N_PERIODS];
        for p in 0..N_PERIODS {
            let (blo, bhi) = period_slice(tb, base, p);
            b_blo[p] = blo;
            b_bhi[p] = bhi;
            if blo < bhi {
                let m_b = (bhi - blo) as u32;
                sum_ln_c[p] += ln_c_of_m(m_b);
                n_b[p] += 1;
                let base_p = (1.0 - t_us as f64 / x_bar[p]).clamp(0.0, 1.0);
                null_hit_acc[p] += 1.0 - base_p.powf(m_b as f64);
            }
        }
        // j-walk（绝对时间域，等价 fused 相对域；全事件流无时段过滤）
        let mut j = 0usize;
        let mut tp_b: i64 = i64::MIN;
        let mut walk = |ta_seg: &[i64], wa_seg: &[f64], s: usize| {
            for idx in 0..ta_seg.len() {
                let a = ta_seg[idx];
                let wi = wa_seg[idx];
                while j < tb_len && tb[j] <= a {
                    tp_b = tb[j];
                    j += 1;
                }
                let tj_b = if j < tb_len { tb[j] } else { i64::MAX };
                let bbw = |d: i64| (d.max(0) as u64 / 1_000_000) as usize;
                match s {
                    0 => {
                        if tp_b >= base + lo_s[0] {
                            let d = (a - tp_b) as u64;
                            bwd.push(0b0001, bbw(a - tp_b), wi, d, t_us);
                        }
                        if tj_b < base + hi_s[0] {
                            let d = (tj_b - a) as u64;
                            fwd.push(0b0001, bbw(tj_b - a), wi, d, t_us);
                        }
                    }
                    1 => {
                        if tp_b >= base + lo_s[1] {
                            let d = (a - tp_b) as u64;
                            bwd.push(0b0011, bbw(a - tp_b), wi, d, t_us);
                        } else if tp_b >= base + lo_s[0] {
                            let d = (a - tp_b) as u64;
                            bwd.push(0b0001, bbw(a - tp_b), wi, d, t_us);
                        }
                        if tj_b < base + hi_s[1] {
                            let d = (tj_b - a) as u64;
                            fwd.push(0b0011, bbw(tj_b - a), wi, d, t_us);
                        } else if tj_b < base + hi_s[0] {
                            let d = (tj_b - a) as u64;
                            fwd.push(0b0001, bbw(tj_b - a), wi, d, t_us);
                        }
                    }
                    2 => {
                        if tp_b >= base + lo_s[3] {
                            let d = (a - tp_b) as u64;
                            bwd.push(0b1011, bbw(a - tp_b), wi, d, t_us);
                        } else if tp_b >= base + lo_s[1] {
                            let d = (a - tp_b) as u64;
                            bwd.push(0b0011, bbw(a - tp_b), wi, d, t_us);
                        } else if tp_b >= base + lo_s[0] {
                            let d = (a - tp_b) as u64;
                            bwd.push(0b0001, bbw(a - tp_b), wi, d, t_us);
                        }
                        if tj_b < base + hi_s[1] {
                            let d = (tj_b - a) as u64;
                            fwd.push(0b1011, bbw(tj_b - a), wi, d, t_us);
                        } else if tj_b < base + hi_s[0] {
                            let d = (tj_b - a) as u64;
                            fwd.push(0b1001, bbw(tj_b - a), wi, d, t_us);
                        } else if tj_b < base + hi_s[3] {
                            let d = (tj_b - a) as u64;
                            fwd.push(0b1000, bbw(tj_b - a), wi, d, t_us);
                        }
                    }
                    3 => {
                        if tp_b >= base + lo_s[2] {
                            let d = (a - tp_b) as u64;
                            bwd.push(0b1101, bbw(a - tp_b), wi, d, t_us);
                        } else if tp_b >= base + lo_s[3] {
                            let d = (a - tp_b) as u64;
                            bwd.push(0b1001, bbw(a - tp_b), wi, d, t_us);
                        } else if tp_b >= base + lo_s[0] {
                            let d = (a - tp_b) as u64;
                            bwd.push(0b0001, bbw(a - tp_b), wi, d, t_us);
                        }
                        if tj_b < base + hi_s[0] {
                            let d = (tj_b - a) as u64;
                            fwd.push(0b1101, bbw(tj_b - a), wi, d, t_us);
                        } else if tj_b < base + hi_s[3] {
                            let d = (tj_b - a) as u64;
                            fwd.push(0b1000, bbw(tj_b - a), wi, d, t_us);
                        }
                    }
                    _ => {
                        if tp_b >= base + lo_s[3] {
                            let d = (a - tp_b) as u64;
                            bwd.push(0b1000, bbw(a - tp_b), wi, d, t_us);
                        }
                        if tj_b < base + hi_s[3] {
                            let d = (tj_b - a) as u64;
                            fwd.push(0b1000, bbw(tj_b - a), wi, d, t_us);
                        }
                    }
                }
            }
        };
        // 段内 wa 与 ta 同区间切片
        for s in 0..5 {
            let (lo, hi) = seg[s];
            walk(&ta[lo..hi], &wa[lo..hi], s);
        }
    }
    // 统计输出
    let mut v = Vec::with_capacity(60);
    for p in 0..N_PERIODS {
        if rate[p] == 0.0 {
            v.push(0.0);
            for _ in 0..14 {
                v.push(f64::NAN);
            }
            continue;
        }
        let null_med = if n_b[p] > 0 {
            (sum_ln_x[p] / k_a[p]).exp() * (sum_ln_c[p] / n_b[p] as f64).exp() / 1e6
        } else {
            f64::NAN
        };
        let null_hit = if n_b[p] > 0 {
            null_hit_acc[p] / n_b[p] as f64
        } else {
            f64::NAN
        };
        let mut n_f = 0u64;
        let mut n_bw = 0u64;
        for i in 0..=fwd.max_b[p] {
            n_f += fwd.cnt[p][i] as u64;
        }
        for i in 0..=bwd.max_b[p] {
            n_bw += bwd.cnt[p][i] as u64;
        }
        let kf = ((n_f as f64) * prm.fast_q).round().max(1.0) as u64;
        let kb = ((n_bw as f64) * prm.fast_q).round().max(1.0) as u64;
        let (med_f, _, wmed_f) = if n_f > 0 {
            fwd.locate_p(p, n_f / 2, fwd.wsum[p] / 2.0, kf)
        } else {
            (0, 0, 0)
        };
        let (med_b, _, wmed_b) = if n_bw > 0 {
            bwd.locate_p(p, n_bw / 2, bwd.wsum[p] / 2.0, kb)
        } else {
            (0, 0, 0)
        };
        let st_f = fwd.stats_p(p, n_f, kf, med_f, wmed_f);
        let st_b = bwd.stats_p(p, n_bw, kb, med_b, wmed_b);
        v.push(rate[p]);
        for dd in [&st_f, &st_b] {
            v.push(dd[0]);
            v.push(dd[1]);
            v.push(dd[2]);
            v.push(dd[3]);
            v.push(dd[4]);
            v.push(dd[0] / null_med); // rmed
            v.push(dd[2] / null_hit); // rhit
        }
    }
    Some(v)
}

// ---------------- 因子名 ----------------

pub fn ext_names() -> Vec<String> {
    let mut names = Vec::with_capacity(1740 * 2 + 696);
    for e in 0..N_EVENTS {
        for p in 0..N_PERIODS {
            for (pref, pool) in [("yhyb_ext_indtop10", ""), ("yhyb_ext_mkttop310", "")] {
                names.push(format!("{pref}_e{e:02}_p{p}_rate"));
                for dir in ["fwd", "bwd"] {
                    for mt in ["med", "mean", "hit", "fast5", "wmed", "rmed", "rhit"] {
                        names.push(format!("{pref}_e{e:02}_p{p}_{mt}_{dir}"));
                    }
                }
                let _ = pool;
            }
        }
    }
    // G3：spec = G1b − 初版全市场（hit/rhit/rmed × fwd/bwd）
    for e in 0..N_EVENTS {
        for p in 0..N_PERIODS {
            for dir in ["fwd", "bwd"] {
                for mt in ["hit", "rmed", "rhit"] {
                    names.push(format!("yhyb_ext_mkttop310_spec_e{e:02}_p{p}_{mt}_{dir}"));
                }
            }
        }
    }
    names
}

// ---------------- 主计算 ----------------

/// 池版补充因子计算：返回 (codes, vals)。
/// mode: 0 = G1a+G1b+G3（4176）；1 = 验证模式（池=全市场，1740 对照初版）。
/// G3（方案 A，用户批准）：pipeline 内调初版无 L4 聚合重算全市场基线，因子级做差
/// （G3 = G1b − 初版，hit/rmed/rhit × fwd/bwd），口径天然一致；总耗时 ~77s/天，
/// 达标口径 = 不慢于初版（82s）。
pub fn compute_yhyb_ext(date: i64, prm: &YhybParams, mode: usize) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    crate::yhyb_metrics::ensure_threads();
    let (codes, streams) = load_streams(date, prm)?;
    let (slices, _null_t) = build_slices_and_null(&streams);
    let n = codes.len();
    let (indtop10, mkttop310) = build_pools(date, &codes);
    // 验证模式：池 = 全市场
    let all_pool: Vec<Vec<usize>> = if mode == 1 {
        (0..n).map(|i| (0..n).filter(|&x| x != i).collect()).collect()
    } else {
        Vec::new()
    };
    let n_factors = if mode == 1 {
        N_EVENTS * N_PERIODS * 15
    } else {
        N_EVENTS * N_PERIODS * 15 * 2 + N_EVENTS * N_PERIODS * 6
    };
    let mut rows: Vec<Option<Vec<f64>>> = (0..n * N_EVENTS)
        .into_par_iter()
        .map(|idx| {
            let ai = idx / N_EVENTS;
            let e = idx % N_EVENTS;
            if mode == 1 {
                agg_pool(&streams, &slices, ai, e, &all_pool[ai], prm)
            } else {
                let ga = agg_pool(&streams, &slices, ai, e, &indtop10[ai], prm);
                let gb = agg_pool(&streams, &slices, ai, e, &mkttop310[ai], prm);
                match (ga, gb) {
                    (Some(a), Some(b)) => {
                        let mut v = a;
                        v.extend(b);
                        Some(v)
                    }
                    _ => None,
                }
            }
        })
        .collect();
    // G3：初版全市场基线（无 L4 现成函数，口径天然一致）
    let (base_codes, base_vals) = if mode == 0 {
        crate::yhyb_metrics::compute_from_streams(&codes, &streams, prm)
    } else {
        (Vec::new(), Vec::new())
    };
    let base_pos: std::collections::HashMap<&str, usize> = base_codes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();
    // G3 统计列（1740 布局内）：hit_f, rmed_f, rhit_f, hit_b, rmed_b, rhit_b
    const SPEC_COLS: [usize; 6] = [3, 6, 7, 10, 13, 14];
    let mut out_codes = Vec::new();
    let mut vals: Vec<f32> = Vec::with_capacity(n * n_factors);
    for ai in 0..n {
        // rows[ai*N_EVENTS+e] = [G1a 60 值, G1b 60 值]（mode 0）；组装时按因子组分块：
        // 先 G1a 全部事件，再 G1b 全部事件（与 ext_names 的分块布局一致）。
        let mut g1a = Vec::with_capacity(N_EVENTS * 60);
        let mut g1b = Vec::with_capacity(N_EVENTS * 60);
        let mut ok = true;
        for e in 0..N_EVENTS {
            match &rows[ai * N_EVENTS + e] {
                Some(v) if mode == 1 => g1a.extend_from_slice(v),
                Some(v) if v.len() == 120 => {
                    g1a.extend_from_slice(&v[..60]);
                    g1b.extend_from_slice(&v[60..]);
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            let mut row = g1a;
            if mode == 0 {
                row.extend(g1b);
                // G3 = G1b − 初版（6 统计 × 29 事件 × 4 时段）
                if let Some(&bp) = base_pos.get(codes[ai].as_str()) {
                    for e in 0..N_EVENTS {
                        for p in 0..N_PERIODS {
                            let bo = (e * N_PERIODS + p) * 15;
                            let go = (e * N_PERIODS + p) * 15;
                            for &c in SPEC_COLS.iter() {
                                let bv = base_vals[bp * (N_EVENTS * N_PERIODS * 15) + bo + c] as f64;
                                let gv = row[1740 + go + c];
                                row.push(gv - bv);
                            }
                        }
                    }
                } else {
                    for _ in 0..(N_EVENTS * N_PERIODS * 6) {
                        row.push(f64::NAN);
                    }
                }
            }
            if row.len() == n_factors {
                out_codes.push(codes[ai].clone());
                vals.extend(row.iter().map(|&x| x as f32));
            }
        }
    }
    Ok((out_codes, vals))
}

/// Python 入口：计算某日补充因子（mode 0 = 3480；mode 1 = 验证对照 1740）。
#[pyo3::pyfunction]
#[pyo3(signature = (date, mode=0))]
pub fn py_yhyb_ext(date: i64, mode: usize) -> pyo3::PyResult<(Vec<String>, Vec<f32>)> {
    let prm = YhybParams::default();
    compute_yhyb_ext(date, &prm, mode).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

/// Python 拿补充因子名（4176）。
#[pyo3::pyfunction]
pub fn py_yhyb_ext_names() -> Vec<String> {
    ext_names()
}
