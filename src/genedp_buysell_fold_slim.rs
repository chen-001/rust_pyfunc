//! 基因动态规划 · 方向 4「买卖配对折叠」精简版（67,467 列）：算法与完整版 genedp_buysell_fold 相同，
//! 由沙箱 sandbox_align_fold/src/align_fold_v3_slim.rs 搬入，保留列的数值与完整版逐字节一致。
//! 精简依据是 20241231 全市场排名相关普查（同一因子只换一个参数时两列有多像）：
//!   完全重复/常数列（未配对净方向、计数类规则的得分=配对比例、恒定列）→ 删；
//!   行业内中性、二层折叠、ordjump 的 0.5/2 倍门槛、块内 |相关|>0.95 的近似重复 → 删；
//!   对朴素量取残差整层、时段差「尾盘减下午」、6 种成交/撤单字母的 0.5 倍门槛、块内 0.85 以上近亲 → 删。
//! 被删的字母版本和二层折叠不再计算；残差与行业中性层整层不算，所以不读行业，也没有横截面变换。
//! 计算核、字母化、特征、读取器全部复用 crate::genedp_buysell_fold 的子模块（完整版不改动）。
//! 列名约定：凡依赖撤单数据的列名都含 cxl；撤单数据不存在的日期（沪市 20211105 前、深市 20160104 前）这些列写空值。
use crate::genedp_buysell_fold::reader::{read_market_fast, read_trade_fast, MarketRecord, TradeRecord};
use crate::genedp_buysell_fold::v3_feats::{arc_feats, naive_feats, pair_feats};
use crate::genedp_buysell_fold::v3_kernels::{self, cross_match, hpk, make_ctx, nested, prob_fast, Kern, Mode};
use crate::genedp_buysell_fold::v3_letters::{
    base_variant, build_cancel_map, cancel_arcs_with_map, cxl_available, is_cxl, variants, windows,
    Bars, Seq, Variant, Win, HK_BARS, TIER_BARS,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

const NV: usize = 38;
const NW: usize = 3;
const NG: usize = 2;
const NS: usize = NW * NG * NV; // 228 条序列
const NF: usize = 17; // 无交叉规则的特征数
const BLOCK: usize = 268;
const O_PK2: usize = 12 * NF; // 204
const O_CROSS: usize = O_PK2 + 19; // 223
const O_PROB: usize = O_CROSS + 19; // 242
const O_EXC: usize = O_PROB + 8; // 250
const O_NAIVE: usize = O_EXC + 9; // 259
const N_DIFFABLE: usize = O_NAIVE; // 259：参与时段差的列
const EXC_RULES: [usize; 3] = [0, 1, 2];
const EXC_FEATS: [usize; 3] = [0, 5, 6];
const XD_F: usize = NF + 1; // 跨日配对：17 + 跨日配对占比
const CO_PAIRS: [(usize, usize); 6] = [(0, 5), (0, 6), (0, 4), (5, 6), (1, 2), (3, 7)]; // 基础字母下标
const CO_RULES: [usize; 3] = [0, 1, 2];
const WDIFF: [(&str, usize, usize); 3] = [("pm_minus_full", 1, 0), ("tail_minus_full", 2, 0), ("tail_minus_pm", 2, 1)];
const RDIFF: [(&str, usize, usize); 8] = [
    ("profit_minus_loss", 4 * NF + 1, 5 * NF + 1),
    ("bf_minus_sf", 6 * NF + 1, 7 * NF + 1),
    ("short_minus_cnt", 8 * NF + 1, 1),
    ("dec_minus_cnt", NF + 1, 1),
    ("pk2_minus_cnt", O_PK2 + 1, 1),
    ("cross_minus_dec", O_CROSS + 1, NF + 1),
    ("stack_minus_cnt", 11 * NF + 1, 1),
    ("gain_minus_profit", 10 * NF + 1, 4 * NF + 1),
];

const OFF_HK: usize = NS * BLOCK;
const OFF_XD: usize = OFF_HK + NW * NV * 19;
const OFF_CO: usize = OFF_XD + 16 * NW * 12 * XD_F;
const OFF_ARC: usize = OFF_CO + CO_PAIRS.len() * NW * CO_RULES.len() * NF;
const OFF_WD: usize = OFF_ARC + 3 * NW * 19;
const OFF_RD: usize = OFF_WD + 3 * NV * NG * N_DIFFABLE;
const ROW: usize = OFF_RD + NS * RDIFF.len();
/// 精简版输出列数（固定）
pub const N_FACTORS: usize = 67_467;

#[inline]
fn sidx(w: usize, g: usize, v: usize) -> usize {
    (w * NG + g) * NV + v
}

// ---------------- 精简版保留规则（按列名判断；与 20241231 普查定下的 67,467 列名单逐一核对过） ----------------

/// 整个字母版本都删掉（不再计算）：6 种成交/撤单字母的 0.5 倍门槛、ordjump 的 0.5 和 2 倍门槛
const DROP_VARIANTS: [&str; 8] = ["flow_d05", "count_d05", "size_d05", "cxl_vol_d05", "volbar_d05", "cxl_cnt_d05", "ordjump_d05", "ordjump_d20"];
/// 得分恰好等于配对比例的规则（每对得分都是 1）
const COUNT_RULES: [&str; 8] = ["cnt", "profit", "loss", "bf", "sf", "short", "h10", "pk2"];
/// 块内 |相关| > 0.95 的近似重复
const NEAR_DUP: [&str; 7] = ["_cross_branch", "_cross_span_mean", "_gain_gain_bps", "_short_hairpin", "_prob_free_energy", "_excess_cnt_span_mean", "_excess_vol_span_mean"];
/// 块内 0.85 以上近亲（每组留一个代表，这里列的是被代替的那些）
const KIN_085: [&str; 34] = [
    "bf_span_mean", "cnt_span_mean", "cross_depth_max", "cross_depth_mean", "cross_hairpin", "cross_pair_center", "cross_span_p90", "dec_branch",
    "dec_pair_ratio", "dec_stem", "excess_dec_span_mean", "gain_span_p50", "h10_pair_ratio", "h10_span_mean", "h10_span_p50", "h10_stem",
    "loss_span_mean", "naive_active_share", "prob_exp_pair_ratio", "profit_span_mean", "rna_pair_ratio", "rna_score", "rna_span_mean", "rna_stem",
    "sf_span_mean", "short_branch", "short_pair_center", "short_pair_ratio", "short_span_p50", "short_stem", "stack_gain_bps", "stack_pair_ratio",
    "stack_span_mean", "vol_span_mean",
];

fn variant_kept(v: &str) -> bool {
    !DROP_VARIANTS.contains(&v)
}

/// 列名里是否出现某个字母版本（开头或以下划线隔开）
fn has_variant(n: &str, v: &str) -> bool {
    n.starts_with(&format!("{v}_")) || n.contains(&format!("_{v}_"))
}

/// 块内后缀：第一个 _g1_ / _g2_ 之后的部分（原始块与时段差适用）
fn block_suffix_of(n: &str) -> Option<&str> {
    let p = [n.find("_g1_"), n.find("_g2_")].into_iter().flatten().min()?;
    Some(&n[p + 4..])
}

fn col_kept(n: &str) -> bool {
    let layer = if n.starts_with("rs_") { "rs" } else if n.starts_with("in_") { "in" } else if n.starts_with("wd_") { "wd" }
        else if n.starts_with("rd_") { "rd" } else if n.starts_with("xd_") { "xd" } else if n.starts_with("co_") { "co" }
        else if n.starts_with("cxlarc_") { "arc" } else if n.contains("_hk_hpk_") { "hk" } else { "block" };
    // 第一档：完全重复与常数
    if n.ends_with("_unp_net") {
        let dup = match layer { "block" => true, "hk" => !n.contains("_full_"), _ => !n.ends_with("_cnt_unp_net") };
        if dup { return false; }
    }
    if n.ends_with("naive_buy_share") || n.ends_with("naive_walk_end") { return false; }
    if !n.contains("excess_") && COUNT_RULES.iter().any(|r| n.ends_with(&format!("_{r}_score"))) { return false; }
    if n.ends_with("_hk_hpk_score") || n.ends_with("_bf_bf_share") || n.ends_with("_sf_bf_share") || n.ends_with("_cross_pair_ratio") { return false; }
    // 第二档：整层或整块高度雷同
    if layer == "in" || layer == "rs" { return false; }
    if n.contains("_pk2_") || n.ends_with("pk2_minus_cnt") { return false; }
    if DROP_VARIANTS.iter().any(|v| has_variant(n, v)) { return false; }
    if NEAR_DUP.iter().any(|x| n.ends_with(x)) { return false; }
    // 精简档
    if n.starts_with("wd_tail_minus_pm") { return false; }
    if layer == "block" || layer == "wd" {
        if let Some(s) = block_suffix_of(n) {
            if KIN_085.contains(&s) { return false; }
        }
    }
    true
}

/// 精简版保留的列在完整版原始层（前 ROW 列）里的位置及列名，按完整版原顺序；只算一次
fn kept() -> &'static (Vec<usize>, Vec<String>) {
    static K: OnceLock<(Vec<usize>, Vec<String>)> = OnceLock::new();
    K.get_or_init(|| {
        let full = crate::genedp_buysell_fold::genedp_buysell_fold_names();
        let (idx, nm): (Vec<usize>, Vec<String>) = full.into_iter().enumerate().take(ROW).filter(|(_, n)| col_kept(n)).unzip();
        assert_eq!(idx.len(), N_FACTORS, "精简版列数应恒为 {N_FACTORS}");
        (idx, nm)
    })
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn permute(s: &Seq, seed: u64) -> Seq {
    let n = s.side.len();
    let mut idx: Vec<usize> = (0..n).collect();
    let mut x = seed | 1;
    for i in (1..n).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        idx.swap(i, (x % (i as u64 + 1)) as usize);
    }
    Seq {
        side: idx.iter().map(|&i| s.side[i]).collect(),
        strength: idx.iter().map(|&i| s.strength[i]).collect(),
        vol: idx.iter().map(|&i| s.vol[i]).collect(),
        price: idx.iter().map(|&i| s.price[i]).collect(),
    }
}

fn valid(c: &v3_kernels::Ctx) -> bool {
    c.n >= 20 && c.n_buy >= 3 && c.n_sell >= 3
}

/// 一条序列的 268 列
fn series_block(mode: Mode, k: &mut Kern, seq: &Seq, seed: u64, out: &mut [f32]) {
    let c = make_ctx(seq, None);
    naive_feats(&c, &mut out[O_NAIVE..BLOCK]);
    if !valid(&c) { return; }
    for r in 0..12 {
        let (sn, pairs) = nested(mode, k, &c, r);
        pair_feats(&c, &pairs, sn, true, &mut out[r * NF..(r + 1) * NF]);
    }
    // 二层折叠（pk2）在精简版里整块删除，不再计算，对应列留空且不输出
    let (tot, pc) = cross_match(&c);
    let ub = 1000.0 * c.n_buy.min(c.n_sell) as f64;
    pair_feats(&c, &pc, (tot as f64 / ub) as f32, false, &mut out[O_CROSS..O_PROB]);
    let pr = prob_fast(&c, &mut k.prob);
    out[O_PROB..O_EXC].copy_from_slice(&pr);
    let mut sums = [[0f64; 3]; 3];
    let mut cnts = [[0u32; 3]; 3];
    let mut buf = [f32::NAN; NF];
    for rep in 0..3u64 {
        let p = permute(seq, seed.wrapping_add((rep + 1).wrapping_mul(0x9E3779B97F4A7C15)));
        let cp = make_ctx(&p, None);
        for (i, &r) in EXC_RULES.iter().enumerate() {
            let (sn, pairs) = nested(mode, k, &cp, r);
            pair_feats(&cp, &pairs, sn, true, &mut buf);
            for (j, &f) in EXC_FEATS.iter().enumerate() {
                if buf[f].is_finite() { sums[i][j] += buf[f] as f64; cnts[i][j] += 1; }
            }
        }
    }
    for (i, &r) in EXC_RULES.iter().enumerate() {
        for (j, &f) in EXC_FEATS.iter().enumerate() {
            let v = out[r * NF + f];
            out[O_EXC + i * 3 + j] = if cnts[i][j] > 0 && v.is_finite() { v - (sums[i][j] / cnts[i][j] as f64) as f32 } else { f32::NAN };
        }
    }
}

type StockYdaySeqs = [[Option<Seq>; 16]; NW];
type DayYdayMap = HashMap<String, StockYdaySeqs>;

fn yday_cache() -> &'static Mutex<VecDeque<((String, u64, i64), Arc<DayYdayMap>)>> {
    static C: OnceLock<Mutex<VecDeque<((String, u64, i64), Arc<DayYdayMap>)>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(VecDeque::with_capacity(3)))
}

fn get_cached_yday(key: &(String, u64, i64)) -> Option<Arc<DayYdayMap>> {
    let guard = yday_cache().lock().ok()?;
    guard.iter().find(|(k, _)| k == key).map(|(_, v)| Arc::clone(v))
}

fn put_cached_yday(key: (String, u64, i64), map: Arc<DayYdayMap>) {
    if std::env::var("RUST_PYFUNC_DISABLE_GENEDP_CACHE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return;
    }
    if let Ok(mut guard) = yday_cache().lock() {
        if let Some(pos) = guard.iter().position(|(k, _)| *k == key) {
            guard.remove(pos);
        }
        while guard.len() >= 2 {
            guard.pop_front();
        }
        guard.push_back((key, map));
    }
}

fn extract_yday_only(code: &str, day: i64, vs: &[Variant], wins: &[Win; 3]) -> Option<StockYdaySeqs> {
    let trp = read_trade_fast(code, day).ok()?;
    let mkp = read_market_fast(code, day).unwrap_or_default();
    let cxl_ok_prev = cxl_available(code, day);
    let mut out: StockYdaySeqs = std::array::from_fn(|_| std::array::from_fn(|_| None));
    for w in 0..NW {
        let g = if w == 2 { 0 } else { 1 };
        if let Some(bp) = Bars::new(&trp, &mkp, &wins[w].with_bar(TIER_BARS[w][g])) {
            for a in 0..16 {
                let v = base_variant(a);
                out[w][a] = bp.seq(&vs[v], cxl_ok_prev);
            }
        }
    }
    Some(out)
}

/// 一只股票的完整版原始层（ROW 列，被删部分为空值），同时顺带产出供下一交易日跨日复用的 16 基准字母序列
fn per_stock_with_yday(
    code: &str,
    date: i64,
    prev: Option<i64>,
    prev_yday: Option<&StockYdaySeqs>,
    vs: &[Variant],
    wins: &[Win; 3],
    base_vars: &[usize; 16],
    mode: Mode,
) -> (Vec<f32>, Option<StockYdaySeqs>) {
    let mut row = vec![f32::NAN; ROW];
    v3_kernels::set_hpk_fast(mode == Mode::Fast);
    let tr: Vec<TradeRecord> = match read_trade_fast(code, date) {
        Ok(t) => t,
        Err(_) => return (row, None),
    };
    let mk: Vec<MarketRecord> = read_market_fast(code, date).unwrap_or_default();
    let cxl_ok = cxl_available(code, date);
    let code_seed = fnv(code.as_bytes());
    let mut k = Kern::new();
    // 两档粒度的字母序列（被删的 8 个字母版本直接跳过不构建，节省 21% 中位数与分桶开销）
    let mut seqs: Vec<Vec<Option<Seq>>> = Vec::with_capacity(NW * NG);
    for w in 0..NW {
        for g in 0..NG {
            let win = wins[w].with_bar(TIER_BARS[w][g]);
            let bars = Bars::new(&tr, &mk, &win);
            let ss: Vec<Option<Seq>> = vs
                .iter()
                .map(|v| {
                    if variant_kept(&v.name) {
                        bars.as_ref().and_then(|b| b.seq(v, cxl_ok))
                    } else {
                        None
                    }
                })
                .collect();
            for v in 0..NV {
                if !variant_kept(&vs[v].name) {
                    continue;
                }
                if let Some(s) = &ss[v] {
                    let si = sidx(w, g, v);
                    series_block(
                        mode,
                        &mut k,
                        s,
                        code_seed ^ ((si as u64 + 1) << 24),
                        &mut row[si * BLOCK..(si + 1) * BLOCK],
                    );
                }
            }
            seqs.push(ss);
        }
    }
    // 顺手保存当天用于跨日配对的 16 个基准字母序列（零额外计算）
    let today_yday: StockYdaySeqs = std::array::from_fn(|w| {
        let g = if w == 2 { 0 } else { 1 };
        std::array::from_fn(|a| seqs[w * NG + g][base_vars[a]].clone())
    });

    // H 型假结：每个时段用约 120 个字母的粒度（全天 120 秒、下午 60 秒、尾盘 15 秒）
    for w in 0..NW {
        let hk_owned: Option<Vec<Option<Seq>>> = if HK_BARS[w] == TIER_BARS[w][1] || HK_BARS[w] == TIER_BARS[w][0] {
            None
        } else {
            let b = Bars::new(&tr, &mk, &wins[w].with_bar(HK_BARS[w]));
            Some(
                vs.iter()
                    .map(|v| {
                        if variant_kept(&v.name) {
                            b.as_ref().and_then(|b| b.seq(v, cxl_ok))
                        } else {
                            None
                        }
                    })
                    .collect(),
            )
        };
        let hk_seqs: &[Option<Seq>] = if HK_BARS[w] == TIER_BARS[w][1] {
            &seqs[w * NG + 1]
        } else if HK_BARS[w] == TIER_BARS[w][0] {
            &seqs[w * NG]
        } else {
            hk_owned.as_deref().unwrap()
        };
        for v in 0..NV {
            if !variant_kept(&vs[v].name) {
                continue;
            }
            let Some(s) = &hk_seqs[v] else { continue };
            let c = make_ctx(s, None);
            if !valid(&c) {
                continue;
            }
            let (best, pairs) = hpk(&c, &mut k.hpk);
            let sn = (best as f64 / c.n_buy.min(c.n_sell) as f64) as f32;
            let o = OFF_HK + (w * NV + v) * 19;
            pair_feats(&c, &pairs, sn, false, &mut row[o..o + 19]);
        }
    }
    // 跨日配对：昨天与今天同一时段的字母首尾相接（全天、下午 60 秒，尾盘 15 秒），只用基准阈值版本
    if let (Some(p), Some(yday_arr)) = (prev, prev_yday) {
        let cxl_both = cxl_ok && cxl_available(code, p);
        for w in 0..NW {
            let g = if w == 2 { 0 } else { 1 };
            for a in 0..16 {
                if is_cxl(a) && !cxl_both {
                    continue;
                }
                let v = base_vars[a];
                let (Some(today), Some(yday)) = (&seqs[w * NG + g][v], &yday_arr[w][a]) else {
                    continue;
                };
                let cat = Seq::concat(yday, today);
                let c = make_ctx(&cat, None);
                if !valid(&c) {
                    continue;
                }
                let ny = yday.side.len();
                for r in 0..12 {
                    let (sn, pairs) = nested(mode, &mut k, &c, r);
                    let o = OFF_XD + ((a * NW + w) * 12 + r) * XD_F;
                    pair_feats(&c, &pairs, sn, true, &mut row[o..o + NF]);
                    if !pairs.is_empty() {
                        let st = pairs.iter().filter(|&&(x, y)| x < ny && y >= ny).count();
                        row[o + NF] = st as f32 / pairs.len() as f32;
                    }
                }
            }
        }
    }
    // 两字母联合配对：只有两种字母在两端都方向相反才可配对
    for w in 0..NW {
        for (p, &(a, b)) in CO_PAIRS.iter().enumerate() {
            let (Some(sa), Some(sb)) = (&seqs[w * NG][base_vars[a]], &seqs[w * NG][base_vars[b]]) else {
                continue;
            };
            let c = make_ctx(sa, Some(&sb.side));
            if !valid(&c) {
                continue;
            }
            for (ri, &r) in CO_RULES.iter().enumerate() {
                let (sn, pairs) = nested(mode, &mut k, &c, r);
                let o = OFF_CO + ((p * NW + w) * CO_RULES.len() + ri) * NF;
                pair_feats(&c, &pairs, sn, true, &mut row[o..o + NF]);
            }
        }
    }
    // 真实挂撤弧（撤单数据存在才算，三个时段共享一次主动委托映射排序）
    if cxl_ok && !tr.is_empty() {
        let day0 = tr[0].time_us - tr[0].time_us.rem_euclid(86_400_000_000);
        let cancel_map = build_cancel_map(&tr);
        for w in 0..NW {
            let win: &Win = &wins[w];
            let arcs = cancel_arcs_with_map(&tr, &cancel_map, win);
            let lo = day0 + win.adj_lo * 1_000_000;
            let len = (win.adj_hi - win.adj_lo) * 1_000_000;
            for (si, filt) in [Some(true), Some(false), None].iter().enumerate() {
                let sel: Vec<(i64, i64, bool, f64, bool, bool)> = arcs
                    .iter()
                    .filter(|a| filt.map_or(true, |b| a.buy == b))
                    .map(|a| (a.start_us, a.end_us, a.buy, a.vol, !a.clipped, a.unmapped))
                    .collect();
                let o = OFF_ARC + (si * NW + w) * 19;
                arc_feats(&sel, lo, len, TIER_BARS[w][0] * 1_000_000, &mut row[o..o + 19]);
            }
        }
    }
    // 单股内的时段差（d=2 tail_minus_pm 整层不保留，跳过）与规则差
    for (d, &(_, wa, wb)) in WDIFF.iter().enumerate().take(2) {
        for v in 0..NV {
            if !variant_kept(&vs[v].name) {
                continue;
            }
            for g in 0..NG {
                let (sa, sb) = (sidx(wa, g, v) * BLOCK, sidx(wb, g, v) * BLOCK);
                let o = OFF_WD + ((d * NV + v) * NG + g) * N_DIFFABLE;
                for c in 0..N_DIFFABLE {
                    row[o + c] = row[sa + c] - row[sb + c];
                }
            }
        }
    }
    for w in 0..NW {
        for g in 0..NG {
            for v in 0..NV {
                if !variant_kept(&vs[v].name) {
                    continue;
                }
                let s = sidx(w, g, v);
                for (q, &(_, a, b)) in RDIFF.iter().enumerate() {
                    if q == 4 {
                        continue; // pk2_minus_cnt 整列不保留
                    }
                    row[OFF_RD + s * RDIFF.len() + q] = row[s * BLOCK + a] - row[s * BLOCK + b];
                }
            }
        }
    }
    (row, Some(today_yday))
}

fn stock_codes(root: &str, date: i64) -> Result<(Vec<String>, String, u64), String> {
    let map = crate::data_paths::basic_info_path("symbol_map.csv");
    crate::genedp_common::stock_codes_whitelisted(root, &map, date, "transaction")
}

/// 全部列名（67,467 列，与输出逐列对齐）
pub fn genedp_buysell_fold_slim_names() -> Vec<String> {
    kept().1.clone()
}

/// 核心唯一真相源：读今天（和前一交易日）全市场原始数据 → 配对折叠特征 → 取精简版保留列。
/// 返回 (股票代码, 行优先展平的因子值)，每股 N_FACTORS 列。用当前 rayon 线程池并行。
pub fn compute_genedp_buysell_fold_slim_full(date: i64) -> std::io::Result<(Vec<String>, Vec<f32>)> {
    let root = crate::genedp_common::level2_root_string();
    let (codes, _map_path, map_hash) = stock_codes(&root, date).map_err(crate::genedp_common::invalid)?;
    let prev = crate::genedp_common::prev_level2_date(date);
    let vs = variants();
    let wins = windows();
    let base_vars: [usize; 16] = std::array::from_fn(base_variant);
    for &bv in &base_vars {
        debug_assert!(variant_kept(&vs[bv].name));
    }

    let prev_map: Option<Arc<DayYdayMap>> = if let Some(p) = prev {
        let prev_key = (root.clone(), map_hash, p);
        if let Some(hit) = get_cached_yday(&prev_key) {
            Some(hit)
        } else {
            let mut union_codes: Vec<String> = match stock_codes(&root, p) {
                Ok((pc, _, _)) => pc,
                Err(_) => Vec::new(),
            };
            let existing: std::collections::HashSet<String> = union_codes.iter().cloned().collect();
            for c in &codes {
                if !existing.contains(c) {
                    union_codes.push(c.clone());
                }
            }
            let built: DayYdayMap = union_codes
                .par_iter()
                .filter_map(|c| extract_yday_only(c, p, &vs, &wins).map(|yd| (c.clone(), yd)))
                .collect();
            let arc = Arc::new(built);
            put_cached_yday(prev_key, Arc::clone(&arc));
            Some(arc)
        }
    } else {
        None
    };

    let keep = &kept().0;
    let mut vals = vec![f32::NAN; codes.len() * N_FACTORS];
    let today_yday_vec: Vec<Option<(String, StockYdaySeqs)>> = vals
        .par_chunks_mut(N_FACTORS)
        .zip(codes.par_iter())
        .map(|(out, c)| {
            let py = prev_map.as_deref().and_then(|m| m.get(c));
            let (row, ty) = per_stock_with_yday(c, date, prev, py, &vs, &wins, &base_vars, Mode::Fast);
            for (o, &k) in out.iter_mut().zip(keep) {
                *o = row[k];
            }
            ty.map(|y| (c.clone(), y))
        })
        .collect();

    let today_map: DayYdayMap = today_yday_vec.into_iter().flatten().collect();
    put_cached_yday((root, map_hash, date), Arc::new(today_map));

    Ok((codes, vals))
}

/// 单日全市场调试入口（50 线程）：返回 (股票代码, 展平因子值)，每股 N_FACTORS 列
#[pyfunction]
pub fn py_genedp_buysell_fold_slim(py: Python<'_>, date: i64) -> PyResult<(Vec<String>, Vec<f32>)> {
    py.allow_threads(|| crate::genedp_common::run_in_50_threads(|| compute_genedp_buysell_fold_slim_full(date)))
        .and_then(|r| r)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{e:?}")))
}

#[pyfunction]
pub fn py_genedp_buysell_fold_slim_names() -> Vec<String> {
    genedp_buysell_fold_slim_names()
}
