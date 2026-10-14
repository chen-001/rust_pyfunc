//! v11：跨股票非负稀疏重构（sandbox）。
//! 对象：每只股票的日内画像 = 4 个字段块 × 48 个五分钟桶（时间分布、主买净流、每桶振幅、笔数密度），
//! 一律用本股全天总量归一，只留「形状」。
//! 核心构造：把本股画像用其余股票的画像做一个非负稀疏线性组合重构（投影梯度 + L1），
//! 残差 = 别人解释不掉的自己。库里的共同因子都是手写的等权市场收益（hm100/hm110），
//! 没有数据驱动的重构，也没有非负稀疏这一层。
//! 加工层：残差范数/稀疏度/集中度/时间形状/跨桶自相关；同伴网络（入度、互惠、集中度）；
//! 同伴集合在不同迭代与不同罚强下的稳定性；残差自身再分解一次的二次残差份额（特异度谱）。
//! 构建: cd /home/chenzongwei/rust_pyfunc/sandbox_oidflow && cargo build --release
//! 运行: ./target/release/oidflow_sandbox 20220819 > out_v11_20220819.json
mod fast_csv_reader;
mod v9_stats;
use rayon::prelude::*;
use serde::Serialize;
use v9_stats::*;

const B: usize = 48;          // 5 分钟桶
const BLOCKS: usize = 4;      // 字段块
const KDIM: usize = B * BLOCKS;
const START_SEC: f64 = 9.5 * 3600.0;
const END_SEC: f64 = 15.0 * 3600.0;
const LUNCH: f64 = 1.5 * 3600.0;

#[derive(Serialize)]
struct Out {
    names: Vec<String>,
    codes: Vec<String>,
    vals: Vec<Vec<f32>>,
    diag: Vec<Vec<f32>>,
}

fn stock_codes_for_day(
    dir: &std::path::Path,
    symbol_map: &std::path::Path,
    date: i64,
    subdir: &str,
) -> std::io::Result<Vec<String>> {
    use std::io::{Error, ErrorKind};
    let invalid = |e: &str| Error::new(ErrorKind::InvalidData, e.to_string());
    let text = std::fs::read_to_string(symbol_map).map_err(|e| invalid(&e.to_string()))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| invalid("股票表为空"))?;
    let col = header
        .split(',')
        .position(|h| h.trim() == "symbol")
        .ok_or_else(|| invalid("股票表缺少 symbol 列"))?;
    let mut stocks = std::collections::BTreeSet::new();
    for line in lines {
        let code = line.split(',').nth(col).unwrap_or("").trim();
        if code.is_empty() { continue; }
        if code.len() > 6 || !code.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid("股票表含无效代码"));
        }
        stocks.insert(format!("{code:0>6}"));
    }
    if stocks.is_empty() || !dir.is_dir() {
        return Err(invalid("股票白名单为空或原始数据目录不存在"));
    }
    Ok(stocks.into_iter()
        .filter(|code| dir.join(format!("{code}_{date}_{subdir}.csv")).is_file())
        .collect())
}

fn list_codes(date: i64) -> std::io::Result<Vec<String>> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let basic =
        std::env::var("RUST_PYFUNC_BASIC_INFO_DIR").unwrap_or("/ssd_data/data/basic_info".into());
    let dir = std::path::Path::new(&root).join(date.to_string()).join("transaction");
    let map = std::path::Path::new(&basic).join("symbol_map.csv");
    stock_codes_for_day(&dir, &map, date, "transaction")
}

/// 连续交易时间（把午休压掉）-> 统一轴的秒数。
#[inline]
fn axis_sec(h: f64, m: f64, s: f64) -> Option<f64> {
    let sec = h * 3600.0 + m * 60.0 + s;
    if sec < START_SEC || sec >= END_SEC { return None; }
    Some(if sec > 12.0 * 3600.0 { sec - LUNCH } else { sec })
}

fn profile(code: &str, date: i64) -> Option<[f64; KDIM]> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let path = format!("{}/{}/transaction/{}_{}_transaction.csv", root, date, code, date);
    let text = std::fs::read_to_string(&path).ok()?;
    let span = END_SEC - LUNCH - START_SEC;
    let mut amt = vec![0.0f64; B];
    let mut signed = vec![0.0f64; B];
    let mut hivol = vec![0.0f64; B];
    let mut lovol = vec![0.0f64; B];
    let mut cnt = vec![0.0f64; B];
    let (mut tot, mut n_trade) = (0.0f64, 0.0f64);
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 15 { continue; }
        let flag: i32 = match f[10].parse() { Ok(v) => v, Err(_) => continue };
        if flag != 66 && flag != 83 { continue; }
        let vol: f64 = match f[8].parse() { Ok(v) => v, Err(_) => continue };
        let price: f64 = match f[7].parse() { Ok(v) => v, Err(_) => continue };
        if !(vol > 0.0) || !(price > 0.0) { continue; }
        let sec = match f[1].split(' ').nth(1).and_then(|t| {
            let mut it = t.split(':');
            let h: f64 = it.next()?.parse().ok()?;
            let m: f64 = it.next()?.parse().ok()?;
            let s: f64 = it.next()?.parse().ok()?;
            Some(h * 3600.0 + m * 60.0 + s)
        }) { Some(v) => v, None => continue };
        let sec = if sec > 12.0 * 3600.0 { sec - LUNCH } else { sec };
        if sec < START_SEC || sec >= END_SEC { continue; }
        let b = (((sec - START_SEC) / span) * B as f64) as usize;
        let b = b.min(B - 1);
        amt[b] += vol * price;
        signed[b] += if flag == 66 { vol * price } else { -vol * price };
        cnt[b] += 1.0;
        if hivol[b] == 0.0 { hivol[b] = price; lovol[b] = price; }
        else {
            if price > hivol[b] { hivol[b] = price; }
            if price < lovol[b] { lovol[b] = price; }
        }
        tot += vol * price;
        n_trade += 1.0;
    }
    if n_trade < 200.0 || tot <= 0.0 { return None; }
    let mut out = [0.0f64; KDIM];
    for b in 0..B {
        // 块 0：成交额的时间分布占比
        out[b] = amt[b] / tot;
        // 块 1：主买净流占比
        out[B + b] = signed[b] / tot;
        // 块 2：桶内振幅（相对该桶均价）
        let rng = if lovol[b] > 0.0 { (hivol[b] - lovol[b]) / lovol[b] } else { 0.0 };
        out[2 * B + b] = rng;
        // 块 3：笔数密度（每桶 300 秒）
        out[3 * B + b] = cnt[b] / 300.0;
    }
    Some(out)
}

/// 稳健缩放：按每一维的中位数与 1.5 倍分位距缩，再截尾。
fn robust_scale(cols: &mut Vec<[f64; KDIM]>, dims: usize) {
    let n = cols.len();
    for d in 0..dims {
        let mut xs: Vec<f64> = (0..n).map(|i| cols[i][d]).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = quantile(&xs, 0.5);
        let lo = quantile(&xs, 0.05);
        let hi = quantile(&xs, 0.95);
        let scale = if hi - lo > 1e-12 { (hi - lo) / 2.0 } else { 1.0 };
        for i in 0..n {
            cols[i][d] = ((cols[i][d] - med) / scale).max(-5.0).min(5.0);
        }
    }
}

/// 非负稀疏重构：坐标下降解 min ||y - Zw||^2 + lam*||w||_1，w >= 0，并去掉自己那一列。
/// 每列的模长先算好，闭式更新 w_k = max(0, (rho_k - lam/2)) / ||z_k||^2，rho_k 用残差增量维护。
fn reconstruct(z: &[[f64; KDIM]], colnorm2: &[f64], yi: usize, lam: f64, epochs: usize) -> Vec<f64> {
    let n = z.len();
    let mut w = vec![0.0f64; n];
    // 初始：只用与 y 最相关的一列（带 L1 惩罚后的最大者）
    let mut best = (0usize, f64::NEG_INFINITY);
    for k in 0..n {
        if k == yi { continue; }
        let mut dot = 0.0;
        for d in 0..KDIM { dot += z[k][d] * z[yi][d]; }
        if dot > best.1 { best = (k, dot); }
    }
    if best.1 > lam / 2.0 {
        w[best.0] = (best.1 - lam / 2.0) / colnorm2[best.0].max(1e-12);
    }
    let mut resid = z[yi];
    for d in 0..KDIM { resid[d] -= w[best.0] * z[best.0][d]; }
    for _ in 0..epochs {
        let mut max_delta = 0.0f64;
        for k in 0..n {
            if k == yi { continue; }
            let mut rho = 0.0;
            let mut rho = 0.0;
            for d in 0..KDIM { rho += z[k][d] * resid[d]; }
            let nk = colnorm2[k].max(1e-12);
            // 加上本列当前贡献，得到「去掉本列后」与 y 的内积
            let rho_full = rho + w[k] * nk;
            let new = ((rho_full - lam / 2.0).max(0.0)) / nk;
            let delta = new - w[k];
            if delta.abs() > max_delta { max_delta = delta.abs(); }
            if delta != 0.0 {
                for d in 0..KDIM { resid[d] -= delta * z[k][d]; }
                w[k] = new;
            }
        }
        if max_delta < 1e-6 { break; }
    }
    w
}

fn residual_norm(z: &[[f64; KDIM]], w: &[f64], yi: usize) -> (f64, f64) {
    let mut pred = [0.0f64; KDIM];
    for (k, v) in w.iter().enumerate() {
        if *v == 0.0 { continue; }
        for d in 0..KDIM { pred[d] += v * z[k][d]; }
    }
    let (mut nres, mut ny) = (0.0, 0.0);
    for d in 0..KDIM {
        let r = z[yi][d] - pred[d];
        nres += r * r;
        ny += z[yi][d] * z[yi][d];
    }
    (nres.sqrt(), ny.sqrt())
}

fn main() {
    let date: i64 = std::env::args().nth(1).expect("用法: oidflow_sandbox <date>")
        .parse().expect("date 必须是整数");
    let codes = list_codes(date).expect("股票范围或原始目录检查失败");
    eprintln!("codes = {}", codes.len());

    // ① 每股画像（逐笔读 + 分桶，rayon 并行）
    let prof: Vec<(String, [f64; KDIM])> = codes
        .par_iter()
        .filter_map(|c| profile(c, date).map(|p| (c.clone(), p)))
        .collect();
    let n = prof.len();
    eprintln!("profiles = {}", n);
    if n < 200 { eprintln!("股票太少"); return; }
    let mut z: Vec<[f64; KDIM]> = prof.iter().map(|x| x.1).collect();
    robust_scale(&mut z, KDIM);

    // ② 逐股非负稀疏重构（rayon 并行；同时跑两个罚强做稳定性）
    // 每列的模长，用于坐标下降的闭式更新与罚强标定
    let mut colnorm2 = vec![0.0f64; n];
    for k in 0..n {
        let mut s = 0.0;
        for d in 0..KDIM { s += z[k][d] * z[k][d]; }
        colnorm2[k] = s;
    }
    let med_norm: f64 = {
        let mut v: Vec<f64> = colnorm2.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        quantile(&v, 0.5)
    };
    let lam_a = 0.35 * med_norm;
    let lam_b = 0.70 * med_norm;
    let iters = 14usize;
    let res: Vec<(Vec<f64>, Vec<f64>, [f64; 2])> = (0..n)
        .into_par_iter()
        .map(|i| {
            let wa = reconstruct(&z, &colnorm2, i, lam_a, iters);
            let wb = reconstruct(&z, &colnorm2, i, lam_b, iters);
            let (ra, _) = residual_norm(&z, &wa, i);
            let (rb, _) = residual_norm(&z, &wb, i);
            (wa, wb, [ra, rb])
        })
        .collect();

    // ③ 网络：谁被谁用（入度）
    let mut indeg = vec![0.0f64; n];
    let mut idx_of: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, (c, _)) in prof.iter().enumerate() { idx_of.insert(c.clone(), i); }
    let mut recip_num = 0.0f64;
    let mut edges = 0.0f64;
    for (i, (wa, _, _)) in res.iter().enumerate() {
        for k in 0..n {
            if wa[k] > 1e-6 && k != i {
                indeg[k] += 1.0;
                edges += 1.0;
                if res[k].0[i] > 1e-6 { recip_num += 1.0; }
            }
        }
    }
    let recip = if edges > 0.0 { recip_num / edges } else { f64::NAN };

    // ④ 每股输出
    let mut names: Vec<String> = Vec::new();
    let mut vals: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut diags: Vec<[f64; 4]> = Vec::with_capacity(n);
    for (i, (wa, wb, rr)) in res.iter().enumerate() {
        let (ra, rb) = (rr[0], rr[1]);
        let (_, ny) = residual_norm(&z, wa, i);
        let resid = if ny > 1e-12 { ra / ny } else { f64::NAN };
        let resid_b = if ny > 1e-12 { rb / ny } else { f64::NAN };
        // 活跃同伴
        let mut act: Vec<f64> = wa.iter().cloned().filter(|x| *x > 1e-6).collect();
        let n_act = act.len() as f64;
        act.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let sum: f64 = act.iter().sum();
        let p: Vec<f64> = if sum > 0.0 { act.iter().map(|x| x / sum).collect() } else { vec![] };
        let ent = if !p.is_empty() { p.iter().map(|x| if *x > 0.0 { -x * x.ln() } else { 0.0 }).sum::<f64>() } else { f64::NAN };
        let eff = ent.exp();
        let g = gini(&act);
        let top1 = p.first().cloned().unwrap_or(f64::NAN);
        // 残差的时间形状（按桶聚合 4 个块）
        let mut pred = [0.0f64; KDIM];
        for (k, v) in wa.iter().enumerate() {
            if *v == 0.0 { continue; }
            for d in 0..KDIM { pred[d] += v * z[k][d]; }
        }
        let mut rb_b = vec![0.0f64; B];
        let mut rb_abs = vec![0.0f64; B];
        let mut rpos = 0.0f64;
        let mut rneg = 0.0f64;
        for d in 0..KDIM {
            let r = z[i][d] - pred[d];
            rb_b[d % B] += r;
            rb_abs[d % B] += r.abs();
            if r > 0.0 { rpos += r; } else { rneg -= r; }
        }
        let rabs = ts21(&rb_abs);
        let rsum = ts21(&rb_b);
        // 残差最大块
        let (mxb, _) = rb_abs.iter().enumerate().fold((0usize, 0.0f64), |acc, (b, v)| if *v > acc.1 { (b, *v) } else { acc });
        // 罚强敏感性：两种罚下的活跃数与残差差
        let act_b = wb.iter().filter(|x| **x > 1e-6).count() as f64;
        let vals_i = vec![
            resid, resid_b, n_act, act_b, n_act / act_b.max(1e-9),
            eff, g, top1, ent,
            rpos / (rpos + rneg).max(1e-12),
            rabs[0], rabs[2], rabs[3], rabs[11], rabs[13], rabs[18], rabs[19], rabs[20],
            rsum[0], rsum[2], rsum[11], rsum[13],
            mxb as f64 / B as f64,
            // 同伴网络
            indeg[i], indeg[i] / n as f64, recip,
            // 残差与画像自身的重合度
            {
                let mut dot = 0.0; let mut a2 = 0.0; let mut b2 = 0.0;
                for d in 0..KDIM {
                    let r = z[i][d] - pred[d];
                    dot += r * z[i][d]; a2 += r * r; b2 += z[i][d] * z[i][d];
                }
                if a2 > 1e-12 && b2 > 1e-12 { dot / (a2.sqrt() * b2.sqrt()) } else { f64::NAN }
            },
            // 残差在四个块上的份额
            {
                let mut blk = [0.0f64; BLOCKS];
                for d in 0..KDIM { blk[d / B] += (z[i][d] - pred[d]).powi(2); }
                let t: f64 = blk.iter().sum();
                if t > 1e-12 { blk[0] / t } else { f64::NAN }
            },
            {
                let mut blk = [0.0f64; BLOCKS];
                for d in 0..KDIM { blk[d / B] += (z[i][d] - pred[d]).powi(2); }
                let t: f64 = blk.iter().sum();
                if t > 1e-12 { blk[1] / t } else { f64::NAN }
            },
        ];
        vals.push(vals_i);
        diags.push([n_act, ra, rb, indeg[i]]);
        if names.is_empty() {
            names = vec![
                "resid_l1", "resid_lam_hi", "n_active", "n_active_hi", "active_ratio",
                "eff_peers", "peer_gini", "peer_top1", "peer_entropy",
                "resid_pos_share",
                "res_abs_mean", "res_abs_std", "res_abs_skew", "res_abs_ac1", "res_abs_trend", "res_abs_lz", "res_abs_entropy", "res_abs_maxrange",
                "res_signed_mean", "res_signed_std", "res_signed_ac1", "res_signed_trend",
                "res_argmax_bucket",
                "indegree", "indegree_share", "reciprocity",
                "resid_self_corr", "res_share_block0", "res_share_block1",
            ].into_iter().map(|s| s.to_string()).collect();
        }
    }

    let mut out = Out {
        names,
        codes: prof.iter().map(|x| x.0.clone()).collect(),
        vals: vals.into_iter().map(|v| v.into_iter().map(|x| x as f32).collect()).collect(),
        diag: diags.into_iter().map(|d| d.iter().map(|&x| x as f32).collect()).collect(),
    };
    eprintln!("valid = {} cols = {}", out.codes.len(), out.names.len());
    let _ = idx_of;
    println!("{}", serde_json::to_string(&out).unwrap());
}
