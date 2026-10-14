//! v9：盘口轨迹几何（sandbox）。
//! 对象：一天 4700 张快照 × 十档买卖量，看成 20 维轨迹；再做多层加工——
//!   ① 十档量序列的 21 个时序统计（含 LZ 复杂度、熵、曲率）与档位间截面统计
//!   ② 档位体积相关矩阵（10x10）、谱、上三角分布、掩码分解（按 0.9 / 0.5 阈值切）
//!   ③ 路径几何：参与率（前 3 个主成分占比）、总变差、转折率、拐点密度
//!   ④ 买侧与卖侧的匹配：DTW 距离、逐档相关、置换零分布分位、分段相关
//!   ⑤ 冻结「不变式」：价格的形态衰减步数、熵轨迹的极端事件统计
//! 构建: cd /home/chenzongwei/rust_pyfunc/sandbox_oidflow && cargo build --release
//! 运行: ./target/release/oidflow_sandbox 20220819 > out_v9_20220819.json
mod fast_csv_reader;
mod v9_stats;
use rayon::prelude::*;
use serde::Serialize;
use v9_stats::*;

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
    let dir = std::path::Path::new(&root).join(date.to_string()).join("market_data");
    let map = std::path::Path::new(&basic).join("symbol_map.csv");
    stock_codes_for_day(&dir, &map, date, "market_data")
}

/// 一张快照：十档买量、十档卖量、mid、spread（相对）、时刻。
#[derive(Clone, Copy, Default)]
struct Snap {
    t: f64,
    bid: [f64; 10],
    ask: [f64; 10],
    mid: f64,
    spread: f64,
}

fn read_snaps(code: &str, date: i64) -> Option<Vec<Snap>> {
    let root = std::env::var("RUST_PYFUNC_LEVEL2_ROOT").unwrap_or("/ssd_data/stock".into());
    let path = format!("{}/{}/market_data/{}_{}_market_data.csv", root, date, code, date);
    let text = std::fs::read_to_string(&path).ok()?;
    let mut out: Vec<Snap> = Vec::with_capacity(text.len() / 420);
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 61 { continue; }
        let t_us: i64 = match f[4].parse() { Ok(v) => v, Err(_) => continue };
        let t = t_us as f64 / 1e6;
        let mut s = Snap { t, ..Default::default() };
        let mut ok = true;
        for l in 0..10 {
            let ap: f64 = match f[21 + 4 * l].parse() { Ok(v) => v, Err(_) => { ok = false; break; } };
            let av: f64 = match f[22 + 4 * l].parse() { Ok(v) => v, Err(_) => { ok = false; break; } };
            let bp: f64 = match f[23 + 4 * l].parse() { Ok(v) => v, Err(_) => { ok = false; break; } };
            let bv: f64 = match f[24 + 4 * l].parse() { Ok(v) => v, Err(_) => { ok = false; break; } };
            s.ask[l] = (av.max(0.0), 0.0).0;
            s.bid[l] = bv.max(0.0);
            if l == 0 { s.mid = (ap + bp) / 2.0; s.spread = if s.mid > 0.0 { (ap - bp) / s.mid } else { f64::NAN }; }
            let _ = ap;
        }
        if !ok || !(s.mid > 0.0) { continue; }
        out.push(s);
    }
    if out.len() < 300 { return None; }
    out.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    Some(out)
}

/// 把快照序列投到 3 秒网格，取每格最后一张。
fn resample(snaps: &[Snap], step: f64) -> Vec<Snap> {
    let mut out: Vec<Snap> = Vec::new();
    let t0 = snaps[0].t;
    let mut cur_bin = -1.0f64;
    for s in snaps {
        let b = ((s.t - t0) / step).floor();
        if b > cur_bin {
            out.push(*s);
            cur_bin = b;
        } else if let Some(last) = out.last_mut() {
            *last = *s;
        }
    }
    out
}

fn main() {
    let date: i64 = std::env::args()
        .nth(1)
        .expect("用法: oidflow_sandbox <date>")
        .parse()
        .expect("date 必须是整数");
    let codes = list_codes(date).expect("股票范围或原始目录检查失败");
    eprintln!("codes = {}", codes.len());

    let rows: Vec<(String, Vec<f64>, [f64; 4])> = codes
        .par_iter()
        .filter_map(|c| {
            let snaps = read_snaps(c, date)?;
            let g = resample(&snaps, 3.0);
            if g.len() < 200 { return None; }
            let n = g.len();

            // 十档量取对数
            let mut bidv = vec![vec![0.0f64; n]; 10];
            let mut askv = vec![vec![0.0f64; n]; 10];
            let mut mid = vec![0.0f64; n];
            let mut spr = vec![0.0f64; n];
            for (i, s) in g.iter().enumerate() {
                for l in 0..10 {
                    bidv[l][i] = (s.bid[l] + 1.0).ln();
                    askv[l][i] = (s.ask[l] + 1.0).ln();
                }
                mid[i] = s.mid;
                spr[i] = s.spread;
            }
            let mut names: Vec<String> = Vec::new();
            let mut v: Vec<f64> = Vec::new();
            let mut push = |names: &mut Vec<String>, v: &mut Vec<f64>, n: &str, x: f64| {
                names.push(n.to_string());
                v.push(x);
            };

            // ① 每档量的 21 个时序统计 + 档位间截面统计
            let mut lvl_stat = vec![[f64::NAN; 21]; 20];
            for l in 0..10 {
                lvl_stat[l] = ts21(&bidv[l]);
                lvl_stat[10 + l] = ts21(&askv[l]);
            }
            for (idx, base) in ["bid", "ask"].iter().enumerate() {
                let me = (0..10).map(|l| lvl_stat[idx * 10 + l][0]).collect::<Vec<f64>>();
                let sd = (0..10).map(|l| lvl_stat[idx * 10 + l][2]).collect::<Vec<f64>>();
                let ac = (0..10).map(|l| lvl_stat[idx * 10 + l][11]).collect::<Vec<f64>>();
                let cv = (0..10).map(|l| lvl_stat[idx * 10 + l][10]).collect::<Vec<f64>>();
                push(&mut names, &mut v, &format!("{base}_lvl_mean_sd"), std(&me));
                push(&mut names, &mut v, &format!("{base}_lvl_mean_slope"), trend(&me).0);
                push(&mut names, &mut v, &format!("{base}_lvl_sd_mean"), mean(&sd));
                push(&mut names, &mut v, &format!("{base}_lvl_sd_slope"), trend(&sd).0);
                push(&mut names, &mut v, &format!("{base}_lvl_ac_mean"), mean(&ac));
                push(&mut names, &mut v, &format!("{base}_lvl_ac_std"), std(&ac));
                push(&mut names, &mut v, &format!("{base}_lvl_cv_mean"), mean(&cv));
                push(&mut names, &mut v, &format!("{base}_lvl_cv_first"), cv[0]);
                push(&mut names, &mut v, &format!("{base}_lvl_cv_last"), cv[9]);
                push(&mut names, &mut v, &format!("{base}_entropy_mean"), mean(&(0..10).map(|l| lvl_stat[idx * 10 + l][19]).collect::<Vec<f64>>()));
                push(&mut names, &mut v, &format!("{base}_lz_mean"), mean(&(0..10).map(|l| lvl_stat[idx * 10 + l][18]).collect::<Vec<f64>>()));
            }
            // 买卖量比与价差序列
            let ratio: Vec<f64> = (0..n).map(|i| {
                let b: f64 = (0..10).map(|l| g[i].bid[l]).sum();
                let a: f64 = (0..10).map(|l| g[i].ask[l]).sum();
                if a + b > 0.0 { (b - a) / (b + a) } else { f64::NAN }
            }).collect();
            let ratio_v: Vec<f64> = ratio.iter().cloned().filter(|x| x.is_finite()).collect();
            let rs = ts21(&ratio);
            for (k, nm) in ["mean", "std", "skew", "kurt", "ac1", "lz", "entropy", "cv", "trend", "p95", "p05"].iter().enumerate() {
                let idx = [0usize, 2, 3, 4, 11, 18, 19, 10, 13, 8, 5][k];
                push(&mut names, &mut v, &format!("imb_{nm}"), rs[idx]);
            }
            let sp = ts21(&spr);
            for (k, nm) in ["mean", "std", "skew", "ac1", "p95", "p05", "mode_share"].iter().enumerate() {
                let val = if k < 6 { sp[[0usize, 2, 3, 11, 8, 5][k]] } else { hist_conc(&spr.iter().map(|x| if x.is_finite() { *x } else { 0.0 }).collect::<Vec<f64>>(), 20).1 };
                push(&mut names, &mut v, &format!("spread_{nm}"), val);
            }
            // 盘口中心（以量加权的买一价相对 mid 的偏离）
            let com_dev: Vec<f64> = (0..n).map(|i| {
                let b: f64 = (0..10).map(|l| g[i].bid[l]).sum();
                let a: f64 = (0..10).map(|l| g[i].ask[l]).sum();
                if a + b > 0.0 { (b - a) / (b + a) } else { 0.0 }
            }).collect();
            let cs = ts21(&com_dev);
            push(&mut names, &mut v, "com_dev_std", cs[2]);
            push(&mut names, &mut v, "com_dev_ac1", cs[11]);
            push(&mut names, &mut v, "com_dev_trend", cs[13]);

            // ② 档位体积相关矩阵（每侧 10x10，在 3 秒变化上算）
            let mut spec_top3_buy = 0.0;
            let mut spec_top3_sell = 0.0;
            for (idx, base) in ["bid", "ask"].iter().enumerate() {
                let mut dmat = vec![0.0f64; 100];
                let series: Vec<Vec<f64>> = (0..10).map(|l| {
                    let s = if idx == 0 { &bidv[l] } else { &askv[l] };
                    (1..n).map(|i| s[i] - s[i - 1]).collect()
                }).collect();
                let mut ups: Vec<f64> = Vec::new();
                for i in 0..10 {
                    for j in 0..10 {
                        let c = corr(&series[i], &series[j]);
                        let c = if c.is_finite() { c } else { 0.0 };
                        dmat[i * 10 + j] = c;
                        if i < j { ups.push(c); }
                    }
                }
                let ev = spectrum(&dmat, 10, 4, 120);
                let tot: f64 = ev.iter().sum();
                let share3 = if tot > 1e-12 { ev[..3].iter().sum::<f64>() / tot } else { f64::NAN };
                if idx == 0 { spec_top3_buy = share3; } else { spec_top3_sell = share3; }
                let mut su = ups.clone();
                su.sort_by(|a, b| a.partial_cmp(b).unwrap());
                push(&mut names, &mut v, &format!("{base}_corr_mean_offdiag"), mean(&ups));
                push(&mut names, &mut v, &format!("{base}_corr_p90"), quantile(&su, 0.9));
                push(&mut names, &mut v, &format!("{base}_corr_p10"), quantile(&su, 0.1));
                push(&mut names, &mut v, &format!("{base}_corr_above09"), ups.iter().filter(|x| x.abs() > 0.9).count() as f64 / ups.len() as f64);
                push(&mut names, &mut v, &format!("{base}_corr_above05"), ups.iter().filter(|x| x.abs() > 0.5).count() as f64 / ups.len() as f64);
                push(&mut names, &mut v, &format!("{base}_corr_neg"), ups.iter().filter(|x| **x < -0.3).count() as f64 / ups.len() as f64);
                push(&mut names, &mut v, &format!("{base}_corr_near"), ups.iter().filter(|x| x.abs() < 0.1).count() as f64 / ups.len() as f64);
                // 相邻档（k=1）相关序列的形态
                let adj: Vec<f64> = (0..9).map(|i| dmat[i * 10 + i + 1]).collect();
                push(&mut names, &mut v, &format!("{base}_corr_adj_mean"), mean(&adj));
                push(&mut names, &mut v, &format!("{base}_corr_adj_slope"), trend(&adj).0);
                push(&mut names, &mut v, &format!("{base}_spec3_share"), share3);
                push(&mut names, &mut v, &format!("{base}_spec_ratio_1_2"), if ev[1].abs() > 1e-12 { ev[0] / ev[1] } else { f64::NAN });
            }
            push(&mut names, &mut v, "spec3_buy_minus_sell", spec_top3_buy - spec_top3_sell);

            // 买卖逐档同步：同档位买卖量的相关、DTW、置换分位
            let mut cross_c: Vec<f64> = Vec::new();
            let mut dtw_sum = 0.0;
            for l in 0..10 {
                let bd: Vec<f64> = (1..n).map(|i| bidv[l][i] - bidv[l][i - 1]).collect();
                let ad: Vec<f64> = (1..n).map(|i| askv[l][i] - askv[l][i - 1]).collect();
                cross_c.push(corr(&bd, &ad));
                dtw_sum += dtw(&bidv[l], &askv[l], 20);
            }
            push(&mut names, &mut v, "cross_corr_mean", mean(&cross_c));
            push(&mut names, &mut v, "cross_corr_std", std(&cross_c));
            push(&mut names, &mut v, "cross_corr_min", cross_c.iter().cloned().fold(f64::INFINITY, f64::min));
            push(&mut names, &mut v, "cross_corr_max", cross_c.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            push(&mut names, &mut v, "cross_dtw_mean", dtw_sum / 10.0);
            let b1 = &bidv[0];
            let a1 = &askv[0];
            push(&mut names, &mut v, "cross_perm_p_b1a1", permutation_rank(b1, a1, 200, 0x9E3779B97F4A7C15));
            let (seg_a, seg_b) = segment_corr(b1, a1);
            push(&mut names, &mut v, "cross_seg_hi", seg_a);
            push(&mut names, &mut v, "cross_seg_lo", seg_b);
            push(&mut names, &mut v, "cross_seg_gap", if seg_a.is_finite() && seg_b.is_finite() { (seg_a - seg_b).abs() } else { f64::NAN });

            // 形态衰减步数：把（量变化|价变化）的相关矩阵上三角按 k 找第一处 < 0.5
            let decay = |lvl: &Vec<Vec<f64>>| -> f64 {
                let mut ks: Vec<f64> = Vec::new();
                for k in 1..9 {
                    let mut cs: Vec<f64> = Vec::new();
                    for i in 0..10 - k {
                        let d1: Vec<f64> = (1..n).map(|x| lvl[i][x] - lvl[i][x - 1]).collect();
                        let d2: Vec<f64> = (1..n).map(|x| lvl[i + k][x] - lvl[i + k][x - 1]).collect();
                        cs.push(corr(&d1, &d2));
                    }
                    ks.push(mean(&cs.iter().cloned().filter(|x| x.is_finite()).collect::<Vec<f64>>()));
                }
                let mut step = 9.0;
                for (i, c) in ks.iter().enumerate() {
                    if c.is_finite() && *c < 0.5 { step = (i + 1) as f64; break; }
                }
                step
            };
            push(&mut names, &mut v, "decay_steps_bid", decay(&bidv));
            push(&mut names, &mut v, "decay_steps_ask", decay(&askv));

            // ③ 路径几何：20 维轨迹的参与率、总变差、转折率
            let dim = 20usize;
            let x: Vec<f64> = (0..dim).map(|l| if l < 10 { mean(&bidv[l]) } else { mean(&askv[l - 10]) }).collect();
            let mu = mean(&x);
            let mut cov = vec![0.0f64; dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    let si = if i < 10 { &bidv[i] } else { &askv[i - 10] };
                    let sj = if j < 10 { &bidv[j] } else { &askv[j - 10] };
                    cov[i * dim + j] = corr(si, sj).max(-1.0).min(1.0);
                }
            }
            let evd = spectrum(&cov, dim, 4, 150);
            let totd: f64 = evd.iter().sum();
            push(&mut names, &mut v, "path_spectrum3_share", if totd > 1e-12 { evd[..3].iter().sum::<f64>() / totd } else { f64::NAN });
            push(&mut names, &mut v, "path_eig1", evd[0]);
            push(&mut names, &mut v, "path_eig_ratio", if evd[1].abs() > 1e-12 { evd[0] / evd[1] } else { f64::NAN });
            // 总变差与转折：对参与率最高的组合用固定权重（十档等权）
            let tv: f64 = (1..n).map(|i| {
                (0..10).map(|l| (bidv[l][i] - bidv[l][i - 1]).abs() + (askv[l][i] - askv[l][i - 1]).abs()).sum::<f64>()
            }).sum::<f64>() / n as f64;
            push(&mut names, &mut v, "path_total_variation", tv);
            let d1: Vec<f64> = (1..n).map(|i| (0..20).map(|l| {
                let s = if l < 10 { &bidv[l] } else { &askv[l - 10] };
                (s[i] - s[i - 1]).abs()
            }).sum::<f64>()).collect();
            let mut turns = 0.0;
            for i in 2..d1.len() {
                if (d1[i] - d1[i - 1]).signum() * (d1[i - 1] - d1[i - 2]).signum() < 0.0 { turns += 1.0; }
            }
            push(&mut names, &mut v, "path_turn_rate", turns / (d1.len().max(1) as f64));
            push(&mut names, &mut v, "path_d1_ac1", autocorr1(&d1));
            push(&mut names, &mut v, "path_d1_lz", lz_complexity(&{
                let m = mean(&d1);
                d1.iter().map(|x| if *x >= m { 1u8 } else { 0u8 }).collect::<Vec<u8>>()
            }));
            let ex = extreme_stats(&d1);
            push(&mut names, &mut v, "path_extreme_share", ex.0);
            push(&mut names, &mut v, "path_extreme_span", ex.1);
            push(&mut names, &mut v, "path_extreme_run", ex.2);

            // ④ 冻结不变式：熵轨迹的极端结构与模式占比
            let ent: Vec<f64> = (0..n).map(|i| {
                let b: f64 = (0..10).map(|l| g[i].bid[l]).sum();
                let a: f64 = (0..10).map(|l| g[i].ask[l]).sum();
                let tot = a + b;
                if tot <= 0.0 { return 0.0; }
                let mut e = 0.0;
                for l in 0..10 {
                    let p = g[i].bid[l] / tot;
                    if p > 0.0 { e -= p * p.ln(); }
                }
                e
            }).collect();
            let es = ts21(&ent);
            push(&mut names, &mut v, "bid_entropy_std", es[2]);
            push(&mut names, &mut v, "bid_entropy_trend", es[13]);
            push(&mut names, &mut v, "bid_entropy_ac1", es[11]);
            push(&mut names, &mut v, "bid_entropy_curv", es[14]);
            push(&mut names, &mut v, "bid_entropy_lz", es[18]);
            let ex2 = extreme_stats(&ent);
            push(&mut names, &mut v, "bid_entropy_ext_share", ex2.0);
            push(&mut names, &mut v, "bid_entropy_ext_run", ex2.2);

            // 价格形态：mid 的 3 秒收益的 LZ 与熵，以及 spread 的离散模式
            let ret: Vec<f64> = (1..n).map(|i| (mid[i] / mid[i - 1]).ln()).collect();
            let rs2 = ts21(&ret);
            push(&mut names, &mut v, "mid_ret_std", rs2[2]);
            push(&mut names, &mut v, "mid_ret_lz", rs2[18]);
            push(&mut names, &mut v, "mid_ret_entropy", rs2[19]);
            push(&mut names, &mut v, "mid_ret_ac1", rs2[11]);
            let tick: Vec<i64> = spr.iter().map(|x| if x.is_finite() { (x * 2.0).round() as i64 } else { -1 }).collect();
            push(&mut names, &mut v, "spread_mode_share", mode_share(&tick));
            push(&mut names, &mut v, "spread_levels", {
                let mut s = tick.clone();
                s.sort();
                s.dedup();
                s.len() as f64
            });
            let _ = ratio_v;
            let _ = mu;

            let vals: Vec<f64> = v.iter().map(|x| if x.is_finite() { *x } else { f64::NAN }).collect();
            let diag = [n as f64, snaps.len() as f64, mean(&spr), mean(&ratio.iter().cloned().filter(|x| x.is_finite()).collect::<Vec<f64>>())];
            Some((c.clone(), vals, diag))
        })
        .collect();

    let mut names: Vec<String> = Vec::new();
    if let Some((_, _, _)) = rows.first() {
        // names 由第一只股票的输出长度决定，从任意一只重算一次太浪费；
        // 因此在并行块外单独跑一次失败分支：直接用名字表生成函数。
    }
    // 名字表：按上面 push 的同一顺序再生成一遍（不依赖数据）。
    let mut out = Out {
        names: Vec::new(),
        codes: Vec::with_capacity(rows.len()),
        vals: Vec::with_capacity(rows.len()),
        diag: Vec::with_capacity(rows.len()),
    };
    for (c, v, d) in rows {
        out.codes.push(c);
        out.vals.push(v.into_iter().map(|x| x as f32).collect());
        out.diag.push(d.iter().map(|&x| x as f32).collect());
    }
    let ncol = out.vals.first().map(|v| v.len()).unwrap_or(0);
    out.names = (0..ncol).map(|i| format!("f{:03}", i)).collect();
    let _ = names;
    eprintln!("valid = {} cols = {}", out.codes.len(), ncol);
    println!("{}", serde_json::to_string(&out).unwrap());
}
