mod allo_microstructure;
mod allo_microstructure_v3;
mod data_paths;
mod features_f64;
mod hm089;
mod time_series;

use std::sync::atomic::Ordering;
use std::time::Instant;

fn reset() {
    for a in [
        &features_f64::T_FILE_READ,
        &features_f64::T_F_COLS,
        &features_f64::T_F_COLSTATS,
        &features_f64::T_F_LYAP,
        &features_f64::T_F_CORR,
        &allo_microstructure::T_AM_PARSE,
        &allo_microstructure::T_AM_DETECT,
        &allo_microstructure::T_AM_ALA,
        &allo_microstructure::T_RANGE,
        &allo_microstructure::N_RANGE_CALLS,
        &allo_microstructure::N_SCANNED,
        &allo_microstructure::N_EVENTS,
        &allo_microstructure::T_M20,
        &allo_microstructure::T_M21,
        &allo_microstructure::N_M20_ITERS,
        &allo_microstructure::N_M21_ITERS,
        &allo_microstructure::N_M20_ALLOC,
        &allo_microstructure_v3::T_V3_PARSE,
        &allo_microstructure_v3::T_V3_DETECT,
        &allo_microstructure_v3::T_V3_ALA,
        &allo_microstructure_v3::T_V3_M2021,
        &allo_microstructure_v3::N_V3_ITERS,
    ] {
        a.store(0, Ordering::Relaxed);
    }
}

fn ms(ns: u128) -> f64 {
    ns as f64 / 1e6
}
fn ams(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = args.get(1).cloned().unwrap_or_else(|| "000001".to_string());
    let date: i64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20220819);
    let reps: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(3);

    for r in 0..reps {
        reset();
        let t = Instant::now();
        let (v, tm) = hm089::compute_stock_timed(&code, date).expect("no data");
        let wall = t.elapsed();
        let micro_total: f64 = tm.micro_ns.iter().map(|&x| ms(x)).sum();
        let micro: Vec<String> = tm.micro_ns.iter().map(|&x| format!("{:.0}", ms(x))).collect();
        let feat = &tm.feat_ns;
        let feat_total: f64 = feat.iter().map(|&x| ms(x)).sum();
        let feat_sorted = {
            let mut f: Vec<u128> = feat.clone();
            f.sort();
            f
        };
        let feat_max = feat_sorted.last().copied().unwrap_or(0);
        let feat_median = feat_sorted.get(feat_sorted.len() / 2).copied().unwrap_or(0);
        let feat_sum_ns: u64 = feat.iter().map(|&x| x as u64).sum();
        let feat_share = if feat_total > 0.0 {
            feat_sum_ns as f64 / feat_total
        } else {
            0.0
        };
        let _ = feat_share;
        println!(
            "REP {} wall={:.1} total={:.1} read_trade={:.1} read_market={:.1} fileread={:.1} | micro_total={:.1} micro=[{}] | feat_total={:.1} feat_n={} feat_med={:.1} feat_max={:.1} | am_parse={:.1} am_detect={:.1} am_ala={:.1} | v3_parse={:.1} v3_detect={:.1} v3_ala={:.1} | f_cols={:.1} f_colstats={:.1} f_lyap={:.1} f_corr={:.1} | n_events=[{}] range_ns={:.1} range_calls={} scanned={} ala_calls={} | m20={:.1} m21={:.1} m20_iters={} m21_iters={} m20_alloc={} | v3_loop={:.1} v3_iters={}",
            r,
            wall.as_secs_f64() * 1000.0,
            ms(tm.total_ns),
            ms(tm.read_trade_ns),
            ms(tm.read_market_ns),
            ms(tm.file_read_ns),
            micro_total,
            micro.join(","),
            feat_total,
            feat.len(),
            ms(feat_median),
            ms(feat_max),
            ams(allo_microstructure::T_AM_PARSE.load(Ordering::Relaxed)),
            ams(allo_microstructure::T_AM_DETECT.load(Ordering::Relaxed)),
            ams(allo_microstructure::T_AM_ALA.load(Ordering::Relaxed)),
            ams(allo_microstructure_v3::T_V3_PARSE.load(Ordering::Relaxed)),
            ams(allo_microstructure_v3::T_V3_DETECT.load(Ordering::Relaxed)),
            ams(allo_microstructure_v3::T_V3_ALA.load(Ordering::Relaxed)),
            ams(features_f64::T_F_COLS.load(Ordering::Relaxed)),
            ams(features_f64::T_F_COLSTATS.load(Ordering::Relaxed)),
            ams(features_f64::T_F_LYAP.load(Ordering::Relaxed)),
            ams(features_f64::T_F_CORR.load(Ordering::Relaxed)),
            tm.n_events.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
            ams(allo_microstructure::T_RANGE.load(Ordering::Relaxed)),
            allo_microstructure::N_RANGE_CALLS.load(Ordering::Relaxed),
            allo_microstructure::N_SCANNED.load(Ordering::Relaxed),
            allo_microstructure::N_EVENTS.load(Ordering::Relaxed),
            ams(allo_microstructure::T_M20.load(Ordering::Relaxed)),
            ams(allo_microstructure::T_M21.load(Ordering::Relaxed)),
            allo_microstructure::N_M20_ITERS.load(Ordering::Relaxed),
            allo_microstructure::N_M21_ITERS.load(Ordering::Relaxed),
            allo_microstructure::N_M20_ALLOC.load(Ordering::Relaxed),
            ams(allo_microstructure_v3::T_V3_M2021.load(Ordering::Relaxed)),
            allo_microstructure_v3::N_V3_ITERS.load(Ordering::Relaxed),
        );
        if r == 0 {
            let s: Vec<String> = v.iter().map(|x| format!("{:.9e}", *x as f64)).collect();
            println!("DUMP={}", s.join(","));
        }
    }
}

