//! Super-Factor ("黑旋风四人团牌阵") high-throughput screening, assembly, and materialization engine.
//!
//! Exports zero-copy C-ABI entry points (`sf_*`) directly from `rust_pyfunc`'s shared library
//! for full-history 50-face dual-track (`FixedCal` + `TurnEq`) expansion, double-buffered store
//! scanning, 4-member `{1.0, 0.5}` squad assembly, and exact `10-Barra` / `ColdV2` evaluation.

pub mod kernels;
pub mod store_direct;

use rayon::prelude::*;
use std::sync::OnceLock;

static SF_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

#[inline]
pub(crate) fn pool() -> &'static rayon::ThreadPool {
    SF_POOL.get_or_init(|| {
        let default_n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(64)
            .clamp(1, 400);
        rayon::ThreadPoolBuilder::new()
            .num_threads(default_n)
            .build()
            .expect("failed to build SF_POOL")
    })
}

#[no_mangle]
pub extern "C" fn sf_init_threads(n_threads: usize) {
    let n = n_threads.clamp(1, 512);
    let _ = SF_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to build SF_POOL")
    });
}

/// Computes individual-stock dynamic lookback window lengths `K in [2..40]` for cumulative
/// free-float turnover thresholds `[15%, 30%, 60%]`.
#[no_mangle]
pub unsafe extern "C" fn sf_build_turnover_win_lens(
    fturn_ptr: *const f32, // (t_len, n_stocks)
    t_len: usize,
    n_stocks: usize,
    out_win_ptr: *mut u8,  // (3, t_len, n_stocks)
) {
    if t_len == 0 || n_stocks == 0 || fturn_ptr.is_null() || out_win_ptr.is_null() {
        return;
    }
    let tn = t_len * n_stocks;
    let fturn = std::slice::from_raw_parts(fturn_ptr, tn);
    let out_win = std::slice::from_raw_parts_mut(out_win_ptr, 3 * tn);
    let thetas = [15.0f32, 30.0f32, 60.0f32];

    pool().install(|| {
        out_win
            .par_chunks_mut(n_stocks)
            .enumerate()
            .for_each(|(row_idx, win_row)| {
                let hi = row_idx / t_len;
                let t = row_idx % t_len;
                let theta = thetas[hi];
                let max_k = (t + 1).min(40);
                for s in 0..n_stocks {
                    let mut acc = 0.0f32;
                    let mut chosen = max_k;
                    for k in 0..max_k {
                        let ft = fturn[(t - k) * n_stocks + s];
                        if ft.is_finite() && ft > 0.0 {
                            acc += ft;
                        }
                        if acc >= theta {
                            chosen = (k + 1).max(2).min(max_k);
                            break;
                        }
                    }
                    win_row[s] = chosen as u8;
                }
            });
    });
}
