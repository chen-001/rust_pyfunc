//! tail_v8_selfcheck：v8 各模块与生产实现的逐位对账入口（调试用 pyfunction）。
//!
//! 用法（Python）：
//!   import rust_pyfunc as rp
//!   print(rp.tail_v8_selfcheck("roll", "/home/chenzongwei/neu_lab/data_yupei_real"))
//!   print(rp.tail_v8_selfcheck("pf",   ...))
//!   print(rp.tail_v8_selfcheck("bt",   ...))
//!   print(rp.tail_v8_selfcheck("neu",  ...))
//!
//! 每个模块在自己的文件里实现 `selfcheck(data_dir) -> String`。

use pyo3::prelude::*;
use pyo3::types::PyString;

/// 生产 preflight 的公开包装（原函数在 tail_v5_pipeline 内，已改为 pub(crate)）。
pub fn prod_preflight(
    slot: &ndarray::ArrayView2<f32>,
    restrict: &ndarray::ArrayView2<f32>,
    thr_maj: f64,
    thr_zero: f64,
    thr_nan: f64,
) -> crate::tail_v5_pipeline::PreflightReport {
    crate::tail_v5_pipeline::preflight_quality_check(slot, restrict, thr_maj, thr_zero, thr_nan)
}

#[pyfunction]
#[pyo3(signature = (module, data_dir))]
pub fn tail_v8_selfcheck<'py>(
    py: Python<'py>,
    module: String,
    data_dir: String,
) -> PyResult<&'py PyString> {
    let out = py.allow_threads(|| match module.as_str() {
        "roll" => crate::tail_v8_roll::selfcheck(&data_dir),
        "pf" => crate::tail_v8_preflight::selfcheck(&data_dir),
        "bt" => crate::tail_v8_backtest::selfcheck(&data_dir),
        "neu" => selfcheck_neu(&data_dir),
        other => format!("未知模块 {other}（可选 roll / pf / bt / neu）"),
    });
    Ok(PyString::new(py, &out))
}

/// 中性化按日期块入口的对账：把 13 个面按块调用拼起来，与整张调用逐位比较。
fn selfcheck_neu(data_dir: &str) -> String {
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    let dates: Vec<i32> = match read_npy::<_, ndarray::Array1<i32>>(format!("{data_dir}/dates.npy"))
    {
        Ok(v) => v.to_vec(),
        Err(e) => return format!("[neu] 读 dates 失败: {e}"),
    };
    // stocks.npy 是 numpy object 数组，ndarray-npy 读不了；改用同目录的纯文本 stocks.txt
    let stocks: Vec<String> = match std::fs::read_to_string(format!("{data_dir}/stocks.txt")) {
        Ok(t) => t.lines().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        Err(e) => return format!("[neu] 读 stocks.txt 失败: {e}"),
    };
    let restrict: Array2<f32> = read_npy(format!("{data_dir}/restrict.npy")).unwrap();
    let industry: Array2<f64> = read_npy(format!("{data_dir}/industry.npy")).unwrap();
    let style_path = std::env::var("V8_STYLE_PATH").unwrap_or_else(|_| {
        "/home/chenzongwei/database/barra/barra_daily_together_jason.parquet".to_string()
    });
    let style = match crate::factor_neutralization_io_optimized::IOOptimizedStyleData::
        load_from_parquet_io_optimized(&style_path)
    {
        Ok(s) => s,
        Err(e) => return format!("[neu] 加载风格数据失败: {e}"),
    };
    let shared = match crate::factor_neutralize_std::neutralize_std_precompute(
        &industry, &restrict, &style, &dates, &stocks,
    ) {
        Ok(s) => s,
        Err(e) => return format!("[neu] 预计算失败: {e}"),
    };
    let names: Vec<String> = std::fs::read_to_string(format!("{data_dir}/sample_names.txt"))
        .unwrap_or_default()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let (t, n) = restrict.dim();
    let mut lines = Vec::new();
    let mut all_pass = true;
    let mut n_cmp = 0usize;
    for nm in names.iter().take(2) {
        let raw: Array2<f32> = read_npy(format!("{data_dir}/factor_{nm}.npy")).unwrap();
        let ranked =
            crate::tail_v5_pipeline::rank_and_fill_missing_cross_sectional_median(&raw, &restrict);
        for &bs in &[64usize, 997, 2818] {
            let mut t0 = 0usize;
            let mut cat = Array2::<f32>::from_elem((t, n), f32::NAN);
            while t0 < t {
                let t1 = (t0 + bs).min(t);
                let blk = ranked.slice(ndarray::s![t0..t1, ..]);
                let outs = crate::factor_neutralize_std::neutralize_std_slots_f32_v2_resid_batch_range(
                    &[blk],
                    &shared,
                    true,
                    t0,
                    t1,
                )
                .unwrap();
                cat.slice_mut(ndarray::s![t0..t1, ..]).assign(&outs[0]);
                t0 = t1;
            }
            let full = crate::factor_neutralize_std::neutralize_std_slots_f32_v2_resid_batch(
                &[ranked.view()],
                &shared,
                true,
            )
            .unwrap();
            n_cmp += 1;
            let mut mm = 0usize;
            for (a, b) in cat.iter().zip(full[0].iter()) {
                let same = (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits();
                if !same {
                    mm += 1;
                }
            }
            if mm > 0 {
                all_pass = false;
            }
            lines.push(format!("  {nm} block={bs}: 不一致 {mm} 格"));
        }
    }
    // ---- 计时（单线程）：13 面「整张 batch 一次」vs「按 bs 分块 range 拼起来」----
    // 13 面 = smooth_1 + w=5/10/20 × {mean,max,min,std}（与生产 variant 的面集合一致）
    if let Some(nm) = names.first() {
        let raw: Array2<f32> = read_npy(format!("{data_dir}/factor_{nm}.npy")).unwrap();
        let ranked =
            crate::tail_v5_pipeline::rank_and_fill_missing_cross_sectional_median(&raw, &restrict);
        let mut slots: Vec<Array2<f32>> = vec![ranked.clone()];
        for &w in &[5usize, 10, 20] {
            let (m, x, mn, sd) =
                crate::tail_v2_rank_roll_factor::rolling_stats_f32_serial(&ranked, w, w / 2);
            slots.push(m);
            slots.push(x);
            slots.push(mn);
            slots.push(sd);
        }
        let ns = slots.len();
        let views: Vec<ndarray::ArrayView2<f32>> = slots.iter().map(|s| s.view()).collect();
        let tmr = std::time::Instant::now();
        let full13 = crate::factor_neutralize_std::neutralize_std_slots_f32_v2_resid_batch(
            &views, &shared, true,
        )
        .unwrap();
        let t_full = tmr.elapsed().as_secs_f64();
        lines.push(format!("  [计时] {ns} 面整张 batch: {t_full:.3}s"));
        for &bs in &[64usize, 997, 2818] {
            let mut cat13: Vec<Array2<f32>> =
                (0..ns).map(|_| Array2::<f32>::from_elem((t, n), f32::NAN)).collect();
            let tmr = std::time::Instant::now();
            let mut a = 0usize;
            while a < t {
                let b1 = (a + bs).min(t);
                let blk: Vec<ndarray::ArrayView2<f32>> =
                    slots.iter().map(|s| s.slice(ndarray::s![a..b1, ..])).collect();
                let outs =
                    crate::factor_neutralize_std::neutralize_std_slots_f32_v2_resid_batch_range(
                        &blk, &shared, true, a, b1,
                    )
                    .unwrap();
                for k in 0..ns {
                    cat13[k].slice_mut(ndarray::s![a..b1, ..]).assign(&outs[k]);
                }
                a = b1;
            }
            let t_blk = tmr.elapsed().as_secs_f64();
            let mut mm13 = 0usize;
            for k in 0..ns {
                for (x, y) in cat13[k].iter().zip(full13[k].iter()) {
                    if !((x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits()) {
                        mm13 += 1;
                    }
                }
            }
            lines.push(format!(
                "  [计时] {ns} 面按块 range bs={bs}: {t_blk:.3}s (相对整张 {:.2}x) 不一致 {mm13} 格",
                t_blk / t_full
            ));
        }
    }
    lines.insert(
        0,
        format!(
            "[neu range] 逐位一致: {}  ({n_cmp} 组比较)",
            if all_pass { "PASS" } else { "FAIL" }
        ),
    );
    lines.join("\n")
}
