use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::{Datelike, NaiveDateTime};
use crossbeam::channel::{unbounded, Receiver, RecvTimeoutError, Sender};
#[cfg(feature = "hdf5")]
use hdf5_metno as hdf5;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use ndarray_npy::{read_npy, write_npy};
use numpy::IntoPyArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

use crate::factor_neutralization_io_optimized::IOOptimizedStyleData;
use crate::tail_v2_rank_roll_factor::rank_roll_block_f32_with_parallel;

#[cfg(target_family = "unix")]
use nix::sys::signal::{kill, Signal};
#[cfg(target_family = "unix")]
use nix::unistd::Pid;

pub(crate) const EPS: f64 = 1e-12;

fn default_true() -> bool {
    true
}

fn default_nan_f64() -> f64 {
    f64::NAN
}
const DEFAULT_TAIL_V4_FULLTEST_IDLE_TIMEOUT_SECS: u64 = 1800;
const TAIL_V5_FULLTEST_RESULT_POLL_SECS: u64 = 1;

type TailV4PidRegistry = Arc<Mutex<HashMap<usize, u32>>>;

struct TailV4WorkerPidGuard {
    registry: TailV4PidRegistry,
    worker_id: usize,
}

impl TailV4WorkerPidGuard {
    fn new(registry: TailV4PidRegistry, worker_id: usize, pid: u32) -> Self {
        if let Ok(mut guard) = registry.lock() {
            guard.insert(worker_id, pid);
        }
        Self {
            registry,
            worker_id,
        }
    }
}

impl Drop for TailV4WorkerPidGuard {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.registry.lock() {
            guard.remove(&self.worker_id);
        }
    }
}

pub(crate) fn format_hms(total_secs: u64) -> (u64, u64, u64) {
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    (hours, minutes, seconds)
}

#[derive(Clone)]
pub(crate) struct TailSelectionConfig {
    pub(crate) cover_rate: f64,
    pub(crate) ret_point_neu_gap5: f64,
    pub(crate) ret_point_neu_gap1: f64,
    pub(crate) ic_point_neu_gap5: f64,
    pub(crate) ic_point_neu_gap1: f64,
    pub(crate) ret_point_gap5: f64,
    pub(crate) ret_point_gap1: f64,
    pub(crate) ic_point_gap5: f64,
    pub(crate) ic_point_gap1: f64,
    pub(crate) ic_more_important_gap5: Option<f64>,
    pub(crate) ic_more_important_gap1: Option<f64>,
    pub(crate) majority_count_threshold: f64,
    pub(crate) zero_max_threshold: f64,
    pub(crate) nan_max_threshold: f64,
    pub(crate) save_all_metrics: bool,
    /// ic_only 模式：跳过收益回测（十分组/多空组合），只计算 IC 序列与 IC 汇总。
    /// 收益相关 summary 字段填 0.0（无意义值，避免 serde_json 拒绝 NaN）；
    /// 筛选时仅按中性化 IC 通道选取，Python 侧由 cut1_rate=None 等触发。
    pub(crate) ic_only: bool,
}

#[derive(Clone)]
pub(crate) struct SharedInputs {
    pub(crate) dates: Arc<Vec<i32>>,
    pub(crate) stocks: Arc<Vec<String>>,
    pub(crate) windows: Arc<Vec<usize>>,
    pub(crate) fold: bool,
    pub(crate) min_valid: usize,
    pub(crate) backtest_start: i32,
    pub(crate) legacy_style_data: Arc<IOOptimizedStyleData>,
    pub(crate) industry_neutralize: bool,
    /// 生产标准中性化所需的行业码矩阵 (T,N, 模板轴); None=使用旧 neutralize 路径
    pub(crate) industry: Option<Arc<Array2<f64>>>,
    /// 标准中性化管线中不随因子变化的预计算量 (barra/size rank、行业分级码、
    /// restrict、模板轴映射)，一次性展开 Arc 共享，避免每因子重建 barra(10×T×N)。
    /// None=未启用标准中性化 (industry 为 None 的旧路径)。
    pub(crate) neutralize_std_shared:
        Option<Arc<crate::factor_neutralize_std::NeutralizeStdShared>>,
    pub(crate) ret_gap1: Arc<Array2<f32>>,
    pub(crate) ret_sum_gap1: Arc<Array2<f32>>,
    pub(crate) ret_gap5: Arc<Array2<f32>>,
    pub(crate) ret_sum_gap5: Arc<Array2<f32>>,
    pub(crate) restrict: Arc<Array2<f32>>,
    pub(crate) index_ret: Arc<Array1<f32>>,
    pub(crate) config: Arc<TailSelectionConfig>,
    /// O1 优化 (2026-09): 收益秩预计算。orders[g][date] = ret_sum 全行按 (值, index)
    /// 排序的股票索引; 回测时对当日有效子集 walk 出子集内 ordinal 秩, 替代每个
    /// slot 重新对收益子集排序 (与 legacy_spearman_correlation 的排序语义一致)。
    pub(crate) bt_pre: Option<Arc<BtPrecomputed>>,
    /// v8 一档：restrict → 1 字节/格可交易掩码（全 run 一次，所有面复用）
    pub(crate) free_mask: Option<Arc<crate::tail_v8_preflight::FreeMask>>,
    /// v8 二档：v3 中性化（比 v2 快 2.7~4.2×，数值逐位一致）的派生索引，全 run 建一次
    pub(crate) v3_shared: Option<Arc<crate::tail_v8_neu_v3::V3Shared>>,
}

/// O1 收益秩预计算 (因子无关, 全 run 一次)。
#[derive(Clone, Default)]
pub struct BtPrecomputed {
    pub(crate) orders_g1: Vec<Vec<u32>>,
    pub(crate) orders_g5: Vec<Vec<u32>>,
}

#[derive(Clone)]
pub(crate) struct TailTask {
    pub(crate) source_factor: String,
    pub(crate) factor_path: String,
}

impl TailTask {
    pub(crate) fn new(source_factor: String, factor_path: String) -> Self {
        Self {
            source_factor,
            factor_path,
        }
    }
}

/// 跨模块构造 SharedInputs（供 tail_backtest_engine 调用）
pub(crate) fn build_shared_inputs(
    dates: Vec<i32>,
    stocks: Vec<String>,
    windows: Vec<usize>,
    fold: bool,
    min_valid: usize,
    backtest_start: i32,
    industry_neutralize: bool,
    industry: Option<Array2<f64>>,
    style_vars_dir: &str,
    ret_gap1_path: &str,
    ret_sum_gap1_path: &str,
    ret_gap5_path: &str,
    ret_sum_gap5_path: &str,
    restrict_path: &str,
    index_ret_path: &str,
    config: TailSelectionConfig,
) -> Result<SharedInputs, String> {
    // ---- P2 跨 run 固定预计算缓存 ----
    // 键 = dates + stocks + style sha256 + restrict sha256 + ret_sum sha256 ×2 + industry sha256。
    // 缓存目录 = dirname(restrict.npy)/_engine_shared_cache/（不随 cache_root 被 force_restart 清掉）。
    // 任何失败（目录不可写、文件缺失等）都退化为「不缓存」，行为与改动前一致。
    let shared_cache = if crate::tail_shared_cache::cache_enabled() {
        match crate::tail_shared_cache::SharedCache::open(
            restrict_path,
            style_vars_dir,
            ret_sum_gap1_path,
            ret_sum_gap5_path,
            &dates,
            &stocks,
            industry.as_ref(),
        ) {
            Ok(c) => Some(c),
            Err(e) => {
                println!("⚠️ [shared-cache] 缓存不可用，本次全量重算: {}", e);
                None
            }
        }
    } else {
        None
    };

    let t0 = Instant::now();
    let style_data = match shared_cache.as_ref().and_then(|c| c.load_style()) {
        Some(d) => {
            println!(
                "✅ [shared-cache] style 命中 (读盘 {:.2}s)",
                t0.elapsed().as_secs_f64()
            );
            d
        }
        None => {
            let d = IOOptimizedStyleData::load_from_vars_h5(style_vars_dir)
                .map_err(|e| e.to_string())?;
            if let Some(c) = &shared_cache {
                if let Err(e) = c.store_style(&d) {
                    println!("⚠️ [shared-cache] style 落盘失败: {}", e);
                }
            }
            println!(
                "❄️ [shared-cache] style 未命中，H5 解析 {:.2}s",
                t0.elapsed().as_secs_f64()
            );
            d
        }
    };
    // npy 的 fortran_order 是文件自带的：`load_tail_v2_backtest_inputs` 里 ret 由
    // `DataFrame.to_numpy(...).T` 得来（F 序），而 `np.save` **保留**内存序，于是同一批
    // 缓存目录里既有 C 序也有 F 序的 npy。v8 回测逐日按行访问 ret/restrict，
    // 非标准布局会让 `row().as_slice()` 拿不到切片、`row.iter()` 退化成跨步访问。
    // 这里一次性转成标准行主序（已经是标准布局就不拷贝）。
    fn to_standard_layout(a: Array2<f32>) -> Array2<f32> {
        if a.is_standard_layout() {
            a
        } else {
            a.as_standard_layout().into_owned()
        }
    }
    let restrict: Array2<f32> = to_standard_layout(
        read_npy(restrict_path).map_err(|e| format!("读取 restrict.npy 失败: {}", e))?,
    );
    // v8 一档：restrict → 1 字节/格可交易掩码，全 run 建一次
    let free_mask = Some(Arc::new(crate::tail_v8_preflight::build_free_mask(
        &restrict.view(),
    )));
    let ret_g1: Array2<f32> = to_standard_layout(
        read_npy(ret_gap1_path).map_err(|e| format!("读取 ret_gap1.npy 失败: {}", e))?,
    );
    let ret_s1: Array2<f32> = to_standard_layout(
        read_npy(ret_sum_gap1_path).map_err(|e| format!("读取 ret_sum_gap1.npy 失败: {}", e))?,
    );
    let ret_g5: Array2<f32> = to_standard_layout(
        read_npy(ret_gap5_path).map_err(|e| format!("读取 ret_gap5.npy 失败: {}", e))?,
    );
    let ret_s5: Array2<f32> = to_standard_layout(
        read_npy(ret_sum_gap5_path).map_err(|e| format!("读取 ret_sum_gap5.npy 失败: {}", e))?,
    );
    let index_v: Array1<f32> =
        read_npy(index_ret_path).map_err(|e| format!("读取 index_ret.npy 失败: {}", e))?;

    // 标准中性化预计算：一次性展开 barra/size/行业分级码 (模板轴)，Arc 共享。
    // 仅在启用标准中性化 (industry 非 None) 时预计算；旧路径 (industry=None) 不触发。
    let neutralize_std_shared = match &industry {
        Some(ind) => {
            let t0 = Instant::now();
            let ns = match shared_cache.as_ref().and_then(|c| c.load_neutral(ind)) {
                Some(v) => {
                    println!(
                        "✅ [shared-cache] neutralize 命中 (读盘 {:.2}s)",
                        t0.elapsed().as_secs_f64()
                    );
                    v
                }
                None => {
                    let v = crate::factor_neutralize_std::neutralize_std_precompute(
                        ind,
                        &restrict,
                        &style_data,
                        &dates,
                        &stocks,
                    )?;
                    if let Some(c) = &shared_cache {
                        if let Err(e) = c.store_neutral(&v) {
                            println!("⚠️ [shared-cache] neutralize 落盘失败: {}", e);
                        }
                    }
                    println!(
                        "❄️ [shared-cache] neutralize 未命中，重算 {:.2}s",
                        t0.elapsed().as_secs_f64()
                    );
                    v
                }
            };
            Some(Arc::new(ns))
        }
        None => None,
    };

    // v8 二档：v3 中性化派生索引（依赖 neutralize_std_shared，全 run 建一次）
    let v3_shared = match &neutralize_std_shared {
        Some(ns) => Some(Arc::new(crate::tail_v8_neu_v3::V3Shared::build(ns.clone())?)),
        None => None,
    };

    // O1 收益秩预计算 (因子无关): 每日期对 ret_sum_gap1/ret_sum_gap5 全行按
    // (值, index) 排序。按 (mono_key32(v), index) 的 radix 稳定排序, 与
    // ordinal_ranks 的 sort_by 语义一致 (-0/+0 合并、NaN 置末)。
    let bt_pre = {
        let t0 = Instant::now();
        Some(Arc::new(
            match shared_cache.as_ref().and_then(|c| c.load_bt()) {
                Some(v) => {
                    println!(
                        "✅ [shared-cache] bt_pre 命中 (读盘 {:.2}s)",
                        t0.elapsed().as_secs_f64()
                    );
                    v
                }
                None => {
                    let v = build_bt_precomputed(&ret_s1, &ret_s5)?;
                    if let Some(c) = &shared_cache {
                        if let Err(e) = c.store_bt(&v) {
                            println!("⚠️ [shared-cache] bt_pre 落盘失败: {}", e);
                        }
                    }
                    println!(
                        "❄️ [shared-cache] bt_pre 未命中，重算 {:.2}s",
                        t0.elapsed().as_secs_f64()
                    );
                    v
                }
            },
        ))
    };
    if let Some(c) = &shared_cache {
        println!("[shared-cache] key={} dir={}", c.key(), c.dir_display());
    }

    Ok(SharedInputs {
        dates: Arc::new(dates),
        stocks: Arc::new(stocks),
        windows: Arc::new(windows),
        fold,
        min_valid,
        backtest_start,
        legacy_style_data: Arc::new(style_data),
        industry_neutralize,
        industry: industry.map(Arc::new),
        neutralize_std_shared,
        ret_gap1: Arc::new(ret_g1),
        ret_sum_gap1: Arc::new(ret_s1),
        ret_gap5: Arc::new(ret_g5),
        ret_sum_gap5: Arc::new(ret_s5),
        restrict: Arc::new(restrict),
        index_ret: Arc::new(index_v),
        config: Arc::new(config),
        bt_pre,
        free_mask,
        v3_shared,
    })
}

pub(crate) fn build_selection_config(
    cover_rate: f64,
    ret_point_neu_gap5: f64,
    ret_point_neu_gap1: f64,
    ic_point_neu_gap5: f64,
    ic_point_neu_gap1: f64,
    ret_point_gap5: f64,
    ret_point_gap1: f64,
    ic_point_gap5: f64,
    ic_point_gap1: f64,
    ic_more_important_gap5: Option<f64>,
    ic_more_important_gap1: Option<f64>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
    save_all_metrics: bool,
    ic_only: bool,
) -> TailSelectionConfig {
    TailSelectionConfig {
        cover_rate,
        ret_point_neu_gap5,
        ret_point_neu_gap1,
        ic_point_neu_gap5,
        ic_point_neu_gap1,
        ret_point_gap5,
        ret_point_gap1,
        ic_point_gap5,
        ic_point_gap1,
        ic_more_important_gap5,
        ic_more_important_gap1,
        majority_count_threshold,
        zero_max_threshold,
        nan_max_threshold,
        save_all_metrics,
        ic_only,
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct SummaryRowRecord {
    pub(crate) factor_name: String,
    pub(crate) stage: String,
    pub(crate) gap: i32,
    pub(crate) source_factor: String,
    #[serde(default = "default_true")]
    pub(crate) preflight_passed: bool,
    #[serde(rename = "IC_mean")]
    pub(crate) ic_mean: f64,
    #[serde(rename = "IR")]
    pub(crate) ir: f64,
    pub(crate) annualized_return: f64,
    pub(crate) sharpe_ratio: f64,
    pub(crate) max_drawdown: f64,
    pub(crate) date_size: i32,
    pub(crate) ratio_mean: f64,
    pub(crate) hedge_annualized_return: f64,
    pub(crate) hedge_annualized_sharpe_ratio: f64,
    pub(crate) hedge_max_drawdown: f64,
    /// 双边单调度 SSM（5 段版）。ic_only 模式或旧缓存回读时为 NaN。
    #[serde(rename = "SSM", default = "default_nan_f64")]
    pub(crate) ssm: f64,
    /// 概率版双边单调度 MPROB。ic_only 模式或旧缓存回读时为 NaN。
    #[serde(rename = "MPROB", default = "default_nan_f64")]
    pub(crate) mprob: f64,
    /// Balanced Rank Concordance = min(BRC_S, BRC_L)。无可用日时为 NaN。
    #[serde(rename = "BRC", default = "default_nan_f64")]
    pub(crate) brc: f64,
    /// BRC 的「从低端切」半段均值 BRC_S。
    #[serde(rename = "BRC_S", default = "default_nan_f64")]
    pub(crate) brc_s: f64,
    /// BRC 的「从高端切」半段均值 BRC_L。
    #[serde(rename = "BRC_L", default = "default_nan_f64")]
    pub(crate) brc_l: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct IcRecord {
    pub(crate) factor_name: String,
    #[serde(default)]
    pub(crate) dates: Vec<i32>,
    pub(crate) values: Vec<f32>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub(crate) struct TailTaskResult {
    pub(crate) source_factor: String,
    pub(crate) raw_summary_gap1: Vec<SummaryRowRecord>,
    pub(crate) raw_summary_gap5: Vec<SummaryRowRecord>,
    pub(crate) neu_summary_gap1: Vec<SummaryRowRecord>,
    pub(crate) neu_summary_gap5: Vec<SummaryRowRecord>,
    pub(crate) raw_ic_gap1: Vec<IcRecord>,
    pub(crate) raw_ic_gap5: Vec<IcRecord>,
    pub(crate) neu_ic_gap1: Vec<IcRecord>,
    pub(crate) neu_ic_gap5: Vec<IcRecord>,
    pub(crate) all_raw_summary_gap1: Vec<SummaryRowRecord>,
    pub(crate) all_raw_summary_gap5: Vec<SummaryRowRecord>,
    pub(crate) all_neu_summary_gap1: Vec<SummaryRowRecord>,
    pub(crate) all_neu_summary_gap5: Vec<SummaryRowRecord>,
    pub(crate) all_raw_ic_gap1: Vec<IcRecord>,
    pub(crate) all_raw_ic_gap5: Vec<IcRecord>,
    pub(crate) all_neu_ic_gap1: Vec<IcRecord>,
    pub(crate) all_neu_ic_gap5: Vec<IcRecord>,
    #[serde(default = "default_nan_f64")]
    pub(crate) raw_cover_before_fill: f64,
    #[serde(default = "default_nan_f64")]
    pub(crate) raw_cover_after_fill: f64,
    pub(crate) derived_factor_count: usize,
    #[serde(default)]
    pub(crate) passed: bool,
    #[serde(default)]
    pub(crate) eliminated_by_raw_cover: bool,
    #[serde(default)]
    pub(crate) any_window_passed_preflight: bool,
    #[serde(default)]
    pub(crate) preflight_maj_failed_windows: usize,
    #[serde(default)]
    pub(crate) preflight_zero_failed_windows: usize,
    #[serde(default)]
    pub(crate) preflight_nan_failed_windows: usize,
}

#[derive(Default)]
pub(crate) struct AggregatedCandidates {
    pub(crate) raw_summary_gap1: Vec<SummaryRowRecord>,
    pub(crate) raw_summary_gap5: Vec<SummaryRowRecord>,
    pub(crate) neu_summary_gap1: Vec<SummaryRowRecord>,
    pub(crate) neu_summary_gap5: Vec<SummaryRowRecord>,
    pub(crate) raw_ic_gap1: HashMap<String, IcRecord>,
    pub(crate) raw_ic_gap5: HashMap<String, IcRecord>,
    pub(crate) neu_ic_gap1: HashMap<String, IcRecord>,
    pub(crate) neu_ic_gap5: HashMap<String, IcRecord>,
    pub(crate) all_raw_summary_gap1: Vec<SummaryRowRecord>,
    pub(crate) all_raw_summary_gap5: Vec<SummaryRowRecord>,
    pub(crate) all_neu_summary_gap1: Vec<SummaryRowRecord>,
    pub(crate) all_neu_summary_gap5: Vec<SummaryRowRecord>,
    pub(crate) all_raw_ic_gap1: HashMap<String, IcRecord>,
    pub(crate) all_raw_ic_gap5: HashMap<String, IcRecord>,
    pub(crate) all_neu_ic_gap1: HashMap<String, IcRecord>,
    pub(crate) all_neu_ic_gap5: HashMap<String, IcRecord>,
}

impl AggregatedCandidates {
    pub(crate) fn merge_task(&mut self, task: TailTaskResult) {
        self.raw_summary_gap1.extend(task.raw_summary_gap1);
        self.raw_summary_gap5.extend(task.raw_summary_gap5);
        self.neu_summary_gap1.extend(task.neu_summary_gap1);
        self.neu_summary_gap5.extend(task.neu_summary_gap5);
        for record in task.raw_ic_gap1 {
            self.raw_ic_gap1.insert(record.factor_name.clone(), record);
        }
        for record in task.raw_ic_gap5 {
            self.raw_ic_gap5.insert(record.factor_name.clone(), record);
        }
        for record in task.neu_ic_gap1 {
            self.neu_ic_gap1.insert(record.factor_name.clone(), record);
        }
        for record in task.neu_ic_gap5 {
            self.neu_ic_gap5.insert(record.factor_name.clone(), record);
        }
        self.all_raw_summary_gap1.extend(task.all_raw_summary_gap1);
        self.all_raw_summary_gap5.extend(task.all_raw_summary_gap5);
        self.all_neu_summary_gap1.extend(task.all_neu_summary_gap1);
        self.all_neu_summary_gap5.extend(task.all_neu_summary_gap5);
        for record in task.all_raw_ic_gap1 {
            self.all_raw_ic_gap1.insert(record.factor_name.clone(), record);
        }
        for record in task.all_raw_ic_gap5 {
            self.all_raw_ic_gap5.insert(record.factor_name.clone(), record);
        }
        for record in task.all_neu_ic_gap1 {
            self.all_neu_ic_gap1.insert(record.factor_name.clone(), record);
        }
        for record in task.all_neu_ic_gap5 {
            self.all_neu_ic_gap5.insert(record.factor_name.clone(), record);
        }
    }
}

#[derive(Clone)]
pub(crate) struct LegacyBacktestResult {
    /// 前 10 个是历史字段（IC_mean/IR/annualized_return/.../hedge_max_drawdown），
    /// 第 11 个（下标 10）是双边单调度 SSM，第 12 个（下标 11）是概率版 MPROB，
    /// 第 13/14/15 个（下标 12/13/14）是 BRC / BRC_S / BRC_L。
    /// ic_only 模式下没有组收益，SSM/MPROB 填 NaN；BRC 不依赖十分组收益，两种模式都算。
    pub(crate) summary: [f64; 15],
    pub(crate) ic_dates: Vec<i32>,
    pub(crate) ic_values: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
struct TailV4FulltestTask {
    task_key: String,
    factor_name: String,
    stage: String,
    gap: i32,
}

#[derive(Serialize, Deserialize)]
struct TailV4FulltestWorkerConfig {
    ver: String,
    temp_root: String,
    source_dir: String,
    factor_names: Vec<String>,
    start_date: String,
    backtest_start_date: String,
    end_date: String,
    style_vars_dir: String,
    min_valid: usize,
    index_name: String,
}

#[derive(Serialize, Deserialize)]
struct TailV4FulltestWorkerResult {
    ok: bool,
    task_key: String,
    factor_name: String,
    stage: String,
    gap: i32,
    error: Option<String>,
}

pub(crate) fn nanmean_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if !value.is_nan() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

pub(crate) fn nanstd_population(values: &[f64]) -> f64 {
    let mean = nanmean_f64(values);
    if mean.is_nan() {
        return f64::NAN;
    }
    let mut sq_sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if !value.is_nan() {
            let delta = value - mean;
            sq_sum += delta * delta;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        (sq_sum / count as f64).sqrt()
    }
}

pub(crate) fn sample_std(values: &[f64]) -> f64 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for &value in values {
        if !value.is_nan() {
            sum += value;
            count += 1;
        }
    }
    if count < 2 {
        return f64::NAN;
    }
    let mean = sum / count as f64;
    let mut sq_sum = 0.0;
    for &value in values {
        if !value.is_nan() {
            let delta = value - mean;
            sq_sum += delta * delta;
        }
    }
    (sq_sum / (count as f64 - 1.0)).sqrt()
}

pub(crate) fn annualized_sharpe_sample(values: &[f64]) -> f64 {
    let std = sample_std(values);
    if std.is_nan() || std <= EPS {
        return f64::NAN;
    }
    nanmean_f64(values) / std * 250.0_f64.sqrt()
}

pub(crate) fn max_drawdown_from_returns(values: &[f64]) -> f64 {
    let mut cumulative = 0.0;
    let mut peak = 0.0;
    let mut max_drawdown = 0.0;
    for &value in values {
        if !value.is_nan() {
            cumulative += value;
        }
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = peak - cumulative;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    max_drawdown
}

fn average_ranks(values: &[f32]) -> Vec<f64> {
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    indexed.sort_by(|lhs, rhs| {
        lhs.1
            .partial_cmp(&rhs.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.0.cmp(&rhs.0))
    });

    let mut ranks = vec![f64::NAN; values.len()];
    let mut start = 0usize;
    while start < indexed.len() {
        let value = indexed[start].1;
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == value {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for item in indexed.iter().take(end).skip(start) {
            ranks[item.0] = avg_rank;
        }
        start = end;
    }
    ranks
}

fn ordinal_ranks(values: &[f32]) -> Vec<i64> {
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f32)>>();
    indexed.sort_by(|lhs, rhs| match (lhs.1.is_nan(), rhs.1.is_nan()) {
        (true, true) => lhs.0.cmp(&rhs.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => lhs
            .1
            .partial_cmp(&rhs.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.0.cmp(&rhs.0)),
    });
    let mut ranks = vec![0i64; values.len()];
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = rank as i64;
    }
    ranks
}

fn legacy_spearman_correlation(x: &[f32], y: &[f32]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let xx = ordinal_ranks(x);
    let yy = ordinal_ranks(y);
    let n = x.len() as f64;
    let mut diff_sq_sum = 0.0;
    for idx in 0..x.len() {
        let diff = xx[idx] - yy[idx];
        diff_sq_sum += (diff * diff) as f64;
    }
    1.0 - 6.0 * diff_sq_sum / (n * (n * n - 1.0))
}

// ============================================================================
// O1 优化 (2026-09): f32 radix 秩 + 收益秩预排序。
// 语义与 ordinal_ranks / average_ranks / legacy_spearman_correlation 完全一致:
//   - 排序键 = (mono_key32(v), index), 稳定 radix;
//   - -0.0 规范化为 +0.0 (与 f32 == 判等语义一致);
//   - 调用方保证输入无 NaN (filtered 集合已过滤信号/收益有限)。
// ============================================================================

#[inline]
fn mono_key32(v: f32) -> u32 {
    let v = if v == 0.0 { 0.0 } else { v };
    let bits = v.to_bits();
    if bits >> 31 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    }
}

/// 按 u32 单调位序 key 的 8-bit LSD 稳定 radix 排序 (4 趟)。
/// 初始 order 按 index 升序时等价 (key, index) 总序 —— 与 radix_sort_keys64 (8 趟)
/// 的排序结果逐位一致, 排序开销减半。
fn radix_sort_u32_keys(keys: &[u32], order: &mut Vec<usize>, tmp: &mut Vec<usize>) {
    let n = order.len();
    if n < 2 {
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut count = [0usize; 256];
    for shift in (0..32).step_by(8) {
        count.fill(0);
        for &i in order.iter() {
            count[((keys[i] >> shift) & 0xff) as usize] += 1;
        }
        let mut acc = 0usize;
        for c in count.iter_mut() {
            let t = *c;
            *c = acc;
            acc += t;
        }
        for &i in order.iter() {
            let b = ((keys[i] >> shift) & 0xff) as usize;
            tmp[count[b]] = i;
            count[b] += 1;
        }
        std::mem::swap(order, tmp);
    }
}

/// ordinal 秩 (与 ordinal_ranks 一致: 等值按 index 稳定序, 无 NaN)。
/// O1b (2026-09): u32 4-pass radix, 与旧 (key,index) 8-pass 总序逐位一致。
/// P3a: 直接复用 rank_both_radix_into（与 rank_both_radix 同一份实现，结果不变）。
fn ordinal_ranks_radix(values: &[f32]) -> Vec<i64> {
    let mut keys: Vec<u32> = Vec::new();
    let mut order: Vec<usize> = Vec::new();
    let mut tmp: Vec<usize> = Vec::new();
    let mut ordinal: Vec<i64> = Vec::new();
    let mut avg: Vec<f64> = Vec::new();
    rank_both_radix_into(values, &mut keys, &mut order, &mut tmp, &mut ordinal, &mut avg);
    ordinal
}

/// 平均秩 (与 average_ranks 一致: 等值组取平均, f32 == 判等)。
/// O1b (2026-09): u32 4-pass radix, 与旧 (key,index) 8-pass 总序逐位一致。
/// P3a: 直接复用 rank_both_radix_into（与 rank_both_radix 同一份实现，结果不变）。
pub(crate) fn average_ranks_radix(values: &[f32]) -> Vec<f64> {
    let mut keys: Vec<u32> = Vec::new();
    let mut order: Vec<usize> = Vec::new();
    let mut tmp: Vec<usize> = Vec::new();
    let mut ordinal: Vec<i64> = Vec::new();
    let mut avg: Vec<f64> = Vec::new();
    rank_both_radix_into(values, &mut keys, &mut order, &mut tmp, &mut ordinal, &mut avg);
    avg
}

/// O1b: 一次排序同时产出 ordinal 秩与平均秩 (与分别调用上述两个函数逐位一致)。
///
/// 复用缓冲版 (P3a): keys/order/tmp/ordinal/avg 全部由调用方持有并跨行复用，
/// 热路径零堆分配。语义与旧实现逐位相同（旧实现即本函数的薄包装）：
///   - 排序键 = (mono_key32(v), index)，稳定 radix，初始 order 按 index 升序；
///   - -0.0 规范化为 +0.0；等值组（f32 ==）取平均秩；调用方保证无 NaN。
#[allow(clippy::too_many_arguments)]
pub(crate) fn rank_both_radix_into(
    values: &[f32],
    keys: &mut Vec<u32>,
    order: &mut Vec<usize>,
    tmp: &mut Vec<usize>,
    ordinal: &mut Vec<i64>,
    avg: &mut Vec<f64>,
) {
    let n = values.len();
    keys.clear();
    keys.resize(n, 0u32);
    for (i, &v) in values.iter().enumerate() {
        keys[i] = mono_key32(v);
    }
    order.clear();
    order.extend(0..n);
    radix_sort_u32_keys(keys, order, tmp);
    ordinal.clear();
    ordinal.resize(n, 0i64);
    for (rank, &idx) in order.iter().enumerate() {
        ordinal[idx] = rank as i64;
    }
    avg.clear();
    avg.resize(n, f64::NAN);
    let mut start = 0usize;
    while start < n {
        let value = values[order[start]];
        let mut end = start + 1;
        while end < n && values[order[end]] == value {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for &idx in order[start..end].iter() {
            avg[idx] = avg_rank;
        }
        start = end;
    }
}

/// 薄包装：每次调用新建 5 个缓冲（保留给未走 BtAcc 的调用点，行为不变）。
pub(crate) fn rank_both_radix(values: &[f32]) -> (Vec<i64>, Vec<f64>) {
    let mut keys: Vec<u32> = Vec::new();
    let mut order: Vec<usize> = Vec::new();
    let mut tmp: Vec<usize> = Vec::new();
    let mut ordinal: Vec<i64> = Vec::new();
    let mut avg: Vec<f64> = Vec::new();
    rank_both_radix_into(values, &mut keys, &mut order, &mut tmp, &mut ordinal, &mut avg);
    (ordinal, avg)
}

/// BtPrecomputed 构建: 每日期 ret_sum 全行按 (值, index) 排序 (NaN 置末)。
/// O1b (2026-09): u32 4-pass (稳定 + 初始 index 升序 ≡ (key, index) 总序)。
pub fn build_bt_precomputed(
    ret_sum_g1: &Array2<f32>,
    ret_sum_g5: &Array2<f32>,
) -> Result<BtPrecomputed, String> {
    let n_dates = ret_sum_g1.nrows();
    let n_stocks = ret_sum_g1.ncols();
    if ret_sum_g5.dim() != (n_dates, n_stocks) {
        return Err("O1 预计算: ret_sum_gap1/ret_sum_gap5 形状不一致".to_string());
    }
    let mut build = |ret_sum: &Array2<f32>| -> Result<Vec<Vec<u32>>, String> {
        let mut orders = Vec::with_capacity(n_dates);
        for d in 0..n_dates {
            let mut keys: Vec<u32> = Vec::with_capacity(n_stocks);
            for (j, &v) in ret_sum.row(d).iter().enumerate() {
                let k = if v.is_nan() { u32::MAX } else { mono_key32(v) };
                keys.push(k);
            }
            let mut order: Vec<usize> = (0..n_stocks).collect();
            let mut tmp: Vec<usize> = Vec::new();
            radix_sort_u32_keys(&keys, &mut order, &mut tmp);
            orders.push(order.into_iter().map(|x| x as u32).collect());
        }
        Ok(orders)
    };
    Ok(BtPrecomputed {
        orders_g1: build(ret_sum_g1)?,
        orders_g5: build(ret_sum_g5)?,
    })
}

fn count_open_symbols(restrict_row: ArrayView1<'_, f32>) -> usize {
    restrict_row
        .iter()
        .filter(|&&value| value.is_finite() && value == 0.0)
        .count()
}

pub(crate) fn precompute_open_symbol_counts(restrict: &ArrayView2<'_, f32>) -> Vec<usize> {
    (0..restrict.shape()[0])
        .map(|row_idx| count_open_symbols(restrict.row(row_idx)))
        .collect()
}

fn has_enough_unique_values(
    factor: &ArrayView3<'_, f32>,
    slot_idx: usize,
    min_unique: usize,
) -> bool {
    let mut seen = HashSet::<u32>::new();
    for raw_idx in 0..factor.shape()[0].saturating_sub(1) {
        for stock_idx in 0..factor.shape()[1] {
            let value = factor[[raw_idx, stock_idx, slot_idx]];
            if value.is_finite() {
                seen.insert(value.to_bits());
                if seen.len() >= min_unique {
                    return true;
                }
            }
        }
    }
    false
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PreflightReport {
    pub(crate) passed: bool,
    pub(crate) majority_count_mean: f64,
    pub(crate) zero_ratio_mean: f64,
    pub(crate) nan_ratio_mean: f64,
}

pub(crate) fn default_legacy_backtest_result() -> LegacyBacktestResult {
    LegacyBacktestResult {
        summary: [f64::NAN; 15],
        ic_dates: Vec::new(),
        ic_values: Vec::new(),
    }
}

/// 双边单调度 SSM（5 段版）。
///
/// 输入 `group_returns[d][t]` 是第 d 组的逐日收益。先对时间取平均得到 10 组组均收益，
/// 把曲线统一成向上（末组低于首组时整条倒序），再对五段各自算「净位移 ÷ 段内台阶幅度之和」：
/// 整体（组 1-10）、空头段（组 1-5）、多头段（组 6-10）、首段（组 1-4）、尾段（组 7-10），
/// 取五段里最小的那个。
///
/// 为什么这样设计：
/// - 每一段各自归一，所以哪一侧幅度大都不额外加分，空头半边跨度放大 50 倍仍是 1.000；
/// - 首段/尾段只含 3 个台阶，端点那一步不被后面的大台阶稀释，所以第 1 组或第 10 组
///   不是极值时扣分更重（这正是「希望 1、10 组是收益最高与最低」那一条）；
/// - 只对组间差敏感，全市场涨跌不影响。
///
/// 值域 [-1, 1]，9 个台阶全顺着走时为 1.000。10 组纯噪声下 5% 分位约 -1.000、
/// 中位约 -0.347、95% 分位约 0.098，所以及格线取 0.098 而不是 0。
/// `ic_only=true` 时组收益为空，返回 NaN。
pub(crate) fn compute_ssm(group_returns: &[Vec<f64>], portf_num: usize) -> f64 {
    if portf_num != 10 || group_returns.len() != portf_num {
        return f64::NAN;
    }
    let n = group_returns[0].len();
    if n == 0 || group_returns.iter().any(|v| v.len() != n) {
        return f64::NAN;
    }
    let mut r = [0.0_f64; 10];
    for (d, col) in group_returns.iter().enumerate() {
        r[d] = col.iter().sum::<f64>() / n as f64;
    }
    if !r.iter().all(|v| v.is_finite()) {
        return f64::NAN;
    }
    if r[9] < r[0] {
        r.reverse();
    }
    // 1 起的闭区间 [i, j]：净位移 ÷ 段内台阶幅度之和。
    let seg = |i: usize, j: usize| -> f64 {
        let s = &r[i - 1..j];
        let mut den = 0.0_f64;
        for k in 1..s.len() {
            den += (s[k] - s[k - 1]).abs();
        }
        if den <= 0.0 {
            return 0.0;
        }
        (s[s.len() - 1] - s[0]) / den
    };
    let mut ssm = seg(1, 10);
    for v in [seg(1, 5), seg(6, 10), seg(1, 4), seg(7, 10)] {
        if v < ssm {
            ssm = v;
        }
    }
    ssm
}

/// Abramowitz & Stegun 7.1.26 的误差函数近似（五系数），与 `copula.rs::erf` 同一份系数。
/// 最大绝对误差 1.5e-7。
fn erf_as_7_1_26(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// 标准正态分布函数 `Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))`，erf 走 A&S 7.1.26。
fn std_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_as_7_1_26(x / std::f64::consts::SQRT_2))
}

/// 概率版双边单调度 MPROB。
///
/// 输入 `group_returns[d][t]` 是第 d 组的逐日收益。与 [`compute_ssm`] 用同一套端点规则定向
/// （第 10 组时间均值低于第 1 组时整条倒序），然后对 45 个组对各自算一条价差序列的
/// Newey-West（1994 自动选阶）t 统计量，转成正态概率后等权平均：
///
/// ```text
/// d_t     = group_returns[idx[j]][t] - group_returns[idx[i]][t]      （i < j，45 对）
/// mean    = Σ d_t / n,   e_t = d_t - mean
/// gamma_0 = Σ e_t² / n,  gamma_k = Σ_{t=k..n-1} e_t·e_{t-k} / n
/// nw_var  = (gamma_0 + 2·Σ_{k=1..L} (1 - k/(L+1))·gamma_k) / n
/// 该对贡献 = 2·Φ(mean / sqrt(nw_var)) - 1
/// MPROB   = Σ(45 对贡献) / 45
/// ```
///
/// `L = min(n-1, floor(4·(n/100)^(2/9)))` 是 Newey-West 的自动滞后阶（n=2424 时 L=8）。
///
/// 与 SSM 的分工：SSM 只看组均收益曲线的形状（样本均值的符号比较），MPROB 看每条价差
/// 序列的信噪比，所以「10bp ± 1bp」会比「0.01bp ± 5bp」拿到更高的分。值域 [-1, 1]；
/// 定向规则与 SSM 相同，因此整条曲线取负时严格反号（`Φ(-t) = 1-Φ(t)`），是方向无关的形状分。
///
/// 精度：erf 用 A&S 7.1.26 近似（最大绝对误差 1.5e-7）→ Φ 的绝对误差 ≤ 7.5e-8
/// → 每个组对 `2·Φ(t)-1` 的绝对误差 ≤ 1.5e-7（系数 2 把它放大一倍），45 对等权平均后
/// MPROB 的绝对误差 ≤ 1.5e-7（各对误差同向的最坏情形）。
///
/// 无法计算时返回 `NaN`：组数不是 10、任一列长度不等、日期数 n < 2、出现 NaN/±Inf。
///
/// 单对 `nw_var` 非有限或 ≤ 0 时该对贡献 0.0，这是**刻意的保守兜底**：Newey-West 方差
/// 为 0 说明该价差序列在这段窗口里没有可用波动，给不出置信度；为负则是强均值回复下
/// Bartlett 加权和把 gamma_0 抵消掉的结果，同样是「没有可用信息」。所以严格恒定的阶梯
/// （任意两组价差逐日不变，e_t ≡ 0 → gamma_0 = 0 → nw_var = 0）得 0.0 而不是 1.0，
/// 值域上界 1.0 只在「价差均值远大于其 Newey-West 标准误」的极限下取到。
/// 真实数据的组间价差逐日有波动，这条只会在「整段恒定」或「强均值回复」两种病态情形触发。
pub(crate) fn compute_mprob(group_returns: &[Vec<f64>], portf_num: usize) -> f64 {
    if portf_num != 10 || group_returns.len() != portf_num {
        return f64::NAN;
    }
    let n = group_returns[0].len();
    if n < 2 || group_returns.iter().any(|v| v.len() != n) {
        return f64::NAN;
    }
    if group_returns
        .iter()
        .any(|col| col.iter().any(|v| !v.is_finite()))
    {
        return f64::NAN;
    }
    // 时间均值 + 定向：与 compute_ssm 同一套端点规则。
    let mut r = [0.0_f64; 10];
    for (d, col) in group_returns.iter().enumerate() {
        r[d] = col.iter().sum::<f64>() / n as f64;
    }
    let mut idx = [0usize; 10];
    for (d, slot) in idx.iter_mut().enumerate() {
        *slot = if r[9] < r[0] { 9 - d } else { d };
    }
    // Newey-West(1994) 自动选阶。
    let l = (((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor()) as usize).min(n - 1);
    let l_f = l as f64;
    // e_t 缓冲复用（n 最大到回测窗口长度，45 对循环里不再分配）。
    let mut e = vec![0.0_f64; n];
    let mut acc = 0.0_f64;
    for i in 0..10 {
        for j in (i + 1)..10 {
            let a = &group_returns[idx[i]];
            let b = &group_returns[idx[j]];
            let mut mean = 0.0_f64;
            for t in 0..n {
                mean += b[t] - a[t];
            }
            mean /= n as f64;
            for t in 0..n {
                e[t] = b[t] - a[t] - mean;
            }
            let mut gamma0 = 0.0_f64;
            for v in e.iter() {
                gamma0 += v * v;
            }
            gamma0 /= n as f64;
            let mut acov = 0.0_f64;
            for k in 1..=l {
                let mut gamma_k = 0.0_f64;
                for t in k..n {
                    gamma_k += e[t] * e[t - k];
                }
                gamma_k /= n as f64;
                acov += (1.0 - k as f64 / (l_f + 1.0)) * gamma_k;
            }
            let nw_var = (gamma0 + 2.0 * acov) / n as f64;
            if !nw_var.is_finite() || nw_var <= 0.0 {
                continue;
            }
            acc += 2.0 * std_normal_cdf(mean / nw_var.sqrt()) - 1.0;
        }
    }
    acc / 45.0
}

/// Balanced Rank Concordance（BRC）的单日「半段均值」，返回 `(S_t, L_t)`。
///
/// 输入是某一天的**有效股票子集**：`signal` = 因子值 f，`future` = 未来收益 r
/// （与 Rank IC 同一目标，即 `ret_sum`），长度 n。定义（n = 子集长度）：
///
/// ```text
/// 1. r 转升序平均秩 R（并列取平均秩，取值 1..n）；
/// 2. 按下标稳定排序 f 得 order（f 并列时保持原始下标顺序）；
/// 3. W_k = Σ_{i<k} R[order[i]]，U_k = W_k - k(k+1)/2，
///    D_k = 1 - 2·U_k/(k·(n-k))，k = 1..n-1；
///    D_k = 2·A_k - 1，A_k = P(r_i < r_j | i 在低 factor 组、j 在高 factor 组)；
/// 4. m = floor(n/2)，S_t = mean_{k=1..m} D_k，L_t = mean_{k=1..m} D_{n-k}。
/// ```
///
/// S_t 只看「从低端切」，L_t 只看「从高端切」（两者共用同一个 f 升序排列）。值域都是
/// [-1, 1]：U_k = 0（低 factor 组的收益秩恰好是最小的 k 个）时 D_k = 1，
/// U_k = k(n-k)（完全反序）时 D_k = -1。
///
/// **不做任何「整条曲线取负」的定向处理**：D_k > 0 表示因子低值对应低收益。
/// 两条取负恒等式：
/// - 把 r 取负：每个 D_k 严格反号 → `(S_t, L_t) → (-S_t, -L_t)`；
/// - 把 f 取负（因子序整体反转）：`(S_t, L_t) → (-L_t, -S_t)`。
///
/// 两者都让 `min(S_t, L_t) → -max(S_t, L_t)`。
///
/// 复杂度 O(n log n)：两次排序（r 的平均秩、f 的稳定序），D 的累加是 O(n) 一趟。
/// 无法计算时返回 `(NaN, NaN)`：n < 2、两侧长度不等、或任一输入含 NaN/±Inf。
///
/// 实现走复用缓冲 + radix（[`brc_day_halves_into`]），结果与旧版（每次新建 Vec +
/// 比较排序）逐位一致；`brc_day_halves` 只是「新建一次缓冲」的薄包装。
pub(crate) fn brc_day_halves(signal: &[f32], future: &[f32]) -> (f64, f64) {
    let mut buf = BrcBuffers::default();
    brc_day_halves_into(signal, future, &mut buf)
}

/// 「未来收益升序平均秩」的 radix 缓冲（语义与 [`rank_both_radix_into`] 的五个参数一致）。
/// 只给**没有 orders 的通用/参考路径**用；opt 与 v8 走 [`brc_day_halves_from_ranks_into`]，
/// 不需要这些缓冲。
#[derive(Default)]
pub(crate) struct BrcRankBuffers {
    keys: Vec<u32>,
    order: Vec<usize>,
    tmp: Vec<usize>,
    ordinal: Vec<i64>,
    avg: Vec<f64>,
}

/// BRC 单日计算的复用缓冲：跨日复用，热路径零堆分配。
///
/// - `rank`：通用路径（无 orders，参考实现/pyfunction）的 radix 秩缓冲；
/// - `factor_order`：signal 升序稳定序（下标，长度 n）；
/// - `future_order` / `future_avg`：快速路径（opt/v8）由 walk 的 ordinal 秩逆置换 +
///   并列段取平均得到的 future 平均秩。
#[derive(Default)]
pub(crate) struct BrcBuffers {
    rank: BrcRankBuffers,
    factor_order: Vec<u32>,
    future_order: Vec<u32>,
    future_avg: Vec<f64>,
}

/// W_k / D_k 收口（两条路径共用同一份累加，保证逐位一致）。
///
/// 一趟扫 k = 1..n-1：W_k 递推，k ≤ m 进 S，k ≥ n-m 进 L（L 的第 k 项是 D_{n-k}）。
#[inline]
fn brc_halves_from_orders(factor_order: &[u32], future_avg: &[f64]) -> (f64, f64) {
    let n = factor_order.len();
    let nf = n as f64;
    let m = n / 2;
    let mut w = 0.0_f64;
    let mut s_sum = 0.0_f64;
    let mut l_sum = 0.0_f64;
    for k in 1..n {
        w += future_avg[factor_order[k - 1] as usize];
        let kf = k as f64;
        let u = w - kf * (kf + 1.0) / 2.0;
        let d = 1.0 - 2.0 * u / (kf * (nf - kf));
        if k <= m {
            s_sum += d;
        }
        if k >= n - m {
            l_sum += d;
        }
    }
    let mf = m as f64;
    (s_sum / mf, l_sum / mf)
}

/// 通用路径核心：给定「因子升序稳定序」`factor_order`（长度 n，元素是 0..n-1 的下标）
/// 与 `future`，返回 `(S_t, L_t)`。future 的升序平均秩走 radix
/// （[`rank_both_radix_into`]，已注明与 `average_ranks` 逐位一致）。
fn brc_from_factor_order(
    factor_order: &[u32],
    future: &[f32],
    rank_buf: &mut BrcRankBuffers,
) -> (f64, f64) {
    let n = future.len();
    if n < 2 || factor_order.len() != n {
        return (f64::NAN, f64::NAN);
    }
    if future.iter().any(|v| !v.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    rank_both_radix_into(
        future,
        &mut rank_buf.keys,
        &mut rank_buf.order,
        &mut rank_buf.tmp,
        &mut rank_buf.ordinal,
        &mut rank_buf.avg,
    );
    brc_halves_from_orders(factor_order, &rank_buf.avg)
}

/// 秩→下标逆置换（O(n)）：把 ordinal 秩转成升序下标序，写进 `order`。
/// ordinal 越界时返回 false（调用方按「该日不可算」处理）。
#[inline]
fn invert_ordinal_into(ordinal: &[i64], n: usize, order: &mut Vec<u32>) -> bool {
    order.clear();
    order.resize(n, 0u32);
    for (idx, &rank) in ordinal.iter().enumerate() {
        let r = rank as usize;
        if rank < 0 || r >= n {
            return false;
        }
        order[r] = idx as u32;
    }
    true
}

/// 复用缓冲版：由 `(signal, future)` 算单日 `(S_t, L_t)`，不新建任何堆缓冲。
///
/// 与旧实现逐位一致：因子升序稳定序与 future 的平均秩都走 radix
/// （[`rank_both_radix_into`] 与 `average_ranks` / `(值, index)` 稳定序逐位一致）。
pub(crate) fn brc_day_halves_into(
    signal: &[f32],
    future: &[f32],
    buf: &mut BrcBuffers,
) -> (f64, f64) {
    let n = signal.len();
    if n < 2 || future.len() != n {
        return (f64::NAN, f64::NAN);
    }
    if signal.iter().any(|v| !v.is_finite()) || future.iter().any(|v| !v.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    rank_both_radix_into(
        signal,
        &mut buf.rank.keys,
        &mut buf.rank.order,
        &mut buf.rank.tmp,
        &mut buf.rank.ordinal,
        &mut buf.rank.avg,
    );
    if !invert_ordinal_into(&buf.rank.ordinal, n, &mut buf.factor_order) {
        return (f64::NAN, f64::NAN);
    }
    let factor_order: &[u32] = &buf.factor_order;
    brc_from_factor_order(factor_order, future, &mut buf.rank)
}

/// **快速路径（opt / v8 生产用）**：signal 与 future 的子集 ordinal 秩都已由引擎现成给出，
/// BRC 不再做任何排序、不新建任何堆缓冲，每日只多一趟 O(n)。
///
/// - `signal_ordinal[i]` = 子集第 i 位股票在 signal 升序里的位置（0..n-1，来自
///   [`rank_both_radix_into`] 的输出）；
/// - `future_ordinal[i]` = 同上，但按 ret_sum 升序（来自 `orders[raw_eff_idx]` 的
///   gen/stamp walk，见 `legacy_backtest_single_factor_with_effective_opt`）；
/// - `future[i]` = 子集的 ret_sum 值，只用于「并列段」判定。
///
/// future 的平均秩由 `future_ordinal` 逆置换出升序下标序后，对连续等值段取
/// `(start+1+end)/2` —— 与 [`average_ranks`] 的 `(值, index)` 稳定序 + 等值段取平均
/// **逐位相同**（orders 就是按 (值, index) 预排序的，子集内并列段连续且组内 ordinal 连续）。
/// 调用方保证 signal 全有限（回测过滤已保证），`future` 的非有限判定在这里做。
pub(crate) fn brc_day_halves_from_ranks_into(
    signal_ordinal: &[i64],
    future_ordinal: &[i64],
    future: &[f32],
    buf: &mut BrcBuffers,
) -> (f64, f64) {
    let n = future.len();
    if n < 2 || signal_ordinal.len() != n || future_ordinal.len() != n {
        return (f64::NAN, f64::NAN);
    }
    if future.iter().any(|v| !v.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    if !invert_ordinal_into(signal_ordinal, n, &mut buf.factor_order) {
        return (f64::NAN, f64::NAN);
    }
    if !invert_ordinal_into(future_ordinal, n, &mut buf.future_order) {
        return (f64::NAN, f64::NAN);
    }
    // 并列段取平均秩：与 average_ranks 的等值段分组（f32 ==）和 (start+1+end)/2 完全同式。
    buf.future_avg.clear();
    buf.future_avg.resize(n, f64::NAN);
    let mut start = 0usize;
    while start < n {
        let value = future[buf.future_order[start] as usize];
        let mut end = start + 1;
        while end < n && future[buf.future_order[end] as usize] == value {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for k in start..end {
            let idx = buf.future_order[k] as usize;
            buf.future_avg[idx] = avg_rank;
        }
        start = end;
    }
    let factor_order: &[u32] = &buf.factor_order;
    let future_avg: &[f64] = &buf.future_avg;
    brc_halves_from_orders(factor_order, future_avg)
}

/// 把逐日 `(S_t, L_t)` 的累加和收口成 `(BRC_S, BRC_L, BRC)`。
///
/// `days` 是可计算的天数（[`brc_day_halves`] 返回有限值的天数）；一天都没有时三列都是 NaN。
/// `BRC = min(BRC_S, BRC_L)`，与 SSM/MPROB 不同，这里不做定向，取两个半段的保守下界。
pub(crate) fn finalize_brc(s_sum: f64, l_sum: f64, days: usize) -> (f64, f64, f64) {
    if days == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let brc_s = s_sum / days as f64;
    let brc_l = l_sum / days as f64;
    (brc_s, brc_l, brc_s.min(brc_l))
}

/// 调试/独立验证用：直接调 [`brc_day_halves`]（不经过回测引擎）。
#[pyfunction]
pub fn tail_brc_halves_f32(signal: Vec<f32>, future: Vec<f32>) -> (f64, f64) {
    brc_day_halves(&signal, &future)
}

fn effective_raw_indices_for_slot(
    factor: &ArrayView3<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    slot_idx: usize,
) -> Vec<usize> {
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    let mut effective_raw_indices = Vec::<usize>::new();
    for raw_eff_idx in 1..n_dates {
        if dates[raw_eff_idx] <= backtest_start {
            continue;
        }
        let signal_row_idx = raw_eff_idx - 1;
        let mut all_nan = true;
        for stock_idx in 0..n_stocks {
            if factor[[signal_row_idx, stock_idx, slot_idx]].is_finite() {
                all_nan = false;
                break;
            }
        }
        if !all_nan {
            effective_raw_indices.push(raw_eff_idx);
        }
    }
    effective_raw_indices
}

pub(crate) fn preflight_quality_check(
    raw_values: &ArrayView2<f32>,
    restrict: &ArrayView2<f32>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PreflightReport {
    let n_dates = raw_values.shape()[0];
    let n_stocks = raw_values.shape()[1];
    let mut majority_sum: f64 = 0.0;
    let mut nan_ratio_sum: f64 = 0.0;
    let mut zero_ratio_sum: f64 = 0.0;
    let mut valid_date_count: usize = 0;

    // O1b 优化: majority 检查只关心"最多重复值个数"。
    // (a) 阈值 >= 股票数时判定恒真 (max 计数 <= n_stocks), 跳过全部计数;
    // (b) 否则用 u32 bits 稳定归并计数 (radix 4-pass) 替代逐日 HashMap。
    // 分组按 value.to_bits() 等价, 结果与旧 HashMap 完全一致。
    let count_majority = majority_count_threshold < n_stocks as f64;

    if !count_majority {
        // 快路径: 只统计 zero/nan
        for t in 0..n_dates {
            let mut free_count: usize = 0;
            let mut nan_count: usize = 0;
            let mut zero_count: usize = 0;
            for s in 0..n_stocks {
                let val = raw_values[[t, s]];
                let is_free = restrict[[t, s]].is_finite() && restrict[[t, s]] == 0.0;
                if is_free {
                    free_count += 1;
                    if !val.is_finite() {
                        nan_count += 1;
                    } else if val == 0.0 {
                        zero_count += 1;
                    }
                }
            }
            if free_count > 0 {
                nan_ratio_sum += nan_count as f64 / free_count as f64;
                zero_ratio_sum += zero_count as f64 / free_count as f64;
                valid_date_count += 1;
            }
        }
        let nan_ratio_mean = if valid_date_count > 0 {
            nan_ratio_sum / valid_date_count as f64
        } else {
            0.0
        };
        let zero_ratio_mean = if valid_date_count > 0 {
            zero_ratio_sum / valid_date_count as f64
        } else {
            0.0
        };
        return PreflightReport {
            // majority 恒真 (threshold >= n_stocks); majority_count_mean 不参与任何
            // 可观测输出 (仅用于判定与计数), 返回 n_stocks 占位。
            passed: zero_ratio_mean < zero_max_threshold && nan_ratio_mean < nan_max_threshold,
            majority_count_mean: n_stocks as f64,
            zero_ratio_mean,
            nan_ratio_mean,
        };
    }

    let mut keys: Vec<u32> = Vec::with_capacity(n_stocks);
    let mut order: Vec<usize> = Vec::with_capacity(n_stocks);
    let mut tmp: Vec<usize> = Vec::with_capacity(n_stocks);
    for t in 0..n_dates {
        keys.clear();
        let mut free_count: usize = 0;
        let mut nan_count: usize = 0;
        let mut zero_count: usize = 0;
        for s in 0..n_stocks {
            let val = raw_values[[t, s]];
            let is_free = restrict[[t, s]].is_finite() && restrict[[t, s]] == 0.0;
            if is_free {
                free_count += 1;
                if !val.is_finite() {
                    nan_count += 1;
                } else if val == 0.0 {
                    zero_count += 1;
                }
            }
            if val.is_finite() {
                keys.push(val.to_bits());
            }
        }
        // 按 bits 分组计数 (与 HashMap 的 to_bits 分组一致); 顺序无关, 只需相邻相等
        let n = keys.len();
        order.clear();
        order.extend(0..n);
        if n >= 2 {
            radix_sort_u32_keys(&keys, &mut order, &mut tmp);
        }
        let mut max_count = 0usize;
        let mut start = 0usize;
        while start < n {
            let key = keys[order[start]];
            let mut end = start + 1;
            while end < n && keys[order[end]] == key {
                end += 1;
            }
            let c = end - start;
            if c > max_count {
                max_count = c;
            }
            start = end;
        }
        majority_sum += max_count as f64;
        if free_count > 0 {
            nan_ratio_sum += nan_count as f64 / free_count as f64;
            zero_ratio_sum += zero_count as f64 / free_count as f64;
            valid_date_count += 1;
        }
    }

    let majority_count_mean = if n_dates > 0 {
        majority_sum / n_dates as f64
    } else {
        0.0
    };
    let nan_ratio_mean = if valid_date_count > 0 {
        nan_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };
    let zero_ratio_mean = if valid_date_count > 0 {
        zero_ratio_sum / valid_date_count as f64
    } else {
        0.0
    };

    PreflightReport {
        passed: majority_count_mean <= majority_count_threshold
            && zero_ratio_mean < zero_max_threshold
            && nan_ratio_mean < nan_max_threshold,
        majority_count_mean,
        zero_ratio_mean,
        nan_ratio_mean,
    }
}

pub(crate) fn compute_raw_cover_rate(
    raw_values: &ArrayView2<f32>,
    restrict: &ArrayView2<f32>,
    ret: &ArrayView2<f32>,
    min_stocks: usize,
) -> f64 {
    let n_dates = raw_values.shape()[0];
    let n_stocks = raw_values.shape()[1];
    let mut ratios: Vec<f64> = Vec::new();

    for t in 0..n_dates {
        let mut free_count: usize = 0;
        let mut valid_count: usize = 0;
        for s in 0..n_stocks {
            let is_free = restrict[[t, s]].is_finite() && restrict[[t, s]] == 0.0;
            if is_free {
                free_count += 1;
                if !raw_values[[t, s]].is_nan() && ret[[t, s]].is_finite() {
                    valid_count += 1;
                }
            }
        }
        if free_count > 0 && valid_count >= min_stocks {
            ratios.push(valid_count as f64 / free_count as f64);
        }
    }

    if ratios.is_empty() {
        1.0
    } else {
        ratios.iter().sum::<f64>() / ratios.len() as f64
    }
}

fn legacy_backtest_single_factor_with_effective(
    factor: &ArrayView3<'_, f32>,
    ret: &ArrayView2<'_, f32>,
    ret_sum: &ArrayView2<'_, f32>,
    restrict: &ArrayView2<'_, f32>,
    index: &ArrayView1<'_, f32>,
    dates: &[i32],
    slot_idx: usize,
    gap: usize,
    portf_num: usize,
    effective_raw_indices: &[usize],
    open_symbol_counts: &[usize],
    ic_only: bool,
) -> LegacyBacktestResult {
    if effective_raw_indices.is_empty() {
        return default_legacy_backtest_result();
    }
    let n_stocks = factor.shape()[1];

    let date_size = effective_raw_indices.len();
    let mut group_returns = if ic_only {
        Vec::new()
    } else {
        vec![vec![0.0_f64; date_size]; portf_num]
    };
    let mut ratio_values = vec![f64::NAN; date_size];
    let mut ic_dates = Vec::<i32>::new();
    let mut ic_values_f64 = Vec::<f64>::new();
    let mut ic_values_f32 = Vec::<f32>::new();
    let mut filtered_signal = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_ret = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_future = Vec::<f32>::with_capacity(n_stocks);
    let mut group_sums = vec![0.0_f64; portf_num];
    let mut group_counts = vec![0usize; portf_num];
    let mut held_signal_row_idx = effective_raw_indices[0] - 1;
    let mut held_restrict_row_idx = effective_raw_indices[0] - 1;
    // BRC：IC 采样格点上逐日 (S_t, L_t) 的累加（与 IC 同一批日子）。
    let mut brc_s_sum = 0.0_f64;
    let mut brc_l_sum = 0.0_f64;
    let mut brc_days = 0usize;
    let mut brc_buf = BrcBuffers::default();

    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        if local_t % gap == 0 {
            held_signal_row_idx = raw_eff_idx - 1;
            held_restrict_row_idx = raw_eff_idx - 1;
        }

        filtered_signal.clear();
        filtered_ret.clear();
        filtered_future.clear();
        for stock_idx in 0..n_stocks {
            let signal_value = factor[[held_signal_row_idx, stock_idx, slot_idx]];
            let ret_value = ret[[raw_eff_idx, stock_idx]];
            let is_open = restrict[[held_restrict_row_idx, stock_idx]].is_finite()
                && restrict[[held_restrict_row_idx, stock_idx]] == 0.0;
            if signal_value.is_finite() && ret_value.is_finite() && is_open {
                filtered_signal.push(signal_value);
                filtered_ret.push(ret_value);
                filtered_future.push(ret_sum[[raw_eff_idx, stock_idx]]);
            }
        }

        if (local_t + 1) % gap == 0 {
            let ic_value = legacy_spearman_correlation(&filtered_future, &filtered_signal);
            ic_dates.push(dates[raw_eff_idx]);
            ic_values_f64.push(ic_value);
            ic_values_f32.push(ic_value as f32);
            let (s_t, l_t) = brc_day_halves_into(&filtered_signal, &filtered_future, &mut brc_buf);
            if s_t.is_finite() && l_t.is_finite() {
                brc_s_sum += s_t;
                brc_l_sum += l_t;
                brc_days += 1;
            }
        }

        let stocks_num = filtered_signal.len();
        if stocks_num < portf_num {
            continue;
        }

        let valid_symbol_num = open_symbol_counts
            .get(raw_eff_idx - 1)
            .copied()
            .unwrap_or(0);
        if valid_symbol_num > 0 {
            ratio_values[local_t] = stocks_num as f64 / valid_symbol_num as f64;
        }

        if ic_only {
            continue;
        }

        group_sums.fill(0.0);
        group_counts.fill(0);
        let ranks = average_ranks(&filtered_signal);
        for idx in 0..stocks_num {
            let pct = ranks[idx] / stocks_num as f64;
            let mut bucket = (pct * portf_num as f64).floor() as usize;
            if bucket >= portf_num {
                bucket = portf_num - 1;
            }
            group_sums[bucket] += filtered_ret[idx] as f64;
            group_counts[bucket] += 1;
        }
        for bucket in 0..portf_num {
            group_returns[bucket][local_t] = if group_counts[bucket] == 0 {
                0.0
            } else {
                group_sums[bucket] / group_counts[bucket] as f64
            };
        }
    }

    let ic_mean = nanmean_f64(&ic_values_f64);
    let ic_std = nanstd_population(&ic_values_f64);
    let ir = if ic_std.is_nan() || ic_std <= EPS {
        f64::NAN
    } else {
        ic_mean.abs() / ic_std * (250.0 / gap as f64).sqrt()
    };
    let (brc_s, brc_l, brc) = finalize_brc(brc_s_sum, brc_l_sum, brc_days);
    let summary = if ic_only {
        // ic_only 模式：跳过收益回测（十分组/多空组合），收益字段填 0.0
        // （该模式下无意义；serde_json 拒绝 NaN，故不用 NaN 占位）。
        // 末尾两项（下标 10 SSM、11 MPROB）没有组收益可算，填 NaN；
        // BRC 三列不依赖十分组收益，两种模式都算且逐位相同。
        [
            ic_mean,
            ir,
            0.0,
            0.0,
            0.0,
            date_size as f64,
            nanmean_f64(&ratio_values),
            0.0,
            0.0,
            0.0,
            f64::NAN,
            f64::NAN,
            brc,
            brc_s,
            brc_l,
        ]
    } else {
        let first_leg_cum = group_returns[0].iter().sum::<f64>();
        let last_leg_cum = group_returns[portf_num - 1].iter().sum::<f64>();
        let (long_idx, short_idx) = if first_leg_cum > last_leg_cum {
            (0usize, portf_num - 1)
        } else {
            (portf_num - 1, 0usize)
        };

        let mut ls_returns = vec![0.0_f64; date_size];
        let mut hedge_returns = vec![0.0_f64; date_size];
        for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
            let long_ret = group_returns[long_idx][local_t];
            let short_ret = group_returns[short_idx][local_t];
            ls_returns[local_t] = long_ret - short_ret;
            hedge_returns[local_t] = long_ret - index[raw_eff_idx] as f64;
        }

        [
            ic_mean,
            ir,
            nanmean_f64(&ls_returns) * 250.0,
            annualized_sharpe_sample(&ls_returns),
            max_drawdown_from_returns(&ls_returns),
            date_size as f64,
            nanmean_f64(&ratio_values),
            nanmean_f64(&hedge_returns) * 250.0,
            annualized_sharpe_sample(&hedge_returns),
            max_drawdown_from_returns(&hedge_returns),
            compute_ssm(&group_returns, portf_num),
            compute_mprob(&group_returns, portf_num),
            brc,
            brc_s,
            brc_l,
        ]
    };
    LegacyBacktestResult {
        summary,
        ic_dates,
        ic_values: ic_values_f32,
    }
}


/// O1 优化回测 (2026-09): 与 legacy_backtest_single_factor_with_effective 数值逐位一致,
/// 仅两处实现替换:
///   1. IC 的收益秩: 由"每个 slot 对收益子集重新排序" → "按预计算的全行序 walk 出
///      子集 ordinal 秩" (pre.orders_g1/g5), 与 ordinal_ranks 的 (值, index) 稳定序一致;
///   2. 信号秩: ordinal_ranks / average_ranks → radix 版本 (同排序语义)。
/// 过滤条件、ratio 计算、十分组、summary 完全同原实现。
#[allow(clippy::too_many_arguments)]
fn legacy_backtest_single_factor_with_effective_opt(
    factor: &ArrayView3<'_, f32>,
    ret: ArrayView2<'_, f32>,
    ret_sum: ArrayView2<'_, f32>,
    restrict: ArrayView2<'_, f32>,
    index: ArrayView1<'_, f32>,
    dates: &[i32],
    slot_idx: usize,
    gap: usize,
    portf_num: usize,
    effective_raw_indices: &[usize],
    open_symbol_counts: &[usize],
    ic_only: bool,
    pre: &BtPrecomputed,
) -> LegacyBacktestResult {
    if effective_raw_indices.is_empty() {
        return default_legacy_backtest_result();
    }
    let n_stocks = factor.shape()[1];
    let date_size = effective_raw_indices.len();
    let mut group_returns = if ic_only {
        Vec::new()
    } else {
        vec![vec![0.0_f64; date_size]; portf_num]
    };
    let mut ratio_values = vec![f64::NAN; date_size];
    let mut ic_dates = Vec::<i32>::new();
    let mut ic_values_f64 = Vec::<f64>::new();
    let mut ic_values_f32 = Vec::<f32>::new();
    let mut filtered_signal = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_ret = Vec::<f32>::with_capacity(n_stocks);
    let mut filtered_stock_idx = Vec::<u32>::with_capacity(n_stocks);
    // BRC 专用：只在 IC 采样格点上按 filtered_stock_idx 顺序收集 ret_sum，
    // 元素序与参考实现 filtered_future 完全一致（stock_idx 升序）。
    let mut filtered_future = Vec::<f32>::with_capacity(n_stocks);
    let mut group_sums = vec![0.0_f64; portf_num];
    let mut group_counts = vec![0usize; portf_num];
    let mut held_signal_row_idx = effective_raw_indices[0] - 1;
    let mut held_restrict_row_idx = effective_raw_indices[0] - 1;
    let mut gen = vec![0u32; n_stocks];
    let mut stamp = vec![0u32; n_stocks];
    let mut walk_buf = Vec::<i64>::with_capacity(n_stocks);
    let mut gen_id: u32 = 0;
    // 信号秩的复用缓冲（与 v8 累积器同一套写法：跨日复用，热路径零堆分配）。
    let mut rk_keys = Vec::<u32>::new();
    let mut rk_order = Vec::<usize>::new();
    let mut rk_tmp = Vec::<usize>::new();
    let mut rk_ordinal = Vec::<i64>::new();
    let mut rk_avg = Vec::<f64>::new();
    // BRC：IC 采样格点上逐日 (S_t, L_t) 的累加（与 IC 同一批日子）。
    let mut brc_s_sum = 0.0_f64;
    let mut brc_l_sum = 0.0_f64;
    let mut brc_days = 0usize;
    let mut brc_buf = BrcBuffers::default();
    let orders = if gap == 1 {
        &pre.orders_g1
    } else {
        &pre.orders_g5
    };

    for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
        if local_t % gap == 0 {
            held_signal_row_idx = raw_eff_idx - 1;
            held_restrict_row_idx = raw_eff_idx - 1;
        }
        filtered_signal.clear();
        filtered_ret.clear();
        filtered_stock_idx.clear();
        // 过滤与生产原实现完全一致 (全扫描; restrict[held] 与 ret[当前行] 判定)
        for stock_idx in 0..n_stocks {
            let signal_value = factor[[held_signal_row_idx, stock_idx, slot_idx]];
            let ret_value = ret[[raw_eff_idx, stock_idx]];
            let is_open = restrict[[held_restrict_row_idx, stock_idx]].is_finite()
                && restrict[[held_restrict_row_idx, stock_idx]] == 0.0;
            if signal_value.is_finite() && ret_value.is_finite() && is_open {
                filtered_signal.push(signal_value);
                filtered_ret.push(ret_value);
                filtered_stock_idx.push(stock_idx as u32);
            }
        }
        // O1b: 每日一次排序同时产出 ordinal 秩与平均秩（缓冲复用，零堆分配）。
        // gap 日的排序结果供 IC、BRC 与十分组复用; 非 gap 日只排一次用于十分组。
        let ranked_this_row = (local_t + 1) % gap == 0;
        if ranked_this_row {
            // BRC 的输入：按 filtered_stock_idx 顺序取当日 ret_sum 子集
            // （与参考实现 filtered_future 的取值与顺序一致）。
            filtered_future.clear();
            let ret_sum_row = ret_sum.row(raw_eff_idx);
            for &stk in filtered_stock_idx.iter() {
                filtered_future.push(ret_sum_row[stk as usize]);
            }
            // 收益秩: 预排序全行 walk 出子集 ordinal 秩 (gen 代标记防跨日串扰)
            gen_id += 1;
            let order = &orders[raw_eff_idx];
            for (pos, &stk) in filtered_stock_idx.iter().enumerate() {
                gen[stk as usize] = gen_id;
                stamp[stk as usize] = (pos + 1) as u32;
            }
            walk_buf.clear();
            walk_buf.resize(filtered_stock_idx.len(), 0);
            let mut counter = 0usize;
            for &stk in order {
                if gen[stk as usize] == gen_id {
                    walk_buf[stamp[stk as usize] as usize - 1] = counter as i64;
                    counter += 1;
                }
            }
            rank_both_radix_into(
                &filtered_signal,
                &mut rk_keys,
                &mut rk_order,
                &mut rk_tmp,
                &mut rk_ordinal,
                &mut rk_avg,
            );
            let n = filtered_signal.len() as f64;
            let mut diff_sq_sum = 0.0;
            for idx in 0..filtered_signal.len() {
                let diff = walk_buf[idx] - rk_ordinal[idx];
                diff_sq_sum += (diff * diff) as f64;
            }
            let ic_value = if n < 2.0 {
                f64::NAN
            } else {
                1.0 - 6.0 * diff_sq_sum / (n * (n * n - 1.0))
            };
            ic_dates.push(dates[raw_eff_idx]);
            ic_values_f64.push(ic_value);
            ic_values_f32.push(ic_value as f32);
            // BRC 走快速路径：signal 的 ordinal 秩（rk_ordinal）与 ret_sum 的子集 ordinal 秩
            // （walk_buf，上面刚 walk 出来）都是现成的 → 不排序、不分配，只多一趟 O(n)。
            let (s_t, l_t) = brc_day_halves_from_ranks_into(
                &rk_ordinal,
                &walk_buf,
                &filtered_future,
                &mut brc_buf,
            );
            if s_t.is_finite() && l_t.is_finite() {
                brc_s_sum += s_t;
                brc_l_sum += l_t;
                brc_days += 1;
            }
        }
        let stocks_num = filtered_signal.len();
        if stocks_num < portf_num {
            continue;
        }
        let valid_symbol_num = open_symbol_counts
            .get(raw_eff_idx - 1)
            .copied()
            .unwrap_or(0);
        if valid_symbol_num > 0 {
            ratio_values[local_t] = stocks_num as f64 / valid_symbol_num as f64;
        }
        if ic_only {
            continue;
        }
        group_sums.fill(0.0);
        group_counts.fill(0);
        if !ranked_this_row {
            // 非 gap 日：本行只做一次排序，结果直接用于十分组（与旧实现一致）。
            rank_both_radix_into(
                &filtered_signal,
                &mut rk_keys,
                &mut rk_order,
                &mut rk_tmp,
                &mut rk_ordinal,
                &mut rk_avg,
            );
        }
        let ranks: &[f64] = &rk_avg;
        for idx in 0..stocks_num {
            let pct = ranks[idx] / stocks_num as f64;
            let mut bucket = (pct * portf_num as f64).floor() as usize;
            if bucket >= portf_num {
                bucket = portf_num - 1;
            }
            group_sums[bucket] += filtered_ret[idx] as f64;
            group_counts[bucket] += 1;
        }
        for bucket in 0..portf_num {
            group_returns[bucket][local_t] = if group_counts[bucket] == 0 {
                0.0
            } else {
                group_sums[bucket] / group_counts[bucket] as f64
            };
        }
    }

    let ic_mean = nanmean_f64(&ic_values_f64);
    let ic_std = nanstd_population(&ic_values_f64);
    let ir = if ic_std.is_nan() || ic_std <= EPS {
        f64::NAN
    } else {
        ic_mean.abs() / ic_std * (250.0 / gap as f64).sqrt()
    };
    let (brc_s, brc_l, brc) = finalize_brc(brc_s_sum, brc_l_sum, brc_days);
    let summary = if ic_only {
        [
            ic_mean,
            ir,
            0.0,
            0.0,
            0.0,
            date_size as f64,
            nanmean_f64(&ratio_values),
            0.0,
            0.0,
            0.0,
            f64::NAN,
            f64::NAN,
            brc,
            brc_s,
            brc_l,
        ]
    } else {
        let first_leg_cum = group_returns[0].iter().sum::<f64>();
        let last_leg_cum = group_returns[portf_num - 1].iter().sum::<f64>();
        let (long_idx, short_idx) = if first_leg_cum > last_leg_cum {
            (0usize, portf_num - 1)
        } else {
            (portf_num - 1, 0usize)
        };
        let mut ls_returns = vec![0.0_f64; date_size];
        let mut hedge_returns = vec![0.0_f64; date_size];
        for (local_t, &raw_eff_idx) in effective_raw_indices.iter().enumerate() {
            let long_ret = group_returns[long_idx][local_t];
            let short_ret = group_returns[short_idx][local_t];
            ls_returns[local_t] = long_ret - short_ret;
            hedge_returns[local_t] = long_ret - index[raw_eff_idx] as f64;
        }
        [
            ic_mean,
            ir,
            nanmean_f64(&ls_returns) * 250.0,
            annualized_sharpe_sample(&ls_returns),
            max_drawdown_from_returns(&ls_returns),
            date_size as f64,
            nanmean_f64(&ratio_values),
            nanmean_f64(&hedge_returns) * 250.0,
            annualized_sharpe_sample(&hedge_returns),
            max_drawdown_from_returns(&hedge_returns),
            compute_ssm(&group_returns, portf_num),
            compute_mprob(&group_returns, portf_num),
            brc,
            brc_s,
            brc_l,
        ]
    };
    LegacyBacktestResult {
        summary,
        ic_dates,
        ic_values: ic_values_f32,
    }
}

/// O1 优化的单 slot gap1/gap5 回测包装 (与 legacy_backtest_gap1_gap5_single_slot 对应)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn legacy_backtest_gap1_gap5_single_slot_opt(
    slot: ArrayView2<'_, f32>,
    ret_gap1: ArrayView2<'_, f32>,
    ret_sum_gap1: ArrayView2<'_, f32>,
    ret_gap5: ArrayView2<'_, f32>,
    ret_sum_gap5: ArrayView2<'_, f32>,
    restrict: ArrayView2<'_, f32>,
    index: ArrayView1<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    portf_num: usize,
    open_symbol_counts: &[usize],
    ic_only: bool,
    pre: &BtPrecomputed,
) -> (LegacyBacktestResult, LegacyBacktestResult) {
    let n_dates = slot.nrows();
    let slot_block = slot.insert_axis(ndarray::Axis(2));
    if n_dates < 2 || !has_enough_unique_values(&slot_block, 0, 10) {
        return (
            default_legacy_backtest_result(),
            default_legacy_backtest_result(),
        );
    }
    let effective_raw_indices =
        effective_raw_indices_for_slot(&slot_block, dates, backtest_start, 0);
    (
        legacy_backtest_single_factor_with_effective_opt(
            &slot_block,
            ret_gap1,
            ret_sum_gap1,
            restrict,
            index,
            dates,
            0,
            1,
            portf_num,
            &effective_raw_indices,
            open_symbol_counts,
            ic_only,
            pre,
        ),
        legacy_backtest_single_factor_with_effective_opt(
            &slot_block,
            ret_gap5,
            ret_sum_gap5,
            restrict,
            index,
            dates,
            0,
            5,
            portf_num,
            &effective_raw_indices,
            open_symbol_counts,
            ic_only,
            pre,
        ),
    )
}

#[allow(dead_code)]
fn legacy_backtest_single_factor(
    factor: &ArrayView3<'_, f32>,
    ret: &ArrayView2<'_, f32>,
    ret_sum: &ArrayView2<'_, f32>,
    restrict: &ArrayView2<'_, f32>,
    index: &ArrayView1<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    slot_idx: usize,
    gap: usize,
    portf_num: usize,
    ic_only: bool,
) -> LegacyBacktestResult {
    if factor.shape()[0] < 2 || gap == 0 || portf_num == 0 {
        return default_legacy_backtest_result();
    }
    if !has_enough_unique_values(factor, slot_idx, 10) {
        return default_legacy_backtest_result();
    }
    let effective_raw_indices =
        effective_raw_indices_for_slot(factor, dates, backtest_start, slot_idx);
    let open_symbol_counts = precompute_open_symbol_counts(restrict);
    legacy_backtest_single_factor_with_effective(
        factor,
        ret,
        ret_sum,
        restrict,
        index,
        dates,
        slot_idx,
        gap,
        portf_num,
        &effective_raw_indices,
        &open_symbol_counts,
        ic_only,
    )
}

#[allow(dead_code)]
fn legacy_backtest_block_f32(
    factor: ArrayView3<'_, f32>,
    ret: ArrayView2<'_, f32>,
    ret_sum: ArrayView2<'_, f32>,
    restrict: ArrayView2<'_, f32>,
    index: ArrayView1<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    gap: usize,
    portf_num: usize,
    ic_only: bool,
) -> Result<Vec<LegacyBacktestResult>, String> {
    if gap == 0 {
        return Err("gap 必须大于 0".to_string());
    }
    if portf_num == 0 {
        return Err("portf_num 必须大于 0".to_string());
    }
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    if ret.shape() != [n_dates, n_stocks]
        || ret_sum.shape() != [n_dates, n_stocks]
        || restrict.shape() != [n_dates, n_stocks]
        || index.len() != n_dates
        || dates.len() != n_dates
    {
        return Err("legacy backtest 输入形状不匹配".to_string());
    }

    let n_factors = factor.shape()[2];
    let mut results = Vec::with_capacity(n_factors);
    for slot_idx in 0..n_factors {
        results.push(legacy_backtest_single_factor(
            &factor,
            &ret,
            &ret_sum,
            &restrict,
            &index,
            dates,
            backtest_start,
            slot_idx,
            gap,
            portf_num,
            ic_only,
        ));
    }
    Ok(results)
}

fn legacy_backtest_gap1_gap5_selected_slots_f32(
    factor: ArrayView3<'_, f32>,
    ret_gap1: ArrayView2<'_, f32>,
    ret_sum_gap1: ArrayView2<'_, f32>,
    ret_gap5: ArrayView2<'_, f32>,
    ret_sum_gap5: ArrayView2<'_, f32>,
    restrict: ArrayView2<'_, f32>,
    index: ArrayView1<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    selected_slots: &[usize],
    portf_num: usize,
    ic_only: bool,
) -> Result<(Vec<LegacyBacktestResult>, Vec<LegacyBacktestResult>), String> {
    if portf_num == 0 {
        return Err("portf_num 必须大于 0".to_string());
    }
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    if ret_gap1.shape() != [n_dates, n_stocks]
        || ret_sum_gap1.shape() != [n_dates, n_stocks]
        || ret_gap5.shape() != [n_dates, n_stocks]
        || ret_sum_gap5.shape() != [n_dates, n_stocks]
        || restrict.shape() != [n_dates, n_stocks]
        || index.len() != n_dates
        || dates.len() != n_dates
    {
        return Err("legacy backtest 输入形状不匹配".to_string());
    }

    let n_factors = factor.shape()[2];
    let open_symbol_counts = precompute_open_symbol_counts(&restrict);
    let mut gap1_results = Vec::with_capacity(selected_slots.len());
    let mut gap5_results = Vec::with_capacity(selected_slots.len());
    for &slot_idx in selected_slots {
        if slot_idx >= n_factors {
            return Err(format!("selected slot 越界: {} >= {}", slot_idx, n_factors));
        }
        if n_dates < 2 || !has_enough_unique_values(&factor, slot_idx, 10) {
            gap1_results.push(default_legacy_backtest_result());
            gap5_results.push(default_legacy_backtest_result());
            continue;
        }
        let effective_raw_indices =
            effective_raw_indices_for_slot(&factor, dates, backtest_start, slot_idx);
        gap1_results.push(legacy_backtest_single_factor_with_effective(
            &factor,
            &ret_gap1,
            &ret_sum_gap1,
            &restrict,
            &index,
            dates,
            slot_idx,
            1,
            portf_num,
            &effective_raw_indices,
            &open_symbol_counts,
            ic_only,
        ));
        gap5_results.push(legacy_backtest_single_factor_with_effective(
            &factor,
            &ret_gap5,
            &ret_sum_gap5,
            &restrict,
            &index,
            dates,
            slot_idx,
            5,
            portf_num,
            &effective_raw_indices,
            &open_symbol_counts,
            ic_only,
        ));
    }
    Ok((gap1_results, gap5_results))
}

fn select_factor_slots_block(
    factor: ArrayView3<'_, f32>,
    selected_slots: &[usize],
) -> Result<Array3<f32>, String> {
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    let n_slots = factor.shape()[2];
    let mut output = Array3::<f32>::from_elem((n_dates, n_stocks, selected_slots.len()), f32::NAN);
    for (local_idx, &slot_idx) in selected_slots.iter().enumerate() {
        if slot_idx >= n_slots {
            return Err(format!("selected slot 越界: {} >= {}", slot_idx, n_slots));
        }
        let src = factor.slice(s![.., .., slot_idx]);
        let mut dst = output.slice_mut(s![.., .., local_idx]);
        dst.assign(&src);
    }
    Ok(output)
}

fn collect_preflight_passed_slots(
    rolled_block: ArrayView3<'_, f32>,
    restrict: ArrayView2<'_, f32>,
    derived_names: &[String],
    config: &TailSelectionConfig,
    result: &mut TailTaskResult,
) -> Vec<usize> {
    let mut selected_slots = Vec::new();
    for (slot_idx, derived_name) in derived_names.iter().enumerate() {
        let slot_view = rolled_block.slice(s![.., .., slot_idx]);
        let pre_report = preflight_quality_check(
            &slot_view,
            &restrict,
            config.majority_count_threshold,
            config.zero_max_threshold,
            config.nan_max_threshold,
        );
        if !pre_report.passed {
            if pre_report.majority_count_mean > config.majority_count_threshold {
                result.preflight_maj_failed_windows += 1;
            }
            if pre_report.zero_ratio_mean >= config.zero_max_threshold {
                result.preflight_zero_failed_windows += 1;
            }
            if pre_report.nan_ratio_mean >= config.nan_max_threshold {
                result.preflight_nan_failed_windows += 1;
            }
            let mut reasons: Vec<String> = Vec::new();
            if pre_report.majority_count_mean > config.majority_count_threshold {
                reasons.push(format!(
                    "majority_count_mean={:.2}, 标准<={:.2}",
                    pre_report.majority_count_mean, config.majority_count_threshold,
                ));
            }
            if pre_report.zero_ratio_mean >= config.zero_max_threshold {
                reasons.push(format!(
                    "zero_ratio_mean={:.4}, 标准<{:.4}",
                    pre_report.zero_ratio_mean, config.zero_max_threshold,
                ));
            }
            if pre_report.nan_ratio_mean >= config.nan_max_threshold {
                reasons.push(format!(
                    "nan_ratio_mean={:.4}, 标准<{:.4}",
                    pre_report.nan_ratio_mean, config.nan_max_threshold,
                ));
            }
            println!(
                "[preflight] 剔除因子 {}，该因子指标不达标: {}",
                derived_name,
                reasons.join("; "),
            );
            continue;
        }
        result.any_window_passed_preflight = true;
        selected_slots.push(slot_idx);
    }
    selected_slots
}

pub(crate) fn factor_result_path(task_results_dir: &Path, source_factor: &str) -> PathBuf {
    task_results_dir.join(format!("{}.msgpack", source_factor))
}

pub(crate) fn write_task_result(path: &Path, result: &TailTaskResult) -> Result<(), String> {
    let tmp_path = path.with_extension("msgpack.tmp");
    let bytes =
        rmp_serde::to_vec_named(result).map_err(|e| format!("序列化任务结果失败: {}", e))?;
    fs::write(&tmp_path, bytes).map_err(|e| format!("写入任务结果失败: {}", e))?;
    fs::rename(&tmp_path, path).map_err(|e| format!("原子替换任务结果失败: {}", e))?;
    Ok(())
}

pub(crate) fn read_task_result(path: &Path) -> Result<TailTaskResult, String> {
    let bytes = fs::read(path).map_err(|e| format!("读取任务结果失败: {}", e))?;
    rmp_serde::from_slice::<TailTaskResult>(&bytes).map_err(|e| format!("解析任务结果失败: {}", e))
}

pub(crate) fn append_completed_source(
    completed_log_path: &Path,
    source_factor: &str,
) -> Result<(), String> {
    if let Some(parent) = completed_log_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("创建日志目录失败: {}", e))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(completed_log_path)
        .map_err(|e| format!("打开 completed log 失败: {}", e))?;
    writeln!(file, "{}", source_factor).map_err(|e| format!("写 completed log 失败: {}", e))?;
    Ok(())
}

fn sanitize_task_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => ch,
            _ => '_',
        })
        .collect()
}

fn fulltest_task_key(stage: &str, gap: i32, factor_name: &str) -> String {
    format!(
        "{}_gap{}_{}",
        stage,
        gap,
        sanitize_task_component(factor_name)
    )
}

fn fulltest_done_path(done_dir: &Path, task: &TailV4FulltestTask) -> PathBuf {
    done_dir.join(format!("{}.json", task.task_key))
}

fn write_fulltest_done(path: &Path, task: &TailV4FulltestTask) -> Result<(), String> {
    let tmp_path = path.with_extension("json.tmp");
    let payload = serde_json::json!({
        "task_key": task.task_key,
        "factor_name": task.factor_name,
        "stage": task.stage,
        "gap": task.gap,
    });
    fs::write(
        &tmp_path,
        serde_json::to_vec_pretty(&payload)
            .map_err(|e| format!("序列化 fulltest done 失败: {}", e))?,
    )
    .map_err(|e| format!("写入 fulltest done 失败: {}", e))?;
    fs::rename(&tmp_path, path).map_err(|e| format!("原子替换 fulltest done 失败: {}", e))?;
    Ok(())
}

fn tail_v4_fulltest_idle_timeout() -> Duration {
    std::env::var("TAIL_V4_FULLTEST_IDLE_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(DEFAULT_TAIL_V4_FULLTEST_IDLE_TIMEOUT_SECS))
}

fn kill_tail_v4_fulltest_workers(pid_registry: &TailV4PidRegistry) {
    let pids: Vec<u32> = pid_registry
        .lock()
        .map(|guard| guard.values().copied().collect())
        .unwrap_or_default();

    #[cfg(target_family = "unix")]
    for pid in pids {
        let _ = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);
    }
}

fn create_tail_v4_fulltest_worker_script() -> String {
    r#"#!/usr/bin/env python3
import os
import sys

project_root = os.environ.get("TAIL_V4_PROJECT_ROOT", "/home/chenzongwei")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from design_whatever.tail_v4 import run_tail_v4_fulltest_worker

if __name__ == "__main__":
    run_tail_v4_fulltest_worker()
"#
    .to_string()
}

fn build_fulltest_tasks(
    gap5_selected: &[String],
    gap1_selected: &[String],
) -> Vec<TailV4FulltestTask> {
    let mut ordered_factors = Vec::<String>::new();
    let mut seen = HashSet::<String>::new();
    for factor_name in gap5_selected.iter().chain(gap1_selected.iter()) {
        if seen.insert(factor_name.clone()) {
            ordered_factors.push(factor_name.clone());
        }
    }

    let gap5_set = gap5_selected.iter().cloned().collect::<HashSet<_>>();
    let gap1_set = gap1_selected.iter().cloned().collect::<HashSet<_>>();
    let mut tasks = Vec::new();
    for factor_name in ordered_factors {
        if gap5_set.contains(&factor_name) {
            tasks.push(TailV4FulltestTask {
                task_key: fulltest_task_key("raw", 5, &factor_name),
                factor_name: factor_name.clone(),
                stage: "raw".to_string(),
                gap: 5,
            });
            tasks.push(TailV4FulltestTask {
                task_key: fulltest_task_key("neu", 5, &factor_name),
                factor_name: factor_name.clone(),
                stage: "neu".to_string(),
                gap: 5,
            });
        }
        if gap1_set.contains(&factor_name) {
            tasks.push(TailV4FulltestTask {
                task_key: fulltest_task_key("raw", 1, &factor_name),
                factor_name: factor_name.clone(),
                stage: "raw".to_string(),
                gap: 1,
            });
            tasks.push(TailV4FulltestTask {
                task_key: fulltest_task_key("neu", 1, &factor_name),
                factor_name: factor_name.clone(),
                stage: "neu".to_string(),
                gap: 1,
            });
        }
    }
    tasks
}

fn increment_fulltest_bucket(bucket_counts: &mut HashMap<String, usize>, stage: &str, gap: i32) {
    let key = match (stage, gap) {
        ("raw", 5) => "g5-raw",
        ("neu", 5) => "g5-neu",
        ("raw", 1) => "g1-raw",
        ("neu", 1) => "g1-neu",
        _ => return,
    };
    *bucket_counts.entry(key.to_string()).or_insert(0) += 1;
}

fn render_fulltest_progress(
    processed: usize,
    restored: usize,
    total: usize,
    started: Instant,
    bucket_counts: &HashMap<String, usize>,
) -> Result<(), String> {
    let finished = processed + restored;
    let progress = if total > 0 {
        finished as f64 / total as f64
    } else {
        1.0
    };
    let elapsed = started.elapsed();
    let estimated_total_secs = if progress > 0.0 {
        elapsed.as_secs_f64() / progress
    } else {
        elapsed.as_secs_f64()
    };
    let remaining_secs = if estimated_total_secs > elapsed.as_secs_f64() {
        (estimated_total_secs - elapsed.as_secs_f64()) as u64
    } else {
        0
    };
    let (elapsed_h, elapsed_m, elapsed_s) = format_hms(elapsed.as_secs());
    let (remaining_h, remaining_m, remaining_s) = format_hms(remaining_secs);
    let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let g5_raw = bucket_counts.get("g5-raw").copied().unwrap_or(0);
    let g5_neu = bucket_counts.get("g5-neu").copied().unwrap_or(0);
    let g1_raw = bucket_counts.get("g1-raw").copied().unwrap_or(0);
    let g1_neu = bucket_counts.get("g1-neu").copied().unwrap_or(0);
    let lead = if is_terminal() { "\r" } else { "" };
    let trail = if is_terminal() { "" } else { "\n" };
    print!(
        "{lead}[{}] Fulltest 进度 {}/{} ({:.1}%)，已恢复 {} 个，g5-raw {}，g5-neu {}，g1-raw {}，g1-neu {}，已用{}h{}m{}s，预计剩余{}h{}m{}s{trail}",
        current_time,
        finished,
        total,
        progress * 100.0,
        restored,
        g5_raw,
        g5_neu,
        g1_raw,
        g1_neu,
        elapsed_h,
        elapsed_m,
        elapsed_s,
        remaining_h,
        remaining_m,
        remaining_s,
    );
    std::io::stdout()
        .flush()
        .map_err(|e| format!("刷新 fulltest 进度失败: {}", e))?;
    Ok(())
}

fn run_tail_v4_fulltest_worker_process(
    worker_id: usize,
    task_queue: Arc<Mutex<VecDeque<TailV4FulltestTask>>>,
    result_sender: Sender<Result<TailV4FulltestWorkerResult, String>>,
    stop_flag: Arc<AtomicBool>,
    pid_registry: TailV4PidRegistry,
    python_path: String,
    worker_config_json: String,
    verbose: bool,
) {
    let script_content = create_tail_v4_fulltest_worker_script();
    let script_path = format!("/tmp/tail_v4_fulltest_worker_{}.py", worker_id);
    if let Err(err) = fs::write(&script_path, script_content) {
        let _ = result_sender.send(Err(format!("创建 fulltest worker 脚本失败: {}", err)));
        return;
    }

    let mut command = Command::new(&python_path);
    command
        .arg(&script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("NUMEXPR_NUM_THREADS", "1")
        .env("VECLIB_MAXIMUM_THREADS", "1")
        .env("BLIS_NUM_THREADS", "1")
        .env("POLARS_MAX_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .env("TAIL_V4_PROJECT_ROOT", "/home/chenzongwei")
        .env("PYTHONPATH", "/home/chenzongwei")
        .env("TAIL_V4_FULLTEST_WORKER_CONFIG", worker_config_json);

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(err) => {
            let _ = result_sender.send(Err(format!("启动 fulltest worker 失败: {}", err)));
            let _ = fs::remove_file(&script_path);
            return;
        }
    };
    let _pid_guard = TailV4WorkerPidGuard::new(Arc::clone(&pid_registry), worker_id, child.id());

    let mut stdin = match child.stdin.take() {
        Some(stdin) => stdin,
        None => {
            let _ = result_sender.send(Err("获取 fulltest worker stdin 失败".to_string()));
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_file(&script_path);
            return;
        }
    };
    let mut stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = result_sender.send(Err("获取 fulltest worker stdout 失败".to_string()));
            drop(stdin);
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_file(&script_path);
            return;
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            let _ = result_sender.send(Err("获取 fulltest worker stderr 失败".to_string()));
            drop(stdin);
            drop(stdout);
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_file(&script_path);
            return;
        }
    };

    let stderr_worker_id = worker_id;
    let stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            match line {
                Ok(text) => {
                    if verbose {
                        eprintln!("[Tail V4 Fulltest][worker {}] {}", stderr_worker_id, text);
                    }
                }
                Err(_) => break,
            }
        }
    });

    loop {
        if stop_flag.load(AtomicOrdering::Relaxed) {
            break;
        }
        let next_task = {
            let mut guard = match task_queue.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    let _ = result_sender.send(Err("获取 fulltest 任务队列锁失败".to_string()));
                    stop_flag.store(true, AtomicOrdering::Relaxed);
                    break;
                }
            };
            guard.pop_front()
        };
        let Some(task) = next_task else {
            break;
        };

        let packed_data = match rmp_serde::to_vec_named(&task) {
            Ok(data) => data,
            Err(err) => {
                let _ = result_sender.send(Err(format!("序列化 fulltest 任务失败: {}", err)));
                stop_flag.store(true, AtomicOrdering::Relaxed);
                break;
            }
        };
        let length_bytes = (packed_data.len() as u32).to_le_bytes();
        if stdin.write_all(&length_bytes).is_err()
            || stdin.write_all(&packed_data).is_err()
            || stdin.flush().is_err()
        {
            let _ = result_sender.send(Err(format!("发送 fulltest 任务失败: {}", task.task_key)));
            stop_flag.store(true, AtomicOrdering::Relaxed);
            break;
        }

        let mut len_buf = [0u8; 4];
        if stdout.read_exact(&mut len_buf).is_err() {
            let _ = result_sender.send(Err(format!(
                "读取 fulltest worker 结果长度失败: {}",
                task.task_key
            )));
            stop_flag.store(true, AtomicOrdering::Relaxed);
            break;
        }
        let result_len = u32::from_le_bytes(len_buf) as usize;
        let mut result_data = vec![0u8; result_len];
        if stdout.read_exact(&mut result_data).is_err() {
            let _ = result_sender.send(Err(format!(
                "读取 fulltest worker 结果失败: {}",
                task.task_key
            )));
            stop_flag.store(true, AtomicOrdering::Relaxed);
            break;
        }
        let result = match rmp_serde::from_slice::<TailV4FulltestWorkerResult>(&result_data) {
            Ok(result) => result,
            Err(err) => {
                let _ = result_sender.send(Err(format!("解析 fulltest worker 结果失败: {}", err)));
                stop_flag.store(true, AtomicOrdering::Relaxed);
                break;
            }
        };
        if !result.ok {
            let _ = result_sender.send(Err(format!(
                "fulltest 任务 {} 失败: {}",
                result.task_key,
                result.error.unwrap_or_else(|| "未知错误".to_string())
            )));
            stop_flag.store(true, AtomicOrdering::Relaxed);
            break;
        }
        if result_sender.send(Ok(result)).is_err() {
            stop_flag.store(true, AtomicOrdering::Relaxed);
            break;
        }
    }

    let _ = stdin.write_all(&[0u8; 4]);
    let _ = stdin.flush();

    // 等待子进程退出，超时后强制kill防止僵尸进程
    match child.try_wait() {
        Ok(Some(_)) => {}
        _ => {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    break;
                }
                if child.try_wait().ok().flatten().is_some() {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
    let _ = child.wait();

    // join stderr 线程确保资源释放
    let _ = stderr_handle.join();

    let _ = fs::remove_file(&script_path);
}

fn timestamp_ns_to_date_key(value: i64) -> Result<i32, String> {
    let secs = value.div_euclid(1_000_000_000);
    let nanos = value.rem_euclid(1_000_000_000) as u32;
    let dt = NaiveDateTime::from_timestamp_opt(secs, nanos)
        .ok_or_else(|| format!("无效 timestamp 纳秒值: {}", value))?;
    Ok((dt.year() * 10000 + dt.month() as i32 * 100 + dt.day() as i32) as i32)
}

fn extract_date_keys(
    batch: &arrow::record_batch::RecordBatch,
    date_col_idx: usize,
) -> Result<Vec<i32>, String> {
    let date_column = batch.column(date_col_idx);
    if let Some(array) = date_column.as_any().downcast_ref::<Int32Array>() {
        return Ok((0..array.len()).map(|idx| array.value(idx)).collect());
    }
    if let Some(array) = date_column.as_any().downcast_ref::<Int64Array>() {
        return Ok((0..array.len())
            .map(|idx| array.value(idx) as i32)
            .collect());
    }
    if let Some(array) = date_column
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
    {
        return (0..array.len())
            .map(|idx| timestamp_ns_to_date_key(array.value(idx)))
            .collect();
    }
    if let Some(array) = date_column
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
    {
        return (0..array.len())
            .map(|idx| timestamp_ns_to_date_key(array.value(idx) * 1_000))
            .collect();
    }
    if let Some(array) = date_column
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
    {
        return (0..array.len())
            .map(|idx| timestamp_ns_to_date_key(array.value(idx) * 1_000_000))
            .collect();
    }
    Err("不支持的 date 列类型".to_string())
}

fn load_factor_to_template(
    factor_path: &str,
    template_dates: &[i32],
    template_stocks: &[String],
) -> Result<Array2<f32>, String> {
    // 新列式存储分支：路径格式 "<store_dir>::<col_idx>"（如 "/hdd/x/factors.colblk::3"）
    // store_dir 是含 factors.colblk + factors.idx 的目录。
    if factor_path.contains("::") {
        let parts: Vec<&str> = factor_path.splitn(2, "::").collect();
        let store_path = parts[0];
        let col_idx: usize = parts[1]
            .parse()
            .map_err(|e| format!("解析 col_idx 失败 '{}': {}", parts[1], e))?;
        // store_path 可能是目录或 factors.colblk 文件；取其父目录作为 store_dir
        let store_dir = if store_path.ends_with(".colblk") {
            Path::new(store_path)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| store_path.to_string())
        } else {
            store_path.to_string()
        };
        let reader = crate::factor_store_v5::FactorStoreReader::open(&store_dir)?;
        return reader.read_factor_to_matrix(col_idx, template_dates, template_stocks);
    }
    if factor_path.ends_with(".h5") {
        load_h5_factor_to_template(factor_path, template_dates, template_stocks)
    } else {
        load_parquet_factor_to_template(factor_path, template_dates, template_stocks)
    }
}

fn load_parquet_factor_to_template(
    factor_path: &str,
    template_dates: &[i32],
    template_stocks: &[String],
) -> Result<Array2<f32>, String> {
    let file =
        File::open(factor_path).map_err(|e| format!("打开因子文件失败 {}: {}", factor_path, e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("创建 parquet 读取器失败 {}: {}", factor_path, e))?;
    let schema = builder.schema();
    let mut date_col_idx = None;
    let stock_pos_map: HashMap<&str, usize> = template_stocks
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect();
    let mut stock_cols = Vec::<(usize, usize)>::new();
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let name = field.name();
        if name == "date" {
            date_col_idx = Some(col_idx);
        } else if let Some(&stock_pos) = stock_pos_map.get(name.as_str()) {
            stock_cols.push((col_idx, stock_pos));
        }
    }
    let date_col_idx =
        date_col_idx.ok_or_else(|| format!("因子文件缺少 date 列: {}", factor_path))?;
    let date_pos_map: HashMap<i32, usize> = template_dates
        .iter()
        .enumerate()
        .map(|(idx, value)| (*value, idx))
        .collect();
    let reader = builder
        .with_batch_size(8192)
        .build()
        .map_err(|e| format!("构建 parquet 批读取器失败 {}: {}", factor_path, e))?;

    let mut output =
        Array2::<f32>::from_elem((template_dates.len(), template_stocks.len()), f32::NAN);
    for batch_result in reader {
        let batch =
            batch_result.map_err(|e| format!("读取 parquet batch 失败 {}: {}", factor_path, e))?;
        let date_keys = extract_date_keys(&batch, date_col_idx)?;
        for (row_idx, date_key) in date_keys.iter().enumerate() {
            let Some(&date_pos) = date_pos_map.get(date_key) else {
                continue;
            };
            for &(col_idx, stock_pos) in &stock_cols {
                let array = batch.column(col_idx);
                let value = if let Some(col) = array.as_any().downcast_ref::<Float64Array>() {
                    if col.is_null(row_idx) {
                        f32::NAN
                    } else {
                        col.value(row_idx) as f32
                    }
                } else if let Some(col) = array.as_any().downcast_ref::<Float32Array>() {
                    if col.is_null(row_idx) {
                        f32::NAN
                    } else {
                        col.value(row_idx)
                    }
                } else {
                    return Err(format!(
                        "股票列类型不是 float: {} / col_idx={}",
                        factor_path, col_idx
                    ));
                };
                output[[date_pos, stock_pos]] = if value.is_finite() { value } else { f32::NAN };
            }
        }
    }
    Ok(output)
}

/// 解析 calendar_map.csv：header 为 "date_min"，每行一个日期字符串，行号即 row index
fn parse_calendar_map(path: &Path) -> Result<HashMap<i32, usize>, String> {
    let file = File::open(path).map_err(|e| format!("打开 calendar_map.csv 失败: {}", e))?;
    let reader = BufReader::new(file);
    let mut map = HashMap::new();
    for (idx, line_result) in reader.lines().enumerate() {
        if idx == 0 {
            continue;
        }
        let line =
            line_result.map_err(|e| format!("读取 calendar_map 第 {} 行失败: {}", idx, e))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let date_int: i32 = trimmed
            .parse()
            .map_err(|e| format!("无效日期 '{}': {}", trimmed, e))?;
        map.insert(date_int, idx - 1);
    }
    Ok(map)
}

/// 解析 symbol_map.csv：支持两种格式
/// - 单列：header 为 "symbol"，每行一个股票代码，行号即列位置（与 calendar_map.csv 一致）
/// - 双列：header 为 "symbol,pos"，每行格式 "000001,0"
fn parse_symbol_map(path: &Path) -> Result<HashMap<String, usize>, String> {
    let file = File::open(path).map_err(|e| format!("打开 symbol_map.csv 失败: {}", e))?;
    let reader = BufReader::new(file);
    let mut map = HashMap::new();
    for (idx, line_result) in reader.lines().enumerate() {
        if idx == 0 {
            continue;
        }
        let line = line_result.map_err(|e| format!("读取 symbol_map 第 {} 行失败: {}", idx, e))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        let (symbol, pos) = if parts.len() == 1 {
            (parts[0].to_string(), idx - 1)
        } else if parts.len() == 2 {
            let pos: usize = parts[1]
                .parse()
                .map_err(|e| format!("无效位置 '{}': {}", parts[1], e))?;
            (parts[0].to_string(), pos)
        } else {
            return Err(format!("symbol_map 格式错误: {}", trimmed));
        };
        map.insert(symbol, pos);
    }
    Ok(map)
}

/// 去掉股票代码后缀（如 "000060.SZ" → "000060"）
fn strip_stock_suffix(code: &str) -> &str {
    code.split('.').next().unwrap_or(code)
}

/// 从 H5 文件读取因子数据到模板矩阵
#[cfg(feature = "hdf5")]
fn load_h5_factor_to_template(
    factor_path: &str,
    template_dates: &[i32],
    template_stocks: &[String],
) -> Result<Array2<f32>, String> {
    let h5_dir = Path::new(factor_path)
        .parent()
        .ok_or_else(|| format!("无法获取 H5 文件所在目录: {}", factor_path))?;

    let calendar_map_path = h5_dir.join("calendar_map.csv");
    let symbol_map_path = h5_dir.join("symbol_map.csv");

    let date_to_row = parse_calendar_map(&calendar_map_path)?;
    let symbol_to_col = parse_symbol_map(&symbol_map_path)?;

    // 构建 stock 查找映射：template_stocks 可能带后缀（如 "000060.SZ"），需去掉后缀
    let stock_col_indices: Vec<Option<usize>> = template_stocks
        .iter()
        .map(|s| {
            let bare = strip_stock_suffix(s);
            symbol_to_col.get(bare).copied()
        })
        .collect();

    let file = hdf5::File::open(factor_path)
        .map_err(|e| format!("打开 H5 文件失败 {}: {}", factor_path, e))?;
    let dataset = file
        .dataset("data")
        .map_err(|e| format!("H5 文件缺少 data dataset {}: {}", factor_path, e))?;

    let full_data: Array2<f64> = dataset
        .read_2d()
        .map_err(|e| format!("读取 H5 数据失败 {}: {}", factor_path, e))?;

    let n_dates = template_dates.len();
    let n_stocks = template_stocks.len();
    let mut output = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);

    for (date_idx, &date_val) in template_dates.iter().enumerate() {
        let Some(&row_idx) = date_to_row.get(&date_val) else {
            continue;
        };
        for (stock_idx, col_opt) in stock_col_indices.iter().enumerate() {
            let Some(col_idx) = col_opt else { continue };
            let val = full_data[[row_idx, *col_idx]];
            output[[date_idx, stock_idx]] = if val.is_finite() {
                val as f32
            } else {
                f32::NAN
            };
        }
    }
    Ok(output)
}

#[cfg(not(feature = "hdf5"))]
fn load_h5_factor_to_template(
    _factor_path: &str,
    _template_dates: &[i32],
    _template_stocks: &[String],
) -> Result<Array2<f32>, String> {
    Err("HDF5 支持未启用（此 wheel 编译时未包含 hdf5-metno）".to_string())
}

pub(crate) fn build_fold_values(raw_values: &Array2<f32>) -> Array2<f32> {
    let n_dates = raw_values.nrows();
    let n_stocks = raw_values.ncols();
    let mut folded = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
    for date_idx in 0..n_dates {
        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for stock_idx in 0..n_stocks {
            let value = raw_values[[date_idx, stock_idx]];
            if value.is_finite() {
                sum += value as f64;
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        let mean = (sum / count as f64) as f32;
        for stock_idx in 0..n_stocks {
            let value = raw_values[[date_idx, stock_idx]];
            if value.is_finite() {
                folded[[date_idx, stock_idx]] = (value - mean).abs();
            }
        }
    }
    folded
}

pub(crate) fn derived_names_for_variant(source_factor: &str, windows: &[usize]) -> Vec<String> {
    let mut names = vec![format!("{}_smooth_1", source_factor)];
    for &window in windows {
        names.push(format!("{}_mean_smooth_{}", source_factor, window));
        names.push(format!("{}_max_smooth_{}", source_factor, window));
        names.push(format!("{}_min_smooth_{}", source_factor, window));
        names.push(format!("{}_std_smooth_{}", source_factor, window));
    }
    names
}

fn legacy_rank_values(values: &[f64]) -> Vec<f64> {
    let mut indexed_values = Vec::with_capacity(values.len());
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_nan() {
            indexed_values.push((idx, value));
        }
    }
    indexed_values
        .sort_unstable_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal));
    let mut ranks = vec![f64::NAN; values.len()];
    for (rank, &(original_idx, _)) in indexed_values.iter().enumerate() {
        ranks[original_idx] = (rank + 1) as f64;
    }
    ranks
}

fn neutralize_block_legacy_exact(
    legacy_style_data: &IOOptimizedStyleData,
    factor: ArrayView3<'_, f32>,
    dates: &[i32],
    stocks: &[String],
    rank_before: bool,
    min_valid: usize,
    industry_neutralize: bool,
) -> Result<Array3<f32>, String> {
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    let n_factors = factor.shape()[2];
    if dates.len() != n_dates || stocks.len() != n_stocks {
        return Err("legacy neutralize exact 输入形状不匹配".to_string());
    }

    let mut all_style_stocks = std::collections::HashSet::new();
    for day_data in legacy_style_data.data_by_date.values() {
        for stock in day_data.stocks.iter() {
            all_style_stocks.insert(stock.clone());
        }
    }

    let mut ordered_factor_stocks: Vec<(usize, String)> = stocks
        .iter()
        .enumerate()
        .filter_map(|(stock_idx, stock)| {
            let stock_code = stock.get(..6).unwrap_or(stock.as_str()).to_string();
            if all_style_stocks.contains(&stock_code) {
                Some((stock_idx, stock_code))
            } else {
                None
            }
        })
        .collect();
    ordered_factor_stocks.sort_unstable_by(|lhs, rhs| lhs.1.cmp(&rhs.1));

    let mut output = Array3::<f32>::from_elem((n_dates, n_stocks, n_factors), f32::NAN);
    for date_idx in 0..n_dates {
        let date_key = dates[date_idx] as i64;
        let Some(day_data) = legacy_style_data.data_by_date.get(&date_key) else {
            continue;
        };
        let regression_matrix = if industry_neutralize {
            day_data.regression_matrix.as_ref()
        } else {
            day_data.regression_matrix_style_only.as_ref()
        };
        let Some(regression_matrix) = regression_matrix else {
            continue;
        };
        let mut template_style_positions = vec![None; n_stocks];
        for (stock_idx, stock) in stocks.iter().enumerate() {
            let stock_code = stock.get(..6).unwrap_or(stock.as_str());
            if let Some(&style_idx) = day_data.stock_index_map.get(stock_code) {
                template_style_positions[stock_idx] = Some(style_idx);
            }
        }

        let mut daily_factor_values = Vec::<f64>::with_capacity(n_stocks);
        let mut valid_stock_indices = Vec::<usize>::with_capacity(n_stocks);
        let mut valid_style_indices = Vec::<usize>::with_capacity(n_stocks);
        let n_features = if industry_neutralize {
            day_data.style_matrix.ncols()
        } else {
            10
        };
        let mut beta_values = vec![0.0_f64; n_features];

        for factor_idx in 0..n_factors {
            daily_factor_values.clear();
            valid_stock_indices.clear();
            valid_style_indices.clear();
            beta_values.fill(0.0);

            for &(stock_idx, _) in ordered_factor_stocks.iter() {
                let Some(style_idx) = template_style_positions[stock_idx] else {
                    continue;
                };
                let value = factor[[date_idx, stock_idx, factor_idx]];
                if value.is_finite() {
                    daily_factor_values.push(value as f64);
                    valid_stock_indices.push(stock_idx);
                    valid_style_indices.push(style_idx);
                }
            }
            if daily_factor_values.len() < min_valid {
                continue;
            }
            let ranked_values = if rank_before {
                legacy_rank_values(&daily_factor_values)
            } else {
                daily_factor_values.clone()
            };

            for (col_idx, &style_idx) in valid_style_indices.iter().enumerate() {
                let y_value = ranked_values[col_idx];
                for feature_idx in 0..n_features {
                    beta_values[feature_idx] +=
                        regression_matrix[(feature_idx, style_idx)] * y_value;
                }
            }

            for (row_idx, &stock_idx) in valid_stock_indices.iter().enumerate() {
                let style_idx = valid_style_indices[row_idx];
                let mut predicted = 0.0_f64;
                for feature_idx in 0..n_features {
                    predicted +=
                        day_data.style_matrix[(style_idx, feature_idx)] * beta_values[feature_idx];
                }
                output[[date_idx, stock_idx, factor_idx]] =
                    (ranked_values[row_idx] - predicted) as f32;
            }
        }
    }
    Ok(output)
}

/// V7 neutralize：日期维度 rayon 并行（35945 次回归从串行到多核并行）。
/// 计算逻辑与原版完全一致，只是把 for date_idx 改为 par_chunks_mut（每天独立写不重叠的内存区域）。
/// 结果逐元素相同（确定性计算，并行不改变内容）。
fn neutralize_block_legacy_exact_v7(
    legacy_style_data: &IOOptimizedStyleData,
    factor: ArrayView3<'_, f32>,
    dates: &[i32],
    stocks: &[String],
    rank_before: bool,
    min_valid: usize,
    industry_neutralize: bool,
) -> Result<Array3<f32>, String> {
    let n_dates = factor.shape()[0];
    let n_stocks = factor.shape()[1];
    let n_factors = factor.shape()[2];
    if dates.len() != n_dates || stocks.len() != n_stocks {
        return Err("legacy neutralize exact v7 输入形状不匹配".to_string());
    }

    // 预计算：风格股票集合 + 排序（只算一次）
    let mut all_style_stocks = std::collections::HashSet::new();
    for day_data in legacy_style_data.data_by_date.values() {
        for stock in day_data.stocks.iter() {
            all_style_stocks.insert(stock.clone());
        }
    }
    let mut ordered_factor_stocks: Vec<(usize, String)> = stocks
        .iter()
        .enumerate()
        .filter_map(|(stock_idx, stock)| {
            let stock_code = stock.get(..6).unwrap_or(stock.as_str()).to_string();
            if all_style_stocks.contains(&stock_code) {
                Some((stock_idx, stock_code))
            } else {
                None
            }
        })
        .collect();
    ordered_factor_stocks.sort_unstable_by(|lhs, rhs| lhs.1.cmp(&rhs.1));

    // V7：flat output，按日期 chunk 并行（每天 n_stocks*n_factors 个元素，非重叠）
    let day_size = n_stocks * n_factors;
    let mut output_flat = vec![f32::NAN; n_dates * day_size];

    output_flat
        .par_chunks_mut(day_size)
        .enumerate()
        .for_each(|(date_idx, day_slice)| {
            let date_key = dates[date_idx] as i64;
            let Some(day_data) = legacy_style_data.data_by_date.get(&date_key) else {
                return;
            };
            let regression_matrix = if industry_neutralize {
                day_data.regression_matrix.as_ref()
            } else {
                day_data.regression_matrix_style_only.as_ref()
            };
            let Some(regression_matrix) = regression_matrix else {
                return;
            };
            // 当天的 stock → style 映射
            let mut template_style_positions = vec![None; n_stocks];
            for (stock_idx, stock) in stocks.iter().enumerate() {
                let stock_code = stock.get(..6).unwrap_or(stock.as_str());
                if let Some(&style_idx) = day_data.stock_index_map.get(stock_code) {
                    template_style_positions[stock_idx] = Some(style_idx);
                }
            }

            let mut daily_factor_values = Vec::<f64>::with_capacity(n_stocks);
            let mut valid_stock_indices = Vec::<usize>::with_capacity(n_stocks);
            let mut valid_style_indices = Vec::<usize>::with_capacity(n_stocks);
            let n_features = if industry_neutralize {
                day_data.style_matrix.ncols()
            } else {
                10
            };
            let mut beta_values = vec![0.0_f64; n_features];

            for factor_idx in 0..n_factors {
                daily_factor_values.clear();
                valid_stock_indices.clear();
                valid_style_indices.clear();
                beta_values.fill(0.0);

                for &(stock_idx, _) in ordered_factor_stocks.iter() {
                    let Some(style_idx) = template_style_positions[stock_idx] else {
                        continue;
                    };
                    let value = factor[[date_idx, stock_idx, factor_idx]];
                    if value.is_finite() {
                        daily_factor_values.push(value as f64);
                        valid_stock_indices.push(stock_idx);
                        valid_style_indices.push(style_idx);
                    }
                }
                if daily_factor_values.len() < min_valid {
                    continue;
                }
                let ranked_values = if rank_before {
                    legacy_rank_values(&daily_factor_values)
                } else {
                    daily_factor_values.clone()
                };

                for (col_idx, &style_idx) in valid_style_indices.iter().enumerate() {
                    let y_value = ranked_values[col_idx];
                    for feature_idx in 0..n_features {
                        beta_values[feature_idx] +=
                            regression_matrix[(feature_idx, style_idx)] * y_value;
                    }
                }

                for (row_idx, &stock_idx) in valid_stock_indices.iter().enumerate() {
                    let style_idx = valid_style_indices[row_idx];
                    let mut predicted = 0.0_f64;
                    for feature_idx in 0..n_features {
                        predicted += day_data.style_matrix[(style_idx, feature_idx)]
                            * beta_values[feature_idx];
                    }
                    // V7: 写 flat slice（day_slice[stock_idx * n_factors + factor_idx]）
                    day_slice[stock_idx * n_factors + factor_idx] =
                        (ranked_values[row_idx] - predicted) as f32;
                }
            }
        });

    Ok(
        Array3::from_shape_vec((n_dates, n_stocks, n_factors), output_flat)
            .map_err(|e| format!("neutralize v7 输出形状错误: {e}"))?,
    )
}

#[pyclass]
pub struct TailV5LegacyStyleData {
    style_data: Arc<IOOptimizedStyleData>,
}

#[pymethods]
impl TailV5LegacyStyleData {
    #[new]
    fn new(style_vars_dir: String) -> PyResult<Self> {
        let style_data = IOOptimizedStyleData::load_from_vars_h5(&style_vars_dir)?;
        Ok(Self {
            style_data: Arc::new(style_data),
        })
    }

    #[pyo3(signature = (dates, stocks, factor_block, rank_before=true, min_valid=12, industry_neutralize=true))]
    fn neutralize_block_exact<'py>(
        &self,
        py: Python<'py>,
        dates: Vec<i32>,
        stocks: Vec<String>,
        factor_block: numpy::PyReadonlyArray3<'py, f32>,
        rank_before: bool,
        min_valid: usize,
        industry_neutralize: bool,
    ) -> PyResult<Py<numpy::PyArray3<f32>>> {
        let factor = factor_block.as_array();
        let output = py
            .allow_threads(|| {
                neutralize_block_legacy_exact(
                    self.style_data.as_ref(),
                    factor,
                    &dates,
                    &stocks,
                    rank_before,
                    min_valid,
                    industry_neutralize,
                )
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(output.into_pyarray(py).to_owned())
    }
}

#[pyfunction]
#[pyo3(signature = (style_vars_dir, dates, stocks, factor_block, rank_before=true, min_valid=12, industry_neutralize=true))]
pub fn tail_v5_neutralize_block_exact<'py>(
    py: Python<'py>,
    style_vars_dir: String,
    dates: Vec<i32>,
    stocks: Vec<String>,
    factor_block: numpy::PyReadonlyArray3<'py, f32>,
    rank_before: bool,
    min_valid: usize,
    industry_neutralize: bool,
) -> PyResult<Py<numpy::PyArray3<f32>>> {
    let factor = factor_block.as_array();
    let style_data = IOOptimizedStyleData::load_from_vars_h5(&style_vars_dir)?;
    let output = py
        .allow_threads(|| {
            neutralize_block_legacy_exact(
                &style_data,
                factor,
                &dates,
                &stocks,
                rank_before,
                min_valid,
                industry_neutralize,
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(output.into_pyarray(py).to_owned())
}

pub(crate) fn summary_from_row(
    factor_name: &str,
    stage: &str,
    gap: i32,
    source_factor: &str,
    values: &[f64],
) -> SummaryRowRecord {
    SummaryRowRecord {
        factor_name: factor_name.to_string(),
        stage: stage.to_string(),
        gap,
        source_factor: source_factor.to_string(),
        preflight_passed: true,
        ic_mean: values[0],
        ir: values[1],
        annualized_return: values[2],
        sharpe_ratio: values[3],
        max_drawdown: values[4],
        date_size: values[5] as i32,
        ratio_mean: values[6],
        hedge_annualized_return: values[7],
        hedge_annualized_sharpe_ratio: values[8],
        hedge_max_drawdown: values[9],
        ssm: values.get(10).copied().unwrap_or(f64::NAN),
        mprob: values.get(11).copied().unwrap_or(f64::NAN),
        brc: values.get(12).copied().unwrap_or(f64::NAN),
        brc_s: values.get(13).copied().unwrap_or(f64::NAN),
        brc_l: values.get(14).copied().unwrap_or(f64::NAN),
    }
}

pub(crate) fn qualify_raw(summary: &SummaryRowRecord, gap: usize, cfg: &TailSelectionConfig) -> bool {
    match gap {
        1 => {
            summary.hedge_annualized_return >= cfg.ret_point_gap1
                || summary.ic_mean.abs() >= cfg.ic_point_gap1
        }
        5 => {
            summary.hedge_annualized_return >= cfg.ret_point_gap5
                || summary.ic_mean.abs() >= cfg.ic_point_gap5
        }
        _ => false,
    }
}

pub(crate) fn qualify_neu(summary: &SummaryRowRecord, gap: usize, cfg: &TailSelectionConfig) -> bool {
    let (ret_point, ic_point, ic_more) = match gap {
        1 => (
            cfg.ret_point_neu_gap1,
            cfg.ic_point_neu_gap1,
            cfg.ic_more_important_gap1,
        ),
        5 => (
            cfg.ret_point_neu_gap5,
            cfg.ic_point_neu_gap5,
            cfg.ic_more_important_gap5,
        ),
        _ => return false,
    };
    let mut qualifies =
        summary.hedge_annualized_return >= ret_point || summary.ic_mean.abs() >= ic_point;
    if let Some(ic_more_value) = ic_more {
        qualifies = qualifies
            || (summary.hedge_annualized_return >= ret_point
                && summary.ic_mean.abs() >= ic_more_value);
    }
    qualifies
}

fn process_task(task: &TailTask, shared: &SharedInputs) -> Result<TailTaskResult, String> {
    let raw_values = load_factor_to_template(
        &task.factor_path,
        shared.dates.as_slice(),
        shared.stocks.as_slice(),
    )?;
    process_task_with_values(task, raw_values, shared)
}

// ===== 插桩：各步骤耗时累计（AtomicU64 线程安全，热路径开销 ~ns）=====
use std::sync::atomic::AtomicU64;
static PROF_RAW_COVER: AtomicU64 = AtomicU64::new(0);
static PROF_FOLD: AtomicU64 = AtomicU64::new(0);
static PROF_RANK_ROLL: AtomicU64 = AtomicU64::new(0);
static PROF_BT_RAW: AtomicU64 = AtomicU64::new(0);
static PROF_NEU: AtomicU64 = AtomicU64::new(0);
static PROF_BT_NEU: AtomicU64 = AtomicU64::new(0);
static PROF_COUNT: AtomicU64 = AtomicU64::new(0);

fn prof_dump(tag: &str) {
    let rc = PROF_RAW_COVER.load(AtomicOrdering::Relaxed);
    let fd = PROF_FOLD.load(AtomicOrdering::Relaxed);
    let rr = PROF_RANK_ROLL.load(AtomicOrdering::Relaxed);
    let br = PROF_BT_RAW.load(AtomicOrdering::Relaxed);
    let ne = PROF_NEU.load(AtomicOrdering::Relaxed);
    let bn = PROF_BT_NEU.load(AtomicOrdering::Relaxed);
    let n = PROF_COUNT.load(AtomicOrdering::Relaxed);
    let total = rc + fd + rr + br + ne + bn;
    let pct = |x: u64| {
        if total > 0 {
            x as f64 / total as f64 * 100.0
        } else {
            0.0
        }
    };
    let avg = if n > 0 {
        total as f64 / n as f64 / 1e9
    } else {
        0.0
    };
    eprintln!(
        "[PROF {} n={} avg={:.2}s] raw_cover={:.1}% fold={:.1}% rank_roll={:.1}% bt_raw={:.1}% neutralize={:.1}% bt_neu={:.1}% (其余=preflight+汇总)",
        tag, n, avg, pct(rc), pct(fd), pct(rr), pct(br), pct(ne), pct(bn)
    );
}

/// 用已读好的因子矩阵执行回测（IO/CPU 分离架构：IO 线程预读后调用此函数）。
fn process_task_with_values(
    task: &TailTask,
    raw_values: Array2<f32>,
    shared: &SharedInputs,
) -> Result<TailTaskResult, String> {
    let _t = std::time::Instant::now();
    let raw_cover_rate = compute_raw_cover_rate(
        &raw_values.view(),
        &shared.restrict.view(),
        &shared.ret_gap1.view(),
        10,
    );
    PROF_RAW_COVER.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    if raw_cover_rate < shared.config.cover_rate {
        println!(
            "[raw_cover] 剔除因子 {}，原始覆盖率不达标: raw_cover_rate={:.4}, 标准>={:.4}",
            task.source_factor, raw_cover_rate, shared.config.cover_rate,
        );
        let _n = PROF_COUNT.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if _n % 20 == 0 {
            prof_dump("raw_cover剔除");
        }
        return Ok(TailTaskResult {
            source_factor: task.source_factor.clone(),
            eliminated_by_raw_cover: true,
            ..TailTaskResult::default()
        });
    }
    let _t = std::time::Instant::now();
    let mut variants = vec![(task.source_factor.clone(), raw_values)];
    if shared.fold {
        let folded = build_fold_values(&variants[0].1);
        variants.push((format!("{}_fold", task.source_factor), folded));
    }
    PROF_FOLD.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);

    let mut result = TailTaskResult {
        source_factor: task.source_factor.clone(),
        ..TailTaskResult::default()
    };

    for (variant_name, variant_values) in variants {
        let _t = std::time::Instant::now();
        let rolled_block =
            rank_roll_block_f32_with_parallel(&variant_values, shared.windows.as_slice(), false)?;
        PROF_RANK_ROLL.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        let derived_names = derived_names_for_variant(&variant_name, shared.windows.as_slice());
        result.derived_factor_count += derived_names.len();

        let selected_slots = collect_preflight_passed_slots(
            rolled_block.view(),
            shared.restrict.view(),
            &derived_names,
            &shared.config,
            &mut result,
        );
        if selected_slots.is_empty() {
            continue;
        }

        let _t = std::time::Instant::now();
        let (raw_gap1_results, raw_gap5_results) = legacy_backtest_gap1_gap5_selected_slots_f32(
            rolled_block.view(),
            shared.ret_gap1.view(),
            shared.ret_sum_gap1.view(),
            shared.ret_gap5.view(),
            shared.ret_sum_gap5.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            shared.dates.as_slice(),
            shared.backtest_start,
            &selected_slots,
            10,
            false,
        )?;
        PROF_BT_RAW.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);

        let selected_rolled_block =
            select_factor_slots_block(rolled_block.view(), &selected_slots)?;
        let _t = std::time::Instant::now();
        let neutralized = neutralize_block_legacy_exact(
            shared.legacy_style_data.as_ref(),
            selected_rolled_block.view(),
            shared.dates.as_slice(),
            shared.stocks.as_slice(),
            true,
            shared.min_valid,
            shared.industry_neutralize,
        )?;
        PROF_NEU.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        let local_slots = (0..selected_slots.len()).collect::<Vec<_>>();
        let _t = std::time::Instant::now();
        let (neu_gap1_results, neu_gap5_results) = legacy_backtest_gap1_gap5_selected_slots_f32(
            neutralized.view(),
            shared.ret_gap1.view(),
            shared.ret_sum_gap1.view(),
            shared.ret_gap5.view(),
            shared.ret_sum_gap5.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            shared.dates.as_slice(),
            shared.backtest_start,
            &local_slots,
            10,
            false,
        )?;
        PROF_BT_NEU.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);

        for (local_idx, &slot_idx) in selected_slots.iter().enumerate() {
            let derived_name = &derived_names[slot_idx];
            let raw_gap1_row = summary_from_row(
                derived_name,
                "rolled",
                1,
                &variant_name,
                &raw_gap1_results[local_idx].summary,
            );
            let raw_gap5_row = summary_from_row(
                derived_name,
                "rolled",
                5,
                &variant_name,
                &raw_gap5_results[local_idx].summary,
            );
            let neu_gap1_row = summary_from_row(
                derived_name,
                "neu",
                1,
                &variant_name,
                &neu_gap1_results[local_idx].summary,
            );
            let neu_gap5_row = summary_from_row(
                derived_name,
                "neu",
                5,
                &variant_name,
                &neu_gap5_results[local_idx].summary,
            );

            let raw_gap1_keep = qualify_raw(&raw_gap1_row, 1, &shared.config);
            let raw_gap5_keep = qualify_raw(&raw_gap5_row, 5, &shared.config);
            let neu_gap1_keep = qualify_neu(&neu_gap1_row, 1, &shared.config);
            let neu_gap5_keep = qualify_neu(&neu_gap5_row, 5, &shared.config);

            if raw_gap1_keep || raw_gap5_keep || neu_gap1_keep || neu_gap5_keep {
                result.passed = true;
            }

            if raw_gap1_keep {
                result.raw_summary_gap1.push(raw_gap1_row.clone());
                result.raw_ic_gap1.push(IcRecord {
                    factor_name: derived_name.clone(),
                    dates: raw_gap1_results[local_idx].ic_dates.clone(),
                    values: raw_gap1_results[local_idx].ic_values.clone(),
                });
            }
            if raw_gap5_keep {
                result.raw_summary_gap5.push(raw_gap5_row.clone());
                result.raw_ic_gap5.push(IcRecord {
                    factor_name: derived_name.clone(),
                    dates: raw_gap5_results[local_idx].ic_dates.clone(),
                    values: raw_gap5_results[local_idx].ic_values.clone(),
                });
            }
            if neu_gap1_keep {
                result.neu_summary_gap1.push(neu_gap1_row.clone());
            }
            if neu_gap5_keep {
                result.neu_summary_gap5.push(neu_gap5_row.clone());
            }
            if raw_gap1_keep || neu_gap1_keep {
                result.neu_ic_gap1.push(IcRecord {
                    factor_name: derived_name.clone(),
                    dates: neu_gap1_results[local_idx].ic_dates.clone(),
                    values: neu_gap1_results[local_idx].ic_values.clone(),
                });
            }
            if raw_gap5_keep || neu_gap5_keep {
                result.neu_ic_gap5.push(IcRecord {
                    factor_name: derived_name.clone(),
                    dates: neu_gap5_results[local_idx].ic_dates.clone(),
                    values: neu_gap5_results[local_idx].ic_values.clone(),
                });
            }
        }
    }

    let _n = PROF_COUNT.fetch_add(1, AtomicOrdering::Relaxed) + 1;
    if _n % 20 == 0 {
        prof_dump("完整回测");
    }
    Ok(result)
}

/// 原子写用的同目录临时路径（同目录保证 rename 不跨文件系统）。
fn atomic_tmp_path(path: &Path) -> Result<PathBuf, String> {
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("非法输出路径: {}", path.display()))?
        .to_string_lossy()
        .to_string();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    Ok(parent.join(format!("{}.tmp.{}.{}", file_name, std::process::id(), nanos)))
}

/// summary 产物：Rust 直接写 parquet。
///
/// 替代旧路径「Rust 写 573MB JSON → Python `json.load` → `pd.DataFrame` → `to_parquet`」。
///
/// schema（列序 = `SummaryRowRecord` 字段声明序 = 旧 JSON 的键序 = 旧
/// `_load_summary_json_full` 得到的 DataFrame 列序）：
///
/// | # | 列名 | Arrow 类型 | pandas dtype |
/// |---|---|---|---|
/// | 0 | factor_name | Utf8 | object |
/// | 1 | stage | Utf8 | object |
/// | 2 | gap | Int64 | int64 |
/// | 3 | source_factor | Utf8 | object |
/// | 4 | preflight_passed | Boolean | bool |
/// | 5 | IC_mean | Float64 | float64 |
/// | 6 | IR | Float64 | float64 |
/// | 7 | annualized_return | Float64 | float64 |
/// | 8 | sharpe_ratio | Float64 | float64 |
/// | 9 | max_drawdown | Float64 | float64 |
/// | 10 | date_size | Int64 | int64 |
/// | 11 | ratio_mean | Float64 | float64 |
/// | 12 | hedge_annualized_return | Float64 | float64 |
/// | 13 | hedge_annualized_sharpe_ratio | Float64 | float64 |
/// | 14 | hedge_max_drawdown | Float64 | float64 |
/// | 15 | SSM | Float64 | float64 |
/// | 16 | MPROB | Float64 | float64 |
/// | 17 | BRC | Float64 | float64 |
/// | 18 | BRC_S | Float64 | float64 |
/// | 19 | BRC_L | Float64 | float64 |
///
/// 行序 = `rows` 原序（与旧 JSON 数组顺序相同）。全部列 nullable=true，但实际不产生 null。
/// 非有限 f64 一律写 NaN：旧路径经 serde_json 会把 NaN/±Inf 序列化成 `null`，Python
/// `json.load` 得 None、建 DataFrame 后即 NaN，故语义逐字段一致。
fn write_summary_parquet(path: &Path, rows: &[SummaryRowRecord]) -> Result<(), String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("factor_name", DataType::Utf8, true),
        Field::new("stage", DataType::Utf8, true),
        Field::new("gap", DataType::Int64, true),
        Field::new("source_factor", DataType::Utf8, true),
        Field::new("preflight_passed", DataType::Boolean, true),
        Field::new("IC_mean", DataType::Float64, true),
        Field::new("IR", DataType::Float64, true),
        Field::new("annualized_return", DataType::Float64, true),
        Field::new("sharpe_ratio", DataType::Float64, true),
        Field::new("max_drawdown", DataType::Float64, true),
        Field::new("date_size", DataType::Int64, true),
        Field::new("ratio_mean", DataType::Float64, true),
        Field::new("hedge_annualized_return", DataType::Float64, true),
        Field::new("hedge_annualized_sharpe_ratio", DataType::Float64, true),
        Field::new("hedge_max_drawdown", DataType::Float64, true),
        Field::new("SSM", DataType::Float64, true),
        Field::new("MPROB", DataType::Float64, true),
        Field::new("BRC", DataType::Float64, true),
        Field::new("BRC_S", DataType::Float64, true),
        Field::new("BRC_L", DataType::Float64, true),
    ]));
    let fin = |v: f64| if v.is_finite() { v } else { f64::NAN };
    let f64_col = |get: fn(&SummaryRowRecord) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from(
            rows.iter().map(|r| fin(get(r))).collect::<Vec<_>>(),
        ))
    };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.factor_name.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.stage.as_str()),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.gap as i64).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.source_factor.as_str()),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|r| r.preflight_passed).collect::<Vec<_>>(),
        )),
        f64_col(|r| r.ic_mean),
        f64_col(|r| r.ir),
        f64_col(|r| r.annualized_return),
        f64_col(|r| r.sharpe_ratio),
        f64_col(|r| r.max_drawdown),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.date_size as i64).collect::<Vec<_>>(),
        )),
        f64_col(|r| r.ratio_mean),
        f64_col(|r| r.hedge_annualized_return),
        f64_col(|r| r.hedge_annualized_sharpe_ratio),
        f64_col(|r| r.hedge_max_drawdown),
        f64_col(|r| r.ssm),
        f64_col(|r| r.mprob),
        f64_col(|r| r.brc),
        f64_col(|r| r.brc_s),
        f64_col(|r| r.brc_l),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| format!("构造 summary RecordBatch 失败: {}", e))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("创建 metrics 目录失败: {}", e))?;
    }
    let tmp = atomic_tmp_path(path)?;
    let write_res = (|| -> Result<(), String> {
        let file = File::create(&tmp).map_err(|e| format!("创建 summary parquet 失败: {}", e))?;
        // SNAPPY = pyarrow `to_parquet` 的默认压缩，与旧路径产出的 parquet 同一量级。
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| format!("创建 ArrowWriter 失败: {}", e))?;
        writer
            .write(&batch)
            .map_err(|e| format!("写 summary parquet 失败: {}", e))?;
        writer
            .close()
            .map_err(|e| format!("关闭 summary parquet 失败: {}", e))?;
        Ok(())
    })();
    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        format!("summary parquet 原子改名失败 {}: {}", path.display(), e)
    })
}

fn write_ic_outputs(
    matrix_path: &Path,
    names_path: &Path,
    dates_path: &Path,
    store: &HashMap<String, IcRecord>,
) -> Result<(), String> {
    if let Some(parent) = matrix_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("创建 ic 目录失败: {}", e))?;
    }
    let factor_names = {
        let mut names = store.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    };
    let all_dates = {
        let mut date_set = BTreeSet::<i32>::new();
        for record in store.values() {
            for &date in &record.dates {
                date_set.insert(date);
            }
        }
        date_set.into_iter().collect::<Vec<_>>()
    };
    let date_positions = all_dates
        .iter()
        .enumerate()
        .map(|(idx, date)| (*date, idx))
        .collect::<HashMap<_, _>>();
    let matrix = if factor_names.is_empty() {
        Array2::<f32>::from_shape_vec((all_dates.len(), 0), Vec::new())
            .map_err(|e| format!("构造空 ic 矩阵失败: {}", e))?
    } else {
        let mut data = vec![f32::NAN; all_dates.len() * factor_names.len()];
        for (col_idx, name) in factor_names.iter().enumerate() {
            let record = store
                .get(name)
                .ok_or_else(|| format!("缺少候选 IC 数据: {}", name))?;
            for (&date, &value) in record.dates.iter().zip(record.values.iter()) {
                if let Some(&row_idx) = date_positions.get(&date) {
                    data[row_idx * factor_names.len() + col_idx] = value;
                }
            }
        }
        Array2::<f32>::from_shape_vec((all_dates.len(), factor_names.len()), data)
            .map_err(|e| format!("构造 ic 矩阵失败: {}", e))?
    };
    write_npy(matrix_path, &matrix).map_err(|e| format!("写 ic npy 失败: {}", e))?;
    let file = File::create(names_path).map_err(|e| format!("创建 ic names json 失败: {}", e))?;
    serde_json::to_writer(BufWriter::new(file), &factor_names)
        .map_err(|e| format!("写 ic names json 失败: {}", e))?;
    let dates_array = Array1::<i32>::from_vec(all_dates);
    write_npy(dates_path, &dates_array).map_err(|e| format!("写 ic dates npy 失败: {}", e))?;
    Ok(())
}

pub(crate) fn write_aggregated_outputs(
    cache_root: &Path,
    aggregated: &AggregatedCandidates,
) -> Result<(), String> {
    let metrics_dir = cache_root.join("metrics");
    let ic_dir = cache_root.join("ic_ts");
    write_summary_parquet(
        &metrics_dir.join("summary_rolled_gap1_candidates.parquet"),
        &aggregated.raw_summary_gap1,
    )?;
    write_summary_parquet(
        &metrics_dir.join("summary_rolled_gap5_candidates.parquet"),
        &aggregated.raw_summary_gap5,
    )?;
    write_summary_parquet(
        &metrics_dir.join("summary_neu_gap1_candidates.parquet"),
        &aggregated.neu_summary_gap1,
    )?;
    write_summary_parquet(
        &metrics_dir.join("summary_neu_gap5_candidates.parquet"),
        &aggregated.neu_summary_gap5,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_rolled_gap1.npy"),
        &ic_dir.join("ic_rolled_gap1_names.json"),
        &ic_dir.join("ic_rolled_gap1_dates.npy"),
        &aggregated.raw_ic_gap1,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_rolled_gap5.npy"),
        &ic_dir.join("ic_rolled_gap5_names.json"),
        &ic_dir.join("ic_rolled_gap5_dates.npy"),
        &aggregated.raw_ic_gap5,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_neu_gap1.npy"),
        &ic_dir.join("ic_neu_gap1_names.json"),
        &ic_dir.join("ic_neu_gap1_dates.npy"),
        &aggregated.neu_ic_gap1,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_neu_gap5.npy"),
        &ic_dir.join("ic_neu_gap5_names.json"),
        &ic_dir.join("ic_neu_gap5_dates.npy"),
        &aggregated.neu_ic_gap5,
    )?;

    // metrics-only 全量产物：任何通过 raw_cover 的 derived slot 都会出现在这里，
    // 即使 preflight 不过、收益/IC 不达候选阈值。文件名以 _all 区分 candidates。
    // 注意：即使整批全部 raw_cover 失败、聚合为空，也强制写出空文件，
    // 这样 skill B 才能完整闭环“每个 expected source factor 都有结论”。
    write_summary_parquet(
        &metrics_dir.join("summary_rolled_gap1_all.parquet"),
        &aggregated.all_raw_summary_gap1,
    )?;
    write_summary_parquet(
        &metrics_dir.join("summary_rolled_gap5_all.parquet"),
        &aggregated.all_raw_summary_gap5,
    )?;
    write_summary_parquet(
        &metrics_dir.join("summary_neu_gap1_all.parquet"),
        &aggregated.all_neu_summary_gap1,
    )?;
    write_summary_parquet(
        &metrics_dir.join("summary_neu_gap5_all.parquet"),
        &aggregated.all_neu_summary_gap5,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_rolled_gap1_all.npy"),
        &ic_dir.join("ic_rolled_gap1_all_names.json"),
        &ic_dir.join("ic_rolled_gap1_all_dates.npy"),
        &aggregated.all_raw_ic_gap1,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_rolled_gap5_all.npy"),
        &ic_dir.join("ic_rolled_gap5_all_names.json"),
        &ic_dir.join("ic_rolled_gap5_all_dates.npy"),
        &aggregated.all_raw_ic_gap5,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_neu_gap1_all.npy"),
        &ic_dir.join("ic_neu_gap1_all_names.json"),
        &ic_dir.join("ic_neu_gap1_all_dates.npy"),
        &aggregated.all_neu_ic_gap1,
    )?;
    write_ic_outputs(
        &ic_dir.join("ic_neu_gap5_all.npy"),
        &ic_dir.join("ic_neu_gap5_all_names.json"),
        &ic_dir.join("ic_neu_gap5_all_dates.npy"),
        &aggregated.all_neu_ic_gap5,
    )?;
    Ok(())
}

pub(crate) struct ProcessStats {
    pub(crate) restored_pass: usize,
    pub(crate) restored_raw_cov: usize,
    pub(crate) restored_preflight: usize,
    pub(crate) restored_ret_ic: usize,
    pub(crate) restored_unknown: usize,

    pub(crate) done: usize,
    pub(crate) done_pass: usize,
    pub(crate) done_raw_cov: usize,
    pub(crate) done_preflight: usize,
    pub(crate) done_ret_ic: usize,
    pub(crate) done_unknown: usize,

    pub(crate) preflight_maj_windows: usize,
    pub(crate) preflight_zero_windows: usize,
    pub(crate) preflight_nan_windows: usize,
}

impl Default for ProcessStats {
    fn default() -> Self {
        ProcessStats {
            restored_pass: 0,
            restored_raw_cov: 0,
            restored_preflight: 0,
            restored_ret_ic: 0,
            restored_unknown: 0,
            done: 0,
            done_pass: 0,
            done_raw_cov: 0,
            done_preflight: 0,
            done_ret_ic: 0,
            done_unknown: 0,
            preflight_maj_windows: 0,
            preflight_zero_windows: 0,
            preflight_nan_windows: 0,
        }
    }
}

#[cfg(unix)]
fn terminal_height() -> u16 {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_row > 0 {
            ws.ws_row
        } else {
            24
        }
    }
}

#[cfg(windows)]
fn terminal_height() -> u16 {
    24
}

/// stdout 是否为真实终端。任务管理系统把 stdout 重定向到文件时返回 false，
/// 此时 ANSI 光标定位/滚动区转义序列全部失效（无法原地刷新），进度必须改用 println! 换行输出，
/// 否则进度数据被行缓冲吞掉，Web UI 日志弹窗只能看到淘汰日志、看不到进度栏。
#[cfg(unix)]
pub(crate) fn is_terminal() -> bool {
    unsafe { libc::isatty(1) != 0 }
}

#[cfg(windows)]
pub(crate) fn is_terminal() -> bool {
    // Windows 用 console handle 判断；保守起见统一直接到 print 分支
    unsafe { libc::isatty(1) != 0 }
}

pub(crate) fn init_status_line() {
    if !is_terminal() {
        return;
    }
    let h = terminal_height();
    if h > 4 {
        print!("\x1B[1;{}r", h - 3);
    }
    let _ = std::io::stdout().flush();
}

/// 非 TTY（重定向到文件/Web UI）时的节流计数器：每 N 次调用输出一行，避免 61344 个因子刷屏。
static NON_TTY_STATUS_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const NON_TTY_STATUS_EVERY: u64 = 50;

pub(crate) fn update_status_line(l1: &str, l2: &str, l3: &str) {
    if !is_terminal() {
        let n = NON_TTY_STATUS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n % NON_TTY_STATUS_EVERY == 0 {
            println!("{l1} | {l2} | {l3}");
        }
        return;
    }
    let h = terminal_height();
    if h > 3 {
        print!("\x1B7");
        print!("\x1B[{};1H\x1B[K{}", h - 2, l1);
        print!("\x1B[{};1H\x1B[K{}", h - 1, l2);
        print!("\x1B[{};1H\x1B[K{}", h, l3);
        print!("\x1B8");
    }
    let _ = std::io::stdout().flush();
}

pub(crate) fn reset_status_line() {
    if !is_terminal() {
        return;
    }
    print!("\x1B[r");
    let h = terminal_height();
    for i in 0..3 {
        print!("\x1B[{};1H\x1B[K", h - 2 + i);
    }
    let _ = std::io::stdout().flush();
}

#[pyfunction]
#[pyo3(signature = (
    factor_names,
    factor_paths,
    dates,
    stocks,
    windows,
    fold,
    n_jobs,
    min_valid,
    cache_root,
    style_vars_dir,
    ret_gap1_path,
    ret_sum_gap1_path,
    ret_gap5_path,
    ret_sum_gap5_path,
    restrict_path,
    index_ret_path,
    backtest_start,
    cover_rate=0.97,
    ret_point_neu_gap5=0.055,
    ret_point_neu_gap1=0.08,
    ic_point_neu_gap5=0.01,
    ic_point_neu_gap1=0.006,
    ret_point_gap5=0.1,
    ret_point_gap1=0.13,
    ic_point_gap5=0.03,
    ic_point_gap1=0.02,
    ic_more_important_gap5=0.01,
    ic_more_important_gap1=0.006,
    majority_count_threshold=200.0,
    zero_max_threshold=0.01,
    nan_max_threshold=0.04,
))]
pub fn tail_v5_run_candidates<'py>(
    py: Python<'py>,
    factor_names: Vec<String>,
    factor_paths: Vec<String>,
    dates: Vec<i32>,
    stocks: Vec<String>,
    windows: Vec<usize>,
    fold: bool,
    n_jobs: usize,
    min_valid: usize,
    cache_root: String,
    style_vars_dir: String,
    ret_gap1_path: String,
    ret_sum_gap1_path: String,
    ret_gap5_path: String,
    ret_sum_gap5_path: String,
    restrict_path: String,
    index_ret_path: String,
    backtest_start: i32,
    cover_rate: f64,
    ret_point_neu_gap5: f64,
    ret_point_neu_gap1: f64,
    ic_point_neu_gap5: f64,
    ic_point_neu_gap1: f64,
    ret_point_gap5: f64,
    ret_point_gap1: f64,
    ic_point_gap5: f64,
    ic_point_gap1: f64,
    ic_more_important_gap5: Option<f64>,
    ic_more_important_gap1: Option<f64>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PyResult<PyObject> {
    if factor_names.len() != factor_paths.len() {
        return Err(PyValueError::new_err(
            "factor_names 和 factor_paths 长度必须一致",
        ));
    }
    if n_jobs == 0 {
        return Err(PyValueError::new_err("n_jobs 必须大于 0"));
    }
    if (ic_more_important_gap5.is_some()) != (ic_more_important_gap1.is_some()) {
        return Err(PyValueError::new_err(
            "ic_more_important_gap5 和 ic_more_important_gap1 必须同时有值或同时为 None",
        ));
    }

    let output = py.allow_threads(|| -> Result<(usize, usize, HashMap<String, usize>), String> {
        let started = Instant::now();
        let cache_root_path = PathBuf::from(&cache_root);
        let task_results_dir = cache_root_path.join("task_results");
        let logs_dir = cache_root_path.join("logs");
        let completed_log_path = logs_dir.join("completed_sources.txt");
        fs::create_dir_all(&task_results_dir).map_err(|e| format!("创建 task_results 目录失败: {}", e))?;
        fs::create_dir_all(&logs_dir).map_err(|e| format!("创建 logs 目录失败: {}", e))?;

        let shared = SharedInputs {
            dates: Arc::new(dates),
            stocks: Arc::new(stocks),
            windows: Arc::new(windows),
            fold,
            min_valid,
            backtest_start,
            legacy_style_data: Arc::new(
                IOOptimizedStyleData::load_from_vars_h5(&style_vars_dir)
                    .map_err(|e| e.to_string())?
            ),
            industry_neutralize: true,
            industry: None,
            neutralize_std_shared: None,
            ret_gap1: Arc::new(read_npy(&ret_gap1_path).map_err(|e| format!("读取 ret_gap1.npy 失败: {}", e))?),
            ret_sum_gap1: Arc::new(read_npy(&ret_sum_gap1_path).map_err(|e| format!("读取 ret_sum_gap1.npy 失败: {}", e))?),
            ret_gap5: Arc::new(read_npy(&ret_gap5_path).map_err(|e| format!("读取 ret_gap5.npy 失败: {}", e))?),
            ret_sum_gap5: Arc::new(read_npy(&ret_sum_gap5_path).map_err(|e| format!("读取 ret_sum_gap5.npy 失败: {}", e))?),
            restrict: Arc::new(read_npy(&restrict_path).map_err(|e| format!("读取 restrict.npy 失败: {}", e))?),
            index_ret: Arc::new(read_npy(&index_ret_path).map_err(|e| format!("读取 index_ret.npy 失败: {}", e))?),
            config: Arc::new(TailSelectionConfig {
                cover_rate,
                ret_point_neu_gap5,
                ret_point_neu_gap1,
                ic_point_neu_gap5,
                ic_point_neu_gap1,
                ret_point_gap5,
                ret_point_gap1,
                ic_point_gap5,
                ic_point_gap1,
                ic_more_important_gap5,
                ic_more_important_gap1,
                majority_count_threshold,
                zero_max_threshold,
                nan_max_threshold,
                save_all_metrics: false,
                ic_only: false,
            }),
            bt_pre: None,
        free_mask: None,
        v3_shared: None,
        };

        let mut aggregated = AggregatedCandidates::default();
        let mut completed_sources = HashSet::<String>::new();
        let mut stats = ProcessStats::default();
        for (source_factor, factor_path) in factor_names.iter().zip(factor_paths.iter()) {
            let result_path = factor_result_path(&task_results_dir, source_factor);
            if result_path.exists() {
                if let Ok(mut task_result) = read_task_result(&result_path) {
                    if !task_result.passed
                        && (!task_result.raw_summary_gap1.is_empty()
                            || !task_result.raw_summary_gap5.is_empty()
                            || !task_result.neu_summary_gap1.is_empty()
                            || !task_result.neu_summary_gap5.is_empty())
                    {
                        task_result.passed = true;
                    }
                    if task_result.passed {
                        stats.restored_pass += 1;
                    } else if task_result.eliminated_by_raw_cover {
                        stats.restored_raw_cov += 1;
                    } else if !task_result.any_window_passed_preflight
                        && !task_result.raw_summary_gap1.is_empty() == false
                        && !task_result.raw_summary_gap5.is_empty() == false
                        && !task_result.neu_summary_gap1.is_empty() == false
                        && !task_result.neu_summary_gap5.is_empty() == false
                        && !task_result.passed
                    {
                        // 区分 preflight 淘汰 vs 未知:
                        // 旧缓存没有 any_window_passed_preflight(默认false)
                        // 也没有 eliminated_by_raw_cover(默认false)
                        // 如果所有 summary vecs 都为空且 passed=false
                        // 我们通过是否有 preflight 失败记录来判断
                        if task_result.preflight_maj_failed_windows > 0
                            || task_result.preflight_zero_failed_windows > 0
                            || task_result.preflight_nan_failed_windows > 0
                        {
                            stats.restored_preflight += 1;
                        } else {
                            stats.restored_unknown += 1;
                        }
                    } else {
                        // any_window_passed_preflight=true, passed=false → ret_ic淘汰
                        stats.restored_ret_ic += 1;
                    }
                    stats.preflight_maj_windows += task_result.preflight_maj_failed_windows;
                    stats.preflight_zero_windows += task_result.preflight_zero_failed_windows;
                    stats.preflight_nan_windows += task_result.preflight_nan_failed_windows;
                    aggregated.merge_task(task_result);
                    completed_sources.insert(source_factor.clone());
                    continue;
                }
            }
            let _ = factor_path;
        }
        let restored_sources =
            stats.restored_pass + stats.restored_raw_cov + stats.restored_preflight
            + stats.restored_ret_ic + stats.restored_unknown;

        let pending_tasks = factor_names
            .iter()
            .zip(factor_paths.iter())
            .filter_map(|(source_factor, factor_path)| {
                if completed_sources.contains(source_factor) {
                    None
                } else {
                    Some(TailTask {
                        source_factor: source_factor.clone(),
                        factor_path: factor_path.clone(),
                    })
                }
            })
            .collect::<Vec<_>>();

        let total_pending = pending_tasks.len();
        if total_pending > 0 {
            init_status_line();
            let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            let total = factor_names.len();
            let total_elim = restored_sources - stats.restored_pass;
            let l1 = format!("[{}] Tail V4 启动，待处理 {}/{} 个原始因子", current_time, total_pending, total);
            let l2 = format!("累计通过 {} ({}%) | 淘汰 raw_cov={} preflight={} ret_ic={} 未知={}",
                stats.restored_pass,
                if restored_sources > 0 { stats.restored_pass * 100 / restored_sources } else { 0 },
                stats.restored_raw_cov, stats.restored_preflight,
                stats.restored_ret_ic, stats.restored_unknown);
            let _ = total_elim; // suppress unused warning
            let l3 = format!("maj={}w zero={}w nan={}w | 恢复 {} 个 | 即将开始处理...",
                stats.preflight_maj_windows, stats.preflight_zero_windows,
                stats.preflight_nan_windows, restored_sources);
            update_status_line(&l1, &l2, &l3);
        }
        let (task_sender, task_receiver): (Sender<TailTask>, Receiver<TailTask>) = unbounded();
        let (result_sender, result_receiver) = unbounded::<Result<TailTaskResult, (String, String)>>();
        for task in pending_tasks {
            task_sender.send(task).map_err(|e| format!("发送任务失败: {}", e))?;
        }
        drop(task_sender);

        let shared_arc = Arc::new(shared);

        // 检测是否为列式存储模式（任一 task 路径含 "::"）。
        // 若是，启用 IO/CPU 分离架构：少量 IO 线程顺序读因子 → 有界内存队列 → n_jobs 计算线程。
        // 否则保持原逻辑：n_jobs 线程各自读盘 + 计算（兼容 parquet/h5）。
        let is_colblk_mode = factor_paths.iter().any(|p| p.contains("::"));

        if is_colblk_mode {
            // ---- IO/CPU 分离架构 ----
            // IO 线程数：匹配 HDD 物理盘条带并行度，默认 8（8 盘 LVM）。
            let n_io_threads = if n_jobs >= 200 { 8 } else { 4 };
            // 已读好的 (TailTask, Array2<f32>) 队列，有界实现反压（计算跟不上则 IO 阻塞，不撑爆内存）
            let (loaded_tx, loaded_rx) =
                crossbeam::channel::bounded::<(TailTask, Array2<f32>)>(16);

            // 解析唯一的 store_dir（所有 colblk task 共享同一 Reader）
            let store_dir_for_reader = factor_paths
                .iter()
                .find_map(|p| {
                    if p.contains("::") {
                        let sp = p.splitn(2, "::").next().unwrap_or("");
                        if sp.ends_with(".colblk") {
                            Path::new(sp).parent().map(|x| x.to_string_lossy().to_string())
                        } else {
                            Some(sp.to_string())
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            // 启动 IO 读取线程（顺序读，单 Reader 共享）
            let mut io_handles = Vec::with_capacity(n_io_threads);
            let task_rx_io = task_receiver.clone();
            for io_idx in 0..n_io_threads {
                let task_rx = task_rx_io.clone();
                let loaded_tx = loaded_tx.clone();
                let store_dir = store_dir_for_reader.clone();
                let dates = shared_arc.dates.clone();
                let stocks = shared_arc.stocks.clone();
                io_handles.push(thread::spawn(move || {
                    // 每个 IO 线程独立打开 Reader（pread，只读，多线程安全）
                    let reader = match crate::factor_store_v5::FactorStoreReader::open(&store_dir) {
                        Ok(r) => r,
                        Err(_) => return,
                    };
                    // 预计算 scatter 映射（只算一次，61344 个因子复用）
                    let scatter_maps = reader.precompute_scatter_maps(
                        dates.as_slice(),
                        stocks.as_slice(),
                    );
                    // 线程间按 task 顺序取，天然趋向顺序读投影区
                    while let Ok(task) = task_rx.recv() {
                        // 解析 col_idx
                        let col_idx = match task.factor_path.splitn(2, "::").nth(1) {
                            Some(s) => match s.parse::<usize>() {
                                Ok(v) => v,
                                Err(_) => continue,
                            },
                            None => continue,
                        };
                        let matrix = match reader.read_factor_to_matrix_fast(
                            col_idx,
                            dates.as_slice(),
                            stocks.as_slice(),
                            &scatter_maps,
                        ) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };
                        if loaded_tx.send((task, matrix)).is_err() {
                            break;
                        }
                    }
                    let _ = io_idx; // 标识用
                }));
            }
            drop(loaded_tx);

            // 启动 n_jobs 计算线程（纯内存，完全不碰盘）
            let mut handles = Vec::with_capacity(n_jobs);
            for _ in 0..n_jobs {
                let loaded_rx = loaded_rx.clone();
                let tx = result_sender.clone();
                let shared_clone = shared_arc.clone();
                handles.push(thread::spawn(move || {
                    while let Ok((task, raw_values)) = loaded_rx.recv() {
                        let task_name = task.source_factor.clone();
                        let outcome = process_task_with_values(&task, raw_values, &shared_clone)
                            .map_err(|err| (task_name, err));
                        if tx.send(outcome).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(result_sender);

            for h in io_handles {
                let _ = h.join();
            }
            for h in handles {
                let _ = h.join();
            }
        } else {
            // ---- 原架构：n_jobs 线程各自读盘 + 计算（兼容 parquet/h5）----
            let mut handles = Vec::with_capacity(n_jobs);
            for _ in 0..n_jobs {
                let rx = task_receiver.clone();
                let tx = result_sender.clone();
                let shared_clone = shared_arc.clone();
                handles.push(thread::spawn(move || {
                    while let Ok(task) = rx.recv() {
                        let task_name = task.source_factor.clone();
                        let outcome = process_task(&task, &shared_clone)
                            .map_err(|err| (task_name, err));
                        if tx.send(outcome).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(result_sender);
            for h in handles {
                let _ = h.join();
            }
        }

        let mut processed_sources = 0usize;
        while let Ok(task_outcome) = result_receiver.recv() {
            match task_outcome {
                Ok(task_result) => {
                    let result_path = factor_result_path(&task_results_dir, &task_result.source_factor);
                    let is_passed = task_result.passed;
                    let is_raw_cov = task_result.eliminated_by_raw_cover;
                    let any_window = task_result.any_window_passed_preflight;
                    let preflight_maj = task_result.preflight_maj_failed_windows;
                    let preflight_zero = task_result.preflight_zero_failed_windows;
                    let preflight_nan = task_result.preflight_nan_failed_windows;
                    write_task_result(&result_path, &task_result)?;
                    append_completed_source(&completed_log_path, &task_result.source_factor)?;
                    aggregated.merge_task(task_result);
                    processed_sources += 1;

                    if is_passed {
                        stats.done_pass += 1;
                    } else if is_raw_cov {
                        stats.done_raw_cov += 1;
                    } else if !any_window {
                        stats.done_preflight += 1;
                    } else {
                        stats.done_ret_ic += 1;
                    }
                    stats.done = processed_sources;
                    stats.preflight_maj_windows += preflight_maj;
                    stats.preflight_zero_windows += preflight_zero;
                    stats.preflight_nan_windows += preflight_nan;

                    if total_pending > 0 {
                        let elapsed = started.elapsed();
                        let elapsed_secs = elapsed.as_secs();
                        let progress = processed_sources as f64 / total_pending as f64;
                        let estimated_total_secs = if progress > 0.0 {
                            elapsed.as_secs_f64() / progress
                        } else {
                            elapsed.as_secs_f64()
                        };
                        let remaining_secs = if estimated_total_secs > elapsed.as_secs_f64() {
                            (estimated_total_secs - elapsed.as_secs_f64()) as u64
                        } else {
                            0
                        };
                        let (elapsed_h, elapsed_m, elapsed_s) = format_hms(elapsed_secs);
                        let (remaining_h, remaining_m, remaining_s) = format_hms(remaining_secs);
                        let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

                        let cum_pass = stats.restored_pass + stats.done_pass;
                        let cum_total = restored_sources + processed_sources;
                        let cum_raw_cov = stats.restored_raw_cov + stats.done_raw_cov;
                        let cum_preflight = stats.restored_preflight + stats.done_preflight;
                        let cum_ret_ic = stats.restored_ret_ic + stats.done_ret_ic;
                        let cum_unknown = stats.restored_unknown + stats.done_unknown;

                        let l1 = format!(
                            "[{}] Tail V4 进度 {}/{} ({:.1}%)，已用{}h{}m{}s，预计剩余{}h{}m{}s",
                            current_time, processed_sources, total_pending,
                            progress * 100.0,
                            elapsed_h, elapsed_m, elapsed_s,
                            remaining_h, remaining_m, remaining_s,
                        );
                        let l2 = format!(
                            "累计通过 {} ({:.0}%) | 淘汰 raw_cov={} preflight={} ret_ic={} 未知={}",
                            cum_pass,
                            if cum_total > 0 { cum_pass as f64 * 100.0 / cum_total as f64 } else { 0.0 },
                            cum_raw_cov, cum_preflight, cum_ret_ic, cum_unknown,
                        );
                        let l3 = format!(
                            "maj={}w zero={}w nan={}w | 本次 {}(通过{}) | 恢复 {}(通过{})",
                            stats.preflight_maj_windows, stats.preflight_zero_windows,
                            stats.preflight_nan_windows,
                            stats.done, stats.done_pass,
                            restored_sources, stats.restored_pass,
                        );
                        update_status_line(&l1, &l2, &l3);
                    }
                }
                Err((task_name, err)) => {
                    reset_status_line();
                    return Err(format!("处理因子 {} 失败: {}", task_name, err));
                }
            }
        }

        if total_pending > 0 {
            println!();
            reset_status_line();
        }

        // 注意：worker 线程在上方 IO/CPU 分离分支或原架构分支内已全部 join 完毕。

        write_aggregated_outputs(
            &cache_root_path,
            &aggregated,
        )?;

        let mut candidate_counts = HashMap::new();
        candidate_counts.insert("rolled_gap1".to_string(), aggregated.raw_summary_gap1.len());
        candidate_counts.insert("rolled_gap5".to_string(), aggregated.raw_summary_gap5.len());
        candidate_counts.insert("neu_gap1".to_string(), aggregated.neu_summary_gap1.len());
        candidate_counts.insert("neu_gap5".to_string(), aggregated.neu_summary_gap5.len());
        Ok((processed_sources, restored_sources, candidate_counts))
    }).map_err(PyRuntimeError::new_err)?;

    let info = PyDict::new(py);
    info.set_item("processed_sources", output.0)?;
    info.set_item("restored_sources", output.1)?;
    let candidate_counts = PyDict::new(py);
    for (key, value) in output.2 {
        candidate_counts.set_item(key, value)?;
    }
    info.set_item("candidate_counts", candidate_counts)?;
    Ok(info.into())
}

#[pyfunction]
#[pyo3(signature = (
    ver,
    gap5_selected,
    gap1_selected,
    cache_root,
    temp_root,
    source_dir,
    style_vars_dir,
    min_valid=12,
    start_date="2016-01-01",
    backtest_start_date="2016-02-01",
    end_date="2024-12-31",
    fulltest_jobs=32,
    python_path="/home/chenzongwei/.conda/envs/chenzongwei311/bin/python",
    resume=true,
    index_name="000905",
    verbose=false
))]
pub fn tail_v5_run_fulltest_queue<'py>(
    py: Python<'py>,
    ver: String,
    gap5_selected: Vec<String>,
    gap1_selected: Vec<String>,
    cache_root: String,
    temp_root: String,
    source_dir: String,
    style_vars_dir: String,
    min_valid: usize,
    start_date: &str,
    backtest_start_date: &str,
    end_date: &str,
    fulltest_jobs: usize,
    python_path: &str,
    resume: bool,
    index_name: &str,
    verbose: bool,
) -> PyResult<PyObject> {
    if fulltest_jobs == 0 {
        return Err(PyValueError::new_err("fulltest_jobs 必须大于 0"));
    }

    let output = py
        .allow_threads(
            || -> Result<(usize, usize, HashMap<String, usize>), String> {
                let started = Instant::now();
                let cache_root_path = PathBuf::from(&cache_root);
                let postprocess_root = cache_root_path.join("postprocess_fulltest");
                let done_dir = postprocess_root.join("done");
                if !resume && postprocess_root.exists() {
                    fs::remove_dir_all(&postprocess_root)
                        .map_err(|e| format!("删除旧 fulltest 恢复目录失败: {}", e))?;
                }
                fs::create_dir_all(&done_dir)
                    .map_err(|e| format!("创建 fulltest done 目录失败: {}", e))?;

                let all_tasks = build_fulltest_tasks(&gap5_selected, &gap1_selected);
                let total_tasks = all_tasks.len();

                let mut bucket_counts = HashMap::<String, usize>::new();
                let mut pending_tasks = VecDeque::<TailV4FulltestTask>::new();
                let mut restored_tasks = 0usize;
                for task in all_tasks {
                    let done_path = fulltest_done_path(&done_dir, &task);
                    if resume && done_path.exists() {
                        restored_tasks += 1;
                        increment_fulltest_bucket(&mut bucket_counts, &task.stage, task.gap);
                    } else {
                        pending_tasks.push_back(task);
                    }
                }
                let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
                let lead = if is_terminal() { "\r" } else { "" };
                let trail = if is_terminal() { "" } else { "\n" };
                print!(
                    "{lead}[{}] Fulltest 启动，待处理 {}/{} 个任务，已恢复 {} 个{trail}",
                    current_time,
                    pending_tasks.len(),
                    total_tasks,
                    restored_tasks,
                );
                std::io::stdout()
                    .flush()
                    .map_err(|e| format!("刷新 fulltest 启动进度失败: {}", e))?;

                if pending_tasks.is_empty() {
                    render_fulltest_progress(
                        0,
                        restored_tasks,
                        total_tasks,
                        started,
                        &bucket_counts,
                    )?;
                    println!();
                    return Ok((0, restored_tasks, bucket_counts));
                }

                let worker_config = TailV4FulltestWorkerConfig {
                    ver: ver.clone(),
                    temp_root,
                    source_dir,
                    factor_names: {
                        let mut factor_names = Vec::new();
                        let mut seen = HashSet::<String>::new();
                        for factor_name in gap5_selected.iter().chain(gap1_selected.iter()) {
                            if seen.insert(factor_name.clone()) {
                                factor_names.push(factor_name.clone());
                            }
                        }
                        factor_names
                    },
                    start_date: start_date.to_string(),
                    backtest_start_date: backtest_start_date.to_string(),
                    end_date: end_date.to_string(),
                    style_vars_dir,
                    min_valid,
                    index_name: index_name.to_string(),
                };
                let worker_config_json = serde_json::to_string(&worker_config)
                    .map_err(|e| format!("序列化 worker 配置失败: {}", e))?;

                let pending_total = pending_tasks.len();
                let task_queue = Arc::new(Mutex::new(pending_tasks));
                let stop_flag = Arc::new(AtomicBool::new(false));
                let pid_registry: TailV4PidRegistry = Arc::new(Mutex::new(HashMap::new()));
                let (result_sender, result_receiver) =
                    unbounded::<Result<TailV4FulltestWorkerResult, String>>();
                let worker_count = fulltest_jobs.min(pending_total.max(1));
                let mut handles = Vec::with_capacity(worker_count);
                for worker_id in 0..worker_count {
                    let queue_clone = Arc::clone(&task_queue);
                    let sender_clone = result_sender.clone();
                    let stop_clone = Arc::clone(&stop_flag);
                    let pid_registry_clone = Arc::clone(&pid_registry);
                    let python_path_owned = python_path.to_string();
                    let worker_config_json_clone = worker_config_json.clone();
                    handles.push(thread::spawn(move || {
                        run_tail_v4_fulltest_worker_process(
                            worker_id,
                            queue_clone,
                            sender_clone,
                            stop_clone,
                            pid_registry_clone,
                            python_path_owned,
                            worker_config_json_clone,
                            verbose,
                        );
                    }));
                }
                drop(result_sender);

                let mut processed_tasks = 0usize;
                let mut fatal_error: Option<String> = None;
                let idle_timeout = tail_v4_fulltest_idle_timeout();
                let mut last_progress_at = Instant::now();
                while processed_tasks < pending_total {
                    match result_receiver
                        .recv_timeout(Duration::from_secs(TAIL_V5_FULLTEST_RESULT_POLL_SECS))
                    {
                        Ok(Ok(result)) => {
                            let task = TailV4FulltestTask {
                                task_key: result.task_key,
                                factor_name: result.factor_name,
                                stage: result.stage,
                                gap: result.gap,
                            };
                            let done_path = fulltest_done_path(&done_dir, &task);
                            write_fulltest_done(&done_path, &task)?;
                            processed_tasks += 1;
                            last_progress_at = Instant::now();
                            increment_fulltest_bucket(&mut bucket_counts, &task.stage, task.gap);
                            render_fulltest_progress(
                                processed_tasks,
                                restored_tasks,
                                total_tasks,
                                started,
                                &bucket_counts,
                            )?;
                        }
                        Ok(Err(err)) => {
                            fatal_error = Some(err);
                            stop_flag.store(true, AtomicOrdering::Relaxed);
                            break;
                        }
                        Err(RecvTimeoutError::Timeout) => {
                            if last_progress_at.elapsed() >= idle_timeout {
                                fatal_error = Some(format!(
                                    "fulltest 超过 {} 秒无进度，已强制终止 worker",
                                    idle_timeout.as_secs()
                                ));
                                stop_flag.store(true, AtomicOrdering::Relaxed);
                                break;
                            }
                        }
                        Err(RecvTimeoutError::Disconnected) => {
                            fatal_error = Some("fulltest worker 通道提前关闭".to_string());
                            stop_flag.store(true, AtomicOrdering::Relaxed);
                            break;
                        }
                    }
                }

                // 外部 supervisor: 当出现错误或长时间无进度时，强制终止所有活跃 worker 进程。
                if fatal_error.is_some() {
                    kill_tail_v4_fulltest_workers(&pid_registry);
                }

                for handle in handles {
                    let _ = handle.join();
                }
                println!();

                if let Some(err) = fatal_error {
                    return Err(err);
                }

                Ok((processed_tasks, restored_tasks, bucket_counts))
            },
        )
        .map_err(PyRuntimeError::new_err)?;

    let info = PyDict::new(py);
    info.set_item("processed_tasks", output.0)?;
    info.set_item("restored_tasks", output.1)?;
    let bucket_counts = PyDict::new(py);
    for (key, value) in output.2 {
        bucket_counts.set_item(key, value)?;
    }
    info.set_item("bucket_counts", bucket_counts)?;
    Ok(info.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ShapeBuilder};

    /// 把一组组均收益（每天完全相同）包成 compute_ssm 需要的输入形状。
    fn one_day(means: &[f64]) -> Vec<Vec<f64>> {
        means.iter().map(|&v| vec![v]).collect()
    }

    /// SSM 的五段定义自检。期望值由独立的 Python 参考实现算出
    /// （段内 = 净位移 ÷ 台阶幅度之和，五段取最小）。
    #[test]
    fn test_compute_ssm_definition() {
        // 完美等距阶梯 → 1.000（分母是浮点累加，用容差而不是精确相等）
        let got = compute_ssm(
            &one_day(&[0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0]),
            10,
        );
        assert!((got - 1.0).abs() < 1e-12, "got {}", got);
        // 完全反向的阶梯：先统一方向再算，同样 1.000（镜像不变）
        let got = compute_ssm(
            &one_day(&[72.0, 64.0, 56.0, 48.0, 40.0, 32.0, 24.0, 16.0, 8.0, 0.0]),
            10,
        );
        assert!((got - 1.0).abs() < 1e-12, "got {}", got);
        // 空头半边跨度放大 50 倍：仍然 1.000（每一段各自归一，幅度大不加分）。
        // 实测值 0.9999999999999997（分母浮点累加 203.20000000000005），所以必须用容差。
        let got = compute_ssm(
            &one_day(&[-200.0, -160.0, -120.0, -80.0, -40.0, 0.0, 0.8, 1.6, 2.4, 3.2]),
            10,
        );
        assert!((got - 1.0).abs() < 1e-12, "got {}", got);
        // 第 10 组掉到第 3 名 → 0.142857142857143
        let got = compute_ssm(
            &one_day(&[-45.0, -35.0, -25.0, -15.0, -5.0, 1.0, 3.0, 5.0, 7.0, 4.0]),
            10,
        );
        assert!((got - 0.142857142857143).abs() < 1e-12, "got {}", got);
        // 正中断裂（第 6 组低于第 5 组）→ 0.866666666666667
        let got = compute_ssm(
            &one_day(&[-45.0, -35.0, -25.0, -15.0, -5.0, -9.0, 1.0, 3.0, 5.0, 7.0]),
            10,
        );
        assert!((got - 0.866666666666667).abs() < 1e-12, "got {}", got);
        // 真实因子实例（多头最后几组掉头）→ 0.121212121212121
        let got = compute_ssm(
            &one_day(&[-40.9, -10.9, 3.4, 15.9, 23.1, 28.3, 32.5, 34.7, 36.2, 33.3]),
            10,
        );
        assert!((got - 0.121212121212121).abs() < 1e-12, "got {}", got);
        // 全平 → 0.0（分母为 0 的兜底）
        assert_eq!(compute_ssm(&one_day(&[5.0; 10]), 10), 0.0);
        // 组数不是 10 或没有日期 → NaN
        assert!(compute_ssm(&one_day(&[1.0; 10]), 5).is_nan());
        assert!(compute_ssm(&[], 10).is_nan());
        let mut empty = one_day(&[1.0; 10]);
        for col in empty.iter_mut() {
            col.clear();
        }
        assert!(compute_ssm(&empty, 10).is_nan());
        // 多日：取时间平均，结果与单日同值一致
        let mut multi = vec![Vec::new(); 10];
        for (d, col) in multi.iter_mut().enumerate() {
            col.push(d as f64 * 8.0);
            col.push(d as f64 * 8.0 + 2.0);
        }
        let got = compute_ssm(&multi, 10);
        assert!((got - 1.0).abs() < 1e-12, "got {}", got);
    }

    /// MPROB 的定义自检。期望值由独立的 Python 参考实现（同一套 A&S erf）算出。
    #[test]
    fn test_compute_mprob_definition() {
        // 阶梯 + 确定性小扰动 → 1.000：每对价差的均值远大于其 Newey-West 标准误。
        // 正反两向同值，验证值域上界与「定向后方向无关」。
        let ladder_perturbed: Vec<Vec<f64>> = (0..10)
            .map(|d| {
                (0..30)
                    .map(|t| 8.0 * d as f64 + 1e-7 * (((d * 37 + t * 17) % 11) as f64))
                    .collect()
            })
            .collect();
        let got = compute_mprob(&ladder_perturbed, 10);
        assert!((got - 1.0).abs() < 1e-12, "got {}", got);
        let mut reversed = ladder_perturbed.clone();
        reversed.reverse();
        let got = compute_mprob(&reversed, 10);
        assert!((got - 1.0).abs() < 1e-12, "got {}", got);

        // 严格恒定的阶梯 → 0.0，不是 1.0：任意两组价差逐日不变 → e_t ≡ 0 → gamma_0 = 0
        // → nw_var = 0 → 该对按「这段窗口没有可用波动」贡献 0.0（刻意的保守兜底）。
        let ladder_constant: Vec<Vec<f64>> = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0]
            .iter()
            .map(|&v| vec![v; 30])
            .collect();
        assert_eq!(compute_mprob(&ladder_constant, 10), 0.0);

        // 全平（多日）→ 0.0（同样走 nw_var = 0 的兜底）
        let flat: Vec<Vec<f64>> = (0..10).map(|_| vec![5.0; 30]).collect();
        assert_eq!(compute_mprob(&flat, 10), 0.0);

        // NaN 分支：组数不对、空输入、10 组但没有日期、只有 1 天、含 NaN、列长不齐。
        let ten = |v: f64| -> Vec<Vec<f64>> { (0..10).map(|_| vec![v]).collect() };
        assert!(compute_mprob(&ten(1.0), 5).is_nan());
        assert!(compute_mprob(&[], 10).is_nan());
        let mut empty = ten(1.0);
        for col in empty.iter_mut() {
            col.clear();
        }
        assert!(compute_mprob(&empty, 10).is_nan());
        assert!(compute_mprob(&ten(1.0), 10).is_nan()); // n = 1 < 2
        let mut with_nan = flat.clone();
        with_nan[3][7] = f64::NAN;
        assert!(compute_mprob(&with_nan, 10).is_nan());
        let mut ragged = flat.clone();
        ragged[9].pop();
        assert!(compute_mprob(&ragged, 10).is_nan());

        // 真实感用例（数据与期望值由 mprob-verifier 的独立 Python 参考实现提供）：10 组 × 120 天，
        // 由 numpy.default_rng(20240501).standard_normal((10, 120)) 加组 d 的 0.02*d 漂移生成。
        // 输入逐位取自其落盘文件 realcase_A.rs.txt；期望值 0.20324338436656386 由 A&S 7.1.26 版
        // 参考实现算出（math.erf 版是 0.20324340391338103，两者差 1.95e-8，所以 1e-9 容差只能对 A&S 版）。
        #[rustfmt::skip]
        const REALCASE_A: [[f64; 120]; 10] = [
            [
                -0.3269673054182518, -0.9743151135198309, 0.49458774095422053, 0.42498994033700543, -0.44121902020898357, -0.09967550426459997,
                -1.8036922783781175, -0.8823798545837249, 0.21658812346361556, 0.5955474978335934, -0.008974761211923903, -0.8227503016608018,
                -0.3551098248782777, 0.5253162876577246, -1.3667069647716388, 1.2047733146667488, -0.22859893036546997, -0.8050304136214149,
                -1.0394599209058064, -1.1291653373479724, 0.7629746997291696, -1.354787036952232, -0.8425118302056921, 0.10194122219697868,
                -1.2129252464974594, -1.1553775988290782, 1.6878272494896474, -0.14485180631145442, -0.373648688063378, 2.1487890977684714,
                0.9965057841476685, 0.5764315854877531, 0.8677575065279608, 0.5677175981658187, 0.13106426052178835, -2.337758760493535,
                0.6500776310130784, -0.13908204148055395, 1.521884709866391, 0.997745445926278, 1.357931026881183, -0.4024104980929273,
                0.4556712627839348, 1.3139817871765809, -0.997074517440322, -0.13498749252047593, -0.5412459601406389, 0.19888471713878428,
                -0.08134038110135416, -0.3488623888194531, 0.8324790007648996, 0.5166775303296615, 0.8265469924560145, 0.658775989576322,
                -1.31542842680326, 0.1751903387864707, -0.7019448315190892, 1.5635738101917487, -0.077655645653741, 1.1173122223445964,
                0.6671685304258267, 0.12373210871073526, -1.3461794261299154, -0.2801778320976253, 0.3954696032939856, 0.954701317440315,
                1.7428395662626062, 0.4723043900390911, 2.090307056983361, -0.3661502665466895, 0.3035933304604479, -0.35086852849010947,
                -0.6919364099966178, -1.6983977395950134, -1.893771232676127, 1.3705947001072538, -1.6054375593516916, -0.8582988625947567,
                -1.2986501975166043, 0.0037701923679036345, -1.4343579720098192, -1.363339178541941, 0.3771711222892208, 0.15024896988109498,
                -0.40702159064493165, 1.1919206808083136, 1.3915735929952004, 0.37985206530590215, -0.7478270611468948, 0.6671836243084436,
                0.4361364137796937, 2.239778437267028, 0.40383964864369054, 0.8682003574372636, -0.08882070070448538, 2.30102662084735,
                -0.010199280295198666, 1.7960165208630179, 2.4571347701848505, -0.8173568837919022, -1.2137366895333659, -0.8972281061066102,
                -0.9474323753090923, -0.1755135830894321, -1.5377376673584122, 0.6755739166532263, 0.33124296002203074, -1.1094679954914317,
                -0.35480301663644676, 2.199026054378178, 0.0965229346455339, 0.31366644292532664, 0.38376884535989714, -1.5714687966471739,
                0.9198171333296077, 1.2537598932464105, -1.9638683961655559, 0.6382435094840049, -0.9633747535105812, 0.012488561482287493,
            ],
            [
                -0.5061735440221719, -0.28729935038536203, -2.4225719174145506, -1.5058047750228105, 0.1209912132623996, -1.0985319977132435,
                0.25286169692242777, -0.8112370601563405, -0.28228501440695086, 0.7325937923968454, 1.003157797957303, -0.38510250864973056,
                2.1228667665461107, 0.02096932386184219, 0.4914823110345154, -0.4225340448383003, -0.8702762883283774, 2.502776639784766,
                -1.847069206302621, -1.2239021977355962, 0.9448633192234952, -1.6183567346793508, 0.03611298955107911, -0.5731578640915036,
                -2.1506958822132383, 1.2342040852989888, 1.4025935485359948, -0.9122772832905222, -0.2718222338941078, -0.5082715580021882,
                0.28138750897101666, -0.6802526987340537, 0.3227188142375991, 0.8839538727845517, 0.06792462437814059, -0.7408211570357225,
                0.5495427627152663, 0.6622024800131516, -0.5717758287863892, -0.574149925562729, -0.6459499322458412, -1.009516809648286,
                2.057873918536257, 0.9241325027253838, -1.9267702033737777, -0.5191765715283014, -0.3198894889877703, -1.7918152972486112,
                0.9757628670645611, 1.0146804324834056, 0.27463668745125824, 0.8535212045269781, -0.14811033320353353, 0.6303584476613625,
                -0.21003320473657494, -0.8496492598577295, 2.802134576146795, -0.8008343709207127, -2.1164525720718728, 0.9727419126190936,
                1.1097930996289755, 0.8613447930394121, 0.625496750320373, -1.1280008698493615, -1.3945976860726752, 0.8136186771308469,
                0.7387843258880259, -0.020626928432415505, 0.44128148073194967, 0.9060187125564008, -1.4654949488084503, 1.128207792055286,
                -0.2185765700064054, 1.1123012347607775, -0.7987729305375936, 0.5812737566234928, 1.554379789977898, -1.1112426123370924,
                1.0414606501364516, -1.4722292673603585, -2.553965628111452, -0.3620133191650579, 0.5741280929939243, -1.0373213996493074,
                -0.7394988391908881, -0.7055593550155157, -0.3436181130748683, 1.2651459325294672, 1.4728262686546219, -0.43986353507037246,
                -0.881686723246029, 0.48468091845989086, 1.2704168283179404, 1.5407340883653156, 0.9953244514317668, -0.7154032673614052,
                0.8196787786473241, -0.42809800017170796, -0.276349949981606, -0.09687386535259117, -2.0778332909059203, -1.6922475377338428,
                -0.08616781235768246, 0.45394266504553554, 2.4345624779822743, -1.1543387179074576, -0.12344165996248156, -0.08077901245457066,
                -0.8774669029449343, 0.1063182865801946, -0.031328635639827795, 0.23278719684114824, -1.166214254871035, -2.8777148214520327,
                1.0301683921585885, 0.4737724897041966, -0.6369633860324962, 0.6246808998622602, 0.8361195471564892, 0.3179314854055286,
            ],
            [
                0.9460039103892771, 0.07478513366607333, -1.5180246949210034, -0.469608519710545, 0.4680737863665844, -0.4221743102118408,
                -0.8972088560540828, 0.15675885883494686, -1.6633629003225805, 0.46372825957327274, -1.0104400565405331, 0.3423154600745079,
                -1.1834409223243898, -0.9786421100869829, 0.5869448265223858, -0.9509173618838892, -1.2704013718118514, 1.7694160945083406,
                -2.9329970823359264, -0.6192427354444267, 1.3901261615199554, 1.0497429748604314, 1.6285878268469003, 0.2964595933676158,
                -1.1869086404241866, -0.14778566794353998, 0.667365081678231, 0.637925146287138, 0.35551532885648035, -0.6914242165688702,
                0.14632787136367167, -1.6438878442869975, 1.0611653881422747, -0.45462514183676034, 0.3403160940267365, -0.9303537019434819,
                0.4323947373700432, 0.9732104077371919, -0.6790302850526237, -1.738122908863935, -0.9424451918152272, -0.39744003263407446,
                -0.40195926334240367, 0.9819878672415077, -0.02008668696975787, 0.09701742149411166, 1.0631628003134947, 0.2958329878268448,
                -1.764791821622685, -0.19100020844002005, 1.4457373761343404, 0.3244505856305899, -1.2894813386569046, 2.0326223162368455,
                -1.14420095420147, 0.21392869815542592, 0.4221197283010353, -0.740584480454702, -0.2628512640217825, 1.7268269059934562,
                0.09815367523920382, 1.8183100832800179, -0.2255828969596497, -0.7278127881134073, 1.2653204253446928, -1.4052001909832466,
                0.21295126440975456, 0.930719398047528, -0.9883017285683955, 0.7329460450609562, -2.502322238755081, -0.6458580229096368,
                0.07792840350567883, -1.8446510291973355, 0.09849534647555888, 0.31442127540482595, -0.22148363807562657, -0.9258763802316695,
                -1.0630858729039887, 1.6904901085742372, 1.1444598710417822, -0.4630447633235592, -0.3526978042766477, 0.9095179488862859,
                -0.5606307408798664, -0.006540811981807684, 0.08065034168455801, -0.019965641824682752, -0.6358496345673436, 1.7576951740018778,
                1.5573530116175331, 1.4412007052181577, 1.4574939537795029, 0.38618254137471253, 0.31980900220996444, -1.1604126138608983,
                1.1825547905373421, -1.2304798325056592, -0.07131054801360787, -0.11531505922149834, 1.4123699831776837, 0.33834164566773095,
                0.5341643588198808, -0.6411780708572197, -0.9391775866397112, -0.06269190183517001, 1.2249402161089646, 0.43637337979255786,
                1.5617628306266698, -0.33606107455097484, -0.27846672564926106, 1.8270982933590754, -0.8655643121765033, -0.672610182515622,
                0.3684500894298271, 0.3332562545973332, -1.2261113329034463, 0.427317804697133, -2.0668550590505292, -1.0899042207317964,
            ],
            [
                -0.488735339778898, -1.6115755950971533, 0.35842135186673124, -0.7194419105183654, 0.6020557945334726, -0.659947043555746,
                -1.7538518591263685, -0.24167015953895826, 0.22220096068067194, 1.3495488046596436, 1.8575972515783306, -1.0363114873057135,
                -0.4980955630929255, -0.35963821072103136, -1.527356636898633, -1.4476012587110878, 0.8289403196329825, 0.025917194845543425,
                2.267861790803735, 0.6390528943304123, -0.49013490440662594, -1.203313650573267, 0.6043154215878757, -1.0748977722065014,
                -0.8373591119212271, -0.5548337204070946, 0.5949698229101712, -0.9472017550157175, 0.50685636709768, 0.7233749362079938,
                -0.49919836156113867, -1.3180141335477336, -0.6976949919515112, 1.6470478982414811, 0.6779446630803037, 2.1032930449237357,
                0.6286239364315824, 0.20627350680222614, -0.5718699961613649, 0.201809216749943, 0.7690081359951335, 1.6055424404701997,
                0.17363482959416665, -0.14335444962641333, 0.11208295800730407, -0.25517813602921485, 0.5763645097923535, 1.9905984407342807,
                -0.2909365015077812, -0.36739797198174956, 0.8997495651207865, 0.24349531087408616, -0.25755249736691954, 2.4982379897225924,
                0.9543988821290887, -1.90259376799431, -0.16220090273166182, 0.7371562427807383, -0.9250427322867498, 1.265086760265672,
                0.13194418428599758, 0.5530841576014509, -0.6119256538790243, 0.6712654976962187, -1.0916786915356438, 0.7334841124846316,
                0.1041958421986392, 0.05913818410925724, 1.209109289702292, -1.0476107411354871, -1.7846697602577957, 0.88232115441754,
                -0.2705371113486818, 0.5566946594791646, 0.11727379897463902, 0.4552310315292946, -1.9595045260325925, -0.1737588939529929,
                -2.2589730411374545, -0.3966994175700807, 0.012206370852695506, 0.40160187294145017, 0.5733592213395438, 0.8124349244009523,
                0.9966202907175201, -0.4027654467635453, 1.1134883781910536, 1.1783458856144655, 1.0987321216332655, 1.1162015219530788,
                -1.2629927321235936, -0.4128410922771567, 0.1951595318379315, -0.3841680999045546, -0.8395915783467394, 0.2603484575439779,
                -0.7455827651885305, -0.5843231416421788, -1.0544257576414071, 1.1753027708742865, -0.5371452784810089, 0.7997011158260354,
                -1.0201922756812676, 0.493500510397515, 0.5013141077931349, -1.2626979526409814, 0.4429910522590519, 0.42285183707002993,
                0.9862118307890044, -1.548681905778713, -0.04163371900862918, 1.607988794343037, 1.9258093245997128, 0.22971216927931074,
                1.3772361197931733, -0.8012760399809544, -0.2025903857076326, 0.20255903191778837, 0.08361998819265429, 0.3711303676419777,
            ],
            [
                0.4382039377445785, 1.0930582327812404, -0.18263798672087989, 0.7297141761334092, 0.33356484557491894, -1.4618685838417231,
                0.5703496217323041, -0.5201039697981743, 1.4177563764900778, -0.7717926222111179, 1.1487031135958634, 1.661191489643667,
                1.6112748118497355, -0.7098354212288494, 1.092256974165952, -0.7800685043575167, 0.7586295834083455, 0.4810474253845395,
                0.49792160909562555, -0.4647727220578795, 1.1717592221949058, 1.8004150300258996, 0.4285383592144285, 1.3848193530943471,
                0.049837447850851736, -1.0400117395124426, -0.19069309446459404, 0.7619298251352241, 0.6953834551619268, 1.0820889970571752,
                0.5305919462785639, -0.6186886268860653, -1.102441662480493, 1.5998656248620844, 0.38073368652599904, -2.1214390222957067,
                0.9960099749088464, 1.4480288439569213, 0.5114643816070381, -0.32977325414166025, -0.6941500611750364, -0.930125920063558,
                0.12965264105561913, -2.397132520779201, -0.939673418304629, -2.1716609964729625, -0.8156389877511965, -0.17454008891282735,
                1.7040316762976802, 0.4953433366904164, 0.9509038011586075, 0.8723013339544886, 1.032290326581847, -0.637026899458879,
                1.7411903262512483, 0.8632468791337747, -0.36563268944108984, -1.4153266153560595, -0.6134921021532018, 1.0100311441078347,
                -0.040206622834085756, -0.7472326806077563, 0.8712729674336469, -0.9633923180363203, -1.4651851420596502, 0.6599930279765357,
                0.7484172139587982, 0.6205952034265009, 0.6433818691762351, 0.23761921430256244, 0.7076817115903359, -0.25407865885057646,
                -0.13717300284037776, -0.5956532508236985, -0.7744207985620654, 0.8277547427333429, 0.31632897668297055, 0.1367475724797428,
                1.5128572275605918, -1.3468100394190885, 0.6003975925635681, 0.27754020315655525, 0.2510413402886767, -0.25472694690587094,
                0.18525662285747796, 0.7870082665485392, 0.5344561367580933, -1.833911147562546, -1.4618788198385693, 0.21300863069920273,
                0.15764884882029545, -1.5436981924323514, -0.4572070892208892, -0.37459934775326736, -0.4733577298615251, -0.3983102036181813,
                -0.04761113295569365, -1.1397986856077431, 0.8252389276990634, 0.7749927138156992, 0.46136145212960994, -0.2685025481931801,
                0.39320895457203736, 2.1603529194022757, -0.41559567660521385, -0.3161857360870219, -1.4067330291928293, 0.06713641461819558,
                0.7893666534306689, 2.158832472828033, -0.03594556345319054, 0.21346325512308045, -1.1628549852241148, -0.4044931795181153,
                0.14458325293668609, 1.0947645583473304, 0.07014712375380236, -0.15798113527227386, 1.0834182386866682, -0.6823858385551314,
            ],
            [
                0.08910470046267774, 0.7076533236058394, -0.7424700227950826, 0.09986949197260875, -0.04797738543895916, 1.6140768115487851,
                -0.7377344183875808, -0.5381109664694987, -0.6463457587467811, 1.5638320831675618, -0.10092678537832714, -0.32232023525144726,
                1.3604802938528688, -1.4022443556624875, 0.9622203173459323, -0.41880751748324097, -0.6684469230832871, -1.3552838714268762,
                -1.3062140078895097, 0.9827035380762443, -0.7309351005537618, 1.2804528307914955, -1.103823261704575, 0.26325299697953386,
                -0.922194324608027, 0.8917829081427278, 1.7379456898645138, -1.0706629591881043, 1.6215782662283016, 1.0357799079244006,
                0.07981558226304035, 1.202345082460899, 0.2263891539364807, -0.33641685993903625, 0.4142583546755725, 0.4947551151855779,
                -0.13324745788242295, -0.705097849344255, 2.4233546136725925, 0.19241475415844628, -1.8098395630994963, -2.202022878548712,
                1.9199824138881048, 0.5152691252847107, -0.07886960890443612, 0.3997130947862412, 1.5693510483517186, 0.8039042692361752,
                -0.8453747568000708, -0.8260710520397114, 0.03559024011665114, -1.7445984376489085, -0.3392267789670518, -2.9076260446752094,
                -0.6464760865244253, -0.2873168573221848, 1.036193587777824, 1.3544655684811555, -0.5267269457748529, -1.047495987872548,
                -1.0022115548842077, -0.38413914851429787, -0.39992761085277406, 1.5043283228375006, -0.681868542205505, 0.5477045630134844,
                -1.4555747933927896, 1.201739628802785, 0.8880513533740797, -0.09905130977311774, -1.113108024954639, -0.2584879485488699,
                1.476239845937511, -1.4808420385203802, -0.22240211917025696, -0.09532184847522474, 0.23050913975117, 0.010139001517426346,
                -0.0542761338766109, -0.7244086034144713, 1.7549783928095801, -2.066301831762991, 1.61909307761384, -1.1263464530378138,
                -0.17196513472158811, -0.8828347926635908, -0.317623144865549, -1.8192087006812452, -0.8895588436298343, -0.6713472458590135,
                0.6572151233866247, -0.6628877503698212, -0.36140678040070906, 0.17865884157702988, -0.10167249352407359, -1.216412581043622,
                -0.9126347149420132, 1.8396931020260323, 0.15240598354661886, 0.4361480624010242, -1.6855442413077861, 0.9561187095016103,
                -0.28264567379356054, -0.5270637655265777, 0.9914808228479411, -0.9453986467599821, -0.03397232492714719, 1.5149258051195296,
                -1.033344215231703, -0.9158578399253249, -1.4089351615837544, -0.3458564958149387, -0.9913366387418278, -0.45437057874821074,
                -0.1096295551138535, -1.169092954574302, -0.6484189314104153, 0.4986339371497487, 1.089130110766389, 0.30334594332228837,
            ],
            [
                0.9839245962149531, 0.11918813210015149, -1.1745855157262435, -1.7596838531475996, 0.6007819449698619, 1.689434866311053,
                -1.9464759112002135, 0.3397525101576229, 0.9801353858542828, 0.07551467619110303, -0.5482392791553298, -0.24591589615525306,
                1.4991252718045458, -0.7193128713186178, 0.48596666614110956, -1.2254428827480992, 1.8606417325423035, -0.9364459991635764,
                1.474513645617122, -0.7848637516998094, -1.4929498884782868, 0.5945909101403788, -1.454576639301762, 0.5096697602485738,
                0.26570605836254835, 0.5346526215545592, -0.09448804799323843, -0.8424273618539794, -0.8402793958389148, -0.2521145254999503,
                -0.9387475938136948, 0.8819121184877737, -1.8735736554317328, -0.7925539235568403, 0.7937836445755977, -0.2720720438367878,
                1.6008013008275208, -0.3164189529509294, 0.3731920463792998, 1.4136586725139897, -0.09845620493247476, -0.08029630905858726,
                -1.3591638147629261, 1.2603696903885493, 1.7268456425255758, -0.17463353177511293, -3.5615057308525793, -2.6601297279149567,
                -1.4647009260509707, -0.265398903218689, -1.4759508505103418, -0.24851801716517286, 0.40692652868898627, 1.8903288821915032,
                -0.08426147971928016, 0.7907508932382814, 0.5464815820689689, -0.6960324402444045, -1.0296367358076308, -1.1011463648058268,
                0.3480562257544043, 0.4776982896998778, 2.1014142735939556, -0.7138807119602091, -0.27413538308452234, -0.9859175903281198,
                0.6809628454112785, 2.7159226041373654, 0.6104381927585241, -1.2980317054335302, 0.999788850820766, 0.3773193273572951,
                0.6470383011250942, 1.944503845101453, -0.4385429807630655, -2.279169161607187, -1.2555408080149228, 0.7937848231744932,
                -0.9730168105704914, -0.20690212315084938, 0.11965511921279862, 0.5568514245204219, -0.8320906510343969, 0.4140213483929676,
                -0.909558297831048, 0.5290650365545955, -0.6322895513318246, 0.042810498886496906, 0.9891207763055011, -1.4421518426511906,
                -2.8504925591282397, -1.0597720831453166, -1.5370805917887091, 2.0085448844113816, -0.22102432240270364, 1.562167692687464,
                2.3102142550726636, 0.6804428331798424, -1.816475586347492, -0.3394188913978779, 0.7300838092087056, 0.1374008115208389,
                1.1301287861214866, 1.4531653035955814, 0.4269008656149899, 0.8318933012139135, 2.3659544401729495, -0.3640347752380025,
                -0.8902561364147007, -0.7099589827322224, -0.27020809828573733, 0.9019656698055745, 1.465888230643837, 0.12033875975608528,
                0.09869047502158701, 2.0480549175434666, 1.394674578160331, -0.9056503628087521, -0.1783026463403556, -1.0646868781047854,
            ],
            [
                -0.8499159885256395, -1.3642710072456667, 0.20604653691433777, 1.4135564367079887, 0.534459901431841, 0.092579569224677,
                -0.6577967204654228, 0.8554938332309319, 0.24677734475436466, 0.6525090390954992, -1.133900350187159, 0.9452431536682749,
                0.3977739382132172, 2.627383033936437, -1.2018854376498602, -0.7548218352975584, -0.8909788598151466, -0.45848397122878737,
                -1.274508416427159, -0.6314615450543763, 0.971945090873947, -0.7732499193399954, -0.035428709630479394, -1.390542615333373,
                -0.015441779928621197, 0.021700605863125347, -0.03241479285652782, -1.7427915315277196, 1.2261755788192876, 1.5395478385083927,
                0.14384394461273728, -0.8495457752305451, 0.4195655340981268, -0.6398787303444092, 0.467329501342649, 0.6811020340577227,
                -0.7589451221295302, -0.8779507865820745, -1.0117811316339695, 0.9382186084201986, 0.615002087419825, -0.5230389792255075,
                -0.3172316832341296, 0.6783297836071147, 0.6651014228215039, 2.666737808143118, 1.6099613392677146, 1.4877366871279478,
                1.2915841618426724, -1.2475158376068483, -1.0963708076426997, -0.18316900840399586, -0.012122216348514875, 0.04779605951579885,
                -0.5095132021289217, 0.22993958443067686, -0.22670260300993134, 0.14383006984624083, 0.12418285472743834, 0.5791342864656248,
                1.1806547181909433, 0.44714516560678785, 1.0358172940506494, -1.0699362763939484, 0.769575081558355, -0.831004555147157,
                -0.6802889260298229, 1.7754745714531688, 0.7429100835795929, 0.2887602873787124, 2.0957148508181533, 1.257491665358692,
                -0.14706593553271058, 2.572256113396677, -0.6682652300473383, -1.0695119994434852, 1.7964660192082667, -0.6682165373526001,
                -1.096342937953036, 1.0740077418954477, 1.1464435822638679, -0.03197312539431468, 0.7821022969440448, 0.06443982548054848,
                -0.3003947276670705, 0.21961231189859043, -2.690836474070076, -0.32044365984766204, -0.6261802888724567, -0.28044339844451877,
                0.22804350948618182, 0.6012644831131841, 0.4511701710833965, 0.8515790837609375, 0.1858299201313875, 0.25533614488009115,
                -0.1524349669947091, -1.620465390790387, 0.5347019206542367, -0.7622242350117765, -1.5824094582059147, 0.5905569714929193,
                -1.9056865515161268, -1.505091214718933, -0.04151086972528337, -0.5878107699776897, -0.09532220117413259, 0.131152653000742,
                1.0137144699701865, 0.5115871793426381, 1.2079988147334562, 1.0665180069505675, -0.08328040935011269, 0.16434030793468535,
                1.4552172281373332, 0.776456848969006, 1.4230471007341996, -0.8448941517699533, -2.0237191823544576, -0.2353252986320739,
            ],
            [
                1.2554837689363765, 0.157292098845215, 1.6836639740283115, 0.3976010265502773, -0.5139431706973612, -1.2076262204289059,
                -0.35221354095936985, -0.8484226183332092, -1.215617299341103, -1.8921531637423479, 0.20466078876224772, -0.6320503310008859,
                1.050774563244058, 0.49865096220061544, 1.9942281694119612, -1.7591276295125704, -1.4841602424118023, 0.6699384412395829,
                -0.6237880697118025, 1.7185804879710078, -0.3160862866220221, 0.36667127945200695, 0.6665227443079975, -0.7421802002574677,
                1.4500077302938825, 1.2011387388991195, -0.700699024618414, -0.24140685694895367, 0.07846925831377242, -0.5336397167057819,
                0.18461169240563904, -0.2706313514797478, -0.22398389914137237, 1.63268553599132, 0.23585220552375083, 0.7312107209788126,
                -0.19829972878705052, -2.2674594780686608, -1.3590932585818938, -0.3439323107092469, -1.349785816394389, 0.3520423460268348,
                -0.17724748624334333, 1.2454312219547836, 0.2296571414433104, 0.9472440991306894, -2.1244035528852296, 2.3114265707958945,
                -0.27939101821502954, -0.4341928900269596, 1.145576254886822, 0.681432588792488, -1.7884256134409429, 0.5605228091638625,
                -0.04595710114698781, 2.9022458854368782, -1.84966240622436, -0.15482284232990842, -0.48693960288216165, 0.22254604277248066,
                -0.673689123011425, -1.8524079778206464, -0.8472653598756049, 2.893694733547345, 0.12126627594672823, 0.07494043628018225,
                0.9857540215458647, -0.18828439162285923, 0.14406054241147281, -0.24009892624076848, -0.9450088956065815, 0.8369888469039252,
                2.174896853753119, -0.20777404311432215, 1.4139216205738971, -0.6494379400384065, 0.7821254615517167, 1.2369405282118944,
                -0.19371398543209775, 2.407916574172478, 0.09531836493243075, 0.20273022496231546, -0.06101338918119589, 0.3987677844577132,
                -0.3797566485715199, -0.04859848074219014, 1.4022996251508824, 1.0423054734698, -0.24799523390025088, 2.317345179655827,
                -1.1748809219622998, 0.4055208531043172, 0.9421174582612082, -0.2737700858811595, -0.5387080740682298, -0.2486242900623802,
                -1.114879159536882, 0.8961331888885958, -1.3969040033740208, 0.40995542374000626, -0.8332862530640559, 1.9985792640019984,
                -0.668625051688448, 0.14409703171491092, 1.443436947961339, 0.19306816446631825, 0.025231146418829398, -1.4337371023144725,
                1.6399041872360152, 0.47041411371505837, -1.0351450503256325, 0.1880784246128703, 0.3943154867812839, -0.18002638399967377,
                -2.051471965248319, 0.4139026852802241, -1.2684627874410597, -0.632891882082498, 1.8708777938220187, -0.05333361441045886,
            ],
            [
                -0.4361867016924284, 0.627554744788359, 0.18942965171201942, 0.06205224594785674, 0.33986930073695404, 0.538059799118259,
                0.5241394134178964, 0.12472449986295246, -1.9219183094154786, 0.8603010987126904, -0.49508758339829756, -0.17567283640169057,
                1.3903626674664495, -2.459284694267654, 1.244202545279143, -1.183253548915499, -2.069487246706529, 0.08784273003211787,
                -0.8394447128534812, 0.8624360087632053, 1.8196726489352868, 2.1270389934787906, -0.6005873116693043, 0.8603978664955998,
                0.7838609015136098, 0.6922140317021479, -0.005802723649376534, -0.2247592656611639, -0.8186253014960849, 0.2609202314944603,
                0.17133488278960537, 1.239912834903095, -0.3832391017807199, -0.2951945306636347, -0.36548960885407006, 1.2012061478522968,
                -1.0884439795556116, -0.41518503566910053, -0.005172441686977947, 0.9029650233621993, -0.3045992860961977, -0.844063270527454,
                1.122009637972884, 1.1814990795231215, 0.41268305965956253, -0.4004367152930655, 0.20065084634541655, 1.5840973558995355,
                -0.4872100849946434, 0.835185517480457, -0.4110838118130134, 0.7488998068332848, 1.8729953510221369, -0.20468169381835738,
                -0.1936427972724567, 1.1538455318001462, -0.17646671608313208, 0.07455221591399863, -0.4607315835977895, 0.9287224464720183,
                -0.08078073673638664, -0.3992259751984851, 0.9223981021046332, 1.5570106372319306, -0.5746704255160888, -0.015446992538628523,
                -0.11147828596321319, -0.9343526421102857, -0.07135224841110416, -0.9096941161689083, 0.9168144953413371, 0.7495614311781245,
                -0.4775221955662537, -0.9670851212180771, -2.778477678057662, 0.13592572379063092, 0.004078571954045068, -0.4316576149078219,
                -1.1940660786306903, 0.04466186458809429, -1.2996121003700771, -1.2177391273726426, -0.900317460589038, -0.7351106004025199,
                -1.8704256547449216, 0.605155558141822, 1.0753277513602517, -1.3074148141671265, -1.1457122412768208, 0.1477393835949203,
                0.43841622574871636, -0.7651710992569858, 0.23243539629199667, -0.3999676516737514, 1.0044405601210034, 0.6535410748204704,
                1.5840713467297294, 1.6752368868713108, -0.5727860804805585, 0.700633586153288, 1.6377792288698667, 1.4934480263654086,
                -0.11110319991034523, -0.02499422403272311, 0.9775444253087238, 0.26803455073714116, 0.3413436583992662, 0.4151610145073482,
                -0.17490425491931755, 0.14568984683593217, 1.1548971699999293, 1.323589424026647, -0.13156590993532408, 1.1645237600565062,
                -1.7949256512894456, 0.7510780421548449, 2.3963282234177217, 0.7888463462062898, -0.49847506696439775, -0.8603181058952714,
            ],
        ];
        let realistic: Vec<Vec<f64>> = REALCASE_A.iter().map(|r| r.to_vec()).collect();
        let got = compute_mprob(&realistic, 10);
        assert!((got - 0.20324338436656386).abs() < 1e-9, "got {}", got);
    }

    #[test]
    fn open_symbol_counts_support_fortran_order_rows() {
        let restrict =
            Array2::from_shape_vec((2, 3).f(), vec![0.0_f32, 1.0, 0.0, 0.0, f32::NAN, 1.0])
                .unwrap();
        assert!(!restrict.row(0).is_standard_layout());

        assert_eq!(precompute_open_symbol_counts(&restrict.view()), vec![2, 1]);
    }

    fn make_test_data() -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let raw = Array2::from_shape_vec(
            (3, 5),
            vec![
                1.0_f32,
                2.0,
                f32::NAN,
                4.0,
                5.0,
                6.0,
                f32::NAN,
                f32::NAN,
                9.0,
                10.0,
                f32::NAN,
                12.0,
                13.0,
                14.0,
                f32::NAN,
            ],
        )
        .unwrap();
        let restrict = Array2::from_shape_vec(
            (3, 5),
            vec![
                0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let ret = Array2::from_shape_vec(
            (3, 5),
            vec![
                0.01_f32,
                0.02,
                0.03,
                0.04,
                0.05,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.01,
                0.02,
                f32::NAN,
                0.04,
                0.05,
            ],
        )
        .unwrap();
        (raw, restrict, ret)
    }

    fn assert_backtest_result_eq(a: &LegacyBacktestResult, b: &LegacyBacktestResult) {
        for (x, y) in a.summary.iter().zip(b.summary.iter()) {
            if x.is_nan() {
                assert!(y.is_nan());
            } else {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
        assert_eq!(a.ic_dates, b.ic_dates);
        assert_eq!(a.ic_values.len(), b.ic_values.len());
        for (x, y) in a.ic_values.iter().zip(b.ic_values.iter()) {
            if x.is_nan() {
                assert!(y.is_nan());
            } else {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }

    #[test]
    fn single_slot_backtest_matches_batch_exact() {
        let (t, n, f) = (14usize, 20usize, 4usize);
        let mut seed = 0x0123_4567_89ab_cdefu64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 as f32) / 500.0 - 1000.0
        };
        let mut block = Array3::<f32>::from_elem((t, n, f), f32::NAN);
        for fi in 0..f {
            for i in 0..t {
                for j in 0..n {
                    if (i * n + j + fi) % 6 != 0 {
                        block[[i, j, fi]] = next();
                    }
                }
            }
        }
        let mut restrict = Array2::<f32>::from_elem((t, n), 0.0);
        for i in 0..t {
            for j in 0..n {
                if (i * n + j) % 8 == 0 {
                    restrict[[i, j]] = 1.0;
                }
            }
        }
        let mut ret_gap1 = Array2::<f32>::from_elem((t, n), f32::NAN);
        let mut ret_sum_gap1 = Array2::<f32>::from_elem((t, n), f32::NAN);
        let mut ret_gap5 = Array2::<f32>::from_elem((t, n), f32::NAN);
        let mut ret_sum_gap5 = Array2::<f32>::from_elem((t, n), f32::NAN);
        for i in 0..t {
            for j in 0..n {
                ret_gap1[[i, j]] = next();
                ret_sum_gap1[[i, j]] = next();
                ret_gap5[[i, j]] = next();
                ret_sum_gap5[[i, j]] = next();
            }
        }
        // index 是「指数日收益」序列，长度必须等于日期数（原测试误用股票数 n，
        // 会在形状校验处直接 Err，逐位对账根本没跑起来）。
        let index = Array1::<f32>::from_shape_fn(t, |_| next());
        let dates: Vec<i32> = (0..t as i32).map(|d| 20200101 + d).collect();
        let selected_slots = vec![0usize, 1, 2, 3];
        let (batch_g1, batch_g5) = legacy_backtest_gap1_gap5_selected_slots_f32(
            block.view(),
            ret_gap1.view(),
            ret_sum_gap1.view(),
            ret_gap5.view(),
            ret_sum_gap5.view(),
            restrict.view(),
            index.view(),
            &dates,
            dates[2],
            &selected_slots,
            10,
            false,
        )
        .unwrap();
        let open_symbol_counts = precompute_open_symbol_counts(&restrict.view());
        for (local, &slot_idx) in selected_slots.iter().enumerate() {
            let (g1, g5) = legacy_backtest_gap1_gap5_single_slot(
                block.slice(s![.., .., slot_idx]),
                ret_gap1.view(),
                ret_sum_gap1.view(),
                ret_gap5.view(),
                ret_sum_gap5.view(),
                restrict.view(),
                index.view(),
                &dates,
                dates[2],
                10,
                &open_symbol_counts,
                false,
            );
            assert_backtest_result_eq(&batch_g1[local], &g1);
            assert_backtest_result_eq(&batch_g5[local], &g5);
        }
    }

    #[test]
    fn iconly_single_slot_keeps_ic_exact() {
        // ic_only 模式必须与正常模式产出完全一致的 IC 序列 / IC_mean / IR / ratio_mean，
        // 收益字段（十分组/多空组合）应为 0.0。
        let (t, n) = (30usize, 40usize);
        let mut seed = 0xfeed_beef_cafe_1234u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 as f32) / 300.0 - 200.0
        };
        let mut block3 = Array3::<f32>::from_elem((t, n, 1), f32::NAN);
        for i in 0..t {
            for j in 0..n {
                if (i * n + j) % 5 != 0 {
                    block3[[i, j, 0]] = next();
                }
            }
        }
        let mut restrict = Array2::<f32>::from_elem((t, n), 0.0);
        for i in 0..t {
            for j in 0..n {
                if (i * n + j) % 7 == 0 {
                    restrict[[i, j]] = 1.0;
                }
            }
        }
        let mut ret_gap1 = Array2::<f32>::from_elem((t, n), f32::NAN);
        let mut ret_sum_gap1 = Array2::<f32>::from_elem((t, n), f32::NAN);
        let mut ret_gap5 = Array2::<f32>::from_elem((t, n), f32::NAN);
        let mut ret_sum_gap5 = Array2::<f32>::from_elem((t, n), f32::NAN);
        for i in 0..t {
            for j in 0..n {
                ret_gap1[[i, j]] = next();
                ret_sum_gap1[[i, j]] = next();
                ret_gap5[[i, j]] = next();
                ret_sum_gap5[[i, j]] = next();
            }
        }
        let index = Array1::<f32>::from_shape_fn(n, |_| next());
        let dates: Vec<i32> = (0..t as i32).map(|d| 20200101 + d).collect();
        let open_symbol_counts = precompute_open_symbol_counts(&restrict.view());
        let (full_g1, full_g5) = legacy_backtest_gap1_gap5_single_slot(
            block3.slice(s![.., .., 0]),
            ret_gap1.view(),
            ret_sum_gap1.view(),
            ret_gap5.view(),
            ret_sum_gap5.view(),
            restrict.view(),
            index.view(),
            &dates,
            dates[2],
            10,
            &open_symbol_counts,
            false,
        );
        let (ic_g1, ic_g5) = legacy_backtest_gap1_gap5_single_slot(
            block3.slice(s![.., .., 0]),
            ret_gap1.view(),
            ret_sum_gap1.view(),
            ret_gap5.view(),
            ret_sum_gap5.view(),
            restrict.view(),
            index.view(),
            &dates,
            dates[2],
            10,
            &open_symbol_counts,
            true,
        );
        // IC 序列完全一致
        assert_eq!(full_g1.ic_dates, ic_g1.ic_dates);
        assert_eq!(full_g5.ic_dates, ic_g5.ic_dates);
        for (x, y) in full_g1.ic_values.iter().zip(ic_g1.ic_values.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for (x, y) in full_g5.ic_values.iter().zip(ic_g5.ic_values.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        // IC_mean / IR / date_size / ratio_mean 一致；BRC/BRC_S/BRC_L（12/13/14）
        // 不依赖十分组收益，两种模式必须逐位相同。
        for idx in [0usize, 1, 5, 6, 12, 13, 14] {
            assert_eq!(
                full_g1.summary[idx].to_bits(),
                ic_g1.summary[idx].to_bits(),
                "gap1 summary[{}]",
                idx
            );
            assert_eq!(
                full_g5.summary[idx].to_bits(),
                ic_g5.summary[idx].to_bits(),
                "gap5 summary[{}]",
                idx
            );
        }
        // 收益相关字段 ic_only 为 0.0
        for idx in [2usize, 3, 4, 7, 8, 9] {
            assert_eq!(ic_g1.summary[idx], 0.0);
            assert_eq!(ic_g5.summary[idx], 0.0);
        }
    }

    #[test]
    fn test_compute_raw_cover_rate_all_dates_low_threshold() {
        let (raw, restrict, ret) = make_test_data();
        let result = compute_raw_cover_rate(&raw.view(), &restrict.view(), &ret.view(), 0);
        // day0: free=5, valid(raw!=NaN & ret.finite & free)=4 (/val0=NaN → exclude)
        //       ratio=4/5=0.8
        // day1: free=3 (restrict[1,3]=1, [1,4]=1), valid=1 (/val1=NaN /val2=NaN → exclude)
        //       ratio=1/3
        // day2: free=4 (restrict[2,2]=1), valid=3 (/val0=NaN, /val4=NaN, ret[2,2]=NaN)
        //       ratio=3/4=0.75
        // result = (0.8 + 1/3 + 0.75) / 3
        let expected = (0.8_f64 + 1.0 / 3.0 + 0.75) / 3.0;
        assert!(
            (result - expected).abs() < 0.0001,
            "{} vs {}",
            result,
            expected
        );
    }

    #[test]
    fn test_compute_raw_cover_rate_min_stocks_2() {
        let (raw, restrict, ret) = make_test_data();
        let result = compute_raw_cover_rate(&raw.view(), &restrict.view(), &ret.view(), 2);
        // day1: valid=1 < 2 → excluded
        // day0: 4/5=0.8, day2: 3/4=0.75
        let expected = (0.8_f64 + 0.75) / 2.0;
        assert!(
            (result - expected).abs() < 0.0001,
            "{} vs {}",
            result,
            expected
        );
    }

    #[test]
    fn test_compute_raw_cover_rate_min_stocks_5() {
        let (raw, restrict, ret) = make_test_data();
        let result = compute_raw_cover_rate(&raw.view(), &restrict.view(), &ret.view(), 5);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_compute_raw_cover_rate_against_python_small() {
        // 用Python脚本生成的数据: restrict (22,5419), ret_gap1 (22,5419)
        // 构造一个确定性的小因子测试
        let mut raw_data = vec![0.0_f32; 22 * 5419];
        let n_dates = 22_usize;
        let n_stocks = 5419_usize;
        for t in 0..n_dates {
            for s in 0..n_stocks {
                if s % 2 == 0 && t % 3 == 0 {
                    raw_data[t * n_stocks + s] = f32::NAN;
                } else {
                    raw_data[t * n_stocks + s] = ((s * t) % 100) as f32;
                }
            }
        }
        let raw = Array2::from_shape_vec((n_dates, n_stocks), raw_data).unwrap();
        // 全0的restrict (所有股票可交易)
        let restrict = Array2::from_elem((n_dates, n_stocks), 0.0_f32);
        // 全1的ret (所有收益有效)
        let ret = Array2::from_elem((n_dates, n_stocks), 0.01_f32);
        let result = compute_raw_cover_rate(&raw.view(), &restrict.view(), &ret.view(), 10);
        // 每3天有1天 (t%3==0) 偶数股票为NaN, 其他天全部有效
        // nan股票 = ceil(5419/2) = 2710, 有效股票 = 2709
        // ratio_some_nan = 2709/5419, ratio_all_valid = 1.0
        // dates with t%3==0: t=0,3,6,9,12,15,18,21 (8 days)
        // dates with t%3!=0: 22-8 = 14 days
        let expected = (8.0 * (2709.0 / 5419.0) + 14.0 * 1.0) / 22.0;
        assert!(
            (result - expected).abs() < 0.001,
            "result={:.6}, expected={:.6}",
            result,
            expected
        );
    }

    #[test]
    fn test_fill_missing_rank_only_fills_normal_restrict() {
        // 25 天 x 4 股，restrict==0(可交易) 才填：
        // 股0: restrict 全 0, 每天有值 → 全程有限
        // 股1: restrict 全 0, 第0天有值之后全 NaN → 缺口全填（可交易日缺数据=缺失）
        // 股2: restrict 全 1(停牌/涨跌停), 全 NaN → 永不填（不适用）
        // 股3: restrict 前10天全 NaN(未上市)、第10天起 0 → 之前不填(不适用), 之后有限
        let n_dates = 25usize;
        let n_stocks = 4usize;
        let mut raw = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
        let mut restrict = Array2::<f32>::from_elem((n_dates, n_stocks), f32::NAN);
        for t in 0..n_dates {
            raw[[t, 0]] = 1.0; // 股0 恒有值
            restrict[[t, 0]] = 0.0;
            restrict[[t, 1]] = 0.0; // 股1 恒可交易
            restrict[[t, 2]] = 1.0; // 股2 恒不可交易（停牌/涨跌停）
            restrict[[t, 3]] = if t >= 10 { 0.0 } else { f32::NAN }; // 股3 第10天上市
        }
        raw[[0, 1]] = 2.0; // 股1 仅第0天有值, 之后缺口但可交易 → 应填
        for t in 10..n_dates {
            raw[[t, 3]] = 3.0; // 股3 第10天起有值
        }
        let ranked = rank_and_fill_missing_cross_sectional_median(&raw, &restrict);
        // 股0 全程有限；股1 可交易日缺口全填（不再有"20天窗口"限制）
        for t in 0..n_dates {
            assert!(ranked[[t, 0]].is_finite());
            assert!(ranked[[t, 1]].is_finite(), "可交易日缺口应填充");
            assert!(ranked[[t, 2]].is_nan(), "不可交易(restrict=1)不应填充");
        }
        // 股3: 未上市期(restrict NaN)不填, 上市后(restrict=0)有限
        for t in 0..10 {
            assert!(ranked[[t, 3]].is_nan(), "未上市期间不应填充");
        }
        for t in 10..n_dates {
            assert!(ranked[[t, 3]].is_finite());
        }
        // 中位 rank 验证: 第1天有效值=股0(1.0) 1个 → med=(1+1)/2=1.0 → 股1缺口填1.0
        assert_eq!(ranked[[1, 1]], 1.0);
    }

    #[test]
    fn test_fill_missing_rank_all_outside_universe() {
        // 全 NaN 矩阵（从未有值/未上市）：restrict 全 1（不可交易）与全 NaN（未上市）
        // 两种口径，填充后必须仍全 NaN
        let raw = Array2::<f32>::from_elem((2, 3), f32::NAN);
        let restrict_all_one = Array2::<f32>::from_elem((2, 3), 1.0_f32);
        let ranked = rank_and_fill_missing_cross_sectional_median(&raw, &restrict_all_one);
        assert!(ranked.iter().all(|v| v.is_nan()));
        let restrict_all_nan = Array2::<f32>::from_elem((2, 3), f32::NAN);
        let ranked = rank_and_fill_missing_cross_sectional_median(&raw, &restrict_all_nan);
        assert!(ranked.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_ret_nan_filters_correctly() {
        // 所有signal有效, 但部分ret为NaN
        let raw = Array2::from_elem((2, 10), 1.0_f32);
        let restrict = Array2::from_elem((2, 10), 0.0_f32);
        let ret = Array2::from_shape_vec(
            (2, 10),
            vec![
                0.01_f32,
                0.02,
                f32::NAN,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.10,
                f32::NAN,
                f32::NAN,
                f32::NAN,
                f32::NAN,
                f32::NAN,
                0.06,
                0.07,
                0.08,
                0.09,
                0.10,
            ],
        )
        .unwrap();
        let result = compute_raw_cover_rate(&raw.view(), &restrict.view(), &ret.view(), 0);
        // day0: free=10, valid=9 (ret[2]=NaN) → 9/10=0.9
        // day1: free=10, valid=5 (ret[0..5]=NaN) → 5/10=0.5
        let expected = (0.9_f64 + 0.5) / 2.0;
        assert!(
            (result - expected).abs() < 0.0001,
            "{} vs {}",
            result,
            expected
        );
    }

    #[test]
    fn test_restrict_filters_correctly() {
        let raw = Array2::from_elem((2, 5), 1.0_f32);
        let restrict = Array2::from_shape_vec(
            (2, 5),
            vec![0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let ret = Array2::from_elem((2, 5), 0.01_f32);
        let result = compute_raw_cover_rate(&raw.view(), &restrict.view(), &ret.view(), 0);
        // day0: restrict[0]=free, restrict[1]=1(not free), restrict[2]=free, restrict[3]=1(not free), restrict[4]=free
        //       free=3, valid=3 → 3/3=1.0
        // day1: free=3, valid=3 → 1.0
        assert!((result - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_restrict_nan_not_counted_as_free() {
        let raw = Array2::from_elem((1, 5), 1.0_f32);
        let restrict =
            Array2::from_shape_vec((1, 5), vec![0.0_f32, 0.0, f32::NAN, 0.0, 0.0]).unwrap();
        let ret = Array2::from_elem((1, 5), 0.01_f32);
        let result = compute_raw_cover_rate(&raw.view(), &restrict.view(), &ret.view(), 0);
        // day0: free=4 (restrict[2]=NaN → not free), valid=4 → 4/4=1.0
        assert!((result - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_no_valid_dates_returns_1() {
        let raw = Array2::from_elem((1, 5), f32::NAN);
        let restrict = Array2::from_elem((1, 5), 0.0_f32);
        let ret = Array2::from_elem((1, 5), 0.01_f32);
        let result = compute_raw_cover_rate(&raw.view(), &restrict.view(), &ret.view(), 10);
        assert_eq!(result, 1.0);
    }

    // ==================== BRC（Balanced Rank Concordance）====================

    /// n=4 的手算例子：完美单调 / 完全反序 / r 含并列 / f 含并列 / S≠L。
    #[test]
    fn brc_halves_n4_hand_computed() {
        // 完美单调（f 与 r 同序）：U_k ≡ 0 → D_k ≡ 1 → (S, L) = (1, 1)
        let (s, l) = brc_day_halves(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s, 1.0);
        assert_eq!(l, 1.0);
        // 完全反序：U_k = k(n-k) → D_k ≡ -1 → (S, L) = (-1, -1)
        let (s, l) = brc_day_halves(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]);
        assert_eq!(s, -1.0);
        assert_eq!(l, -1.0);
        // r 含并列：f=[1,2,3,4]，r=[1,1,2,2] → R=[1.5,1.5,3.5,3.5]
        // D_1 = 1 - 2*0.5/(1*3) = 2/3，D_2 = 1，D_3 = 1 - 2*0.5/(3*1) = 2/3
        // S = (D_1+D_2)/2 = 5/6，L = (D_3+D_2)/2 = 5/6
        let (s, l) = brc_day_halves(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 2.0, 2.0]);
        assert!((s - 5.0 / 6.0).abs() < 1e-15, "s={}", s);
        assert!((l - 5.0 / 6.0).abs() < 1e-15, "l={}", l);
        // f 含并列（稳定序保持原始下标）：f=[1,1,2,2]，r=[2,1,4,3] → R=[2,1,4,3]
        // D_1 = 1 - 2*1/(1*3) = 1/3，D_2 = 1，D_3 = 1/3 → S = L = 2/3
        let (s, l) = brc_day_halves(&[1.0, 1.0, 2.0, 2.0], &[2.0, 1.0, 4.0, 3.0]);
        assert!((s - 2.0 / 3.0).abs() < 1e-15, "s={}", s);
        assert!((l - 2.0 / 3.0).abs() < 1e-15, "l={}", l);
        // S ≠ L 的手算例子：f=[1,2,3,4]，r=[1,2,4,3] → R=[1,2,4,3]
        // D_1 = 1，D_2 = 1，D_3 = 1 - 2*1/(3*1) = 1/3
        // S = (D_1+D_2)/2 = 1，L = (D_3+D_2)/2 = 2/3
        let (s, l) = brc_day_halves(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 4.0, 3.0]);
        assert_eq!(s, 1.0);
        assert!((l - 2.0 / 3.0).abs() < 1e-15, "l={}", l);
    }

    /// 两条取负恒等式（契约勘误后的口径）：
    /// - future 取负 → 每个 D_k 严格反号 → (S, L) → (-S, -L)，BRC → -max(S, L)；
    /// - signal 取负（因子序反转）→ (S, L) → (-L, -S)，BRC → -max(S, L)；
    /// - 两条叠加 → (S, L) → (L, S)。
    ///
    /// 注意不是「future 取负 → (-L, -S)」：那个说法只在 S = L 的对称样本上凑巧成立，
    /// 下面这组 S = 1、L = 2/3 就是反例。
    #[test]
    fn brc_halves_negation_identity() {
        let signal = [1.0_f32, 2.0, 3.0, 4.0];
        let future = [1.0_f32, 2.0, 4.0, 3.0];
        let (s, l) = brc_day_halves(&signal, &future);
        assert_eq!(s, 1.0);
        assert!((l - 2.0 / 3.0).abs() < 1e-15, "l={}", l);

        // future 取负：D_k → -D_k → (S, L) → (-S, -L)
        let neg_future: Vec<f32> = future.iter().map(|v| -v).collect();
        let (s1, l1) = brc_day_halves(&signal, &neg_future);
        assert!((s1 - (-s)).abs() < 1e-15, "s1={}", s1);
        assert!((l1 - (-l)).abs() < 1e-15, "l1={}", l1);
        assert!((s1.min(l1) - (-s.max(l))).abs() < 1e-15);
        // 反例核对：(-L, -S) 与实测不符（S ≠ L 时）
        assert!((s1 - (-l)).abs() > 1e-3, "s1={} -L={}", s1, -l);

        // signal 取负（因子序反转）：S → -L、L → -S
        let neg_signal: Vec<f32> = signal.iter().map(|v| -v).collect();
        let (s2, l2) = brc_day_halves(&neg_signal, &future);
        assert!((s2 - (-l)).abs() < 1e-15, "s2={}", s2);
        assert!((l2 - (-s)).abs() < 1e-15, "l2={}", l2);
        assert!((s2.min(l2) - (-s.max(l))).abs() < 1e-15);

        // 两侧同时取负：两条恒等式叠加后 (S, L) → (L, S)（S ↔ L 互换）
        let (s3, l3) = brc_day_halves(&neg_signal, &neg_future);
        assert!((s3 - l).abs() < 1e-15, "s3={}", s3);
        assert!((l3 - s).abs() < 1e-15, "l3={}", l3);
    }

    /// n < 2、长度不等、含 NaN/±Inf → (NaN, NaN)（该日跳过）。
    #[test]
    fn brc_halves_rejects_short_and_nonfinite() {
        assert!(brc_day_halves(&[1.0], &[1.0]).0.is_nan());
        assert!(brc_day_halves(&[], &[]).0.is_nan());
        assert!(brc_day_halves(&[1.0, 2.0], &[1.0]).0.is_nan());
        assert!(brc_day_halves(&[1.0, f32::NAN], &[1.0, 2.0]).0.is_nan());
        assert!(brc_day_halves(&[1.0, 2.0], &[1.0, f32::INFINITY]).1.is_nan());
        assert!(brc_day_halves(&[f32::NEG_INFINITY, 2.0], &[1.0, 2.0]).0.is_nan());
    }

    /// 优化前的实现（两次比较排序 + 每天新建 Vec），只作为逐位对账的参照留在测试里。
    fn brc_day_halves_reference(signal: &[f32], future: &[f32]) -> (f64, f64) {
        let n = signal.len();
        if n < 2 || future.len() != n {
            return (f64::NAN, f64::NAN);
        }
        if signal.iter().any(|v| !v.is_finite()) || future.iter().any(|v| !v.is_finite()) {
            return (f64::NAN, f64::NAN);
        }
        let ranks = average_ranks(future);
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&a, &b| {
            signal[a as usize]
                .partial_cmp(&signal[b as usize])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let nf = n as f64;
        let m = n / 2;
        let mut w = 0.0_f64;
        let mut s_sum = 0.0_f64;
        let mut l_sum = 0.0_f64;
        for k in 1..n {
            w += ranks[order[k - 1] as usize];
            let kf = k as f64;
            let u = w - kf * (kf + 1.0) / 2.0;
            let d = 1.0 - 2.0 * u / (kf * (nf - kf));
            if k <= m {
                s_sum += d;
            }
            if k >= n - m {
                l_sum += d;
            }
        }
        let mf = m as f64;
        (s_sum / mf, l_sum / mf)
    }

    /// 优化前 vs 优化后（radix + 复用缓冲）：400 组随机数据逐位相等。
    ///
    /// 覆盖：n 从 2 到 79；一半样本把值量化成少数档制造并列（signal 并列与 future 并列
    /// 都有）；每 7 组掺一个 NaN 或 ±Inf（走「该日跳过」分支）。
    #[test]
    fn brc_buffered_matches_reference_exact() {
        let mut seed = 0x5eed_2026_0918_u64;
        let mut next_f32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 as f32) / 1.0e6 - 1000.0
        };
        let mut mismatches = 0usize;
        let mut buf = BrcBuffers::default();
        for trial in 0..400usize {
            let n = 2 + (trial * 7 + 3) % 78;
            let mut signal = Vec::<f32>::with_capacity(n);
            let mut future = Vec::<f32>::with_capacity(n);
            for _ in 0..n {
                let mut s = next_f32();
                let mut f = next_f32();
                if trial % 2 == 0 {
                    // 量化到 3 档 → 大量并列
                    s = (s / 500.0).round();
                    f = (f / 500.0).round();
                }
                if trial % 4 == 1 {
                    // 只让 signal 有并列
                    s = (s / 500.0).round();
                }
                if trial % 4 == 3 {
                    // 只让 future 有并列
                    f = (f / 500.0).round();
                }
                signal.push(s);
                future.push(f);
            }
            if trial % 7 == 0 {
                let bad = trial % n;
                match trial % 21 {
                    0 => future[bad] = f32::NAN,
                    7 => future[bad] = f32::INFINITY,
                    _ => signal[bad] = f32::NAN,
                }
            }
            let want = brc_day_halves_reference(&signal, &future);
            let got = brc_day_halves_into(&signal, &future, &mut buf);
            let same = |x: f64, y: f64| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
            if !same(want.0, got.0) || !same(want.1, got.1) {
                mismatches += 1;
                if mismatches <= 3 {
                    println!(
                        "trial={trial} n={n} want=({:?},{:?}) got=({:?},{:?})",
                        want.0, want.1, got.0, got.1
                    );
                }
            }
        }
        assert_eq!(mismatches, 0, "优化前后有 {mismatches}/400 组不一致");
    }

    /// 复刻生产 `build_bt_precomputed` 的全行序 + opt/v8 的 gen/stamp walk：
    /// 返回「子集（下标升序）按值升序」的 ordinal 秩（0..n-1）。
    ///
    /// 与 `build_bt_precomputed` 完全同式：NaN 键取 u32::MAX（置末），其余走 mono_key32，
    /// 4 趟稳定 radix + 初始下标升序 ⇒ (值, index) 总序。
    fn walk_subset_ordinal(full_values: &[f32], subset: &[u32]) -> Vec<i64> {
        let mut keys: Vec<u32> = full_values
            .iter()
            .map(|&v| if v.is_nan() { u32::MAX } else { mono_key32(v) })
            .collect();
        let mut order: Vec<usize> = (0..full_values.len()).collect();
        let mut tmp: Vec<usize> = Vec::new();
        radix_sort_u32_keys(&keys, &mut order, &mut tmp);

        let mut gen = vec![0u32; full_values.len()];
        let mut stamp = vec![0u32; full_values.len()];
        let gen_id = 1u32;
        for (pos, &stk) in subset.iter().enumerate() {
            gen[stk as usize] = gen_id;
            stamp[stk as usize] = (pos + 1) as u32;
        }
        let mut out = vec![0i64; subset.len()];
        let mut counter = 0usize;
        for &stk in order.iter() {
            if gen[stk] == gen_id {
                out[stamp[stk] as usize - 1] = counter as i64;
                counter += 1;
            }
        }
        out
    }

    /// 核心主张：orders 的 walk（(值, index) 预排序）逆置换后按连续等值段取平均，
    /// 与 `average_ranks` 逐位相同。
    ///
    /// 覆盖大量并列、-0.0 / +0.0 混排、极端重复（整段同值）、子集只取一小部分等情形。
    #[test]
    fn brc_walk_ranks_match_average_ranks_exact() {
        let mut seed = 0x2026_0918_5eed_u64;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let mut buf = BrcBuffers::default();
        let mut mismatches = 0usize;
        for trial in 0..400usize {
            let n_full = 5 + (trial * 11 + 7) % 60;
            let mut full = Vec::<f32>::with_capacity(n_full);
            for i in 0..n_full {
                let r = next_u32();
                let v = match trial % 5 {
                    // 极端重复：只有 2~3 个不同值
                    0 => (r % 3) as f32,
                    // 大量并列 + -0.0 / +0.0
                    1 => match r % 6 {
                        0 => -0.0_f32,
                        1 => 0.0_f32,
                        k => (k as f32) - 2.0,
                    },
                    // 小数并列
                    2 => ((r % 20) as f32) / 4.0,
                    // 几乎全同值
                    3 => {
                        if i % 7 == 0 {
                            (r % 100) as f32
                        } else {
                            5.0_f32
                        }
                    }
                    // 一般随机
                    _ => (r as f32) / 1.0e6 - 1000.0,
                };
                full.push(v);
            }
            // 子集：随机挑一部分（保持下标升序，与生产 filtered_stock_idx 一致）
            let mut subset: Vec<u32> = (0..n_full as u32)
                .filter(|&s| (next_u32() as usize + s as usize) % 3 != 0)
                .collect();
            if subset.len() < 2 {
                subset = (0..n_full as u32).collect();
            }
            let subset_values: Vec<f32> = subset.iter().map(|&s| full[s as usize]).collect();
            let future_ordinal = walk_subset_ordinal(&full, &subset);
            // 用假 signal（值 = 子集下标倒序）跑快速路径，只为让 future_avg 被填出来
            let signal: Vec<f32> = (0..subset.len()).map(|i| -(i as f32)).collect();
            let signal_ordinal = ordinal_ranks_radix(&signal);
            buf.future_avg.clear();
            let _ = brc_day_halves_from_ranks_into(
                &signal_ordinal,
                &future_ordinal,
                &subset_values,
                &mut buf,
            );
            let want = average_ranks(&subset_values);
            if buf.future_avg.len() != want.len() {
                mismatches += 1;
                continue;
            }
            for (i, &w) in want.iter().enumerate() {
                if buf.future_avg[i].to_bits() != w.to_bits() {
                    mismatches += 1;
                    if mismatches <= 3 {
                        println!(
                            "trial={trial} i={i} walk={:?} average_ranks={:?}",
                            buf.future_avg[i], w
                        );
                    }
                    break;
                }
            }
        }
        assert_eq!(mismatches, 0, "walk 平均秩与 average_ranks 有 {mismatches}/400 组不一致");
    }

    /// 快速路径（walk 的 ordinal 秩 + signal ordinal，零排序零分配）与旧参考实现逐位相等。
    #[test]
    fn brc_fast_path_matches_reference_exact() {
        let mut seed = 0x1234_5678_9abc_def0_u64;
        let mut next_u32 = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let mut buf = BrcBuffers::default();
        let mut mismatches = 0usize;
        for trial in 0..400usize {
            let n_full = 4 + (trial * 13 + 5) % 70;
            let mut full = Vec::<f32>::with_capacity(n_full);
            for _ in 0..n_full {
                let r = next_u32();
                full.push(if trial % 4 == 0 {
                    (r % 4) as f32
                } else if trial % 4 == 1 {
                    match r % 5 {
                        0 => -0.0_f32,
                        1 => 0.0_f32,
                        k => (k as f32) - 2.0,
                    }
                } else {
                    (r as f32) / 1.0e5 - 1000.0
                });
            }
            let mut subset: Vec<u32> = (0..n_full as u32)
                .filter(|&s| (next_u32() as usize + s as usize) % 4 != 0)
                .collect();
            if subset.len() < 2 {
                subset = (0..n_full as u32).collect();
            }
            let n = subset.len();
            // signal 值（子集内），量化制造并列
            let signal: Vec<f32> = (0..n)
                .map(|i| {
                    let r = next_u32();
                    if trial % 3 == 0 {
                        (r % 5) as f32
                    } else {
                        (r as f32) / 1.0e5 - 1000.0 + i as f32 * 1.0e-3
                    }
                })
                .collect();
            let future: Vec<f32> = subset.iter().map(|&s| full[s as usize]).collect();
            let signal_ordinal = ordinal_ranks_radix(&signal);
            let future_ordinal = walk_subset_ordinal(&full, &subset);

            let want = brc_day_halves_reference(&signal, &future);
            let got = brc_day_halves_from_ranks_into(
                &signal_ordinal,
                &future_ordinal,
                &future,
                &mut buf,
            );
            let same = |x: f64, y: f64| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
            if !same(want.0, got.0) || !same(want.1, got.1) {
                mismatches += 1;
                if mismatches <= 3 {
                    println!(
                        "trial={trial} n={n} want=({:?},{:?}) got=({:?},{:?})",
                        want.0, want.1, got.0, got.1
                    );
                }
            }
        }
        assert_eq!(mismatches, 0, "快速路径与参考实现有 {mismatches}/400 组不一致");
    }

    /// 生产 opt 路径与参考实现逐位一致（含新增 BRC/BRC_S/BRC_L 三列），两种 ic_only 都比。
    #[test]
    fn opt_backtest_matches_reference_exact() {
        let (t, n) = (14usize, 20usize);
        let mut seed = 0x0bad_c0de_dead_beefu64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 as f32) / 500.0 - 1000.0
        };
        let mut slot = Array2::<f32>::from_elem((t, n), f32::NAN);
        for i in 0..t {
            for j in 0..n {
                if (i * n + j + 3) % 6 != 0 {
                    slot[[i, j]] = next();
                }
            }
        }
        let mut restrict = Array2::<f32>::from_elem((t, n), 0.0);
        for i in 0..t {
            for j in 0..n {
                if (i * n + j) % 8 == 0 {
                    restrict[[i, j]] = 1.0;
                }
            }
        }
        let mut ret_gap1 = Array2::<f32>::from_elem((t, n), f32::NAN);
        let mut ret_sum_gap1 = Array2::<f32>::from_elem((t, n), f32::NAN);
        let mut ret_gap5 = Array2::<f32>::from_elem((t, n), f32::NAN);
        let mut ret_sum_gap5 = Array2::<f32>::from_elem((t, n), f32::NAN);
        for i in 0..t {
            for j in 0..n {
                ret_gap1[[i, j]] = next();
                ret_sum_gap1[[i, j]] = next();
                ret_gap5[[i, j]] = next();
                ret_sum_gap5[[i, j]] = next();
            }
        }
        let index = Array1::<f32>::from_shape_fn(t, |_| next());
        let dates: Vec<i32> = (0..t as i32).map(|d| 20200101 + d).collect();
        let pre = build_bt_precomputed(&ret_sum_gap1, &ret_sum_gap5).unwrap();
        let open_symbol_counts = precompute_open_symbol_counts(&restrict.view());
        for &ic_only in &[false, true] {
            let (ref_g1, ref_g5) = legacy_backtest_gap1_gap5_single_slot(
                slot.view(),
                ret_gap1.view(),
                ret_sum_gap1.view(),
                ret_gap5.view(),
                ret_sum_gap5.view(),
                restrict.view(),
                index.view(),
                &dates,
                dates[2],
                10,
                &open_symbol_counts,
                ic_only,
            );
            let (opt_g1, opt_g5) = legacy_backtest_gap1_gap5_single_slot_opt(
                slot.view(),
                ret_gap1.view(),
                ret_sum_gap1.view(),
                ret_gap5.view(),
                ret_sum_gap5.view(),
                restrict.view(),
                index.view(),
                &dates,
                dates[2],
                10,
                &open_symbol_counts,
                ic_only,
                &pre,
            );
            assert_backtest_result_eq(&ref_g1, &opt_g1);
            assert_backtest_result_eq(&ref_g5, &opt_g5);
        }
    }
}

// ==================== v6：在线转置回测（不读投影区，从 colblk 批量转置，回测核心零改动）====================
#[pyfunction]
#[pyo3(signature = (
    factor_names,
    factor_paths,
    dates,
    stocks,
    windows,
    fold,
    n_jobs,
    min_valid,
    cache_root,
    style_vars_dir,
    ret_gap1_path,
    ret_sum_gap1_path,
    ret_gap5_path,
    ret_sum_gap5_path,
    restrict_path,
    index_ret_path,
    backtest_start,
    cover_rate=0.97,
    ret_point_neu_gap5=0.055,
    ret_point_neu_gap1=0.08,
    ic_point_neu_gap5=0.01,
    ic_point_neu_gap1=0.006,
    ret_point_gap5=0.1,
    ret_point_gap1=0.13,
    ic_point_gap5=0.03,
    ic_point_gap1=0.02,
    ic_more_important_gap5=0.01,
    ic_more_important_gap1=0.006,
    majority_count_threshold=200.0,
    zero_max_threshold=0.01,
    nan_max_threshold=0.04,
))]
pub fn tail_v5_run_candidates_online<'py>(
    py: Python<'py>,
    factor_names: Vec<String>,
    factor_paths: Vec<String>,
    dates: Vec<i32>,
    stocks: Vec<String>,
    windows: Vec<usize>,
    fold: bool,
    n_jobs: usize,
    min_valid: usize,
    cache_root: String,
    style_vars_dir: String,
    ret_gap1_path: String,
    ret_sum_gap1_path: String,
    ret_gap5_path: String,
    ret_sum_gap5_path: String,
    restrict_path: String,
    index_ret_path: String,
    backtest_start: i32,
    cover_rate: f64,
    ret_point_neu_gap5: f64,
    ret_point_neu_gap1: f64,
    ic_point_neu_gap5: f64,
    ic_point_neu_gap1: f64,
    ret_point_gap5: f64,
    ret_point_gap1: f64,
    ic_point_gap5: f64,
    ic_point_gap1: f64,
    ic_more_important_gap5: Option<f64>,
    ic_more_important_gap1: Option<f64>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PyResult<PyObject> {
    if factor_names.len() != factor_paths.len() {
        return Err(PyValueError::new_err(
            "factor_names 和 factor_paths 长度必须一致",
        ));
    }
    if n_jobs == 0 {
        return Err(PyValueError::new_err("n_jobs 必须大于 0"));
    }
    if (ic_more_important_gap5.is_some()) != (ic_more_important_gap1.is_some()) {
        return Err(PyValueError::new_err(
            "ic_more_important_gap5 和 ic_more_important_gap1 必须同时有值或同时为 None",
        ));
    }

    let output = py.allow_threads(|| -> Result<(usize, usize, HashMap<String, usize>), String> {
        let started = Instant::now();
        let cache_root_path = PathBuf::from(&cache_root);
        let task_results_dir = cache_root_path.join("task_results");
        let logs_dir = cache_root_path.join("logs");
        let completed_log_path = logs_dir.join("completed_sources.txt");
        fs::create_dir_all(&task_results_dir).map_err(|e| format!("创建 task_results 目录失败: {}", e))?;
        fs::create_dir_all(&logs_dir).map_err(|e| format!("创建 logs 目录失败: {}", e))?;

        let shared = SharedInputs {
            dates: Arc::new(dates),
            stocks: Arc::new(stocks),
            windows: Arc::new(windows),
            fold,
            min_valid,
            backtest_start,
            legacy_style_data: Arc::new(
                IOOptimizedStyleData::load_from_vars_h5(&style_vars_dir)
                    .map_err(|e| e.to_string())?
            ),
            industry_neutralize: true,
            industry: None,
            neutralize_std_shared: None,
            ret_gap1: Arc::new(read_npy(&ret_gap1_path).map_err(|e| format!("读取 ret_gap1.npy 失败: {}", e))?),
            ret_sum_gap1: Arc::new(read_npy(&ret_sum_gap1_path).map_err(|e| format!("读取 ret_sum_gap1.npy 失败: {}", e))?),
            ret_gap5: Arc::new(read_npy(&ret_gap5_path).map_err(|e| format!("读取 ret_gap5.npy 失败: {}", e))?),
            ret_sum_gap5: Arc::new(read_npy(&ret_sum_gap5_path).map_err(|e| format!("读取 ret_sum_gap5.npy 失败: {}", e))?),
            restrict: Arc::new(read_npy(&restrict_path).map_err(|e| format!("读取 restrict.npy 失败: {}", e))?),
            index_ret: Arc::new(read_npy(&index_ret_path).map_err(|e| format!("读取 index_ret.npy 失败: {}", e))?),
            config: Arc::new(TailSelectionConfig {
                cover_rate,
                ret_point_neu_gap5,
                ret_point_neu_gap1,
                ic_point_neu_gap5,
                ic_point_neu_gap1,
                ret_point_gap5,
                ret_point_gap1,
                ic_point_gap5,
                ic_point_gap1,
                ic_more_important_gap5,
                ic_more_important_gap1,
                majority_count_threshold,
                zero_max_threshold,
                nan_max_threshold,
                save_all_metrics: false,
                ic_only: false,
            }),
            bt_pre: None,
        free_mask: None,
        v3_shared: None,
        };

        let mut aggregated = AggregatedCandidates::default();
        let mut completed_sources = HashSet::<String>::new();
        let mut stats = ProcessStats::default();
        for (source_factor, factor_path) in factor_names.iter().zip(factor_paths.iter()) {
            let result_path = factor_result_path(&task_results_dir, source_factor);
            if result_path.exists() {
                if let Ok(mut task_result) = read_task_result(&result_path) {
                    if !task_result.passed
                        && (!task_result.raw_summary_gap1.is_empty()
                            || !task_result.raw_summary_gap5.is_empty()
                            || !task_result.neu_summary_gap1.is_empty()
                            || !task_result.neu_summary_gap5.is_empty())
                    {
                        task_result.passed = true;
                    }
                    if task_result.passed {
                        stats.restored_pass += 1;
                    } else if task_result.eliminated_by_raw_cover {
                        stats.restored_raw_cov += 1;
                    } else if !task_result.any_window_passed_preflight
                        && !task_result.raw_summary_gap1.is_empty() == false
                        && !task_result.raw_summary_gap5.is_empty() == false
                        && !task_result.neu_summary_gap1.is_empty() == false
                        && !task_result.neu_summary_gap5.is_empty() == false
                        && !task_result.passed
                    {
                        // 区分 preflight 淘汰 vs 未知:
                        // 旧缓存没有 any_window_passed_preflight(默认false)
                        // 也没有 eliminated_by_raw_cover(默认false)
                        // 如果所有 summary vecs 都为空且 passed=false
                        // 我们通过是否有 preflight 失败记录来判断
                        if task_result.preflight_maj_failed_windows > 0
                            || task_result.preflight_zero_failed_windows > 0
                            || task_result.preflight_nan_failed_windows > 0
                        {
                            stats.restored_preflight += 1;
                        } else {
                            stats.restored_unknown += 1;
                        }
                    } else {
                        // any_window_passed_preflight=true, passed=false → ret_ic淘汰
                        stats.restored_ret_ic += 1;
                    }
                    stats.preflight_maj_windows += task_result.preflight_maj_failed_windows;
                    stats.preflight_zero_windows += task_result.preflight_zero_failed_windows;
                    stats.preflight_nan_windows += task_result.preflight_nan_failed_windows;
                    aggregated.merge_task(task_result);
                    completed_sources.insert(source_factor.clone());
                    continue;
                }
            }
            let _ = factor_path;
        }
        let restored_sources =
            stats.restored_pass + stats.restored_raw_cov + stats.restored_preflight
            + stats.restored_ret_ic + stats.restored_unknown;

        let pending_tasks = factor_names
            .iter()
            .zip(factor_paths.iter())
            .filter_map(|(source_factor, factor_path)| {
                if completed_sources.contains(source_factor) {
                    None
                } else {
                    Some(TailTask {
                        source_factor: source_factor.clone(),
                        factor_path: factor_path.clone(),
                    })
                }
            })
            .collect::<Vec<_>>();

        let total_pending = pending_tasks.len();
        if total_pending > 0 {
            init_status_line();
            let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            let total = factor_names.len();
            let total_elim = restored_sources - stats.restored_pass;
            let l1 = format!("[{}] Tail V4 启动，待处理 {}/{} 个原始因子", current_time, total_pending, total);
            let l2 = format!("累计通过 {} ({}%) | 淘汰 raw_cov={} preflight={} ret_ic={} 未知={}",
                stats.restored_pass,
                if restored_sources > 0 { stats.restored_pass * 100 / restored_sources } else { 0 },
                stats.restored_raw_cov, stats.restored_preflight,
                stats.restored_ret_ic, stats.restored_unknown);
            let _ = total_elim; // suppress unused warning
            let l3 = format!("maj={}w zero={}w nan={}w | 恢复 {} 个 | 即将开始处理...",
                stats.preflight_maj_windows, stats.preflight_zero_windows,
                stats.preflight_nan_windows, restored_sources);
            update_status_line(&l1, &l2, &l3);
        }
        let (task_sender, task_receiver): (Sender<TailTask>, Receiver<TailTask>) = unbounded();
        let (result_sender, result_receiver) = unbounded::<Result<TailTaskResult, (String, String)>>();
        for task in pending_tasks {
            task_sender.send(task).map_err(|e| format!("发送任务失败: {}", e))?;
        }
        drop(task_sender);

        let shared_arc = Arc::new(shared);

        // 检测是否为列式存储模式（任一 task 路径含 "::"）。
        // 若是，启用 IO/CPU 分离架构：少量 IO 线程顺序读因子 → 有界内存队列 → n_jobs 计算线程。
        // 否则保持原逻辑：n_jobs 线程各自读盘 + 计算（兼容 parquet/h5）。
        let is_colblk_mode = factor_paths.iter().any(|p| p.contains("::"));

        // v6 在线转置：单 IO 线程批量转置 colblk（不读投影区）→ 有界 channel → n_jobs 计算线程。
        // 分批按 col_idx 顺序遍历 chunk，每批因子数由 500GB 内存预算动态决定。
        if is_colblk_mode {
            let (loaded_tx, loaded_rx) =
                crossbeam::channel::bounded::<(TailTask, ndarray::Array2<f32>)>(16);

            // 解析唯一 store_dir（所有 colblk task 共享同一 Reader）
            let store_dir_for_reader = factor_paths
                .iter()
                .find_map(|p| {
                    if p.contains("::") {
                        let sp = p.splitn(2, "::").next().unwrap_or("");
                        if sp.ends_with(".colblk") {
                            Path::new(sp).parent().map(|x| x.to_string_lossy().to_string())
                        } else {
                            Some(sp.to_string())
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            // 解析 col_idx → task 映射；收集全部 col_idx（去重排序，假设连续以高效读 chunk 内因子段）
            let mut col_idx_to_task: HashMap<usize, TailTask> = HashMap::new();
            for (source_factor, factor_path) in factor_names.iter().zip(factor_paths.iter()) {
                if let Some(s) = factor_path.splitn(2, "::").nth(1) {
                    if let Ok(col_idx) = s.parse::<usize>() {
                        col_idx_to_task.insert(
                            col_idx,
                            TailTask {
                                source_factor: source_factor.clone(),
                                factor_path: factor_path.clone(),
                            },
                        );
                    }
                }
            }
            let mut all_col_idx: Vec<usize> = col_idx_to_task.keys().copied().collect();
            all_col_idx.sort_unstable();
            let total_factors = all_col_idx.len();

            // 流式转置：逐因子读 colblk（遍历 chunk 一次，每因子立即推 channel）
            // 内存：channel 缓冲 16 个矩阵 + process 中间量 ≈ 50GB
            eprintln!(
                "🔧 v6 流式转置（逐因子）：{} 因子，遍历 colblk 逐因子转置+推channel",
                total_factors
            );

            // IO 线程：逐因子读 colblk → 转置 → 推 channel
            // 复用 read_factor_to_matrix_fast（预计算 scatter 映射，避免每行 HashMap 查找）
            let io_handle = {
                let dates = shared_arc.dates.clone();
                let stocks = shared_arc.stocks.clone();
                thread::spawn(move || {
                    let reader = match crate::factor_store_v5::FactorStoreReader::open(
                        &store_dir_for_reader,
                    ) {
                        Ok(r) => r,
                        Err(_) => return,
                    };
                    let scatter_maps = reader.precompute_scatter_maps(
                        dates.as_slice(),
                        stocks.as_slice(),
                    );
                    for &col_idx in all_col_idx.iter() {
                        let matrix = match reader.read_factor_to_matrix_fast(
                            col_idx,
                            dates.as_slice(),
                            stocks.as_slice(),
                            &scatter_maps,
                        ) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };
                        if let Some(task) = col_idx_to_task.get(&col_idx) {
                            if loaded_tx.send((task.clone(), matrix)).is_err() {
                                return;
                            }
                        }
                    }
                })
            };

            // n_jobs 计算线程（纯内存，与原版完全一致——零改动复用 process_task_with_values）
            let mut handles = Vec::with_capacity(n_jobs);
            for _ in 0..n_jobs {
                let loaded_rx = loaded_rx.clone();
                let tx = result_sender.clone();
                let shared_clone = shared_arc.clone();
                handles.push(thread::spawn(move || {
                    while let Ok((task, raw_values)) = loaded_rx.recv() {
                        let task_name = task.source_factor.clone();
                        let outcome = process_task_with_values(&task, raw_values, &shared_clone)
                            .map_err(|err| (task_name, err));
                        if tx.send(outcome).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(result_sender);

            let _ = io_handle.join();
            for h in handles {
                let _ = h.join();
            }
        } else {
            // ---- 非 colblk 模式：原架构（n_jobs 线程各自读盘 + 计算，兼容 parquet/h5）----
            let mut handles = Vec::with_capacity(n_jobs);
            for _ in 0..n_jobs {
                let rx = task_receiver.clone();
                let tx = result_sender.clone();
                let shared_clone = shared_arc.clone();
                handles.push(thread::spawn(move || {
                    while let Ok(task) = rx.recv() {
                        let task_name = task.source_factor.clone();
                        let outcome = process_task(&task, &shared_clone)
                            .map_err(|err| (task_name, err));
                        if tx.send(outcome).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(result_sender);
            for h in handles {
                let _ = h.join();
            }
        }

        let mut processed_sources = 0usize;
        while let Ok(task_outcome) = result_receiver.recv() {
            match task_outcome {
                Ok(task_result) => {
                    let result_path = factor_result_path(&task_results_dir, &task_result.source_factor);
                    let is_passed = task_result.passed;
                    let is_raw_cov = task_result.eliminated_by_raw_cover;
                    let any_window = task_result.any_window_passed_preflight;
                    let preflight_maj = task_result.preflight_maj_failed_windows;
                    let preflight_zero = task_result.preflight_zero_failed_windows;
                    let preflight_nan = task_result.preflight_nan_failed_windows;
                    write_task_result(&result_path, &task_result)?;
                    append_completed_source(&completed_log_path, &task_result.source_factor)?;
                    aggregated.merge_task(task_result);
                    processed_sources += 1;

                    if is_passed {
                        stats.done_pass += 1;
                    } else if is_raw_cov {
                        stats.done_raw_cov += 1;
                    } else if !any_window {
                        stats.done_preflight += 1;
                    } else {
                        stats.done_ret_ic += 1;
                    }
                    stats.done = processed_sources;
                    stats.preflight_maj_windows += preflight_maj;
                    stats.preflight_zero_windows += preflight_zero;
                    stats.preflight_nan_windows += preflight_nan;

                    if total_pending > 0 {
                        let elapsed = started.elapsed();
                        let elapsed_secs = elapsed.as_secs();
                        let progress = processed_sources as f64 / total_pending as f64;
                        let estimated_total_secs = if progress > 0.0 {
                            elapsed.as_secs_f64() / progress
                        } else {
                            elapsed.as_secs_f64()
                        };
                        let remaining_secs = if estimated_total_secs > elapsed.as_secs_f64() {
                            (estimated_total_secs - elapsed.as_secs_f64()) as u64
                        } else {
                            0
                        };
                        let (elapsed_h, elapsed_m, elapsed_s) = format_hms(elapsed_secs);
                        let (remaining_h, remaining_m, remaining_s) = format_hms(remaining_secs);
                        let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

                        let cum_pass = stats.restored_pass + stats.done_pass;
                        let cum_total = restored_sources + processed_sources;
                        let cum_raw_cov = stats.restored_raw_cov + stats.done_raw_cov;
                        let cum_preflight = stats.restored_preflight + stats.done_preflight;
                        let cum_ret_ic = stats.restored_ret_ic + stats.done_ret_ic;
                        let cum_unknown = stats.restored_unknown + stats.done_unknown;

                        let l1 = format!(
                            "[{}] Tail V4 进度 {}/{} ({:.1}%)，已用{}h{}m{}s，预计剩余{}h{}m{}s",
                            current_time, processed_sources, total_pending,
                            progress * 100.0,
                            elapsed_h, elapsed_m, elapsed_s,
                            remaining_h, remaining_m, remaining_s,
                        );
                        let l2 = format!(
                            "累计通过 {} ({:.0}%) | 淘汰 raw_cov={} preflight={} ret_ic={} 未知={}",
                            cum_pass,
                            if cum_total > 0 { cum_pass as f64 * 100.0 / cum_total as f64 } else { 0.0 },
                            cum_raw_cov, cum_preflight, cum_ret_ic, cum_unknown,
                        );
                        let l3 = format!(
                            "maj={}w zero={}w nan={}w | 本次 {}(通过{}) | 恢复 {}(通过{})",
                            stats.preflight_maj_windows, stats.preflight_zero_windows,
                            stats.preflight_nan_windows,
                            stats.done, stats.done_pass,
                            restored_sources, stats.restored_pass,
                        );
                        update_status_line(&l1, &l2, &l3);
                    }
                }
                Err((task_name, err)) => {
                    reset_status_line();
                    return Err(format!("处理因子 {} 失败: {}", task_name, err));
                }
            }
        }

        if total_pending > 0 {
            println!();
            reset_status_line();
        }

        // 注意：worker 线程在上方 IO/CPU 分离分支或原架构分支内已全部 join 完毕。

        write_aggregated_outputs(
            &cache_root_path,
            &aggregated,
        )?;

        let mut candidate_counts = HashMap::new();
        candidate_counts.insert("rolled_gap1".to_string(), aggregated.raw_summary_gap1.len());
        candidate_counts.insert("rolled_gap5".to_string(), aggregated.raw_summary_gap5.len());
        candidate_counts.insert("neu_gap1".to_string(), aggregated.neu_summary_gap1.len());
        candidate_counts.insert("neu_gap5".to_string(), aggregated.neu_summary_gap5.len());
        Ok((processed_sources, restored_sources, candidate_counts))
    }).map_err(PyRuntimeError::new_err)?;

    let info = PyDict::new(py);
    info.set_item("processed_sources", output.0)?;
    info.set_item("restored_sources", output.1)?;
    let candidate_counts = PyDict::new(py);
    for (key, value) in output.2 {
        candidate_counts.set_item(key, value)?;
    }
    info.set_item("candidate_counts", candidate_counts)?;
    Ok(info.into())
}

// ==================== V7：批量顺序 pread 回测（消除 HDD 寻道，回测核心零改动）====================
#[pyfunction]
#[pyo3(signature = (
    factor_names,
    factor_paths,
    dates,
    stocks,
    windows,
    fold,
    n_jobs,
    min_valid,
    cache_root,
    style_vars_dir,
    ret_gap1_path,
    ret_sum_gap1_path,
    ret_gap5_path,
    ret_sum_gap5_path,
    restrict_path,
    index_ret_path,
    backtest_start,
    cover_rate=0.97,
    ret_point_neu_gap5=0.055,
    ret_point_neu_gap1=0.08,
    ic_point_neu_gap5=0.01,
    ic_point_neu_gap1=0.006,
    ret_point_gap5=0.1,
    ret_point_gap1=0.13,
    ic_point_gap5=0.03,
    ic_point_gap1=0.02,
    ic_more_important_gap5=0.01,
    ic_more_important_gap1=0.006,
    majority_count_threshold=200.0,
    zero_max_threshold=0.01,
    nan_max_threshold=0.04,
))]
pub fn tail_v5_run_candidates_v7<'py>(
    py: Python<'py>,
    factor_names: Vec<String>,
    factor_paths: Vec<String>,
    dates: Vec<i32>,
    stocks: Vec<String>,
    windows: Vec<usize>,
    fold: bool,
    n_jobs: usize,
    min_valid: usize,
    cache_root: String,
    style_vars_dir: String,
    ret_gap1_path: String,
    ret_sum_gap1_path: String,
    ret_gap5_path: String,
    ret_sum_gap5_path: String,
    restrict_path: String,
    index_ret_path: String,
    backtest_start: i32,
    cover_rate: f64,
    ret_point_neu_gap5: f64,
    ret_point_neu_gap1: f64,
    ic_point_neu_gap5: f64,
    ic_point_neu_gap1: f64,
    ret_point_gap5: f64,
    ret_point_gap1: f64,
    ic_point_gap5: f64,
    ic_point_gap1: f64,
    ic_more_important_gap5: Option<f64>,
    ic_more_important_gap1: Option<f64>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PyResult<PyObject> {
    if factor_names.len() != factor_paths.len() {
        return Err(PyValueError::new_err(
            "factor_names 和 factor_paths 长度必须一致",
        ));
    }
    if n_jobs == 0 {
        return Err(PyValueError::new_err("n_jobs 必须大于 0"));
    }
    if (ic_more_important_gap5.is_some()) != (ic_more_important_gap1.is_some()) {
        return Err(PyValueError::new_err(
            "ic_more_important_gap5 和 ic_more_important_gap1 必须同时有值或同时为 None",
        ));
    }

    let output = py.allow_threads(|| -> Result<(usize, usize, HashMap<String, usize>), String> {
        let started = Instant::now();
        let cache_root_path = PathBuf::from(&cache_root);
        let task_results_dir = cache_root_path.join("task_results");
        let logs_dir = cache_root_path.join("logs");
        let completed_log_path = logs_dir.join("completed_sources.txt");
        fs::create_dir_all(&task_results_dir).map_err(|e| format!("创建 task_results 目录失败: {}", e))?;
        fs::create_dir_all(&logs_dir).map_err(|e| format!("创建 logs 目录失败: {}", e))?;

        let shared = SharedInputs {
            dates: Arc::new(dates),
            stocks: Arc::new(stocks),
            windows: Arc::new(windows),
            fold,
            min_valid,
            backtest_start,
            legacy_style_data: Arc::new(
                IOOptimizedStyleData::load_from_vars_h5(&style_vars_dir)
                    .map_err(|e| e.to_string())?
            ),
            industry_neutralize: true,
            industry: None,
            neutralize_std_shared: None,
            ret_gap1: Arc::new(read_npy(&ret_gap1_path).map_err(|e| format!("读取 ret_gap1.npy 失败: {}", e))?),
            ret_sum_gap1: Arc::new(read_npy(&ret_sum_gap1_path).map_err(|e| format!("读取 ret_sum_gap1.npy 失败: {}", e))?),
            ret_gap5: Arc::new(read_npy(&ret_gap5_path).map_err(|e| format!("读取 ret_gap5.npy 失败: {}", e))?),
            ret_sum_gap5: Arc::new(read_npy(&ret_sum_gap5_path).map_err(|e| format!("读取 ret_sum_gap5.npy 失败: {}", e))?),
            restrict: Arc::new(read_npy(&restrict_path).map_err(|e| format!("读取 restrict.npy 失败: {}", e))?),
            index_ret: Arc::new(read_npy(&index_ret_path).map_err(|e| format!("读取 index_ret.npy 失败: {}", e))?),
            config: Arc::new(TailSelectionConfig {
                cover_rate,
                ret_point_neu_gap5,
                ret_point_neu_gap1,
                ic_point_neu_gap5,
                ic_point_neu_gap1,
                ret_point_gap5,
                ret_point_gap1,
                ic_point_gap5,
                ic_point_gap1,
                ic_more_important_gap5,
                ic_more_important_gap1,
                majority_count_threshold,
                zero_max_threshold,
                nan_max_threshold,
                save_all_metrics: false,
                ic_only: false,
            }),
            bt_pre: None,
        free_mask: None,
        v3_shared: None,
        };

        let mut aggregated = AggregatedCandidates::default();
        let mut completed_sources = HashSet::<String>::new();
        let mut stats = ProcessStats::default();
        for (source_factor, factor_path) in factor_names.iter().zip(factor_paths.iter()) {
            let result_path = factor_result_path(&task_results_dir, source_factor);
            if result_path.exists() {
                if let Ok(mut task_result) = read_task_result(&result_path) {
                    if !task_result.passed
                        && (!task_result.raw_summary_gap1.is_empty()
                            || !task_result.raw_summary_gap5.is_empty()
                            || !task_result.neu_summary_gap1.is_empty()
                            || !task_result.neu_summary_gap5.is_empty())
                    {
                        task_result.passed = true;
                    }
                    if task_result.passed {
                        stats.restored_pass += 1;
                    } else if task_result.eliminated_by_raw_cover {
                        stats.restored_raw_cov += 1;
                    } else if !task_result.any_window_passed_preflight
                        && !task_result.raw_summary_gap1.is_empty() == false
                        && !task_result.raw_summary_gap5.is_empty() == false
                        && !task_result.neu_summary_gap1.is_empty() == false
                        && !task_result.neu_summary_gap5.is_empty() == false
                        && !task_result.passed
                    {
                        // 区分 preflight 淘汰 vs 未知:
                        // 旧缓存没有 any_window_passed_preflight(默认false)
                        // 也没有 eliminated_by_raw_cover(默认false)
                        // 如果所有 summary vecs 都为空且 passed=false
                        // 我们通过是否有 preflight 失败记录来判断
                        if task_result.preflight_maj_failed_windows > 0
                            || task_result.preflight_zero_failed_windows > 0
                            || task_result.preflight_nan_failed_windows > 0
                        {
                            stats.restored_preflight += 1;
                        } else {
                            stats.restored_unknown += 1;
                        }
                    } else {
                        // any_window_passed_preflight=true, passed=false → ret_ic淘汰
                        stats.restored_ret_ic += 1;
                    }
                    stats.preflight_maj_windows += task_result.preflight_maj_failed_windows;
                    stats.preflight_zero_windows += task_result.preflight_zero_failed_windows;
                    stats.preflight_nan_windows += task_result.preflight_nan_failed_windows;
                    aggregated.merge_task(task_result);
                    completed_sources.insert(source_factor.clone());
                    continue;
                }
            }
            let _ = factor_path;
        }
        let restored_sources =
            stats.restored_pass + stats.restored_raw_cov + stats.restored_preflight
            + stats.restored_ret_ic + stats.restored_unknown;

        let pending_tasks = factor_names
            .iter()
            .zip(factor_paths.iter())
            .filter_map(|(source_factor, factor_path)| {
                if completed_sources.contains(source_factor) {
                    None
                } else {
                    Some(TailTask {
                        source_factor: source_factor.clone(),
                        factor_path: factor_path.clone(),
                    })
                }
            })
            .collect::<Vec<_>>();

        let total_pending = pending_tasks.len();
        if total_pending > 0 {
            init_status_line();
            let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            let total = factor_names.len();
            let total_elim = restored_sources - stats.restored_pass;
            let l1 = format!("[{}] Tail V4 启动，待处理 {}/{} 个原始因子", current_time, total_pending, total);
            let l2 = format!("累计通过 {} ({}%) | 淘汰 raw_cov={} preflight={} ret_ic={} 未知={}",
                stats.restored_pass,
                if restored_sources > 0 { stats.restored_pass * 100 / restored_sources } else { 0 },
                stats.restored_raw_cov, stats.restored_preflight,
                stats.restored_ret_ic, stats.restored_unknown);
            let _ = total_elim; // suppress unused warning
            let l3 = format!("maj={}w zero={}w nan={}w | 恢复 {} 个 | 即将开始处理...",
                stats.preflight_maj_windows, stats.preflight_zero_windows,
                stats.preflight_nan_windows, restored_sources);
            update_status_line(&l1, &l2, &l3);
        }
        let (task_sender, task_receiver): (Sender<TailTask>, Receiver<TailTask>) = unbounded();
        let (result_sender, result_receiver) = unbounded::<Result<TailTaskResult, (String, String)>>();
        for task in pending_tasks {
            task_sender.send(task).map_err(|e| format!("发送任务失败: {}", e))?;
        }
        drop(task_sender);

        let shared_arc = Arc::new(shared);

        // 检测是否为列式存储模式（任一 task 路径含 "::"）。
        // 若是，启用 IO/CPU 分离架构：少量 IO 线程顺序读因子 → 有界内存队列 → n_jobs 计算线程。
        // 否则保持原逻辑：n_jobs 线程各自读盘 + 计算（兼容 parquet/h5）。
        let is_colblk_mode = factor_paths.iter().any(|p| p.contains("::"));

        // V7 批量顺序 pread：IO 线程一次读 N 个连续因子的投影段（顺序 pread + fadvise）
        // 消除 HDD 随机寻道。计算线程零改动（复用 process_task_with_values）。
        if is_colblk_mode {
            let (loaded_tx, loaded_rx) =
                crossbeam::channel::bounded::<(TailTask, ndarray::Array2<f32>)>(16);

            let store_dir_for_reader = factor_paths
                .iter()
                .find_map(|p| {
                    if p.contains("::") {
                        let sp = p.splitn(2, "::").next().unwrap_or("");
                        if sp.ends_with(".colblk") {
                            Path::new(sp).parent().map(|x| x.to_string_lossy().to_string())
                        } else {
                            Some(sp.to_string())
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            // col_idx → task 映射
            let mut col_idx_to_task: HashMap<usize, TailTask> = HashMap::new();
            for (source_factor, factor_path) in factor_names.iter().zip(factor_paths.iter()) {
                if let Some(s) = factor_path.splitn(2, "::").nth(1) {
                    if let Ok(col_idx) = s.parse::<usize>() {
                        col_idx_to_task.insert(
                            col_idx,
                            TailTask {
                                source_factor: source_factor.clone(),
                                factor_path: factor_path.clone(),
                            },
                        );
                    }
                }
            }
            let mut all_col_idx: Vec<usize> = col_idx_to_task.keys().copied().collect();
            all_col_idx.sort_unstable();
            let total_factors = all_col_idx.len();

            // V7 批量大小：一次 pread 读 BATCH 个连续因子（投影段连续）
            const V7_BATCH: usize = 50;
            eprintln!(
                "🚀 V7 批量顺序 pread：{} 因子，每 {} 个一批（顺序 pread + fadvise）",
                total_factors, V7_BATCH
            );

            let io_handle = {
                let dates = shared_arc.dates.clone();
                let stocks = shared_arc.stocks.clone();
                thread::spawn(move || {
                    let reader = match crate::factor_store_v5::FactorStoreReader::open(
                        &store_dir_for_reader,
                    ) {
                        Ok(r) => r,
                        Err(_) => return,
                    };
                    let scatter_maps = reader.precompute_scatter_maps(
                        dates.as_slice(),
                        stocks.as_slice(),
                    );
                    let mut start = 0usize;
                    while start < total_factors {
                        let col_start = all_col_idx[start];
                        let end = (start + V7_BATCH).min(total_factors);
                        let col_end = all_col_idx[end - 1] + 1; // 连续区间
                        // V7 批量 pread：一次读 [col_start, col_end) 的投影段
                        let matrices = match reader.read_factors_batch_v7_fast(
                            col_start,
                            col_end,
                            dates.as_slice(),
                            stocks.as_slice(),
                            &scatter_maps,
                        ) {
                            Ok(m) => m,
                            Err(e) => {
                                eprintln!("⚠️ V7 批量 pread 失败 [{col_start},{col_end}): {e}");
                                start = end;
                                continue;
                            }
                        };
                        // 逐因子推 channel
                        for (bi, matrix) in matrices.into_iter().enumerate() {
                            let col_idx = col_start + bi;
                            if let Some(task) = col_idx_to_task.get(&col_idx) {
                                if loaded_tx.send((task.clone(), matrix)).is_err() {
                                    return;
                                }
                            }
                        }
                        start = end;
                    }
                })
            };

            // n_jobs 计算线程（零改动）
            let mut handles = Vec::with_capacity(n_jobs);
            for _ in 0..n_jobs {
                let loaded_rx = loaded_rx.clone();
                let tx = result_sender.clone();
                let shared_clone = shared_arc.clone();
                handles.push(thread::spawn(move || {
                    while let Ok((task, raw_values)) = loaded_rx.recv() {
                        let task_name = task.source_factor.clone();
                        let outcome = process_task_with_values_v7(&task, raw_values, &shared_clone)
                            .map_err(|err| (task_name, err));
                        if tx.send(outcome).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(result_sender);

            let _ = io_handle.join();
            for h in handles {
                let _ = h.join();
            }
        } else {
            // ---- 非 colblk 模式：原架构 ----
            let mut handles = Vec::with_capacity(n_jobs);
            for _ in 0..n_jobs {
                let rx = task_receiver.clone();
                let tx = result_sender.clone();
                let shared_clone = shared_arc.clone();
                handles.push(thread::spawn(move || {
                    while let Ok(task) = rx.recv() {
                        let task_name = task.source_factor.clone();
                        let outcome = process_task(&task, &shared_clone)
                            .map_err(|err| (task_name, err));
                        if tx.send(outcome).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(result_sender);
            for h in handles {
                let _ = h.join();
            }
        }

        let mut processed_sources = 0usize;
        while let Ok(task_outcome) = result_receiver.recv() {
            match task_outcome {
                Ok(task_result) => {
                    let result_path = factor_result_path(&task_results_dir, &task_result.source_factor);
                    let is_passed = task_result.passed;
                    let is_raw_cov = task_result.eliminated_by_raw_cover;
                    let any_window = task_result.any_window_passed_preflight;
                    let preflight_maj = task_result.preflight_maj_failed_windows;
                    let preflight_zero = task_result.preflight_zero_failed_windows;
                    let preflight_nan = task_result.preflight_nan_failed_windows;
                    write_task_result(&result_path, &task_result)?;
                    append_completed_source(&completed_log_path, &task_result.source_factor)?;
                    aggregated.merge_task(task_result);
                    processed_sources += 1;

                    if is_passed {
                        stats.done_pass += 1;
                    } else if is_raw_cov {
                        stats.done_raw_cov += 1;
                    } else if !any_window {
                        stats.done_preflight += 1;
                    } else {
                        stats.done_ret_ic += 1;
                    }
                    stats.done = processed_sources;
                    stats.preflight_maj_windows += preflight_maj;
                    stats.preflight_zero_windows += preflight_zero;
                    stats.preflight_nan_windows += preflight_nan;

                    if total_pending > 0 {
                        let elapsed = started.elapsed();
                        let elapsed_secs = elapsed.as_secs();
                        let progress = processed_sources as f64 / total_pending as f64;
                        let estimated_total_secs = if progress > 0.0 {
                            elapsed.as_secs_f64() / progress
                        } else {
                            elapsed.as_secs_f64()
                        };
                        let remaining_secs = if estimated_total_secs > elapsed.as_secs_f64() {
                            (estimated_total_secs - elapsed.as_secs_f64()) as u64
                        } else {
                            0
                        };
                        let (elapsed_h, elapsed_m, elapsed_s) = format_hms(elapsed_secs);
                        let (remaining_h, remaining_m, remaining_s) = format_hms(remaining_secs);
                        let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

                        let cum_pass = stats.restored_pass + stats.done_pass;
                        let cum_total = restored_sources + processed_sources;
                        let cum_raw_cov = stats.restored_raw_cov + stats.done_raw_cov;
                        let cum_preflight = stats.restored_preflight + stats.done_preflight;
                        let cum_ret_ic = stats.restored_ret_ic + stats.done_ret_ic;
                        let cum_unknown = stats.restored_unknown + stats.done_unknown;

                        let l1 = format!(
                            "[{}] Tail V4 进度 {}/{} ({:.1}%)，已用{}h{}m{}s，预计剩余{}h{}m{}s",
                            current_time, processed_sources, total_pending,
                            progress * 100.0,
                            elapsed_h, elapsed_m, elapsed_s,
                            remaining_h, remaining_m, remaining_s,
                        );
                        let l2 = format!(
                            "累计通过 {} ({:.0}%) | 淘汰 raw_cov={} preflight={} ret_ic={} 未知={}",
                            cum_pass,
                            if cum_total > 0 { cum_pass as f64 * 100.0 / cum_total as f64 } else { 0.0 },
                            cum_raw_cov, cum_preflight, cum_ret_ic, cum_unknown,
                        );
                        let l3 = format!(
                            "maj={}w zero={}w nan={}w | 本次 {}(通过{}) | 恢复 {}(通过{})",
                            stats.preflight_maj_windows, stats.preflight_zero_windows,
                            stats.preflight_nan_windows,
                            stats.done, stats.done_pass,
                            restored_sources, stats.restored_pass,
                        );
                        update_status_line(&l1, &l2, &l3);
                    }
                }
                Err((task_name, err)) => {
                    reset_status_line();
                    return Err(format!("处理因子 {} 失败: {}", task_name, err));
                }
            }
        }

        if total_pending > 0 {
            println!();
            reset_status_line();
        }

        // 注意：worker 线程在上方 IO/CPU 分离分支或原架构分支内已全部 join 完毕。

        write_aggregated_outputs(
            &cache_root_path,
            &aggregated,
        )?;

        let mut candidate_counts = HashMap::new();
        candidate_counts.insert("rolled_gap1".to_string(), aggregated.raw_summary_gap1.len());
        candidate_counts.insert("rolled_gap5".to_string(), aggregated.raw_summary_gap5.len());
        candidate_counts.insert("neu_gap1".to_string(), aggregated.neu_summary_gap1.len());
        candidate_counts.insert("neu_gap5".to_string(), aggregated.neu_summary_gap5.len());
        Ok((processed_sources, restored_sources, candidate_counts))
    }).map_err(PyRuntimeError::new_err)?;

    let info = PyDict::new(py);
    info.set_item("processed_sources", output.0)?;
    info.set_item("restored_sources", output.1)?;
    let candidate_counts = PyDict::new(py);
    for (key, value) in output.2 {
        candidate_counts.set_item(key, value)?;
    }
    info.set_item("candidate_counts", candidate_counts)?;
    Ok(info.into())
}

// ==================== V7 process：流式 rank_roll + 单 slot 中性化/回测（低内存版） ====================

/// 单 slot 的 gap1/gap5 回测。通过 insert_axis 零拷贝复用 block 版底层函数，
/// 保证与 legacy_backtest_gap1_gap5_selected_slots_f32 中同一 slot 的结果逐位一致。
fn legacy_backtest_gap1_gap5_single_slot(
    slot: ArrayView2<'_, f32>,
    ret_gap1: ArrayView2<'_, f32>,
    ret_sum_gap1: ArrayView2<'_, f32>,
    ret_gap5: ArrayView2<'_, f32>,
    ret_sum_gap5: ArrayView2<'_, f32>,
    restrict: ArrayView2<'_, f32>,
    index: ArrayView1<'_, f32>,
    dates: &[i32],
    backtest_start: i32,
    portf_num: usize,
    open_symbol_counts: &[usize],
    ic_only: bool,
) -> (LegacyBacktestResult, LegacyBacktestResult) {
    let n_dates = slot.nrows();
    let slot_block = slot.insert_axis(ndarray::Axis(2));
    if n_dates < 2 || !has_enough_unique_values(&slot_block, 0, 10) {
        return (
            default_legacy_backtest_result(),
            default_legacy_backtest_result(),
        );
    }
    let effective_raw_indices =
        effective_raw_indices_for_slot(&slot_block, dates, backtest_start, 0);
    (
        legacy_backtest_single_factor_with_effective(
            &slot_block,
            &ret_gap1,
            &ret_sum_gap1,
            &restrict,
            &index,
            dates,
            0,
            1,
            portf_num,
            &effective_raw_indices,
            open_symbol_counts,
            ic_only,
        ),
        legacy_backtest_single_factor_with_effective(
            &slot_block,
            &ret_gap5,
            &ret_sum_gap5,
            &restrict,
            &index,
            dates,
            0,
            5,
            portf_num,
            &effective_raw_indices,
            open_symbol_counts,
            ic_only,
        ),
    )
}

/// 流式处理单个 derived slot：preflight → raw 回测 → 标准中性化 → neu 回测 → 汇总。
/// 每个 slot 处理完立即释放，不再同时持有 selected block 和 neutralized block。
#[allow(clippy::too_many_arguments)]
/// raw 段 (preflight + 原始回测): 返回 None 表示 preflight 未过且非 metrics-only (提前退出)。
struct SlotRawOut {
    pre_report: PreflightReport,
    raw_gap1: LegacyBacktestResult,
    raw_gap5: LegacyBacktestResult,
}

fn process_v7_slot_raw(
    slot_idx: usize,
    slot_values: ArrayView2<'_, f32>,
    variant_name: &str,
    derived_names: &[String],
    shared: &SharedInputs,
    open_symbol_counts: &[usize],
    result: &mut TailTaskResult,
) -> Result<Option<SlotRawOut>, String> {
    let derived_name = &derived_names[slot_idx];

    // ---- preflight（与 collect_preflight_passed_slots 完全一致） ----
    let pre_report = preflight_quality_check(
        &slot_values,
        &shared.restrict.view(),
        shared.config.majority_count_threshold,
        shared.config.zero_max_threshold,
        shared.config.nan_max_threshold,
    );
    if !pre_report.passed {
        if pre_report.majority_count_mean > shared.config.majority_count_threshold {
            result.preflight_maj_failed_windows += 1;
        }
        if pre_report.zero_ratio_mean >= shared.config.zero_max_threshold {
            result.preflight_zero_failed_windows += 1;
        }
        if pre_report.nan_ratio_mean >= shared.config.nan_max_threshold {
            result.preflight_nan_failed_windows += 1;
        }
        let mut reasons: Vec<String> = Vec::new();
        if pre_report.majority_count_mean > shared.config.majority_count_threshold {
            reasons.push(format!(
                "majority_count_mean={:.2}, 标准<={:.2}",
                pre_report.majority_count_mean, shared.config.majority_count_threshold,
            ));
        }
        if pre_report.zero_ratio_mean >= shared.config.zero_max_threshold {
            reasons.push(format!(
                "zero_ratio_mean={:.4}, 标准<{:.4}",
                pre_report.zero_ratio_mean, shared.config.zero_max_threshold,
            ));
        }
        if pre_report.nan_ratio_mean >= shared.config.nan_max_threshold {
            reasons.push(format!(
                "nan_ratio_mean={:.4}, 标准<{:.4}",
                pre_report.nan_ratio_mean, shared.config.nan_max_threshold,
            ));
        }
        println!(
            "[preflight] 剔除因子 {}，该因子指标不达标: {}",
            derived_name,
            reasons.join("; "),
        );
        if !shared.config.save_all_metrics {
            return Ok(None);
        }
        // metrics-only 模式继续跑 raw/neu 回测，但 summary 中 preflight_passed=false，
        // 判定器会把这些 derived 因子视为不可入选。
    } else {
        result.any_window_passed_preflight = true;
    }

    // ---- raw gap1/gap5 回测（ic_only 模式跳过：只保留中性化 IC 路径） ----
    // O1 (2026-09): 有 bt_pre 时用 _opt 版 (收益秩预排序 walk + radix 秩, 数值逐位一致);
    // 旧入口 (tail_v5_run_candidates 等) 未构建 bt_pre 时回退原实现, 行为不变。
    let _t = Instant::now();
    let (raw_gap1_result, raw_gap5_result) = if shared.config.ic_only {
        (
            default_legacy_backtest_result(),
            default_legacy_backtest_result(),
        )
    } else {
        match &shared.bt_pre {
            Some(pre) => legacy_backtest_gap1_gap5_single_slot_opt(
                slot_values,
                shared.ret_gap1.view(),
                shared.ret_sum_gap1.view(),
                shared.ret_gap5.view(),
                shared.ret_sum_gap5.view(),
                shared.restrict.view(),
                shared.index_ret.view(),
                shared.dates.as_slice(),
                shared.backtest_start,
                10,
                open_symbol_counts,
                false,
                pre,
            ),
            None => legacy_backtest_gap1_gap5_single_slot(
                slot_values,
                shared.ret_gap1.view(),
                shared.ret_sum_gap1.view(),
                shared.ret_gap5.view(),
                shared.ret_sum_gap5.view(),
                shared.restrict.view(),
                shared.index_ret.view(),
                shared.dates.as_slice(),
                shared.backtest_start,
                10,
                open_symbol_counts,
                false,
            ),
        }
    };
    PROF_BT_RAW.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    Ok(Some(SlotRawOut {
        pre_report,
        raw_gap1: raw_gap1_result,
        raw_gap5: raw_gap5_result,
    }))
}

/// neu 段 (中性化后回测 + 汇总 + 入结果)。
#[allow(clippy::too_many_arguments)]
fn process_v7_slot_neu(
    slot_idx: usize,
    neutralized_slot: &Array2<f32>,
    raw_out: SlotRawOut,
    variant_name: &str,
    derived_names: &[String],
    shared: &SharedInputs,
    open_symbol_counts: &[usize],
    result: &mut TailTaskResult,
) -> Result<(), String> {
    let derived_name = &derived_names[slot_idx];
    let pre_report = raw_out.pre_report;
    let raw_gap1_result = raw_out.raw_gap1;
    let raw_gap5_result = raw_out.raw_gap5;

    // ---- neu gap1/gap5 回测 ----
    let _t = Instant::now();
    let (neu_gap1_result, neu_gap5_result) = match &shared.bt_pre {
        Some(pre) => legacy_backtest_gap1_gap5_single_slot_opt(
            neutralized_slot.view(),
            shared.ret_gap1.view(),
            shared.ret_sum_gap1.view(),
            shared.ret_gap5.view(),
            shared.ret_sum_gap5.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            shared.dates.as_slice(),
            shared.backtest_start,
            10,
            open_symbol_counts,
            shared.config.ic_only,
            pre,
        ),
        None => legacy_backtest_gap1_gap5_single_slot(
            neutralized_slot.view(),
            shared.ret_gap1.view(),
            shared.ret_sum_gap1.view(),
            shared.ret_gap5.view(),
            shared.ret_sum_gap5.view(),
            shared.restrict.view(),
            shared.index_ret.view(),
            shared.dates.as_slice(),
            shared.backtest_start,
            10,
            open_symbol_counts,
            shared.config.ic_only,
        ),
    };
    PROF_BT_NEU.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);

    // ---- 汇总（顺序/条件与旧实现完全一致） ----
    let mut raw_gap1_row = summary_from_row(
        derived_name,
        "rolled",
        1,
        variant_name,
        &raw_gap1_result.summary,
    );
    let mut raw_gap5_row = summary_from_row(
        derived_name,
        "rolled",
        5,
        variant_name,
        &raw_gap5_result.summary,
    );
    let mut neu_gap1_row = summary_from_row(
        derived_name,
        "neu",
        1,
        variant_name,
        &neu_gap1_result.summary,
    );
    let mut neu_gap5_row = summary_from_row(
        derived_name,
        "neu",
        5,
        variant_name,
        &neu_gap5_result.summary,
    );
    raw_gap1_row.preflight_passed = pre_report.passed;
    raw_gap5_row.preflight_passed = pre_report.passed;
    neu_gap1_row.preflight_passed = pre_report.passed;
    neu_gap5_row.preflight_passed = pre_report.passed;

    let raw_gap1_keep = pre_report.passed && qualify_raw(&raw_gap1_row, 1, &shared.config);
    let raw_gap5_keep = pre_report.passed && qualify_raw(&raw_gap5_row, 5, &shared.config);
    let neu_gap1_keep = pre_report.passed && qualify_neu(&neu_gap1_row, 1, &shared.config);
    let neu_gap5_keep = pre_report.passed && qualify_neu(&neu_gap5_row, 5, &shared.config);

    if raw_gap1_keep || raw_gap5_keep || neu_gap1_keep || neu_gap5_keep {
        result.passed = true;
    }

    // metrics-only：无论是否达到候选阈值，都保存该 derived slot 的全部指标。
    // ic_only 模式 raw 未回测（summary 为 NaN），跳过 raw all 产物，避免写含 NaN 的 JSON。
    if shared.config.save_all_metrics {
        if !shared.config.ic_only {
            result.all_raw_summary_gap1.push(raw_gap1_row.clone());
            result.all_raw_summary_gap5.push(raw_gap5_row.clone());
            result.all_raw_ic_gap1.push(IcRecord {
                factor_name: derived_name.clone(),
                dates: raw_gap1_result.ic_dates.clone(),
                values: raw_gap1_result.ic_values.clone(),
            });
            result.all_raw_ic_gap5.push(IcRecord {
                factor_name: derived_name.clone(),
                dates: raw_gap5_result.ic_dates.clone(),
                values: raw_gap5_result.ic_values.clone(),
            });
        }
        result.all_neu_summary_gap1.push(neu_gap1_row.clone());
        result.all_neu_summary_gap5.push(neu_gap5_row.clone());
        result.all_neu_ic_gap1.push(IcRecord {
            factor_name: derived_name.clone(),
            dates: neu_gap1_result.ic_dates.clone(),
            values: neu_gap1_result.ic_values.clone(),
        });
        result.all_neu_ic_gap5.push(IcRecord {
            factor_name: derived_name.clone(),
            dates: neu_gap5_result.ic_dates.clone(),
            values: neu_gap5_result.ic_values.clone(),
        });
    }

    if raw_gap1_keep {
        result.raw_summary_gap1.push(raw_gap1_row.clone());
        result.raw_ic_gap1.push(IcRecord {
            factor_name: derived_name.clone(),
            dates: raw_gap1_result.ic_dates.clone(),
            values: raw_gap1_result.ic_values.clone(),
        });
    }
    if raw_gap5_keep {
        result.raw_summary_gap5.push(raw_gap5_row.clone());
        result.raw_ic_gap5.push(IcRecord {
            factor_name: derived_name.clone(),
            dates: raw_gap5_result.ic_dates.clone(),
            values: raw_gap5_result.ic_values.clone(),
        });
    }
    if neu_gap1_keep {
        result.neu_summary_gap1.push(neu_gap1_row.clone());
    }
    if neu_gap5_keep {
        result.neu_summary_gap5.push(neu_gap5_row.clone());
    }
    if raw_gap1_keep || neu_gap1_keep {
        result.neu_ic_gap1.push(IcRecord {
            factor_name: derived_name.clone(),
            dates: neu_gap1_result.ic_dates.clone(),
            values: neu_gap1_result.ic_values.clone(),
        });
    }
    if raw_gap5_keep || neu_gap5_keep {
        result.neu_ic_gap5.push(IcRecord {
            factor_name: derived_name.clone(),
            dates: neu_gap5_result.ic_dates.clone(),
            values: neu_gap5_result.ic_values.clone(),
        });
    }
    Ok(())
}

/// B=1 单面流程 (slot 0 等): raw → 中性化 → neu。行为与旧 process_v7_slot 完全一致。
fn process_v7_slot(
    slot_idx: usize,
    slot_values: ArrayView2<'_, f32>,
    variant_name: &str,
    derived_names: &[String],
    shared: &SharedInputs,
    open_symbol_counts: &[usize],
    result: &mut TailTaskResult,
) -> Result<(), String> {
    let raw_out = match process_v7_slot_raw(
        slot_idx,
        slot_values,
        variant_name,
        derived_names,
        shared,
        open_symbol_counts,
        result,
    )? {
        Some(o) => o,
        None => return Ok(()), // preflight 未过且非 metrics-only
    };
    // ---- 标准中性化：只物化当前 slot 的 (T,N) ----
    // O2 (2026-09): precompute 已带 orders/per_date, v2 用预计算排序与逐日 X'X
    // (数值逐位一致; 因子侧 NaN 日自动回退生产原路径)。
    let _t = Instant::now();
    let neutralized_slot = {
        let ns = shared
            .neutralize_std_shared
            .as_ref()
            .expect("tail_backtest_engine 中性化需要预计算的 neutralize_std_shared");
        match &shared.bt_pre {
            Some(_) => crate::factor_neutralize_std::neutralize_std_slot_f32_v2_resid(
                slot_values,
                ns,
                shared.industry_neutralize,
            )?,
            None => crate::factor_neutralize_std::neutralize_std_slot_f32(
                slot_values,
                ns,
                shared.industry_neutralize,
            )?,
        }
    };
    PROF_NEU.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    process_v7_slot_neu(
        slot_idx,
        &neutralized_slot,
        raw_out,
        variant_name,
        derived_names,
        shared,
        open_symbol_counts,
        result,
    )
}

/// P3: B 个面按日期批量中性化 (组内 raw 段先行, 中性化一次批处理, neu 段后行)。
fn process_v7_slots_batch(
    group: &[(usize, Array2<f32>)],
    variant_name: &str,
    derived_names: &[String],
    shared: &SharedInputs,
    open_symbol_counts: &[usize],
    result: &mut TailTaskResult,
) -> Result<(), String> {
    // Phase A: preflight + raw bt (保持 slot 矩阵存活)
    let mut raw_outs: Vec<Option<SlotRawOut>> = Vec::with_capacity(group.len());
    for (slot_idx, matrix) in group.iter() {
        let raw = process_v7_slot_raw(
            *slot_idx,
            matrix.view(),
            variant_name,
            derived_names,
            shared,
            open_symbol_counts,
            result,
        )?;
        raw_outs.push(raw);
    }
    // Phase B: 对需要中性化的面做批处理 (preflight 未过且非 metrics-only 的不需要)
    let need: Vec<usize> = (0..group.len())
        .filter(|&i| raw_outs[i].is_some())
        .collect();
    if !need.is_empty() {
        let ns = shared
            .neutralize_std_shared
            .as_ref()
            .expect("tail_backtest_engine 中性化需要预计算的 neutralize_std_shared");
        let _t = Instant::now();
        let views: Vec<ArrayView2<f32>> = need.iter().map(|&i| group[i].1.view()).collect();
        let neutrals = match &shared.bt_pre {
            Some(_) => crate::factor_neutralize_std::neutralize_std_slots_f32_v2_resid_batch(
                &views,
                ns,
                shared.industry_neutralize,
            )?,
            None => views
                .iter()
                .map(|v| {
                    crate::factor_neutralize_std::neutralize_std_slot_f32(
                        v.clone(),
                        ns,
                        shared.industry_neutralize,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?,
        };
        PROF_NEU.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        // Phase C: neu 段
        for (k, &i) in need.iter().enumerate() {
            let raw_out = raw_outs[i].take().expect("need 已过滤 Some");
            process_v7_slot_neu(
                group[i].0,
                &neutrals[k],
                raw_out,
                variant_name,
                derived_names,
                shared,
                open_symbol_counts,
                result,
            )?;
        }
    }
    Ok(())
}

/// P3 批大小 (进程级缓存; env TAIL_BATCH_B, 缺省 4)。
fn batch_size_b() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("TAIL_BATCH_B").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(4))
}

/// 把 rank 后的缺失值用当日横截面中位 rank 填充。
/// 只填"缺失"（当日 Restrict 显示可交易的股票当天的缺口）；"不适用"保持 NaN。
/// Restrict 语义：0=可交易（正常，当日缺口视为"缺失"应填充）；1=不可交易
/// （停牌∪涨跌停，当日无数据是合理的"不适用"）；NaN=未上市/数据不存在（不适用）。
/// 不可用位置不是"缺失"，填充必须跳过——否则每天会给宇宙外股票凭空编造中位 rank
/// （全模板 9000+ 只"有值"），打爆 preflight 众数统计并污染覆盖率诊断。
/// 平均 rank 的取值是 1..=n_valid，中位数就是 (n_valid+1)/2。
/// `restrict` 是与 ranked 同形状的限制状态矩阵（来自 S_RESTRICT）。
fn fill_missing_rank_with_cross_sectional_median(ranked: &mut Array2<f32>, restrict: &Array2<f32>) {
    let n_dates = ranked.nrows();
    let n_stocks = ranked.ncols();
    for date_idx in 0..n_dates {
        let mut valid_count = 0usize;
        for stock_idx in 0..n_stocks {
            if ranked[[date_idx, stock_idx]].is_finite() {
                valid_count += 1;
            }
        }
        if valid_count == 0 {
            continue;
        }
        let median_rank = ((valid_count + 1) as f32) / 2.0;
        for stock_idx in 0..n_stocks {
            if !ranked[[date_idx, stock_idx]].is_finite() {
                let is_normal = restrict[[date_idx, stock_idx]] == 0.0;
                if is_normal {
                    ranked[[date_idx, stock_idx]] = median_rank;
                }
            }
        }
    }
}

/// rank + 缺失值填充（回测预处理第二层保障）。
/// 顺序约定：先横截面 rank，再填充"缺失"的 rank（Restrict 可交易股票当天的缺口）；
/// 不可交易（停牌/涨跌停）与未上市（不适用）保持 NaN。随后才做 raw_cover / preflight / 回测。
pub(crate) fn rank_and_fill_missing_cross_sectional_median(
    variant_values: &Array2<f32>,
    restrict: &Array2<f32>,
) -> Array2<f32> {
    let mut ranked =
        crate::tail_v2_rank_roll_factor::rank_axis1_average_f32_serial(variant_values);
    fill_missing_rank_with_cross_sectional_median(&mut ranked, restrict);
    ranked
}

/// 流式处理一个 variant（raw 或 _fold）。
/// 不再先 rank_roll 成 13 面大 block，而是逐 slot 生成、处理、释放。
/// `ranked` 已由调用方完成 rank + 缺失填充。
fn process_v7_variant(
    variant_name: &str,
    ranked: Array2<f32>,
    shared: &SharedInputs,
    open_symbol_counts: &[usize],
    result: &mut TailTaskResult,
) -> Result<(), String> {
    let (n_dates, n_stocks) = ranked.dim();
    if shared.ret_gap1.dim() != (n_dates, n_stocks)
        || shared.ret_sum_gap1.dim() != (n_dates, n_stocks)
        || shared.ret_gap5.dim() != (n_dates, n_stocks)
        || shared.ret_sum_gap5.dim() != (n_dates, n_stocks)
        || shared.restrict.dim() != (n_dates, n_stocks)
        || shared.index_ret.len() != n_dates
        || shared.dates.len() != n_dates
    {
        return Err("legacy backtest 输入形状不匹配".to_string());
    }

    let derived_names = derived_names_for_variant(variant_name, shared.windows.as_slice());
    result.derived_factor_count += derived_names.len();

    let mut slot_idx = 0usize;

    // slot 0: _smooth_1 = ranked 本身
    process_v7_slot(
        slot_idx,
        ranked.view(),
        variant_name,
        &derived_names,
        shared,
        open_symbol_counts,
        result,
    )?;
    slot_idx += 1;

    // window 派生 slot：P3 小批量多面 —— 按 TAIL_BATCH_B 分组的流水线
    // (rolling 面按序入缓冲, 满 B 个即 batch 处理; 组内 raw → 批中性化 → neu;
    //  与逐面处理逐位一致)。
    let b = batch_size_b().max(1);
    let mut buffer: Vec<(usize, Array2<f32>)> = Vec::with_capacity(b);
    for &window in shared.windows.as_slice() {
        if window == 0 {
            return Err("window 必须大于 0".to_string());
        }
        let min_periods = std::cmp::max(1, window / 2);
        let (mean, max, min, std) =
            crate::tail_v2_rank_roll_factor::rolling_stats_f32_serial(&ranked, window, min_periods);
        for (slot_mat, stat_idx) in [(mean, 0usize), (max, 1), (min, 2), (std, 3)].into_iter() {
            buffer.push((slot_idx + stat_idx, slot_mat));
            if buffer.len() >= b {
                process_v7_slots_batch(
                    &buffer,
                    variant_name,
                    &derived_names,
                    shared,
                    open_symbol_counts,
                    result,
                )?;
                buffer.clear();
            }
        }
        slot_idx += 4;
    }
    if !buffer.is_empty() {
        process_v7_slots_batch(
            &buffer,
            variant_name,
            &derived_names,
            shared,
            open_symbol_counts,
            result,
        )?;
    }
    drop(ranked);
    Ok(())
}

pub(crate) fn process_task_with_values_v7(
    task: &TailTask,
    raw_values: Array2<f32>,
    shared: &SharedInputs,
) -> Result<TailTaskResult, String> {
    let _t = Instant::now();

    // 填充前覆盖率：用于识别“局部宇宙因子伪装成全市场因子”。
    let raw_cover_before_fill = compute_raw_cover_rate(
        &raw_values.view(),
        &shared.restrict.view(),
        &shared.ret_gap1.view(),
        10,
    );

    // 回测前预处理（第二层覆盖率保障）：先横截面 rank + 填充缺失 rank，
    // 再做 raw_cover / preflight。填充只针对"缺失"（当日 Restrict=0 可交易、
    // 仅当天缺值）；不可交易（停牌/涨跌停）与未上市（不适用）保持 NaN，绝不编造。
    let ranked_raw = rank_and_fill_missing_cross_sectional_median(&raw_values, &shared.restrict);

    let raw_cover_rate = compute_raw_cover_rate(
        &ranked_raw.view(),
        &shared.restrict.view(),
        &shared.ret_gap1.view(),
        10,
    );
    PROF_RAW_COVER.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    if raw_cover_rate < shared.config.cover_rate {
        println!(
            "[raw_cover] 剔除因子 {}，填充后覆盖率仍不达标: raw_cover_rate={:.4}, 标准>={:.4}",
            task.source_factor, raw_cover_rate, shared.config.cover_rate,
        );
        let _n = PROF_COUNT.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if _n % 20 == 0 {
            prof_dump("raw_cover剔除");
        }
        return Ok(TailTaskResult {
            source_factor: task.source_factor.clone(),
            eliminated_by_raw_cover: true,
            raw_cover_before_fill,
            raw_cover_after_fill: raw_cover_rate,
            ..TailTaskResult::default()
        });
    }

    let mut result = TailTaskResult {
        source_factor: task.source_factor.clone(),
        raw_cover_before_fill,
        raw_cover_after_fill: raw_cover_rate,
        ..TailTaskResult::default()
    };

    // open_symbol_counts 只依赖 restrict，整个任务算一次，raw/neu 回测共用。
    let open_symbol_counts = precompute_open_symbol_counts(&shared.restrict.view());

    // raw variant
    process_v7_variant(
        task.source_factor.as_str(),
        ranked_raw,
        shared,
        &open_symbol_counts,
        &mut result,
    )?;

    // fold variant。fold 矩阵用完 raw 后立刻释放，不再让 raw + folded 全程共存。
    if shared.fold {
        let _t = Instant::now();
        let folded = build_fold_values(&raw_values);
        let ranked_fold = rank_and_fill_missing_cross_sectional_median(&folded, &shared.restrict);
        PROF_FOLD.fetch_add(_t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        drop(raw_values);
        drop(folded);
        process_v7_variant(
            &format!("{}_fold", task.source_factor),
            ranked_fold,
            shared,
            &open_symbol_counts,
            &mut result,
        )?;
    }

    let _n = PROF_COUNT.fetch_add(1, AtomicOrdering::Relaxed) + 1;
    if _n % 20 == 0 {
        prof_dump("完整回测");
    }
    Ok(result)
}

// ==================== V7b：原版 8 IO 线程 + V7 process 优化 ====================
// ==================== V7：批量顺序 pread 回测（消除 HDD 寻道，回测核心零改动）====================
#[pyfunction]
#[pyo3(signature = (
    factor_names,
    factor_paths,
    dates,
    stocks,
    windows,
    fold,
    n_jobs,
    min_valid,
    cache_root,
    style_vars_dir,
    ret_gap1_path,
    ret_sum_gap1_path,
    ret_gap5_path,
    ret_sum_gap5_path,
    restrict_path,
    index_ret_path,
    backtest_start,
    cover_rate=0.97,
    ret_point_neu_gap5=0.055,
    ret_point_neu_gap1=0.08,
    ic_point_neu_gap5=0.01,
    ic_point_neu_gap1=0.006,
    ret_point_gap5=0.1,
    ret_point_gap1=0.13,
    ic_point_gap5=0.03,
    ic_point_gap1=0.02,
    ic_more_important_gap5=0.01,
    ic_more_important_gap1=0.006,
    majority_count_threshold=200.0,
    zero_max_threshold=0.01,
    nan_max_threshold=0.04,
))]
pub fn tail_v5_run_candidates_v7b<'py>(
    py: Python<'py>,
    factor_names: Vec<String>,
    factor_paths: Vec<String>,
    dates: Vec<i32>,
    stocks: Vec<String>,
    windows: Vec<usize>,
    fold: bool,
    n_jobs: usize,
    min_valid: usize,
    cache_root: String,
    style_vars_dir: String,
    ret_gap1_path: String,
    ret_sum_gap1_path: String,
    ret_gap5_path: String,
    ret_sum_gap5_path: String,
    restrict_path: String,
    index_ret_path: String,
    backtest_start: i32,
    cover_rate: f64,
    ret_point_neu_gap5: f64,
    ret_point_neu_gap1: f64,
    ic_point_neu_gap5: f64,
    ic_point_neu_gap1: f64,
    ret_point_gap5: f64,
    ret_point_gap1: f64,
    ic_point_gap5: f64,
    ic_point_gap1: f64,
    ic_more_important_gap5: Option<f64>,
    ic_more_important_gap1: Option<f64>,
    majority_count_threshold: f64,
    zero_max_threshold: f64,
    nan_max_threshold: f64,
) -> PyResult<PyObject> {
    if factor_names.len() != factor_paths.len() {
        return Err(PyValueError::new_err(
            "factor_names 和 factor_paths 长度必须一致",
        ));
    }
    if n_jobs == 0 {
        return Err(PyValueError::new_err("n_jobs 必须大于 0"));
    }
    if (ic_more_important_gap5.is_some()) != (ic_more_important_gap1.is_some()) {
        return Err(PyValueError::new_err(
            "ic_more_important_gap5 和 ic_more_important_gap1 必须同时有值或同时为 None",
        ));
    }

    let output = py.allow_threads(|| -> Result<(usize, usize, HashMap<String, usize>), String> {
        let started = Instant::now();
        let cache_root_path = PathBuf::from(&cache_root);
        let task_results_dir = cache_root_path.join("task_results");
        let logs_dir = cache_root_path.join("logs");
        let completed_log_path = logs_dir.join("completed_sources.txt");
        fs::create_dir_all(&task_results_dir).map_err(|e| format!("创建 task_results 目录失败: {}", e))?;
        fs::create_dir_all(&logs_dir).map_err(|e| format!("创建 logs 目录失败: {}", e))?;

        let shared = SharedInputs {
            dates: Arc::new(dates),
            stocks: Arc::new(stocks),
            windows: Arc::new(windows),
            fold,
            min_valid,
            backtest_start,
            legacy_style_data: Arc::new(
                IOOptimizedStyleData::load_from_vars_h5(&style_vars_dir)
                    .map_err(|e| e.to_string())?
            ),
            industry_neutralize: true,
            industry: None,
            neutralize_std_shared: None,
            ret_gap1: Arc::new(read_npy(&ret_gap1_path).map_err(|e| format!("读取 ret_gap1.npy 失败: {}", e))?),
            ret_sum_gap1: Arc::new(read_npy(&ret_sum_gap1_path).map_err(|e| format!("读取 ret_sum_gap1.npy 失败: {}", e))?),
            ret_gap5: Arc::new(read_npy(&ret_gap5_path).map_err(|e| format!("读取 ret_gap5.npy 失败: {}", e))?),
            ret_sum_gap5: Arc::new(read_npy(&ret_sum_gap5_path).map_err(|e| format!("读取 ret_sum_gap5.npy 失败: {}", e))?),
            restrict: Arc::new(read_npy(&restrict_path).map_err(|e| format!("读取 restrict.npy 失败: {}", e))?),
            index_ret: Arc::new(read_npy(&index_ret_path).map_err(|e| format!("读取 index_ret.npy 失败: {}", e))?),
            config: Arc::new(TailSelectionConfig {
                cover_rate,
                ret_point_neu_gap5,
                ret_point_neu_gap1,
                ic_point_neu_gap5,
                ic_point_neu_gap1,
                ret_point_gap5,
                ret_point_gap1,
                ic_point_gap5,
                ic_point_gap1,
                ic_more_important_gap5,
                ic_more_important_gap1,
                majority_count_threshold,
                zero_max_threshold,
                nan_max_threshold,
                save_all_metrics: false,
                ic_only: false,
            }),
            bt_pre: None,
        free_mask: None,
        v3_shared: None,
        };

        let mut aggregated = AggregatedCandidates::default();
        let mut completed_sources = HashSet::<String>::new();
        let mut stats = ProcessStats::default();
        for (source_factor, factor_path) in factor_names.iter().zip(factor_paths.iter()) {
            let result_path = factor_result_path(&task_results_dir, source_factor);
            if result_path.exists() {
                if let Ok(mut task_result) = read_task_result(&result_path) {
                    if !task_result.passed
                        && (!task_result.raw_summary_gap1.is_empty()
                            || !task_result.raw_summary_gap5.is_empty()
                            || !task_result.neu_summary_gap1.is_empty()
                            || !task_result.neu_summary_gap5.is_empty())
                    {
                        task_result.passed = true;
                    }
                    if task_result.passed {
                        stats.restored_pass += 1;
                    } else if task_result.eliminated_by_raw_cover {
                        stats.restored_raw_cov += 1;
                    } else if !task_result.any_window_passed_preflight
                        && !task_result.raw_summary_gap1.is_empty() == false
                        && !task_result.raw_summary_gap5.is_empty() == false
                        && !task_result.neu_summary_gap1.is_empty() == false
                        && !task_result.neu_summary_gap5.is_empty() == false
                        && !task_result.passed
                    {
                        // 区分 preflight 淘汰 vs 未知:
                        // 旧缓存没有 any_window_passed_preflight(默认false)
                        // 也没有 eliminated_by_raw_cover(默认false)
                        // 如果所有 summary vecs 都为空且 passed=false
                        // 我们通过是否有 preflight 失败记录来判断
                        if task_result.preflight_maj_failed_windows > 0
                            || task_result.preflight_zero_failed_windows > 0
                            || task_result.preflight_nan_failed_windows > 0
                        {
                            stats.restored_preflight += 1;
                        } else {
                            stats.restored_unknown += 1;
                        }
                    } else {
                        // any_window_passed_preflight=true, passed=false → ret_ic淘汰
                        stats.restored_ret_ic += 1;
                    }
                    stats.preflight_maj_windows += task_result.preflight_maj_failed_windows;
                    stats.preflight_zero_windows += task_result.preflight_zero_failed_windows;
                    stats.preflight_nan_windows += task_result.preflight_nan_failed_windows;
                    aggregated.merge_task(task_result);
                    completed_sources.insert(source_factor.clone());
                    continue;
                }
            }
            let _ = factor_path;
        }
        let restored_sources =
            stats.restored_pass + stats.restored_raw_cov + stats.restored_preflight
            + stats.restored_ret_ic + stats.restored_unknown;

        let pending_tasks = factor_names
            .iter()
            .zip(factor_paths.iter())
            .filter_map(|(source_factor, factor_path)| {
                if completed_sources.contains(source_factor) {
                    None
                } else {
                    Some(TailTask {
                        source_factor: source_factor.clone(),
                        factor_path: factor_path.clone(),
                    })
                }
            })
            .collect::<Vec<_>>();

        let total_pending = pending_tasks.len();
        if total_pending > 0 {
            init_status_line();
            let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            let total = factor_names.len();
            let total_elim = restored_sources - stats.restored_pass;
            let l1 = format!("[{}] Tail V4 启动，待处理 {}/{} 个原始因子", current_time, total_pending, total);
            let l2 = format!("累计通过 {} ({}%) | 淘汰 raw_cov={} preflight={} ret_ic={} 未知={}",
                stats.restored_pass,
                if restored_sources > 0 { stats.restored_pass * 100 / restored_sources } else { 0 },
                stats.restored_raw_cov, stats.restored_preflight,
                stats.restored_ret_ic, stats.restored_unknown);
            let _ = total_elim; // suppress unused warning
            let l3 = format!("maj={}w zero={}w nan={}w | 恢复 {} 个 | 即将开始处理...",
                stats.preflight_maj_windows, stats.preflight_zero_windows,
                stats.preflight_nan_windows, restored_sources);
            update_status_line(&l1, &l2, &l3);
        }
        let (task_sender, task_receiver): (Sender<TailTask>, Receiver<TailTask>) = unbounded();
        let (result_sender, result_receiver) = unbounded::<Result<TailTaskResult, (String, String)>>();
        for task in pending_tasks {
            task_sender.send(task).map_err(|e| format!("发送任务失败: {}", e))?;
        }
        drop(task_sender);

        let shared_arc = Arc::new(shared);

        // 检测是否为列式存储模式（任一 task 路径含 "::"）。
        // 若是，启用 IO/CPU 分离架构：少量 IO 线程顺序读因子 → 有界内存队列 → n_jobs 计算线程。
        // 否则保持原逻辑：n_jobs 线程各自读盘 + 计算（兼容 parquet/h5）。
        let is_colblk_mode = factor_paths.iter().any(|p| p.contains("::"));

        // V7b：原版 8 IO 线程（每线程独立 Reader，pread 读投影区）+ V7 process 优化
        // 8 IO 线程分布在 8 块 HDD 上并行读，供料给 200 计算线程
        let mut io_handles_to_join = Vec::new();
        let worker_handles_to_join;
        if is_colblk_mode {
            let n_io_threads = if n_jobs >= 200 { 8 } else { 4 };
            let (loaded_tx, loaded_rx) =
                crossbeam::channel::bounded::<(TailTask, ndarray::Array2<f32>)>(16);

            let store_dir_for_reader = factor_paths
                .iter()
                .find_map(|p| {
                    if p.contains("::") {
                        let sp = p.splitn(2, "::").next().unwrap_or("");
                        if sp.ends_with(".colblk") {
                            Path::new(sp).parent().map(|x| x.to_string_lossy().to_string())
                        } else {
                            Some(sp.to_string())
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            eprintln!(
                "🚀 V7b：{} IO 线程 pread 投影区 + {} 计算线程（V7 process 优化）",
                n_io_threads, n_jobs
            );

            // 8 IO 线程：每线程独立 Reader，逐因子 pread 读投影区 value 段
            let mut io_handles = Vec::with_capacity(n_io_threads);
            let task_rx_io = task_receiver.clone();
            for _io_idx in 0..n_io_threads {
                let task_rx = task_rx_io.clone();
                let loaded_tx = loaded_tx.clone();
                let store_dir = store_dir_for_reader.clone();
                let dates = shared_arc.dates.clone();
                let stocks = shared_arc.stocks.clone();
                io_handles.push(thread::spawn(move || {
                    let reader = match crate::factor_store_v5::FactorStoreReader::open(
                        &store_dir,
                    ) {
                        Ok(r) => r,
                        Err(_) => return,
                    };
                    let scatter_maps = reader.precompute_scatter_maps(
                        dates.as_slice(),
                        stocks.as_slice(),
                    );
                    while let Ok(task) = task_rx.recv() {
                        let col_idx = match task.factor_path.splitn(2, "::").nth(1) {
                            Some(s) => match s.parse::<usize>() {
                                Ok(v) => v,
                                Err(_) => continue,
                            },
                            None => continue,
                        };
                        let matrix = match reader.read_factor_to_matrix_fast(
                            col_idx,
                            dates.as_slice(),
                            stocks.as_slice(),
                            &scatter_maps,
                        ) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };
                        if loaded_tx.send((task, matrix)).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(loaded_tx);

            // n_jobs 计算线程（零改动）
            let mut handles = Vec::with_capacity(n_jobs);
            for _ in 0..n_jobs {
                let loaded_rx = loaded_rx.clone();
                let tx = result_sender.clone();
                let shared_clone = shared_arc.clone();
                handles.push(thread::spawn(move || {
                    while let Ok((task, raw_values)) = loaded_rx.recv() {
                        let task_name = task.source_factor.clone();
                        let outcome = process_task_with_values_v7(&task, raw_values, &shared_clone)
                            .map_err(|err| (task_name, err));
                        if tx.send(outcome).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(result_sender);

            io_handles_to_join = io_handles;
            worker_handles_to_join = handles;
        } else {
            // ---- 非 colblk 模式：原架构 ----
            let mut handles = Vec::with_capacity(n_jobs);
            for _ in 0..n_jobs {
                let rx = task_receiver.clone();
                let tx = result_sender.clone();
                let shared_clone = shared_arc.clone();
                handles.push(thread::spawn(move || {
                    while let Ok(task) = rx.recv() {
                        let task_name = task.source_factor.clone();
                        let outcome = process_task(&task, &shared_clone)
                            .map_err(|err| (task_name, err));
                        if tx.send(outcome).is_err() {
                            break;
                        }
                    }
                }));
            }
            drop(result_sender);
            worker_handles_to_join = handles;
        }

        let mut processed_sources = 0usize;
        while let Ok(task_outcome) = result_receiver.recv() {
            match task_outcome {
                Ok(task_result) => {
                    let result_path = factor_result_path(&task_results_dir, &task_result.source_factor);
                    let is_passed = task_result.passed;
                    let is_raw_cov = task_result.eliminated_by_raw_cover;
                    let any_window = task_result.any_window_passed_preflight;
                    let preflight_maj = task_result.preflight_maj_failed_windows;
                    let preflight_zero = task_result.preflight_zero_failed_windows;
                    let preflight_nan = task_result.preflight_nan_failed_windows;
                    write_task_result(&result_path, &task_result)?;
                    append_completed_source(&completed_log_path, &task_result.source_factor)?;
                    aggregated.merge_task(task_result);
                    processed_sources += 1;

                    if is_passed {
                        stats.done_pass += 1;
                    } else if is_raw_cov {
                        stats.done_raw_cov += 1;
                    } else if !any_window {
                        stats.done_preflight += 1;
                    } else {
                        stats.done_ret_ic += 1;
                    }
                    stats.done = processed_sources;
                    stats.preflight_maj_windows += preflight_maj;
                    stats.preflight_zero_windows += preflight_zero;
                    stats.preflight_nan_windows += preflight_nan;

                    if total_pending > 0 {
                        let elapsed = started.elapsed();
                        let elapsed_secs = elapsed.as_secs();
                        let progress = processed_sources as f64 / total_pending as f64;
                        let estimated_total_secs = if progress > 0.0 {
                            elapsed.as_secs_f64() / progress
                        } else {
                            elapsed.as_secs_f64()
                        };
                        let remaining_secs = if estimated_total_secs > elapsed.as_secs_f64() {
                            (estimated_total_secs - elapsed.as_secs_f64()) as u64
                        } else {
                            0
                        };
                        let (elapsed_h, elapsed_m, elapsed_s) = format_hms(elapsed_secs);
                        let (remaining_h, remaining_m, remaining_s) = format_hms(remaining_secs);
                        let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

                        let cum_pass = stats.restored_pass + stats.done_pass;
                        let cum_total = restored_sources + processed_sources;
                        let cum_raw_cov = stats.restored_raw_cov + stats.done_raw_cov;
                        let cum_preflight = stats.restored_preflight + stats.done_preflight;
                        let cum_ret_ic = stats.restored_ret_ic + stats.done_ret_ic;
                        let cum_unknown = stats.restored_unknown + stats.done_unknown;

                        let l1 = format!(
                            "[{}] Tail V5 V7b 进度 {}/{} ({:.1}%)，已用{}h{}m{}s，预计剩余{}h{}m{}s",
                            current_time, processed_sources, total_pending,
                            progress * 100.0,
                            elapsed_h, elapsed_m, elapsed_s,
                            remaining_h, remaining_m, remaining_s,
                        );
                        let l2 = format!(
                            "累计通过 {} ({:.0}%) | 淘汰 raw_cov={} preflight={} ret_ic={} 未知={}",
                            cum_pass,
                            if cum_total > 0 { cum_pass as f64 * 100.0 / cum_total as f64 } else { 0.0 },
                            cum_raw_cov, cum_preflight, cum_ret_ic, cum_unknown,
                        );
                        let l3 = format!(
                            "maj={}w zero={}w nan={}w | 本次 {}(通过{}) | 恢复 {}(通过{})",
                            stats.preflight_maj_windows, stats.preflight_zero_windows,
                            stats.preflight_nan_windows,
                            stats.done, stats.done_pass,
                            restored_sources, stats.restored_pass,
                        );
                        update_status_line(&l1, &l2, &l3);
                    }
                }
                Err((task_name, err)) => {
                    reset_status_line();
                    return Err(format!("处理因子 {} 失败: {}", task_name, err));
                }
            }
        }

        if total_pending > 0 {
            println!();
            reset_status_line();
        }

        for h in io_handles_to_join {
            let _ = h.join();
        }
        for h in worker_handles_to_join {
            let _ = h.join();
        }

        write_aggregated_outputs(
            &cache_root_path,
            &aggregated,
        )?;

        let mut candidate_counts = HashMap::new();
        candidate_counts.insert("rolled_gap1".to_string(), aggregated.raw_summary_gap1.len());
        candidate_counts.insert("rolled_gap5".to_string(), aggregated.raw_summary_gap5.len());
        candidate_counts.insert("neu_gap1".to_string(), aggregated.neu_summary_gap1.len());
        candidate_counts.insert("neu_gap5".to_string(), aggregated.neu_summary_gap5.len());
        Ok((processed_sources, restored_sources, candidate_counts))
    }).map_err(PyRuntimeError::new_err)?;

    let info = PyDict::new(py);
    info.set_item("processed_sources", output.0)?;
    info.set_item("restored_sources", output.1)?;
    let candidate_counts = PyDict::new(py);
    for (key, value) in output.2 {
        candidate_counts.set_item(key, value)?;
    }
    info.set_item("candidate_counts", candidate_counts)?;
    Ok(info.into())
}
