//! rust_pyfunc_worker —— 多进程因子流水线的 worker 二进制入口。
//!
//! 由主进程（run_factor_pipeline mode="multiprocess"）通过 Command::new spawn。
//! 通信协议：stdin/stdout pipe + 4字节长度前缀 + bincode。
//!
//! 生命周期：
//!   1. 启动时从环境变量 RUST_PYFUNC_CORE_AFFINITY_IDX 读取核绑定索引（可选）
//!   2. 读 Init 消息（获取 pipeline_name、params、trading_days），回复 Ready
//!   3. while loop { 读 Task → 按 pipeline_name 分发计算 → 写 Result }
//!   4. 收到 Shutdown（长度0）则退出
//!   5. 计算错误不 panic，回传 Error 消息（避免进程崩溃重启开销）
use rust_pyfunc::backup_reader::TaskResult;
use rust_pyfunc::factor_pipeline::{
    ipc_read_result, ipc_read_task, ipc_write, ipc_write_result, pipeline_anneal_volume,
    pipeline_anneal_volume_market,
    pipeline_cross_section_example, pipeline_distill, pipeline_distill_tick, pipeline_drop_event,
    pipeline_extreme_point_fit, pipeline_hot_stock_pool, pipeline_hot_stock_pool_v2,
    pipeline_individual_order_ratio,
    pipeline_long_order, pipeline_microstructure_capm, pipeline_observable_order,
    pipeline_order_pair_hm90, pipeline_orderbook_imb_refactor, pipeline_urgency, ResultMessage,
    TaskMessage,
};
use std::io::{BufReader, BufWriter};

fn main() {
    // 核绑定（可选）
    if let Ok(idx_str) = std::env::var("RUST_PYFUNC_CORE_AFFINITY_IDX") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            let core_ids = core_affinity::get_core_ids().unwrap_or_default();
            if idx < core_ids.len() {
                let _ = core_affinity::set_for_current(core_ids[idx]);
            }
        }
    }

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    // IPC 优化：1MB 缓冲
    let mut reader = BufReader::with_capacity(1 << 20, stdin.lock());
    let mut writer = BufWriter::with_capacity(1 << 20, stdout.lock());

    // 1. 读 Init 消息
    let (pipeline_name, params, oo_params, trading_days, expected_len) =
        match ipc_read_task(&mut reader) {
            Ok(Some(TaskMessage::Init {
                pipeline_name,
                params,
                oo_params,
                trading_days,
                expected_len,
            })) => (pipeline_name, params, oo_params, trading_days, expected_len),
            _ => {
                eprintln!("worker: 未收到 Init 消息，退出");
                std::process::exit(1);
            }
        };

    // 回复 Ready
    if ipc_write_result(&mut writer, &ResultMessage::Ready).is_err() {
        eprintln!("worker: 无法回复 Ready，退出");
        std::process::exit(1);
    }

    // 2. 主循环：读任务 → 按 pipeline_name 分发计算 → 写结果
    loop {
        // 分两种任务类型：Level2 per-(date,code) 和 分钟 per-date
        match ipc_read_task(&mut reader) {
            Ok(Some(TaskMessage::Task { date, code })) => {
                // ---- Level2 单股任务 ----
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if pipeline_name == "observable_order" {
                        pipeline_observable_order(
                            date,
                            &code,
                            &oo_params,
                            &trading_days,
                            expected_len,
                        )
                    } else if pipeline_name == "individual_order_ratio" {
                        pipeline_individual_order_ratio(
                            date,
                            &code,
                            &oo_params,
                            &trading_days,
                            expected_len,
                        )
                    } else if pipeline_name == "orderbook_imb_refactor" {
                        pipeline_orderbook_imb_refactor(
                            date,
                            &code,
                            &oo_params,
                            &trading_days,
                            expected_len,
                        )
                    } else if pipeline_name == "extreme_point_fit" {
                        pipeline_extreme_point_fit(date, &code, &trading_days, expected_len)
                    } else if pipeline_name == "distill" {
                        pipeline_distill(date, &code, &trading_days, expected_len)
                    } else if pipeline_name == "distill_tick" {
                        pipeline_distill_tick(date, &code, &trading_days, expected_len)
                    } else if pipeline_name == "anneal_volume" {
                        pipeline_anneal_volume(date, &code, &trading_days, expected_len)
                    } else if pipeline_name == "anneal_volume_market" {
                        pipeline_anneal_volume_market(
                            date,
                            &code,
                            &trading_days,
                            expected_len,
                        )
                    } else if pipeline_name == "hidden_arrange" {
                        match rust_pyfunc::hidden_arrange_metrics::compute_hidden_arrange_full(
                            &code, date,
                        ) {
                            Ok((_n, v)) => v,
                            Err(_) => vec![f32::NAN; expected_len],
                        }
                    } else {
                        pipeline_order_pair_hm90(date, &code, &params, &trading_days, expected_len)
                    }
                }));

                match result {
                    Ok(vals) => {
                        let msg = ResultMessage::Result(TaskResult {
                            date,
                            code,
                            timestamp: 0,
                            facs: vals.iter().map(|&v| v as f32).collect(),
                        });
                        if ipc_write_result(&mut writer, &msg).is_err() {
                            break;
                        }
                    }
                    Err(_) => {
                        let _ = ipc_write_result(
                            &mut writer,
                            &ResultMessage::Error {
                                date,
                                code,
                                msg: "panic".to_string(),
                            },
                        );
                    }
                }
            }
            Ok(Some(TaskMessage::MinuteTask { date })) => {
                // ---- 分钟 per-date 任务：一次算全市场 ----
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if pipeline_name == "minute_example" {
                        crate_logic::pipeline_minute_example(date, expected_len)
                    } else if pipeline_name == "minute_capm" {
                        crate_logic::pipeline_minute_capm(date, expected_len)
                    } else if pipeline_name == "cross_section_example" {
                        pipeline_cross_section_example(date, expected_len)
                    } else if pipeline_name == "long_order" {
                        pipeline_long_order(date, expected_len)
                    } else if pipeline_name == "microstructure_capm" {
                        pipeline_microstructure_capm(date, expected_len)
                    } else if pipeline_name == "urgency" {
                        pipeline_urgency(date, expected_len)
                    } else if pipeline_name == "drop_event" {
                        pipeline_drop_event(date, expected_len)
                    } else if pipeline_name == "hot_stock_pool" {
                        pipeline_hot_stock_pool(date, expected_len)
                    } else if pipeline_name == "hot_stock_pool_v2" {
                        pipeline_hot_stock_pool_v2(date, expected_len)
                    } else {
                        Vec::new()
                    }
                }));

                match result {
                    Ok(batch) => {
                        let msg = ResultMessage::MinuteBatch(batch);
                        if ipc_write_result(&mut writer, &msg).is_err() {
                            break;
                        }
                    }
                    Err(_) => {
                        let _ = ipc_write_result(
                            &mut writer,
                            &ResultMessage::Error {
                                date,
                                code: String::new(),
                                msg: "minute pipeline panic".to_string(),
                            },
                        );
                    }
                }
            }
            Ok(Some(TaskMessage::Shutdown)) | Ok(None) => break,
            Ok(Some(TaskMessage::Init { .. })) => {
                eprintln!("worker: 意外收到第二个 Init，忽略");
                continue;
            }
            Err(_) => {
                eprintln!("worker: 读取任务失败，退出");
                break;
            }
        }
    }
}

/// 分钟 pipeline 分发逻辑（内联在 worker 中，避免跨 crate 依赖具体因子模块）。
mod crate_logic {
    use rust_pyfunc::backup_reader::TaskResult;
    use rust_pyfunc::minute_capm_metrics;
    use rust_pyfunc::minute_example_metrics;

    /// 分钟示例因子：返回整天全市场的 TaskResult 列表。
    pub fn pipeline_minute_example(date: i64, expected_len: usize) -> Vec<TaskResult> {
        match minute_example_metrics::compute_minute_example_full(date) {
            Ok((codes, vals)) => {
                // fan-out: codes × vals → Vec<TaskResult>
                let n_factors = expected_len;
                vals.chunks(n_factors)
                    .zip(codes.iter())
                    .map(|(facs, code)| TaskResult {
                        date,
                        code: code.clone(),
                        timestamp: 0,
                        facs: facs.to_vec(),
                    })
                    .collect()
            }
            Err(e) => {
                eprintln!("minute_example error [{date}]: {e:?}");
                Vec::new()
            }
        }
    }

    /// 两阶段分钟 CAPM：返回全市场逐股日内均值。
    pub fn pipeline_minute_capm(date: i64, expected_len: usize) -> Vec<TaskResult> {
        if expected_len != minute_capm_metrics::N_FACTORS {
            eprintln!(
                "minute_capm expected_len 错误 [{date}]: {expected_len} != {}",
                minute_capm_metrics::N_FACTORS
            );
            return Vec::new();
        }
        match minute_capm_metrics::compute_minute_capm_full(date) {
            Ok((codes, vals)) => vals
                .chunks(expected_len)
                .zip(codes.iter())
                .map(|(facs, code)| TaskResult {
                    date,
                    code: code.clone(),
                    timestamp: 0,
                    facs: facs.to_vec(),
                })
                .collect(),
            Err(e) => {
                eprintln!("minute_capm error [{date}]: {e:?}");
                Vec::new()
            }
        }
    }
}
