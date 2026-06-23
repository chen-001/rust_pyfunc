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
    ipc_read_result, ipc_read_task, ipc_write, ipc_write_result, pipeline_observable_order,
    pipeline_order_pair_hm90, ResultMessage, TaskMessage,
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
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());

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
        let (date, code) = match ipc_read_task(&mut reader) {
            Ok(Some(TaskMessage::Task { date, code })) => (date, code),
            Ok(Some(TaskMessage::Shutdown)) | Ok(None) => break,
            Ok(Some(TaskMessage::Init { .. })) => {
                eprintln!("worker: 意外收到第二个 Init，忽略");
                continue;
            }
            Err(_) => {
                eprintln!("worker: 读取任务失败，退出");
                break;
            }
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if pipeline_name == "observable_order" {
                pipeline_observable_order(date, &code, &oo_params, &trading_days, expected_len)
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
                    facs: vals,
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
}
