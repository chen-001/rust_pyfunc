//! rust_pyfunc_worker —— 多进程因子流水线的 worker 二进制入口。
//!
//! 由主进程（run_factor_pipeline mode="multiprocess"）通过 Command::new spawn。
//! 通信协议：stdin/stdout pipe + 4字节长度前缀 + bincode。
//!
//! 生命周期：
//!   1. 启动时从环境变量 RUST_PYFUNC_CORE_AFFINITY_IDX 读取核绑定索引（可选）
//!   2. 读 Init 消息（获取 params、trading_days、expected_len），回复 Ready
//!   3. while loop { 读 Task → 调 pipeline_order_pair_hm90 → 写 Result }
//!   4. 收到 Shutdown（长度0）则退出
//!   5. 计算错误不 panic，回传 Error 消息（避免进程崩溃重启开销）
use rust_pyfunc::factor_pipeline::{
    ipc_read_result, ipc_read_task, ipc_write, ipc_write_result,
    pipeline_order_pair_hm90, Hm90Params, ResultMessage, TaskMessage,
};
use std::io::{BufReader, BufWriter};

fn main() {
    // 核绑定（可选）
    if let Ok(idx_str) = std::env::var("RUST_PYFUNC_CORE_AFFINITY_IDX") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(cores) = core_affinity::get_core_ids() {
                if idx < cores.len() {
                    let _ = core_affinity::set_for_current(cores[idx].clone());
                }
            }
        }
    }

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    // stdin 用锁避免竞争，BufReader 缓冲
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());

    // 1. 读 Init 消息
    let (params, trading_days, _expected_len) = match ipc_read_task(&mut reader) {
        Ok(Some(TaskMessage::Init { params, trading_days, expected_len })) => {
            (params, trading_days, expected_len)
        }
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

    // 2. 主循环：读任务 → 计算 → 写结果
    loop {
        let msg = match ipc_read_task(&mut reader) {
            Ok(Some(TaskMessage::Task { date, code })) => (date, code),
            Ok(Some(TaskMessage::Shutdown)) | Ok(None) => {
                // 优雅退出
                break;
            }
            Ok(Some(TaskMessage::Init { .. })) => {
                // 忽略重复 Init
                continue;
            }
            Err(e) => {
                eprintln!("worker: 读取任务失败: {}, 退出", e);
                break;
            }
        };

        let (date, code) = msg;

        // 调用流水线计算
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pipeline_order_pair_hm90(date, &code, &params, &trading_days, 0)
        }));

        match result {
            Ok(facs) => {
                let task_result = rust_pyfunc::backup_reader::TaskResult {
                    date,
                    code,
                    timestamp: 0,
                    facs,
                };
                if ipc_write_result(&mut writer, &ResultMessage::Result(task_result)).is_err() {
                    eprintln!("worker: 写结果失败，退出");
                    break;
                }
            }
            Err(_) => {
                // panic 被捕获，回传 Error
                let _ = ipc_write_result(&mut writer, &ResultMessage::Error {
                    date,
                    code,
                    msg: "计算 panic".to_string(),
                });
            }
        }
    }
}
