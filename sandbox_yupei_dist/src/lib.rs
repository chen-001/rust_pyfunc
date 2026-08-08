//! yupei_dist: 关联-差异矩阵（衰减事件场交互能量） + 降维指标（网络因子）纯 Rust 库。
//! 用法见 main.rs 子命令。

pub mod fast_csv_reader;
pub mod indicator_ctx;
pub mod indicators;
pub mod industry;
pub mod matrix_stage;
pub mod matrix_store;

/// 从备份目录加载上下文（矩阵 + 统计 + 前一日）
pub fn load_ctx(outdir: &str, date: i64, prev_date: Option<i64>) -> std::io::Result<indicator_ctx::IndicatorCtx<'static>> {
    use indicator_ctx::{BackupSet, IndicatorCtx, PrevDay};
    let set = BackupSet::load(std::path::Path::new(outdir), date, true)?;
    let prev = match prev_date {
        Some(d) => {
            let pset = BackupSet::load(std::path::Path::new(outdir), d, true)?;
            // 前一日只需要动态类模块用到的矩阵, 全部加载也无妨（内存按需 mmap）
            let prev = PrevDay::new(Box::leak(Box::new(pset)));
            Some(prev)
        }
        None => None,
    };
    Ok(IndicatorCtx::new(Box::leak(Box::new(set)), prev))
}
