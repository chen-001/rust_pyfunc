//! tail_perf_bench_merge：opt3 归并式 rank2 原型的独立可复现 crate。
//! 用法:
//!   tail_perf_bench_merge v2m                    对账(逐位) + 单线程测速
//!   tail_perf_bench_merge v2mmt <workers,..> <rounds> <batch>   多线程吞吐对照
//!   tail_perf_bench_merge v2                     单槽分步耗时分解 (opt2)
mod engine;
mod neu;
mod npy;
mod v2;

const DATA: &str = "/home/chenzongwei/pythoncode/tail_perf_lab/data";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("v2m");
    match mode {
        "v2m" => v2::run_v2_merge_bench(DATA),
        "v2mmt" => {
            let workers: Vec<usize> = args
                .get(2)
                .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or_else(|| vec![1, 96, 192]);
            let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
            let batch: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
            v2::run_v2_merge_mt(DATA, &workers, rounds, batch);
        }
        "v2" => v2::run_v2_bench(DATA),
        "v2b" => v2::run_v2_batch_bench(DATA),
        _ => eprintln!("unknown mode: {mode}"),
    }
}
