//! 把 synth_input/{dates.npy, codes.txt, values.npy} 写成列式因子 store（含投影），
//! 供回测引擎“低覆盖放行”验证使用。
//! 用法：make_synth_store <输入目录> <输出 store 目录>
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use rust_pyfunc::backup_reader::TaskResult;
use rust_pyfunc::factor_store_v5::FactorStoreWriter;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (inp, out) = (&args[1], &args[2]);
    let dates: Array1<i64> = read_npy(format!("{inp}/dates.npy")).expect("读 dates");
    let values: Array2<f32> = read_npy(format!("{inp}/values.npy")).expect("读 values");
    let codes: Vec<String> = std::fs::read_to_string(format!("{inp}/codes.txt"))
        .expect("读 codes")
        .lines()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(dates.len(), codes.len());
    assert_eq!(dates.len(), values.nrows());
    let names: Vec<String> =
        ["plain_full", "cxl_masked", "ctrl_masked"].iter().map(|s| s.to_string()).collect();
    assert_eq!(values.ncols(), names.len());
    let mut w = FactorStoreWriter::open(out, &names, false).expect("打开 store");
    let mut batch: Vec<TaskResult> = Vec::with_capacity(1 << 16);
    for i in 0..dates.len() {
        batch.push(TaskResult {
            date: dates[i],
            code: codes[i].clone(),
            timestamp: 0,
            facs: values.row(i).to_vec(),
        });
        if batch.len() == (1 << 16) {
            w.append_batch(&batch).expect("写入");
            batch.clear();
        }
    }
    w.append_batch(&batch).expect("写入");
    w.finish_and_project(16).expect("投影");
    eprintln!("写入 {} 行 → {out}", dates.len());
}
