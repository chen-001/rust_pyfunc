//! 临时验证：copy_subset_write 的模板轴继承（全 NaN 股票/日期）。
//! 运行: cargo run --example verify_copy_subset_axes
use rust_pyfunc::backup_reader::TaskResult;
use rust_pyfunc::factor_store_v5::{copy_subset_write, FactorStoreReader, FactorStoreWriter};

fn main() {
    let dir = std::env::temp_dir().join(format!("copy_subset_verify_{}", std::process::id()));
    let src = dir.join("src");
    let dst_stock = dir.join("dst_stock");
    let dst_date = dir.join("dst_date");
    let _ = std::fs::remove_dir_all(&dir);
    let names: Vec<String> = vec!["f0".to_string(), "f1".to_string(), "f2".to_string()];
    let dates = [20230103i64, 20230104];
    let codes = ["000001", "000002", "000003"];
    let mut results: Vec<TaskResult> = Vec::new();
    for &d in &dates {
        for (ci, &c) in codes.iter().enumerate() {
            let base = ci as f32 * 10.0 + if d == 20230103 { 0.0 } else { 1.0 };
            let f0 = base;
            let f1 = if c == "000002" { f32::NAN } else { base + 2.0 };
            let f2 = if d == 20230104 { f32::NAN } else { base + 4.0 };
            results.push(TaskResult {
                date: d,
                code: c.to_string(),
                timestamp: 0,
                facs: vec![f0, f1, f2],
            });
        }
    }
    let mut w = FactorStoreWriter::open(src.to_str().unwrap(), &names).unwrap();
    w.append_batch(&results).unwrap();
    w.finish_and_project(0).unwrap();
    let src_axes = FactorStoreReader::open(src.to_str().unwrap()).unwrap().template_axes();
    println!("src axes: {:?} / {:?}", src_axes.0, src_axes.1);

    // f1: 股票 000002 全 NaN → 继承股票轴
    copy_subset_write(src.to_str().unwrap(), dst_stock.to_str().unwrap(), &["f1".to_string()]).unwrap();
    let a1 = FactorStoreReader::open(dst_stock.to_str().unwrap()).unwrap().template_axes();
    println!("dst(f1) axes: {:?} / {:?}", a1.0, a1.1);
    assert_eq!(src_axes, a1, "全 NaN 股票轴必须继承");

    // f2: 日期 20230104 全 NaN → 继承日期轴
    copy_subset_write(src.to_str().unwrap(), dst_date.to_str().unwrap(), &["f2".to_string()]).unwrap();
    let a2 = FactorStoreReader::open(dst_date.to_str().unwrap()).unwrap().template_axes();
    println!("dst(f2) axes: {:?} / {:?}", a2.0, a2.1);
    assert_eq!(src_axes, a2, "全 NaN 日期轴必须继承");

    let _ = std::fs::remove_dir_all(&dir);
    println!("✅ copy_subset 模板轴继承验证通过");
}
