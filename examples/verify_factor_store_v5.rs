//! 列式因子存储验证程序（独立编译，验证写入→投影→读取一致性）。
//! 运行：cargo run --release --example verify_factor_store_v5

use rust_pyfunc::backup_reader::TaskResult;
use rust_pyfunc::factor_store_v5::{FactorStoreReader, FactorStoreWriter};

fn main() {
    println!("=== 列式因子存储 RPFBINV5 验证 ===\n");

    let tmp = tempfile::tempdir().unwrap();
    let store_dir = tmp.path().to_str().unwrap().to_string();

    // 构造测试数据：5 日期 × 3 股票 × 4 因子
    let dates: Vec<i64> = vec![20230101, 20230102, 20230103, 20230104, 20230105];
    let codes: Vec<&str> = vec!["000001", "600519", "000858"];
    let factor_count = 4;
    let factor_names: Vec<String> = (0..factor_count).map(|i| format!("f{i}")).collect();

    let mut results = Vec::new();
    for (di, &date) in dates.iter().enumerate() {
        for (ci, code) in codes.iter().enumerate() {
            let facs: Vec<f64> = (0..factor_count)
                .map(|f| (di as f64) * 1000.0 + (ci as f64) * 10.0 + f as f64)
                .collect();
            results.push(TaskResult {
                date,
                code: code.to_string(),
                timestamp: 0,
                facs,
            });
        }
    }

    // 1. 分两批写入（模拟增量追加）
    println!("[1] 写入 {} 条记录（分两批）...", results.len());
    let mut writer = FactorStoreWriter::open(&store_dir, &factor_names).unwrap();
    let mid = results.len() / 2;
    writer.append_batch(&results[..mid]).unwrap();
    writer.append_batch(&results[mid..]).unwrap();
    assert_eq!(writer.record_count(), results.len() as u64);
    println!("    记录数: {}, 因子数: {}", writer.record_count(), factor_count);

    // 2. 断点续算验证：重新打开应能识别已写入记录
    drop(writer);
    println!("\n[2] 断点续算验证（重新打开 Writer）...");
    let writer2 = FactorStoreWriter::open(&store_dir, &factor_names).unwrap();
    assert_eq!(writer2.record_count(), results.len() as u64);
    let completed = writer2.check_completed().unwrap();
    assert_eq!(completed.len(), results.len());
    println!("    check_completed 识别出 {} 条已完成记录 ✅", completed.len());
    drop(writer2);

    // 3. 投影
    println!("\n[3] 投影（finish_and_project）...");
    let mut writer3 = FactorStoreWriter::open(&store_dir, &factor_names).unwrap();
    writer3.finish_and_project(4).unwrap();
    drop(writer3);

    // 4. 读取校验
    println!("\n[4] 读取并校验每个因子值...");
    let reader = FactorStoreReader::open(&store_dir).unwrap();
    assert!(reader.is_projected());
    assert_eq!(reader.factor_names(), &factor_names);

    let template_dates: Vec<i32> = dates.iter().map(|&d| d as i32).collect();
    let template_stocks: Vec<String> = codes.iter().map(|c| format!("{c}.SZ")).collect();

    for f_idx in 0..factor_count {
        let matrix = reader
            .read_factor_to_matrix(f_idx, &template_dates, &template_stocks)
            .unwrap();
        assert_eq!(matrix.shape(), &[dates.len(), codes.len()]);
        for (di, _) in dates.iter().enumerate() {
            for (ci, _) in codes.iter().enumerate() {
                let expected = (di as f32) * 1000.0 + (ci as f32) * 10.0 + f_idx as f32;
                let actual = matrix[[di, ci]];
                assert!(
                    (actual - expected).abs() < 0.01,
                    "因子{f_idx} (date_idx={di},code_idx={ci}): 期望{expected}, 实际{actual}"
                );
            }
        }
    }
    println!("    4 个因子 × 15 个值全部正确 ✅");

    // 5. 容量信息
    let colblk_size = std::fs::metadata(tmp.path().join("factors.colblk")).unwrap().len();
    let idx_size = std::fs::metadata(tmp.path().join("factors.idx")).unwrap().len();
    println!("\n[5] 文件大小: factors.colblk={} 字节, factors.idx={} 字节", colblk_size, idx_size);

    println!("\n🎉 全部验证通过！");
}
