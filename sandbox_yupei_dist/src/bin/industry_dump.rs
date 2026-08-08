//! industry_dump <date> <out.csv>  （--features hdf5 构建）
//! 从 /ssd_data/data/vars/SzBa/industry.h5（申万一级, 31 类）提取指定日期的
//! 每股行业代码 → CSV（code,ind），供指标阶段使用。
//! 行序: industry.h5 的第 r 行对应 calendar_map.csv 第 r 个日期;
//! 列序: 第 c 列对应 symbol_map.csv 中 pos=c 的股票。

use hdf5_metno as hdf5;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: industry_dump <date> <out.csv>");
        std::process::exit(1);
    }
    let date: i64 = args[1].parse().unwrap();
    let out = &args[2];

    // 1) 读 calendar_map.csv → 日期序列
    let cal = std::fs::read_to_string("/ssd_data/data/vars/SzBa/calendar_map.csv").unwrap();
    let dates: Vec<String> = cal.lines().skip(1).map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
    let row = dates.iter().position(|d| d == &date.to_string()).unwrap_or_else(|| {
        // 取最后一个 <= date 的日期
        dates.iter().rposition(|d| d.as_str() <= &date.to_string()).unwrap_or(0)
    });
    eprintln!("date {date} → h5 row {row}");

    // 2) 读 symbol_map.csv → 代码序列（pos → code）
    let sym = std::fs::read_to_string("/ssd_data/data/basic_info/symbol_map.csv").unwrap();
    let mut code_by_pos: Vec<String> = Vec::new();
    for line in sym.lines().skip(1) {
        let mut it = line.split(',');
        let code = it.next().unwrap_or("").trim();
        if !code.is_empty() {
            code_by_pos.push(code.to_string());
        }
    }
    eprintln!("symbols = {}", code_by_pos.len());

    // 3) 读 h5（整表读入, 取目标行）
    let f = hdf5::File::open("/ssd_data/data/vars/SzBa/industry.h5").unwrap();
    let ds = f.dataset("data").unwrap();
    let shape = ds.shape();
    eprintln!("h5 shape = {:?}", shape);
    let arr: ndarray::Array2<f64> = ds.read_2d().unwrap();
    let row_data: Vec<f64> = arr.row(row).to_vec();

    // 4) 输出 code,ind（ind = 行业编号, -1 未知）
    let mut out_s = String::new();
    for (pos, code) in code_by_pos.iter().enumerate() {
        let v = row_data[pos];
        let ind = if v.is_nan() { -1 } else { v as i16 };
        out_s.push_str(&format!("{code},{ind}\n"));
    }
    std::fs::write(out, out_s).unwrap();
    eprintln!("written {out}");
}
