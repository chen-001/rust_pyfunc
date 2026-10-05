// 自动生成，不要手改。生成器：lib/okay_database/gen_registry.py

pub mod hm013;
pub mod hm016;
pub mod hm018;
pub mod hm029;
pub mod hm036;

/// 逐股（TaskMessage::Task）分发：series 形如 "hm003"。
pub fn dispatch_stock(series: &str, code: &str, date: i64, expected_len: usize) -> Vec<f32> {
    let out: Option<Vec<f32>> = match series {
        "hm013" => hm013::compute_stock(code, date),
        "hm016" => hm016::compute_stock(code, date),
        "hm018" => hm018::compute_stock(code, date),
        "hm029" => hm029::compute_stock(code, date),
        "hm036" => hm036::compute_stock(code, date),
        _ => None,
    };
    match out {
        Some(v) => fit(v, expected_len),
        None => nan_stock(expected_len),
    }
}

/// 原样返回（不补 NaN），供单股调试入口区分「系列没有实现」和「该股该日算不出来」。
pub fn dispatch_stock_raw(series: &str, code: &str, date: i64) -> Option<Vec<f32>> {
    match series {
        "hm013" => hm013::compute_stock(code, date),
        "hm016" => hm016::compute_stock(code, date),
        "hm018" => hm018::compute_stock(code, date),
        "hm029" => hm029::compute_stock(code, date),
        "hm036" => hm036::compute_stock(code, date),
        _ => None,
    }
}

/// 横截面 / 分钟（TaskMessage::MinuteTask）分发：一次算一天全市场。
pub fn dispatch_cross(series: &str, date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    match series {
        "hm013" => hm013::compute_cross(date),
        "hm016" => hm016::compute_cross(date),
        "hm018" => hm018::compute_cross(date),
        "hm029" => hm029::compute_cross(date),
        "hm036" => hm036::compute_cross(date),
        _ => None,
    }
}

/// 该系列声明的因子名（用于核对 expected_len）。
pub fn series_names(series: &str) -> Option<Vec<String>> {
    match series {
        "hm013" => Some(hm013::names()),
        "hm016" => Some(hm016::names()),
        "hm018" => Some(hm018::names()),
        "hm029" => Some(hm029::names()),
        "hm036" => Some(hm036::names()),
        _ => None,
    }
}
