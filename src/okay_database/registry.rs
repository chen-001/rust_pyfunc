// 自动生成，不要手改。生成器：lib/okay_database/gen_registry.py

pub mod hm003;
pub mod hm008;
pub mod hm009;
pub mod hm010;
pub mod hm011;
pub mod hm013;
pub mod hm014;
pub mod hm015;
pub mod hm016;
pub mod hm017;
pub mod hm018;
pub mod hm020;
pub mod hm021;
pub mod hm022;
pub mod hm023;
pub mod hm025;
pub mod hm026;
pub mod hm028;
pub mod hm029;
pub mod hm031;
pub mod hm034;
pub mod hm035;
pub mod hm036;
pub mod hm039;
pub mod hm044;
pub mod hm046;
pub mod hm061;

/// 逐股（TaskMessage::Task）分发：series 形如 "hm003"。
pub fn dispatch_stock(series: &str, code: &str, date: i64, expected_len: usize) -> Vec<f32> {
    let out: Option<Vec<f32>> = match series {
        "hm003" => hm003::compute_stock(code, date),
        "hm008" => hm008::compute_stock(code, date),
        "hm009" => hm009::compute_stock(code, date),
        "hm010" => hm010::compute_stock(code, date),
        "hm011" => hm011::compute_stock(code, date),
        "hm013" => hm013::compute_stock(code, date),
        "hm014" => hm014::compute_stock(code, date),
        "hm015" => hm015::compute_stock(code, date),
        "hm016" => hm016::compute_stock(code, date),
        "hm017" => hm017::compute_stock(code, date),
        "hm018" => hm018::compute_stock(code, date),
        "hm020" => hm020::compute_stock(code, date),
        "hm021" => hm021::compute_stock(code, date),
        "hm022" => hm022::compute_stock(code, date),
        "hm023" => hm023::compute_stock(code, date),
        "hm025" => hm025::compute_stock(code, date),
        "hm026" => hm026::compute_stock(code, date),
        "hm028" => hm028::compute_stock(code, date),
        "hm029" => hm029::compute_stock(code, date),
        "hm031" => hm031::compute_stock(code, date),
        "hm034" => hm034::compute_stock(code, date),
        "hm035" => hm035::compute_stock(code, date),
        "hm036" => hm036::compute_stock(code, date),
        "hm039" => hm039::compute_stock(code, date),
        "hm044" => hm044::compute_stock(code, date),
        "hm046" => hm046::compute_stock(code, date),
        "hm061" => hm061::compute_stock(code, date),
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
        "hm003" => hm003::compute_stock(code, date),
        "hm008" => hm008::compute_stock(code, date),
        "hm009" => hm009::compute_stock(code, date),
        "hm010" => hm010::compute_stock(code, date),
        "hm011" => hm011::compute_stock(code, date),
        "hm013" => hm013::compute_stock(code, date),
        "hm014" => hm014::compute_stock(code, date),
        "hm015" => hm015::compute_stock(code, date),
        "hm016" => hm016::compute_stock(code, date),
        "hm017" => hm017::compute_stock(code, date),
        "hm018" => hm018::compute_stock(code, date),
        "hm020" => hm020::compute_stock(code, date),
        "hm021" => hm021::compute_stock(code, date),
        "hm022" => hm022::compute_stock(code, date),
        "hm023" => hm023::compute_stock(code, date),
        "hm025" => hm025::compute_stock(code, date),
        "hm026" => hm026::compute_stock(code, date),
        "hm028" => hm028::compute_stock(code, date),
        "hm029" => hm029::compute_stock(code, date),
        "hm031" => hm031::compute_stock(code, date),
        "hm034" => hm034::compute_stock(code, date),
        "hm035" => hm035::compute_stock(code, date),
        "hm036" => hm036::compute_stock(code, date),
        "hm039" => hm039::compute_stock(code, date),
        "hm044" => hm044::compute_stock(code, date),
        "hm046" => hm046::compute_stock(code, date),
        "hm061" => hm061::compute_stock(code, date),
        _ => None,
    }
}

/// 横截面 / 分钟（TaskMessage::MinuteTask）分发：一次算一天全市场。
pub fn dispatch_cross(series: &str, date: i64) -> Option<(Vec<String>, Vec<f32>)> {
    match series {
        "hm003" => hm003::compute_cross(date),
        "hm008" => hm008::compute_cross(date),
        "hm009" => hm009::compute_cross(date),
        "hm010" => hm010::compute_cross(date),
        "hm011" => hm011::compute_cross(date),
        "hm013" => hm013::compute_cross(date),
        "hm014" => hm014::compute_cross(date),
        "hm015" => hm015::compute_cross(date),
        "hm016" => hm016::compute_cross(date),
        "hm017" => hm017::compute_cross(date),
        "hm018" => hm018::compute_cross(date),
        "hm020" => hm020::compute_cross(date),
        "hm021" => hm021::compute_cross(date),
        "hm022" => hm022::compute_cross(date),
        "hm023" => hm023::compute_cross(date),
        "hm025" => hm025::compute_cross(date),
        "hm026" => hm026::compute_cross(date),
        "hm028" => hm028::compute_cross(date),
        "hm029" => hm029::compute_cross(date),
        "hm031" => hm031::compute_cross(date),
        "hm034" => hm034::compute_cross(date),
        "hm035" => hm035::compute_cross(date),
        "hm036" => hm036::compute_cross(date),
        "hm039" => hm039::compute_cross(date),
        "hm044" => hm044::compute_cross(date),
        "hm046" => hm046::compute_cross(date),
        "hm061" => hm061::compute_cross(date),
        _ => None,
    }
}

/// 该系列声明的因子名（用于核对 expected_len）。
pub fn series_names(series: &str) -> Option<Vec<String>> {
    match series {
        "hm003" => Some(hm003::names()),
        "hm008" => Some(hm008::names()),
        "hm009" => Some(hm009::names()),
        "hm010" => Some(hm010::names()),
        "hm011" => Some(hm011::names()),
        "hm013" => Some(hm013::names()),
        "hm014" => Some(hm014::names()),
        "hm015" => Some(hm015::names()),
        "hm016" => Some(hm016::names()),
        "hm017" => Some(hm017::names()),
        "hm018" => Some(hm018::names()),
        "hm020" => Some(hm020::names()),
        "hm021" => Some(hm021::names()),
        "hm022" => Some(hm022::names()),
        "hm023" => Some(hm023::names()),
        "hm025" => Some(hm025::names()),
        "hm026" => Some(hm026::names()),
        "hm028" => Some(hm028::names()),
        "hm029" => Some(hm029::names()),
        "hm031" => Some(hm031::names()),
        "hm034" => Some(hm034::names()),
        "hm035" => Some(hm035::names()),
        "hm036" => Some(hm036::names()),
        "hm039" => Some(hm039::names()),
        "hm044" => Some(hm044::names()),
        "hm046" => Some(hm046::names()),
        "hm061" => Some(hm061::names()),
        _ => None,
    }
}
