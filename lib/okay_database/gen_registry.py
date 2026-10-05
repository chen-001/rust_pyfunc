"""扫描 src/okay_database/ 下的系列文件，生成 registry.rs。

新增一个系列只需要：
  1. 往 src/okay_database/ 放一个 hmXXX.rs（提供 names / compute_stock / compute_cross）
  2. 跑一次本脚本
  3. 重新构建

这样多个 agent 并行改写时不会互相改到同一个文件。
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path("/home/chenzongwei/rust_pyfunc/src/okay_database")
OUT = SRC / "registry.rs"
PATTERN = re.compile(r"^hm\d+\.rs$")


def main() -> int:
    files = sorted(p for p in SRC.iterdir() if p.is_file() and PATTERN.match(p.name))
    mods = [p.stem for p in files]

    lines = [
        "// 自动生成，不要手改。生成器：lib/okay_database/gen_registry.py",
        "",
    ]
    for m in mods:
        lines.append(f"pub mod {m};")
    lines += [
        "",
        "/// 逐股（TaskMessage::Task）分发：series 形如 \"hm003\"。",
        "pub fn dispatch_stock(series: &str, code: &str, date: i64, expected_len: usize) -> Vec<f32> {",
        "    let out: Option<Vec<f32>> = match series {",
    ]
    for m in mods:
        lines.append(f'        "{m}" => {m}::compute_stock(code, date),')
    lines += [
        "        _ => None,",
        "    };",
        "    match out {",
        "        Some(v) => fit(v, expected_len),",
        "        None => nan_stock(expected_len),",
        "    }",
        "}",
        "",
        "/// 原样返回（不补 NaN），供单股调试入口区分「系列没有实现」和「该股该日算不出来」。",
        "pub fn dispatch_stock_raw(series: &str, code: &str, date: i64) -> Option<Vec<f32>> {",
        "    match series {",
    ]
    for m in mods:
        lines.append(f'        "{m}" => {m}::compute_stock(code, date),')
    lines += [
        "        _ => None,",
        "    }",
        "}",
        "",
        "/// 横截面 / 分钟（TaskMessage::MinuteTask）分发：一次算一天全市场。",
        "pub fn dispatch_cross(series: &str, date: i64) -> Option<(Vec<String>, Vec<f32>)> {",
        "    match series {",
    ]
    for m in mods:
        lines.append(f'        "{m}" => {m}::compute_cross(date),')
    lines += [
        "        _ => None,",
        "    }",
        "}",
        "",
        "/// 该系列声明的因子名（用于核对 expected_len）。",
        "pub fn series_names(series: &str) -> Option<Vec<String>> {",
        "    match series {",
    ]
    for m in mods:
        lines.append(f'        "{m}" => Some({m}::names()),')
    lines += [
        "        _ => None,",
        "    }",
        "}",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"registry.rs 已生成，收录 {len(mods)} 个系列：{', '.join(mods) if mods else '（空）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
