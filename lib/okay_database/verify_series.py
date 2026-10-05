"""新旧对照验证：跑旧框架（design_whatever_out）的因子函数，和 Rust 复现值逐值比。

用法：
    python lib/okay_database/verify_series.py --series hm029 --okay /path/hm29_okay.py \
        --pairs 20250303:000001,20250304:600519

原理：把旧 okay 文件当模块导入，临时替换掉 dwo.run_factor_out，
只截获它里面定义的 Go 类，不触发任何写盘。然后用旧框架自己的读取与计算跑单股单日，
再和 rp.okay_compute_stock 比。

判定：
    - 两边同为 NaN 的位置算一致
    - 非 NaN 位置按相对误差比，默认阈值 1e-4（全链路 f32，比 f64 松一档）
    - 输出每个位置的最大误差，以及不一致的位置清单
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PYCODE = "/home/chenzongwei/pythoncode"


def load_old(okay_path: str):
    """导入旧 okay 模块，返回 (module, captured dict)。"""
    sys.path.insert(0, PYCODE)
    sys.path.insert(0, str(Path(okay_path).parent))
    import design_whatever_out as dwo

    captured: dict = {}
    original = dwo.run_factor_out

    def fake(*args, **kwargs):
        captured.update(kwargs)
        captured["Go"] = kwargs.get("Go") or (args[0] if args else None)
        return None

    dwo.run_factor_out = fake
    name = Path(okay_path).stem
    mod = importlib.import_module(name)
    try:
        mod.run_base(20240102, 20240102, 1)
    except Exception as exc:  # noqa: BLE001
        captured.setdefault("error", repr(exc))
    finally:
        dwo.run_factor_out = original
    return mod, captured


def old_values(mod, captured, code: str, date: int):
    """按旧框架的调用方式算单股单日的原始值。"""
    if "Go" in captured and captured["Go"] is not None:
        cls = captured["Go"]
        kwargs = dict(
            read_full=captured.get("read_full", 0) or 0,
            get_many=captured.get("get_many", 0) or 0,
            change_time=captured.get("change_time", 1),
        )
        if getattr(cls, "check_my_kind", "level2") == "minute":
            rc = cls(None, date, date, 1, captured.get("base_ver"), **kwargs)
        else:
            rc = cls(1, date, date, 1, symbol=code, **kwargs)
        res = rc.get_factor()
        return None if res is None else list(res)
    if hasattr(mod, "go"):
        res = mod.go(date, code)
        return None if res is None else list(res)
    raise RuntimeError("既没截获到 Go 类，也没有 go 函数")


def compare(old, new, rtol: float):
    if old is None:
        old = []
    n = max(len(old), len(new))
    bad = []
    nan_old = nan_new = 0
    for i in range(n):
        a = old[i] if i < len(old) else float("nan")
        b = new[i] if i < len(new) else float("nan")
        a = float(a) if a is not None else float("nan")
        b = float(b)
        if math.isnan(a) and math.isnan(b):
            nan_old += 1
            nan_new += 1
            continue
        if math.isnan(a) or math.isnan(b):
            bad.append((i, a, b, "一边是 NaN"))
            continue
        denom = max(abs(a), abs(b), 1e-12)
        rel = abs(a - b) / denom
        if not (rel <= rtol):
            bad.append((i, a, b, f"相对误差 {rel:.3e}"))
    return dict(n_old=len(old), n_new=len(new), nan=nan_old, bad=bad)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True, help="如 hm029")
    ap.add_argument("--okay", required=True)
    ap.add_argument("--pairs", default="20250303:000001,20250304:600519,20250305:000002")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import rust_pyfunc as rp

    mod, captured = load_old(args.okay)
    if "error" in captured:
        print(f"[提示] 旧 run_base 报错（可能只是写入路径不可用）: {captured['error']}")

    report = {"series": args.series, "okay": args.okay, "pairs": [], "rtol": args.rtol}
    all_ok = True
    for item in args.pairs.split(","):
        # 支持「日期:股票」和「股票:日期」两种写法
        left, _, right = item.strip().partition(":")
        if len(left) == 8 and left.isdigit():
            date, code = int(left), right
        else:
            code, date = left, int(right)
        code = code.strip().zfill(6)
        old = old_values(mod, captured, code, date)
        try:
            new = rp.okay_compute_stock(args.series, code, date)
        except Exception as exc:  # noqa: BLE001
            print(f"{code} {date}: Rust 侧失败 {exc}")
            all_ok = False
            continue
        res = compare(old, new, args.rtol)
        res.update(code=code, date=date)
        report["pairs"].append(res)
        flag = "一致" if not res["bad"] else f"有 {len(res['bad'])} 处不一致"
        print(f"{code} {date}: 旧 {res['n_old']} 值 / 新 {res['n_new']} 值，NaN {res['nan']}，{flag}")
        for i, a, b, why in res["bad"][:10]:
            print(f"    #{i}: 旧={a:.10g} 新={b:.10g} ({why})")
        all_ok = all_ok and not res["bad"]

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("结论:", "通过" if all_ok else "不通过")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
