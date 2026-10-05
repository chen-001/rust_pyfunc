"""will 型系列（今天要跟**上一个交易日**的盘口帧比）的新旧对照验证。

和 verify_series.py 的区别
--------------------------
verify_series.py 的 old_values 只 new 一个 Go 实例、调一次 get_factor()，不传 will。
对 with_will=1 的系列（hm049 / hm050）这样做时，旧代码走的是 `self.will is None`
那条分支：只把当天的 df 存进 self.will 就 return，拿不到任何因子值。
本脚本按 design_whatever_out.cal_factor_one_stock 的真实调用顺序补上这一步：

    prev = last_n_trading_date(date, with_will_back_days)   # 默认回看 1 个交易日
    rc   = Go(prev)   →  rc.get_factor() 只更新 rc.will、返回 None
    will = copy.deepcopy(rc.will)
    rc2  = Go(date, will=will)  →  rc2.get_factor() 才是目标日的真值

其余（读数据、比数、判定阈值）与 verify_series.py 一致：
两边同为 NaN 算一致，非 NaN 按相对误差比，默认 1e-4；近零位置另有 1e-9 的绝对兜底。

用法
----
    python lib/okay_database/verify_will.py --series hm050 \
        --okay /home/chenzongwei/pythoncode/成果hm50_昨日重现/hm50_okay.py \
        --pairs "20250304:000001,20250304:600519,20180601:000002,20231020:300750"

环境变量 OKAY_VERIFY_DATA_ROOT（可选）：指向形如 <root>/stock 的数据根，
用法与 verify_series.py 完全相同（设了就把旧框架的 path_to_be_decided 和 Rust 侧的
RUST_PYFUNC_LEVEL2_ROOT 同时指过去）。
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PYCODE = "/home/chenzongwei/pythoncode"
DATA_ROOT_ENV = "OKAY_VERIFY_DATA_ROOT"


def data_root() -> str:
    return os.environ.get(DATA_ROOT_ENV, "").strip()


def load_old(okay_path: str):
    """导入旧 okay 模块，截获它定义的 Go 类（不触发任何写盘）。"""
    sys.path.insert(0, PYCODE)
    sys.path.insert(0, str(Path(okay_path).parent))
    import design_whatever_out as dwo

    root = data_root()
    if root:
        dwo.path_to_be_decided = root
        print(f"[数据] 旧框架数据根目录已覆盖为 {root}（{DATA_ROOT_ENV}）")

    captured: dict = {}
    original = dwo.run_factor_out

    def fake(*args, **kwargs):
        captured.update(kwargs)
        captured["Go"] = kwargs.get("Go") or (args[0] if args else None)
        return None

    dwo.run_factor_out = fake
    _orig_rust_bin = getattr(dwo, "run_factor_rust_bin", None)

    def fake_rust_bin(*a, **k):
        return None

    if _orig_rust_bin is not None:
        dwo.run_factor_rust_bin = fake_rust_bin
    name = Path(okay_path).stem
    mod = importlib.import_module(name)
    try:
        cal = os.environ.get("OKAY_VERIFY_CAL_RANGE", "20150105:20261231").split(":")
        mod.run_base(int(cal[0]), int(cal[1]), 1)
    except Exception as exc:  # noqa: BLE001
        captured.setdefault("error", repr(exc))
    finally:
        dwo.run_factor_out = original
        if _orig_rust_bin is not None:
            dwo.run_factor_rust_bin = _orig_rust_bin
    return mod, captured


def old_values(mod, captured, code: str, date: int):
    """按 cal_factor_one_stock 的 with_will 顺序跑两日，返回目标日的真值。"""
    import design_whatever_out as dwo

    if "Go" not in captured or captured["Go"] is None:
        raise RuntimeError("没截获到 Go 类，旧 okay 文件可能不是 run_factor_out 形态")
    cls = captured["Go"]
    kwargs = dict(
        read_full=captured.get("read_full", 0) or 0,
        get_many=captured.get("get_many", 0) or 0,
        change_time=captured.get("change_time", 1),
    )
    back = int(captured.get("with_will_back_days", 1) or 1)
    prev = date
    for _ in range(back):
        prev = int(dwo.last_n_trading_date(prev, 1))
    rc = cls(1, prev, prev, 1, symbol=code, **kwargs)
    rc.get_factor()  # 只更新 will，返回值无用（回看日没有上一日可看）
    will = copy.deepcopy(rc.will)
    rc2 = cls(1, date, date, 1, symbol=code, will=will, **kwargs)
    return prev, rc2.get_factor()


def compare(old, new, rtol: float, atol: float = 1e-9):
    if old is None:
        old = []
    bad, nan, near_zero = [], 0, 0
    for i in range(max(len(old), len(new))):
        a = old[i] if i < len(old) else float("nan")
        b = float(new[i]) if i < len(new) else float("nan")
        a = float(a) if a is not None else float("nan")
        if math.isnan(a) and math.isnan(b):
            nan += 1
            continue
        if math.isnan(a) or math.isnan(b):
            bad.append((i, a, b, "一边是 NaN"))
            continue
        if math.isinf(a) or math.isinf(b):
            if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
                continue
            bad.append((i, a, b, "无穷不一致"))
            continue
        if abs(a) < atol and abs(b) < atol:
            near_zero += 1
            continue
        rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
        if not (rel <= rtol):
            bad.append((i, a, b, f"相对误差 {rel:.3e}"))
    return dict(n_old=len(old), n_new=len(new), nan=nan, near_zero=near_zero, bad=bad)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True, help="如 hm049 / hm050")
    ap.add_argument("--okay", required=True)
    ap.add_argument("--pairs", default="20250304:000001,20180601:000002")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = data_root()
    if root:
        os.environ["RUST_PYFUNC_LEVEL2_ROOT"] = str(Path(root) / "stock")

    import rust_pyfunc as rp

    mod, captured = load_old(args.okay)
    if "error" in captured:
        print(f"[提示] 旧 run_base 报错（通常只是写盘路径不可用）: {captured['error']}")
    names = rp.okay_series_names(args.series)

    report = {"series": args.series, "okay": args.okay, "pairs": [], "rtol": args.rtol}
    all_ok = True
    for item in args.pairs.split(","):
        left, _, right = item.strip().partition(":")
        if len(left) == 8 and left.isdigit():
            date, code = int(left), right
        else:
            code, date = left, int(right)
        code = code.strip().zfill(6)
        try:
            prev, old = old_values(mod, captured, code, date)
        except Exception as exc:  # noqa: BLE001
            print(f"{code} {date}: 旧框架失败 {exc!r}")
            all_ok = False
            continue
        try:
            new = list(rp.okay_compute_stock(args.series, code, date))
        except Exception as exc:  # noqa: BLE001
            print(f"{code} {date}: Rust 侧失败 {exc}")
            all_ok = False
            continue
        res = compare(old, new, args.rtol, args.atol)
        res.update(code=code, date=date, prev=prev)
        report["pairs"].append(res)
        flag = "一致" if not res["bad"] else f"有 {len(res['bad'])} 处不一致"
        print(
            f"{code} {date}（上一交易日 {prev}）: 旧 {res['n_old']} 值 / 新 {res['n_new']} 值，"
            f"NaN {res['nan']}，近零放行 {res['near_zero']}，{flag}"
        )
        for i, a, b, why in res["bad"][:12]:
            nm = names[i] if i < len(names) else "?"
            print(f"    #{i} {nm}: 旧={a:.10g} 新={b:.10g} ({why})")
        all_ok = all_ok and not res["bad"]

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("结论:", "通过" if all_ok else "不通过")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
