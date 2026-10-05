"""新旧对照验证：跑旧框架（design_whatever_out）的因子函数，和 Rust 复现值逐值比。

用法：
    python lib/okay_database/verify_series.py --series hm029 --okay /path/hm29_okay.py \
        --pairs 20250303:000001,20250304:600519

原理：把旧 okay 文件当模块导入，临时替换掉 dwo.run_factor_out，
只截获它里面定义的 Go 类，不触发任何写盘。然后用旧框架自己的读取与计算跑单股单日，
再和 rp.okay_compute_stock 比。

两种系列都支持：
    逐股系列（compute_stock）：旧 get_factor() 返回一串数字，直接逐值比。
    横截面系列（compute_cross）：旧 get_factor() 返回的是一串 pandas Series
        （index 是股票代码），按 index 取出目标股票那一项再逐值比；
        Rust 侧走 rp.okay_compute_cross(series, date)，返回值是**股票优先**排布的
        （每只股票连续 expected_len 个值），按 codes 里目标股票的位置切出来。

环境变量 OKAY_VERIFY_DATA_ROOT（可选，默认为空 = 用真实全量数据）：
    指向一个形如 <root>/stock/<date>/transaction/*.csv 与 <root>/basic_info/*.csv 的目录树。
    设了它就把旧框架的 dwo.path_to_be_decided 和 Rust 侧的 RUST_PYFUNC_LEVEL2_ROOT
    同时指过去，两边看到的是同一批股票文件，对照才成立。
    用途：横截面系列的旧代码一次要读**一天全市场**（6000 只股票、约 23GB、几十分钟），
    调试和验收时先拿一个只放几百只股票软链接的子集目录把逻辑对齐，
    最后再不打这个变量、用真实全量数据复核一组。
    子集目录里 stock/<date>/transaction/ 放软链接，basic_info/ 放 calendar.csv 与
    symbol_map.csv 的软链接（名单保持全量没关系，读到哪些文件就看目录里有哪些）。

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
DATA_ROOT_ENV = "OKAY_VERIFY_DATA_ROOT"


def data_root() -> str:
    """可选的数据目录覆盖（见文件头）。没设就返回空串。"""
    return os.environ.get(DATA_ROOT_ENV, "").strip()


def load_old(okay_path: str):
    """导入旧 okay 模块，返回 (module, captured dict)。"""
    sys.path.insert(0, PYCODE)
    sys.path.insert(0, str(Path(okay_path).parent))
    import design_whatever_out as dwo

    root = data_root()
    if root:
        # 旧框架的 read_trade_data_full_market 按运行时全局查 path_to_be_decided，
        # 所以在调用 get_factor 之前改这个全局就能把整条链指到子集目录。
        dwo.path_to_be_decided = root
        print(f"[数据] 旧框架数据根目录已覆盖为 {root}（{DATA_ROOT_ENV}）")

    captured: dict = {}
    original = dwo.run_factor_out

    def fake(*args, **kwargs):
        captured.update(kwargs)
        captured["Go"] = kwargs.get("Go") or (args[0] if args else None)
        return None

    dwo.run_factor_out = fake
    # 有的旧 okay 文件 run_base 里还直接调 dwo.run_factor_rust_bin（不经过 run_factor_out），
    # 那是真的去跑全历史，会把对账挂死几个小时。这里一并拦掉，只取它定义的类/函数。
    _orig_rust_bin = getattr(dwo, "run_factor_rust_bin", None)

    def fake_rust_bin(*a, **k):
        captured.setdefault("run_factor_rust_bin_called", True)
        return None

    if _orig_rust_bin is not None:
        dwo.run_factor_rust_bin = fake_rust_bin
    name = Path(okay_path).stem
    mod = importlib.import_module(name)
    try:
        # 旧模块的日历/日频表是按这里的起止日期建的。范围太窄会让「回看 N 个交易日前」
        # 这类因子在窗口外取不到值（旧代码 KeyError 被 try 吞成 NaN），于是对账看起来通过
        # 其实两边都是 NaN。默认给一个覆盖全历史的宽范围，需要时用 OKAY_VERIFY_CAL_RANGE 改。
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
    """按旧框架的调用方式算单股单日的原始值。

    返回 (values, is_cross)：
      - 逐股系列：values 是一串数字，is_cross=False
      - 横截面系列：旧的 get_factor() 返回一串 pandas Series（index 是股票代码），
        values 是按 index 取出的目标股票那一串，is_cross=True
    """
    import pandas as pd

    if "Go" in captured and captured["Go"] is not None:
        cls = captured["Go"]
        kwargs = dict(
            read_full=captured.get("read_full", 0) or 0,
            get_many=captured.get("get_many", 0) or 0,
            change_time=captured.get("change_time", 1),
        )
        if getattr(cls, "check_my_kind", "level2") == "minute":
            # 分钟系列的 Go 继承的是 dwo.FactorDesign，它的 __init__ 不认识 change_time
            # （那是 TickDataBase 的参数），传进去只会 TypeError。
            kwargs.pop("change_time", None)
            rc = cls(None, date, date, 1, captured.get("base_ver"), **kwargs)
        else:
            rc = cls(1, date, date, 1, symbol=code, **kwargs)
        res = rc.get_factor()
        if res is None:
            return None, False
        res = list(res)
        if res and isinstance(res[0], pd.Series):
            return [float(s.get(code, float("nan"))) for s in res], True
        if res and isinstance(res[0], pd.DataFrame):
            # 分钟系列：get_factor() 返回的是「一天一行 × 全市场列」的 DataFrame 列表
            # （旧代码里 `i.to_frame(self.end_date).T`），按列取出目标股票那一个值。
            out = []
            for s in res:
                out.append(float(s[code].iloc[0]) if code in s.columns else float("nan"))
            return out, True
        return [None if v is None else float(v) for v in res], False
    if hasattr(mod, "go"):
        res = mod.go(date, code)
        return (None if res is None else list(res)), False
    raise RuntimeError("既没截获到 Go 类，也没有 go 函数")


def compare(old, new, rtol: float, atol: float = 1e-9):
    if old is None:
        old = []
    n = max(len(old), len(new))
    bad = []
    nan_old = nan_new = 0
    near_zero = 0
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
        # 无穷要单独判：inf - inf 是 nan，按相对误差比会把「两边都是 inf」误判成不一致
        if math.isinf(a) or math.isinf(b):
            if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
                continue
            bad.append((i, a, b, "无穷不一致"))
            continue
        if abs(a) < atol and abs(b) < atol:
            # 两边都落在数值零附近：这种量是求和顺序造成的舍入噪声，相对误差没有意义，
            # 数学上恒等于 0 的量（比如标准化后的回归截距）永远过不了相对误差门槛。
            # 单独记一笔，回报时要说清有多少个位置是靠这条放行的。
            near_zero += 1
            continue
        denom = max(abs(a), abs(b), 1e-12)
        rel = abs(a - b) / denom
        if not (rel <= rtol):
            bad.append((i, a, b, f"相对误差 {rel:.3e}"))
    return dict(n_old=len(old), n_new=len(new), nan=nan_old, near_zero=near_zero, bad=bad)


def cross_values(series: str, code: str, date: int, rp):
    """横截面系列：调 okay_compute_cross，按「股票优先」切出目标股票的 expected_len 个值。"""
    codes, vals = rp.okay_compute_cross(series, date)
    n = len(rp.okay_series_names(series))
    if n == 0:
        raise RuntimeError(f"系列 {series} 没声明因子名")
    if len(codes) == 0:
        return [float("nan")] * n
    assert len(vals) == len(codes) * n, (
        f"横截面返回值长度 {len(vals)} 与 {len(codes)} 只股票 × {n} 个因子对不上"
    )
    if code not in codes:
        return [float("nan")] * n
    i = codes.index(code)
    return list(vals[i * n:(i + 1) * n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True, help="如 hm029")
    ap.add_argument("--okay", required=True)
    ap.add_argument("--pairs", default="20250303:000001,20250304:600519,20250305:000002")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-9,
                    help="数值零附近的绝对误差兜底；两边都小于它就按舍入噪声放行")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = data_root()
    if root:
        # Rust 侧的数据根目录：<root>/stock/<date>/transaction/...
        os.environ["RUST_PYFUNC_LEVEL2_ROOT"] = str(Path(root) / "stock")

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
        old, is_cross = old_values(mod, captured, code, date)
        try:
            if is_cross:
                new = cross_values(args.series, code, date, rp)
            else:
                new = rp.okay_compute_stock(args.series, code, date)
        except Exception as exc:  # noqa: BLE001
            print(f"{code} {date}: Rust 侧失败 {exc}")
            all_ok = False
            continue
        res = compare(old, new, args.rtol, args.atol)
        res.update(code=code, date=date, cross=is_cross)
        report["pairs"].append(res)
        flag = "一致" if not res["bad"] else f"有 {len(res['bad'])} 处不一致"
        kind = "横截面" if is_cross else "逐股"
        print(f"{code} {date} [{kind}]: 旧 {res['n_old']} 值 / 新 {res['n_new']} 值，NaN {res['nan']}，{flag}")
        for i, a, b, why in res["bad"][:10]:
            print(f"    #{i}: 旧={a:.10g} 新={b:.10g} ({why})")
        all_ok = all_ok and not res["bad"]

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("结论:", "通过" if all_ok else "不通过")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
