"""用 sandbox 里的 Rust 实现复刻 TLQ，并对账上线因子报告里的 base_TLQ。

数据：/nas197/binary/stock/sz_alpha（hf_data/hm94 的 35 个 gap5 因子 + vars 面板）
口径：与同事 factor_gate_pipeline_v20260825.py 一致（见 sandbox/src/tlq.rs 头部注释）
"""
import os
import h5py
import numpy as np
import pandas as pd

import dev_sandbox

ROOT = "/nas197/binary/stock/sz_alpha"
FACTOR_DIR = f"{ROOT}/hf_data/hm94"
REPORT = "/nas197/user_home_share/上线因子/factor_report_20261010/hm94/summary_hm94.csv"
CACHE = "/tmp/tlq_panels"
START = "2016-01-04"
END = "2026-07-30"
BARRA_NAMES = [
    "residual_volatility", "momentum", "earnings_yield", "growth", "liquidity",
    "size", "leverage", "book_to_price", "beta", "non_linear_size",
]


def windcode(symbol: str) -> str:
    code = str(symbol).strip().upper().split(".")[0].zfill(6)
    if code.startswith(("0", "3")):
        return code + ".SZ"
    if code.startswith("6"):
        return code + ".SH"
    if code.startswith("8"):
        return code + ".BJ"
    return code


def _read_h5(path: str, rows: np.ndarray, n_cols: int) -> np.ndarray:
    with h5py.File(path, "r") as store:
        return np.asarray(store["data"][rows, :n_cols], dtype=np.float64)


def build_panels(force: bool = False) -> dict:
    calendar = pd.read_csv(f"{ROOT}/basic_info/calendar.csv", header=None).iloc[:, 0]
    all_dates = pd.to_datetime(calendar.astype(str), format="%Y%m%d")
    symbols = pd.read_csv(f"{ROOT}/basic_info/symbol_map.csv", dtype={"symbol": str})
    all_stocks = pd.Index([windcode(s) for s in symbols["symbol"]])
    start, end = pd.Timestamp(START), pd.Timestamp(END)
    end_pos = int(np.searchsorted(all_dates.values, end.to_datetime64(), side="right"))
    end_pos = min(len(all_dates), end_pos + 20)
    mask = (all_dates >= start) & (np.arange(len(all_dates)) < end_pos)
    rows = np.nonzero(mask)[0]
    dates = pd.DatetimeIndex(all_dates[mask])
    n_cols = len(all_stocks)

    spec = {
        "restrict": "vars/Base/S_RESTRICT_bt.h5",
        "industry": "vars/Base/SW_IND_CODE.h5",
        "returns": "vars/Base/S_DQ_RET_VWAP_0930_1000.h5",
        "size": "vars/SzBa/size.h5",
    }
    for name in BARRA_NAMES:
        spec[f"barra_{name}"] = f"vars/SzBa/{name}.h5"

    panels = {}
    for key, rel in spec.items():
        path = f"{CACHE}/{key}.npy"
        if not force and os.path.exists(path):
            panels[key] = np.load(path)
        else:
            panels[key] = _read_h5(f"{ROOT}/{rel}", rows, n_cols)
            np.save(path, panels[key])
    panels["_dates"] = dates
    panels["_mean_size"] = n_cols
    return panels


def build_derived(panels: dict) -> tuple[np.ndarray, np.ndarray]:
    restrict = panels["restrict"]
    logret = np.log1p(panels["returns"])
    prev_ok = np.vstack([np.zeros((1, restrict.shape[1]), dtype=bool), restrict[:-1] == 0])
    market_mean = np.nanmean(np.where(prev_ok, logret, np.nan), axis=1)
    log_exret = logret - market_mean[:, None]

    dates = panels["_dates"]
    idx = pd.Series(np.arange(len(dates)), index=dates)
    week_first = (
        idx.resample("W").first().resample("D").first().shift(-6).ffill().reindex(idx.index)
    )
    week_first = week_first.where(dates > pd.Timestamp(START), -1).fillna(-1).astype(np.int32)
    return log_exret, week_first.to_numpy()


def main(names: list[str]) -> pd.DataFrame:
    panels = build_panels()
    log_exret, week_first = build_derived(panels)
    dates = panels["_dates"]
    barra_ranked = dev_sandbox.sandbox_interleave_styles(
        [dev_sandbox.sandbox_rank_pct_rows(panels[f"barra_{n}"]) for n in BARRA_NAMES],
        panels["restrict"].shape[1],
    )
    size_ranked = dev_sandbox.sandbox_rank_pct_rows(panels["size"])
    restrict = panels["restrict"]
    industry = panels["industry"]
    size = panels["size"]
    group_to = int(np.searchsorted(dates.values, pd.Timestamp(END).to_datetime64(), side="right"))
    print(f"面板就绪 dates={len(dates)} stocks={restrict.shape[1]} TLQ 区间=0..{group_to}", flush=True)

    with h5py.File(f"{FACTOR_DIR}/{names[0]}.h5", "r") as f:
        shape = f["data"].shape
    rows = None
    report = pd.read_csv(REPORT).set_index("factor_name")
    out = []
    for name in names:
        t0 = pd.Timestamp.now()
        if rows is None:
            calendar = pd.read_csv(f"{ROOT}/basic_info/calendar.csv", header=None).iloc[:, 0]
            all_dates = pd.to_datetime(calendar.astype(str), format="%Y%m%d")
            end_pos = int(np.searchsorted(all_dates.values, pd.Timestamp(END).to_datetime64(), side="right"))
            end_pos = min(len(all_dates), end_pos + 20)
            rows = np.nonzero(
                (all_dates >= pd.Timestamp(START)) & (np.arange(len(all_dates)) < end_pos)
            )[0]
        factor = _read_h5(f"{FACTOR_DIR}/{name}.h5", rows, restrict.shape[1])
        tlq, rets, curves = dev_sandbox.sandbox_tlq_base_resid(
            factor, log_exret, restrict, industry, barra_ranked, len(BARRA_NAMES),
            size_ranked, week_first, 0, group_to, 160,
        )
        out.append(
            {
                "factor_name": name,
                "我的TLQ": float(tlq),
                "报告base_TLQ": float(report.loc[name, "base_TLQ"]),
                "秒": round((pd.Timestamp.now() - t0).total_seconds(), 1),
            }
        )
        print(out[-1], flush=True)
        np.save(f"{CACHE}/curves_{name}.npy", curves)
    res = pd.DataFrame(out)
    res["差"] = (res["我的TLQ"] - res["报告base_TLQ"]).round(4)
    return res


if __name__ == "__main__":
    report = pd.read_csv(REPORT).set_index("factor_name")
    ordered = report["base_TLQ"].sort_values(ascending=False)
    picks = [
        ordered.index[0], ordered.index[1], ordered.index[2],
        ordered.index[len(ordered) // 2], ordered.index[-2], ordered.index[-1],
    ]
    result = main(picks)
    print(result.round(4).to_string(index=False), flush=True)
    result.to_csv(f"{CACHE}/tlq_replicate_result.csv", index=False)
