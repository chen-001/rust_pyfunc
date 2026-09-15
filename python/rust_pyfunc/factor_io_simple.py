"""dwo 因子 IO/衍生 简化版 —— 不依赖 szalpha 研究框架，纯 pandas + h5py。

替代 design_whatever_out 的 save_factor / read_factor / get_abs / base_to_final，
供 _okay.py 实盘每日更新文件使用（rp.save_factor / rp.read_factor / rp.get_abs / rp.base_to_final）。

存储约定：与 szalpha FactorWriter/FactorReader 的 h5 目录逐项一致，
{hdf5_dir_here}/{factor_version}/ 下：
    symbol_map.csv    股票列表（symbol,pos），从 basic_info 复制
    calendar_map.csv  交易日列表（date_min），从 basic_info/calendar.csv 复制
    capacity.csv      容量（name,count：symbol_capacity / date_capacity）
    fields.csv        该版本已写入的因子名（name）
    {name}.h5         dataset "data"，float64，shape=(date_capacity, symbol_capacity)，
                      行 = 交易日在日历里的位置，列 = symbol_map 的顺序，没写过的位置是空值
所以 dwo / szalpha 的读取端能直接读本模块写的文件，反过来也一样，不需要 szalpha。

与 design_whatever_out 的行为对齐点：
- save_factor：只写 [next_trading_day_tricky(start_date), last_trading_day_tricky(end_date)]
  这一段交易日，按日历位置整块写；块内缺失的日期写空值；列按 symbol_map 顺序对齐；
  输入是 DatetimeIndex 时先转成 int 日期
- read_factor：起始行 = start_date 往前多算 look_back_window-1 个交易日（默认就是 start_date
  当天），早于 20150105 的截到 20150105（对齐 szalpha 的 largest_hf_start_date）
- get_abs：每行（截面）减行均值取绝对值（square/quantile 分支一致）
- base_to_final：{base}_{action}_smooth_{days} 命名解析 + rank(axis=1) + 滚动算子，
  回看 start_date 前 60 个交易日；smooth_1 直接输出 rank（含 dwo 的 endswith('1') 语义）
"""
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import shutil

import h5py
import numpy as np
import pandas as pd

from .trading_day import (
    _get_td,
    _resolve_calendar_path,
    last_n_trading_date,
    last_trading_day_tricky,
    next_trading_day_tricky,
)

HF_START_DATE = 20150105  # szalpha CalculatorBase.largest_hf_start_date


def _read_symbols(data_dir: str) -> list:
    """版本目录里的股票列表（symbol_map.csv 的 symbol 列，顺序即 h5 的列顺序）。"""
    df = pd.read_csv(os.path.join(data_dir, "symbol_map.csv"), dtype={"symbol": str})
    return df["symbol"].tolist()


def _read_calendar(data_dir: str) -> tuple:
    """版本目录里的交易日列表 + 日期到行号的映射。"""
    dates = pd.read_csv(os.path.join(data_dir, "calendar_map.csv"), dtype=np.int64)["date_min"].to_numpy()
    return dates, dict(zip(dates.tolist(), range(len(dates))))


def _read_capacity(data_dir: str) -> tuple:
    """版本目录里的 (日期容量, 股票容量)。"""
    df = pd.read_csv(os.path.join(data_dir, "capacity.csv"), index_col="name")
    return int(df.at["date_capacity", "count"]), int(df.at["symbol_capacity", "count"])


def _ensure_version_dir(data_dir: str, name: str) -> str:
    """按 szalpha 的 h5 目录约定补齐配置文件和 {name}.h5，返回 h5 路径。"""
    os.makedirs(data_dir, exist_ok=True)

    symbol_file = os.path.join(data_dir, "symbol_map.csv")
    if not os.path.exists(symbol_file):
        shutil.copyfile(_resolve_calendar_path("symbol_map.csv"), symbol_file)

    calendar_file = os.path.join(data_dir, "calendar_map.csv")
    if not os.path.exists(calendar_file):
        pd.read_csv(_resolve_calendar_path("calendar.csv"), header=None, names=["date_min"]).to_csv(
            calendar_file, index=False
        )

    capacity_file = os.path.join(data_dir, "capacity.csv")
    if not os.path.exists(capacity_file):
        cfg = pd.read_csv(_resolve_calendar_path("capacity_config.csv"))
        pd.DataFrame(
            {
                "name": ["symbol_capacity", "date_capacity"],
                "count": [cfg["symbol_capacity"][0], cfg["date_capacity"][0]],
            }
        ).to_csv(capacity_file, index=False)

    fields_file = os.path.join(data_dir, "fields.csv")
    names = pd.read_csv(fields_file)["name"].tolist() if os.path.exists(fields_file) else []
    if name not in names:
        pd.DataFrame({"name": names + [name]}).to_csv(fields_file, index=False)

    h5_file = os.path.join(data_dir, f"{name}.h5")
    if not os.path.exists(h5_file):
        date_capacity, symbol_capacity = _read_capacity(data_dir)
        with h5py.File(h5_file, "w") as f:
            f.create_dataset(
                "data",
                shape=(date_capacity, symbol_capacity),
                dtype=np.float64,
                maxshape=(date_capacity, None),
                chunks=True,
                fillvalue=np.nan,
            )
    return h5_file


def _refresh_calendar(data_dir: str) -> None:
    """日历变长时补齐版本目录里的 calendar_map.csv（对齐 szalpha 的 update_latest_calendar）。"""
    path = os.path.join(data_dir, "calendar_map.csv")
    dates = _get_td().trading_days
    if not os.path.exists(path) or len(pd.read_csv(path, dtype=np.int64)) < len(dates):
        pd.DataFrame({"date_min": dates}).to_csv(path, index=False)


def _index_to_int(index: pd.Index) -> pd.Index:
    """统一日期索引为 int YYYYMMDD（DatetimeIndex 转 int，int 原样保留）。"""
    if isinstance(index, pd.DatetimeIndex):
        return index.strftime("%Y%m%d").astype(np.int64)
    return index.astype(np.int64)


def save_factor(
    factor_df: pd.DataFrame,
    factor_version: str,
    name: str,
    start_date: int = 20160101,
    end_date: int = 20240229,
    look_back_window: int = 1,
    hdf5_dir_here: str = None,
) -> None:
    """写因子 h5：只写 [start_date, end_date] 之间的交易日，行列按版本目录的日历和股票表对齐。

    目录 {hdf5_dir_here}/{factor_version}/ 不存在时自动建好配置文件和 {name}.h5。
    look_back_window 只对读取端有意义，写入端和 dwo 一样忽略它。
    """
    data_dir = os.path.join(hdf5_dir_here, factor_version)
    h5_file = _ensure_version_dir(data_dir, name)
    _refresh_calendar(data_dir)
    dates, date_pos = _read_calendar(data_dir)
    symbols = _read_symbols(data_dir)
    start_pos = date_pos[next_trading_day_tricky(start_date)]
    end_pos = date_pos[last_trading_day_tricky(end_date)] + 1
    block = factor_df.set_axis(_index_to_int(factor_df.index), axis=0).reindex(
        index=dates[start_pos:end_pos], columns=symbols
    )
    with h5py.File(h5_file, "r+") as f:
        f["data"][start_pos:end_pos, : len(symbols)] = block.to_numpy(np.float64)


def read_factor(
    factor_version: str,
    name: str,
    start_date: int = 20160101,
    end_date: int = 20240229,
    look_back_window: int = 1,
    hdf5_dir_here: str = None,
) -> pd.DataFrame:
    """读因子 h5，返回 index=交易日(int) / columns=股票 的 DataFrame。

    起始行是 start_date 往前 look_back_window-1 个交易日，早于 20150105 的截到 20150105。
    """
    data_dir = os.path.join(hdf5_dir_here, factor_version)
    dates, date_pos = _read_calendar(data_dir)
    symbols = _read_symbols(data_dir)
    start_pos = max(date_pos[next_trading_day_tricky(start_date)] - look_back_window + 1, 0)
    if dates[start_pos] < HF_START_DATE:
        start_pos = date_pos[HF_START_DATE]
    end_pos = date_pos[last_trading_day_tricky(end_date)] + 1
    with h5py.File(os.path.join(data_dir, f"{name}.h5"), "r") as f:
        data = f["data"][start_pos:end_pos, : len(symbols)]
    return pd.DataFrame(data, index=dates[start_pos:end_pos], columns=symbols)


def get_abs(df: pd.DataFrame, quantile: float = None, square: int = 0) -> pd.DataFrame:
    """均值距离化：每行（截面）减去行均值后取绝对值；square=1 取平方，quantile 改算到分位点距离。

    与 dwo.get_abs 表达式逐位一致。
    """
    if not square:
        if quantile is not None:
            return np.abs((df.T - df.T.quantile(quantile)).T)
        return np.abs((df.T - df.T.mean()).T)
    if quantile is not None:
        return ((df.T - df.T.quantile(quantile)).T) ** 2
    return ((df.T - df.T.mean()).T) ** 2


def _autocorr_df(a: pd.DataFrame, backsee: int) -> pd.DataFrame:
    """滚动窗口内逐列 1 阶自相关：窗口 [i-backsee+1..i] 的 corrwith(shift(1))，与 dwo 完全一致。"""
    dates = list(a.index)
    res = []
    for num, i in enumerate(dates[backsee - 1:]):
        son = a.iloc[num:num + backsee, :]
        res.append(son.corrwith(son.shift(1)).to_frame(i).T)
    return pd.concat(res)


def base_to_final(
    names_okay: list[str],
    start_date: int,
    end_date: int,
    base_factor_ver: str,
    base_hdf5_dir: str,
) -> dict[str, pd.DataFrame]:
    """base 因子 → final 因子（与 dwo.base_to_final 行为一致）。

    {base}_{action}_smooth_{days} 命名解析：
      smooth_1            → rank(axis=1) 直接输出
      mean/min/max/std    → rank 后 rolling(days, min_periods=days//2).{action}()
      autocorr            → rank 后滚动窗口内 1 阶自相关
    回看 start_date 前 60 个交易日（max(..., 20150105)），从 base parquet 读取。
    """
    start_date0 = max(last_n_trading_date(start_date, 60), 20150105)
    finals = {}
    for name in names_okay:
        infos = name.split("_")
        days = int(infos[-1])
        mdays = int(days / 2)
        action = infos[-3]
        if name.endswith("1"):
            n = "_".join(name.split("_")[:-2])
            df = read_factor(base_factor_ver, n, start_date0, end_date, 1, base_hdf5_dir).rank(axis=1)
            finals[name] = df
        elif action == "mean":
            n = "_".join(name.split("_")[:-3])
            df = read_factor(base_factor_ver, n, start_date0, end_date, 1, base_hdf5_dir).rank(axis=1)
            finals[name] = df.rolling(days, min_periods=mdays).mean()
        elif action == "min":
            n = "_".join(name.split("_")[:-3])
            df = read_factor(base_factor_ver, n, start_date0, end_date, 1, base_hdf5_dir).rank(axis=1)
            finals[name] = df.rolling(days, min_periods=mdays).min()
        elif action == "max":
            n = "_".join(name.split("_")[:-3])
            df = read_factor(base_factor_ver, n, start_date0, end_date, 1, base_hdf5_dir).rank(axis=1)
            finals[name] = df.rolling(days, min_periods=mdays).max()
        elif action == "std":
            n = "_".join(name.split("_")[:-3])
            df = read_factor(base_factor_ver, n, start_date0, end_date, 1, base_hdf5_dir).rank(axis=1)
            finals[name] = df.rolling(days, min_periods=mdays).std()
        elif action == "autocorr":
            n = "_".join(name.split("_")[:-3])
            df = read_factor(base_factor_ver, n, start_date0, end_date, 1, base_hdf5_dir).rank(axis=1)
            finals[name] = _autocorr_df(df, days)
    return finals


# ───────────────────── _okay.py 工具函数（替代 okay 文件内的下划线辅助函数） ─────────────────────

def get_factor_names(names_function: str) -> list:
    """调用 Rust 因子名函数（rp.py_xxx_names() 形式），返回全部因子名列表。"""
    import rust_pyfunc as rp

    return list(getattr(rp, names_function)())


def read_factor_from_colblk(
    store_dir: str, name: str, start_date: int, end_date: int
) -> pd.DataFrame:
    """从 colblk 列式存储读取单个因子，返回 DataFrame(date×stock)。

    替代 okay 文件里的 _read_factor_from_colblk：按 name 取子集，无需重算。
    """
    import rust_pyfunc as rp

    info = rp.factor_store_v5_info(store_dir)
    name_to_idx = {n: i for i, n in enumerate(info["factor_names"])}
    if name not in name_to_idx:
        raise KeyError(f"因子 {name} 不在 colblk 存储中")

    tmpl = rp.factor_store_v5_template(store_dir)
    all_dates = np.asarray(tmpl["dates"], dtype=np.int64)
    all_stocks = list(tmpl["stocks"])

    tri = rp.factor_store_v5_read_factor(store_dir, name_to_idx[name])
    mat = np.full((len(all_dates), len(all_stocks)), np.nan)
    mat[tri["date_id"].astype(int), tri["code_id"].astype(int)] = tri["factor"]

    mask = (all_dates >= start_date) & (all_dates <= end_date)
    return pd.DataFrame(
        mat[mask],
        index=pd.to_datetime(all_dates[mask].astype(str)),
        columns=all_stocks,
    )


def probe_update_window(
    colblk_store_dir: str, start_date: int, end_date: int
) -> tuple:
    """探测增量更新窗口（运行时动态，可移植）。

    start = colblk 存储已有最大日期（无存储则返回 start_date，适配新服务器首次全量算）
    end   = /ssd_data/stock 最新原始数据日期（无则返回 end_date）
    """
    import rust_pyfunc as rp

    start = start_date
    if os.path.isdir(colblk_store_dir):
        tmpl = rp.factor_store_v5_template(colblk_store_dir)
        dates = [int(d) for d in tmpl["dates"]]
        start = max(dates) if dates else start_date
    end = end_date
    stock_dir = "/ssd_data/stock"
    if os.path.isdir(stock_dir):
        raw_dates = [int(d) for d in os.listdir(stock_dir) if d.isdigit()]
        if raw_dates:
            end = max(raw_dates)
    return start, end


def read_level2_list(start_date: int, end_date: int) -> list:
    """生成 [start_date, end_date] 内真实存在的 (date, code) 任务列表（per_stock pipeline 用）。

    替代 dw.read_level2_list：日期对齐交易日 → symbol_map.csv 全股票列表 →
    按日目录扫描 transaction 文件集合过滤存在性。每日更新仅 1 天，直接扫描无性能问题。
    """
    import rust_pyfunc as rp

    dates = sorted(
        rp.td.get_range(
            rp.next_trading_day_tricky(start_date),
            rp.last_trading_day_tricky(end_date),
        )
    )
    symbol_map_path = "/ssd_data/data/basic_info/symbol_map.csv"
    symbol_list = pd.read_csv(symbol_map_path).symbol.astype(str).str.zfill(6).tolist()
    pairs = []
    for date in dates:
        day_dir = f"/ssd_data/stock/{date}/transaction"
        if not os.path.isdir(day_dir):
            continue
        files = {f for f in os.listdir(day_dir) if f.endswith("_transaction.csv")}
        for symbol in symbol_list:
            if f"{symbol}_{date}_transaction.csv" in files:
                pairs.append([date, symbol])
    return pairs


def cleanup_factor_store(colblk_store_dir: str, ver: str, script_dir: str = None) -> None:
    """清理批量计算产生的备份数据文件与 colblk 存储目录。

    backup_{ver}* 与 colblk 存储目录是批量计算（全量几千几万个因子）的中间产物，
    names_save 已单独物化到 hdf5 base，其余因子后续用不上，执行完毕删除。
    script_dir 默认取调用方脚本目录；colblk 存储目录按传入值清理。
    """
    import glob
    import shutil

    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    removed = []
    for path in glob.glob(os.path.join(script_dir, f"backup_{ver}*")):
        if os.path.isfile(path):
            os.remove(path)
            removed.append(os.path.basename(path))
    if os.path.isdir(colblk_store_dir):
        shutil.rmtree(colblk_store_dir)
        removed.append(os.path.basename(colblk_store_dir.rstrip("/")))
    if removed:
        print(f"🧹 已清理计算过程备份: {removed}")
