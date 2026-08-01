"""dwo 因子 IO/衍生 简化版 —— 不依赖 szalpha 研究框架，纯 pandas + parquet。

替代 design_whatever_out 的 save_factor / read_factor / get_abs / base_to_final，
供 _okay.py 实盘每日更新文件使用（rp.save_factor / rp.read_factor / rp.get_abs / rp.base_to_final）。

存储约定：因子统一存 parquet（列 = ['date'(int YYYYMMDD), 股票1, 股票2, ...]）。
szalpha 的 FactorReader.read_date_data 优先读 {name}.parquet，因此本模块的产物
可被既有 szalpha 生态直接读取；反过来本模块不读 h5（历史 h5 数据请先用 dwo 读出再转存）。

与 design_whatever_out 的行为对齐点：
- save_factor：日期窗口 [start_date, end_date] 过滤后落盘；index 统一转 int 日期
- read_factor：返回 index=date(int) / columns=股票 的 DataFrame，按窗口过滤
- get_abs：每行（截面）减行均值取绝对值（square/quantile 分支一致）
- base_to_final：{base}_{action}_smooth_{days} 命名解析 + rank(axis=1) + 滚动算子，
  回看 start_date 前 60 个交易日；smooth_1 直接输出 rank（含 dwo 的 endswith('1') 语义）
"""
import os

import numpy as np
import pandas as pd

from .trading_day import last_n_trading_date


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
    """写因子 parquet：date 列（int YYYYMMDD）+ 股票列，仅保留 [start_date, end_date] 窗口。

    等价于 dwo.save_factor 的窗口过滤 + 落盘；目录 {hdf5_dir_here}/{factor_version}/ 自动创建。
    """
    out_dir = os.path.join(hdf5_dir_here, factor_version)
    os.makedirs(out_dir, exist_ok=True)
    df = factor_df[(factor_df.index >= start_date) & (factor_df.index <= end_date)]
    out = df.copy()
    out.index = _index_to_int(out.index)
    out = out.reset_index()
    out = out.rename(columns={out.columns[0]: "date"})
    out.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)


def read_factor(
    factor_version: str,
    name: str,
    start_date: int = 20160101,
    end_date: int = 20240229,
    look_back_window: int = 1,
    hdf5_dir_here: str = None,
) -> pd.DataFrame:
    """读因子 parquet，返回 index=date(int) / columns=股票 的 DataFrame，按日期窗口过滤。

    与 szalpha FactorReader.read_date_data 的 parquet 分支逐行等价。
    """
    df = pd.read_parquet(os.path.join(hdf5_dir_here, factor_version, f"{name}.parquet"))
    if "date" in df.columns:
        df = df.set_index("date")
    return df[(df.index >= start_date) & (df.index <= end_date)]


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
