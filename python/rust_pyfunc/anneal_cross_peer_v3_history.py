"""柜员游戏拓展3 v3 组合 pipeline：第二轮开始前，把第一轮的中间数据取到一个临时目录。

中间数据的正式存放位置只有一处：研究期是第一轮的因子库，生产期是 h5（write_selected_factor_base 写的
base 目录）。这里产生的目录只是运行时缓存：第二轮读完就可以删，不需要长期维护。

两个入口只差数据来源，产出完全相同：
  prepare_anneal_cross_peer_v3_history_from_store  研究期，从第一轮的因子库读
  prepare_anneal_cross_peer_v3_history_from_h5     生产期，从 base h5 读
都会设置环境变量 ACP3_PROFILE_DIR，第二轮的 worker 进程会继承它。
第二轮任一 task 需要的过去 60 个交易日里缺任何一天，这里直接报错，不带着残缺数据继续。
"""
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HIST_DAYS = 60
_MAGIC = b"ACP3"


def _axes():
    import rust_pyfunc as rp

    names = list(rp.py_anneal_cross_peer_v3_profile_names())
    n_prof = sum(1 for n in names if not n.startswith("acp3p_shape_"))
    return rp, names, n_prof


def _needed_dates(task_dates, trading_days):
    days = sorted(int(d) for d in trading_days)
    pos = {d: i for i, d in enumerate(days)}
    need = set()
    for d in task_dates:
        d = int(d)
        if d not in pos:
            raise ValueError(f"{d} 不是交易日")
        if pos[d] < HIST_DAYS:
            raise ValueError(f"{d} 之前的交易日不足 {HIST_DAYS} 天")
        need.update(days[pos[d] - HIST_DAYS : pos[d]])
    return sorted(need), days


def _reset_dir(out_dir, days):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "calendar.txt"), "w") as f:
        f.write("\n".join(str(d) for d in days))


def _write_day(out_dir, date, codes, values, n_prof):
    """values: 股票 × 全部中间列。全天成交额有效的股票才写入；分钟占比的空值是真实的 0（该分钟无成交）。"""
    keep = np.isfinite(values[:, 0])
    if not keep.any():
        raise ValueError(f"中间数据 {date} 整天为空")
    codes = [c for c, k in zip(codes, keep) if k]
    v = values[keep]
    prof = v[:, :n_prof].astype("<f4")
    shape = np.nan_to_num(v[:, n_prof:], nan=0.0).astype("<f4")
    tmp = os.path.join(out_dir, f".{date}.tmp")
    with open(tmp, "wb") as f:
        f.write(_MAGIC)
        for x in (len(codes), prof.shape[1], shape.shape[1]):
            f.write(int(x).to_bytes(4, "little"))
        f.write("".join(codes).encode())
        f.write(prof.tobytes())
        f.write(shape.tobytes())
    os.replace(tmp, os.path.join(out_dir, f"{date}.bin"))


def _read_col(args):
    import rust_pyfunc as rp

    store_dir, col = args
    tri = rp.factor_store_v5_read_factor(store_dir, col)
    return (
        col,
        np.asarray(tri["date_id"]).astype(np.int64),
        np.asarray(tri["code_id"]).astype(np.int64),
        np.asarray(tri["factor"], dtype=np.float32),
    )


def prepare_anneal_cross_peer_v3_history_from_store(store_dir, task_dates, out_dir, n_proc=16):
    """研究期：从第一轮的因子库取第二轮 task_dates 需要的中间数据。"""
    rp, names, n_prof = _axes()
    need, days = _needed_dates(task_dates, rp.td.trading_days)
    info = rp.factor_store_v5_info(store_dir)
    if list(info["factor_names"]) != names:
        raise ValueError(f"{store_dir} 的列名与中间数据列名不一致")
    tmpl = rp.factor_store_v5_template(store_dir)
    store_dates = [int(d) for d in tmpl["dates"]]
    missing = sorted(set(need) - set(store_dates))
    if missing:
        raise ValueError(f"中间数据缺 {len(missing)} 天: {missing[:10]}")
    pos = {d: i for i, d in enumerate(need)}
    date_map = np.array([pos.get(d, -1) for d in store_dates], dtype=np.int64)
    codes = [str(s)[:6] for s in tmpl["stocks"]]
    cube = np.full((len(need), len(codes), len(names)), np.nan, dtype=np.float32)
    with ProcessPoolExecutor(n_proc) as ex:
        for col, di, si, val in ex.map(_read_col, [(store_dir, c) for c in range(len(names))]):
            r = date_map[di]
            ok = r >= 0
            cube[r[ok], si[ok], col] = val[ok]
    _reset_dir(out_dir, days)
    for r, d in enumerate(need):
        _write_day(out_dir, d, codes, cube[r], n_prof)
    os.environ["ACP3_PROFILE_DIR"] = out_dir
    return out_dir


def prepare_anneal_cross_peer_v3_history_from_h5(base_factor_ver, base_hdf5_dir, task_dates, out_dir):
    """生产期：从 base h5 取第二轮 task_dates 需要的中间数据。"""
    rp, names, n_prof = _axes()
    need, days = _needed_dates(task_dates, rp.td.trading_days)
    frames = [
        rp.read_factor(base_factor_ver, n, need[0], need[-1], 1, base_hdf5_dir).reindex(index=need)
        for n in names
    ]
    codes = [str(c).zfill(6)[:6] for c in frames[0].columns]
    cube = np.stack([f.to_numpy(dtype=np.float32) for f in frames], axis=2)
    _reset_dir(out_dir, days)
    for r, d in enumerate(need):
        _write_day(out_dir, d, codes, cube[r], n_prof)
    os.environ["ACP3_PROFILE_DIR"] = out_dir
    return out_dir
