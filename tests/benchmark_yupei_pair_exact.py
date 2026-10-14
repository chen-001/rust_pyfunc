"""50 线程、默认全量因子测速；用 float32 位模式比较，包含 NaN/Inf/正负零。

用 PYTHONPATH 指定待测包；在新进程中运行，避免 Rayon 已初始化。
输出目录保存结果，用 --reference 指定另一轮输出目录进行严格对比。
"""
import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
import time

os.environ["RAYON_NUM_THREADS"] = "50"
import numpy as np
import rust_pyfunc as rp
import rust_pyfunc.rust_pyfunc as ext

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--reference", type=Path)
p.add_argument("--date", type=int, default=20241231)
p.add_argument("--mats", nargs="+")
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
meta = dict(module=ext.__file__, threads=50, date=args.date, mats=args.mats,
            sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
            load_before=os.getloadavg())
print(json.dumps(meta), flush=True)
t = time.perf_counter()
codes, names, vals = rp.py_yupei_pair(args.date, args.mats)
meta.update(seconds=time.perf_counter() - t, load_after=os.getloadavg(),
            codes=codes, names=names)
meta["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
v = np.asarray(vals, dtype=np.float32).reshape(len(codes), len(names))
del vals
np.save(args.output / "values.npy", v)
if args.reference:
    ref = json.loads((args.reference / "result.json").read_text())
    assert ref.get("date", args.date) == args.date, "基准日期不同"
    old = np.load(args.reference / "values.npy", mmap_mode="r")
    meta["order_equal"] = codes == ref["codes"] and names == ref["names"]
    assert meta["order_equal"] and v.shape == old.shape, "股票/列顺序或形状改变"
    meta["bit_differences"] = int(np.count_nonzero(v.view("u4") != old.view("u4")))
    meta["nan_mismatches"] = int(np.count_nonzero(np.isnan(v) != np.isnan(old)))
(args.output / "result.json").write_text(json.dumps(meta, ensure_ascii=False))
print(json.dumps({k: x for k, x in meta.items() if k not in ("codes", "names")}), flush=True)
if args.reference:
    assert meta["bit_differences"] == 0, "存在位模式不同的结果"
