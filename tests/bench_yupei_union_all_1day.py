"""yupei_union_all 单日计时。用法: python tests/bench_yupei_union_all_1day.py [date] [threads]

threads 通过 RAYON_NUM_THREADS 控制（整个函数的内部并行度）。
分段耗时由 YUPEI_STAGE=1 打到 stderr，本脚本负责墙钟、产物形状、结果指纹与全空列清单。
输出 JSON 写到 reports/yupei_union_all_bench/run_<date>_<threads>.json。
"""
import hashlib
import json
import os
import sys
import time

date = int(sys.argv[1]) if len(sys.argv) > 1 else 20241231
os.environ["RAYON_NUM_THREADS"] = sys.argv[2] if len(sys.argv) > 2 else "50"
threads = os.environ["RAYON_NUM_THREADS"]

import numpy as np
import rust_pyfunc as rp

print(f"date={date} RAYON_NUM_THREADS={threads}", flush=True)

t = time.perf_counter()
codes, names, vals = rp.py_yupei_union_all(date)
t_call = time.perf_counter() - t

t = time.perf_counter()
U = np.asarray(vals, dtype=np.float32).reshape(len(codes), -1)
t_np = time.perf_counter() - t

allnan = np.isnan(U).all(axis=0)
report = {
    "date": date,
    "threads": int(threads),
    "shape": list(U.shape),
    "call_seconds": round(t_call, 1),
    "numpy_seconds": round(t_np, 1),
    "sha256_f32": hashlib.sha256(np.ascontiguousarray(U).tobytes()).hexdigest(),
    "allnan_columns": [n for n, bad in zip(names, allnan) if bad],
    "allnan_rows": int(np.isnan(U).all(axis=1).sum()),
    "finite_fraction": round(float(1 - np.isnan(U).mean()), 4),
    "inf_count": int(np.isinf(U).sum()),
}
os.makedirs("reports/yupei_union_all_bench", exist_ok=True)
out = f"reports/yupei_union_all_bench/run_{date}_{threads}.json"
with open(out, "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)

print(f"shape={U.shape}  调用墙钟={t_call:.1f}s  转 numpy={t_np:.1f}s", flush=True)
print(f"全空列={len(report['allnan_columns'])}  全空行={report['allnan_rows']}", flush=True)
print(f"非空比例={report['finite_fraction']}  无穷={report['inf_count']}", flush=True)
print(f"指纹={report['sha256_f32'][:16]}  报告={out}", flush=True)
