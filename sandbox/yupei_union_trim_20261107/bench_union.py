# 跑一天 yupei_union_all（沙箱版），结果存 results/<tag>_<date>.npy/.meta
# 用法: RAYON_NUM_THREADS=50 YUPEI_STAGE=1 python -I bench_union.py <tag> [date]
import sys, time, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D + "/package")
import rust_pyfunc as mod
assert mod.__file__.startswith(D), mod.__file__
tag = sys.argv[1]; date = int(sys.argv[2]) if len(sys.argv) > 2 else 20241231
t = time.perf_counter()
codes, names, vals = mod.py_yupei_union_all(date)
wall = time.perf_counter() - t
v = np.asarray(vals, dtype=np.float32).reshape(len(codes), len(names))
out = f"{D}/results/{tag}_{date}"
np.save(out + ".npy", v)
with open(out + ".meta", "w") as f:
    f.write("\n".join(codes) + "\n##\n" + "\n".join(names))
print(f"RESULT tag={tag} date={date} wall={wall:.1f}s shape={v.shape} load={os.getloadavg()[0]:.0f}")
