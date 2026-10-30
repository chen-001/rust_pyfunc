"""One-day benchmark; set RAYON_NUM_THREADS=50 before Python starts.

YUPEI_BENCH_PACKAGE inserts an isolated package directory before import.
The call timer includes Python return conversion; array conversion/save are separate.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import sys
import time

import numpy as np

if os.environ.get("YUPEI_BENCH_PACKAGE"):
    sys.path.insert(0, os.environ["YUPEI_BENCH_PACKAGE"])
import rust_pyfunc as rp

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
date = int(sys.argv[2]) if len(sys.argv) > 2 else 20260717
ks = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
meta = {"date": date, "package": rp.__file__, "threads": os.environ.get("RAYON_NUM_THREADS"),
        "opt": os.environ.get("YUPEI_MULTI_OPT", "0"), "ks": ks,
        "ks_origin": "explicit" if ks is not None else "library_default",
        "load_before": os.getloadavg()}
so = next(Path(rp.__file__).parent.glob("rust_pyfunc*.so"))
meta["module_sha256"] = hashlib.sha256(so.read_bytes()).hexdigest()
meta["switches"] = {k: os.environ.get(k) for k in (
    "YUPEI_MULTI_CACHE", "YUPEI_MULTI_SVD", "YUPEI_SVD_CHUNK", "YUPEI_SORT100_SIMD",
    "YUPEI_NET_PROFILE", "YUPEI_STAGE", "YUPEI_VERBOSE", "YUPEI_TOPK_HIGH_FIRST",
    "YUPEI_SHARE_RHO_REDUCE", "YUPEI_NET_MIRROR", "YUPEI_MASS_AVX512",
    "YUPEI_MATRIX_TWO_TARGET", "YUPEI_R6_STREAMING")}
print(json.dumps(meta), flush=True)
t0 = time.perf_counter()
codes, names, vals = rp.py_yupei_multi_v5(date, ks=ks)
meta["call_seconds"] = time.perf_counter() - t0
if ks is None:
    meta["ks"] = sorted({int(m.group(1)) for name in names
                         if (m := re.search(r"(?:x0topk|gtopk)(\d+)_", name))})
t1 = time.perf_counter()
a = np.asarray(vals, dtype=np.float32).reshape(len(codes), len(names))
meta.update(array_seconds=time.perf_counter() - t1, stocks=len(codes), factors=len(names),
            finite=int(np.isfinite(a).sum()), nan=int(np.isnan(a).sum()), inf=int(np.isinf(a).sum()),
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            sha256=hashlib.sha256(a.tobytes()).hexdigest(), load_after=os.getloadavg())
np.save(out / "values.npy", a)
(out / "labels.json").write_text(json.dumps({"codes": codes, "names": names}))
(out / "timing.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2), flush=True)
