import os, sys, json, time, hashlib
from pathlib import Path
os.environ.setdefault("RAYON_NUM_THREADS", "50")
import numpy as np
import rust_pyfunc as rp

outdir = Path(sys.argv[1]); date = int(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
t0 = time.time()
codes, names, vals = rp.py_yupei_pair(date)
dt = time.time() - t0
a = np.array(vals, dtype=np.float32).reshape(len(codes), len(names))
np.save(outdir / "values.npy", a)
meta = {"module": rp.__file__, "date": date, "threads": int(os.environ["RAYON_NUM_THREADS"]),
        "seconds": dt, "stocks": len(codes), "columns": len(names),
        "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
        "codes": codes, "names": names}
(outdir / "result.json").write_text(json.dumps(meta))
print(json.dumps({k: v for k, v in meta.items() if k not in ("codes", "names")}), flush=True)
