"""Full-market benchmark; each CLI call uses a fresh 50-thread process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BOX = ROOT / "sandbox/relaynet_perf_20261102"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["baseline", "fast", "candidate"])
    parser.add_argument("date", type=int)
    parser.add_argument("--label", default="")
    parser.add_argument("--python-list", action="store_true",
                        help="Time the production py_relaynet Python-list return interface")
    args = parser.parse_args()
    if args.python_list and args.variant == "baseline":
        parser.error("--python-list requires fast or candidate")
    os.environ["RAYON_NUM_THREADS"] = "50"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, str(BOX))
    # Record the compiled source snapshot before entering the long calculation;
    # source files can be edited while an earlier binary remains mapped.
    manifest_path = BOX / "relaynet_perf_20261102.build.json"
    build = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    binary_sha = hashlib.sha256((BOX / "relaynet_perf_20261102.so").read_bytes()).hexdigest()
    if build and build["binary_sha256"] != binary_sha:
        raise RuntimeError("Library/build manifest mismatch; wait for build completion")
    runtime_environment = {k: v for k, v in os.environ.items()
                           if k.startswith(("RLG_", "RELAYNET_")) or k == "RAYON_NUM_THREADS"}
    import numpy as np
    import relaynet_perf_20261102 as engine

    dest = ROOT / "reports/relaynet_perf_20261102"
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_{args.date}" + (f"_{args.label}" if args.label else "")
    started = time.perf_counter()
    if args.python_list:
        call = engine.py_relaynet if args.variant == 'candidate' else engine.py_relaynet_experimental
        codes, values = call(args.date)
    else:
        codes, values = engine.run(args.date, args.variant)
    elapsed = time.perf_counter() - started
    names = engine.names()
    values = np.asarray(values, dtype=np.float32).reshape(len(codes), len(names))
    np.save(dest / f"{stem}.npy", values)
    meta = dict(variant=args.variant, date=args.date, threads=50, seconds=elapsed,
                interface="production-list" if args.python_list else "numpy",
                shape=values.shape, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                binary_sha256=binary_sha, build=build, runtime_environment=runtime_environment,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                codes=codes, names=names)
    (dest / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("codes", "names")}), flush=True)


if __name__ == "__main__":
    main()
