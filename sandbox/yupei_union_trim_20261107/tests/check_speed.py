"""Full-universe timing and exact comparison against the frozen trimmed module.

Run in a fresh process with python -I; each variant keeps its own module binary.
The original output is f32; this harness does not lower its precision.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "speed"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run(args):
    os.environ["RAYON_NUM_THREADS"] = "50"
    os.environ["YUPEI_STAGE"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    saved_package = ROOT / "frozen/variants" / args.variant / "package"
    package = ROOT / "frozen/package" if args.variant == "base" else (
        saved_package if saved_package.is_dir() else ROOT / "package")
    sys.path.insert(0, str(package))
    import rust_pyfunc as rp
    assert Path(rp.__file__).resolve().is_relative_to(package.resolve())
    before_load = os.getloadavg()
    if before_load[0] > 50 and not args.validation_only:
        raise RuntimeError(f"load1 exceeds timing limit: {before_load}")
    expected = rp.py_yupei_union_all_names()
    assert len(expected) == 18758, len(expected)
    module = package / "rust_pyfunc/rust_pyfunc.abi3.so"
    module_hash = hashlib.sha256(module.read_bytes()).hexdigest()
    print(json.dumps({"event": "start", "date": args.date,
                      "variant": args.variant, "repeat": args.repeat,
                      "load": before_load, "module_sha256": module_hash}), flush=True)
    cpu_start = time.process_time()
    start = time.perf_counter()
    codes, names, values = rp.py_yupei_union_all(args.date)
    call_seconds = time.perf_counter() - start
    cpu_seconds = time.process_time() - cpu_start
    assert names == expected, "returned names differ from declared names"
    start = time.perf_counter()
    matrix = np.asarray(values, dtype=np.float32).reshape(len(codes), len(names))
    array_seconds = time.perf_counter() - start
    del values
    prefix = OUT / f"{args.variant}_{args.date}_r{args.repeat}"
    start = time.perf_counter()
    np.save(str(prefix) + ".npy", matrix)
    save_json(Path(str(prefix) + ".labels.json"), {"codes": codes, "names": names})
    write_seconds = time.perf_counter() - start
    report = {"date": args.date, "variant": args.variant, "repeat": args.repeat,
              "shape": list(matrix.shape), "threads": 50,
              "call_seconds": call_seconds, "cpu_seconds": cpu_seconds,
              "numpy_seconds": array_seconds, "write_seconds": write_seconds,
              "load_before": before_load, "load_after": os.getloadavg(),
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "module_sha256": module_hash, "validation_only": args.validation_only,
              "experiment_settings": {key: value for key, value in sorted(os.environ.items())
                                      if key.startswith("YUPEI_") and key not in
                                      {"YUPEI_STAGE", "YUPEI_VALIDATE_ONLY"}},
              "nan_count": int(np.isnan(matrix).sum()),
              "inf_count": int(np.isinf(matrix).sum())}
    save_json(Path(str(prefix) + ".json"), report)
    print(json.dumps({"event": "complete", **report}), flush=True)


def compare(args):
    left = OUT / args.left
    right = OUT / args.right
    a = np.load(str(left) + ".npy", mmap_mode="r")
    b = np.load(str(right) + ".npy", mmap_mode="r")
    al = json.loads(Path(str(left) + ".labels.json").read_text())
    bl = json.loads(Path(str(right) + ".labels.json").read_text())
    assert al["codes"] == bl["codes"], "stock codes/order changed"
    assert al["names"] == bl["names"], "factor names/order changed"
    assert a.shape == b.shape and a.shape[1] == 18758, (a.shape, b.shape)
    nan_same = True
    value_same = True
    bits_same = True
    max_abs = max_rel = 0.0
    mismatches = 0
    for start in range(0, a.shape[0], 64):
        x, y = a[start:start + 64], b[start:start + 64]
        nan_same &= bool(np.array_equal(np.isnan(x), np.isnan(y)))
        equal = (x == y) | (np.isnan(x) & np.isnan(y))
        value_same &= bool(equal.all())
        bits_same &= bool(np.array_equal(x.view(np.uint32), y.view(np.uint32)))
        mismatches += int((~equal).sum())
        finite = np.isfinite(x) & np.isfinite(y)
        xd, yd = x[finite].astype(np.float64), y[finite].astype(np.float64)
        if xd.size:
            diff = np.abs(xd - yd)
            max_abs = max(max_abs, float(diff.max()))
            max_rel = max(max_rel, float((diff / np.maximum(np.abs(xd), 1e-12)).max()))
    report = {"left": args.left, "right": args.right, "shape": list(a.shape),
              "names_exact": True, "codes_exact": True, "nan_exact": nan_same,
              "array_equal_nan": value_same, "bits_equal": bits_same,
              "mismatched_cells": mismatches, "max_abs_diff": max_abs,
              "max_rel_diff": max_rel}
    save_json(OUT / f"compare_{args.left}__{args.right}.json", report)
    print(json.dumps(report), flush=True)
    assert nan_same and value_same and bits_same, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("run")
    bench.add_argument("variant")
    bench.add_argument("date", type=int)
    bench.add_argument("repeat", type=int)
    bench.add_argument("--validation-only", action="store_true",
                       help="check all values under load; exclude this run from speed conclusions")
    check = sub.add_parser("compare")
    check.add_argument("left")
    check.add_argument("right")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (run if args.command == "run" else compare)(args)
