"""Full-market GPT scheduling A/B, always using the sandbox and 50 threads.

Run separately, on an idle host, after rebuilding the sandbox:
  python -I tests/gpt_family_bench.py all --batch all
  python -I tests/gpt_family_bench.py batch1 --batch 1 --baseline results/speed/gpt_all_20241231.json
The timed boundary is py_yupei_gpt(date), including input reads; saving and
comparison are timed separately. All 491 GPT columns are retained, including
leadlag, unlike the union's existing 455-column GPT selection.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--date", type=int, default=20241231)
    parser.add_argument("--batch", default="all", help="all (unset) or a positive family count")
    parser.add_argument("--baseline", type=Path, help="JSON metadata from the reference run")
    args = parser.parse_args()
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.tag):
        parser.error("tag must contain only letters, digits, underscores, or hyphens")
    if args.batch == "all":
        os.environ.pop("YUPEI_GPT_FAMILY_BATCH", None)
        batch = None
    else:
        try:
            batch = int(args.batch)
        except ValueError:
            parser.error("batch must be all or a positive integer")
        if batch <= 0:
            parser.error("batch must be all or a positive integer")
        os.environ["YUPEI_GPT_FAMILY_BATCH"] = str(batch)
    os.environ["RAYON_NUM_THREADS"] = "50"
    os.environ["YUPEI_STAGE"] = "1"
    os.environ.pop("YUPEI_GPT_SORT_BASELINE", None)
    os.environ.pop("YUPEI_GPT_SNAP_DIR", None)
    root = Path(__file__).resolve().parents[1]
    outdir = root / "results" / "speed"
    prefix = outdir / f"gpt_{args.tag}_{args.date}"
    json_path = prefix.with_suffix(".json")
    array_path = prefix.with_suffix(".npy")
    if json_path.exists() or array_path.exists():
        raise SystemExit(f"Choose a new tag; output already exists: {prefix}")
    load_start = os.getloadavg()[0]
    if load_start > 50:
        raise SystemExit(f"Benchmark not started: load1={load_start:.3f} exceeds 50")
    sys.path.insert(0, str(root / "package"))
    import numpy as np
    import rust_pyfunc as rp
    extension = importlib.import_module("rust_pyfunc.rust_pyfunc")
    if not Path(extension.__file__).resolve().is_relative_to(root / "package"):
        raise RuntimeError(f"Wrong extension imported: {extension.__file__}")
    names = list(rp.py_yupei_gpt_names())
    t0, cpu0 = time.perf_counter(), time.process_time()
    codes, values = rp.py_yupei_gpt(args.date)
    wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
    result = np.asarray(values, dtype=np.float32).reshape(len(codes), len(names))
    if len(names) != 491:
        raise RuntimeError(f"Expected all 491 GPT columns, got {len(names)}")
    save_start = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(array_path, result)
    metadata = {
        "tag": args.tag, "date": args.date, "family_batch": batch,
        "rayon_threads": 50, "function": "py_yupei_gpt", "wall_s": wall,
        "process_cpu_s": cpu, "load1_start": load_start, "load1_end": os.getloadavg()[0],
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "shape": list(result.shape), "dtype": str(result.dtype),
        "codes": list(codes), "names": names, "array_path": str(array_path),
        "extension_path": extension.__file__,
        "extension_mtime_ns": Path(extension.__file__).stat().st_mtime_ns,
        "save_array_s": time.perf_counter() - save_start,
    }
    if args.baseline:
        compare_start = time.perf_counter()
        reference = json.loads(args.baseline.read_text())
        old = np.load(reference["array_path"], mmap_mode="r")
        if reference["codes"] != metadata["codes"] or reference["names"] != names or old.shape != result.shape:
            raise RuntimeError("Reference codes, names, or shape differ")
        same_nan = bool(np.array_equal(np.isnan(old), np.isnan(result)))
        defined = ~np.isnan(old) & ~np.isnan(result)
        same_bits = bool(np.array_equal(old.view(np.uint32)[defined], result.view(np.uint32)[defined]))
        finite = np.isfinite(old) & np.isfinite(result)
        before, after = old[finite].astype(np.float64), result[finite].astype(np.float64)
        absolute = np.abs(after - before)
        metadata["comparison"] = {
            "reference_json": str(args.baseline.resolve()), "nan_positions_equal": same_nan,
            "defined_values_bits_equal": same_bits,
            "array_equal_equal_nan": bool(np.array_equal(old, result, equal_nan=True)),
            "max_abs_diff": float(absolute.max(initial=0.)),
            "max_rel_diff": float((absolute / np.maximum(np.abs(before), 1e-12)).max(initial=0.)),
            "elapsed_s": time.perf_counter() - compare_start,
        }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("codes", "names")}, ensure_ascii=False), flush=True)
    if args.baseline and not (same_nan and same_bits):
        raise SystemExit("Full-market output changed; see saved comparison")


if __name__ == "__main__":
    main()
