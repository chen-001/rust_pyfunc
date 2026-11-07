"""Full-market GPT scheduling A/B, using frozen candidate 3 and 50 threads.

Run separately on an idle host; the current sandbox may contain candidate 2:
  python -I tests/gpt_family_bench.py all --batch all
  python -I tests/gpt_family_bench.py batch1 --batch 1 --baseline results/speed/gpt_all_20241231.json
The timed boundary is py_yupei_gpt(date), including input reads; saving and
comparison are timed separately. All 491 GPT columns are retained, including
leadlag, unlike the union's existing 455-column GPT selection.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--date", type=int, default=20241231)
    parser.add_argument("--batch", default="all", help="all (unset) or a positive family count")
    parser.add_argument("--share-rho", choices=("off", "on"), default="off")
    parser.add_argument("--package-root", type=Path, default=root / "frozen/variants/v3/package")
    parser.add_argument("--source-manifest", type=Path, default=root / "results/speed/candidate3_sources.json")
    parser.add_argument("--expected-module-sha256", default="a4b504e261dd5da5f98607259fb046ace908e7971e879f53f1071c1e7d23a759")
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
    if args.share_rho == "on":
        os.environ["YUPEI_GPT_SHARE_RHO"] = "1"
    else:
        os.environ.pop("YUPEI_GPT_SHARE_RHO", None)
    os.environ["RAYON_NUM_THREADS"] = "50"
    os.environ["YUPEI_STAGE"] = "1"
    os.environ.pop("YUPEI_GPT_SORT_BASELINE", None)
    os.environ.pop("YUPEI_GPT_SNAP_DIR", None)
    outdir = root / "results" / "speed"
    prefix = outdir / f"gpt_{args.tag}_{args.date}"
    json_path = prefix.with_suffix(".json")
    array_path = prefix.with_suffix(".npy")
    load_path = prefix.with_suffix(".load.jsonl")
    if json_path.exists() or array_path.exists() or load_path.exists():
        raise SystemExit(f"Choose a new tag; output already exists: {prefix}")
    load_start = os.getloadavg()[0]
    if load_start > 50:
        raise SystemExit(f"Benchmark not started: load1={load_start:.3f} exceeds 50")
    package_root = args.package_root.resolve()
    sys.path.insert(0, str(package_root))
    import numpy as np
    import rust_pyfunc as rp
    extension = importlib.import_module("rust_pyfunc.rust_pyfunc")
    if not Path(extension.__file__).resolve().is_relative_to(package_root):
        raise RuntimeError(f"Wrong extension imported: {extension.__file__}")
    extension_path = Path(extension.__file__)
    module_sha = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    if args.expected_module_sha256 and module_sha != args.expected_module_sha256:
        raise RuntimeError(f"Extension changed: {module_sha}")
    # Candidate 3's changed files are archived separately; every other GPT file
    # comes from the original freeze. Never read current candidate 2 sources.
    # Backup mtimes do not establish provenance, so use the saved content hashes.
    manifest_path = args.source_manifest.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = {row["path"]: row["sha256"] for row in json.loads(manifest_bytes)}
    source_names = {str(p.relative_to(root / "frozen")) for p in (root / "frozen/src/yupei_gpt").glob("*.rs")}
    source_names.update(name for name in manifest if name.startswith("src/yupei_gpt/"))
    source_state = {}
    for name in sorted(source_names):
        changed = name in manifest
        source_path = root / ("frozen/candidate3_src" if changed else "frozen") / name
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if changed and source_sha != manifest[name]:
            raise RuntimeError(f"Candidate 3 source hash mismatch: {source_path}")
        source_state[name] = {"sha256": source_sha, "physical_path": str(source_path),
            "layer": "candidate3_changed" if changed else "original_frozen_unchanged",
            "manifest_sha256": manifest.get(name)}
    names = list(rp.py_yupei_gpt_names())
    load_start = os.getloadavg()[0]
    if load_start > 50:
        raise SystemExit(f"Benchmark not started after import: load1={load_start:.3f} exceeds 50")
    outdir.mkdir(parents=True, exist_ok=True)
    # A separate lightweight observer is necessary because the extension can
    # hold Python's GIL. It samples load only and adds no factor worker threads.
    monitor_code = "import json,os,time\nwhile True:\n print(json.dumps({'unix_time':time.time(),'load1':os.getloadavg()[0]}),flush=True)\n time.sleep(10)\n"
    monitor_file = load_path.open("x")
    monitor = subprocess.Popen([sys.executable, "-I", "-c", monitor_code], stdout=monitor_file, stderr=subprocess.DEVNULL)
    t0, cpu0 = time.perf_counter(), time.process_time()
    try:
        codes, values = rp.py_yupei_gpt(args.date)
        wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
        load_end = os.getloadavg()[0]
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        monitor_file.close()
    load_samples = [json.loads(line) for line in load_path.read_text().splitlines() if line.strip()]
    max_load = max([load_start, load_end] + [row["load1"] for row in load_samples])
    result = np.asarray(values, dtype=np.float32).reshape(len(codes), len(names))
    if len(names) != 491:
        raise RuntimeError(f"Expected all 491 GPT columns, got {len(names)}")
    if args.date == 20241231 and len(codes) != 5107:
        raise RuntimeError(f"Expected 5107 stocks on 20241231, got {len(codes)}")
    save_start = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(array_path, result)
    metadata = {
        "tag": args.tag, "date": args.date, "family_batch": batch, "share_rho": args.share_rho,
        "rayon_threads": 50, "function": "py_yupei_gpt", "wall_s": wall,
        "process_cpu_s": cpu, "load1_start": load_start, "load1_end": load_end,
        "sampled_load1_max": max_load, "load_sample_interval_s": 10,
        "load_samples": load_samples, "load_log": str(load_path),
        "performance_valid_by_load": max_load <= 100,
        "performance_invalid_reason": "sampled load1 exceeded 100" if max_load > 100 else None,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "shape": list(result.shape), "dtype": str(result.dtype),
        "codes": list(codes), "names": names, "array_path": str(array_path),
        "extension_path": extension.__file__,
        "extension_mtime_ns": Path(extension.__file__).stat().st_mtime_ns,
        "extension_sha256": module_sha, "gpt_source_state": source_state,
        "package_root": str(package_root), "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
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
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("codes", "names", "load_samples", "gpt_source_state")}, ensure_ascii=False), flush=True)
    if args.baseline and not (same_nan and same_bits):
        raise SystemExit("Full-market output changed; see saved comparison")


if __name__ == "__main__":
    main()
