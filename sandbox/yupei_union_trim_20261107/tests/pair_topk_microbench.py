"""Opt-in microbenchmark; run only in the coordinator's exclusive idle window.

python tests/pair_topk_microbench.py --run-benchmark
The selector is single-threaded inside each factor worker, so pin this kernel
benchmark to one CPU. This is not a 50-thread whole-market acceptance test.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "frozen/candidate3_src"
MANIFEST = ROOT / "results/speed/candidate3_sources.json"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-benchmark", action="store_true", help="Explicitly enter the reserved benchmark window")
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--iterations", type=int, default=1024)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--cpu", type=int)
args = parser.parse_args()
if not args.run_benchmark:
    parser.error("No benchmark started. Pass --run-benchmark only after obtaining the reserved idle window.")
if min(args.samples, args.iterations, args.repeats) <= 0:
    parser.error("samples, iterations, and repeats must all be positive")
expected = {row["path"]: row["sha256"] for row in json.loads(MANIFEST.read_text())}
candidate_sources = [FROZEN_ROOT / f"src/yupei_pair/{name}.rs"
                     for name in ("simd", "bitonic_sort", "net")]
for source in candidate_sources:
    key = str(source.relative_to(FROZEN_ROOT))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected.get(key):
        raise RuntimeError(f"Candidate 3 SHA256 mismatch: {source}: {actual} != {expected.get(key)}")
allowed_cpus = os.sched_getaffinity(0)
cpu = args.cpu if args.cpu is not None else max(allowed_cpus)
if cpu not in allowed_cpus:
    parser.error(f"CPU {cpu} is not available in this process affinity")
report = {
    "created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "measurement": "alternating_top100_threshold_kernel", "n": 5107, "take": 100,
    "cpu": cpu, "kernel_threads": 1, "input_samples": args.samples,
    "iterations_per_process": args.iterations, "repeats": args.repeats,
    "diagnostics_excluded_from_kernel_timing": True,
    "candidate_source_manifest": str(MANIFEST.relative_to(ROOT)),
    "candidate_source_manifest_verified": True,
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in candidate_sources + [ROOT / "tests/pair_topk_microbench.rs"]},
    "records": [],
}
destination = ROOT / "results/speed/pair_topk_candidate3_microbench.json"

def save():
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

def require_idle():
    load = os.getloadavg()
    if load[0] > 50:
        report.update(status="deferred_high_load", last_load=load)
        save()
        raise SystemExit(f"No further benchmark started: load1={load[0]:.2f} exceeds 50")
    return load

require_idle()
with tempfile.TemporaryDirectory(prefix="pair_topk_bench_") as tmp:
    binary = str(Path(tmp) / "benchmark")
    command = ["rustc", "--edition=2021", "tests/pair_topk_microbench.rs", "-O", "-C", "target-cpu=native", "-o", binary]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    report["build"] = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        report["status"] = "build_failed"
        save()
        raise SystemExit(result.stderr)
    kinds = ["uniform_positive", "uniform_signed", "repeated_32", "narrow_near_ties", "sample_heavy_fallback"]
    for repeat in range(1, args.repeats + 1):
        # Reverse order on even repeats to avoid systematically favoring one policy.
        modes = ("0", "1") if repeat % 2 else ("1", "0")
        for kind in kinds:
            for wide in modes:
                load = require_idle()
                env = dict(os.environ, YUPEI_TOPK_HIGH_FIRST="1", YUPEI_TOPK100_WIDE_FIRST=wide,
                           YUPEI_TOPK_DIAGNOSTICS="0")
                command = [binary, kind, str(args.samples), str(args.iterations)]
                started = time.perf_counter()
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                        preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
                process_wall = time.perf_counter() - started
                if result.returncode:
                    report.update(status="run_failed", failed_command=command, failed_exit_code=result.returncode,
                                  failed_stdout=result.stdout, failed_stderr=result.stderr)
                    save()
                    raise SystemExit(result.stderr or result.stdout)
                row = json.loads(result.stdout)
                diag = row["diagnostics"]
                calls = max(diag["high_first_calls"], 1)
                row.update(repeat=repeat, wide_first=wide == "1", load_before=load, exit_code=result.returncode,
                           process_wall_seconds=process_wall,
                           first_shortfall_rate=diag["first_shortfalls"] / calls,
                           mean_first_candidates=diag["first_candidates"] / calls,
                           mean_retry_candidates=(diag["retry_candidates"] / diag["retry_scans"]
                                                  if diag["retry_scans"] else None),
                           full_key_fallback_rate=diag["full_key_fallbacks"] / calls)
                report["records"].append(row)
    report["summaries"] = []
    for kind in kinds:
        rows = [r for r in report["records"] if r["kind"] == kind]
        before = [r["seconds"] for r in rows if not r["wide_first"]]
        after = [r["seconds"] for r in rows if r["wide_first"]]
        report["summaries"].append({"kind": kind, "old_min": min(before), "old_median": statistics.median(before),
                                    "wide_min": min(after), "wide_median": statistics.median(after),
                                    "median_speedup": statistics.median(before) / statistics.median(after),
                                    "all_complete_keys_exactly_equal": all(r["complete_keys_exactly_equal"] for r in rows)})
    report["status"] = "completed"
    save()
    print(json.dumps({"output": str(destination), "summaries": report["summaries"]}, ensure_ascii=False))
