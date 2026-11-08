"""Alternating single-core scan16 microbenchmark; launch in a reserved idle window.

This measures only the standalone synthetic-input kernel. It cannot establish
whole-market factor speed, and does not rebuild or install the actual library.
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
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-benchmark", action="store_true")
parser.add_argument("--cpu", type=int)
parser.add_argument("--iterations", type=int, default=50000)
args = parser.parse_args()
if not args.run_benchmark:
    parser.error("Pass --run-benchmark only after the parent grants the timing window")
if args.iterations <= 0:
    parser.error("iterations must be positive")
cpus = os.sched_getaffinity(0)
cpu = args.cpu if args.cpu is not None else max(cpus)
if cpu not in cpus:
    parser.error("cpu must be in the current allowed CPU set")
destination = ROOT / "results/speed/pair_scan16_microbench.json"
if destination.exists():
    raise SystemExit(f"Refusing to overwrite existing evidence: {destination}")
reference = ROOT / "frozen/src/yupei_pair/simd.rs"
expected = "84284220c4b8adbfdfc5fe0edbac6c5e87dc13c33d2eaf5e4fda49f4098dbabc"
if hashlib.sha256(reference.read_bytes()).hexdigest() != expected:
    raise RuntimeError("Frozen reference changed")
report = {
    "created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "measurement": "synthetic_scan16_single_core_kernel",
    "whole_market_performance_claim": False,
    "cpu": cpu,
    "kernel_threads": 1,
    "repeats_per_mode_and_kind": 3,
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (reference, ROOT / "tests/pair_scan16_candidate.rs", ROOT / "tests/pair_scan16_bench.rs")},
    "records": [],
}

def save():
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

def require_idle():
    load = list(os.getloadavg())
    if load[0] > 50:
        report.update(status="deferred_high_load", last_load=load)
        save()
        raise SystemExit(f"No further benchmark started: load1={load[0]:.2f} > 50")
    return load

require_idle()
with tempfile.TemporaryDirectory(prefix="pair_scan16_bench_") as tmp:
    binary = str(Path(tmp) / "bench")
    command = ["rustc", "--edition=2021", "tests/pair_scan16_bench.rs", "-O", "-C", "target-cpu=native", "-o", binary]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
    report["build"] = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        report["status"] = "build_failed"
        save()
        raise SystemExit(result.stderr)
    for repeat in range(1, 4):
        modes = ("reference", "candidate") if repeat % 2 else ("candidate", "reference")
        for kind in ("positive", "signed", "boundary"):
            for mode in modes:
                load = require_idle()
                command = [binary, mode, kind, str(args.iterations)]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                        preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
                after = list(os.getloadavg())
                if result.returncode:
                    report.update(status="run_failed", failed_command=command, failed_exit_code=result.returncode,
                                  failed_stdout=result.stdout, failed_stderr=result.stderr)
                    save()
                    raise SystemExit(result.stderr or result.stdout)
                row = json.loads(result.stdout)
                row.update(repeat=repeat, exit_code=result.returncode, load_before=load, load_after=after,
                           performance_valid_by_observed_load=after[0] <= 50)
                report["records"].append(row)
                if after[0] > 50:
                    report["status"] = "deferred_high_load"
                    save()
                    raise SystemExit("Load exceeded 50 during a kernel process; do not claim its speed")
    report["summaries"] = []
    for kind in ("positive", "signed", "boundary"):
        rows = [r for r in report["records"] if r["kind"] == kind]
        before = [r["seconds"] for r in rows if r["mode"] == "reference"]
        after = [r["seconds"] for r in rows if r["mode"] == "candidate"]
        report["summaries"].append({"kind": kind, "reference_first": before[0], "reference_min": min(before),
                                    "reference_median": statistics.median(before), "candidate_first": after[0],
                                    "candidate_min": min(after), "candidate_median": statistics.median(after),
                                    "median_speedup": statistics.median(before) / statistics.median(after)})
    report["status"] = "completed"
    save()
    print(json.dumps({"output": str(destination), "summaries": report["summaries"]}, ensure_ascii=False))
