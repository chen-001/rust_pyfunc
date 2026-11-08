"""Check exact 1--4-pair mass batching; optionally time the isolated kernel.

Launch only in the coordinator's reserved window. Checks exercise both original
backend settings in separate processes. Timing defaults to AVX512, uses a fixed
CPU, alternates order over three repeats, and accounts separately for copying
the same h/g bytes into each implementation's natural staging buffers.
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
parser.add_argument("--integrated", action="store_true", help="check the actual sandbox mass-batch source")
parser.add_argument("--cpu", type=int)
parser.add_argument("--iterations", type=int, default=20000)
parser.add_argument("--tag", default="pair_mass4")
args = parser.parse_args()
if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in args.tag):
    parser.error("tag must contain only letters, digits, or underscores")
if args.iterations <= 0:
    parser.error("iterations must be positive")
allowed = os.sched_getaffinity(0)
cpu = args.cpu if args.cpu is not None else max(allowed)
if cpu not in allowed:
    parser.error("CPU is outside the current allowed set")
tag = args.tag + ("_integrated" if args.integrated else "")
destination = ROOT / "results/speed" / (tag + ("_microbench.json" if args.run_benchmark else "_correctness.json"))
if destination.exists():
    raise SystemExit(f"Refusing to overwrite existing evidence: {destination}")
reference = ROOT / "frozen/src/yupei_pair/simd.rs"
candidate = ROOT / ("src/yupei_pair_mass_batch.rs" if args.integrated else "tests/pair_mass4_candidate.rs")
expected = "84284220c4b8adbfdfc5fe0edbac6c5e87dc13c33d2eaf5e4fda49f4098dbabc"
if hashlib.sha256(reference.read_bytes()).hexdigest() != expected:
    raise RuntimeError("Frozen mass reference changed")
report = {
    "created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "measurement": "one_to_four_pair_mass_batch",
    "performance_benchmark_run": args.run_benchmark,
    "integrated_source": args.integrated,
    "whole_market_performance_claim": False,
    "cpu": cpu, "kernel_threads": 1, "repeats_per_setting": 3,
    "scope": "Mass update only. Direct and staged inputs have identical values; staged variants copy equal h/g bytes. The extra working-set interaction with score production and TopK remains unmeasured.",
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (reference, candidate, ROOT / "tests/pair_mass4_probe.rs")},
    "correctness": [], "records": [],
}

def save():
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

def require_idle():
    load = list(os.getloadavg())
    if load[0] > 50:
        report.update(status="deferred_high_load", last_load=load)
        save()
        raise SystemExit(f"No benchmark started: load1={load[0]:.2f} > 50")
    return load

def execute(command, env=None):
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                          preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))

if args.run_benchmark:
    require_idle()
with tempfile.TemporaryDirectory(prefix="pair_mass4_") as tmp:
    binary = str(Path(tmp) / "probe")
    command = ["rustc", "--edition=2021", "tests/pair_mass4_probe.rs", "-O", "-C", "target-cpu=native", "-o", binary]
    if args.integrated:
        command.extend(["--cfg", "integrated_mass4"])
    result = execute(command)
    report["build"] = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        report["status"] = "build_failed"
        save()
        raise SystemExit(result.stderr)
    for backend in ("1", "0"):
        result = execute([binary], dict(os.environ, YUPEI_MASS_AVX512=backend))
        row = {"mass_avx512": backend, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        report["correctness"].append(row)
        if result.returncode:
            report["status"] = "correctness_failed"
            save()
            raise SystemExit(result.stderr or result.stdout)
        row["result"] = json.loads(result.stdout)
    if not args.run_benchmark:
        report["status"] = "passed"
        save()
        print(json.dumps({"output": str(destination), "correctness": [r["result"] for r in report["correctness"]]}))
        raise SystemExit(0)
    for repeat in range(1, 4):
        modes = ("reference", "candidate") if repeat % 2 else ("candidate", "reference")
        for access in ("direct", "staged"):
            for kind in ("normal", "mixed_flags"):
                for mode in modes:
                    before = require_idle()
                    result = execute([binary, "--bench", mode, access, kind, str(args.iterations)],
                                     dict(os.environ, YUPEI_MASS_AVX512="1"))
                    after = list(os.getloadavg())
                    if result.returncode:
                        report.update(status="run_failed", failed_exit_code=result.returncode,
                                      failed_stdout=result.stdout, failed_stderr=result.stderr)
                        save()
                        raise SystemExit(result.stderr or result.stdout)
                    row = json.loads(result.stdout)
                    row.update(repeat=repeat, exit_code=0, mass_avx512="1", load_before=before, load_after=after,
                               performance_valid_by_observed_load=after[0] <= 50)
                    report["records"].append(row)
                    if after[0] > 50:
                        report["status"] = "deferred_high_load"
                        save()
                        raise SystemExit("Load rose above 50; this kernel run is not a timing result")
    report["summaries"] = []
    for access in ("direct", "staged"):
        for kind in ("normal", "mixed_flags"):
            rows = [r for r in report["records"] if r["access"] == access and r["kind"] == kind]
            before = [r["seconds"] for r in rows if r["mode"] == "reference"]
            after = [r["seconds"] for r in rows if r["mode"] == "candidate"]
            report["summaries"].append({"access": access, "kind": kind,
                "reference_first": before[0], "reference_min": min(before), "reference_median": statistics.median(before),
                "candidate_first": after[0], "candidate_min": min(after), "candidate_median": statistics.median(after),
                "median_speedup": statistics.median(before) / statistics.median(after)})
    report["status"] = "completed"
    save()
    print(json.dumps({"output": str(destination), "summaries": report["summaries"]}))
