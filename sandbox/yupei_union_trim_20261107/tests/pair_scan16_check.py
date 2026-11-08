"""Compile and run only the standalone scan16 correctness comparison.

No performance claim is derived from test duration. Run only when the parent
coordinator has released its exclusive timing window. All child processes are
pinned to a single available CPU, and no main-library build is performed.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "frozen/src/yupei_pair/simd.rs"
ACTIVE = ROOT / "src/yupei_pair/simd.rs"
EXPECTED = "84284220c4b8adbfdfc5fe0edbac6c5e87dc13c33d2eaf5e4fda49f4098dbabc"
for source in (REFERENCE, ACTIVE):
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError(f"The exact reference changed: {source}")
cpu = max(os.sched_getaffinity(0))
destination = ROOT / "results/speed/pair_scan16_correctness.json"
report = {
    "created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "measurement": "standalone_scan16_all_statistics_exactness",
    "performance_benchmark_run": False,
    "kernel_threads": 1,
    "cpu": cpu,
    "reference_sha256": EXPECTED,
    "source_sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (REFERENCE, ROOT / "frozen/src/yupei_pair/bitonic_sort.rs",
                  ROOT / "tests/pair_scan16_candidate.rs", ROOT / "tests/pair_scan16_check.rs")
    },
    "records": [],
}
with tempfile.TemporaryDirectory(prefix="pair_scan16_check_") as tmp:
    binary = str(Path(tmp) / "check")
    commands = [
        ["rustc", "--edition=2021", "tests/pair_scan16_check.rs", "-O", "-C",
         "target-cpu=native", "-o", binary],
        [binary],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
        report["records"].append({"command": command, "exit_code": result.returncode,
                                  "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            report["status"] = "failed"
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            raise SystemExit(result.stderr or result.stdout)
    report["result"] = json.loads(report["records"][-1]["stdout"])
    report["status"] = "passed"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), **report["result"]}, ensure_ascii=False))
