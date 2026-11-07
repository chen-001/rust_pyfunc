"""Standalone correctness only; no performance timing is claimed by this test."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "frozen/candidate3_src"
MANIFEST = ROOT / "results/speed/candidate3_sources.json"
expected = {row["path"]: row["sha256"] for row in json.loads(MANIFEST.read_text())}
candidate_sources = [FROZEN_ROOT / f"src/yupei_pair/{name}.rs"
                     for name in ("simd", "bitonic_sort", "net")]
for source in candidate_sources:
    key = str(source.relative_to(FROZEN_ROOT))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected.get(key):
        raise RuntimeError(f"Candidate 3 SHA256 mismatch: {source}: {actual} != {expected.get(key)}")
records = []
with tempfile.TemporaryDirectory(prefix="pair_topk_thresholds_") as tmp:
    binary = str(Path(tmp) / "test")
    command = ["rustc", "--edition=2021", "--test", "tests/pair_topk_thresholds.rs", "-O", "-C", "target-cpu=native", "-o", binary]
    built = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    records.append({"command": command, "exit_code": built.returncode, "stdout": built.stdout, "stderr": built.stderr})
    if built.returncode:
        raise RuntimeError(built.stderr)
    for wide, high in [("0", "1"), ("1", "1"), ("1", "0")]:
        env = dict(os.environ, YUPEI_TOPK100_WIDE_FIRST=wide, YUPEI_TOPK_HIGH_FIRST=high)
        result = subprocess.run([binary, "--nocapture"], cwd=ROOT, env=env, capture_output=True, text=True)
        counts = re.search(r"(\d+) passed; (\d+) failed", result.stdout)
        records.append({"wide_first": wide, "high_first": high, "exit_code": result.returncode,
                        "tests_passed": int(counts[1]) if counts else None,
                        "tests_failed": int(counts[2]) if counts else None,
                        "stdout": result.stdout, "stderr": result.stderr})
    report = {"created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "measurement": "standalone_topk_full_key_correctness", "performance_benchmark_run": False,
              "candidate_source_manifest": str(MANIFEST.relative_to(ROOT)),
              "candidate_source_manifest_verified": True,
              "records": records,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in candidate_sources + [ROOT / "tests/pair_topk_thresholds.rs"]}}
    destination = ROOT / "results/speed/pair_topk_candidate3_tests.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "cases": [{k: v for k, v in r.items() if k in (
        "wide_first", "high_first", "exit_code", "tests_passed", "tests_failed")} for r in records[1:]]}), flush=True)
    if any(r["exit_code"] != 0 for r in records):
        raise SystemExit(1)
