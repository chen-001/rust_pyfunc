"""Test the exact helper extracted from reduce2.rs; optionally microbenchmark it."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--bench", action="store_true")
args = parser.parse_args()
source_path = ROOT / "src/yupei_pair/reduce2.rs"
source = source_path.read_text()
helper = source.split("// BEGIN PAIR_BITSET_ROW_OVERLAP\n", 1)[1].split("// END PAIR_BITSET_ROW_OVERLAP", 1)[0]
rayon_lib, = (ROOT / "target/release-fast/deps").glob("librayon-*.rlib")
records = []

def run(command, env=None):
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    records.append({"command": list(map(str, command)), "exit_code": result.returncode,
                    "wall_seconds": time.perf_counter() - started, "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout

with tempfile.TemporaryDirectory(prefix="pair_bitset_") as tmp:
    temp = Path(tmp)
    implementation = temp / "implementation.rs"
    implementation.write_text(helper)
    binary = temp / "test"
    env = dict(os.environ, PAIR_BITSET_IMPLEMENTATION=str(implementation))
    run(["rustc", "--edition=2021", "--test", "tests/pair_bitset_counter.rs", "-O", "-C", "target-cpu=native",
         "--extern", f"rayon={rayon_lib}", "-L", f"dependency={rayon_lib.parent}", "-o", str(binary)], env=env)
    output = run([str(binary)])
    passed, failed = map(int, re.search(r"(\d+) passed; (\d+) failed", output).groups())
    report = {"created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "measurement": "extracted_production_bitset_helper", "tests_passed": passed, "tests_failed": failed,
              "helper_sha256": hashlib.sha256(helper.encode()).hexdigest(), "commands": records.copy()}
    destination = ROOT / "results/speed/pair_bitset_tests.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "tests_passed": passed, "tests_failed": failed}))
    if args.bench:
        load = os.getloadavg()
        if load[0] > 50:
            destination = ROOT / "results/speed/pair_bitset_microbench.json"
            deferred = {"created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                        "measurement": "synthetic_bitset_overlap_one_graph", "status": "deferred_high_load",
                        "load_before": load, "max_load1": 50, "benchmark_executed": False}
            destination.write_text(json.dumps(deferred, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"output": str(destination), **deferred}), flush=True)
            raise SystemExit(2)
        output = run([str(binary), "--ignored", "--nocapture"])
        data = json.loads(re.search(r"PAIR_BITSET_BENCH=(\{.*\})", output).group(1))
        old = [r["old_seconds"] for r in data["records"]]
        new = [r["new_seconds"] for r in data["records"]]
        data.update(created_at=datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), load_before=load,
                    helper_sha256=report["helper_sha256"], command=records[-1],
                    old_min=min(old), old_median=statistics.median(old), new_min=min(new), new_median=statistics.median(new),
                    median_speedup=statistics.median(old) / statistics.median(new))
        destination = ROOT / "results/speed/pair_bitset_microbench.json"
        destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"output": str(destination), "old_median": data["old_median"], "new_median": data["new_median"],
                          "median_speedup": data["median_speedup"]}))
