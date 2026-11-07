"""Run full-day checks serially so separate runs never compete for CPU/I/O."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/speed"
PYTHON = "/home/chenzongwei/.conda/envs/chenzongwei311/bin/python"
OUT.mkdir(parents=True, exist_ok=True)

for task in sys.argv[1:]:
    variant, date, repeat = task.split(":")
    prefix = f"{variant}_{date}_r{repeat}"
    if (OUT / f"{prefix}.json").exists():
        raise RuntimeError(f"Refusing to overwrite completed run {prefix}")
    validation_only = os.environ.get("YUPEI_VALIDATE_ONLY") == "1"
    while os.getloadavg()[0] > 50 and not validation_only:
        print(json.dumps({"event": "waiting_load", "task": task,
                          "load": os.getloadavg()}), flush=True)
        time.sleep(10)
    command = ["/usr/bin/time", "-v", PYTHON, "-I",
               str(ROOT / "tests/check_speed.py"), "run", variant, date, repeat]
    if validation_only:
        command.append("--validation-only")
        exclusion_path = OUT / "timing_exclusions.json"
        exclusions = json.loads(exclusion_path.read_text()) if exclusion_path.exists() else {}
        exclusions[prefix] = {"exclude_from_timing_summary": True,
                              "reason": "Validation only under external machine load; not a speed benchmark."}
        exclusion_path.write_text(json.dumps(exclusions, ensure_ascii=False, indent=2) + "\n")
    environment = dict(os.environ, RAYON_NUM_THREADS="50", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(json.dumps({"event": "start", "task": task}), flush=True)
    start = time.perf_counter()
    load_samples = [[0.0, *os.getloadavg()]]
    with (OUT / f"{prefix}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        while True:
            try:
                exit_code = process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                load_samples.append([time.perf_counter() - start, *os.getloadavg()])
    load_samples.append([time.perf_counter() - start, *os.getloadavg()])
    status = {"task": task, "command": command, "exit_code": exit_code,
              "process_wall_seconds": time.perf_counter() - start,
              "load_samples": load_samples,
              "max_load1": max(sample[1] for sample in load_samples)}
    # Our calculation uses 50 threads. A load far beyond that indicates other
    # work arrived after the initial guard; retain values but reject its timing.
    if status["max_load1"] > 100 and not validation_only:
        exclusion_path = OUT / "timing_exclusions.json"
        exclusions = json.loads(exclusion_path.read_text()) if exclusion_path.exists() else {}
        exclusions[prefix] = {"exclude_from_timing_summary": True,
                              "reason": "Machine load1 exceeded 100 during a fixed-50-thread run.",
                              "max_load1": status["max_load1"]}
        exclusion_path.write_text(json.dumps(exclusions, ensure_ascii=False, indent=2) + "\n")
    (OUT / f"{prefix}.status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({key: value for key, value in status.items() if key != "load_samples"}), flush=True)
    if exit_code:
        sys.exit(exit_code)
    reference = OUT / f"base_{date}_r1.json"
    if reference.exists() and prefix != f"base_{date}_r1":
        command = [PYTHON, "-I", str(ROOT / "tests/check_speed.py"), "compare",
                   f"base_{date}_r1", prefix]
        result = subprocess.run(command)
        if result.returncode:
            sys.exit(result.returncode)
