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
    while os.getloadavg()[0] > 50:
        print(json.dumps({"event": "waiting_load", "task": task,
                          "load": os.getloadavg()}), flush=True)
        time.sleep(10)
    command = ["/usr/bin/time", "-v", PYTHON, "-I",
               str(ROOT / "tests/check_speed.py"), "run", variant, date, repeat]
    environment = dict(os.environ, RAYON_NUM_THREADS="50", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(json.dumps({"event": "start", "task": task}), flush=True)
    start = time.perf_counter()
    with (OUT / f"{prefix}.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    status = {"task": task, "command": command, "exit_code": result.returncode,
              "process_wall_seconds": time.perf_counter() - start}
    (OUT / f"{prefix}.status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    if result.returncode:
        sys.exit(result.returncode)
    reference = OUT / f"base_{date}_r1.json"
    if reference.exists() and prefix != f"base_{date}_r1":
        command = [PYTHON, "-I", str(ROOT / "tests/check_speed.py"), "compare",
                   f"base_{date}_r1", prefix]
        result = subprocess.run(command)
        if result.returncode:
            sys.exit(result.returncode)
