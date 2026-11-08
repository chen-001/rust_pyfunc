"""Serial, explicit GPT screening of frozen candidate 3; no automatic tuning.

Run only after the coordinator releases the performance window:
  python -I tests/run_gpt_screening.py --run --prefix ab_v3_screen_20261108
The three settings are all/off, batch1/off and all/on. One run each is screening
evidence, never a stable speed claim. The same lock as the acceptance suite
prevents overlap. No compilation, deployment, source edits, or Git operations.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/speed"
EXPECTED_SHA = "a4b504e261dd5da5f98607259fb046ace908e7971e879f53f1071c1e7d23a759"
SETTINGS = (("all_off", "all", "off"), ("batch1_off", "1", "off"), ("all_on", "all", "on"))


def idle_window(max_wait):
    deadline = time.monotonic() + max_wait
    quiet_since = None
    while True:
        now = time.monotonic()
        load = os.getloadavg()[0]
        quiet_since = (quiet_since if quiet_since is not None else now) if load <= 40 else None
        if quiet_since is not None and now - quiet_since >= 30:
            return
        if now >= deadline:
            raise RuntimeError(f"No idle window within {max_wait}s; latest load1={load:.2f}")
        time.sleep(min(5, max(0, deadline - now)))


def validate_result(metadata, log, setting, baseline):
    batch, share = setting
    if metadata.get("shape") != [5107, 491] or metadata.get("rayon_threads") != 50:
        raise RuntimeError("Wrong whole-market shape or requested thread count")
    counts = re.findall(r"(?:thread_count|rayon_threads|threads)=(\d+)", log)
    if not counts or any(int(value) != 50 for value in counts):
        raise RuntimeError("Missing or wrong actual logged Rayon thread count")
    if metadata.get("extension_sha256") != EXPECTED_SHA:
        raise RuntimeError("Wrong frozen candidate module")
    if metadata.get("family_batch") != (None if batch == "all" else int(batch)) or metadata.get("share_rho") != share:
        raise RuntimeError("Wrong experiment settings")
    duration = metadata.get("wall_s")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Missing or invalid elapsed time")
    if (metadata.get("load1_start", float("inf")) > 50
            or metadata.get("sampled_load1_max", float("inf")) > 100
            or not metadata.get("performance_valid_by_load")):
        raise RuntimeError("Overloaded run retained as evidence; no performance conclusion")
    if baseline:
        comparison = metadata.get("comparison", {})
        if (not all(comparison.get(name) for name in
                    ("codes_exact", "names_exact", "shape_exact", "nan_positions_equal",
                     "defined_values_bits_equal", "array_equal_equal_nan"))
                or comparison.get("max_abs_diff") != 0 or comparison.get("max_rel_diff") != 0):
            raise RuntimeError("Missing or failed complete-output numerical comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--max-idle-wait", type=int, default=600,
                        help="Maximum seconds to wait for load1<=40 continuously for 30s before each run")
    args = parser.parse_args()
    if not args.run:
        parser.error("Nothing started; --run is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        parser.error("prefix must contain only letters, digits, underscores or hyphens")
    if args.max_idle_wait < 30:
        parser.error("max-idle-wait must be at least 30")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".acceptance_queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Performance window is owned by another run; no output changed")
        tags = [f"{args.prefix}_{label}" for label, _, _ in SETTINGS]
        state_path = OUT / f"gpt_{args.prefix}.driver.json"
        for tag in tags:
            if any(OUT.glob(f"gpt_{tag}_20241231.*")):
                raise SystemExit(f"Existing artifacts for {tag}; choose a new prefix")
        # Exclusive creation reserves this complete three-run experiment.
        with state_path.open("x") as handle:
            json.dump({"status": "reserved", "pid": os.getpid()}, handle)
        state = {"status": "waiting", "pid": os.getpid(), "scope": "one-run screening only",
                 "module_sha256": EXPECTED_SHA, "records": []}

        def save():
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(state_path)

        environment = {key: value for key, value in os.environ.items() if not key.startswith("YUPEI_")}
        environment.update(RAYON_NUM_THREADS="50", OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        baseline = OUT / f"gpt_{tags[0]}_20241231.json"
        try:
            for index, ((label, batch, share), tag) in enumerate(zip(SETTINGS, tags)):
                state.update(status="waiting_for_idle", next_setting=label)
                save()
                idle_window(args.max_idle_wait)
                command = [sys.executable, "-I", str(ROOT / "tests/gpt_family_bench.py"), tag,
                           "--date", "20241231", "--batch", batch, "--share-rho", share,
                           "--expected-module-sha256", EXPECTED_SHA]
                if index:
                    command.extend(["--baseline", str(baseline)])
                log_path = OUT / f"gpt_{tag}_20241231.log"
                state.update(status="running", next_setting=label)
                save()
                print(json.dumps({"event": "start", "setting": label, "command": command}), flush=True)
                with log_path.open("x") as handle:
                    result = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                row = {"setting": label, "tag": tag, "command": command,
                       "exit_code": result.returncode, "log": str(log_path)}
                state["records"].append(row)
                save()
                if result.returncode:
                    raise RuntimeError(f"{label} exited {result.returncode}; see {log_path}")
                result_path = OUT / f"gpt_{tag}_20241231.json"
                metadata = json.loads(result_path.read_text())
                log_text = log_path.read_text()
                validate_result(metadata, log_text, (batch, share), index > 0)
                stages = [json.loads(line.split("]", 1)[1].strip())
                          for line in log_text.splitlines() if line.startswith("[gpt-stage]")]
                row.update(json=str(result_path), wall_s=metadata["wall_s"],
                           process_cpu_s=metadata["process_cpu_s"], max_rss_kib=metadata["max_rss_kib"],
                           max_load1=metadata["sampled_load1_max"], gpt_stages=stages,
                           numerical_check="reference" if not index else "exact")
                save()
                print(json.dumps({"event": "complete", **row}), flush=True)
            state.update(status="screening_complete_pending_review",
                         note="One run per setting. No source selection or stable speed claim was made.")
            save()
        except BaseException as error:
            state.update(status="needs_review", reason=f"{type(error).__name__}: {error}")
            save()
            raise


if __name__ == "__main__":
    main()
