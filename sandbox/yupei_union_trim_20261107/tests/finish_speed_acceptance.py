"""Finish fixed-50-thread speed measurements after other work frees the host.

Only measures the frozen baseline and the three-date-verified v2 package.
No source edits, deployment, Git writes, or changes to other processes.
Writes a fresh XLSX from measured evidence; completion is not a speed claim.
"""
import hashlib
import json
import fcntl
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
OUT = ROOT / "results/speed"
STATE = OUT / "acceptance_queue.json"
STOP = OUT / "acceptance_queue.stop"
PYTHON = "/home/chenzongwei/.conda/envs/chenzongwei311/bin/python"
EXPECTED = {
    "base": "c3a192b8495ab36e66d56df28733797d57a91455e28f41a9eca32971fcec8066",
    "v2": "b7c4fbafd8f934d4db82d5ff8d812aaf3b2202a5447d2073bcfda4b8849100d9",
}
EXCLUDED_ATTEMPTS = []


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def record(status, **extra):
    value = {"status": status, "pid": os.getpid(), "updated_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
             "load": list(os.getloadavg()), "threads": 50, "date": 20241231,
             "target_valid_repeats_per_variant": 3,
             "expected_modules": EXPECTED, "excluded_attempts": EXCLUDED_ATTEMPTS, **extra}
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(STATE)


def ineligible_reasons(variant, path):
    excluded = read(OUT / "timing_exclusions.json", {})
    value = read(path)
    status = read(path.with_name(path.stem + ".status.json"), {})
    reasons = []
    seconds = value.get("call_seconds")
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        reasons.append("missing_or_invalid_duration")
    if value.get("validation_only"):
        reasons.append("validation_only")
    if value.get("shape") != [5107, 18758] or value.get("threads") != 50:
        reasons.append("shape_or_thread_count")
    if value.get("variant") != variant or value.get("date") != 20241231:
        reasons.append("wrong_variant_or_date")
    if value.get("load_before", [float("inf")])[0] > 50:
        reasons.append("high_initial_load")
    if status.get("exit_code") != 0:
        reasons.append("unsuccessful_exit")
    # The original frozen reference predates sampling. Keep this one explicit
    # legacy exception; its initial/end load and original run log are retained.
    legacy_reference = path.stem == "base_20241231_r1"
    max_load = status.get("max_load1")
    if max_load is None and not legacy_reference:
        reasons.append("missing_load_monitor")
    elif max_load is not None and max_load > 100:
        reasons.append("high_run_load")
    if value.get("module_sha256") != EXPECTED[variant] or value.get("experiment_settings", {}):
        reasons.append("wrong_module_or_settings")
    if excluded.get(path.stem, {}).get("exclude_from_timing_summary"):
        reasons.append("explicit_timing_exclusion")
    log_path = path.with_suffix(".log")
    counts = re.findall(r"(?:thread_count|rayon_threads|threads)=(\d+)", log_path.read_text() if log_path.exists() else "")
    if not counts or any(int(count) != 50 for count in counts):
        reasons.append("missing_or_wrong_logged_threads")
    if not legacy_reference:
        comparison = read(OUT / f"compare_base_20241231_r1__{path.stem}.json", {})
        if (comparison.get("shape") != [5107, 18758]
                or not all(comparison.get(key) for key in
                           ("names_exact", "codes_exact", "nan_exact", "bits_equal", "array_equal_nan"))
                or comparison.get("mismatched_cells") != 0
                or comparison.get("max_abs_diff") != 0
                or comparison.get("max_rel_diff") != 0
                or status.get("comparison_exit_code") != 0):
            reasons.append("missing_or_failed_exact_comparison")
    return reasons


def eligible(variant):
    accepted = []
    for path in OUT.glob(f"{variant}_20241231_r*.json"):
        if path.name.endswith((".status.json", ".labels.json")):
            continue
        if ineligible_reasons(variant, path):
            continue
        accepted.append(path.stem)
    return sorted(accepted)


def await_idle(task):
    quiet_since = None
    while True:
        if STOP.exists():
            raise InterruptedError("Stop marker present; no further calculation started")
        if os.getloadavg()[0] <= 40:
            quiet_since = quiet_since or time.monotonic()
        else:
            quiet_since = None
        quiet = 0 if quiet_since is None else time.monotonic() - quiet_since
        record("waiting_for_stable_low_load", next_task=task,
               required_load1_at_most=40, required_quiet_seconds=120,
               observed_quiet_seconds=quiet,
               valid_runs={v: eligible(v) for v in EXPECTED})
        if quiet >= 120:
            return
        time.sleep(10)


def report():
    subprocess.run([PYTHON, str(PROJECT / "tests/make_speed_report.py")], check=True)


def main():
    # A rerun may resume completed evidence; output arrays are never overwritten.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("YUPEI_")}
    environment.update(RAYON_NUM_THREADS="50", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for variant, digest in EXPECTED.items():
        package = ROOT / ("frozen/package" if variant == "base" else "frozen/variants/v2/package")
        binary = package / "rust_pyfunc/rust_pyfunc.abi3.so"
        if hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen {variant} module changed")
    EXCLUDED_ATTEMPTS.extend(read(STATE, {}).get("excluded_attempts", []))
    if len(EXCLUDED_ATTEMPTS) >= 3:
        record("needs_review", reason="Prior excluded attempts require review before resuming")
        return
    for variant in EXPECTED:
        while len(eligible(variant)) < 3:
            repeat = 1
            while any(OUT.glob(f"{variant}_20241231_r{repeat}.*")):
                repeat += 1
            task = f"{variant}:20241231:{repeat}"
            await_idle(task)
            # Reallocate after waiting; run_suite atomically reserves the prefix.
            while any(OUT.glob(f"{variant}_20241231_r{repeat}.*")):
                repeat += 1
            task = f"{variant}:20241231:{repeat}"
            record("running", task=task)
            result = subprocess.run([PYTHON, "-I", str(ROOT / "tests/run_suite.py"), task], env=environment)
            if result.returncode:
                raise RuntimeError(f"{task} exited {result.returncode}; inspect its log")
            report()
            if f"{variant}_20241231_r{repeat}" not in eligible(variant):
                reasons = ineligible_reasons(variant, OUT / f"{variant}_20241231_r{repeat}.json")
                EXCLUDED_ATTEMPTS.append({"task": task, "reasons": reasons})
                if len(EXCLUDED_ATTEMPTS) >= 3 or any(reason not in {
                        "high_initial_load", "high_run_load", "explicit_timing_exclusion"} for reason in reasons):
                    record("needs_review", reason="Completed evidence needs review; see excluded_attempts",
                           valid_runs={v: eligible(v) for v in EXPECTED})
                    return
    report()
    record("measurements_complete_pending_review", valid_runs={v: eligible(v) for v in EXPECTED},
           note="Measured output, speed targets, and any further optimization still require review; no deployment performed.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / ".acceptance_queue.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another acceptance queue already owns the lock; no state or outputs changed")
    try:
        main()
    except BaseException as error:
        record("stopped" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
               reason=f"{type(error).__name__}: {error}")
        raise
