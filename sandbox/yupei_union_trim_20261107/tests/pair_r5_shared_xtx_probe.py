"""Prepare/compare exact same-carrier/(a,b) R5 X'X sharing in a reserved window.

Nothing runs without --run. --run-benchmark adds three alternating single-core
measurements: three original fits versus one cache build plus three cached fits.
Each measurement includes median preparation, X'y and the unchanged solver;
residual row output is checked for exactness but is outside the timed scope.
All inputs are synthetic. No production/source changes or whole-day claim.

Future integration, only after this probe and real-date checks pass:
  reduce.rs: SharedRhoReduce/prepare_shared_rho_reduce, r5_residual_4 and caller;
  split.rs, yupei_multi/mod.rs: pass lnt and the same group-scoped cache.
Keep the original fallback, f32 sdir round trip, mask and i/j/a/b order. Do not
apply this to reduce2's 7x7 regression: its response-finite mask is a dependency.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "frozen/src/yupei_pair/reduce.rs"
EXPECTED = "9a9e8fe6ea0cd7c976d67776b4f650108929eb5155e19e94b01bea5eaaa87246"


def function(source, name):
    start = source.index(f"pub(crate) fn {name}(")
    return source[start:source.index("\n}", start) + 2]


def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Frozen extraction anchor changed: {old!r}")
    return source.replace(old, new, 1)


def generated_source():
    source = FROZEN.read_text()
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError("Frozen R5 reference changed")
    current = (ROOT / "src/yupei_pair/reduce.rs").read_text()
    for name in ("r5_residual_4", "nanmedian", "solve5", "row_stats"):
        if function(current, name) != function(source, name):
            raise RuntimeError(f"Current {name} differs from frozen reference; review before running")
    if "pub(crate) const K_TOPK: usize = 20;" not in source:
        raise RuntimeError("Reference row-stat top-K changed")
    original = function(source, "r5_residual_4")
    body = original.split(") -> [R5; 4] {\n", 1)[1]
    prepare, tail = body.split("    let n_ok =", 1)
    prepare = once(prepare, "    let t0 = Instant::now();\n", "")
    impute = prepare[:prepare.index("    let mut xtx =")]
    xtx_loop = "                for b in 0..5 {\n                    xtx[a][b] += d[a] * d[b];\n                }\n"
    fit_head = "pub(crate) fn reference_fit(n: usize, wmats: [&[f32]; 4], rho: &[f32], lnt: &[f64], ok: &[bool]) -> Fit {\n"
    fit_end = "    Fit { ln, xtx, xty, npair, betas }\n}\n"
    reference = "#[inline(never)]\n" + fit_head + prepare + fit_end
    cached = once(prepare, impute, "    let ln = cache.ln.clone();\n")
    cached = once(cached, "    let mut xtx = [[0.0f64; 5]; 5];\n", "    let xtx = cache.xtx;\n")
    cached = once(cached, xtx_loop, "")
    cached_head = "pub(crate) fn cached_fit(n: usize, wmats: [&[f32]; 4], rho: &[f32], ok: &[bool], cache: &SharedXtx) -> Fit {\n"
    cached = "#[inline(never)]\n" + cached_head + cached + "    assert_eq!(npair, cache.npair);\n" + fit_end
    shared = prepare[:prepare.index("    let mut a = xtx;")]
    shared = once(shared, "    let mut xty = [[0.0f64; 5]; 4];\n", "")
    for k in range(4):
        shared = once(shared, f"        let w{k} = &wmats[{k}][i * n..i * n + n];\n", "")
        shared = once(shared, f"            let y{k} = w{k}[j] as f64;\n", "")
        shared = once(shared, f"                xty[{k}][a] += d[a] * y{k};\n", "")
    shared_head = "pub(crate) fn prepare_shared(n: usize, rho: &[f32], lnt: &[f64], ok: &[bool]) -> SharedXtx {\n"
    shared = "#[inline(never)]\n" + shared_head + shared + "    SharedXtx { ln, xtx, npair }\n}\n"
    tail = "    let n_ok =" + tail.split("    if super::verbose()", 1)[0]
    if tail.count(".into_par_iter()") != 1:
        raise RuntimeError("Residual row schedule changed")
    tail = tail.replace(".into_par_iter()", ".into_iter()")
    finish = ("pub(crate) fn finish_r5(n: usize, wmats: [&[f32]; 4], rho: &[f32], ok: &[bool], fit: &Fit) -> [R5; 4] {\n"
              "    let ln = &fit.ln;\n    let betas = fit.betas;\n" + tail + "    out_arr\n}\n")
    struct_start = source.index("pub(crate) struct R5 {")
    r5_struct = source[struct_start:source.index("\n}", struct_start) + 2]
    types = """
pub(crate) struct Fit {
    pub ln: Vec<f64>, pub xtx: [[f64; 5]; 5], pub xty: [[f64; 5]; 4],
    pub npair: usize, pub betas: [[f64; 5]; 4],
}
pub(crate) struct SharedXtx { ln: Vec<f64>, xtx: [[f64; 5]; 5], npair: usize }
const K_TOPK: usize = 20;
"""
    return "\n".join([types, r5_struct, *(function(source, name) for name in
                       ("nanmedian", "solve5", "row_stats")), reference, shared, cached, finish])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if not args.run:
        parser.error("Prepared only: --run is required to compile or execute")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag) or args.iterations <= 0:
        parser.error("Use a unique simple tag and positive iterations")
    allowed = os.sched_getaffinity(0)
    cpu = args.cpu if args.cpu is not None else max(allowed)
    if cpu not in allowed:
        parser.error("CPU is outside allowed affinity")
    out = ROOT / "results/speed"
    destination = out / f"{args.tag}.json"
    with (out / ".acceptance_queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Acceptance window busy; no compilation or calculation started")
        if os.getloadavg()[0] > 50:
            raise SystemExit("load1 > 50; no compilation or calculation started")
        generated = generated_source()
        files = (FROZEN, ROOT / "tests/pair_r5_shared_xtx_probe.rs", Path(__file__))
        report = {
            "created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "status": "reserved", "cpu": cpu, "kernel_threads": 1,
            "performance_benchmark_run": args.run_benchmark, "repeats_per_mode": 3,
            "whole_market_performance_claim": False, "fixture": "synthetic same carrier+(a,b), three different c responses",
            "scope": "Normal-equation preparation and solve only; cached mode includes cache construction. Residual outputs checked, not timed. Not reduce2 7x7.",
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "generated_reference_sha256": hashlib.sha256(generated.encode()).hexdigest(),
            "records": [],
        }
        # Exclusive reservation: an existing evidence file is never overwritten.
        with destination.open("x") as evidence:
            def save():
                evidence.seek(0)
                json.dump(report, evidence, ensure_ascii=False, indent=2)
                evidence.write("\n")
                evidence.truncate()
                evidence.flush()

            def execute(command, environment):
                before = list(os.getloadavg())
                if before[0] > 50:
                    raise RuntimeError("load1 > 50 before subprocess; deferred")
                samples = [before[0]]
                with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True,
                                      preexec_fn=lambda: os.sched_setaffinity(0, {cpu})) as process:
                    while True:
                        try:
                            stdout, stderr = process.communicate(timeout=0.25)
                            break
                        except subprocess.TimeoutExpired:
                            samples.append(os.getloadavg()[0])
                    after = list(os.getloadavg())
                    samples.append(after[0])
                    return {"command": command, "exit_code": process.returncode,
                            "stdout": stdout, "stderr": stderr, "load_before": before,
                            "load_after": after, "sampled_load1_max": max(samples),
                            "performance_valid_by_observed_load": max(samples) <= 50}

            def require_success(row):
                if row["exit_code"] != 0:
                    raise RuntimeError(f"Subprocess exited {row['exit_code']}; see recorded stderr")
                if not row["performance_valid_by_observed_load"]:
                    raise RuntimeError("Observed load1 > 50; timing is invalid, stopped")

            save()
            try:
                with tempfile.TemporaryDirectory(prefix="pair_r5_xtx_") as tmp:
                    reference = Path(tmp) / "reference.rs"
                    reference.write_text(generated)
                    binary = str(Path(tmp) / "probe")
                    env = dict(os.environ, R5_GENERATED_REFERENCE=str(reference), RAYON_NUM_THREADS="1",
                               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    command = ["rustc", "--edition=2021", "tests/pair_r5_shared_xtx_probe.rs",
                               "-O", "-C", "target-cpu=native", "-o", binary]
                    report["build"] = execute(command, env)
                    save()
                    require_success(report["build"])
                    report["correctness"] = execute([binary], env)
                    save()
                    require_success(report["correctness"])
                    report["correctness"]["result"] = json.loads(report["correctness"]["stdout"])
                    if args.run_benchmark:
                        for repeat in range(1, 4):
                            modes = ("reference", "shared") if repeat % 2 else ("shared", "reference")
                            for mode in modes:
                                row = execute([binary, "--bench", mode, str(args.iterations)], env)
                                row.update(repeat=repeat, mode=mode)
                                report["records"].append(row)
                                save()
                                require_success(row)
                                row["result"] = json.loads(row["stdout"])
                                save()
                        groups = {mode: [r["result"]["seconds"] for r in report["records"]
                                         if r["mode"] == mode] for mode in ("reference", "shared")}
                        report["summary"] = {mode: {"first": v[0], "min": min(v), "median": statistics.median(v)}
                                             for mode, v in groups.items()}
                        report["summary"]["median_speedup"] = statistics.median(groups["reference"]) / statistics.median(groups["shared"])
                    report["status"] = "completed" if args.run_benchmark else "correctness_passed"
                    save()
            except BaseException as error:
                report.update(status="needs_review", reason=f"{type(error).__name__}: {error}")
                save()
                raise
    print(json.dumps({"output": str(destination), "status": report["status"], "summary": report.get("summary")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
