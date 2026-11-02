"""Installed full-day validation: python tests/test_relaynet_six_people_deploy.py direct|pipeline 20241231|20220819."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {20241231: "candidate_20241231_clean_b", 20220819: "candidate_20220819_clean_a"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_names(candidate, actual):
    """Only the later occurrence of each audited PM split name may change."""
    suffixes = ["deg", "str", "bet", "harm", "pagerank", "kcore", "dist_mkt", "top1", "compsize", "neighind"]
    expected, renamed = list(candidate), []
    for offset, suffix in enumerate(suffixes):
        old = f"rlg_pm30_rd_thr20_{suffix}"
        positions = [i for i, name in enumerate(candidate) if name == old]
        assert len(positions) == 2 and positions[-1] == 13530 + offset, (old, positions)
        new = f"rlg_pm30_split_rd_thr20_{suffix}"
        expected[positions[-1]] = new
        renamed.append(dict(column_index=positions[-1], old=old, new=new))
    assert len(actual) == len(set(actual)) == len(candidate), "new factor names must be unique"
    assert actual == expected, "unexpected name/position change beyond the ten audited renames"
    assert sum(a != b for a, b in zip(candidate, actual)) == 10
    return renamed


def check_bad_factor_count(rp, output, date, names):
    """A bad column count must fail before even an explicit clear touches a store."""
    store = output / "invalid_count_marker_store"
    store.mkdir()
    marker = store / "keep.txt"
    content = b"existing store must survive invalid expected_result_length\n"
    marker.write_bytes(content)
    before = marker.stat()
    try:
        rp.run_factor_pipeline_cross_section(
            pipeline="relaynet_six_people", tasks=[], n_jobs=50, n_workers=1,
            expected_result_length=len(names) - 1, trading_days=[date], bind_cores=False,
            store_dir=str(store), store_factor_names=names, update_mode=False, force_clear=True)
    except ValueError as error:
        assert "relaynet_six_people" in str(error) and "expected_result_length" in str(error)
        message = str(error)
    else:
        raise AssertionError("incorrect factor count did not raise ValueError")
    assert list(store.iterdir()) == [marker] and marker.read_bytes() == content
    after = marker.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    return dict(value_error=message, existing_marker_unchanged=True)


def compare_all(actual, expected):
    """Compare every finite f32 bit, including signed zero, and all nonfinite masks."""
    import numpy as np
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    counts = dict(finite=0, nan=0, positive_inf=0, negative_inf=0)
    for start in range(0, expected.shape[1], 64):
        a = np.asarray(actual[:, start:start + 64], dtype=np.float32)
        b = np.asarray(expected[:, start:start + 64], dtype=np.float32)
        for key, operation in [("nan", np.isnan), ("positive_inf", np.isposinf),
                               ("negative_inf", np.isneginf)]:
            mask = operation(b)
            assert np.array_equal(operation(a), mask), (start, key)
            counts[key] += int(mask.sum())
        finite = np.isfinite(b)
        assert np.array_equal(a.view(np.uint32)[finite], b.view(np.uint32)[finite]), (
            "finite f32 bits differ", start)
        counts["finite"] += int(finite.sum())
    return counts


def projection_values(store, date, names, codes):
    """Read v3 raw values: the public factor reader drops NaN and both infinities."""
    import numpy as np
    import zstandard
    shard = store / f"shard_{date % 8}"
    dates = list(dict.fromkeys(np.fromfile(shard / "dates.bin", dtype="<i8").tolist()))
    raw_codes = (shard / "codes.bin").read_bytes()
    assert len(raw_codes) % 16 == 0
    dictionary = list(dict.fromkeys(raw_codes[i:i + 16].split(b"\0", 1)[0].decode()
                                   for i in range(0, len(raw_codes), 16)))
    path = shard / "factors.proj"
    with path.open("rb") as stream:
        version, columns, rows, row_offset, row_bytes = struct.unpack("<IIQQQ", stream.read(32))
        assert (version, columns, rows) == (3, len(names), len(codes))
        segments = np.frombuffer(stream.read(columns * 16), dtype="<u8").reshape(columns, 2)
        stream.seek(row_offset)
        order = zstandard.ZstdDecompressor().decompress(
            stream.read(row_bytes), max_output_size=rows * 8)
    order = np.frombuffer(order, dtype="<u4").reshape(rows, 2)
    assert all(dates[int(i)] == date for i in order[:, 0])
    assert [dictionary[int(i)] for i in order[:, 1]] == codes, "worker/projection stock order"
    assert np.all(segments[:, 1] == rows * 4)
    assert np.array_equal(segments[:, 0], segments[0, 0] + np.arange(columns) * rows * 4)
    assert int(segments[-1, 0] + segments[-1, 1]) == path.stat().st_size
    assert not list(store.glob("shard_*/factors.proj.delta.*")), "fresh store must have base projection"
    return np.memmap(path, dtype="<f4", mode="r", offset=int(segments[0, 0]),
                     shape=(columns, rows)).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["direct", "pipeline"])
    parser.add_argument("date", type=int, choices=REFERENCES)
    args = parser.parse_args()
    # One fresh process, one whole date, one worker with 50 computation threads.
    os.environ["RAYON_NUM_THREADS"] = "50"
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import numpy as np
    import rust_pyfunc as rp

    reference = ROOT / "reports/relaynet_perf_20261102" / REFERENCES[args.date]
    meta = json.loads(reference.with_suffix(".json").read_text())
    expected = np.load(reference.with_suffix(".npy"), mmap_mode="r")
    names = rp.py_relaynet_six_people_names()
    renamed = check_names(meta["names"], names)
    assert meta["date"] == args.date and list(expected.shape) == meta["shape"]
    assert expected.dtype == np.float32 and len(names) == 13574
    output_root = ROOT / "reports/relaynet_six_people_deploy"
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{args.mode}_{args.date}_", dir=output_root))
    library = Path(importlib.import_module("rust_pyfunc.rust_pyfunc").__file__).resolve()
    report = dict(mode=args.mode, date=args.date, threads=50, reference=str(reference),
                  shape=list(expected.shape), library=str(library), library_sha256=sha256(library),
                  renamed_columns=renamed)
    try:
        report["invalid_count_guard"] = check_bad_factor_count(rp, output, args.date, names)
        if args.mode == "direct":
            started = time.perf_counter()
            codes, values = rp.py_relaynet_six_people(args.date)
            report["seconds"] = time.perf_counter() - started
            assert codes == meta["codes"], "direct stock order differs from candidate reference"
            actual = np.asarray(values, dtype=np.float32).reshape(expected.shape)
            del values
        else:
            worker = (Path(rp.__file__).resolve().parent / "rust_pyfunc_worker").resolve()
            assert worker.is_file(), f"deployed worker missing: {worker}"
            os.environ["RUST_PYFUNC_WORKER_BIN"] = str(worker)
            report.update(worker=str(worker), worker_sha256=sha256(worker), n_workers=1)
            store = output / "store"
            started = time.perf_counter()
            rp.run_factor_pipeline_cross_section(
                pipeline="relaynet_six_people", tasks=[args.date], n_jobs=50, n_workers=1,
                expected_result_length=len(names), trading_days=[args.date], bind_cores=False,
                store_dir=str(store), store_factor_names=names, update_mode=False,
                incremental_projection=False)
            report["seconds"] = time.perf_counter() - started
            info = rp.factor_store_v5_info(str(store))
            assert info["is_projected"] and info["record_count"] == len(meta["codes"])
            assert info["factor_count"] == len(names) and info["factor_names"] == names
            template = rp.factor_store_v5_template(str(store))
            assert list(template["dates"]) == [args.date]
            assert [code.split(".")[0] for code in template["stocks"]] == meta["codes"]
            assert (store / "_completed_dates").read_text().split() == [str(args.date)]
            actual = projection_values(store, args.date, names, meta["codes"])
            report.update(store=str(store), is_projected=True)
        report["compared"] = compare_all(actual, expected)
        report["passed"] = True
    except Exception as error:
        report.update(passed=False, error=repr(error))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(report, report=str(output / "result.json"))), flush=True)


if __name__ == "__main__":
    main()
