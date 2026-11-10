"""Compare the installed trimmed Yupei entry points with the frozen sandbox v6."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "sandbox/yupei_union_trim_20261107/results/speed"
N_FACTORS = 18_758


def compare(actual, reference, row_order):
    import numpy as np

    assert actual.shape == reference.shape
    nans_equal, bits_equal = True, True
    differing_values = 0
    max_abs = max_rel = 0.0
    digest = hashlib.sha256()
    for start in range(0, len(actual), 64):
        end = min(start + 64, len(actual))
        a = actual[row_order[start:end]]
        b = reference[start:end]
        digest.update(a.tobytes(order="C"))
        nans_equal &= bool(np.array_equal(np.isnan(a), np.isnan(b)))
        bits_equal &= bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
        differing_values += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        finite = np.isfinite(a) & np.isfinite(b)
        af, bf = a[finite].astype(np.float64), b[finite].astype(np.float64)
        if bf.size:
            diff = np.abs(af - bf)
            max_abs = max(max_abs, float(diff.max()))
            max_rel = max(max_rel, float((diff / np.maximum(np.abs(bf), 1e-12)).max()))
    return dict(nans_equal=nans_equal, bits_equal=bits_equal, differing_values=differing_values,
                values_sha256=digest.hexdigest(),
                max_abs_diff=max_abs, max_rel_diff=max_rel)


def worker_result(rp, store, date, names):
    import numpy as np
    import zstandard

    started = time.perf_counter()
    rp.run_factor_pipeline_cross_section(
        "yupei_union_trim", [date], n_jobs=50, n_workers=1,
        expected_result_length=len(names), trading_days=[date],
        store_dir=str(store), store_factor_names=names, direct_write=True,
    )
    elapsed = time.perf_counter() - started
    info = rp.factor_store_v5_info(str(store))
    assert info["storage_layout"] == "direct_segments" and list(info["factor_names"]) == names
    manifest = json.loads((store / "segments.manifest.json").read_text())
    assert manifest["factor_names"] == names
    assert manifest["completed_dates"] == [date] and manifest["dates"] == [date]
    assert len(manifest["parts"]) == 1, manifest["parts"]
    part = manifest["parts"][0]
    segment = store / part["file"]
    assert {p.name for p in store.iterdir()} - {".segments.writer.lock"} == {
        "segments.manifest.json", segment.name,
    }
    with segment.open("rb") as f:
        version, cols, rows, row_offset, row_bytes = struct.unpack("<IIQQQ", f.read(32))
        assert version == 3 and cols == N_FACTORS and rows == part["rows"]
        assert row_offset == 32 + 16 * cols
        offsets = np.frombuffer(f.read(16 * cols), dtype="<u8").reshape(cols, 2)
        data_start = row_offset + row_bytes
        assert (offsets[:, 0] == data_start + np.arange(cols) * rows * 4).all()
        assert (offsets[:, 1] == rows * 4).all()
        raw = zstandard.ZstdDecompressor().decompress(f.read(row_bytes), max_output_size=rows * 8)
        row_ids = np.frombuffer(raw, dtype="<u4").reshape(rows, 2)
    assert segment.stat().st_size == data_start + rows * cols * 4
    assert (row_ids[:, 0] == 0).all()
    codes = [manifest["codes"][i] for i in row_ids[:, 1]]
    values = np.memmap(segment, dtype="<f4", mode="r", offset=data_start, shape=(cols, rows)).T
    return codes, values, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["direct", "worker"])
    parser.add_argument("date", type=int)
    parser.add_argument("--legacy-names", type=Path, required=True)
    parser.add_argument("--store-parent", type=Path, default=Path("/hdd/user_home_unsafe/chenzongwei"))
    args = parser.parse_args()

    for key in list(os.environ):
        if key.startswith("YUPEI_"):
            del os.environ[key]
    os.environ.pop("RUST_PYFUNC_WORKER_BIN", None)
    os.environ.pop("RAYON_NUM_THREADS", None)
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", YUPEI_STAGE="1")
    import numpy as np
    import rust_pyfunc as rp
    from rust_pyfunc import rust_pyfunc as extension

    package = Path(rp.__file__).resolve().parent
    assert package == ROOT / "python/rust_pyfunc", package
    worker = package / "rust_pyfunc_worker"
    assert worker.is_file()
    names = rp.py_yupei_union_trim_names()
    legacy = rp.py_yupei_union_all_names()
    assert legacy == json.loads(args.legacy_names.read_text()) and len(legacy) == 20_716
    assert len(names) == len(set(names)) == N_FACTORS
    prefix = REFERENCE / f"v6_{args.date}_r1"
    labels = json.loads(prefix.with_suffix(".labels.json").read_text())
    assert names == labels["names"]
    os.environ["YUPEI_BF16"] = "1"
    try:
        rp.py_yupei_union_trim(args.date)
    except OSError as e:
        assert "YUPEI_BF16" in str(e)
    else:
        raise AssertionError("accepted reduced-precision experiment")
    finally:
        del os.environ["YUPEI_BF16"]
    reference = np.load(prefix.with_suffix(".npy"), mmap_mode="r")
    load_before = os.getloadavg()[0]
    with tempfile.TemporaryDirectory(prefix="verify_yupei_union_trim_", dir=args.store_parent) as tmp:
        if args.mode == "worker":
            store = Path(tmp) / "verify_direct"
            try:
                rp.run_factor_pipeline_cross_section(
                    "yupei_union_trim", [args.date], n_jobs=50, n_workers=1,
                    expected_result_length=N_FACTORS - 1, trading_days=[args.date], store_dir=str(store),
                )
            except ValueError as e:
                assert "18758" in str(e) and not store.exists()
            else:
                raise AssertionError("accepted wrong output width")
            codes, values, seconds = worker_result(rp, store, args.date, names)
        else:
            start = time.perf_counter()
            codes, returned_names, vals = rp.py_yupei_union_trim(args.date)
            seconds = time.perf_counter() - start
            assert returned_names == names and codes == labels["codes"]
            values = np.asarray(vals, dtype=np.float32).reshape(len(codes), N_FACTORS)
            del vals
        row_order = np.argsort(codes)
        assert np.asarray(codes)[row_order].tolist() == labels["codes"]
        result = compare(values, reference, row_order)
        result.update(date=args.date, mode=args.mode, shape=list(values.shape),
                      names_exact=True, codes_exact=True, legacy_names_exact=True,
                      validation_only=True, elapsed_seconds=seconds, threads=50, load1_before=load_before,
                      load1_after=os.getloadavg()[0], extension_path=extension.__file__,
                      extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
                      worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest())
        print("YUPEI_FORMAL_CHECK " + json.dumps(result), flush=True)
        assert result["bits_equal"], result


if __name__ == "__main__":
    main()
