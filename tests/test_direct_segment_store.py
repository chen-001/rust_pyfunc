"""Direct-segment storage checks. All factor files are disposable and stay on HDD.

Build: bash sandbox/direct_segments/build.sh
Run: python -m pytest -p no:cacheprovider tests/test_direct_segment_store.py -q
Set DIRECT_SEGMENT_BENCH=1 to include the 256 MiB write/read comparison.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

SANDBOX = Path(__file__).resolve().parents[1] / "sandbox" / "direct_segments"
sys.path.insert(0, str(SANDBOX))
sb = pytest.importorskip("direct_segments_sandbox", reason="build sandbox/direct_segments first")
NAMES = ["f0", "f1", "f2"]


@pytest.fixture
def work():
    root = Path(os.environ.get("DIRECT_SEGMENT_TEST_ROOT", "/hdd/user_home_unsafe/chenzongwei")).resolve()
    if not root.is_relative_to(Path("/hdd")):
        pytest.fail("factor test files must stay on /hdd (including symlink targets)")
    with tempfile.TemporaryDirectory(prefix="codex_direct_segments_", dir=root) as path:
        yield Path(path)


def child(code, *args):
    env = dict(os.environ, PYTHONPATH=str(SANDBOX))
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], env=env,
                          capture_output=True, text=True, timeout=60)


def write(path, rows, budget=36, names=NAMES):
    writer = sb.Writer(str(path), names, budget)
    if rows:
        writer.append_rows([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
    writer.finish()
    writer.close()


def assert_values(path, rows, names=NAMES):
    ds, cs, actual = sb.read_matrix(str(path), list(range(len(names))), True)
    expected = np.full((len(names), len(ds), len(cs)), np.nan, np.float32)
    for date, code, values in rows:
        values = np.asarray(values, np.float32)
        expected[:, ds.index(date), cs.index(code)] = np.where(np.isfinite(values), values, np.nan)
    np.testing.assert_array_equal(np.asarray(actual).reshape(expected.shape), expected)
    np.testing.assert_array_equal(actual, sb.read_matrix(str(path), list(range(len(names))), False)[2])
    reads, cached = sb.read_cached(str(path), list(range(len(names))), 2)
    assert reads == (len(names) + 1) // 2
    np.testing.assert_array_equal(actual, cached)


def test_multisegment_python_reference_and_legacy(work):
    rows = [(20230104, "600001", [1, np.nan, 3]), (20230102, "000002", [4, 5, np.inf]),
            (20230103, "000001", [7, 8, -0.0]), (20230102, "600001", [np.nan, 11, 12]),
            (20230104, "000001", [-13, 14, -np.inf]), (20230103, "600001", [16, 17, 18]),
            (20230105, "000002", [19, 20, 21])]
    direct, legacy = work / "direct", work / "legacy"
    write(direct, rows)
    info = sb.factor_store_v5_info(str(direct))
    assert info["is_projected"] and info["record_count"] == len(rows)
    assert info["storage_layout"] == "direct_segments" and info["direct_segment_count"] == 3
    assert_values(direct, rows)
    sb.write_legacy(str(legacy), [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], NAMES)
    assert_values(legacy, rows)
    np.testing.assert_array_equal(sb.read_matrix(str(direct), [0, 1, 2], True)[2],
                                  sb.read_matrix(str(legacy), [0, 1, 2], True)[2])
    files = {p.name for p in direct.iterdir()}
    assert files == {".segments.writer.lock", "segments.manifest.json",
                     "segment_00000000.bin", "segment_00000001.bin", "segment_00000002.bin"}


def test_empty_store_and_factor_names(work):
    path = work / "empty"
    write(path, [])
    info = sb.factor_store_v5_info(str(path))
    assert info["record_count"] == 0 and info["is_projected"] and info["direct_segment_count"] == 0
    assert sb.read_matrix(str(path), [0, 1, 2], True)[2] == [[], [], []]
    before = (path / "segments.manifest.json").read_bytes()
    with pytest.raises(OSError, match="因子名|顺序"):
        sb.Writer(str(path), NAMES[::-1], 36)
    assert (path / "segments.manifest.json").read_bytes() == before
    with pytest.raises(OSError):
        sb.Writer(str(work / "duplicate_names"), ["a", "a"], 36)
    with pytest.raises(OSError):
        sb.Writer(str(work / "tiny_budget"), NAMES, 1)


def test_crash_discards_only_unpublished_rows_and_releases_lock(work):
    path = work / "crash"
    result = child("import os,sys,direct_segments_sandbox as s; "
                   "w=s.Writer(sys.argv[1],['f0','f1','f2'],1024); "
                   "w.append_rows([20230102],['000001'],[[1,2,3]]); w.finish(); "
                   "w.append_rows([20230103],['000002'],[[4,5,6]]); os._exit(17)", path)
    assert result.returncode == 17, result.stderr
    writer = sb.Writer(str(path), NAMES, 1024)
    assert writer.completed() == [(20230102, "000001")]
    assert sb.factor_store_v5_info(str(path))["record_count"] == 1
    writer.append_rows([20230102, 20230103], ["000001", "000002"], [[1, 2, 3], [4, 5, 6]])
    writer.finish()
    writer.close()
    assert_values(path, [(20230102, "000001", [1, 2, 3]), (20230103, "000002", [4, 5, 6])])
    assert sb.factor_store_v5_info(str(path))["record_count"] == 2


def test_day_split_pending_day_and_resume(work):
    path = work / "dates"
    writer = sb.Writer(str(path), NAMES, 24)  # Two rows per segment; day crosses segments.
    writer.append_date(20230102, ["000003", "000001", "000002"], [[7, 8, 9], [1, 2, 3], [4, 5, 6]])
    assert writer.completed_dates() == []
    writer.close()  # First segment durable; last row was only in memory.
    writer = sb.Writer(str(path), NAMES, 24)
    assert len(writer.completed()) == 2 and writer.completed_dates() == []
    writer.append_date(20230102, ["000002", "000003", "000001"], [[4, 5, 6], [7, 8, 9], [1, 2, 3]])
    writer.finish()
    assert writer.completed_dates() == [20230102] and len(writer.completed()) == 3
    writer.close()
    assert_values(path, [(20230102, "000003", [7, 8, 9]), (20230102, "000001", [1, 2, 3]),
                         (20230102, "000002", [4, 5, 6])])


def test_writer_lock_and_input_validation(work):
    path = work / "locked"
    writer = sb.Writer(str(path), NAMES, 36)
    result = child("import sys,direct_segments_sandbox as s; s.Writer(sys.argv[1],['f0','f1','f2'],36)", path)
    assert result.returncode and "写入者" in result.stderr
    with pytest.raises(OSError):
        writer.append_rows([20230102], ["000001"], [[1, 2]])
    with pytest.raises(OSError):
        writer.append_rows([20230102, 20230102], ["000001", "000001"], [[1, 2, 3]] * 2)
    with pytest.raises(OSError):
        writer.append_date(20230102, [], [])
    writer.append_date(20230102, ["000001"], [[1, 2, 3]])
    writer.finish()
    writer.close()
    reopened = sb.Writer(str(path), NAMES, 36)
    assert reopened.completed_dates() == [20230102]
    with pytest.raises(OSError, match="已整日提交"):
        reopened.append_rows([20230102], ["000002"], [[4, 5, 6]])
    with pytest.raises(OSError, match="已整日提交"):
        reopened.append_date(20230102, ["000001", "000002"], [[1, 2, 3], [4, 5, 6]])
    reopened.close()


@pytest.mark.parametrize("damage", ["manifest", "header", "truncate", "missing"])
def test_published_corruption_is_an_error(work, damage):
    path = work / damage
    write(path, [(20230102, "000001", [1, 2, 3])])
    segment = path / "segment_00000000.bin"
    if damage == "manifest":
        (path / "segments.manifest.json").write_text("{broken")
    elif damage == "missing":
        segment.unlink()
    else:
        raw = segment.read_bytes()
        segment.write_bytes(raw[:-4] if damage == "truncate" else b"\0" * 4 + raw[4:])
    with pytest.raises(OSError):
        sb.read_matrix(str(path), [0], False)
    with pytest.raises(OSError):
        sb.Writer(str(path), NAMES, 36)


def test_unpublished_orphans_and_mixed_layout_rejected(work):
    path = work / "orphans"
    write(path, [(20230102, "000001", [1, 2, 3])])
    shutil.copyfile(path / "segment_00000000.bin", path / "segment_00000001.bin")
    (path / "segment_00000002.tmp").write_bytes(b"partial")
    assert sb.factor_store_v5_info(str(path))["record_count"] == 1
    writer = sb.Writer(str(path), NAMES, 36)
    assert len(writer.completed()) == 1
    writer.append_rows([20230103], ["000002"], [[4, 5, 6]])
    writer.finish()
    writer.close()
    assert sb.factor_store_v5_info(str(path))["record_count"] == 2
    legacy = work / "legacy"
    sb.write_legacy(str(legacy), [20230102], ["000001"], [[1, 2, 3]], NAMES)
    with pytest.raises(OSError):
        sb.Writer(str(legacy), NAMES, 36)
    with pytest.raises(OSError):
        sb.write_legacy(str(path), [20230104], ["000003"], [[7, 8, 9]], NAMES)


def test_mixed_groups_preserve_batch_reads(work):
    root = work / "groups"
    rows = [(20230102, "000001", [1, 2, 3]), (20230103, "600001", [4, 5, 6])]
    write(root / "direct", rows, 12)
    sb.write_legacy(str(root / "legacy"), [r[0] for r in rows], [r[1] for r in rows],
                    [r[2] for r in rows], ["g0", "g1", "g2"])
    sb.factor_store_v5_register_group(str(root), "direct", "direct")
    sb.factor_store_v5_register_group(str(root), "legacy", "legacy")
    assert sb.factor_store_v5_info(str(root))["factor_count"] == 6
    expected = sb.read_matrix(str(root / "direct"), [0, 1, 2], True)[2] * 2
    np.testing.assert_array_equal(sb.read_matrix(str(root), list(range(6)), True)[2], expected)
    count, matrices = sb.read_cached(str(root), list(range(6)), 64)
    assert count == 2  # A batch cannot cross factor-group boundaries.
    np.testing.assert_array_equal(matrices, expected)


@pytest.mark.skipif(os.environ.get("DIRECT_SEGMENT_BENCH") != "1", reason="opt-in HDD benchmark")
def test_256mib_write_read_comparison(work):
    # Equal data, one warm process, persisted end-to-end wall times; no cache dropping.
    names = [f"f{i}" for i in range(512)]
    n = 131072
    dates = [20230102 + i // 8192 for i in range(n)]
    codes = [f"{i % 8192:06}" for i in range(n)]
    values = np.random.default_rng(19).normal(size=(n, len(names))).astype(np.float32)
    values[::997, ::13] = np.nan
    rows = values.tolist()
    measurements = {}
    for mode in ["legacy", "direct"]:
        path = work / mode
        started = time.perf_counter()
        if mode == "legacy":
            sb.write_legacy(str(path), dates, codes, rows, names)
        else:
            writer = sb.Writer(str(path), names, 64 * 1024 * 1024)
            writer.append_rows(dates, codes, rows)
            writer.finish()
            writer.close()
        measurements[mode + "_write_seconds"] = time.perf_counter() - started
        measurements[mode + "_disk_bytes"] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        started = time.perf_counter()
        read = sb.read_matrix(str(path), list(range(16)), True)[2]
        measurements[mode + "_read16_seconds"] = time.perf_counter() - started
        if mode == "legacy":
            baseline = read
        else:
            np.testing.assert_array_equal(read, baseline)
    assert measurements["direct_disk_bytes"] < measurements["legacy_disk_bytes"] * 0.6
    print("DIRECT_SEGMENT_BENCH " + json.dumps(measurements, sort_keys=True))
