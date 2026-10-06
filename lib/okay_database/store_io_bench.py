"""对比「边算边存」写入 和「回测读因子」读取，在 NAS 和本地 HDD 上的速度。

两段 IO 形态都是从生产代码里抄出来的：

写入（对应 factor_store_v5 的 ShardedBackupSink）：
  8 个分片各自顺序追加 chunk，chunk 内部是按列拼好后 zstd 压缩的一块；
  文件用 fallocate 预留连续空间；header 每 N 个 chunk 同步一次，不是每块都 fsync。

读取（对应 factor_store_v5_read_factor / 回测引擎的 thread-per-factor）：
  每个因子读一段连续的值区（rows × 4 字节），另加一段共享行序（每次调用都读一遍）。

用法：python lib/okay_database/store_io_bench.py
"""
from __future__ import annotations

import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import rust_pyfunc as rp
import zstandard

HDD_STORE = "/hdd/user_home_unsafe/chenzongwei/.speedtest_store"
NAS_STORE = "/nas197/user_home/chenzongwei/.speedtest_store"

HDD_WRITE = Path("/hdd/user_home_unsafe/chenzongwei/.speedtest_write")
NAS_WRITE = Path("/nas197/user_home/chenzongwei/.speedtest_write")

N_SHARDS = 8
CHUNK_ROWS = 2000
CHUNK_FACTORS = 40
CHUNKS_PER_SHARD = 40          # 每分片 ≈ 40 × 2000 × 40 × 4B ≈ 12.8 MB 压缩前
READ_FACTORS = 120
READ_THREADS = 4


def drop_cache(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def bench_write(root: Path) -> dict:
    """8 分片并行追加 zstd chunk，模仿 collector 的写盘路径。"""
    import shutil
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    cctx = zstandard.ZstdCompressor(level=3)
    rng = np.random.default_rng(0)
    # 预先造好一块数据（列优先的 f32），避免把随机数生成的耗时算进写盘
    raw = rng.standard_normal(CHUNK_ROWS * CHUNK_FACTORS).astype(np.float32).tobytes()
    blobs = [cctx.compress(raw) for _ in range(CHUNKS_PER_SHARD)]
    total = sum(len(b) for b in blobs) * N_SHARDS
    uncompressed = len(raw) * CHUNKS_PER_SHARD * N_SHARDS

    files = []
    for s in range(N_SHARDS):
        f = open(root / f"shard_{s}.colblk", "wb", buffering=0)
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.fallocate(f.fileno(), 0, 0, 2 * 1024 ** 3)
        except Exception:
            pass
        files.append(f)

    err = []

    def worker(idx: int) -> None:
        f = files[idx]
        try:
            for b in blobs:
                f.write(b)
        except Exception as exc:  # noqa: BLE001
            err.append(str(exc))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_SHARDS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for f in files:
        os.fsync(f.fileno())
        f.close()
    dt = time.perf_counter() - t0
    if err:
        raise RuntimeError(err[0])
    return {"墙钟秒": dt, "实际写入MB": total / 1e6,
            "写入MB/s": total / 1e6 / dt, "等价未压缩MB/s": uncompressed / 1e6 / dt}


def bench_read(store: str, n: int = READ_FACTORS, threads: int = READ_THREADS) -> dict:
    proj = Path(store) / "shard_0" / "factors.proj"
    drop_cache(proj)
    results: list[float] = []
    lock = threading.Lock()
    idx = list(range(n))
    pos = [0]

    def worker() -> None:
        while True:
            with lock:
                if pos[0] >= len(idx):
                    return
                i = idx[pos[0]]
                pos[0] += 1
            t0 = time.perf_counter()
            rp.factor_store_v5_read_factor(store, i)
            with lock:
                results.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    n_rows = 1_413_690
    useful = n * n_rows * 4
    return {"墙钟秒": dt, "因子数": n, "并行度": threads,
            "单因子均值ms": statistics.mean(results) * 1000,
            "值区吞吐MB/s": useful / 1e6 / dt}


def main() -> int:
    out = {}
    for label, wroot, store in (("HDD(本地btrfs)", HDD_WRITE, HDD_STORE),
                                ("NAS197(NFS)", NAS_WRITE, NAS_STORE)):
        print(f"=== {label}", flush=True)
        try:
            w = bench_write(wroot)
            print(f"   写入：8 分片并行追加 {w['实际写入MB']:.0f} MB（压缩后），"
                  f"{w['墙钟秒']:.2f}s，{w['写入MB/s']:.1f} MB/s（压缩后），"
                  f"等价未压缩 {w['等价未压缩MB/s']:.1f} MB/s", flush=True)
        except Exception as exc:  # noqa: BLE001
            w = {"错误": str(exc)}
            print("   写入失败:", exc, flush=True)
        try:
            r = bench_read(store)
            print(f"   读取：{r['并行度']} 线程读 {r['因子数']} 个因子，"
                  f"{r['墙钟秒']:.2f}s，单因子均值 {r['单因子均值ms']:.1f} ms，"
                  f"值区吞吐 {r['值区吞吐MB/s']:.1f} MB/s", flush=True)
        except Exception as exc:  # noqa: BLE001
            r = {"错误": str(exc)}
            print("   读取失败:", exc, flush=True)
        out[label] = (w, r)
    import json
    Path("/tmp/store_io_bench.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
