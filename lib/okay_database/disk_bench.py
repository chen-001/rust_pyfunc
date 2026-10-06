"""对比 /nas197/user_home/chenzongwei（NFS）和 /hdd/user_home_unsafe/chenzongwei（本地机械盘 btrfs）的读写速度。

测六项，每项两边用同一份代码、同一份数据、同样的顺序，只换目录：
  1. 顺序写 512 MB 不可压缩数据（模仿因子矩阵，随机字节）
  2. 顺序写 512 MB 可压缩数据（全零，看本地 zstd 压缩能省多少时间）
  3. 顺序读 512 MB（读之前用 posix_fadvise 把这一页缓存丢掉，避免测到内存）
  4. 随机读 4 KB × 2000 次（同样先丢缓存）
  5. 元数据：创建 2000 个 1 KB 小文件 → 遍历 stat → 删除
  6. 单次写 4 KB 并 fsync 的延迟（小文件落地要多久）

注意：跑的时候机器上还有 400 并行的因子计算任务，负载 80 以上，两个路径都会被拖慢。
数字只做两边相对比较，不代表空载性能。
"""
from __future__ import annotations

import os
import random
import shutil
import time
from pathlib import Path

SIZE = 512 * 1024 * 1024
BLOCK = 1024 * 1024
SMALL_N = 2000
RAND_READS = 2000
RAND_BLOCK = 4096

TARGETS = [
    ("NAS197(NFS)", Path("/nas197/user_home/chenzongwei/.speedtest_tmp")),
    ("HDD(本地btrfs)", Path("/hdd/user_home_unsafe/chenzongwei/.speedtest_tmp")),
]


def drop_cache(path: Path) -> None:
    """把文件从页缓存里踢掉，让读测试真的走存储。"""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def mbps(nbytes: int, secs: float) -> float:
    return nbytes / 1024 / 1024 / secs if secs > 0 else float("inf")


def bench(root: Path) -> dict:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    res = {}

    # 1. 顺序写：不可压缩
    rnd_path = root / "seq_rand.bin"
    chunk = os.urandom(BLOCK)
    t0 = time.perf_counter()
    with open(rnd_path, "wb", buffering=0) as fh:
        for _ in range(SIZE // BLOCK):
            fh.write(chunk)
        os.fsync(fh.fileno())
    res["顺序写(不可压缩)"] = mbps(SIZE, time.perf_counter() - t0)

    # 2. 顺序写：可压缩（全零）
    zero_path = root / "seq_zero.bin"
    zero = bytes(BLOCK)
    t0 = time.perf_counter()
    with open(zero_path, "wb", buffering=0) as fh:
        for _ in range(SIZE // BLOCK):
            fh.write(zero)
        os.fsync(fh.fileno())
    res["顺序写(可压缩)"] = mbps(SIZE, time.perf_counter() - t0)

    # 3. 顺序读
    drop_cache(rnd_path)
    t0 = time.perf_counter()
    with open(rnd_path, "rb", buffering=0) as fh:
        while fh.read(BLOCK):
            pass
    res["顺序读"] = mbps(SIZE, time.perf_counter() - t0)

    # 4. 随机读 4K
    drop_cache(rnd_path)
    offsets = [random.randrange(0, SIZE - RAND_BLOCK, RAND_BLOCK) for _ in range(RAND_READS)]
    t0 = time.perf_counter()
    with open(rnd_path, "rb", buffering=0) as fh:
        for off in offsets:
            fh.seek(off)
            fh.read(RAND_BLOCK)
    secs = time.perf_counter() - t0
    res["随机读4K(次/秒)"] = RAND_READS / secs

    # 5. 元数据：建 2000 个小文件 + stat + 删
    small = root / "many"
    small.mkdir()
    t0 = time.perf_counter()
    for i in range(SMALL_N):
        (small / f"f{i}.txt").write_bytes(b"x" * 1024)
    create = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(SMALL_N):
        (small / f"f{i}.txt").stat()
    stat_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    shutil.rmtree(small)
    delete = time.perf_counter() - t0
    res["建小文件(个/秒)"] = SMALL_N / create
    res["遍历stat(个/秒)"] = SMALL_N / stat_s
    res["删小文件(个/秒)"] = SMALL_N / delete

    # 6. 单次 4K 写 + fsync 延迟
    lat_path = root / "lat.bin"
    n = 100
    t0 = time.perf_counter()
    with open(lat_path, "ab", buffering=0) as fh:
        for _ in range(n):
            fh.write(b"y" * 4096)
            os.fsync(fh.fileno())
    res["4K写+fsync(毫秒/次)"] = (time.perf_counter() - t0) / n * 1000

    # 7. 删掉大文件本身要多久（顺带看看删除代价）
    t0 = time.perf_counter()
    rnd_path.unlink()
    zero_path.unlink()
    lat_path.unlink()
    root.rmdir()
    res["删512MB×2+收尾(秒)"] = time.perf_counter() - t0
    return res


def main() -> int:
    out = {}
    for label, root in TARGETS:
        print(f"=== {label}  {root}", flush=True)
        try:
            out[label] = bench(root)
        except Exception as exc:  # noqa: BLE001
            out[label] = {"错误": str(exc)}
            print("   失败:", exc, flush=True)
        for k, v in out[label].items():
            print(f"   {k:<22} {v:,.1f}", flush=True)

    keys = [k for k in out[TARGETS[0][0]] if k != "错误"]
    print("\n=== 对比（NAS197 / HDD，倍数）")
    for k in keys:
        a = out[TARGETS[0][0]].get(k)
        b = out[TARGETS[1][0]].get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
            print(f"{k:<24} NAS {a:>12,.1f}   HDD {b:>12,.1f}   倍数 {a / b:>8.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
