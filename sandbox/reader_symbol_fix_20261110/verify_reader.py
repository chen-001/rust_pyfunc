"""读取器“代码列按数值比较”修复的前后对照。

用法：python verify_reader.py <date> <标签>
  对白名单内、当天有文件的每只股票，用当前安装的 rust_pyfunc 读逐笔（含撤单、含下午平移）、
  逐笔微秒版、盘口，记录行数和数组字节哈希，写到 results/<date>_<标签>.json。
修复前后各跑一次，再用 compare 子命令对比：
  python verify_reader.py compare <date> <标签A> <标签B>
最多 50 个进程。
"""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[k] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROOT = "/nvme/level2"
SYMBOL_MAP = "/ssd_data/data/basic_info/symbol_map.csv"


def stock_codes(date):
    import csv
    with open(SYMBOL_MAP) as f:
        allowed = {r["symbol"].strip().zfill(6) for r in csv.DictReader(f)}
    out = set()
    for sub, suf in (("transaction", "transaction"), ("market_data", "market_data")):
        d = os.path.join(ROOT, str(date), sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            code = name.split("_")[0]
            if code in allowed and name.endswith(f"_{date}_{suf}.csv"):
                out.add(code)
    return sorted(out)


def feed(h, obj):
    """递归把数组或（嵌套）字典按键名顺序写进哈希，返回第一个数组的行数。"""
    if isinstance(obj, dict):
        rows = None
        for k in sorted(obj):
            h.update(k.encode())
            r = feed(h, obj[k])
            rows = r if rows is None else rows
        return -2 if rows is None else rows
    if hasattr(obj, "tobytes"):
        h.update(obj.tobytes())
        return int(obj.shape[0])
    h.update(repr(obj).encode())
    return -2


def one(args):
    code, date = args
    import rust_pyfunc as rp
    rec = {}
    for key, fn in (
        ("trade", lambda: rp.read_trade_fast(code, date, 1, True)),
        ("trade_us", lambda: rp.read_trade_fast_us(code, date, 1, True)),
        ("market", lambda: rp.read_market_fast(code, date, 1, True)),
        ("market_pair", lambda: rp.read_market_pair_fast(code, date)),
    ):
        try:
            a = fn()
            h = hashlib.sha1()
            rows = feed(h, a)
            rec[key] = [rows, h.hexdigest()]
        except Exception as e:  # 没有文件等
            rec[key] = [-1, type(e).__name__]
    return code, rec


def run(date, tag):
    codes = stock_codes(date)
    with Pool(50) as p:
        res = dict(p.map(one, [(c, date) for c in codes], chunksize=8))
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"{date}_{tag}.json"), "w") as f:
        json.dump(res, f)
    for key in ("trade", "trade_us", "market", "market_pair"):
        n_rows = sum(1 for v in res.values() if v[key][0] > 0)
        print(f"{date} {tag} {key}: 有文件股票 {len(res)}，读到非空 {n_rows}")


def compare(date, a, b):
    A = json.load(open(os.path.join(RES, f"{date}_{a}.json")))
    B = json.load(open(os.path.join(RES, f"{date}_{b}.json")))
    assert A.keys() == B.keys(), "股票集合不同"
    for key in ("trade", "trade_us", "market", "market_pair"):
        diff = [c for c in A if A[c][key] != B[c][key]]
        na = sum(1 for c in A if A[c][key][0] > 0)
        nb = sum(1 for c in B if B[c][key][0] > 0)
        print(f"{date} {key}: {a} 非空 {na} / {b} 非空 {nb}；结果不同的股票 {len(diff)}"
              + (f"，例 {diff[:5]}" if diff else "，全部逐字节相同"))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    else:
        run(int(sys.argv[1]), sys.argv[2])
