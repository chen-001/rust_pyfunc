"""第三版验证：只读 Rust 输出，不参与任何因子合成。
检查列数与命名、撤单列空值规则、2016 年读取修复、全空列、输出大小与全历史存储估计。"""
import json, os, sys
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v3")
def load(tag):
    m = json.load(open(f"{R}/{tag}.json"))
    X = np.memmap(f"{R}/{tag}.f32", dtype="<f4", mode="r", shape=(m["rows"], m["cols"]))
    return m, X

for tag in ["opt_20241231", "opt_20211105", "opt_20160630"]:
    m, X = load(tag)
    names = np.array(m["names"]); codes = np.array(m["codes"])
    cxl = np.array(["cxl" in n for n in names])
    sh = np.array([c.startswith("6") for c in codes]); sz = ~sh
    fin = np.isfinite(X)
    print(f"\n=== {tag}: 股票 {m['rows']}  列 {m['cols']}  列名唯一 {len(set(m['names'])) == m['cols']}  含 cxl 列 {cxl.sum()}")
    print(f"  沪市股票 {sh.sum()}  深市股票 {sz.sum()}")
    print(f"  cxl 列有值的格子：沪市 {int(fin[sh][:, cxl].sum())}  深市 {int(fin[sz][:, cxl].sum())}")
    print(f"  非 cxl 列有值占比：沪市 {fin[sh][:, ~cxl].mean():.3f}  深市 {fin[sz][:, ~cxl].mean():.3f}")
    print(f"  cxl 列有值占比：沪市 {fin[sh][:, cxl].mean():.3f}  深市 {fin[sz][:, cxl].mean():.3f}")
    allnan = ~fin.any(axis=0)
    print(f"  整列为空的列数 {int(allnan.sum())}（其中 cxl {int((allnan & cxl).sum())}）", list(names[allnan][:5]))
    zero_cxl = int((X[:, cxl] == 0).sum())
    print(f"  cxl 列中恰为 0 的格子数 {zero_cxl}（撤单缺失时应为空值，不应是 0；有撤单的日子可以出现真实的 0）")
    if tag == "opt_20160630":
        i = list(codes).index("000001")
        print(f"  000001 有值的列数 {int(fin[i].sum())} / {m['cols']}")
        # 2016 年代码写法：统计文件第一行数据的代码是否是 6 位
        d = "/nvme/level2/20160630/transaction"
        short = []
        for c in codes:
            with open(f"{d}/{c}_20160630_transaction.csv") as f:
                f.readline(); line = f.readline()
            if line and line.split(",")[0] != c: short.append(c)
        rows_ok = [int(fin[list(codes).index(c)].sum() > 0) for c in short]
        print(f"  代码不是 6 位写法的文件 {len(short)} 个（例 {short[:5]}），其中按数值比较后有因子值的 {sum(rows_ok)} 个")
        print(f"  全天有值的股票 {int(fin.any(axis=1).sum())} / {m['rows']}")

# 输出大小与全历史存储估计（按每年第一个交易日的股票文件数近似全年）
root = "/nvme/level2"
days = sorted(d for d in os.listdir(root) if d.isdigit() and len(d) == 8)
by_year = {}
for d in days: by_year.setdefault(d[:4], []).append(d)
total = 0
print("\n年份 交易日数 年初逐笔文件数")
for y, ds in sorted(by_year.items()):
    n = len(os.listdir(f"{root}/{ds[0]}/transaction"))
    total += len(ds) * n
    print(y, len(ds), n)
cols = 42924
print(f"全历史 {len(days)} 个交易日，股票-日约 {total:,}，因子值未压缩约 {total * cols * 4 / 1e12:.2f} TB")
