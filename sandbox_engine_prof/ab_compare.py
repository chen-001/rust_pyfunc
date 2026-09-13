"""ab_before vs ab_after: task_results msgpack 逐位对比 + 汇总。"""
import sys, os, json

def files(d):
    return {f: os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".msgpack")}

a = files("/tmp/ab_before_task_results")
b = files("/tmp/ab_after_task_results")
print(f"before={len(a)} after={len(b)}")
assert set(a) == set(b), f"文件集合不一致: {set(a) ^ set(b)}"

diff_count = 0
for name in a:
    with open(a[name], "rb") as fa, open(b[name], "rb") as fb:
        ba, bb = fa.read(), fb.read()
    if ba != bb:
        # 定位第一个差异位置, 并比较长度
        common = 0
        for x, y in zip(ba, bb):
            if x != y:
                break
            common += 1
        print(f"DIFF {name}: len {len(ba)} vs {len(bb)}, first_diff_at={common}")
        diff_count += 1
print(f"identical={len(a) - diff_count}/{len(a)} diff={diff_count}")
