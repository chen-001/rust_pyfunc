import re
src = open('/home/chenzongwei/rust_pyfunc/src/observable_order_metrics.rs').read().split('\n')
# DELETE_COLS
m = re.search(r'pub const DELETE_COLS: \[usize; 84\] = \[(.*?)\];', src and '\n'.join(src), re.S)
dele = [int(x) for x in re.findall(r'(\d+)', m.group(1))]
print('DELETE 列数', len(dele), '范围', min(dele), max(dele))
keep = [i for i in range(147) if i not in set(dele)]
print('保留列数', len(keep))
# 找函数体里的块注释与「小计」
start = next(i for i,l in enumerate(src) if 'fn compute_segment_features' in l) if any('fn compute_segment_features' in l for l in src) else None
print('函数起始行', start+1 if start is not None else '?')
blocks = []
off = 0
pending = []
for i in range(600, 890):
    l = src[i]
    s = l.strip()
    if s.startswith('//'):
        pending.append((i+1, s))
    mm = re.match(r'// 小计 (\d+)', s)
    if mm:
        n = int(mm.group(1))
        label = [t for _, t in pending[:-1] if t.startswith('// ===')]
        lab = [t for t in label if t]
        blocks.append((off, off+n, n, pending[-2][1] if len(pending) >= 2 else '', pending[-1][1]))
        off += n
        pending = []
print('解析出块数', len(blocks), '总列数', off)
import collections
full_drop = partial = full_keep = 0
detail = []
for lo, hi, n, lab, _ in blocks:
    d = [i for i in range(lo, hi) if i in set(dele)]
    k = n - len(d)
    if len(d) == n: full_drop += 1; cls='整块被删'
    elif len(d) == 0: full_keep += 1; cls='整块保留'
    else: partial += 1; cls='部分被删'
    detail.append((cls, lo, hi, n, len(d), lab[:60]))
print('整块被删 %d 块 / 部分被删 %d 块 / 整块保留 %d 块' % (full_drop, partial, full_keep))
print()
print('--- 整块被删的块（能直接不算）---')
for c, lo, hi, n, nd, lab in detail:
    if c == '整块被删': print('  列 %3d-%3d (%d 列) %s' % (lo, hi-1, n, lab))
print()
print('--- 部分被删的块（前 25 条）---')
cnt = 0
for c, lo, hi, n, nd, lab in detail:
    if c == '部分被删':
        print('  列 %3d-%3d 共 %2d 列，删 %2d 列 %s' % (lo, hi-1, n, nd, lab)); cnt += 1
        if cnt >= 25: break
