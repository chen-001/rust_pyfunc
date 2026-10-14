import os, json
import numpy as np
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
meta = json.load(open(OUT + '/meta.json'))
names = np.array(meta['names'])
PREF = ['cnt_t005','cnt_t01','cnt_t02','cnt_t05','cnt_t1','cnt_t3','cnt_t5','cnt_t10','cnt_t30',
        'vol_t02','vol_t05','vol_t1','vol_t3','vol_t5','vol_t30']
F = len(names)
pre_arr = np.zeros(F, np.int8); met = np.empty(F, object)
for i, nm in enumerate(names):
    for k, p in enumerate(PREF):
        if nm.startswith(p + '_'): pre_arr[i] = k; met[i] = nm[len(p)+1:]; break
corr = np.load(OUT + '/corr.npy')
ov = np.load(OUT + '/overlap.npy')
nonfin = ~np.isfinite(corr)
np.fill_diagonal(nonfin, False)
frac = nonfin.sum(1) / F
print('每个因子与多少比例的其它因子算不出相关系数（分子分母为零）:')
print('  中位 %.4f, p90 %.4f, >0.5 的因子数 %d, >0.9 的 %d, 全部 %d' %
      (np.median(frac), np.percentile(frac, 90), (frac > 0.5).sum(), (frac > 0.9).sum(), (frac == 1.0).sum()))
degen = np.where(frac > 0.9)[0]
print('  退化因子例子:', [str(names[i]) for i in degen[:8]])
import collections
print('  按指标名归类（退化因子 top）:', collections.Counter([str(met[i]) for i in degen]).most_common(8))

bad = nonfin | (ov < 500)
corr[bad] = np.nan
np.fill_diagonal(corr, np.nan)
alive = np.load(OUT + '/alive_0.9900_abs.npy')
drop = ~alive
print()
print('== 0.99 阈值被丢弃 %d 列，按指标名统计 top 25 ==' % drop.sum())
c = collections.Counter([str(met[i]) for i in np.where(drop)[0]])
for m, n in c.most_common(25): print('   %-32s %d' % (m, n))
print('   不同指标名被删到 %d 个' % len(c))
print()
print('== 被丢弃最多的指标名在 701 个里占的比例 ==')
for m, n in c.most_common(12):
    tot = int((met == m).sum())
    print('   %-32s 删 %3d / 共 %3d' % (m, n, tot))
json.dump({'degen_frac_median': float(np.median(frac)), 'degen_gt09': int((frac > 0.9).sum()),
           'degen_all': int((frac == 1.0).sum()),
           'drop_by_metric': {m: int(n) for m, n in c.items()}},
          open(OUT + '/drop_by_metric.json', 'w'), ensure_ascii=False, indent=1)
d = json.load(open(OUT + '/analysis.json'))
print()
print('== 0.99 最大组 ==')
for g in d['greedy']['0.9900_abs'].get('largest_groups', [])[:8]:
    print('  size %d  代表 %s  组内 |corr| %.4f~%.4f' % (g['size'], g['rep'], g['min_abs'], g['max_abs']))
    print('     ', ' | '.join(g['members'][:8]))
