import json, numpy as np, collections
OUT='/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
meta=json.load(open(OUT+'/meta.json')); names=meta['names']
d2=json.load(open(OUT+'/zero_check_20241129.json'))
cols=d2['cols']; V2=np.load(OUT+'/zero_check_20241129.npy')
V1=np.load(OUT+'/vals.npy', mmap_mode='r')
n1={n:i for i,n in enumerate(names)}
def stat(col):
    f=np.isfinite(col); nf=int(f.sum()); nz=int((col[f]!=0).sum())
    u=int(np.unique(col[f]).size) if nf else 0
    return nf,nz,u
rows=[]
for k,c in enumerate(cols):
    a=stat(np.asarray(V1[:,n1[c]],dtype=np.float64)); b=stat(np.asarray(V2[:,k],dtype=np.float64))
    rows.append((c,a,b))
def cat(s):
    nf,nz,u=s
    if nf==0: return '全NaN'
    if nz==0: return '全0'
    if u==1: return '常数非零'
    return '有区分度'
c1=collections.Counter(cat(r[1]) for r in rows); c2=collections.Counter(cat(r[2]) for r in rows)
print('两天比对：%d 列（20241231 vs 20241129）' % len(rows))
print('20241231：', dict(c1))
print('20241129：', dict(c2))
print()
still0=[r for r in rows if cat(r[1])=='全0' and cat(r[2])=='全0']
now0 =[r for r in rows if cat(r[1])=='全0' and cat(r[2])=='有区分度']
const=[r for r in rows if cat(r[1])=='全0' and cat(r[2])=='常数非零']
print('第一天全 0 的 %d 列里：' % sum(1 for r in rows if cat(r[1])=='全0'))
print('  第二天还是全 0：%d 列' % len(still0))
print('  第二天变成有区分度：%d 列' % len(now0))
print('  第二天变成常数非零：%d 列' % len(const))
g=lambda c: 'degen' in c
print()
print('按名字里带不带 degen 拆：')
for tag,sel in [('带 degen', [r for r in rows if g(r[0])]), ('不带 degen', [r for r in rows if not g(r[0])])]:
    cc=collections.Counter(cat(r[2]) for r in sel)
    print('  %s：%d 列，20241129 分类 %s' % (tag, len(sel), dict(cc)))
print()
print('--- 第二天不再全 0 的列（这些是「一天全 0 不等于一直全 0」的证据）---')
for c,a,b in now0[:25]:
    print('  %-46s 1231: 非NaN=%d 非零=%d 唯一=%d | 1129: 非NaN=%d 非零=%d 唯一=%d' % (c,a[0],a[1],a[2],b[0],b[1],b[2]))
print()
met=collections.Counter('_'.join(r[0].split('_')[2:]) for r in still0)
print('--- 两天都全 0 的指标名（前 15）---')
for k,v in met.most_common(15): print('  %-34s %d 档' % (k,v))
json.dump({'still_zero':[r[0] for r in still0],'became_normal':[r[0] for r in now0],'became_const':[r[0] for r in const]},
          open(OUT+'/zero_two_day.json','w'), ensure_ascii=False)
