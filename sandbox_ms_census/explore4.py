import numpy as np
from load import *
np.set_printoptions(linewidth=220, precision=3, suppress=True)
Ms={};As={}
for d in DATES: Ms[d],As[d]=load_day(d)
ths=[10,20,30,50,75,100,150,200,300]
for d in DATES:
    c=Ms[d]['cross'];r=np.array(c['real_ex']);f=np.array(c['fake_ex'])
    top=np.array(c['top'][:5000]); tot=top[:,2]+top[:,3]
    print(d,'real max',np.flatnonzero(r)[-1],'fake max avg',np.mean(c['fake_max_ex']),'max',max(c['fake_max_ex']),'ratio>=th',[(t,int(r[t:].sum()),round(float(f[t:].sum()),1)) for t in ths])
    print('  sh real max',np.flatnonzero(c['real_sh'])[-1],'fake',np.flatnonzero(c['fake_sh'])[-1],'sz',np.flatnonzero(c['real_sz'])[-1],np.flatnonzero(c['fake_sz'])[-1])
    t100=top[:100]; idx=t100[:,0]
    print('  top100: sec%60==0',np.mean(idx//100%60==0),'cells',np.bincount((idx%100)//10,minlength=10),'buy share',np.mean(t100[:,1]==0),'sh share of stocks',t100[:,2].sum()/(t100[:,2]+t100[:,3]).sum(),'am',np.mean(idx<720000))
    print('  top5',[(int(a)//100//60,int(a)//100%60,int(a)%100,int(b),int(c1),int(c2)) for a,b,c1,c2 in top[:5]])
# hour rhythm of top1000
all_top=np.vstack([np.array(Ms[d]['cross']['top'][:1000]) for d in DATES]); i=all_top[:,0]; m=(i//100)%60
print('top1000x8 sec-of-minute hist',np.bincount(m,minlength=60));print('cell hist/10',np.bincount((i%100)//10,minlength=10))
# 10ms hole per day
for d in DATES:
    mk=np.array([c[0] for c in Ms[d]['codes']])
    o=[]
    for m in '603':
        a=As[d]['gap'][mk==m].sum(0).astype(float);b=As[d]['shuf'][mk==m].sum(0).astype(float); o.append((m,np.round(a[:4]/b[:4],2).tolist(),np.round(a[300]/b[300],2)))
    print(d,o)
