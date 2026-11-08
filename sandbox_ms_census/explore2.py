import numpy as np, json
from load import *
np.set_printoptions(linewidth=220, precision=2, suppress=True)
Ms={};As={}
for d in DATES:
    Ms[d],As[d]=load_day(d)
mks={d:np.array([c[0] for c in Ms[d]['codes']]) for d in DATES}
print('--- SH ev intensity by day (pooled), every cell peaks; autocorr at lag 25')
for m in '603':
  print('market',m)
  for d in DATES:
    x=As[d]['ev'][mks[d]==m].sum(0).astype(float); x/=x.mean()
    y=x-np.convolve(np.r_[x,x,x],np.ones(25)/25,'same')[100:200]
    f=np.abs(np.fft.rfft(x-x.mean()));
    top=np.argsort(-f[1:])[:4]+1
    print(d,'max cell',x.argmax()*10,round(x.max(),3),'min',x.argmin()*10,round(x.min(),3),'top fft freq(cycles/s)',top,np.round(f[top]/f[1:].sum(),3),'peaks',[int(i)*10 for i in np.argsort(-y)[:5]])
print('--- median stock-day intensity vs pooled, market 6 (all days)')
ev=np.vstack([As[d]['ev'] for d in DATES]).astype(float);mk=np.concatenate([mks[d] for d in DATES])
for m in '603':
    i=mk==m; a=ev[i]; rate=a/a.mean(1,keepdims=True); med=np.median(rate,0); pool=a.sum(0)/a.sum()*100
    print(m,'corr med/pool',np.corrcoef(med,pool)[0,1]); print(med)
    # high-activity vs low activity
    n=a.sum(1); q=np.quantile(n,[.2,.8]); lo=np.median(rate[n<=q[0]],0); hi=np.median(rate[n>=q[1]],0); print(' lo-hi corr',np.corrcoef(lo,hi)[0,1])
print('--- snapshot phases by market/date')
for d in DATES:
    S=As[d]['scal']; mkx=mks[d]
    out=[]
    for m in '603':
        i=mkx==m; ph=S[i,5:8]; sh=ph.max(1)/np.maximum(ph.sum(1),1); dom=ph.argmax(1)
        out.append((m,int(i.sum()),np.bincount(dom,minlength=3).tolist(),round(float(np.median(sh)),3)))
    print(d,out)
