import numpy as np, json
from load import *
np.set_printoptions(linewidth=200, precision=3, suppress=True)
Ms={};As={}
for d in DATES:
    Ms[d],As[d]=load_day(d)
S=np.vstack([As[d]['scal'] for d in DATES]); print('stock-days',len(S),'rec',S[:,0].sum(),'ev',S[:,1].sum(),'rec/ev',S[:,0].sum()/S[:,1].sum(),'ticks',S[:,2].sum(),'multi-order tick share',S[:,3].sum()/S[:,2].sum(),'unsorted',S[:,8].sum(),'edge',S[:,9].sum())
# recon
D=daily()
r=np.concatenate([As[d]['amt_all']/1000/D['amt'].loc[d,Ms[d]['codes']].values for d in DATES]); print('recon quantiles',np.nanquantile(r,[0,.01,.05,.5,.95,.99,1]), 'off>2%',np.mean(np.abs(r-1)>.02))
mk=np.concatenate([[c[0] for c in Ms[d]['codes']] for d in DATES])
g=lambda k:np.vstack([As[d][k] for d in DATES]).astype(float)
rec=g('rec');ev=g('ev');amt=g('amt100');ev0=g('ev0');evx=g('evx')
print('--- pooled rec share*100 per cell');p=rec.sum(0)/rec.sum()*100;print(p);print('max',p.argmax(),p.max(),'min',p.argmin(),p.min(),'ratio',p.max()/p.min())
pa=amt.sum(0)/amt.sum()*100;print('amt max',pa.argmax(),pa.max(),'min',pa.argmin(),pa.min()); print(pa)
print('per-day argmax/argmin of rec');
for d in DATES:
    x=As[d]['rec'].sum(0).astype(float);x=x/x.mean();print(d,x.argmax()*10,round(x.max(),3),x.argmin()*10,round(x.min(),3),'top5',np.argsort(-x)[:5]*10,'bot5',np.argsort(x)[:5]*10)
print('--- by market pooled ev intensity')
for m in '063':
    x=ev[mk==m].sum(0);print(m,(x/x.mean()))
print('--- second0 vs other: median stock-day intensity')
for m in '063':
    i=mk==m; b=(ev[i].sum(1,keepdims=True)/100/14220)
    i0=ev0[i]/237/b; io=(ev[i]-ev0[i])/(14220-237)/b
    print(m,'ev0 pooled',(ev0[i].sum(0)/237/(ev[i].sum()/1422000))[:30]);print(m,'ev0 tail',(ev0[i].sum(0)/237/(ev[i].sum()/1422000))[30:]);print(m,'oth',((ev[i]-ev0[i]).sum(0)/(14220-237)/(ev[i].sum()/1422000))[:15])
