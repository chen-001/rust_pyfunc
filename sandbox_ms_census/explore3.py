import numpy as np, json
from load import *
np.set_printoptions(linewidth=220, precision=3, suppress=True)
Ms={};As={}
for d in DATES:
    Ms[d],As[d]=load_day(d)
g=lambda k:np.vstack([As[d][k] for d in DATES]).astype(float)
mk=np.concatenate([[c[0] for c in Ms[d]['codes']] for d in DATES])
snap,occ,clk,S=g('snap'),g('occ'),g('clk'),g('scal')
print('--- snapshot aligned pooled intensity (per 100ms avg over 300 bins)')
for m in '603':
    i=mk==m; r=snap[i].sum(0)/occ[i].sum(0); r/=r.mean(); print(m,r.reshape(30,10).mean(1)); print(' first 30 cells',r[:30])
print('--- clock 3s, 100ms avg')
dom=S[:,5:8].argmax(1)
for m in '603':
    i=mk==m; r=clk[i].sum(0); r=r/r.mean(); print(m,r.reshape(30,10).mean(1))
print('--- SH by snapshot phase: clock 3s, 1s avg, and per-300ms')
for p in range(3):
    i=(mk=='6')&(dom==p); r=clk[i].sum(0); r/=r.mean(); print(p,int(i.sum()),'sec avg',r.reshape(3,100).mean(1),r.reshape(30,10).mean(1)[:12])
print('--- same SH, aligned to own snapshot, by phase'); 
for p in range(3):
    i=(mk=='6')&(dom==p); r=snap[i].sum(0)/occ[i].sum(0); r/=r.mean(); print(p,r.reshape(30,10).mean(1)[:15])
gap,shuf=g('gap'),g('shuf')
print('--- gaps pooled real / shuf, all');
for m in '603':
    i=mk==m; a=gap[i].sum(0); b=shuf[i].sum(0); ratio=a/b
    print(m,'n',a.sum(),b.sum(),'share<50ms real',a[:5].sum()/a.sum(),'shuf',b[:5].sum()/b.sum());print(' ratio first 30',ratio[:30]);
    for k in [25,50,100,200,300,400,499]: print('  k',k*10,'ms: ratio neighbors',ratio[max(0,k-2):k+3])
    # spike score: ratio over local median
    med=np.array([np.median(ratio[max(0,j-5):j+6]) for j in range(500)]); sc=ratio/med; top=np.argsort(-sc[3:])[:12]+3; print(' top spikes (ms, spike ratio)',[(int(j)*10,round(float(sc[j]),3)) for j in top])
print('--- similarity');
pair,sd,sdv,bday,bmin=g('pair'),g('sd'),g('sdv'),g('bday'),g('bmin')
e=Ms[DATES[0]]['gap_edges'];print(e)
for m in ['6','0','3']:
    i=mk==m;P=pair[i].sum(0);print(m,'pairs',P);print(' sdv',sdv[i].sum(0)/P);print(' base_day',bday[i].sum(0)/P);print(' base_min',bmin[i].sum(0)/P);print(' sd',sd[i].sum(0)/P)
print('--- cross')
for d in DATES:
    c=Ms[d]['cross'];r=np.array(c['real_ex']);f=np.array(c['fake_ex'])
    top=c['top'][:8]; lab=[(t//100,t%100,'B' if dd==0 else 'S',a,b) for t,dd,a,b in top]
    nz=np.flatnonzero(r); nf=np.flatnonzero(f)
    print(d,'real max',nz[-1],'fake max avg',np.mean(c['fake_max_ex']),'max of fakes',max(c['fake_max_ex']));print('  counts >=threshold real/fake',[(th,int(r[th:].sum()),round(float(f[th:].sum()),1)) for th in [20,50,100,200,500]]);print('  top ticks (secIdx,cell,dir,sh,sz):',lab)
