# -*- coding: utf-8 -*-
import sys, pickle; sys.path.insert(0,'..')
from common import *
from load import load_day, DATES as D8
import probe2
import matplotlib; matplotlib.use('Agg')
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
np.set_printoptions(linewidth=220,precision=3,suppress=True)
N={}
df=pd.read_pickle('probe4.pkl').sort_values('date').reset_index(drop=True)
def lagdeg(a): return 2*np.pi*(a%25)/25
def Rstat(T,P): return np.abs(np.exp(1j*lagdeg(P-T)).mean())
def era(a,b): return df[(df.date>=a)&(df.date<=b)]
rng=np.random.default_rng(3)
for name,(a,b) in {'2023-06至2024-09':(20230601,20240902),'2024-12至2026-09':(20241202,20260901)}.items():
    e=era(a,b); T=e['T'].values; P=e.peak.values; lag=((P-T)%25)*10
    R=Rstat(T,P); perm=[Rstat(rng.permutation(T),P) for _ in range(20000)]
    N[name]=dict(天数=int(len(e)),T中位ms=float(np.median(T)*10),T最小ms=int(T.min()*10),T最大ms=int(T.max()*10),高峰中位ms=float(np.median(P)*10),间隔中位ms=float(np.median(lag)),间隔10分位=float(np.percentile(lag,10)),间隔90分位=float(np.percentile(lag,90)),同步度R=float(R),打乱日期R的99分位=float(np.percentile(perm,99)),打乱后不低于真实的比例=float(np.mean(np.array(perm)>=R)),间隔明细=[int(x) for x in lag])
print(json.dumps(N,ensure_ascii=False,indent=0))
N['最近两月']=df[df.date>=20261001][['date','share','dT24_26','T','peak']].to_dict('records')
print(N['最近两月'])
# 上一批有无成交：用正确的边界
B={int(r.date):(int(r['T'])+1)%25 for _,r in df[df.date.isin(D8)].iterrows()}
jobs=[(d,c,B[d]) for d in D8 for c in sample(d,'6',200)]
R={}
with mp.Pool(50) as p:
    for d,o in p.imap_unordered(probe2.p2,jobs,chunksize=2):
        for k,(occ,e,ca) in o.items():
            A=R.setdefault(k,[0,0,0]); A[0]+=occ;A[1]=A[1]+e;A[2]=A[2]+ca
React={}
for k,(occ,e,ca) in R.items():
    y=e/occ; z=ca/occ; React[k]=dict(窗数=occ,每窗主动单=float(y.sum()),每窗撤单=float(z.sum()),主动单形状=(y/y.mean()).tolist(),撤单形状=(z/z.mean()).tolist(),主动单最高比最低=float(y.max()/y.min()),最高位置ms=int(y.argmax())*10,撤单最高比最低=float(z.max()/z.min()))
    print(k,occ,'ev per win',round(y.sum(),3),'ev shape max/min',round(y.max()/y.min(),2),'at',int(y.argmax())*10,'ca',round(z.max()/z.min(),2)); print('  ',(y/y.mean()).round(2))
N['上一批有无成交']=React
# 深市：非快照秒 vs 快照秒；每只股票日是否 4Hz
Ms,As={},{}
for d in D8: Ms[d],As[d]=load_day(d)
mk=np.concatenate([[c[0] for c in Ms[d]['codes']] for d in D8]); clk=np.vstack([As[d]['clk'] for d in D8]).astype(float); ev=np.vstack([As[d]['ev'] for d in D8]).astype(float)
for m,nm in zip('063',['深市主板','创业板']) if False else [('0','深市主板'),('3','创业板'),('6','沪市')]:
    c=clk[mk==m].sum(0); s0=c[:100]; s1=c[100:200]; s2=c[200:300]; f=lambda x:(x/x.mean())
    ns=(s1+s2); 
    N[nm+'_按3秒周期']=dict(整3秒那一秒_最高ms=int(f(s0).argmax())*10,整3秒那一秒_最高=float(f(s0).max()),其余两秒_最高ms=int(f(ns).argmax())*10,其余两秒_最高=float(f(ns).max()),其余两秒_最低ms=int(f(ns).argmin())*10,其余两秒_最低=float(f(ns).min()),其余两秒_前100毫秒均值=float(f(ns)[:10].mean()),其余两秒_后100毫秒均值=float(f(ns)[90:].mean()),其余两秒_150ms前后=[float(f(ns)[14]),float(f(ns)[15])],其余两秒曲线=f(ns).tolist())
    print(nm,{k:v for k,v in N[nm+'_按3秒周期'].items() if k!='其余两秒曲线'})
Fq={}
for m,nm in [('6','沪市'),('0','深市主板'),('3','创业板')]:
    a=ev[mk==m]; n=a.sum(1); a=a[n>=np.quantile(n,.6)]; x=a/a.mean(1,keepdims=True); f=np.abs(np.fft.rfft(x-1,axis=1)); top=f[:,1:].argmax(1)+1
    Fq[nm]=dict(最活跃四成_股票日数=int(len(a)),最强周期是每秒4次的比例=float((top==4).mean()),最强周期是每秒1次的比例=float((top==1).mean()),每秒4次平均振幅占比=float((f[:,4]/f[:,1:].sum(1)).mean())); print(nm,Fq[nm])
N['逐股周期']=Fq
# 到达间隔图的数据
def gaps(a):
    d,c=a
    t=pd.read_csv(f'/ssd_data/stock/{d}/transaction/{c}_{d}_transaction.csv',usecols=['time','localtime']); t=t[session_mask(t.time)]
    dl=np.diff(t.localtime.values.astype(float)); return c[0],np.bincount(np.clip((np.maximum(dl,0)/10000).astype(int),0,60),minlength=61)
A={}
for d in [20240329,20251127]:
  for m in '60':
    A[(d,m)]=0
J=[(d,c) for d in [20240329,20251127] for p in '60' for c in sample(d,p,150)]
def gapj(a): return (a[0],)+gaps(a)
with mp.Pool(50) as p:
    for d,m,g in p.imap_unordered(gapj,J,chunksize=2): A[(d,m)]=A[(d,m)]+g
N['到达间隔']={f'{d}_{m}':np.asarray(A[(d,m)]).tolist() for d,m in A}
for k,v in A.items(): v=np.asarray(v).astype(float); print(k,'<10ms',round(v[0]/v.sum(),3),'240-259ms',round((v[24]+v[25])/v.sum(),3), '10-239 ms',round(v[1:24].sum()/v.sum(),3))
json.dump(N,open('numbers.json','w'),ensure_ascii=False,default=float,indent=0)
