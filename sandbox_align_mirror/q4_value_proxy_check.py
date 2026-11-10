"""第二轮 Q4：用真实因子值，检验“只靠 IC 序列估算合成效果”到底准不准（只读因子库）。
真实合成 = 两个因子各自排名后等权相加再排名。"""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
import sys
import numpy as np, pandas as pd, rust_pyfunc as rp
sys.path.insert(0,"/home/chenzongwei"); sys.path.insert(0,"/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as ddr
from design_whatever.tlq_gate import make_store_reader
store=sys.argv[1]; NF=int(sys.argv[2]) if len(sys.argv)>2 else 60
SEED=20261109; rng=np.random.default_rng(SEED)
info=rp.factor_store_v5_info(store); names=list(info["factor_names"]); t=rp.factor_store_v5_template(store)
dates=np.asarray(t["dates"]).astype(np.int64); stocks=[str(x) for x in t["stocks"]]
print("因子",len(names),"日期",len(dates),dates[0],dates[-1],"股票",len(stocks))
pick=rng.choice(len(names),NF,replace=False); reader=make_store_reader(store,dates,stocks)
c=ddr.close(start=int(dates[0]),end=int(dates[-1])+20000 if False else 20270101)
codes=[x.split(".")[0] for x in stocks]
c=c.reindex(columns=codes)
print("收盘价对上股票数",int(c.notna().any().sum()),"示例代码",stocks[:2])
idx={int(d):i for i,d in enumerate(c.index)}
vals=[]; kept=[]
for p in pick:
    v=reader(names[p]); v=np.asarray(v,dtype=np.float32)
    if np.isfinite(v).mean()>0.2: vals.append(v); kept.append(names[p])
print("读入",len(kept))
days=np.arange(10,len(dates)-10,max(1,(len(dates)-20)//120))
C=c.to_numpy(float)
def rank(x):
    o=np.full(x.shape,np.nan); m=np.isfinite(x); r=x[m].argsort().argsort().astype(float); o[m]=(r+.5)/m.sum(); return o
def corr(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<300: return np.nan
    a,b=a[m],b[m]
    return np.corrcoef(a,b)[0,1]
R=[];ICs=[]
for t_ in days:
    d=int(dates[t_])
    if d not in idx or idx[d]+5>=len(C): R.append(None); ICs.append(None); continue
    r=C[idx[d]+5]/C[idx[d]]-1; rr=rank(r)
    rs=[rank(v[t_]) for v in vals]
    R.append((rs,rr)); ICs.append([corr(x,rr) for x in rs])
ok=[i for i,x in enumerate(R) if x is not None]; IC=np.array([ICs[i] for i in ok],float); print("样本日",len(ok),IC.shape)
s=np.sign(np.nanmean(IC,0)); s[s==0]=1; IC*=s
pairs=[(i,j) for i in range(len(kept)) for j in range(i+1,len(kept))]
pairs=[pairs[k] for k in rng.choice(len(pairs),min(400,len(pairs)),replace=False)]
rows=[]
for i,j in pairs:
    act=[];rho=[]
    for k,idx_ok in enumerate(ok):
        rs,rr=R[idx_ok]
        a=rs[i]*s[i]; b=rs[j]*s[j]
        x=np.where(np.isfinite(a)&np.isfinite(b),a+b,np.nan)
        act.append(corr(rank(x),rr)); rho.append(corr(rank(a),rank(b)))
    act=np.array(act);rho=np.array(rho);icsum=IC[:,i]+IC[:,j]
    m=np.isfinite(act)&np.isfinite(rho)&np.isfinite(icsum)
    if m.sum()<30: continue
    a=act[m];r_=rho[m];ii=IC[m,i];jj=IC[m,j]; pxr=(ii+jj)/np.sqrt(2*(1+r_)); px0=(ii+jj)/np.sqrt(2)
    ric=np.corrcoef(ii,jj)[0,1]
    rows.append(dict(i=i,j=j,act=a.mean(),px_rho=pxr.mean(),px_0=px0.mean(),act_ir=a.mean()/a.std(),ic_ir=(ii+jj).mean()/(ii+jj).std(),rho_x=r_.mean(),rho_ic=ric,best1=max(ii.mean(),jj.mean()),gain_act=a.mean()-max(ii.mean(),jj.mean())))
D=pd.DataFrame(rows); D.to_csv("results/q4_value_proxy.csv",index=False)
print("配对",len(D))
print("真实合成IC vs 带截面相关的估算: 相关",round(D.act.corr(D.px_rho),3),"平均误差",round((D.act-D.px_rho).mean(),4),"误差绝对值中位",round((D.act-D.px_rho).abs().median(),4))
print("真实合成IC vs 不知道截面相关(取0)的估算: 相关",round(D.act.corr(D.px_0),3),"平均误差",round((D.act-D.px_0).mean(),4),"误差绝对值中位",round((D.act-D.px_0).abs().median(),4))
print("真实合成IR vs 只用IC序列的IR: 相关",round(D.act_ir.corr(D.ic_ir),3),"秩相关",round(D.act_ir.corr(D.ic_ir,method='spearman'),3))
print("真实合成增益(真实-较强单因子) vs IC序列估算增益: 秩相关",round(D.gain_act.rank().corr((D.px_0-D.best1).rank()),3))
print("IC序列相关 vs 截面值相关: 相关",round(D.rho_ic.corr(D.rho_x),3),"差绝对值中位",round((D.rho_ic-D.rho_x).abs().median(),3),"截面相关均值",round(D.rho_x.mean(),3),"IC序列相关均值",round(D.rho_ic.mean(),3))
print("真实合成IC > 较强单因子IC 的占比",round((D.gain_act>0).mean(),3),"截面相关<0.3 的对中占比",round((D[D.rho_x<0.3].gain_act>0).mean(),3),"其中配对数",int((D.rho_x<0.3).sum()))
