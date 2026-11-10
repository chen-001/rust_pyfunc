"""只读模拟：纯撤单因子沪市 20211105 前缺失时，回测引擎各道闸门的读数。不改任何回测代码。
闸门语义抄自 tail_v8_pipeline.rs / tail_v5_pipeline.rs / tail_v8_preflight.rs。"""
import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]: os.environ[k]="1"
import sys; import numpy as np; import pandas as pd
sys.path.insert(0,"/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as ddr
R=ddr.restricts(20160101,20241231); dates=R.index.values.astype(int)
sh=np.array([str(c).startswith(("60","68")) for c in R.columns]); r=R.values.astype(np.float32)
free=(r==0); print("dates",len(dates),dates[0],dates[-1],"stocks",r.shape[1])
start=20211105; miss=dates<start; print("缺沪市撤单的交易日",int(miss.sum()),"占比",round(float(miss.mean()),4))
fsh=(free&sh[None,:]).sum(1); fsz=(free&~sh[None,:]).sum(1)
print("沪市可交易占比 缺失期均值",round(float((fsh[miss]/(fsh[miss]+fsz[miss])).mean()),4),"缺失期沪市可交易只数均值",round(float(fsh[miss].mean()),1))
rng=np.random.default_rng(7); v=rng.random(r.shape,dtype=np.float32)
v[~free]=np.nan
v[np.ix_(miss,sh)]=np.nan
rk=pd.DataFrame(v).rank(axis=1,method="average").values.astype(np.float32)
before=np.where((fsh+fsz)>0,np.isfinite(rk).sum(1)/(fsh+fsz),np.nan)
print("填充前覆盖率(分母=当日可交易) 全样本均值",round(float(np.nanmean(before)),4),"缺失期",round(float(np.nanmean(before[miss])),4),"之后",round(float(np.nanmean(before[~miss])),4),"缺失期最低",round(float(np.nanmin(before[miss])),4))
f=rk.copy()
for t in range(f.shape[0]):
    n=int(np.isfinite(f[t]).sum())
    if n==0: continue
    f[t][(~np.isfinite(f[t]))&free[t]]=(n+1)/2.0
after=np.where((fsh+fsz)>0,(np.isfinite(f)&free).sum(1)/(fsh+fsz),np.nan)
print("填充后覆盖率 均值",round(float(np.nanmean(after)),4),"nan占比",round(float(1-np.nanmean(after)),6))
def maj(a):
    out=[]
    for row in a:
        x=row[np.isfinite(row)]
        if x.size==0: out.append(0); continue
        u,c=np.unique(x,return_counts=True); out.append(int(c.max()))
    return np.array(out)
m0=maj(f); print("众数计数 第0面 全样本均值",round(float(m0.mean()),1),"缺失期",round(float(m0[miss].mean()),1),"之后",round(float(m0[~miss].mean()),1),"门槛200")
for w in (5,20):
    mp=max(1,w//2); out=np.full(f.shape,np.nan,np.float32); s=np.zeros(f.shape[1]); c=np.zeros(f.shape[1],int)
    for t in range(f.shape[0]):
        x=np.nan_to_num(f[t],nan=0.0); ok=np.isfinite(f[t]); s+=x; c+=ok
        if t>=w:
            y=np.nan_to_num(f[t-w],nan=0.0); oy=np.isfinite(f[t-w]); s-=y; c-=oy
        out[t]=np.where(c>=mp,(s/np.maximum(c,1)).astype(np.float32),np.nan)
    m=maj(out); print(f"众数计数 滚动均值窗{w} 全样本均值",round(float(m.mean()),1))
# 回测起点推迟到沪市有撤单之后
late=dates>=start; print("仅 20211105 起：众数均值",round(float(m0[late].mean()),1),"覆盖",round(float(np.nanmean(before[late])),4),"天数",int(late.sum()))
# 深市-only 分母
print("若分母改为可计算域（缺失期只算深市）：覆盖",round(float(np.nanmean(np.where(fsz[miss]>0,(np.isfinite(rk[miss])&free[miss]).sum(1)/fsz[miss],np.nan))),4))
