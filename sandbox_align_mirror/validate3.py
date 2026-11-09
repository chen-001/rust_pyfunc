"""方向 3 补充：最佳候选 buyfirst/signed 相对朴素前后段差的两种增量（只读 Rust 产物）。"""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
import json,sys
import numpy as np,pandas as pd
sys.path.insert(0,"/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as ddr
def load(p):
    d=json.load(open(p)); return pd.DataFrame(np.array(d["vals"],dtype=float),index=d["codes"],columns=d["names"])
def rc(a,b):
    x=pd.concat([a,b],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    return x.iloc[:,0].rank().corr(x.iloc[:,1].rank()) if len(x)>=50 else np.nan
def neu(y,X):
    df=pd.concat([y.rename("y"),X],axis=1).replace([np.inf,-np.inf],np.nan).dropna().rank()
    M=np.column_stack([np.ones(len(df)),df.drop(columns="y").values]);b,*_=np.linalg.lstsq(M,df["y"].values,rcond=None)
    return pd.Series(df["y"].values-M@b,index=df.index)
A=load("results/mirror_20241231.json");B=load("results/mirror_20241230.json")
r=ddr.ret(start=20241220,end=20250110);r1=r.loc[20250102].reindex(A.index);r0=r.loc[20241231].reindex(B.index)
for c in ["mir_int_vo_all_buyfirst","mir_int_vo_all_signed","mir_tempo_vo_all_buyfirst","mir_int_od_all_signed","mir_int_od_tail_npeaks","mir_jump_od_tail_npeaks"]:
    w=c.split("_")[3]; fb=f"naive_fbdiff_{w}"; ac=f"naive_anticorr_{w}"
    x=pd.concat([A[c],A[fb]],axis=1,keys=["a","b"]).dropna()
    bins=pd.qcut(x.b,20,labels=False,duplicates="drop"); ratio=(x.a-x.groupby(bins).a.transform("mean")).std()/x.a.std()
    n1=neu(A[c],A[[fb,ac]]);n0=neu(B[c],B[[fb,ac]])
    print(f"{c:28s} cov={int(A[c].notna().sum())} stab={rc(A[c],B[c]):+.3f} 对fbdiff秩相关={rc(A[c],A[fb]):+.3f} 对anticorr={rc(A[c],A[ac]):+.3f} 数值残差比={ratio:.3f} IC={rc(A[c],r1):+.3f}/{rc(B[c],r0):+.3f} 去朴素后IC={rc(n1,r1):+.3f}/{rc(n0,r0):+.3f}")
for w in ("all","pm","tail"):
    f=f"naive_fbdiff_{w}";print(f,"IC",round(rc(A[f],r1),3),round(rc(B[f],r0),3))
print("市场当日截面收益均值",round(r1.mean(),4),round(r0.mean(),4))
