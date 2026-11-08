# -*- coding: utf-8 -*-
"""新图六的例子：2024-03-29 当天最挤的那 10 毫秒及其前后 150 毫秒。"""
import sys; sys.path.insert(0,'..')
from common import *
from load import load_day
d=20240329; m,_=load_day(d); top=m['cross']['top'][0]; idx,dr,_sh,_sz=top; W=15
secs=idx//100; t0=(93000000 if idx<720000 else 130000000)
print('top',top)
def hhmmss(i):
    base=9*3600+30*60 if i<720000 else 13*3600
    s=base+(i if i<720000 else i-720000)//100
    return f'{s//3600:02d}:{s//60%60:02d}:{s%60:02d}.{i%100:02d}0'
print(hhmmss(idx))
fl=66 if dr==0 else 83
def job(c):
    t=pd.read_csv(f'/ssd_data/stock/{d}/transaction/{c}_{d}_transaction.csv',usecols=['exchtime','flag','turnover'])
    t=t[(t.flag==fl)&(t.turnover>0)]; cs=cs_of(t.exchtime.values.astype(float)); i=np.where(cs<4140000,cs-3420000,cs-4680000+720000)
    x=i[(i>=idx-W)&(i<=idx+W)]-(idx-W); return np.bincount(np.unique(x),minlength=2*W+1)
codes=m['codes']
with mp.Pool(50) as p: tot=sum(p.imap_unordered(job,codes,chunksize=20))
hist=np.array(m['cross']['real_ex']); cum=np.cumsum(hist); med=int(np.searchsorted(cum,cum[-1]/2)); nz=hist[1:]; nzc=np.cumsum(nz); medn=int(np.searchsorted(nzc,nzc[-1]/2))+1
json.dump(dict(date=d,dir=int(dr),time=hhmmss(idx),n_codes=len(codes),count=tot.tolist(),median_all=med,median_nonzero=medn),open('fig6_example.json','w'),ensure_ascii=False)
print(tot.tolist(),med,medn)
