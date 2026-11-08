from common import *
np.set_printoptions(linewidth=220,precision=2,suppress=True)
DAYS=[20240329,20240813,20241008]
def job(a):
    d,c=a
    t=trades(d,c); o=orders(d,c)
    r={}
    tr=t[t.flag.isin([66,83])&(t.turnover>0)&session_mask(t.time)].copy()
    ca=t[(t.flag==32)&session_mask(t.time)]
    cell=lambda x:(cs_of(x.exchtime.values)%100)
    # 主动单
    tr['cs']=cs_of(tr.exchtime.values); tr['oid']=np.where(tr.flag==66,tr.bid_order,tr.ask_order).astype(np.int64)
    ev=tr.groupby(['cs','flag','oid'],sort=False).agg(v=('volume','sum'),n=('volume','size')).reset_index()
    r['ev']=np.bincount(ev.cs%100,minlength=100)
    r['ev_small']=np.bincount(ev.cs[ev.v<=500]%100,minlength=100); r['ev_big']=np.bincount(ev.cs[ev.v>=5000]%100,minlength=100)
    r['ev_buy']=np.bincount(ev.cs[ev.flag==66]%100,minlength=100); r['ev_sell']=np.bincount(ev.cs[ev.flag==83]%100,minlength=100)
    r['cancel']=np.bincount(cell(ca),minlength=100)
    oo=o[session_mask(o.time)].copy(); oo['cs']=cs_of(oo.exchtime.values)
    aggr=set(ev.oid.values)
    isag=oo.order.astype(np.int64).isin(aggr).values
    r['ord_all']=np.bincount(oo.cs%100,minlength=100); r['ord_pass']=np.bincount(oo.cs[~isag]%100,minlength=100); r['ord_aggr']=np.bincount(oo.cs[isag]%100,minlength=100); r['n_ord']=len(oo); r['n_match']=int(isag.sum()); r['n_ev']=len(ev)
    # localtime：相邻记录的本地时间差，是否成批到达
    lt=t[session_mask(t.time)].localtime.values.astype(float)
    dl=np.diff(lt); r['lt_same1ms']=float((np.abs(dl)<1000).mean()); r['lat_med']=float(np.median(t.localtime.values[session_mask(t.time)].astype(float)-t.exchtime.values[session_mask(t.time)].astype(float)))
    r['lt250']=np.bincount((np.floor((lt%250000)/5000)).astype(int),minlength=50)
    r['gap_ms_lt']=np.bincount(np.clip((np.maximum(dl,0)/10000).astype(int),0,99),minlength=100)
    return d,c[0],r
jobs=[(d,c) for d in DAYS for p,n in [('6',150),('0',150)] for c in sample(d,p,n)]
if __name__=='__main__':
    R={}
    with mp.Pool(50) as p:
        for d,m,r in p.imap_unordered(job,jobs,chunksize=2):
            A=R.setdefault((d,m),{})
            for k,v in r.items():
                A[k]=A.get(k,0)+(np.asarray(v) if not isinstance(v,float) else v)
    for key in sorted(R):
        d,m=key; A=R[key]; nm={'6':'SH','0':'SZ'}[m]; print('=====',d,nm,'orders',A['n_ord'],'matched aggressor orders',A['n_match'],'events',A['n_ev'])
        for k in ['ev','ev_small','ev_big','ev_buy','ev_sell','cancel','ord_all','ord_pass','ord_aggr']:
            x=A[k].astype(float); x=x/x.mean(); f=np.abs(np.fft.rfft(x-x.mean()))
            print(f'{k:9s} n={int(A[k].sum()):>9d} top freq {np.argsort(-f[1:])[:3]+1} max {x.argmax()*10} {x.max():.2f} min {x.argmin()*10} {x.min():.2f} | peaks @',[int(x[i:i+25].argmax()+i)*10 for i in (0,25,50,75)])
        print(' same-localtime(<1ms) share',A['lt_same1ms']/150,' median local-exch us',A['lat_med']/150)
        x=A['lt250'].astype(float); print(' localtime mod 250ms (5ms bins, x/mean):',(x/x.mean())[:50])
        g=A['gap_ms_lt'].astype(float); print(' localtime gap hist(10ms bins) share:',(g/g.sum())[:30])
