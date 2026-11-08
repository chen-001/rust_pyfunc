from common import *
import pickle
np.set_printoptions(linewidth=220,precision=3,suppress=True)
D8=[20240205,20240329,20240813,20240930,20241008,20250407,20251127,20260311]
dirs=sorted(int(d) for d in os.listdir('/ssd_data/stock') if d.isdigit() and int(d)>=20220101)
seen=set();M=[]
for d in dirs:
    k=d//100
    if k not in seen: seen.add(k);M.append(d)
DATES=sorted(set(M)|set(D8))
def files(d,pre,n):
    fs=sorted(f[:6] for f in os.listdir(f'/ssd_data/stock/{d}/transaction') if f.startswith(pre) and f.endswith('_transaction.csv'))
    random.Random(11).shuffle(fs); return fs[:n]
def read(d,c):
    t=pd.read_csv(f'/ssd_data/stock/{d}/transaction/{c}_{d}_transaction.csv',usecols=['time','exchtime','localtime','turnover','flag'])
    return t[session_mask(t.time)]
def job(d):
    X=[read(d,c) for c in files(d,'6',300)]
    t=pd.concat([x for x in X if len(x)],ignore_index=True)
    ex=t.exchtime.values.astype(float); lt=t.localtime.values.astype(float); o=np.argsort(lt,kind='stable'); ex=ex[o];lt=lt[o]
    cs=cs_of(ex); cut=np.r_[0,np.flatnonzero(np.diff(lt)>40000)+1,len(lt)]
    mx=np.maximum.reduceat(cs,cut[:-1]); mn=np.minimum.reduceat(cs,cut[:-1]); n=np.diff(cut); L0=lt[cut[:-1]]
    big=n>=20
    hmax=np.bincount(mx[big]%25,minlength=25); hmin=np.bincount(mn[big]%25,minlength=25)
    Tstar=int(hmax.argmax())
    # 批的结束时间戳之间的间隔（10毫秒格）
    dT=np.diff(mx[big]); hd=np.bincount(np.clip(dT,0,60),minlength=61)
    # 成交相位
    x=t[t.flag.isin([66,83])&(t.turnover>0)]; c=np.bincount(cs_of(x.exchtime.values.astype(float))%25,minlength=25).astype(float); c/=c.mean()
    peak=int(c.argmax())
    # 数据到手的迟滞：批第一条本地时间 - 批内最晚交易所时间
    ex_end=(mx*1e4)  # cs -> 东八区微秒
    L_cn=(L0+8*3600e6)%86400e6
    hold=L_cn[big]-ex_end[big]
    # 深市对照：本地时间-交易所时间 的中位数（连续到达，等于时钟差+传输）
    Z=[read(d,c) for c in files(d,'00',100)]
    z=pd.concat([x for x in Z if len(x)],ignore_index=True); zl=(z.localtime.values.astype(float)-z.exchtime.values.astype(float))
    zdl=np.diff(np.sort(z.localtime.values.astype(float))); 
    return dict(date=d,nbatch=int(big.sum()),rows_batch=float(n[big].mean()),T=Tstar,share=float(hmax.max()/hmax.sum()),share3=float((hmax[(Tstar-1)%25]+hmax[Tstar]+hmax[(Tstar+1)%25])/hmax.sum()),
        dT25=float(hd[25]/hd.sum()),dT24_26=float(hd[24:27].sum()/hd.sum()),minspan=float(np.median((mx-mn)[big])),peak=peak,amp=float(c.max()/c.min()),lag=(peak-Tstar)%25,
        hold=float(np.median(hold)),hold_p10=float(np.percentile(hold,10)),hold_p90=float(np.percentile(hold,90)),sz_lag=float(np.median(zl)),sz_lag_p10=float(np.percentile(zl,10)),hmax=hmax.tolist(),curve=c.tolist())
if __name__=='__main__':
    print(len(DATES))
    with mp.Pool(45) as p: R=p.map(job,DATES,chunksize=1)
    df=pd.DataFrame(R); df.to_pickle('probe4.pkl')
    df['net_hold_ms']=(df.hold-df.sz_lag)/1000
    df['peak_after_T_ms']=df.lag*10
    pd.set_option('display.width',240,'display.max_rows',200,'display.float_format',lambda v:f'{v:.3f}')
    print(df[['date','nbatch','rows_batch','T','share','share3','dT25','dT24_26','peak','amp','peak_after_T_ms','hold','sz_lag','net_hold_ms']].to_string())
