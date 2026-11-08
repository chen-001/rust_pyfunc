from common import *
np.set_printoptions(linewidth=220,precision=3,suppress=True)
dirs=sorted([d for d in os.listdir('/ssd_data/stock') if d.isdigit()])
# 每季度取第一个交易日（2015 起），2023 起每月
pick=[]; seen=set()
for d in dirs:
    y,m=int(d[:4]),int(d[4:6]); key=(y,m) if y>=2023 else (y,(m-1)//3)
    if key not in seen: seen.add(key); pick.append(int(d))
def job(a):
    d,c=a
    p=f'/ssd_data/stock/{d}/transaction/{c}_{d}_transaction.csv'
    try:
        t=pd.read_csv(p,usecols=['time','exchtime','localtime','turnover','flag'])
    except Exception: return None
    t=t[session_mask(t.time)]; 
    if len(t)<200: return None
    x=t[t.flag.isin([66,83])&(t.turnover>0)]
    cell=np.bincount(cs_of(x.exchtime.values.astype(float))%100,minlength=100)
    lt=t.localtime.values.astype(float); dl=np.diff(lt); 
    g=np.bincount(np.clip((np.maximum(dl,0)/10000).astype(int),0,99),minlength=100)
    # 按 250ms 分批的话，24/25 格的本地间隔很多
    return d,c[0],cell,g
if __name__=='__main__':
    jobs=[]
    for d in pick:
        fs=os.listdir(f'/ssd_data/stock/{d}/transaction')
        for pre in ['60','00']:
            cs=sorted(f[:6] for f in fs if f.startswith(pre) and f.endswith('_transaction.csv'))
            random.Random(9).shuffle(cs); jobs+=[(d,c) for c in cs[:80]]
    print('dates',len(pick),'jobs',len(jobs))
    R={}
    with mp.Pool(50) as p:
        for r in p.imap_unordered(job,jobs,chunksize=4):
            if r is None: continue
            d,m,c,g=r; A=R.setdefault((d,m),[0,0]); A[0]=A[0]+c;A[1]=A[1]+g
    rows=[]
    for d in pick:
        for m in '60':
            if (d,m) not in R: continue
            c,g=R[(d,m)]; x=c.astype(float); x=x/x.mean(); f=np.fft.rfft(x-x.mean()); a=np.abs(f); 
            ph=(-np.angle(f[4])/(2*np.pi)*25)%25*10  # 4Hz 峰的相位(毫秒,0~250)
            rows.append(dict(date=d,mk=m,topf=int(a[1:].argmax())+1,amp4=float(a[4]/a[1:].sum()),phase4=float(ph),gapshare_240_260=float((g[24]+g[25])/g.sum()),gap0share=float(g[0]/g.sum()),peak=int(x[:25].argmax())*10))
    df=pd.DataFrame(rows); df.to_pickle('timeline.pkl')
    pd.set_option('display.width',200,'display.max_rows',500)
    print(df[df.mk=='6'].to_string()); print(df[df.mk=='0'][['date','topf','amp4','gapshare_240_260','gap0share']].to_string())
