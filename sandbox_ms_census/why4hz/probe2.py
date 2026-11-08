from common import *
import sys
np.set_printoptions(linewidth=220,precision=3,suppress=True)
DAYS=[20240205,20240329,20240813,20240930,20241008,20250407,20251127,20260311]
def load(d,c):
    t=trades(d,c); t=t[session_mask(t.time)].copy(); t['cs']=cs_of(t.exchtime.values); return t
def batches(t):
    lt=t.localtime.values.astype(float); cs=t.cs.values
    cut=np.r_[0,np.flatnonzero(np.diff(lt)>50000)+1,len(lt)]
    return lt,cs,cut
def p1(a):
    d,c=a; t=load(d,c); lt,cs,cut=batches(t)
    h=np.zeros(25,int); lag=[]; nb=len(cut)-1; g=np.diff(lt)
    for i in range(nb):
        x=cs[cut[i]:cut[i+1]]
        if len(x)==0: continue
        h[int(x.max()%25)]+=1
        lag.append((lt[cut[i]]+0-0))  # 占位
    last=np.array([cs[cut[i]:cut[i+1]].max() for i in range(nb) if cut[i+1]>cut[i]])
    first_lt=np.array([lt[cut[i]] for i in range(nb) if cut[i+1]>cut[i]])
    exch_last_us=(last*1e4)-8*3600e6   # 仅相对量
    span=np.array([cs[cut[i]:cut[i+1]].max()-cs[cut[i]:cut[i+1]].min() for i in range(nb) if cut[i+1]>cut[i]])
    bg=np.diff(first_lt)
    return d,h,nb,len(t),np.bincount(np.clip((bg/10000).astype(int),0,99),minlength=100),np.bincount(np.clip(span,0,60),minlength=61)
def p2(a):
    d,c,b=a; t=load(d,c)
    tr=t[t.flag.isin([66,83])&(t.turnover>0)].copy(); ca=t[t.flag==32]
    tr['oid']=np.where(tr.flag==66,tr.bid_order,tr.ask_order).astype(np.int64)
    ev=tr.groupby(['cs','flag','oid'],sort=False).size().reset_index()
    # 用真实连续竞价序号
    def idx(x):
        x=np.asarray(x); return np.where(x<4140000+1,x-3420000,x-4680000+720000)
    ei=idx(ev.cs.values); cai=idx(ca.cs.values)
    W=1422000//25+2
    def win(i): 
        q=(i-b); return q//25, q%25
    w,o=win(ei); wc,oc=win(cai)
    n=np.bincount(w+1,minlength=W+3)  # 每窗主动单数（下标 w+1）
    prev=np.r_[0,n[:-1]]              # 上一窗
    out={}
    for name,mask in [('prev0',prev==0),('prev1',(prev>=1)&(prev<3)),('prev3',prev>=3)]:
        occ=int(mask[1:W+1].sum())
        pick=mask[w+1]; 
        ev_c=np.bincount(o[pick],minlength=25)
        pc=mask[wc+1]; c_c=np.bincount(oc[pc],minlength=25)
        out[name]=(occ,ev_c,c_c)
    return d,out
if __name__=='__main__':
    SH=[(d,c) for d in DAYS for c in sample(d,'6',200)]
    Hs={};N={}
    with mp.Pool(50) as p:
        for d,h,nb,n,bg,sp in p.imap_unordered(p1,SH,chunksize=2):
            A=Hs.setdefault(d,dict(h=0,nb=0,n=0,bg=0,sp=0)); A['h']=A['h']+h;A['nb']+=nb;A['n']+=n;A['bg']=A['bg']+bg;A['sp']=A['sp']+sp
    B={}
    for d in DAYS:
        A=Hs[d]; h=A['h']; c=int(h.argmax()); B[d]=(c+1)%25
        print(d,'batches/stock',A['nb']/200,'rows/batch',round(A['n']/A['nb'],1),'cut cell(last exch cell mod 25)=',c,'next window starts at cell',B[d],'share at mode',round(h.max()/h.sum(),3),'share mode±1',round((h[(c-1)%25]+h[c]+h[(c+1)%25])/h.sum(),3))
        print('   h/sum',(h/h.sum()).round(3)); print('   batch-to-batch gap bins(10ms) top',[(int(i)*10,round(float(A['bg'][i]/A['bg'].sum()),3)) for i in np.argsort(-A['bg'])[:5]],' span cells',(A['sp']/A['sp'].sum())[:8].round(3))
    json.dump({str(k):int(v) for k,v in B.items()},open('bound.json','w'))
    jobs=[(d,c,B[d]) for d,c in SH]
    R={}
    with mp.Pool(50) as p:
        for d,o in p.imap_unordered(p2,jobs,chunksize=2):
            for k,(occ,e,ca) in o.items():
                A=R.setdefault((d,k),[0,0,0]); A[0]+=occ;A[1]=A[1]+e;A[2]=A[2]+ca
    import pickle
    pickle.dump({'B':B,'R':R,'Hs':Hs},open('probe2.pkl','wb'))
    for d in DAYS:
        print('==',d)
        for k in ['prev0','prev1','prev3']:
            occ,e,ca=R[(d,k)]; e=e/occ;ca=ca/occ;y=e/e.mean();z=ca/ca.mean()
            print(f' {k} windows {occ} per-window ev {e.sum():.3f} cancel {ca.sum():.3f} shape ev max/min {y.max()/y.min():.2f} argmax {y.argmax()}'); print('   ev',y.round(2)); print('   ca',z.round(2))
