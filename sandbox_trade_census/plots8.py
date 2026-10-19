import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC']; plt.rcParams['axes.unicode_minus']=False
plt.rcParams['axes.edgecolor']='#d0d0d0'; plt.rcParams['axes.linewidth']=.9

D=['20240205','20240329','20240813','20240930','20241008','20250407','20251127','20260311']
LEG={'20240205':'24-02-05 小盘股崩盘','20240329':'24-03-29 常态','20240813':'24-08-13 地量',
     '20240930':'24-09-30 924 行情','20241008':'24-10-08 历史天量','20250407':'25-04-07 关税暴跌',
     '20251127':'25-11-27 常态','20260311':'26-03-11 常态'}
TIT={'20240205':'2024-02-05\n小盘股流动性危机','20240329':'2024-03-29\n常态','20240813':'2024-08-13\n地量',
     '20240930':'2024-09-30\n924 行情','20241008':'2024-10-08\n历史天量','20250407':'2025-04-07\n关税暴跌',
     '20251127':'2025-11-27\n常态','20260311':'2026-03-11\n常态'}
C={'20240205':'#2b6cb0','20240329':'#8a94a6','20240813':'#38a169','20240930':'#dd6b20',
   '20241008':'#c53030','20250407':'#805ad5','20251127':'#0f766e','20260311':'#45505f'}
J={d:json.load(open(f'out/{d}.json')) for d in D}
N={d:np.array([x['n'] for x in J[d]['stats']],float) for d in D}
A={d:np.array([x['amount'] for x in J[d]['stats']],float) for d in D}
def mins(d):
    m=np.array(J[d]['minute_cnt'],float); return np.r_[m[:237],[m[237]+m[238]+m[239]]]
def smooth(v,k=5):
    w=np.ones(k)/k; return np.convolve(v,w,mode='same')/np.convolve(np.ones_like(v),w,mode='same')
def curves(d):
    j=J[d]; hc=np.array(j['hist_cnt'],float); x=10**(j['hist_lo']+j['hist_w']*np.arange(len(hc)))
    o=np.argsort(-x); return np.cumsum(hc[o])/hc.sum()*100, np.cumsum((x*hc)[o])/(x*hc).sum()*100
TICK=[1e2,1e3,1e4,1e5,1e6]; LBL=['100','1千','1万','10万','100万']

# ---------- 图1 ----------
fig,axes=plt.subplots(2,4,figsize=(19,9.4),sharex=True)
for ax,d in zip(axes.flat,D):
    n=N[d]; med=np.median(n)
    ax.hist(np.log10(n),bins=60,color=C[d],alpha=.92)
    ax.axvline(np.log10(med),color='#1a1a1a',ls='--',lw=1.3)
    ax.set_title(TIT[d],fontsize=12,pad=6)
    ax.text(.04,.94,f'中位 {med:,.0f} 笔',transform=ax.transAxes,fontsize=11,va='top',
            bbox=dict(fc='white',ec='none',alpha=.85,pad=1.5))
    ax.set_yticks([]); ax.set_xlim(2,6); ax.grid(axis='x',alpha=.25)
    ax.xaxis.set_major_locator(FixedLocator([np.log10(t) for t in TICK]))
    ax.xaxis.set_major_formatter(FixedFormatter(LBL))
fig.supxlabel('一只股票一天的成交笔数（对数刻度）',fontsize=13,y=.02)
fig.suptitle('同样的形状，不同的位置：八个交易日、五千只股票的成交笔数分布',fontsize=16.5,y=.985)
plt.tight_layout(rect=[0,.045,1,.955]); plt.savefig('figs/f1_count_dist.png',dpi=125,bbox_inches='tight'); plt.close()

# ---------- 图2 ----------
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(16.5,10.5),sharex=True,gridspec_kw={'height_ratios':[1,1],'hspace':.2})
h=[]
for d in D:
    m=mins(d); h.append(ax1.plot(m,color=C[d],lw=1.5,label=LEG[d])[0]); ax2.plot(smooth(m/m.mean()),color=C[d],lw=1.7)
ax1.set_yscale('log'); ax1.set_ylabel('每分钟成交笔数（对数）',fontsize=12)
ax1.set_title('绝对水平：最忙的一天和最闲的一天差三倍多',fontsize=13.5,loc='left',pad=8)
ax2.set_yscale('log'); ax2.set_ylabel('相对全天均值（倍）',fontsize=12)
ax2.set_title('各自除以全天均值（5 分钟平滑）：八条线几乎重合',fontsize=13.5,loc='left',pad=8)
ax2.axhline(1,color='#444',lw=.9,ls=':')
for ax in (ax1,ax2):
    ax.axvline(119.5,color='#999',lw=1,ls='--'); ax.set_xlim(0,237); ax.grid(alpha=.22)
    ax.xaxis.set_major_locator(FixedLocator([0,30,60,90,119.5,150,180,210,237]))
    ax.xaxis.set_major_formatter(FixedFormatter(['9:30','10:00','10:30','11:00','11:30 | 13:00','13:30','14:00','14:30','15:00']))
ax1.annotate('开盘第一分钟是全天均值的 6 到 8 倍',xy=(1.5,4.4e6),xytext=(24,2.0e6),fontsize=11,
             arrowprops=dict(arrowstyle='->',color='#444',lw=1))
fig.legend(h,[LEG[d] for d in D],fontsize=11,ncol=4,loc='lower center',frameon=False,bbox_to_anchor=(.5,.004))
fig.suptitle('一天之内的成交节奏：开盘一根尖峰，午休一道断崖，尾盘再翘一下',fontsize=16,y=.965)
plt.tight_layout(rect=[0,.06,1,.945]); plt.savefig('figs/f2_minute.png',dpi=125,bbox_inches='tight'); plt.close()

# ---------- 图3 ----------
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(17.5,7.6),gridspec_kw={'width_ratios':[1,1.05]})
def dens(cnt,k=10):
    cs=np.convolve(cnt,np.ones(k),mode='same')/k
    return cs/cnt.sum()*k*100
for d in D:
    j=J[d]; c=np.array(j['hist_cnt'],float); x=10**(j['hist_lo']+j['hist_w']*np.arange(len(c)))
    ax1.plot(x,dens(c),color=C[d],lw=1.7,label=LEG[d])
ax1.set_xscale('log'); ax1.set_xlim(100,1e6); ax1.set_xlabel('单笔成交金额 / 元（对数刻度）',fontsize=12)
ax1.set_ylabel('落在该金额附近的成交笔数占比 / %',fontsize=12); ax1.grid(alpha=.22)
ax1.set_title('单笔有多大：峰值永远落在 1 千到 5 千元',fontsize=13,loc='left',pad=8)
ax1.legend(fontsize=9.5,ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.13))

xg=np.logspace(np.log10(.05),2,500); vals=[]
for d in D:
    cn,ca=curves(d); ax2.plot(cn,ca,color=C[d],lw=1.7)
    vals.append(np.interp(np.log10(xg),np.log10(cn),ca))
vals=np.array(vals)
ax2.fill_between(xg,vals.min(0),vals.max(0),color='#6b7280',alpha=.20,label='八天的范围')
ax2.plot(xg,xg,color='#999',ls=':',lw=1.1)
ax2.set_xscale('log'); ax2.set_xlim(.05,100); ax2.set_ylim(0,100); ax2.grid(alpha=.22)
ax2.set_xlabel('成交笔数从大到小累计 / %（对数刻度）',fontsize=12)
ax2.set_ylabel('这些笔数贡献的成交金额 / %',fontsize=12)
ax2.set_title('钱在哪里：最贵的 1% 笔数拿走两成多的金额',fontsize=13,loc='left',pad=8)
exact={1:[],5:[],10:[]}
for d in D:
    cn,ca=curves(d); cn=cn/100; ca=ca/100
    for q in (1,5,10):
        exact[q].append(ca[np.searchsorted(cn,q/100)]*100)
for q in (1,5,10):
    v=np.array(exact[q]); lo,hi=v.min(),v.max()
    i=np.argmin(np.abs(np.log10(xg)-np.log10(q)))
    ax2.axvline(q,color='#666',lw=.9,ls='--',alpha=.8)
    ax2.plot([q]*8,v,'o',ms=5,color='#333',alpha=.5,zorder=6)
    ax2.annotate(f'{q}% 的笔数\n{lo:.1f}% ~ {hi:.1f}% 金额',xy=(q,v.mean()),xytext=(q*1.3,v.mean()-16),
                 fontsize=10.5,arrowprops=dict(arrowstyle='-',color='#777',lw=.8))
ax2.text(.055,3,'如果每一笔一样大，就贴着这条对角线',fontsize=9.5,color='#666')
fig.suptitle('最小的一手和最大的单：单笔金额的两个视角',fontsize=16,y=.98)
plt.tight_layout(rect=[0,0,1,.945]); plt.savefig('figs/f3_amt_dist.png',dpi=125,bbox_inches='tight'); plt.close()

# ---------- 图4 ----------
fig,ax=plt.subplots(figsize=(15,8))
for d in D:
    n=N[d]; y=A[d]; b=np.logspace(3,np.log10(n.max()),18); idx=np.digitize(n,b)-1
    xs,ys=[],[]
    for k in range(len(b)):
        m=idx==k
        if m.sum()>15: xs.append(np.median(n[m])); ys.append(np.median(y[m]))
    ax.plot(xs,ys,color=C[d],lw=2,marker='o',ms=3.5,label=LEG[d],alpha=.95)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(8e3,4e5); ax.set_ylim(2e7,2e10)
ax.set_xlabel('一只股票一天的成交笔数（对数）',fontsize=12)
ax.set_ylabel('这只股票一天的成交额 / 元（对数）',fontsize=12)
ax.grid(alpha=.22); ax.legend(fontsize=10,ncol=4,frameon=False,loc='upper left')
ax.set_title('成交额几乎就是笔数乘一个常数：八条线平行，斜率都接近 1',fontsize=14.5,loc='left',pad=10)
plt.tight_layout(); plt.savefig('figs/f4_amount_vs_count.png',dpi=125,bbox_inches='tight'); plt.close()
print('ok')
