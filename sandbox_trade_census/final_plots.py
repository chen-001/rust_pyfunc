import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC']; plt.rcParams['axes.unicode_minus']=False
plt.rcParams['axes.edgecolor']='#d0d0d0'; plt.rcParams['axes.linewidth']=.9
plt.rcParams['figure.facecolor']='white'; plt.rcParams['axes.facecolor']='white'

D=['20240205','20240329','20240813','20240930','20241008','20250407']
SHORT={'20240205':'02-05 小盘股崩盘','20240329':'03-29 常态','20240813':'08-13 地量',
       '20240930':'09-30 924 行情','20241008':'10-08 历史天量','20250407':'04-07 关税暴跌'}
NAME={k:v.replace(' ','\n',1) for k,v in SHORT.items()}
NAME={'20240205':'2024-02-05\n小盘股流动性危机','20240329':'2024-03-29\n常态','20240813':'2024-08-13\n地量',
      '20240930':'2024-09-30\n924 行情','20241008':'2024-10-08\n历史天量','20250407':'2025-04-07\n关税暴跌'}
C={'20240205':'#2b6cb0','20240329':'#8a94a6','20240813':'#38a169','20240930':'#dd6b20','20241008':'#c53030','20250407':'#805ad5'}
J={d:json.load(open(f'out/{d}.json')) for d in D}
N={d:np.array([x['n'] for x in J[d]['stats']],dtype=float) for d in D}
A={d:np.array([x['amount'] for x in J[d]['stats']],dtype=float) for d in D}
def mins(d):
    m=np.array(J[d]['minute_cnt'],dtype=float); return np.r_[m[:237],[m[237]+m[238]+m[239]]]
TICK=[1e2,1e3,1e4,1e5,1e6]; LBL=['100','1千','1万','10万','100万']

# ---------- 图1 ----------
fig,axes=plt.subplots(2,3,figsize=(16,8.6),sharex=True)
for ax,d in zip(axes.flat,D):
    n=N[d]; lg=np.log10(n); med=np.median(n)
    ax.hist(lg,bins=70,color=C[d],alpha=.92)
    ax.axvline(np.log10(med),color='#1a1a1a',ls='--',lw=1.4)
    ax.set_title(NAME[d],fontsize=12.5,pad=6)
    ax.text(.03,.93,f'中位 {med:,.0f} 笔',transform=ax.transAxes,fontsize=11.5,
            va='top',bbox=dict(fc='white',ec='none',alpha=.85,pad=2))
    ax.set_yticks([]); ax.set_xlim(2,6); ax.grid(axis='x',alpha=.25)
    ax.xaxis.set_major_locator(FixedLocator([np.log10(t) for t in TICK]))
    ax.xaxis.set_major_formatter(FixedFormatter(LBL))
fig.supxlabel('一只股票一天的成交笔数（对数刻度）',fontsize=13,y=.015)
fig.suptitle('同样的形状，不同的位置：六个交易日、五千只股票的成交笔数分布',fontsize=16,y=.985)
plt.tight_layout(rect=[0,.045,1,.955]); plt.savefig('figs/f1_count_dist.png',dpi=130,bbox_inches='tight'); plt.close()

# ---------- 图2 ----------
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(16,10),sharex=True,gridspec_kw={'height_ratios':[1,1],'hspace':.18})
for d in D:
    m=mins(d); ax1.plot(m,color=C[d],lw=1.5,label=SHORT[d]); ax2.plot(m/m.mean(),color=C[d],lw=1.5)
ax1.set_yscale('log'); ax1.set_ylabel('每分钟成交笔数（对数）',fontsize=12)
ax1.set_title('绝对水平：最忙和最闲的日子差三倍多',fontsize=13.5,loc='left',pad=8)
ax2.set_yscale('log'); ax2.set_ylabel('相对全天均值（倍）',fontsize=12)
ax2.set_title('各自除以全天均值之后：形状几乎重合，只有 10-08 在 10:30 多出一块',fontsize=13.5,loc='left',pad=8)
ax2.axhline(1,color='#444',lw=.9,ls=':')
for ax in (ax1,ax2):
    ax.axvline(119.5,color='#999',lw=1,ls='--'); ax.set_xlim(0,237); ax.grid(alpha=.22)
    ax.xaxis.set_major_locator(FixedLocator([0,30,60,90,119.5,150,180,210,237]))
    ax.xaxis.set_major_formatter(FixedFormatter(['9:30','10:00','10:30','11:00','11:30 | 13:00','13:30','14:00','14:30','15:00']))
ax1.legend(fontsize=11,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.02),frameon=False)
fig.suptitle('一天之内的成交节奏：开盘一根尖峰，午休一道断崖',fontsize=16,y=.965)
plt.tight_layout(rect=[0,0,1,.945]); plt.savefig('figs/f2_minute.png',dpi=130,bbox_inches='tight'); plt.close()

# ---------- 图3 双面板 ----------
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(17,7.2),gridspec_kw={'width_ratios':[1,1]})
for d in D:
    j=J[d]; c=np.array(j['hist_cnt'],dtype=float); x=10**(j['hist_lo']+j['hist_w']*np.arange(len(c)))
    ax1.plot(x,c/c.sum(),color=C[d],lw=1.8,label=SHORT[d])
ax1.set_xscale('log'); ax1.set_xlim(100,1e6); ax1.set_xlabel('单笔成交金额 / 元（对数刻度）',fontsize=12)
ax1.set_ylabel('占全部成交笔数的比例',fontsize=12); ax1.grid(alpha=.22)
ax1.set_title('单笔有多大：峰值永远落在 1 千到 5 千元',fontsize=13,loc='left',pad=8)
ax1.legend(fontsize=10,ncol=2,frameon=False,loc='upper right')
for d in D:
    j=J[d]; hc=np.array(j['hist_cnt'],dtype=float); x=10**(j['hist_lo']+j['hist_w']*np.arange(len(hc)))
    order=np.argsort(-x); cn=np.cumsum(hc[order])/hc.sum(); ca=np.cumsum((x*hc)[order])/(x*hc).sum()
    ax2.plot(np.r_[0,cn]*100,np.r_[0,ca]*100,color=C[d],lw=1.8)
ax2.set_xlim(0,100); ax2.set_ylim(0,100); ax2.grid(alpha=.22)
ax2.set_xlabel('成交笔数从大到小累计，占全部笔数的比例 / %',fontsize=12)
ax2.set_ylabel('对应的成交金额占全部金额的比例 / %',fontsize=12)
ax2.set_title('钱在哪里：最大的 1% 笔数拿走约 22% 的金额',fontsize=13,loc='left',pad=8)
ax2.plot([0,100],[0,100],color='#999',ls=':',lw=1)
for d in D:
    j=J[d]; hc=np.array(j['hist_cnt'],dtype=float); x=10**(j['hist_lo']+j['hist_w']*np.arange(len(hc)))
    order=np.argsort(-x); cn=np.cumsum(hc[order])/hc.sum(); ca=np.cumsum((x*hc)[order])/(x*hc).sum()
    i=np.searchsorted(cn,0.01)
    if d=='20240930': ax2.plot(1,ca[i]*100,'o',color=C[d],ms=7,zorder=5)
ax2.annotate('前 1% 的笔数\n≈ 22% 的金额',xy=(1,22),xytext=(12,16),fontsize=11,
             arrowprops=dict(arrowstyle='->',color='#444',lw=1))
fig.suptitle('最小的一手和最大的单：单笔金额的两个视角',fontsize=16,y=.985)
plt.tight_layout(rect=[0,.005,1,.945]); plt.savefig('figs/f3_amt_dist.png',dpi=130,bbox_inches='tight'); plt.close()

# ---------- 图4 笔数 vs 平均单笔 ----------
fig,ax=plt.subplots(figsize=(15,7.5))
for d in D:
    n=N[d]; y=A[d]/n
    b=np.logspace(np.log10(max(n.min(),50)),np.log10(n.max()),25)
    idx=np.digitize(n,b)-1
    xs,ys=[],[]
    for k in range(len(b)):
        m=idx==k
        if m.sum()>20: xs.append(np.median(n[m])); ys.append(np.median(y[m]))
    ax.plot(xs,ys,color=C[d],lw=2.2,marker='o',ms=3.5,label=SHORT[d])
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('一只股票一天的成交笔数（对数）',fontsize=12)
ax.set_ylabel('平均每一笔的成交金额 / 元（对数）',fontsize=12)
ax.grid(alpha=.22); ax.legend(fontsize=10.5,ncol=2,frameon=False)
ax.set_title('越活跃的股票，每一笔也越大：笔数翻十倍，单笔金额翻三倍',fontsize=14.5,loc='left',pad=10)
plt.tight_layout(); plt.savefig('figs/f4_count_vs_size.png',dpi=130,bbox_inches='tight'); plt.close()
print('ok')
