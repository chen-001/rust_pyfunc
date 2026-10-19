import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
plt.rcParams['font.sans-serif']=['Noto Sans CJK SC']; plt.rcParams['axes.unicode_minus']=False
plt.rcParams['axes.edgecolor']='#d0d0d0'; plt.rcParams['axes.linewidth']=.9
D=['20240205','20240329','20240813','20240930','20241008','20250407','20251127','20260311']
TIT={'20240205':'2024-02-05\n小盘股流动性危机','20240329':'2024-03-29\n常态','20240813':'2024-08-13\n地量',
     '20240930':'2024-09-30\n924 行情','20241008':'2024-10-08\n历史天量','20250407':'2025-04-07\n关税暴跌',
     '20251127':'2025-11-27\n常态','20260311':'2026-03-11\n常态'}
C={'20240205':'#2b6cb0','20240329':'#8a94a6','20240813':'#38a169','20240930':'#dd6b20',
   '20241008':'#c53030','20250407':'#805ad5','20251127':'#0f766e','20260311':'#45505f'}
J={d:json.load(open(f'out/{d}.json')) for d in D}
N={d:np.array([x['n'] for x in J[d]['stats']],float) for d in D}
A={d:np.array([x['amount'] for x in J[d]['stats']],float) for d in D}

fig,axes=plt.subplots(2,4,figsize=(19,9.6),sharex=True,sharey=True)
for ax,d in zip(axes.flat,D):
    n,a=N[d],A[d]; k=a.sum()/n.sum()
    ax.scatter(n,a,s=3.2,alpha=.22,color=C[d],edgecolors='none',rasterized=True)
    xs=np.array([20,1.2e6]); ax.plot(xs,1e4*xs,ls='--',color='#111',lw=1.1,alpha=.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(20,1.2e6); ax.set_ylim(1e5,3e11)
    ax.set_title(f'{TIT[d]}\n当天平均每笔 {k:,.0f} 元',fontsize=11,pad=6)
    ax.grid(alpha=.18)
    ax.xaxis.set_major_locator(FixedLocator([1e2,1e3,1e4,1e5,1e6]))
    ax.xaxis.set_major_formatter(FixedFormatter(['100','1千','1万','10万','100万']))
    ax.yaxis.set_major_locator(FixedLocator([1e6,1e8,1e10]))
    ax.yaxis.set_major_formatter(FixedFormatter(['百万','亿','百亿']))
fig.supxlabel('这只股票一天的成交笔数（对数刻度）',fontsize=13,y=.02)
fig.supylabel('这只股票一天的成交额（对数刻度）',fontsize=13,x=.008)
fig.suptitle('一只股票一个点，虚线是一笔一万元',fontsize=16,y=.985)
plt.tight_layout(rect=[.015,.045,1,.955]); plt.savefig('figs/f4_amount_vs_count.png',dpi=125,bbox_inches='tight'); plt.close()
print('ok')
