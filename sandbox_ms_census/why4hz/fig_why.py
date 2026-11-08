# -*- coding: utf-8 -*-
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
for fp in fm.findSystemFonts():
    if 'NotoSansCJK' in fp or 'NotoSansSC' in fp: fm.fontManager.addfont(fp)
plt.rcParams.update({'font.family':['Noto Sans CJK SC','DejaVu Sans'],'axes.unicode_minus':False,'figure.facecolor':'white','axes.facecolor':'white','axes.edgecolor':'#BBBBBB','axes.grid':True,'grid.color':'#E9E9E9','axes.axisbelow':True,'axes.spines.top':False,'axes.spines.right':False})
N=json.load(open('numbers.json')); df=pd.read_pickle('probe4.pkl').sort_values('date')
SH,SZ,G='#1F618D','#C0392B','#95A5A6'
fig,ax=plt.subplots(1,3,figsize=(16,4.9))
a=ax[0]; x=np.arange(1,31)*10
for k,c,l in [('20240329_6',SH,'沪市'),('20240329_0',SZ,'深市主板')]:
    v=np.array(N['到达间隔'][k],float); a.plot(x+5,v[1:31]/v.sum()*100,color=c,lw=2,label=l)
a.set_xlabel('相邻两条逐笔数据到达的时间差（毫秒）'); a.set_ylabel('占全部相邻数据的 %'); a.set_xlim(10,310); a.set_xticks([50,100,150,200,250,300]); a.legend(frameon=False)
a.set_title('沪市的数据隔 250 毫秒来一批，深市连续到达',loc='left',fontsize=11.5)
a=ax[1]; rng=np.random.default_rng(1)
e1=df[(df.date>=20230601)&(df.date<=20240902)]; e2=df[(df.date>=20241202)&(df.date<=20260901)]
for e,c,l in [(e1,'#7FB3D5','2023-06 到 2024-09'),(e2,SH,'2024-12 到 2026-09')]:
    T=e['T'].values; a.scatter(T*10,((e.peak.values-T)%25)*10,s=34,color=c,label=l,zorder=3)
T=e2['T'].values; P=rng.permutation(e2.peak.values); a.scatter(T*10,((P-T)%25)*10,s=22,color=G,alpha=.65,label='把日期打乱（对照）',zorder=2)
a.set_xlabel('这一天分批的结束位置（一个 250 毫秒周期里的毫秒数）'); a.set_ylabel('成交高峰比结束位置晚多少毫秒'); a.set_xlim(-5,250); a.set_ylim(-10,250); a.legend(frameon=False,fontsize=9,loc='upper right')
a.set_title('分批的位置每天在变，高峰总紧跟在后面',loc='left',fontsize=11.5)
a=ax[2]; R=N['上一批有无成交']; xx=np.arange(25)*10+5
for k,c,l in [('prev0',G,'上一批本股没有成交'),('prev1','#7FB3D5','上一批本股 1 到 2 张主动单'),('prev3',SH,'上一批本股 3 张及以上')]:
    a.plot(xx,R[k]['主动单形状'],color=c,lw=2.2,label=l)
a.axhline(1,color='#333333',lw=1,ls='--'); a.set_xlim(0,250); a.set_xticks([0,50,100,150,200,250]); a.set_xlabel('距离这一批结束位置的毫秒数'); a.set_ylabel('相对这 250 毫秒平均的倍数'); a.legend(frameon=False,fontsize=9)
a.set_title('上一批里自己有成交，这一批的高峰才明显',loc='left',fontsize=11.5)
fig.suptitle('沪市一秒四个高峰的来源：逐笔数据每 250 毫秒来一批，交易跟着这一批走',x=.01,ha='left',fontsize=14)
fig.tight_layout(rect=(0,0,1,.94)); fig.savefig('why4hz.png',dpi=170)
