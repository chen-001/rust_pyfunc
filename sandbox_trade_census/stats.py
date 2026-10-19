import json, numpy as np
D=['20240205','20240329','20240813','20240930','20241008','20250407']
for d in D:
    j=json.load(open(f'out/{d}.json')); s=j['stats']
    n=np.array([x['n'] for x in s]); a=np.array([x['amount'] for x in s])
    mi=np.array(j['minute_cnt'],dtype=float); ma=np.array(j['minute_amt'])
    tot_n=n.sum(); tot_a=a.sum()
    lg=np.log10(n)
    # 全局前1%笔数占金额：用直方图从大到小累加
    hc=np.array(j['hist_cnt'],dtype=float); x=10**(j['hist_lo']+j['hist_w']*np.arange(len(hc)))
    ws=x*hc  # 每 bin 的金额近似
    order=np.argsort(-x); cum_n=np.cumsum(hc[order]); cum_a=np.cumsum(ws[order])
    i1=np.searchsorted(cum_n, tot_n*0.01)
    top1_glob=cum_a[i1]/tot_a
    i10=np.searchsorted(cum_n, tot_n*0.10)
    top10_glob=cum_a[i10]/tot_a
    # 最忙 10 分钟 / 开盘第一分钟
    srt=np.sort(mi)[::-1]
    print(f'{d}  总笔数 {tot_n/1e8:.3f}亿  总金额 {tot_a/1e8:.0f}亿')
    print(f'   笔数 中位{np.median(n):.0f} P10 {np.percentile(n,10):.0f} P90 {np.percentile(n,90):.0f} 最大 {n.max():.0f} | P90/P10 {np.percentile(n,90)/np.percentile(n,10):.1f}x  最大/中位 {n.max()/np.median(n):.0f}x')
    print(f'   log10笔数 偏度 {(((lg-lg.mean())**3).mean()/lg.std()**3):.2f}  峰度 {(((lg-lg.mean())**4).mean()/lg.std()**4-3):.2f}')
    print(f'   单笔中位 {np.median([x["p50"] for x in s]):.0f} 元 | 全局前1%笔数占金额 {top1_glob*100:.1f}%  前10% {top10_glob*100:.1f}%')
    print(f'   分钟: 开盘1分占比 {mi[0]/mi.sum()*100:.1f}%  最忙10分 {srt[:10].sum()/mi.sum()*100:.1f}%  上午(tot) {mi[:120].sum()/mi.sum()*100:.1f}%  下午 {mi[120:].sum()/mi.sum()*100:.1f}%')
    print(f'   集合竞价占比 {sum(x["auc_amt"] for x in s)/tot_a*100:.2f}%  收盘竞价 {sum(x["close_amt"] for x in s)/tot_a*100:.2f}%')
    # 笔数 vs 金额 秩相关
    from scipy.stats import spearmanr
    print(f'   笔数与金额 秩相关 {spearmanr(n,a).statistic:.3f}   金额/笔数 中位 {np.median(a/n):.0f} 元')
    print()
