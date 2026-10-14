import json, collections
SLOT=['mean','std','skew','ac1','trend']
# 成本权重：遍数（mean1, std1, skew2+powi/powf, ac1 1, trend 1+建一个等长 Vec）
W={'mean':1.0,'std':1.2,'skew':2.0,'ac1':1.2,'trend':1.5}
stmts=[(611,20,'P1 盘口失衡 4 档'),(616,10,'P2 10档trend'),(621,10,'P3 10档std'),
       (627,10,'P7 rolling10 det'),(633,10,'P8 rolling20 det'),(699,5,'T1 订单编号差值原始'),
       (700,5,'T1 |差值|'),(714,5,'T2 按秒主买占比'),(756,25,'T3 买/卖/差/|差|/全量'),
       (842,25,'T4 买均/卖均/差/|差|/全量')]
dele=set(range(0,0))
import re
src=open('/home/chenzongwei/rust_pyfunc/src/observable_order_metrics.rs').read()
dele=set(int(x) for x in re.findall(r'(\d+)', re.search(r'pub const DELETE_COLS: \[usize; 84\] = \[(.*?)\];', src, re.S).group(1)))
# 列号起点（与上一版一致，按手工账单累加）
offsets=[(611,0,20),(616,20,10),(621,30,10),(627,40,10),(633,50,10),(639,60,1),(647,61,1),(656,62,1),
         (663,63,1),(664,64,1),(665,65,1),(666,66,1),(667,67,1),(668,68,1),(669,69,1),(670,70,1),(675,71,1),
         (699,72,5),(700,77,5),(714,82,5),(756,87,25),(842,112,25),
         (856,137,1),(857,138,1),(860,139,1),(861,140,1),(864,141,1),(865,142,1),(867,143,1),(868,144,1)]
om=dict((a,(b,c)) for a,b,c in offsets)
tot_slots=0; drop_slots=0; wsum=0.0; wdrop=0.0
detail=[]
for ln,n,desc in stmts:
    c0,_=om[ln]
    dslots=[]; 
    for s in range(n//5):
        for k in range(5):
            col=c0+s*5+k; tot_slots+=1; wsum+=W[SLOT[k]]
            if col in dele:
                drop_slots+=1; wdrop+=W[SLOT[k]]; dslots.append(SLOT[k])
    cnt=collections.Counter(dslots)
    detail.append((ln,desc,n//5,cnt))
print('聚合语句共有槽', tot_slots, '被删槽', drop_slots, '(%.0f%%)' % (100.0*drop_slots/tot_slots))
print('按每槽成本加权：被删掉的权重占 %.0f%%' % (100.0*wdrop/wsum))
print()
print('| 语句 | 源数 | 被删的槽（×源数） |')
print('|---|---|---|')
for ln,desc,ns,cnt in detail:
    print('| 行 %d %s | %d | %s |' % (ln, desc, ns, '，'.join('%s×%d' % (k,v) for k,v in sorted(cnt.items()))))
print()
print('整条可跳的标量语句 12 条 / 12 列（行 647、664-670、856、860、864、867）')
