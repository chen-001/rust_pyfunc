import re, json
p='/home/chenzongwei/rust_pyfunc/src/observable_order_metrics.rs'
lines=open(p).read().split('\n')
dele=set(int(x) for x in re.findall(r'(\d+)', re.search(r'pub const DELETE_COLS: \[usize; 84\] = \[(.*?)\];', '\n'.join(lines), re.S).group(1)))
# 每个产出语句手工确认后的 (行号, 列数, 说明)
stmts=[
 (611,20,'P1 盘口失衡 4 档 ×5（for dim in 0..4）'),
 (616,10,'P2 10档trend 同侧/对手 ×5'),
 (621,10,'P3 10档std 同侧/对手 ×5'),
 (627,10,'P7 rolling10 det/|det| ×5'),
 (633,10,'P8 rolling20 det/|det| ×5'),
 (639,1,'P4 价格波动率'),
 (647,1,'P5 间隔成交量 std（另一分支 649）'),
 (656,1,'P6 间隔成交量 trend'),
 (663,1,'P9 d_same'),(664,1,'P9 |d_same|'),(665,1,'P9 d_opp'),(666,1,'P9 |d_opp|'),
 (667,1,'P9 差'),(668,1,'P9 |差|'),(669,1,'P9 差绝对值'),(670,1,'P9 |差绝对值|'),
 (675,1,'P10 d（另一分支 676）'),
 (699,5,'T1 订单编号差值 原始 ×5'),(700,5,'T1 |差值| ×5'),
 (714,5,'T2 按秒主买占比 ×5'),
 (756,25,'T3 买/卖/差/|差|/全量 各 ×5（5 个源）'),
 (842,25,'T4 买均/卖均/差/|差|/全量 各 ×5（5 个源）'),
 (856,1,'主买 全量 d_all'),(857,1,'主买 全量 |d|'),
 (860,1,'主买 d_buy'),(861,1,'主买 |d_buy|'),
 (864,1,'主卖 d_sell'),(865,1,'主卖 |d_sell|'),
 (867,1,'主买-主卖 差'),(868,1,'主买-主卖 |差|'),
]
col=0; out=[]
for ln,n,desc in stmts:
    idx=list(range(col,col+n)); d=[k for k in idx if k in dele]
    kind='整条可跳' if len(d)==n else ('部分被删' if d else '整条保留')
    out.append((ln,col,n,desc,kind,len(d)))
    col+=n
print('账单列总数', col, '（应为 147）')
full=[x for x in out if x[4]=='整条可跳']; part=[x for x in out if x[4]=='部分被删']; keep=[x for x in out if x[4]=='整条保留']
print('整条可跳 %d 条语句 / %d 列' % (len(full), sum(x[2] for x in full)))
print('部分被删 %d 条语句 / 涉及删 %d 列（这些语句里聚合是共享的，跳不了）' % (len(part), sum(x[5] for x in part)))
print('整条保留 %d 条语句 / %d 列' % (len(keep), sum(x[2] for x in keep)))
print('合计被删列', sum(x[5] for x in out))
print()
print('--- 整条可跳的语句（真能省的部分）---')
for ln,c0,n,desc,kind,nd in full:
    print('  行 %4d  列 %3d 起 %d 列  删 %d  %s' % (ln, c0, n, nd, desc))
print()
print('--- 部分被删的语句（跳不了）---')
for ln,c0,n,desc,kind,nd in part:
    print('  行 %4d  列 %3d 起 %2d 列  删 %2d  %s' % (ln, c0, n, nd, desc))
json.dump([{'line':a,'col0':b,'n':c,'desc':d,'kind':e,'dropped':f} for a,b,c,d,e,f in out],
          open('/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out/oom_stmt_class.json','w'), ensure_ascii=False, indent=1)
