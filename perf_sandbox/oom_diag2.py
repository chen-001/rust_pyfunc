p='/home/chenzongwei/rust_pyfunc/src/observable_order_metrics.rs'
lines=open(p).read().split('\n')
body=[(i+1, lines[i]) for i in range(603, 884)]
hits=[(ln,l.strip()) for ln,l in body if ('out' in l.split('//')[0]) and 'out.' not in l.split('//')[0]]
print('含 out 但不是 out. 的行数', len(hits))
for ln,l in hits[:25]: print('%4d %s' % (ln, l[:110]))
print()
print('--- 677-700 行原文 ---')
for i in range(676, 700): print('%4d %s' % (i+1, lines[i][:110]))
