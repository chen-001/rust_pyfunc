import re
p='/home/chenzongwei/rust_pyfunc/src/observable_order_metrics.rs'
lines=open(p).read().split('\n')
body=[(i+1, lines[i]) for i in range(603, 884)]
cnt_push=sum(l.count('out.push(') for _,l in body)
cnt_ext=sum(l.count('out.extend_from_slice(') for _,l in body)
cnt_exto=sum(l.count('out.extend(') for _,l in body) - cnt_ext
print('out.push 出现次数', cnt_push, '| extend_from_slice', cnt_ext, '| 其它 extend', cnt_exto)
print()
print('--- 含 out. 的行（未被上一版解析覆盖的重点看有没有循环）---')
for ln, l in body:
    s=l.split('//')[0]
    if 'out.' in s:
        mark = '<<<' if re.search(r'(for |in \[|in 0\.\.)', s) else ''
        print('%4d %s' % (ln, s.strip()[:100]))
