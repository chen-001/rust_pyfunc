import re
lines = open('/home/chenzongwei/rust_pyfunc/src/observable_order_metrics.rs').read().split('\n')
body = lines[586:884]   # compute_segment_features 函数体附近
dele = set(int(x) for x in re.findall(r'(\d+)', re.search(r'pub const DELETE_COLS: \[usize; 84\] = \[(.*?)\];', '\n'.join(lines), re.S).group(1)))
col = 0
stmts = []
pend_comment = ''
for i, l in enumerate(body):
    s = l.strip()
    if s.startswith('//'):
        if not s.startswith('// ===='): pend_comment = s
        continue
    if s.startswith('out.extend_from_slice('):
        stmts.append((col, 5, pend_comment, i+587)); col += 5
    elif s.startswith('out.push('):
        # push 可能跨多行；只按单行计数
        if s.endswith(';') or s.endswith(');'):
            stmts.append((col, 1, pend_comment, i+587)); col += 1
    elif s.startswith('out.extend('):
        stmts.append((col, 0, 'extend(?) ' + pend_comment, i+587))
print('按语句累加列数:', col)
full_drop = [s for s in stmts if s[1] and all(c in dele for c in range(s[0], s[0]+s[1]))]
part = [s for s in stmts if s[1] and any(c in dele for c in range(s[0], s[0]+s[1])) and not all(c in dele for c in range(s[0], s[0]+s[1]))]
keep = [s for s in stmts if s[1] and not any(c in dele for c in range(s[0], s[0]+s[1]))]
print('语句总数', len(stmts), '整段被删', len(full_drop), '部分被删', len(part), '整段保留', len(keep))
print('整段被删占的列数', sum(s[1] for s in full_drop), '部分被删里的被删列数', sum(len([c for c in range(s[0], s[0]+s[1]) if c in dele]) for s in part))
print()
print('--- 整段被删、可以不算的语句（前 30 条，含行号）---')
for s in full_drop[:30]:
    print('  列 %3d 起 %d 列  行 %d  %s' % (s[0], s[1], s[3], s[2][:58]))
print()
print('--- 部分被删的语句（×5 聚合类，前 20 条）---')
for s in part[:20]:
    d = [c for c in range(s[0], s[0]+s[1]) if c in dele]
    print('  列 %3d 起 %d 列，删 %s  行 %d  %s' % (s[0], s[1], d, s[3], s[2][:45]))
