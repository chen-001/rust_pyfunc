import re, json
p='/home/chenzongwei/rust_pyfunc/src/observable_order_metrics.rs'
lines=open(p).read().split('\n')
dele=set(int(x) for x in re.findall(r'(\d+)', re.search(r'pub const DELETE_COLS: \[usize; 84\] = \[(.*?)\];', '\n'.join(lines), re.S).group(1)))
body=lines[603:884]          # 604..883（1-based）
# 去注释
clean=[]
for i,l in enumerate(body):
    s=l.split('//')[0].rstrip()
    if s.strip(): clean.append((604+i, s))
# 语句切分：按行扫描，遇到 for 头压栈，遇到 '}' 弹栈；push 语句跨行累积
stmts=[]; col=0; stack=[]; buf=''; bline=None
i=0
def mult_of(header):
    m=re.search(r'in\s+0\.\.(\d+)', header)
    if m: return int(m.group(1))
    m=re.search(r'in\s+\[([^\]]*)\]', header)
    if m: return len([x for x in m.group(1).split(',') if x.strip()])
    return 1
while i < len(clean):
    ln, s = clean[i]
    st=s.strip()
    if st.startswith('for '):
        stack.append(mult_of(st)); i+=1
        # 同行的 for 头后面可能有花括号，跳过
        continue
    if st == '}' or st.startswith('} else'):
        if stack: stack.pop()
        i+=1; continue
    if 'out.push(' in s or 'out.extend_from_slice(' in s or 'out.extend(' in s:
        blk=s; j=i; 
        while ';' not in blk and j+1 < len(clean):
            j+=1; blk += ' ' + clean[j][1]
        m=1
        for x in stack: m*=x
        n_push=blk.count('out.push(')
        n_ext=blk.count('out.extend_from_slice(')
        n_ext2=blk.count('out.extend(')-n_ext
        cnt=n_push+n_ext*5
        tag='push' if n_push and not n_ext else ('ext5' if n_ext and not n_push else 'mixed')
        stmts.append((ln, clean[j][0], m, cnt, col, tag, n_push, n_ext, n_ext2, blk.strip()[:70]))
        col += cnt*m
        i=j+1; continue
    i+=1
print('解析出的列总数', col, '（应等于 147）')
full=[]; part=[]; keep=[]
for st in stmts:
    ln0, ln1, m, cnt, c0, tag, np_, ne, ne2, txt = st
    total=cnt*m
    idxs=list(range(c0, c0+total))
    d=[k for k in idxs if k in dele]
    if total and len(d)==total: full.append((st, len(d)))
    elif d: part.append((st, len(d), total))
    elif total: keep.append((st, total))
print('语句数', len(stmts), '| 整条可跳的语句', len(full), '列数', sum(x[1] for x in full))
print('部分被删的语句', len(part), '涉及删掉', sum(x[1] for x in part), '列（含共用计算，不能整条跳）')
print('完全保留的语句', len(keep), '列数', sum(x[1] for x in keep))
print()
print('--- 整条可跳（这几条改完就能不执行）---')
for st,nd in full:
    ln0,ln1,m,cnt,c0,tag,_,_,_,txt=st
    print('  行 %4d-%-4d 倍数 %d  列 %3d 起 %d 列  删 %d 列  %s' % (ln0,ln1,m,c0,cnt*m,nd,txt[:56]))
print()
print('--- 部分被删（跳不了，只能省被删那几列自己的收尾）---')
for st,nd,total in part:
    ln0,ln1,m,cnt,c0,tag,_,_,_,txt=st
    print('  行 %4d-%-4d 倍数 %d  列 %3d 起 %d 列  删 %d/%d  %s' % (ln0,ln1,m,c0,total,nd,total,txt[:48]))
json.dump({'full_drop':[{'line0':s[0][0],'line1':s[0][1],'mult':s[0][2],'cols':s[0][3]*s[0][2],'start':s[0][4],'dropped':s[1]} for s in full],
           'partial':[{'line0':s[0][0],'line1':s[0][1],'mult':s[0][2],'total':s[2],'start':s[0][4],'dropped':s[1]} for s in part],
           'total_cols':col}, open('/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out/oom_stmt_class.json','w'), ensure_ascii=False)
