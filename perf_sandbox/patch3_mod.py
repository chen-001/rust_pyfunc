p='/home/chenzongwei/rust_pyfunc/src/yupei_pair/mod.rs'
s=open(p).read(); done=[]
def rep(old,new,tag):
    global s
    if new in s: done.append('SKIP '+tag); return
    assert old in s, 'MISS '+tag
    s=s.replace(old,new,1); done.append('OK '+tag)
rep("            let net = net::net_one_mat(sdir, n, &dec, &mask, &stock_idx, null_mc, exz_bins, mat);",
    "            let mut net = net::net_one_mat(\n                sdir, n, &dec, &mask, &stock_idx, null_mc, exz_bins, mat, Some(&KS_DEFAULT),\n            );", 'call net')
rep("            let ms = matset_from_net(net, n);",
    "            // topk 的累计结果已经从同一趟扫描里拿出来了，先取走再 move net\n            let tk_total = net.topk_total.take();\n            let ms = matset_from_net(net, n);", 'take topk')
rep("            let (tn, tc) = topk_one_mat(&ms.rho, &ms.e, n, &mask, &KS_DEFAULT, &pre);",
    "            let (tn, tc) = match tk_total {\n                Some(t) => super::topk::finalize_total(&t, n, &mask, &KS_DEFAULT, &pre),\n                None => topk_one_mat(&ms.rho, &ms.e, n, &mask, &KS_DEFAULT, &pre),\n            };", 'finalize')
open(p,'w').write(s)
print('\n'.join(done))
