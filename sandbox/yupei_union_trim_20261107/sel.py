# 四段入选名单（去 _fold 后缀）与四合一列名分段；被 exec 引用
import json, re, collections, rust_pyfunc as rp
P='/home/chenzongwei/pythoncode/'
fs={'lite':'寻找玉佩-双层版/yupei_dist_lite_tail_v4_selected.json','split':'寻找玉佩-双层版/yupei_pair_split_tail_v4_selected.json','gpt':'寻找玉佩双层版 gpt 改进/yupei_gpt_tail_v4_selected.json','v5':'寻找玉佩双层版-多种同步度/yupei_multi_v5_tail_v4_selected.json'}
sel={k:set(json.load(open(P+f))['source_names']) for k,f in fs.items()}
names=rp.py_yupei_union_all_names()
seg={'lite':names[:741],'gpt':names[741:1232],'split':names[1232:11027],'v5':names[11027:]}
v5full=rp.py_yupei_multi_v5_names() if hasattr(rp,'py_yupei_multi_v5_names') else None
allsel=set()
for s in sel.values():
    for n in s: allsel.add(n[:-5] if n.endswith('_fold') else n)
nameset=set(names)
missing=[n for n in allsel if n not in nameset]
