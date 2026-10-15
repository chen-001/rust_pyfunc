"""Check a benchmark output against the full output, allowing only specified Top-K deletions."""
from pathlib import Path
import argparse, json, re
import numpy as np
p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('reference',type=Path);a=p.parse_args()
c=json.loads((a.candidate/'result.json').read_text());r=json.loads((a.reference/'result.json').read_text())
removed={1,2,10,50,200,500}
def keep(name):
    m=re.search(r'_(?:x0topk|gtopk)(\d+)_',name)
    return not m or int(m[1]) not in removed
indices=[i for i,name in enumerate(r['names']) if keep(name)]
assert c['codes']==r['codes'] and c['date']==r['date']
assert c['names']==[r['names'][i] for i in indices]
x=np.load(a.candidate/'values.npy',mmap_mode='r');y=np.load(a.reference/'values.npy',mmap_mode='r')
assert x.shape==(len(c['codes']),len(indices))
diff=sum(int(np.count_nonzero(x[:,j].view('u4')!=y[:,i].view('u4'))) for j,i in enumerate(indices))
result=dict(date=c['date'],threads=c['threads'],seconds=c['seconds'],stocks=len(c['codes']),columns=len(indices),removed_columns=len(r['names'])-len(indices),bit_differences=diff,order_equal=True,kept_k=[5,20,100],sha256=c['sha256'])
(a.candidate/'subset_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result));assert diff==0
