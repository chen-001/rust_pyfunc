"""分段确认不得改变投影产物的字节内容：段大小从 1 GiB 到 1 字节，factors.proj 必须逐字节相同。"""
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
import storage_sandbox as sb

names=['f0','f1','f2','f3']
codes=[f'{i:06}' for i in range(1000)]
values=(np.arange(1000,dtype=np.float32)[:,None]*np.float32(.001)+np.arange(4,dtype=np.float32)).tolist()
expected=[np.sort(np.tile(np.asarray(values,dtype=np.float32)[:,c],5)) for c in range(4)]

def build(chunk):
 os.environ['RUST_PYFUNC_PROJ_SYNC_CHUNK']=str(chunk)
 with tempfile.TemporaryDirectory(prefix='codex_proj_chunk_',dir='/nas197/user_home/chenzongwei') as tmp:
  store=str(Path(tmp)/'store')
  for day in range(5):sb.append_store(store,[20230102+day]*1000,codes,values,names,True)
  proj=Path(store)/'factors.proj'
  match=all(np.array_equal(np.sort(np.asarray(sb.factor_store_v5_read_factor(store,c)['factor'],dtype=np.float32)),expected[c]) for c in range(4))
  assert match,'读回数值与输入不一致'
  return hashlib.sha256(proj.read_bytes()).hexdigest(),proj.stat().st_size

rows=[]
for chunk in (1<<30,4096,1):
 h,size=build(chunk);rows.append((chunk,size,h))
 print(json.dumps({'chunk_bytes':chunk,'proj_bytes':size,'sha256':h[:16]}),flush=True)
assert len({h for _,_,h in rows})==1,[r[2] for r in rows]
out=Path('/tmp/proj_chunk_invariance.csv')
out.write_text('chunk_bytes,proj_bytes,sha256,values_match\n'+''.join(f'{c},{s},{h},True\n' for c,s,h in rows))
print(f'✅ 三种段大小（1 GiB / 4096 / 1 字节）投影产物逐字节相同，读回数值与输入一致 → {out}')
