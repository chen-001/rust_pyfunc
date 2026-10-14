"""Same Rust reader loop and checksum; vary only baseline/strict/grouped reads."""
import json,os,random,tempfile
from pathlib import Path
import storage_sandbox as s
out=[]
for label,root in [('NAS','/nas197/user_home/chenzongwei'),('HDD','/hdd/user_home_unsafe/chenzongwei')]:
 with tempfile.TemporaryDirectory(prefix='codex_storage_bench_',dir=root) as tmp:
  store=str(Path(tmp)/'store');s.factor_store_v5_smoke_proj(store,300000,32)
  cols=list(range(32));random.Random(42).shuffle(cols)
  for rep in range(3):
   for mode in (['baseline','strict','batch'] if rep%2==0 else ['batch','strict','baseline']):
    for p in Path(store).glob('factors.*'):
     fd=os.open(p,os.O_RDONLY);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);os.close(fd)
    sec,checksum,reads=s.bench_read(store,cols,mode)
    out.append(dict(device=label,rep=rep,mode=mode,seconds=sec,checksum=checksum,logical_reads=reads))
    print(out[-1],flush=True)
assert len({v['checksum'] for v in out})==1
Path('/tmp/storage_adaptive_bench.json').write_text(json.dumps(out,indent=2))
