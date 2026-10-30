"""Reproduce the three R6 full-day trials after building the isolated sandbox.

Requires default_after reference arrays and the four earlier candidate runs.
Each variant runs in a fresh process; saved output is checked before continuing.
"""
import os, subprocess, json, hashlib
from pathlib import Path
root=Path('/home/chenzongwei/rust_pyfunc')
r=root/'reports/yupei_multi_ks5_20_20261030'
sandbox=root/'sandbox/yupei_multi_r6_20261030'
py='/home/chenzongwei/.conda/envs/chenzongwei311/bin/python'
so=sandbox/'package/rust_pyfunc/rust_pyfunc.abi3.so'
manifest=json.loads((r/'sandbox_build.json').read_text())
manifest['module_sha256']=hashlib.sha256(so.read_bytes()).hexdigest()
manifest['source_sha256']={str(p.relative_to(sandbox)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [sandbox/name for name in manifest['source_sha256']] + [sandbox/'src/yupei_pair/r6_streaming.rs']}
manifest['switches'].append('YUPEI_R6_STREAMING')
(r/'r6_build.json').write_text(json.dumps(manifest,indent=2))
for run,flag in [('r6_control','0'),('r6_fast','1'),('r6_repeat','1')]:
 env=os.environ.copy()
 env.update(RAYON_NUM_THREADS='50',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',YUPEI_STAGE='1',YUPEI_BENCH_PACKAGE=str(sandbox/'package'),YUPEI_R6_STREAMING=flag)
 env.update({k:'1' for k in ['YUPEI_TOPK_HIGH_FIRST','YUPEI_MASS_AVX512','YUPEI_MATRIX_TWO_TARGET','YUPEI_SHARE_RHO_REDUCE','YUPEI_NET_MIRROR']})
 with (r/(run+'.log')).open('w') as log:
  subprocess.run([py,str(root/'tests/yupei_multi_perf_20261030/bench.py'),str(r/run),'20260717'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
 with (r/(run+'_comparison.log')).open('w') as log:
  subprocess.run([py,str(root/'tests/yupei_multi_perf_20261030/compare_topk.py'),str(r/'default_after'),str(r/run),str(r/(run+'_comparison.json'))],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 cmp=json.loads((r/(run+'_comparison.json')).read_text())
 assert cmp['same_column_order'] and not cmp['added'] and not cmp['removed'] and all(g['bit_mismatches']==0 for g in cmp['groups'].values()),run
 t=json.loads((r/run/'timing.json').read_text())
 print(run,t['call_seconds'],'seconds; exact-output PASS',flush=True)
ids=['sandbox_base','kernel_fast','shared_fast','mirror_fast','r6_control','r6_fast','r6_repeat']
ts={n:json.loads((r/n/'timing.json').read_text())['call_seconds'] for n in ids}
# Recommendation is reviewed separately, accounting for matched controls and noise.
(r/'r6_trial_times.json').write_text(json.dumps(ts,indent=2))
