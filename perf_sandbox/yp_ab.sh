#!/bin/bash
PY=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
export RAYON_NUM_THREADS=10
echo "=== A：生产库（改前，全家族）==="
$PY -u -c "
import os, time, rust_pyfunc as rp
print('module', rp.__file__)
t=time.time(); c,n,v = rp.py_yupei_pair(20241231, ['cnt_t1']); dt=time.time()-t
print('A_PROD_SECONDS %.1f  F=%d' % (dt, len(n)))
"
echo "=== B：沙箱裁剪版（改后，只算 cnt）==="
PYTHONPATH=/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_pkg $PY -u -c "
import os, time, rust_pyfunc as rp
print('module', rp.__file__)
t=time.time(); c,n,v = rp.py_yupei_pair(20241231, ['cnt_t1']); dt=time.time()-t
print('B_PATCHED_SECONDS %.1f  F=%d' % (dt, len(n)))
"
echo DONE
