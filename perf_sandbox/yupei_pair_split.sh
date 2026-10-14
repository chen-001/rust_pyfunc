#!/bin/bash
cd /home/chenzongwei/rust_pyfunc/perf_sandbox
export RAYON_NUM_THREADS=50
PY=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
echo "=== load ==="; cat /proc/loadavg
echo "=== T([]) 只要上游 ==="
$PY -u -c "
import time, rust_pyfunc as rp
t0=time.time()
try:
    rp.py_yupei_pair(20241231, [])
except Exception as e:
    print('  (预期报错: %s)' % str(e)[:60])
print('UPSTREAM_SECONDS %.2f' % (time.time()-t0))
"
echo "=== 裸读 23GB 逐笔的耗时（I/O 下限）==="
sync; S=$(date +%s.%N); cat /ssd_data/stock/20241231/transaction/*.csv > /dev/null; E=$(date +%s.%N)
echo "CAT_SECONDS $(echo "$E - $S" | bc)"
echo "=== T(15) 默认 15 张矩阵 ==="
$PY -u -c "
import time, rust_pyfunc as rp
t0=time.time(); c,n,v = rp.py_yupei_pair(20241231); dt=time.time()-t0
print('DEFAULT_SECONDS %.2f  X=%d F=%d' % (dt, len(c), len(n)))
" 2>/dev/null
echo DONE
