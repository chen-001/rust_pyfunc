#!/bin/bash
PY=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
export RAYON_NUM_THREADS=10
run() {
  local tag="$1"; local pp="$2"
  if [ -z "$pp" ]; then
    $PY -u -c "
import time, rust_pyfunc as rp
t=time.time(); c,n,v = rp.py_yupei_pair(20241231, ['cnt_t1']); dt=time.time()-t
print('$tag %.1f' % dt)
" 2>/dev/null
  else
    PYTHONPATH=$pp $PY -u -c "
import time, rust_pyfunc as rp
t=time.time(); c,n,v = rp.py_yupei_pair(20241231, ['cnt_t1']); dt=time.time()-t
print('$tag %.1f' % dt)
" 2>/dev/null
  fi
}
run A1_prod ""
run B_patched "/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_pkg"
run A2_prod ""
echo DONE
