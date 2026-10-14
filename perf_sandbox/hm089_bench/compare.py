import sys
import numpy as np
import rust_pyfunc as rp
p = sys.argv[1]
code = sys.argv[2]
date = int(sys.argv[3])
for line in open(p):
    if line.startswith('DUMP='):
        dump = np.array([float(x) for x in line.strip()[5:].split(',')])
lib = np.array(rp.okay_compute_stock('hm089', code, date), dtype=np.float64)
a = dump.astype(np.float32); b = lib.astype(np.float32)
print('文件 %s  比对 %s@%d' % (p.split('/')[-1], code, date))
print('len bench=%d lib=%d' % (len(dump), len(lib)))
print('f32 逐位相同: %d / %d' % (int((a == b).sum()), len(a)))
d = np.abs(dump - lib); i = int(np.argmax(d))
print('最大绝对差 %.3e @idx%d  bench=%.17g lib=%.17g' % (d[i], i, dump[i], lib[i]))
