import os, numpy as np
import rust_pyfunc as rp
v = rp.okay_compute_stock('hm076', '000001', 20220819)
print('n', len(v))
print('DUMP=' + ','.join('%.17e' % x for x in v))
