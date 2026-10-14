import re, numpy as np
def load(path, tag):
    for line in open(path):
        if line.startswith(tag):
            return np.array([float(x) for x in line.strip()[len(tag):].split(',')])
    return None
a = load('/home/chenzongwei/rust_pyfunc/perf_sandbox/hm076_bench/bench_out.txt', 'DUMP=')
b = load('/home/chenzongwei/rust_pyfunc/perf_sandbox/hm076_bench/ref_out.txt', 'DUMP=')
print('bench n', None if a is None else len(a), 'lib n', None if b is None else len(b))
if a is not None and b is not None:
    d = np.abs(a - b)
    print('max |bench - lib| = %.6e' % d.max())
    print('相同位数(逐值相等个数) = %d / %d' % (int((a == b).sum()), len(a)))
    i = int(d.argmax())
    print('最大差那一项: bench=%.17e lib=%.17e' % (a[i], b[i]))
    print('值域 |lib| max = %.6e' % np.abs(b).max())
