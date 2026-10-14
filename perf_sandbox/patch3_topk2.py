p='/home/chenzongwei/rust_pyfunc/src/yupei_pair/topk.rs'
s=open(p).read()
old = """            cols.push(c_ms);
        }
    }
    eprintln!(
        "[topk] {nlev} 档 × 2 口径 {:.1}s，pairs={pairs}",
        t0.elapsed().as_secs_f64()
    );
    (names, cols)
}"""
new = """            cols.push(c_ms);
        }
    }
    (names, cols)
}"""
assert old in s, 'MISS tail'
s = s.replace(old, new, 1)
open(p,'w').write(s)
print('OK tail fixed')
