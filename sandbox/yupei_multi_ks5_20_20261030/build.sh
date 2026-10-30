#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Recreate the frozen source only when absent; never overwrite an active experiment.
if [ ! -f src/lib.rs ]; then
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python - <<'PYBASE'
from pathlib import Path
import hashlib, json, shutil
root = Path('../..')
for name, expected in json.loads(Path('baseline_sources.json').read_text()).items():
    actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual == expected, f'Baseline changed: {name}: {actual} != {expected}'
shutil.copytree(root/'src', Path('src'), dirs_exist_ok=True)
for name in ('Cargo.toml', 'Cargo.lock', 'build.rs'):
    if (root/name).exists():
        shutil.copy2(root/name, Path(name))
PYBASE
  patch -p1 < optimization.patch
fi
# Independent source and package, shared compiler cache. No install or worker deployment.
export PATH="/home/chenzongwei/.cargo/bin:$PATH"
export PYO3_PYTHON=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
cargo build --lib --profile release-fast --jobs 16 --target-dir /home/chenzongwei/rust_pyfunc/target
mkdir -p package/rust_pyfunc
cp /home/chenzongwei/rust_pyfunc/target/release-fast/librust_pyfunc.so package/rust_pyfunc/rust_pyfunc.next.so
mv package/rust_pyfunc/rust_pyfunc.next.so package/rust_pyfunc/rust_pyfunc.abi3.so
printf 'from .rust_pyfunc import *\n' > package/rust_pyfunc/__init__.py
