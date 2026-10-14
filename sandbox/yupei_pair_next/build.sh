#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Recreate the isolated source snapshot after checkout; refuse an unrelated baseline.
if [ ! -f src/lib.rs ]; then
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python - <<'PYBASE'
from pathlib import Path
import hashlib, json
root = Path('../..')
for name, expected in json.loads(Path('baseline_sources.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, f'Baseline changed: {name}'
PYBASE
  cp -a ../../src/. ./src/
  cp ../../Cargo.toml ../../Cargo.lock ./
  # bitonic_sort.rs is already tracked in this sandbox; apply only the two integrations.
  patch -p1 < <(sed '/^--- \/dev\/null/,$d' bitonic.patch)
fi
export PATH="/home/chenzongwei/.cargo/bin:$PATH"
export PYO3_PYTHON=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
# Independent source, shared dependency/build cache. Never installs into Python or deploys worker.
cargo build --lib --profile release-fast --jobs 16 --target-dir /home/chenzongwei/rust_pyfunc/target
mkdir -p package/rust_pyfunc
cp /home/chenzongwei/rust_pyfunc/target/release-fast/librust_pyfunc.so package/rust_pyfunc/rust_pyfunc.next.so
mv package/rust_pyfunc/rust_pyfunc.next.so package/rust_pyfunc/rust_pyfunc.abi3.so
printf 'from .rust_pyfunc import *\n' > package/rust_pyfunc/__init__.py
