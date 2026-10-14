#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/chenzongwei/.cargo/bin:$PATH"
export PYO3_PYTHON=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
# Independent source, shared dependency/build cache. Never installs into Python or deploys worker.
cargo build --lib --profile release-fast --jobs 16 --target-dir /home/chenzongwei/rust_pyfunc/target
mkdir -p package/rust_pyfunc
cp /home/chenzongwei/rust_pyfunc/target/release-fast/librust_pyfunc.so package/rust_pyfunc/rust_pyfunc.next.so
mv package/rust_pyfunc/rust_pyfunc.next.so package/rust_pyfunc/rust_pyfunc.abi3.so
printf 'from .rust_pyfunc import *\n' > package/rust_pyfunc/__init__.py
