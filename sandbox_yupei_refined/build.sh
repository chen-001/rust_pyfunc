#!/usr/bin/env bash
# Build ONLY this independent sandbox; never install or rebuild rust_pyfunc.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v cmake >/dev/null 2>&1; then
    export PATH="/home/chenzongwei/.conda/envs/chenzongwei311/bin:$PATH"
fi
cargo build --release --locked "$@"
