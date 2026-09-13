#!/bin/bash
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:/home/chenzongwei/.conda/envs/chenzongwei311/bin:$HOME/.local/bin:$PATH"
export CONDA_PREFIX=/home/chenzongwei/.conda/envs/chenzongwei311
export VIRTUAL_ENV=$CONDA_PREFIX
/home/chenzongwei/.local/bin/mold -run /home/chenzongwei/.conda/envs/chenzongwei311/bin/maturin develop 2>&1
